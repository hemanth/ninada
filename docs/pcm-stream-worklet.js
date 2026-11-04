/* Ninada Real-Time Continuous Ring-Buffer AudioWorkletProcessor (C1 Hermite Upsampling + Zero-Click Jitter Guard) */
class NinadaStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunks = [];
    this.headOffset = 0;
    this.queuedSamples = 0;
    this.srcRate = 24000;
    this.watermarkSamples = 1080; // default 45ms @ 24kHz
    this.state = 'idle'; // 'idle' | 'buffering' | 'playing'
    this.eos = false;

    // Continuous 4-point Hermite cubic resampler state across chunk boundaries
    this.hist = new Float32Array(4);
    this.histPrimed = false;
    this.drainRemaining = 2;
    this.phase = 0.0;

    // Smooth click-free envelope for starvation guard
    this.envGain = 1.0;
    this.sessionId = 0;
    this.playedSourceSamples = 0;
    this.totalPushedSamples = 0;
    this.underrunCount = 0;
    this.minBufferedPlayingSamples = Infinity;
    this.quantaCounter = 0;
    this.recoveryWatermarkSec = 0.24;

    this.port.onmessage = (ev) => {
      const msg = ev.data || {};
      if (msg.type === 'reset') {
        this.sessionId = Number(msg.sessionId) || 0;
        this.chunks = [];
        this.headOffset = 0;
        this.queuedSamples = 0;
        this.srcRate = Number(msg.sampleRate) || 24000;
        this.recoveryWatermarkSec = Number(msg.recoveryWatermarkSec) || this.recoveryWatermarkSec || 0.24;
        this.watermarkSamples = Math.max(128, Number(msg.watermarkSamples) || Math.round(0.045 * this.srcRate));
        this.state = 'buffering';
        this.eos = false;
        this.hist.fill(0);
        this.histPrimed = false;
        this.drainRemaining = 2;
        this.phase = 0.0;
        this.envGain = 1.0;
        this.playedSourceSamples = 0;
        this.totalPushedSamples = 0;
        this.underrunCount = 0;
        this.minBufferedPlayingSamples = Infinity;
        this.quantaCounter = 0;
        return;
      }

      if (msg.type === 'watermark') {
        if (typeof msg.watermarkSamples === 'number' && msg.watermarkSamples >= 128) {
          this.watermarkSamples = Math.round(msg.watermarkSamples);
        }
        return;
      }

      if (msg.type === 'push') {
        const pcm = msg.pcm;
        if (pcm && pcm.length > 0) {
          if (msg.sessionId && this.sessionId === 0) this.sessionId = Number(msg.sessionId) || 0;
          if (msg.sampleRate) this.srcRate = Number(msg.sampleRate);
          this.chunks.push(pcm);
          this.queuedSamples += pcm.length;
          this.totalPushedSamples += pcm.length;
          if (this.state === 'idle') {
            this.state = 'buffering';
          }
        }
        return;
      }

      if (msg.type === 'eos') {
        if (msg.sessionId && this.sessionId !== 0 && msg.sessionId !== this.sessionId) return;
        if (msg.sessionId && this.sessionId === 0) this.sessionId = Number(msg.sessionId) || 0;
        this.eos = true;
        if (this.state === 'buffering' && this.queuedSamples > 0 && !Number.isFinite(this.minBufferedPlayingSamples)) {
          this.minBufferedPlayingSamples = this.queuedSamples;
        }
        if (this.state === 'buffering' && this.queuedSamples === 0) {
          this.state = 'idle';
          this.port.postMessage({
            type: 'ended',
            sessionId: this.sessionId,
            playedSourceSamples: this.playedSourceSamples,
            underrunCount: this.underrunCount,
            minBufferedSamples: Number.isFinite(this.minBufferedPlayingSamples) ? this.minBufferedPlayingSamples : 0,
          });
        }
        return;
      }

      if (msg.type === 'stop') {
        this.chunks = [];
        this.headOffset = 0;
        this.queuedSamples = 0;
        this.state = 'idle';
        this.eos = true;
        this.hist.fill(0);
        this.histPrimed = false;
      }
    };
  }

  _shiftOneSourceSample() {
    if (this.queuedSamples <= 0 || this.chunks.length === 0) {
      return 0.0;
    }
    const head = this.chunks[0];
    const rawVal = head[this.headOffset];
    this.headOffset += 1;
    this.queuedSamples -= 1;
    this.playedSourceSamples += 1;
    if (this.headOffset >= head.length) {
      this.chunks.shift();
      this.headOffset = 0;
    }
    return rawVal;
  }

  _primeHistory() {
    if (this.histPrimed) return;
    const s0 = this._shiftOneSourceSample();
    const s1 = this.queuedSamples > 0 ? this._shiftOneSourceSample() : s0;
    const s2 = this.queuedSamples > 0 ? this._shiftOneSourceSample() : s1;
    this.hist[0] = s0;
    this.hist[1] = s0;
    this.hist[2] = s1;
    this.hist[3] = s2;
    this.phase = 0.0;
    this.histPrimed = true;
  }

  _advanceHistory() {
    this.hist[0] = this.hist[1];
    this.hist[1] = this.hist[2];
    this.hist[2] = this.hist[3];
    this.hist[3] = this.queuedSamples > 0 ? this._shiftOneSourceSample() : 0.0;
  }

  process(_inputs, outputs) {
    const out = outputs[0];
    if (!out || out.length === 0) return true;
    const ch0 = out[0];
    const nFrames = ch0.length;

    if (this.state === 'idle') {
      ch0.fill(0);
      for (let c = 1; c < out.length; c++) out[c].fill(0);
      return true;
    }

    let justStarted = false;
    if (this.state === 'buffering') {
      if (this.queuedSamples >= this.watermarkSamples || (this.eos && this.queuedSamples > 0)) {
        this._primeHistory();
        this.state = 'playing';
        justStarted = true;
        if (!Number.isFinite(this.minBufferedPlayingSamples)) {
          this.minBufferedPlayingSamples = this.queuedSamples;
        }
      } else {
        ch0.fill(0);
        for (let c = 1; c < out.length; c++) out[c].fill(0);
        return true;
      }
    }

    const ratio = this.srcRate / sampleRate;

    if (!this.eos && this.queuedSamples < this.minBufferedPlayingSamples) {
      this.minBufferedPlayingSamples = this.queuedSamples;
    }

    for (let i = 0; i < nFrames; i++) {
      // Starvation guard: smoothly fade to 0 instead of clicking if network/CPU ever stalls mid-utterance
      if (!this.eos && this.queuedSamples < 48) {
        this.envGain *= 0.94;
        if (this.queuedSamples === 0 && this.envGain < 0.002) {
          this.envGain = 0.0;
          this.underrunCount += 1;
          this.watermarkSamples = Math.round((this.recoveryWatermarkSec || 0.24) * this.srcRate);
          this.hist.fill(0);
          this.histPrimed = false;
          this.state = 'buffering';
          this.port.postMessage({
            type: 'underrun',
            sessionId: this.sessionId,
            underrunCount: this.underrunCount,
            playedSourceSamples: this.playedSourceSamples,
          });
          for (let k = i; k < nFrames; k++) ch0[k] = 0;
          break;
        }
      } else if (this.envGain < 1.0) {
        this.envGain = Math.min(1.0, this.envGain + 0.015);
      }

      // 4-point C1 Hermite cubic interpolation: exact pass-through at integer phase, continuous slope across seams
      const ym1 = this.hist[0];
      const y0 = this.hist[1];
      const y1 = this.hist[2];
      const y2 = this.hist[3];
      const t = this.phase;
      const c1 = 0.5 * (y1 - ym1);
      const c2 = ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2;
      const c3 = 0.5 * (y2 - ym1) + 1.5 * (y0 - y1);
      let sample = (((c3 * t + c2) * t + c1) * t + y0) * this.envGain;
      if (sample > 1.0) sample = 1.0;
      else if (sample < -1.0) sample = -1.0;
      ch0[i] = sample;

      this.phase += ratio;
      while (this.phase >= 1.0) {
        this.phase -= 1.0;
        if (this.queuedSamples === 0 && this.eos) {
          this.hist[0] = this.hist[1];
          this.hist[1] = this.hist[2];
          this.hist[2] = this.hist[3];
          this.hist[3] = 0.0;
          this.drainRemaining -= 1;
          if (this.drainRemaining < 0) {
            for (let k = i + 1; k < nFrames; k++) ch0[k] = 0;
            this.state = 'idle';
            this.eos = false;
            this.hist.fill(0);
            this.histPrimed = false;
            this.drainRemaining = 2;
            this.phase = 0.0;
            this.envGain = 1.0;
            this.port.postMessage({
              type: 'ended',
              sessionId: this.sessionId,
              playedSourceSamples: this.playedSourceSamples,
              underrunCount: this.underrunCount,
              minBufferedSamples: Number.isFinite(this.minBufferedPlayingSamples) ? this.minBufferedPlayingSamples : this.queuedSamples,
            });
            i = nFrames;
            break;
          }
        } else {
          this._advanceHistory();
        }
      }
    }

    for (let c = 1; c < out.length; c++) {
      out[c].set(ch0);
    }

    this.quantaCounter += 1;
    if (justStarted || this.quantaCounter >= 12) {
      this.quantaCounter = 0;
      this.port.postMessage({
        type: 'progress',
        sessionId: this.sessionId,
        playedSourceSamples: this.playedSourceSamples,
        bufferedSourceSamples: this.queuedSamples,
        minBufferedSamples: Number.isFinite(this.minBufferedPlayingSamples) ? this.minBufferedPlayingSamples : this.queuedSamples,
        underrunCount: this.underrunCount,
      });
    }

    return true;
  }
}

registerProcessor('ninada-stream-processor', NinadaStreamProcessor);
