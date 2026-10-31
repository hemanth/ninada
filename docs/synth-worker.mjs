import * as ortNs from './ort/ort.wasm.min.mjs';
import ninada from './ninada.mjs';

const ort = ortNs?.default?.InferenceSession ? ortNs.default : ortNs;

const ENC_URLS = [
  './encoder_flow_quantized.onnx',
  'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/encoder_flow_quantized.onnx',
];
const VOC_URLS = [
  './vocoder_quantized.onnx',
  'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/vocoder_quantized.onnx',
];
const MONO_URLS = [
  './model_quantized.onnx',
  'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/model_quantized.onnx',
];

let voiceEngine = null;
let initPromise = null;
let activeAbort = null;
let activeReqId = 0;
let runQueue = Promise.resolve();

async function fetchSingleBuffer(urls, expectedBytes, onDelta) {
  for (const url of urls) {
    try {
      const res = await fetch(url);
      if (!res.ok) continue;
      const exactLen = Number(res.headers.get('content-length')) || 0;
      if (!res.body) {
        const ab = await res.arrayBuffer();
        onDelta(ab.byteLength);
        return new Uint8Array(ab);
      }
      const reader = res.body.getReader();
      if (exactLen > 0) {
        const out = new Uint8Array(exactLen);
        let offset = 0;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (offset + value.byteLength <= out.byteLength) {
            out.set(value, offset);
          } else {
            const grown = new Uint8Array(offset + value.byteLength);
            grown.set(out.subarray(0, offset), 0);
            grown.set(value, offset);
            return grown;
          }
          offset += value.byteLength;
          onDelta(value.byteLength);
        }
        return offset === out.byteLength ? out : out.subarray(0, offset);
      }

      const parts = [];
      let loaded = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
        loaded += value.byteLength;
        onDelta(value.byteLength);
      }
      const merged = new Uint8Array(loaded || expectedBytes);
      let off = 0;
      for (const p of parts) {
        merged.set(p, off);
        off += p.byteLength;
      }
      return merged.subarray(0, off);
    } catch {
      // Try next mirror
    }
  }
  return null;
}

const SESSION_OPTS = {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
  enableCpuMemArena: false,
};

async function ensureReady() {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    voiceEngine = await ninada('sa');
    if (!ort || !ort.InferenceSession) {
      self.postMessage({
        type: 'ready',
        isNeural: false,
        isStreamingNeural: false,
        totalMB: 0,
        reason: 'ONNX Runtime WASM unavailable',
      });
      return voiceEngine;
    }

    ort.env.wasm.wasmPaths = new URL('./ort/', import.meta.url).href;
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads =
      typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated
        ? Math.min(4, self.navigator?.hardwareConcurrency || 2)
        : 1;

    let loadedBytes = 0;
    const expectedTotal = 56.6 * 1048576;
    const reportProgress = (delta) => {
      loadedBytes += delta;
      const loadedMB = loadedBytes / 1048576;
      const totalMB = expectedTotal / 1048576;
      const pct = Math.min(97, Math.round((loadedBytes / expectedTotal) * 100));
      self.postMessage({ type: 'progress', phase: 'download', loadedMB, totalMB, pct });
    };

    try {
      // Stage 1: Fetch & compile Encoder+Flow sequentially, then immediately free JS buffer
      let encBytes = await fetchSingleBuffer(ENC_URLS, 29.0 * 1048576, reportProgress);
      if (encBytes) {
        const encLen = encBytes.byteLength;
        self.postMessage({
          type: 'progress',
          phase: 'compile',
          loadedMB: encLen / 1048576,
          totalMB: expectedTotal / 1048576,
          pct: 52,
        });
        const encSession = await ort.InferenceSession.create(encBytes, SESSION_OPTS);
        encBytes = null;
        await new Promise((r) => setTimeout(r, 10));

        // Stage 2: Fetch & compile Halo-Chunked HiFi-GAN Vocoder, then immediately free JS buffer
        let vocBytes = await fetchSingleBuffer(VOC_URLS, 27.6 * 1048576, reportProgress);
        if (vocBytes) {
          const totalMB = (encLen + vocBytes.byteLength) / 1048576;
          self.postMessage({ type: 'progress', phase: 'compile', loadedMB: totalMB, totalMB, pct: 99 });
          const vocSession = await ort.InferenceSession.create(vocBytes, SESSION_OPTS);
          vocBytes = null;

          voiceEngine = await ninada('sa', {
            encoderSession: encSession,
            vocoderSession: vocSession,
            ort,
          });
          self.postMessage({
            type: 'ready',
            isNeural: voiceEngine.isNeural,
            isStreamingNeural: voiceEngine.isStreamingNeural,
            totalMB,
          });
          return voiceEngine;
        }
      }

      // Fallback to monolithic graph if split streaming graphs are unavailable
      loadedBytes = 0;
      let monoBytes = await fetchSingleBuffer(MONO_URLS, expectedTotal, reportProgress);
      if (monoBytes) {
        const totalMB = monoBytes.byteLength / 1048576;
        self.postMessage({ type: 'progress', phase: 'compile', loadedMB: totalMB, totalMB, pct: 99 });
        const monoSession = await ort.InferenceSession.create(monoBytes, SESSION_OPTS);
        monoBytes = null;
        voiceEngine = await ninada('sa', { session: monoSession, ort });
        self.postMessage({
          type: 'ready',
          isNeural: voiceEngine.isNeural,
          isStreamingNeural: false,
          totalMB,
        });
        return voiceEngine;
      }
    } catch (err) {
      self.postMessage({
        type: 'ready',
        isNeural: false,
        isStreamingNeural: false,
        totalMB: 0,
        reason: `WASM fallback (${err?.message || 'memory limit'}) — running Klatt Tier-2`,
      });
      return voiceEngine;
    }

    self.postMessage({
      type: 'ready',
      isNeural: false,
      isStreamingNeural: false,
      totalMB: 0,
      reason: 'Model weights not found — running Klatt Tier-2',
    });
    return voiceEngine;
  })();
  return initPromise;
}

self.addEventListener('message', (ev) => {
  const msg = ev.data || {};
  if (msg.type === 'init') {
    ensureReady().catch((err) => {
      self.postMessage({ type: 'error', reqId: 0, message: err?.message || String(err) });
    });
    return;
  }

  if (msg.type === 'cancel') {
    if (activeAbort && (!msg.reqId || msg.reqId === activeReqId)) {
      activeAbort.abort();
    }
    return;
  }

  if (msg.type === 'stream') {
    const reqId = Number(msg.reqId || Date.now());
    if (activeAbort) {
      activeAbort.abort();
    }
    const controller = new AbortController();
    activeAbort = controller;
    activeReqId = reqId;

    runQueue = runQueue
      .then(async () => {
        if (controller.signal.aborted) return;
        const engine = await ensureReady();
        if (controller.signal.aborted) return;

        const t0 = performance.now();
        let ttfaMs = null;
        let chunkCount = 0;
        let totalSamples = 0;
        let finalTier = 'klatt_zero_dep';
        let finalLang = msg.lang || 'sa';

        for await (const ch of engine.stream(msg.text || '', {
          lang: msg.lang || 'sa',
          rate: Number(msg.rate ?? 1.0),
          noise: Number(msg.noise ?? 0.32),
          tier: msg.tier || 'auto',
          firstChunkFrames: 40,
          chunkFrames: Number(msg.chunkFrames || 96),
          haloFrames: 8,
          signal: controller.signal,
        })) {
          if (controller.signal.aborted) return;
          if (ttfaMs === null) ttfaMs = performance.now() - t0;
          chunkCount += 1;
          totalSamples += ch.pcm.length;
          finalTier = ch.tier;
          finalLang = ch.lang;

          const pcmTransfer =
            ch.pcm.byteOffset === 0 && ch.pcm.byteLength === ch.pcm.buffer.byteLength
              ? ch.pcm
              : ch.pcm.slice();

          self.postMessage(
            {
              type: 'chunk',
              reqId,
              pcm: pcmTransfer,
              sampleRate: ch.sampleRate,
              index: ch.index,
              clauseIndex: ch.clauseIndex,
              clauseCount: ch.clauseCount,
              isFinal: ch.isFinal,
              durationSec: ch.durationSec,
              latencyMs: ch.latencyMs,
              ttfaMs,
              lang: ch.lang,
              script: ch.script,
              tier: ch.tier,
              aksharas: ch.aksharas,
              durations: ch.durations,
            },
            [pcmTransfer.buffer]
          );
        }

        if (controller.signal.aborted) return;
        const totalMs = performance.now() - t0;
        self.postMessage({
          type: 'done',
          reqId,
          totalMs,
          ttfaMs: ttfaMs ?? totalMs,
          chunkCount,
          totalSamples,
          durationSec: totalSamples / 16000,
          tier: finalTier,
          isNeural: engine.isNeural,
          isStreamingNeural: engine.isStreamingNeural,
          lang: finalLang,
        });
      })
      .catch((err) => {
        if (!controller.signal.aborted) {
          self.postMessage({
            type: 'error',
            reqId,
            message: err?.message || String(err),
          });
        }
      });
  }
});
