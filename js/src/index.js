/**
 * ninada — Streaming Pan-Indic & Classical Sanskrit Neural TTS (Browser WASM SIMD & Node.js)
 *
 * Streaming quick start:
 *   import ninada from 'ninada';
 *   const voice = await ninada('sanskrit', {
 *     encoderModel: './encoder_flow_quantized.onnx',
 *     vocoderModel: './vocoder_quantized.onnx',
 *   });
 *   for await (const chunk of voice.stream('प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् ।')) {
 *     // chunk.pcm (Float32Array @ 16kHz), chunk.ttfaMs (<250ms!)
 *   }
 */
import {
  VOCAB_TOKENS,
  TOKEN_TO_ID,
  BLANK_ID,
  LANGUAGE_IDS,
  detectScriptAndLang,
  iastToDevanagari,
  brahmicToDevanagari,
  applyPhonologicalRules,
  segmentAksharas,
  textToIds,
} from './g2p.js';
import { SAMPLE_RATE, synthesizeKlattPcm, encodeWavBuffer } from './klatt.js';

const HOP_LENGTH = 256;

const LANG_ALIASES = {
  sa: 'sa', sanskrit: 'sa',
  hi: 'hi', hindi: 'hi',
  kn: 'kn', kannada: 'kn',
  ta: 'ta', tamil: 'ta',
  te: 'te', telugu: 'te',
  ml: 'ml', malayalam: 'ml',
};

function resolveLang(val) {
  if (!val || typeof val !== 'string') return 'sa';
  const k = val.trim().toLowerCase();
  return LANG_ALIASES[k] || 'sa';
}

function splitProsodicClauses(text, maxChars = 160) {
  const cleaned = String(text || '').replace(/\s+/g, ' ').trim();
  if (!cleaned) return [];
  const rawParts = cleaned.split(/(?<=\|\||[\u0964\u0965|.?!;\n])\s+/g);
  const clauses = [];
  for (const part of rawParts) {
    const p = (part || '').trim();
    if (!p) continue;
    // Merge standalone punctuation markers (e.g. trailing danda) into previous clause
    if (/^[\u0964\u0965|.?!;,\s-]+$/.test(p)) {
      if (clauses.length > 0) clauses[clauses.length - 1] += ` ${p}`;
      continue;
    }
    if (p.length <= maxChars) {
      clauses.push(p);
      continue;
    }
    const words = p.split(' ');
    let buf = '';
    for (const w of words) {
      if (!buf) buf = w;
      else if (buf.length + 1 + w.length <= maxChars) buf = `${buf} ${w}`;
      else {
        clauses.push(buf);
        buf = w;
      }
    }
    if (buf) clauses.push(buf);
  }
  return clauses.length > 0 ? clauses : [cleaned];
}

async function resolveOrt(explicitOrt) {
  if (explicitOrt && explicitOrt.InferenceSession) return explicitOrt;
  if (typeof globalThis !== 'undefined' && globalThis.ort && globalThis.ort.InferenceSession) {
    return globalThis.ort;
  }
  for (const pkg of ['onnxruntime-node', 'onnxruntime-web']) {
    try {
      const mod = await import(pkg);
      const candidate = mod.default && mod.default.InferenceSession ? mod.default : mod;
      if (candidate && candidate.InferenceSession) return candidate;
    } catch {}
  }
  return null;
}

async function createOrtSession(ort, source, providers) {
  if (!source || !ort || !ort.InferenceSession) return null;
  const isNode = typeof process !== 'undefined' && Boolean(process.versions?.node) && typeof window === 'undefined';
  let resolved = source;
  if (isNode && typeof source === 'string' && !/^https?:\/\//i.test(source) && ort.env?.wasm) {
    const fsPkg = 'node:fs/promises';
    const fsMod = await import(fsPkg);
    const buf = await fsMod.readFile(source);
    resolved = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  const useProviders = providers || (ort.env?.wasm ? ['wasm'] : ['cpu']);
  return ort.InferenceSession.create(resolved, {
    executionProviders: useProviders,
    graphOptimizationLevel: 'all',
    enableCpuMemArena: false,
  });
}

function nowMs() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

export default async function ninada(input = 'sa', opts = {}) {
  let defaultLang = 'sa';
  let modelSource = opts.model || null;
  let encSource = opts.encoderModel || null;
  let vocSource = opts.vocoderModel || null;

  if (typeof input === 'string') {
    const k = input.trim().toLowerCase();
    if (LANG_ALIASES[k]) {
      defaultLang = LANG_ALIASES[k];
    } else {
      modelSource = input;
    }
  } else if (input instanceof ArrayBuffer || ArrayBuffer.isView(input)) {
    modelSource = input;
  } else if (input && typeof input === 'object') {
    opts = input;
    defaultLang = resolveLang(opts.lang);
    modelSource = opts.model || null;
    encSource = opts.encoderModel || null;
    vocSource = opts.vocoderModel || null;
  }

  let session = opts.session || null;
  let encSession = opts.encoderSession || null;
  let vocSession = opts.vocoderSession || null;

  const needOrt = !session && !encSession && Boolean(modelSource || (encSource && vocSource));
  const ort = needOrt || session || encSession ? await resolveOrt(opts.ort) : null;

  if (!encSession && !vocSession && encSource && vocSource && ort) {
    encSession = await createOrtSession(ort, encSource, opts.providers);
    encSource = null;
    vocSession = await createOrtSession(ort, vocSource, opts.providers);
    vocSource = null;
  } else if (!session && modelSource && ort) {
    session = await createOrtSession(ort, modelSource, opts.providers);
    modelSource = null;
  }

  // Drop raw Uint8Array/ArrayBuffer references immediately so GC frees ~57MB JS heap
  modelSource = null;
  encSource = null;
  vocSource = null;

  const listeners = new Map();
  const emit = (ev, payload) => {
    const arr = listeners.get(ev);
    if (arr) for (const fn of arr) fn(payload);
  };

  return {
    get lang() {
      return defaultLang;
    },
    get isNeural() {
      return Boolean((encSession && vocSession) || session);
    },
    get isStreamingNeural() {
      return Boolean(encSession && vocSession);
    },
    on(event, handler) {
      if (!listeners.has(event)) listeners.set(event, []);
      listeners.get(event).push(handler);
      return this;
    },
    inspect(text, callOpts = {}) {
      return textToIds(text, { lang: callOpts.lang ? resolveLang(callOpts.lang) : defaultLang });
    },
    async *stream(text, callOpts = {}) {
      const lang = callOpts.lang ? resolveLang(callOpts.lang) : defaultLang;
      const rate = callOpts.rate ?? opts.rate ?? 1.0;
      const noise = callOpts.noise ?? opts.noise ?? 0.32;
      const tier = callOpts.tier || 'auto';
      const firstChunkFrames = Math.max(20, Number(callOpts.firstChunkFrames || 40));
      const chunkFrames = Math.max(32, Number(callOpts.chunkFrames || 96));
      const haloFrames = Math.max(4, Number(callOpts.haloFrames || 8));
      const signal = callOpts.signal || null;

      const clauses = splitProsodicClauses(text, callOpts.maxClauseChars || 160);
      const tStart = nowMs();
      let ttfaMs = null;
      let chunkIndex = 0;

      // Path 1: True Halo-Windowed Neural Streaming (Encoder+Flow ~35ms -> 40-frame HiFi-GAN chunks)
      if (tier !== 'klatt' && encSession && vocSession && ort) {
        for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
          if (signal?.aborted) return;
          const parsed = textToIds(clauses[cIdx], { lang });
          const feeds = {
            input_ids: new ort.Tensor('int64', BigInt64Array.from(parsed.inputIds.map(BigInt)), [1, parsed.inputIds.length]),
            attention_mask: new ort.Tensor('int64', BigInt64Array.from(parsed.attentionMask.map(BigInt)), [1, parsed.attentionMask.length]),
            lang_id: new ort.Tensor('int64', BigInt64Array.from([BigInt(parsed.langId)]), [1]),
            scales: new ort.Tensor('float32', Float32Array.from([Number(rate), Number(noise)]), [2]),
          };

          const encOut = await encSession.run(feeds);
          feeds.input_ids.dispose?.();
          feeds.attention_mask.dispose?.();
          feeds.lang_id.dispose?.();
          feeds.scales.dispose?.();
          if (signal?.aborted) {
            encOut.latents.dispose?.();
            encOut.voice_cond.dispose?.();
            encOut.durations.dispose?.();
            return;
          }

          const latData = Float32Array.from(encOut.latents.data);
          const totalFrames = encOut.latents.dims[2];
          const channels = encOut.latents.dims[1]; // 192
          const voiceCondData = Float32Array.from(encOut.voice_cond.data);
          const voiceCondDims = encOut.voice_cond.dims.slice();
          const dursArr = Array.from(encOut.durations.data, Number);
          encOut.latents.dispose?.();
          encOut.voice_cond.dispose?.();
          encOut.durations.dispose?.();
          // Adaptive first-chunk streaming: fast 40-frame first chunk for low TTFA, 96-frame steady chunks for low halo overhead
          const bounds = [];
          for (let cur = 0; cur < totalFrames; ) {
            const step = cur === 0 && cIdx === 0 ? firstChunkFrames : chunkFrames;
            const nxt = Math.min(totalFrames, cur + step);
            bounds.push([cur, nxt]);
            cur = nxt;
          }
          const numSub = bounds.length;

          for (let subIdx = 0; subIdx < numSub; subIdx++) {
            if (signal?.aborted) return;
            const [st, ed] = bounds[subIdx];
            const wSt = Math.max(0, st - haloFrames);
            const wEd = Math.min(totalFrames, ed + haloFrames);
            const winLen = wEd - wSt;

            // Slice [1, 192, winLen] from [1, 192, totalFrames]
            const sliceBuf = new Float32Array(channels * winLen);
            for (let ch = 0; ch < channels; ch++) {
              const srcBase = ch * totalFrames + wSt;
              const dstBase = ch * winLen;
              for (let f = 0; f < winLen; f++) {
                sliceBuf[dstBase + f] = latData[srcBase + f];
              }
            }

            const tChunk = nowMs();
            const latTensor = new ort.Tensor('float32', sliceBuf, [1, channels, winLen]);
            const vcTensor = new ort.Tensor('float32', voiceCondData, voiceCondDims);
            const vocOut = await vocSession.run({
              latents: latTensor,
              voice_cond: vcTensor,
            });
            latTensor.dispose?.();
            vcTensor.dispose?.();
            if (signal?.aborted) {
              vocOut.waveform.dispose?.();
              return;
            }

            const raw = vocOut.waveform.data;
            const cropL = (st - wSt) * HOP_LENGTH;
            const cropLen = (ed - st) * HOP_LENGTH;
            const pcm = new Float32Array(cropLen);
            for (let i = 0; i < cropLen; i++) {
              const v = (raw[cropL + i] || 0) * 0.90;
              pcm[i] = v > 0.98 ? 0.98 : v < -0.98 ? -0.98 : v;
            }
            vocOut.waveform.dispose?.();

            const elapsedNow = nowMs() - tStart;
            if (ttfaMs === null) ttfaMs = elapsedNow;
            const isFinal = cIdx === clauses.length - 1 && subIdx === numSub - 1;
            const chunkObj = {
              pcm,
              wav: encodeWavBuffer(pcm, SAMPLE_RATE),
              sampleRate: SAMPLE_RATE,
              index: chunkIndex++,
              clauseIndex: cIdx,
              clauseCount: clauses.length,
              isFinal,
              durationSec: pcm.length / SAMPLE_RATE,
              latencyMs: nowMs() - tChunk,
              ttfaMs,
              lang: parsed.lang,
              script: parsed.script,
              tier: 'neural_onnx',
              aksharas: subIdx === 0 ? parsed.aksharas : [],
              durations: subIdx === 0 ? dursArr : [],
            };
            emit('chunk', chunkObj);
            yield chunkObj;
          }
        }
        return;
      }

      // Path 2: Clause-Level Neural Streaming over Monolithic ONNX
      if (tier !== 'klatt' && session && ort) {
        for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
          if (signal?.aborted) return;
          const tChunk = nowMs();
          const parsed = textToIds(clauses[cIdx], { lang });
          const feeds = {
            input_ids: new ort.Tensor('int64', BigInt64Array.from(parsed.inputIds.map(BigInt)), [1, parsed.inputIds.length]),
            attention_mask: new ort.Tensor('int64', BigInt64Array.from(parsed.attentionMask.map(BigInt)), [1, parsed.attentionMask.length]),
            lang_id: new ort.Tensor('int64', BigInt64Array.from([BigInt(parsed.langId)]), [1]),
            scales: new ort.Tensor('float32', Float32Array.from([Number(rate), Number(noise)]), [2]),
          };
          const out = await session.run(feeds);
          if (signal?.aborted) return;
          const raw = out.waveform.data;
          const pcm = new Float32Array(raw.length);
          let peak = 1e-5;
          for (let i = 0; i < raw.length; i++) {
            const a = Math.abs(raw[i]);
            if (a > peak) peak = a;
          }
          const gain = peak > 1e-4 ? 0.88 / Math.max(1.0, peak) : 1.0;
          for (let i = 0; i < raw.length; i++) pcm[i] = raw[i] * gain;

          if (ttfaMs === null) ttfaMs = nowMs() - tStart;
          const chunkObj = {
            pcm,
            wav: encodeWavBuffer(pcm, SAMPLE_RATE),
            sampleRate: SAMPLE_RATE,
            index: chunkIndex++,
            clauseIndex: cIdx,
            clauseCount: clauses.length,
            isFinal: cIdx === clauses.length - 1,
            durationSec: pcm.length / SAMPLE_RATE,
            latencyMs: nowMs() - tChunk,
            ttfaMs,
            lang: parsed.lang,
            script: parsed.script,
            tier: 'neural_onnx',
            aksharas: parsed.aksharas,
            durations: Array.from(out.durations.data, Number),
          };
          emit('chunk', chunkObj);
          yield chunkObj;
        }
        return;
      }

      // Path 3: Zero-Dependency Klatt Tier-2 Clause Streaming
      for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
        if (signal?.aborted) return;
        const tChunk = nowMs();
        const { pcm, meta: baseMeta } = synthesizeKlattPcm(clauses[cIdx], { lang, rate });
        if (ttfaMs === null) ttfaMs = nowMs() - tStart;
        const chunkObj = {
          pcm,
          wav: encodeWavBuffer(pcm, SAMPLE_RATE),
          sampleRate: SAMPLE_RATE,
          index: chunkIndex++,
          clauseIndex: cIdx,
          clauseCount: clauses.length,
          isFinal: cIdx === clauses.length - 1,
          durationSec: pcm.length / SAMPLE_RATE,
          latencyMs: nowMs() - tChunk,
          ttfaMs,
          lang: baseMeta.lang,
          script: baseMeta.script,
          tier: 'klatt_zero_dep',
          aksharas: baseMeta.aksharas,
          durations: [],
        };
        emit('chunk', chunkObj);
        yield chunkObj;
      }
    },
    async synthesize(text, callOpts = {}) {
      const lang = callOpts.lang ? resolveLang(callOpts.lang) : defaultLang;
      const parsed = textToIds(text, { lang });
      const t0 = nowMs();
      const chunks = [];
      const allDurations = [];
      let totalLen = 0;
      let activeTier = 'klatt_zero_dep';
      let ttfaMs = 0;

      for await (const ch of this.stream(text, callOpts)) {
        chunks.push(ch.pcm);
        totalLen += ch.pcm.length;
        activeTier = ch.tier;
        ttfaMs = ch.ttfaMs;
        if (ch.durations && ch.durations.length > 0) {
          allDurations.push(...ch.durations);
        }
      }

      const pcm = new Float32Array(Math.max(totalLen, 1));
      let offset = 0;
      for (const c of chunks) {
        pcm.set(c, offset);
        offset += c.length;
      }

      let peak = 1e-5;
      for (let i = 0; i < pcm.length; i++) {
        const a = Math.abs(pcm[i]);
        if (a > peak) peak = a;
      }
      if (peak > 0.98) {
        const g = 0.95 / peak;
        for (let i = 0; i < pcm.length; i++) pcm[i] *= g;
      }

      const elapsedMs = nowMs() - t0;
      const durationSec = pcm.length / SAMPLE_RATE;
      const meta = {
        ...parsed,
        tier: activeTier,
        sampleRate: SAMPLE_RATE,
        durationSec,
        latencyMs: elapsedMs,
        ttfaMs,
        chunks: chunks.length,
        rtf: elapsedMs / 1000 / Math.max(0.01, durationSec),
        durations: allDurations,
      };
      emit('done', meta);
      return { pcm, wav: encodeWavBuffer(pcm, SAMPLE_RATE), meta };
    },
    async speak(text, callOpts = {}) {
      const { wav } = await this.synthesize(text, callOpts);
      return wav;
    },
  };
}

export {
  ninada,
  SAMPLE_RATE,
  HOP_LENGTH,
  VOCAB_TOKENS,
  TOKEN_TO_ID,
  BLANK_ID,
  LANGUAGE_IDS,
  splitProsodicClauses,
  detectScriptAndLang,
  iastToDevanagari,
  brahmicToDevanagari,
  applyPhonologicalRules,
  segmentAksharas,
  textToIds,
  synthesizeKlattPcm,
  encodeWavBuffer,
};
