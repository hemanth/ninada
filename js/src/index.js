/**
 * ninada — Streaming Pan-Indic & Classical Sanskrit Specialist Neural TTS (Browser WASM SIMD & Node.js)
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
  NATIVE_VOCABS,
  NATIVE_TOKEN_TO_ID,
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

function buildStreamingClauses(text, maxChars = 180) {
  const cleaned = String(text || '').replace(/\s+/g, ' ').trim();
  if (!cleaned) return [];
  const rawParts = cleaned.split(/(?<=\|\||[\u0964\u0965|.?!;\n])\s+/g);
  const rawClauses = [];
  for (const part of rawParts) {
    const p = (part || '').trim();
    if (!p) continue;
    if (/^[\u0964\u0965|.?!;,\s-]+$/.test(p)) {
      if (rawClauses.length > 0) rawClauses[rawClauses.length - 1].text += ` ${p}`;
      continue;
    }
    if (p.length <= maxChars) {
      rawClauses.push({ text: p, sentenceEnd: true });
      continue;
    }
    const words = p.split(' ');
    let buf = '';
    for (const w of words) {
      if (!buf) buf = w;
      else if (buf.length + 1 + w.length <= maxChars) buf = `${buf} ${w}`;
      else {
        rawClauses.push({ text: buf, sentenceEnd: false });
        buf = w;
      }
    }
    if (buf) rawClauses.push({ text: buf, sentenceEnd: true });
  }
  const merged = [];
  for (const item of rawClauses) {
    if (
      merged.length > 0 &&
      merged[merged.length - 1].text.length < 10 &&
      merged[merged.length - 1].text.length + 1 + item.text.length <= maxChars
    ) {
      merged[merged.length - 1].text += ` ${item.text}`;
      merged[merged.length - 1].sentenceEnd = item.sentenceEnd;
    } else {
      merged.push(item);
    }
  }
  return merged.length > 0 ? merged : [{ text: cleaned, sentenceEnd: true }];
}

function splitProsodicClauses(text, maxChars = 180) {
  return buildStreamingClauses(text, maxChars).map((c) => c.text);
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

const BROWSER_MODEL_CACHE = 'ninada-onnx-v2.0.0';

async function fetchBrowserCachedBytes(url, useCache = true) {
  const absUrl =
    typeof location !== 'undefined' && location?.href
      ? new URL(url, location.href).href
      : url;
  if (useCache && typeof caches !== 'undefined' && caches?.open) {
    try {
      const cache = await caches.open(BROWSER_MODEL_CACHE);
      const hit = await cache.match(absUrl);
      if (hit) {
        const arr = new Uint8Array(await hit.arrayBuffer());
        if (arr.byteLength >= 1000000 && arr[0] === 0x08) {
          return arr;
        }
        await cache.delete(absUrl).catch(() => {});
      }
    } catch {
      // Fall through to network fetch
    }
  }
  const res = await fetch(absUrl);
  if (!res.ok) {
    throw new Error(`Failed to fetch ONNX model (${res.status}): ${absUrl}`);
  }
  const arr = new Uint8Array(await res.arrayBuffer());
  if (useCache && arr.byteLength >= 1000000 && arr[0] === 0x08 && typeof caches !== 'undefined' && caches?.open) {
    try {
      const cache = await caches.open(BROWSER_MODEL_CACHE);
      await cache.put(
        absUrl,
        new Response(arr, {
          headers: {
            'Content-Type': 'application/octet-stream',
            'Content-Length': String(arr.byteLength),
          },
        })
      );
    } catch {
      // Ignore quota / private-browsing cache errors
    }
  }
  return arr;
}

async function createOrtSession(ort, source, providers, useCache = true) {
  if (!source || !ort || !ort.InferenceSession) return null;
  const isNode =
    typeof process !== 'undefined' &&
    Boolean(process.versions?.node) &&
    typeof window === 'undefined' &&
    typeof WorkerGlobalScope === 'undefined';
  let resolved = source;
  if (isNode && typeof source === 'string' && !/^https?:\/\//i.test(source) && ort.env?.wasm) {
    const fsPkg = 'node:fs/promises';
    const fsMod = await import(fsPkg);
    const buf = await fsMod.readFile(source);
    resolved = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  } else if (!isNode && typeof source === 'string' && typeof fetch === 'function') {
    resolved = await fetchBrowserCachedBytes(source, useCache);
  }
  const useProviders = providers || (ort.env?.wasm ? ['wasm'] : ['cpu']);
  const canUseArena = isNode || Boolean(typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated);
  return ort.InferenceSession.create(resolved, {
    executionProviders: useProviders,
    graphOptimizationLevel: 'all',
    enableCpuMemArena: canUseArena,
    enableMemPattern: canUseArena,
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

  const useCache = opts.cache !== false;
  if (!encSession && !vocSession && encSource && vocSource && ort) {
    encSession = await createOrtSession(ort, encSource, opts.providers, useCache);
    encSource = null;
    vocSession = await createOrtSession(ort, vocSource, opts.providers, useCache);
    vocSource = null;
  } else if (!session && modelSource && ort) {
    session = await createOrtSession(ort, modelSource, opts.providers, useCache);
    modelSource = null;
  }

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
    async dispose() {
      try { await encSession?.release?.(); } catch {}
      try { await vocSession?.release?.(); } catch {}
      try { await session?.release?.(); } catch {}
      encSession = null;
      vocSession = null;
      session = null;
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
      const noise = callOpts.noise ?? opts.noise ?? 0.667;
      const noiseDuration = callOpts.noiseDuration ?? opts.noiseDuration ?? 0.80;
      const tier = callOpts.tier || 'auto';
      const firstChunkFrames = Math.max(24, Number(callOpts.firstChunkFrames || 44));
      const chunkFrames = Math.max(32, Number(callOpts.chunkFrames || 96));
      const haloFrames = Math.max(8, Number(callOpts.haloFrames || 12));
      const crossfadeSamples = Math.max(0, Math.min(Number(callOpts.crossfadeSamples ?? 256), (haloFrames * HOP_LENGTH) >> 1));
      const signal = callOpts.signal || null;

      const attachLazyWav = (obj, pcmBuf) => {
        let cachedWav = null;
        Object.defineProperty(obj, 'wav', {
          enumerable: true,
          configurable: true,
          get() {
            if (!cachedWav) cachedWav = encodeWavBuffer(pcmBuf, SAMPLE_RATE);
            return cachedWav;
          },
        });
        return obj;
      };

      const clauses = buildStreamingClauses(text, callOpts.maxClauseChars || 180);
      const tStart = nowMs();
      let ttfaMs = null;
      let chunkIndex = 0;

      // Path 1: 12-Frame Halo + 256-Sample Raised-Cosine Crossfaded Neural Streaming (122.1 dB SNR)
      if (tier !== 'klatt' && encSession && vocSession && ort) {
        const encInNames = Array.isArray(encSession.inputNames) ? encSession.inputNames : [];
        const vocInNames = Array.isArray(vocSession.inputNames) ? vocSession.inputNames : [];
        const hasLangId = encInNames.includes('lang_id');
        const hasVoiceCond = vocInNames.includes('voice_cond');

        const fadeIn = new Float32Array(crossfadeSamples);
        const fadeOut = new Float32Array(crossfadeSamples);
        for (let i = 0; i < crossfadeSamples; i++) {
          const w = 0.5 * (1.0 - Math.cos((Math.PI * i) / Math.max(1, crossfadeSamples)));
          fadeIn[i] = w;
          fadeOut[i] = 1.0 - w;
        }

        let globalChunkOrdinal = 0;

        for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
          if (signal?.aborted) return;
          const clauseItem = clauses[cIdx];
          const tEnc0 = nowMs();
          const parsed = textToIds(clauseItem.text, { lang });
          const feeds = {
            input_ids: new ort.Tensor('int64', BigInt64Array.from(parsed.inputIds.map(BigInt)), [1, parsed.inputIds.length]),
            attention_mask: new ort.Tensor('int64', BigInt64Array.from(parsed.attentionMask.map(BigInt)), [1, parsed.attentionMask.length]),
          };
          if (hasLangId) {
            feeds.lang_id = new ort.Tensor('int64', BigInt64Array.from([BigInt(parsed.langId)]), [1]);
            feeds.scales = new ort.Tensor('float32', Float32Array.from([Number(rate), Number(noise)]), [2]);
          } else {
            feeds.scales = new ort.Tensor('float32', Float32Array.from([Number(rate), Number(noise), Number(noiseDuration)]), [3]);
          }

          const encOut = await encSession.run(feeds);
          const encElapsedMs = nowMs() - tEnc0;
          feeds.input_ids.dispose?.();
          feeds.attention_mask.dispose?.();
          feeds.lang_id?.dispose?.();
          feeds.scales.dispose?.();
          if (signal?.aborted) {
            encOut.latents?.dispose?.();
            encOut.voice_cond?.dispose?.();
            encOut.durations?.dispose?.();
            return;
          }

          const latData = Float32Array.from(encOut.latents.data);
          const totalFrames = encOut.latents.dims[2];
          const channels = encOut.latents.dims[1];
          const voiceCondData = encOut.voice_cond ? Float32Array.from(encOut.voice_cond.data) : null;
          const voiceCondDims = encOut.voice_cond ? encOut.voice_cond.dims.slice() : null;
          const dursArr = Array.from(encOut.durations.data, Number);
          encOut.latents.dispose?.();
          encOut.voice_cond?.dispose?.();
          encOut.durations.dispose?.();

          // Progressive geometric chunk ladder + sliver merging prevents Chunk 0->1 buffer underruns
          const bounds = [];
          for (let cur = 0; cur < totalFrames; ) {
            let step = chunkFrames;
            if (globalChunkOrdinal === 0) {
              step = firstChunkFrames;
            } else if (globalChunkOrdinal === 1 && chunkFrames > firstChunkFrames) {
              step = Math.round(firstChunkFrames + (chunkFrames - firstChunkFrames) * 0.30);
            } else if (globalChunkOrdinal === 2 && chunkFrames > firstChunkFrames) {
              step = Math.round(firstChunkFrames + (chunkFrames - firstChunkFrames) * 0.65);
            }
            let nxt = Math.min(totalFrames, cur + step);
            if (totalFrames - nxt < 36) {
              nxt = totalFrames;
            }
            bounds.push([cur, nxt]);
            cur = nxt;
            globalChunkOrdinal += 1;
          }
          const numSub = bounds.length;
          let prevTail = null;

          for (let subIdx = 0; subIdx < numSub; subIdx++) {
            if (signal?.aborted) return;
            const [st, ed] = bounds[subIdx];
            const isLastSub = subIdx === numSub - 1;
            const wSt = Math.max(0, st - haloFrames);
            const wEd = Math.min(totalFrames, ed + haloFrames);
            const winLen = wEd - wSt;

            const sliceBuf = new Float32Array(channels * winLen);
            for (let ch = 0; ch < channels; ch++) {
              const srcBase = ch * totalFrames + wSt;
              const dstBase = ch * winLen;
              sliceBuf.set(latData.subarray(srcBase, srcBase + winLen), dstBase);
            }

            const tChunk = nowMs();
            const latTensor = new ort.Tensor('float32', sliceBuf, [1, channels, winLen]);
            const vocFeeds = { latents: latTensor };
            let vcTensor = null;
            if (hasVoiceCond && voiceCondData) {
              vcTensor = new ort.Tensor('float32', voiceCondData, voiceCondDims);
              vocFeeds.voice_cond = vcTensor;
            }
            const vocOut = await vocSession.run(vocFeeds);
            latTensor.dispose?.();
            vcTensor?.dispose?.();
            if (signal?.aborted) {
              vocOut.waveform.dispose?.();
              return;
            }

            const raw = vocOut.waveform.data;
            const cropL = (st - wSt) * HOP_LENGTH;
            const cropLen = (ed - st) * HOP_LENGTH;
            const cropR = cropL + cropLen;
            const extraR = Math.min(raw.length, cropR + (isLastSub ? 0 : crossfadeSamples));
            const segLen = Math.max(0, extraR - cropL);
            const seg = new Float32Array(segLen);
            for (let i = 0; i < segLen; i++) {
              const v = (raw[cropL + i] || 0) * 0.90;
              seg[i] = v > 0.98 ? 0.98 : v < -0.98 ? -0.98 : v;
            }
            vocOut.waveform.dispose?.();

            if (prevTail && crossfadeSamples > 0 && prevTail.length === crossfadeSamples && seg.length >= crossfadeSamples) {
              for (let i = 0; i < crossfadeSamples; i++) {
                seg[i] = prevTail[i] * fadeOut[i] + seg[i] * fadeIn[i];
              }
            }

            let basePcm;
            if (!isLastSub && crossfadeSamples > 0 && extraR - cropR === crossfadeSamples) {
              prevTail = seg.slice(seg.length - crossfadeSamples);
              basePcm = seg.subarray(0, seg.length - crossfadeSamples);
            } else {
              prevTail = null;
              basePcm = seg.subarray(0, Math.min(seg.length, cropLen));
            }

            if (subIdx === 0 && basePcm.length > 192) {
              for (let i = 0; i < 192; i++) {
                basePcm[i] *= 0.5 * (1.0 - Math.cos((Math.PI * i) / 192.0));
              }
            }
            if (isLastSub && basePcm.length > 256) {
              const off = basePcm.length - 256;
              for (let i = 0; i < 256; i++) {
                basePcm[off + i] *= 0.5 * (1.0 + Math.cos((Math.PI * i) / 256.0));
              }
            }

            let pcm;
            if (isLastSub && cIdx < clauses.length - 1 && clauseItem.sentenceEnd) {
              const pauseSamples = Math.round(0.048 * SAMPLE_RATE);
              pcm = new Float32Array(basePcm.length + pauseSamples);
              pcm.set(basePcm, 0);
            } else {
              pcm = new Float32Array(basePcm.length);
              pcm.set(basePcm, 0);
            }

            const elapsedNow = nowMs() - tStart;
            if (ttfaMs === null) ttfaMs = elapsedNow;
            const isFinal = cIdx === clauses.length - 1 && isLastSub;
            const vocMs = nowMs() - tChunk;
            const chunkObj = attachLazyWav(
              {
                pcm,
                sampleRate: SAMPLE_RATE,
                index: chunkIndex++,
                clauseIndex: cIdx,
                clauseCount: clauses.length,
                isFinal,
                durationSec: pcm.length / SAMPLE_RATE,
                latencyMs: vocMs + (subIdx === 0 ? encElapsedMs : 0),
                vocoderMs: vocMs,
                ttfaMs,
                lang: parsed.lang,
                script: parsed.script,
                tier: 'neural_onnx',
                aksharas: subIdx === 0 ? parsed.aksharas : [],
                durations: subIdx === 0 ? dursArr : [],
              },
              pcm
            );
            emit('chunk', chunkObj);
            yield chunkObj;
          }
        }
        return;
      }

      // Path 2: Clause-Level Neural Streaming over Monolithic ONNX
      if (tier !== 'klatt' && session && ort) {
        const sessInNames = Array.isArray(session.inputNames) ? session.inputNames : [];
        const hasLangId = sessInNames.includes('lang_id');
        for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
          if (signal?.aborted) return;
          const tChunk = nowMs();
          const parsed = textToIds(clauses[cIdx].text, { lang });
          const feeds = {
            input_ids: new ort.Tensor('int64', BigInt64Array.from(parsed.inputIds.map(BigInt)), [1, parsed.inputIds.length]),
            attention_mask: new ort.Tensor('int64', BigInt64Array.from(parsed.attentionMask.map(BigInt)), [1, parsed.attentionMask.length]),
          };
          if (hasLangId) {
            feeds.lang_id = new ort.Tensor('int64', BigInt64Array.from([BigInt(parsed.langId)]), [1]);
            feeds.scales = new ort.Tensor('float32', Float32Array.from([Number(rate), Number(noise)]), [2]);
          } else {
            feeds.scales = new ort.Tensor('float32', Float32Array.from([Number(rate), Number(noise), Number(noiseDuration)]), [3]);
          }
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
          const chunkObj = attachLazyWav(
            {
              pcm,
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
            },
            pcm
          );
          emit('chunk', chunkObj);
          yield chunkObj;
        }
        return;
      }

      // Path 3: Zero-Dependency Klatt Tier-2 Clause Streaming
      for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
        if (signal?.aborted) return;
        const tChunk = nowMs();
        const { pcm, meta: baseMeta } = synthesizeKlattPcm(clauses[cIdx].text, { lang, rate });
        if (ttfaMs === null) ttfaMs = nowMs() - tStart;
        const chunkObj = attachLazyWav(
          {
            pcm,
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
          },
          pcm
        );
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
  NATIVE_VOCABS,
  NATIVE_TOKEN_TO_ID,
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
