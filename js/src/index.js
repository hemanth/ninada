/**
 * ninada v4.0 — Streaming 24 kHz Native Per-Language Studio VITS Neural TTS
 * (6 Dedicated Native 24,000 Hz Studio VITS Specialists + Zero-Dep Klatt Fallback for Browser WASM SIMD & Node.js)
 *
 * Streaming quick start:
 *   import ninada from 'ninada';
 *   const voice = await ninada('kannada', {
 *     models: { kn: './models/kn.onnx', sa: './models/sa.onnx' },
 *   });
 *   for await (const chunk of voice.stream('ನಮಸ್ಕಾರ, ಇಂದು ಬೆಂಗಳೂರಿನಲ್ಲಿ ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.')) {
 *     // chunk.pcm (Float32Array @ 24kHz), chunk.ttfaMs
 *   }
 */
import {
  VOCAB_TOKENS,
  TOKEN_TO_ID,
  IPA_VOCAB,
  IPA_ID_TO_TOKEN,
  RETROFLEX_LATERAL_ID,
  BLANK_ID,
  LANGUAGE_IDS,
  NATIVE_VOCABS,
  NATIVE_TOKEN_TO_ID,
  detectScriptAndLang,
  iastToDevanagari,
  brahmicToDevanagari,
  applyPhonologicalRules,
  segmentAksharas,
  toNativeModelText,
  textToIpa,
  ipaToIds,
  textToIds,
} from './g2p.js';
import {
  STUDIO_HOP_LENGTH,
  STUDIO_VOCABS,
  cleanStudioText,
  studioTextToIds,
} from './studio_g2p.js';
import { SAMPLE_RATE, synthesizeKlattPcm, encodeWavBuffer } from './klatt.js';

const HOP_LENGTH = 600;
const VOICE_BANK_FLOATS = 510 * 256;
const VOICE_BANK_BYTES = VOICE_BANK_FLOATS * 4;

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

function buildStreamingClauses(text, maxChars = 180, fastFirstClause = false) {
  const cleaned = String(text || '').replace(/\s+/g, ' ').trim();
  if (!cleaned) return [];
  const effMax = maxChars;
  const rawParts = cleaned.split(/(?<=\|\||[\u0964\u0965|.?!;\n])\s+/g);
  const rawClauses = [];
  for (const part of rawParts) {
    const p = (part || '').trim();
    if (!p) continue;
    if (/^[\u0964\u0965|.?!;,\s-]+$/.test(p)) {
      if (rawClauses.length > 0) rawClauses[rawClauses.length - 1].text += ` ${p}`;
      continue;
    }
    if (p.length <= effMax) {
      rawClauses.push({ text: p, sentenceEnd: true });
      continue;
    }
    const words = p.split(' ');
    let buf = '';
    for (const w of words) {
      if (!buf) buf = w;
      else if (buf.length + 1 + w.length <= effMax) buf = `${buf} ${w}`;
      else {
        const bufWithComma = /[,;—–:|\u0964\u0965.?!]$/.test(buf) ? buf : `${buf},`;
        rawClauses.push({ text: bufWithComma, sentenceEnd: false });
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
      merged[merged.length - 1].text.length + 1 + item.text.length <= effMax
    ) {
      merged[merged.length - 1].text += ` ${item.text}`;
      merged[merged.length - 1].sentenceEnd = item.sentenceEnd;
    } else {
      merged.push(item);
    }
  }
  const result = merged.length > 0 ? merged : [{ text: cleaned, sentenceEnd: true }];

  if (fastFirstClause && result.length > 0 && result[0].text.length > 34) {
    const first = result[0];
    const subParts = first.text.split(/(?<=[,;—–:])\s+/g).filter((s) => s && s.trim());
    if (subParts.length >= 2 && subParts[0].length >= 12 && subParts[0].length <= 54) {
      const rest = subParts.slice(1).join(' ');
      result.splice(
        0,
        1,
        { text: subParts[0].trim(), sentenceEnd: false },
        { text: rest.trim(), sentenceEnd: first.sentenceEnd }
      );
    } else {
      const words = first.text.split(' ').filter(Boolean);
      if (words.length >= 2) {
        let splitIdx = words[0].length >= 16 ? 1 : Math.max(2, Math.floor(words.length / 2));
        for (let i = 2; i < words.length - 1; i++) {
          if (/^(ಮತ್ತು|और|तथा|மற்றும்|మరియు|കൂടാതെ|च)$/.test(words[i])) {
            splitIdx = i;
            break;
          }
        }
        const foot = words.slice(0, splitIdx).join(' ');
        const rest = words.slice(splitIdx).join(' ');
        const restLetters = rest.replace(/[\u0964\u0965|.?!;,\s-]+/g, '');
        if (foot.length >= 14 && restLetters.length >= 8) {
          const footWithComma = /[,;—–:|\u0964\u0965.?!]$/.test(foot) ? foot : `${foot},`;
          result.splice(
            0,
            1,
            { text: footWithComma, sentenceEnd: false },
            { text: rest, sentenceEnd: first.sentenceEnd }
          );
        }
      }
    }
  }
  return result;
}

function splitProsodicClauses(text, maxChars = 180) {
  return buildStreamingClauses(text, maxChars, false).map((c) => c.text);
}

function generateRoomTone(numSamples = 1920, seed = 1337) {
  let s = seed >>> 0;
  const randU = () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return (s + 1.0) / 4294967296.0;
  };
  const nTot = Math.max(1, Math.floor(numSamples));
  const buf = new Float32Array(nTot);
  let y = 0.0;
  const alpha = 0.28;
  const raw = new Float32Array(nTot);
  let sum = 0.0;
  for (let i = 0; i < nTot; i++) {
    const u1 = randU();
    const u2 = randU();
    const n = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * 0.0035;
    y = y + alpha * (n - y);
    raw[i] = y;
    sum += y;
  }
  const mean = sum / nTot;
  const fadeLen = Math.min(192, Math.floor(nTot / 4));
  for (let i = 0; i < nTot; i++) {
    let env = 1.0;
    if (fadeLen > 0) {
      if (i < fadeLen) {
        env = 0.5 * (1.0 - Math.cos((Math.PI * i) / fadeLen));
      } else if (i >= nTot - fadeLen) {
        env = 0.5 * (1.0 + Math.cos((Math.PI * (i - (nTot - fadeLen))) / fadeLen));
      }
    }
    buf[i] = (raw[i] - mean) * env;
  }
  return buf;
}

function hasPronounceableContent(parsed) {
  const ak = parsed?.aksharas;
  if (!Array.isArray(ak) || ak.length === 0) return false;
  for (let i = 0; i < ak.length; i++) {
    const core = String(ak[i]?.akshara || '').replace(/\u094d/g, '').trim();
    if (core.length > 0) return true;
  }
  return false;
}

function isNodeEnv() {
  return (
    typeof process !== 'undefined' &&
    Boolean(process.versions?.node) &&
    typeof window === 'undefined' &&
    typeof WorkerGlobalScope === 'undefined'
  );
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

const BROWSER_MODEL_CACHE = 'ninada-onnx-v4.0.0';

async function fetchBrowserCachedBytes(url, useCache = true, validator = null) {
  const absUrl =
    typeof location !== 'undefined' && location?.href
      ? new URL(url, location.href).href
      : url;
  const isOk =
    validator ||
    ((arr) => arr.byteLength >= 1000000 && arr[0] === 0x08);

  if (useCache && typeof caches !== 'undefined' && caches?.open) {
    try {
      const cache = await caches.open(BROWSER_MODEL_CACHE);
      const hit = await cache.match(absUrl);
      if (hit) {
        const arr = new Uint8Array(await hit.arrayBuffer());
        if (isOk(arr)) {
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
    throw new Error(`Failed to fetch binary asset (${res.status}): ${absUrl}`);
  }
  const arr = new Uint8Array(await res.arrayBuffer());
  if (useCache && isOk(arr) && typeof caches !== 'undefined' && caches?.open) {
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

function toFloat32VoiceBank(raw) {
  if (!raw) return null;
  if (raw instanceof Float32Array) {
    return raw.length === VOICE_BANK_FLOATS ? raw : null;
  }
  if (raw instanceof ArrayBuffer) {
    return raw.byteLength === VOICE_BANK_BYTES ? new Float32Array(raw) : null;
  }
  if (ArrayBuffer.isView(raw)) {
    if (raw.byteLength !== VOICE_BANK_BYTES) return null;
    if (raw.byteOffset % 4 === 0) {
      return new Float32Array(raw.buffer, raw.byteOffset, VOICE_BANK_FLOATS);
    }
    const copy = new Uint8Array(VOICE_BANK_BYTES);
    copy.set(new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength));
    return new Float32Array(copy.buffer);
  }
  return null;
}

async function loadVoiceBankSource(source, useCache = true) {
  if (!source) return null;
  const direct = toFloat32VoiceBank(source);
  if (direct) return direct;
  if (typeof source === 'string') {
    if (isNodeEnv() && !/^https?:\/\//i.test(source)) {
      try {
        const fsMod = await import('node:fs/promises');
        const buf = await fsMod.readFile(source);
        return toFloat32VoiceBank(buf);
      } catch {
        return null;
      }
    }
    if (typeof fetch === 'function') {
      const bytes = await fetchBrowserCachedBytes(
        source,
        useCache,
        (arr) => arr.byteLength === VOICE_BANK_BYTES
      );
      return toFloat32VoiceBank(bytes);
    }
  }
  return null;
}

async function createOrtSession(ort, source, providers, useCache = true) {
  if (!source || !ort || !ort.InferenceSession) return null;
  const isNode = isNodeEnv();
  let resolved = source;
  if (isNode && typeof source === 'string' && !/^https?:\/\//i.test(source) && ort.env?.wasm) {
    const fsMod = await import('node:fs/promises');
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

function isV4StudioSession(sess) {
  if (!sess) return false;
  const names = Array.isArray(sess.inputNames) ? sess.inputNames : [];
  return names.length === 2 && names.includes('input_ids') && names.includes('scales');
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

  const modelBasePath = typeof modelSource === 'string' && !/^https?:\/\//i.test(modelSource) ? modelSource : null;

  let session = opts.session || null;
  let encSession = opts.encoderSession || null;
  let vocSession = opts.vocoderSession || null;
  const getSessionHook = typeof opts.getSession === 'function' ? opts.getSession : null;

  const sessionsMap = new Map();
  if (opts.sessions) {
    const entries = opts.sessions instanceof Map ? opts.sessions.entries() : Object.entries(opts.sessions);
    for (const [k, s] of entries) {
      if (s) sessionsMap.set(String(k).trim().toLowerCase(), s);
    }
  }

  const modelSourcesMap = new Map();
  if (opts.models && typeof opts.models === 'object') {
    for (const [k, v] of Object.entries(opts.models)) {
      if (v) modelSourcesMap.set(String(k).trim().toLowerCase(), v);
    }
  }

  const hasNeuralConfig = Boolean(
    session ||
    encSession ||
    getSessionHook ||
    sessionsMap.size > 0 ||
    modelSourcesMap.size > 0 ||
    modelSource ||
    (encSource && vocSource)
  );
  const ort = hasNeuralConfig ? await resolveOrt(opts.ort) : null;

  const useCache = opts.cache !== false;
  if (!encSession && !vocSession && encSource && vocSource && ort) {
    encSession = await createOrtSession(ort, encSource, opts.providers, useCache);
    encSource = null;
    vocSession = await createOrtSession(ort, vocSource, opts.providers, useCache);
    vocSource = null;
  } else if (!session && modelSource && ort) {
    session = await createOrtSession(ort, modelSource, opts.providers, useCache);
    modelSource = null;
  } else if (!session && !getSessionHook && modelSourcesMap.has(defaultLang) && ort) {
    session = await createOrtSession(ort, modelSourcesMap.get(defaultLang), opts.providers, useCache);
    if (session) sessionsMap.set(defaultLang, session);
  }

  modelSource = null;
  encSource = null;
  vocSource = null;

  async function resolveV4Session(lang, voiceOverride = null) {
    const lc = resolveLang(lang);
    const vNorm = typeof voiceOverride === 'string' ? voiceOverride.trim().toLowerCase() : '';
    const femKey = `${lc}_female`;
    const targetKey = vNorm === 'female' ? femKey : lc;

    if (getSessionHook) {
      const hooked = await getSessionHook(lc, vNorm || 'male');
      if (isV4StudioSession(hooked)) return hooked;
    }
    if (sessionsMap.has(targetKey)) {
      const hit = sessionsMap.get(targetKey);
      if (isV4StudioSession(hit)) return hit;
    }
    if (modelSourcesMap.has(targetKey) && ort) {
      try {
        const loaded = await createOrtSession(ort, modelSourcesMap.get(targetKey), opts.providers, useCache);
        if (isV4StudioSession(loaded)) {
          sessionsMap.set(targetKey, loaded);
          return loaded;
        }
      } catch {}
    }
    if (vNorm === 'female') {
      if (sessionsMap.has(lc) && isV4StudioSession(sessionsMap.get(lc))) return sessionsMap.get(lc);
      if (modelSourcesMap.has(lc) && ort) {
        try {
          const loaded = await createOrtSession(ort, modelSourcesMap.get(lc), opts.providers, useCache);
          if (isV4StudioSession(loaded)) {
            sessionsMap.set(lc, loaded);
            return loaded;
          }
        } catch {}
      }
    }
    if (isV4StudioSession(session)) return session;
    return null;
  }

  // Legacy voice banks: Map<lang, Float32Array(510 * 256)>
  const voiceBanks = new Map();
  const voiceSources = new Map();
  if (opts.voices && typeof opts.voices === 'object') {
    for (const [k, v] of Object.entries(opts.voices)) {
      const lc = resolveLang(k);
      const direct = toFloat32VoiceBank(v);
      if (direct) voiceBanks.set(lc, direct);
      else if (v) voiceSources.set(lc, v);
    }
  }
  const initialVoice = opts.voiceBank || (opts.voice && typeof opts.voice !== 'string' ? opts.voice : null) ||
    (typeof opts.voice === 'string' && /\.bin$/i.test(opts.voice) ? opts.voice : null);
  if (initialVoice) {
    const loaded = await loadVoiceBankSource(initialVoice, useCache);
    if (loaded) voiceBanks.set(defaultLang, loaded);
  } else if (voiceSources.has(defaultLang)) {
    const loaded = await loadVoiceBankSource(voiceSources.get(defaultLang), useCache);
    if (loaded) voiceBanks.set(defaultLang, loaded);
  }

  async function resolveVoiceStyle(lang, numIpaTokens, voiceOverride = null) {
    const vNorm = typeof voiceOverride === 'string' ? voiceOverride.trim().toLowerCase() : '';
    const key = vNorm === 'female' ? 'female' : resolveLang(lang);
    let bank = voiceBanks.get(key);
    if (!bank && voiceSources.has(key)) {
      bank = await loadVoiceBankSource(voiceSources.get(key), useCache);
      if (bank) voiceBanks.set(key, bank);
    }
    if (!bank && modelBasePath && isNodeEnv()) {
      try {
        const pathMod = await import('node:path');
        const dir = pathMod.dirname(modelBasePath);
        for (const cand of [
          pathMod.join(dir, 'voices', `${key}.bin`),
          pathMod.join(dir, '..', 'voices', `${key}.bin`),
        ]) {
          bank = await loadVoiceBankSource(cand, false);
          if (bank) {
            voiceBanks.set(key, bank);
            break;
          }
        }
      } catch {}
    }
    if (!bank && key === 'female') {
      return resolveVoiceStyle(lang, numIpaTokens, null);
    }
    if (!bank && voiceBanks.has(defaultLang)) {
      bank = voiceBanks.get(defaultLang);
    }
    if (!bank) {
      return new Float32Array(256);
    }
    const row = Math.min(509, Math.max(0, (numIpaTokens | 0) - 2));
    const offset = row * 256;
    return bank.subarray(offset, offset + 256);
  }

  function isUnified24kSession() {
    if (!session) return false;
    const names = Array.isArray(session.inputNames) ? session.inputNames : [];
    return names.includes('style') && names.includes('speed');
  }

  function hasV4Neural() {
    return Boolean(
      getSessionHook ||
      isV4StudioSession(session) ||
      sessionsMap.size > 0 ||
      modelSourcesMap.size > 0
    );
  }

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
      return Boolean(hasV4Neural() || (encSession && vocSession) || session);
    },
    get isStreamingNeural() {
      return Boolean(hasV4Neural() || isUnified24kSession() || (encSession && vocSession));
    },
    setVoiceBank(lang, bank) {
      const kNorm = typeof lang === 'string' && lang.trim().toLowerCase() === 'female' ? 'female' : resolveLang(lang);
      const f32 = toFloat32VoiceBank(bank);
      if (f32) {
        voiceBanks.set(kNorm, f32);
      }
      return this;
    },
    async dispose() {
      try { await encSession?.release?.(); } catch {}
      try { await vocSession?.release?.(); } catch {}
      try { await session?.release?.(); } catch {}
      for (const s of sessionsMap.values()) {
        try { await s?.release?.(); } catch {}
      }
      encSession = null;
      vocSession = null;
      session = null;
      sessionsMap.clear();
      voiceBanks.clear();
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
      const effRate = Math.max(0.45, Math.min(2.5, Number(callOpts.rate ?? opts.rate ?? 1.0) || 1.0));
      const tier = callOpts.tier || 'auto';
      const firstChunkFrames = Math.max(24, Number(callOpts.firstChunkFrames || 44));
      const chunkFrames = Math.max(24, Number(callOpts.chunkFrames || 72));
      const haloFrames = Math.max(8, Number(callOpts.haloFrames || 12));
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

      const v4InitialSess = tier !== 'klatt' && ort ? await resolveV4Session(lang, callOpts.voice || null) : null;
      const isV4 = Boolean(v4InitialSess && ort);
      const is24k = !isV4 && tier !== 'klatt' && isUnified24kSession() && Boolean(ort);
      const fastFirst = is24k && callOpts.fastFirstClause !== false;
      const clauses = buildStreamingClauses(text, callOpts.maxClauseChars || (isV4 || is24k ? 140 : 180), fastFirst);
      const tStart = nowMs();
      let ttfaMs = null;
      let chunkIndex = 0;

      let dcXPrev = 0.0;
      let dcYPrev = 0.0;
      const dcR = 0.9965;

      // Path 0: Primary v4.0 Per-Language Native 24,000 Hz Studio VITS Specialist (`input_ids`, `scales`)
      if (isV4) {
        if (clauses.length === 0) {
          const silent = generateRoomTone(Math.round(0.10 * SAMPLE_RATE), 1337);
          ttfaMs = nowMs() - tStart;
          const chunkObj = attachLazyWav(
            {
              pcm: silent,
              sampleRate: SAMPLE_RATE,
              index: 0,
              clauseIndex: 0,
              clauseCount: 1,
              isFinal: true,
              durationSec: silent.length / SAMPLE_RATE,
              latencyMs: ttfaMs,
              ttfaMs,
              lang,
              script: 'devanagari',
              tier: 'neural_onnx',
              aksharas: [],
              durations: [],
            },
            silent
          );
          emit('chunk', chunkObj);
          yield chunkObj;
          return;
        }

        const effNoise = Math.max(0.0, Math.min(1.2, Number(callOpts.noise ?? opts.noise ?? 0.60)));
        const effNoiseDp = Math.max(
          0.0,
          Math.min(
            1.2,
            Number(callOpts.noiseDuration ?? callOpts.noise_duration ?? opts.noiseDuration ?? opts.noise_duration ?? 0.0)
          )
        );
        const chunkSamples = chunkFrames * HOP_LENGTH;
        const firstChunkSamples = firstChunkFrames * HOP_LENGTH;
        const minTailSamples = 16 * STUDIO_HOP_LENGTH;

        for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
          if (signal?.aborted) return;
          const clauseItem = clauses[cIdx];
          const tClause = nowMs();
          const parsed = textToIds(clauseItem.text, { lang });

          if (!hasPronounceableContent(parsed)) {
            const silent = generateRoomTone(Math.round(0.10 * SAMPLE_RATE), 1337 + cIdx);
            if (ttfaMs === null) ttfaMs = nowMs() - tStart;
            const chunkObj = attachLazyWav(
              {
                pcm: silent,
                sampleRate: SAMPLE_RATE,
                index: chunkIndex++,
                clauseIndex: cIdx,
                clauseCount: clauses.length,
                isFinal: cIdx === clauses.length - 1,
                durationSec: silent.length / SAMPLE_RATE,
                latencyMs: nowMs() - tClause,
                ttfaMs,
                lang: parsed.lang,
                script: parsed.script,
                tier: 'neural_onnx',
                aksharas: parsed.aksharas,
                durations: [],
              },
              silent
            );
            emit('chunk', chunkObj);
            yield chunkObj;
            continue;
          }

          const { studioIds, lang: resolvedLc } = studioTextToIds(clauseItem.text, parsed.lang);
          const activeSess = (await resolveV4Session(resolvedLc, callOpts.voice || null)) || v4InitialSess;
          const idTensor = new ort.Tensor('int64', BigInt64Array.from(studioIds.map(BigInt)), [1, studioIds.length]);
          const scalesTensor = new ort.Tensor('float32', Float32Array.from([1.0 / effRate, effNoise, effNoiseDp]), [3]);

          const out = await activeSess.run({
            input_ids: idTensor,
            scales: scalesTensor,
          });
          const clauseElapsedMs = nowMs() - tClause;
          idTensor.dispose?.();
          scalesTensor.dispose?.();

          if (signal?.aborted) {
            out.waveform?.dispose?.();
            out.durations?.dispose?.();
            return;
          }

          const rawWav = out.waveform.data;
          const dursArr = out.durations ? Array.from(out.durations.data, Number) : [];
          const hasPause = cIdx < clauses.length - 1;
          const pauseSec = hasPause ? (clauseItem.sentenceEnd ? 0.052 : 0.020) : 0.480;
          const pauseSamples = Math.round((pauseSec / effRate) * SAMPLE_RATE);
          const clausePcm = new Float32Array(rawWav.length + pauseSamples);

          let dcMeanSum = 0.0;
          for (let i = 0; i < rawWav.length; i++) {
            const x = rawWav[i] || 0;
            const y = x - dcXPrev + dcR * dcYPrev;
            dcXPrev = x;
            dcYPrev = y;
            clausePcm[i] = y;
            dcMeanSum += y;
          }
          if (Math.abs(dcYPrev) < 1e-15) dcYPrev = 0.0;
          out.waveform?.dispose?.();
          out.durations?.dispose?.();

          const dcMean = rawWav.length > 0 ? dcMeanSum / rawWav.length : 0.0;
          let peak = 1e-5;
          for (let i = 0; i < rawWav.length; i++) {
            const v = clausePcm[i] - dcMean;
            clausePcm[i] = v;
            const a = Math.abs(v);
            if (a > peak) peak = a;
          }

          const gain = peak > 0.96 ? 0.92 / peak : 1.0;
          if (gain !== 1.0) {
            for (let i = 0; i < rawWav.length; i++) clausePcm[i] *= gain;
          }
          if (rawWav.length > 192) {
            for (let i = 0; i < 192; i++) {
              clausePcm[i] *= 0.5 * (1.0 - Math.cos((Math.PI * i) / 192.0));
            }
          }
          if (rawWav.length > 256) {
            const off = rawWav.length - 256;
            for (let i = 0; i < 256; i++) {
              clausePcm[off + i] *= 0.5 * (1.0 + Math.cos((Math.PI * i) / 256.0));
            }
          }

          const totalS = clausePcm.length;
          const bounds = [];
          for (let cur = 0; cur < totalS; ) {
            const step = chunkIndex === 0 && bounds.length === 0 ? Math.min(chunkSamples, firstChunkSamples) : chunkSamples;
            let nxt = Math.min(totalS, cur + step);
            if (totalS - nxt < minTailSamples) nxt = totalS;
            bounds.push([cur, nxt]);
            cur = nxt;
          }
          if (bounds.length === 0) bounds.push([0, totalS]);

          for (let subIdx = 0; subIdx < bounds.length; subIdx++) {
            if (signal?.aborted) return;
            const [stS, edS] = bounds[subIdx];
            const pcm = clausePcm.slice(stS, edS);
            const isLastSub = subIdx === bounds.length - 1;
            if (ttfaMs === null) ttfaMs = nowMs() - tStart;
            const chunkObj = attachLazyWav(
              {
                pcm,
                sampleRate: SAMPLE_RATE,
                index: chunkIndex++,
                clauseIndex: cIdx,
                clauseCount: clauses.length,
                isFinal: cIdx === clauses.length - 1 && isLastSub,
                durationSec: pcm.length / SAMPLE_RATE,
                clauseDurationSec: totalS / SAMPLE_RATE,
                latencyMs: subIdx === 0 ? clauseElapsedMs : 0.05,
                vocoderMs: subIdx === 0 ? clauseElapsedMs : 0.05,
                ttfaMs,
                lang: resolvedLc,
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

      const noise = callOpts.noise ?? opts.noise ?? 0.667;
      const noiseDuration = callOpts.noiseDuration ?? callOpts.noise_duration ?? opts.noiseDuration ?? opts.noise_duration ?? 0.80;

      // Path 1: Legacy Unified 24 kHz Studio Neural Streaming
      if (is24k) {
        if (clauses.length === 0) {
          const silent = generateRoomTone(Math.round(0.10 * SAMPLE_RATE), 1337);
          ttfaMs = nowMs() - tStart;
          const chunkObj = attachLazyWav(
            {
              pcm: silent,
              sampleRate: SAMPLE_RATE,
              index: 0,
              clauseIndex: 0,
              clauseCount: 1,
              isFinal: true,
              durationSec: silent.length / SAMPLE_RATE,
              latencyMs: ttfaMs,
              ttfaMs,
              lang,
              script: 'devanagari',
              tier: 'neural_onnx',
              aksharas: [],
              durations: [],
            },
            silent
          );
          emit('chunk', chunkObj);
          yield chunkObj;
          return;
        }

        const chunkSamples = chunkFrames * HOP_LENGTH;
        const firstChunkSamples = firstChunkFrames * HOP_LENGTH;
        const minTailSamples = 16 * HOP_LENGTH;

        for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
          if (signal?.aborted) return;
          const clauseItem = clauses[cIdx];
          const tClause = nowMs();
          const parsed = textToIds(clauseItem.text, { lang });

          if (!hasPronounceableContent(parsed)) {
            const silent = generateRoomTone(Math.round(0.10 * SAMPLE_RATE), 1337 + cIdx);
            if (ttfaMs === null) ttfaMs = nowMs() - tStart;
            const chunkObj = attachLazyWav(
              {
                pcm: silent,
                sampleRate: SAMPLE_RATE,
                index: chunkIndex++,
                clauseIndex: cIdx,
                clauseCount: clauses.length,
                isFinal: cIdx === clauses.length - 1,
                durationSec: silent.length / SAMPLE_RATE,
                latencyMs: nowMs() - tClause,
                ttfaMs,
                lang: parsed.lang,
                script: parsed.script,
                tier: 'neural_onnx',
                aksharas: parsed.aksharas,
                durations: [],
              },
              silent
            );
            emit('chunk', chunkObj);
            yield chunkObj;
            continue;
          }

          const ipaIds = parsed.ipaIds;
          const styleSlice = await resolveVoiceStyle(parsed.lang, ipaIds.length, callOpts.voice || null);
          const idTensor = new ort.Tensor('int64', BigInt64Array.from(ipaIds.map(BigInt)), [1, ipaIds.length]);
          const styleTensor = new ort.Tensor('float32', new Float32Array(styleSlice), [1, 256]);
          const speedTensor = new ort.Tensor('float32', Float32Array.from([effRate]), [1]);

          const out = await session.run({
            input_ids: idTensor,
            style: styleTensor,
            speed: speedTensor,
          });
          const clauseElapsedMs = nowMs() - tClause;
          idTensor.dispose?.();
          styleTensor.dispose?.();
          speedTensor.dispose?.();

          if (signal?.aborted) {
            out.waveform?.dispose?.();
            out.durations?.dispose?.();
            return;
          }

          const rawWav = out.waveform.data;
          const dursArr = out.durations ? Array.from(out.durations.data, Number) : [];
          const hasPause = cIdx < clauses.length - 1;
          const pauseSamples = hasPause ? Math.round((clauseItem.sentenceEnd ? 0.048 : 0.018) * SAMPLE_RATE) : 0;
          const clausePcm = new Float32Array(rawWav.length + pauseSamples);

          let peak = 1e-5;
          for (let i = 0; i < rawWav.length; i++) {
            const x = rawWav[i] || 0;
            const y = x - dcXPrev + dcR * dcYPrev;
            dcXPrev = x;
            dcYPrev = y;
            clausePcm[i] = y;
            const a = Math.abs(y);
            if (a > peak) peak = a;
          }
          if (Math.abs(dcYPrev) < 1e-15) dcYPrev = 0.0;
          out.waveform?.dispose?.();
          out.durations?.dispose?.();

          const gain = peak > 0.96 ? 0.92 / peak : 1.0;
          if (gain !== 1.0) {
            for (let i = 0; i < rawWav.length; i++) clausePcm[i] *= gain;
          }
          if (rawWav.length > 192) {
            for (let i = 0; i < 192; i++) {
              clausePcm[i] *= 0.5 * (1.0 - Math.cos((Math.PI * i) / 192.0));
            }
          }
          if (rawWav.length > 256) {
            const off = rawWav.length - 256;
            for (let i = 0; i < 256; i++) {
              clausePcm[off + i] *= 0.5 * (1.0 + Math.cos((Math.PI * i) / 256.0));
            }
          }

          const totalS = clausePcm.length;
          const bounds = [];
          for (let cur = 0; cur < totalS; ) {
            const step = chunkIndex === 0 && bounds.length === 0 ? Math.min(chunkSamples, firstChunkSamples) : chunkSamples;
            let nxt = Math.min(totalS, cur + step);
            if (totalS - nxt < minTailSamples) nxt = totalS;
            bounds.push([cur, nxt]);
            cur = nxt;
          }
          if (bounds.length === 0) bounds.push([0, totalS]);

          for (let subIdx = 0; subIdx < bounds.length; subIdx++) {
            if (signal?.aborted) return;
            const [stS, edS] = bounds[subIdx];
            const pcm = clausePcm.slice(stS, edS);
            const isLastSub = subIdx === bounds.length - 1;
            if (ttfaMs === null) ttfaMs = nowMs() - tStart;
            const chunkObj = attachLazyWav(
              {
                pcm,
                sampleRate: SAMPLE_RATE,
                index: chunkIndex++,
                clauseIndex: cIdx,
                clauseCount: clauses.length,
                isFinal: cIdx === clauses.length - 1 && isLastSub,
                durationSec: pcm.length / SAMPLE_RATE,
                clauseDurationSec: totalS / SAMPLE_RATE,
                latencyMs: subIdx === 0 ? clauseElapsedMs : 0.05,
                vocoderMs: subIdx === 0 ? clauseElapsedMs : 0.05,
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

      // Path 2: Legacy 2-stage Encoder + Vocoder Streaming (if explicitly configured)
      if (tier !== 'klatt' && encSession && vocSession && ort) {
        const legacyHop = 256;
        const crossfadeSamples = Math.max(0, Math.min(Number(callOpts.crossfadeSamples ?? 256), (haloFrames * legacyHop) >> 1));
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
            feeds.scales = new ort.Tensor('float32', Float32Array.from([effRate, Number(noise)]), [2]);
          } else {
            feeds.scales = new ort.Tensor('float32', Float32Array.from([effRate, Number(noise), Number(noiseDuration)]), [3]);
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

          const bounds = [];
          for (let cur = 0; cur < totalFrames; ) {
            const step = globalChunkOrdinal === 0 ? firstChunkFrames : chunkFrames;
            let nxt = Math.min(totalFrames, cur + step);
            if (totalFrames - nxt < 36) nxt = totalFrames;
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
            const cropL = (st - wSt) * legacyHop;
            const cropLen = (ed - st) * legacyHop;
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

            const pcm = new Float32Array(basePcm.length);
            for (let idx = 0; idx < basePcm.length; idx++) {
              const x = basePcm[idx];
              const y = x - dcXPrev + dcR * dcYPrev;
              dcXPrev = x;
              dcYPrev = y;
              pcm[idx] = y;
            }
            if (Math.abs(dcYPrev) < 1e-15) dcYPrev = 0.0;

            if (ttfaMs === null) ttfaMs = nowMs() - tStart;
            const vocMs = nowMs() - tChunk;
            const chunkObj = attachLazyWav(
              {
                pcm,
                sampleRate: SAMPLE_RATE,
                index: chunkIndex++,
                clauseIndex: cIdx,
                clauseCount: clauses.length,
                isFinal: cIdx === clauses.length - 1 && isLastSub,
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

      // Path 3: Legacy Monolithic VITS ONNX with attention_mask
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
            feeds.scales = new ort.Tensor('float32', Float32Array.from([effRate, Number(noise)]), [2]);
          } else {
            feeds.scales = new ort.Tensor('float32', Float32Array.from([effRate, Number(noise), Number(noiseDuration)]), [3]);
          }
          const out = await session.run(feeds);
          if (signal?.aborted) return;
          const raw = out.waveform.data;
          const pcm = new Float32Array(raw.length);
          for (let idx = 0; idx < raw.length; idx++) {
            const x = raw[idx];
            const y = x - dcXPrev + dcR * dcYPrev;
            dcXPrev = x;
            dcYPrev = y;
            pcm[idx] = y;
          }
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

      // Path 4: Zero-Dependency Klatt Tier-2 Clause Streaming
      const klattClauses = clauses.length > 0 ? clauses : [{ text: '', sentenceEnd: true }];
      for (let cIdx = 0; cIdx < klattClauses.length; cIdx++) {
        if (signal?.aborted) return;
        const tChunk = nowMs();
        const { pcm: rawPcm, meta: baseMeta } = synthesizeKlattPcm(klattClauses[cIdx].text, { lang, rate: effRate });
        const pcm = new Float32Array(rawPcm.length);
        for (let idx = 0; idx < pcm.length; idx++) {
          const x = rawPcm[idx];
          const y = x - dcXPrev + dcR * dcYPrev;
          dcXPrev = x;
          dcYPrev = y;
          pcm[idx] = y;
        }
        if (Math.abs(dcYPrev) < 1e-15) dcYPrev = 0.0;
        if (ttfaMs === null) ttfaMs = nowMs() - tStart;
        const chunkObj = attachLazyWav(
          {
            pcm,
            sampleRate: SAMPLE_RATE,
            index: chunkIndex++,
            clauseIndex: cIdx,
            clauseCount: klattClauses.length,
            isFinal: cIdx === klattClauses.length - 1,
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

      for await (const ch of this.stream(text, { ...callOpts, fastFirstClause: false })) {
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

      let dcSum = 0.0;
      for (let idx = 0; idx < pcm.length; idx++) dcSum += pcm[idx];
      const mean = dcSum / pcm.length;
      for (let idx = 0; idx < pcm.length; idx++) pcm[idx] -= mean;

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
  STUDIO_HOP_LENGTH,
  STUDIO_VOCABS,
  VOCAB_TOKENS,
  TOKEN_TO_ID,
  IPA_VOCAB,
  IPA_ID_TO_TOKEN,
  RETROFLEX_LATERAL_ID,
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
  toNativeModelText,
  textToIpa,
  ipaToIds,
  textToIds,
  cleanStudioText,
  studioTextToIds,
  synthesizeKlattPcm,
  encodeWavBuffer,
  generateRoomTone,
};
