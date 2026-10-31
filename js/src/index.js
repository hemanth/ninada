/**
 * ninada — Pan-Indic & Classical Sanskrit Neural TTS (Hemanth-style minimal functional SDK)
 *
 * Three-line quick start:
 *   import ninada from 'ninada';
 *   const voice = await ninada('sanskrit');
 *   const wav = await voice.speak('praj\u00f1\u0101p\u016brvakavy\u0101p\u0101ravattva\u1e41 j\u012bvatvam');
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

export default async function ninada(input = 'sa', opts = {}) {
  let defaultLang = 'sa';
  let modelSource = opts.model || null;

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
  }

  let session = opts.session || null;
  let ort = opts.ort || (typeof globalThis !== 'undefined' ? globalThis.ort : null);

  if (!session && modelSource && !ort) {
    for (const pkg of ['onnxruntime-node', 'onnxruntime-web']) {
      try {
        const mod = await import(pkg);
        const candidate = mod.default && mod.default.InferenceSession ? mod.default : mod;
        if (candidate && candidate.InferenceSession) {
          ort = candidate;
          break;
        }
      } catch {}
    }
  }

  if (!session && modelSource && ort && ort.InferenceSession) {
    const isNode = typeof process !== 'undefined' && Boolean(process.versions?.node) && typeof window === 'undefined';
    let resolvedSource = modelSource;
    if (isNode && typeof modelSource === 'string' && !/^https?:\/\//i.test(modelSource) && ort.env?.wasm) {
      const fsPkg = 'node:fs/promises';
      const fsMod = await import(fsPkg);
      const buf = await fsMod.readFile(modelSource);
      resolvedSource = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    }
    const providers = opts.providers || (ort.env?.wasm ? ['wasm'] : ['cpu']);
    session = await ort.InferenceSession.create(resolvedSource, { executionProviders: providers });
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
      return Boolean(session);
    },
    on(event, handler) {
      if (!listeners.has(event)) listeners.set(event, []);
      listeners.get(event).push(handler);
      return this;
    },
    inspect(text, callOpts = {}) {
      return textToIds(text, { lang: callOpts.lang ? resolveLang(callOpts.lang) : defaultLang });
    },
    async synthesize(text, callOpts = {}) {
      const lang = callOpts.lang ? resolveLang(callOpts.lang) : defaultLang;
      const rate = callOpts.rate ?? opts.rate ?? 1.0;
      const noise = callOpts.noise ?? opts.noise ?? 0.32;
      const tier = callOpts.tier || 'auto';

      if (tier !== 'klatt' && session && ort) {
        const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
        const parsed = textToIds(text, { lang });
        const idsBig = BigInt64Array.from(parsed.inputIds.map(BigInt));
        const maskBig = BigInt64Array.from(parsed.attentionMask.map(BigInt));
        const langBig = BigInt64Array.from([BigInt(parsed.langId)]);
        const scalesArr = Float32Array.from([Number(rate), Number(noise)]);

        const feeds = {
          input_ids: new ort.Tensor('int64', idsBig, [1, parsed.inputIds.length]),
          attention_mask: new ort.Tensor('int64', maskBig, [1, parsed.attentionMask.length]),
          lang_id: new ort.Tensor('int64', langBig, [1]),
          scales: new ort.Tensor('float32', scalesArr, [2]),
        };

        const out = await session.run(feeds);
        const raw = out.waveform.data;
        const pcm = new Float32Array(raw.length);
        let peak = 1e-5;
        for (let i = 0; i < raw.length; i++) {
          const a = Math.abs(raw[i]);
          if (a > peak) peak = a;
        }
        const gain = peak > 1e-4 ? 0.88 / Math.max(1.0, peak) : 1.0;
        for (let i = 0; i < raw.length; i++) pcm[i] = raw[i] * gain;

        const elapsedMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
        const durationSec = pcm.length / SAMPLE_RATE;
        const meta = {
          ...parsed,
          tier: 'neural_onnx',
          sampleRate: SAMPLE_RATE,
          durationSec,
          latencyMs: elapsedMs,
          rtf: elapsedMs / 1000 / Math.max(0.01, durationSec),
          durations: Array.from(out.durations.data),
        };
        emit('chunk', pcm);
        emit('done', meta);
        return { pcm, wav: encodeWavBuffer(pcm, SAMPLE_RATE), meta };
      }

      const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
      const { pcm, meta: baseMeta } = synthesizeKlattPcm(text, { lang, rate });
      const elapsedMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
      const durationSec = pcm.length / SAMPLE_RATE;
      const meta = {
        ...baseMeta,
        sampleRate: SAMPLE_RATE,
        durationSec,
        latencyMs: elapsedMs,
        rtf: elapsedMs / 1000 / Math.max(0.01, durationSec),
      };
      emit('chunk', pcm);
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
  synthesizeKlattPcm,
  encodeWavBuffer,
};
