/**
 * ninada v5.0 — Streaming 24 kHz Native Per-Language Studio VITS Neural TTS
 * (24 Dedicated Male & Female Native 24,000 Hz Studio VITS Specialists across 12 Languages
 *  + Zero-Dep Klatt Fallback for Browser WASM SIMD & Node.js)
 *
 * Streaming quick start:
 *   import ninada from 'ninada';
 *   const voice = await ninada('english', {
 *     models: { en: './models/en.onnx', kn: './models/kn.onnx', sa: './models/sa.onnx' },
 *   });
 *   for await (const chunk of voice.stream('Hello from Bengaluru, welcome to Ninada Studio.')) {
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
const DEFAULT_HF_REPO = 'gnumanth/ninada';
const ALL_SPECIALIST_KEYS = [
  'sa', 'kn', 'hi', 'ta', 'te', 'ml', 'en', 'mr', 'bn', 'gu', 'pa', 'or',
  'sa_female', 'kn_female', 'hi_female', 'ta_female', 'te_female', 'ml_female',
  'en_female', 'mr_female', 'bn_female', 'gu_female', 'pa_female', 'or_female',
];

const LANG_ALIASES = {
  sa: 'sa', sanskrit: 'sa',
  hi: 'hi', hindi: 'hi',
  kn: 'kn', kannada: 'kn',
  ta: 'ta', tamil: 'ta',
  te: 'te', telugu: 'te',
  ml: 'ml', malayalam: 'ml',
  en: 'en', english: 'en', 'indian-english': 'en', 'en-in': 'en',
  mr: 'mr', marathi: 'mr',
  bn: 'bn', bengali: 'bn', bangla: 'bn',
  gu: 'gu', gujarati: 'gu',
  pa: 'pa', punjabi: 'pa', gurmukhi: 'pa',
  or: 'or', odia: 'or', oriya: 'or',
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
          if (/^(ಮತ್ತು|और|तथा|மற்றும்|మరియు|കൂടാതെ|च|and|आणि|এবং|અને|ਅਤੇ|ଏବଂ)$/i.test(words[i])) {
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

function unwrapSessionOrPipeline(candidate) {
  if (!candidate) return null;
  if (Array.isArray(candidate.sessions) && candidate.sessions[0]) {
    return candidate.sessions[0];
  }
  if (typeof candidate.run === 'function') {
    return candidate;
  }
  return null;
}

class DuckOrtTensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
    this.size = data.length;
  }
  dispose() {}
}

async function resolveWebmlKit(explicitWebml, skipAutoImport = false) {
  if (explicitWebml && typeof explicitWebml.createOnnxPipeline === 'function') {
    return explicitWebml;
  }
  if (
    typeof globalThis !== 'undefined' &&
    globalThis.webml &&
    typeof globalThis.webml.createOnnxPipeline === 'function'
  ) {
    return globalThis.webml;
  }
  if (skipAutoImport) return null;
  try {
    const pkg = 'webml-kit';
    const mod = await import(pkg);
    const candidate = mod?.createOnnxPipeline ? mod : mod?.default;
    if (candidate && typeof candidate.createOnnxPipeline === 'function') {
      return candidate;
    }
  } catch {}
  return null;
}

async function resolveOrt(explicitOrt, webmlKit = null) {
  if (explicitOrt && (explicitOrt.InferenceSession || explicitOrt.Tensor)) return explicitOrt;
  if (typeof globalThis !== 'undefined' && globalThis.ort && (globalThis.ort.InferenceSession || globalThis.ort.Tensor)) {
    return globalThis.ort;
  }
  if (webmlKit?.ort && (webmlKit.ort.InferenceSession || webmlKit.ort.Tensor)) {
    return webmlKit.ort;
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

const BROWSER_MODEL_CACHE = 'ninada-onnx-v5.0.0';

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

function yieldMacrotask() {
  if (typeof MessageChannel !== 'undefined') {
    return new Promise((resolve) => {
      const ch = new MessageChannel();
      ch.port1.onmessage = () => {
        ch.port1.onmessage = null;
        resolve();
      };
      ch.port2.postMessage(0);
    });
  }
  return new Promise((resolve) => setTimeout(resolve, 0));
}

const _UTF8_DEC = typeof TextDecoder !== 'undefined' ? new TextDecoder() : null;
const _UTF8_ENC = typeof TextEncoder !== 'undefined' ? new TextEncoder() : null;

// Static ValueInfoProto type suffixes:
//   FLOAT [1, 192, "T"] -> type { tensor_type { elem_type=1, shape { 1, 192, "T" } } }
const _VI_SUFFIX_FLOAT_1_192_T = new Uint8Array([
  18, 20, 10, 18, 8, 1, 18, 14, 10, 2, 8, 1, 10, 3, 8, 192, 1, 10, 3, 18, 1, 84,
]);
//   FLOAT [1, 1, "N"]   -> type { tensor_type { elem_type=1, shape { 1, 1, "N" } } }
const _VI_SUFFIX_FLOAT_1_1_N = new Uint8Array([
  18, 19, 10, 17, 8, 1, 18, 13, 10, 2, 8, 1, 10, 2, 8, 1, 10, 3, 18, 1, 78,
]);

function _pbReadVarint(buf, pos) {
  let val = 0;
  let mult = 1;
  const len = buf.length;
  while (pos < len) {
    const b = buf[pos++];
    val += (b & 0x7f) * mult;
    if ((b & 0x80) === 0) return [val, pos];
    mult *= 128;
  }
  throw new Error('Truncated ONNX protobuf varint');
}

function _pbSkipField(buf, pos, wireType) {
  if (wireType === 0) {
    while (pos < buf.length && (buf[pos++] & 0x80) !== 0) {}
    return pos;
  }
  if (wireType === 1) return pos + 8;
  if (wireType === 2) {
    const [len, p2] = _pbReadVarint(buf, pos);
    return p2 + len;
  }
  if (wireType === 5) return pos + 4;
  throw new Error(`Unsupported protobuf wire type: ${wireType}`);
}

function _pbEncodeVarint(n) {
  const out = [];
  let v = Math.floor(n);
  while (v >= 128) {
    out.push((v % 128) | 0x80);
    v = Math.floor(v / 128);
  }
  out.push(v);
  return new Uint8Array(out);
}

function _pbWrapLengthDelimited(fieldNum, chunks) {
  let payloadLen = 0;
  for (let i = 0; i < chunks.length; i++) payloadLen += chunks[i].byteLength;
  const tagBytes = _pbEncodeVarint(fieldNum * 8 + 2);
  const lenBytes = _pbEncodeVarint(payloadLen);
  const out = new Uint8Array(tagBytes.byteLength + lenBytes.byteLength + payloadLen);
  let off = 0;
  out.set(tagBytes, off);
  off += tagBytes.byteLength;
  out.set(lenBytes, off);
  off += lenBytes.byteLength;
  for (let i = 0; i < chunks.length; i++) {
    out.set(chunks[i], off);
    off += chunks[i].byteLength;
  }
  return out;
}

function _pbMakeValueInfoField(graphFieldNum, tensorName, typeSuffix) {
  const nameBytes = _UTF8_ENC
    ? _UTF8_ENC.encode(tensorName)
    : Uint8Array.from(Array.from(tensorName, (c) => c.charCodeAt(0) & 0xff));
  const nameHeader = _pbEncodeVarint(nameBytes.byteLength);
  const viPayload = new Uint8Array(1 + nameHeader.byteLength + nameBytes.byteLength + typeSuffix.byteLength);
  let off = 0;
  viPayload[off++] = 10; // field 1 (name), wire 2
  viPayload.set(nameHeader, off);
  off += nameHeader.byteLength;
  viPayload.set(nameBytes, off);
  off += nameBytes.byteLength;
  viPayload.set(typeSuffix, off);
  return _pbWrapLengthDelimited(graphFieldNum, [viPayload]);
}

function _pbDecodeUtf8(buf, start, end) {
  if (_UTF8_DEC) return _UTF8_DEC.decode(buf.subarray(start, end));
  let s = '';
  for (let i = start; i < end; i++) s += String.fromCharCode(buf[i]);
  return s;
}

function _pbScanGraphNodeInputs(buf, gStart, gEnd, outInputs) {
  let pos = gStart;
  while (pos < gEnd) {
    const [tag, p1] = _pbReadVarint(buf, pos);
    const fNum = Math.floor(tag / 8);
    const wt = tag & 7;
    const fEnd = _pbSkipField(buf, p1, wt);
    if (fNum === 1 && wt === 2) {
      const [nLen, nStart] = _pbReadVarint(buf, p1);
      const nEnd = nStart + nLen;
      let np = nStart;
      while (np < nEnd) {
        const [nTag, np1] = _pbReadVarint(buf, np);
        const nf = Math.floor(nTag / 8);
        const nwt = nTag & 7;
        const nfEnd = _pbSkipField(buf, np1, nwt);
        if (nf === 1 && nwt === 2) {
          const [sLen, sStart] = _pbReadVarint(buf, np1);
          if (sLen > 0) outInputs.push(_pbDecodeUtf8(buf, sStart, sStart + sLen));
        }
        np = nfEnd;
      }
    }
    pos = fEnd;
  }
}

/**
 * Splits a monolithic Studio VITS 24 kHz `.onnx` binary (`input_ids, scales -> waveform, durations`)
 * in memory (< 8ms, zero weight duplication) into:
 *   - `encFlowBytes`: `TextEncoder + DurationPredictor + NormalizingFlow` (`input_ids, scales -> latent, durations`)
 *   - `vocoderBytes`: `HiFi-GAN WaveformDecoder` (`latent -> wav3d`)
 * Returns `null` if the ONNX graph is not a Studio VITS model with `/waveform_decoder/conv_pre/Conv`.
 */
function splitStudioVitsOnnxBuffer(rawBuffer) {
  try {
    let buf = null;
    if (rawBuffer instanceof Uint8Array) {
      buf = rawBuffer;
    } else if (rawBuffer instanceof ArrayBuffer) {
      buf = new Uint8Array(rawBuffer);
    } else if (ArrayBuffer.isView(rawBuffer)) {
      buf = new Uint8Array(rawBuffer.buffer, rawBuffer.byteOffset, rawBuffer.byteLength);
    }
    if (!buf || buf.byteLength < 65536 || buf[0] !== 0x08) return null;

    const modelMetaSlices = [];
    let gStart = -1;
    let gEnd = -1;

    let pos = 0;
    while (pos < buf.byteLength) {
      const fStart = pos;
      const [tag, p1] = _pbReadVarint(buf, pos);
      const fNum = Math.floor(tag / 8);
      const wt = tag & 7;
      const fEnd = _pbSkipField(buf, p1, wt);
      if (fEnd > buf.byteLength) return null;
      if (fNum === 7 && wt === 2) {
        const [gLen, gs] = _pbReadVarint(buf, p1);
        gStart = gs;
        gEnd = gs + gLen;
      } else {
        modelMetaSlices.push(buf.subarray(fStart, fEnd));
      }
      pos = fEnd;
    }
    if (gStart < 0 || gEnd > buf.byteLength) return null;

    const nodes = [];
    const initializers = [];
    const graphInputs = [];
    const graphOutputs = [];
    let graphNameSlice = null;

    pos = gStart;
    while (pos < gEnd) {
      const fStart = pos;
      const [tag, p1] = _pbReadVarint(buf, pos);
      const fNum = Math.floor(tag / 8);
      const wt = tag & 7;
      const fEnd = _pbSkipField(buf, p1, wt);
      if (fEnd > gEnd) return null;

      if (fNum === 1 && wt === 2) {
        const [nLen, nStart] = _pbReadVarint(buf, p1);
        const nEnd = nStart + nLen;
        const inputs = [];
        const outputs = [];
        const extraInputs = [];
        let name = '';
        let np = nStart;
        while (np < nEnd) {
          const [nTag, np1] = _pbReadVarint(buf, np);
          const nf = Math.floor(nTag / 8);
          const nwt = nTag & 7;
          const nfEnd = _pbSkipField(buf, np1, nwt);
          if (nf === 1 && nwt === 2) {
            const [sLen, sStart] = _pbReadVarint(buf, np1);
            inputs.push(sLen > 0 ? _pbDecodeUtf8(buf, sStart, sStart + sLen) : '');
          } else if (nf === 2 && nwt === 2) {
            const [sLen, sStart] = _pbReadVarint(buf, np1);
            outputs.push(sLen > 0 ? _pbDecodeUtf8(buf, sStart, sStart + sLen) : '');
          } else if (nf === 3 && nwt === 2) {
            const [sLen, sStart] = _pbReadVarint(buf, np1);
            name = _pbDecodeUtf8(buf, sStart, sStart + sLen);
          } else if (nf === 5 && nwt === 2) {
            // Check AttributeProto field 5 (`g`, GraphProto) for subgraph references
            const [aLen, aStart] = _pbReadVarint(buf, np1);
            const aEnd = aStart + aLen;
            let ap = aStart;
            while (ap < aEnd) {
              const [aTag, ap1] = _pbReadVarint(buf, ap);
              const af = Math.floor(aTag / 8);
              const awt = aTag & 7;
              const afEnd = _pbSkipField(buf, ap1, awt);
              if (af === 5 && awt === 2) {
                const [sgLen, sgStart] = _pbReadVarint(buf, ap1);
                _pbScanGraphNodeInputs(buf, sgStart, sgStart + sgLen, extraInputs);
              }
              ap = afEnd;
            }
          }
          np = nfEnd;
        }
        nodes.push({ raw: buf.subarray(fStart, fEnd), inputs, outputs, extraInputs, name });
      } else if (fNum === 2 && wt === 2) {
        graphNameSlice = buf.subarray(fStart, fEnd);
      } else if (fNum === 5 && wt === 2) {
        const [tLen, tStart] = _pbReadVarint(buf, p1);
        const tEnd = tStart + tLen;
        let tName = '';
        let tp = tStart;
        while (tp < tEnd) {
          const [tTag, tp1] = _pbReadVarint(buf, tp);
          const tf = Math.floor(tTag / 8);
          const twt = tTag & 7;
          const tfEnd = _pbSkipField(buf, tp1, twt);
          if (tf === 8 && twt === 2) {
            const [sLen, sStart] = _pbReadVarint(buf, tp1);
            tName = _pbDecodeUtf8(buf, sStart, sStart + sLen);
            break;
          }
          tp = tfEnd;
        }
        initializers.push({ raw: buf.subarray(fStart, fEnd), name: tName });
      } else if ((fNum === 11 || fNum === 12) && wt === 2) {
        const [vLen, vStart] = _pbReadVarint(buf, p1);
        const vEnd = vStart + vLen;
        let vName = '';
        let vp = vStart;
        while (vp < vEnd) {
          const [vTag, vp1] = _pbReadVarint(buf, vp);
          const vf = Math.floor(vTag / 8);
          const vwt = vTag & 7;
          const vfEnd = _pbSkipField(buf, vp1, vwt);
          if (vf === 1 && vwt === 2) {
            const [sLen, sStart] = _pbReadVarint(buf, vp1);
            vName = _pbDecodeUtf8(buf, sStart, sStart + sLen);
            break;
          }
          vp = vfEnd;
        }
        const rec = { raw: buf.subarray(fStart, fEnd), name: vName };
        if (fNum === 11) graphInputs.push(rec);
        else graphOutputs.push(rec);
      }
      pos = fEnd;
    }

    const inNameSet = new Set(graphInputs.map((i) => i.name));
    if (!inNameSet.has('input_ids') || !inNameSet.has('scales')) return null;

    let convPreNode = null;
    let lastDecNode = null;
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      if (!convPreNode && (n.name === '/waveform_decoder/conv_pre/Conv' || n.name.endsWith('waveform_decoder/conv_pre/Conv'))) {
        convPreNode = n;
      }
      if (n.name.startsWith('/waveform_decoder') && n.outputs.length > 0) {
        lastDecNode = n;
      }
    }
    if (!convPreNode || !lastDecNode || !convPreNode.inputs[0] || !lastDecNode.outputs[0]) {
      return null;
    }

    const latentName = convPreNode.inputs[0];
    const wav3dName = lastDecNode.outputs[0];
    const durOutputRec = graphOutputs.find((o) => o.name === 'durations');
    if (!durOutputRec) return null;

    const producerByTensor = new Map();
    for (let i = 0; i < nodes.length; i++) {
      for (const o of nodes[i].outputs) {
        if (o) producerByTensor.set(o, i);
      }
    }

    function collectSubgraph(targetTensors, stopTensors) {
      const visited = new Uint8Array(nodes.length);
      const usedInputs = new Set();
      const stack = [];
      for (const t of targetTensors) {
        if (!stopTensors.has(t)) {
          const pIdx = producerByTensor.get(t);
          if (pIdx !== undefined) stack.push(pIdx);
        }
      }
      while (stack.length > 0) {
        const idx = stack.pop();
        if (visited[idx]) continue;
        visited[idx] = 1;
        const nd = nodes[idx];
        const allIn = nd.extraInputs.length > 0 ? nd.inputs.concat(nd.extraInputs) : nd.inputs;
        for (let k = 0; k < allIn.length; k++) {
          const inp = allIn[k];
          if (!inp || stopTensors.has(inp)) continue;
          usedInputs.add(inp);
          const pIdx = producerByTensor.get(inp);
          if (pIdx !== undefined && !visited[pIdx]) {
            stack.push(pIdx);
          }
        }
      }
      const indices = [];
      for (let i = 0; i < nodes.length; i++) {
        if (visited[i]) indices.push(i);
      }
      return { indices, usedInputs };
    }

    const encSub = collectSubgraph([latentName, 'durations'], new Set(['input_ids', 'scales']));
    const vocSub = collectSubgraph([wav3dName], new Set([latentName]));
    if (encSub.indices.length === 0 || vocSub.indices.length === 0) return null;

    const encGraphParts = [];
    for (const idx of encSub.indices) encGraphParts.push(nodes[idx].raw);
    if (graphNameSlice) encGraphParts.push(graphNameSlice);
    for (const init of initializers) {
      if (encSub.usedInputs.has(init.name)) encGraphParts.push(init.raw);
    }
    for (const gi of graphInputs) {
      if (gi.name === 'input_ids' || gi.name === 'scales') encGraphParts.push(gi.raw);
    }
    encGraphParts.push(_pbMakeValueInfoField(12, latentName, _VI_SUFFIX_FLOAT_1_192_T));
    encGraphParts.push(durOutputRec.raw);

    const vocGraphParts = [];
    for (const idx of vocSub.indices) vocGraphParts.push(nodes[idx].raw);
    if (graphNameSlice) vocGraphParts.push(graphNameSlice);
    for (const init of initializers) {
      if (vocSub.usedInputs.has(init.name)) vocGraphParts.push(init.raw);
    }
    vocGraphParts.push(_pbMakeValueInfoField(11, latentName, _VI_SUFFIX_FLOAT_1_192_T));
    vocGraphParts.push(_pbMakeValueInfoField(12, wav3dName, _VI_SUFFIX_FLOAT_1_1_N));

    const encGraphField = _pbWrapLengthDelimited(7, encGraphParts);
    const vocGraphField = _pbWrapLengthDelimited(7, vocGraphParts);

    const encModelParts = [...modelMetaSlices, encGraphField];
    const vocModelParts = [...modelMetaSlices, vocGraphField];

    let encLen = 0;
    for (const p of encModelParts) encLen += p.byteLength;
    const encFlowBytes = new Uint8Array(encLen);
    let eOff = 0;
    for (const p of encModelParts) {
      encFlowBytes.set(p, eOff);
      eOff += p.byteLength;
    }

    let vocLen = 0;
    for (const p of vocModelParts) vocLen += p.byteLength;
    const vocoderBytes = new Uint8Array(vocLen);
    let vOff = 0;
    for (const p of vocModelParts) {
      vocoderBytes.set(p, vOff);
      vOff += p.byteLength;
    }

    return {
      encFlowBytes,
      vocoderBytes,
      latentName,
      wav3dName,
    };
  } catch {
    return null;
  }
}

async function createSplitStudioVitsSession(ort, rawBytes, sessionOpts) {
  if (!ort || !ort.InferenceSession) return null;
  const split = splitStudioVitsOnnxBuffer(rawBytes);
  if (!split) {
    return ort.InferenceSession.create(rawBytes, sessionOpts);
  }
  const { latentName, wav3dName } = split;
  let encFlowBytes = split.encFlowBytes;
  let vocoderBytes = split.vocoderBytes;
  split.encFlowBytes = null;
  split.vocoderBytes = null;

  let encFlowSession = null;
  let vocoderSession = null;
  try {
    encFlowSession = await ort.InferenceSession.create(encFlowBytes, sessionOpts);
    encFlowBytes = null;
    vocoderSession = await ort.InferenceSession.create(vocoderBytes, sessionOpts);
    vocoderBytes = null;
  } catch (err) {
    try { await encFlowSession?.release?.(); } catch {}
    try { await vocoderSession?.release?.(); } catch {}
    return ort.InferenceSession.create(rawBytes, sessionOpts);
  }

  return {
    inputNames: ['input_ids', 'scales'],
    outputNames: ['waveform', 'durations'],
    isSplitStudioVits: true,
    encFlowSession,
    vocoderSession,
    latentName,
    wav3dName,
    async run(feeds) {
      const encOut = await encFlowSession.run(feeds);
      const zTensor = encOut[latentName];
      try {
        const vocOut = await vocoderSession.run({ [latentName]: zTensor });
        const wavTensor = vocOut[wav3dName];
        return {
          waveform: wavTensor,
          durations: encOut.durations,
        };
      } finally {
        zTensor?.dispose?.();
      }
    },
    async release() {
      try { await encFlowSession?.release?.(); } catch {}
      try { await vocoderSession?.release?.(); } catch {}
    },
  };
}

async function createOrtSession(ort, source, providers, useCache = true, webmlOpts = null) {
  const unwrapped = unwrapSessionOrPipeline(source);
  if (unwrapped) return unwrapped;
  if (!source) return null;

  const isNode = isNodeEnv();
  const webmlKit = webmlOpts?.webmlKit || null;
  const preferWebml = Boolean(
    webmlKit &&
    typeof source === 'string' &&
    (!isNode || webmlOpts?.explicitWebml || typeof webmlOpts?.onProgress === 'function' || !ort?.InferenceSession)
  );

  if (preferWebml) {
    try {
      const pipe = await webmlKit.createOnnxPipeline(
        {
          modelId: webmlOpts?.hubRepo || DEFAULT_HF_REPO,
          modelFile: source,
          task: 'raw-onnx',
          device: webmlOpts?.device || 'wasm',
          cache: useCache,
          cacheName: BROWSER_MODEL_CACHE,
        },
        webmlOpts?.onProgress
      );
      const sess = unwrapSessionOrPipeline(pipe);
      if (sess) return sess;
    } catch {
      // Fall back to direct ort.InferenceSession.create below
    }
  }

  if (!ort || !ort.InferenceSession) return null;
  let resolved = source;
  if (typeof source === 'string' && hubRepoPathNeedsExpand(source, webmlOpts?.hubRepo)) {
    resolved = `https://huggingface.co/${webmlOpts?.hubRepo || DEFAULT_HF_REPO}/resolve/main/${source}`;
  }
  if (isNode && typeof resolved === 'string' && !/^https?:\/\//i.test(resolved)) {
    try {
      const fsMod = await import('node:fs/promises');
      const buf = await fsMod.readFile(resolved);
      resolved = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    } catch {
      // Allow mock path strings in unit tests to fall through to ort.InferenceSession.create
    }
  } else if (isNode && typeof resolved === 'string' && /^https?:\/\//i.test(resolved) && typeof fetch === 'function') {
    const res = await fetch(resolved);
    if (!res.ok) throw new Error(`Failed to fetch ONNX model (${res.status}): ${resolved}`);
    resolved = new Uint8Array(await res.arrayBuffer());
  } else if (!isNode && typeof resolved === 'string' && typeof fetch === 'function') {
    resolved = await fetchBrowserCachedBytes(resolved, useCache);
  }
  const useProviders = providers || (ort.env?.wasm ? ['wasm'] : ['cpu']);
  const canUseArena = isNode || Boolean(typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated);
  const sessionOpts = {
    executionProviders: useProviders,
    graphOptimizationLevel: 'all',
    enableCpuMemArena: canUseArena,
    enableMemPattern: canUseArena,
  };
  if (resolved instanceof Uint8Array || resolved instanceof ArrayBuffer || ArrayBuffer.isView(resolved)) {
    return createSplitStudioVitsSession(ort, resolved, sessionOpts);
  }
  return ort.InferenceSession.create(resolved, sessionOpts);
}

function hubRepoPathNeedsExpand(src, hubRepo) {
  if (!hubRepo || typeof src !== 'string') return false;
  if (/^https?:\/\//i.test(src) || src.startsWith('/') || src.startsWith('./') || src.startsWith('../')) {
    return false;
  }
  return src.startsWith('onnx/');
}

function isHubRepoId(val) {
  return (
    typeof val === 'string' &&
    /^[a-z0-9_-]+\/[a-z0-9_.-]+$/i.test(val.trim()) &&
    !/\.onnx$/i.test(val.trim())
  );
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
  let hubRepo = typeof opts.hub === 'string' ? opts.hub.trim() : opts.hub === true ? DEFAULT_HF_REPO : null;

  if (typeof input === 'string') {
    const k = input.trim().toLowerCase();
    if (LANG_ALIASES[k]) {
      defaultLang = LANG_ALIASES[k];
    } else if (isHubRepoId(input)) {
      hubRepo = input.trim();
      defaultLang = resolveLang(opts.lang);
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
    if (!hubRepo) {
      hubRepo = typeof opts.hub === 'string' ? opts.hub.trim() : opts.hub === true ? DEFAULT_HF_REPO : null;
    }
  }

  const defaultVoiceMode =
    typeof opts.voice === 'string' && !/\.bin$/i.test(opts.voice)
      ? opts.voice.trim().toLowerCase()
      : null;

  const modelBasePath = typeof modelSource === 'string' && !/^https?:\/\//i.test(modelSource) ? modelSource : null;

  let session = unwrapSessionOrPipeline(opts.pipeline || opts.session) || null;
  let encSession = unwrapSessionOrPipeline(opts.encoderSession) || null;
  let vocSession = unwrapSessionOrPipeline(opts.vocoderSession) || null;
  const getSessionHook = typeof opts.getSession === 'function' ? opts.getSession : null;

  const sessionsMap = new Map();
  if (opts.sessions) {
    const entries = opts.sessions instanceof Map ? opts.sessions.entries() : Object.entries(opts.sessions);
    for (const [k, s] of entries) {
      const unwrapped = unwrapSessionOrPipeline(s);
      if (unwrapped) sessionsMap.set(String(k).trim().toLowerCase(), unwrapped);
    }
  }
  if (opts.pipelines && typeof opts.pipelines === 'object') {
    const entries = opts.pipelines instanceof Map ? opts.pipelines.entries() : Object.entries(opts.pipelines);
    for (const [k, p] of entries) {
      const unwrapped = unwrapSessionOrPipeline(p);
      if (unwrapped) sessionsMap.set(String(k).trim().toLowerCase(), unwrapped);
    }
  }

  const modelSourcesMap = new Map();
  if (hubRepo) {
    for (const specKey of ALL_SPECIALIST_KEYS) {
      modelSourcesMap.set(specKey, `onnx/${specKey}.onnx`);
    }
  }
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
  const skipAutoWebmlImport = Boolean(opts.ort && (getSessionHook || session || sessionsMap.size > 0));
  const webmlKit = hasNeuralConfig ? await resolveWebmlKit(opts.webml, skipAutoWebmlImport) : null;
  const ort = hasNeuralConfig ? await resolveOrt(opts.ort, webmlKit) : null;
  const TensorCtor = ort?.Tensor || DuckOrtTensor;
  const webmlSessionOpts = {
    webmlKit,
    explicitWebml: Boolean(opts.webml),
    hubRepo,
    device: opts.device || 'wasm',
    onProgress: typeof opts.onProgress === 'function' ? opts.onProgress : undefined,
  };

  const useCache = opts.cache !== false;
  const canCreateSessions = Boolean(ort?.InferenceSession || webmlKit?.createOnnxPipeline);
  const initialTargetKey = defaultVoiceMode === 'female' ? `${defaultLang}_female` : defaultLang;
  if (!encSession && !vocSession && encSource && vocSource && canCreateSessions) {
    encSession = await createOrtSession(ort, encSource, opts.providers, useCache, webmlSessionOpts);
    encSource = null;
    vocSession = await createOrtSession(ort, vocSource, opts.providers, useCache, webmlSessionOpts);
    vocSource = null;
  } else if (!session && modelSource && canCreateSessions) {
    session = await createOrtSession(ort, modelSource, opts.providers, useCache, webmlSessionOpts);
    modelSource = null;
  } else if (!session && !getSessionHook && modelSourcesMap.has(initialTargetKey) && canCreateSessions) {
    session = await createOrtSession(ort, modelSourcesMap.get(initialTargetKey), opts.providers, useCache, webmlSessionOpts);
    if (session) sessionsMap.set(initialTargetKey, session);
  } else if (!session && !getSessionHook && modelSourcesMap.has(defaultLang) && canCreateSessions) {
    session = await createOrtSession(ort, modelSourcesMap.get(defaultLang), opts.providers, useCache, webmlSessionOpts);
    if (session) sessionsMap.set(defaultLang, session);
  }

  modelSource = null;
  encSource = null;
  vocSource = null;

  async function resolveV4Session(lang, voiceOverride = null) {
    const lc = resolveLang(lang);
    const effVoice = voiceOverride || defaultVoiceMode;
    const vNorm = typeof effVoice === 'string' ? effVoice.trim().toLowerCase() : '';
    const femKey = `${lc}_female`;
    const targetKey = vNorm === 'female' ? femKey : lc;

    if (getSessionHook) {
      const hooked = unwrapSessionOrPipeline(await getSessionHook(lc, vNorm || 'male'));
      if (isV4StudioSession(hooked)) return hooked;
    }
    if (sessionsMap.has(targetKey)) {
      const hit = sessionsMap.get(targetKey);
      if (isV4StudioSession(hit)) return hit;
    }
    if (modelSourcesMap.has(targetKey) && canCreateSessions) {
      try {
        const loaded = await createOrtSession(ort, modelSourcesMap.get(targetKey), opts.providers, useCache, webmlSessionOpts);
        if (isV4StudioSession(loaded)) {
          sessionsMap.set(targetKey, loaded);
          return loaded;
        }
      } catch {}
    }
    if (vNorm === 'female') {
      if (sessionsMap.has(lc) && isV4StudioSession(sessionsMap.get(lc))) return sessionsMap.get(lc);
      if (modelSourcesMap.has(lc) && canCreateSessions) {
        try {
          const loaded = await createOrtSession(ort, modelSourcesMap.get(lc), opts.providers, useCache, webmlSessionOpts);
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

      const v4InitialSess = tier !== 'klatt' ? await resolveV4Session(lang, callOpts.voice || null) : null;
      const isV4 = Boolean(v4InitialSess && TensorCtor);
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
          let activeSess = (await resolveV4Session(resolvedLc, callOpts.voice || null)) || v4InitialSess;
          const idTensor = new TensorCtor('int64', BigInt64Array.from(studioIds.map(BigInt)), [1, studioIds.length]);
          const scalesTensor = new TensorCtor('float32', Float32Array.from([1.0 / effRate, effNoise, effNoiseDp]), [3]);

          // Sub-Clause Latent-Window HiFi-GAN Streaming Path (when session was split into encFlow + vocoder)
          if (activeSess?.isSplitStudioVits && !callOpts._monolithicVocoder) {
            const tEnc0 = nowMs();
            const encOut = await activeSess.encFlowSession.run({
              input_ids: idTensor,
              scales: scalesTensor,
            });
            const encFlowMs = nowMs() - tEnc0;
            idTensor.dispose?.();
            scalesTensor.dispose?.();

            const zTensor = encOut[activeSess.latentName];
            const dursTensor = encOut.durations;
            if (signal?.aborted || !zTensor) {
              zTensor?.dispose?.();
              dursTensor?.dispose?.();
              return;
            }

            const zDims = zTensor.dims || [1, 192, Math.floor(zTensor.data.length / 192)];
            const latentChannels = Number(zDims[1] || 192);
            const totalLatentFrames = Math.max(1, Number(zDims[2] || Math.floor(zTensor.data.length / latentChannels)));
            const zFlat = new Float32Array(zTensor.data);
            const dursArr = dursTensor ? Array.from(dursTensor.data, Number) : [];
            zTensor.dispose?.();
            dursTensor?.dispose?.();

            const hasPause = cIdx < clauses.length - 1;
            const isFinalClause = cIdx === clauses.length - 1;
            const pauseSec = hasPause ? (clauseItem.sentenceEnd ? 0.220 : 0.030) : 0.480;
            const pauseSamples = Math.round((pauseSec / effRate) * SAMPLE_RATE);
            const clauseTotalSamples = totalLatentFrames * STUDIO_HOP_LENGTH + pauseSamples;
            const totalClauseChars = Math.max(
              1,
              clauses.reduce((acc, item) => acc + Math.max(1, (item.text || '').trim().length), 0)
            );
            const thisClauseChars = Math.max(1, (clauseItem.text || '').trim().length);
            const estimatedTotalSec =
              clauses.length === 1
                ? clauseTotalSamples / SAMPLE_RATE
                : (clauseTotalSamples / SAMPLE_RATE) * (totalClauseChars / thisClauseChars);

            const isSingleThread =
              Boolean(callOpts.singleThread) ||
              (ort?.env?.wasm?.numThreads === 1 && !isNodeEnv());
            const haloLatent = 10;
            const xfadeSamples = 256;
            const firstLatent = Math.max(
              20,
              Math.min(40, Number(callOpts.firstLatentFrames || (isSingleThread ? 28 : 32)))
            );

            const fBounds = [];
            for (let fCur = 0; fCur < totalLatentFrames; ) {
              const remFrames = totalLatentFrames - fCur;
              let stepF;
              if (chunkIndex === 0 && fBounds.length === 0) {
                stepF = firstLatent;
              } else if (isSingleThread) {
                if (isFinalClause && remFrames <= 112 && remFrames > 54) {
                  stepF = Math.ceil(remFrames / 2);
                } else if (isFinalClause && remFrames <= 54) {
                  stepF = remFrames;
                } else if (chunkIndex === 0 && fBounds.length === 1) {
                  stepF = 124;
                } else if (!isFinalClause) {
                  stepF = 124;
                } else {
                  stepF = 98;
                }
              } else if (fBounds.length === 1) {
                stepF = 40;
              } else if (fBounds.length === 2) {
                stepF = 56;
              } else if (fBounds.length === 3) {
                stepF = 72;
              } else {
                stepF = 96;
              }
              let fNext = Math.min(totalLatentFrames, fCur + stepF);
              if (totalLatentFrames - fNext < 20) fNext = totalLatentFrames;
              fBounds.push([fCur, fNext]);
              fCur = fNext;
            }

            let prevRightHalo = null;
            const shouldYield = callOpts.yieldBetweenWindows !== false && !isNodeEnv();

            for (let wIdx = 0; wIdx < fBounds.length; wIdx++) {
              if (shouldYield) {
                await yieldMacrotask();
              }
              if (signal?.aborted) return;

              const [fStart, fEnd] = fBounds[wIdx];
              const validF = fEnd - fStart;
              const lHalo = Math.min(fStart, haloLatent);
              const rHalo = Math.min(totalLatentFrames - fEnd, haloLatent);
              const sliceStart = fStart - lHalo;
              const sliceF = validF + lHalo + rHalo;

              const zSliceBuf = new Float32Array(latentChannels * sliceF);
              for (let ch = 0; ch < latentChannels; ch++) {
                const srcOff = ch * totalLatentFrames + sliceStart;
                zSliceBuf.set(zFlat.subarray(srcOff, srcOff + sliceF), ch * sliceF);
              }

              const tVoc0 = nowMs();
              const zSliceTensor = new TensorCtor('float32', zSliceBuf, [1, latentChannels, sliceF]);
              const vocOut = await activeSess.vocoderSession.run({
                [activeSess.latentName]: zSliceTensor,
              });
              const vocMs = nowMs() - tVoc0;
              zSliceTensor.dispose?.();

              const wavTensor = vocOut[activeSess.wav3dName];
              if (signal?.aborted || !wavTensor) {
                wavTensor?.dispose?.();
                return;
              }

              const rawSlice = wavTensor.data;
              const validStartSamp = lHalo * STUDIO_HOP_LENGTH;
              const validLenSamp = validF * STUDIO_HOP_LENGTH;
              const validEndSamp = validStartSamp + validLenSamp;
              const isLastSub = wIdx === fBounds.length - 1;
              const pcm = new Float32Array(validLenSamp + (isLastSub ? pauseSamples : 0));

              pcm.set(rawSlice.subarray(validStartSamp, validEndSamp), 0);

              if (prevRightHalo && validLenSamp >= xfadeSamples) {
                for (let i = 0; i < xfadeSamples; i++) {
                  const alpha = 0.5 * (1.0 - Math.cos((Math.PI * (i + 0.5)) / xfadeSamples));
                  pcm[i] = (1.0 - alpha) * prevRightHalo[i] + alpha * pcm[i];
                }
              }

              if (!isLastSub && rawSlice.length >= validEndSamp + xfadeSamples) {
                prevRightHalo = Float32Array.from(rawSlice.subarray(validEndSamp, validEndSamp + xfadeSamples));
              } else {
                prevRightHalo = null;
              }
              wavTensor.dispose?.();

              for (let i = 0; i < validLenSamp; i++) {
                const x = pcm[i] || 0;
                let y = x - dcXPrev + dcR * dcYPrev;
                dcXPrev = x;
                dcYPrev = y;
                if (y > 0.96) y = 0.96 + 0.03 * Math.tanh((y - 0.96) / 0.04);
                else if (y < -0.96) y = -0.96 - 0.03 * Math.tanh((-y - 0.96) / 0.04);
                pcm[i] = y;
              }
              if (Math.abs(dcYPrev) < 1e-15) dcYPrev = 0.0;

              if (wIdx === 0 && validLenSamp > 192) {
                for (let i = 0; i < 192; i++) {
                  pcm[i] *= 0.5 * (1.0 - Math.cos((Math.PI * i) / 192.0));
                }
              }
              if (isLastSub && validLenSamp > 256) {
                const off = validLenSamp - 256;
                for (let i = 0; i < 256; i++) {
                  pcm[off + i] *= 0.5 * (1.0 + Math.cos((Math.PI * i) / 256.0));
                }
              }

              if (ttfaMs === null) ttfaMs = nowMs() - tStart;

              let nextWindowSec = 0;
              if (!isLastSub) {
                const [nfS, nfE] = fBounds[wIdx + 1];
                const nL = Math.min(nfS, haloLatent);
                const nR = Math.min(totalLatentFrames - nfE, haloLatent);
                nextWindowSec = ((nfE - nfS + nL + nR) * STUDIO_HOP_LENGTH) / SAMPLE_RATE;
              }

              const chunkObj = attachLazyWav(
                {
                  pcm,
                  sampleRate: SAMPLE_RATE,
                  index: chunkIndex++,
                  clauseIndex: cIdx,
                  clauseCount: clauses.length,
                  isFinal: cIdx === clauses.length - 1 && isLastSub,
                  durationSec: pcm.length / SAMPLE_RATE,
                  clauseDurationSec: clauseTotalSamples / SAMPLE_RATE,
                  estimatedTotalSec,
                  remainingClauseSec: ((totalLatentFrames - fEnd) * STUDIO_HOP_LENGTH) / SAMPLE_RATE,
                  nextWindowSec,
                  vocoderWindowAudioSec: (sliceF * STUDIO_HOP_LENGTH) / SAMPLE_RATE,
                  encFlowMs: wIdx === 0 ? encFlowMs : 0,
                  latencyMs: vocMs + (wIdx === 0 ? encFlowMs : 0),
                  vocoderMs: vocMs,
                  ttfaMs,
                  lang: resolvedLc,
                  script: parsed.script,
                  tier: 'neural_onnx',
                  aksharas: wIdx === 0 ? parsed.aksharas : [],
                  durations: wIdx === 0 ? dursArr : [],
                },
                pcm
              );
              emit('chunk', chunkObj);
              yield chunkObj;
            }
            continue;
          }

          let out;
          try {
            out = await activeSess.run({
              input_ids: idTensor,
              scales: scalesTensor,
            });
          } catch (runErr) {
            const vNorm = typeof (callOpts.voice || defaultVoiceMode) === 'string'
              ? (callOpts.voice || defaultVoiceMode).trim().toLowerCase()
              : '';
            const targetKey = vNorm === 'female' ? `${resolvedLc}_female` : resolvedLc;
            if (webmlSessionOpts.device === 'webgpu' && modelSourcesMap.has(targetKey)) {
              webmlSessionOpts.device = 'wasm';
              const wasmSess = await createOrtSession(
                ort,
                modelSourcesMap.get(targetKey),
                ['wasm'],
                useCache,
                webmlSessionOpts
              );
              if (isV4StudioSession(wasmSess)) {
                sessionsMap.set(targetKey, wasmSess);
                activeSess = wasmSess;
                out = await activeSess.run({
                  input_ids: idTensor,
                  scales: scalesTensor,
                });
              } else {
                throw runErr;
              }
            } else {
              throw runErr;
            }
          }
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
          const pauseSec = hasPause ? (clauseItem.sentenceEnd ? 0.220 : 0.030) : 0.480;
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

      for await (const ch of this.stream(text, { ...callOpts, fastFirstClause: false, _monolithicVocoder: true })) {
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

const Ninada = { create: ninada };
ninada.create = ninada;

export {
  ninada,
  Ninada,
  DEFAULT_HF_REPO,
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
  splitStudioVitsOnnxBuffer,
  createSplitStudioVitsSession,
  synthesizeKlattPcm,
  encodeWavBuffer,
  generateRoomTone,
};
