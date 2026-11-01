/**
 * Tier-2 Zero-Dependency Cascade Klatt-Akshara Synthesizer + 16-bit PCM WAV Encoder
 */
import { textToIds } from './g2p.js';

export const SAMPLE_RATE = 24000;

const VOWEL_FORMANTS = {
  '\u0905': [640, 1240, 2550, 0.085],
  '\u0906': [760, 1180, 2520, 0.155],
  '\u093e': [760, 1180, 2520, 0.155],
  '\u0907': [310, 2240, 2920, 0.085],
  '\u093f': [310, 2240, 2920, 0.085],
  '\u0908': [285, 2340, 3000, 0.155],
  '\u0940': [285, 2340, 3000, 0.155],
  '\u0909': [340, 860, 2420, 0.085],
  '\u0941': [340, 860, 2420, 0.085],
  '\u090a': [310, 780, 2380, 0.155],
  '\u0942': [310, 780, 2380, 0.155],
  '\u090b': [440, 1480, 1950, 0.105],
  '\u0943': [440, 1480, 1950, 0.105],
  '\u0960': [440, 1480, 1950, 0.165],
  '\u0944': [440, 1480, 1950, 0.165],
  '\u090e': [460, 1960, 2720, 0.090],
  '\u0946': [460, 1960, 2720, 0.090],
  '\u090f': [410, 2080, 2780, 0.150],
  '\u0947': [410, 2080, 2780, 0.150],
  '\u0910': [580, 1780, 2700, 0.165],
  '\u0948': [580, 1780, 2700, 0.165],
  '\u0912': [480, 940, 2480, 0.090],
  '\u094a': [480, 940, 2480, 0.090],
  '\u0913': [430, 860, 2450, 0.150],
  '\u094b': [430, 860, 2450, 0.150],
  '\u0914': [600, 1020, 2480, 0.165],
  '\u094c': [600, 1020, 2480, 0.165],
};

function biquadResonator(samples, freq, bw, sr = SAMPLE_RATE) {
  const r = Math.exp((-Math.PI * bw) / sr);
  const c = -(r * r);
  const b = 2 * r * Math.cos((2 * Math.PI * freq) / sr);
  const a = 1 - b - c;
  let y1 = 0;
  let y2 = 0;
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const y = a * samples[i] + b * y1 + c * y2;
    y2 = y1;
    y1 = y;
    out[i] = y;
  }
  return out;
}

export function synthesizeKlattPcm(text, { lang = null, rate = 1.0, f0Base = 132.0 } = {}) {
  const parsed = textToIds(text, { lang });
  const aksharas = parsed.aksharas;
  const speed = Math.max(0.45, Math.min(2.5, Number(rate) || 1.0));
  const chunks = [];
  let totalSamples = 0;
  let phase = 0;
  const totalAk = Math.max(1, aksharas.length);

  let seed = 20261030;
  const randU = () => {
    seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
    return seed / 2147483648.0 - 1.0;
  };

  for (let idx = 0; idx < aksharas.length; idx++) {
    const { akshara: s, weight } = aksharas[idx];
    const decl = 1.0 - 0.10 * (idx / totalAk);
    const accent = weight === 'G' ? 1.07 : (weight === 'C' ? 0.96 : 1.0);
    const f0 = f0Base * decl * accent;

    let [f1, f2, f3, durV] = VOWEL_FORMANTS['\u0905'];
    if (s.endsWith('\u094d')) {
      durV = 0.000;
    } else {
      for (const ch of s) {
        if (VOWEL_FORMANTS[ch]) [f1, f2, f3, durV] = VOWEL_FORMANTS[ch];
      }
    }

    const firstCp = s.codePointAt(0);
    const isSibilant = s[0] === '\u0936' || s[0] === '\u0937' || s[0] === '\u0938';
    const isAspirated = '\u0916\u0918\u091b\u091d\u0920\u0922\u0925\u0927\u092b\u092d\u0939'.includes(s[0]);
    const isNasal = '\u0919\u091e\u0923\u0928\u092e\u0929'.includes(s[0]) || s.includes('\u0902');
    const hasVisarga = s.includes('\u0903');
    const isRetroflex = (firstCp >= 0x091F && firstCp <= 0x0923) || '\u0937\u0933\u0934'.includes(s[0]);
    if (isRetroflex) f3 = Math.min(f3, 1980);

    const onsetSec = (isSibilant ? 0.048 : (isAspirated ? 0.036 : 0.022)) / speed;
    const nucleusSec = (durV + (hasVisarga ? 0.035 : 0)) / speed;
    const nOnset = Math.max(16, Math.round(onsetSec * SAMPLE_RATE));
    const nNuc = Math.max(32, Math.round(nucleusSec * SAMPLE_RATE));
    const nTot = nOnset + nNuc;

    const exc = new Float32Array(nTot);
    for (let t = 0; t < nTot; t++) {
      let env = t < 0.12 * nTot ? Math.sin(Math.PI * Math.min(1, t / Math.max(1, Math.floor(0.12 * nTot)))) : 1.0;
      if (t > 0.82 * nTot) env *= Math.max(0, (nTot - t) / Math.max(1, 0.18 * nTot));
      if (t < nOnset) {
        if (isSibilant) exc[t] = 0.28 * randU() * env;
        else if (isAspirated) exc[t] = 0.18 * randU() * env;
        else if (isNasal) {
          phase = (phase + f0 / SAMPLE_RATE) % 1.0;
          exc[t] = 0.22 * Math.sin(2 * Math.PI * phase) * env;
        } else {
          exc[t] = 0.04 * randU() * env;
        }
      } else {
        phase = (phase + f0 / SAMPLE_RATE) % 1.0;
        const glottal =
          0.52 * Math.sin(2 * Math.PI * phase) +
          0.28 * Math.sin(4 * Math.PI * phase) +
          0.14 * Math.sin(6 * Math.PI * phase) +
          0.07 * Math.sin(8 * Math.PI * phase);
        const breath = hasVisarga && t > nOnset + Math.floor(0.65 * nNuc) ? 0.12 * randU() : 0.015 * randU();
        exc[t] = (glottal + breath) * env;
      }
    }

    const r1 = biquadResonator(exc, f1, isNasal ? 150 : 90);
    const r2 = biquadResonator(r1, f2, 120);
    const r3 = biquadResonator(r2, f3, 170);
    const gap = new Float32Array(Math.max(8, Math.round((0.008 * SAMPLE_RATE) / speed)));
    chunks.push(r3, gap);
    totalSamples += r3.length + gap.length;
  }

  if (parsed.normalizedDevanagari.includes('\u0964') || parsed.normalizedDevanagari.includes('\u0965')) {
    const tail = new Float32Array(Math.round((0.14 * SAMPLE_RATE) / speed));
    chunks.push(tail);
    totalSamples += tail.length;
  }

  if (totalSamples === 0) {
    return { pcm: new Float32Array(2400), meta: { ...parsed, tier: 'klatt_zero_dep' } };
  }

  const out = new Float32Array(totalSamples);
  let offset = 0;
  for (const ch of chunks) {
    out.set(ch, offset);
    offset += ch.length;
  }

  // Remove DC bias from cascaded resonators before peak scaling
  let sum = 0.0;
  for (let idx = 0; idx < out.length; idx++) sum += out[idx];
  const mean = sum / out.length;
  for (let idx = 0; idx < out.length; idx++) out[idx] -= mean;

  let peak = 1e-5;
  for (let idx = 0; idx < out.length; idx++) {
    const a = Math.abs(out[idx]);
    if (a > peak) peak = a;
  }
  const gain = 0.82 / peak;
  for (let idx = 0; idx < out.length; idx++) out[idx] *= gain;

  return { pcm: out, meta: { ...parsed, tier: 'klatt_zero_dep' } };
}

export function encodeWavBuffer(pcm, sr = SAMPLE_RATE) {
  const numSamples = pcm.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);
  const writeStr = (off, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  let pos = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    view.setInt16(pos, Math.round(s * 32767), true);
    pos += 2;
  }
  return buffer;
}
