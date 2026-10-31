/**
 * Pan-Indic Akshara & Classical/Vedic Sanskrit Phoneme G2P Front-End
 * Zero-dependency JS mirror of python/src/ninada/g2p.py (87 tokens).
 */

export const VOCAB_TOKENS = [
  '\u092b', '4', '1', '-', '\u0905', '\u0907', '8', ' ', '0', '\u091b',
  '\u0928', '\u090f', '\u0910', '\u093c', '\u0937', '\u093f', '\u0901', '\u091a', '2', '\u0911',
  '\u0925', '\u092d', '\u0940', '\u200d', '\u0945', '3', '\u091e', '\u0948', '\u0941', '\u0920',
  '\u0902', '\u0949', '\u0909', '_', '\u0908', '\u0903', '\u0939', '\u0927', '\u0932', '\u0930',
  '\u0938', '\u092c', '\u0916', '\u0923', "'", '`', '\u0935', '\u0918', '\u092a', '\u0917',
  '\u0922', '\u092f', '\u0947', '\u094d', '\u093e', '\u0906', '\u0921', '\u091c', '\u091d', '\u0936',
  '\u0914', '\u094b', '\u0926', '\u0943', '\u094c', '\u090a', '\u0942', '\u0913', '\u091f', '\u0924',
  '\u0915', '\u092e', '<unk>',
  // Extended Pan-Indic & Classical/Vedic Sanskrit symbols (73..86)
  '\u0933', '\u0934', '\u0931', '\u0929', '\u090e', '\u0912', '\u0946', '\u094a', '\u0960', '\u0944', '\u0964', '\u0965', '\u090b', '\u0919'
];

export const TOKEN_TO_ID = Object.fromEntries(VOCAB_TOKENS.map((t, i) => [t, i]));
export const BLANK_ID = 0;

export const LANGUAGE_IDS = {
  sa: 0,
  hi: 1,
  kn: 2,
  ta: 3,
  te: 4,
  ml: 5,
};

const BRAHMIC_BASES = [
  [0x0900, 0x097F, 'devanagari'],
  [0x0980, 0x09FF, 'bengali'],
  [0x0A00, 0x0A7F, 'gurmukhi'],
  [0x0A80, 0x0AFF, 'gujarati'],
  [0x0B00, 0x0B7F, 'odia'],
  [0x0B80, 0x0BFF, 'tamil'],
  [0x0C00, 0x0C7F, 'telugu'],
  [0x0C80, 0x0CFF, 'kannada'],
  [0x0D00, 0x0D7F, 'malayalam'],
];

const MALAYALAM_CHILLU_CP = {
  0x0D7A: '\u0923\u094d',
  0x0D7B: '\u0928\u094d',
  0x0D7C: '\u0930\u094d',
  0x0D7D: '\u0932\u094d',
  0x0D7E: '\u0933\u094d',
  0x0D7F: '\u0915\u094d',
};

const VIRAMA = '\u094d';
const ANUSVARA = '\u0902';
const VISARGA = '\u0903';
const CHANDRABINDU = '\u0901';

function isDevConsonant(ch) {
  if (!ch) return false;
  const cp = ch.codePointAt(0);
  return (cp >= 0x0915 && cp <= 0x0939) || cp === 0x0931 || cp === 0x0934 || cp === 0x0929;
}

function isDevIndepVowel(ch) {
  if (!ch) return false;
  const cp = ch.codePointAt(0);
  return (cp >= 0x0904 && cp <= 0x0914) || cp === 0x0960 || cp === 0x090c;
}

function isDevMatra(ch) {
  if (!ch) return false;
  const cp = ch.codePointAt(0);
  return (cp >= 0x093E && cp <= 0x094C) || cp === 0x0944 || cp === 0x0962 || cp === 0x0963;
}

const IAST_CONSONANTS = [
  ['k\u1e63', '\u0915\u094d\u0937'],
  ['j\u00f1', '\u091c\u094d\u091e'],
  ['kh', '\u0916'], ['gh', '\u0918'], ['\u1e45', '\u0919'],
  ['ch', '\u091b'], ['jh', '\u091d'], ['\u00f1', '\u091e'],
  ['\u1e6dh', '\u0920'], ['\u1e0dh', '\u0922'], ['\u1e47', '\u0923'], ['\u1e6d', '\u091f'], ['\u1e0d', '\u0921'],
  ['th', '\u0925'], ['dh', '\u0927'],
  ['ph', '\u092b'], ['bh', '\u092d'],
  ['\u015b', '\u0936'], ['\u1e63', '\u0937'], ['\u1e3b', '\u0934'], ['\u1e37', '\u0933'], ['\u1e5f', '\u0931'], ['\u1e49', '\u0929'],
  ['k', '\u0915'], ['g', '\u0917'], ['c', '\u091a'], ['j', '\u091c'],
  ['t', '\u0924'], ['d', '\u0926'], ['n', '\u0928'],
  ['p', '\u092a'], ['b', '\u092c'], ['m', '\u092e'],
  ['y', '\u092f'], ['r', '\u0930'], ['l', '\u0932'], ['v', '\u0935'], ['w', '\u0935'],
  ['s', '\u0938'], ['h', '\u0939'],
];

const IAST_VOWELS = [
  ['ai', '\u0910', '\u0948'], ['au', '\u0914', '\u094c'],
  ['\u0101', '\u0906', '\u093e'], ['\u012b', '\u0908', '\u0940'], ['\u016b', '\u090a', '\u0942'],
  ['\u1e5d', '\u0960', '\u0944'], ['\u1e5b', '\u090b', '\u0943'],
  ['\u0113', '\u090f', '\u0947'], ['\u014d', '\u0913', '\u094b'],
  ['a', '\u0905', ''], ['i', '\u0907', '\u093f'], ['u', '\u0909', '\u0941'],
  ['e', '\u090f', '\u0947'], ['o', '\u0913', '\u094b'],
];

export function detectScriptAndLang(text, defaultLang = 'sa') {
  const counts = {};
  for (const ch of String(text)) {
    const cp = ch.codePointAt(0);
    for (const [lo, hi, sname] of BRAHMIC_BASES) {
      if (cp >= lo && cp <= hi) {
        counts[sname] = (counts[sname] || 0) + 1;
        break;
      }
    }
  }
  const entries = Object.entries(counts);
  if (entries.length === 0) {
    const lower = String(text).toLowerCase();
    const hasDiacritic = /[\u0101\u012b\u016b\u1e5b\u1e5d\u1e37\u015b\u1e63\u1e6d\u1e0d\u1e47\u1e45\u00f1\u1e41\u1e25]/.test(lower);
    return [hasDiacritic ? 'iast' : 'latin', defaultLang];
  }
  entries.sort((a, b) => b[1] - a[1]);
  const dom = entries[0][0];
  const map = { kannada: 'kn', tamil: 'ta', telugu: 'te', malayalam: 'ml' };
  return [dom, map[dom] || defaultLang];
}

export function iastToDevanagari(text) {
  let s = String(text).normalize('NFC').trim().toLowerCase();
  s = s.replace(/o\u1e41/g, '\u0913\u092e\u094d').replace(/\bom\b/g, '\u0913\u092e\u094d');
  const out = [];
  let i = 0;
  const n = s.length;
  let prevCons = false;

  while (i < n) {
    const c = s[i];
    if (/\s/.test(c)) {
      if (prevCons) { out.push(VIRAMA); prevCons = false; }
      out.push(' ');
      i += 1;
      continue;
    }
    if (c === '|' || c === '/') {
      if (prevCons) { out.push(VIRAMA); prevCons = false; }
      if (i + 1 < n && s[i + 1] === c) {
        out.push('\u0965');
        i += 2;
      } else {
        out.push('\u0964');
        i += 1;
      }
      continue;
    }
    if (c === "'" || c === '\u2019') {
      out.push('\u093d');
      prevCons = false;
      i += 1;
      continue;
    }
    if (c === '\u1e41' || c === '\u1e43') {
      out.push(ANUSVARA);
      prevCons = false;
      i += 1;
      continue;
    }
    if (c === '\u1e25') {
      out.push(VISARGA);
      prevCons = false;
      i += 1;
      continue;
    }

    let matchedV = false;
    for (const [rom, indep, matra] of IAST_VOWELS) {
      if (s.startsWith(rom, i)) {
        if (prevCons) {
          if (matra) out.push(matra);
        } else {
          out.push(indep);
        }
        prevCons = false;
        i += rom.length;
        matchedV = true;
        break;
      }
    }
    if (matchedV) continue;

    let matchedC = false;
    for (const [rom, dev] of IAST_CONSONANTS) {
      if (s.startsWith(rom, i)) {
        if (prevCons) out.push(VIRAMA);
        out.push(dev);
        prevCons = true;
        i += rom.length;
        matchedC = true;
        break;
      }
    }
    if (matchedC) continue;

    if (prevCons) { out.push(VIRAMA); prevCons = false; }
    if (c === '-' || c === ',' || c === '.') out.push('-');
    i += 1;
  }
  if (prevCons) out.push(VIRAMA);
  return out.join('');
}

export function brahmicToDevanagari(text) {
  const norm = String(text).normalize('NFC');
  const out = [];
  for (const ch of norm) {
    const cp = ch.codePointAt(0);
    if (MALAYALAM_CHILLU_CP[cp]) {
      out.push(MALAYALAM_CHILLU_CP[cp]);
      continue;
    }
    if (cp >= 0x0900 && cp <= 0x097F) {
      out.push(ch);
      continue;
    }
    let mapped = false;
    for (let b = 1; b < BRAHMIC_BASES.length; b++) {
      const [lo, hi] = BRAHMIC_BASES[b];
      if (cp >= lo && cp <= hi) {
        const offset = cp - lo;
        if (offset >= 0x01 && offset <= 0x4D) {
          out.push(String.fromCodePoint(0x0900 + offset));
        } else if (offset >= 0x66 && offset <= 0x6F) {
          out.push(String(offset - 0x66));
        } else {
          out.push(' ');
        }
        mapped = true;
        break;
      }
    }
    if (!mapped) out.push(ch);
  }
  return out.join('');
}

function homorganicNasal(nextConsonant) {
  const cp = nextConsonant.codePointAt(0);
  if (cp >= 0x0915 && cp <= 0x0919) return '\u0919\u094d';
  if (cp >= 0x091A && cp <= 0x091E) return '\u091e\u094d';
  if (cp >= 0x091F && cp <= 0x0923) return '\u0923\u094d';
  if (cp >= 0x0924 && cp <= 0x0928) return '\u0928\u094d';
  if (cp >= 0x092A && cp <= 0x092E) return '\u092e\u094d';
  return ANUSVARA;
}

export function applyPhonologicalRules(devText, lang = 'sa') {
  let s = String(devText)
    .replace(/[\u0951-\u0954\u1cd0-\u1cf4\u1cf7-\u1cf9]/g, '')
    .replace(/[\u1cf5\u1cf6]/g, VISARGA)
    .replace(/\u0950/g, '\u0913\u092e\u094d')
    .replace(/([\u0947\u094b\u093e\u090f\u0913\u0906])\u093d+/g, '$1')
    .replace(/\u093d/g, '\u0905')
    .replace(/\u0962/g, '\u093f')
    .replace(/\u0963/g, '\u0940')
    .replace(/\u090c/g, '\u0932\u0943');

  const dOut = [];
  for (const ch of s) {
    const cp = ch.codePointAt(0);
    if (cp >= 0x0966 && cp <= 0x096F) dOut.push(String(cp - 0x0966));
    else dOut.push(ch);
  }
  s = dOut.join('');

  if (lang === 'sa') {
    const chars = [...s];
    const res = [];
    for (let idx = 0; idx < chars.length; idx++) {
      const ch = chars[idx];
      if (ch === ANUSVARA && idx + 1 < chars.length && isDevConsonant(chars[idx + 1])) {
        const nas = homorganicNasal(chars[idx + 1]);
        if (TOKEN_TO_ID[nas[0]] !== undefined) res.push(nas);
        else res.push(ANUSVARA);
      } else {
        res.push(ch);
      }
    }
    s = res.join('');
  } else if (lang === 'ta') {
    const pairs = [
      ['\u0919\u094d\u0915', '\u0919\u094d\u0917'], ['\u0902\u0915', '\u0902\u0917'],
      ['\u091e\u094d\u091b', '\u091e\u094d\u091d'], ['\u091e\u094d\u091a', '\u091e\u094d\u091c'], ['\u0902\u091a', '\u0902\u091c'],
      ['\u0923\u094d\u091f', '\u0923\u094d\u0921'], ['\u0902\u091f', '\u0902\u0921'],
      ['\u0928\u094d\u0924', '\u0928\u094d\u0926'], ['\u0929\u094d\u0924', '\u0928\u094d\u0926'], ['\u0902\u0924', '\u0902\u0926'],
      ['\u092e\u094d\u092a', '\u092e\u094d\u092c'], ['\u0902\u092a', '\u0902\u092c'],
    ];
    for (const [src, dst] of pairs) s = s.split(src).join(dst);
  }

  s = s.replace(/\|\|/g, '\u0965').replace(/\|/g, '\u0964');
  s = s.replace(/[,;:!?]+/g, ' - ').replace(/\s+/g, ' ').trim();
  return s;
}

const LONG_VOWELS = new Set(['\u0906', '\u0908', '\u090a', '\u0960', '\u090f', '\u0910', '\u0913', '\u0914']);
const LONG_MATRAS = new Set(['\u093e', '\u0940', '\u0942', '\u0944', '\u0947', '\u0948', '\u094b', '\u094c']);

export function segmentAksharas(devText) {
  const chars = [...String(devText)];
  const aksharas = [];
  let i = 0;
  const n = chars.length;

  while (i < n) {
    const ch = chars[i];
    if (/\s/.test(ch) || ch === '-' || ch === '\u0964' || ch === '\u0965') {
      i += 1;
      continue;
    }
    if (isDevIndepVowel(ch)) {
      let buf = ch;
      i += 1;
      let isLong = LONG_VOWELS.has(ch);
      while (i < n && (chars[i] === ANUSVARA || chars[i] === VISARGA || chars[i] === CHANDRABINDU)) {
        if (chars[i] === ANUSVARA || chars[i] === VISARGA) isLong = true;
        buf += chars[i];
        i += 1;
      }
      aksharas.push({ akshara: buf, weight: isLong ? 'G' : 'L' });
      continue;
    }
    if (isDevConsonant(ch)) {
      let buf = ch;
      i += 1;
      while (i + 1 < n && chars[i] === VIRAMA && isDevConsonant(chars[i + 1])) {
        buf += chars[i] + chars[i + 1];
        i += 2;
      }
      let isLong = false;
      if (i < n && chars[i] === VIRAMA) {
        buf += VIRAMA;
        i += 1;
        if (aksharas.length > 0) aksharas[aksharas.length - 1].weight = 'G';
        aksharas.push({ akshara: buf, weight: 'C' });
        continue;
      }
      if (i < n && isDevMatra(chars[i])) {
        if (LONG_MATRAS.has(chars[i])) isLong = true;
        buf += chars[i];
        i += 1;
      }
      while (i < n && (chars[i] === ANUSVARA || chars[i] === VISARGA || chars[i] === CHANDRABINDU)) {
        if (chars[i] === ANUSVARA || chars[i] === VISARGA) isLong = true;
        buf += chars[i];
        i += 1;
      }
      let j = i;
      while (j < n && /\s/.test(chars[j])) j += 1;
      if (j + 1 < n && isDevConsonant(chars[j]) && chars[j + 1] === VIRAMA) {
        isLong = true;
      }
      aksharas.push({ akshara: buf, weight: isLong ? 'G' : 'L' });
      continue;
    }
    i += 1;
  }
  return aksharas;
}

export function textToIds(text, { lang = null, addBlank = true, minLen = 8 } = {}) {
  const [detectedScript, detectedLang] = detectScriptAndLang(text, lang || 'sa');
  const activeLang = lang && LANGUAGE_IDS[lang] !== undefined ? lang : detectedLang;
  const dev = (detectedScript === 'iast' || detectedScript === 'latin')
    ? iastToDevanagari(text)
    : brahmicToDevanagari(text);
  const phonemic = applyPhonologicalRules(dev, activeLang);

  const rawIds = [];
  const tokens = [];
  for (const ch of phonemic) {
    if (TOKEN_TO_ID[ch] !== undefined) {
      rawIds.push(TOKEN_TO_ID[ch]);
      tokens.push(ch);
    }
  }
  if (rawIds.length === 0) {
    rawIds.push(TOKEN_TO_ID['\u0905']);
    tokens.push('\u0905');
  }

  let inputIds;
  if (addBlank) {
    inputIds = [BLANK_ID];
    for (const tid of rawIds) {
      inputIds.push(tid, BLANK_ID);
    }
  } else {
    inputIds = [...rawIds];
  }
  while (inputIds.length < minLen) inputIds.push(BLANK_ID);

  return {
    inputIds,
    attentionMask: new Array(inputIds.length).fill(1),
    lang: activeLang,
    langId: LANGUAGE_IDS[activeLang],
    script: detectedScript,
    normalizedDevanagari: phonemic,
    tokens,
    aksharas: segmentAksharas(phonemic),
  };
}
