/**
 * Pan-Indic Akshara, Per-Language Native Brahmic & Classical/Vedic Sanskrit Phoneme G2P Front-End
 * Zero-dependency JS mirror of python/src/ninada/g2p.py.
 */
import { IPA_VOCAB, IPA_ID_TO_TOKEN, RETROFLEX_LATERAL_ID, textToIpa, ipaToIds } from './ipa_g2p.js';
export { IPA_VOCAB, IPA_ID_TO_TOKEN, RETROFLEX_LATERAL_ID, textToIpa, ipaToIds };

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
export const PAD_TOKEN = '_';
export const BLANK_ID = TOKEN_TO_ID[PAD_TOKEN] ?? 33;
export const PAD_TOKEN_ID = {
  sa: 49,
  kn: 49,
  hi: 33,
  ta: 4,
  te: 45,
  ml: 43,
  en: 0,
  mr: 33,
  bn: 0,
  gu: 0,
  pa: 0,
  or: 0,
};
export const VIRAMA_TOKEN_IDS = {
  sa: 15,  // ್ (\u0ccd)
  kn: 15,  // ್ (\u0ccd)
  hi: 53,  // ् (\u094d)
  ta: 18,  // ் (\u0bcd)
  te: 48,  // ్ (\u0c4d)
  ml: 36,  // ് (\u0d4d)
  en: -1,
  mr: 53,  // ् (\u094d)
  bn: 56,  // ् (\u09cd)
  gu: 60,  // ् (\u0acd)
  pa: 53,  // ् (\u0a4d)
  or: 58,  // ् (\u0b4d)
};

export const LANGUAGE_IDS = {
  sa: 0,
  hi: 1,
  kn: 2,
  ta: 3,
  te: 4,
  ml: 5,
  en: 6,
  mr: 7,
  bn: 8,
  gu: 9,
  pa: 10,
  or: 11,
};

function makeNativeVocab(tokens, donorChars = []) {
  const arr = [...tokens];
  const donorSet = new Set(donorChars);
  const origIncludes = arr.includes.bind(arr);
  arr.includes = (item, fromIndex) => origIncludes(item, fromIndex) || (fromIndex === undefined && donorSet.has(item));
  return arr;
}

const KN_VOCAB = makeNativeVocab([
  "'", 'ಓ', 'ಮ', 'ಆ', 'ಭ', 'ಋ', 'ಚ', 'ಟ', 'ು', 'ೇ',
  'ಜ', 'ಏ', 'ಾ', 'ಪ', 'ಶ', '್', 'ಈ', 'ಔ', 'ಞ', '1',
  'ನ', 'ಎ', 'ಸ', 'ವ', 'ಇ', 'ಘ', '5', 'ಊ', 'ಳ', 'ಖ',
  'ೃ', 'ೆ', '-', 'ತ', 'ಝ', 'ಢ', ' ', 'ಅ', 'ಧ', 'ೌ',
  'ಹ', 'ಯ', '9', 'ಫ', 'ಛ', '8', '6', 'ಲ', 'ಣ', '_',
  'ದ', 'ಬ', 'ಠ', 'ಡ', 'ಃ', '3', 'ೋ', 'ಂ', 'ಒ', 'ೀ',
  'ಕ', 'ಿ', 'ಥ', 'ರ', 'ೂ', 'ಐ', 'ೈ', 'ಉ', '2', '0',
  'ಗ', 'ೊ', 'ಷ', '4', '7', '<unk>',
], ['\u0c99']);

export const NATIVE_VOCABS = {
  kn: KN_VOCAB,
  sa: [...KN_VOCAB, '\u0c99', '\u0ce0', '\u0cc4'],
  hi: [
    'फ', '4', '1', '-', 'अ', 'इ', '8', ' ', '0', 'छ',
    'न', 'ए', 'ऐ', '़', 'ष', 'ि', 'ँ', 'च', '2', 'ऑ',
    'थ', 'भ', 'ी', '\u200d', 'ॅ', '3', 'ञ', 'ै', 'ु', 'ठ',
    'ं', 'ॉ', 'उ', '_', 'ई', 'ः', 'ह', 'ध', 'ल', 'र',
    'स', 'ब', 'ख', 'ण', "'", '`', 'व', 'घ', 'प', 'ग',
    'ढ', 'य', 'े', '्', 'ा', 'आ', 'ड', 'ज', 'झ', 'श',
    'औ', 'ो', 'द', 'ृ', 'ौ', 'ऊ', 'ू', 'ओ', 'ट', 'त',
    'क', 'म', '<unk>'
  ],
  ta: [
    '3', 'அ', 'ஈ', 'உ', '_', 'ந', 'க', ' ', 'ே', 'ா',
    'ெ', 'ஊ', 'ீ', "'", '7', 'a', 'ஏ', 'வ', '்', 'ஹ',
    'ூ', 'ல', 'ட', '2', 'ழ', 'ர', 'ம', 'ஒ', 'ற', 'ஞ',
    'ச', '9', 'ி', 'ஸ', 'ை', 'ன', '6', 'எ', 'இ', 'ய',
    'ௌ', 'த', 'ோ', 'ள', '1', 'ஆ', 'ப', '0', 'ண', 'ஓ',
    'ங', 'ு', 'ஐ', 'ஜ', '4', 'ஷ', 'ொ', '5', '<unk>'
  ],
  te: makeNativeVocab([
    'త', 'ఞ', 'భ', 'థ', 'ర', 'న', 'ో', 'ఘ', "'", 'ఐ',
    'గ', 'ే', 'జ', 'ఔ', 'హ', 'ా', 'డ', 'మ', 'స', ' ',
    'క', 'ఒ', 'ూ', 'వ', 'ధ', 'శ', 'ౌ', 'ప', 'ః', 'ఠ',
    'ఖ', 'ష', '–', 'ణ', 'ఫ', 'ి', 'ొ', 'ఏ', 'ఎ', 'ృ',
    'ఓ', 'య', 'అ', 'ీ', 'ె', '_', 'చ', 'ు', '్', 'ద',
    'ల', 'ఉ', 'ఢ', '-', 'బ', 'ఈ', 'ట', 'ఛ', 'ళ', 'ఊ',
    '6', 'ం', 'ఇ', 'ఆ', 'ై', '<unk>',
  ], ['\u0c1d']),
  ml: makeNativeVocab([
    '—', 'ത', 'ര', 'ള', 'ഴ', 'ഒ', '0', 'ൂ', 'ഖ', 'ഞ',
    'ഇ', 'ഡ', 'ി', 'ഊ', 'ച', 'ാ', 'യ', 'ഷ', 'i', 'ഉ',
    'ഫ', 'ല', 'ഠ', '4', 'െ', 'ൗ', 'ആ', 'സ', 'ു', 'ം',
    'm', 'ജ', 'ധ', '-', 'ഹ', 'ഛ', '്', 'ട', 'ഃ', 'ഗ',
    'ണ', 'ൽ', 'ോ', '_', 'റ', 'ഏ', '6', 'എ', 'ർ', 'ഥ',
    'ഘ', 'ഓ', 'ീ', 'ൈ', '2', 'േ', 'ക', '1', 'ൾ', 'ങ',
    'ഭ', ' ', 'ഢ', 'അ', 'മ', 'ബ', '3', 'ഈ', 'ഔ', 'പ',
    'ൺ', 'ന', 'ൊ', 'o', 'ൃ', 'c', 'വ', 'ൻ', 'ദ', '5',
    'ഐ', "'", 'q', 'ശ', '<unk>',
  ], ['\u0d1d']),
};

function buildRangeVocab(lo, hi) {
  const letters = [];
  for (let cp = lo; cp < hi; cp++) {
    const ch = String.fromCodePoint(cp);
    if (/[\p{L}\p{M}]/u.test(ch)) letters.push(ch);
  }
  letters.sort();
  return ['_', ...[..." ',.-?"], ...letters, '<unk>'];
}

NATIVE_VOCABS.en = ['_', ...[..." abcdefghijklmnopqrstuvwxyz'-,.?!"], '<unk>'];
NATIVE_VOCABS.mr = [...NATIVE_VOCABS.hi.slice(0, -1), '\u0933', '<unk>'];
NATIVE_VOCABS.bn = buildRangeVocab(0x0981, 0x09fa);
NATIVE_VOCABS.gu = buildRangeVocab(0x0a81, 0x0af2);
NATIVE_VOCABS.pa = buildRangeVocab(0x0a01, 0x0a76);
NATIVE_VOCABS.or = buildRangeVocab(0x0b01, 0x0b73);
for (const [lc, vir] of [['bn', '\u09cd'], ['gu', '\u0acd'], ['pa', '\u0a4d'], ['or', '\u0b4d']]) {
  const idx = NATIVE_VOCABS[lc].indexOf(vir);
  if (idx >= 0) VIRAMA_TOKEN_IDS[lc] = idx;
}

export const NATIVE_TOKEN_TO_ID = Object.fromEntries(
  Object.entries(NATIVE_VOCABS).map(([lc, toks]) => [lc, Object.fromEntries(toks.map((t, i) => [t, i]))])
);

export const NATIVE_DONOR_MAP = {
  sa: {
    '\u0c99': '\u0ca3',  // ಙ (ṅa) <- ಣ (ṇa)
    '\u0ce0': '\u0c8b',  // ೠ (ṝ)  <- ಋ (ṛ)
    '\u0cc4': '\u0cc3',  // ೄ (ṝ)  <- ೃ (ṛ)
  },
  kn: {
    '\u0c99': '\u0ca8',  // ಙ (ṅa) <- ನ (na)
  },
  te: {
    '\u0c1d': '\u0c1c',  // ఝ (jha) <- జ (ja)
  },
  ml: {
    '\u0d1d': '\u0d1c',  // ഝ (jha) <- ജ (ja)
  },
};

export const VEDIC_SVARA_MARKS = {
  '\u0951': 'udatta',
  '\u0952': 'anudatta',
  '\u0953': 'svarita',
  '\u0954': 'svarita',
  '\u1cd4': 'udatta',
  '\u1cda': 'udatta',
  '\u1cdb': 'udatta',
  '\u1cdc': 'anudatta',
  '\u1cdd': 'svarita',
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

const MALAYALAM_ZWJ_CHILLU = [
  ['\u0d23\u0d4d\u200d', '\u0d7a'],
  ['\u0d28\u0d4d\u200d', '\u0d7b'],
  ['\u0d30\u0d4d\u200d', '\u0d7c'],
  ['\u0d32\u0d4d\u200d', '\u0d7d'],
  ['\u0d33\u0d4d\u200d', '\u0d7e'],
];

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
  const map = {
    kannada: 'kn',
    tamil: 'ta',
    telugu: 'te',
    malayalam: 'ml',
    bengali: 'bn',
    gurmukhi: 'pa',
    gujarati: 'gu',
    odia: 'or',
  };
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

const EN_DIGIT_DEV = {
  '0': 'जीरो', '1': 'वन', '2': 'टू', '3': 'थ्री', '4': 'फोर',
  '5': 'फाइव', '6': 'सिक्स', '7': 'सेवन', '8': 'एट', '9': 'नाइन',
};

function englishToDevanagari(text) {
  let s = String(text).normalize('NFC').trim().toLowerCase();
  for (const [d, w] of Object.entries(EN_DIGIT_DEV)) {
    s = s.replace(new RegExp(`\\b${d}\\b`, 'g'), ` ${w} `);
  }
  s = s
    .replaceAll('tion', 'shan')
    .replaceAll('sion', 'zhan')
    .replaceAll('qu', 'kw')
    .replaceAll('ck', 'k')
    .replaceAll('wh', 'w');
  const enVows = [
    ['oo', '\u090a', '\u0942'],
    ['ee', '\u0908', '\u0940'],
    ['ea', '\u0908', '\u0940'],
    ['ai', '\u090f', '\u0947'],
    ['ay', '\u090f', '\u0947'],
    ['ou', '\u0914', '\u094c'],
    ['ow', '\u0913', '\u094b'],
    ['oa', '\u0913', '\u094b'],
    ['oi', '\u0913\u0907', '\u094b\u0907'],
    ['oy', '\u0913\u0907', '\u094b\u0907'],
    ['a', '\u0905', ''],
    ['e', '\u090f', '\u0947'],
    ['i', '\u0907', '\u093f'],
    ['o', '\u0913', '\u094b'],
    ['u', '\u0905', ''],
  ];
  const enCons = [
    ['tch', '\u091a'], ['sh', '\u0936'], ['ch', '\u091a'], ['th', '\u0925'],
    ['ph', '\u092b'], ['gh', '\u0917'], ['ng', '\u0919'], ['zh', '\u091d'],
    ['kh', '\u0916'], ['bh', '\u092d'], ['dh', '\u0927'],
    ['b', '\u092c'], ['d', '\u0921'], ['f', '\u092b'], ['g', '\u0917'],
    ['h', '\u0939'], ['j', '\u091c'], ['k', '\u0915'], ['l', '\u0932'],
    ['m', '\u092e'], ['n', '\u0928'], ['p', '\u092a'], ['q', '\u0915'],
    ['r', '\u0930'], ['s', '\u0938'], ['t', '\u091f'], ['v', '\u0935'],
    ['w', '\u0935'], ['x', '\u0915\u094d\u0938'], ['z', '\u091c'],
  ];
  const out = [];
  let i = 0;
  const n = s.length;
  let prevC = false;
  while (i < n) {
    const ch = s[i];
    if (/\s/.test(ch)) {
      if (prevC) { out.push(VIRAMA); prevC = false; }
      out.push(' ');
      i += 1;
      continue;
    }
    if (EN_DIGIT_DEV[ch]) {
      if (prevC) { out.push(VIRAMA); prevC = false; }
      out.push(EN_DIGIT_DEV[ch]);
      i += 1;
      continue;
    }
    if (ch === 'c') {
      const nxt = i + 1 < n ? s[i + 1] : '';
      if (prevC) out.push(VIRAMA);
      out.push(nxt === 'e' || nxt === 'i' || nxt === 'y' ? '\u0938' : '\u0915');
      prevC = true;
      i += 1;
      continue;
    }
    if (ch === 'y') {
      const nxt = i + 1 < n ? s[i + 1] : '';
      if (nxt && 'aeiou'.includes(nxt)) {
        if (prevC) out.push(VIRAMA);
        out.push('\u092f');
        prevC = true;
      } else {
        out.push(prevC ? '\u0940' : '\u0908');
        prevC = false;
      }
      i += 1;
      continue;
    }
    let mv = false;
    for (const [rom, indep, matra] of enVows) {
      if (s.startsWith(rom, i)) {
        if (rom === 'e' && i + 1 === n && prevC && out.length >= 2) {
          out.push(VIRAMA);
          prevC = false;
          i += 1;
          mv = true;
          break;
        }
        if (prevC) {
          if (matra) out.push(matra);
        } else {
          out.push(indep);
        }
        prevC = false;
        i += rom.length;
        mv = true;
        break;
      }
    }
    if (mv) continue;
    let mc = false;
    for (const [rom, dev] of enCons) {
      if (s.startsWith(rom, i)) {
        if (prevC) out.push(VIRAMA);
        out.push(dev);
        prevC = true;
        i += rom.length;
        mc = true;
        break;
      }
    }
    if (mc) continue;
    if (prevC && ch !== "'" && ch !== '\u2019') {
      out.push(VIRAMA);
      prevC = false;
    }
    i += 1;
  }
  if (prevC) out.push(VIRAMA);
  return out.join('');
}

export function brahmicToDevanagari(text) {
  let norm = String(text).normalize('NFC');
  for (const [zwjSeq, atomic] of MALAYALAM_ZWJ_CHILLU) {
    norm = norm.split(zwjSeq).join(atomic);
  }
  norm = norm
    .replace(/\u0d4c/g, '\u0d57')
    .replaceAll('\u0a74', '\u0907\u0915 \u0913\u0902\u0915\u093e\u0930')
    .replaceAll('\u09af\u09bc', '\u09df')
    .replaceAll('\u0a71', '');
  const specialCp = {
    0x09ce: '\u0924\u094d',
    0x09d7: '\u094c',
    0x09dc: '\u0921\u093c',
    0x09dd: '\u0922\u093c',
    0x09df: '\u092f',
    0x09f0: '\u0930',
    0x09f1: '\u0935',
    0x0a5c: '\u0921\u093c',
    0x0a70: '\u0902',
    0x0a72: '\u0907',
    0x0a73: '\u0909',
    0x0b56: '\u0948',
    0x0b57: '\u094c',
    0x0b5c: '\u0921\u093c',
    0x0b5d: '\u0922\u093c',
    0x0b5f: '\u092f',
    0x0b71: '\u0935',
  };
  const out = [];
  for (const ch of norm) {
    const cp = ch.codePointAt(0);
    if (MALAYALAM_CHILLU_CP[cp]) {
      out.push(MALAYALAM_CHILLU_CP[cp]);
      continue;
    }
    if (cp === 0x0D57) {
      out.push('\u094c');
      continue;
    }
    if (specialCp[cp]) {
      out.push(specialCp[cp]);
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
        if ((offset >= 0x01 && offset <= 0x4D) || offset === 0x50 || offset === 0x60 || offset === 0x62 || offset === 0x63) {
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

export function deleteHindiSchwaWord(word) {
  const mLead = word.match(/^[^\w\u0900-\u097f]+/);
  const lead = mLead ? mLead[0] : '';
  const mTrail = word.match(/[^\w\u0900-\u097f]+$/);
  const trail = mTrail ? mTrail[0] : '';
  const core = lead || trail ? word.slice(lead.length, word.length - trail.length) : word;
  if (!core) return word;

  const chars = [...core];
  const n = chars.length;
  const tokens = [];
  let i = 0;
  while (i < n) {
    const c = chars[i];
    let hasNukta = false;
    if (i + 1 < n && chars[i + 1] === '\u093c') {
      hasNukta = true;
    }
    const isC = isDevConsonant(c);
    const isV = isDevIndepVowel(c);
    const isM = isDevMatra(c);
    const isVir = (c === VIRAMA);
    tokens.push({
      char: c,
      nukta: hasNukta,
      isC,
      isV,
      isM,
      isVir,
      hasSchwa: false,
      deleteSchwa: false,
    });
    i += 1;
    if (hasNukta) i += 1;
  }

  const numTok = tokens.length;
  for (let idx = 0; idx < numTok; idx++) {
    const t = tokens[idx];
    if (t.isC) {
      if (idx + 1 < numTok) {
        const nextT = tokens[idx + 1];
        if (nextT.isVir || nextT.isM) {
          t.hasSchwa = false;
          continue;
        }
      }
      t.hasSchwa = true;
    }
  }

  const totalNuclei = tokens.filter((t) => t.isV || t.isM || t.hasSchwa).length;
  if (totalNuclei <= 1) return word;

  // 1. Word-Final Schwa Deletion (with Sanskrit Tatsama / final cluster protection)
  let lastCIdx = null;
  for (let idx = numTok - 1; idx >= 0; idx--) {
    if (tokens[idx].isC) {
      lastCIdx = idx;
      break;
    } else if (tokens[idx].isV || tokens[idx].isM) {
      break;
    }
  }

  if (lastCIdx !== null && tokens[lastCIdx].hasSchwa) {
    if (!(lastCIdx > 0 && tokens[lastCIdx - 1].isVir)) {
      tokens[lastCIdx].hasSchwa = false;
      tokens[lastCIdx].deleteSchwa = true;
    }
  }

  // 2. Medial Schwa Deletion (VC_CV evaluated right-to-left)
  let firstNucleusIdx = null;
  for (let idx = 0; idx < numTok; idx++) {
    if (tokens[idx].isV || tokens[idx].isM || tokens[idx].hasSchwa) {
      firstNucleusIdx = idx;
      break;
    }
  }

  for (let idx = numTok - 1; idx >= 0; idx--) {
    const t = tokens[idx];
    if (!t.hasSchwa) continue;
    if (t.char === '\u0939') continue; // 'ह' resists deletion
    if (firstNucleusIdx !== null && idx <= firstNucleusIdx) continue;

    if (idx > 0 && tokens[idx - 1].isVir) continue;
    if (idx >= 2 && tokens[idx - 2].isVir) continue;
    if (idx + 2 < numTok && tokens[idx + 2].isVir) continue;

    let hasPrecV = false;
    for (let p = idx - 1; p >= 0; p--) {
      if (tokens[p].isV || tokens[p].isM || tokens[p].hasSchwa) {
        hasPrecV = true;
        break;
      }
    }
    if (!hasPrecV) continue;

    let hasFollCv = false;
    let follC = null;
    for (let f = idx + 1; f < numTok; f++) {
      if (tokens[f].isC) {
        follC = f;
        break;
      }
    }
    if (follC !== null) {
      if (follC + 1 < numTok && (tokens[follC + 1].isM || tokens[follC + 1].isV)) {
        hasFollCv = true;
      }
    }

    if (hasFollCv) {
      t.hasSchwa = false;
      t.deleteSchwa = true;
    }
  }

  const out = [];
  for (const t of tokens) {
    out.push(t.char);
    if (t.nukta) out.push('\u093c');
    if (t.deleteSchwa) out.push(VIRAMA);
  }
  return lead + out.join('') + trail;
}

export function deleteHindiSchwa(text) {
  return String(text)
    .split(' ')
    .map(deleteHindiSchwaWord)
    .join(' ');
}

export function applyPhonologicalRules(devText, lang = 'sa') {
  let s = String(devText)
    .normalize('NFC')
    .replace(/[\u0951-\u0954\u1cd0-\u1cf4\u1cf7-\u1cf9]/g, '')
    .replace(/[\u1cf5\u1cf6]/g, VISARGA)
    .replace(/\u0950/g, '\u0913\u092e\u094d')
    .replace(/([\u0905-\u0914\u093e-\u094c\u0915-\u0939])\s*\u093d+/g, '$1')
    .replace(/\u093d+/g, '')
    .replace(/\u0962/g, '\u0943')
    .replace(/\u0963/g, '\u0944')
    .replace(/\u090c/g, '\u0932\u0943');

  const dOut = [];
  for (const ch of s) {
    const cp = ch.codePointAt(0);
    if (cp >= 0x0966 && cp <= 0x096F) dOut.push(String(cp - 0x0966));
    else dOut.push(ch);
  }
  s = dOut.join('');

  if (lang === 'hi' || lang === 'mr' || lang === 'gu' || lang === 'pa') {
    s = deleteHindiSchwa(s);
  } else if (lang === 'sa') {
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
  s = s.replace(/[,;:!?]+/g, ' ').replace(/\s+/g, ' ').trim();
  return s;
}

const TA_FOLD = {
  '\u0916': '\u0915', '\u0917': '\u0915', '\u0918': '\u0915',
  '\u091b': '\u091a', '\u091d': '\u091c',
  '\u0920': '\u091f', '\u0921': '\u091f', '\u0922': '\u091f',
  '\u0925': '\u0924', '\u0926': '\u0924', '\u0927': '\u0924',
  '\u092b': '\u092a', '\u092c': '\u092a', '\u092d': '\u092a',
  '\u0936': '\u0938', '\u090b': '\u0930\u093f', '\u0943': '\u0930\u093f',
  '\u0902': '\u092e\u094d', '\u0903': '',
};

function devanagariToTargetScript(devText, targetLang) {
  if (targetLang === 'hi' || targetLang === 'mr') {
    return devText.replace(/[\u0964\u0965]/g, ' ');
  }
  if (targetLang === 'en') {
    const revMap = {
      '\u0905': 'a', '\u0906': 'aa', '\u0907': 'i', '\u0908': 'ee', '\u0909': 'u', '\u090a': 'oo',
      '\u090f': 'e', '\u0910': 'ai', '\u0913': 'o', '\u0914': 'au',
      '\u093e': 'aa', '\u093f': 'i', '\u0940': 'ee', '\u0941': 'u', '\u0942': 'oo',
      '\u0947': 'e', '\u0948': 'ai', '\u094b': 'o', '\u094c': 'au',
      '\u0902': 'm', '\u0901': 'n', '\u0903': 'h',
      '\u0915': 'k', '\u0916': 'kh', '\u0917': 'g', '\u0918': 'gh', '\u0919': 'ng',
      '\u091a': 'ch', '\u091b': 'chh', '\u091c': 'j', '\u091d': 'jh', '\u091e': 'ny',
      '\u091f': 't', '\u0920': 'th', '\u0921': 'd', '\u0922': 'dh', '\u0923': 'n',
      '\u0924': 't', '\u0925': 'th', '\u0926': 'd', '\u0927': 'dh', '\u0928': 'n',
      '\u092a': 'p', '\u092b': 'f', '\u092c': 'b', '\u092d': 'bh', '\u092e': 'm',
      '\u092f': 'y', '\u0930': 'r', '\u0932': 'l', '\u0933': 'l', '\u0935': 'v',
      '\u0936': 'sh', '\u0937': 'sh', '\u0938': 's', '\u0939': 'h',
    };
    const chars = [...devText];
    const romOut = [];
    for (let i = 0; i < chars.length; i++) {
      const ch = chars[i];
      if (ch === '\u094d' || ch === '\u093c') continue;
      if (ch === '\u0964' || ch === '\u0965') { romOut.push(' '); continue; }
      romOut.push(revMap[ch] !== undefined ? revMap[ch] : ch);
      if (isDevConsonant(ch)) {
        const nxt = i + 1 < chars.length ? chars[i + 1] : '';
        if (!isDevMatra(nxt) && nxt !== VIRAMA) romOut.push('a');
      }
    }
    return romOut.join('').toLowerCase();
  }
  const baseMap = {
    sa: 0x0C80, kn: 0x0C80, ta: 0x0B80, te: 0x0C00, ml: 0x0D00,
    bn: 0x0980, pa: 0x0A00, gu: 0x0A80, or: 0x0B00,
  };
  const targetBase = baseMap[targetLang] ?? 0x0C80;
  let s = devText.replace(/\u0901/g, '\u0902');
  if (targetLang === 'bn' || targetLang === 'or') {
    s = s.replaceAll('\u0935', '\u092c');
  } else if (targetLang === 'pa') {
    s = s.replaceAll('\u0937', '\u0936').replaceAll('\u090b', '\u0930\u093f').replaceAll('\u0943', '\u0930\u093f');
  }
  if (targetLang === 'ta') {
    s = [...s].map((c) => (TA_FOLD[c] !== undefined ? TA_FOLD[c] : c)).join('');
  }
  const out = [];
  for (const ch of s) {
    if (ch === '\u0964' || ch === '\u0965') {
      out.push(' ');
      continue;
    }
    const cp = ch.codePointAt(0);
    if (cp === 0x093C) {
      if (targetLang === 'bn' || targetLang === 'pa' || targetLang === 'or') {
        out.push(String.fromCodePoint(targetBase + 0x3C));
      }
      continue;
    }
    if (cp >= 0x0901 && cp <= 0x0963) {
      const targetCh = String.fromCodePoint(targetBase + (cp - 0x0900));
      if (/[\p{L}\p{M}]/u.test(targetCh)) out.push(targetCh);
    } else {
      out.push(ch);
    }
  }
  return out.join('');
}

export function toNativeModelText(text, lang, detectedScript, phonemicDev) {
  const sRaw = String(text).normalize('NFC');
  const scriptForLang = {
    kn: 'kannada', hi: 'devanagari', mr: 'devanagari', ta: 'tamil', te: 'telugu',
    ml: 'malayalam', en: 'latin', bn: 'bengali', gu: 'gujarati', pa: 'gurmukhi', or: 'odia',
  };
  let native;

  if (lang === 'sa') {
    // Preserve homorganic nasal+stop clusters (ಙ್, ಞ್, ಣ್, ನ್, ಮ್) for accurate Sanskrit articulation
    native = devanagariToTargetScript(phonemicDev, 'sa');
  } else if (lang === 'en') {
    if (detectedScript === 'latin' || detectedScript === 'iast') {
      native = sRaw.toLowerCase();
    } else {
      native = devanagariToTargetScript(phonemicDev, 'en');
    }
  } else if (detectedScript === scriptForLang[lang]) {
    native = sRaw;
    if (lang === 'hi') {
      native = deleteHindiSchwa(native);
    } else if (lang === 'ml') {
      for (const [zwjSeq, atomic] of MALAYALAM_ZWJ_CHILLU) {
        native = native.split(zwjSeq).join(atomic);
      }
      native = native
        .replace(/[\u200c\u200d]/g, '')
        .replace(/\u0d4c/g, '\u0d57')
        .replace(/\u0d7f/g, '\u0d15\u0d4d')
        .replace(/\u0d0b/g, '\u0d31\u0d41');
    } else if (lang === 'te') {
      native = native
        .replace(/\u0c0b/g, '\u0c30\u0c41')
        .replace(/\u0c19/g, '\u0c02');
    } else if (lang === 'ta') {
      native = native.toLowerCase().replace(/\u0b83/g, '\u0bb9\u0bcd');
    } else if (lang === 'pa') {
      native = native.replaceAll('\u0a74', '\u0a07\u0a71\u0a15 \u0a13\u0a05\u0a70\u0a15\u0a3e\u0a30');
    }
  } else {
    native = devanagariToTargetScript(phonemicDev, lang);
    if (lang === 'hi') {
      native = deleteHindiSchwa(native);
    } else if (lang === 'ml') {
      native = native
        .replace(/\u0d4c/g, '\u0d57')
        .replace(/\u0d0b/g, '\u0d31\u0d41');
    } else if (lang === 'te') {
      native = native
        .replace(/\u0c0b/g, '\u0c30\u0c41')
        .replace(/\u0c19/g, '\u0c02');
    } else if (lang === 'kn') {
      // Preserve ಙ
    }
  }

  const normDigits = [];
  for (const ch of native) {
    const cp = ch.codePointAt(0);
    let mappedD = null;
    for (const [lo] of BRAHMIC_BASES) {
      if (cp >= lo + 0x66 && cp <= lo + 0x6F) {
        mappedD = String(cp - (lo + 0x66));
        break;
      }
    }
    normDigits.push(mappedD !== null ? mappedD : ch);
  }
  native = normDigits.join('');

  native = native.replace(/[\u0964\u0965|.,;:!?]+/g, ' ');
  native = native.replace(/\s+/g, ' ').trim();
  return native;
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
      let akSvara = null;
      while (
        i < n &&
        (chars[i] === ANUSVARA ||
          chars[i] === VISARGA ||
          chars[i] === CHANDRABINDU ||
          VEDIC_SVARA_MARKS[chars[i]] !== undefined)
      ) {
        if (VEDIC_SVARA_MARKS[chars[i]] !== undefined) {
          akSvara = VEDIC_SVARA_MARKS[chars[i]];
        } else {
          if (chars[i] === ANUSVARA || chars[i] === VISARGA) isLong = true;
          buf += chars[i];
        }
        i += 1;
      }
      aksharas.push({ akshara: buf, weight: isLong ? 'G' : 'L', svara: akSvara });
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
        aksharas.push({ akshara: buf, weight: 'C', svara: null });
        continue;
      }
      if (i < n && isDevMatra(chars[i])) {
        if (LONG_MATRAS.has(chars[i])) isLong = true;
        buf += chars[i];
        i += 1;
      }
      let akSvara = null;
      while (
        i < n &&
        (chars[i] === ANUSVARA ||
          chars[i] === VISARGA ||
          chars[i] === CHANDRABINDU ||
          VEDIC_SVARA_MARKS[chars[i]] !== undefined)
      ) {
        if (VEDIC_SVARA_MARKS[chars[i]] !== undefined) {
          akSvara = VEDIC_SVARA_MARKS[chars[i]];
        } else {
          if (chars[i] === ANUSVARA || chars[i] === VISARGA) isLong = true;
          buf += chars[i];
        }
        i += 1;
      }
      let j = i;
      while (j < n && /\s/.test(chars[j])) j += 1;
      if (j + 1 < n && isDevConsonant(chars[j]) && chars[j + 1] === VIRAMA) {
        isLong = true;
      }
      aksharas.push({ akshara: buf, weight: isLong ? 'G' : 'L', svara: akSvara });
      continue;
    }
    i += 1;
  }
  return aksharas;
}

export function textToIds(
  text,
  {
    lang = null,
    addBlank = true,
    minLen = 8,
    blankId = null,
    legacyBlank = false,
  } = {}
) {
  const [detectedScript, detectedLang] = detectScriptAndLang(text, lang || 'sa');
  const activeLang = lang && LANGUAGE_IDS[lang] !== undefined ? lang : detectedLang;
  const dev = activeLang === 'en' && detectedScript === 'latin'
    ? englishToDevanagari(text)
    : (detectedScript === 'iast' || detectedScript === 'latin')
    ? iastToDevanagari(text)
    : brahmicToDevanagari(text);
  const aksharas = segmentAksharas(dev);
  const phonemic = applyPhonologicalRules(dev, activeLang);

  const tokens = [];
  for (const ch of phonemic) {
    if (TOKEN_TO_ID[ch] !== undefined) tokens.push(ch);
  }
  if (tokens.length === 0) tokens.push('\u0905');

  const nativeText = toNativeModelText(text, activeLang, detectedScript, phonemic);
  const langMap = NATIVE_TOKEN_TO_ID[activeLang];
  const donorMap = NATIVE_DONOR_MAP[activeLang] ?? {};
  const rawIds = [];
  const nativeTokens = [];
  for (const ch of nativeText) {
    if (langMap[ch] !== undefined) {
      rawIds.push(langMap[ch]);
      nativeTokens.push(ch);
    } else if (donorMap[ch] !== undefined && langMap[donorMap[ch]] !== undefined) {
      rawIds.push(langMap[donorMap[ch]]);
      nativeTokens.push(ch);
    }
  }
  if (rawIds.length === 0) {
    const fb = {
      sa: '\u0c85', kn: '\u0c85', hi: '\u0905', mr: '\u0905', ta: '\u0b85',
      te: '\u0c05', ml: '\u0d05', en: 'a', bn: '\u0985', gu: '\u0a85', pa: '\u0a05', or: '\u0b05',
    }[activeLang];
    rawIds.push(langMap[fb]);
    nativeTokens.push(fb);
  }

  const effectiveBlankId = legacyBlank
    ? 0
    : blankId !== null
    ? blankId
    : PAD_TOKEN_ID[activeLang] ?? BLANK_ID;

  let inputIds;
  let isBlank;
  let tokenTypeIds;

  if (addBlank) {
    const spId = langMap[' '];
    if (spId !== undefined && rawIds[rawIds.length - 1] !== spId) {
      rawIds.push(spId);
      nativeTokens.push(' ');
    }
    inputIds = [effectiveBlankId];
    isBlank = [true];
    tokenTypeIds = [1];
    for (const tid of rawIds) {
      inputIds.push(tid, effectiveBlankId);
      isBlank.push(false, true);
      tokenTypeIds.push(0, 1);
    }
  } else {
    inputIds = [...rawIds];
    isBlank = new Array(inputIds.length).fill(false);
    tokenTypeIds = new Array(inputIds.length).fill(0);
  }
  while (inputIds.length < minLen) {
    inputIds.push(effectiveBlankId);
    isBlank.push(true);
    tokenTypeIds.push(1);
  }

  const viramaTok = {
    sa: '\u0ccd',
    kn: '\u0ccd',
    hi: '\u094d',
    mr: '\u094d',
    ta: '\u0bcd',
    te: '\u0c4d',
    ml: '\u0d4d',
    bn: '\u09cd',
    gu: '\u0acd',
    pa: '\u0a4d',
    or: '\u0b4d',
  }[activeLang] ?? '';
  const viramaId = viramaTok ? (langMap[viramaTok] ?? -1) : -1;
  const isVirama = inputIds.map((tid) => (tid === viramaId ? 1 : 0));

  const ipa = textToIpa(text, activeLang, activeLang !== 'en' ? dev : null);
  const ipaIds = ipaToIds(ipa);

  return {
    inputIds,
    attentionMask: new Array(inputIds.length).fill(1),
    ipa,
    ipaIds,
    ipa_ids: ipaIds,
    isBlank,
    is_blank: isBlank,
    tokenTypeIds,
    token_type_ids: tokenTypeIds,
    isVirama,
    is_virama: isVirama,
    svaras: aksharas.map((a) => a.svara ?? null),
    lang: activeLang,
    langId: LANGUAGE_IDS[activeLang],
    script: detectedScript,
    normalizedDevanagari: phonemic,
    nativeText,
    tokens,
    nativeTokens,
    aksharas,
  };
}
