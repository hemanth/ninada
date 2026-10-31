// Auto-generated ESM bundle for ninada
/**
 * Pan-Indic Akshara, Per-Language Native Brahmic & Classical/Vedic Sanskrit Phoneme G2P Front-End
 * Zero-dependency JS mirror of python/src/ninada/g2p.py.
 */

const VOCAB_TOKENS = [
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

const TOKEN_TO_ID = Object.fromEntries(VOCAB_TOKENS.map((t, i) => [t, i]));
const BLANK_ID = 0;

const LANGUAGE_IDS = {
  sa: 0,
  hi: 1,
  kn: 2,
  ta: 3,
  te: 4,
  ml: 5,
};

const KN_VOCAB = [
  "'", 'ಓ', 'ಮ', 'ಆ', 'ಭ', 'ಋ', 'ಚ', 'ಟ', 'ು', 'ೇ',
  'ಜ', 'ಏ', 'ಾ', 'ಪ', 'ಶ', '್', 'ಈ', 'ಔ', 'ಞ', '1',
  'ನ', 'ಎ', 'ಸ', 'ವ', 'ಇ', 'ಘ', '5', 'ಊ', 'ಳ', 'ಖ',
  'ೃ', 'ೆ', '-', 'ತ', 'ಝ', 'ಢ', ' ', 'ಅ', 'ಧ', 'ೌ',
  'ಹ', 'ಯ', '9', 'ಫ', 'ಛ', '8', '6', 'ಲ', 'ಣ', '_',
  'ದ', 'ಬ', 'ಠ', 'ಡ', 'ಃ', '3', 'ೋ', 'ಂ', 'ಒ', 'ೀ',
  'ಕ', 'ಿ', 'ಥ', 'ರ', 'ೂ', 'ಐ', 'ೈ', 'ಉ', '2', '0',
  'ಗ', 'ೊ', 'ಷ', '4', '7', '<unk>'
];

const NATIVE_VOCABS = {
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
  te: [
    'త', 'ఞ', 'భ', 'థ', 'ర', 'న', 'ో', 'ఘ', "'", 'ఐ',
    'గ', 'ే', 'జ', 'ఔ', 'హ', 'ా', 'డ', 'మ', 'స', ' ',
    'క', 'ఒ', 'ూ', 'వ', 'ధ', 'శ', 'ౌ', 'ప', 'ః', 'ఠ',
    'ఖ', 'ష', '–', 'ణ', 'ఫ', 'ి', 'ొ', 'ఏ', 'ఎ', 'ృ',
    'ఓ', 'య', 'అ', 'ీ', 'ె', '_', 'చ', 'ు', '్', 'ద',
    'ల', 'ఉ', 'ఢ', '-', 'బ', 'ఈ', 'ట', 'ఛ', 'ళ', 'ఊ',
    '6', 'ం', 'ఇ', 'ఆ', 'ై', '<unk>'
  ],
  ml: [
    '—', 'ത', 'ര', 'ള', 'ഴ', 'ഒ', '0', 'ൂ', 'ഖ', 'ഞ',
    'ഇ', 'ഡ', 'ി', 'ഊ', 'ച', 'ാ', 'യ', 'ഷ', 'i', 'ഉ',
    'ഫ', 'ല', 'ഠ', '4', 'െ', 'ൗ', 'ആ', 'സ', 'ു', 'ം',
    'm', 'ജ', 'ധ', '-', 'ഹ', 'ഛ', '്', 'ട', 'ഃ', 'ഗ',
    'ണ', 'ൽ', 'ോ', '_', 'റ', 'ഏ', '6', 'എ', 'ർ', 'ഥ',
    'ഘ', 'ഓ', 'ീ', 'ൈ', '2', 'േ', 'ക', '1', 'ൾ', 'ങ',
    'ഭ', ' ', 'ഢ', 'അ', 'മ', 'ബ', '3', 'ഈ', 'ഔ', 'പ',
    'ൺ', 'ന', 'ൊ', 'o', 'ൃ', 'c', 'വ', 'ൻ', 'ദ', '5',
    'ഐ', "'", 'q', 'ശ', '<unk>'
  ],
};

const NATIVE_TOKEN_TO_ID = Object.fromEntries(
  Object.entries(NATIVE_VOCABS).map(([lc, toks]) => [lc, Object.fromEntries(toks.map((t, i) => [t, i]))])
);

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

function detectScriptAndLang(text, defaultLang = 'sa') {
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

function iastToDevanagari(text) {
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

function brahmicToDevanagari(text) {
  let norm = String(text).normalize('NFC');
  for (const [zwjSeq, atomic] of MALAYALAM_ZWJ_CHILLU) {
    norm = norm.split(zwjSeq).join(atomic);
  }
  norm = norm.replace(/\u0d4c/g, '\u0d57');
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
    if (cp >= 0x0900 && cp <= 0x097F) {
      out.push(ch);
      continue;
    }
    let mapped = false;
    for (let b = 1; b < BRAHMIC_BASES.length; b++) {
      const [lo, hi] = BRAHMIC_BASES[b];
      if (cp >= lo && cp <= hi) {
        const offset = cp - lo;
        if ((offset >= 0x01 && offset <= 0x4D) || offset === 0x60 || offset === 0x62 || offset === 0x63) {
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

function applyPhonologicalRules(devText, lang = 'sa') {
  let s = String(devText)
    .normalize('NFC')
    .replace(/[\u0951-\u0954\u1cd0-\u1cf4\u1cf7-\u1cf9]/g, '')
    .replace(/[\u1cf5\u1cf6]/g, VISARGA)
    .replace(/\u0950/g, '\u0913\u092e\u094d')
    .replace(/([\u0947\u094b\u093e\u090f\u0913\u0906])\u093d+/g, '$1')
    .replace(/\u093d/g, '\u0905')
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
  if (targetLang === 'hi') {
    return devText.replace(/[\u0964\u0965]/g, ' - ');
  }
  const baseMap = { sa: 0x0C80, kn: 0x0C80, ta: 0x0B80, te: 0x0C00, ml: 0x0D00 };
  const targetBase = baseMap[targetLang] ?? 0x0C80;
  let s = devText.replace(/\u0901/g, '\u0902');
  if (targetLang === 'ta') {
    s = [...s].map((c) => (TA_FOLD[c] !== undefined ? TA_FOLD[c] : c)).join('');
  }
  const out = [];
  for (const ch of s) {
    if (ch === '\u0964' || ch === '\u0965') {
      out.push(targetLang === 'ta' ? ' ' : ' - ');
      continue;
    }
    const cp = ch.codePointAt(0);
    if (cp === 0x093C) continue;
    if (cp >= 0x0901 && cp <= 0x0963) {
      out.push(String.fromCodePoint(targetBase + (cp - 0x0900)));
    } else {
      out.push(ch);
    }
  }
  return out.join('');
}

function toNativeModelText(text, lang, detectedScript, phonemicDev) {
  const sRaw = String(text).normalize('NFC');
  const scriptForLang = { kn: 'kannada', hi: 'devanagari', ta: 'tamil', te: 'telugu', ml: 'malayalam' };
  let native;

  if (lang === 'sa') {
    native = devanagariToTargetScript(phonemicDev, 'sa').replace(
      /[\u0c99\u0c9e\u0ca3\u0ca8\u0cae]\u0ccd(?=[\u0c95-\u0c98\u0c9a-\u0c9d\u0c9f-\u0ca2\u0ca4-\u0ca7\u0caa-\u0cad])/g,
      '\u0c82'
    );
  } else if (detectedScript === scriptForLang[lang]) {
    native = sRaw;
    if (lang === 'ml') {
      for (const [zwjSeq, atomic] of MALAYALAM_ZWJ_CHILLU) {
        native = native.split(zwjSeq).join(atomic);
      }
      native = native
        .replace(/[\u200c\u200d]/g, '')
        .replace(/\u0d4c/g, '\u0d57')
        .replace(/\u0d7f/g, '\u0d15\u0d4d')
        .replace(/\u0d0b/g, '\u0d31\u0d41')
        .replace(/\u0d1d/g, '\u0d1c');
    } else if (lang === 'te') {
      native = native
        .replace(/\u0c0b/g, '\u0c30\u0c41')
        .replace(/\u0c1d/g, '\u0c1c')
        .replace(/\u0c19/g, '\u0c02');
    } else if (lang === 'ta') {
      native = native.toLowerCase().replace(/\u0b83/g, '\u0bb9\u0bcd');
    } else if (lang === 'kn') {
      native = native.replace(/\u0c99/g, '\u0ca8');
    }
  } else {
    native = devanagariToTargetScript(phonemicDev, lang);
    if (lang === 'ml') {
      native = native
        .replace(/\u0d4c/g, '\u0d57')
        .replace(/\u0d0b/g, '\u0d31\u0d41')
        .replace(/\u0d1d/g, '\u0d1c');
    } else if (lang === 'te') {
      native = native
        .replace(/\u0c0b/g, '\u0c30\u0c41')
        .replace(/\u0c1d/g, '\u0c1c')
        .replace(/\u0c19/g, '\u0c02');
    } else if (lang === 'kn') {
      native = native.replace(/\u0c99/g, '\u0ca8');
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

  const sep = lang === 'ta' ? ' ' : ' - ';
  native = native.replace(/[\u0964\u0965|.,;:!?]+/g, sep);
  if (lang !== 'ta') {
    native = native.replace(/\s*-\s*(?:-\s*)*/g, ' - ');
  }
  native = native.replace(/\s+/g, ' ').replace(/^[ -]+|[ -]+$/g, '');
  return native;
}

const LONG_VOWELS = new Set(['\u0906', '\u0908', '\u090a', '\u0960', '\u090f', '\u0910', '\u0913', '\u0914']);
const LONG_MATRAS = new Set(['\u093e', '\u0940', '\u0942', '\u0944', '\u0947', '\u0948', '\u094b', '\u094c']);

function segmentAksharas(devText) {
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

function textToIds(text, { lang = null, addBlank = true, minLen = 8 } = {}) {
  const [detectedScript, detectedLang] = detectScriptAndLang(text, lang || 'sa');
  const activeLang = lang && LANGUAGE_IDS[lang] !== undefined ? lang : detectedLang;
  const dev = (detectedScript === 'iast' || detectedScript === 'latin')
    ? iastToDevanagari(text)
    : brahmicToDevanagari(text);
  const phonemic = applyPhonologicalRules(dev, activeLang);

  const tokens = [];
  for (const ch of phonemic) {
    if (TOKEN_TO_ID[ch] !== undefined) tokens.push(ch);
  }
  if (tokens.length === 0) tokens.push('\u0905');

  const nativeText = toNativeModelText(text, activeLang, detectedScript, phonemic);
  const langMap = NATIVE_TOKEN_TO_ID[activeLang];
  const rawIds = [];
  const nativeTokens = [];
  for (const ch of nativeText) {
    if (langMap[ch] !== undefined) {
      rawIds.push(langMap[ch]);
      nativeTokens.push(ch);
    }
  }
  if (rawIds.length === 0) {
    const fb = { sa: '\u0c85', kn: '\u0c85', hi: '\u0905', ta: '\u0b85', te: '\u0c05', ml: '\u0d05' }[activeLang];
    rawIds.push(langMap[fb]);
    nativeTokens.push(fb);
  }

  let inputIds;
  if (addBlank) {
    const spId = langMap[' '];
    if (spId !== undefined && rawIds[rawIds.length - 1] !== spId) {
      rawIds.push(spId);
      nativeTokens.push(' ');
    }
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
    nativeText,
    tokens,
    nativeTokens,
    aksharas: segmentAksharas(phonemic),
  };
}

/**
 * Tier-2 Zero-Dependency Cascade Klatt-Akshara Synthesizer + 16-bit PCM WAV Encoder
 */


const SAMPLE_RATE = 16000;

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

function synthesizeKlattPcm(text, { lang = null, rate = 1.0, f0Base = 132.0 } = {}) {
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
      durV = 0.032;
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
    return { pcm: new Float32Array(1600), meta: { ...parsed, tier: 'klatt_zero_dep' } };
  }

  const out = new Float32Array(totalSamples);
  let offset = 0;
  let peak = 1e-5;
  for (const ch of chunks) {
    for (let i = 0; i < ch.length; i++) {
      const a = Math.abs(ch[i]);
      if (a > peak) peak = a;
    }
    out.set(ch, offset);
    offset += ch.length;
  }
  const gain = 0.82 / peak;
  for (let i = 0; i < out.length; i++) out[i] *= gain;

  return { pcm: out, meta: { ...parsed, tier: 'klatt_zero_dep' } };
}

function encodeWavBuffer(pcm, sr = SAMPLE_RATE) {
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
  return ort.InferenceSession.create(resolved, {
    executionProviders: useProviders,
    graphOptimizationLevel: 'all',
    enableCpuMemArena: false,
    enableMemPattern: false,
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
      const firstChunkFrames = Math.max(24, Number(callOpts.firstChunkFrames || 48));
      const chunkFrames = Math.max(32, Number(callOpts.chunkFrames || 96));
      const haloFrames = Math.max(8, Number(callOpts.haloFrames || 16));
      const crossfadeSamples = Math.max(0, Math.min(Number(callOpts.crossfadeSamples ?? 256), (haloFrames * HOP_LENGTH) >> 1));
      const signal = callOpts.signal || null;

      const clauses = splitProsodicClauses(text, callOpts.maxClauseChars || 160);
      const tStart = nowMs();
      let ttfaMs = null;
      let chunkIndex = 0;

      // Path 1: 16-Frame Halo + 256-Sample Raised-Cosine Crossfaded Neural Streaming
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

        for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
          if (signal?.aborted) return;
          const parsed = textToIds(clauses[cIdx], { lang });
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
            const step = cur === 0 && cIdx === 0 ? firstChunkFrames : chunkFrames;
            const nxt = Math.min(totalFrames, cur + step);
            bounds.push([cur, nxt]);
            cur = nxt;
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
              for (let f = 0; f < winLen; f++) {
                sliceBuf[dstBase + f] = latData[srcBase + f];
              }
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
            if (isLastSub && cIdx < clauses.length - 1) {
              const pauseSamples = Math.round(0.08 * SAMPLE_RATE);
              pcm = new Float32Array(basePcm.length + pauseSamples);
              pcm.set(basePcm, 0);
            } else {
              pcm = new Float32Array(basePcm.length);
              pcm.set(basePcm, 0);
            }

            const elapsedNow = nowMs() - tStart;
            if (ttfaMs === null) ttfaMs = elapsedNow;
            const isFinal = cIdx === clauses.length - 1 && isLastSub;
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
        const sessInNames = Array.isArray(session.inputNames) ? session.inputNames : [];
        const hasLangId = sessInNames.includes('lang_id');
        for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
          if (signal?.aborted) return;
          const tChunk = nowMs();
          const parsed = textToIds(clauses[cIdx], { lang });
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
