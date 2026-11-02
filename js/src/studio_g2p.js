/**
 * Ninada v4.0 — Native Per-Language 24,000 Hz Studio VITS Character Tokenizer (Zero-Dep ESM).
 */
import { detectScriptAndLang, iastToDevanagari, toNativeModelText } from './g2p.js';

export const STUDIO_HOP_LENGTH = 256;
export const STUDIO_SAMPLE_RATE = 24000;

export const STUDIO_SPECS = Object.freeze({
  sa: {
    pad: '<PAD>',
    punctuations: "!¡'(),-.:;¿? ",
    characters: ')ऊभॐब८•औॲ?z५ी/ऩठअवइ\',डळत3° ञोnआफ़ॊच:․छऍई;ओuगझल0२ॆऑऱ!ॕ"(मर|9…न-ॄु5ृहेॅःएघ७शऎ1ङॠं।१ष४cॉधाऽ\u200aकि2.ऋउ–ूa्स६जखदथ०68णऐैफ़ँs९३4ौ7—टपढयऌॡॢॣ॥',
    blank: '<BLNK>',
  },
  hi: {
    pad: '<PAD>',
    punctuations: "!¡'(),-.:;¿? ",
    characters: '८ढी५ू)बख”ंनग़ँक़ङत1।व2ट(ऑओग़फज9्थुख़०छञड5ड़औ१ैचफ़रण?ॐ२लॠदय़म‘–"6भॉ7’ईय४७इशोठ3९उ“ज़ष३िौ4ऊःहढ़ऋअॅआऐए0क६घझधस,\'ृपे 8ा',
    blank: '<BLNK>',
  },
  kn: {
    pad: '<PAD>',
    punctuations: "!¡'(),-.:;¿? ",
    characters: 'ಆಽ*ಬಚಠಸಂುಣೂೆಖಔಢಭ(ಡಿಝದಈಫೊ ಛಅಐಗಉ?ನೃಧಇೀಜಥೌತಊಾೇೕಹಷಕಲೈಒಳಯ,ಟಪಓಎ\'೦ಋ"್ಶರಘೋಞಏವಮ',
    blank: '<BLNK>',
  },
  ta: {
    pad: '_',
    punctuations: '.,!?; ',
    characters: "ஂஃஅஆஇஈஉஊஎஏஐஒஓஔகஙசஜஞடணதநனபமயரறலளழவஷஸஹாிீுூெேைொோௌ்ௗ'-()",
    blank: '<BLNK>',
  },
  te: {
    pad: '<PAD>',
    punctuations: "!¡'(),-.:;¿? ",
    characters: 'ఆధఏఘbః…ఈడమఖకలఊ’దఒ,:తీఞ·గైం.ొిఔ•్ృయ।ఱూ/[ఢేసఝ;\'ఛు-టనళఓఐథ“భపాషఋఙోఠ–ఉణ!జబ("రa”ఎఁశఇఫ|అహ*—ౌ?ʼ)఼‘ వచె౦',
    blank: '<BLNK>',
  },
  ml: {
    pad: '<PAD>',
    punctuations: "!¡'(),-.:;¿? ",
    characters: 'ംഃഅആഇഈഉഊഋഌഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനഩപഫബഭമയരറലളഴവശഷസഹഽാിീുൂൃെേൈൊോൌ്ൗൠൡ൦൧൨൩൪൫൬൭൮൯ൺൻർൽൾൿ',
    blank: '<BLNK>',
  },
});

export const STUDIO_VOCABS = Object.freeze(
  Object.fromEntries(
    Object.entries(STUDIO_SPECS).map(([lc, spec]) => {
      const vocab = [spec.pad, ...Array.from(spec.punctuations), ...Array.from(spec.characters), spec.blank];
      const charToId = {};
      for (let i = 0; i < vocab.length; i++) {
        charToId[vocab[i]] = i;
      }
      return [
        lc,
        Object.freeze({
          lang: lc,
          vocab: Object.freeze(vocab),
          charToId: Object.freeze(charToId),
          padId: charToId[spec.pad],
          blankId: charToId[spec.blank],
          numChars: vocab.length,
        }),
      ];
    })
  )
);

const SCRIPT_RANGES = {
  sa: [0x0900, 0x097f],
  hi: [0x0900, 0x097f],
  kn: [0x0c80, 0x0cff],
  ta: [0x0b80, 0x0bff],
  te: [0x0c00, 0x0c7f],
  ml: [0x0d00, 0x0d7f],
};

const MALAYALAM_CHILLU_PAIRS = [
  ['\u0d23\u0d4d\u200d', '\u0d7a'],
  ['\u0d28\u0d4d\u200d', '\u0d7b'],
  ['\u0d30\u0d4d\u200d', '\u0d7c'],
  ['\u0d32\u0d4d\u200d', '\u0d7d'],
  ['\u0d33\u0d4d\u200d', '\u0d7e'],
  ['\u0d15\u0d4d\u200d', '\u0d7f'],
  ['\u0d23\u0d4d\u200c', '\u0d7a'],
  ['\u0d28\u0d4d\u200c', '\u0d7b'],
  ['\u0d30\u0d4d\u200c', '\u0d7c'],
  ['\u0d32\u0d4d\u200c', '\u0d7d'],
  ['\u0d33\u0d4d\u200c', '\u0d7e'],
  ['\u0d15\u0d4d\u200c', '\u0d7f'],
];

export function cleanStudioText(text, lang = 'sa') {
  const lc = STUDIO_VOCABS[lang] ? lang : detectScriptAndLang(text, 'sa')[1];
  let t = String(text ?? '').normalize('NFC').trim();
  if (!t) return '';

  const [lo, hi] = SCRIPT_RANGES[lc];
  let hasTarget = false;
  for (const ch of t) {
    const cp = ch.codePointAt(0);
    if (cp >= lo && cp <= hi) {
      hasTarget = true;
      break;
    }
  }
  if (!hasTarget) {
    t = lc === 'sa' ? iastToDevanagari(t) : toNativeModelText(t, lc);
  }

  if (lc === 'sa') {
    t = t.replace(/[\u0951-\u0954\u1cd0-\u1cff]/g, '');
    t = t.replaceAll('॥', '.').replaceAll('।', ',');
  }
  if (lc === 'kn') {
    t = t.replace(/\u0cbf\u0c83(?=[\s,.;:!?।॥]|$)/g, 'ಿಹಿ');
    t = t.replace(/\u0cc1\u0c83(?=[\s,.;:!?।॥]|$)/g, 'ುಹು');
    t = t.replace(/\u0c83(?=[\s,.;:!?।॥]|$)/g, 'ಹ');
    t = t.replaceAll('\u0c83', 'ಹ್');
    t = t.replaceAll('\u0c99\u0ccd\u0cae', 'ಂಗ್ಮ');
    t = t.replaceAll('\u0c99\u0ccd', 'ಂ').replaceAll('\u0c99', 'ಂಗ');
    t = t
      .replaceAll('\u0cb1', 'ರ')
      .replaceAll('\u0cc4', 'ೃ')
      .replaceAll('\u0cde', 'ಳ')
      .replaceAll('\u0cd6', 'ೈ')
      .replaceAll('\u0ce0', 'ಋ')
      .replaceAll('\u0c8c', 'ಲ')
      .replaceAll('\u0ce1', 'ಲ')
      .replaceAll('\u0c81', 'ಂ');
  }
  if (lc === 'ta') {
    t = t.replaceAll('\u0bd0', 'ஓம்').replaceAll('\u0bb6', 'ஸ');
  }
  if (lc === 'ml') {
    for (const [seq, atomic] of MALAYALAM_CHILLU_PAIRS) {
      t = t.replaceAll(seq, atomic);
    }
    t = t
      .replaceAll('\u0d4e', '\u0d7c')
      .replaceAll('\u0d54', 'ം')
      .replaceAll('\u0d3a', 'റ')
      .replaceAll('\u0d44', 'ൃ');
  }

  t = t.replace(/[\u200c\u200d\ufeff]/g, '');
  t = t
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/–/g, ' ')
    .replace(/—/g, ', ')
    .replace(/…/g, '.')
    .replace(/[;:]/g, ',')
    .replace(/-/g, ' ');

  if (lc === 'kn' || lc === 'ta' || lc === 'te' || lc === 'ml') {
    t = t.replace(/[॥।]/g, '.');
  } else if (lc === 'hi') {
    t = t.replaceAll('॥', '।');
  }

  t = t.toLowerCase();
  t = t.replace(/[<>()[\]"]+/g, '');
  t = t.replace(/\s+/g, ' ').trim();
  if ((lc === 'ml' || lc === 'ta') && t) {
    if (t.endsWith(',')) {
      const stem = t.replace(/[,;\s]+$/, '').trim();
      if (stem) t = `${stem} ,`;
    } else {
      const stem = t.replace(/[.,!?;\s]+$/, '').trim();
      if (stem) t = `${stem} . .`;
    }
  }
  return t;
}

export function studioTextToIds(text, lang = null) {
  const [, resolvedLang] = detectScriptAndLang(text, lang);
  const lc = STUDIO_VOCABS[resolvedLang] ? resolvedLang : 'sa';
  const spec = STUDIO_VOCABS[lc];
  const c2i = spec.charToId;
  const blankId = spec.blankId;

  const cleaned = cleanStudioText(text, lc);
  const rawIds = [];
  for (const ch of cleaned) {
    if (Object.prototype.hasOwnProperty.call(c2i, ch)) {
      rawIds.push(c2i[ch]);
    }
  }
  if (rawIds.length === 0) {
    rawIds.push(Object.prototype.hasOwnProperty.call(c2i, ' ') ? c2i[' '] : spec.padId);
  }

  const interleaved = [blankId];
  for (let i = 0; i < rawIds.length; i++) {
    interleaved.push(rawIds[i], blankId);
  }
  return { studioIds: interleaved, lang: lc, cleanedText: cleaned };
}
