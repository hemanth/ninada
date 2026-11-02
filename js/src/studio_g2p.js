/**
 * Ninada v5.0 — Native Per-Language 24,000 Hz Studio VITS Character Tokenizer (Zero-Dep ESM).
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
  en: {
    pad: '<PAD>',
    punctuations: "!¡'(),-.:;¿? ",
    characters: 'â,pGYI&“US7(9…Z–v1yadHND`@lw”80ti?42mOgCE_XRMz\ufe0fk3‘VeP5jè"rK#F’)scQ\'Jôfq xA»bTouhnBWL6àé',
    blank: '<BLNK>',
  },
  mr: {
    pad: '<PAD>',
    punctuations: "!¡'(),-.:;¿? ",
    characters: ')ऊभॐब८•औॲ?z५ी/ऩठअवइ\',डळत3° ञोnआफ़ॊच:․छऍई;ओuगझल0२ॆऑऱ!ॕ"(मर|9…न-ॄु5ृहेॅःएघ७शऎ1ङॠं।१ष४cॉधाऽ\u200aकि2.ऋउ–ूa्स६जखदथ०68णऐैफ़ँs९३4ौ7—टपढयऌॡॢॣ॥',
    blank: '<BLNK>',
  },
  bn: {
    pad: '<PAD>',
    punctuations: "!¡'(),-.:;¿? ",
    characters: 'এাসৎঢঊচঢ়রখৰঘড়ঋ"ঔহনবভঅঠ্ঃৌছৃিড়কতীেওলং, জউ?থঞ\'ফুআঐয়ইূৈযগঁঙঝ।ঈৗণষধশটপদোমঌঽৄৠৡ০১২৩৪৫৬৭৮৯',
    blank: '<BLNK>',
  },
  gu: {
    pad: '<PAD>',
    punctuations: "!¡'(),-.:;¿? ",
    characters: 'ધઋલ૭ઉઊઝઈછઍત6ગेફ“78ખથरયજન૧ળણ,ૉ}ઞ૪ઔશચ3ા૦"(ૅ્મષૠ૩સ‘अઇઠુक૫દોર\'ીપૈ૬઼मઆ”ૌ4)ૂિહ{ઑઓ કઢઃ2ૃેવભि?અએ5૯ઘટંડઐ1૮ઁબ’09૨ઌઙઽૄૐૡ',
    blank: '<BLNK>',
  },
  pa: {
    pad: '<PAD>',
    punctuations: "!¡'(),-.:;¿? ",
    characters: 'ਁਂਃਅਆਇਈਉਊਏਐਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਵਸਹ਼ਾਿੀੁੂੇੈੋੌ੍ੜੰੱੲੳ੦੧੨੩੪੫੬੭੮੯',
    blank: '<BLNK>',
  },
  or: {
    pad: '<PAD>',
    punctuations: "!¡'(),-.:;¿? ",
    characters: 'ଁଂଃଅଆଇଈଉଊଋଌଏଐଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଳଵଶଷସହ଼ଽାିୀୁୂୃୄେୈୋୌ୍ୖୗୟୠୡୱ୦୧୨୩୪୫୬୭୮୯',
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
  mr: [0x0900, 0x097f],
  bn: [0x0980, 0x09ff],
  pa: [0x0a00, 0x0a7f],
  gu: [0x0a80, 0x0aff],
  or: [0x0b00, 0x0b7f],
  ta: [0x0b80, 0x0bff],
  te: [0x0c00, 0x0c7f],
  kn: [0x0c80, 0x0cff],
  ml: [0x0d00, 0x0d7f],
  en: [0x0041, 0x007a],
};

const EN_ONES = [
  'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
  'seventeen', 'eighteen', 'nineteen',
];
const EN_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
const EN_ORD_MAP = {
  one: 'first', two: 'second', three: 'third', five: 'fifth',
  eight: 'eighth', nine: 'ninth', twelve: 'twelfth',
};
const EN_ABBREVS = [
  [/\bmrs\./gi, 'missus'],
  [/\bmr\./gi, 'mister'],
  [/\bms\./gi, 'miss'],
  [/\bdr\./gi, 'doctor'],
  [/\bprof\./gi, 'professor'],
  [/\bvs\./gi, 'versus'],
  [/\betc\./gi, 'et cetera'],
  [/\be\.g\./gi, 'for example'],
  [/\bi\.e\./gi, 'that is'],
  [/\brs\./gi, 'rupees'],
];

function enIntToWords(n) {
  if (n < 0) return `minus ${enIntToWords(-n)}`;
  if (n < 20) return EN_ONES[n];
  if (n < 100) {
    const t = Math.floor(n / 10);
    const r = n % 10;
    return r === 0 ? EN_TENS[t] : `${EN_TENS[t]} ${EN_ONES[r]}`;
  }
  if (n < 1000) {
    const h = Math.floor(n / 100);
    const r = n % 100;
    return r === 0 ? `${EN_ONES[h]} hundred` : `${EN_ONES[h]} hundred ${enIntToWords(r)}`;
  }
  if (n >= 1900 && n <= 2099 && n % 100 !== 0 && !(n >= 2001 && n <= 2009)) {
    const hi = Math.floor(n / 100);
    const lo = n % 100;
    const loStr = lo < 10 ? `oh ${EN_ONES[lo]}` : enIntToWords(lo);
    return `${enIntToWords(hi)} ${loStr}`;
  }
  if (n < 1_000_000) {
    const k = Math.floor(n / 1000);
    const r = n % 1000;
    return r === 0 ? `${enIntToWords(k)} thousand` : `${enIntToWords(k)} thousand ${enIntToWords(r)}`;
  }
  return [...String(n)].map((d) => EN_ONES[Number(d)]).join(' ');
}

function enOrdinalToWords(n) {
  const words = enIntToWords(n).split(' ');
  const last = words[words.length - 1];
  if (EN_ORD_MAP[last]) words[words.length - 1] = EN_ORD_MAP[last];
  else if (last.endsWith('y')) words[words.length - 1] = `${last.slice(0, -1)}ieth`;
  else words[words.length - 1] = `${last}th`;
  return words.join(' ');
}

function normalizeEnglishText(t) {
  let out = t;
  for (const [pat, rep] of EN_ABBREVS) {
    out = out.replace(pat, rep);
  }
  out = out
    .replaceAll('&', ' and ')
    .replaceAll('%', ' percent ')
    .replaceAll('₹', ' rupees ')
    .replaceAll('$', ' dollars ');
  out = out.replace(/(?<=\d),(?=\d{3}\b)/g, '');
  out = out.replace(/\b(\d{1,6})(st|nd|rd|th)\b/gi, (_, num) => enOrdinalToWords(Number(num)));
  out = out.replace(/\b(\d{1,6})\b/g, (_, num) => enIntToWords(Number(num)));
  return out;
}

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

  if (lc === 'en') {
    t = normalizeEnglishText(t);
  } else {
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
      if (lc === 'sa') {
        t = iastToDevanagari(t);
      } else {
        const [detSc] = detectScriptAndLang(t, lc);
        const dev = detSc === 'iast' || detSc === 'latin' ? iastToDevanagari(t) : t;
        t = toNativeModelText(t, lc, detSc, dev);
      }
    }
  }

  if (lc === 'sa') {
    t = t.replace(/[\u0951-\u0954\u1cd0-\u1cff]/g, '');
    t = t.replaceAll('॥', '.').replaceAll('।', ',');
  }
  if (lc === 'mr') {
    t = t.replace(/[\u0951-\u0954\u1cd0-\u1cff]/g, '');
    t = t.replaceAll('॥', '।');
  }
  if (lc === 'bn') {
    t = t.replaceAll('\u09f1', 'ব').replaceAll('॥', '।');
  }
  if (lc === 'gu') {
    t = t.replaceAll('॥', '.').replaceAll('।', '.');
  }
  if (lc === 'pa') {
    t = t.replaceAll('\u0a74', 'ਇੱਕ ਓਅੰਕਾਰ');
    t = t.replaceAll('॥', '.').replaceAll('।', '.');
  }
  if (lc === 'or') {
    t = t.replaceAll('॥', '.').replaceAll('।', '.');
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

  if (lc === 'kn' || lc === 'ta' || lc === 'te' || lc === 'ml' || lc === 'en' || lc === 'gu' || lc === 'pa' || lc === 'or') {
    t = t.replace(/[॥।]/g, '.');
  } else if (lc === 'hi' || lc === 'mr' || lc === 'bn') {
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
