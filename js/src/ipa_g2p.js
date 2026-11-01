/**
 * Zero-dependency Pan-Indic & Classical/Vedic Sanskrit -> 24 kHz IPA Phonemizer for Ninada v3.0.
 * Exact 1-to-1 JavaScript mirror of python/src/ninada/ipa_g2p.py.
 */

export const IPA_VOCAB = {
  ';': 1, ':': 2, ',': 3, '.': 4, '!': 5, '?': 6, '—': 9, '…': 10,
  '"': 11, '(': 12, ')': 13, '“': 14, '”': 15, ' ': 16, '̃': 17,
  'ʣ': 18, 'ʥ': 19, 'ʦ': 20, 'ʨ': 21, 'ᵝ': 22, 'ꭧ': 23,
  'A': 24, 'I': 25, 'O': 31, 'Q': 33, 'S': 35, 'T': 36, 'W': 39, 'Y': 41,
  'ᵊ': 42, 'a': 43, 'b': 44, 'c': 45, 'd': 46, 'e': 47, 'f': 48, 'h': 50,
  'i': 51, 'j': 52, 'k': 53, 'l': 54, 'm': 55, 'n': 56, 'o': 57, 'p': 58,
  'q': 59, 'r': 60, 's': 61, 't': 62, 'u': 63, 'v': 64, 'w': 65, 'x': 66,
  'y': 67, 'z': 68, 'ɑ': 69, 'ɐ': 70, 'ɒ': 71, 'æ': 72, 'β': 75, 'ɔ': 76,
  'ɕ': 77, 'ç': 78, 'ɭ': 79, 'ɖ': 80, 'ð': 81, 'ʤ': 82, 'ə': 83, 'ɚ': 85,
  'ɛ': 86, 'ɜ': 87, 'ɟ': 90, 'ɡ': 92, 'ɥ': 99, 'ɨ': 101, 'ɪ': 102, 'ʝ': 103,
  'ɯ': 110, 'ɰ': 111, 'ŋ': 112, 'ɳ': 113, 'ɲ': 114, 'ɴ': 115, 'ø': 116,
  'ɸ': 118, 'θ': 119, 'œ': 120, 'ɹ': 123, 'ɾ': 125, 'ɻ': 126, 'ʁ': 128,
  'ɽ': 129, 'ʂ': 130, 'ʃ': 131, 'ʈ': 132, 'ʧ': 133, 'ʊ': 135, 'ʋ': 136,
  'ʌ': 138, 'ɣ': 139, 'ɤ': 140, 'χ': 142, 'ʎ': 143, 'ʒ': 147, 'ʔ': 148,
  'ˈ': 156, 'ˌ': 157, 'ː': 158, 'ʰ': 162, 'ʲ': 164,
  '↓': 169, '→': 171, '↗': 172, '↘': 173, 'ᵻ': 177,
};

export const IPA_ID_TO_TOKEN = Object.fromEntries(
  Object.entries(IPA_VOCAB).map(([k, v]) => [v, k])
);
export const RETROFLEX_LATERAL_ID = 79;

const VIRAMA = '\u094d';
const NUKTA = '\u093c';
const ANUSVARA = '\u0902';
const CHANDRABINDU = '\u0901';
const VISARGA = '\u0903';
const AVAGRAHA = '\u093d';

const NUKTA_PRECOMPOSED = {
  '\u0958': 'q',
  '\u0959': 'x',
  '\u095a': 'ɣ',
  '\u095b': 'z',
  '\u095c': 'ɽ',
  '\u095d': 'ɽʰ',
  '\u095e': 'f',
  '\u095f': 'j',
};

const NUKTA_BASE_MAP = {
  'क': 'q',
  'ख': 'x',
  'ग': 'ɣ',
  'ज': 'z',
  'ड': 'ɽ',
  'ढ': 'ɽʰ',
  'फ': 'f',
};

const CONSONANT_IPA = {
  'क': 'k', 'ख': 'kʰ', 'ग': 'ɡ', 'घ': 'ɡʰ', 'ङ': 'ŋ',
  'च': 'ʧ', 'छ': 'ʧʰ', 'ज': 'ʤ', 'झ': 'ʤʰ', 'ञ': 'ɲ',
  'ट': 'ʈ', 'ठ': 'ʈʰ', 'ड': 'ɖ', 'ढ': 'ɖʰ', 'ण': 'ɳ',
  'त': 't', 'थ': 'tʰ', 'द': 'd', 'ध': 'dʰ', 'न': 'n', '\u0929': 'n',
  'प': 'p', 'फ': 'pʰ', 'ब': 'b', 'भ': 'bʰ', 'म': 'm',
  'य': 'j', 'र': 'ɾ', '\u0931': 'r', 'ल': 'l', 'ळ': 'ɭ', '\u0934': 'ɻ',
  'व': 'ʋ', 'श': 'ɕ', 'ष': 'ʂ', 'स': 's', 'ह': 'h',
};

const STOP_HOMORGANIC_NASAL = {
  'क': 'ŋ', 'ख': 'ŋ', 'ग': 'ŋ', 'घ': 'ŋ', 'ङ': 'ŋ',
  'च': 'ɲ', 'छ': 'ɲ', 'ज': 'ɲ', 'झ': 'ɲ', 'ञ': 'ɲ',
  'ट': 'ɳ', 'ठ': 'ɳ', 'ड': 'ɳ', 'ढ': 'ɳ', 'ण': 'ɳ',
  'त': 'n', 'थ': 'n', 'द': 'n', 'ध': 'n', 'न': 'n', '\u0929': 'n',
  'प': 'm', 'फ': 'm', 'ब': 'm', 'भ': 'm', 'म': 'm',
};

const ML_CHILLU_IPA = {
  '\u0d7a': 'ɳ',
  '\u0d7b': 'n',
  '\u0d7c': 'ɾ',
  '\u0d7d': 'l',
  '\u0d7e': 'ɭ',
  '\u0d7f': 'k',
};

const INDEP_VOWELS_COMMON = {
  'अ': 'ə',
  'आ': 'ɑː',
  'इ': 'ɪ',
  'ई': 'iː',
  'उ': 'ʊ',
  'ऊ': 'uː',
  'ऋ': 'ɾɪ',
  'ॠ': 'ɾiː',
  'ऌ': 'lɪ',
  'ॡ': 'liː',
  '\u090e': 'ɛ',
  'ए': 'eː',
  'ऍ': 'æː',
  '\u0912': 'o',
  'ओ': 'oː',
  'ऑ': 'ɔː',
};

const MATRAS_COMMON = {
  'ा': 'ɑː',
  'ि': 'ɪ',
  'ी': 'iː',
  'ु': 'ʊ',
  'ू': 'uː',
  'ृ': 'ɾɪ',
  'ॄ': 'ɾiː',
  'ॢ': 'lɪ',
  'ॣ': 'liː',
  '\u0946': 'ɛ',
  'े': 'eː',
  'ॅ': 'æː',
  '\u094a': 'o',
  'ो': 'oː',
  'ॉ': 'ɔː',
};

const LONG_VOWEL_IPA = new Set(['ɑː', 'iː', 'uː', 'ɾiː', 'eː', 'æː', 'oː', 'ɔː', 'aɪ', 'əɪ', 'aʊ']);
const VOWEL_CHARS = new Set([...'əɑɪiuʊeɛæoɔaᵊ']);

function vowelIpa(ch, isMatra, lang) {
  if (isMatra) {
    if (ch === 'ै') return lang === 'hi' ? 'æː' : 'aɪ';
    if (ch === 'ौ') return lang === 'hi' ? 'ɔː' : 'aʊ';
    return MATRAS_COMMON[ch] || '';
  }
  if (ch === 'ऐ') return lang === 'hi' ? 'æː' : 'aɪ';
  if (ch === 'औ') return lang === 'hi' ? 'ɔː' : 'aʊ';
  return INDEP_VOWELS_COMMON[ch] || '';
}

function normalizeScriptWordToDev(word) {
  const out = [];
  for (const ch of String(word).normalize('NFC')) {
    if (ML_CHILLU_IPA[ch] !== undefined) {
      out.push(ch);
      continue;
    }
    const cp = ch.codePointAt(0);
    if (cp >= 0x0b80 && cp <= 0x0bff) {
      if (ch === '\u0b83') out.push('ख़');
      else out.push(String.fromCodePoint(cp - 0x0b80 + 0x0900));
    } else if (cp >= 0x0c00 && cp <= 0x0c7f) {
      out.push(String.fromCodePoint(cp - 0x0c00 + 0x0900));
    } else if (cp >= 0x0c80 && cp <= 0x0cff) {
      out.push(String.fromCodePoint(cp - 0x0c80 + 0x0900));
    } else if (cp >= 0x0d00 && cp <= 0x0d7f) {
      if (ch === '\u0d3a') out.push('ट');
      else out.push(String.fromCodePoint(cp - 0x0d00 + 0x0900));
    } else {
      out.push(ch);
    }
  }
  return out.join('');
}

function isModMark(ch) {
  return (
    ch === ANUSVARA ||
    ch === CHANDRABINDU ||
    ch === VISARGA ||
    ch === '॑' ||
    ch === '॒' ||
    ch === '॓'
  );
}

function parseWordUnits(devWord, lang) {
  const chars = [...devWord];
  const n = chars.length;
  const units = [];
  let i = 0;

  while (i < n) {
    const ch = chars[i];
    if (ch === '\u200c' || ch === '\u200d') {
      i += 1;
      continue;
    }
    if (ML_CHILLU_IPA[ch] !== undefined) {
      units.push({
        kind: 'C',
        cons: ML_CHILLU_IPA[ch],
        rawC: ch,
        vowel: '',
        inherent: false,
        nasal: '',
        visarga: false,
        stress: false,
        finalVirama: true,
      });
      i += 1;
      continue;
    }
    if (ch === '॑' || ch === '॓') {
      if (units.length > 0) units[units.length - 1].stress = true;
      i += 1;
      continue;
    }
    if (ch === '॒') {
      i += 1;
      continue;
    }
    if (ch === AVAGRAHA) {
      if (units.length > 0 && units[units.length - 1].vowel) {
        const v = units[units.length - 1].vowel;
        if (!v.endsWith('ː')) units[units.length - 1].vowel = `${v}ː`;
      }
      i += 1;
      continue;
    }

    const vInd = vowelIpa(ch, false, lang);
    if (vInd) {
      i += 1;
      let nasal = '';
      let visarga = false;
      let stress = false;
      while (i < n && isModMark(chars[i])) {
        const m = chars[i];
        if (m === CHANDRABINDU) nasal = '̃';
        else if (m === ANUSVARA) nasal = 'M';
        else if (m === VISARGA) visarga = true;
        else if (m === '॑' || m === '॓') stress = true;
        i += 1;
      }
      units.push({
        kind: 'V',
        cons: '',
        rawC: '',
        vowel: vInd,
        inherent: false,
        nasal,
        visarga,
        stress,
        finalVirama: false,
      });
      continue;
    }

    let cIpa = NUKTA_PRECOMPOSED[ch];
    const rawC = ch;
    if (cIpa !== undefined) {
      i += 1;
    } else if (CONSONANT_IPA[ch] !== undefined) {
      if (i + 1 < n && chars[i + 1] === NUKTA) {
        cIpa = NUKTA_BASE_MAP[ch] || CONSONANT_IPA[ch];
        i += 2;
      } else {
        cIpa = CONSONANT_IPA[ch];
        if (lang === 'ta' && ch === '\u0931') cIpa = 'ɾ';
        i += 1;
      }
    } else {
      i += 1;
      continue;
    }

    if (i < n && chars[i] === VIRAMA) {
      i += 1;
      units.push({
        kind: 'C',
        cons: cIpa,
        rawC,
        vowel: '',
        inherent: false,
        nasal: '',
        visarga: false,
        stress: false,
        finalVirama: i === n,
      });
      continue;
    }

    let vowel = 'ə';
    let inherent = true;
    if (i < n) {
      const vMat = vowelIpa(chars[i], true, lang);
      if (vMat) {
        vowel = vMat;
        inherent = false;
        i += 1;
      }
    }

    let nasal = '';
    let visarga = false;
    let stress = false;
    while (i < n && isModMark(chars[i])) {
      const m = chars[i];
      if (m === CHANDRABINDU) nasal = '̃';
      else if (m === ANUSVARA) nasal = 'M';
      else if (m === VISARGA) visarga = true;
      else if (m === '॑' || m === '॓') stress = true;
      i += 1;
    }

    units.push({
      kind: 'C',
      cons: cIpa,
      rawC,
      vowel,
      inherent,
      nasal,
      visarga,
      stress,
      finalVirama: false,
    });
  }

  return units;
}

const HI_FRIC_SON = new Set(['श', 'ष', 'स', 'ह', 'य', 'र', 'ल', 'व']);
const TA_VOICELESS_STOPS = new Set(['क', 'च', 'ट', 'त', 'प']);
const TA_NASALS = new Set(['ङ', 'ञ', 'ण', 'न', '\u0929', 'म']);
const ML_NASALS = new Set(['ङ', 'ञ', 'ण', 'न', 'म', '\u0d7a', '\u0d7b']);
const IPA_NASALS = new Set(['ŋ', 'ɲ', 'ɳ', 'n', 'm']);
const TA_SONORANTS = new Set(['र', 'ल', 'ळ', '\u0934', 'य']);
const POST_NASAL_VOICING = { 'क': 'ɡ', 'च': 'ʤ', 'ट': 'ɖ', 'त': 'd', 'प': 'b' };
const TA_LENITION = { 'क': 'ɡ', 'च': 's', 'ट': 'ɖ', 'त': 'ð', 'प': 'b' };

function applyLanguageAllophony(units, lang) {
  if (!units || units.length === 0) return;

  for (let idx = 0; idx < units.length; idx++) {
    const u = units[idx];
    if (u.nasal === 'M') {
      const nxtRaw = idx + 1 < units.length ? units[idx + 1].rawC : '';
      if (STOP_HOMORGANIC_NASAL[nxtRaw]) {
        u.nasal = STOP_HOMORGANIC_NASAL[nxtRaw];
      } else if (lang === 'hi' && HI_FRIC_SON.has(nxtRaw)) {
        u.nasal = '̃';
      } else if (lang === 'hi' && idx === units.length - 1) {
        u.nasal = '̃';
      } else {
        u.nasal = 'm';
      }
    }
  }

  if (lang === 'hi') {
    const nuclei = [];
    for (let idx = 0; idx < units.length; idx++) {
      if (units[idx].vowel) nuclei.push(idx);
    }
    if (nuclei.length >= 2) {
      const lastIdx = nuclei[nuclei.length - 1];
      const uLast = units[lastIdx];
      if (lastIdx === units.length - 1 && uLast.inherent && !uLast.nasal && !uLast.visarga) {
        const prevIsClusterY = lastIdx >= 1 && units[lastIdx - 1].vowel === '' && uLast.rawC === 'य';
        if (!prevIsClusterY) {
          uLast.vowel = '';
          uLast.inherent = false;
        }
      }
    }
    for (let idx = 1; idx < units.length - 1; idx++) {
      const u = units[idx];
      if (!u.inherent || u.nasal || u.visarga) continue;
      const prevU = units[idx - 1];
      const nextU = units[idx + 1];
      if (prevU.vowel && nextU.vowel && !nextU.inherent) {
        u.vowel = '';
        u.inherent = false;
      }
    }
  } else if (lang === 'ta') {
    for (let idx = 0; idx < units.length; idx++) {
      const u = units[idx];
      const raw = u.rawC;
      const prevU = idx > 0 ? units[idx - 1] : null;
      const prevRaw = prevU ? prevU.rawC : '';
      const prevBare = Boolean(prevU && prevU.vowel === '');

      if (raw === '\u0931' && prevBare) {
        if (prevRaw === '\u0931') {
          prevU.cons = 't';
          u.cons = 'tɾ';
          continue;
        }
        if (prevRaw === '\u0929' || prevRaw === 'न') {
          prevU.cons = 'n';
          u.cons = 'dɾ';
          continue;
        }
      }

      if (prevBare && prevRaw === raw && TA_VOICELESS_STOPS.has(raw)) continue;
      if (idx + 1 < units.length && u.vowel === '' && units[idx + 1].rawC === raw) continue;

      const prevHasHomorganicNasal = Boolean(
        (prevBare && TA_NASALS.has(prevRaw)) || (prevU && IPA_NASALS.has(prevU.nasal))
      );
      if (prevHasHomorganicNasal) {
        if (POST_NASAL_VOICING[raw]) u.cons = POST_NASAL_VOICING[raw];
        continue;
      }

      if (idx === 0) {
        if (raw === 'च') u.cons = 's';
        continue;
      }

      const prevVoicedSonorant = Boolean(
        (prevU && prevU.vowel !== '') || (prevBare && TA_SONORANTS.has(prevRaw))
      );
      if (prevVoicedSonorant && u.vowel !== '') {
        if (TA_LENITION[raw]) u.cons = TA_LENITION[raw];
      }
    }

    for (let idx = 0; idx < units.length; idx++) {
      if (idx > 0 && units[idx].vowel === 'aɪ') {
        units[idx].vowel = 'əɪ';
      }
    }
  } else if (lang === 'ml') {
    for (let idx = 0; idx < units.length; idx++) {
      const u = units[idx];
      const raw = u.rawC;
      const prevU = idx > 0 ? units[idx - 1] : null;
      const prevRaw = prevU ? prevU.rawC : '';
      const prevBare = Boolean(prevU && prevU.vowel === '');

      if (raw === '\u0931' && prevBare) {
        if (prevRaw === '\u0931') {
          prevU.cons = 't';
          u.cons = 't';
        } else if (prevRaw === 'न' || prevRaw === '\u0d7b') {
          prevU.cons = 'n';
          u.cons = 'd';
        }
      } else if (TA_VOICELESS_STOPS.has(raw)) {
        const prevNasal = Boolean(
          (prevBare && ML_NASALS.has(prevRaw)) || (prevU && IPA_NASALS.has(prevU.nasal))
        );
        if (prevNasal && POST_NASAL_VOICING[raw]) {
          u.cons = POST_NASAL_VOICING[raw];
        }
      }
    }
    const last = units[units.length - 1];
    if (last.finalVirama && ML_CHILLU_IPA[last.rawC] === undefined) {
      last.vowel = 'ə';
    }
  }

  if (!units.some((u) => u.stress)) {
    const vowelIndices = [];
    for (let idx = 0; idx < units.length; idx++) {
      if (units[idx].vowel) vowelIndices.push(idx);
    }
    if (vowelIndices.length > 0) {
      let chosen = vowelIndices[0];
      for (const vIdx of vowelIndices) {
        const u = units[vIdx];
        const isHeavy =
          LONG_VOWEL_IPA.has(u.vowel) ||
          Boolean(u.nasal) ||
          Boolean(u.visarga) ||
          (vIdx + 1 < units.length && units[vIdx + 1].vowel === '');
        if (isHeavy) {
          chosen = vIdx;
          break;
        }
      }
      units[chosen].stress = true;
    }
  }
}

function renderUnitsToIpa(units, lang) {
  const parts = [];
  for (let idx = 0; idx < units.length; idx++) {
    const u = units[idx];
    if (u.stress && u.vowel) parts.push(`${u.cons}ˈ${u.vowel}`);
    else parts.push(`${u.cons}${u.vowel}`);

    if (u.nasal) parts.push(u.nasal);

    if (u.visarga) {
      const nxtRaw = idx + 1 < units.length ? units[idx + 1].rawC : '';
      if (nxtRaw === 'च' || nxtRaw === 'छ') parts.push('ɕ');
      else if (nxtRaw === 'ट' || nxtRaw === 'ठ') parts.push('ʂ');
      else if (nxtRaw === 'त' || nxtRaw === 'थ') parts.push('s');
      else if (idx === units.length - 1 && lang === 'sa' && u.vowel) {
        if (u.vowel === 'ə') parts.push('hᵊ');
        else if (u.vowel === 'ɪ') parts.push('hɪ');
        else if (u.vowel === 'ʊ') parts.push('hʊ');
        else parts.push('h');
      } else {
        parts.push('h');
      }
    }
  }
  return parts.join('');
}

const DIGIT_WORDS_HI = {
  '0': 'ɕˈuːnjə', '1': 'ˈeːk', '2': 'dˈoː', '3': 'tˈiːn', '4': 'ʧˈɑːɾ',
  '5': 'pˈɑ̃ːʧ', '6': 'ʧʰˈɛː', '7': 'sˈɑːt', '8': 'ˈɑːʈʰ', '9': 'nˈɔː',
};
const INDIC_ZERO_BASES = [0x0966, 0x0be6, 0x0c66, 0x0ce6, 0x0d66];
const LATIN_CHAR_IPA = {
  a: 'ə', b: 'b', c: 'k', d: 'd', e: 'e', f: 'f', g: 'ɡ',
  h: 'h', i: 'ɪ', j: 'ʤ', k: 'k', l: 'l', m: 'm', n: 'n',
  o: 'o', p: 'p', q: 'k', r: 'ɾ', s: 's', t: 't', u: 'ʊ',
  v: 'v', w: 'w', x: 'ks', y: 'j', z: 'z',
};

function latinWordToIpa(word) {
  let w = String(word || '').toLowerCase();
  for (const [dg, rep] of [['tion', 'ʃən'], ['sh', 'ʃ'], ['ch', 'ʧ'], ['th', 'tʰ'], ['ph', 'f'], ['ng', 'ŋ'], ['ee', 'iː'], ['oo', 'uː']]) {
    w = w.split(dg).join(rep);
  }
  let out = '';
  for (const ch of w) {
    if (LATIN_CHAR_IPA[ch] !== undefined) out += LATIN_CHAR_IPA[ch];
    else if (IPA_VOCAB[ch] !== undefined) out += ch;
  }
  return out;
}

function normalizeDigits(text) {
  const out = [];
  for (const ch of String(text)) {
    const cp = ch.codePointAt(0);
    let replaced = false;
    for (const base of INDIC_ZERO_BASES) {
      if (cp >= base && cp <= base + 9) {
        out.push(String(cp - base));
        replaced = true;
        break;
      }
    }
    if (!replaced) out.push(ch);
  }
  return out.join('');
}

const GUARD_PUNCT = new Set(['.', ',', '!', '?', ';', ':']);

export function textToIpa(text, lang = 'sa', normalizedDevFallback = null) {
  let raw = normalizeDigits(String(text || '')).trim();
  if (!raw) return 'ə';

  raw = raw.replace(/\u0965/g, '.').replace(/\|\|/g, '.');
  if (lang === 'sa' && (raw.includes('\u0964') || raw.includes('|'))) {
    raw = raw.replace(/[\u0964|]\s*$/, '.');
    raw = raw.replace(/\u0964/g, ';').replace(/\|/g, ';');
  } else {
    raw = raw.replace(/\u0964/g, '.').replace(/\|/g, '.');
  }
  raw = raw.replace(/\.\.\./g, '…').replace(/–/g, '—');

  let work = raw;
  if (normalizedDevFallback && /[A-Za-z\u0100-\u024F]/.test(raw)) {
    work = String(normalizedDevFallback).replace(/\u0965/g, '.').replace(/\u0964/g, ',');
  }

  const tokens = work.match(/[^\s,.;:!?—…"()“”]+|[,.;:!?—…"()“”]+|\s+/g) || [];
  const outChunks = [];

  for (const tok of tokens) {
    if (/^\s+$/.test(tok)) {
      outChunks.push(' ');
      continue;
    }
    if (/^[,.;:!?—…"()“”]+$/.test(tok)) {
      const p = tok[0];
      const prev = outChunks.length > 0 ? outChunks[outChunks.length - 1] : '';
      const lastChar = prev ? [...prev].pop() : '';
      if (lastChar && VOWEL_CHARS.has(lastChar) && GUARD_PUNCT.has(p)) {
        outChunks.push(` ${p}`);
      } else {
        outChunks.push(p);
      }
      continue;
    }
    if (/^\d+$/.test(tok)) {
      outChunks.push([...tok].map((d) => DIGIT_WORDS_HI[d] || 'ɕˈuːnjə').join(' '));
      continue;
    }
    if (/^[A-Za-z']+$/.test(tok)) {
      const latIpa = latinWordToIpa(tok);
      if (latIpa) outChunks.push(latIpa);
      continue;
    }

    const devWord = normalizeScriptWordToDev(tok);
    const units = parseWordUnits(devWord, lang);
    if (units.length === 0) continue;
    applyLanguageAllophony(units, lang);
    const ipaWord = renderUnitsToIpa(units, lang);
    if (ipaWord) outChunks.push(ipaWord);
  }

  const ipa = outChunks.join('').replace(/\s+/g, ' ').trim();
  const filteredChars = [];
  for (const ch of ipa) {
    if (IPA_VOCAB[ch] !== undefined) filteredChars.push(ch);
  }
  let filtered = filteredChars.join('');
  if (!filtered) return 'ə';
  const lastFiltered = filteredChars[filteredChars.length - 1];
  if (VOWEL_CHARS.has(lastFiltered)) {
    filtered = `${filtered} .`;
  }
  return filtered;
}

export function ipaToIds(ipa, maxTokens = 510) {
  const core = [];
  for (const ch of String(ipa || '')) {
    if (IPA_VOCAB[ch] !== undefined) core.push(IPA_VOCAB[ch]);
  }
  if (core.length === 0) core.push(IPA_VOCAB['ə']);
  if (core.length > maxTokens) core.length = maxTokens;
  return [0, ...core, 0];
}
