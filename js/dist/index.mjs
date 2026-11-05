// Auto-generated ESM bundle for ninada v5.0
const { IPA_VOCAB, IPA_ID_TO_TOKEN, RETROFLEX_LATERAL_ID, textToIpa, ipaToIds } = (() => {
/**
 * Zero-dependency Pan-Indic & Classical/Vedic Sanskrit -> 24 kHz IPA Phonemizer for Ninada v3.0.
 * Exact 1-to-1 JavaScript mirror of python/src/ninada/ipa_g2p.py.
 */

const IPA_VOCAB = {
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

const IPA_ID_TO_TOKEN = Object.fromEntries(
  Object.entries(IPA_VOCAB).map(([k, v]) => [v, k])
);
const RETROFLEX_LATERAL_ID = 79;

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
  'व': 'ʋ', 'श': 'ʃ', 'ष': 'ʂ', 'स': 's', 'ह': 'h',
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

const LONG_VOWEL_IPA = new Set(['ɑː', 'iː', 'uː', 'ɾiː', 'eː', 'ɛː', 'æː', 'oː', 'ɔː', 'aɪ', 'əɪ', 'aʊ', 'əʊ']);
const VOWEL_CHARS = new Set([...'əɐʌɑɪiuʊeɛæoɔaᵊ']);
const RELEASE_TAIL_CHARS = new Set([...VOWEL_CHARS, 'ː', '\u0303']);

function vowelIpa(ch, isMatra, lang) {
  if (isMatra) {
    if (ch === 'ै') return lang === 'hi' ? 'ɛː' : 'əɪ';
    if (ch === 'ौ') return lang === 'hi' ? 'ɔː' : 'əʊ';
    return MATRAS_COMMON[ch] || '';
  }
  if (ch === 'ऐ') return lang === 'hi' ? 'ɛː' : 'əɪ';
  if (ch === 'औ') return lang === 'hi' ? 'ɔː' : 'əʊ';
  return INDEP_VOWELS_COMMON[ch] || '';
}

function normalizeScriptWordToDev(word) {
  const out = [];
  const norm = [...String(word).normalize('NFC')];
  const n = norm.length;
  for (let i = 0; i < n; i++) {
    const ch = norm[i];
    if (ML_CHILLU_IPA[ch] !== undefined) {
      out.push(ch);
      continue;
    }
    const cp = ch.codePointAt(0);
    if (cp >= 0x0980 && cp <= 0x09ff) {
      if (cp === 0x09ce) out.push('त्');
      else if (cp === 0x09f0) out.push('र');
      else if (cp === 0x09f1) out.push('ब');
      else if (cp === 0x09d7) out.push('ौ');
      else if (cp === 0x09b0) out.push('र');
      else out.push(String.fromCodePoint(cp - 0x0980 + 0x0900));
    } else if (cp >= 0x0a00 && cp <= 0x0a7f) {
      if (cp === 0x0a70) out.push('ं');
      else if (cp === 0x0a71) {
        if (i + 1 < n) {
          const ncp = norm[i + 1].codePointAt(0);
          if (ncp >= 0x0a15 && ncp <= 0x0a39) {
            out.push(`${String.fromCodePoint(ncp - 0x0a00 + 0x0900)}्`);
          }
        }
      } else if (cp === 0x0a5c) out.push('ड़');
      else if (cp === 0x0a72) out.push('इ');
      else if (cp === 0x0a73) out.push('उ');
      else out.push(String.fromCodePoint(cp - 0x0a00 + 0x0900));
    } else if (cp >= 0x0a80 && cp <= 0x0aff) {
      out.push(String.fromCodePoint(cp - 0x0a80 + 0x0900));
    } else if (cp >= 0x0b00 && cp <= 0x0b7f) {
      if (cp === 0x0b5f) out.push('य');
      else if (cp === 0x0b71) out.push('व');
      else if (cp === 0x0b56) out.push('ै');
      else if (cp === 0x0b57) out.push('ौ');
      else out.push(String.fromCodePoint(cp - 0x0b00 + 0x0900));
    } else if (cp >= 0x0b80 && cp <= 0x0bff) {
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

  for (let idx = 0; idx < units.length - 1; idx++) {
    const uCur = units[idx];
    const uNxt = units[idx + 1];
    if (uCur.rawC === 'ज' && uCur.vowel === '' && uNxt.rawC === 'ञ') {
      uCur.cons = 'ɡ';
      uNxt.cons = lang === 'hi' ? 'j' : 'ɲ';
    }
  }

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

  if (lang === 'bn' || lang === 'or') {
    for (const u of units) {
      if (u.inherent && u.vowel === 'ə') u.vowel = 'ɔ';
      if (lang === 'bn' && (u.rawC === 'श' || u.rawC === 'ष' || u.rawC === 'स') && (u.cons === 'ʂ' || u.cons === 's')) {
        u.cons = 'ʃ';
      }
    }
  }
  if (lang === 'hi' || lang === 'mr' || lang === 'gu' || lang === 'pa') {
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
        const isGeminate = Boolean(
          (prevBare && prevRaw === raw) ||
          (idx + 1 < units.length && u.vowel === '' && units[idx + 1].rawC === raw)
        );
        const prevNasal = Boolean(
          (prevBare && ML_NASALS.has(prevRaw)) || (prevU && IPA_NASALS.has(prevU.nasal))
        );
        if (prevNasal && POST_NASAL_VOICING[raw]) {
          u.cons = POST_NASAL_VOICING[raw];
        } else if (!isGeminate && idx > 0 && u.vowel !== '' && prevU && prevU.vowel !== '') {
          if (raw === 'ट' || raw === 'त') {
            u.cons = POST_NASAL_VOICING[raw];
          }
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
      let foundLong = false;
      for (const vIdx of vowelIndices) {
        if (LONG_VOWEL_IPA.has(units[vIdx].vowel)) {
          chosen = vIdx;
          foundLong = true;
          break;
        }
      }
      if (!foundLong) {
        for (const vIdx of vowelIndices) {
          const u = units[vIdx];
          const isClosed =
            Boolean(u.nasal) ||
            Boolean(u.visarga) ||
            (vIdx + 1 < units.length && units[vIdx + 1].vowel === '');
          if (isClosed) {
            chosen = vIdx;
            break;
          }
        }
      }
      units[chosen].stress = true;
    }
  }

  for (let idx = 0; idx < units.length; idx++) {
    const u = units[idx];
    if (u.stress && u.vowel === 'ə') {
      u.vowel = 'ʌ';
    } else if (u.stress && u.vowel === 'əɪ') {
      u.vowel = 'ʌɪ';
    } else if (u.stress && u.vowel === 'əʊ') {
      u.vowel = 'ʌʊ';
    } else if (
      lang !== 'hi' &&
      idx > 0 &&
      idx < units.length - 1 &&
      u.inherent &&
      u.vowel === 'ə' &&
      !u.stress &&
      !u.nasal &&
      !u.visarga
    ) {
      u.vowel = 'ɐ';
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
  '0': 'ʃˈuːnjə', '1': 'ˈeːk', '2': 'dˈoː', '3': 'tˈiːn', '4': 'ʧˈɑːɾ',
  '5': 'pˈɑ̃ːʧ', '6': 'ʧʰˈɛː', '7': 'sˈɑːt', '8': 'ˈɑːʈʰ', '9': 'nˈɔː',
};
const DIGIT_WORDS_EN = {
  '0': 'zˈɪːɾoː', '1': 'wˈʌn', '2': 'tˈuː', '3': 'tʰɾˈiː', '4': 'fˈɔːɾ',
  '5': 'fˈaɪv', '6': 'sˈɪks', '7': 'sˈɛvən', '8': 'ˈeːt', '9': 'nˈaɪn',
};
const INDIC_ZERO_BASES = [0x0966, 0x09e6, 0x0a66, 0x0ae6, 0x0b66, 0x0be6, 0x0c66, 0x0ce6, 0x0d66];
const LATIN_CHAR_IPA = {
  a: 'ə', b: 'b', c: 'k', d: 'd', e: 'ɛ', f: 'f', g: 'ɡ',
  h: 'h', i: 'ɪ', j: 'ʤ', k: 'k', l: 'l', m: 'm', n: 'n',
  o: 'o', p: 'p', q: 'k', r: 'ɾ', s: 's', t: 'ʈ', u: 'ʊ',
  v: 'ʋ', w: 'ʋ', x: 'ks', y: 'j', z: 'z',
};

function latinWordToIpa(word) {
  let w = String(word || '').toLowerCase().replace(/^'+|'+$/g, '');
  if (!w) return '';
  for (const [dg, rep] of [
    ['tion', 'ʃən'], ['sion', 'ʒən'], ['igh', 'aɪ'], ['tch', 'ʧ'],
    ['sh', 'ʃ'], ['ch', 'ʧ'], ['th', 'tʰ'], ['ph', 'f'], ['wh', 'ʋ'],
    ['ng', 'ŋ'], ['ee', 'iː'], ['ea', 'iː'], ['oo', 'uː'], ['ou', 'aʊ'],
    ['ai', 'eː'], ['ay', 'eː'], ['oa', 'oː'], ['ow', 'oː'], ['ar', 'ɑːɾ'],
    ['er', 'əɾ'], ['ir', 'əɾ'], ['ur', 'əɾ'], ['or', 'ɔːɾ'], ['ck', 'k'],
  ]) {
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

function textToIpa(text, lang = 'sa', normalizedDevFallback = null) {
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
  if (lang !== 'en' && normalizedDevFallback && /[A-Za-z\u0100-\u024F]/.test(raw)) {
    work = String(normalizedDevFallback).replace(/\u0965/g, '.').replace(/\u0964/g, ',');
  }

  const tokens = work.match(/[^\s,.;:!?—…"()“”]+|[,.;:!?—…"()“”]+|\s+/g) || [];
  const outChunks = [];
  const digitMap = lang === 'en' ? DIGIT_WORDS_EN : DIGIT_WORDS_HI;

  for (const tok of tokens) {
    if (/^\s+$/.test(tok)) {
      outChunks.push(' ');
      continue;
    }
    if (/^[,.;:!?—…"()“”]+$/.test(tok)) {
      const p = tok[0];
      const prev = outChunks.length > 0 ? outChunks[outChunks.length - 1] : '';
      const lastChar = prev ? [...prev].pop() : '';
      if (lastChar && RELEASE_TAIL_CHARS.has(lastChar) && GUARD_PUNCT.has(p)) {
        outChunks.push(` ${p}`);
      } else {
        outChunks.push(p);
      }
      continue;
    }
    if (/^\d+$/.test(tok)) {
      outChunks.push([...tok].map((d) => digitMap[d] || 'ʃˈuːnjə').join(' '));
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
  if (RELEASE_TAIL_CHARS.has(lastFiltered)) {
    filtered = `${filtered} .`;
  }
  filtered = filtered.replace(/e (?=[.,!?;:])/g, 'ɛ ');
  return filtered;
}

function ipaToIds(ipa, maxTokens = 510) {
  const core = [];
  for (const ch of String(ipa || '')) {
    if (IPA_VOCAB[ch] !== undefined) core.push(IPA_VOCAB[ch]);
  }
  if (core.length === 0) core.push(IPA_VOCAB['ə']);
  if (core.length > maxTokens) core.length = maxTokens;
  return [0, ...core, 0];
}

  return { IPA_VOCAB, IPA_ID_TO_TOKEN, RETROFLEX_LATERAL_ID, textToIpa, ipaToIds };
})();
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
const PAD_TOKEN = '_';
const BLANK_ID = TOKEN_TO_ID[PAD_TOKEN] ?? 33;
const PAD_TOKEN_ID = {
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
const VIRAMA_TOKEN_IDS = {
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

const LANGUAGE_IDS = {
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
    if (/[\p{L}\p{M}\p{N}]/u.test(ch)) letters.push(ch);
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

const NATIVE_TOKEN_TO_ID = Object.fromEntries(
  Object.entries(NATIVE_VOCABS).map(([lc, toks]) => [lc, Object.fromEntries(toks.map((t, i) => [t, i]))])
);

const NATIVE_DONOR_MAP = {
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

const VEDIC_SVARA_MARKS = {
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
    else if (c >= '0' && c <= '9') out.push(c);
    i += 1;
  }
  if (prevCons) out.push(VIRAMA);
  return out.join('');
}

const EN_DIGIT_DEV = {
  '0': 'जीरो', '1': 'वन', '2': 'टू', '3': 'थ्री', '4': 'फोर',
  '5': 'फाइव', '6': 'सिक्स', '7': 'सेवन', '8': 'एट', '9': 'नाइन',
};

const EN_DIGIT_WORDS = {
  '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
  '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
};

const PA_DIGIT_DEV = {
  '0': 'सिफ़र',
  '1': 'इक्क',
  '2': 'दो',
  '3': 'तिन्न',
  '4': 'चार',
  '5': 'पंज',
  '6': 'छे',
  '7': 'सत्त',
  '8': 'अट्ਠ',
  '9': 'नौं',
};

function englishToDevanagari(text) {
  let s = String(text).normalize('NFC').trim();
  s = s
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .replace(/([A-Za-z])(\d)/g, '$1 $2')
    .replace(/(\d)([A-Za-z])/g, '$1 $2')
    .toLowerCase();
  for (const [d, w] of Object.entries(EN_DIGIT_WORDS)) {
    s = s.replace(new RegExp(`\\b${d}\\b`, 'g'), ` ${w} `);
  }
  s = s.replace(/\s+/g, ' ').trim();
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
        if (rom === 'e' && prevC && out.length >= 2 && (i + 1 === n || /\s/.test(s[i + 1]) || /[,.?!]/.test(s[i + 1]))) {
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

function brahmicToDevanagari(text) {
  let norm = String(text).normalize('NFC');
  for (const [zwjSeq, atomic] of MALAYALAM_ZWJ_CHILLU) {
    norm = norm.split(zwjSeq).join(atomic);
  }
  norm = norm
    .replace(/\u0d4c/g, '\u0d57')
    .replaceAll('\u0a74', '\u0907\u0915 \u0913\u0902\u0915\u093e\u0930')
    .replaceAll('\u09af\u09bc', '\u09df');
  norm = norm.replace(/\u0a71([\u0a15-\u0a39\u0a59-\u0a5e]\u0a3c?)/g, '$1\u0a4d$1');
  norm = norm.replaceAll('\u0a71', '');
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

function deleteHindiSchwaWord(word) {
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

function deleteHindiSchwa(text) {
  return String(text)
    .split(' ')
    .map(deleteHindiSchwaWord)
    .join(' ');
}

function applyPhonologicalRules(devText, lang = 'sa') {
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

function toNativeModelText(text, lang, detectedScript, phonemicDev) {
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
    if (/\d/.test(sRaw) && !/\p{L}/u.test(sRaw)) {
      native = sRaw;
    } else {
      native = devanagariToTargetScript(phonemicDev, lang);
    }
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
        if (lang === 'pa') {
          mappedD = String.fromCodePoint(0x0a66 + (cp - (lo + 0x66)));
        } else {
          mappedD = String(cp - (lo + 0x66));
        }
        break;
      }
    }
    if (mappedD === null && lang === 'pa' && ch >= '0' && ch <= '9') {
      mappedD = String.fromCodePoint(0x0a66 + Number(ch));
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

function textToIds(
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
  let dev = activeLang === 'en' && detectedScript === 'latin'
    ? englishToDevanagari(text)
    : (detectedScript === 'iast' || detectedScript === 'latin')
    ? iastToDevanagari(text)
    : brahmicToDevanagari(text);
  if (activeLang === 'pa') {
    dev = [...dev].map((ch) => PA_DIGIT_DEV[ch] || ch).join('');
    dev = brahmicToDevanagari(dev);
  }
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

const { STUDIO_HOP_LENGTH, STUDIO_VOCABS, cleanStudioText, studioTextToIds } = (() => {
/**
 * Ninada v5.0 — Native Per-Language 24,000 Hz Studio VITS Character Tokenizer (Zero-Dep ESM).
 */


{ deleteHindiSchwa, deleteHindiSchwaWord };

const STUDIO_HOP_LENGTH = 256;
const STUDIO_SAMPLE_RATE = 24000;

const STUDIO_SPECS = Object.freeze({
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

const STUDIO_VOCABS = Object.freeze(
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
  [/\bst\./gi, 'saint'],
  [/\bvs\./gi, 'versus'],
  [/\binc\./gi, 'incorporated'],
  [/\bcorp\./gi, 'corporation'],
  [/\bltd\./gi, 'limited'],
  [/\bco\./gi, 'company'],
  [/\bjr\./gi, 'junior'],
  [/\bsr\./gi, 'senior'],
  [/\bdept\./gi, 'department'],
  [/\buniv\./gi, 'university'],
  [/\bno\./gi, 'number'],
  [/\bapprox\./gi, 'approximately'],
  [/\betc\./gi, 'et cetera'],
  [/\be\.g\./gi, 'for example'],
  [/\bi\.e\./gi, 'that is'],
  [/\brs\./gi, 'rupees'],
  [/\bcached\b/gi, 'cashed'],
  [/\bcaches\b/gi, 'cashes'],
  [/\bcaching\b/gi, 'cashing'],
  [/\bcache\b/gi, 'cash'],
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
  if (n < 1_000_000_000) {
    const m = Math.floor(n / 1_000_000);
    const r = n % 1_000_000;
    return r === 0 ? `${enIntToWords(m)} million` : `${enIntToWords(m)} million ${enIntToWords(r)}`;
  }
  if (n < 1_000_000_000_000) {
    const b = Math.floor(n / 1_000_000_000);
    const r = n % 1_000_000_000;
    return r === 0 ? `${enIntToWords(b)} billion` : `${enIntToWords(b)} billion ${enIntToWords(r)}`;
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
  let out = t
    .replace(/(?<=\d)\s*kHz\b/gi, ' kilohertz')
    .replace(/(?<=\d)\s*MHz\b/gi, ' megahertz')
    .replace(/(?<=\d)\s*GHz\b/gi, ' gigahertz')
    .replace(/\bkHz\b/gi, 'kilohertz')
    .replace(/\bMHz\b/gi, 'megahertz')
    .replace(/\bGHz\b/gi, 'gigahertz')
    .replace(/(?<=\d)\s*Hz\b/gi, ' hertz')
    .replace(/(?<=\d)\s*ms\b/gi, ' milliseconds')
    .replace(/(?<=\d)\s*MB\b/gi, ' megabytes')
    .replace(/(?<=\d)\s*GB\b/gi, ' gigabytes')
    .replace(/(?<=\d)\s*KB\b/gi, ' kilobytes')
    .replace(/\b(\d{1,12})(st|nd|rd|th)\b/gi, (_, num) => enOrdinalToWords(Number(num)))
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .replace(/([A-Za-z])(\d)/g, '$1 $2')
    .replace(/(\d)([A-Za-z])/g, '$1 $2');
  for (const [pat, rep] of EN_ABBREVS) {
    out = out.replace(pat, rep);
  }
  out = out.replace(/(?<=\d),(?=\d{3}\b)/g, '');

  const currMap = {
    $: ['dollar', 'dollars', 'cent', 'cents'],
    '€': ['euro', 'euros', 'cent', 'cents'],
    '₹': ['rupee', 'rupees', 'paise', 'paise'],
  };

  out = out.replace(/([$€₹])(\d+)\.(\d{2})\b/g, (_, sym, dStr, cStr) => {
    const dollars = Number(dStr);
    const cents = Number(cStr);
    const [cSing, cPlur, sSing, sPlur] = currMap[sym] || ['dollar', 'dollars', 'cent', 'cents'];
    const parts = [];
    if (dollars > 0 || cents === 0) {
      parts.push(`${enIntToWords(dollars)} ${dollars === 1 ? cSing : cPlur}`);
    }
    if (cents > 0) {
      const sName = cents === 1 ? sSing : sPlur;
      if (parts.length > 0) {
        parts.push(`and ${enIntToWords(cents)} ${sName}`);
      } else {
        parts.push(`${enIntToWords(cents)} ${sName}`);
      }
    }
    return parts.length > 0 ? parts.join(' ') : `zero ${cPlur}`;
  });

  out = out.replace(/([$€₹])(\d+)\b/g, (_, sym, valStr) => {
    const val = Number(valStr);
    const [cSing, cPlur] = currMap[sym] || ['dollar', 'dollars'];
    return `${enIntToWords(val)} ${val === 1 ? cSing : cPlur}`;
  });

  out = out
    .replaceAll('&', ' and ')
    .replaceAll('%', ' percent ')
    .replaceAll('₹', ' rupees ')
    .replaceAll('$', ' dollars ')
    .replaceAll('€', ' euros ');

  out = out.replace(/\b\d+(?:\.\d+)+\b/g, (s) => {
    const parts = s.split('.');
    if (parts.length === 2) {
      const whole = enIntToWords(Number(parts[0]));
      const fracWords = [...parts[1]].map((d) => EN_ONES[Number(d)]).join(' ');
      return `${whole} point ${fracWords}`;
    }
    const outParts = [];
    for (const p of parts) {
      if (p === '0') {
        outParts.push('zero');
      } else if (p.length > 1 && p.startsWith('0')) {
        outParts.push([...p].map((d) => EN_ONES[Number(d)]).join(' '));
      } else {
        outParts.push(enIntToWords(Number(p)));
      }
    }
    return outParts.join(' point ');
  });

  out = out.replace(/\b(\d{1,12})\b/g, (_, num) => enIntToWords(Number(num)));
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

function cleanStudioText(text, lang = 'sa') {
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
    t = t.replace(/\s*ऽ+/g, 'ऽ');
    t = t.replaceAll('||', '॥').replaceAll('|', '।');
  }
  if (lc === 'hi') {
    t = t.replace(/[\u0951-\u0954\u1cd0-\u1cff]/g, '');
    t = t.replaceAll('||', '।').replaceAll('|', '।').replaceAll('।।', '।');
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

function studioTextToIds(text, lang = null) {
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

  return { STUDIO_HOP_LENGTH, STUDIO_VOCABS, cleanStudioText, studioTextToIds };
})();
/**
 * Tier-2 Zero-Dependency Cascade Klatt-Akshara Synthesizer + 16-bit PCM WAV Encoder
 */


const SAMPLE_RATE = 24000;

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

  // Remove DC bias from cascaded resonators before nominal gain scaling
  let sum = 0.0;
  for (let idx = 0; idx < out.length; idx++) sum += out[idx];
  const mean = sum / out.length;
  for (let idx = 0; idx < out.length; idx++) out[idx] -= mean;

  for (let idx = 0; idx < out.length; idx++) {
    let y = 0.70 * out[idx];
    if (y > 0.96) {
      y = 0.96 + 0.03 * Math.tanh((y - 0.96) / 0.04);
    } else if (y < -0.96) {
      y = -0.96 - 0.03 * Math.tanh((-y - 0.96) / 0.04);
    }
    out[idx] = y;
  }

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

function splitLongUnpunctuated(segment, effMax) {
  const text = (segment || '').trim();
  if (!text) return [];
  if (text.length <= effMax) return [{ text, sentenceEnd: true }];

  const subParts = text.split(/(?<=[,:;—–])\s+/g).filter((s) => s && s.trim());
  if (subParts.length > 1) {
    const packed = [];
    let cur = '';
    for (const rawSub of subParts) {
      const s = rawSub.trim();
      if (!cur) {
        cur = s;
      } else if (cur.length + 1 + s.length <= effMax) {
        cur = `${cur} ${s}`;
      } else {
        const subChunks = splitLongUnpunctuated(cur, effMax);
        for (const sc of subChunks) sc.sentenceEnd = false;
        packed.push(...subChunks);
        cur = s;
      }
    }
    if (cur) {
      packed.push(...splitLongUnpunctuated(cur, effMax));
    }
    if (
      packed.length >= 2 &&
      packed[packed.length - 1].text.split(' ').length <= 2 &&
      packed[packed.length - 2].text.length + 1 + packed[packed.length - 1].text.length <= Math.ceil(effMax * 1.2)
    ) {
      const tail = packed.pop();
      packed[packed.length - 1].text = `${packed[packed.length - 1].text} ${tail.text}`;
      packed[packed.length - 1].sentenceEnd = tail.sentenceEnd;
    }
    return packed;
  }

  const words = text.split(' ').filter(Boolean);
  const nChunks = Math.max(2, Math.ceil(text.length / Math.max(40, effMax)));
  const targetChars = Math.ceil(text.length / nChunks);
  const out = [];
  let buf = '';
  for (const w of words) {
    if (!buf) {
      buf = w;
    } else if (buf.length + 1 + w.length <= targetChars || out.length === nChunks - 1) {
      buf = `${buf} ${w}`;
    } else {
      out.push({ text: buf, sentenceEnd: false });
      buf = w;
    }
  }
  if (buf) out.push({ text: buf, sentenceEnd: true });
  if (out.length >= 2 && out[out.length - 1].text.split(' ').length <= 2) {
    const tail = out.pop();
    out[out.length - 1].text = `${out[out.length - 1].text} ${tail.text}`;
    out[out.length - 1].sentenceEnd = true;
  } else if (out.length > 0) {
    out[out.length - 1].sentenceEnd = true;
  }
  return out;
}

const ABBREV_PROTECT_RE = /\b(?:Dr|Mr|Mrs|Ms|Prof|St|vs|Inc|Corp|Ltd|Co|Jr|Sr|Dept|Univ|No|etc|e\.g|i\.e|ie|approx|Rs)\./gi;

function buildStreamingClauses(text, maxChars = 100, fastFirstClause = false) {
  const cleaned = String(text || '').replace(/\s+/g, ' ').trim();
  if (!cleaned) return [];
  const effMax = maxChars;
  const protectedStr = cleaned.replace(ABBREV_PROTECT_RE, (m) => m.replace(/\./g, '\uE000'));
  const rawParts = protectedStr.split(
    /(?<=\|\|)(?:\s+|(?=[^\s|]))|(?<=[\u0964\u0965])(?:\s+|(?=[^\s\u0964\u0965]))|(?<=[\n?!;:]|(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|St|vs|Inc|Corp|Ltd|Co|Jr|Sr|Dept|Univ|No|etc|e\.g|i\.e|ie|approx|Rs))\.)(?:\s+|(?=[^\s\d.?!;:,]))|(?<=[,;—–])\s+/gi
  );
  const rawClauses = [];
  for (const part of rawParts) {
    const p = (part || '').trim();
    if (!p) continue;
    if (/^[\u0964\u0965|.?!;,\s\-—–:\uE000]+$/.test(p)) {
      if (rawClauses.length > 0) rawClauses[rawClauses.length - 1].text += ` ${p}`;
      continue;
    }
    if (p.length <= effMax) {
      rawClauses.push({ text: p, sentenceEnd: true });
      continue;
    }
    rawClauses.push(...splitLongUnpunctuated(p, effMax));
  }
  const merged = [];
  for (const item of rawClauses) {
    if (
      merged.length > 0 &&
      (merged[merged.length - 1].text.length < 6 ||
        (merged[merged.length - 1].text.length < 10 && !merged[merged.length - 1].text.includes(' '))) &&
      merged[merged.length - 1].text.length + 1 + item.text.length <= effMax
    ) {
      merged[merged.length - 1].text += ` ${item.text}`;
      merged[merged.length - 1].sentenceEnd = item.sentenceEnd;
    } else {
      merged.push(item);
    }
  }
  for (const item of merged) {
    item.text = item.text.replace(/\uE000/g, '.');
  }
  const result = merged.length > 0 ? merged : [{ text: cleaned, sentenceEnd: true }];

  if (
    result.length >= 2 &&
    result[result.length - 1].text.trim().split(/\s+/).length <= 2 &&
    /[,—–]\s*$/.test(result[result.length - 2].text)
  ) {
    const tail = result.pop();
    result[result.length - 1].text = `${result[result.length - 1].text} ${tail.text}`;
    result[result.length - 1].sentenceEnd = tail.sentenceEnd;
  }

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

function splitProsodicClauses(text, maxChars = 100) {
  return buildStreamingClauses(text, maxChars, false).map((c) => c.text);
}

function buildStudioSentenceBlocks(text, maxChars = 340) {
  const cleaned = String(text || '').replace(/\s+/g, ' ').trim();
  if (!cleaned) return [];
  const effMax = Math.max(80, Number(maxChars || 340));
  const protectedStr = cleaned.replace(ABBREV_PROTECT_RE, (m) => m.replace(/\./g, '\uE000'));
  // Split strictly on full sentence boundaries (dandas, periods, ?, !, newlines).
  // Never split on mid-sentence commas or semicolons so Studio VITS models phrase prosody natively.
  const rawParts = protectedStr.split(
    /(?<=\|\|)(?:\s+|(?=[^\s|]))|(?<=[\u0964\u0965])(?:\s+|(?=[^\s\u0964\u0965]))|(?<=[\n?!]|(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|St|vs|Inc|Corp|Ltd|Co|Jr|Sr|Dept|Univ|No|etc|e\.g|i\.e|ie|approx|Rs))\.)(?:\s+|(?=[^\s\d.?!,]))/gi
  );
  const rawBlocks = [];
  for (const part of rawParts) {
    const p = (part || '').trim();
    if (!p) continue;
    if (/^[\u0964\u0965|.?!;,\s\-—–:\uE000]+$/.test(p)) {
      if (rawBlocks.length > 0) rawBlocks[rawBlocks.length - 1].text += ` ${p}`;
      continue;
    }
    if (p.length <= effMax) {
      rawBlocks.push({ text: p, sentenceEnd: true });
    } else {
      rawBlocks.push(...splitLongUnpunctuated(p, effMax));
    }
  }
  const merged = [];
  for (const item of rawBlocks) {
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
  for (const item of merged) {
    item.text = item.text.replace(/\uE000/g, '.');
  }
  return merged.length > 0 ? merged : [{ text: cleaned, sentenceEnd: true }];
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
//   FLOAT [1, 1, "T"]   -> type { tensor_type { elem_type=1, shape { 1, 1, "T" } } }
const _VI_SUFFIX_FLOAT_1_1_T = new Uint8Array([
  18, 19, 10, 17, 8, 1, 18, 13, 10, 2, 8, 1, 10, 2, 8, 1, 10, 3, 18, 1, 84,
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

function _pbConcatParts(parts) {
  let total = 0;
  for (let i = 0; i < parts.length; i++) total += parts[i].byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (let i = 0; i < parts.length; i++) {
    out.set(parts[i], off);
    off += parts[i].byteLength;
  }
  return out;
}

/**
 * Splits a monolithic Studio VITS 24 kHz `.onnx` binary (`input_ids, scales -> waveform, durations`)
 * in memory (< 8ms, zero weight duplication) into three disjoint stages:
 *   - `priorBytes`: `TextEncoder + DurationPredictor` (`input_ids, scales -> z_p, durations`)
 *   - `flowBytes`: `NormalizingFlow` (`z_p, y_mask -> latent`)
 *   - `vocoderBytes`: `HiFi-GAN WaveformDecoder` (`latent -> wav3d`)
 * Plus `encFlowBytes` (2-way `prior + flow` union) when requested or when `/flow/` is absent.
 * Returns `null` if the ONNX graph is not a Studio VITS model with `/waveform_decoder/conv_pre/Conv`.
 */
function splitStudioVitsOnnxBuffer(rawBuffer, splitOpts = {}) {
  try {
    const unwrappedBuf =
      rawBuffer && typeof rawBuffer === 'object' && 'bytes' in rawBuffer ? rawBuffer.bytes : rawBuffer;
    let buf = null;
    if (unwrappedBuf instanceof Uint8Array) {
      buf = unwrappedBuf;
    } else if (unwrappedBuf instanceof ArrayBuffer) {
      buf = new Uint8Array(unwrappedBuf);
    } else if (ArrayBuffer.isView(unwrappedBuf)) {
      buf = new Uint8Array(unwrappedBuf.buffer, unwrappedBuf.byteOffset, unwrappedBuf.byteLength);
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
        let opType = '';
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
          } else if (nf === 4 && nwt === 2) {
            const [sLen, sStart] = _pbReadVarint(buf, np1);
            opType = _pbDecodeUtf8(buf, sStart, sStart + sLen);
          } else if (nf === 5 && nwt === 2) {
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
        nodes.push({ raw: buf.subarray(fStart, fEnd), inputs, outputs, extraInputs, name, opType });
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

    let firstFlowNode = null;
    let convPreIdx = -1;
    let convPreNode = null;
    let lastDecNode = null;
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      if (!firstFlowNode && n.name.startsWith('/flow/') && n.opType !== 'Constant') {
        firstFlowNode = n;
      }
      if (convPreIdx < 0 && (n.name === '/waveform_decoder/conv_pre/Conv' || n.name.endsWith('waveform_decoder/conv_pre/Conv'))) {
        convPreIdx = i;
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

    function buildSubmodelBytes(sub, inputValueInfos, outputValueInfos) {
      const graphParts = [];
      for (const idx of sub.indices) graphParts.push(nodes[idx].raw);
      if (graphNameSlice) graphParts.push(graphNameSlice);
      for (const init of initializers) {
        if (sub.usedInputs.has(init.name)) graphParts.push(init.raw);
      }
      for (const vi of inputValueInfos) graphParts.push(vi);
      for (const vi of outputValueInfos) graphParts.push(vi);
      const graphField = _pbWrapLengthDelimited(7, graphParts);
      return _pbConcatParts([...modelMetaSlices, graphField]);
    }

    const vocSub = collectSubgraph([wav3dName], new Set([latentName]));
    if (vocSub.indices.length === 0) return null;

    const vocoderBytes = buildSubmodelBytes(
      vocSub,
      [_pbMakeValueInfoField(11, latentName, _VI_SUFFIX_FLOAT_1_192_T)],
      [_pbMakeValueInfoField(12, wav3dName, _VI_SUFFIX_FLOAT_1_1_N)]
    );

    const latentProdIdx = producerByTensor.get(latentName);
    const latentProdNode = latentProdIdx !== undefined ? nodes[latentProdIdx] : null;
    const candidateZp = firstFlowNode ? firstFlowNode.inputs.find(Boolean) : '';
    const candidateYMask =
      latentProdNode && latentProdNode.opType === 'Mul' && latentProdNode.inputs.length >= 2
        ? latentProdNode.inputs[1]
        : '';

    let priorBytes = null;
    let flowBytes = null;
    let zpName = null;
    let yMaskName = null;

    if (candidateZp && candidateYMask) {
      const priorSub = collectSubgraph([candidateZp, 'durations'], new Set(['input_ids', 'scales']));
      const flowSub = collectSubgraph([latentName], new Set([candidateZp, candidateYMask]));
      // Verify flowSub has zero dynamic dependencies back to input_ids or scales
      const flowHasTokenDeps =
        flowSub.usedInputs.has('input_ids') || flowSub.usedInputs.has('scales');
      if (priorSub.indices.length > 0 && flowSub.indices.length > 0 && !flowHasTokenDeps) {
        zpName = candidateZp;
        yMaskName = candidateYMask;
        const priorIns = graphInputs
          .filter((gi) => gi.name === 'input_ids' || gi.name === 'scales')
          .map((gi) => gi.raw);
        priorBytes = buildSubmodelBytes(
          priorSub,
          priorIns,
          [_pbMakeValueInfoField(12, zpName, _VI_SUFFIX_FLOAT_1_192_T), durOutputRec.raw]
        );
        flowBytes = buildSubmodelBytes(
          flowSub,
          [
            _pbMakeValueInfoField(11, zpName, _VI_SUFFIX_FLOAT_1_192_T),
            _pbMakeValueInfoField(11, yMaskName, _VI_SUFFIX_FLOAT_1_1_T),
          ],
          [_pbMakeValueInfoField(12, latentName, _VI_SUFFIX_FLOAT_1_192_T)]
        );
      }
    }

    let encFlowBytes = null;
    if (!priorBytes || !flowBytes || !splitOpts.skipCombinedEncFlow) {
      const encSub = collectSubgraph([latentName, 'durations'], new Set(['input_ids', 'scales']));
      if (encSub.indices.length === 0) return null;
      const encIns = graphInputs
        .filter((gi) => gi.name === 'input_ids' || gi.name === 'scales')
        .map((gi) => gi.raw);
      encFlowBytes = buildSubmodelBytes(
        encSub,
        encIns,
        [_pbMakeValueInfoField(12, latentName, _VI_SUFFIX_FLOAT_1_192_T), durOutputRec.raw]
      );
    }

    return {
      priorBytes,
      flowBytes,
      encFlowBytes,
      vocoderBytes,
      zpName,
      yMaskName,
      latentName,
      wav3dName,
    };
  } catch {
    return null;
  }
}

async function createSplitStudioVitsSession(ort, rawBytesOrBox, sessionOpts) {
  if (!ort || !ort.InferenceSession) return null;
  const isBoxed = Boolean(rawBytesOrBox && typeof rawBytesOrBox === 'object' && 'bytes' in rawBytesOrBox);
  const split = splitStudioVitsOnnxBuffer(rawBytesOrBox, { skipCombinedEncFlow: true });
  if (!split) {
    const fallbackRaw = isBoxed ? rawBytesOrBox.bytes : rawBytesOrBox;
    if (isBoxed) rawBytesOrBox.bytes = null;
    return ort.InferenceSession.create(fallbackRaw, sessionOpts);
  }

  // Release caller's 28 MB raw buffer immediately before compiling WASM/WebGPU sessions
  if (isBoxed) {
    rawBytesOrBox.bytes = null;
  }

  const { zpName, yMaskName, latentName, wav3dName } = split;
  let priorBytes = split.priorBytes;
  let flowBytes = split.flowBytes;
  let encFlowBytes = split.encFlowBytes;
  let vocoderBytes = split.vocoderBytes;
  split.priorBytes = null;
  split.flowBytes = null;
  split.encFlowBytes = null;
  split.vocoderBytes = null;

  const baseProviders = Array.isArray(sessionOpts?.executionProviders)
    ? sessionOpts.executionProviders
    : ort.env?.wasm
    ? ['wasm']
    : ['cpu'];
  const vocProviders = Array.isArray(sessionOpts?.vocoderExecutionProviders)
    ? sessionOpts.vocoderExecutionProviders
    : baseProviders;
  const wantsWebGpuVocoder = vocProviders.some(
    (p) => p === 'webgpu' || (p && typeof p === 'object' && p.name === 'webgpu')
  );
  const cpuBackend = baseProviders.some((p) => p === 'cpu') && !ort.env?.wasm ? ['cpu'] : ['wasm'];

  // Keep INT8 weights compact on CPU/WASM (saves >250 MB RSS by avoiding FP32 ConstantFolding + prepacking bloat)
  const wasmStageOpts = {
    executionProviders: cpuBackend,
    graphOptimizationLevel: cpuBackend[0] === 'wasm' ? 'disabled' : 'all',
    enableCpuMemArena: false,
    enableMemPattern: false,
  };

  const TensorCtor = ort.Tensor || DuckOrtTensor;
  let priorSession = null;
  let flowSession = null;
  let encFlowSession = null;
  let vocoderSession = null;
  let vocoderBackend = 'wasm';

  try {
    if (priorBytes && flowBytes && zpName && yMaskName) {
      priorSession = await ort.InferenceSession.create(priorBytes, wasmStageOpts);
      priorBytes = null;
      flowSession = await ort.InferenceSession.create(flowBytes, wasmStageOpts);
      flowBytes = null;
    } else if (encFlowBytes) {
      encFlowSession = await ort.InferenceSession.create(encFlowBytes, wasmStageOpts);
      encFlowBytes = null;
    }

    if (wantsWebGpuVocoder) {
      try {
        vocoderSession = await ort.InferenceSession.create(vocoderBytes, {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
          enableCpuMemArena: true,
          enableMemPattern: true,
        });
        vocoderBackend = 'webgpu';
      } catch (gpuErr) {
        if (sessionOpts?.strictWebGpu) {
          throw gpuErr;
        }
        vocoderSession = await ort.InferenceSession.create(vocoderBytes, wasmStageOpts);
        vocoderBackend = cpuBackend[0];
      }
    } else {
      vocoderSession = await ort.InferenceSession.create(vocoderBytes, wasmStageOpts);
      vocoderBackend = cpuBackend[0];
    }
    vocoderBytes = null;
  } catch (err) {
    try { await priorSession?.release?.(); } catch {}
    try { await flowSession?.release?.(); } catch {}
    try { await encFlowSession?.release?.(); } catch {}
    try { await vocoderSession?.release?.(); } catch {}
    throw err;
  }

  // Compatibility wrapper when 3-way split is active so external callers using encFlowSession.run() still work
  if (!encFlowSession && priorSession && flowSession) {
    encFlowSession = {
      async run(feeds) {
        const pOut = await priorSession.run(feeds);
        const zpT = pOut[zpName];
        const T = Number(zpT?.dims?.[2] || Math.max(1, Math.floor((zpT?.data?.length || 192) / 192)));
        const maskT = new TensorCtor('float32', new Float32Array(T).fill(1.0), [1, 1, T]);
        try {
          const fOut = await flowSession.run({ [zpName]: zpT, [yMaskName]: maskT });
          return {
            [latentName]: fOut[latentName],
            durations: pOut.durations,
          };
        } finally {
          zpT?.dispose?.();
          maskT?.dispose?.();
        }
      },
      async release() {},
    };
  }

  return {
    inputNames: ['input_ids', 'scales'],
    outputNames: ['waveform', 'durations'],
    isSplitStudioVits: true,
    isThreeWaySplit: Boolean(priorSession && flowSession && zpName && yMaskName),
    priorSession,
    flowSession,
    encFlowSession,
    vocoderSession,
    vocoderBackend,
    zpName,
    yMaskName,
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
      try { await priorSession?.release?.(); } catch {}
      try { await flowSession?.release?.(); } catch {}
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
  const sessionOpts = {
    executionProviders: useProviders,
    graphOptimizationLevel: useProviders[0] === 'wasm' ? 'disabled' : 'all',
    enableCpuMemArena: false,
    enableMemPattern: false,
  };
  if (resolved instanceof Uint8Array || resolved instanceof ArrayBuffer || ArrayBuffer.isView(resolved)) {
    const rawBox = { bytes: resolved };
    resolved = null;
    return createSplitStudioVitsSession(ort, rawBox, sessionOpts);
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
      const clauses = isV4
        ? buildStudioSentenceBlocks(text, callOpts.maxClauseChars || 220)
        : buildStreamingClauses(text, callOpts.maxClauseChars || (is24k ? 140 : 220), fastFirst);
      const tStart = nowMs();
      let ttfaMs = null;
      let chunkIndex = 0;
      let sampleOffset = 0;

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
              sampleOffset: 0,
              sample_offset: 0,
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

        const rawNoise = callOpts.noise ?? opts.noise;
        const effNoise = Math.max(
          0.0,
          Math.min(1.2, Number(rawNoise ?? (lang === 'en' ? 0.56 : 0.40)))
        );
        const rawNoiseDp =
          callOpts.noiseDuration ??
          callOpts.noise_duration ??
          opts.noiseDuration ??
          opts.noise_duration;
        const effNoiseDp = Math.max(
          0.0,
          Math.min(
            1.2,
            Number(rawNoiseDp ?? (lang === 'en' ? Math.min(0.80, effNoise * 1.10) : 0.0))
          )
        );
        const chunkSamples = chunkFrames * HOP_LENGTH;
        const firstChunkSamples = firstChunkFrames * HOP_LENGTH;
        const minTailSamples = 16 * STUDIO_HOP_LENGTH;

        const precomputedStudio = clauses.map((item) => {
          const parsedItem = textToIds(item.text, { lang });
          if (!hasPronounceableContent(parsedItem)) {
            return { parsed: parsedItem, studioIds: null, resolvedLc: parsedItem.lang, weight: 0 };
          }
          const st = studioTextToIds(item.text, parsedItem.lang);
          return {
            parsed: parsedItem,
            studioIds: st.studioIds,
            resolvedLc: st.lang,
            weight: Math.max(8, st.studioIds.length + 18),
          };
        });
        const totalStudioWeight = Math.max(
          1,
          precomputedStudio.reduce((acc, item) => acc + item.weight, 0)
        );

        for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
          if (signal?.aborted) return;
          const clauseItem = clauses[cIdx];
          const tClause = nowMs();
          const preItem = precomputedStudio[cIdx];
          const parsed = preItem.parsed;

          if (!preItem.studioIds) {
            const silent = generateRoomTone(Math.round(0.10 * SAMPLE_RATE), 1337 + cIdx);
            if (ttfaMs === null) ttfaMs = nowMs() - tStart;
            const chunkObj = attachLazyWav(
              {
                pcm: silent,
                sampleRate: SAMPLE_RATE,
                index: chunkIndex++,
                sampleOffset: sampleOffset,
                sample_offset: sampleOffset,
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
            sampleOffset += silent.length;
            emit('chunk', chunkObj);
            yield chunkObj;
            continue;
          }

          const studioIds = preItem.studioIds;
          const resolvedLc = preItem.resolvedLc;
          let activeSess = (await resolveV4Session(resolvedLc, callOpts.voice || null)) || v4InitialSess;
          const idTensor = new TensorCtor('int64', BigInt64Array.from(studioIds.map(BigInt)), [1, studioIds.length]);
          const scalesTensor = new TensorCtor('float32', Float32Array.from([1.0 / effRate, effNoise, effNoiseDp]), [3]);

          // Sub-Clause 3-Stage Latent-Window Streaming Path (prior -> incremental flow -> static-bucket HiFi-GAN)
          if (activeSess?.isSplitStudioVits && !callOpts._monolithicVocoder) {
            const isThreeWay = Boolean(
              activeSess.isThreeWaySplit &&
              activeSess.priorSession &&
              activeSess.flowSession &&
              activeSess.zpName &&
              activeSess.yMaskName
            );
            const isWebGpuVoc = activeSess.vocoderBackend === 'webgpu';
            const isMobileWasm = !isWebGpuVoc && Boolean(callOpts.mobileWasmStreaming);
            const haloLatent = 12;
            const flowHaloLatent = 24;
            const firstBucket = Math.max(
              36,
              Number(callOpts.firstBucketFrames || (isWebGpuVoc ? 44 : 56))
            );
            const isSingleThreadWasm = isMobileWasm && Boolean(callOpts.singleThread);
            const steadyBucket = isWebGpuVoc ? 112 : isSingleThreadWasm ? 168 : isMobileWasm ? 120 : 96;
            const steadyValid = steadyBucket - 2 * haloLatent;
            const xfadeSamples = 512;

            let latentChannels = 192;
            let totalLatentFrames = 1;
            let zpFlat = null;
            let zFlat = null;
            let zComputedUpTo = 0;
            let dursArr = [];
            let encFlowMs = 0;

            if (isThreeWay) {
              const tPrior0 = nowMs();
              const priorOut = await activeSess.priorSession.run({
                input_ids: idTensor,
                scales: scalesTensor,
              });
              encFlowMs = nowMs() - tPrior0;
              idTensor.dispose?.();
              scalesTensor.dispose?.();

              const zpTensor = priorOut[activeSess.zpName];
              const dursTensor = priorOut.durations;
              if (signal?.aborted || !zpTensor) {
                zpTensor?.dispose?.();
                dursTensor?.dispose?.();
                return;
              }
              const zpDims = zpTensor.dims || [1, 192, Math.floor(zpTensor.data.length / 192)];
              latentChannels = Number(zpDims[1] || 192);
              totalLatentFrames = Math.max(1, Number(zpDims[2] || Math.floor(zpTensor.data.length / latentChannels)));
              zpFlat = new Float32Array(zpTensor.data);
              dursArr = dursTensor ? Array.from(dursTensor.data, Number) : [];
              zpTensor.dispose?.();
              dursTensor?.dispose?.();
              zFlat = new Float32Array(latentChannels * totalLatentFrames);
            } else {
              const tEnc0 = nowMs();
              const encOut = await activeSess.encFlowSession.run({
                input_ids: idTensor,
                scales: scalesTensor,
              });
              encFlowMs = nowMs() - tEnc0;
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
              latentChannels = Number(zDims[1] || 192);
              totalLatentFrames = Math.max(1, Number(zDims[2] || Math.floor(zTensor.data.length / latentChannels)));
              zFlat = new Float32Array(zTensor.data);
              zComputedUpTo = totalLatentFrames;
              dursArr = dursTensor ? Array.from(dursTensor.data, Number) : [];
              zTensor.dispose?.();
              dursTensor?.dispose?.();
            }

            const ensureFlowUpTo = async (targetFrameEnd) => {
              if (!isThreeWay || zComputedUpTo >= targetFrameEnd || !zpFlat) return 0;
              const tFl0 = nowMs();
              // Compute bounded flow blocks (<= 192 valid frames) to cap peak activation RAM on low-memory devices
              while (zComputedUpTo < targetFrameEnd) {
                const isFirstFlow = zComputedUpTo === 0;
                const firstFlowCap = isWebGpuVoc
                  ? firstBucket
                  : isSingleThreadWasm
                  ? firstBucket
                  : isMobileWasm
                  ? 128
                  : firstBucket <= 44
                  ? 64
                  : 88;
                const blockValidCap = isFirstFlow
                  ? Math.max(targetFrameEnd, firstFlowCap)
                  : Math.max(targetFrameEnd - zComputedUpTo, 192);
                const nextValidEnd =
                  totalLatentFrames - zComputedUpTo <= blockValidCap + flowHaloLatent
                    ? totalLatentFrames
                    : Math.min(totalLatentFrames, zComputedUpTo + blockValidCap);
                const lFHalo = Math.min(zComputedUpTo, flowHaloLatent);
                const rFHalo = Math.min(totalLatentFrames - nextValidEnd, flowHaloLatent);
                const fSliceStart = zComputedUpTo - lFHalo;
                const fSliceLen = (nextValidEnd - zComputedUpTo) + lFHalo + rFHalo;

                const subZp = new Float32Array(latentChannels * fSliceLen);
                for (let ch = 0; ch < latentChannels; ch++) {
                  const sOff = ch * totalLatentFrames + fSliceStart;
                  subZp.set(zpFlat.subarray(sOff, sOff + fSliceLen), ch * fSliceLen);
                }
                const subZpTensor = new TensorCtor('float32', subZp, [1, latentChannels, fSliceLen]);
                const subMaskTensor = new TensorCtor(
                  'float32',
                  new Float32Array(fSliceLen).fill(1.0),
                  [1, 1, fSliceLen]
                );
                const flowOut = await activeSess.flowSession.run({
                  [activeSess.zpName]: subZpTensor,
                  [activeSess.yMaskName]: subMaskTensor,
                });
                subZpTensor.dispose?.();
                subMaskTensor.dispose?.();

                const flowLatTensor = flowOut[activeSess.latentName];
                if (flowLatTensor?.data) {
                  const fd = flowLatTensor.data;
                  const copyFrames = nextValidEnd - zComputedUpTo;
                  for (let ch = 0; ch < latentChannels; ch++) {
                    const srcBase = ch * fSliceLen + lFHalo;
                    zFlat.set(fd.subarray(srcBase, srcBase + copyFrames), ch * totalLatentFrames + zComputedUpTo);
                  }
                }
                flowLatTensor?.dispose?.();
                zComputedUpTo = nextValidEnd;
              }
              if (zComputedUpTo >= totalLatentFrames) {
                zpFlat = null;
              }
              return nowMs() - tFl0;
            };

            const isFinalClause = cIdx === clauses.length - 1;
            const endsWithDouble = /(?:॥|\|\||।।|।\s*।|\.\s*\.)\s*$/.test(clauseItem.text || '');
            // Indic Studio VITS emits ~230ms trailing breath at sentence boundaries; English trims tightly so add a natural 165ms breath pause.
            const interPauseSec =
              resolvedLc === 'en' || lang === 'en'
                ? clauseItem.sentenceEnd === false
                  ? 0.075
                  : 0.165
                : 0.035;
            const pauseSec = isFinalClause ? 0.120 : endsWithDouble ? Math.max(0.140, interPauseSec) : interPauseSec;
            const pauseSamples = Math.round((pauseSec / effRate) * SAMPLE_RATE);
            const clauseTotalSamples = totalLatentFrames * STUDIO_HOP_LENGTH + pauseSamples;
            const thisStudioWeight = Math.max(1, preItem.weight);
            const estimatedTotalSec =
              clauses.length === 1
                ? clauseTotalSamples / SAMPLE_RATE
                : (clauseTotalSamples / SAMPLE_RATE) * (totalStudioWeight / thisStudioWeight);

            const fBounds = [];
            for (let fCur = 0; fCur < totalLatentFrames; ) {
              const remFrames = totalLatentFrames - fCur;
              const useFastBucket = isMobileWasm
                ? fCur === 0
                : fCur === 0 || (!isWebGpuVoc && fBounds.length === 1);
              if (useFastBucket) {
                if (fCur === 0) {
                  const firstValidCap = firstBucket - haloLatent;
                  const fNext = totalLatentFrames <= firstBucket ? totalLatentFrames : Math.min(totalLatentFrames, firstValidCap);
                  fBounds.push([fCur, fNext, firstBucket]);
                  fCur = fNext;
                } else {
                  const secondValidCap = firstBucket - 2 * haloLatent;
                  const fNext =
                    remFrames <= firstBucket - haloLatent
                      ? totalLatentFrames
                      : fCur + secondValidCap;
                  fBounds.push([fCur, fNext, firstBucket]);
                  fCur = fNext;
                }
              } else if (isMobileWasm && fBounds.length === 1 && remFrames > 92) {
                // Zero-flow Bridge Chunk 1 (~0.64s audio, 84 vocoded frames, 0ms flow since Chunk 0 pre-computed 128 frames)
                const fNext = fCur + 60;
                fBounds.push([fCur, fNext, steadyBucket]);
                fCur = fNext;
              } else if (isMobileWasm && fBounds.length === 2 && remFrames > 104) {
                // Step-up Chunk 2 (~0.85s audio, 104 vocoded frames) smoothly transitions into 96-frame steady state
                const fNext = fCur + 80;
                fBounds.push([fCur, fNext, steadyBucket]);
                fCur = fNext;
              } else {
                let fNext;
                if (remFrames <= steadyBucket - haloLatent) {
                  fNext = totalLatentFrames;
                } else if (remFrames < steadyValid + 16) {
                  fNext = fCur + Math.ceil(remFrames / 2);
                } else {
                  fNext = fCur + steadyValid;
                }
                fBounds.push([fCur, fNext, steadyBucket]);
                fCur = fNext;
              }
            }

            const firstSliceBuf = new Float32Array(latentChannels * firstBucket);
            const steadySliceBuf = new Float32Array(latentChannels * steadyBucket);
            let prevRightHalo = null;
            const shouldYield = callOpts.yieldBetweenWindows !== false && !isNodeEnv();

            for (let wIdx = 0; wIdx < fBounds.length; wIdx++) {
              if (shouldYield && (wIdx > 0 || cIdx > 0)) {
                await yieldMacrotask();
              }
              if (signal?.aborted) return;

              const [fStart, fEnd, bucketF] = fBounds[wIdx];
              const validF = fEnd - fStart;
              const lHalo = Math.min(fStart, haloLatent);
              const rHalo = Math.min(totalLatentFrames - fEnd, haloLatent);
              const sliceStart = fStart - lHalo;
              const rawSliceF = validF + lHalo + rHalo;

              const flowStepMs = await ensureFlowUpTo(sliceStart + rawSliceF);
              if (signal?.aborted) return;

              const effBucketF = isMobileWasm ? rawSliceF : bucketF;
              const backingBuf = bucketF === firstBucket ? firstSliceBuf : steadySliceBuf;
              const zSliceBuf = effBucketF === bucketF ? backingBuf : backingBuf.subarray(0, latentChannels * effBucketF);
              for (let ch = 0; ch < latentChannels; ch++) {
                const srcOff = ch * totalLatentFrames + sliceStart;
                const dstOff = ch * effBucketF;
                zSliceBuf.set(zFlat.subarray(srcOff, srcOff + rawSliceF), dstOff);
                if (rawSliceF < effBucketF) {
                  const edgeVal = zFlat[srcOff + rawSliceF - 1] || 0.0;
                  zSliceBuf.fill(edgeVal, dstOff + rawSliceF, dstOff + effBucketF);
                }
              }

              const tVoc0 = nowMs();
              const zSliceTensor = new TensorCtor('float32', zSliceBuf, [1, latentChannels, effBucketF]);
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
              const unpaddedSamp = rawSliceF * STUDIO_HOP_LENGTH;
              const isLastSub = wIdx === fBounds.length - 1;
              const pcm = new Float32Array(validLenSamp + (isLastSub ? pauseSamples : 0));

              pcm.set(rawSlice.subarray(validStartSamp, validEndSamp), 0);

              if (prevRightHalo && validLenSamp >= xfadeSamples) {
                for (let i = 0; i < xfadeSamples; i++) {
                  const alpha = 0.5 * (1.0 - Math.cos((Math.PI * (i + 0.5)) / xfadeSamples));
                  pcm[i] = (1.0 - alpha) * prevRightHalo[i] + alpha * pcm[i];
                }
              }

              if (!isLastSub && unpaddedSamp >= validEndSamp + xfadeSamples) {
                prevRightHalo = Float32Array.from(rawSlice.subarray(validEndSamp, validEndSamp + xfadeSamples));
              } else {
                prevRightHalo = null;
              }
              wavTensor.dispose?.();

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

              if (isLastSub && pauseSamples > 0) {
                pcm.set(generateRoomTone(pauseSamples, 1337 + cIdx), validLenSamp);
              }

              for (let i = 0; i < pcm.length; i++) {
                const x = pcm[i] || 0;
                let y = x - dcXPrev + dcR * dcYPrev;
                dcXPrev = x;
                dcYPrev = y;
                if (y > 0.96) y = 0.96 + 0.03 * Math.tanh((y - 0.96) / 0.04);
                else if (y < -0.96) y = -0.96 - 0.03 * Math.tanh((-y - 0.96) / 0.04);
                pcm[i] = y;
              }
              if (Math.abs(dcYPrev) < 1e-15) dcYPrev = 0.0;

              const tTaper = Math.min(64, Math.floor(pcm.length / 2));
              if (tTaper > 0) {
                if (wIdx === 0) {
                  for (let k = 0; k < tTaper; k++) {
                    const wIn = 0.5 * (1.0 - Math.cos((Math.PI * (k + 0.5)) / tTaper));
                    pcm[k] *= wIn;
                  }
                }
                if (isLastSub) {
                  for (let k = 0; k < tTaper; k++) {
                    const wOut = 0.5 * (1.0 - Math.cos((Math.PI * (k + 0.5)) / tTaper));
                    pcm[pcm.length - 1 - k] *= wOut;
                  }
                }
              }

              if (pcm.length > 0) {
                let sum = 0.0;
                for (let i = 0; i < pcm.length; i++) sum += pcm[i];
                const dcMean = sum / pcm.length;
                if (Math.abs(dcMean) > 1e-6) {
                  const nSamp = pcm.length;
                  for (let i = 0; i < nSamp; i++) {
                    const wDc = 1.0 - Math.cos((2.0 * Math.PI * (i + 0.5)) / nSamp);
                    pcm[i] -= dcMean * wDc;
                  }
                }
              }

              if (ttfaMs === null) ttfaMs = nowMs() - tStart;

              let nextWindowSec = 0;
              if (!isLastSub) {
                const [nfStart, nfEnd] = fBounds[wIdx + 1];
                nextWindowSec = ((nfEnd - nfStart) * STUDIO_HOP_LENGTH) / SAMPLE_RATE;
              }

              const stageEncMs = wIdx === 0 ? encFlowMs + flowStepMs : flowStepMs;
              const chunkObj = attachLazyWav(
                {
                  pcm,
                  sampleRate: SAMPLE_RATE,
                  index: chunkIndex++,
                  sampleOffset: sampleOffset,
                  sample_offset: sampleOffset,
                  clauseIndex: cIdx,
                  clauseCount: clauses.length,
                  isFinal: cIdx === clauses.length - 1 && isLastSub,
                  durationSec: pcm.length / SAMPLE_RATE,
                  clauseDurationSec: clauseTotalSamples / SAMPLE_RATE,
                  estimatedTotalSec,
                  remainingClauseSec: ((totalLatentFrames - fEnd) * STUDIO_HOP_LENGTH) / SAMPLE_RATE,
                  nextWindowSec,
                  vocoderWindowAudioSec: (effBucketF * STUDIO_HOP_LENGTH) / SAMPLE_RATE,
                  vocoderBucketFrames: effBucketF,
                  vocoderBackend: activeSess.vocoderBackend || 'wasm',
                  encFlowMs: stageEncMs,
                  latencyMs: vocMs + stageEncMs,
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
              sampleOffset += pcm.length;
              emit('chunk', chunkObj);
              yield chunkObj;
            }
            zFlat = null;
            zpFlat = null;
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
          const isFinalClause = cIdx === clauses.length - 1;
          const endsWithDouble = /(?:॥|\|\||।।|।\s*।|\.\s*\.)\s*$/.test(clauseItem.text || '');
          const interPauseSec =
            resolvedLc === 'en' || lang === 'en'
              ? clauseItem.sentenceEnd === false
                ? 0.075
                : 0.165
              : 0.035;
          const pauseSec = isFinalClause ? 0.120 : (endsWithDouble ? Math.max(0.140, interPauseSec) : interPauseSec);
          const pauseSamples = Math.round((pauseSec / effRate) * SAMPLE_RATE);
          const clausePcm = new Float32Array(rawWav.length + pauseSamples);
          if (pauseSamples > 0) {
            clausePcm.set(generateRoomTone(pauseSamples, 1337 + cIdx), rawWav.length);
          }

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

          if (rawWav.length > 0) {
            const dcMean = dcMeanSum / rawWav.length;
            if (Math.abs(dcMean) > 1e-6) {
              const nSamp = rawWav.length;
              for (let i = 0; i < nSamp; i++) {
                const wDc = 1.0 - Math.cos((2.0 * Math.PI * (i + 0.5)) / nSamp);
                clausePcm[i] -= dcMean * wDc;
              }
            }
          }

          for (let i = 0; i < clausePcm.length; i++) {
            let val = clausePcm[i];
            if (val > 0.96) {
              val = 0.96 + 0.03 * Math.tanh((val - 0.96) / 0.04);
            } else if (val < -0.96) {
              val = -0.96 - 0.03 * Math.tanh((-val - 0.96) / 0.04);
            }
            clausePcm[i] = val;
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
                sampleOffset: sampleOffset,
                sample_offset: sampleOffset,
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
            sampleOffset += pcm.length;
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
              sampleOffset: 0,
              sample_offset: 0,
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
                sampleOffset: sampleOffset,
                sample_offset: sampleOffset,
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
            sampleOffset += silent.length;
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
                sampleOffset: sampleOffset,
                sample_offset: sampleOffset,
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
            sampleOffset += pcm.length;
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
                sampleOffset: sampleOffset,
                sample_offset: sampleOffset,
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
            sampleOffset += pcm.length;
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
              sampleOffset: sampleOffset,
              sample_offset: sampleOffset,
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
          sampleOffset += pcm.length;
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
          let y = x - dcXPrev + dcR * dcYPrev;
          dcXPrev = x;
          dcYPrev = y;
          if (y > 0.96) y = 0.96 + 0.03 * Math.tanh((y - 0.96) / 0.04);
          else if (y < -0.96) y = -0.96 - 0.03 * Math.tanh((-y - 0.96) / 0.04);
          pcm[idx] = y;
        }
        if (Math.abs(dcYPrev) < 1e-15) dcYPrev = 0.0;

        // Nominal gain factor (matching python/src/ninada/engine.py:442)
        for (let idx = 0; idx < pcm.length; idx++) {
          pcm[idx] = 0.70 * pcm[idx];
        }

        // 64-sample raised-cosine chunk edge taper
        const tTaper = Math.min(64, Math.floor(pcm.length / 2));
        if (tTaper > 0) {
          for (let k = 0; k < tTaper; k++) {
            const wIn = 0.5 * (1.0 - Math.cos((Math.PI * (k + 0.5)) / tTaper));
            pcm[k] *= wIn;
            pcm[pcm.length - 1 - k] *= wIn;
          }
        }

        // Windowed Hann DC removal (applied after taper to guarantee zero residual DC)
        if (pcm.length > 0) {
          let sum = 0.0;
          for (let i = 0; i < pcm.length; i++) sum += pcm[i];
          const dcMean = sum / pcm.length;
          if (Math.abs(dcMean) > 1e-6) {
            const nSamp = pcm.length;
            for (let i = 0; i < nSamp; i++) {
              const wDc = 1.0 - Math.cos((2.0 * Math.PI * (i + 0.5)) / nSamp);
              pcm[i] -= dcMean * wDc;
            }
          }
        }

        if (ttfaMs === null) ttfaMs = nowMs() - tStart;
        const chunkObj = attachLazyWav(
          {
            pcm,
            sampleRate: SAMPLE_RATE,
            index: chunkIndex++,
            sampleOffset: sampleOffset,
            sample_offset: sampleOffset,
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
        sampleOffset += pcm.length;
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
  buildStudioSentenceBlocks,
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
  deleteHindiSchwa,
  deleteHindiSchwaWord,
};
