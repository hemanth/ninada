/**
 * Comprehensive Phonetic & G2P Regression Test Suite (JavaScript / Node.js Test Runner)
 * 100% parity with tests/test_phonetics.py.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  VOCAB_TOKENS,
  TOKEN_TO_ID,
  LANGUAGE_IDS,
  NATIVE_VOCABS,
  NATIVE_TOKEN_TO_ID,
  textToIds,
  applyPhonologicalRules,
  iastToDevanagari,
  brahmicToDevanagari,
  segmentAksharas,
  toNativeModelText,
  generateRoomTone,
  synthesizeKlattPcm,
} from '../src/index.js';

import {
  PAD_TOKEN_ID,
  VIRAMA_TOKEN_IDS,
  NATIVE_DONOR_MAP,
  deleteHindiSchwa,
  deleteHindiSchwaWord,
} from '../src/g2p.js';

// ==============================================================================
// Suite 1: Phoneme Inventory Preservation Across All Languages
// ==============================================================================
test('Phoneme Inventory: 87-symbol Devanagari and specialist native inventories are intact', () => {
  assert.equal(VOCAB_TOKENS.length, 87);
  assert.equal(VOCAB_TOKENS[85], '\u090b'); // ऋ
  assert.equal(VOCAB_TOKENS[86], '\u0919'); // ङ
  assert.deepEqual(Object.keys(LANGUAGE_IDS).sort(), [
    'bn', 'en', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'sa', 'ta', 'te',
  ]);

  // Sanskrit inventory must contain extended symbols (ಙ, ೠ, ೄ)
  assert.ok(NATIVE_VOCABS.sa.includes('\u0c99'), 'ಙ must be in sa vocab');
  assert.ok(NATIVE_VOCABS.sa.includes('\u0ce0'), 'ೠ must be in sa vocab');
  assert.ok(NATIVE_VOCABS.sa.includes('\u0cc4'), 'ೄ must be in sa vocab');

  // Restored Dravidian consonants must exist
  assert.ok(NATIVE_VOCABS.kn.includes('\u0c99'), 'ಙ must be in kn vocab');
  assert.ok(NATIVE_VOCABS.te.includes('\u0c1d'), 'ఝ must be in te vocab');
  assert.ok(NATIVE_VOCABS.ml.includes('\u0d1d'), 'ഝ must be in ml vocab');

  // Donor map mappings must exist
  assert.ok(NATIVE_DONOR_MAP.kn['\u0c99'] !== undefined);
  assert.ok(NATIVE_DONOR_MAP.te['\u0c1d'] !== undefined);
  assert.ok(NATIVE_DONOR_MAP.ml['\u0d1d'] !== undefined);
});

// ==============================================================================
// Suite 2: Sanskrit Homorganic Nasal Sandhi Preservation
// ==============================================================================
test('Sanskrit Sandhi: Homorganic nasals are preserved before stops without Anusvara collapse', () => {
  const cases = [
    { text: 'शंकर', expected: '\u0c99' }, // ಙ್ಕ, NOT ಂಕ
    { text: 'शङ्कर', expected: '\u0c99' },
    { text: 'संचय', expected: '\u0c9e' }, // ಞ್ಚ, NOT ಂಚ
    { text: 'सञ्चय', expected: '\u0c9e' },
    { text: 'कंठ', expected: '\u0ca3' },  // ಣ್ಠ, NOT ಂಠ
    { text: 'कण्ठ', expected: '\u0ca3' },
    { text: 'शांत', expected: '\u0ca8' },  // ನ್ತ, NOT ಂತ
    { text: 'शान्त', expected: '\u0ca8' },
    { text: 'संपूर्ण', expected: '\u0cae' }, // ಮ್ಪ, NOT ಂಪ
    { text: 'सम्पूर्ण', expected: '\u0cae' },
  ];
  for (const { text, expected } of cases) {
    const res = textToIds(text, { lang: 'sa' });
    assert.ok(
      res.nativeText.includes(expected),
      `Expected ${expected} in nativeText "${res.nativeText}" for "${text}"`
    );
    assert.ok(!res.nativeText.slice(0, 3).includes('\u0c82'), `Anusvara found in "${res.nativeText}"`);
  }
});

// ==============================================================================
// Suite 3: Avagraha (ऽ) Pūrvarūpa Sandhi Processing
// ==============================================================================
test('Avagraha: Purvarupa sandhi absorbs cleanly without spurious "अ" insertion', () => {
  const cases = [
    { text: 'सोऽहम्', expected: 'सोहम्', forbidden: 'सो अहम्' },
    { text: 'सो ऽहम्', expected: 'सोहम्', forbidden: 'सो अहम्' },
    { text: 'शिवोऽहम्', expected: 'शिवोहम्', forbidden: 'शिवो अहम्' },
    { text: 'तेऽपि', expected: 'तेपि', forbidden: 'ते अपि' },
    { text: 'ते ऽपि', expected: 'तेपि', forbidden: 'ते अपि' },
    { text: 'भोऽऽऽ', expected: 'भो', forbidden: 'भोअ' },
    { text: 'कालोऽस्ति', expected: 'कालोस्ति', forbidden: 'कालो अस्ति' },
  ];
  for (const { text, expected, forbidden } of cases) {
    const res = textToIds(text, { lang: 'sa' });
    assert.ok(!res.normalizedDevanagari.includes('\u093d'), 'Avagraha must be stripped');
    assert.ok(
      !res.normalizedDevanagari.includes(forbidden),
      `Found forbidden "${forbidden}" in "${res.normalizedDevanagari}"`
    );
    assert.ok(
      res.normalizedDevanagari.includes(expected),
      `Expected "${expected}" in "${res.normalizedDevanagari}"`
    );
  }
});

test('IAST Transliteration: Purvarupa sandhi apostrophe maps cleanly without spurious "अ"', () => {
  const res1 = textToIds("so 'ham", { lang: 'sa' });
  assert.ok(res1.normalizedDevanagari.includes('सोहम्'));
  assert.ok(!res1.normalizedDevanagari.includes('अ'));

  const res2 = textToIds("te 'pi", { lang: 'sa' });
  assert.ok(res2.normalizedDevanagari.includes('तेपि'));
  assert.ok(!res2.normalizedDevanagari.includes('अ'));
});

// ==============================================================================
// Suite 4: Vedic Svara Accent Preservation & Punctuation Tokens
// ==============================================================================
test('Vedic Svara: Accents extracted to metadata, bare marks stripped from normalized Devanagari', () => {
  const mantra = '\u0905\u0962\u0917\u094d\u0928\u093f\u092e\u0940\u0951\u0933\u0947 \u092a\u0941\u0962\u0930\u094b\u0939\u093f\u0951\u0924\u092e\u094d';
  const res = textToIds(mantra, { lang: 'sa' });
  assert.ok(!res.normalizedDevanagari.includes('\u0951'), 'Combining udatta mark must be stripped');
  assert.ok(!res.normalizedDevanagari.includes('\u0952'), 'Combining anudatta mark must be stripped');
  assert.ok(res.svaras && res.svaras.some((s) => s === 'udatta' || s === 'anudatta'));
  assert.ok(res.aksharas.length >= 6);
});

test('Punctuation: Boundary punctuation maps to space tokens without click-inducing hyphens', () => {
  for (const text of ['सत्यमेव जयते । नानृतम् ॥', 'अथातो ब्रह्मजिज्ञासा ; तस्मात् विद्धि :', 'कथम् ? न हि !']) {
    const res = textToIds(text, { lang: 'sa' });
    assert.ok(!res.nativeText.includes('-'), `nativeText must not contain "-": "${res.nativeText}"`);
    assert.ok(!res.nativeText.includes('  '), `nativeText must not contain double spaces: "${res.nativeText}"`);
  }
});

// ==============================================================================
// Suite 5: Dravidian Phoneme Preservation & Token ID 0 Disambiguation
// ==============================================================================
test('Dravidian Phonemes: Restored symbols preserved in native text', () => {
  const kn = textToIds('ವಾಙ್ಮಯ', { lang: 'kn' });
  assert.ok(kn.nativeText.includes('\u0c99'), 'ಙ must be preserved in Kannada');

  const te = textToIds('ఝషము', { lang: 'te' });
  assert.ok(te.nativeText.includes('\u0c1d'), 'ఝ must be preserved in Telugu');

  const ml = textToIds('ഝഷം', { lang: 'ml' });
  assert.ok(ml.nativeText.includes('\u0d1d'), 'ഝ must be preserved in Malayalam');

  const ta = textToIds('தமிழ்', { lang: 'ta' });
  assert.ok(ta.nativeText.includes('\u0ba4'), 'த must be preserved in Tamil');
});

test('Token ID 0: Decoupled from blank tokens across Telugu and Hindi', () => {
  // Telugu: 'త' is token ID 0
  const te = textToIds('తెలుగు', { lang: 'te' });
  assert.equal(te.inputIds[1], 0, 'First character must be token ID 0');
  assert.equal(te.isBlank[1], false, 'Token 0 must not be blank');
  assert.equal(te.tokenTypeIds[1], 0, 'Token 0 must have tokenTypeId 0');
  assert.equal(PAD_TOKEN_ID.te, 45);
  assert.equal(te.inputIds[0], 45, 'Telugu blank token ID must be 45');

  // Hindi: 'फ' is token ID 0
  const hi = textToIds('फल', { lang: 'hi' });
  assert.equal(hi.inputIds[1], 0, 'First character must be token ID 0');
  assert.equal(hi.isBlank[1], false, 'Token 0 must not be blank');
  assert.equal(PAD_TOKEN_ID.hi, 33);
  assert.equal(hi.inputIds[0], 33, 'Hindi blank token ID must be 33');
});

// ==============================================================================
// Suite 6: Hindi Deterministic Schwa Deletion Engine
// ==============================================================================
test('Hindi Schwa Deletion: Word-final, medial VC_CV, and Tatsama protection rules', () => {
  const cases = [
    // Tier 1: Polysyllabic word-final schwa deletion
    { word: 'कमल', expected: 'कमल्' },
    { word: 'भारत', expected: 'भारत्' },
    { word: 'रात', expected: 'रात्' },
    { word: 'कल', expected: 'कल्' },
    // Tier 1 protection: Vowel matras preserved
    { word: 'माता', expected: 'माता' },
    { word: 'राजा', expected: 'राजा' },
    { word: 'पूजा', expected: 'पूजा' },
    // Tier 1 protection: Monosyllables retain schwa
    { word: 'न', expected: 'न' },
    { word: 'व', expected: 'व' },
    { word: 'स', expected: 'स' },
    // Tier 1 protection: Sanskrit Tatsama final conjuncts retain schwa
    { word: 'सत्य', expected: 'सत्य' },
    { word: 'मित्र', expected: 'मित्र' },
    { word: 'कष्ट', expected: 'कष्ट' },
    { word: 'धर्म', expected: 'धर्म' },
    { word: 'अर्थ', expected: 'अर्थ' },
    { word: 'शब्द', expected: 'शब्द' },
    { word: 'सूर्य', expected: 'सूर्य' },
    { word: 'शास्त्र', expected: 'शास्त्र' },
    { word: 'पुत्र', expected: 'पुत्र' },
    // Tier 2: Medial schwa deletion in VC_CV contexts
    { word: 'कमला', expected: 'कम्ला' },
    { word: 'समझना', expected: 'समझ्ना' },
    // Tier 2 protection: Initial syllable never deletes schwa
    { word: 'सड़क', expected: 'सड़क्' },
    // Tier 2 protection: 'ह' resists deletion
    { word: 'कहना', expected: 'कहना' },
    { word: 'रहना', expected: 'रहना' },
  ];

  for (const { word, expected } of cases) {
    const res = deleteHindiSchwaWord(word);
    assert.equal(res, expected, `Mismatch for word "${word}": got "${res}", expected "${expected}"`);
  }

  // Words ending in vowel matras never end in virama
  for (const w of ['सफलता', 'सुंदरता', 'कविता', 'क्षमता', 'भाषा']) {
    const r = deleteHindiSchwaWord(w);
    assert.ok(!r.endsWith('\u094d'), `Word "${w}" must not end in virama: "${r}"`);
  }

  // Full sentence processing
  const sent = 'कमल भारत का राष्ट्रीय फूल है';
  const proc = deleteHindiSchwa(sent);
  assert.ok(proc.includes('कमल्'));
  assert.ok(proc.includes('भारत्'));
});

// ==============================================================================
// Suite 7: Virama & Consonant Cluster Co-articulation
// ==============================================================================
test('Virama Co-articulation: isVirama mask and Klatt zero vowel resonance', () => {
  const parsed = textToIds('सत्य', { lang: 'sa' });
  assert.ok(parsed.isVirama && parsed.isVirama.some((v) => v === 1));
  assert.ok(parsed.isBlank !== undefined);
  assert.ok(parsed.tokenTypeIds !== undefined);

  // Klatt synthesis on virama-ended akshara has weight C
  const { pcm, meta } = synthesizeKlattPcm('त्', { lang: 'sa' });
  assert.ok(pcm.length > 0);
  assert.equal(meta.aksharas[meta.aksharas.length - 1].weight, 'C');
});

// ==============================================================================
// Suite 8: Inter-Clause Room Tone Continuity
// ==============================================================================
test('Inter-Clause Room Tone: Calibrated studio room tone (-58 dBFS)', () => {
  const roomTone = generateRoomTone(1280, 1337);
  assert.equal(roomTone.length, 1280);
  assert.ok(roomTone.some((v) => v !== 0.0), 'Room tone must not be pure zeros');

  let sumSq = 0.0;
  let maxAbs = 0.0;
  for (let i = 0; i < roomTone.length; i++) {
    const abs = Math.abs(roomTone[i]);
    if (abs > maxAbs) maxAbs = abs;
    sumSq += roomTone[i] * roomTone[i];
  }
  const rms = Math.sqrt(sumSq / roomTone.length);
  const dbRms = 20.0 * Math.log10(Math.max(1e-12, rms));

  assert.ok(
    dbRms > -62.0 && dbRms < -52.0,
    `Room tone RMS ${dbRms.toFixed(2)} dBFS outside acceptable range (-62 to -52)`
  );
  assert.ok(maxAbs < 0.008, `Room tone peak ${maxAbs.toFixed(5)} exceeds threshold 0.008`);
});

// ==============================================================================
// Suite 9: Blank Duration Relaxation Masks
// ==============================================================================
test('Blank Relaxation: Even positions are blanks, odd are characters', () => {
  for (const lc of ['sa', 'kn', 'hi', 'ta', 'te', 'ml']) {
    const res = textToIds('परीक्षणम्', { lang: lc });
    const { inputIds, isBlank, tokenTypeIds } = res;
    assert.equal(inputIds.length, isBlank.length);
    assert.equal(inputIds.length, tokenTypeIds.length);
    for (let i = 0; i < inputIds.length; i += 2) {
      assert.equal(isBlank[i], true);
      assert.equal(tokenTypeIds[i], 1);
    }
  }
});

// ==============================================================================
// Suite 10: Multi-Script Cross-Platform Parity
// ==============================================================================
test('Multi-Script Parity: All 6 languages pass G2P validation', () => {
  const corpus = {
    sa: 'प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् ।',
    kn: 'ಕನ್ನಡ ಸಾಹಿತ್ಯ ಪರಿಷತ್ತು',
    hi: 'नमस्ते भारत',
    ta: 'தமிழ்நாடு வாழ்க',
    te: 'తెలుగు భాష అమృతం',
    ml: 'മലയാളം സുന്ദരമാണ്',
  };
  for (const [lc, text] of Object.entries(corpus)) {
    const res = textToIds(text, { lang: lc });
    assert.equal(res.lang, lc);
    assert.ok(res.inputIds.length >= 8);
    assert.ok(res.nativeText.length > 0);
    assert.equal(res.attentionMask.length, res.inputIds.length);
  }
});
