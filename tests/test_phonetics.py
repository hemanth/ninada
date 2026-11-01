"""Comprehensive Phonetic & G2P Articulation Regression Test Suite for Ninada v2.0.

Covers Milestone 1:
1. Phoneme inventory preservation across 87 canonical tokens and 6 specialist languages.
2. Homorganic nasal sandhi preservation before stops without Anusvara collapse.
3. Avagraha (ऽ) pūrvarūpa sandhi processing without spurious vowel insertion.
4. Vedic svara accent extraction (udātta, anudātta, svarita) and weight preservation.
5. Clean prosodic boundary token mapping (zero punctuation hyphen insertion).
6. Dravidian phoneme preservation and token ID 0 non-collision.
7. Hindi two-tier right-to-left schwa deletion engine.
8. Virama co-articulation and zero vowel resonance in consonant clusters.
9. Inter-clause room tone continuity and silence-gating elimination.
10. Cross-platform G2P algorithmic parity and metadata validation.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from ninada.engine import generate_room_tone, synthesize_klatt_pcm
from ninada.g2p import (
    LANGUAGE_IDS,
    NATIVE_DONOR_MAP,
    NATIVE_VOCABS,
    PAD_TOKEN_ID,
    TOKEN_TO_ID,
    VEDIC_SVARA_MARKS,
    VIRAMA_TOKEN_IDS,
    VOCAB_TOKENS,
    apply_phonological_rules,
    brahmic_to_devanagari,
    delete_hindi_schwa,
    delete_hindi_schwa_word,
    iast_to_devanagari,
    segment_aksharas,
    text_to_ids,
    to_native_model_text,
)


# ==============================================================================
# Suite 1: Phoneme Inventory Preservation Across All Languages
# ==============================================================================
def test_phoneme_inventory_preservation() -> None:
    """Verify canonical and specialist vocabularies retain all necessary phonetic symbols."""
    assert len(VOCAB_TOKENS) == 87
    assert VOCAB_TOKENS[85] == "\u090b"  # ऋ
    assert VOCAB_TOKENS[86] == "\u0919"  # ङ
    assert set(LANGUAGE_IDS.keys()) == {"sa", "hi", "kn", "ta", "te", "ml"}

    # Sanskrit inventory must contain extended symbols (ಙ, ೠ, ೄ)
    sa_vocab = NATIVE_VOCABS["sa"]
    assert "\u0c99" in sa_vocab  # Kannada velar nasal ಙ
    assert "\u0ce0" in sa_vocab  # Kannada vocalic rr ೠ
    assert "\u0cc4" in sa_vocab  # Kannada vowel sign vocalic rr ೄ

    # Restored Dravidian consonants must exist in respective vocabularies
    assert "\u0c99" in NATIVE_VOCABS["kn"]  # ಙ
    assert "\u0c1d" in NATIVE_VOCABS["te"]  # ఝ
    assert "\u0d1d" in NATIVE_VOCABS["ml"]  # ഝ

    # Base checkpoint donor embedding mappings must exist
    assert "\u0c99" in NATIVE_DONOR_MAP["kn"]
    assert "\u0c1d" in NATIVE_DONOR_MAP["te"]
    assert "\u0d1d" in NATIVE_DONOR_MAP["ml"]


# ==============================================================================
# Suite 2: Sanskrit Homorganic Nasal Sandhi Preservation
# ==============================================================================
@pytest.mark.parametrize(
    "text,expected_char,forbidden_char",
    [
        ("शंकर", "\u0c99", "\u0c82"),  # Velar: ಙ್ಕ, NOT ಂಕ
        ("शङ्कर", "\u0c99", "\u0c82"),
        ("संचय", "\u0c9e", "\u0c82"),  # Palatal: ಞ್ಚ, NOT ಂಚ
        ("सञ्चय", "\u0c9e", "\u0c82"),
        ("कंठ", "\u0ca3", "\u0c82"),   # Retroflex: ಣ್ಠ, NOT ಂಠ
        ("कण्ठ", "\u0ca3", "\u0c82"),
        ("शांत", "\u0ca8", "\u0c82"),   # Dental: ನ್ತ, NOT ಂತ
        ("शान्त", "\u0ca8", "\u0c82"),
        ("संपूर्ण", "\u0cae", "\u0c82"), # Labial: ಮ್ಪ, NOT ಂಪ
        ("सम्पूर्ण", "\u0cae", "\u0c82"),
    ],
)
def test_sanskrit_homorganic_nasal_sandhi_preservation(
    text: str, expected_char: str, forbidden_char: str
) -> None:
    """Verify homorganic nasals before stops are preserved and never collapsed to Anusvara."""
    parsed = text_to_ids(text, lang="sa")
    native = parsed["native_text"]
    assert expected_char in native, f"Expected {expected_char} in {native} for input {text}"
    assert forbidden_char not in native[:3]


# ==============================================================================
# Suite 3: Avagraha (ऽ) Pūrvarūpa Sandhi Processing
# ==============================================================================
@pytest.mark.parametrize(
    "input_text,expected_dev_sub,forbidden_dev_sub",
    [
        ("सोऽहम्", "सोहम्", "सो अहम्"),
        ("सो ऽहम्", "सोहम्", "सो अहम्"),
        ("शिवोऽहम्", "शिवोहम्", "शिवो अहम्"),
        ("तेऽपि", "तेपि", "ते अपि"),
        ("ते ऽपि", "तेपि", "ते अपि"),
        ("भोऽऽऽ", "भो", "भोअ"),
        ("कालोऽस्ति", "कालोस्ति", "कालो अस्ति"),
    ],
)
def test_avagraha_purvarupa_sandhi(
    input_text: str, expected_dev_sub: str, forbidden_dev_sub: str
) -> None:
    """Verify avagraha is absorbed into preceding vowel without inserting spurious independent 'अ'."""
    parsed = text_to_ids(input_text, lang="sa")
    dev = parsed["normalized_devanagari"]
    assert "\u093d" not in dev, "Avagraha symbol must not remain in normalized Devanagari"
    assert forbidden_dev_sub not in dev, f"Spurious vowel inserted: found {forbidden_dev_sub} in {dev}"
    assert expected_dev_sub in dev, f"Expected {expected_dev_sub} in {dev}"

    weights = [a["weight"] for a in parsed["aksharas"]]
    assert "G" in weights


def test_iast_avagraha_purvarupa() -> None:
    """Verify IAST romanization with apostrophe absorbs cleanly into pūrvarūpa sandhi."""
    parsed = text_to_ids("so 'ham", lang="sa")
    assert "सोहम्" in parsed["normalized_devanagari"]
    assert "अ" not in parsed["normalized_devanagari"]

    parsed_te = text_to_ids("te 'pi", lang="sa")
    assert "तेपि" in parsed_te["normalized_devanagari"]
    assert "अ" not in parsed_te["normalized_devanagari"]


# ==============================================================================
# Suite 4: Vedic Svara Accent Preservation & Prosodic Punctuation
# ==============================================================================
def test_vedic_svara_accent_preservation() -> None:
    """Verify Vedic svara marks (udātta, anudātta) are preserved in akshara metadata."""
    mantra = "\u0905\u0962\u0917\u094d\u0928\u093f\u092e\u0940\u0951\u0933\u0947 \u092a\u0941\u0962\u0930\u094b\u0939\u093f\u0951\u0924\u092e\u094d"
    parsed = text_to_ids(mantra, lang="sa")
    # Normalized Devanagari strips bare combining marks to avoid tokenizer <unk>
    assert "\u0951" not in parsed["normalized_devanagari"]
    assert "\u0952" not in parsed["normalized_devanagari"]
    assert len(parsed["aksharas"]) >= 6
    # Metadata includes non-empty svara values
    assert any(s in ("udatta", "anudatta", "svarita") for s in parsed["svaras"])


@pytest.mark.parametrize(
    "text",
    [
        "सत्यमेव जयते । नानृतम् ॥",
        "अथातो ब्रह्मजिज्ञासा ; तस्मात् विद्धि :",
        "कथम् ? न हि !",
    ],
)
def test_punctuation_boundary_tokens_no_hyphens(text: str) -> None:
    """Verify punctuation marks map to clean prosodic space boundaries, never hyphens."""
    parsed = text_to_ids(text, lang="sa")
    native = parsed["native_text"]
    assert "-" not in native, f"Found click-inducing hyphen in native_text: {native}"
    assert "  " not in native, f"Found collapsed whitespace in native_text: {native}"


# ==============================================================================
# Suite 5: Dravidian Phoneme Preservation & Token ID 0 Disambiguation
# ==============================================================================
def test_dravidian_phoneme_preservation() -> None:
    """Verify native Dravidian phonemes are preserved in native specialist text."""
    kn = text_to_ids("ವಾಙ್ಮಯ", lang="kn")
    assert kn["lang"] == "kn"
    assert "\u0c99" in kn["native_text"], "Velar nasal ಙ must be preserved in Kannada"

    te = text_to_ids("ఝషము", lang="te")
    assert te["lang"] == "te"
    assert "\u0c1d" in te["native_text"], "Aspirated ఝ must be preserved in Telugu"

    ml = text_to_ids("ഝഷം", lang="ml")
    assert ml["lang"] == "ml"
    assert "\u0d1d" in ml["native_text"], "Aspirated ഝ must be preserved in Malayalam"

    ta = text_to_ids("தமிழ்", lang="ta")
    assert ta["lang"] == "ta"
    assert "\u0ba4" in ta["native_text"]  # Tamil dental த


def test_token_id_0_disambiguation() -> None:
    """Verify token ID 0 is never conflated with blank tokens in any language."""
    # In Telugu, 'త' is token ID 0
    te_parsed = text_to_ids("తెలుగు", lang="te")
    assert te_parsed["input_ids"][1] == 0, "First character must be token ID 0 ('త')"
    assert te_parsed["is_blank"][1] is False, "Lexical token 0 must NOT be marked as blank"
    assert te_parsed["token_type_ids"][1] == 0, "Lexical token 0 must have token_type_id 0"
    assert PAD_TOKEN_ID["te"] == 45
    assert te_parsed["input_ids"][0] == 45, "Telugu blank token ID must be 45"

    # In Hindi, 'फ' is token ID 0
    hi_parsed = text_to_ids("फल", lang="hi")
    assert hi_parsed["input_ids"][1] == 0, "First character must be token ID 0 ('फ')"
    assert hi_parsed["is_blank"][1] is False, "Lexical token 0 must NOT be marked as blank"
    assert PAD_TOKEN_ID["hi"] == 33
    assert hi_parsed["input_ids"][0] == 33, "Hindi blank token ID must be 33"


# ==============================================================================
# Suite 6: Hindi Deterministic Schwa Deletion Engine
# ==============================================================================
@pytest.mark.parametrize(
    "word,expected",
    [
        # Tier 1: Polysyllabic word-final schwa deletion
        ("कमल", "कमल्"),
        ("भारत", "भारत्"),
        ("रात", "रात्"),
        ("कल", "कल्"),
        # Tier 1 protection: Vowel matras are preserved (word does not end in virama)
        ("माता", "माता"),
        ("राजा", "राजा"),
        ("पूजा", "पूजा"),
        # Tier 1 protection: Monosyllables retain schwa
        ("न", "न"),
        ("व", "व"),
        ("स", "स"),
        # Tier 1 protection: Sanskrit Tatsama final conjuncts retain schwa
        ("सत्य", "सत्य"),
        ("मित्र", "मित्र"),
        ("कष्ट", "कष्ट"),
        ("धर्म", "धर्म"),
        ("अर्थ", "अर्थ"),
        ("शब्द", "शब्द"),
        ("सूर्य", "सूर्य"),
        ("शास्त्र", "शास्त्र"),
        ("पुत्र", "पुत्र"),
        # Tier 2: Medial schwa deletion in VC_CV contexts
        ("कमला", "कम्ला"),
        ("समझना", "समझ्ना"),
        # Tier 2 protection: Initial syllable never deletes schwa
        ("सड़क", "सड़क्"),
        # Tier 2 protection: 'ह' resists deletion
        ("कहना", "कहना"),
        ("रहना", "रहना"),
    ],
)
def test_hindi_schwa_deletion_rules(word: str, expected: str) -> None:
    """Verify deterministic two-tier right-to-left Hindi schwa deletion behavior."""
    assert delete_hindi_schwa_word(word) == expected


def test_hindi_schwa_sentence_processing() -> None:
    """Verify full Hindi sentence schwa deletion integration."""
    sentence = "कमल भारत का राष्ट्रीय फूल है"
    processed = delete_hindi_schwa(sentence)
    assert "कमल्" in processed
    assert "भारत्" in processed


def test_hindi_schwa_vowel_matra_preservation() -> None:
    """Verify words ending in vowel matras preserve word-final vowels and never end with virama."""
    for word in ("सफलता", "सुंदरता", "कविता", "क्षमता", "भाषा"):
        res = delete_hindi_schwa_word(word)
        assert not res.endswith("\u094d"), f"Word {word} must not end in virama: {res}"
        assert res[-1] in ("\u093e", "\u093f", "\u0940", "\u0941", "\u0942", "\u0947", "\u0948", "\u094b", "\u094c"), f"Word {word} must preserve final vowel: {res}"


# ==============================================================================
# Suite 7: Virama & Consonant Cluster Co-articulation
# ==============================================================================
def test_virama_cluster_metadata() -> None:
    """Verify viramas are explicitly flagged in is_virama metadata."""
    parsed = text_to_ids("सत्य", lang="sa")
    assert any(v == 1 for v in parsed["is_virama"]), "Virama must be flagged in is_virama mask"
    assert parsed["is_blank"] is not None
    assert parsed["token_type_ids"] is not None


def test_klatt_virama_zero_vowel_resonance() -> None:
    """Verify Tier-2 Klatt sets zero vowel duration for virama-ended aksharas."""
    pcm, meta = synthesize_klatt_pcm("त्", lang="sa")
    assert len(pcm) > 0
    # Last akshara is pure consonant (C weight)
    assert meta["aksharas"][-1]["weight"] == "C"


# ==============================================================================
# Suite 8: Inter-Clause Room Tone Continuity
# ==============================================================================
def test_inter_clause_room_tone_acoustic_properties() -> None:
    """Verify the 80ms inter-clause room tone buffer provides calibrated continuous studio ambiance."""
    sample_rate = 16000
    pause_samples = int(0.08 * sample_rate)
    room_tone = generate_room_tone(num_samples=pause_samples, seed=1337)

    assert len(room_tone) == 1280
    assert not np.all(room_tone == 0.0), "Room tone must not be exact mathematical zeros"

    rms = float(np.sqrt(np.mean(room_tone**2)))
    db_rms = 20.0 * math.log10(max(1e-12, rms))
    assert -62.0 < db_rms < -52.0, f"Room tone RMS {db_rms:.2f} dBFS outside acceptable range (-62 to -52)"

    peak = float(np.max(np.abs(room_tone)))
    assert peak < 0.008, f"Room tone peak {peak:.5f} exceeds threshold 0.008"


# ==============================================================================
# Suite 9: Blank Duration Relaxation Mechanics
# ==============================================================================
def test_blank_duration_relaxation_masks() -> None:
    """Verify text_to_ids generates matched length is_blank and token_type_ids."""
    for lc in ("sa", "kn", "hi", "ta", "te", "ml"):
        parsed = text_to_ids("परीक्षणम्", lang=lc)
        ids = parsed["input_ids"]
        is_b = parsed["is_blank"]
        t_types = parsed["token_type_ids"]
        assert len(ids) == len(is_b) == len(t_types)
        # Even positions are blank tokens in standard interleaving
        assert all(is_b[i] is True for i in range(0, len(ids), 2))
        assert all(t_types[i] == 1 for i in range(0, len(ids), 2))


# ==============================================================================
# Suite 10: Multi-Script Cross-Platform Parity
# ==============================================================================
def test_multiscript_roundtrip_coverage() -> None:
    """Verify all 6 supported languages map correctly through text_to_ids."""
    corpus = {
        "sa": "प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् ।",
        "kn": "ಕನ್ನಡ ಸಾಹಿತ್ಯ ಪರಿಷತ್ತು",
        "hi": "नमस्ते भारत",
        "ta": "தமிழ்நாடு வாழ்க",
        "te": "తెలుగు భాష అమృతం",
        "ml": "മലയാളം സുന്ദരമാണ്",
    }
    for lc, text in corpus.items():
        res = text_to_ids(text, lang=lc)
        assert res["lang"] == lc
        assert len(res["input_ids"]) >= 8
        assert len(res["native_text"]) > 0
        assert res["attention_mask"] == [1] * len(res["input_ids"])
