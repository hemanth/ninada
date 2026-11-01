"""Tier 2: Boundary Value Analysis & Corner Cases End-to-End Tests.

Covers empty strings, single characters/aksharas, extreme length compounds,
punctuation clusters, Vedic svara markers, avagraha variations, multi-script switches,
and restricted terminology verification across documentation.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ninada import Ninada, encode_wav_bytes, split_prosodic_clauses
from tests.e2e.helpers import verify_pcm_samples, verify_wav_header

REPO_ROOT = Path(__file__).resolve().parents[2]


# ==============================================================================
# 1. Empty Strings and Whitespace Inputs
# ==============================================================================
@pytest.mark.parametrize(
    "blank_text",
    [
        "",
        "   ",
        "\t\t",
        "\n\n\n",
        "  \t \n \r  ",
    ],
)
def test_empty_and_whitespace_inputs(engine_sa: Ninada, blank_text: str) -> None:
    """Verify empty and whitespace-only inputs do not raise unhandled exceptions."""
    pcm, meta = engine_sa.synthesize(blank_text, lang="sa", tier="klatt")
    assert isinstance(pcm, list)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] >= 0.0


# ==============================================================================
# 2. Single Characters and Minimal Aksharas
# ==============================================================================
@pytest.mark.parametrize(
    "single_char,lang",
    [
        ("ॐ", "sa"),
        ("अ", "sa"),
        ("क", "sa"),
        ("ं", "sa"),
        ("ಃ", "kn"),
        ("த", "ta"),
        ("తె", "te"),
        ("മ", "ml"),
    ],
)
def test_single_characters_and_aksharas(
    single_char: str, lang: str
) -> None:
    """Verify single-character and isolated grapheme inputs synthesize valid audio."""
    engine = Ninada(lang=lang)
    pcm, meta = engine.synthesize(single_char, lang=lang, tier="klatt")
    verify_pcm_samples(pcm, min_length=100)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.01


# ==============================================================================
# 3. Extreme Long Compounds and Long Passages
# ==============================================================================
def test_extreme_length_continuous_compound(engine_sa: Ninada) -> None:
    """Verify 100+ character continuous Sanskrit compound synthesizes without buffer issues."""
    compound = (
        "प्रज्ञापूर्वकव्यापारवत्त्वं"
        "यथार्थज्ञानप्रमाणत्वं"
        "निरतिशयानन्दस्वरूपत्वं"
        "सच्चिदानन्दविग्रहात्मकत्वं"
        "समस्तसद्गुणगणालङ्कृतत्वम्"
    )
    assert len(compound) > 100
    pcm, meta = engine_sa.synthesize(compound, lang="sa", tier="klatt")
    verify_pcm_samples(pcm, min_length=20000)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 1.2


def test_long_multiclause_splitting_and_streaming(engine_sa: Ninada) -> None:
    """Verify arbitrarily long multi-clause verse partitions and streams sequentially."""
    long_passage = (
        "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । "
        "मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥ "
        "दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा । "
        "आचार्यमुपसङ्गम्य राजा वचनमब्रवीत् ॥ "
        "पश्यैतां पाण्डुपुत्राणामाचार्य महतीं चमूम् । "
        "व्यूढां द्रुपदपुत्रेण तव शिष्येण धीमता ॥"
    )
    clauses = split_prosodic_clauses(long_passage)
    assert len(clauses) >= 6

    chunks = list(engine_sa.stream(long_passage, lang="sa", tier="klatt"))
    assert len(chunks) == len(clauses)
    assert chunks[0]["index"] == 0
    assert chunks[-1]["is_final"] is True
    for idx, chunk in enumerate(chunks):
        assert chunk["clause_index"] == idx
        verify_pcm_samples(chunk["pcm"], min_length=1000)


# ==============================================================================
# 4. Punctuation Clusters and Irregular Delimiters
# ==============================================================================
@pytest.mark.parametrize(
    "punct_input",
    [
        "॥ ॥ ॥",
        "? ! . , ; -",
        "सत्यमेव जयते...???!!!",
        "«धर्मक्षेत्रे» - (कुरुक्षेत्रे)!",
        "॥ ॐ नमः शिवाय ॥",
    ],
)
def test_punctuation_clusters_handling(engine_sa: Ninada, punct_input: str) -> None:
    """Verify punctuation clusters and delimiter strings synthesize gracefully."""
    pcm, meta = engine_sa.synthesize(punct_input, lang="sa", tier="klatt")
    assert isinstance(pcm, list)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] >= 0.05


# ==============================================================================
# 5. Vedic Svara Accent Markers
# ==============================================================================
@pytest.mark.parametrize(
    "text_with_svara",
    [
        "अ॒ग्निमी॑ळे पु॒रोहि॑तम्",
        "य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म्",
        "होता॑रं रत्न॒धात॑मम्",
        "इ॒षे त्वो॒र्जे त्वा॑",
        "स॒हस्र॑शीर्षा॒ पुरु॑षः",
    ],
)
def test_vedic_svara_accent_markers(engine_sa: Ninada, text_with_svara: str) -> None:
    """Verify Vedic accents (udatta \u0951, anudatta \u0952) synthesize smoothly."""
    pcm, meta = engine_sa.synthesize(text_with_svara, lang="sa", tier="klatt")
    verify_pcm_samples(pcm, min_length=2000)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.15


# ==============================================================================
# 6. Avagraha Variations and Pluta Prolongation
# ==============================================================================
@pytest.mark.parametrize(
    "avagraha_text",
    [
        "सोऽहम्",
        "शिवोऽहम्",
        "कोऽऽपि तत्र नास्ति",
        "ऽहम्",
        "तेऽपि गच्छन्ति",
    ],
)
def test_avagraha_variations_synthesis(
    engine_sa: Ninada, avagraha_text: str
) -> None:
    """Verify single, double, initial, and medial avagrahas synthesize properly."""
    pcm, meta = engine_sa.synthesize(avagraha_text, lang="sa", tier="klatt")
    verify_pcm_samples(pcm, min_length=1500)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.10


# ==============================================================================
# 7. Multi-Script Inputs and IAST Transliteration
# ==============================================================================
@pytest.mark.parametrize(
    "mixed_text,lang",
    [
        ("dharmakṣetre kurukṣetre samavetā yuyutsavaḥ", "sa"),
        ("prajñāpūrvakavyāpāravattvaṃ jīvatvam", "sa"),
        ("ಸಂಸ್ಕೃತಮ್ ಮತ್ತು ಕನ್ನಡ", "kn"),
        ("ಕನ್ನಡ 123 ನುಡಿ", "kn"),
        ("தமிழ் வாழ்க 2026", "ta"),
    ],
)
def test_multiscript_and_iast_inputs(
    mixed_text: str, lang: str
) -> None:
    """Verify IAST romanization and multi-script Brahmic text synthesize cleanly."""
    engine = Ninada(lang=lang)
    pcm, meta = engine.synthesize(mixed_text, lang=lang, tier="klatt")
    verify_pcm_samples(pcm, min_length=1500)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.10


# ==============================================================================
# 8. Restricted Terminology Cleansing in Docs and README
# ==============================================================================
def test_restricted_terms_absence_in_documentation() -> None:
    """Verify docs/ and README.md contain strictly zero restricted term occurrences."""
    prohibited_terms = [
        re.compile(r"su-srota", re.IGNORECASE),
        re.compile(r"l4\s+gpus", re.IGNORECASE),
    ]

    target_paths: list[Path] = [REPO_ROOT / "README.md"]
    docs_dir = REPO_ROOT / "docs"
    if docs_dir.exists():
        for p in docs_dir.rglob("*"):
            if p.is_file() and p.suffix in (".md", ".html", ".js", ".mjs", ".json", ".txt"):
                target_paths.append(p)

    violations: list[str] = []
    for fpath in target_paths:
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for term_pattern in prohibited_terms:
            matches = term_pattern.findall(content)
            if matches:
                violations.append(f"{fpath}: matched {term_pattern.pattern} ({len(matches)} times)")

    assert not violations, f"Prohibited terminology found:\n" + "\n".join(violations)
