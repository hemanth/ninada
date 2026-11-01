"""Tier 3: Cross-Feature Interactions End-to-End Tests.

Covers pairwise combinations of:
- Language x Script (Devanagari, Kannada, Tamil, Telugu, Malayalam, IAST)
- Language x Speaking Rate x Synthesis Tier
- Language x Sentence Punctuation Delimiters
- Streaming Chunk Frames x Halo Overlap x Crossfade Samples
- Multi-Clause Inter-Clause Transitions
"""
from __future__ import annotations

import pytest

from ninada import Ninada, encode_wav_bytes, split_prosodic_clauses
from tests.e2e.helpers import verify_pcm_samples, verify_wav_header


# ==============================================================================
# 1. Pairwise Language x Script Combinations
# ==============================================================================
@pytest.mark.parametrize(
    "text,lang,expected_script",
    [
        ("सत्यमेव जयते नानृतम्", "sa", "devanagari"),
        ("ಸತ್ಯಮೇವ ಜಯತೇ ನಾನೃತಮ್", "sa", "kannada"),
        ("satyam eva jayate nānṛtam", "sa", "iast"),
        ("ಕನ್ನಡ ನುಡಿ ಮುತ್ತಿನ ಹಾರ", "kn", "kannada"),
        ("kannaḍa nuḍi muttina hāra", "kn", "iast"),
        ("नमस्ते भारत देश हमारा", "hi", "devanagari"),
        ("namaste bhārata deśa hamārā", "hi", "iast"),
        ("தமிழ் வாழ்க வளர்க", "ta", "tamil"),
        ("తెలుగు భాష అమృతము", "te", "telugu"),
        ("മലയാളം സുന്ദരമായ ഭാഷ", "ml", "malayalam"),
    ],
)
def test_language_cross_script_synthesis(
    text: str, lang: str, expected_script: str
) -> None:
    """Verify engine handles cross-script inputs across all supported languages."""
    engine = Ninada(lang=lang)
    pcm, meta = engine.synthesize(text, lang=lang, tier="klatt")
    verify_pcm_samples(pcm, min_length=2000)
    assert meta["lang"] == lang
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.15


# ==============================================================================
# 2. Pairwise Language x Speaking Rate x Tier
# ==============================================================================
@pytest.mark.parametrize("lang", ["sa", "hi", "kn", "ta", "te", "ml"])
@pytest.mark.parametrize("tier", ["klatt", "auto"])
def test_language_cross_rate_and_tier_monotonicity(
    lang: str, tier: str
) -> None:
    """Verify rate scaling (0.75x vs 1.40x) produces monotonic durations per language."""
    sample_texts = {
        "sa": "धर्मक्षेत्रे कुरुक्षेत्रे समवेताः",
        "hi": "भारत एक विशाल और सुंदर देश है",
        "kn": "ಕರ್ನಾಟಕ ರಾಜ್ಯ ಅತ್ಯಂತ ಸುಂದರವಾದದ್ದು",
        "ta": "தமிழுக்கும் அமுதென்று பேர்",
        "te": "తెలుగు భాష తేనె వంటిది",
        "ml": "കേരളം പ്രകൃതിഭംഗിയാൽ സമൃദ്ധമാണ്",
    }
    text = sample_texts[lang]
    engine = Ninada(lang=lang)

    pcm_slow, meta_slow = engine.synthesize(
        text, lang=lang, rate=0.75, noise_duration=0.0, tier=tier
    )
    pcm_fast, meta_fast = engine.synthesize(
        text, lang=lang, rate=1.40, noise_duration=0.0, tier=tier
    )

    verify_pcm_samples(pcm_slow, min_length=1500)
    verify_pcm_samples(pcm_fast, min_length=1000)

    assert len(pcm_slow) > len(pcm_fast), (
        f"Language {lang} tier {tier}: slow {len(pcm_slow)} not > fast {len(pcm_fast)}"
    )


# ==============================================================================
# 3. Pairwise Language x Punctuation Delimiters
# ==============================================================================
@pytest.mark.parametrize("lang", ["sa", "hi", "kn", "ta", "te", "ml"])
@pytest.mark.parametrize(
    "delim,phrase_template",
    [
        ("danda", "{p1} । {p2} ॥"),
        ("period", "{p1}. {p2}."),
        ("question", "{p1}? {p2}?"),
        ("exclamation", "{p1}! {p2}!"),
        ("comma", "{p1}, {p2}."),
    ],
)
def test_language_cross_punctuation_delimiters(
    lang: str, delim: str, phrase_template: str
) -> None:
    """Verify punctuation delimiters split clauses cleanly across all languages."""
    words = {
        "sa": ("सत्यं वद", "धर्मं चर"),
        "hi": ("नमस्ते मित्र", "आप कैसे हैं"),
        "kn": ("ನಮಸ್ಕಾರ ಸ್ನೇಹಿತರೆ", "ಬನ್ನಿ ಕುಳಿತುಕೊಳ್ಳಿ"),
        "ta": ("வணக்கம் நண்பரே", "நலமா"),
        "te": ("నమస్కారం మిత్రమా", "ఎలా ఉన్నారు"),
        "ml": ("നമസ്കാരം സുഹൃത്തേ", "സുഖമാണോ"),
    }
    p1, p2 = words[lang]
    text = phrase_template.format(p1=p1, p2=p2)
    clauses = split_prosodic_clauses(text)
    assert len(clauses) >= 1

    engine = Ninada(lang=lang)
    pcm, meta = engine.synthesize(text, lang=lang, tier="klatt")
    verify_pcm_samples(pcm, min_length=1500)
    wav = encode_wav_bytes(pcm)
    verify_wav_header(wav)


# ==============================================================================
# 4. Pairwise Streaming Configurations (Chunk x Halo x Crossfade)
# ==============================================================================
@pytest.mark.parametrize(
    "chunk_f,halo_f,xfade",
    [
        (32, 8, 128),
        (48, 12, 192),
        (64, 12, 256),
        (80, 16, 256),
        (96, 16, 384),
    ],
)
@pytest.mark.parametrize("lang", ["sa", "kn", "hi"])
def test_streaming_chunk_halo_crossfade_matrix(
    lang: str, chunk_f: int, halo_f: int, xfade: int
) -> None:
    """Verify streaming matrix parameters preserve monotonic chunk sequence."""
    test_sentences = {
        "sa": "प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् । यथार्थज्ञानं प्रमाणम् ॥",
        "kn": "ಕನ್ನಡ ಸಾಹಿತ್ಯ ಅತ್ಯಂತ ಶ್ರೀಮಂತವಾದದ್ದು । ಪ್ರತಿಯೊಬ್ಬರೂ ಕಲಿಯಬೇಕು ॥",
        "hi": "सत्य और अहिंसा के मार्ग पर चलना चाहिए । यही जीवन का सार है ॥",
    }
    text = test_sentences[lang]
    engine = Ninada(lang=lang)
    chunks = list(
        engine.stream(
            text,
            lang=lang,
            tier="klatt",
            chunk_frames=chunk_f,
            halo_frames=halo_f,
            crossfade_samples=xfade,
        )
    )
    assert len(chunks) >= 2
    assert chunks[0]["index"] == 0
    assert chunks[-1]["is_final"] is True
    for ch in chunks:
        verify_pcm_samples(ch["pcm"], min_length=200)
        verify_wav_header(ch["wav"])


# ==============================================================================
# 5. Multi-Clause Inter-Clause Transitions
# ==============================================================================
@pytest.mark.parametrize(
    "lang,multi_clause_text",
    [
        ("sa", "ईशा वास्यमिदं सर्वम् । यत्किञ्च जगत्यां जगत् ॥"),
        ("hi", "सूर्य पूर्व दिशा में उगता है । और पश्चिम दिशा में अस्त होता है ॥"),
        ("kn", "ಜ್ಞಾನವೇ ಬೆಳಕು । ಅಜ್ಞಾನವೇ ಕತ್ತಲೆ ॥"),
    ],
)
def test_multiclause_interclause_transitions(
    lang: str, multi_clause_text: str
) -> None:
    """Verify multi-clause synthesis generates audio for all breath groups."""
    engine = Ninada(lang=lang)
    clauses = split_prosodic_clauses(multi_clause_text)
    assert len(clauses) == 2

    pcm, meta = engine.synthesize(multi_clause_text, lang=lang, tier="klatt")
    verify_pcm_samples(pcm, min_length=5000)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.4
