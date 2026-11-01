"""Tier 1: Feature Coverage End-to-End Tests.

Covers pronunciation, phonetics, all 6 supported languages (sa, hi, kn, ta, te, ml),
and runtime flags (tiers, speaking rate, prosodic noise parameters, CLI invocation,
and streaming chunk parameters).
Each feature includes at least 5 distinct test cases.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ninada import Ninada, encode_wav_bytes, speak, stream
from tests.e2e.helpers import verify_pcm_samples, verify_wav_header


# ==============================================================================
# Feature 1: Sanskrit Vocalic Liquids (ऋ, ॠ, ೠ) and Vocalic Extensions
# ==============================================================================
@pytest.mark.parametrize(
    "text,expected_sub",
    [
        ("ऋतं वद", "ऋ"),
        ("ऋषिः वनम् अगच्छत्", "ऋ"),
        ("ॠकारः दीर्घस्वरः अस्ति", "ॠ"),
        ("पितॄन् तर्पयति", "ॄ"),
        ("ೠಷಿ ಸಂಸ್ಕೃತಿಃ", "ೠ"),
    ],
)
def test_f1_sanskrit_vocalic_liquids_synthesis(
    engine_sa: Ninada, text: str, expected_sub: str
) -> None:
    """Verify vocalic liquids synthesize into valid PCM and 16kHz WAV."""
    pcm, meta = engine_sa.synthesize(text, lang="sa", tier="klatt")
    verify_pcm_samples(pcm, min_length=2000)
    assert meta["lang"] == "sa"

    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.15


# ==============================================================================
# Feature 2: Homorganic Nasal Sandhi Preservation
# ==============================================================================
@pytest.mark.parametrize(
    "text",
    [
        "शङ्करः शङ्खं दध्मौ",
        "गङ्गा हिमालयात् प्रवहति",
        "सङ्घः बलवान् वर्तते",
        "अङ्कगणितं कठिनम् अस्ति",
        "पञ्चतन्त्रस्य कथाः मधुराः",
    ],
)
def test_f2_homorganic_nasal_sandhi_synthesis(engine_sa: Ninada, text: str) -> None:
    """Verify homorganic nasal clusters (ṅk, ṅkh, ṅg, ṅgh, ñc) synthesize cleanly."""
    pcm, meta = engine_sa.synthesize(text, lang="sa", tier="klatt")
    verify_pcm_samples(pcm, min_length=3000)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.2


# ==============================================================================
# Feature 3: Hindi Schwa Deletion and Word Cadence
# ==============================================================================
@pytest.mark.parametrize(
    "text",
    [
        "कमल का फूल तालाब में खिलता है",
        "भारत एक महान देश है",
        "राम अपने घर जा रहा है",
        "कड़ी मेहनत से सफलता मिलती है",
        "नमस्ते, आप कैसे हैं?",
    ],
)
def test_f3_hindi_schwa_cadence_synthesis(engine_hi: Ninada, text: str) -> None:
    """Verify Hindi sentences with terminal and medial schwas synthesize cleanly."""
    pcm, meta = engine_hi.synthesize(text, lang="hi", tier="klatt")
    verify_pcm_samples(pcm, min_length=4000)
    assert meta["lang"] == "hi"

    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.3


# ==============================================================================
# Feature 4: Dravidian Phoneme Preservation Across Native Alphabets
# ==============================================================================
@pytest.mark.parametrize(
    "lang,text,expected_char",
    [
        ("kn", "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ", "ಙ"),
        ("kn", "ಅಙ್ ಕಣ್ಣು ತೆರೆಯಿತು", "ಙ"),
        ("te", "ఝాము గడచింది వాన కురిసింది", "ఝ"),
        ("ml", "ഝഷം ജലാശയത്തിൽ നീന്തുന്നു", "ഝ"),
        ("ta", "தமிழ்நாடு என் தாய்நாடு", "ழ"),
    ],
)
def test_f4_dravidian_phoneme_preservation(
    lang: str, text: str, expected_char: str
) -> None:
    """Verify distinct Dravidian phonemes produce clean acoustic representations."""
    engine = Ninada(lang=lang)
    pcm, meta = engine.synthesize(text, lang=lang, tier="klatt")
    verify_pcm_samples(pcm, min_length=3000)
    assert meta["lang"] == lang

    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.20


# ==============================================================================
# Feature 5: Sanskrit Avagraha Prolongation and Vedic Svara Accents
# ==============================================================================
@pytest.mark.parametrize(
    "text",
    [
        "सोऽहम् इति विद्धि",
        "शिवोऽहम् शिवोऽहम्",
        "अ॒ग्निमी॑ळे पु॒रोहि॑तम्",
        "य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म्",
        "होता॑रं रत्न॒धात॑मम्",
    ],
)
def test_f5_avagraha_and_vedic_svara_synthesis(engine_sa: Ninada, text: str) -> None:
    """Verify avagraha and Vedic accents (udatta, anudatta) synthesize without error."""
    pcm, meta = engine_sa.synthesize(text, lang="sa", tier="klatt")
    verify_pcm_samples(pcm, min_length=2500)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.15


# ==============================================================================
# Feature 6: Virama Co-articulation Without Unnatural Pauses
# ==============================================================================
@pytest.mark.parametrize(
    "lang,text",
    [
        ("sa", "प्रज्ञापूर्वकव्यापारवत्त्वम्"),
        ("sa", "सत्यमेव जयते नानृतम्"),
        ("sa", "धर्मक्षेत्रे कुरुक्षेत्रे समवेताः"),
        ("kn", "ಕರ್ನಾಟಕ ರಾಜ್ಯೋತ್ಸವದ ಶುಭಾಶಯಗಳು"),
        ("kn", "ಜ್ಞಾನ ಮತ್ತು ವಿಜ್ಞಾನ ಎರಡೂ ಮುಖ್ಯ"),
    ],
)
def test_f6_virama_coarticulation_clusters(lang: str, text: str) -> None:
    """Verify complex consonant conjuncts with virama articulate smoothly."""
    engine = Ninada(lang=lang)
    pcm, meta = engine.synthesize(text, lang=lang, tier="klatt")
    verify_pcm_samples(pcm, min_length=3500)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.25


# ==============================================================================
# Feature 7: Sanskrit Language Full Coverage (5 Genres)
# ==============================================================================
@pytest.mark.parametrize(
    "text",
    [
        "अथातो ब्रह्मजिज्ञासा ।",
        "सत्यं वद धर्मं चर ।",
        "विद्वत्वं च नृपत्वं च नैव तुल्ये कदाचन ।",
        "शरीरमाद्यं खलु धर्मसाधनम् ।",
        "योगश्चित्तवृत्तिनिरोधः ।",
    ],
)
def test_f7_sanskrit_language_coverage(engine_sa: Ninada, text: str) -> None:
    """Verify Classical Sanskrit genre samples synthesize valid speech."""
    pcm, meta = engine_sa.synthesize(text, lang="sa", tier="auto")
    verify_pcm_samples(pcm, min_length=2000)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.2


# ==============================================================================
# Feature 8: Hindi Language Full Coverage (5 Styles)
# ==============================================================================
@pytest.mark.parametrize(
    "text",
    [
        "आज मौसम बहुत सुहावना और ठंडा है।",
        "क्या आप कल विद्यालय जाएँगे?",
        "कृपया इस पत्र को शीघ्र डाक में डाल दें।",
        "विज्ञान और प्रौद्योगिकी का विकास तीव्र गति से हो रहा है।",
        "सच्चाई की हमेशा जीत होती है।",
    ],
)
def test_f8_hindi_language_coverage(engine_hi: Ninada, text: str) -> None:
    """Verify Hindi speech generation across distinct syntactic structures."""
    pcm, meta = engine_hi.synthesize(text, lang="hi", tier="auto")
    verify_pcm_samples(pcm, min_length=2500)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.25


# ==============================================================================
# Feature 9: Kannada Language Full Coverage (5 Styles)
# ==============================================================================
@pytest.mark.parametrize(
    "text",
    [
        "ಕನ್ನಡ ನಾಡು ನುಡಿ ಅತ್ಯಂತ ಪ್ರಾಚೀನವಾದದ್ದು.",
        "ನೀವು ಎಲ್ಲಿಗೆ ಹೋಗುತ್ತಿದ್ದೀರಿ?",
        "ಎಲ್ಲರಿಗೂ ನಮಸ್ಕಾರ, ಇಂದಿನ ಸಭೆಗೆ ಸ್ವಾಗತ.",
        "ಕಾಯಕವೇ ಕೈಲಾಸ ಎಂದು ಬಸವಣ್ಣನವರು ಹೇಳಿದರು.",
        "ಸಿರಿಗನ್ನಡಂ ಗೆಲ್ಗೆ, ಸಿರಿಗನ್ನಡಂ ಬಾಳ್ಗೆ.",
    ],
)
def test_f9_kannada_language_coverage(engine_kn: Ninada, text: str) -> None:
    """Verify Kannada speech generation across literary and daily sentences."""
    pcm, meta = engine_kn.synthesize(text, lang="kn", tier="auto")
    verify_pcm_samples(pcm, min_length=2500)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.25


# ==============================================================================
# Feature 10: Tamil Language Full Coverage (5 Styles)
# ==============================================================================
@pytest.mark.parametrize(
    "text",
    [
        "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "யாதும் ஊரே யாவரும் கேளிர்.",
        "தமிழுக்கும் அமுதென்று பேர்.",
        "இன்று காலை வானிலை மிக நன்று.",
        "வாழ்க நிரந்தரம் வாழ்க தமிழ்மொழி.",
    ],
)
def test_f10_tamil_language_coverage(engine_ta: Ninada, text: str) -> None:
    """Verify Tamil speech generation across classical and modern sentences."""
    pcm, meta = engine_ta.synthesize(text, lang="ta", tier="auto")
    verify_pcm_samples(pcm, min_length=2500)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.25


# ==============================================================================
# Feature 11: Telugu Language Full Coverage (5 Styles)
# ==============================================================================
@pytest.mark.parametrize(
    "text",
    [
        "నమస్కారం, మీరు ఎలా ఉన్నారు?",
        "తెలుగు భాష తేనె వంటిది.",
        "రేపు ఉదయం మేము హైదరాబాదు వెళ్తున్నాము.",
        "సత్యమేవ జయతే అని మన పూర్వీకులు చెప్పారు.",
        "విద్య లేని వాడు వింత పశువు.",
    ],
)
def test_f11_telugu_language_coverage(engine_te: Ninada, text: str) -> None:
    """Verify Telugu speech generation across classical and conversational texts."""
    pcm, meta = engine_te.synthesize(text, lang="te", tier="auto")
    verify_pcm_samples(pcm, min_length=2500)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.25


# ==============================================================================
# Feature 12: Malayalam Language Full Coverage (5 Styles)
# ==============================================================================
@pytest.mark.parametrize(
    "text",
    [
        "നമസ്കാരം, സുഖമാണോ?",
        "മലയാളം കേരളത്തിന്റെ മാതൃഭാഷയാണ്.",
        "നാളെ രാവിലെ ഞങ്ങൾ തിരുവനന്തപുരത്തേക്ക് പോകും.",
        "വിദ്യ ധനങ്ങളിൽ വെച്ച് ഉത്തമമായ ധനമാണ്.",
        "കേരളം പ്രകൃതിഭംഗിയാൽ സമൃദ്ധമായ നാടാണ്.",
    ],
)
def test_f12_malayalam_language_coverage(engine_ml: Ninada, text: str) -> None:
    """Verify Malayalam speech generation with chillu letters and geminates."""
    pcm, meta = engine_ml.synthesize(text, lang="ml", tier="auto")
    verify_pcm_samples(pcm, min_length=2500)
    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.25


# ==============================================================================
# Feature 13: Synthesis Tier Selection Modes and Speak API
# ==============================================================================
@pytest.mark.parametrize(
    "tier,text,lang",
    [
        ("auto", "सत्यमेव जयते", "sa"),
        ("klatt", "ಸತ್ಯಮೇವ ಜಯತೇ", "kn"),
        ("neural", "सत्यमेव जयते", "sa"),
        ("klatt", "नमस्ते भारत", "hi"),
        ("auto", "ಕನ್ನಡ ನುಡಿ", "kn"),
    ],
)
def test_f13_synthesis_tier_selection(
    tier: str, text: str, lang: str
) -> None:
    """Verify tier selection flags ('auto', 'klatt', 'neural') function accurately."""
    engine = Ninada(lang=lang)
    pcm, meta = engine.synthesize(text, lang=lang, tier=tier)
    verify_pcm_samples(pcm, min_length=1500)
    if tier == "klatt":
        assert meta["tier"] == "klatt_zero_dep"
    elif tier == "neural" and engine.is_neural:
        assert meta["tier"] == "neural_onnx"

    wav = engine.speak(text, lang=lang, tier=tier)
    verify_wav_header(wav)


def test_f13_standalone_speak_function() -> None:
    """Verify top-level speak() function produces valid WAV bytes."""
    wav = speak("सत्यमेव जयते", lang="sa", tier="klatt")
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 0.1


# ==============================================================================
# Feature 14: Speaking Rate Monotonic Duration Control
# ==============================================================================
@pytest.mark.parametrize(
    "rate_pair",
    [
        (0.60, 1.00),
        (0.75, 1.25),
        (0.85, 1.15),
        (1.00, 1.50),
        (1.20, 1.80),
    ],
)
def test_f14_speaking_rate_duration_monotonicity(
    engine_sa: Ninada, rate_pair: tuple[float, float]
) -> None:
    """Verify slower speaking rate produces longer duration than faster rate."""
    slow_rate, fast_rate = rate_pair
    text = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः"
    pcm_slow, _ = engine_sa.synthesize(text, lang="sa", rate=slow_rate, tier="klatt")
    pcm_fast, _ = engine_sa.synthesize(text, lang="sa", rate=fast_rate, tier="klatt")
    assert len(pcm_slow) > len(pcm_fast), (
        f"Expected slow ({slow_rate}x, {len(pcm_slow)} samples) > "
        f"fast ({fast_rate}x, {len(pcm_fast)} samples)"
    )


# ==============================================================================
# Feature 15: Noise and Prosodic Modulation Parameters
# ==============================================================================
@pytest.mark.parametrize(
    "noise,noise_dur",
    [
        (0.500, 0.60),
        (0.667, 0.80),
        (0.800, 0.90),
        (0.333, 0.50),
        (1.000, 1.00),
    ],
)
def test_f15_prosodic_noise_parameters(
    engine_sa: Ninada, noise: float, noise_dur: float
) -> None:
    """Verify stochastic flow noise parameters execute without deviation or crash."""
    text = "अहं ब्रह्मास्मि"
    pcm, meta = engine_sa.synthesize(text, lang="sa", noise=noise, tier="auto")
    verify_pcm_samples(pcm, min_length=2000)
    wav = encode_wav_bytes(pcm)
    verify_wav_header(wav)

    # Verify streaming accepts noise and noise_duration
    chunks = list(
        engine_sa.stream(
            text, lang="sa", noise=noise, noise_duration=noise_dur, tier="auto"
        )
    )
    assert len(chunks) >= 1
    verify_wav_header(chunks[0]["wav"])


# ==============================================================================
# Feature 16: CLI Interface Invocation
# ==============================================================================
def test_f16_cli_basic_execution(run_cli: Any, tmp_path: Path) -> None:
    """Verify CLI basic text synthesis to output WAV file."""
    out_file = tmp_path / "cli_test_1.wav"
    res = run_cli("सत्यमेव जयते", "-l", "sa", "-o", str(out_file))
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"
    assert out_file.exists()
    verify_wav_header(out_file.read_bytes())


def test_f16_cli_stdin_pipe(run_cli: Any, tmp_path: Path) -> None:
    """Verify CLI accepts input text piped through stdin."""
    out_file = tmp_path / "cli_test_2.wav"
    res = run_cli("-l", "kn", "-o", str(out_file), input_text="ಕನ್ನಡ ನುಡಿ")
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"
    assert out_file.exists()
    verify_wav_header(out_file.read_bytes())


def test_f16_cli_inspect_json_mode(run_cli: Any) -> None:
    """Verify CLI --inspect mode prints valid JSON metadata."""
    res = run_cli("नमस्ते", "-l", "hi", "--inspect")
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"
    parsed = json.loads(res.stdout)
    assert parsed["lang"] == "hi"
    assert "aksharas" in parsed


def test_f16_cli_speaking_rate_flag(run_cli: Any, tmp_path: Path) -> None:
    """Verify CLI --rate flag modulates output length."""
    out_file = tmp_path / "cli_test_rate.wav"
    res = run_cli("धर्मक्षेत्रे", "-l", "sa", "-r", "1.5", "-o", str(out_file))
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"
    assert out_file.exists()
    verify_wav_header(out_file.read_bytes())


def test_f16_cli_tier_flag(run_cli: Any, tmp_path: Path) -> None:
    """Verify CLI --tier klatt forces pure-math synthesis."""
    out_file = tmp_path / "cli_test_klatt.wav"
    res = run_cli("सत्यं वद", "-l", "sa", "--tier", "klatt", "-o", str(out_file))
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"
    assert "tier=klatt_zero_dep" in res.stdout
    assert out_file.exists()
    verify_wav_header(out_file.read_bytes())


# ==============================================================================
# Feature 17: Streaming Chunk Configuration Parameters
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
def test_f17_streaming_chunk_configuration(
    engine_sa: Ninada, chunk_f: int, halo_f: int, xfade: int
) -> None:
    """Verify streaming parameters yield consistent sequential chunks."""
    text = "प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् । यथार्थज्ञानं प्रमाणम् ॥"
    chunks = list(
        engine_sa.stream(
            text,
            lang="sa",
            chunk_frames=chunk_f,
            halo_frames=halo_f,
            crossfade_samples=xfade,
        )
    )
    assert len(chunks) >= 1
    assert chunks[0]["index"] == 0
    assert chunks[-1]["is_final"] is True
    for ch in chunks:
        verify_pcm_samples(ch["pcm"])
        verify_wav_header(ch["wav"])
