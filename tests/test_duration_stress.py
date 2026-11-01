"""Empirical Stress Testing Harness for Ninada v3.0 24 kHz Studio Prosody & Duration Predictor.

Evaluates duration scaling linearity across 4 length tiers, ordinal survival duration bounds,
speaking rate monotonicity (0.5x to 2.0x), and 24 kHz iSTFTNet waveform integrity across all six
languages (sa, hi, kn, ta, te, ml).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ninada import HOP_LENGTH, SAMPLE_RATE, Ninada
from ninada.g2p import text_to_ids

LANGUAGES = ["sa", "kn", "hi", "ta", "te", "ml"]

TEST_TEXTS: dict[str, dict[str, str]] = {
    "sa": {
        "single_word": "धर्मः",
        "short": "तेन किञ्चित् साहाय्यं कर्तुं शक्यसे वा?",
        "compound": "परन्तु तदीयमहिलाचरित्राणां काठिन्येन एव व्याप्तिः कदाचित् अस्ति इति भाति। प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् इति नैयायिकाः मन्यन्ते।",
        "paragraph": "प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् । यथार्थज्ञानं प्रमाणम् । परन्तु तदीयमहिलाचरित्राणां काठिन्येन एव व्याप्तिः कदाचित् अस्ति इति भाति । धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ।",
    },
    "kn": {
        "single_word": "ನಮಸ್ಕಾರ",
        "short": "ನಮ್ಮ ಧರ್ಮಾಚಾರ್ಯನಾದ ಬೃಹಸ್ಪತಿಯು ಎಲ್ಲಿಯೋ ಅಡಗಿಕೊಂಡಿರುವನು.",
        "compound": "ನೃಪೇಂದ್ರ ನಾರಾಯಣ ಭೂಪ್ ಬಹದ್ದೂರ್ ಮತ್ತು ಕೂಚ್ ಬೆಹಾರ್ನ ಸುನೀತಿ ದೇವಿ ಆಕೆಯ ತಂದೆ ಕಡೆಯ ಅಜ್ಜ ಅಜ್ಜಿ ಆಗಿದ್ದರು ಎಂದು ಇತಿಹಾಸ ಹೇಳುತ್ತದೆ.",
        "paragraph": "ನೃಪೇಂದ್ರ ನಾರಾಯಣ ಭೂಪ್ ಬಹದ್ದೂರ್ ಮತ್ತು ಕೂಚ್ ಬೆಹಾರ್ನ ಸುನೀತಿ ದೇವಿ ಆಕೆಯ ತಂದೆ ಕಡೆಯ ಅಜ್ಜ ಅಜ್ಜಿ ಆಗಿದ್ದರು ಎಂದು ಇತಿಹಾಸ ಹೇಳುತ್ತದೆ. ಆತ ನಟರಾದ ನಾಸಿರುದ್ದೀನ್ ಷಾ ಮತ್ತು ರತ್ನಾ ಪಾಠಕ್ ಷಾ ಅವರ ಪುತ್ರ. ನಮ್ಮ ಧರ್ಮಾಚಾರ್ಯನಾದ ಬೃಹಸ್ಪತಿಯು ಎಲ್ಲಿಯೋ ಅಡಗಿಕೊಂಡಿರುವನು.",
    },
    "hi": {
        "single_word": "नमस्ते",
        "short": "परेड का समापन वायु सेना के विमानों के फ़्लाईपास्ट से हुआ।",
        "compound": "वे प्रयोग करने योग्य बनाने के लिए ओवन और थर्मोस्टैट्स जैसे उपकरणों पर डायल को बड़े आकार और उच्च कंट्रास्ट चिह्नों के साथ चिह्नित करते हैं।",
        "paragraph": "परेड का समापन वायु सेना के विमानों के फ़्लाईपास्ट से हुआ। वे प्रयोग करने योग्य बनाने के लिए ओवन और थर्मोस्टैट्स जैसे उपकरणों पर डायल को बड़े आकार और उच्च कंट्रास्ट चिह्नों के साथ चिह्नित करते हैं। उन्होंने कहा कि कालीघाट में क्रीक पर्यटन को बढ़ावा दिया जाएगा तथा निश्चिंतापुर में सुधार होगा।",
    },
    "ta": {
        "single_word": "வணக்கம்",
        "short": "இப்போது நாம் நுண்ணுயிரியல் ஆபத்துக்கு வருகிறோம்.",
        "compound": "அதேமாதிரி, ஸ்போர்ட்ஸ்ல ஈடுபட்டா எதிர்காலத்துல திறமையான விளையாட்டு வீரனா ஆகமுடியும்னு அவங்களுக்குத் தெரியலை.",
        "paragraph": "அதேமாதிரி, ஸ்போர்ட்ஸ்ல ஈடுபட்டா எதிர்காலத்துல திறமையான விளையாட்டு வீரனா ஆகமுடியும்னு அவங்களுக்குத் தெரியலை. இப்போது நாம் நுண்ணுயிரியல் ஆபத்துக்கு வருகிறோம். வந்தியத்தேவனும் அவளைப் பார்த்துவிட்டான்.",
    },
    "te": {
        "single_word": "నమస్కారం",
        "short": "ఇక్కడి నుండి కొరియారు జలపాతం చేరుకోవాలంటే కొండల మార్గంలో నడవాలి.",
        "compound": "వరద ప్రవాహం పెరగడం వల్ల ప్రాజెక్టు ఆరు గేట్లు తెరిచి దిగువకు నీటిని విడుదల చేస్తున్నట్లు ప్రాజెక్టు అధికారులు తెలిపారు.",
        "paragraph": "వరద ప్రవాహం పెరగడం వల్ల ప్రాజెక్టు ఆరు గేట్లు తెరిచి దిగువకు నీటిని విడుదల చేస్తున్నట్లు ప్రాజెక్టు అధికారులు తెలిపారు. ఇక్కడి నుండి కొరియారు జలపాతం చేరుకోవాలంటే కొండల మార్గంలో నడవాలి. నేను నా తమిళనాడు మర్కంటైల్ బ్యాంక్ ఖాతా నుండి ఎంత డ్రా చేసుకోవచ్చు?",
    },
    "ml": {
        "single_word": "നമസ്കാരം",
        "short": "മുംബൈയിലെ നന്നായി അറിയപ്പെടുന്ന ഒരു ട്രെയിനാണത്.",
        "compound": "രണ്ടായിരത്തി പതിനൊന്നിൽ പുറത്തിറങ്ങിയ മിഷ്കിൻ ചിത്രം യുദ്ധം സെയ് ആണ് അദ്ദേഹത്തിന്റെ ആദ്യ തമിഴ് ചിത്രം എന്ന് രേഖപ്പെടുത്തിയിട്ടുണ്ട്.",
        "paragraph": "രണ്ടായിരത്തി പതിനൊന്നിൽ പുറത്തിറങ്ങിയ മിഷ്കിൻ ചിത്രം യുദ്ധം സെയ് ആണ് അദ്ദേഹത്തിന്റെ ആദ്യ തമിഴ് ചിത്രം എന്ന് രേഖപ്പെടുത്തിയിട്ടുണ്ട്. മുംബൈയിലെ നന്നായി അറിയപ്പെടുന്ന ഒരു ട്രെയിനാണത്. വിധി രേഖപ്പെടുത്തുകയും പിന്നീടുണ്ടായ സമിതി തലത്തിൽ അംഗീകരിക്കുകയും ചെയ്തു.",
    },
}


@pytest.fixture(scope="session")
def studio_engine() -> Ninada:
    engine = Ninada(lang="sa")
    assert engine.is_neural, "Neural ONNX session required for duration stress suite"
    return engine


@pytest.mark.parametrize("lang", LANGUAGES)
def test_duration_scaling_linearity_and_drift(studio_engine: Ninada, lang: str) -> None:
    """Verify durations scale linearly with IPA token counts with zero runaway or negative frames."""
    texts = TEST_TEXTS[lang]
    tiers = ["single_word", "short", "compound", "paragraph"]

    token_counts: list[int] = []
    dur_frames: list[float] = []
    dur_seconds: list[float] = []

    max_sec_by_tier = {
        "single_word": 3.5,
        "short": 9.0,
        "compound": 22.0,
        "paragraph": 50.0,
    }

    for tier in tiers:
        txt = texts[tier]
        t = text_to_ids(txt, lang=lang)
        pcm, meta = studio_engine.synthesize(txt, lang=lang, rate=1.0, tier="neural")
        durs = np.asarray(meta["durations"], dtype=np.float64)
        assert len(durs) > 0, f"Empty durations in {lang}/{tier}"
        assert not np.isnan(durs).any(), f"NaN duration in {lang}/{tier}"
        assert not np.isinf(durs).any(), f"Inf duration in {lang}/{tier}"
        assert durs.min() >= 1.0, f"Sub-frame duration in {lang}/{tier}: {durs.min()}"

        frames = float(durs.sum())
        sec = len(pcm) / float(SAMPLE_RATE)
        assert 0.15 < sec < max_sec_by_tier[tier], (
            f"Duration out of bounds in {lang}/{tier}: {sec:.2f}s"
        )

        token_counts.append(len(t["ipa_ids"]))
        dur_frames.append(frames)
        dur_seconds.append(sec)

    for i in range(len(dur_frames) - 1):
        assert dur_frames[i] < dur_frames[i + 1], (
            f"{lang}: Tier {tiers[i]} ({dur_frames[i]} frames) >= {tiers[i+1]} ({dur_frames[i+1]} frames)"
        )

    corr = float(np.corrcoef(token_counts, dur_frames)[0, 1])
    assert corr >= 0.98, f"{lang}: Duration scaling non-linear (Pearson r = {corr:.4f} < 0.98)"

    for i, tier in enumerate(tiers):
        sec_per_token = dur_seconds[i] / max(1, token_counts[i])
        assert 0.025 <= sec_per_token <= 0.180, (
            f"{lang}/{tier}: Seconds per IPA token {sec_per_token:.4f} outside [0.025, 0.180]"
        )


@pytest.mark.parametrize("lang", LANGUAGES)
def test_speaking_rate_monotonicity_extreme(studio_engine: Ninada, lang: str) -> None:
    """Verify duration decreases strictly monotonically across extreme rates (0.5x to 2.0x)."""
    text = TEST_TEXTS[lang]["compound"]
    rates = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    durations_sec: list[float] = []

    for r in rates:
        pcm, _ = studio_engine.synthesize(text, lang=lang, rate=r, tier="neural")
        durations_sec.append(len(pcm) / float(SAMPLE_RATE))

    for i in range(len(rates) - 1):
        assert durations_sec[i] > durations_sec[i + 1], (
            f"{lang}: Non-monotonic rate scaling: rate {rates[i]} ({durations_sec[i]:.3f}s) <= rate {rates[i+1]} ({durations_sec[i+1]:.3f}s)"
        )

    scaling_ratio = durations_sec[0] / durations_sec[-1]
    assert durations_sec[-1] > 0.5, f"{lang}: Fast speech collapsed: {durations_sec[-1]:.3f}s"
    assert 2.0 <= scaling_ratio <= 4.8, (
        f"{lang}: Rate scaling ratio {scaling_ratio:.2f} outside [2.0, 4.8]"
    )


@pytest.mark.parametrize("lang", LANGUAGES)
def test_ordinal_duration_bounds_and_determinism(studio_engine: Ninada, lang: str) -> None:
    """Verify deterministic ordinal survival duration predictor stays strictly bounded in [1, 50] frames at 1.0x."""
    text = TEST_TEXTS[lang]["short"]
    _, meta1 = studio_engine.synthesize(text, lang=lang, rate=1.0, tier="neural")
    _, meta2 = studio_engine.synthesize(text, lang=lang, rate=1.0, tier="neural")

    d1 = np.asarray(meta1["durations"], dtype=np.int64)
    d2 = np.asarray(meta2["durations"], dtype=np.int64)
    assert np.array_equal(d1, d2), f"{lang}: Non-deterministic durations across identical runs"
    assert d1.min() >= 1, f"{lang}: Token duration dropped below 1 frame ({25}ms)"
    assert d1.max() <= 50, f"{lang}: Token duration {d1.max()} exceeded 50-frame ordinal cap"
    assert HOP_LENGTH == 600 and SAMPLE_RATE == 24000


@pytest.mark.parametrize("lang", LANGUAGES)
def test_waveform_audio_integrity_under_stress(studio_engine: Ninada, lang: str) -> None:
    """Verify synthesized 24 kHz audio waveform exhibits zero NaNs, clean studio RMS, and zero DC offset."""
    text = TEST_TEXTS[lang]["short"]
    for rate in (0.6, 1.0, 1.8):
        pcm, _ = studio_engine.synthesize(text, lang=lang, rate=rate, tier="neural")
        audio = np.asarray(pcm, dtype=np.float32)

        assert not np.isnan(audio).any(), f"{lang}/rate={rate}: NaNs in decoded audio"
        assert not np.isinf(audio).any(), f"{lang}/rate={rate}: Infs in decoded audio"
        assert len(audio) > 4800, f"{lang}/rate={rate}: Audio truncated ({len(audio)} samples)"

        rms = float(np.sqrt(np.mean(audio**2)))
        assert rms > 0.015, f"{lang}/rate={rate}: Silent or weak audio: RMS {rms:.5f}"

        peak = float(np.max(np.abs(audio)))
        assert peak <= 1.01, f"{lang}/rate={rate}: Peak {peak:.3f} exceeds [-1, 1]"
        assert float(np.abs(np.mean(audio))) < 1.0e-4, f"{lang}/rate={rate}: DC offset >= 1.0e-4"
