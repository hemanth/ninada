"""Tier 4: Real-World Application Scenarios End-to-End Tests.

Covers:
- Scenario 1: Classical Sanskrit literature verses (Bhagavad Gita, Upanishads, Vedic chants).
- Scenario 2: Hindi conversational dialogue sentences and announcements.
- Scenario 3: Multilingual technical, scientific, and educational passages (kn, ta, te, ml, sa).
- Scenario 4: Streaming synthesis workloads (TTFA, chunk monotonicity, and termination).
"""
from __future__ import annotations

import pytest

from ninada import Ninada, encode_wav_bytes, split_prosodic_clauses
from tests.e2e.helpers import verify_pcm_samples, verify_wav_header


# ==============================================================================
# Scenario 1: Classical Sanskrit Literature Verses
# ==============================================================================
@pytest.mark.parametrize(
    "title,verse_text",
    [
        (
            "Bhagavad_Gita_1_1",
            "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥",
        ),
        (
            "Isha_Upanishad_1",
            "ईशा वास्यमिदं सर्वं यत्किञ्च जगत्यां जगत् । तेन त्यक्तेन भुञ्जीथा मा गृधः कस्यस्विद्धनम् ॥",
        ),
        (
            "Rigvedic_Gayatri",
            "ॐ भूर्भुवः स्वः । तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि । धियो यो नः प्रचोदयात् ॥",
        ),
        (
            "Mandukya_Peace_Chant",
            "ॐ भद्रं कर्णेभिः शृणुयाम देवाः । भद्रं पश्येमाक्षभिर्यजत्राः ॥",
        ),
        (
            "Subhashita_Truth_Dharma",
            "सत्यं ब्रूयात् प्रियं ब्रूयात् न ब्रूयात् सत्यमप्रियम् । प्रियं च नानृतं ब्रूयात् एष धर्मः सनातनः ॥",
        ),
    ],
)
def test_scenario1_classical_sanskrit_verses(
    engine_sa: Ninada, title: str, verse_text: str
) -> None:
    """Verify classical Sanskrit literature recitations synthesize into high-fidelity audio."""
    clauses = split_prosodic_clauses(verse_text)
    assert len(clauses) >= 2, f"Expected multi-clause shloka, got {len(clauses)}"

    pcm, meta = engine_sa.synthesize(verse_text, lang="sa", tier="klatt")
    verify_pcm_samples(pcm, min_length=20000)
    assert meta["lang"] == "sa"

    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 1.5, f"Verse audio {info['duration_sec']}s too short"


# ==============================================================================
# Scenario 2: Hindi Conversational Dialogue Sentences
# ==============================================================================
@pytest.mark.parametrize(
    "dialogue_name,hindi_text",
    [
        (
            "Welcome_Greeting",
            "नमस्ते! आपका हमारे इस नए कार्यक्रम में हार्दिक स्वागत है। आशा है आपको यह पसंद आएगा।",
        ),
        (
            "Inquiry_Scheduling",
            "क्या आप मुझे बता सकते हैं कि अगला सत्र कितने बजे शुरू होगा? धन्यवाद!",
        ),
        (
            "Customer_Service",
            "आपकी समस्या का समाधान शीघ्र किया जाएगा। कृपया कुछ समय प्रतीक्षा करें।",
        ),
        (
            "Weather_Report",
            "आज राजधानी दिल्ली में तेज हवाओं के साथ बारिश होने की संभावना है।",
        ),
        (
            "Farewell_Gratitude",
            "आप सभी का बहुत-बहुत धन्यवाद। हम फिर मिलेंगे, तब तक के लिए नमस्कार।",
        ),
    ],
)
def test_scenario2_hindi_conversational_sentences(
    engine_hi: Ninada, dialogue_name: str, hindi_text: str
) -> None:
    """Verify Hindi conversational dialogue synthesizes with natural speech duration."""
    pcm, meta = engine_hi.synthesize(hindi_text, lang="hi", tier="klatt")
    verify_pcm_samples(pcm, min_length=15000)
    assert meta["lang"] == "hi"

    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 1.0


# ==============================================================================
# Scenario 3: Multilingual Technical & Educational Passages
# ==============================================================================
@pytest.mark.parametrize(
    "lang,topic,passage_text",
    [
        (
            "kn",
            "Kannada_AI_Tech",
            "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಮತ್ತು ಯಂತ್ರ ಕಲಿಕೆ ಇಂದಿನ ತಂತ್ರಜ್ಞಾನ ಯುಗದಲ್ಲಿ ಮಹತ್ವದ ಪಾತ್ರ ವಹಿಸುತ್ತಿವೆ.",
        ),
        (
            "ta",
            "Tamil_Literature",
            "தமிழ் மொழி தொன்மையான இலக்கிய வளமும் செழுமையான பண்பாடும் கொண்ட உலகளாவிய மொழியாகும்.",
        ),
        (
            "te",
            "Telugu_Astronomy",
            "సూర్యుని చుట్టూ గ్రహాలు నిర్దిష్ట కక్ష్యలో తిరుగుతూ ఉంటాయి, ఇది గురుత్వాకర్షణ శక్తి వల్ల జరుగుతుంది.",
        ),
        (
            "ml",
            "Malayalam_Education",
            "വിജ്ഞാന സമൂഹത്തിന്റെ വളർച്ചയ്ക്ക് ഗുണനിലവಾರമുള്ള വിദ്യാഭ്യാസം അത്യന്താപേക്ഷിതമാണ്.",
        ),
        (
            "sa",
            "Sanskrit_Aryabhatiya",
            "प्रणिपत्यैकमनेकं कं सत्यां देवतां परं ब्रह्म । आर्यभटस्त्रीणि गदति गणितं कालक्रिया गोलम् ॥",
        ),
    ],
)
def test_scenario3_multilingual_passages(
    lang: str, topic: str, passage_text: str
) -> None:
    """Verify technical, scientific, and educational domain passages synthesize cleanly."""
    engine = Ninada(lang=lang)
    pcm, meta = engine.synthesize(passage_text, lang=lang, tier="klatt")
    verify_pcm_samples(pcm, min_length=15000)
    assert meta["lang"] == lang

    wav = encode_wav_bytes(pcm)
    info = verify_wav_header(wav)
    assert info["duration_sec"] > 1.0


# ==============================================================================
# Scenario 4: Streaming Synthesis Workloads
# ==============================================================================
def test_scenario4_streaming_workload_metrics(engine_sa: Ninada) -> None:
    """Verify interactive streaming session produces valid chunks, TTFA, and termination."""
    passage = (
        "प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् । "
        "यथार्थज्ञानं प्रमाणम् । "
        "सत्यमेव जयते नानृतम् ॥"
    )
    chunks = list(
        engine_sa.stream(
            passage,
            lang="sa",
            tier="klatt",
            chunk_frames=48,
            halo_frames=12,
            crossfade_samples=192,
        )
    )

    assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"

    first_chunk = chunks[0]
    assert first_chunk["index"] == 0
    assert first_chunk["clause_index"] == 0
    assert first_chunk["ttfa_ms"] > 0
    assert first_chunk["is_final"] is False

    last_chunk = chunks[-1]
    assert last_chunk["index"] == len(chunks) - 1
    assert last_chunk["is_final"] is True

    total_streamed_samples = 0
    for idx, ch in enumerate(chunks):
        assert ch["index"] == idx, f"Chunk index gap: expected {idx}, got {ch['index']}"
        verify_pcm_samples(ch["pcm"], min_length=100)
        verify_wav_header(ch["wav"])
        total_streamed_samples += len(ch["pcm"])

    assert total_streamed_samples > 10000, "Streamed audio duration insufficient"
