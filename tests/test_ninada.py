from __future__ import annotations

from pathlib import Path
import numpy as np

from ninada import (
    HOP_LENGTH,
    IPA_VOCAB,
    LANGUAGE_IDS,
    RETROFLEX_LATERAL_ID,
    SAMPLE_RATE,
    VOCAB_TOKENS,
    Ninada,
    speak,
    split_prosodic_clauses,
    stream,
    text_to_ids,
    text_to_ipa,
)


def test_vocab_size_and_sanskrit_extensions() -> None:
    assert SAMPLE_RATE == 24000
    assert HOP_LENGTH == 600
    assert len(VOCAB_TOKENS) == 87
    assert VOCAB_TOKENS[85] == "\u090b"
    assert VOCAB_TOKENS[86] == "\u0919"
    assert IPA_VOCAB["\u026d"] == RETROFLEX_LATERAL_ID == 79
    assert set(LANGUAGE_IDS.keys()) == {"sa", "hi", "kn", "ta", "te", "ml"}


def test_sanskrit_phonology_and_akshara_weights() -> None:
    parsed = text_to_ids("\u090b\u0924\u0951\u0902 \u0938\u094b\u093d\u0939\u092e\u094d \u0936\u0902\u0915\u0930\u0903", lang="sa")
    assert parsed["lang"] == "sa"
    assert "\u090b" in parsed["tokens"]
    assert "\u0919" in parsed["tokens"]
    assert "\u093d" not in parsed["normalized_devanagari"]
    assert "\u0951" not in parsed["normalized_devanagari"]
    assert len(parsed["ipa_ids"]) >= 10
    weights = [a["weight"] for a in parsed["aksharas"]]
    assert "G" in weights


def test_multilingual_brahmic_and_iast_parity() -> None:
    kn = text_to_ids("\u0c85\u0cb0\u0cbf\u0cb5\u0cc7 \u0c97\u0cc1\u0cb0\u0cc1")
    assert kn["lang"] == "kn"
    assert kn["script"] == "kannada"
    assert "\u026d" in text_to_ipa("\u0cb9\u0cb3\u0cc6\u0c97\u0ca8\u0ccd\u0ca8\u0ca1", lang="kn")

    ta = text_to_ids("\u0b9a\u0b99\u0bcd\u0b95\u0ba4\u0bcd \u0ba4\u0bae\u0bbf\u0bb4\u0bcd")
    assert ta["lang"] == "ta"
    assert "\u0917" in ta["normalized_devanagari"]  # post-nasal voicing ṅk -> ṅg
    assert "\u0934" in ta["tokens"]
    assert "\u027b" in ta["ipa"]  # Tamil ழ -> ɻ

    ml = text_to_ids("\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d02")
    assert ml["lang"] == "ml"
    assert "\u026d" in ml["ipa"]  # Malayalam ള -> ɭ (slot 79)

    iast = text_to_ids("praj\u00f1\u0101p\u016brvakavy\u0101p\u0101ravattva\u1e41 j\u012bvatvam", lang="sa")
    assert iast["script"] == "iast"
    assert len(iast["input_ids"]) > 20
    assert len(iast["ipa_ids"]) > 15


def test_prosodic_clause_splitting_and_klatt_stream() -> None:
    shloka = "\u092a\u094d\u0930\u091c\u094d\u091e\u093e\u092a\u0942\u0930\u094d\u0935\u0915\u0935\u094d\u092f\u093e\u092a\u093e\u0930\u0935\u0924\u094d\u0924\u094d\u0935\u0902 \u091c\u0940\u0935\u0924\u094d\u0935\u092e\u094d \u0964 \u092f\u0925\u093e\u0930\u094d\u0925\u091c\u094d\u091e\u093e\u0928\u0902 \u092a\u094d\u0930\u092e\u093e\u0923\u092e\u094d \u0965"
    clauses = split_prosodic_clauses(shloka)
    assert len(clauses) == 2

    chunks = list(stream(shloka, lang="sa", tier="klatt"))
    assert len(chunks) == 2
    assert chunks[0]["sample_rate"] == 24000
    assert chunks[0]["ttfa_ms"] > 0
    assert chunks[-1]["is_final"] is True
    assert chunks[0]["wav"][:4] == b"RIFF"


def test_klatt_zero_dep_wav_synthesis() -> None:
    wav = speak("\u0938\u0924\u094d\u092f\u092e\u0947\u0935 \u091c\u092f\u0924\u0947", lang="sa", tier="klatt")
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert len(wav) > 6000

    engine = Ninada(lang="kn")
    pcm, meta = engine.synthesize("\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \u0ca8\u0cc1\u0ca1\u0cbf", tier="klatt")
    assert meta["tier"] == "klatt_zero_dep"
    assert len(pcm) > 3000


def test_neural_onnx_streaming_and_synthesis_if_present() -> None:
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    studio_p = docs_dir / "ninada_24k_q8f16.onnx"
    enc_p = docs_dir / "encoder_flow_quantized.onnx"
    voc_p = docs_dir / "vocoder_quantized.onnx"

    if studio_p.exists():
        engine = Ninada(lang="sa", model_path=studio_p, voice_dir=docs_dir / "voices")
    elif enc_p.exists() and voc_p.exists():
        engine = Ninada(lang="sa", encoder_path=enc_p, vocoder_path=voc_p)
    else:
        return

    chunks = list(
        engine.stream(
            "\u092a\u094d\u0930\u091c\u094d\u091e\u093e\u092a\u0942\u0930\u094d\u0935\u0915\u0935\u094d\u092f\u093e\u092a\u093e\u0930\u0935\u0924\u094d\u0924\u094d\u0935\u0902 \u091c\u0940\u0935\u0924\u094d\u0935\u092e\u094d \u0964",
            tier="neural",
            chunk_frames=40,
        )
    )
    assert len(chunks) >= 1
    assert chunks[0]["tier"] == "neural_onnx"
    assert chunks[0]["sample_rate"] == 24000
    assert chunks[0]["ttfa_ms"] > 0
    assert chunks[-1]["is_final"] is True

    pcm, meta = engine.synthesize("\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \u0cad\u0cbe\u0cb7\u0cc6\u0caf \u0ca7\u0ccd\u0cb5\u0ca8\u0cbf", lang="kn", tier="neural")
    assert meta["tier"] == "neural_onnx"
    assert len(pcm) > 18000
    wav = engine.speak("\u0938\u0924\u094d\u092f\u092e\u0947\u0935 \u091c\u092f\u0924\u0947")
    assert wav[:4] == b"RIFF"
    assert len(wav) > 36000


def test_24k_studio_dc_offset_compliance() -> None:
    """Verify all 6 languages produce 24 kHz waveforms with DC offset < 1.0e-4."""
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    studio_p = docs_dir / "ninada_24k_q8f16.onnx"
    if not studio_p.exists():
        return

    engine = Ninada(lang="sa", model_path=studio_p, voice_dir=docs_dir / "voices")
    sample_texts = {
        "sa": "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः",
        "hi": "भारत एक विशाल और विविधतापूर्ण देश है",
        "kn": "ಕನ್ನಡ ನಾಡು ನುಡಿ ಸಂಸ್ಕೃತಿ ಅತ್ಯಂತ ಶ್ರೀಮಂತವಾಗಿದೆ",
        "ta": "தமிழ் மொழி உலகின் தொன்மையான மொழிகளில் ஒன்று",
        "te": "తెలుగు భాష చాలా మధురమైనది మరియు సుందరమైనది",
        "ml": "മലയാളം കേരളത്തിന്റെ മാതൃഭാഷയാണ്",
    }
    for lang, txt in sample_texts.items():
        wav, _ = engine.synthesize(txt, lang=lang, tier="neural")
        arr = np.asarray(wav, dtype=np.float32)
        dc = float(np.abs(np.mean(arr)))
        assert dc < 1.0e-4, f"Language [{lang}] DC offset {dc:.6e} exceeds 1.0e-4 limit"


def test_pan_indic_ipa_phonetic_fidelity_and_female_voice() -> None:
    """Regression guard for jna allophony, Dravidian short-e, zero ˈə stress, medial ɐ, and female studio voice."""
    from ninada.ipa_g2p import text_to_ipa

    sa_ipa = text_to_ipa("प्रज्ञा ज्ञानं", lang="sa")
    hi_ipa = text_to_ipa("प्रज्ञा विज्ञान", lang="hi")
    te_ipa = text_to_ipa("తెలుగు భాష", lang="te")
    kn_ipa = text_to_ipa("ಕನ್ನಡ ಭಾಷೆಯ ನೈಸರ್ಗಿಕ ಧ್ವನಿ", lang="kn")
    ml_ipa = text_to_ipa("മലയാള ഭാഷയുടെ സ്വാഗതം", lang="ml")

    assert "ɡɲ" in sa_ipa
    assert "ɡj" in hi_ipa
    assert "tˈɛlʊɡʊ" in te_ipa
    assert "ˈə" not in sa_ipa and "ˈə" not in kn_ipa and "ˈə" not in ml_ipa
    assert "ɐ" in kn_ipa and "ɐ" in ml_ipa
    assert "jʊɖɛ" in ml_ipa

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    studio_p = docs_dir / "ninada_24k_q8f16.onnx"
    if studio_p.exists() and (docs_dir / "voices" / "female.bin").exists():
        engine = Ninada(lang="sa", model_path=studio_p, voice_dir=docs_dir / "voices")
        wav_f, meta_f = engine.synthesize("सत्यमेव जयते ।", lang="sa", voice="female")
        assert meta_f["tier"] == "neural_onnx"
        assert len(wav_f) > 18000

