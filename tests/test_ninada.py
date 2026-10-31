from __future__ import annotations

from ninada import LANGUAGE_IDS, VOCAB_TOKENS, Ninada, speak, text_to_ids


def test_vocab_size_and_sanskrit_extensions() -> None:
    assert len(VOCAB_TOKENS) == 87
    assert VOCAB_TOKENS[85] == "\u090b"
    assert VOCAB_TOKENS[86] == "\u0919"
    assert set(LANGUAGE_IDS.keys()) == {"sa", "hi", "kn", "ta", "te", "ml"}


def test_sanskrit_phonology_and_akshara_weights() -> None:
    parsed = text_to_ids("\u090b\u0924\u0951\u0902 \u0938\u094b\u093d\u0939\u092e\u094d \u0936\u0902\u0915\u0930\u0903", lang="sa")
    assert parsed["lang"] == "sa"
    assert "\u090b" in parsed["tokens"]
    assert "\u0919" in parsed["tokens"]
    assert "\u093d" not in parsed["normalized_devanagari"]
    assert "\u0951" not in parsed["normalized_devanagari"]
    weights = [a["weight"] for a in parsed["aksharas"]]
    assert "G" in weights


def test_multilingual_brahmic_and_iast_parity() -> None:
    kn = text_to_ids("\u0c85\u0cb0\u0cbf\u0cb5\u0cc7 \u0c97\u0cc1\u0cb0\u0cc1")
    assert kn["lang"] == "kn"
    assert kn["script"] == "kannada"

    ta = text_to_ids("\u0b9a\u0b99\u0bcd\u0b95\u0ba4\u0bcd \u0ba4\u0bae\u0bbf\u0bb4\u0bcd")
    assert ta["lang"] == "ta"
    assert "\u0917" in ta["normalized_devanagari"]  # post-nasal voicing ṅk -> ṅg
    assert "\u0934" in ta["tokens"]

    iast = text_to_ids("praj\u00f1\u0101p\u016brvakavy\u0101p\u0101ravattva\u1e41 j\u012bvatvam", lang="sa")
    assert iast["script"] == "iast"
    assert len(iast["input_ids"]) > 20


def test_klatt_zero_dep_wav_synthesis() -> None:
    wav = speak("\u0938\u0924\u094d\u092f\u092e\u0947\u0935 \u091c\u092f\u0924\u0947", lang="sa", tier="klatt")
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert len(wav) > 4000

    engine = Ninada(lang="kn")
    pcm, meta = engine.synthesize("\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \u0ca8\u0cc1\u0ca1\u0cbf", tier="klatt")
    assert meta["tier"] == "klatt_zero_dep"
    assert len(pcm) > 2000


def test_neural_onnx_tier1_synthesis_if_present() -> None:
    from pathlib import Path

    onnx_path = Path(__file__).resolve().parent.parent / "docs" / "model_quantized.onnx"
    if not onnx_path.exists():
        return
    engine = Ninada(lang="sa", model_path=onnx_path)
    pcm, meta = engine.synthesize("\u0938\u0924\u094d\u092f\u092e\u0947\u0935 \u091c\u092f\u0924\u0947", tier="neural")
    assert meta["tier"] == "neural_onnx"
    assert len(pcm) > 16000
    wav = engine.speak("\u0938\u0924\u094d\u092f\u092e\u0947\u0935 \u091c\u092f\u0924\u0947")
    assert wav[:4] == b"RIFF"
    assert len(wav) > 32000
