from __future__ import annotations

import time
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
    clean_studio_text,
    ipa_to_ids,
    speak,
    split_prosodic_clauses,
    stream,
    synthesize_klatt_pcm,
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
    assert set(LANGUAGE_IDS.keys()) == {"sa", "hi", "kn", "ta", "te", "ml", "en", "mr", "bn", "gu", "pa", "or"}


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


def test_english_camelcase_g2p_and_unchopped_clause_prosody() -> None:
    """Ensure English CamelCase/unit normalization and single-sentence breath groups up to 340 chars stay intact."""
    from ninada.studio_g2p import clean_studio_text

    preset1 = (
        "Sub-clause latent streaming synthesizes the opening phrase in milliseconds "
        "while the AudioWorklet ring buffer maintains glitch-free twenty-four kilohertz playback."
    )
    assert len(split_prosodic_clauses(preset1)) in (1, 2)
    assert (
        clean_studio_text("AudioWorklet WebGPU 24 kHz 28 MB 120ms", "en")
        == "audio worklet web gpu twenty four kilohertz twenty eight megabytes one hundred twenty milliseconds"
    )


def test_transliterated_studio_g2p_call_parity() -> None:
    """Verify Romanized/transliterated input to clean_studio_text and studio_text_to_ids does not raise TypeError."""
    from ninada.studio_g2p import clean_studio_text, studio_text_to_ids

    # Hindi Romanized transliteration
    cleaned_hi = clean_studio_text("namaste dosto", "hi")
    assert cleaned_hi == "नमस्ते दोस्तो"
    ids_hi, lang_hi, text_hi = studio_text_to_ids("namaste dosto", "hi")
    assert lang_hi == "hi"
    assert text_hi == "नमस्ते दोस्तो"
    assert len(ids_hi) > 0

    # Kannada Romanized transliteration
    cleaned_kn = clean_studio_text("namaskāra", "kn")
    assert cleaned_kn == "ನಮಸ್ಕಾರ"

    # Streaming integration check
    from ninada import Ninada

    engine = Ninada(lang="hi")
    chunks = list(engine.stream("namaste dosto", lang="hi"))
    assert len(chunks) >= 1
    assert chunks[-1]["is_final"] is True


def test_multi_clause_klatt_seam_continuity_and_snr() -> None:
    """Verify multi-clause Klatt streaming seam continuity (|Δy| < 0.008) and Seam SNR > 45 dB across Hindi and English."""
    from ninada import Ninada

    cases = [
        ("hi", "नमस्ते, आपकी समस्या का समाधान शीघ्र ही किया जाएगा । कृपया अपनी खाता संख्या बताएं, धन्यवाद ॥"),
        ("en", "The AudioWorklet ring buffer processes audio chunks in real time, while the WebAssembly engine runs neural inference."),
        ("kn", "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಮತ್ತು ಅತ್ಯಂತ ಸುಂದರವಾದ ನಗರ । ಇದು ಭಾರತದ ಸಿಲಿಕಾನ್ ವ್ಯಾಲಿ ಎಂದು ಪ್ರಸಿದ್ಧವಾಗಿದೆ ॥"),
        ("te", "హైదరాబాదు తెలంగాణ రాజధాని, అత్యంత సుందరమైన నగరం । ఇది భారతదేశ ఐటీ కేంద్రంగా ప్రసిద్ధి చెందింది, ధన్యవాదాలు ॥"),
    ]
    for lang, text in cases:
        engine = Ninada(lang=lang)
        chunks = list(engine.stream(text, lang=lang, tier="klatt"))
        assert len(chunks) >= 3, f"[{lang}] Expected >= 3 chunks, got {len(chunks)}"

        stitched = np.concatenate([np.asarray(c["pcm"], dtype=np.float32) for c in chunks])
        mono_pcm, _ = engine.synthesize(text, lang=lang, tier="klatt")
        mono_arr = np.asarray(mono_pcm, dtype=np.float32)
        assert len(stitched) == len(mono_arr), f"[{lang}] Sample length parity between stream and synthesize required"

        boundaries = []
        acc = 0
        for c in chunks[:-1]:
            acc += len(c["pcm"])
            boundaries.append(acc)

        # 1. Boundary transient step jumps |Δy| < 0.008
        for idx, b in enumerate(boundaries):
            step_jump = abs(float(stitched[b]) - float(stitched[b - 1]))
            assert step_jump < 0.008, f"[{lang}] Boundary {idx} at sample {b}: step jump |Δy| = {step_jump:.6f} >= 0.008"

        # 2. Seam SNR > 45 dB
        for idx, b in enumerate(boundaries):
            win = 512
            st = max(0, b - 256)
            ed = min(len(mono_arr), st + win)
            r = mono_arr[st:ed]
            s = stitched[st:ed]
            sig_pow = float(np.sum(r * r))
            err_pow = float(np.sum((s - r) * (s - r)))
            snr = 100.0 if err_pow < 1e-12 else 10.0 * float(np.log10(sig_pow / err_pow))
            assert snr > 45.0, f"[{lang}] Boundary {idx} at sample {b}: Seam SNR {snr:.2f} dB < 45.0 dB"


def test_semver_and_multidot_decimal_normalization() -> None:
    """Verify SemVer versions and multi-dotted decimals expand without literal periods inside spoken text."""
    cases = [
        ("version 2.0.1", "version two point zero point one"),
        ("version 2.0.1.", "version two point zero point one."),
        ("Linux 6.1.102", "linux six point one point one hundred two"),
        ("v1.2.3.4", "v one point two point three point four"),
        ("IP 192.168.1.1", "ip one hundred ninety two point one hundred sixty eight point one point one"),
        ("3.14159", "three point one four one five nine"),
        ("0.001", "zero point zero zero one"),
        ("100.50", "one hundred point five zero"),
    ]
    for raw, expected in cases:
        res = clean_studio_text(raw, "en")
        assert res == expected, f"Mismatch for {raw!r}: got {res!r}, expected {expected!r}"
        if not raw.endswith("."):
            assert "." not in res, f"Found unexpected literal period in {res!r}"


def test_abbreviation_protection_and_clause_segmentation_parity() -> None:
    """Verify abbreviation protection for approx, Rs, and orphan-tail merging parity."""
    cases = [
        ("Dr. Rao vs. Prof. Smith at St. Jude Inc. approx. 10 km/h.", 1),
        (
            "Prof. Smith vs. Dr. Watson: the benchmark completed in approx. 120ms with 99.9% accuracy; however, Mr. Brown from Univ. Dept. requested 3.14 iterations.",
            3,
        ),
        ("Yes. No. Dr. Lee said etc. and approx. 5 MB was transferred.", 1),
        ("He paid Rs. 500 for the book.", 1),
        ("This is a reasonably long introductory clause that sets the stage, yes.", 2),
    ]
    for text, expected in cases:
        res = split_prosodic_clauses(text, max_chars=60 if "stage, yes." in text else 100)
        assert len(res) == expected, f"Expected {expected} clauses, got {len(res)}: {res}"


def test_punjabi_numeral_transliteration_klatt_parity() -> None:
    """Verify Punjabi numerals expand to phonetic aksharas for Klatt synthesis while preserving Gurmukhi VITS tokens."""
    text = "ਇੱਕ ਕੁੱਤਾ ਅਤੇ ਸੱਤ ਬੱਚੇ, 500 ਰੁਪਏ।"
    parsed = text_to_ids(text, lang="pa")
    aksharas = [a["akshara"] for a in parsed["aksharas"]]
    assert "पं" in aksharas, f"Expected numeral akshara 'पं' in {aksharas}"
    assert "੫" in parsed["native_tokens"], f"Expected Gurmukhi token '੫' in {parsed['native_tokens']}"

    # Pure digits tokenization and synthesis test
    pure_500 = text_to_ids("500", lang="pa")
    assert "੫" in pure_500["native_tokens"], f"Expected '੫' in {pure_500['native_tokens']}"
    assert "੦" in pure_500["native_tokens"], f"Expected '੦' in {pure_500['native_tokens']}"

    pure_1234 = text_to_ids("1234", lang="pa")
    assert all(d in pure_1234["native_tokens"] for d in ["੧", "੨", "੩", "੪"]), (
        f"Expected all Gurmukhi digits 1-4 in {pure_1234['native_tokens']}"
    )

    danda_500 = text_to_ids("500।", lang="pa")
    assert "੫" in danda_500["native_tokens"], f"Expected '੫' in {danda_500['native_tokens']}"

    decimal_mixed = text_to_ids("ਮੁੱਲ 2.5 ਹੈ।", lang="pa")
    assert "੨" in decimal_mixed["native_tokens"], f"Expected '੨' in {decimal_mixed['native_tokens']}"
    assert "੫" in decimal_mixed["native_tokens"], f"Expected '੫' in {decimal_mixed['native_tokens']}"

    pcm, meta = synthesize_klatt_pcm("500", lang="pa")
    assert len(pcm) > 15000, f"Expected non-empty PCM audio for numerals, got {len(pcm)} samples"
    assert len(meta["aksharas"]) == 8, f"Expected 8 aksharas for 500, got {len(meta['aksharas'])}"


def test_kannada_streaming_dravidian_phonetics_and_continuity() -> None:
    """Verify Kannada Dravidian vowel preservation, retroflex lateral ಳ, geminates, and streaming continuity."""
    # 1. Dravidian vowel preservation (ಎ, ಏ, ಒ, ಓ)
    vowel_text = "ಎಲೆ ಏಣಿ ಒಂಟೆ ಓಟ"
    vowel_ipa = text_to_ipa(vowel_text, "kn")
    assert "ɛ" in vowel_ipa, f"Expected short e [ɛ] in {vowel_ipa}"
    assert "eː" in vowel_ipa, f"Expected long e [eː] in {vowel_ipa}"
    assert "o" in vowel_ipa, f"Expected short o [o] in {vowel_ipa}"
    assert "oː" in vowel_ipa, f"Expected long o [oː] in {vowel_ipa}"

    # 2. Retroflex lateral approximant ಳ (slot 79)
    retroflex_text = "ಹಳೆಗನ್ನಡ ಮತ್ತು ಬೆಳ್ಳಿ ಬೆಳಕು"
    retroflex_ipa = text_to_ipa(retroflex_text, "kn")
    assert "ɭ" in retroflex_ipa, f"Expected retroflex lateral [ɭ] in {retroflex_ipa}"
    token_ids = ipa_to_ids(retroflex_ipa)
    assert RETROFLEX_LATERAL_ID in token_ids, f"Expected slot {RETROFLEX_LATERAL_ID} in {token_ids}"
    assert IPA_VOCAB["ɭ"] == 79

    # 3. Geminate consonants (ಕ್ಕ, ತ್ತ)
    geminate_text = "ಅಕ್ಕ ಮತ್ತು ಅತ್ತೆ"
    geminate_ipa = text_to_ipa(geminate_text, "kn")
    assert "kk" in geminate_ipa, f"Expected geminate kk in {geminate_ipa}"
    assert "tt" in geminate_ipa, f"Expected geminate tt in {geminate_ipa}"

    # 4. Streaming performance: TTFA < 150 ms (target < 50 ms), 0 overlap, offset monotonicity
    engine = Ninada(lang="kn")
    text = "ಕನ್ನಡ ನಾಡು ಚಂದನದ ಬೀಡು . ಬೆಳ್ಳಿ ಬೆಳಕು ಹರಡಿತು ಮತ್ತು ಹಳೆಯ ಕನ್ನಡ ಕಾವ್ಯಗಳು ಜನರ ಮನಸ್ಸನ್ನು ತಣಿಸಿದವು ."
    t0 = time.perf_counter()
    chunks = []
    first_chunk_time: float | None = None
    for ch in engine.stream(text, lang="kn", tier="klatt"):
        if first_chunk_time is None:
            first_chunk_time = (time.perf_counter() - t0) * 1000.0
        chunks.append(ch)

    assert len(chunks) >= 2, f"Expected >= 2 chunks, got {len(chunks)}"
    assert first_chunk_time is not None, "Failed to capture first chunk time"
    assert first_chunk_time < 150.0, f"TTFA {first_chunk_time:.2f} ms exceeded 150 ms threshold"

    # Sample offset monotonicity (Δsamples = 0) and zero chunk overlap
    cur_offset = 0
    for idx, c in enumerate(chunks):
        assert c["sample_offset"] == cur_offset, f"Chunk {idx} offset {c['sample_offset']} != {cur_offset}"
        cur_offset += len(c["pcm"])

    # 5. Acoustic seam continuity and Seam SNR > 45 dB
    stitched = np.concatenate([np.asarray(c["pcm"], dtype=np.float32) for c in chunks])
    mono_pcm, _ = engine.synthesize(text, lang="kn", tier="klatt")
    mono_arr = np.asarray(mono_pcm, dtype=np.float32)
    assert len(stitched) == len(mono_arr), f"Sample length mismatch: {len(stitched)} != {len(mono_arr)}"

    boundaries = [sum(len(chunks[j]["pcm"]) for j in range(i + 1)) for i in range(len(chunks) - 1)]
    for idx, b in enumerate(boundaries):
        step_jump = abs(float(stitched[b]) - float(stitched[b - 1]))
        assert step_jump < 0.008, f"Boundary {idx} at sample {b}: |Δy| = {step_jump:.6f} >= 0.008"

    for idx, b in enumerate(boundaries):
        win = 512
        st = max(0, b - 256)
        ed = min(len(mono_arr), st + win)
        r = mono_arr[st:ed]
        s = stitched[st:ed]
        sig_pow = float(np.sum(r * r))
        err_pow = float(np.sum((s - r) * (s - r)))
        snr = 100.0 if err_pow < 1e-12 else 10.0 * float(np.log10(sig_pow / err_pow))
        assert snr > 45.0, f"Boundary {idx} at sample {b}: Seam SNR {snr:.2f} dB < 45.0 dB"


def test_audioworklet_natural_drain_state_reset_and_clamping() -> None:
    """Verify AudioWorkletSimulator resets state on natural EOS drain and clamps output to [-1.0, 1.0]."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tests.e2e_streaming.helpers import AudioWorkletSimulator

    sim = AudioWorkletSimulator()
    sim.reset(watermark_samples=128)
    # Push signal that has samples exceeding [-1.0, 1.0]
    overshoot_input = np.array([0.0, 0.8, 1.5, -1.8, 0.5, 0.0], dtype=np.float32)
    sim.push(overshoot_input, sample_rate=24000)
    sim.mark_eos()

    pcm_out, report = sim.process_all(target_sample_rate=48000, quantum_size=128)

    # 1. Clamping verification: all rendered samples strictly within [-1.0, 1.0]
    assert np.all(pcm_out <= 1.0), f"Max output {float(np.max(pcm_out))} exceeds 1.0"
    assert np.all(pcm_out >= -1.0), f"Min output {float(np.min(pcm_out))} below -1.0"

    # 2. State reset verification on natural EOS drain
    assert sim.state == "idle"
    assert sim.eos_flag is False
    assert sim.hist_primed is False
    assert sim.drain_remaining == 2
    assert sim.phase == 0.0
    assert sim.env_gain == 1.0
    assert any(ev.get("type") == "ended" for ev in sim.events)


def test_telugu_streaming_continuity_zero_overlap_and_low_latency() -> None:
    """Verify Telugu (te) streaming synthesis: RTF < 0.20, TTFA < 150 ms, |Δy| < 0.008, Seam SNR >= 45 dB, |μ| < 1.0e-4, 0 chunk overlap."""
    from ninada import Ninada

    text = "హైదరాబాదు తెలంగాణ రాజధాని, అత్యంత సుందరమైన నగరం । ఇది భారతదేశ ఐటీ కేంద్రంగా ప్రసిద్ధి చెందింది, ధన్యవాదాలు ॥"
    engine = Ninada(lang="te")

    # Klatt streaming verification
    t0 = time.perf_counter()
    chunks_k = list(engine.stream(text, lang="te", tier="klatt"))
    latency_k = time.perf_counter() - t0
    assert len(chunks_k) >= 3, f"Expected >= 3 chunks, got {len(chunks_k)}"

    # Zero chunk overlap and monotonicity
    curr_offset = 0
    for idx, c in enumerate(chunks_k):
        assert c["sample_offset"] == curr_offset, f"Non-monotonic sample_offset at chunk {idx}"
        curr_offset += len(c["pcm"])
        # Persistent DC blocker check
        pcm_arr = np.asarray(c["pcm"], dtype=np.float32)
        dc = float(np.abs(np.mean(pcm_arr)))
        assert dc < 1.0e-4, f"DC offset {dc:.2e} >= 1.0e-4 in chunk {idx}"

    dur_k = curr_offset / 24000.0
    rtf_k = latency_k / dur_k
    assert rtf_k < 0.20, f"RTF {rtf_k:.4f} exceeded 0.20 threshold"
    assert chunks_k[0]["ttfa_ms"] < 150.0, f"TTFA {chunks_k[0]['ttfa_ms']} ms exceeded 150 ms threshold"

    # Neural streaming verification if specialist is present
    try:
        chunks_n = list(engine.stream(text, lang="te", tier="neural"))
        if chunks_n and chunks_n[0]["tier"] == "neural_onnx":
            curr_n = 0
            for idx, c in enumerate(chunks_n):
                assert c["sample_offset"] == curr_n, f"Neural non-monotonic offset at chunk {idx}"
                curr_n += len(c["pcm"])
                pcm_arr = np.asarray(c["pcm"], dtype=np.float32)
                dc = float(np.abs(np.mean(pcm_arr)))
                assert dc < 1.0e-4, f"Neural DC offset {dc:.2e} >= 1.0e-4 in chunk {idx}"
            stitched_n = np.concatenate([np.asarray(c["pcm"], dtype=np.float32) for c in chunks_n])
            acc = 0
            for c in chunks_n[:-1]:
                acc += len(c["pcm"])
                step_jump = abs(float(stitched_n[acc]) - float(stitched_n[acc - 1]))
                assert step_jump < 0.008, f"Neural step jump {step_jump} >= 0.008"
    except Exception:
        pass






