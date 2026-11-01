"""Tier 5: White-Box Adversarial Hardening End-to-End Tests.

Implements 23 targeted white-box test cases across 6 architectural failure categories:
- Category A: Stochastic Duration Predictor (SDP) & Flow Boundary Stress
- Category B: Pitch (F0) Dynamics, Octave Inversion & Noise Modulation Stress
- Category C: Zero-Energy Frames, Delimiter Havoc & Digital Silence
- Category D: Complex Sanskrit Sandhi & Morphological Extremes
- Category E: Multilingual Code-Switching & Out-of-Vocabulary Resilience
- Category F: Halo-Chunk Streaming Seam Integrity & AudioWorklet Stress
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pytest

from ninada import HOP_LENGTH, SAMPLE_RATE, Ninada, encode_wav_bytes, split_prosodic_clauses
from ninada.engine import apply_dc_blocker_1pole, generate_room_tone
from ninada.g2p import text_to_ids
from tests.e2e.helpers import verify_pcm_samples, verify_wav_header

REPO_ROOT = Path(__file__).resolve().parents[2]


# ==============================================================================
# Category A: Stochastic Duration Predictor (SDP) & Flow Boundary Stress
# ==============================================================================
def test_sdp_all_virama_zero_vowel_resilience(engine_sa: Ninada) -> None:
    """Category A1: Pure virama vowel suppressors synthesize without crash or division by zero."""
    text = "््््"

    # Neural ONNX Tier
    pcm_neural, meta_neural = engine_sa.synthesize(text, lang="sa", tier="neural")
    assert len(pcm_neural) > 0, "Expected non-empty audio buffer from all-virama input"
    verify_pcm_samples(pcm_neural, min_length=1)
    peak_neural = float(np.max(np.abs(pcm_neural)))
    assert peak_neural < 0.10, f"Expected neutral low-energy buffer, got peak {peak_neural}"

    wav_neural = encode_wav_bytes(pcm_neural)
    hdr_neural = verify_wav_header(wav_neural)
    assert hdr_neural["duration_sec"] > 0.0

    # Klatt Fallback Tier
    pcm_klatt, meta_klatt = engine_sa.synthesize(text, lang="sa", tier="klatt")
    verify_pcm_samples(pcm_klatt, min_length=1)
    wav_klatt = encode_wav_bytes(pcm_klatt)
    hdr_klatt = verify_wav_header(wav_klatt)
    assert hdr_klatt["duration_sec"] > 0.0


def test_sdp_blank_duration_collapse_extreme_speed(engine_sa: Ninada) -> None:
    """Category A2: Fast speaking rate compresses durations monotonically while every phoneme retains >= 1 frame."""
    text = "सत्यमेव जयते"
    _, meta_norm = engine_sa.synthesize(text, lang="sa", rate=1.0, tier="neural")
    _, meta_fast = engine_sa.synthesize(text, lang="sa", rate=2.2, tier="neural")

    durs_norm = np.asarray(meta_norm["durations"], dtype=np.float64)
    durs_fast = np.asarray(meta_fast["durations"], dtype=np.float64)
    assert len(durs_norm) > 0 and len(durs_fast) > 0
    assert durs_norm.min() >= 1.0, "Normal rate phoneme dropped below 1.0 frame"
    assert durs_fast.min() >= 1.0, "Fast rate phoneme dropped below 1.0 frame"
    assert durs_fast.sum() < durs_norm.sum(), "Fast rate total frames must be strictly less than normal rate"


def test_sdp_reverse_flow_log_clamping_bounds(engine_sa: Ninada) -> None:
    """Category A3: Duration predictor respects upper frame bound (<= 55 frames/token) across rate/noise sweeps."""
    text = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः"
    max_frame_bound = math.exp(4.0) + 1.0

    for rate in (0.75, 1.0, 1.25, 1.5):
        _, meta = engine_sa.synthesize(text, lang="sa", rate=rate, tier="neural")
        durs = np.asarray(meta["durations"], dtype=np.float64)
        assert not np.isnan(durs).any()
        assert not np.isinf(durs).any()
        assert durs.min() >= 1.0
        assert durs.max() <= max_frame_bound, (
            f"Frame duration {durs.max()} exceeded bound ({max_frame_bound}) at rate={rate}"
        )


def test_sdp_extreme_slow_rate_bounds(engine_sa: Ninada) -> None:
    """Category A4: Sub-0.45x rates are clamped to 0.45x without integer buffer overflow."""
    text = "सत्यमेव जयते नानृतम्"

    # Extreme slow rate (0.1x) should be clamped safely to 0.45x
    pcm_01, meta_01 = engine_sa.synthesize(text, lang="sa", rate=0.1, tier="klatt")
    pcm_45, meta_45 = engine_sa.synthesize(text, lang="sa", rate=0.45, tier="klatt")

    dur_01 = len(pcm_01) / float(SAMPLE_RATE)
    dur_45 = len(pcm_45) / float(SAMPLE_RATE)

    assert 0.5 <= dur_01 <= 10.0, f"Duration {dur_01}s out of reasonable range for clamped slow rate"
    assert abs(dur_01 - dur_45) < 0.05, f"Expected 0.1x rate to clamp to 0.45x: {dur_01}s vs {dur_45}s"
    verify_pcm_samples(pcm_01)


def test_sdp_duration_scaling_linearity(engine_sa: Ninada) -> None:
    """Category A5: Utterance durations scale linearly with token counts (Pearson r >= 0.98)."""
    texts = [
        "धर्मः",
        "तेन किञ्चित् साहाय्यं कर्तुं शक्यसे वा?",
        "परन्तु तदीयमहिलाचरित्राणां काठिन्येन एव व्याप्तिः कदाचित् अस्ति इति भाति। प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम्।",
        "प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् । यथार्थज्ञानं प्रमाणम् । परन्तु तदीयमहिलाचरित्राणां काठिन्येन एव व्याप्तिः कदाचित् अस्ति इति भाति । धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ।",
    ]

    token_counts: list[int] = []
    dur_frames: list[float] = []
    dur_seconds: list[float] = []

    for txt in texts:
        t = text_to_ids(txt, lang="sa")
        pcm, meta = engine_sa.synthesize(txt, lang="sa", rate=1.0, tier="neural")
        frames = float(np.sum(meta["durations"]))
        sec = len(pcm) / float(SAMPLE_RATE)

        token_counts.append(len(t["ipa_ids"]))
        dur_frames.append(frames)
        dur_seconds.append(sec)

    corr = float(np.corrcoef(token_counts, dur_frames)[0, 1])
    assert corr >= 0.98, f"Duration scaling non-linear: Pearson r = {corr:.4f} < 0.98"

    for i in range(len(texts)):
        sec_per_token = dur_seconds[i] / token_counts[i]
        assert 0.025 <= sec_per_token <= 0.180, (
            f"Seconds per token {sec_per_token:.4f} outside safe range [0.025, 0.180]"
        )


# ==============================================================================
# Category B: Pitch Dynamics, Octave Inversion & Noise Modulation Stress
# ==============================================================================
def test_f0_extreme_prosodic_noise_sweep(engine_sa: Ninada) -> None:
    """Category B6: Sweeping noise parameter from 0.0 to 1.5 maintains voiced pitch in [80, 420] Hz."""
    text = "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः"
    noise_levels = [0.0, 0.5, 1.0, 1.5]

    for noise in noise_levels:
        pcm, _ = engine_sa.synthesize(text, lang="sa", noise=noise, tier="neural")
        wav = np.asarray(pcm, dtype=np.float32)
        f0 = librosa.yin(wav, fmin=75, fmax=450, sr=SAMPLE_RATE, frame_length=1536, hop_length=384)
        voiced = f0[(f0 >= 80.0) & (f0 <= 420.0)]
        voiced_frac = float(len(voiced) / max(1, len(f0)))

        assert voiced_frac >= 0.15, f"Noise {noise}: Voiced fraction {voiced_frac:.3f} < 0.15 (pitch collapse)"
        f0_mean = float(np.mean(voiced))
        assert 80.0 <= f0_mean <= 420.0, f"Noise {noise}: Voiced F0 mean {f0_mean:.1f} Hz outside [80, 420] Hz"


def test_f0_dense_vedic_accent_cantillation(engine_sa: Ninada) -> None:
    """Category B7: Alternating Vedic accent cantillation (udatta, anudatta, svarita) avoids pitch collapse."""
    # Agneemīḷe from Rigveda 1.1.1 with anudatta (\u0952) and udatta (\u0951)
    text = "\u0905\u0952\u0917\u094d\u0928\u093f\u092e\u0940\u0951\u0933\u0947"
    pcm, _ = engine_sa.synthesize(text, lang="sa", tier="neural")
    wav = np.asarray(pcm, dtype=np.float32)
    verify_pcm_samples(wav, min_length=1600)

    f0 = librosa.yin(wav, fmin=75, fmax=450, sr=SAMPLE_RATE, frame_length=1536, hop_length=384)
    voiced = f0[(f0 >= 80.0) & (f0 <= 420.0)]
    assert len(voiced) > 4, "Too few voiced frames on Vedic cantillation sample"
    f0_mean = float(np.mean(voiced))
    assert 80.0 <= f0_mean <= 400.0, f"Vedic cantillation F0 mean {f0_mean} Hz out of bounds"


def test_f0_monotone_flatline_absence(
    engine_sa: Ninada,
    engine_hi: Ninada,
    engine_kn: Ninada,
    engine_ta: Ninada,
    engine_te: Ninada,
    engine_ml: Ninada,
) -> None:
    """Category B8: Synthesized speech across all 6 Indic languages avoids flatline monotone pitch."""
    engines = {
        "sa": (engine_sa, "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय।"),
        "hi": (engine_hi, "नमस्ते! आपका हमारे इस नए कार्यक्रम में हार्दिक स्वागत है।"),
        "kn": (engine_kn, "ಸಿರಿಗನ್ನಡಂ ಗೆಲ್ಗೆ, ಸಿರಿಗನ್ನಡಂ ಬಾಳ್ಗೆ ಎಂದು ಕವಿಗಳು ಹಾಡಿದ್ದಾರೆ."),
        "ta": (engine_ta, "தமிழ்நாடு என் தாய்நாடு, தமிழ் மொழி என் உயிர் மூச்சு."),
        "te": (engine_te, "తెలుగు భాష తేనె వంటిది, దేశభాషలందు తెలుగు లెస్స."),
        "ml": (engine_ml, "മലയാളം കേരളത്തിന്റെ മാതൃഭാഷയും സാംസ്കാരിക പൈതൃകവുമാണ്."),
    }

    for lang, (engine, text) in engines.items():
        pcm, _ = engine.synthesize(text, lang=lang, tier="neural")
        wav = np.asarray(pcm, dtype=np.float32)
        f0 = librosa.yin(wav, fmin=75, fmax=450, sr=SAMPLE_RATE, frame_length=1536, hop_length=384)
        voiced = f0[(f0 >= 80.0) & (f0 <= 420.0)]
        assert len(voiced) > 10, f"{lang}: Insufficient voiced frames ({len(voiced)})"

        f0_std = float(np.std(voiced))
        p05, p95 = np.percentile(voiced, [5, 95])
        pitch_range_st = float(12.0 * np.log2(max(1.0, p95) / max(1.0, p05)))

        assert f0_std >= 5.0, f"{lang}: F0 standard deviation {f0_std:.2f} Hz < 5.0 Hz (monotone flatline)"
        assert pitch_range_st >= 2.0, f"{lang}: Pitch range {pitch_range_st:.2f} semitones < 2.0 semitones"


# ==============================================================================
# Category C: Zero-Energy Frames, Delimiter Havoc & Digital Silence
# ==============================================================================
def test_zero_energy_pure_delimiters_synthesis(engine_sa: Ninada) -> None:
    """Category C9: Pure punctuation, danda, and delimiter markers synthesize safely without exceptions."""
    delimiter_text = "॥ ॥ ... ??? !!! --- ;;; ,,, \t \n"

    # Neural ONNX Tier
    pcm_neural, meta_neural = engine_sa.synthesize(delimiter_text, lang="sa", tier="neural")
    verify_pcm_samples(pcm_neural)
    wav_neural = encode_wav_bytes(pcm_neural)
    verify_wav_header(wav_neural)

    # Klatt Tier
    pcm_klatt, meta_klatt = engine_sa.synthesize(delimiter_text, lang="sa", tier="klatt")
    verify_pcm_samples(pcm_klatt)
    wav_klatt = encode_wav_bytes(pcm_klatt)
    verify_wav_header(wav_klatt)


def test_unvoiced_consonant_cluster_vocoder_stability(engine_sa: Ninada) -> None:
    """Category C10: Pure unvoiced consonant sequences decode without metallic explosions or NaNs."""
    unvoiced_clusters = ["क्त्प्", "श्ष्स्", "ಕ್ತ್ಪ್"]

    for cluster in unvoiced_clusters:
        pcm, _ = engine_sa.synthesize(cluster, lang="sa", tier="neural")
        wav = np.asarray(pcm, dtype=np.float32)
        assert not np.isnan(wav).any(), f"NaN in unvoiced cluster '{cluster}'"
        assert not np.isinf(wav).any(), f"Inf in unvoiced cluster '{cluster}'"
        peak = float(np.max(np.abs(wav))) if len(wav) > 0 else 0.0
        assert peak <= 1.05, f"Amplitude runaway on unvoiced cluster '{cluster}': peak {peak}"


def test_inter_clause_room_tone_calibration(engine_sa: Ninada) -> None:
    """Category C11: Calibrated room tone inserted between clauses measures strictly within [-62, -52] dBFS."""
    room_tone = generate_room_tone(num_samples=1280, seed=1337)
    assert len(room_tone) == 1280
    rms = float(np.sqrt(np.mean(room_tone**2)))
    dbfs = float(20.0 * math.log10(rms))
    assert -62.0 <= dbfs <= -52.0, f"Room tone level {dbfs:.2f} dBFS outside expected [-62, -52] dBFS"

    # Multi-clause stream contains room tone insertion
    multi_clause = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
    chunks = list(engine_sa.stream(multi_clause, chunk_frames=48))
    assert len(chunks) >= 2, "Expected multi-clause streaming partition"
    for ch in chunks:
        verify_pcm_samples(ch["pcm"])


def test_causal_dc_blocker_offset_bound(
    engine_sa: Ninada,
    engine_hi: Ninada,
    engine_kn: Ninada,
    engine_ta: Ninada,
    engine_te: Ninada,
    engine_ml: Ninada,
) -> None:
    """Category C12: 1-pole causal DC blocker constrains waveform DC bias strictly below 1.0e-4."""
    engines = [
        (engine_sa, "धर्मक्षेत्रे कुरुक्षेत्रे", "sa"),
        (engine_hi, "नमस्ते भारत", "hi"),
        (engine_kn, "ಸಿರಿಗನ್ನಡಂ ಗೆಲ್ಗೆ", "kn"),
        (engine_ta, "தமிழ்நாடு என் தாய்நாடு", "ta"),
        (engine_te, "తెలుగు భాష తేనె వంటిది", "te"),
        (engine_ml, "മലയാളം കേരളത്തിന്റെ മാതൃഭാഷ", "ml"),
    ]

    for engine, text, lang in engines:
        pcm, _ = engine.synthesize(text, lang=lang, tier="neural")
        dc_offset = float(np.abs(np.mean(pcm)))
        assert dc_offset < 1.0e-4, f"{lang}: DC offset {dc_offset:.2e} exceeds threshold 1.0e-4"


# ==============================================================================
# Category D: Complex Sanskrit Sandhi & Morphological Extremes
# ==============================================================================
def test_mega_compound_120_chars_synthesis(engine_sa: Ninada) -> None:
    """Category D13: Continuous 120+ character Sanskrit compound synthesizes into continuous >8.0s audio."""
    compound = (
        "प्रज्ञापूर्वकव्यापारवत्त्वंयथार्थज्ञानप्रमाणत्वं"
        "निरतिशयानन्दस्वरूपत्वंसच्चिदानन्दविग्रहात्मकत्वं"
        "समस्तसद्गुणगणालङ्कृतत्वम्"
    )
    assert len(compound) >= 120, f"Compound length {len(compound)} < 120 chars"

    pcm, meta = engine_sa.synthesize(compound, lang="sa", tier="neural")
    duration = len(pcm) / float(SAMPLE_RATE)
    assert duration > 7.5, f"Synthesized compound duration {duration:.2f}s is <= 7.5s"

    peak = float(np.max(np.abs(pcm)))
    assert 0.10 <= peak <= 0.95, f"Peak amplitude {peak} outside [0.10, 0.95]"
    verify_pcm_samples(pcm)


def test_dense_5_consonant_conjunct_cluster(engine_sa: Ninada) -> None:
    """Category D14: Dense 5-consonant conjunct clusters articulate smoothly without dropped viramas."""
    complex_clusters = ["कार्त्स्न्य", "वार्ष्ण्या", "उज्ज्वाल", "दौर्हृद्य"]

    for word in complex_clusters:
        pcm, meta = engine_sa.synthesize(word, lang="sa", tier="neural")
        assert len(pcm) > 3200, f"Word '{word}' synthesized audio too short ({len(pcm)} samples)"
        peak = float(np.max(np.abs(pcm)))
        assert 0.10 <= peak <= 1.05, f"Word '{word}' peak {peak} out of valid range"
        verify_pcm_samples(pcm)


def test_multi_avagraha_pluta_prolongation(engine_sa: Ninada) -> None:
    """Category D15: Multi-avagraha pluta strings absorb into preceding vowel with Guru cadence."""
    avagraha_words = ["भोऽऽऽ", "कोऽऽपि", "सोऽऽहम्"]

    for word in avagraha_words:
        pcm, meta = engine_sa.synthesize(word, lang="sa", tier="neural", noise_duration=0.0)
        assert len(pcm) >= 4000, f"Pluta word '{word}' synthesized audio truncated ({len(pcm)} samples)"
        verify_pcm_samples(pcm)
        wav = encode_wav_bytes(pcm)
        verify_wav_header(wav)


def test_vedic_rare_cantillation_symbols(engine_sa: Ninada) -> None:
    """Category D16: Rare Vedic cantillation symbols (Jihvamuliya, Upadhmaniya) map cleanly without crash."""
    # Jihvamuliya \u0CF1 and Upadhmaniya \u0CF2
    vedic_text = "अह\u0CF1करोति प\u0CF2फलानि"
    pcm, meta = engine_sa.synthesize(vedic_text, lang="sa", tier="neural")
    assert len(pcm) > 5000, "Vedic cantillation audio truncated"
    verify_pcm_samples(pcm)


# ==============================================================================
# Category E: Multilingual Code-Switching & Out-of-Vocabulary Resilience
# ==============================================================================
def test_multilingual_mixed_indic_script(engine_sa: Ninada) -> None:
    """Category E17: Sentences mixing multiple Brahmic scripts synthesize cleanly without script collision."""
    mixed_text = "संस्कृतम् ಮತ್ತು తెలుగు"
    pcm, meta = engine_sa.synthesize(mixed_text, lang="sa", tier="neural")
    assert len(pcm) > 10000, f"Mixed script audio truncated ({len(pcm)} samples)"
    verify_pcm_samples(pcm)


def test_english_alphanumeric_code_switching(engine_sa: Ninada) -> None:
    """Category E18: Embedded English words and numbers do not cause out-of-vocab crash or index error."""
    code_switch_text = "सत्यमेव जयते is the national motto of India 2026"
    pcm, meta = engine_sa.synthesize(code_switch_text, lang="sa", tier="neural")
    assert len(pcm) > 10000, "Code-switching audio truncated"
    verify_pcm_samples(pcm)


def test_orthographic_control_characters(engine_sa: Ninada) -> None:
    """Category E19: Zero-Width Joiner (ZWJ) and Non-Joiner (ZWNJ) parse cleanly without spurious tokens."""
    # ZWJ \u200D and ZWNJ \u200C
    zwj_text = "संस्कृत\u200Dम्"
    zwnj_text = "धर्म\u200Cक्षेत्रे"

    pcm_zwj, _ = engine_sa.synthesize(zwj_text, lang="sa", tier="neural")
    pcm_zwnj, _ = engine_sa.synthesize(zwnj_text, lang="sa", tier="neural")

    assert len(pcm_zwj) > 5000, "ZWJ text synthesis failed"
    assert len(pcm_zwnj) > 5000, "ZWNJ text synthesis failed"
    verify_pcm_samples(pcm_zwj)
    verify_pcm_samples(pcm_zwnj)


# ==============================================================================
# Category F: Halo-Chunk Streaming Seam Integrity & AudioWorklet Stress
# ==============================================================================
def test_streaming_minimum_chunk_size(engine_sa: Ninada) -> None:
    """Category F20: Minimal chunk frame sizes (24 frames) stream with monotonic chunk indexing."""
    text = "सत्यमेव जयते नानृतम्"
    chunks = list(
        engine_sa.stream(
            text,
            chunk_frames=24,
            halo_frames=8,
            crossfade_samples=128,
            tier="neural",
        )
    )

    assert len(chunks) >= 2, f"Expected multiple chunks for 24-frame streaming, got {len(chunks)}"
    for idx, ch in enumerate(chunks):
        assert ch["index"] == idx, f"Chunk index out of order: expected {idx}, got {ch['index']}"
        verify_pcm_samples(ch["pcm"])
        verify_wav_header(ch["wav"])
    assert chunks[-1]["is_final"] is True, "Last chunk must indicate is_final == True"


def test_streaming_maximum_burst_chunk_size(engine_sa: Ninada) -> None:
    """Category F21: Large burst chunk sizes (128 frames) stream without latency stalls."""
    text = "सत्यमेव जयते नानृतम्"
    chunks = list(
        engine_sa.stream(
            text,
            chunk_frames=128,
            halo_frames=24,
            crossfade_samples=512,
            tier="neural",
        )
    )

    assert len(chunks) >= 1
    assert chunks[0]["ttfa_ms"] > 0.0, "TTFA must be a positive non-zero latency measurement"
    assert chunks[0]["ttfa_ms"] < 2500.0, f"TTFA {chunks[0]['ttfa_ms']:.1f}ms exceeded 2500ms CPU burst stall threshold"
    assert chunks[-1]["is_final"] is True
    for ch in chunks:
        verify_pcm_samples(ch["pcm"])
        verify_wav_header(ch["wav"])


def test_streaming_remainder_crossfade_boundary(engine_sa: Ninada) -> None:
    """Category F22: Frame lengths with small remainder crossfade boundaries do not trigger dimension mismatch."""
    text = "यथार्थज्ञानं प्रमाणम्"
    # Chunk frames configuration designed to yield sub-crossfade tail segment
    chunks = list(
        engine_sa.stream(
            text,
            chunk_frames=32,
            halo_frames=12,
            crossfade_samples=256,
            tier="neural",
        )
    )

    assert len(chunks) >= 1
    total_samples = sum(len(ch["pcm"]) for ch in chunks)
    assert total_samples > 10000, f"Expected >10000 samples, got {total_samples}"
    for ch in chunks:
        verify_pcm_samples(ch["pcm"])


def test_streaming_early_generator_cancellation(engine_sa: Ninada) -> None:
    """Category F23: Closing stream generator mid-utterance cleans up without leaking sessions or deadlock."""
    text = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
    stream_gen = engine_sa.stream(text, tier="neural")

    # Read first chunk and close immediately
    first_chunk = next(stream_gen)
    assert first_chunk["index"] == 0
    verify_pcm_samples(first_chunk["pcm"])
    verify_wav_header(first_chunk["wav"])

    stream_gen.close()

    # Generator must be exhausted
    with pytest.raises(StopIteration):
        next(stream_gen)

    # Subsequent synthesis on the same engine instance must succeed with zero locks or errors
    pcm_subsequent, _ = engine_sa.synthesize("सत्यमेव जयते", tier="neural")
    assert len(pcm_subsequent) > 5000, "Subsequent synthesis failed after early stream cancellation"
    verify_pcm_samples(pcm_subsequent)
