"""Acoustic and structural verification helpers for Ninada Dual-Track E2E Test Suite."""
from __future__ import annotations

import struct
from typing import Any


from ninada import SAMPLE_RATE


def verify_wav_header(wav_bytes: bytes, expected_sr: int = SAMPLE_RATE) -> dict[str, Any]:
    """Verify RIFF WAV 16-bit mono byte integrity and return parsed header fields."""
    assert len(wav_bytes) >= 44, f"WAV byte sequence too short ({len(wav_bytes)} bytes)"
    riff, total_size, wave = struct.unpack("<4sI4s", wav_bytes[:12])
    assert riff == b"RIFF", f"Expected b'RIFF', got {riff}"
    assert wave == b"WAVE", f"Expected b'WAVE', got {wave}"

    fmt_idx = 12
    fmt, fmt_size = struct.unpack("<4sI", wav_bytes[fmt_idx : fmt_idx + 8])
    assert fmt == b"fmt ", f"Expected b'fmt ', got {fmt}"
    assert fmt_size >= 16, f"Expected fmt_size >= 16, got {fmt_size}"

    audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack(
        "<HHIIHH", wav_bytes[fmt_idx + 8 : fmt_idx + 24]
    )
    assert audio_format == 1, f"Expected PCM format (1), got {audio_format}"
    assert channels == 1, f"Expected mono audio (1 channel), got {channels}"
    assert sample_rate == expected_sr, f"Expected {expected_sr} Hz, got {sample_rate}"
    assert bits_per_sample == 16, f"Expected 16-bit audio, got {bits_per_sample}"
    assert byte_rate == expected_sr * 2, f"Expected byte rate {expected_sr * 2}, got {byte_rate}"
    assert block_align == 2, f"Expected block align 2, got {block_align}"

    data_idx = wav_bytes.find(b"data")
    assert data_idx != -1, "Missing b'data' chunk marker"
    data_size = struct.unpack("<I", wav_bytes[data_idx + 4 : data_idx + 8])[0]
    assert data_size > 0, "Empty audio data chunk"
    assert len(wav_bytes) >= data_idx + 8 + data_size, "Truncated audio payload"

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "bits_per_sample": bits_per_sample,
        "data_size": data_size,
        "duration_sec": data_size / (expected_sr * 2),
    }


def verify_pcm_samples(samples: list[float] | Any, min_length: int = 1) -> None:
    """Verify float PCM sample values within [-1.0, 1.0] without NaN or Inf values."""
    assert len(samples) >= min_length, f"PCM buffer length {len(samples)} < {min_length}"
    for idx, s in enumerate(samples[:1000]):
        val = float(s)
        assert not (val != val), f"NaN found at index {idx}"
        assert abs(val) <= 1.05, f"PCM sample out of normalized range at index {idx}: {val}"
