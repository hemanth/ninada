"""Zero-dependency Klatt-Akshara Formant Synthesizer + Neural ONNX runtime engine for Ninada."""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

from .g2p import LANGUAGE_IDS, text_to_ids

SAMPLE_RATE = 16000

VOWEL_FORMANTS: dict[str, tuple[float, float, float, float]] = {
    "\u0905": (640.0, 1240.0, 2550.0, 0.085),
    "\u0906": (760.0, 1180.0, 2520.0, 0.155),
    "\u093e": (760.0, 1180.0, 2520.0, 0.155),
    "\u0907": (310.0, 2240.0, 2920.0, 0.085),
    "\u093f": (310.0, 2240.0, 2920.0, 0.085),
    "\u0908": (285.0, 2340.0, 3000.0, 0.155),
    "\u0940": (285.0, 2340.0, 3000.0, 0.155),
    "\u0909": (340.0, 860.0, 2420.0, 0.085),
    "\u0941": (340.0, 860.0, 2420.0, 0.085),
    "\u090a": (310.0, 780.0, 2380.0, 0.155),
    "\u0942": (310.0, 780.0, 2380.0, 0.155),
    "\u090b": (440.0, 1480.0, 1950.0, 0.105),
    "\u0943": (440.0, 1480.0, 1950.0, 0.105),
    "\u0960": (440.0, 1480.0, 1950.0, 0.165),
    "\u0944": (440.0, 1480.0, 1950.0, 0.165),
    "\u090e": (460.0, 1960.0, 2720.0, 0.090),
    "\u0946": (460.0, 1960.0, 2720.0, 0.090),
    "\u090f": (410.0, 2080.0, 2780.0, 0.150),
    "\u0947": (410.0, 2080.0, 2780.0, 0.150),
    "\u0910": (580.0, 1780.0, 2700.0, 0.165),
    "\u0948": (580.0, 1780.0, 2700.0, 0.165),
    "\u0912": (480.0, 940.0, 2480.0, 0.090),
    "\u094a": (480.0, 940.0, 2480.0, 0.090),
    "\u0913": (430.0, 860.0, 2450.0, 0.150),
    "\u094b": (430.0, 860.0, 2450.0, 0.150),
    "\u0914": (600.0, 1020.0, 2480.0, 0.165),
    "\u094c": (600.0, 1020.0, 2480.0, 0.165),
}


def _biquad_resonator(samples: list[float], freq: float, bw: float, sr: int = SAMPLE_RATE) -> list[float]:
    r = math.exp(-math.pi * bw / sr)
    c = -(r * r)
    b = 2.0 * r * math.cos(2.0 * math.pi * freq / sr)
    a = 1.0 - b - c
    y1 = 0.0
    y2 = 0.0
    out = [0.0] * len(samples)
    for i, x in enumerate(samples):
        y = a * x + b * y1 + c * y2
        y2 = y1
        y1 = y
        out[i] = y
    return out


def synthesize_klatt_pcm(
    text: str,
    lang: str | None = None,
    rate: float = 1.0,
    f0_base: float = 132.0,
) -> tuple[list[float], dict[str, Any]]:
    """Tier-2 Zero-Dependency Cascade Klatt-Akshara Synthesizer (16 kHz float PCM)."""
    parsed = text_to_ids(text, lang=lang)
    aksharas = parsed["aksharas"]
    speed = max(0.45, min(2.5, float(rate)))

    pcm: list[float] = []
    phase = 0.0
    total_ak = max(1, len(aksharas))

    seed = 20261030

    def rand_u() -> float:
        nonlocal seed
        seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
        return (seed / 2147483648.0) - 1.0

    for idx, ak in enumerate(aksharas):
        s = ak["akshara"]
        weight = ak["weight"]
        decl = 1.0 - 0.10 * (idx / total_ak)
        accent = 1.07 if weight == "G" else (0.96 if weight == "C" else 1.0)
        f0 = f0_base * decl * accent

        f1, f2, f3, dur_v = VOWEL_FORMANTS["\u0905"]
        if s.endswith("\u094d"):
            dur_v = 0.032
        else:
            for ch in s:
                if ch in VOWEL_FORMANTS:
                    f1, f2, f3, dur_v = VOWEL_FORMANTS[ch]

        first_cp = ord(s[0])
        is_sibilant = s[0] in ("\u0936", "\u0937", "\u0938")
        is_aspirated = s[0] in "\u0916\u0918\u091b\u091d\u0920\u0922\u0925\u0927\u092b\u092d\u0939"
        is_nasal = s[0] in "\u0919\u091e\u0923\u0928\u092e\u0929" or "\u0902" in s
        has_visarga = "\u0903" in s
        is_retroflex = (0x091F <= first_cp <= 0x0923) or s[0] in "\u0937\u0933\u0934"
        if is_retroflex:
            f3 = min(f3, 1980.0)

        onset_sec = (0.048 if is_sibilant else (0.036 if is_aspirated else 0.022)) / speed
        nucleus_sec = (dur_v + (0.035 if has_visarga else 0.0)) / speed
        n_onset = max(16, int(round(onset_sec * SAMPLE_RATE)))
        n_nuc = max(32, int(round(nucleus_sec * SAMPLE_RATE)))
        n_tot = n_onset + n_nuc

        exc = [0.0] * n_tot
        for t in range(n_tot):
            if t < n_onset:
                env = math.sin(math.pi * (t / max(1, n_onset)))
                if is_sibilant:
                    exc[t] = 0.42 * env * rand_u()
                elif is_aspirated:
                    exc[t] = 0.30 * env * rand_u()
                elif is_nasal:
                    phase = (phase + f0 / SAMPLE_RATE) % 1.0
                    exc[t] = 0.45 * math.sin(2.0 * math.pi * phase)
                else:
                    exc[t] = 0.08 * env * rand_u()
            else:
                u = (t - n_onset) / max(1, n_nuc)
                env = math.sin(math.pi * min(1.0, u * 0.92 + 0.05))
                phase = (phase + f0 / SAMPLE_RATE) % 1.0
                glot = (
                    0.5 * (1.0 - math.cos(math.pi * phase / 0.42))
                    if phase < 0.42
                    else math.cos(math.pi * (phase - 0.42) / (2.0 * 0.18))
                    if phase < 0.60
                    else 0.0
                )
                breath = (0.28 * rand_u()) if (has_visarga and u > 0.72) else (0.03 * rand_u())
                exc[t] = env * (0.78 * glot + breath)

        r1 = _biquad_resonator(exc, 280.0 if is_nasal else f1, 95.0)
        r2 = _biquad_resonator(r1, f2, 130.0)
        r3 = _biquad_resonator(r2, f3, 185.0)
        pcm.extend(r3)
        pcm.extend([0.0] * int(0.010 * SAMPLE_RATE / speed))

    if not pcm:
        pcm = [0.0] * 1600
    peak = max(1e-6, max(abs(x) for x in pcm))
    pcm = [0.82 * (x / peak) for x in pcm]
    parsed["tier"] = "klatt_zero_dep"
    return pcm, parsed


def encode_wav_bytes(pcm: Any, sr: int = SAMPLE_RATE) -> bytes:
    """Encode float32 [-1, 1] iterable/array into standard 16-bit mono RIFF WAV bytes."""
    buf = bytearray()
    for s in pcm:
        v = max(-1.0, min(1.0, float(s)))
        buf.extend(struct.pack("<h", int(round(v * 32767.0))))
    pcm16 = bytes(buf)
    byte_rate = sr * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm16),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sr,
        byte_rate,
        2,
        16,
        b"data",
        len(pcm16),
    )
    return header + pcm16


class Ninada:
    """Ninada (निनाद): Pan-Indic & Classical Sanskrit TTS engine (Neural VITS ONNX Tier-1 + Zero-Dep Klatt Tier-2)."""

    def __init__(self, model_path: str | Path | None = None, lang: str = "sa") -> None:
        self.default_lang = lang if lang in LANGUAGE_IDS else "sa"
        self.session: Any = None
        self.model_path: Path | None = None

        if model_path is None:
            repo_root = Path(__file__).resolve().parents[3]
            candidates = [
                repo_root / "artifacts" / "ninada" / "onnx" / "model_quantized.onnx",
                repo_root / "docs" / "model_quantized.onnx",
            ]
            for cand in candidates:
                if cand.exists():
                    model_path = cand
                    break
            if model_path is None:
                try:
                    from huggingface_hub import hf_hub_download

                    model_path = Path(
                        hf_hub_download(repo_id="gnumanth/ninada", filename="onnx/model_quantized.onnx")
                    )
                except Exception:
                    model_path = None

        if model_path is not None and Path(model_path).exists():
            try:
                import onnxruntime as ort

                self.model_path = Path(model_path)
                self.session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
            except Exception:
                self.session = None

    def with_lang(self, lang: str) -> "Ninada":
        if lang in LANGUAGE_IDS:
            self.default_lang = lang
        return self

    def speak(
        self,
        text: str,
        lang: str | None = None,
        rate: float = 1.0,
        noise: float = 0.32,
        tier: str = "auto",
    ) -> bytes:
        """Synthesize text into 16 kHz mono WAV bytes."""
        pcm, _ = self.synthesize(text=text, lang=lang, rate=rate, noise=noise, tier=tier)
        return encode_wav_bytes(pcm, sr=SAMPLE_RATE)

    def synthesize(
        self,
        text: str,
        lang: str | None = None,
        rate: float = 1.0,
        noise: float = 0.32,
        tier: str = "auto",
    ) -> tuple[Any, dict[str, Any]]:
        """Synthesize waveform array + linguistic metadata."""
        use_lang = lang or self.default_lang
        if tier != "klatt" and self.session is not None:
            import numpy as np

            parsed = text_to_ids(text, lang=use_lang)
            ids = np.array([parsed["input_ids"]], dtype=np.int64)
            mask = np.array([parsed["attention_mask"]], dtype=np.int64)
            lid = np.array([parsed["lang_id"]], dtype=np.int64)
            scales = np.array([float(rate), float(noise)], dtype=np.float32)
            wav, durs = self.session.run(
                None,
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "lang_id": lid,
                    "scales": scales,
                },
            )
            waveform = wav[0].astype(np.float32)
            peak = float(np.max(np.abs(waveform)))
            if peak > 1e-4:
                waveform = 0.88 * (waveform / max(1.0, peak))
            parsed["tier"] = "neural_onnx"
            parsed["durations"] = durs[0].tolist()
            return waveform, parsed

        pcm, parsed = synthesize_klatt_pcm(text, lang=use_lang, rate=rate)
        parsed["tier"] = "klatt_zero_dep"
        return pcm, parsed


def speak(text: str, lang: str = "sa", rate: float = 1.0, tier: str = "auto") -> bytes:
    """One-line convenience helper to synthesize 16 kHz RIFF WAV bytes."""
    return Ninada(lang=lang).speak(text, lang=lang, rate=rate, tier=tier)
