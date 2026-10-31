"""Streaming Halo-Chunked Specialist Neural ONNX + Zero-Dep Klatt-Akshara Runtime Engine for Ninada v2.0."""
from __future__ import annotations

from collections.abc import Iterator
import math
import re
import struct
import time
from pathlib import Path
from typing import Any

from .g2p import LANGUAGE_IDS, text_to_ids

SAMPLE_RATE = 16000
HOP_LENGTH = 256

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


def split_prosodic_clauses(text: str, max_chars: int = 160) -> list[str]:
    """Split arbitrarily long text into prosodic breath-group clauses (danda, punctuation, or word boundary)."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return []
    raw_parts = re.split(r"(?<=\|\|)\s+|(?<=[\u0964\u0965|.?!;\n])\s+", cleaned)
    clauses: list[str] = []
    for part in raw_parts:
        p = part.strip()
        if not p:
            continue
        if re.fullmatch(r"[\u0964\u0965|.?!;,\s\-]+", p):
            if clauses:
                clauses[-1] = f"{clauses[-1]} {p}"
            continue
        if len(p) <= max_chars:
            clauses.append(p)
            continue
        words = p.split(" ")
        buf = ""
        for w in words:
            if not buf:
                buf = w
            elif len(buf) + 1 + len(w) <= max_chars:
                buf = f"{buf} {w}"
            else:
                clauses.append(buf)
                buf = w
        if buf:
            clauses.append(buf)
    return clauses or [cleaned]


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
    """Ninada (निनाद) v2.0: Streaming Pan-Indic & Classical Sanskrit TTS (Per-Language Specialist ONNX + Klatt Tier-2)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        encoder_path: str | Path | None = None,
        vocoder_path: str | Path | None = None,
        lang: str = "sa",
    ) -> None:
        self.default_lang = lang if lang in LANGUAGE_IDS else "sa"
        self.session: Any = None
        self.enc_session: Any = None
        self.voc_session: Any = None
        self.model_path: Path | None = None
        self._explicit_override = bool(model_path or encoder_path or vocoder_path)
        self._lang_sessions: dict[str, tuple[Any, Any]] = {}
        self._repo_root = Path(__file__).resolve().parents[3]

        if model_path is not None and Path(model_path).is_dir():
            d = Path(model_path)
            if (d / self.default_lang / "encoder_flow_quantized.onnx").exists():
                self._explicit_override = False
                self._onnx_base_dir: Path | None = d
            elif (d / "encoder_flow_quantized.onnx").exists() and (d / "vocoder_quantized.onnx").exists():
                encoder_path = d / "encoder_flow_quantized.onnx"
                vocoder_path = d / "vocoder_quantized.onnx"
                model_path = None
                self._onnx_base_dir = None
            elif (d / "model_quantized.onnx").exists():
                model_path = d / "model_quantized.onnx"
                self._onnx_base_dir = None
            else:
                self._onnx_base_dir = None
        else:
            self._onnx_base_dir = None

        if self._explicit_override:
            try:
                import onnxruntime as ort

                if encoder_path and vocoder_path and Path(encoder_path).exists() and Path(vocoder_path).exists():
                    self.enc_session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
                    self.voc_session = ort.InferenceSession(str(vocoder_path), providers=["CPUExecutionProvider"])
                    self.model_path = Path(encoder_path)
                elif model_path is not None and Path(model_path).exists():
                    self.model_path = Path(model_path)
                    self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            except Exception:
                self.session = None
                self.enc_session = None
                self.voc_session = None
        else:
            pair = self._get_specialist_sessions(self.default_lang)
            if pair is not None:
                self.enc_session, self.voc_session = pair

    def _get_specialist_sessions(self, lang: str) -> tuple[Any, Any] | None:
        lc = lang if lang in LANGUAGE_IDS else "sa"
        if lc in self._lang_sessions:
            return self._lang_sessions[lc]

        enc_p: Path | None = None
        voc_p: Path | None = None

        search_dirs: list[Path] = []
        if self._onnx_base_dir is not None:
            search_dirs.append(self._onnx_base_dir / lc)
            search_dirs.append(self._onnx_base_dir)
        search_dirs.append(self._repo_root / "artifacts" / "ninada" / "onnx" / lc)
        if lc == "sa":
            search_dirs.append(self._repo_root / "artifacts" / "ninada" / "onnx")
            search_dirs.append(self._repo_root / "docs")

        for d in search_dirs:
            ef = d / "encoder_flow_quantized.onnx"
            vf = d / "vocoder_quantized.onnx"
            if ef.exists() and vf.exists():
                enc_p, voc_p = ef, vf
                break

        if enc_p is None:
            try:
                from huggingface_hub import hf_hub_download

                enc_p = Path(
                    hf_hub_download(repo_id="gnumanth/ninada", filename=f"onnx/{lc}/encoder_flow_quantized.onnx")
                )
                voc_p = Path(
                    hf_hub_download(repo_id="gnumanth/ninada", filename=f"onnx/{lc}/vocoder_quantized.onnx")
                )
            except Exception:
                if lc == "sa":
                    try:
                        from huggingface_hub import hf_hub_download

                        enc_p = Path(
                            hf_hub_download(repo_id="gnumanth/ninada", filename="onnx/encoder_flow_quantized.onnx")
                        )
                        voc_p = Path(
                            hf_hub_download(repo_id="gnumanth/ninada", filename="onnx/vocoder_quantized.onnx")
                        )
                    except Exception:
                        enc_p = None

        if enc_p and voc_p and enc_p.exists() and voc_p.exists():
            try:
                import onnxruntime as ort

                es = ort.InferenceSession(str(enc_p), providers=["CPUExecutionProvider"])
                vs = ort.InferenceSession(str(voc_p), providers=["CPUExecutionProvider"])
                self._lang_sessions[lc] = (es, vs)
                if self.model_path is None:
                    self.model_path = enc_p
                return es, vs
            except Exception:
                return None
        return None

    @property
    def is_neural(self) -> bool:
        return (
            bool(self._lang_sessions)
            or (self.enc_session is not None and self.voc_session is not None)
            or (self.session is not None)
        )

    def with_lang(self, lang: str) -> "Ninada":
        if lang in LANGUAGE_IDS:
            self.default_lang = lang
        return self

    def stream(
        self,
        text: str,
        lang: str | None = None,
        rate: float = 1.0,
        noise: float = 0.667,
        noise_duration: float = 0.80,
        tier: str = "auto",
        chunk_frames: int = 64,
        halo_frames: int = 12,
        crossfade_samples: int = 256,
    ) -> Iterator[dict[str, Any]]:
        """Stream audio chunks (<200 ms TTFA) via 12-frame halo + 256-sample raised-cosine crossfaded HiFi-GAN."""
        use_lang = lang if (lang in LANGUAGE_IDS) else self.default_lang
        clauses = split_prosodic_clauses(text)
        t_start = time.perf_counter()
        ttfa_ms: float | None = None
        chunk_idx = 0

        enc_sess = self.enc_session
        voc_sess = self.voc_session
        if not self._explicit_override and tier != "klatt":
            pair = self._get_specialist_sessions(use_lang)
            if pair is not None:
                enc_sess, voc_sess = pair

        if tier != "klatt" and enc_sess is not None and voc_sess is not None:
            import numpy as np

            chunk_f = max(24, int(chunk_frames))
            halo_f = max(8, int(halo_frames))
            xfade = max(0, min(int(crossfade_samples), halo_f * HOP_LENGTH // 2))

            enc_in_names = {i.name for i in enc_sess.get_inputs()}
            voc_in_names = {i.name for i in voc_sess.get_inputs()}

            fade_in = (
                0.5 * (1.0 - np.cos(math.pi * np.arange(xfade, dtype=np.float32) / max(1, xfade)))
                if xfade > 0
                else np.zeros(0, dtype=np.float32)
            )
            fade_out = 1.0 - fade_in

            global_ord = 0
            for c_idx, clause in enumerate(clauses):
                parsed = text_to_ids(clause, lang=use_lang)
                ort_in: dict[str, Any] = {
                    "input_ids": np.array([parsed["input_ids"]], dtype=np.int64),
                    "attention_mask": np.array([parsed["attention_mask"]], dtype=np.int64),
                }
                if "lang_id" in enc_in_names:
                    ort_in["lang_id"] = np.array([parsed["lang_id"]], dtype=np.int64)
                    ort_in["scales"] = np.array([float(rate), float(noise)], dtype=np.float32)
                else:
                    ort_in["scales"] = np.array(
                        [float(rate), float(noise), float(noise_duration)], dtype=np.float32
                    )

                enc_outs = enc_sess.run(None, ort_in)
                if len(enc_outs) == 3:
                    latents, voice_cond, durs = enc_outs
                else:
                    latents, durs = enc_outs
                    voice_cond = None

                total_frames = int(latents.shape[2])
                bounds: list[tuple[int, int]] = []
                cur = 0
                while cur < total_frames:
                    step = min(chunk_f, 48) if global_ord == 0 else chunk_f
                    nxt = min(total_frames, cur + step)
                    if total_frames - nxt < 32:
                        nxt = total_frames
                    bounds.append((cur, nxt))
                    cur = nxt
                    global_ord += 1

                num_sub = max(1, len(bounds))
                prev_tail: np.ndarray | None = None

                for sub_i, (st, ed) in enumerate(bounds):
                    is_last_sub = sub_i == num_sub - 1
                    w_st = max(0, st - halo_f)
                    w_ed = min(total_frames, ed + halo_f)

                    t_c = time.perf_counter()
                    voc_in: dict[str, Any] = {"latents": latents[:, :, w_st:w_ed]}
                    if "voice_cond" in voc_in_names and voice_cond is not None:
                        voc_in["voice_cond"] = voice_cond
                    wav_win = voc_sess.run(None, voc_in)[0][0]

                    crop_l = (st - w_st) * HOP_LENGTH
                    crop_r = crop_l + (ed - st) * HOP_LENGTH
                    extra_r = min(len(wav_win), crop_r + (0 if is_last_sub else xfade))
                    seg = wav_win[crop_l:extra_r].astype(np.float32).copy()

                    if prev_tail is not None and xfade > 0 and len(prev_tail) == xfade and len(seg) >= xfade:
                        seg[:xfade] = prev_tail * fade_out + seg[:xfade] * fade_in

                    if not is_last_sub and xfade > 0 and (extra_r - crop_r) == xfade:
                        prev_tail = seg[-xfade:].copy()
                        pcm = seg[:-xfade]
                    else:
                        prev_tail = None
                        pcm = seg[: crop_r - crop_l]

                    # Micro-envelope on clause start/end + natural breath pause between multi-clause sentences
                    if sub_i == 0 and len(pcm) > 192:
                        r_in = 0.5 * (1.0 - np.cos(math.pi * np.arange(192, dtype=np.float32) / 192.0))
                        pcm[:192] *= r_in
                    if is_last_sub and len(pcm) > 256:
                        r_out = 0.5 * (1.0 + np.cos(math.pi * np.arange(256, dtype=np.float32) / 256.0))
                        pcm[-256:] *= r_out
                        if c_idx < len(clauses) - 1:
                            pcm = np.concatenate([pcm, np.zeros(int(0.08 * SAMPLE_RATE), dtype=np.float32)])

                    peak = float(np.max(np.abs(pcm))) if len(pcm) > 0 else 0.0
                    if peak > 0.98:
                        pcm = (0.94 * (pcm / peak)).astype(np.float32)

                    now_ms = (time.perf_counter() - t_start) * 1000.0
                    if ttfa_ms is None:
                        ttfa_ms = now_ms
                    is_final = (c_idx == len(clauses) - 1) and is_last_sub
                    yield {
                        "pcm": pcm,
                        "wav": encode_wav_bytes(pcm, sr=SAMPLE_RATE),
                        "sample_rate": SAMPLE_RATE,
                        "index": chunk_idx,
                        "clause_index": c_idx,
                        "clause_count": len(clauses),
                        "is_final": is_final,
                        "duration_sec": len(pcm) / SAMPLE_RATE,
                        "latency_ms": (time.perf_counter() - t_c) * 1000.0,
                        "ttfa_ms": ttfa_ms,
                        "lang": parsed["lang"],
                        "tier": "neural_onnx",
                        "aksharas": parsed["aksharas"] if sub_i == 0 else [],
                        "durations": durs[0].tolist() if sub_i == 0 else [],
                    }
                    chunk_idx += 1
            return

        if tier != "klatt" and self.session is not None:
            import numpy as np

            in_names = {i.name for i in self.session.get_inputs()}
            for c_idx, clause in enumerate(clauses):
                t_c = time.perf_counter()
                parsed = text_to_ids(clause, lang=use_lang)
                ort_in = {
                    "input_ids": np.array([parsed["input_ids"]], dtype=np.int64),
                    "attention_mask": np.array([parsed["attention_mask"]], dtype=np.int64),
                }
                if "lang_id" in in_names:
                    ort_in["lang_id"] = np.array([parsed["lang_id"]], dtype=np.int64)
                    ort_in["scales"] = np.array([float(rate), float(noise)], dtype=np.float32)
                else:
                    ort_in["scales"] = np.array(
                        [float(rate), float(noise), float(noise_duration)], dtype=np.float32
                    )
                wav, durs = self.session.run(None, ort_in)
                pcm = wav[0].astype(np.float32)
                peak = float(np.max(np.abs(pcm))) if len(pcm) > 0 else 0.0
                if peak > 1e-4:
                    pcm = 0.88 * (pcm / max(1.0, peak))
                if ttfa_ms is None:
                    ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                yield {
                    "pcm": pcm,
                    "wav": encode_wav_bytes(pcm, sr=SAMPLE_RATE),
                    "sample_rate": SAMPLE_RATE,
                    "index": chunk_idx,
                    "clause_index": c_idx,
                    "clause_count": len(clauses),
                    "is_final": c_idx == len(clauses) - 1,
                    "duration_sec": len(pcm) / SAMPLE_RATE,
                    "latency_ms": (time.perf_counter() - t_c) * 1000.0,
                    "ttfa_ms": ttfa_ms,
                    "lang": parsed["lang"],
                    "tier": "neural_onnx",
                    "aksharas": parsed["aksharas"],
                    "durations": durs[0].tolist(),
                }
                chunk_idx += 1
            return

        for c_idx, clause in enumerate(clauses):
            t_c = time.perf_counter()
            pcm_list, parsed = synthesize_klatt_pcm(clause, lang=use_lang, rate=rate)
            if ttfa_ms is None:
                ttfa_ms = (time.perf_counter() - t_start) * 1000.0
            yield {
                "pcm": pcm_list,
                "wav": encode_wav_bytes(pcm_list, sr=SAMPLE_RATE),
                "sample_rate": SAMPLE_RATE,
                "index": chunk_idx,
                "clause_index": c_idx,
                "clause_count": len(clauses),
                "is_final": c_idx == len(clauses) - 1,
                "duration_sec": len(pcm_list) / SAMPLE_RATE,
                "latency_ms": (time.perf_counter() - t_c) * 1000.0,
                "ttfa_ms": ttfa_ms,
                "lang": parsed["lang"],
                "tier": "klatt_zero_dep",
                "aksharas": parsed["aksharas"],
                "durations": [],
            }
            chunk_idx += 1

    def speak(
        self,
        text: str,
        lang: str | None = None,
        rate: float = 1.0,
        noise: float = 0.667,
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
        noise: float = 0.667,
        tier: str = "auto",
    ) -> tuple[Any, dict[str, Any]]:
        """Synthesize full waveform array + linguistic metadata using halo-crossfaded streaming chunks."""
        use_lang = lang if (lang in LANGUAGE_IDS) else self.default_lang
        parsed = text_to_ids(text, lang=use_lang)

        if tier != "klatt" and (self.is_neural or self._get_specialist_sessions(use_lang) is not None):
            import numpy as np

            pieces: list[np.ndarray] = []
            all_durs: list[int] = []
            ttfa_ms = 0.0
            for ch in self.stream(text=text, lang=use_lang, rate=rate, noise=noise, tier=tier):
                pieces.append(np.asarray(ch["pcm"], dtype=np.float32))
                if ch.get("durations"):
                    all_durs.extend(ch["durations"])
                ttfa_ms = float(ch["ttfa_ms"])
            waveform = np.concatenate(pieces, axis=0) if pieces else np.zeros(1600, dtype=np.float32)
            peak = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0
            if peak > 1e-4:
                waveform = 0.88 * (waveform / max(1.0, peak))
            parsed["tier"] = "neural_onnx"
            parsed["durations"] = all_durs
            parsed["ttfa_ms"] = ttfa_ms
            return waveform, parsed

        pcm, parsed = synthesize_klatt_pcm(text, lang=use_lang, rate=rate)
        parsed["tier"] = "klatt_zero_dep"
        return pcm, parsed


def speak(text: str, lang: str = "sa", rate: float = 1.0, tier: str = "auto") -> bytes:
    """One-line convenience helper to synthesize 16 kHz RIFF WAV bytes."""
    return Ninada(lang=lang).speak(text, lang=lang, rate=rate, tier=tier)


def stream(
    text: str,
    lang: str = "sa",
    rate: float = 1.0,
    noise: float = 0.667,
    tier: str = "auto",
    chunk_frames: int = 64,
) -> Iterator[dict[str, Any]]:
    """One-line convenience generator to stream 16 kHz audio chunks with <200 ms TTFA."""
    yield from Ninada(lang=lang).stream(
        text, lang=lang, rate=rate, noise=noise, tier=tier, chunk_frames=chunk_frames
    )
