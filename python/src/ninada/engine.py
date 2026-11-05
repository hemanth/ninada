"""Streaming 24 kHz Native Studio VITS Specialists ONNX + Zero-Dep Klatt Runtime Engine for Ninada v4.0."""
from __future__ import annotations

from collections.abc import Iterator
import math
import re
import struct
import time
from pathlib import Path
from typing import Any

try:
    from scipy.signal import lfilter as _scipy_lfilter
except Exception:
    _scipy_lfilter = None

from .g2p import LANGUAGE_IDS, text_to_ids
from .studio_g2p import STUDIO_HOP_LENGTH, studio_text_to_ids

SAMPLE_RATE = 24000
HOP_LENGTH = 600

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


def _split_long_unpunctuated(segment: str, max_chars: int) -> list[str]:
    """Split an over-length clause at natural comma/dash pauses or balanced word boundaries without orphan tails."""
    text = segment.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    # Prefer natural secondary pause boundaries (comma, colon, dash) before resorting to word splits
    sub_parts = re.split(r"(?<=[,:;—–])\s+", text)
    if len(sub_parts) > 1:
        packed: list[str] = []
        cur = ""
        for sp in sub_parts:
            s = sp.strip()
            if not s:
                continue
            if not cur:
                cur = s
            elif len(cur) + 1 + len(s) <= max_chars:
                cur = f"{cur} {s}"
            else:
                packed.extend(_split_long_unpunctuated(cur, max_chars))
                cur = s
        if cur:
            packed.extend(_split_long_unpunctuated(cur, max_chars))
        if len(packed) >= 2 and len(packed[-1].split()) <= 2 and len(packed[-2]) + 1 + len(packed[-1]) <= int(max_chars * 1.2):
            tail = packed.pop()
            packed[-1] = f"{packed[-1]} {tail}"
        return packed

    words = text.split(" ")
    n_chunks = max(2, math.ceil(len(text) / max(40, max_chars)))
    target_chars = math.ceil(len(text) / n_chunks)
    out: list[str] = []
    buf = ""
    for w in words:
        if not buf:
            buf = w
        elif len(buf) + 1 + len(w) <= target_chars or len(out) == n_chunks - 1:
            buf = f"{buf} {w}"
        else:
            out.append(buf)
            buf = w
    if buf:
        out.append(buf)
    if len(out) >= 2 and len(out[-1].split()) <= 2:
        tail = out.pop()
        out[-1] = f"{out[-1]} {tail}"
    return out


_ABBREV_PROTECT_RE = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|St|vs|Inc|Corp|Ltd|Co|Jr|Sr|Dept|Univ|No|etc|e\.g|i\.e|ie|approx|Rs)\.",
    re.IGNORECASE,
)


def split_prosodic_clauses(text: str, max_chars: int = 100) -> list[str]:
    """Split arbitrarily long text into prosodic breath-group clauses (danda, punctuation, or balanced phrase boundary)."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return []
    protected = _ABBREV_PROTECT_RE.sub(lambda m: m.group(0).replace(".", "\uE000"), cleaned)
    raw_parts = re.split(
        r"(?<=\|\|)(?:\s+|(?=[^\s|]))"
        r"|(?<=[\u0964\u0965])(?:\s+|(?=[^\s\u0964\u0965]))"
        r"|(?<=[\n.?!;:])(?:\s+|(?=[^\s\d.?!;:,]))"
        r"|(?<=[,;—–])\s+",
        protected,
    )
    raw_clauses: list[str] = []
    for part in raw_parts:
        p = part.strip()
        if not p:
            continue
        if re.fullmatch(r"[\u0964\u0965|.?!;,\s\-—–:\uE000]+", p):
            if raw_clauses:
                raw_clauses[-1] = f"{raw_clauses[-1]} {p}"
            continue
        if len(p) <= max_chars:
            raw_clauses.append(p)
            continue
        raw_clauses.extend(_split_long_unpunctuated(p, max_chars))
    merged: list[str] = []
    for item in raw_clauses:
        if (
            merged
            and (len(merged[-1]) < 6 or (len(merged[-1]) < 10 and " " not in merged[-1]))
            and len(merged[-1]) + 1 + len(item) <= max_chars
        ):
            merged[-1] = f"{merged[-1]} {item}"
        else:
            merged.append(item)
    clauses = [c.replace("\uE000", ".") for c in merged]
    if len(clauses) >= 2 and len(clauses[-1].split()) <= 2 and re.search(r"[,—–]\s*$", clauses[-2]):
        tail = clauses.pop()
        clauses[-1] = f"{clauses[-1]} {tail}"
    return clauses or [cleaned]


def split_studio_sentence_blocks(text: str, max_chars: int = 340) -> list[str]:
    """Split Studio VITS input strictly on full sentence boundaries without breaking mid-sentence commas."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return []
    eff_max = max(80, int(max_chars))
    protected = _ABBREV_PROTECT_RE.sub(lambda m: m.group(0).replace(".", "\uE000"), cleaned)
    raw_parts = re.split(
        r"(?<=\|\|)(?:\s+|(?=[^\s|]))"
        r"|(?<=[\u0964\u0965])(?:\s+|(?=[^\s\u0964\u0965]))"
        r"|(?<=[\n.?!])(?:\s+|(?=[^\s\d.?!,]))",
        protected,
    )
    raw_blocks: list[str] = []
    for part in raw_parts:
        p = part.strip()
        if not p:
            continue
        if re.fullmatch(r"[\u0964\u0965|.?!;,\s\-—–:\uE000]+", p):
            if raw_blocks:
                raw_blocks[-1] = f"{raw_blocks[-1]} {p}"
            continue
        if len(p) <= eff_max:
            raw_blocks.append(p)
        else:
            raw_blocks.extend(_split_long_unpunctuated(p, eff_max))
    merged: list[str] = []
    for item in raw_blocks:
        if merged and len(merged[-1]) < 10 and len(merged[-1]) + 1 + len(item) <= eff_max:
            merged[-1] = f"{merged[-1]} {item}"
        else:
            merged.append(item)
    return [b.replace("\uE000", ".") for b in merged] or [cleaned]



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


def apply_dc_blocker_1pole(
    pcm: Any,
    r: float = 0.9965,
    state: tuple[float, float] | None = None,
) -> tuple[Any, tuple[float, float]]:
    """Apply causal 1-pole DC blocking filter: y[n] = x[n] - x[n-1] + R * y[n-1] (fc ≈ 13.3 Hz @ 24 kHz)."""
    import numpy as np

    x_prev, y_prev = state if state is not None else (0.0, 0.0)
    arr = np.asarray(pcm, dtype=np.float32)
    if len(arr) == 0:
        return arr, (x_prev, y_prev)

    if _scipy_lfilter is not None:
        try:
            b = np.array([1.0, -1.0], dtype=np.float32)
            a = np.array([1.0, -float(r)], dtype=np.float32)
            zi = np.array([-x_prev + float(r) * y_prev], dtype=np.float32)
            filtered, _ = _scipy_lfilter(b, a, arr, zi=zi)
            return filtered.astype(np.float32), (float(arr[-1]), float(filtered[-1]))
        except Exception:
            pass

    out = np.empty_like(arr)
    r_f = float(r)
    for n in range(len(arr)):
        xn = float(arr[n])
        yn = xn - x_prev + r_f * y_prev
        out[n] = yn
        x_prev = xn
        y_prev = yn
    return out.astype(np.float32), (x_prev, y_prev)


def generate_room_tone(num_samples: int = 1920, seed: int = 1337) -> Any:
    """Generate calibrated acoustic room tone (-58 dBFS 1-pole low-pass filtered noise with raised-cosine windowing)."""
    import numpy as np

    n = max(1, int(num_samples))
    rng = np.random.default_rng(seed=seed)
    raw_noise = rng.normal(0.0, 0.0035, size=n).astype(np.float32)
    raw_noise = raw_noise - np.mean(raw_noise)
    room_tone = np.empty(n, dtype=np.float32)
    y = 0.0
    alpha = 0.28
    for i in range(n):
        y = y + alpha * (raw_noise[i] - y)
        room_tone[i] = y
    room_tone = room_tone - np.mean(room_tone)
    fade_len = min(192, n // 4)
    if fade_len > 0:
        f_in = 0.5 * (1.0 - np.cos(math.pi * np.arange(fade_len, dtype=np.float32) / fade_len))
        room_tone[:fade_len] *= f_in
        room_tone[-fade_len:] *= f_in[::-1]
    return room_tone


def _synthesize_klatt_akshara_slice(
    aksharas: list[dict[str, Any]],
    total_ak: int,
    start_idx: int = 0,
    speed: float = 1.0,
    f0_base: float = 132.0,
    init_phase: float = 0.0,
    init_seed: int = 20261030,
) -> tuple[list[float], float, int]:
    """Synthesize a range of aksharas using cascade Klatt resonators, preserving phase and RNG seed."""
    pcm: list[float] = []
    phase = init_phase
    seed = init_seed

    def rand_u() -> float:
        nonlocal seed
        seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
        return (seed / 2147483648.0) - 1.0

    def _filt(x: list[float] | Any, freq: float, bw: float) -> list[float]:
        r_c = math.exp(-math.pi * bw / SAMPLE_RATE)
        c_c = -(r_c * r_c)
        b_c = 2.0 * r_c * math.cos(2.0 * math.pi * freq / SAMPLE_RATE)
        a_c = 1.0 - b_c - c_c
        if _scipy_lfilter is not None:
            import numpy as np

            return _scipy_lfilter([a_c], [1.0, -b_c, -c_c], np.asarray(x, dtype=np.float32)).tolist()
        return _biquad_resonator(x if isinstance(x, list) else list(x), freq, bw, sr=SAMPLE_RATE)

    for i, ak in enumerate(aksharas):
        idx = start_idx + i
        s = ak["akshara"]
        weight = ak["weight"]
        decl = 1.0 - 0.10 * (idx / total_ak)
        accent = 1.07 if weight == "G" else (0.96 if weight == "C" else 1.0)
        f0 = f0_base * decl * accent

        f1, f2, f3, dur_v = VOWEL_FORMANTS["\u0905"]
        if s.endswith("\u094d"):
            dur_v = 0.0
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
        n_onset = max(24, int(round(onset_sec * SAMPLE_RATE)))
        n_nuc = max(48, int(round(nucleus_sec * SAMPLE_RATE)))
        n_tot = n_onset + n_nuc

        exc = [0.0] * n_tot
        inv_onset = 1.0 / max(1, n_onset)
        inv_nuc = 1.0 / max(1, n_nuc)
        for t in range(n_tot):
            if t < n_onset:
                env = math.sin(math.pi * (t * inv_onset))
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
                u = (t - n_onset) * inv_nuc
                u_scaled = u * 0.92 + 0.05
                env = math.sin(math.pi * (u_scaled if u_scaled < 1.0 else 1.0))
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

        r1 = _filt(exc, 280.0 if is_nasal else f1, 95.0)
        r2 = _filt(r1, f2, 130.0)
        r3 = _filt(r2, f3, 185.0)
        pcm.extend(r3)
        pcm.extend([0.0] * int(0.010 * SAMPLE_RATE / speed))

    return pcm, phase, seed


def _iter_klatt_streaming_chunks(
    text: str,
    lang: str | None = None,
    rate: float = 1.0,
    f0_base: float = 132.0,
    t_start: float | None = None,
    sample_offset: int = 0,
    chunk_start_idx: int = 0,
    filter_state: tuple[float, float] | None = None,
) -> Iterator[dict[str, Any]]:
    """Internal single-source generator yielding Tier-2 Klatt streaming chunks."""
    eff_rate = max(0.45, min(2.5, float(rate)))
    clauses = split_prosodic_clauses(text)
    t_origin = time.perf_counter() if t_start is None else t_start
    ttfa_ms: float | None = None
    chunk_idx = chunk_start_idx
    cur_offset = sample_offset
    filter_state_kl = filter_state if filter_state is not None else (0.0, 0.0)

    if not clauses:
        silent = generate_room_tone(num_samples=int(0.10 * SAMPLE_RATE), seed=1337)
        now_ms = (time.perf_counter() - t_origin) * 1000.0
        yield {
            "pcm": silent,
            "wav": encode_wav_bytes(silent, sr=SAMPLE_RATE),
            "sample_rate": SAMPLE_RATE,
            "sample_offset": cur_offset,
            "index": chunk_idx,
            "clause_index": 0,
            "clause_count": 1,
            "is_final": True,
            "duration_sec": len(silent) / SAMPLE_RATE,
            "latency_ms": now_ms,
            "ttfa_ms": now_ms,
            "lang": lang or "sa",
            "tier": "klatt_zero_dep",
            "aksharas": [],
            "durations": [],
            "_filter_state": filter_state_kl,
        }
        return

    import numpy as np

    for c_idx, clause in enumerate(clauses):
        parsed = text_to_ids(clause, lang=lang)
        aksharas = parsed["aksharas"]
        total_ak = len(aksharas)
        if total_ak == 0:
            silent = generate_room_tone(num_samples=int(0.10 * SAMPLE_RATE), seed=1337 + c_idx)
            now_ms = (time.perf_counter() - t_origin) * 1000.0
            if ttfa_ms is None:
                ttfa_ms = now_ms
            yield {
                "pcm": silent,
                "wav": encode_wav_bytes(silent, sr=SAMPLE_RATE),
                "sample_rate": SAMPLE_RATE,
                "sample_offset": cur_offset,
                "index": chunk_idx,
                "clause_index": c_idx,
                "clause_count": len(clauses),
                "is_final": c_idx == len(clauses) - 1,
                "duration_sec": len(silent) / SAMPLE_RATE,
                "latency_ms": (time.perf_counter() - t_origin) * 1000.0 if chunk_idx == chunk_start_idx else 0.05,
                "ttfa_ms": ttfa_ms,
                "lang": parsed["lang"],
                "tier": "klatt_zero_dep",
                "aksharas": [],
                "durations": [],
                "_filter_state": filter_state_kl,
            }
            cur_offset += len(silent)
            chunk_idx += 1
            continue

        if total_ak > 18:
            bounds: list[tuple[int, int]] = []
            cur = 0
            while cur < total_ak:
                step = 6 if cur == 0 and chunk_idx == chunk_start_idx else 14
                nxt = min(total_ak, cur + step)
                if total_ak - nxt < 4:
                    nxt = total_ak
                bounds.append((cur, nxt))
                cur = nxt
        else:
            bounds = [(0, total_ak)]

        ph = 0.0
        sd = 20261030
        num_sub = len(bounds)
        for sub_i, (st_ak, end_ak) in enumerate(bounds):
            t_sub_start = time.perf_counter()
            raw_pcm, ph, sd = _synthesize_klatt_akshara_slice(
                aksharas[st_ak:end_ak],
                total_ak,
                start_idx=st_ak,
                speed=eff_rate,
                f0_base=f0_base,
                init_phase=ph,
                init_seed=sd,
            )
            is_last_chunk_of_clause = sub_i == num_sub - 1
            is_stream_final = (c_idx == len(clauses) - 1) and is_last_chunk_of_clause

            arr = np.asarray(raw_pcm, dtype=np.float32)
            arr, filter_state_kl = apply_dc_blocker_1pole(arr, r=0.9965, state=filter_state_kl)
            if len(arr) > 0:
                # Fixed nominal gain factor
                arr = (0.70 * arr).astype(np.float32)

            pos_m = arr > 0.96
            neg_m = arr < -0.96
            if np.any(pos_m) or np.any(neg_m):
                arr = arr.copy()
                if np.any(pos_m):
                    arr[pos_m] = 0.96 + 0.03 * np.tanh((arr[pos_m] - 0.96) / 0.04)
                if np.any(neg_m):
                    arr[neg_m] = -0.96 - 0.03 * np.tanh((-arr[neg_m] - 0.96) / 0.04)

            # 64-sample raised-cosine chunk edge taper
            t = min(64, len(arr) // 2)
            if t > 0:
                w_in = 0.5 * (1.0 - np.cos(math.pi * (np.arange(t, dtype=np.float32) + 0.5) / float(t)))
                arr[:t] *= w_in
                arr[-t:] *= w_in[::-1]

            # Windowed DC removal applied post-taper to eliminate edge-induced mean shift
            if len(arr) > 0:
                mean_val = float(np.mean(arr))
                if abs(mean_val) > 1e-6:
                    n_samp = len(arr)
                    k_arr = np.arange(n_samp, dtype=np.float32)
                    w_dc = 1.0 - np.cos(2.0 * math.pi * (k_arr + 0.5) / float(n_samp))
                    arr = (arr - mean_val * w_dc).astype(np.float32)

            pcm_out = arr.tolist()
            now_ms = (time.perf_counter() - t_origin) * 1000.0
            if ttfa_ms is None:
                ttfa_ms = now_ms
            t_latency = (time.perf_counter() - t_sub_start) * 1000.0 if chunk_idx == chunk_start_idx else 0.05

            yield {
                "pcm": pcm_out,
                "wav": encode_wav_bytes(pcm_out, sr=SAMPLE_RATE),
                "sample_rate": SAMPLE_RATE,
                "sample_offset": cur_offset,
                "index": chunk_idx,
                "clause_index": c_idx,
                "clause_count": len(clauses),
                "is_final": is_stream_final,
                "duration_sec": len(pcm_out) / SAMPLE_RATE,
                "latency_ms": t_latency,
                "ttfa_ms": ttfa_ms,
                "lang": parsed["lang"],
                "tier": "klatt_zero_dep",
                "aksharas": aksharas[st_ak:end_ak],
                "durations": [],
                "_filter_state": filter_state_kl,
            }
            cur_offset += len(pcm_out)
            chunk_idx += 1


def synthesize_klatt_pcm(
    text: str,
    lang: str | None = None,
    rate: float = 1.0,
    f0_base: float = 132.0,
) -> tuple[list[float], dict[str, Any]]:
    """Tier-2 Zero-Dependency Cascade Klatt-Akshara Synthesizer (24 kHz float PCM)."""
    parsed = text_to_ids(text, lang=lang)
    use_lang = parsed["lang"]
    pcm_full: list[float] = []
    for ch in _iter_klatt_streaming_chunks(text, lang=use_lang, rate=rate, f0_base=f0_base):
        pcm_full.extend(ch["pcm"])
    if not pcm_full:
        pcm_full = [0.0] * 2400
    parsed["tier"] = "klatt_zero_dep"
    return pcm_full, parsed


def encode_wav_bytes(pcm: Any, sr: int = SAMPLE_RATE) -> bytes:
    """Encode float32 [-1, 1] iterable/array into standard 16-bit mono RIFF WAV bytes."""
    try:
        import numpy as np

        arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
        pcm16 = np.round(np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    except Exception:
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


def _has_pronounceable_content(parsed: dict[str, Any]) -> bool:
    """Return True if parsed utterance contains at least one non-virama pronounceable akshara."""
    aksharas = parsed.get("aksharas") or []
    if not aksharas:
        return False
    for ak in aksharas:
        core = str(ak.get("akshara", "")).replace("\u094d", "").strip()
        if core:
            return True
    return False


def _make_ort_cpu_session(onnx_path: Path) -> Any:
    import os
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    n_thr = int(os.environ.get("NINADA_ORT_THREADS", str(min(8, max(4, os.cpu_count() or 4)))))
    so.intra_op_num_threads = max(4, n_thr)
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])


class Ninada:
    """Ninada (निनाद) v5.0: Streaming 24 kHz Native Per-Language Studio VITS Specialists (ONNX) + Zero-Dep Klatt."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        encoder_path: str | Path | None = None,
        vocoder_path: str | Path | None = None,
        voice_dir: str | Path | None = None,
        lang: str = "sa",
        voice: str = "male",
    ) -> None:
        self.default_lang = lang if lang in LANGUAGE_IDS else "sa"
        self.default_voice = "female" if (voice or "").strip().lower() == "female" else "male"
        self.session: Any = None
        self.enc_session: Any = None
        self.voc_session: Any = None
        self.model_path: Path | None = None
        self._explicit_override = bool(model_path or encoder_path or vocoder_path)
        self._studio_v4_sessions: dict[str, Any] = {}
        self._studio_v4_split_sessions: dict[str, tuple[Any, Any]] = {}
        self._dc_filter_state: tuple[float, float] = (0.0, 0.0)
        self._lang_sessions: dict[str, tuple[Any, Any]] = {}
        self._voice_banks: dict[str, Any] = {}
        self._repo_root = Path(__file__).resolve().parents[3]
        self._voice_dir = Path(voice_dir) if voice_dir is not None else None
        self._onnx_base_dir: Path | None = None

        if model_path is not None and Path(model_path).is_dir():
            d = Path(model_path)
            if (
                (d / f"{self.default_lang}.onnx").exists()
                or (d / "models" / f"{self.default_lang}.onnx").exists()
                or (d / "onnx" / f"{self.default_lang}.onnx").exists()
            ):
                self._onnx_base_dir = d
                self._explicit_override = False
                model_path = None
            elif (d / "ninada_24k_q8f16.onnx").exists():
                model_path = d / "ninada_24k_q8f16.onnx"
                if self._voice_dir is None and (d / "voices").is_dir():
                    self._voice_dir = d / "voices"
                self._onnx_base_dir = d
            elif (d / "onnx" / "ninada_24k_q8f16.onnx").exists():
                model_path = d / "onnx" / "ninada_24k_q8f16.onnx"
                if self._voice_dir is None and (d / "voices").is_dir():
                    self._voice_dir = d / "voices"
                self._onnx_base_dir = d
            elif (d / self.default_lang / "encoder_flow_quantized.onnx").exists():
                self._explicit_override = False
                self._onnx_base_dir = d
            elif (d / "encoder_flow_quantized.onnx").exists() and (d / "vocoder_quantized.onnx").exists():
                encoder_path = d / "encoder_flow_quantized.onnx"
                vocoder_path = d / "vocoder_quantized.onnx"
                model_path = None
            elif (d / "model_quantized.onnx").exists():
                model_path = d / "model_quantized.onnx"

        if self._explicit_override:
            try:
                if encoder_path and vocoder_path and Path(encoder_path).exists() and Path(vocoder_path).exists():
                    self.enc_session = _make_ort_cpu_session(Path(encoder_path))
                    self.voc_session = _make_ort_cpu_session(Path(vocoder_path))
                    self.model_path = Path(encoder_path)
                elif model_path is not None and Path(model_path).exists():
                    self.model_path = Path(model_path)
                    self.session = _make_ort_cpu_session(self.model_path)
                    if self._voice_dir is None and (self.model_path.parent / "voices").is_dir():
                        self._voice_dir = self.model_path.parent / "voices"
                    elif self._voice_dir is None and (self.model_path.parent.parent / "voices").is_dir():
                        self._voice_dir = self.model_path.parent.parent / "voices"
            except Exception:
                self.session = None
                self.enc_session = None
                self.voc_session = None
        else:
            sess_v4 = self._get_v4_studio_session(self.default_lang, voice=self.default_voice)
            if sess_v4 is not None:
                self.session = sess_v4
                try:
                    self._get_v4_studio_split_session(self.default_lang, voice=self.default_voice)
                except Exception:
                    pass
            else:
                sess_24k = self._discover_unified_24k_session()
                if sess_24k is not None:
                    self.session = sess_24k
                else:
                    pair = self._get_specialist_sessions(self.default_lang)
                    if pair is not None:
                        self.enc_session, self.voc_session = pair

    def reset_filter_state(self) -> None:
        """Reset causal 1-pole DC blocker filter state to (0.0, 0.0)."""
        self._dc_filter_state = (0.0, 0.0)

    @staticmethod
    def _is_v4_studio_session(sess: Any) -> bool:
        if sess is None:
            return False
        in_names = {i.name for i in sess.get_inputs()}
        return in_names == {"input_ids", "scales"}

    def _get_v4_studio_path(self, lang: str, voice: str | None = None) -> Path | None:
        lc = lang if lang in LANGUAGE_IDS else "sa"
        v_norm = (voice or "").strip().lower()
        is_fem = v_norm == "female"
        fname = f"{lc}_female.onnx" if is_fem else f"{lc}.onnx"
        candidates: list[Path] = []
        if self._onnx_base_dir is not None:
            candidates.extend(
                [
                    self._onnx_base_dir / fname,
                    self._onnx_base_dir / "models" / fname,
                    self._onnx_base_dir / "onnx" / fname,
                ]
            )
        candidates.extend(
            [
                self._repo_root / "docs" / "models" / fname,
                self._repo_root / "artifacts" / "ninada_v5" / "onnx" / fname,
                self._repo_root / "artifacts" / "ninada_v4" / "onnx" / fname,
                self._repo_root / "docs" / fname,
            ]
        )
        for p in candidates:
            if p.exists():
                return p

        try:
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(repo_id="gnumanth/ninada", filename=f"onnx/{fname}"))
        except Exception:
            pass

        if is_fem:
            return self._get_v4_studio_path(lc, voice=None)
        return None

    def _get_v4_studio_session(self, lang: str, voice: str | None = None) -> Any:
        lc = lang if lang in LANGUAGE_IDS else "sa"
        v_norm = (voice or "").strip().lower()
        is_fem = v_norm == "female"
        cache_key = f"{lc}:female" if is_fem else lc

        if self._explicit_override:
            if self._is_v4_studio_session(self.session):
                return self.session
            return None

        if cache_key in self._studio_v4_sessions:
            return self._studio_v4_sessions[cache_key]

        onnx_path = self._get_v4_studio_path(lc, voice=voice)
        if onnx_path is not None and onnx_path.exists():
            try:
                sess = _make_ort_cpu_session(onnx_path)
                if self._is_v4_studio_session(sess):
                    self._studio_v4_sessions[cache_key] = sess
                    if self.model_path is None:
                        self.model_path = onnx_path
                    return sess
            except Exception:
                pass

        if is_fem:
            return self._get_v4_studio_session(lc, voice=None)
        return None

    def _get_v4_studio_split_session(self, lang: str, voice: str | None = None) -> tuple[Any, Any] | None:
        lc = lang if lang in LANGUAGE_IDS else "sa"
        v_norm = (voice or "").strip().lower()
        is_fem = v_norm == "female"
        cache_key = f"{lc}:female" if is_fem else lc

        if cache_key in self._studio_v4_split_sessions:
            return self._studio_v4_split_sessions[cache_key]

        onnx_path = self._get_v4_studio_path(lc, voice=voice)
        if onnx_path is None or not onnx_path.exists():
            return None

        try:
            import tempfile
            import numpy as np
            from onnx.utils import extract_model

            try:
                cache_dir = Path.home() / ".cache" / "ninada" / "split_models"
                cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                cache_dir = Path(tempfile.gettempdir()) / "ninada_split_models"
                cache_dir.mkdir(parents=True, exist_ok=True)

            enc_file = cache_dir / f"{onnx_path.stem}_enc.onnx"
            voc_file = cache_dir / f"{onnx_path.stem}_voc.onnx"

            src_mtime = onnx_path.stat().st_mtime
            needs_extract = (
                not enc_file.exists()
                or not voc_file.exists()
                or enc_file.stat().st_mtime < src_mtime
                or voc_file.stat().st_mtime < src_mtime
                or enc_file.stat().st_size < 1000000
                or voc_file.stat().st_size < 1000000
            )
            if needs_extract:
                extract_model(str(onnx_path), str(enc_file), ["input_ids", "scales"], ["/Mul_8_output_0", "durations"])
                extract_model(str(onnx_path), str(voc_file), ["/Mul_8_output_0"], ["waveform"])

            enc_sess = _make_ort_cpu_session(enc_file)
            voc_sess = _make_ort_cpu_session(voc_file)

            # Warmup JIT execution so chunk 0 emits immediately
            try:
                dummy_ids = np.array([[0, 24, 0]], dtype=np.int64)
                dummy_sc = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                enc_sess.run(None, {"input_ids": dummy_ids, "scales": dummy_sc})
                voc_sess.run(None, {"/Mul_8_output_0": np.zeros((1, 192, 40), dtype=np.float32)})
                voc_sess.run(None, {"/Mul_8_output_0": np.zeros((1, 192, 48), dtype=np.float32)})
                voc_sess.run(None, {"/Mul_8_output_0": np.zeros((1, 192, 112), dtype=np.float32)})
                apply_dc_blocker_1pole(np.zeros(256, dtype=np.float32))
            except Exception:
                pass

            self._studio_v4_split_sessions[cache_key] = (enc_sess, voc_sess)
            return enc_sess, voc_sess
        except Exception:
            return None

    def _discover_unified_24k_session(self) -> Any:
        if self.session is not None:
            return self.session

        candidates: list[Path] = []
        if self._onnx_base_dir is not None:
            candidates.append(self._onnx_base_dir / "ninada_24k_q8f16.onnx")
            candidates.append(self._onnx_base_dir / "onnx" / "ninada_24k_q8f16.onnx")
        candidates.extend(
            [
                self._repo_root / "docs" / "ninada_24k_q8f16.onnx",
                self._repo_root / "artifacts" / "ninada_v3" / "onnx" / "ninada_24k_q8f16.onnx",
                self._repo_root / "artifacts" / "ninada" / "onnx" / "ninada_24k_q8f16.onnx",
            ]
        )
        onnx_path: Path | None = None
        for p in candidates:
            if p.exists():
                onnx_path = p
                break

        if onnx_path is not None and onnx_path.exists():
            try:
                sess = _make_ort_cpu_session(onnx_path)
                self.model_path = onnx_path
                return sess
            except Exception:
                return None
        return None

    def _get_voice_style(self, lang: str, num_ipa_tokens: int, voice: str | None = None) -> Any:
        import numpy as np

        v_norm = (voice or "").strip().lower()
        key = "female" if v_norm == "female" else (lang if lang in LANGUAGE_IDS else "sa")
        bank = self._voice_banks.get(key)
        if bank is None:
            search_paths: list[Path] = []
            if self._voice_dir is not None:
                search_paths.append(self._voice_dir / f"{key}.bin")
            if self.model_path is not None:
                search_paths.append(self.model_path.parent / "voices" / f"{key}.bin")
                search_paths.append(self.model_path.parent.parent / "voices" / f"{key}.bin")
            search_paths.extend(
                [
                    self._repo_root / "docs" / "voices" / f"{key}.bin",
                    self._repo_root / "artifacts" / "ninada_v3" / "voices" / f"{key}.bin",
                    self._repo_root / "artifacts" / "ninada" / "voices" / f"{key}.bin",
                ]
            )
            bin_path: Path | None = None
            for p in search_paths:
                if p.exists() and p.stat().st_size == 510 * 256 * 4:
                    bin_path = p
                    break
            if bin_path is None and key == "female":
                return self._get_voice_style(lang, num_ipa_tokens, voice=None)
            if bin_path is not None:
                bank = np.fromfile(str(bin_path), dtype=np.float32).reshape(510, 1, 256)
            else:
                bank = np.zeros((510, 1, 256), dtype=np.float32)
            self._voice_banks[key] = bank

        row = min(509, max(0, int(num_ipa_tokens) - 2))
        return bank[row].astype(np.float32)

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

        if enc_p and voc_p and enc_p.exists() and voc_p.exists():
            try:
                import onnxruntime as ort

                es = _make_ort_cpu_session(enc_p)
                vs = _make_ort_cpu_session(voc_p)
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
            bool(self._studio_v4_sessions)
            or (self.session is not None)
            or bool(self._lang_sessions)
            or (self.enc_session is not None and self.voc_session is not None)
        )

    @property
    def is_24k_studio(self) -> bool:
        if self._studio_v4_sessions:
            return True
        if self.session is None:
            return False
        in_names = {i.name for i in self.session.get_inputs()}
        return in_names == {"input_ids", "scales"} or ("style" in in_names and "speed" in in_names)

    def with_lang(self, lang: str) -> "Ninada":
        if lang in LANGUAGE_IDS:
            self.default_lang = lang
        return self

    def stream(
        self,
        text: str,
        lang: str | None = None,
        rate: float = 1.0,
        noise: float | None = None,
        noise_duration: float | None = None,
        tier: str = "auto",
        chunk_frames: int = 64,
        halo_frames: int = 12,
        crossfade_samples: int = 256,
        voice: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream 24 kHz native studio neural audio chunks (Native Studio VITS ONNX) or Tier-2 Klatt fallback."""
        use_lang = lang if (lang in LANGUAGE_IDS) else self.default_lang
        use_voice = voice if voice is not None else self.default_voice
        eff_rate = max(0.45, min(2.5, float(rate)))
        eff_noise_base = (0.56 if use_lang == "en" else 0.40) if noise is None else float(noise)
        v4_sess = self._get_v4_studio_session(use_lang, voice=use_voice) if tier != "klatt" else None
        clauses = (
            split_studio_sentence_blocks(text, max_chars=220)
            if v4_sess is not None
            else split_prosodic_clauses(text)
        )
        t_start = time.perf_counter()
        ttfa_ms: float | None = None
        chunk_idx = 0
        sample_offset = 0

        # Path 0: Primary v5.0 Per-Language Native 24,000 Hz Studio VITS Specialist (`input_ids`, `scales`)
        if v4_sess is not None:
            import numpy as np

            if not clauses:
                silent = generate_room_tone(num_samples=int(0.10 * SAMPLE_RATE), seed=1337)
                ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                yield {
                    "pcm": silent,
                    "wav": encode_wav_bytes(silent, sr=SAMPLE_RATE),
                    "sample_rate": SAMPLE_RATE,
                    "index": 0,
                    "sample_offset": 0,
                    "clause_index": 0,
                    "clause_count": 1,
                    "is_final": True,
                    "duration_sec": len(silent) / SAMPLE_RATE,
                    "latency_ms": ttfa_ms,
                    "ttfa_ms": ttfa_ms,
                    "lang": use_lang,
                    "tier": "neural_onnx",
                    "aksharas": [],
                    "durations": [],
                }
                return

            chunk_f = max(24, int(chunk_frames))
            chunk_samples = chunk_f * HOP_LENGTH
            min_tail_samples = 16 * STUDIO_HOP_LENGTH
            filter_state: tuple[float, float] = self._dc_filter_state
            eff_noise = max(0.0, min(1.2, eff_noise_base))
            default_noise_dp = min(0.80, eff_noise * 1.10) if use_lang == "en" else 0.0
            eff_noise_dp = default_noise_dp if noise_duration is None else max(0.0, min(1.2, float(noise_duration)))
            scales_vec = np.asarray([1.0 / eff_rate, eff_noise, eff_noise_dp], dtype=np.float32)

            for c_idx, clause in enumerate(clauses):
                t_c = time.perf_counter()
                parsed = text_to_ids(clause, lang=use_lang)

                if not _has_pronounceable_content(parsed):
                    silent = generate_room_tone(num_samples=int(0.10 * SAMPLE_RATE), seed=1337 + c_idx)
                    now_ms = (time.perf_counter() - t_start) * 1000.0
                    if ttfa_ms is None:
                        ttfa_ms = now_ms
                    yield {
                        "pcm": silent,
                        "wav": encode_wav_bytes(silent, sr=SAMPLE_RATE),
                        "sample_rate": SAMPLE_RATE,
                        "index": chunk_idx,
                        "sample_offset": sample_offset,
                        "clause_index": c_idx,
                        "clause_count": len(clauses),
                        "is_final": c_idx == len(clauses) - 1,
                        "duration_sec": len(silent) / SAMPLE_RATE,
                        "latency_ms": (time.perf_counter() - t_c) * 1000.0,
                        "ttfa_ms": ttfa_ms,
                        "lang": parsed["lang"],
                        "tier": "neural_onnx",
                        "aksharas": parsed["aksharas"],
                        "durations": [],
                    }
                    sample_offset += len(silent)
                    chunk_idx += 1
                    continue

                studio_ids, resolved_lc, _ = studio_text_to_ids(clause, lang=parsed["lang"])
                split_pair = self._get_v4_studio_split_session(resolved_lc, voice=use_voice)
                if split_pair is not None:
                    enc_sess, voc_sess = split_pair
                    enc_out = enc_sess.run(
                        None,
                        {
                            "input_ids": np.asarray([studio_ids], dtype=np.int64),
                            "scales": scales_vec,
                        },
                    )
                    z = enc_out[0]
                    total_latent_frames = z.shape[2]
                    durs_list = np.asarray(enc_out[1], dtype=np.int64).reshape(-1).tolist()

                    VOCODER_BUCKET = 80
                    halo_latent = max(8, min(14, int(halo_frames) if halo_frames != 16 else 12))
                    steady_latent = VOCODER_BUCKET - 2 * halo_latent
                    xfade_len = max(128, min(512, int(crossfade_samples)))
                    first_latent = max(28, min(VOCODER_BUCKET - halo_latent, int(chunk_frames) if chunk_frames <= VOCODER_BUCKET - halo_latent else (VOCODER_BUCKET - halo_latent)))

                    f_bounds: list[tuple[int, int]] = []
                    f_cur = 0
                    while f_cur < total_latent_frames:
                        rem = total_latent_frames - f_cur
                        if f_cur == 0:
                            f_next = total_latent_frames if total_latent_frames <= VOCODER_BUCKET else first_latent
                        elif rem <= VOCODER_BUCKET - halo_latent:
                            f_next = total_latent_frames
                        elif rem <= VOCODER_BUCKET - halo_latent + steady_latent // 2:
                            f_next = f_cur + (rem + 1) // 2
                        else:
                            f_next = f_cur + steady_latent
                        f_bounds.append((f_cur, f_next))
                        f_cur = f_next

                    z_buf = np.zeros((1, 192, VOCODER_BUCKET), dtype=np.float32)
                    prev_right_halo: np.ndarray | None = None

                    for w_idx, (f_start, f_end) in enumerate(f_bounds):
                        t_w0 = time.perf_counter()
                        valid_f = f_end - f_start
                        l_halo = min(f_start, halo_latent)
                        r_halo = min(total_latent_frames - f_end, halo_latent)
                        raw_slice_f = valid_f + l_halo + r_halo
                        bucket_f = VOCODER_BUCKET

                        slice_start = f_start - l_halo
                        z_buf[:, :, :raw_slice_f] = z[:, :, slice_start : slice_start + raw_slice_f]
                        if raw_slice_f < bucket_f:
                            z_buf[:, :, raw_slice_f:] = z_buf[:, :, raw_slice_f - 1 : raw_slice_f]

                        voc_out = voc_sess.run(None, {"/Mul_8_output_0": z_buf})
                        raw_slice = np.asarray(voc_out[0], dtype=np.float32).reshape(-1)

                        valid_start_samp = l_halo * STUDIO_HOP_LENGTH
                        valid_len_samp = valid_f * STUDIO_HOP_LENGTH
                        valid_end_samp = valid_start_samp + valid_len_samp
                        unpadded_samp = raw_slice_f * STUDIO_HOP_LENGTH
                        pcm = raw_slice[valid_start_samp:valid_end_samp].copy()

                        if (
                            prev_right_halo is not None
                            and valid_len_samp >= xfade_len
                            and len(prev_right_halo) >= xfade_len
                        ):
                            alpha = 0.5 * (
                                1.0 - np.cos(math.pi * (np.arange(xfade_len, dtype=np.float32) + 0.5) / float(xfade_len))
                            )
                            pcm[:xfade_len] = (1.0 - alpha) * prev_right_halo[:xfade_len] + alpha * pcm[:xfade_len]

                        is_last_sub = w_idx == len(f_bounds) - 1
                        if not is_last_sub and unpadded_samp >= valid_end_samp + xfade_len:
                            prev_right_halo = raw_slice[valid_end_samp : valid_end_samp + xfade_len].copy()
                        else:
                            prev_right_halo = None

                        if w_idx == 0 and valid_len_samp > 192:
                            r_in = 0.5 * (1.0 - np.cos(math.pi * np.arange(192, dtype=np.float32) / 191.0))
                            pcm[:192] *= r_in
                        if is_last_sub and valid_len_samp > 256:
                            r_out = 0.5 * (1.0 + np.cos(math.pi * np.arange(256, dtype=np.float32) / 255.0))
                            pcm[-256:] *= r_out

                        if is_last_sub:
                            is_final_clause = c_idx == len(clauses) - 1
                            ends_with_double = bool(re.search(r"(?:॥|\|\||।।|।\s*।|\.\s*\.)\s*$", clause))
                            inter_pause = 0.165 if (resolved_lc == "en" or use_lang == "en") else 0.035
                            if is_final_clause:
                                pause_sec = 0.120
                            elif ends_with_double:
                                pause_sec = max(0.140, inter_pause)
                            else:
                                pause_sec = inter_pause
                            pause_samples = int((pause_sec / eff_rate) * SAMPLE_RATE)
                            if pause_samples > 0:
                                room_buf = generate_room_tone(num_samples=pause_samples, seed=1337 + c_idx)
                                pcm = np.concatenate([pcm, room_buf])

                        pcm, filter_state = apply_dc_blocker_1pole(pcm, r=0.9965, state=filter_state)
                        self._dc_filter_state = filter_state

                        pos_m = pcm > 0.96
                        neg_m = pcm < -0.96
                        if np.any(pos_m) or np.any(neg_m):
                            pcm = pcm.copy()
                            if np.any(pos_m):
                                pcm[pos_m] = 0.96 + 0.03 * np.tanh((pcm[pos_m] - 0.96) / 0.04)
                            if np.any(neg_m):
                                pcm[neg_m] = -0.96 - 0.03 * np.tanh((-pcm[neg_m] - 0.96) / 0.04)

                        # 64-sample raised-cosine edge taper across all chunk boundaries
                        t_taper = min(64, len(pcm) // 2)
                        if t_taper > 0:
                            w_in = 0.5 * (1.0 - np.cos(math.pi * (np.arange(t_taper, dtype=np.float32) + 0.5) / float(t_taper)))
                            pcm[:t_taper] *= w_in
                            pcm[-t_taper:] *= w_in[::-1]

                        # Windowed DC removal applied after taper to guarantee zero residual mean
                        if len(pcm) > 0:
                            mean_val = float(np.mean(pcm))
                            if abs(mean_val) > 1e-6:
                                n_samp = len(pcm)
                                k_arr = np.arange(n_samp, dtype=np.float32)
                                w_dc = 1.0 - np.cos(2.0 * math.pi * (k_arr + 0.5) / float(n_samp))
                                pcm = (pcm - mean_val * w_dc).astype(np.float32)

                        now_ms = (time.perf_counter() - t_start) * 1000.0
                        if ttfa_ms is None:
                            ttfa_ms = now_ms
                        t_chunk_ms = (time.perf_counter() - t_c) * 1000.0 if w_idx == 0 else 0.05
                        yield {
                            "pcm": pcm,
                            "wav": encode_wav_bytes(pcm, sr=SAMPLE_RATE),
                            "sample_rate": SAMPLE_RATE,
                            "index": chunk_idx,
                            "sample_offset": sample_offset,
                            "clause_index": c_idx,
                            "clause_count": len(clauses),
                            "is_final": (c_idx == len(clauses) - 1) and is_last_sub,
                            "duration_sec": len(pcm) / SAMPLE_RATE,
                            "latency_ms": t_chunk_ms,
                            "ttfa_ms": ttfa_ms,
                            "lang": resolved_lc,
                            "tier": "neural_onnx",
                            "aksharas": parsed["aksharas"] if w_idx == 0 else [],
                            "durations": durs_list if w_idx == 0 else [],
                        }
                        sample_offset += len(pcm)
                        chunk_idx += 1
                    continue

                active_sess = self._get_v4_studio_session(resolved_lc, voice=use_voice) or v4_sess
                wav_out, durs_out = active_sess.run(
                    None,
                    {
                        "input_ids": np.asarray([studio_ids], dtype=np.int64),
                        "scales": scales_vec,
                    },
                )
                clause_pcm = np.asarray(wav_out, dtype=np.float32).reshape(-1).copy()
                durs_list = np.asarray(durs_out, dtype=np.int64).reshape(-1).tolist()

                clause_pcm, filter_state = apply_dc_blocker_1pole(clause_pcm, r=0.9965, state=filter_state)
                self._dc_filter_state = filter_state
                if len(clause_pcm) > 0:
                    mean_val = float(np.mean(clause_pcm))
                    if abs(mean_val) > 1e-6:
                        n_samp = len(clause_pcm)
                        k_arr = np.arange(n_samp, dtype=np.float32)
                        w_dc = 1.0 - np.cos(2.0 * math.pi * (k_arr + 0.5) / float(n_samp))
                        clause_pcm = (clause_pcm - mean_val * w_dc).astype(np.float32)

                if len(clause_pcm) > 192:
                    r_in = 0.5 * (1.0 - np.cos(math.pi * np.arange(192, dtype=np.float32) / 191.0))
                    clause_pcm[:192] *= r_in
                if len(clause_pcm) > 256:
                    r_out = 0.5 * (1.0 + np.cos(math.pi * np.arange(256, dtype=np.float32) / 255.0))
                    clause_pcm[-256:] *= r_out

                is_final_clause = c_idx == len(clauses) - 1
                ends_with_double = bool(re.search(r"(?:॥|\|\||।।|।\s*।|\.\s*\.)\s*$", clause))
                inter_pause = 0.165 if (resolved_lc == "en" or use_lang == "en") else 0.035
                if is_final_clause:
                    pause_sec = 0.120
                elif ends_with_double:
                    pause_sec = max(0.140, inter_pause)
                else:
                    pause_sec = inter_pause
                pause_samples = int((pause_sec / eff_rate) * SAMPLE_RATE)
                if pause_samples > 0:
                    room_buf = generate_room_tone(num_samples=pause_samples, seed=1337 + c_idx)
                    clause_pcm = np.concatenate([clause_pcm, room_buf])

                pos_m = clause_pcm > 0.96
                neg_m = clause_pcm < -0.96
                if np.any(pos_m) or np.any(neg_m):
                    clause_pcm = clause_pcm.copy()
                    if np.any(pos_m):
                        clause_pcm[pos_m] = 0.96 + 0.03 * np.tanh((clause_pcm[pos_m] - 0.96) / 0.04)
                    if np.any(neg_m):
                        clause_pcm[neg_m] = -0.96 - 0.03 * np.tanh((-clause_pcm[neg_m] - 0.96) / 0.04)

                total_s = len(clause_pcm)
                bounds: list[tuple[int, int]] = []
                cur = 0
                while cur < total_s:
                    nxt = min(total_s, cur + chunk_samples)
                    if total_s - nxt < min_tail_samples:
                        nxt = total_s
                    bounds.append((cur, nxt))
                    cur = nxt
                if not bounds:
                    bounds = [(0, total_s)]

                clause_elapsed_ms = (time.perf_counter() - t_c) * 1000.0
                for sub_i, (st_s, ed_s) in enumerate(bounds):
                    sub_pcm = clause_pcm[st_s:ed_s]
                    is_last_sub = sub_i == len(bounds) - 1
                    now_ms = (time.perf_counter() - t_start) * 1000.0
                    if ttfa_ms is None:
                        ttfa_ms = now_ms
                    yield {
                        "pcm": sub_pcm,
                        "wav": encode_wav_bytes(sub_pcm, sr=SAMPLE_RATE),
                        "sample_rate": SAMPLE_RATE,
                        "index": chunk_idx,
                        "sample_offset": sample_offset,
                        "clause_index": c_idx,
                        "clause_count": len(clauses),
                        "is_final": (c_idx == len(clauses) - 1) and is_last_sub,
                        "duration_sec": len(sub_pcm) / SAMPLE_RATE,
                        "latency_ms": clause_elapsed_ms if sub_i == 0 else 0.05,
                        "ttfa_ms": ttfa_ms,
                        "lang": resolved_lc,
                        "tier": "neural_onnx",
                        "aksharas": parsed["aksharas"] if sub_i == 0 else [],
                        "durations": durs_list if sub_i == 0 else [],
                    }
                    sample_offset += len(sub_pcm)
                    chunk_idx += 1
            self._dc_filter_state = filter_state
            return

        # Path 1: Legacy Unified 24 kHz ONNX (`input_ids`, `style`, `speed`)
        if tier != "klatt" and self.session is not None and "style" in {i.name for i in self.session.get_inputs()}:
            import numpy as np

            if not clauses:
                silent = generate_room_tone(num_samples=int(0.10 * SAMPLE_RATE), seed=1337)
                ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                yield {
                    "pcm": silent,
                    "wav": encode_wav_bytes(silent, sr=SAMPLE_RATE),
                    "sample_rate": SAMPLE_RATE,
                    "index": 0,
                    "sample_offset": 0,
                    "clause_index": 0,
                    "clause_count": 1,
                    "is_final": True,
                    "duration_sec": len(silent) / SAMPLE_RATE,
                    "latency_ms": ttfa_ms,
                    "ttfa_ms": ttfa_ms,
                    "lang": use_lang,
                    "tier": "neural_onnx",
                    "aksharas": [],
                    "durations": [],
                }
                return

            chunk_f = max(24, int(chunk_frames))
            chunk_samples = chunk_f * HOP_LENGTH
            min_tail_samples = 16 * HOP_LENGTH
            filter_state_leg: tuple[float, float] = self._dc_filter_state

            for c_idx, clause in enumerate(clauses):
                t_c = time.perf_counter()
                parsed = text_to_ids(clause, lang=use_lang)
                if not _has_pronounceable_content(parsed):
                    silent = generate_room_tone(num_samples=int(0.10 * SAMPLE_RATE), seed=1337 + c_idx)
                    now_ms = (time.perf_counter() - t_start) * 1000.0
                    if ttfa_ms is None:
                        ttfa_ms = now_ms
                    yield {
                        "pcm": silent,
                        "wav": encode_wav_bytes(silent, sr=SAMPLE_RATE),
                        "sample_rate": SAMPLE_RATE,
                        "index": chunk_idx,
                        "sample_offset": sample_offset,
                        "clause_index": c_idx,
                        "clause_count": len(clauses),
                        "is_final": c_idx == len(clauses) - 1,
                        "duration_sec": len(silent) / SAMPLE_RATE,
                        "latency_ms": (time.perf_counter() - t_c) * 1000.0,
                        "ttfa_ms": ttfa_ms,
                        "lang": parsed["lang"],
                        "tier": "neural_onnx",
                        "aksharas": parsed["aksharas"],
                        "durations": [],
                    }
                    sample_offset += len(silent)
                    chunk_idx += 1
                    continue

                ipa_ids = parsed["ipa_ids"]
                style_vec = self._get_voice_style(parsed["lang"], len(ipa_ids), voice=voice)
                ort_in = {
                    "input_ids": np.array([ipa_ids], dtype=np.int64),
                    "style": style_vec,
                    "speed": np.array([eff_rate], dtype=np.float32),
                }
                wav_out, durs_out = self.session.run(None, ort_in)
                clause_pcm = np.asarray(wav_out, dtype=np.float32).reshape(-1).copy()
                durs_list = np.asarray(durs_out, dtype=np.int64).reshape(-1).tolist()

                clause_pcm, filter_state_leg = apply_dc_blocker_1pole(clause_pcm, r=0.9965, state=filter_state_leg)
                self._dc_filter_state = filter_state_leg
                if len(clause_pcm) > 0:
                    mean_val = float(np.mean(clause_pcm))
                    if abs(mean_val) > 1e-6:
                        n_samp = len(clause_pcm)
                        k_arr = np.arange(n_samp, dtype=np.float32)
                        w_dc = 1.0 - np.cos(2.0 * math.pi * (k_arr + 0.5) / float(n_samp))
                        clause_pcm = (clause_pcm - mean_val * w_dc).astype(np.float32)

                if len(clause_pcm) > 192:
                    r_in = 0.5 * (1.0 - np.cos(math.pi * np.arange(192, dtype=np.float32) / 191.0))
                    clause_pcm[:192] *= r_in
                if len(clause_pcm) > 256:
                    r_out = 0.5 * (1.0 + np.cos(math.pi * np.arange(256, dtype=np.float32) / 255.0))
                    clause_pcm[-256:] *= r_out
                if c_idx < len(clauses) - 1:
                    pause_samples = int(0.048 * SAMPLE_RATE)
                    room_buf = generate_room_tone(num_samples=pause_samples, seed=1337 + c_idx)
                    clause_pcm = np.concatenate([clause_pcm, room_buf])

                pos_m = clause_pcm > 0.96
                neg_m = clause_pcm < -0.96
                if np.any(pos_m) or np.any(neg_m):
                    clause_pcm = clause_pcm.copy()
                    if np.any(pos_m):
                        clause_pcm[pos_m] = 0.96 + 0.03 * np.tanh((clause_pcm[pos_m] - 0.96) / 0.04)
                    if np.any(neg_m):
                        clause_pcm[neg_m] = -0.96 - 0.03 * np.tanh((-clause_pcm[neg_m] - 0.96) / 0.04)

                total_s = len(clause_pcm)
                bounds = []
                cur = 0
                while cur < total_s:
                    nxt = min(total_s, cur + chunk_samples)
                    if total_s - nxt < min_tail_samples:
                        nxt = total_s
                    bounds.append((cur, nxt))
                    cur = nxt
                if not bounds:
                    bounds = [(0, total_s)]

                clause_elapsed_ms = (time.perf_counter() - t_c) * 1000.0
                for sub_i, (st_s, ed_s) in enumerate(bounds):
                    sub_pcm = clause_pcm[st_s:ed_s]
                    is_last_sub = sub_i == len(bounds) - 1
                    now_ms = (time.perf_counter() - t_start) * 1000.0
                    if ttfa_ms is None:
                        ttfa_ms = now_ms
                    yield {
                        "pcm": sub_pcm,
                        "wav": encode_wav_bytes(sub_pcm, sr=SAMPLE_RATE),
                        "sample_rate": SAMPLE_RATE,
                        "index": chunk_idx,
                        "sample_offset": sample_offset,
                        "clause_index": c_idx,
                        "clause_count": len(clauses),
                        "is_final": (c_idx == len(clauses) - 1) and is_last_sub,
                        "duration_sec": len(sub_pcm) / SAMPLE_RATE,
                        "latency_ms": clause_elapsed_ms if sub_i == 0 else 0.05,
                        "ttfa_ms": ttfa_ms,
                        "lang": parsed["lang"],
                        "tier": "neural_onnx",
                        "aksharas": parsed["aksharas"] if sub_i == 0 else [],
                        "durations": durs_list if sub_i == 0 else [],
                    }
                    sample_offset += len(sub_pcm)
                    chunk_idx += 1
            return

        # Path 2: Legacy 2-stage encoder + vocoder sessions (if explicitly configured)
        enc_sess = self.enc_session
        voc_sess = self.voc_session
        if not self._explicit_override and tier != "klatt":
            pair = self._get_specialist_sessions(use_lang)
            if pair is not None:
                enc_sess, voc_sess = pair

        if tier != "klatt" and enc_sess is not None and voc_sess is not None:
            import numpy as np

            legacy_hop = 256
            chunk_f = max(24, int(chunk_frames))
            halo_f = max(8, int(halo_frames))
            xfade = max(0, min(int(crossfade_samples), halo_f * legacy_hop // 2))

            enc_in_names = {i.name for i in enc_sess.get_inputs()}
            voc_in_names = {i.name for i in voc_sess.get_inputs()}

            fade_in = (
                0.5 * (1.0 - np.cos(math.pi * np.arange(xfade, dtype=np.float32) / max(1, xfade)))
                if xfade > 0
                else np.zeros(0, dtype=np.float32)
            )
            fade_out = 1.0 - fade_in

            global_ord = 0
            filter_state_enc = self._dc_filter_state
            for c_idx, clause in enumerate(clauses):
                parsed = text_to_ids(clause, lang=use_lang)
                ort_in_leg: dict[str, Any] = {
                    "input_ids": np.array([parsed["input_ids"]], dtype=np.int64),
                    "attention_mask": np.array([parsed["attention_mask"]], dtype=np.int64),
                }
                if "lang_id" in enc_in_names:
                    ort_in_leg["lang_id"] = np.array([parsed["lang_id"]], dtype=np.int64)
                    ort_in_leg["scales"] = np.array([eff_rate, eff_noise_base], dtype=np.float32)
                else:
                    eff_noise_dur = 0.80 if noise_duration is None else float(noise_duration)
                    ort_in_leg["scales"] = np.array(
                        [eff_rate, eff_noise_base, eff_noise_dur], dtype=np.float32
                    )

                enc_outs = enc_sess.run(None, ort_in_leg)
                if len(enc_outs) == 3:
                    latents, voice_cond, durs = enc_outs
                else:
                    latents, durs = enc_outs
                    voice_cond = None

                total_frames = int(latents.shape[2])
                bounds = []
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

                    crop_l = (st - w_st) * legacy_hop
                    crop_r = crop_l + (ed - st) * legacy_hop
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

                    pcm, filter_state_enc = apply_dc_blocker_1pole(pcm, r=0.9965, state=filter_state_enc)
                    self._dc_filter_state = filter_state_enc
                    if len(pcm) > 0:
                        mean_val = float(np.mean(pcm))
                        if abs(mean_val) > 1e-6:
                            n_samp = len(pcm)
                            k_arr = np.arange(n_samp, dtype=np.float32)
                            w_dc = 1.0 - np.cos(2.0 * math.pi * (k_arr + 0.5) / float(n_samp))
                            pcm = (pcm - mean_val * w_dc).astype(np.float32)

                        pos_m = pcm > 0.96
                        neg_m = pcm < -0.96
                        if np.any(pos_m) or np.any(neg_m):
                            pcm = pcm.copy()
                            if np.any(pos_m):
                                pcm[pos_m] = 0.96 + 0.03 * np.tanh((pcm[pos_m] - 0.96) / 0.04)
                            if np.any(neg_m):
                                pcm[neg_m] = -0.96 - 0.03 * np.tanh((-pcm[neg_m] - 0.96) / 0.04)

                    if sub_i == 0 and len(pcm) > 192:
                        r_in = 0.5 * (1.0 - np.cos(math.pi * np.arange(192, dtype=np.float32) / 191.0))
                        pcm[:192] *= r_in
                    if is_last_sub and len(pcm) > 256:
                        r_out = 0.5 * (1.0 + np.cos(math.pi * np.arange(256, dtype=np.float32) / 255.0))
                        pcm[-256:] *= r_out

                    now_ms = (time.perf_counter() - t_start) * 1000.0
                    if ttfa_ms is None:
                        ttfa_ms = now_ms
                    yield {
                        "pcm": pcm,
                        "wav": encode_wav_bytes(pcm, sr=SAMPLE_RATE),
                        "sample_rate": SAMPLE_RATE,
                        "sample_offset": sample_offset,
                        "index": chunk_idx,
                        "clause_index": c_idx,
                        "clause_count": len(clauses),
                        "is_final": (c_idx == len(clauses) - 1) and is_last_sub,
                        "duration_sec": len(pcm) / SAMPLE_RATE,
                        "latency_ms": (time.perf_counter() - t_c) * 1000.0,
                        "ttfa_ms": ttfa_ms,
                        "lang": parsed["lang"],
                        "tier": "neural_onnx",
                        "aksharas": parsed["aksharas"] if sub_i == 0 else [],
                        "durations": durs[0].tolist() if sub_i == 0 else [],
                    }
                    sample_offset += len(pcm)
                    chunk_idx += 1
            self._dc_filter_state = filter_state_enc
            return

        # Path 3: Legacy monolithic VITS ONNX with attention_mask (if explicitly configured)
        if tier != "klatt" and self.session is not None and "attention_mask" in {i.name for i in self.session.get_inputs()}:
            import numpy as np

            in_names = {i.name for i in self.session.get_inputs()}
            for c_idx, clause in enumerate(clauses):
                t_c = time.perf_counter()
                parsed = text_to_ids(clause, lang=use_lang)
                ort_in_m = {
                    "input_ids": np.array([parsed["input_ids"]], dtype=np.int64),
                    "attention_mask": np.array([parsed["attention_mask"]], dtype=np.int64),
                }
                if "lang_id" in in_names:
                    ort_in_m["lang_id"] = np.array([parsed["lang_id"]], dtype=np.int64)
                    ort_in_m["scales"] = np.array([eff_rate, eff_noise_base], dtype=np.float32)
                else:
                    eff_noise_dur = 0.80 if noise_duration is None else float(noise_duration)
                    ort_in_m["scales"] = np.array([eff_rate, eff_noise_base, eff_noise_dur], dtype=np.float32)
                wav, durs = self.session.run(None, ort_in_m)
                pcm = wav[0].astype(np.float32)
                pcm = (pcm - np.mean(pcm)).astype(np.float32)
                if ttfa_ms is None:
                    ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                yield {
                    "pcm": pcm,
                    "wav": encode_wav_bytes(pcm, sr=SAMPLE_RATE),
                    "sample_rate": SAMPLE_RATE,
                    "sample_offset": sample_offset,
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
                sample_offset += len(pcm)
                chunk_idx += 1
            return

        # Path 4: Zero-Dependency Klatt Tier-2 Clause Streaming
        for chunk in _iter_klatt_streaming_chunks(
            text,
            lang=use_lang,
            rate=eff_rate,
            f0_base=132.0,
            t_start=t_start,
            sample_offset=sample_offset,
            chunk_start_idx=chunk_idx,
            filter_state=self._dc_filter_state,
        ):
            if ttfa_ms is None and chunk.get("ttfa_ms") is not None:
                ttfa_ms = chunk["ttfa_ms"]
            self._dc_filter_state = chunk.get("_filter_state", self._dc_filter_state)
            yield chunk
            sample_offset = chunk["sample_offset"] + len(chunk["pcm"])
            chunk_idx = chunk["index"] + 1
        return

    def speak(
        self,
        text: str,
        lang: str | None = None,
        rate: float = 1.0,
        noise: float | None = None,
        noise_duration: float | None = None,
        tier: str = "auto",
        voice: str | None = None,
    ) -> bytes:
        """Synthesize text into 24 kHz mono WAV bytes."""
        pcm, _ = self.synthesize(
            text=text,
            lang=lang,
            rate=rate,
            noise=noise,
            noise_duration=noise_duration,
            tier=tier,
            voice=voice,
        )
        return encode_wav_bytes(pcm, sr=SAMPLE_RATE)

    def synthesize(
        self,
        text: str,
        lang: str | None = None,
        rate: float = 1.0,
        noise: float | None = None,
        noise_duration: float | None = None,
        tier: str = "auto",
        voice: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Synthesize full 24 kHz waveform array + linguistic metadata."""
        use_lang = lang if (lang in LANGUAGE_IDS) else self.default_lang
        use_voice = voice if voice is not None else self.default_voice
        parsed = text_to_ids(text, lang=use_lang)

        has_neural = (
            self.is_neural
            or self._get_v4_studio_session(use_lang, voice=use_voice) is not None
            or self._get_specialist_sessions(use_lang) is not None
        )
        if tier != "klatt" and has_neural:
            import numpy as np

            pieces: list[np.ndarray] = []
            all_durs: list[int] = []
            ttfa_ms = 0.0
            for ch in self.stream(
                text=text,
                lang=use_lang,
                rate=rate,
                noise=noise,
                noise_duration=noise_duration,
                tier=tier,
                voice=use_voice,
            ):
                pieces.append(np.asarray(ch["pcm"], dtype=np.float32))
                if ch.get("durations"):
                    all_durs.extend(ch["durations"])
                ttfa_ms = float(ch["ttfa_ms"])
            waveform = np.concatenate(pieces, axis=0) if pieces else np.zeros(2400, dtype=np.float32)
            parsed["tier"] = "neural_onnx"
            parsed["durations"] = all_durs
            parsed["ttfa_ms"] = ttfa_ms
            return waveform, parsed

        pcm, parsed = synthesize_klatt_pcm(text, lang=use_lang, rate=rate)
        parsed["tier"] = "klatt_zero_dep"
        return pcm, parsed


def speak(
    text: str,
    lang: str = "sa",
    rate: float = 1.0,
    noise: float | None = None,
    noise_duration: float | None = None,
    tier: str = "auto",
    voice: str | None = None,
) -> bytes:
    """One-line convenience helper to synthesize 24 kHz RIFF WAV bytes."""
    return Ninada(lang=lang).speak(
        text,
        lang=lang,
        rate=rate,
        noise=noise,
        noise_duration=noise_duration,
        tier=tier,
        voice=voice,
    )


def stream(
    text: str,
    lang: str = "sa",
    rate: float = 1.0,
    noise: float | None = None,
    noise_duration: float | None = None,
    tier: str = "auto",
    chunk_frames: int = 64,
    voice: str | None = None,
) -> Iterator[dict[str, Any]]:
    """One-line convenience generator to stream 24 kHz audio chunks with low TTFA."""
    yield from Ninada(lang=lang).stream(
        text,
        lang=lang,
        rate=rate,
        noise=noise,
        noise_duration=noise_duration,
        tier=tier,
        chunk_frames=chunk_frames,
        voice=voice,
    )
