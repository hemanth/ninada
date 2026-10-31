#!/usr/bin/env python3
"""
Curate the Ninada v2.0 Male-Matched Studio Pan-Indic & Classical/Vedic Sanskrit TTS Corpus.

Sources strictly single-speaker/pitch-matched Male neutral studio speech (48 kHz -> 16 kHz)
aligned with the Male acoustic priors (102-170 Hz) of the native VITS HiFi-GAN vocoders:
  1. `prathoshap/sushrota-sanskrit-asr-data` (`recitation_corpus` + `scholar`, gated by
     Su-srota Conformer-CTC `studio_align_cer <= 5.0%` AND Male F0 `115 <= median_f0 <= 195 Hz`)
  2. `ai4bharat/Rasa` 48 kHz anechoic studio Male narration (`Sanskrit`, `Kannada`, `Hindi`,
     `Tamil`, `Telugu`, `Malayalam`; strictly `gender == "Male"` and neutral narration styles
     `BOOK, WIKI, NEWS, CONV, INDIC, ALEXA, BB, UMANG, DIGI`, excluding theatrical emotions)

Produces leakage-free domain-stratified Train (80%) / Validation (10%) / Held-Out Test (10%) splits.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import re
import sys
import unicodedata
from fractions import Fraction
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python" / "src"))

from ninada.g2p import LANGUAGE_IDS, NATIVE_VOCABS, text_to_ids

DATA_DIR = ROOT / "data"
WAV_DIR = DATA_DIR / "wavs"
TARGET_SR = 16000
SUSHROTA_DIR = Path("/Users/lika/labs/sushrota-sanskrit-asr/models")

NEUTRAL_RASA_STYLES = {
    "BOOK",
    "WIKI",
    "NEWS",
    "CONV",
    "INDIC",
    "ALEXA",
    "BB",
    "UMANG",
    "DIGI",
}

HF_SOURCES: list[dict[str, Any]] = [
    {
        "tag": "sa_rasa_male",
        "lang": "sa",
        "domain": "rasa_sanskrit_studio",
        "repo_id": "ai4bharat/Rasa",
        "format": "rasa",
        "shards": [
            "Sanskrit/test-00000-of-00007.parquet",
            "Sanskrit/test-00001-of-00007.parquet",
        ],
        "max_human_cer": 7.5,
        "f0_range": (110.0, 205.0),
        "target_count": 230,
        "max_scan": 700,
    },
    {
        "tag": "sa_sushrota_recit",
        "lang": "sa",
        "domain": "sushrota_recitation_corpus",
        "repo_id": "prathoshap/sushrota-sanskrit-asr-data",
        "format": "audio_dict",
        "shards": [
            "data/train-00000-of-00005.parquet",
            "data/train-00001-of-00005.parquet",
        ],
        "allowed_subsources": {"recitation_corpus"},
        "text_col": "text",
        "max_human_cer": 5.0,
        "f0_range": (115.0, 195.0),
        "target_count": 120,
        "max_scan": 1800,
    },
    {
        "tag": "sa_sushrota_scholar",
        "lang": "sa",
        "domain": "sushrota_scholar_prose",
        "repo_id": "prathoshap/sushrota-sanskrit-asr-data",
        "format": "audio_dict",
        "shards": [
            "data/train-00000-of-00005.parquet",
            "data/train-00001-of-00005.parquet",
        ],
        "allowed_subsources": {"scholar"},
        "text_col": "text",
        "max_human_cer": 5.0,
        "f0_range": (115.0, 195.0),
        "target_count": 100,
        "max_scan": 1800,
    },
    {
        "tag": "kn_rasa_male",
        "lang": "kn",
        "domain": "rasa_kannada_studio",
        "repo_id": "ai4bharat/Rasa",
        "format": "rasa",
        "shards": [
            "Kannada/test-00006-of-00007.parquet",
            "Kannada/test-00005-of-00007.parquet",
        ],
        "f0_range": (95.0, 205.0),
        "target_count": 210,
        "max_scan": 900,
    },
    {
        "tag": "hi_rasa_male",
        "lang": "hi",
        "domain": "rasa_hindi_studio",
        "repo_id": "ai4bharat/Rasa",
        "format": "rasa",
        "shards": [
            "Hindi/test-00000-of-00003.parquet",
        ],
        "f0_range": (95.0, 205.0),
        "target_count": 210,
        "max_scan": 950,
    },
    {
        "tag": "ta_rasa_male",
        "lang": "ta",
        "domain": "rasa_tamil_studio",
        "repo_id": "ai4bharat/Rasa",
        "format": "rasa",
        "shards": [
            "Tamil/test-00007-of-00008.parquet",
            "Tamil/test-00006-of-00008.parquet",
        ],
        "f0_range": (95.0, 205.0),
        "target_count": 210,
        "max_scan": 900,
    },
    {
        "tag": "te_rasa_male",
        "lang": "te",
        "domain": "rasa_telugu_studio",
        "repo_id": "ai4bharat/Rasa",
        "format": "rasa",
        "shards": [
            "Telugu/test-00006-of-00007.parquet",
            "Telugu/test-00005-of-00007.parquet",
        ],
        "f0_range": (95.0, 205.0),
        "target_count": 210,
        "max_scan": 900,
    },
    {
        "tag": "ml_rasa_male",
        "lang": "ml",
        "domain": "rasa_malayalam_studio",
        "repo_id": "ai4bharat/Rasa",
        "format": "rasa",
        "shards": [
            "Malayalam/test-00006-of-00007.parquet",
            "Malayalam/test-00005-of-00007.parquet",
        ],
        "f0_range": (85.0, 200.0),
        "target_count": 210,
        "max_scan": 950,
    },
]


def _edit_distance(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[m]


def _norm_sa_cer(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\u0964\u0965,.;:!?\"'()\-–—0-9\u0966-\u096f]+", " ", s)
    return re.sub(r"\s+", "", s).strip()


class StudioAlignmentGate:
    """Transcript-alignment gate via independent Su-srota Conformer-CTC."""

    def __init__(self, model_dir: Path) -> None:
        self.enabled = (model_dir / "sushrota_sanskrit_ctc_int8.onnx").exists()
        if self.enabled:
            self.prep = ort.InferenceSession(str(model_dir / "preprocessor.onnx"), providers=["CPUExecutionProvider"])
            self.asr = ort.InferenceSession(
                str(model_dir / "sushrota_sanskrit_ctc_int8.onnx"), providers=["CPUExecutionProvider"]
            )
            self.vocab: list[str] = json.loads((model_dir / "sanskrit_vocab.json").read_text(encoding="utf-8"))

    def human_cer(self, wav_16k: np.ndarray, ref_text: str) -> float:
        if not self.enabled or len(wav_16k) < 800:
            return 0.0
        audio = wav_16k.astype(np.float32)
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-4:
            audio = 0.9 * (audio / peak)
        feats, flen = self.prep.run(
            None,
            {
                "audio_signal": audio[np.newaxis, :],
                "length": np.array([len(audio)], dtype=np.int64),
            },
        )
        logits = self.asr.run(None, {"audio_signal": feats, "length": flen})[0]
        pred_ids = np.argmax(logits[0], axis=-1)
        blank_id = len(self.vocab) if logits.shape[-1] > len(self.vocab) else (len(self.vocab) - 1)
        out: list[str] = []
        prev = -1
        for idx in pred_ids:
            i = int(idx)
            if i != prev and i != 0 and i != blank_id and i < len(self.vocab):
                out.append(self.vocab[i])
            prev = i
        hyp = "".join(out).replace("\u2581", " ").strip()
        r_c = list(_norm_sa_cer(ref_text))
        h_c = list(_norm_sa_cer(hyp))
        if not r_c:
            return 100.0
        return _edit_distance(r_c, h_c) / len(r_c) * 100.0


def resample_to_16k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if orig_sr == TARGET_SR:
        return wav
    frac = Fraction(TARGET_SR, orig_sr).limit_denominator(200)
    out = resample_poly(wav, frac.numerator, frac.denominator)
    return out.astype(np.float32)


def trim_and_normalize(wav: np.ndarray, sr: int = TARGET_SR, trim_edges: bool = True) -> np.ndarray | None:
    if wav.size < int(1.0 * sr):
        return None
    wav = wav - float(np.mean(wav))
    peak = float(np.max(np.abs(wav)))
    if peak < 0.015 or not np.isfinite(peak):
        return None
    wav = wav / peak

    win = int(0.02 * sr)
    if trim_edges and wav.size > 4 * win:
        abs_w = np.abs(wav)
        kernel = np.ones(win, dtype=np.float32) / win
        env = np.convolve(abs_w, kernel, mode="same")
        active = np.where(env > 0.0015)[0]
        if active.size < int(0.8 * sr):
            return None
        pad = int(0.180 * sr)
        start = max(0, int(active[0]) - pad)
        end = min(wav.size, int(active[-1]) + pad)
        wav = wav[start:end]

    # Soft 10ms fade-in/out at boundaries to eliminate truncation clicks
    fade = min(int(0.010 * sr), wav.size // 8)
    if fade > 4:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wav[:fade] *= ramp
        wav[-fade:] *= ramp[::-1]

    rms = float(np.sqrt(np.mean(wav * wav)))
    if rms < 0.018:
        return None
    wav = 0.85 * (wav / max(1e-6, float(np.max(np.abs(wav)))))
    return wav.astype(np.float32)


def estimate_median_f0(wav: np.ndarray, sr: int = TARGET_SR) -> float | None:
    f0 = librosa.yin(wav, fmin=70, fmax=360, sr=sr, frame_length=1024, hop_length=256)
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=256)[0]
    n = min(len(f0), len(rms))
    valid = (rms[:n] > 0.025) & (f0[:n] > 75.0) & (f0[:n] < 340.0)
    if int(np.sum(valid)) < 8:
        return None
    return float(np.median(f0[:n][valid]))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    # Remove stale wavs from prior dataset version so local directory is 100% clean Male studio wavs
    for old_wav in WAV_DIR.glob("*.wav"):
        old_wav.unlink(missing_ok=True)

    align_gate = StudioAlignmentGate(SUSHROTA_DIR)
    print(f"Studio Alignment Gate enabled={align_gate.enabled}", flush=True)

    audit: dict[str, Any] = {
        "sources": {},
        "rejected_empty_text": 0,
        "rejected_style_or_gender": 0,
        "rejected_corrupt_or_silent": 0,
        "rejected_duration_outlier": 0,
        "rejected_duplicate_or_unk": 0,
        "rejected_pitch_mismatch": 0,
        "rejected_transcript_misalignment": 0,
    }

    by_source_tag: dict[str, list[dict[str, Any]]] = {}
    seen_texts: set[str] = set()

    for spec in HF_SOURCES:
        tag = spec["tag"]
        lang = spec["lang"]
        repo_id = spec["repo_id"]
        fmt = spec["format"]
        shards = spec["shards"]
        allowed_sub = spec.get("allowed_subsources")
        text_col = spec.get("text_col", "text")
        target_n = spec["target_count"]
        max_scan = spec.get("max_scan", 1000)
        max_cer = spec.get("max_human_cer")
        f0_lo, f0_hi = spec.get("f0_range", (85.0, 210.0))
        unk_id = len(NATIVE_VOCABS[lang]) - 1

        collected = 0
        scanned = 0
        cer_rejected = 0
        f0_rejected = 0
        f0_vals: list[float] = []
        by_source_tag[tag] = []

        for shard in shards:
            if collected >= target_n or scanned >= max_scan:
                break
            parquet_path = hf_hub_download(repo_id=repo_id, filename=shard, repo_type="dataset")
            pf = pq.ParquetFile(parquet_path)

            for batch in pf.iter_batches(batch_size=32):
                cols = batch.to_pydict()
                n_rows = len(cols[text_col])
                for i in range(n_rows):
                    if collected >= target_n or scanned >= max_scan:
                        break

                    if fmt == "rasa":
                        gender = str(cols.get("gender", [""])[i] or "").strip()
                        style = str(cols.get("style", [""])[i] or "").strip().upper()
                        if gender != "Male" or style not in NEUTRAL_RASA_STYLES:
                            audit["rejected_style_or_gender"] += 1
                            continue
                    elif allowed_sub is not None and "source" in cols:
                        if str(cols["source"][i] or "") not in allowed_sub:
                            continue

                    scanned += 1
                    raw_text = str(cols[text_col][i] or "").strip()
                    if not raw_text:
                        audit["rejected_empty_text"] += 1
                        continue

                    try:
                        aud = cols["audio"][i]
                        if isinstance(aud, dict) and aud.get("bytes"):
                            raw_wav, orig_sr = sf.read(io.BytesIO(aud["bytes"]), dtype="float32")
                        elif isinstance(aud, dict) and aud.get("array") is not None:
                            raw_wav = np.asarray(aud["array"], dtype=np.float32)
                            orig_sr = int(aud.get("sampling_rate") or TARGET_SR)
                        elif "bytes" in cols and cols["bytes"][i]:
                            raw_wav, orig_sr = sf.read(io.BytesIO(cols["bytes"][i]), dtype="float32")
                        else:
                            audit["rejected_corrupt_or_silent"] += 1
                            continue
                    except Exception:
                        audit["rejected_corrupt_or_silent"] += 1
                        continue

                    wav_16k = resample_to_16k(raw_wav, int(orig_sr))
                    clean_wav = trim_and_normalize(wav_16k, sr=TARGET_SR, trim_edges=(fmt != "rasa"))
                    if clean_wav is None:
                        audit["rejected_corrupt_or_silent"] += 1
                        continue

                    dur_s = round(clean_wav.size / TARGET_SR, 3)
                    if dur_s < 1.35 or dur_s > 9.6:
                        audit["rejected_duration_outlier"] += 1
                        continue

                    g2p = text_to_ids(raw_text, lang=lang)
                    norm_key = f"{lang}:{g2p['native_text']}"
                    if norm_key in seen_texts or unk_id in g2p["input_ids"]:
                        audit["rejected_duplicate_or_unk"] += 1
                        continue
                    n_native_toks = len(g2p["native_tokens"])
                    if n_native_toks < 8 or len(g2p["input_ids"]) > 360:
                        audit["rejected_duration_outlier"] += 1
                        continue
                    cps = n_native_toks / dur_s
                    if cps < 3.5 or cps > 24.0:
                        audit["rejected_duration_outlier"] += 1
                        continue

                    med_f0 = estimate_median_f0(clean_wav, sr=TARGET_SR)
                    if med_f0 is None or not (f0_lo <= med_f0 <= f0_hi):
                        audit["rejected_pitch_mismatch"] += 1
                        f0_rejected += 1
                        continue

                    human_cer_val = None
                    if max_cer is not None and align_gate.enabled:
                        human_cer_val = round(align_gate.human_cer(clean_wav, raw_text), 2)
                        if human_cer_val > max_cer:
                            audit["rejected_transcript_misalignment"] += 1
                            cer_rejected += 1
                            continue

                    seen_texts.add(norm_key)
                    uid = hashlib.sha1(norm_key.encode("utf-8")).hexdigest()[:12]
                    fname = f"{tag}_{uid}.wav"
                    fpath = WAV_DIR / fname
                    sf.write(str(fpath), clean_wav, TARGET_SR, subtype="PCM_16")

                    f0_vals.append(med_f0)
                    rms = round(float(np.sqrt(np.mean(clean_wav * clean_wav))), 4)
                    rec: dict[str, Any] = {
                        "id": f"{tag}_{uid}",
                        "wav_relpath": f"data/wavs/{fname}",
                        "lang": lang,
                        "lang_id": LANGUAGE_IDS[lang],
                        "gender": "Male",
                        "median_f0_hz": round(med_f0, 2),
                        "domain": spec["domain"],
                        "source_repo": repo_id,
                        "text": raw_text,
                        "native_text": g2p["native_text"],
                        "normalized_devanagari": g2p["normalized_devanagari"],
                        "input_ids": g2p["input_ids"],
                        "num_tokens": n_native_toks,
                        "num_aksharas": len(g2p["aksharas"]),
                        "duration_sec": dur_s,
                        "rms": rms,
                    }
                    if fmt == "rasa":
                        rec["style"] = str(cols.get("style", [""])[i] or "").strip().upper()
                    if human_cer_val is not None:
                        rec["studio_align_cer"] = human_cer_val
                    by_source_tag[tag].append(rec)
                    collected += 1

                if collected >= target_n or scanned >= max_scan:
                    break

        mean_f0 = round(float(np.mean(f0_vals)), 1) if f0_vals else 0.0
        audit["sources"][tag] = {
            "repo_id": repo_id,
            "lang": lang,
            "scanned": scanned,
            "accepted": collected,
            "mean_f0_hz": mean_f0,
            "rejected_f0_gate": f0_rejected,
            "rejected_cer_gate": cer_rejected,
        }
        print(
            f"[{tag:19s}] accepted={collected:3d} / scanned={scanned:4d} "
            f"(mean_f0={mean_f0:5.1f} Hz, f0_rej={f0_rejected:3d}, cer_rej={cer_rejected:3d}) from {repo_id}",
            flush=True,
        )

    # Deterministic domain-stratified split (80% train / 10% val / 10% test) BEFORE fitting any stats
    rng = random.Random(20261031)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    by_lang_all: dict[str, list[dict[str, Any]]] = {k: [] for k in LANGUAGE_IDS}
    by_lang_splits: dict[str, dict[str, int]] = {
        k: {"train": 0, "val": 0, "test": 0} for k in LANGUAGE_IDS
    }

    for tag, items in by_source_tag.items():
        if not items:
            continue
        lang = items[0]["lang"]
        rng.shuffle(items)
        n = len(items)
        n_test = max(3, int(round(0.10 * n)))
        n_val = max(3, int(round(0.10 * n)))
        n_train = max(1, n - n_val - n_test)

        tr = items[:n_train]
        va = items[n_train : n_train + n_val]
        te = items[n_train + n_val :]

        train_rows.extend(tr)
        val_rows.extend(va)
        test_rows.extend(te)
        by_lang_all[lang].extend(items)
        by_lang_splits[lang]["train"] += len(tr)
        by_lang_splits[lang]["val"] += len(va)
        by_lang_splits[lang]["test"] += len(te)

    rng.shuffle(train_rows)

    per_lang_summary: dict[str, Any] = {}
    for lang, items in by_lang_all.items():
        n = len(items)
        total_dur_min = round(sum(r["duration_sec"] for r in items) / 60.0, 2)
        mean_f0_lang = round(float(np.mean([r["median_f0_hz"] for r in items])), 1) if items else 0.0
        per_lang_summary[lang] = {
            "total": n,
            "train": by_lang_splits[lang]["train"],
            "val": by_lang_splits[lang]["val"],
            "test": by_lang_splits[lang]["test"],
            "minutes": total_dur_min,
            "mean_duration_sec": round(sum(r["duration_sec"] for r in items) / max(1, n), 2),
            "mean_f0_hz": mean_f0_lang,
        }

    for split_name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        out_p = DATA_DIR / f"{split_name}_meta.jsonl"
        with open(out_p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_durs = [r["duration_sec"] for r in train_rows]
    train_cps = [r["num_tokens"] / r["duration_sec"] for r in train_rows]
    stats = {
        "dataset_version": "2.0.0",
        "sample_rate": TARGET_SR,
        "speaker_profile": "Male anechoic studio (pitch-matched to native VITS HiFi-GAN vocoders)",
        "total_clips": len(train_rows) + len(val_rows) + len(test_rows),
        "train_clips": len(train_rows),
        "val_clips": len(val_rows),
        "test_clips": len(test_rows),
        "total_duration_minutes": round(
            sum(r["duration_sec"] for r in (train_rows + val_rows + test_rows)) / 60.0, 2
        ),
        "train_only_stats": {
            "mean_duration_sec": round(float(np.mean(train_durs)), 3),
            "std_duration_sec": round(float(np.std(train_durs)), 3),
            "mean_chars_per_sec": round(float(np.mean(train_cps)), 3),
        },
        "per_language": per_lang_summary,
        "hygiene_audit": audit,
    }
    (DATA_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\n=== Ninada v2.0 Male Studio Dataset Curation Complete ===", flush=True)
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
