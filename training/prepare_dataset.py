#!/usr/bin/env python3
"""
Curate the Ninada Pan-Indic & Classical/Vedic Sanskrit studio TTS corpus from
9 real Hugging Face datasets (including 4 distinct Sanskrit speech domains: IISc Su-srota
Vedanta prose, All India Radio Shrutilipi Sanskrit broadcasts, IITB Sanskrit-ASR-84, and
IITK Bhagavad-Gita shlokas), apply LibriTTS / IndicVoices-R Studio Transcript-Alignment
and WadaSNR quality gates, and create leakage-free domain-stratified Train (80%) /
Validation (10%) / Held-Out Test (10%) splits.
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

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python" / "src"))

from ninada.g2p import LANGUAGE_IDS, text_to_ids

DATA_DIR = ROOT / "data"
WAV_DIR = DATA_DIR / "wavs"
TARGET_SR = 16000
SUSHROTA_DIR = Path("/Users/lika/labs/sushrota-sanskrit-asr/models")

HF_SOURCES: list[dict[str, Any]] = [
    {
        "tag": "sa_sushrota",
        "lang": "sa",
        "domain": "sanskrit_recitation_prose",
        "repo_id": "prathoshap/sushrota-sanskrit-asr-data",
        "shard": "data/train-00000-of-00005.parquet",
        "text_col": "text",
        "max_human_cer": 14.0,
        "target_count": 250,
        "max_scan": 550,
    },
    {
        "tag": "sa_shrutilipi",
        "lang": "sa",
        "domain": "sanskrit_air_broadcast",
        "repo_id": "surajp/shrutilipi_sanskrit",
        "shard": "data/train-00000-of-00016-ea3baf304efe85e3.parquet",
        "text_col": "transcriptions",
        "max_human_cer": 14.0,
        "target_count": 140,
        "max_scan": 750,
    },
    {
        "tag": "sa_asr84",
        "lang": "sa",
        "domain": "sanskrit_iitb_spoken",
        "repo_id": "addy88/sanskrit-asr-84",
        "shard": "data/train-00000-of-00030.parquet",
        "text_col": "text",
        "min_wada_snr": 32.0,
        "max_human_cer": 14.0,
        "target_count": 110,
        "max_scan": 800,
    },
    {
        "tag": "sa_gita",
        "lang": "sa",
        "domain": "sanskrit_shloka",
        "repo_id": "yashnbx/gita_supersite_sanskrit_tts",
        "shard": "data/train-00000-of-00001-8c41fcd66e4d371c.parquet",
        "text_col": "text",
        "max_human_cer": 22.0,
        "target_count": 60,
        "max_scan": 320,
    },
    {
        "tag": "hi_syspin",
        "lang": "hi",
        "domain": "hindi_studio",
        "repo_id": "SayantanJoker/SYSPIN_Hindi_Female_TTS",
        "shard": "data/train-00000-of-00035.parquet",
        "text_col": "transcription",
        "target_count": 180,
        "max_scan": 400,
    },
    {
        "tag": "kn_syspin",
        "lang": "kn",
        "domain": "kannada_studio",
        "repo_id": "SayantanJoker/original_data_kannada_tts",
        "shard": "data/train-00000-of-00016.parquet",
        "text_col": "transcription",
        "target_count": 180,
        "max_scan": 400,
    },
    {
        "tag": "ta_syspin",
        "lang": "ta",
        "domain": "tamil_studio",
        "repo_id": "SayantanJoker/original_data_tamil_tts",
        "shard": "data/train-00000-of-00019.parquet",
        "text_col": "transcription",
        "target_count": 180,
        "max_scan": 400,
    },
    {
        "tag": "te_syspin",
        "lang": "te",
        "domain": "telugu_studio",
        "repo_id": "arpit-tiwari/syspin-telugu-tts",
        "shard": "data/test-00000-of-00011.parquet",
        "text_col": "sentence",
        "target_count": 180,
        "max_scan": 400,
    },
    {
        "tag": "ml_iitm",
        "lang": "ml",
        "domain": "malayalam_studio",
        "repo_id": "thennal/indic_tts_ml",
        "shard": "data/train-00000-of-00010-4368680559ac82fd.parquet",
        "text_col": "text",
        "target_count": 180,
        "max_scan": 400,
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
    """LibriTTS / IndicVoices-R style transcript-alignment gate via Su-srota Conformer-CTC."""

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


def trim_and_normalize(wav: np.ndarray, sr: int = TARGET_SR) -> np.ndarray | None:
    if wav.size < int(0.8 * sr):
        return None
    wav = wav - float(np.mean(wav))
    peak = float(np.max(np.abs(wav)))
    if peak < 0.01 or not np.isfinite(peak):
        return None
    wav = wav / peak

    win = int(0.02 * sr)
    if wav.size > 4 * win:
        abs_w = np.abs(wav)
        kernel = np.ones(win, dtype=np.float32) / win
        env = np.convolve(abs_w, kernel, mode="same")
        active = np.where(env > 0.014)[0]
        if active.size < int(0.6 * sr):
            return None
        pad = int(0.045 * sr)
        start = max(0, int(active[0]) - pad)
        end = min(wav.size, int(active[-1]) + pad)
        wav = wav[start:end]

    rms = float(np.sqrt(np.mean(wav * wav)))
    if rms < 0.012:
        return None
    wav = 0.84 * (wav / max(1e-6, float(np.max(np.abs(wav)))))
    return wav.astype(np.float32)


def split_gita_half_shloka(wav: np.ndarray, text: str, sr: int = TARGET_SR) -> tuple[np.ndarray, str] | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and "\u0909\u0935\u093e\u091a" not in ln]
    if not lines:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    dur = wav.size / sr
    if dur <= 9.5:
        return wav, " ".join(lines)

    first_line = lines[0]
    total_chars = max(1, sum(len(x) for x in lines))
    ratio = max(0.24, min(0.58, len(first_line) / total_chars))
    center = int(ratio * wav.size)
    lo = max(int(1.6 * sr), center - int(1.4 * sr))
    hi = min(wav.size - int(1.0 * sr), center + int(1.4 * sr))
    if hi <= lo:
        return None

    hop = 256
    energies = []
    for pos in range(lo, hi, hop):
        seg = wav[pos : pos + 1024]
        energies.append((float(np.mean(seg * seg)), pos))
    if not energies:
        return None
    _, best_cut = min(energies, key=lambda x: x[0])
    sub_wav = wav[:best_cut]
    if 1.4 <= (sub_wav.size / sr) <= 9.8:
        return sub_wav, first_line
    return None


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    align_gate = StudioAlignmentGate(SUSHROTA_DIR)
    print(f"Studio Alignment Gate enabled={align_gate.enabled}")

    audit: dict[str, Any] = {
        "sources": {},
        "rejected_empty_text": 0,
        "rejected_corrupt_or_silent": 0,
        "rejected_duration_outlier": 0,
        "rejected_duplicate_text": 0,
        "rejected_transcript_misalignment": 0,
    }

    by_source_tag: dict[str, list[dict[str, Any]]] = {}
    seen_texts: set[str] = set()

    for spec in HF_SOURCES:
        tag = spec["tag"]
        lang = spec["lang"]
        repo_id = spec["repo_id"]
        shard = spec["shard"]
        text_col = spec["text_col"]
        target_n = spec["target_count"]
        max_scan = spec.get("max_scan", 1000)
        max_cer = spec.get("max_human_cer")

        parquet_path = hf_hub_download(repo_id=repo_id, filename=shard, repo_type="dataset")
        pf = pq.ParquetFile(parquet_path)
        collected = 0
        scanned = 0
        cer_rejected = 0
        by_source_tag[tag] = []

        for batch in pf.iter_batches(batch_size=32):
            cols = batch.to_pydict()
            n_rows = len(cols["audio"])
            for i in range(n_rows):
                if collected >= target_n or scanned >= max_scan:
                    break
                scanned += 1
                raw_text = str(cols[text_col][i] or "").strip()
                if not raw_text:
                    audit["rejected_empty_text"] += 1
                    continue

                aud = cols["audio"][i]
                min_snr = spec.get("min_wada_snr")
                if min_snr is not None and "snr" in cols and cols["snr"][i]:
                    try:
                        wada = float(cols["snr"][i].get("methodDetails", {}).get("snr", 0.0))
                        if wada < min_snr:
                            audit["rejected_corrupt_or_silent"] += 1
                            continue
                    except Exception:
                        pass

                try:
                    if aud.get("bytes"):
                        raw_wav, orig_sr = sf.read(io.BytesIO(aud["bytes"]), dtype="float32")
                    elif aud.get("array") is not None:
                        raw_wav = np.asarray(aud["array"], dtype=np.float32)
                        orig_sr = int(aud.get("sampling_rate") or TARGET_SR)
                    else:
                        audit["rejected_corrupt_or_silent"] += 1
                        continue
                except Exception:
                    audit["rejected_corrupt_or_silent"] += 1
                    continue

                wav_16k = resample_to_16k(raw_wav, int(orig_sr))
                if tag == "sa_gita":
                    seg = split_gita_half_shloka(wav_16k, raw_text, sr=TARGET_SR)
                    if seg is None:
                        audit["rejected_duration_outlier"] += 1
                        continue
                    wav_16k, raw_text = seg

                clean_wav = trim_and_normalize(wav_16k, sr=TARGET_SR)
                if clean_wav is None:
                    audit["rejected_corrupt_or_silent"] += 1
                    continue

                dur_s = round(clean_wav.size / TARGET_SR, 3)
                if dur_s < 1.2 or dur_s > 9.8:
                    audit["rejected_duration_outlier"] += 1
                    continue

                g2p = text_to_ids(raw_text, lang=lang)
                norm_key = f"{lang}:{g2p['normalized_devanagari']}"
                if norm_key in seen_texts:
                    audit["rejected_duplicate_text"] += 1
                    continue
                if len(g2p["tokens"]) < 6 or len(g2p["input_ids"]) > 360:
                    audit["rejected_duration_outlier"] += 1
                    continue
                cps = len(g2p["tokens"]) / dur_s
                if cps < 2.8 or cps > 24.5:
                    audit["rejected_duration_outlier"] += 1
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

                rms = round(float(np.sqrt(np.mean(clean_wav * clean_wav))), 4)
                rec = {
                    "id": f"{tag}_{uid}",
                    "wav_relpath": f"data/wavs/{fname}",
                    "lang": lang,
                    "lang_id": LANGUAGE_IDS[lang],
                    "domain": spec["domain"],
                    "source_repo": repo_id,
                    "text": raw_text,
                    "normalized_devanagari": g2p["normalized_devanagari"],
                    "input_ids": g2p["input_ids"],
                    "num_tokens": len(g2p["tokens"]),
                    "num_aksharas": len(g2p["aksharas"]),
                    "duration_sec": dur_s,
                    "rms": rms,
                }
                if human_cer_val is not None:
                    rec["studio_align_cer"] = human_cer_val
                by_source_tag[tag].append(rec)
                collected += 1
            if collected >= target_n or scanned >= max_scan:
                break

        audit["sources"][tag] = {
            "repo_id": repo_id,
            "lang": lang,
            "scanned": scanned,
            "accepted": collected,
            "rejected_cer_gate": cer_rejected,
        }
        print(
            f"[{tag:13s}] accepted={collected:3d} / scanned={scanned:4d} "
            f"(cer_rejected={cer_rejected:3d}) from {repo_id}"
        )

    # Strict deterministic domain-stratified split (80% train / 10% val / 10% test) BEFORE fitting any stats
    rng = random.Random(20261030)
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
        per_lang_summary[lang] = {
            "total": n,
            "train": by_lang_splits[lang]["train"],
            "val": by_lang_splits[lang]["val"],
            "test": by_lang_splits[lang]["test"],
            "minutes": total_dur_min,
            "mean_duration_sec": round(sum(r["duration_sec"] for r in items) / max(1, n), 2),
        }

    for split_name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        out_p = DATA_DIR / f"{split_name}_meta.jsonl"
        with open(out_p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_durs = [r["duration_sec"] for r in train_rows]
    train_cps = [r["num_tokens"] / r["duration_sec"] for r in train_rows]
    stats = {
        "sample_rate": TARGET_SR,
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
    print("\n=== Ninada Studio Dataset Curation Complete ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
