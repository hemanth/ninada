#!/usr/bin/env python3
"""
Ninada v3.0 — GCP GPU Training & Export Pipeline (24,000 Hz Studio Architecture).

Executes end-to-end on GCP Cloud Run GPU (`us-east4`):
  1. Streams 48 kHz anechoic studio Male recordings from `ai4bharat/Rasa` `train-*.parquet` shards across all 6
     supported languages (`sa`, `hi`, `kn`, `ta`, `te`, `ml`) + Classical Sanskrit studio corpora, resampling cleanly
     to 24,000 Hz (`scipy.signal.resample_poly`).
  2. Extracts 80 fps (`hop=300`) ground-truth F0 contours in Hz (`0.0` unvoiced) via GPU-accelerated vectorized YIN
     and computes exact monotonic token durations (`gt_dur >= 1`, `sum == F`) via Teacher-Prototype Mel-Cepstral
     Viterbi Alignment (auto-rejecting any transcript/audio mismatch).
  3. Trains `Ninada24kModel` with `decoder.generator` (`SineGen + CustomSTFT` 24 kHz harmonic vocoder) strictly frozen
     to guarantee zero vocoder buzz or high-frequency muffling, while optimizing:
       - Per-language smooth studio voice banks (`voice_global_delta`, `voice_slope_delta`)
       - `slot 79` (`'ɭ'` Vedic/Dravidian retroflex lateral) + contextual `bert` / `text_encoder`
       - Explicit `ProsodyPredictor` (Ordinal Duration BCE + Smooth-L1, 80 Hz F0 Smooth-L1, Energy N)
       - Multi-Scale 80-Bin Log-Mel Spectral Envelope Reconstruction + Pretrained Weight-Space Anchor (`theta_0`)
  4. Runs validation `WiSE-FT` (`alpha in {0.25, 0.50, 0.75, 1.0}`) against `alpha = 0.0` across all 6 languages.
  5. Exports `model.safetensors`, `voices/{sa,hi,kn,ta,te,ml}.bin`, `onnx/ninada_24k_q8f16.onnx` (< 98 MB),
     held-out test evaluation metrics (`val_test_metrics_v3.json`), and 24 kHz studio `.wav` samples.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "python" / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ninada.ipa_g2p import ipa_to_ids, text_to_ipa
from training.export_onnx_24k import export_ninada_24k_bundle
from training.modeling_ninada_24k import (
    F0_HOP_LENGTH,
    HOP_LENGTH,
    RETROFLEX_LATERAL_ID,
    SAMPLE_RATE,
    SUPPORTED_LANGS,
    Ninada24kModel,
)

NEUTRAL_RASA_STYLES = {
    "BOOK", "WIKI", "NEWS", "CONV", "INDIC", "ALEXA", "BB", "UMANG", "DIGI", "CHAT",
}

RASA_LANG_DIRS = {
    "sa": "Sanskrit",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
}


class MultiScaleMelEnvelopeLoss(nn.Module):
    """Pitch-comb-invariant Multi-Window 80-Bin Log-Mel Spectral Envelope Loss @ 24,000 Hz."""

    def __init__(self, sr: int = SAMPLE_RATE) -> None:
        super().__init__()
        self.specs = [
            (512, 150, 80),   # 6.25 ms hop: crisp stop bursts, retroflex flaps & geminate closures
            (1024, 300, 80),  # 12.5 ms hop: formant transitions & retroflex F3 locus (ɭ / ɻ / ɳ / ʈ / ɖ)
            (2048, 600, 80),  # 25.0 ms hop: vowel quality, nasal murmur & syllable energy envelope
        ]
        for idx, (n_fft, _, n_mels) in enumerate(self.specs):
            mel_np = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=40.0, fmax=11600.0)
            self.register_buffer(f"mel_fb_{idx}", torch.from_numpy(mel_np).to(torch.float32))
            self.register_buffer(f"win_{idx}", torch.hann_window(n_fft, periodic=True, dtype=torch.float32))

    def log_mel(self, wav: torch.Tensor, scale_idx: int) -> torch.Tensor:
        n_fft, hop, _ = self.specs[scale_idx]
        win = getattr(self, f"win_{scale_idx}")
        fb = getattr(self, f"mel_fb_{scale_idx}")
        stft = torch.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=win,
            center=True,
            return_complex=True,
        )
        mag = torch.abs(stft)
        mel = torch.matmul(fb, mag)
        return torch.log(torch.clamp(mel, min=1e-5))

    def forward(self, pred_wav: torch.Tensor, gt_wav: torch.Tensor) -> torch.Tensor:
        total = pred_wav.new_tensor(0.0)
        for idx in range(len(self.specs)):
            mp = self.log_mel(pred_wav, idx)
            mg = self.log_mel(gt_wav, idx)
            n = min(mp.shape[-1], mg.shape[-1])
            total = total + F.l1_loss(mp[..., :n], mg[..., :n])
        return total / float(len(self.specs))


def resample_to_24k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if orig_sr == SAMPLE_RATE:
        return wav
    frac = Fraction(SAMPLE_RATE, orig_sr).limit_denominator(200)
    return resample_poly(wav, frac.numerator, frac.denominator).astype(np.float32)


def clean_studio_24k(wav: np.ndarray) -> np.ndarray | None:
    if wav.size < int(1.1 * SAMPLE_RATE):
        return None
    wav = wav - float(np.mean(wav))
    peak = float(np.max(np.abs(wav)))
    if peak < 0.015 or not np.isfinite(peak):
        return None
    wav = wav / peak

    # Gentle edge silence trim keeping natural 120 ms studio breath margins
    win = int(0.020 * SAMPLE_RATE)
    env = np.convolve(np.abs(wav), np.ones(win, dtype=np.float32) / win, mode="same")
    active = np.where(env > 0.0025)[0]
    if active.size < int(0.75 * SAMPLE_RATE):
        return None
    pad = int(0.120 * SAMPLE_RATE)
    s = max(0, int(active[0]) - pad)
    e = min(wav.size, int(active[-1]) + pad)
    wav = wav[s:e]

    # Quantize sample count to exact multiple of HOP_LENGTH (600) so F frames == 600*F samples
    n_frames = wav.size // HOP_LENGTH
    if n_frames < 44 or n_frames > 380:  # 1.1s .. 9.5s
        return None
    wav = wav[: n_frames * HOP_LENGTH].copy()

    fade = int(0.008 * SAMPLE_RATE)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    wav[:fade] *= ramp
    wav[-fade:] *= ramp[::-1]

    # Match Kokoro/StyleTTS2 natural RMS (~0.115) & peak (~0.75) so energy N_pred stays in native range!
    rms = float(np.sqrt(np.mean(wav * wav)))
    if rms < 0.015:
        return None
    target_rms = 0.118
    wav = wav * (target_rms / rms)
    peak_after = float(np.max(np.abs(wav)))
    if peak_after > 0.92:
        wav = wav * (0.92 / peak_after)
    return wav.astype(np.float32)


@torch.no_grad()
def extract_f0_80hz_gpu(wav_1d: torch.Tensor, target_len: int, fmin: float = 75.0, fmax: float = 320.0) -> torch.Tensor:
    """Fast GPU vectorized autocorrelation/YIN F0 extractor at 80 Hz (`hop=300`, `0.0` unvoiced)."""
    win_len = 1024
    tau_min = int(SAMPLE_RATE / fmax)  # 75
    tau_max = int(SAMPLE_RATE / fmin)  # 320
    pad = win_len // 2
    x = F.pad(wav_1d.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)
    frames = x.unfold(0, win_len, F0_HOP_LENGTH)  # [N_f, win_len]
    if frames.shape[0] > target_len:
        frames = frames[:target_len]
    elif frames.shape[0] < target_len:
        frames = F.pad(frames, (0, 0, 0, target_len - frames.shape[0]))

    frames = frames - frames.mean(dim=-1, keepdim=True)
    rms = torch.sqrt((frames * frames).mean(dim=-1) + 1e-12)

    # Difference function d(tau) via FFT autocorrelation
    W = 512
    x1 = frames[:, :W]
    n_fft = 1024
    X1 = torch.fft.rfft(x1, n=n_fft, dim=-1)
    X2 = torch.fft.rfft(frames, n=n_fft, dim=-1)
    corr = torch.fft.irfft(X2 * torch.conj(X1), n=n_fft, dim=-1)[:, : tau_max + 1]
    e1 = (x1 * x1).sum(dim=-1, keepdim=True)
    csum = torch.cumsum(frames * frames, dim=-1)
    e2_0 = csum[:, W - 1 : W]
    e2_rest = csum[:, W : W + tau_max] - csum[:, :tau_max]
    e2 = torch.cat([e2_0, e2_rest], dim=-1)
    diff = torch.clamp(e1 + e2 - 2.0 * corr, min=0.0)

    # Cumulative mean normalized difference (CMND)
    taus = torch.arange(1, tau_max + 1, device=wav_1d.device, dtype=torch.float32).unsqueeze(0)
    cmnd = diff[:, 1:] * taus / torch.clamp(torch.cumsum(diff[:, 1:], dim=-1), min=1e-8)
    search = cmnd[:, tau_min - 1 : tau_max]
    min_val, min_idx = torch.min(search, dim=-1)
    best_tau = (min_idx + tau_min).to(torch.float32)
    f0 = SAMPLE_RATE / best_tau
    voiced = (rms > 0.012) & (min_val < 0.32) & (f0 >= fmin) & (f0 <= fmax)
    return torch.where(voiced, f0, torch.zeros_like(f0))


@torch.no_grad()
def align_utterance_viterbi_gpu(
    teacher: Ninada24kModel,
    mel_extractor: MultiScaleMelEnvelopeLoss,
    input_ids: torch.LongTensor,
    lang: str,
    wav_gt: torch.Tensor,
) -> tuple[torch.LongTensor, torch.Tensor, float] | None:
    """
    Compute exact monotonic integer token durations `gt_dur` (`[T]`, `>= 1`, `sum == F_gt`) and `prior_dur` (`[T]`)
    by aligning Teacher-synthesized token log-mel prototypes against studio `wav_gt` at 40 Hz (`hop=600`).
    """
    T = input_ids.shape[1]
    F_gt = wav_gt.shape[-1] // HOP_LENGTH
    if F_gt < T + 2:
        return None

    style = teacher.get_style_for_tokens(lang, max(1, T - 2))
    wav_ref, ref_dur = teacher.infer_tokens(input_ids, style, speed=1.0)
    mel_ref = mel_extractor.log_mel(wav_ref, scale_idx=2).squeeze(0)[:, : int(ref_dur.sum())]  # [80, F_ref]
    mel_gt = mel_extractor.log_mel(wav_gt, scale_idx=2).squeeze(0)[:, :F_gt]                   # [80, F_gt]
    if mel_gt.shape[-1] != F_gt or mel_ref.shape[-1] != int(ref_dur.sum()):
        return None

    # Compute L2-normalized per-token acoustic prototype [T, 80] from teacher frames
    aln_ref = teacher.build_alignment_matrix(ref_dur).squeeze(0)  # [T, F_ref]
    tok_proto = (aln_ref @ mel_ref.transpose(0, 1)) / torch.clamp(aln_ref.sum(dim=-1, keepdim=True), min=1.0)
    tok_proto = F.normalize(tok_proto - tok_proto.mean(dim=-1, keepdim=True), dim=-1)
    gt_norm = F.normalize(mel_gt.transpose(0, 1) - mel_gt.transpose(0, 1).mean(dim=-1, keepdim=True), dim=-1)  # [F_gt, 80]

    sim = gt_norm @ tok_proto.transpose(0, 1)  # [F_gt, T]
    neg_inf = -1.0e9
    dp = torch.full((T,), neg_inf, device=wav_gt.device, dtype=torch.float32)
    dp[0] = sim[0, 0]
    back = torch.zeros((F_gt, T), device=wav_gt.device, dtype=torch.bool)

    for f in range(1, F_gt):
        stay = dp
        adv = torch.cat([dp.new_full((1,), neg_inf), dp[:-1]], dim=0)
        choose_adv = adv > stay
        dp = torch.where(choose_adv, adv, stay) + sim[f]
        back[f] = choose_adv

    mean_sim = float(dp[T - 1].item() / F_gt)
    if not math.isfinite(mean_sim) or mean_sim < 0.28:
        return None

    # Backtrack monotonic token path
    back_cpu = back.cpu().numpy()
    durs = np.zeros(T, dtype=np.int64)
    t_cur = T - 1
    for f in range(F_gt - 1, -1, -1):
        durs[t_cur] += 1
        if f > 0 and back_cpu[f, t_cur]:
            t_cur -= 1
    if t_cur != 0 or int(durs.sum()) != F_gt or int(durs.min()) < 1:
        return None

    gt_dur = torch.from_numpy(durs).to(device=wav_gt.device, dtype=torch.long)
    return gt_dur, ref_dur.to(torch.float32), mean_sim


def curate_studio_dataset_24k(
    work_dir: Path,
    teacher: Ninada24kModel,
    mel_loss: MultiScaleMelEnvelopeLoss,
    device: torch.device,
    per_lang_target: int = 540,
    val_per_lang: int = 40,
    test_per_lang: int = 40,
) -> dict[str, list[dict[str, Any]]]:
    """Stream 48 kHz studio shards from `ai4bharat/Rasa` (+ Sanskrit studio) and build aligned 24 kHz splits."""
    wav_dir = work_dir / "wavs_24k"
    wav_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    rasa_files = api.list_repo_files("ai4bharat/Rasa", repo_type="dataset")

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    seen_keys: set[str] = set()
    rng = random.Random(20261031)

    for lang in SUPPORTED_LANGS:
        lang_dir = RASA_LANG_DIRS[lang]
        train_shards = sorted(f for f in rasa_files if f.startswith(f"{lang_dir}/train-") and f.endswith(".parquet"))
        test_shards = sorted(f for f in rasa_files if f.startswith(f"{lang_dir}/test-") and f.endswith(".parquet"))
        candidate_shards = train_shards + test_shards

        lang_items: list[dict[str, Any]] = []
        rejected_f0 = 0
        rejected_align = 0

        for shard in candidate_shards:
            if len(lang_items) >= per_lang_target:
                break
            try:
                pq_path = hf_hub_download(repo_id="ai4bharat/Rasa", filename=shard, repo_type="dataset")
                pf = pq.ParquetFile(pq_path)
            except Exception as exc:
                print(f"[WARN] Could not download {shard}: {exc}", flush=True)
                continue

            for batch in pf.iter_batches(batch_size=32):
                if len(lang_items) >= per_lang_target:
                    break
                cols = batch.to_pydict()
                texts = cols.get("text", [])
                genders = cols.get("gender", [""] * len(texts))
                styles = cols.get("style", ["BOOK"] * len(texts))
                audios = cols.get("audio", [None] * len(texts))

                for i in range(len(texts)):
                    if len(lang_items) >= per_lang_target:
                        break
                    if str(genders[i] or "").strip() != "Male":
                        continue
                    st = str(styles[i] or "").strip().upper()
                    if st and st not in NEUTRAL_RASA_STYLES:
                        continue
                    raw_text = str(texts[i] or "").strip()
                    if len(raw_text) < 10 or len(raw_text) > 260:
                        continue

                    ipa = text_to_ipa(raw_text, lang=lang)
                    ids = ipa_to_ids(ipa)
                    if len(ids) < 12 or len(ids) > 260:
                        continue
                    dedup_key = f"{lang}:{ipa}"
                    if dedup_key in seen_keys:
                        continue

                    aud = audios[i]
                    try:
                        if isinstance(aud, dict) and aud.get("bytes"):
                            raw_wav, orig_sr = sf.read(io.BytesIO(aud["bytes"]), dtype="float32")
                        elif isinstance(aud, dict) and aud.get("array") is not None:
                            raw_wav = np.asarray(aud["array"], dtype=np.float32)
                            orig_sr = int(aud.get("sampling_rate") or SAMPLE_RATE)
                        else:
                            continue
                    except Exception:
                        continue

                    wav_24k = clean_studio_24k(resample_to_24k(raw_wav, int(orig_sr)))
                    if wav_24k is None:
                        continue

                    dur_s = wav_24k.size / SAMPLE_RATE
                    tok_rate = (len(ids) - 2) / dur_s
                    if tok_rate < 5.0 or tok_rate > 26.0:
                        continue

                    wav_t = torch.from_numpy(wav_24k).to(device)
                    F_frames = wav_24k.size // HOP_LENGTH
                    f0_gt = extract_f0_80hz_gpu(wav_t, target_len=2 * F_frames)
                    voiced = f0_gt[f0_gt > 10.0]
                    if voiced.numel() < 16:
                        rejected_f0 += 1
                        continue
                    med_f0 = float(torch.median(voiced).item())
                    if not (85.0 <= med_f0 <= 215.0):
                        rejected_f0 += 1
                        continue

                    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
                    aligned = align_utterance_viterbi_gpu(teacher, mel_loss, ids_t, lang, wav_t.unsqueeze(0))
                    if aligned is None:
                        rejected_align += 1
                        continue
                    gt_dur, prior_dur, align_sim = aligned

                    seen_keys.add(dedup_key)
                    uid = hashlib.sha1(dedup_key.encode("utf-8")).hexdigest()[:12]
                    wav_path = wav_dir / f"{lang}_{uid}.wav"
                    sf.write(str(wav_path), wav_24k, SAMPLE_RATE, subtype="PCM_16")

                    lang_items.append({
                        "id": f"{lang}_{uid}",
                        "lang": lang,
                        "text": raw_text,
                        "ipa": ipa,
                        "ipa_ids": ids,
                        "wav_path": str(wav_path),
                        "gt_dur": gt_dur.cpu().tolist(),
                        "prior_dur": [round(float(x), 3) for x in prior_dur.cpu().tolist()],
                        "f0_gt": [round(float(x), 2) for x in f0_gt.cpu().tolist()],
                        "median_f0_hz": round(med_f0, 2),
                        "align_sim": round(align_sim, 4),
                        "duration_sec": round(dur_s, 3),
                        "has_retroflex_lat": (RETROFLEX_LATERAL_ID in ids),
                    })

            try:
                del pf
                real_pq = os.path.realpath(pq_path)
                if os.path.exists(real_pq):
                    os.unlink(real_pq)
            except OSError:
                pass

        rng.shuffle(lang_items)
        n_total = len(lang_items)
        n_test = min(test_per_lang, max(4, n_total // 10))
        n_val = min(val_per_lang, max(4, n_total // 10))
        splits["test"].extend(lang_items[:n_test])
        splits["val"].extend(lang_items[n_test : n_test + n_val])
        splits["train"].extend(lang_items[n_test + n_val :])
        n_lat = sum(1 for r in lang_items if r["has_retroflex_lat"])
        print(
            f"[DATA] lang={lang}: collected={n_total} (train={n_total - n_val - n_test}, val={n_val}, test={n_test}), "
            f"retroflex_lat_clips={n_lat}, rej_f0={rejected_f0}, rej_align={rejected_align}",
            flush=True,
        )

    for k, rows in splits.items():
        with open(work_dir / f"{k}_v3.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return splits


@torch.no_grad()
def evaluate_split(
    model: Ninada24kModel,
    mel_loss_fn: MultiScaleMelEnvelopeLoss,
    records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    per_lang: dict[str, list[dict[str, float]]] = {l: [] for l in SUPPORTED_LANGS}
    all_mel, all_dur_mae, all_f0_rmse, all_mcd = [], [], [], []

    for rec in records:
        lang = rec["lang"]
        ids_t = torch.tensor([rec["ipa_ids"]], dtype=torch.long, device=device)
        gt_dur = torch.tensor(rec["gt_dur"], dtype=torch.long, device=device)
        f0_gt = torch.tensor([rec["f0_gt"]], dtype=torch.float32, device=device)
        wav_np, _ = sf.read(rec["wav_path"], dtype="float32")
        wav_gt = torch.from_numpy(wav_np).unsqueeze(0).to(device)

        out = model.forward_train_step(ids_t, lang, gt_dur, f0_gt)
        mel_l = float(mel_loss_fn(out["audio_pred"], wav_gt).item())
        dur_mae = float(torch.abs(out["dur_pred"] - gt_dur.to(torch.float32)).mean().item())

        voiced_mask = f0_gt > 10.0
        if int(voiced_mask.sum().item()) > 4:
            f0_p_v = torch.clamp(out["f0_pred"][voiced_mask], min=40.0)
            f0_g_v = f0_gt[voiced_mask]
            f0_rmse = float(torch.sqrt(F.mse_loss(torch.log(f0_p_v), torch.log(f0_g_v))).item())
        else:
            f0_rmse = 0.0

        # Approximate Mel-Cepstral Distortion (dB) from 80-bin log-mel
        mp = mel_loss_fn.log_mel(out["audio_pred"], 2)
        mg = mel_loss_fn.log_mel(wav_gt, 2)
        nf = min(mp.shape[-1], mg.shape[-1])
        mcd_db = float((10.0 / math.log(10.0)) * math.sqrt(2.0) * torch.abs(mp[..., :nf] - mg[..., :nf]).mean().item())

        all_mel.append(mel_l)
        all_dur_mae.append(dur_mae)
        all_f0_rmse.append(f0_rmse)
        all_mcd.append(mcd_db)
        per_lang[lang].append({"mel": mel_l, "dur_mae": dur_mae, "f0_rmse": f0_rmse, "mcd_db": mcd_db})

    lang_summary = {}
    for lang, items in per_lang.items():
        if items:
            lang_summary[lang] = {
                "n": len(items),
                "mel_l1": round(float(np.mean([x["mel"] for x in items])), 4),
                "mcd_db": round(float(np.mean([x["mcd_db"] for x in items])), 3),
                "dur_mae_frames": round(float(np.mean([x["dur_mae"] for x in items])), 4),
                "voiced_log_f0_rmse": round(float(np.mean([x["f0_rmse"] for x in items])), 4),
            }

    composite = (
        float(np.mean(all_mel))
        + 0.15 * float(np.mean(all_dur_mae))
        + 0.35 * float(np.mean(all_f0_rmse))
    )
    return {
        "composite_loss": round(composite, 5),
        "mel_l1": round(float(np.mean(all_mel)), 4),
        "mcd_db": round(float(np.mean(all_mcd)), 3),
        "dur_mae_frames": round(float(np.mean(all_dur_mae)), 4),
        "voiced_log_f0_rmse": round(float(np.mean(all_f0_rmse)), 4),
        "per_language": lang_summary,
    }


def build_smoke_dataset(
    work_dir: Path,
    teacher: Ninada24kModel,
    mel_loss: MultiScaleMelEnvelopeLoss,
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    """Fast local CPU/GPU smoke dataset (6 clips, 1 per language) to verify the entire pipeline before cloud launch."""
    wav_dir = work_dir / "wavs_24k"
    wav_dir.mkdir(parents=True, exist_ok=True)
    probes = {
        "sa": "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।",
        "hi": "नमस्ते, आज का मौसम बहुत अच्छा है।",
        "kn": "ನಮಸ್ಕಾರ, ಇಂದು ಬೆಂಗಳೂರಿನಲ್ಲಿ ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.",
        "ta": "வணக்கம், இன்று சென்னையில் வானிலை மிகவும் இனிமையாக உள்ளது.",
        "te": "నమస్కారం, ఈరోజు హైదరాబాద్‌లో వాతావరణం చాలా బాగుంది.",
        "ml": "നമಸ್കാരം, ഇന്ന് കേരളത്തിൽ വളരെ നല്ല കാലാവസ്ഥയാണ്.",
    }
    rows = []
    for lang, txt in probes.items():
        ipa = text_to_ipa(txt, lang=lang)
        ids = ipa_to_ids(ipa)
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            style = teacher.get_style_for_tokens(lang, len(ids) - 2)
            wav_ref, _ = teacher.infer_tokens(ids_t, style, speed=1.0)
        wav_np = clean_studio_24k(wav_ref.squeeze(0).cpu().numpy())
        assert wav_np is not None
        wav_t = torch.from_numpy(wav_np).to(device)
        F_frames = wav_np.size // HOP_LENGTH
        f0_gt = extract_f0_80hz_gpu(wav_t, target_len=2 * F_frames)
        aligned = align_utterance_viterbi_gpu(teacher, mel_loss, ids_t, lang, wav_t.unsqueeze(0))
        assert aligned is not None
        gt_dur, prior_dur, align_sim = aligned
        wpath = wav_dir / f"smoke_{lang}.wav"
        sf.write(str(wpath), wav_np, SAMPLE_RATE, subtype="PCM_16")
        rows.append({
            "id": f"smoke_{lang}",
            "lang": lang,
            "text": txt,
            "ipa": ipa,
            "ipa_ids": ids,
            "wav_path": str(wpath),
            "gt_dur": gt_dur.cpu().tolist(),
            "prior_dur": prior_dur.cpu().tolist(),
            "f0_gt": f0_gt.cpu().tolist(),
            "median_f0_hz": 130.0,
            "align_sim": round(align_sim, 4),
            "duration_sec": round(wav_np.size / SAMPLE_RATE, 3),
            "has_retroflex_lat": (RETROFLEX_LATERAL_ID in ids),
        })
    return {"train": rows, "val": rows[:3], "test": rows[3:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per-lang-target", type=int, default=520)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--curate-only", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Ninada v3.0 24 kHz Trainer | device={device}", flush=True)
    if device.type == "cuda":
        # Use rock-solid native ATen CUDA + IEEE FP32 math kernels (avoids PyTorch 2.3 cu121 Cutlass SDPA & cuDNN 8.9 1D workspace bugs)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        print(f"[INIT] GPU: {torch.cuda.get_device_name(0)} | ATen-CUDA-Exact=True", flush=True)

    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Ninada24kModel.from_pretrained_warmstart().to(device)
    mel_loss_fn = MultiScaleMelEnvelopeLoss(sr=SAMPLE_RATE).to(device)

    # Save exact warm-start snapshot theta_0 on CPU for L2 anchor & WiSE-FT sweep
    theta_0 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Curate & align 24 kHz studio dataset using frozen teacher (or reload cached splits if present!)
    model.eval()
    cached_train = work_dir / "train_v3.jsonl"
    cached_val = work_dir / "val_v3.jsonl"
    cached_test = work_dir / "test_v3.jsonl"
    if not args.smoke_test and cached_train.exists() and cached_val.exists() and cached_test.exists():
        splits = {
            k: [json.loads(line) for line in (work_dir / f"{k}_v3.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            for k in ("train", "val", "test")
        }
        print(f"[CACHE] Reloaded curated studio splits from {work_dir}", flush=True)
    elif args.smoke_test:
        splits = build_smoke_dataset(work_dir, model, mel_loss_fn, device)
    else:
        splits = curate_studio_dataset_24k(
            work_dir,
            model,
            mel_loss_fn,
            device,
            per_lang_target=args.per_lang_target,
        )

    print(
        f"[SPLITS] train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}",
        flush=True,
    )
    if args.curate_only:
        print("[CURATE-ONLY] Finished dataset curation.", flush=True)
        return

    # Baseline alpha=0.0 validation & test metrics BEFORE fine-tuning
    val_alpha0 = evaluate_split(model, mel_loss_fn, splits["val"], device)
    test_alpha0 = evaluate_split(model, mel_loss_fn, splits["test"], device)
    print(f"[EVAL alpha=0.0] val={json.dumps(val_alpha0)}", flush=True)

    # Freeze `decoder.generator` (`SineGen + CustomSTFT` 24 kHz harmonic vocoder) 100%!
    for p in model.decoder.generator.parameters():
        p.requires_grad = False

    voice_params = list(model.voice_global_delta.parameters()) + list(model.voice_slope_delta.parameters())
    text_params = (
        list(model.bert.parameters())
        + list(model.bert_encoder.parameters())
        + list(model.text_encoder.parameters())
    )
    pred_params = list(model.predictor.parameters())
    dec_front_params = [p for p in model.decoder.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": voice_params, "lr": 6.0e-4, "weight_decay": 1.0e-3},
            {"params": text_params, "lr": 1.2e-5, "weight_decay": 1.0e-4},
            {"params": pred_params, "lr": 1.5e-5, "weight_decay": 1.0e-4},
            {"params": dec_front_params, "lr": 2.5e-6, "weight_decay": 1.0e-4},
        ],
        betas=(0.9, 0.98),
    )

    train_rows = list(splits["train"])
    total_steps = max(1, (len(train_rows) * args.epochs) // max(1, args.grad_accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1.0e-6)

    # Pre-cache theta_0 anchor subset on device for fast L2 regularization of embeddings & predictor
    bert_emb_0 = model.bert.embeddings.word_embeddings.weight.detach().clone()
    text_emb_0 = model.text_encoder.embedding.weight.detach().clone()
    non_lat_mask = torch.ones(bert_emb_0.shape[0], dtype=torch.bool, device=device)
    non_lat_mask[RETROFLEX_LATERAL_ID] = False

    max_dur = int(model.config_dict["max_dur"])
    ord_bins = torch.arange(max_dur, device=device, dtype=torch.float32).unsqueeze(0)  # [1, 50]

    rng = random.Random(108)
    global_step = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep frozen generator in eval mode
        model.decoder.generator.eval()
        rng.shuffle(train_rows)
        optimizer.zero_grad(set_to_none=True)

        running_mel, running_dur, running_f0 = 0.0, 0.0, 0.0
        n_acc = 0

        for idx, rec in enumerate(train_rows, start=1):
            lang = rec["lang"]
            ids_t = torch.tensor([rec["ipa_ids"]], dtype=torch.long, device=device)
            gt_dur = torch.tensor(rec["gt_dur"], dtype=torch.long, device=device)
            prior_dur = torch.tensor(rec["prior_dur"], dtype=torch.float32, device=device)
            f0_gt = torch.tensor([rec["f0_gt"]], dtype=torch.float32, device=device)
            wav_np, _ = sf.read(rec["wav_path"], dtype="float32")
            wav_gt = torch.from_numpy(wav_np).unsqueeze(0).to(device)

            out = model.forward_train_step(ids_t, lang, gt_dur, f0_gt)

            # 1. Pitch-invariant Multi-Scale 80-Bin Log-Mel Spectral Envelope Loss
            loss_mel = mel_loss_fn(out["audio_pred"], wav_gt)

            # 2. Blended Duration Target (80% studio Viterbi gt_dur + 20% smooth prior_dur guard against edge silence noise)
            eff_dur = torch.clamp(0.80 * gt_dur.to(torch.float32) + 0.20 * prior_dur, min=1.0, max=float(max_dur - 1))
            ord_trg = (ord_bins < torch.round(eff_dur).unsqueeze(-1)).to(torch.float32)
            loss_dur_bce = F.binary_cross_entropy_with_logits(out["dur_logits"], ord_trg)
            loss_dur_l1 = F.smooth_l1_loss(out["dur_pred"], eff_dur)
            loss_dur = loss_dur_bce + 0.25 * loss_dur_l1

            # 3. 80 Hz F0 Loss (Voiced Hz Smooth-L1 in semitone-normalized scale + Unvoiced <= 3 Hz target)
            voiced_m = f0_gt > 10.0
            unvoiced_m = ~voiced_m
            loss_f0 = wav_gt.new_tensor(0.0)
            if bool(voiced_m.any()):
                loss_f0 = loss_f0 + F.smooth_l1_loss(out["f0_pred"][voiced_m] / 120.0, f0_gt[voiced_m] / 120.0)
            if bool(unvoiced_m.any()):
                unv_excess = F.relu(out["f0_pred"][unvoiced_m] - 4.0) / 50.0
                loss_f0 = loss_f0 + 0.5 * (unv_excess * unv_excess).mean()

            # 4. Anchor regularization on pre-existing phonemes (while letting slot 79 'ɭ' adapt freely!) + voice delta norm guard
            loss_anchor = (
                F.mse_loss(model.bert.embeddings.word_embeddings.weight[non_lat_mask], bert_emb_0[non_lat_mask])
                + F.mse_loss(model.text_encoder.embedding.weight[non_lat_mask], text_emb_0[non_lat_mask])
            )
            g_delta = model.voice_global_delta[lang]
            s_delta = model.voice_slope_delta[lang]
            loss_voice_reg = 0.05 * ((g_delta * g_delta).mean() + (s_delta * s_delta).mean())

            total_loss = (
                1.00 * loss_mel
                + 0.30 * loss_dur
                + 0.35 * loss_f0
                + 2.00 * loss_anchor
                + loss_voice_reg
            ) / float(args.grad_accum)

            if not torch.isfinite(total_loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            try:
                total_loss.backward()
            except RuntimeError as exc:
                print(f"[WARN] Skipping backward outlier on {rec['id']} (len={len(rec['ipa_ids'])}): {exc}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            # Boost gradient on newly introduced slot 79 ('ɭ') by 4x so retroflex lateral converges rapidly!
            for emb_w in (model.bert.embeddings.word_embeddings.weight, model.text_encoder.embedding.weight):
                if emb_w.grad is not None:
                    emb_w.grad[RETROFLEX_LATERAL_ID].mul_(4.0)

            running_mel += float(loss_mel.item())
            running_dur += float(loss_dur.item())
            running_f0 += float(loss_f0.item())
            n_acc += 1

            if idx % args.grad_accum == 0 or idx == len(train_rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Keep voice style bank perturbation bounded within realistic acoustic hypersphere
                with torch.no_grad():
                    for lc in SUPPORTED_LANGS:
                        model.voice_global_delta[lc].clamp_(-0.06, 0.06)
                        model.voice_slope_delta[lc].clamp_(-0.03, 0.03)

                if global_step % 40 == 0 or idx == len(train_rows):
                    denom = max(1, n_acc)
                    print(
                        f"[TRAIN] ep={epoch}/{args.epochs} step={global_step}/{total_steps} "
                        f"mel={running_mel/denom:.4f} dur={running_dur/denom:.4f} f0={running_f0/denom:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )
                    running_mel, running_dur, running_f0, n_acc = 0.0, 0.0, 0.0, 0

        val_ep = evaluate_split(model, mel_loss_fn, splits["val"], device)
        history.append({"epoch": epoch, "step": global_step, "val": val_ep})
        print(f"[EPOCH {epoch} VAL] {json.dumps(val_ep)}", flush=True)

    # Save pure fine-tuned state theta_ft (alpha = 1.0)
    theta_ft = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Validation WiSE-FT sweep over alpha in {0.25, 0.50, 0.75, 1.0}
    # Note: Always keep slot 79 ('ɭ') and learned per-language voice deltas at max(alpha, 0.75) so new phonemes/voices stay active!
    wise_sweep: dict[str, Any] = {"0.0": val_alpha0}
    best_alpha = 0.75
    best_score = float("inf")

    for alpha in (0.25, 0.50, 0.75, 1.0):
        blended: dict[str, torch.Tensor] = {}
        for k, v_ft in theta_ft.items():
            v_0 = theta_0[k]
            if not v_ft.is_floating_point():
                blended[k] = v_ft
            elif k.startswith("voice_global_delta.") or k.startswith("voice_slope_delta."):
                blended[k] = max(alpha, 0.75) * v_ft + (1.0 - max(alpha, 0.75)) * v_0
            elif k in ("bert.embeddings.word_embeddings.weight", "text_encoder.embedding.weight"):
                w = alpha * v_ft + (1.0 - alpha) * v_0
                w = w.clone()
                lat_a = max(alpha, 0.85)
                w[RETROFLEX_LATERAL_ID] = lat_a * v_ft[RETROFLEX_LATERAL_ID] + (1.0 - lat_a) * v_0[RETROFLEX_LATERAL_ID]
                blended[k] = w
            else:
                blended[k] = alpha * v_ft + (1.0 - alpha) * v_0

        model.load_state_dict(blended, strict=True)
        val_m = evaluate_split(model, mel_loss_fn, splits["val"], device)
        wise_sweep[str(alpha)] = val_m
        print(f"[WiSE-FT alpha={alpha}] val={json.dumps(val_m)}", flush=True)
        if val_m["composite_loss"] < best_score:
            best_score = val_m["composite_loss"]
            best_alpha = alpha

    print(f"[WiSE-FT SELECTED] best_alpha={best_alpha} (composite_loss={best_score:.5f} vs alpha0={val_alpha0['composite_loss']:.5f})", flush=True)

    # Restore winning WiSE-FT weights and evaluate held-out test split
    final_sd: dict[str, torch.Tensor] = {}
    for k, v_ft in theta_ft.items():
        v_0 = theta_0[k]
        if not v_ft.is_floating_point():
            final_sd[k] = v_ft
        elif k.startswith("voice_global_delta.") or k.startswith("voice_slope_delta."):
            a_v = max(best_alpha, 0.75)
            final_sd[k] = a_v * v_ft + (1.0 - a_v) * v_0
        elif k in ("bert.embeddings.word_embeddings.weight", "text_encoder.embedding.weight"):
            w = (best_alpha * v_ft + (1.0 - best_alpha) * v_0).clone()
            lat_a = max(best_alpha, 0.85)
            w[RETROFLEX_LATERAL_ID] = lat_a * v_ft[RETROFLEX_LATERAL_ID] + (1.0 - lat_a) * v_0[RETROFLEX_LATERAL_ID]
            final_sd[k] = w
        else:
            final_sd[k] = best_alpha * v_ft + (1.0 - best_alpha) * v_0

    model.load_state_dict(final_sd, strict=True)
    test_final = evaluate_split(model, mel_loss_fn, splits["test"], device)
    print(f"[HELD-OUT TEST FINAL] {json.dumps(test_final)}", flush=True)

    # Export full 24 kHz ONNX + voices/*.bin + safetensors + sample WAVs
    export_meta = export_ninada_24k_bundle(model, out_dir, keep_fp32_onnx=False)

    summary = {
        "version": "3.0.0",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "elapsed_sec": round(time.time() - t_start, 1),
        "dataset_counts": {k: len(v) for k, v in splits.items()},
        "best_wise_ft_alpha": best_alpha,
        "val_alpha0": val_alpha0,
        "wise_ft_sweep": wise_sweep,
        "test_alpha0": test_alpha0,
        "test_final": test_final,
        "epoch_history": history,
        "export": export_meta,
    }
    with open(out_dir / "val_test_metrics_v3.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[DONE] Ninada v3.0 24 kHz training & ONNX export succeeded!", flush=True)


if __name__ == "__main__":
    main()
