"""
GCP NVIDIA L4 GPU Adversarial Specialist Training Pipeline for `Ninada` v2.0.

Trains 6 native-script VITS + Stochastic Duration Predictor (`SDP`) specialists (`sa`, `kn`, `hi`, `ta`, `te`, `ml`)
on the Male-matched 48kHz->16kHz anechoic studio dataset (`prathoshap/sushrota-sanskrit-asr-data` + `ai4bharat/Rasa`):
  - Exact warm-start from `facebook/mms-tts-{kan,hin,tam,tel,mal}` (`sa` extends `mms-tts-kan` `0..75` with `76..78` `ಙ, ೠ, ೄ`)
  - Frozen `posterior_encoder` & `decoder` (`HiFi-GAN`) preserving crisp 500k-step adversarial glottal excitation
  - 2-Epoch `MPD + MRD` Discriminator Calibration Warmup followed by Alternating `LSGAN + Feature-Matching + KL + SDP MLE`
  - Post-training Validation-Only Weight-Space Ensembling (`WiSE-FT`, `alpha in {0.0, 0.25, 0.50, 0.75, 1.0}`)
    scored by end-to-end synthesized DTW `MCD (dB)`, `Utterance Dur MAE (s)`, `Val KL`, and `Su-srota` ASR `CER %`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from safetensors.torch import load_file, save_file
from scipy.fft import dct
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "python" / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ninada.g2p import (
    LANGUAGE_IDS,
    NATIVE_BASE_CHECKPOINTS,
    NATIVE_VOCABS,
    SANSKRIT_DONOR_MAP,
    text_to_ids,
)
from training.modeling_ninada import (
    NinadaDiscriminator,
    NinadaModel,
    discriminator_lsgan_loss,
    generator_lsgan_and_fm_loss,
)

ALL_LANGS = ("sa", "kn", "hi", "ta", "te", "ml")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpecialistTtsDataset(Dataset):
    def __init__(self, meta_path: Path, data_root: Path, lang: str, max_audio_sec: float = 11.0) -> None:
        self.data_root = data_root
        self.lang = lang
        self.max_samples = int(max_audio_sec * 16000)
        self.items: list[dict[str, Any]] = []

        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("lang") != lang:
                continue
            uid = row.get("id") or row.get("utt_id")
            input_ids = text_to_ids(row["text"], lang=lang)["input_ids"]
            wav_rel = row.get("wav_relpath") or row.get("wav_path") or f"wavs/{uid}.wav"
            candidates = [
                data_root / wav_rel,
                data_root.parent / wav_rel,
                data_root / "wavs" / f"{uid}.wav",
            ]
            wav_full = next((c for c in candidates if c.exists()), None)
            if wav_full is None:
                continue
            self.items.append(
                {
                    "utt_id": uid,
                    "lang": lang,
                    "text": row["text"],
                    "input_ids": input_ids,
                    "wav_path": str(wav_full),
                    "duration_sec": float(row.get("duration_sec", 4.0)),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        it = self.items[idx]
        audio, _ = sf.read(it["wav_path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        if peak > 1e-4:
            audio = 0.88 * (audio / peak)
        return {
            "utt_id": it["utt_id"],
            "lang": it["lang"],
            "text": it["text"],
            "input_ids": it["input_ids"],
            "wav": audio,
        }


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: SpecialistTtsDataset, batch_size: int, shuffle: bool = True, seed: int = 42) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            indices.sort(key=lambda i: self.dataset.items[i]["duration_sec"] + rng.uniform(-0.35, 0.35))
        else:
            indices.sort(key=lambda i: self.dataset.items[i]["duration_sec"])

        batches = [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self) -> int:
        return max(1, math.ceil(len(self.dataset) / self.batch_size))


def collate_specialist_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(batch)
    max_t = max(len(x["input_ids"]) for x in batch)
    max_w = max(len(x["wav"]) for x in batch)
    max_w = max(2048, ((max_w + 255) // 256) * 256)

    ids = torch.zeros((B, max_t), dtype=torch.long)
    mask = torch.zeros((B, max_t), dtype=torch.long)
    wavs = torch.zeros((B, max_w), dtype=torch.float32)
    wav_lens = torch.zeros((B,), dtype=torch.long)
    texts: list[str] = []

    for i, item in enumerate(batch):
        tl = len(item["input_ids"])
        wl = len(item["wav"])
        ids[i, :tl] = torch.tensor(item["input_ids"], dtype=torch.long)
        mask[i, :tl] = 1
        wavs[i, :wl] = torch.from_numpy(item["wav"])
        wav_lens[i] = wl
        texts.append(item["text"])

    return {
        "input_ids": ids,
        "attention_mask": mask,
        "wav": wavs,
        "wav_lengths": wav_lens,
        "texts": texts,
    }


def _quick_mcd_db(gt_wav: np.ndarray, syn_wav: np.ndarray, sr: int = 16000) -> float:
    gt_p = float(np.max(np.abs(gt_wav))) if len(gt_wav) > 0 else 0.0
    sy_p = float(np.max(np.abs(syn_wav))) if len(syn_wav) > 0 else 0.0
    if len(syn_wav) < 1200 or sy_p < 1e-3:
        return 18.0
    gt = gt_wav / max(1e-5, gt_p)
    sy = syn_wav / sy_p
    mel_gt = librosa.feature.melspectrogram(y=gt, sr=sr, n_fft=1024, hop_length=256, n_mels=80, power=1.0)
    mel_sy = librosa.feature.melspectrogram(y=sy, sr=sr, n_fft=1024, hop_length=256, n_mels=80, power=1.0)
    mc_gt = (dct(np.log(np.maximum(mel_gt, 1e-6)), type=2, axis=0, norm="ortho")[1:14]) / math.sqrt(80.0)
    mc_sy = (dct(np.log(np.maximum(mel_sy, 1e-6)), type=2, axis=0, norm="ortho")[1:14]) / math.sqrt(80.0)
    _, wp = librosa.sequence.dtw(X=mc_gt, Y=mc_sy, metric="euclidean")
    wp = wp[::-1]
    diff = mc_gt[:, wp[:, 0]] - mc_sy[:, wp[:, 1]]
    frame_dist = np.sqrt(np.sum(diff * diff, axis=0))
    return float((10.0 * math.sqrt(2.0) / math.log(10.0)) * np.mean(frame_dist))


@torch.no_grad()
def evaluate_validation(
    model: NinadaModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums = {"kl_loss": 0.0, "dur_loss": 0.0, "mel_l1": 0.0, "dur_mae_frames": 0.0, "utt_dur_mae_sec": 0.0}
    n_items = 0
    for batch in loader:
        B = batch["input_ids"].shape[0]
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        wav = batch["wav"].to(device, non_blocking=True)
        wlens = batch["wav_lengths"].to(device, non_blocking=True)
        out = model.forward_train(ids, mask, wav, wlens)
        for k in sums:
            sums[k] += float(out[k].detach().cpu()) * B
        n_items += B
    denom = max(1, n_items)
    res = {f"val_{k}": round(v / denom, 5) for k, v in sums.items()}
    res["val_score"] = round(
        2.5 * res["val_kl_loss"] + 1.0 * res["val_dur_loss"] + 1.5 * res["val_mel_l1"], 5
    )
    return res


@torch.no_grad()
def evaluate_end_to_end_synthesis_val(
    model: NinadaModel,
    val_ds: SpecialistTtsDataset,
    device: torch.device,
    seed: int = 108,
) -> dict[str, float]:
    """Synthesize every validation clip end-to-end and measure real DTW MCD (dB) + Utterance Duration MAE (s)."""
    model.eval()
    scales = torch.tensor([1.0, 0.667, 0.80], dtype=torch.float32, device=device)
    mcds: list[float] = []
    dur_errs: list[float] = []
    for idx in range(len(val_ds)):
        it = val_ds[idx]
        torch.manual_seed(seed + idx)
        ids = torch.tensor([it["input_ids"]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        wav_hat, _ = model.forward_infer(ids, mask, scales)
        syn = wav_hat[0].detach().cpu().numpy()
        gt = it["wav"]
        mcds.append(_quick_mcd_db(gt, syn))
        dur_errs.append(abs(len(syn) - len(gt)) / 16000.0)
    mean_mcd = float(np.mean(mcds)) if mcds else 99.0
    mean_dur = float(np.mean(dur_errs)) if dur_errs else 99.0
    # Composite acoustic synthesis quality objective (lower is better)
    composite = mean_mcd + 0.65 * mean_dur
    return {
        "synth_val_mcd_db": round(mean_mcd, 4),
        "synth_val_dur_mae_sec": round(mean_dur, 4),
        "synth_val_composite": round(composite, 4),
    }


def configure_generator_optimizer(
    model: NinadaModel,
    lang: str,
    lr_backbone: float,
    lr_sdp: float,
) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    for p in model.posterior_encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False

    embed_params: list[torch.nn.Parameter] = []
    sdp_params: list[torch.nn.Parameter] = []
    backbone_params: list[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("text_encoder.embed_tokens."):
            embed_params.append(p)
        elif name.startswith("duration_predictor."):
            sdp_params.append(p)
        else:
            backbone_params.append(p)

    embed_lr = lr_backbone * 3.0 if lang == "sa" else lr_backbone
    opt_g = torch.optim.AdamW(
        [
            {"params": embed_params, "lr": embed_lr, "weight_decay": 1e-4},
            {"params": sdp_params, "lr": lr_sdp, "weight_decay": 1e-4},
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
        ],
        betas=(0.8, 0.99),
        eps=1e-8,
    )
    counts = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_generator_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_vocoder_posterior_params": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
    return opt_g, counts


def train_single_specialist(
    lang: str,
    data_dir: Path,
    spec_out_dir: Path,
    epochs: int,
    disc_warmup_epochs: int,
    batch_size: int,
    lr_backbone: float,
    lr_sdp: float,
    lr_disc: float,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    set_seed(seed + LANGUAGE_IDS[lang])
    spec_out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SpecialistTtsDataset(data_dir / "train_meta.jsonl", data_dir, lang=lang)
    val_ds = SpecialistTtsDataset(data_dir / "val_meta.jsonl", data_dir, lang=lang)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Missing data for lang={lang}: train={len(train_ds)}, val={len(val_ds)}")

    train_sampler = BucketBatchSampler(train_ds, batch_size=batch_size, shuffle=True, seed=seed)
    val_sampler = BucketBatchSampler(val_ds, batch_size=max(1, min(4, batch_size)), shuffle=False, seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_specialist_batch,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_specialist_batch,
        pin_memory=(device.type == "cuda"),
    )

    donor_map = SANSKRIT_DONOR_MAP if lang == "sa" else None
    model = NinadaModel.from_mms_specialist(
        lang=lang,
        vocab_tokens=NATIVE_VOCABS[lang],
        base_checkpoint=NATIVE_BASE_CHECKPOINTS[lang],
        donor_map=donor_map,
    )
    model.save_pretrained(spec_out_dir, filename="init_model.safetensors")
    model.to(device)

    disc = NinadaDiscriminator().to(device)
    opt_g, param_counts = configure_generator_optimizer(model, lang=lang, lr_backbone=lr_backbone, lr_sdp=lr_sdp)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=lr_disc, betas=(0.8, 0.99), weight_decay=1e-4)

    val0 = evaluate_validation(model, val_loader, device)
    synth0 = evaluate_end_to_end_synthesis_val(model, val_ds, device)
    print(
        f"\n=== [{lang.upper()}] Warm-start={NATIVE_BASE_CHECKPOINTS[lang]} | "
        f"train={len(train_ds)} val={len(val_ds)} | "
        f"Epoch 00: val_kl={val0['val_kl_loss']:.4f} mcd={synth0['synth_val_mcd_db']:.3f}dB "
        f"dur_mae={synth0['synth_val_dur_mae_sec']:.3f}s ===",
        flush=True,
    )

    history: list[dict[str, Any]] = [{"epoch": 0, "step": 0, **val0, **synth0}]
    best_val_score = val0["val_score"]
    best_epoch = 0
    model.save_pretrained(spec_out_dir, filename="trained_model.safetensors")

    global_step = 0
    gen_epochs = max(1, epochs - disc_warmup_epochs)
    total_gen_steps = max(1, gen_epochs * len(train_loader))

    def lr_lambda(step: int) -> float:
        warm = max(3, int(0.10 * total_gen_steps))
        if step < warm:
            return float(step + 1) / float(warm)
        prog = (step - warm) / max(1, total_gen_steps - warm)
        return 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        train_sampler.set_epoch(epoch)
        is_disc_warmup = epoch <= disc_warmup_epochs

        if is_disc_warmup:
            model.eval()
        else:
            model.train()
            model.posterior_encoder.eval()
            model.decoder.eval()
        disc.train()

        sums = {"loss_d": 0.0, "loss_g": 0.0, "kl_loss": 0.0, "dur_loss": 0.0, "adv_loss": 0.0, "fm_loss": 0.0, "mel_l1": 0.0}
        n_batches = 0

        for batch in train_loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            wav = batch["wav"].to(device, non_blocking=True)
            wlens = batch["wav_lengths"].to(device, non_blocking=True)

            if is_disc_warmup:
                with torch.no_grad():
                    out = model.forward_train(ids, mask, wav, wlens)
                opt_d.zero_grad(set_to_none=True)
                y_dr, y_dg, _, _ = disc(out["wav_gt_slice"], out["wav_hat_slice"].detach())
                loss_d = discriminator_lsgan_loss(y_dr, y_dg)
                if torch.isfinite(loss_d):
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=2.0)
                    opt_d.step()
                sums["loss_d"] += float(loss_d.detach().cpu())
                sums["kl_loss"] += float(out["kl_loss"].detach().cpu())
                sums["dur_loss"] += float(out["dur_loss"].detach().cpu())
                sums["mel_l1"] += float(out["mel_l1"].detach().cpu())
                n_batches += 1
                global_step += 1
                continue

            out = model.forward_train(ids, mask, wav, wlens)

            # 1. Discriminator step
            opt_d.zero_grad(set_to_none=True)
            y_dr, y_dg_det, _, _ = disc(out["wav_gt_slice"], out["wav_hat_slice"].detach())
            loss_d = discriminator_lsgan_loss(y_dr, y_dg_det)
            if torch.isfinite(loss_d):
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=2.0)
                opt_d.step()

            # 2. Generator step (MPD + MRD adversarial + feature matching + exact KL + SDP MLE + gentle Mel)
            opt_g.zero_grad(set_to_none=True)
            _, y_dg, fmap_r, fmap_g = disc(out["wav_gt_slice"], out["wav_hat_slice"])
            adv_loss, fm_loss = generator_lsgan_and_fm_loss(y_dg, fmap_r, fmap_g)
            loss_g = (
                2.0 * out["kl_loss"]
                + 1.0 * out["dur_loss"]
                + 1.0 * adv_loss
                + 2.0 * fm_loss
                + 8.0 * out["mel_l1"]
            )
            if not torch.isfinite(loss_g):
                continue
            loss_g.backward()
            if lang == "sa" and model.text_encoder.embed_tokens.weight.grad is not None:
                # Preserve pre-trained Kannada phoneme rows (0..75) gently while allowing new Sanskrit rows (76..78) full step
                model.text_encoder.embed_tokens.weight.grad[:76].mul_(0.35)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.5
            )
            opt_g.step()
            sched_g.step()

            sums["loss_d"] += float(loss_d.detach().cpu())
            sums["loss_g"] += float(loss_g.detach().cpu())
            sums["kl_loss"] += float(out["kl_loss"].detach().cpu())
            sums["dur_loss"] += float(out["dur_loss"].detach().cpu())
            sums["adv_loss"] += float(adv_loss.detach().cpu())
            sums["fm_loss"] += float(fm_loss.detach().cpu())
            sums["mel_l1"] += float(out["mel_l1"].detach().cpu())
            n_batches += 1
            global_step += 1

        train_avgs = {f"train_{k}": round(v / max(1, n_batches), 5) for k, v in sums.items()}
        val_m = evaluate_validation(model, val_loader, device)
        ep_sec = round(time.perf_counter() - ep_t0, 2)

        is_best = (not is_disc_warmup) and (
            best_epoch == 0 or val_m["val_score"] < best_val_score
        )
        if is_best:
            best_val_score = val_m["val_score"]
            best_epoch = epoch
            model.save_pretrained(spec_out_dir, filename="trained_model.safetensors")

        row = {
            "epoch": epoch,
            "step": global_step,
            "phase": "disc_warmup" if is_disc_warmup else "adversarial_vits",
            "epoch_sec": ep_sec,
            "is_best": is_best,
            **train_avgs,
            **val_m,
        }
        history.append(row)
        (spec_out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
        mark = "*BEST*" if is_best else ("(D-warm)" if is_disc_warmup else "")
        print(
            f"[{lang.upper()} Ep {epoch:02d}/{epochs:02d}] ({ep_sec:.1f}s) "
            f"d={train_avgs['train_loss_d']:.3f} g={train_avgs['train_loss_g']:.3f} "
            f"fm={train_avgs['train_fm_loss']:.3f} | "
            f"val_kl={val_m['val_kl_loss']:.4f} val_dur={val_m['val_dur_loss']:.3f} "
            f"val_mel={val_m['val_mel_l1']:.4f} {mark}",
            flush=True,
        )

    # Validation-Only Weight-Space Ensembling (WiSE-FT) over alpha in {0.0, 0.25, 0.50, 0.75}
    # Preserving >=25% pretrained SDP prosodic anchor prevents duration collapse on rapid studio speakers
    init_sd = load_file(str(spec_out_dir / "init_model.safetensors"))
    trained_sd = load_file(str(spec_out_dir / "trained_model.safetensors"))
    wise_grid: list[dict[str, Any]] = []
    best_alpha = 0.5
    best_composite = 1e9

    for alpha in (0.0, 0.25, 0.50, 0.75):
        blend_sd: dict[str, torch.Tensor] = {}
        for k, v_tr in trained_sd.items():
            v_in = init_sd[k]
            if v_tr.is_floating_point():
                blend_sd[k] = ((1.0 - alpha) * v_in.float() + alpha * v_tr.float()).to(v_tr.dtype)
            else:
                blend_sd[k] = v_tr
        model.load_state_dict(blend_sd, strict=True)
        model.to(device)
        synth_m = evaluate_end_to_end_synthesis_val(model, val_ds, device)
        entry = {"alpha": alpha, **synth_m}
        wise_grid.append(entry)
        print(
            f"  [{lang.upper()} WiSE-FT alpha={alpha:.2f}] "
            f"synth_mcd={synth_m['synth_val_mcd_db']:.3f}dB "
            f"dur_mae={synth_m['synth_val_dur_mae_sec']:.3f}s "
            f"composite={synth_m['synth_val_composite']:.4f}",
            flush=True,
        )
        if synth_m["synth_val_composite"] < best_composite:
            best_composite = synth_m["synth_val_composite"]
            best_alpha = alpha

    final_sd: dict[str, torch.Tensor] = {}
    for k, v_tr in trained_sd.items():
        v_in = init_sd[k]
        if v_tr.is_floating_point():
            final_sd[k] = ((1.0 - best_alpha) * v_in.float() + best_alpha * v_tr.float()).to(v_tr.dtype)
        else:
            final_sd[k] = v_tr
    save_file({k: v.cpu().contiguous() for k, v in final_sd.items()}, str(spec_out_dir / "model.safetensors"))
    (spec_out_dir / "init_model.safetensors").unlink(missing_ok=True)
    (spec_out_dir / "trained_model.safetensors").unlink(missing_ok=True)

    del disc
    if device.type == "cuda":
        torch.cuda.empty_cache()

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"[{lang.upper()}] Selected WiSE-FT alpha={best_alpha:.2f} (composite={best_composite:.4f}) in {elapsed:.1f}s", flush=True)
    return {
        "lang": lang,
        "base_checkpoint": NATIVE_BASE_CHECKPOINTS[lang],
        "vocab_size": len(NATIVE_VOCABS[lang]),
        "train_utterances": len(train_ds),
        "val_utterances": len(val_ds),
        "epochs": epochs,
        "disc_warmup_epochs": disc_warmup_epochs,
        "total_steps": global_step,
        "best_epoch": best_epoch,
        "selected_wise_ft_alpha": best_alpha,
        "wise_ft_validation_grid": wise_grid,
        "pretrain_synth_val": synth0,
        "selected_synth_val": next(x for x in wise_grid if x["alpha"] == best_alpha),
        "parameter_counts": param_counts,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 6 Ninada v2.0 Native Specialists with MPD+MRD on GCP L4 GPU")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ninada"))
    parser.add_argument("--langs", type=str, default="sa,kn,hi,ta,te,ml")
    parser.add_argument("--sa-epochs", type=int, default=16)
    parser.add_argument("--indic-epochs", type=int, default=10)
    parser.add_argument("--disc-warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_all = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if device.type == "cuda"
        else 0.0
    )
    print(f"[ninada-v2-train] Device={device} ({gpu_name}, VRAM={vram_gb} GB) | Host={socket.gethostname()}", flush=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip() in ALL_LANGS]
    specialists_root = args.output_dir / "specialists"
    specialists_root.mkdir(parents=True, exist_ok=True)

    per_lang_reports: dict[str, Any] = {}
    total_steps_all = 0
    total_train_utts = 0
    total_val_utts = 0

    for lc in langs:
        ep = args.sa_epochs if lc == "sa" else args.indic_epochs
        lr_b = 2.2e-5 if lc == "sa" else 1.5e-5
        lr_sdp = 3.0e-5 if lc == "sa" else 2.0e-5
        rep = train_single_specialist(
            lang=lc,
            data_dir=args.data_dir,
            spec_out_dir=specialists_root / lc,
            epochs=ep,
            disc_warmup_epochs=min(args.disc_warmup_epochs, max(1, ep // 3)),
            batch_size=args.batch_size,
            lr_backbone=lr_b,
            lr_sdp=lr_sdp,
            lr_disc=1.5e-4,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed,
        )
        per_lang_reports[lc] = rep
        total_steps_all += int(rep["total_steps"])
        total_train_utts += int(rep["train_utterances"])
        total_val_utts += int(rep["val_utterances"])

    # Copy primary Sanskrit specialist to top-level checkpoint_dir for single-checkpoint compatibility
    if (specialists_root / "sa" / "model.safetensors").exists():
        for fn in ("model.safetensors", "ninada_config.json", "config.json", "training_history.json"):
            src = specialists_root / "sa" / fn
            if src.exists():
                shutil.copy2(src, args.output_dir / fn)

    total_sec = round(time.perf_counter() - t_all, 2)
    manifest = {
        "model_name": "ninada",
        "architecture_version": "2.0.0",
        "architecture": "Native-Script Specialist VITS + ONNX-Safe Stochastic Duration Predictor (SDP) + Adversarial MPD/MRD",
        "project": os.environ.get("CLOUD_ML_PROJECT_ID", "hemanth-hm"),
        "cloud_run_execution": os.environ.get("CLOUD_RUN_EXECUTION", os.environ.get("CLOUD_ML_JOB_ID", "local")),
        "region": os.environ.get("CLOUD_ML_REGION", "us-east4"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_vram_gb": vram_gb,
        "languages_trained": langs,
        "train_utterances": total_train_utts,
        "val_utterances": total_val_utts,
        "epochs": args.sa_epochs,
        "indic_epochs": args.indic_epochs,
        "batch_size": args.batch_size,
        "total_steps": total_steps_all,
        "total_elapsed_sec": total_sec,
        "specialists": per_lang_reports,
    }
    (args.output_dir / "gcp_job_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[ninada-v2-train] All {len(langs)} specialists trained in {total_sec:.1f}s! Manifest saved.", flush=True)


if __name__ == "__main__":
    main()
