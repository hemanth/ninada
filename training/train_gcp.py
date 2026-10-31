"""
GCP NVIDIA L4 GPU Training Pipeline for `Ninada` (Disentangled Pan-Indic & Classical Sanskrit VITS + HiFi-GAN).

Trains `NinadaModel` on the curated 9-corpus Hugging Face studio speech dataset (`1,061` train / `133` val):
- Sanskrit (`sa`): `prathoshap/sushrota-sanskrit-asr-data`, `surajp/shrutilipi_sanskrit`,
  `addy88/sanskrit-asr-84`, `yashnbx/gita_supersite_sanskrit_tts`
- Hindi (`hi`): `SayantanJoker/SYSPIN_Hindi_Female_TTS`
- Kannada (`kn`): `SayantanJoker/original_data_kannada_tts`
- Tamil (`ta`): `SayantanJoker/original_data_tamil_tts`
- Telugu (`te`): `arpit-tiwari/syspin-telugu-tts`
- Malayalam (`ml`): `thennal/indic_tts_ml`
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

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "python" / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ninada.g2p import LANGUAGE_IDS, VOCAB_TOKENS, WARMSTART_DONOR_MAP, text_to_ids
from training.modeling_ninada import NinadaConfig, NinadaModel

PRIMARY_STUDIO_DOMAINS = {
    "sanskrit_recitation_prose",
    "hindi_studio",
    "kannada_studio",
    "tamil_studio",
    "telugu_studio",
    "malayalam_studio",
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IndicTtsDataset(Dataset):
    def __init__(self, meta_path: Path, data_root: Path, max_audio_sec: float = 12.5) -> None:
        self.data_root = data_root
        self.max_samples = int(max_audio_sec * 16000)
        self.items: list[dict[str, Any]] = []

        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row.get("id") or row.get("utt_id")
            input_ids = text_to_ids(row["text"], lang=row["lang"])["input_ids"]
            wav_rel = row.get("wav_relpath") or row.get("wav_path") or f"wavs/{uid}.wav"
            candidates = [
                data_root / wav_rel,
                data_root.parent / wav_rel,
                data_root / "wavs" / f"{uid}.wav",
            ]
            wav_full = next((c for c in candidates if c.exists()), None)
            if wav_full is None:
                continue
            domain = row.get("domain", "")
            self.items.append(
                {
                    "utt_id": uid,
                    "lang": row["lang"],
                    "lang_id": LANGUAGE_IDS[row["lang"]],
                    "is_primary_studio": domain in PRIMARY_STUDIO_DOMAINS,
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
            audio = 0.90 * (audio / peak)
        return {
            "utt_id": it["utt_id"],
            "lang": it["lang"],
            "lang_id": it["lang_id"],
            "is_primary_studio": it["is_primary_studio"],
            "input_ids": it["input_ids"],
            "wav": audio,
        }


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: IndicTtsDataset, batch_size: int, shuffle: bool = True, seed: int = 42) -> None:
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
            indices.sort(key=lambda i: self.dataset.items[i]["duration_sec"] + rng.uniform(-0.45, 0.45))
        else:
            indices.sort(key=lambda i: self.dataset.items[i]["duration_sec"])

        batches = [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)


def collate_tts_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(batch)
    max_t = max(len(x["input_ids"]) for x in batch)
    max_w = max(len(x["wav"]) for x in batch)
    max_w = max(2048, ((max_w + 255) // 256) * 256)

    ids = torch.zeros((B, max_t), dtype=torch.long)
    mask = torch.zeros((B, max_t), dtype=torch.long)
    lang_ids = torch.zeros((B,), dtype=torch.long)
    primary_flags = torch.zeros((B,), dtype=torch.bool)
    wavs = torch.zeros((B, max_w), dtype=torch.float32)
    wav_lens = torch.zeros((B,), dtype=torch.long)
    langs: list[str] = []

    for i, item in enumerate(batch):
        tl = len(item["input_ids"])
        wl = len(item["wav"])
        ids[i, :tl] = torch.tensor(item["input_ids"], dtype=torch.long)
        mask[i, :tl] = 1
        lang_ids[i] = int(item["lang_id"])
        primary_flags[i] = bool(item["is_primary_studio"])
        wavs[i, :wl] = torch.from_numpy(item["wav"])
        wav_lens[i] = wl
        langs.append(item["lang"])

    return {
        "input_ids": ids,
        "attention_mask": mask,
        "lang_id": lang_ids,
        "is_primary_studio": primary_flags,
        "wav": wavs,
        "wav_lengths": wav_lens,
        "langs": langs,
    }


@torch.no_grad()
def evaluate_validation(model: NinadaModel, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    keys = ("loss", "kl_loss", "latent_loss", "dur_loss", "mel_l1", "mr_stft", "dur_mae_frames", "utt_dur_mae_sec")
    sums = {k: 0.0 for k in keys}
    n_items = 0
    per_lang: dict[str, dict[str, float]] = {
        lc: {"count": 0.0, "kl_sum": 0.0, "latent_sum": 0.0, "dur_mae_sum": 0.0, "mel_sum": 0.0}
        for lc in LANGUAGE_IDS
    }

    for batch in loader:
        B = batch["input_ids"].shape[0]
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        lid = batch["lang_id"].to(device, non_blocking=True)
        wav = batch["wav"].to(device, non_blocking=True)
        wlens = batch["wav_lengths"].to(device, non_blocking=True)

        out = model.forward_train(ids, mask, lid, wav, wlens)
        for k in keys:
            sums[k] += float(out[k].detach().cpu()) * B
        n_items += B

        for lc in batch["langs"]:
            per_lang[lc]["count"] += 1.0
            per_lang[lc]["kl_sum"] += float(out["kl_loss"].detach().cpu())
            per_lang[lc]["latent_sum"] += float(out["latent_loss"].detach().cpu())
            per_lang[lc]["dur_mae_sum"] += float(out["dur_mae_frames"].detach().cpu())
            per_lang[lc]["mel_sum"] += float(out["mel_l1"].detach().cpu())

    denom = max(1, n_items)
    metrics: dict[str, Any] = {f"val_{k}": round(v / denom, 5) for k, v in sums.items()}
    metrics["per_lang"] = {
        lc: {
            "n": int(d["count"]),
            "val_kl": round(d["kl_sum"] / max(1.0, d["count"]), 4),
            "val_latent": round(d["latent_sum"] / max(1.0, d["count"]), 4),
            "val_dur_mae_frames": round(d["dur_mae_sum"] / max(1.0, d["count"]), 4),
            "val_mel_l1": round(d["mel_sum"] / max(1.0, d["count"]), 4),
        }
        for lc, d in per_lang.items()
        if d["count"] > 0
    }
    return metrics


def configure_trainable_parameters(
    model: NinadaModel,
    lr_head: float = 3.2e-4,
    lr_backbone: float = 4.8e-5,
) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    """
    Freeze adversarial HiFi-GAN (`decoder`) and `posterior_encoder` to preserve crisp consonant bursts,
    while training `dur_predictor`, conditioning layers, `utt_style_encoder`, `text_encoder`, and `flow`.
    """
    for p in model.posterior_encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False

    head_params: list[torch.nn.Parameter] = []
    backbone_params: list[torch.nn.Parameter] = []

    head_prefixes = (
        "dur_predictor.",
        "utt_style_encoder.",
        "lang_embed.",
        "lang_text_gate",
        "lang_dec_proj.",
        "text_encoder.embed_tokens.",
    )

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(pref) for pref in head_prefixes) or ".cond_layer." in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr_head, "weight_decay": 1e-4},
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
        ],
        betas=(0.85, 0.99),
        eps=1e-8,
    )

    counts = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_vocoder_posterior_params": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "head_group_params": sum(p.numel() for p in head_params),
        "backbone_group_params": sum(p.numel() for p in backbone_params),
    }
    return optimizer, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NinadaModel on 9-corpus HF studio dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ninada"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=3.2e-4)
    parser.add_argument("--lr-backbone", type=float, default=4.8e-5)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    t_start = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if device.type == "cuda"
        else 0.0
    )
    print(f"[ninada-train] Device={device} ({gpu_name}, VRAM={vram_gb} GB) | Host={socket.gethostname()}", flush=True)

    train_ds = IndicTtsDataset(args.data_dir / "train_meta.jsonl", args.data_dir)
    val_ds = IndicTtsDataset(args.data_dir / "val_meta.jsonl", args.data_dir)
    print(f"[ninada-train] Loaded train={len(train_ds)} clips, val={len(val_ds)} clips", flush=True)

    train_sampler = BucketBatchSampler(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_sampler = BucketBatchSampler(val_ds, batch_size=max(1, min(4, args.batch_size)), shuffle=False, seed=args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_tts_batch,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_tts_batch,
        pin_memory=(device.type == "cuda"),
    )

    cfg = NinadaConfig()
    model = NinadaModel.from_mms_warmstart(cfg, VOCAB_TOKENS, WARMSTART_DONOR_MAP)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Save exact Epoch-0 warm-start anchor for post-training validation WiSE-FT selection
    model.save_pretrained(args.output_dir, filename="init_model.safetensors")
    model.to(device)

    optimizer, param_counts = configure_trainable_parameters(
        model, lr_head=args.lr_head, lr_backbone=args.lr_backbone
    )
    print(f"[ninada-train] Parameter summary: {json.dumps(param_counts)}", flush=True)

    steps_per_epoch = (
        len(train_loader) if args.max_steps_per_epoch <= 0 else min(len(train_loader), args.max_steps_per_epoch)
    )
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(4, int(0.08 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    val0 = evaluate_validation(model, val_loader, device)
    print(
        f"[Epoch 00/{args.epochs:02d}] (pre-train) val_loss={val0['val_loss']:.4f} "
        f"val_kl={val0['val_kl_loss']:.4f} val_lat={val0['val_latent_loss']:.4f} "
        f"val_mel={val0['val_mel_l1']:.4f} val_dur_mae={val0['val_dur_mae_frames']:.3f}f "
        f"({val0['val_utt_dur_mae_sec']:.2f}s)",
        flush=True,
    )

    history: list[dict[str, Any]] = [{"epoch": 0, "step": 0, **val0}]
    best_val_score = float(
        3.0 * val0["val_kl_loss"] + 14.0 * val0["val_latent_loss"] + 2.5 * val0["val_dur_loss"] + 1.5 * val0["val_mel_l1"]
    )
    best_epoch = 0
    model.save_pretrained(args.output_dir, filename="model.safetensors")

    keys = ("loss", "kl_loss", "latent_loss", "dur_loss", "mel_l1", "mr_stft", "dur_mae_frames", "utt_dur_mae_sec")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        train_sampler.set_epoch(epoch)
        model.train()
        model.posterior_encoder.eval()
        model.decoder.eval()

        ep_sums = {k: 0.0 for k in keys}
        ep_batches = 0

        for b_idx, batch in enumerate(train_loader):
            if args.max_steps_per_epoch > 0 and b_idx >= args.max_steps_per_epoch:
                break

            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            lid = batch["lang_id"].to(device, non_blocking=True)
            prim = batch["is_primary_studio"].to(device, non_blocking=True)
            wav = batch["wav"].to(device, non_blocking=True)
            wlens = batch["wav_lengths"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model.forward_train(ids, mask, lid, wav, wlens, is_primary_studio=prim)
            loss = out["loss"]
            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss at step {global_step}, skipping batch", flush=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.5
            )
            optimizer.step()
            scheduler.step()

            global_step += 1
            ep_batches += 1
            for k in keys:
                ep_sums[k] += float(out[k].detach().cpu())

        train_avgs = {f"train_{k}": round(v / max(1, ep_batches), 5) for k, v in ep_sums.items()}
        val_m = evaluate_validation(model, val_loader, device)
        ep_sec = round(time.perf_counter() - ep_t0, 2)

        val_score = float(
            3.0 * val_m["val_kl_loss"]
            + 14.0 * val_m["val_latent_loss"]
            + 2.5 * val_m["val_dur_loss"]
            + 1.5 * val_m["val_mel_l1"]
        )
        is_best = val_score < best_val_score
        if is_best:
            best_val_score = val_score
            best_epoch = epoch
            model.save_pretrained(args.output_dir, filename="model.safetensors")

        row = {
            "epoch": epoch,
            "step": global_step,
            "epoch_sec": ep_sec,
            "lr_head": round(float(optimizer.param_groups[0]["lr"]), 7),
            "lr_backbone": round(float(optimizer.param_groups[1]["lr"]), 7),
            "is_best": is_best,
            **train_avgs,
            **val_m,
        }
        history.append(row)
        (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))

        mark = "*BEST*" if is_best else ""
        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] ({ep_sec:.1f}s) "
            f"train_loss={train_avgs['train_loss']:.4f} | "
            f"val_loss={val_m['val_loss']:.4f} val_kl={val_m['val_kl_loss']:.4f} "
            f"val_lat={val_m['val_latent_loss']:.4f} val_mel={val_m['val_mel_l1']:.4f} "
            f"val_dur={val_m['val_dur_mae_frames']:.2f}f ({val_m['val_utt_dur_mae_sec']:.2f}s) {mark}",
            flush=True,
        )

    # Preserve un-interpolated best trained checkpoint alongside model.safetensors
    shutil.copy2(args.output_dir / "model.safetensors", args.output_dir / "trained_model.safetensors")

    total_sec = round(time.perf_counter() - t_start, 2)
    manifest = {
        "model_name": "ninada",
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
        "train_utterances": len(train_ds),
        "val_utterances": len(val_ds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "total_steps": global_step,
        "best_epoch": best_epoch,
        "total_elapsed_sec": total_sec,
        "parameter_counts": param_counts,
        "pretrain_baseline": val0,
        "best_epoch_metrics": next((r for r in history if r["epoch"] == best_epoch), history[-1]),
    }
    (args.output_dir / "gcp_job_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ninada-train] Done in {total_sec:.1f}s! Best epoch={best_epoch}. Saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
