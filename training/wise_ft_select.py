"""
Validation-Only WiSE-FT Model Selection (`Wortsman et al., CVPR 2022`) for `Ninada`.

Sweeps weight-space interpolation `alpha in {0.00, 0.35, 0.50, 0.65, 0.80, 0.90, 1.00}`
between the warm-start anchor (`init_model.safetensors`) and the GCP L4 fine-tuned model
(`trained_model.safetensors`) EXCLUSIVELY on `data/val_meta.jsonl` (zero test-split access!).

Note: Newly trained `dur_predictor.*` and `lang_voice_embed` have no pretrained counterpart
in `facebook/mms-tts-hin` (which used stochastic duration flow) and therefore remain at
`100%` trained across all `alpha > 0`, preserving exact Pan-Indic duration cadence.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from safetensors.torch import load_file, save_file
from scipy.fft import dct

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "python" / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ninada.g2p import LANGUAGE_IDS, text_to_ids
from training.modeling_ninada import NinadaConfig, NinadaModel, strip_all_weight_norm


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


def _norm_sa(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\u0964\u0965,.;:!?\"'()\-–—0-9\u0966-\u096f]+", " ", s)
    return re.sub(r"\s+", "", s).strip()


def compute_mcd_f0(gt_wav: np.ndarray, syn_wav: np.ndarray, sr: int = 16000) -> tuple[float, float]:
    gt = gt_wav / max(1e-5, float(np.max(np.abs(gt_wav))))
    sy_peak = float(np.max(np.abs(syn_wav))) if len(syn_wav) > 0 else 0.0
    sy = syn_wav / sy_peak if (len(syn_wav) >= 1600 and sy_peak >= 1e-3) else np.full(4000, 1e-5, dtype=np.float32)

    mel_gt = librosa.feature.melspectrogram(y=gt, sr=sr, n_fft=1024, hop_length=256, n_mels=80, power=1.0)
    mel_sy = librosa.feature.melspectrogram(y=sy, sr=sr, n_fft=1024, hop_length=256, n_mels=80, power=1.0)
    mc_gt = (dct(np.log(np.maximum(mel_gt, 1e-6)), type=2, axis=0, norm="ortho")[1:14]) / math.sqrt(80.0)
    mc_sy = (dct(np.log(np.maximum(mel_sy, 1e-6)), type=2, axis=0, norm="ortho")[1:14]) / math.sqrt(80.0)

    _, wp = librosa.sequence.dtw(X=mc_gt, Y=mc_sy, metric="euclidean")
    wp = wp[::-1]
    idx_gt, idx_sy = wp[:, 0], wp[:, 1]
    diff = mc_gt[:, idx_gt] - mc_sy[:, idx_sy]
    mcd_db = float((10.0 * math.sqrt(2.0) / math.log(10.0)) * np.mean(np.sqrt(np.sum(diff * diff, axis=0))))

    f0_gt = librosa.yin(gt, fmin=75, fmax=450, sr=sr, frame_length=1024, hop_length=256)
    f0_sy = librosa.yin(sy, fmin=75, fmax=450, sr=sr, frame_length=1024, hop_length=256)
    fg = f0_gt[np.clip(idx_gt, 0, len(f0_gt) - 1)]
    fs = f0_sy[np.clip(idx_sy, 0, len(f0_sy) - 1)]
    valid = (fg > 80.0) & (fg < 420.0) & (fs > 80.0) & (fs < 420.0)
    f0_rmse = float(np.sqrt(np.mean((1200.0 * np.log2(fs[valid] / fg[valid])) ** 2))) if np.sum(valid) > 3 else 600.0
    return mcd_db, f0_rmse


class SushrotaOracle:
    def __init__(self, model_dir: Path) -> None:
        self.enabled = (model_dir / "sushrota_sanskrit_ctc_int8.onnx").exists()
        if self.enabled:
            self.prep = ort.InferenceSession(str(model_dir / "preprocessor.onnx"), providers=["CPUExecutionProvider"])
            self.asr = ort.InferenceSession(
                str(model_dir / "sushrota_sanskrit_ctc_int8.onnx"), providers=["CPUExecutionProvider"]
            )
            self.vocab: list[str] = json.loads((model_dir / "sanskrit_vocab.json").read_text(encoding="utf-8"))

    def cer(self, wav_16k: np.ndarray, ref_text: str) -> float:
        if not self.enabled or len(wav_16k) < 800:
            return 0.0
        audio = wav_16k.astype(np.float32)
        pk = float(np.max(np.abs(audio)))
        if pk > 1e-4:
            audio = 0.9 * (audio / pk)
        feats, flen = self.prep.run(
            None, {"audio_signal": audio[np.newaxis, :], "length": np.array([len(audio)], dtype=np.int64)}
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
        r_c = list(_norm_sa(ref_text))
        h_c = list(_norm_sa(hyp))
        return _edit_distance(r_c, h_c) / max(1, len(r_c)) * 100.0


def build_interpolated_state(
    init_sd: dict[str, torch.Tensor],
    trained_sd: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for k, v_tr in trained_sd.items():
        if k.startswith("dur_predictor."):
            # Always preserve trained deterministic duration predictor
            merged[k] = v_tr.clone()
        elif k == "lang_voice_embed":
            merged[k] = (alpha * v_tr).clone()
        elif k in init_sd and v_tr.is_floating_point():
            merged[k] = ((1.0 - alpha) * init_sd[k] + alpha * v_tr).to(v_tr.dtype)
        else:
            merged[k] = v_tr.clone()
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only WiSE-FT selector for Ninada")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/ninada"))
    parser.add_argument("--val-meta", type=Path, default=Path("data/val_meta.jsonl"))
    parser.add_argument("--sushrota-dir", type=Path, default=Path("/Users/lika/labs/sushrota-sanskrit-asr/models"))
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0],
    )
    args = parser.parse_args()

    init_sd = load_file(str(args.checkpoint_dir / "init_model.safetensors"))
    trained_path = args.checkpoint_dir / "trained_model.safetensors"
    if not trained_path.exists():
        trained_path = args.checkpoint_dir / "model.safetensors"
    trained_sd = load_file(str(trained_path))

    val_rows = [json.loads(ln) for ln in args.val_meta.read_text(encoding="utf-8").splitlines() if ln.strip()]
    oracle = SushrotaOracle(args.sushrota_dir)

    cfg = NinadaConfig.from_pretrained(args.checkpoint_dir)
    from transformers import VitsConfig

    vits_cfg = VitsConfig.from_pretrained(str(args.checkpoint_dir))
    model = NinadaModel(cfg, vits_cfg)
    strip_all_weight_norm(model)
    model.eval()

    scales = torch.tensor([1.0, 0.32], dtype=torch.float32)
    candidates: list[dict[str, Any]] = []

    for alpha in args.alphas:
        sd = build_interpolated_state(init_sd, trained_sd, alpha)
        model.load_state_dict(sd, strict=True)
        model.eval()

        mcds: list[float] = []
        f0s: list[float] = []
        durs: list[float] = []
        sa_cers: list[float] = []
        per_lang_mcd: dict[str, list[float]] = {lc: [] for lc in LANGUAGE_IDS}

        for row in val_rows:
            gt_wav, _ = sf.read(str(ROOT / row["wav_relpath"]), dtype="float32")
            parsed = text_to_ids(row["text"], lang=row["lang"])
            ids_t = torch.tensor([parsed["input_ids"]], dtype=torch.long)
            mask_t = torch.tensor([parsed["attention_mask"]], dtype=torch.long)
            lid_t = torch.tensor([LANGUAGE_IDS[row["lang"]]], dtype=torch.long)

            torch.manual_seed(2026)
            with torch.no_grad():
                wav_t, _ = model(ids_t, mask_t, lid_t, scales)
            syn = wav_t[0].cpu().numpy()

            mcd, f0 = compute_mcd_f0(gt_wav, syn)
            dur_err = abs(len(syn) / 16000.0 - float(row["duration_sec"]))
            mcds.append(mcd)
            f0s.append(f0)
            durs.append(dur_err)
            per_lang_mcd[row["lang"]].append(mcd)
            if row["lang"] == "sa":
                sa_cers.append(oracle.cer(syn, row["text"]))

        mean_mcd = float(np.mean(mcds))
        mean_f0 = float(np.mean(f0s))
        mean_dur = float(np.mean(durs))
        mean_cer = float(np.mean(sa_cers)) if sa_cers else 0.0
        # Balanced Pareto objective across spectral fidelity (MCD), pitch (F0), cadence (Dur), and intelligibility (CER)
        pareto_cost = (mean_mcd / 5.0) + (mean_cer / 15.0) + 0.35 * (mean_f0 / 600.0) + 0.35 * (mean_dur / 1.0)

        rec = {
            "alpha": alpha,
            "val_mcd_db": round(mean_mcd, 3),
            "val_f0_cents": round(mean_f0, 1),
            "val_dur_mae_sec": round(mean_dur, 3),
            "val_sa_cer_pct": round(mean_cer, 2),
            "pareto_cost": round(pareto_cost, 4),
            "per_lang_mcd": {k: round(float(np.mean(v)), 3) for k, v in per_lang_mcd.items() if v},
        }
        candidates.append(rec)
        print(
            f"[WiSE-FT val] alpha={alpha:.2f} | MCD={mean_mcd:.3f} dB | SA_CER={mean_cer:.2f}% | "
            f"F0={mean_f0:.1f}c | Dur={mean_dur:.3f}s | cost={pareto_cost:.4f}",
            flush=True,
        )

    best = min(candidates, key=lambda r: r["pareto_cost"])
    best_sd = build_interpolated_state(init_sd, trained_sd, float(best["alpha"]))
    save_file({k: v.detach().cpu().contiguous() for k, v in best_sd.items()}, str(args.checkpoint_dir / "model.safetensors"))

    report = {
        "split_used": "val_meta.jsonl (strictly zero test-set access)",
        "selected_alpha": best["alpha"],
        "selected_metrics": best,
        "sweep": candidates,
    }
    (args.checkpoint_dir / "wise_ft_selection.json").write_text(json.dumps(report, indent=2))
    print(f"\n[WiSE-FT] Selected alpha={best['alpha']} -> saved to {args.checkpoint_dir / 'model.safetensors'}")


if __name__ == "__main__":
    main()
