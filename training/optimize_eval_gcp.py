#!/usr/bin/env python3
"""
Leakage-Free Cloud GPU Validation Subspace Calibration for Ninada v3.0 (24,000 Hz Studio Neural TTS).

Optimizes per-language acoustic (`s_aco`, dims `0..127`) and prosodic (`s_pro`, dims `128..255`) style bank
scaling factors `(alpha_aco, alpha_pro)` strictly on the held-out validation split (`val_v3.jsonl`, `N = 240`,
`40` utterances per language, 0% overlap with `test_v3.jsonl`) while keeping the adversarial `StyleTTS2`
`predictor` and `decoder` weights frozen at `theta_0` for 100% smooth harmonic synthesis and intact word tails.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "python" / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.run_bench import compute_acoustic_metrics_24k, precompute_gt_features_24k
from ninada.ipa_g2p import ipa_to_ids, text_to_ipa
from training.export_onnx_24k import CANONICAL_PROBES, export_ninada_24k_bundle
from training.modeling_ninada_24k import (
    RETROFLEX_LATERAL_ID,
    SAMPLE_RATE,
    SUPPORTED_LANGS,
    Ninada24kModel,
)


def _voiced_tail_rms(pcm: np.ndarray) -> float:
    """Measure RMS energy of the final 320 ms of active speech (guarding against final-syllable unvoicing)."""
    if len(pcm) < 2400:
        return 0.0
    frame_len = 600
    n_f = len(pcm) // frame_len
    frames = pcm[: n_f * frame_len].reshape(n_f, frame_len)
    f_rms = np.sqrt(np.mean(frames * frames, axis=1))
    active = np.nonzero(f_rms > 0.004)[0]
    if len(active) < 4:
        return float(np.sqrt(np.mean(pcm[-2400:] ** 2)))
    end_f = int(active[-1]) + 1
    start_f = max(0, end_f - 13)  # ~325 ms speech tail
    tail = pcm[start_f * frame_len : end_f * frame_len]
    return float(np.sqrt(np.mean(tail * tail)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate Ninada v3.0 per-language style banks on held-out val_v3.jsonl")
    ap.add_argument("--raw-ckpt", type=Path, required=True, help="Raw alpha=1.0 trained model.safetensors")
    ap.add_argument("--val-jsonl", type=Path, required=True, help="Held-out val_v3.jsonl (N=240)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output artifact bundle directory")
    args = ap.parse_args()

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval-opt] Running validation calibration on device={device}", flush=True)

    model = Ninada24kModel.from_pretrained_warmstart()
    theta_0 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    theta_ft = load_file(str(args.raw_ckpt))

    # Build clean base state: frozen adversarial predictor/decoder at theta_0 + norm-matched retroflex lateral slot 79
    base_sd: dict[str, torch.Tensor] = {}
    for k, v_0 in theta_0.items():
        if not v_0.is_floating_point():
            base_sd[k] = v_0.clone()
        elif k == "studio_silence_trim":
            base_sd[k] = torch.tensor(1.0, dtype=torch.float32)
        elif k in ("bert.embeddings.word_embeddings.weight", "text_encoder.embedding.weight") and k in theta_ft:
            w = v_0.clone()
            tn = w[54].norm()
            w79 = theta_ft[k][RETROFLEX_LATERAL_ID].cpu()
            w[RETROFLEX_LATERAL_ID] = w79 * (tn / w79.norm().clamp(min=1e-8))
            base_sd[k] = w
        else:
            base_sd[k] = v_0.clone()

    model.load_state_dict(base_sd, strict=True)
    model.to(device).eval()
    model.decoder.generator.m_source.l_sin_gen.deterministic_eval = True

    # Load and precompute ground-truth validation features once across CPU worker threads
    val_rows = [json.loads(line) for line in args.val_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows_by_lang: dict[str, list[dict[str, Any]]] = {lc: [] for lc in SUPPORTED_LANGS}
    for r in val_rows:
        if r["lang"] in rows_by_lang:
            rows_by_lang[r["lang"]].append(r)

    def _prep_val_item(r: dict[str, Any]) -> dict[str, Any]:
        gt_wav, gt_sr = sf.read(str(r["wav_path"]), dtype="float32")
        if gt_wav.ndim > 1:
            gt_wav = gt_wav.mean(axis=1)
        if gt_sr != SAMPLE_RATE:
            gt_wav = librosa.resample(gt_wav, orig_sr=gt_sr, target_sr=SAMPLE_RATE).astype(np.float32)
        ipa = text_to_ipa(r["text"], lang=r["lang"])
        ipa_ids = ipa_to_ids(ipa)
        return {
            "id": r["id"],
            "lang": r["lang"],
            "gt_wav": gt_wav,
            "gt_dur": len(gt_wav) / float(SAMPLE_RATE),
            "gt_cache": precompute_gt_features_24k(gt_wav),
            "ipa_ids": ipa_ids,
        }

    val_prepped: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for lc in SUPPORTED_LANGS:
            val_prepped[lc] = list(pool.map(_prep_val_item, rows_by_lang[lc]))
    print(
        f"[eval-opt] Precomputed validation acoustic caches in {time.time() - t_start:.1f}s: "
        f"{ {lc: len(val_prepped[lc]) for lc in SUPPORTED_LANGS} }",
        flush=True,
    )

    # Candidate grid: candidate #0 is the verified baseline (0.90, 0.0, 0.0)
    candidates: list[tuple[float, float, float]] = [(0.90, 0.0, 0.0)]
    for a_aco in (0.75, 0.90, 1.00, 1.10):
        for a_pro in (0.0, 0.12, 0.24, 0.36):
            for a_slp in (0.0, 0.40):
                cfg = (a_aco, a_pro, a_slp)
                if cfg not in candidates:
                    candidates.append(cfg)

    selected_per_lang: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        for lc in SUPPORTED_LANGS:
            g_ft = theta_ft[f"voice_global_delta.{lc}"].to(device)
            s_ft = theta_ft[f"voice_slope_delta.{lc}"].to(device)
            g_0 = theta_0[f"voice_global_delta.{lc}"].to(device)
            s_0 = theta_0[f"voice_slope_delta.{lc}"].to(device)

            probe_ids = ipa_to_ids(text_to_ipa(CANONICAL_PROBES[lc], lang=lc))
            probe_t = torch.tensor([probe_ids], dtype=torch.long, device=device)

            def _apply_lang_style(a_aco: float, a_pro: float, a_slp: float) -> None:
                with torch.no_grad():
                    g_new = g_0.clone()
                    g_new[..., :128] = (a_aco * g_ft[..., :128] + (1.0 - a_aco) * g_0[..., :128]).clamp(-0.075, 0.075)
                    g_new[..., 128:] = (a_pro * g_ft[..., 128:] + (1.0 - a_pro) * g_0[..., 128:]).clamp(-0.045, 0.045)
                    s_new = s_0.clone()
                    s_new[..., :128] = (a_slp * a_aco * s_ft[..., :128]).clamp(-0.03, 0.03)
                    s_new[..., 128:] = (a_slp * a_pro * s_ft[..., 128:]).clamp(-0.02, 0.02)
                    model.voice_global_delta[lc].copy_(g_new)
                    model.voice_slope_delta[lc].copy_(s_new)

            # Evaluate baseline candidate #0 first
            _apply_lang_style(*candidates[0])
            with torch.no_grad():
                p_style0 = model.get_style_for_tokens(lc, len(probe_ids) - 2)
                p_wav0_t, _ = model.infer_tokens(probe_t, p_style0, speed=1.0)
                p_wav0 = p_wav0_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
            base_probe_tail = _voiced_tail_rms(p_wav0)
            base_probe_dur = len(p_wav0) / float(SAMPLE_RATE)

            def _eval_lang_config(a_aco: float, a_pro: float, a_slp: float, is_baseline: bool = False) -> dict[str, Any] | None:
                _apply_lang_style(a_aco, a_pro, a_slp)
                with torch.no_grad():
                    ps = model.get_style_for_tokens(lc, len(probe_ids) - 2)
                    pw_t, _ = model.infer_tokens(probe_t, ps, speed=1.0)
                    pw = pw_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
                tail_rms = _voiced_tail_rms(pw)
                p_dur = len(pw) / float(SAMPLE_RATE)
                if not is_baseline:
                    if tail_rms < 0.90 * base_probe_tail or p_dur < 0.86 * base_probe_dur or p_dur > 1.15 * base_probe_dur:
                        return None

                syn_list: list[tuple[dict[str, Any], np.ndarray]] = []
                with torch.no_grad():
                    for item in val_prepped[lc]:
                        ids_t = torch.tensor([item["ipa_ids"]], dtype=torch.long, device=device)
                        st = model.get_style_for_tokens(lc, len(item["ipa_ids"]) - 2)
                        w_t, _ = model.infer_tokens(ids_t, st, speed=1.0)
                        syn_list.append((item, w_t.squeeze(0).detach().cpu().numpy().astype(np.float32)))

                def _score_one(pair: tuple[dict[str, Any], np.ndarray]) -> tuple[float, float, float, float, float]:
                    it, sw = pair
                    mcd, hf, f0c, is_v = compute_acoustic_metrics_24k(
                        it["gt_wav"], sw, syn_sr=SAMPLE_RATE, gt_cache=it["gt_cache"]
                    )
                    sdur = (len(sw) / float(SAMPLE_RATE)) if is_v else 0.0
                    return mcd, hf, f0c, abs(sdur - it["gt_dur"]), (1.0 if is_v else 0.0)

                metrics = list(pool.map(_score_one, syn_list))
                voiced_pct = 100.0 * float(np.mean([m[4] for m in metrics]))
                if voiced_pct < 100.0:
                    return None
                mcd_m = float(np.mean([m[0] for m in metrics]))
                hf_m = float(np.mean([m[1] for m in metrics]))
                f0_m = float(np.mean([m[2] for m in metrics]))
                dur_m = float(np.mean([m[3] for m in metrics]))
                comp = mcd_m + 0.055 * hf_m + 0.35 * dur_m + 0.0012 * f0_m
                return {
                    "alpha_aco": a_aco,
                    "alpha_pro": a_pro,
                    "alpha_slope": a_slp,
                    "val_mcd_db": round(mcd_m, 4),
                    "val_hf_err_db": round(hf_m, 4),
                    "val_f0_rmse_cents": round(f0_m, 2),
                    "val_dur_mae_sec": round(dur_m, 4),
                    "probe_tail_rms": round(tail_rms, 5),
                    "composite_score": round(comp, 5),
                }

            base_res = _eval_lang_config(*candidates[0], is_baseline=True)
            assert base_res is not None
            best_res = base_res

            for cfg in candidates[1:]:
                res = _eval_lang_config(*cfg, is_baseline=False)
                if res is None:
                    continue
                # Strict non-degradation guards on MCD and F0 RMSE relative to baseline
                if res["val_mcd_db"] > base_res["val_mcd_db"] + 0.025:
                    continue
                if res["val_f0_rmse_cents"] > base_res["val_f0_rmse_cents"] + 8.0:
                    continue
                if res["composite_score"] < best_res["composite_score"]:
                    best_res = res

            _apply_lang_style(best_res["alpha_aco"], best_res["alpha_pro"], best_res["alpha_slope"])
            selected_per_lang[lc] = {
                "baseline": base_res,
                "selected": best_res,
            }
            print(
                f"[eval-opt][{lc}] baseline={base_res['composite_score']:.4f} "
                f"(MCD={base_res['val_mcd_db']:.3f}, HF={base_res['val_hf_err_db']:.2f}, Dur={base_res['val_dur_mae_sec']:.3f}s, F0={base_res['val_f0_rmse_cents']:.1f}c) -> "
                f"selected=({best_res['alpha_aco']}, {best_res['alpha_pro']}, {best_res['alpha_slope']}) "
                f"score={best_res['composite_score']:.4f} "
                f"(MCD={best_res['val_mcd_db']:.3f}, HF={best_res['val_hf_err_db']:.2f}, Dur={best_res['val_dur_mae_sec']:.3f}s, F0={best_res['val_f0_rmse_cents']:.1f}c)",
                flush=True,
            )

    export_meta = export_ninada_24k_bundle(model, args.out_dir, keep_fp32_onnx=False)
    report = {
        "elapsed_sec": round(time.time() - t_start, 1),
        "val_utterances": len(val_rows),
        "per_lang_calibration": selected_per_lang,
        "export": export_meta,
    }
    (args.out_dir / "val_calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[eval-opt] Completed validation calibration + ONNX export in {report['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
