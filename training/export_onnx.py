"""
Export trained Ninada v2.0 Native Specialists (`sa`, `kn`, `hi`, `ta`, `te`, `ml`) to Browser-Ready
IEEE FP16 Quantized Streaming ONNX Opset 17:
  - `onnx/{lang}/encoder_flow_quantized.onnx` (~27.3 MB Stage-1 Text+SDP+Flow -> Latents)
  - `onnx/{lang}/vocoder_quantized.onnx` (~27.1 MB Stage-2 Halo-Windowed HiFi-GAN Vocoder)
  - `docs/models_manifest.json` (exact byte lengths + CDN/local URLs for persistent browser CacheStorage)

Zero `ConvInteger` and zero `NonZero` operators — 100% compatible with `onnxruntime-web` WASM SIMD + WebGPU.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "python" / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ninada.g2p import LANGUAGE_IDS, NATIVE_VOCABS, VOCAB_TOKENS, text_to_ids
from training.modeling_ninada import NinadaModel, strip_all_weight_norm

ALL_LANGS = ("sa", "kn", "hi", "ta", "te", "ml")

TEST_SENTENCES: dict[str, str] = {
    "sa": "प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् । यथार्थज्ञानं प्रमाणम् ॥",
    "kn": "ಕನ್ನಡ ಭಾಷೆಯ ನೈಸರ್ಗಿಕ ಧ್ವನಿ ಸಂಶ್ಲೇಷಣೆಗೆ ಸ್ವಾಗತ.",
    "hi": "नमस्ते भारत, यह एक बहुभाषी वाक् संश्लेषण प्रणाली है।",
    "ta": "தமிழ் மொழி ஒலி உருவாக்கம் மிகவும் தெளிவாக உள்ளது.",
    "te": "తెలుగు భాషా ధ్వని సంశ్లేషణ వ్యవస్థకు స్వాగతం.",
    "ml": "മലയാളം ഭാഷാ ശബ്ദ സംയോജന സംവിധാനത്തിലേക്ക് സ്വാഗതം.",
}


class _EncoderFlowWrapper(nn.Module):
    def __init__(self, core: NinadaModel) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.encode_flow_infer(input_ids, attention_mask, scales)


class _VocoderWrapper(nn.Module):
    def __init__(self, core: NinadaModel) -> None:
        super().__init__()
        self.core = core

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.core.decode_latents_infer(latents)


def _compute_snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(len(ref), len(est))
    r = ref[:n].astype(np.float64)
    e = est[:n].astype(np.float64)
    sig = float(np.sum(r * r)) + 1e-12
    err = float(np.sum((r - e) ** 2)) + 1e-12
    return 10.0 * float(np.log10(sig / err))


def compress_onnx_fp16_weights(
    src_onnx_path: Path,
    dst_onnx_path: Path,
) -> dict[str, Any]:
    """Compress FP32 ONNX initializers (>= 256 params) into IEEE FP16 single-file ONNX."""
    model = onnx.load(str(src_onnx_path))
    graph = model.graph

    new_initializers: list[onnx.TensorProto] = []
    prepend_nodes: list[onnx.NodeProto] = []

    fp16_params = 0
    fp32_params = 0

    for init in graph.initializer:
        name = init.name
        if init.data_type != TensorProto.FLOAT:
            new_initializers.append(init)
            continue

        arr = numpy_helper.to_array(init).astype(np.float32)
        numel = int(arr.size)

        if numel >= 256:
            arr16 = np.clip(arr, -65500.0, 65500.0).astype(np.float16)
            h_name = f"{name}__fp16_w"
            new_initializers.append(numpy_helper.from_array(arr16, name=h_name))
            cast_node = helper.make_node(
                "Cast",
                inputs=[h_name],
                outputs=[name],
                name=f"CastFP16_{name}",
                to=TensorProto.FLOAT,
            )
            prepend_nodes.append(cast_node)
            fp16_params += numel
        else:
            new_initializers.append(init)
            fp32_params += numel

    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    existing_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(prepend_nodes + existing_nodes)

    dst_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst_onnx_path))

    op_types = sorted({n.op_type for n in graph.node})
    assert "ConvInteger" not in op_types, "ConvInteger is unsupported in onnxruntime-web WASM!"
    assert "NonZero" not in op_types, "NonZero dynamic indexing must not appear in SDP ONNX!"

    return {
        "fp16_params": fp16_params,
        "fp32_params": fp32_params,
        "fp32_size_mb": round(src_onnx_path.stat().st_size / (1024 * 1024), 2),
        "quantized_size_mb": round(dst_onnx_path.stat().st_size / (1024 * 1024), 2),
        "quantized_bytes": int(dst_onnx_path.stat().st_size),
        "compression_ratio": round(src_onnx_path.stat().st_size / max(1, dst_onnx_path.stat().st_size), 2),
        "op_types": op_types,
    }


def _stream_vocode_with_crossfade(
    sess_enc: ort.InferenceSession,
    sess_voc: ort.InferenceSession,
    ort_inputs: dict[str, np.ndarray],
    chunk_frames: int = 64,
    halo_frames: int = 16,
    hop_length: int = 256,
    crossfade_samples: int = 256,
) -> tuple[np.ndarray, float, float]:
    """Stream HiFi-GAN chunks with 16-frame context halo + 256-sample raised-cosine boundary crossfade."""
    t0 = time.perf_counter()
    latents, _ = sess_enc.run(None, ort_inputs)
    total_frames = int(latents.shape[2])
    pieces: list[np.ndarray] = []
    prev_tail: np.ndarray | None = None
    ttfa_ms = 0.0

    fade_idx = np.arange(crossfade_samples, dtype=np.float32) / max(1, crossfade_samples)
    fade_in = 0.5 * (1.0 - np.cos(math.pi * fade_idx))
    fade_out = 1.0 - fade_in

    bounds = list(range(0, total_frames, chunk_frames))
    for b_i, st in enumerate(bounds):
        ed = min(total_frames, st + chunk_frames)
        is_last = b_i == len(bounds) - 1
        w_st = max(0, st - halo_frames)
        w_ed = min(total_frames, ed + halo_frames)
        wav_win = sess_voc.run(None, {"latents": latents[:, :, w_st:w_ed]})[0][0]
        if st == 0:
            ttfa_ms = (time.perf_counter() - t0) * 1000.0

        crop_l = (st - w_st) * hop_length
        crop_r = crop_l + (ed - st) * hop_length

        # Keep extra `crossfade_samples` past crop_r (from right halo) for seamless overlap with next chunk
        extra_r = min(len(wav_win), crop_r + (0 if is_last else crossfade_samples))
        seg = wav_win[crop_l:extra_r].astype(np.float32).copy()

        if prev_tail is not None and len(prev_tail) == crossfade_samples and len(seg) >= crossfade_samples:
            seg[:crossfade_samples] = prev_tail * fade_out + seg[:crossfade_samples] * fade_in

        if not is_last and (extra_r - crop_r) == crossfade_samples:
            prev_tail = seg[-crossfade_samples:].copy()
            pieces.append(seg[:-crossfade_samples])
        else:
            prev_tail = None
            pieces.append(seg[: crop_r - crop_l])

    total_ms = (time.perf_counter() - t0) * 1000.0
    return np.concatenate(pieces, axis=0), ttfa_ms, total_ms


def export_single_specialist(
    lang: str,
    spec_ckpt_dir: Path,
    lang_onnx_dir: Path,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = NinadaModel.from_pretrained(spec_ckpt_dir)
    strip_all_weight_norm(model)
    model.eval()
    model.cpu()

    lang_onnx_dir.mkdir(parents=True, exist_ok=True)
    enc_fp32_path = lang_onnx_dir / "encoder_flow.onnx"
    enc_quant_path = lang_onnx_dir / "encoder_flow_quantized.onnx"
    voc_fp32_path = lang_onnx_dir / "vocoder.onnx"
    voc_quant_path = lang_onnx_dir / "vocoder_quantized.onnx"

    vocab_len = len(NATIVE_VOCABS[lang])
    ids_dummy = torch.randint(0, vocab_len - 1, (1, 20), dtype=torch.long)
    mask_dummy = torch.ones_like(ids_dummy)
    scales_det = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    base_export_opt: dict[str, Any] = {"opset_version": 17, "do_constant_folding": True}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        base_export_opt["dynamo"] = False

    # 1. Stage-1 Streaming Encoder + SDP + Normalizing Flow
    enc_wrap = _EncoderFlowWrapper(model).eval()
    torch.onnx.export(
        enc_wrap,
        (ids_dummy, mask_dummy, scales_det),
        str(enc_fp32_path),
        input_names=["input_ids", "attention_mask", "scales"],
        output_names=["latents", "durations"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "latents": {0: "batch", 2: "frames"},
            "durations": {0: "batch", 1: "seq"},
        },
        **base_export_opt,
    )
    enc_stats = compress_onnx_fp16_weights(enc_fp32_path, enc_quant_path)
    enc_fp32_path.unlink(missing_ok=True)

    # 2. Stage-2 Halo-Windowed HiFi-GAN Vocoder
    voc_wrap = _VocoderWrapper(model).eval()
    lat_dummy = torch.randn(1, 192, 48, dtype=torch.float32)
    torch.onnx.export(
        voc_wrap,
        (lat_dummy,),
        str(voc_fp32_path),
        input_names=["latents"],
        output_names=["waveform"],
        dynamic_axes={
            "latents": {0: "batch", 2: "frames"},
            "waveform": {0: "batch", 1: "samples"},
        },
        **base_export_opt,
    )
    voc_stats = compress_onnx_fp16_weights(voc_fp32_path, voc_quant_path)
    voc_fp32_path.unlink(missing_ok=True)

    # Deterministic Parity (`noise=0, noise_dur=0`) + Natural Streaming Seam Parity
    sess_enc = ort.InferenceSession(str(enc_quant_path), providers=["CPUExecutionProvider"])
    sess_voc = ort.InferenceSession(str(voc_quant_path), providers=["CPUExecutionProvider"])

    parsed = text_to_ids(TEST_SENTENCES[lang], lang=lang)
    ids_t = torch.tensor([parsed["input_ids"]], dtype=torch.long)
    mask_t = torch.tensor([parsed["attention_mask"]], dtype=torch.long)

    with torch.no_grad():
        pt_wav, _ = model(ids_t, mask_t, scales_det)
    pt_w = pt_wav[0].numpy()

    ort_inputs = {
        "input_ids": ids_t.numpy(),
        "attention_mask": mask_t.numpy(),
        "scales": scales_det.numpy(),
    }
    lat_full, _ = sess_enc.run(None, ort_inputs)
    full_quant_w = sess_voc.run(None, {"latents": lat_full})[0][0]
    stream_w, ttfa_ms, stream_total_ms = _stream_vocode_with_crossfade(sess_enc, sess_voc, ort_inputs)

    quant_snr = _compute_snr_db(pt_w, full_quant_w)
    seam_snr = _compute_snr_db(full_quant_w, stream_w)
    n = min(len(pt_w), len(full_quant_w))
    pearson_r = float(np.corrcoef(pt_w[:n], full_quant_w[:n])[0, 1])

    vocab_meta = {
        "architecture_version": "2.0.0",
        "lang": lang,
        "native_vocab": NATIVE_VOCABS[lang],
        "vocab_tokens": VOCAB_TOKENS,
        "language_ids": LANGUAGE_IDS,
        "sampling_rate": 16000,
        "hop_length": 256,
        "streaming": {
            "chunk_frames": 64,
            "halo_frames": 16,
            "crossfade_samples": 256,
            "encoder_flow_onnx": "encoder_flow_quantized.onnx",
            "vocoder_onnx": "vocoder_quantized.onnx",
        },
    }
    (lang_onnx_dir / "vocab.json").write_text(json.dumps(vocab_meta, ensure_ascii=False, indent=2))

    return {
        "lang": lang,
        "vocab_size": vocab_len,
        "encoder_flow_size_mb": enc_stats["quantized_size_mb"],
        "vocoder_size_mb": voc_stats["quantized_size_mb"],
        "total_size_mb": round(enc_stats["quantized_size_mb"] + voc_stats["quantized_size_mb"], 2),
        "encoder_bytes": enc_stats["quantized_bytes"],
        "vocoder_bytes": voc_stats["quantized_bytes"],
        "quant_snr_db": round(quant_snr, 2),
        "stream_seam_snr_db": round(seam_snr, 2),
        "stream_ttfa_ms": round(ttfa_ms, 2),
        "stream_total_ms": round(stream_total_ms, 2),
        "pearson_r": round(pearson_r, 6),
        "samples": int(len(stream_w)),
        "duration_sec": round(len(stream_w) / 16000.0, 3),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def export_and_verify(
    checkpoint_dir: Path,
    copy_to_docs: Path | None = None,
    hf_repo_id: str = "gnumanth/ninada",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    specialists_dir = checkpoint_dir / "specialists"
    onnx_root = checkpoint_dir / "onnx"
    onnx_root.mkdir(parents=True, exist_ok=True)

    # Remove legacy v1.x monolithic >50MB ONNX if present
    (onnx_root / "model.onnx").unlink(missing_ok=True)
    (onnx_root / "model_quantized.onnx").unlink(missing_ok=True)

    parity_rows: list[dict[str, Any]] = []
    manifest_langs: dict[str, Any] = {}

    for lc in ALL_LANGS:
        spec_dir = specialists_dir / lc
        if not (spec_dir / "model.safetensors").exists():
            continue
        row = export_single_specialist(lc, spec_dir, onnx_root / lc)
        parity_rows.append(row)
        print(
            f"[export {lc.upper()}] size={row['total_size_mb']:.2f}MB "
            f"quant_snr={row['quant_snr_db']:.2f}dB seam_snr={row['stream_seam_snr_db']:.2f}dB "
            f"ttfa={row['stream_ttfa_ms']:.1f}ms r={row['pearson_r']:.6f}",
            flush=True,
        )

        if lc == "sa":
            enc_urls = [
                "./encoder_flow_quantized.onnx",
                f"https://huggingface.co/{hf_repo_id}/resolve/main/onnx/sa/encoder_flow_quantized.onnx",
            ]
            voc_urls = [
                "./vocoder_quantized.onnx",
                f"https://huggingface.co/{hf_repo_id}/resolve/main/onnx/sa/vocoder_quantized.onnx",
            ]
        else:
            enc_urls = [
                f"./onnx/{lc}/encoder_flow_quantized.onnx",
                f"https://huggingface.co/{hf_repo_id}/resolve/main/onnx/{lc}/encoder_flow_quantized.onnx",
            ]
            voc_urls = [
                f"./onnx/{lc}/vocoder_quantized.onnx",
                f"https://huggingface.co/{hf_repo_id}/resolve/main/onnx/{lc}/vocoder_quantized.onnx",
            ]

        manifest_langs[lc] = {
            "lang": lc,
            "encoderBytes": row["encoder_bytes"],
            "vocoderBytes": row["vocoder_bytes"],
            "totalMB": row["total_size_mb"],
            "encoderCacheKey": f"ninada_v2_{lc}_enc_{row['encoder_bytes']}.onnx",
            "vocoderCacheKey": f"ninada_v2_{lc}_voc_{row['vocoder_bytes']}.onnx",
            "encoderUrls": enc_urls,
            "vocoderUrls": voc_urls,
        }

    # Copy primary Sanskrit (`sa`) to top-level `onnx/` for default single-path SDK compatibility
    if (onnx_root / "sa" / "encoder_flow_quantized.onnx").exists():
        for fn in ("encoder_flow_quantized.onnx", "vocoder_quantized.onnx", "vocab.json"):
            shutil.copy2(onnx_root / "sa" / fn, onnx_root / fn)

    models_manifest = {
        "version": "2.0.0",
        "cacheName": "ninada-onnx-v2.0.0",
        "defaultLang": "sa",
        "languages": manifest_langs,
    }
    (onnx_root / "models_manifest.json").write_text(json.dumps(models_manifest, indent=2))
    (checkpoint_dir / "models_manifest.json").write_text(json.dumps(models_manifest, indent=2))

    mean_quant_snr = float(np.mean([r["quant_snr_db"] for r in parity_rows]))
    mean_stream_snr = float(np.mean([r["stream_seam_snr_db"] for r in parity_rows]))
    mean_ttfa_ms = float(np.mean([r["stream_ttfa_ms"] for r in parity_rows]))
    min_pearson = float(np.min([r["pearson_r"] for r in parity_rows]))
    mean_size_mb = float(np.mean([r["total_size_mb"] for r in parity_rows]))

    report = {
        "architecture_version": "2.0.0",
        "quantized_size_mb": round(mean_size_mb, 2),
        "encoder_flow_size_mb": parity_rows[0]["encoder_flow_size_mb"] if parity_rows else 0.0,
        "vocoder_size_mb": parity_rows[0]["vocoder_size_mb"] if parity_rows else 0.0,
        "mean_quant_snr_db": round(mean_quant_snr, 2),
        "mean_stream_seam_snr_db": round(mean_stream_snr, 2),
        "mean_stream_ttfa_ms": round(mean_ttfa_ms, 2),
        "min_pearson_r": round(min_pearson, 6),
        "export_elapsed_sec": round(time.perf_counter() - t0, 2),
        "parity_by_language": parity_rows,
    }
    (onnx_root / "export_report.json").write_text(json.dumps(report, indent=2))

    if copy_to_docs is not None and (onnx_root / "sa" / "encoder_flow_quantized.onnx").exists():
        copy_to_docs.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_root / "sa" / "encoder_flow_quantized.onnx", copy_to_docs / "encoder_flow_quantized.onnx")
        shutil.copy2(onnx_root / "sa" / "vocoder_quantized.onnx", copy_to_docs / "vocoder_quantized.onnx")
        shutil.copy2(onnx_root / "sa" / "vocab.json", copy_to_docs / "vocab.json")
        shutil.copy2(onnx_root / "models_manifest.json", copy_to_docs / "models_manifest.json")
        (copy_to_docs / "model_quantized.onnx").unlink(missing_ok=True)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Ninada v2.0 Native Specialists to Streaming FP16 ONNX")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/ninada"))
    parser.add_argument("--copy-to-docs", type=Path, default=Path("docs"))
    args = parser.parse_args()

    rep = export_and_verify(args.checkpoint_dir, copy_to_docs=args.copy_to_docs)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
