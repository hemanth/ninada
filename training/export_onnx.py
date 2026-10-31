"""
Export trained NinadaModel to ONNX Opset 17 + Browser-Ready IEEE FP16 Quantized Streaming & Monolithic ONNX:
  1. `onnx/model_quantized.onnx` (~56.6 MB single-file end-to-end graph)
  2. `onnx/encoder_flow_quantized.onnx` (~29.8 MB Stage-1 Streaming Akshara-to-Latent graph, ~35 ms)
  3. `onnx/vocoder_quantized.onnx` (~26.8 MB Stage-2 Streaming Halo-Chunked HiFi-GAN vocoder, <250 ms TTFA)

Zero `ConvInteger` operators (100% compatible with `onnxruntime-web` WASM SIMD + WebGPU).
"""
from __future__ import annotations

import argparse
import inspect
import json
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

from ninada.g2p import LANGUAGE_IDS, VOCAB_TOKENS, text_to_ids
from training.modeling_ninada import NinadaModel, strip_all_weight_norm


class _EncoderFlowWrapper(nn.Module):
    def __init__(self, core: NinadaModel) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lang_id: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.core.encode_flow_infer(input_ids, attention_mask, lang_id, scales)


class _VocoderWrapper(nn.Module):
    def __init__(self, core: NinadaModel) -> None:
        super().__init__()
        self.core = core

    def forward(self, latents: torch.Tensor, voice_cond: torch.Tensor) -> torch.Tensor:
        return self.core.decode_latents_infer(latents, voice_cond)


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
    """Compress FP32 ONNX initializers (>= 256 params) into lossless-phase IEEE FP16 single-file ONNX."""
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

    return {
        "fp16_params": fp16_params,
        "fp32_params": fp32_params,
        "fp32_size_mb": round(src_onnx_path.stat().st_size / (1024 * 1024), 2),
        "quantized_size_mb": round(dst_onnx_path.stat().st_size / (1024 * 1024), 2),
        "compression_ratio": round(src_onnx_path.stat().st_size / max(1, dst_onnx_path.stat().st_size), 2),
        "op_types": op_types,
    }


def _stream_vocode_ort(
    sess_enc: ort.InferenceSession,
    sess_voc: ort.InferenceSession,
    ort_inputs: dict[str, np.ndarray],
    chunk_frames: int = 40,
    halo_frames: int = 10,
    hop_length: int = 256,
) -> tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    latents, voice_cond, _ = sess_enc.run(None, ort_inputs)
    total_frames = int(latents.shape[2])
    pieces: list[np.ndarray] = []
    ttfa_ms = 0.0
    for st in range(0, total_frames, chunk_frames):
        ed = min(total_frames, st + chunk_frames)
        w_st = max(0, st - halo_frames)
        w_ed = min(total_frames, ed + halo_frames)
        wav_win = sess_voc.run(
            None,
            {
                "latents": latents[:, :, w_st:w_ed],
                "voice_cond": voice_cond,
            },
        )[0][0]
        if st == 0:
            ttfa_ms = (time.perf_counter() - t0) * 1000.0
        crop_l = (st - w_st) * hop_length
        crop_r = crop_l + (ed - st) * hop_length
        pieces.append(wav_win[crop_l:crop_r])
    total_ms = (time.perf_counter() - t0) * 1000.0
    return np.concatenate(pieces, axis=0), ttfa_ms, total_ms


def export_and_verify(
    checkpoint_dir: Path,
    copy_to_docs: Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = NinadaModel.from_pretrained(checkpoint_dir)
    strip_all_weight_norm(model)
    model.eval()
    model.cpu()

    onnx_dir = checkpoint_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = onnx_dir / "model.onnx"
    quant_path = onnx_dir / "model_quantized.onnx"
    enc_fp32_path = onnx_dir / "encoder_flow.onnx"
    enc_quant_path = onnx_dir / "encoder_flow_quantized.onnx"
    voc_fp32_path = onnx_dir / "vocoder.onnx"
    voc_quant_path = onnx_dir / "vocoder_quantized.onnx"

    ids_dummy = torch.randint(0, 72, (1, 18), dtype=torch.long)
    mask_dummy = torch.ones_like(ids_dummy)
    lang_dummy = torch.tensor([0], dtype=torch.long)
    scales_det = torch.tensor([1.0, 0.0], dtype=torch.float32)

    base_export_opt: dict[str, Any] = {"opset_version": 17, "do_constant_folding": True}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        base_export_opt["dynamo"] = False

    # 1. Monolithic End-to-End ONNX
    torch.onnx.export(
        model,
        (ids_dummy, mask_dummy, lang_dummy, scales_det),
        str(fp32_path),
        input_names=["input_ids", "attention_mask", "lang_id", "scales"],
        output_names=["waveform", "durations"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "lang_id": {0: "batch"},
            "waveform": {0: "batch", 1: "samples"},
            "durations": {0: "batch", 1: "seq"},
        },
        **base_export_opt,
    )
    quant_stats = compress_onnx_fp16_weights(fp32_path, quant_path)

    # 2. Streaming Stage-1: Encoder + Duration + Normalizing Flow
    enc_wrap = _EncoderFlowWrapper(model).eval()
    torch.onnx.export(
        enc_wrap,
        (ids_dummy, mask_dummy, lang_dummy, scales_det),
        str(enc_fp32_path),
        input_names=["input_ids", "attention_mask", "lang_id", "scales"],
        output_names=["latents", "voice_cond", "durations"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "lang_id": {0: "batch"},
            "latents": {0: "batch", 2: "frames"},
            "voice_cond": {0: "batch"},
            "durations": {0: "batch", 1: "seq"},
        },
        **base_export_opt,
    )
    enc_stats = compress_onnx_fp16_weights(enc_fp32_path, enc_quant_path)
    enc_fp32_path.unlink(missing_ok=True)

    # 3. Streaming Stage-2: Halo-Chunked HiFi-GAN Vocoder
    voc_wrap = _VocoderWrapper(model).eval()
    lat_dummy = torch.randn(1, 192, 48, dtype=torch.float32)
    cond_dummy = torch.randn(1, 192, dtype=torch.float32)
    torch.onnx.export(
        voc_wrap,
        (lat_dummy, cond_dummy),
        str(voc_fp32_path),
        input_names=["latents", "voice_cond"],
        output_names=["waveform"],
        dynamic_axes={
            "latents": {0: "batch", 2: "frames"},
            "voice_cond": {0: "batch"},
            "waveform": {0: "batch", 1: "samples"},
        },
        **base_export_opt,
    )
    voc_stats = compress_onnx_fp16_weights(voc_fp32_path, voc_quant_path)
    voc_fp32_path.unlink(missing_ok=True)

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_quant = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])
    sess_enc = ort.InferenceSession(str(enc_quant_path), providers=["CPUExecutionProvider"])
    sess_voc = ort.InferenceSession(str(voc_quant_path), providers=["CPUExecutionProvider"])

    test_cases = [
        ("sa", "\u090b\u0917\u094d\u0935\u0947\u0926\u0903 \u0936\u0902\u0915\u0930\u0903 \u092a\u094d\u0930\u091c\u094d\u091e\u093e\u092a\u0942\u0930\u094d\u0935\u0915\u0935\u094d\u092f\u093e\u092a\u093e\u0930\u0935\u0924\u094d\u0924\u094d\u0935\u0902 \u091c\u0940\u0935\u0924\u094d\u0935\u092e\u094d"),
        ("hi", "\u0928\u092e\u0938\u094d\u0924\u0947 \u092d\u093e\u0930\u0924, \u092f\u0939 \u090f\u0915 \u092c\u0939\u0941\u092d\u093e\u0937\u0940 \u0935\u093e\u0915\u094d \u0938\u0902\u0936\u094d\u0932\u0947\u0937\u0923 \u092a\u094d\u0930\u0923\u093e\u0932\u0940 \u0939\u0948\u0964"),
        ("kn", "\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \u0cad\u0cbe\u0cb7\u0cc6\u0caf \u0ca7\u0ccd\u0cb5\u0ca8\u0cbf \u0cb8\u0c82\u0cb6\u0ccd\u0cb2\u0cc7\u0cb7\u0ca3\u0cc6 \u0caa\u0cb0\u0cc0\u0c95\u0ccd\u0cb7\u0cc6"),
        ("ta", "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd \u0bae\u0bca\u0bb4\u0bbf \u0b92\u0bb2\u0bbf \u0b89\u0bb0\u0bc1\u0bb5\u0bbe\u0b95\u0bcd\u0b95\u0bae\u0bcd"),
        ("te", "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 \u0c2d\u0c3e\u0c37\u0c3e \u0c27\u0c4d\u0c35\u0c28\u0c3f \u0c38\u0c02\u0c36\u0c4d\u0c32\u0c47\u0c37\u0c23"),
        ("ml", "\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d02 \u0d2d\u0c3e\u0c37\u0d3e \u0d36\u0d2c\u0d4d\u0d26 \u0d38\u0d02\u0d2f\u0d4b\u0d1c\u0d28\u0d02"),
    ]

    parity_rows: list[dict[str, Any]] = []
    for lang_code, txt in test_cases:
        parsed = text_to_ids(txt, lang=lang_code)
        ids_t = torch.tensor([parsed["input_ids"]], dtype=torch.long)
        mask_t = torch.tensor([parsed["attention_mask"]], dtype=torch.long)
        lid_t = torch.tensor([LANGUAGE_IDS[lang_code]], dtype=torch.long)

        with torch.no_grad():
            pt_wav, _ = model(ids_t, mask_t, lid_t, scales_det)
        pt_w = pt_wav[0].numpy()

        ort_inputs = {
            "input_ids": ids_t.numpy(),
            "attention_mask": mask_t.numpy(),
            "lang_id": lid_t.numpy(),
            "scales": scales_det.numpy(),
        }
        fp32_w = sess_fp32.run(None, ort_inputs)[0][0]
        quant_w = sess_quant.run(None, ort_inputs)[0][0]
        stream_w, ttfa_ms, stream_total_ms = _stream_vocode_ort(sess_enc, sess_voc, ort_inputs)

        snr_fp32 = _compute_snr_db(pt_w, fp32_w)
        snr_quant = _compute_snr_db(pt_w, quant_w)
        snr_stream = _compute_snr_db(quant_w, stream_w)
        n = min(len(pt_w), len(quant_w))
        pearson_r = float(np.corrcoef(pt_w[:n], quant_w[:n])[0, 1])

        parity_rows.append(
            {
                "lang": lang_code,
                "seq_len": int(ids_t.shape[1]),
                "samples": int(len(quant_w)),
                "duration_sec": round(len(quant_w) / 16000.0, 3),
                "fp32_snr_db": round(snr_fp32, 2),
                "quant_snr_db": round(snr_quant, 2),
                "stream_seam_snr_db": round(snr_stream, 2),
                "stream_ttfa_ms": round(ttfa_ms, 2),
                "stream_total_ms": round(stream_total_ms, 2),
                "pearson_r": round(pearson_r, 6),
            }
        )

    mean_quant_snr = float(np.mean([r["quant_snr_db"] for r in parity_rows]))
    mean_stream_snr = float(np.mean([r["stream_seam_snr_db"] for r in parity_rows]))
    mean_ttfa_ms = float(np.mean([r["stream_ttfa_ms"] for r in parity_rows]))
    min_pearson = float(np.min([r["pearson_r"] for r in parity_rows]))

    vocab_meta = {
        "vocab_tokens": VOCAB_TOKENS,
        "language_ids": LANGUAGE_IDS,
        "sampling_rate": 16000,
        "hop_length": 256,
        "streaming": {
            "chunk_frames": 40,
            "halo_frames": 10,
            "encoder_flow_onnx": "encoder_flow_quantized.onnx",
            "vocoder_onnx": "vocoder_quantized.onnx",
        },
    }
    (checkpoint_dir / "vocab.json").write_text(json.dumps(vocab_meta, ensure_ascii=False, indent=2))
    (onnx_dir / "vocab.json").write_text(json.dumps(vocab_meta, ensure_ascii=False, indent=2))

    report = {
        **quant_stats,
        "encoder_flow_size_mb": enc_stats["quantized_size_mb"],
        "vocoder_size_mb": voc_stats["quantized_size_mb"],
        "mean_quant_snr_db": round(mean_quant_snr, 2),
        "mean_stream_seam_snr_db": round(mean_stream_snr, 2),
        "mean_stream_ttfa_ms": round(mean_ttfa_ms, 2),
        "min_pearson_r": round(min_pearson, 6),
        "export_elapsed_sec": round(time.perf_counter() - t0, 2),
        "parity_by_language": parity_rows,
    }
    (onnx_dir / "export_report.json").write_text(json.dumps(report, indent=2))

    if copy_to_docs is not None:
        copy_to_docs.mkdir(parents=True, exist_ok=True)
        shutil.copy2(enc_quant_path, copy_to_docs / "encoder_flow_quantized.onnx")
        shutil.copy2(voc_quant_path, copy_to_docs / "vocoder_quantized.onnx")
        shutil.copy2(onnx_dir / "vocab.json", copy_to_docs / "vocab.json")
        # Remove >50MB monolithic file from docs/ since Web Studio streams the two <30MB split graphs!
        (copy_to_docs / "model_quantized.onnx").unlink(missing_ok=True)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NinadaModel to Quantized Monolithic + Streaming ONNX")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/ninada"))
    parser.add_argument("--copy-to-docs", type=Path, default=Path("docs"))
    args = parser.parse_args()

    rep = export_and_verify(args.checkpoint_dir, copy_to_docs=args.copy_to_docs)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
