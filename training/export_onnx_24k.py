#!/usr/bin/env python3
"""
Export Ninada v3.0 (24,000 Hz) PyTorch model to:
  1. `model.safetensors`
  2. `voices/{sa,hi,kn,ta,te,ml}.bin` (522 KB per language, [510, 1, 256] float32)
  3. `onnx/ninada_24k_q8f16.onnx` (< 98 MB unified 6-language browser WASM SIMD / WebGPU / Node / Python ONNX)
  4. `samples_v3/{sa,hi,kn,ta,te,ml}.wav` (24 kHz verification audio)
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "python" / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ninada.ipa_g2p import IPA_VOCAB, ipa_to_ids, text_to_ipa
from training.modeling_ninada_24k import (
    HOP_LENGTH,
    SAMPLE_RATE,
    SUPPORTED_LANGS,
    VOICE_MAX_LEN,
    Ninada24kModel,
    Ninada24kONNXWrapper,
)

CANONICAL_PROBES: dict[str, str] = {
    "sa": "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥",
    "hi": "नमस्ते, आज का मौसम बहुत अच्छा है और हम सभी साथ मिलकर खुश हैं।",
    "kn": "ನಮಸ್ಕಾರ, ಇಂದು ಬೆಂಗಳೂರಿನಲ್ಲಿ ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.",
    "ta": "வணக்கம், இன்று சென்னையில் வானிலை மிகவும் இனிமையாக உள்ளது, நன்றி.",
    "te": "నమస్కారం, ఈరోజు హైదరాబాద్‌లో వాతావరణం చాలా బాగుంది.",
    "ml": "നമസ്കാരം, ഇന്ന് കേരളത്തിൽ വളരെ നല്ല കാലാവസ്ഥയാണ്, നന്ദി.",
}


def compress_q8_onnx_with_fp16_weights(
    fp32_onnx_path: Path,
    out_q8f16_path: Path,
    min_elems: int = 512,
    w8_conv_min_elems: int = 2048,
) -> dict[str, Any]:
    """
    Compress FP32 ONNX initializers & Constant tensors into <86 MB unified W8A32/FP16 ONNX with ZERO `ConvInteger`
    or `DynamicQuantizeLinear` ops so `onnxruntime-web` WASM SIMD128 ConstantFolds to 100% native multi-thread `f32x4`!
    """
    model = onnx.load(str(fp32_onnx_path))
    new_inits = []
    prefix_nodes = []
    converted_w8 = 0
    converted_fp16 = 0
    saved_bytes = 0

    def _pack_float_array(arr: np.ndarray, out_name: str, tag: str) -> bool:
        nonlocal converted_w8, converted_fp16, saved_bytes
        if arr.ndim >= 2 and arr.size >= w8_conv_min_elems:
            flat = arr.reshape(arr.shape[0], -1).astype(np.float32)
            max_abs = np.max(np.abs(flat), axis=1)
            scale = np.where(max_abs > 1e-12, max_abs / 127.0, 1.0).astype(np.float32)
            scale_shape = [arr.shape[0]] + [1] * (arr.ndim - 1)
            scale_nd = scale.reshape(scale_shape)
            q_int8 = np.clip(np.round(arr / scale_nd), -127, 127).astype(np.int8)

            q_name = f"{tag}_w8_packed"
            s_name = f"{tag}_w8_scale"
            c_name = f"{tag}_w8_f32"
            new_inits.append(numpy_helper.from_array(q_int8, name=q_name))
            new_inits.append(numpy_helper.from_array(scale_nd, name=s_name))
            prefix_nodes.append(
                helper.make_node("Cast", inputs=[q_name], outputs=[c_name], to=TensorProto.FLOAT)
            )
            prefix_nodes.append(
                helper.make_node("Mul", inputs=[c_name, s_name], outputs=[out_name])
            )
            converted_w8 += 1
            saved_bytes += arr.nbytes - (q_int8.nbytes + scale_nd.nbytes)
            return True
        if arr.size >= min_elems:
            fp16_name = f"{tag}_fp16_packed"
            fp16_arr = arr.astype(np.float16)
            new_inits.append(numpy_helper.from_array(fp16_arr, name=fp16_name))
            prefix_nodes.append(
                helper.make_node("Cast", inputs=[fp16_name], outputs=[out_name], to=TensorProto.FLOAT)
            )
            converted_fp16 += 1
            saved_bytes += arr.nbytes - fp16_arr.nbytes
            return True
        return False

    for i_idx, init in enumerate(model.graph.initializer):
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            if _pack_float_array(arr, init.name, f"init_{i_idx}"):
                continue
        new_inits.append(init)

    kept_nodes = []
    for n_idx, node in enumerate(model.graph.node):
        if node.op_type == "Constant" and len(node.output) == 1:
            replaced = False
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t)
                    if _pack_float_array(arr, node.output[0], f"const_{n_idx}"):
                        replaced = True
                    break
            if replaced:
                continue
        kept_nodes.append(node)

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)
    del model.graph.node[:]
    model.graph.node.extend(prefix_nodes)
    model.graph.node.extend(kept_nodes)

    out_q8f16_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_q8f16_path))
    size_mb = round(out_q8f16_path.stat().st_size / (1024 * 1024), 2)
    return {
        "converted_w8_initializers": converted_w8,
        "converted_fp16_initializers": converted_fp16,
        "saved_mb": round(saved_bytes / (1024 * 1024), 2),
        "size_mb": size_mb,
    }


def export_ninada_24k_bundle(
    model: Ninada24kModel,
    out_dir: Path,
    keep_fp32_onnx: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = out_dir / "onnx"
    voices_dir = out_dir / "voices"
    samples_dir = out_dir / "samples_v3"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    voices_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    model_cpu = model.cpu().eval()

    # 1. Save safetensors weights
    clean_sd = {k: v.detach().cpu().contiguous() for k, v in model_cpu.state_dict().items()}
    safetensors_path = out_dir / "model.safetensors"
    save_file(clean_sd, str(safetensors_path))

    # 2. Export per-language 24 kHz studio voice banks [510, 1, 256] float32
    voice_manifest: dict[str, Any] = {}
    with torch.no_grad():
        for lang in SUPPORTED_LANGS:
            bank = model_cpu.get_voice_bank(lang).detach().cpu().numpy().astype(np.float32)
            assert bank.shape == (VOICE_MAX_LEN, 1, 256), bank.shape
            vpath = voices_dir / f"{lang}.bin"
            bank.tofile(str(vpath))
            voice_manifest[lang] = {
                "file": f"voices/{lang}.bin",
                "bytes": int(vpath.stat().st_size),
                "shape": [VOICE_MAX_LEN, 1, 256],
                "l2_norm_mean": round(float(np.linalg.norm(bank[:, 0, :], axis=-1).mean()), 4),
            }

    # 3. Export dynamic FP32 ONNX -> Quantize Dynamic QUInt8 -> Compress FP16 weights -> ninada_24k_q8f16.onnx
    wrapper = Ninada24kONNXWrapper(model_cpu).eval()
    dummy_ids = torch.LongTensor([[0, 56, 83, 55, 156, 83, 61, 62, 47, 158, 4, 0]])
    dummy_style = model_cpu.get_style_for_tokens("sa", dummy_ids.shape[1] - 2).detach()
    dummy_speed = torch.tensor([1.0], dtype=torch.float32)

    fp32_onnx_path = onnx_dir / "ninada_24k_fp32.onnx"
    q8_tmp_path = onnx_dir / "ninada_24k_q8_tmp.onnx"
    q8f16_onnx_path = onnx_dir / "ninada_24k_q8f16.onnx"

    onnx_export_kwargs: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        onnx_export_kwargs["dynamo"] = False

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_style, dummy_speed),
        str(fp32_onnx_path),
        input_names=["input_ids", "style", "speed"],
        output_names=["waveform", "durations"],
        dynamic_axes={
            "input_ids": {1: "sequence_length"},
            "waveform": {1: "num_samples"},
            "durations": {0: "sequence_length"},
        },
        opset_version=17,
        **onnx_export_kwargs,
    )
    fp32_mb = round(fp32_onnx_path.stat().st_size / (1024 * 1024), 2)

    prep_tmp_path = onnx_dir / "ninada_24k_prep_tmp.onnx"
    quant_pre_process(
        input_model_path=str(fp32_onnx_path),
        output_model_path=str(prep_tmp_path),
        skip_symbolic_shape=True,
    )
    # Keep 24 kHz sample-rate 9-weight harmonic merger (m_source.l_linear, 36 bytes) in exact FP32
    prep_graph = onnx.load(str(prep_tmp_path), load_external_data=False)
    exclude_nodes = [
        n.name
        for n in prep_graph.graph.node
        if n.name and ("m_source/l_linear" in n.name or "m_source.l_linear" in n.name)
    ]
    del prep_graph
    quantize_dynamic(
        model_input=str(prep_tmp_path),
        model_output=str(q8f16_onnx_path),
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=exclude_nodes,
    )
    prep_tmp_path.unlink(missing_ok=True)
    q8_tmp_path.unlink(missing_ok=True)
    if not keep_fp32_onnx:
        fp32_onnx_path.unlink(missing_ok=True)
    q8_mb = round(q8f16_onnx_path.stat().st_size / (1024 * 1024), 2)
    pack_info = {
        "pipeline": "quant_pre_process + quantize_dynamic(QUInt8)",
        "size_mb": q8_mb,
        "saved_mb": round(fp32_mb - q8_mb, 2),
    }

    # 4. Verify ONNX Runtime inference & synthesize canonical 24 kHz studio probes across all 6 languages
    sess = ort.InferenceSession(str(q8f16_onnx_path), providers=["CPUExecutionProvider"])
    probe_report: dict[str, Any] = {}
    for lang, text in CANONICAL_PROBES.items():
        ipa = text_to_ipa(text, lang=lang)
        ids = ipa_to_ids(ipa)
        ids_np = np.asarray([ids], dtype=np.int64)
        bank = np.fromfile(str(voices_dir / f"{lang}.bin"), dtype=np.float32).reshape(VOICE_MAX_LEN, 1, 256)
        style_np = bank[min(VOICE_MAX_LEN - 1, max(0, len(ids) - 2))]
        t0 = time.perf_counter()
        wav_out, dur_out = sess.run(
            None,
            {
                "input_ids": ids_np,
                "style": style_np,
                "speed": np.array([1.0], dtype=np.float32),
            },
        )
        elapsed = time.perf_counter() - t0
        pcm = np.asarray(wav_out, dtype=np.float32).reshape(-1)
        durs = np.asarray(dur_out, dtype=np.int64).reshape(-1)
        assert np.all(np.isfinite(pcm)), f"Non-finite PCM in {lang}"
        rms = float(np.sqrt(np.mean(pcm * pcm)))
        assert rms > 0.02, f"Unexpectedly quiet synthesis for {lang}: rms={rms}"
        assert durs.shape[0] == len(ids), (durs.shape, len(ids))
        assert pcm.shape[0] == int(durs.sum()) * HOP_LENGTH, (pcm.shape[0], int(durs.sum()) * HOP_LENGTH)

        wav_path = samples_dir / f"{lang}.wav"
        sf.write(str(wav_path), pcm, SAMPLE_RATE)
        dur_sec = pcm.shape[0] / SAMPLE_RATE
        probe_report[lang] = {
            "ipa": ipa,
            "num_tokens": len(ids),
            "duration_sec": round(dur_sec, 3),
            "latency_ms": round(elapsed * 1000.0, 2),
            "rtf": round(elapsed / max(1e-6, dur_sec), 4),
            "rms": round(rms, 4),
            "peak": round(float(np.max(np.abs(pcm))), 4),
        }

    export_meta = {
        "architecture": "Ninada-24k-StyleTTS2-PLBERT-iSTFTNet",
        "version": "3.0.0",
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "vocab_size": 178,
        "retroflex_lateral_slot": 79,
        "fp32_size_mb": fp32_mb,
        "q8f16_size_mb": pack_info["size_mb"],
        "pack_info": pack_info,
        "onnx_file": "onnx/ninada_24k_q8f16.onnx",
        "voices": voice_manifest,
        "probes": probe_report,
        "vocab": IPA_VOCAB,
    }
    with open(out_dir / "config_v3.json", "w", encoding="utf-8") as f:
        json.dump(export_meta, f, ensure_ascii=False, indent=2)

    return export_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="", help="Optional .pt/.safetensors state_dict path")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--keep-fp32", action="store_true")
    args = ap.parse_args()

    model = Ninada24kModel.from_pretrained_warmstart()
    if args.checkpoint:
        ckpt_p = Path(args.checkpoint)
        if ckpt_p.suffix == ".safetensors":
            from safetensors.torch import load_file
            sd = load_file(str(ckpt_p))
        else:
            sd = torch.load(str(ckpt_p), map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)

    meta = export_ninada_24k_bundle(model, Path(args.out_dir), keep_fp32_onnx=args.keep_fp32)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
