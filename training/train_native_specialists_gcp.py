#!/usr/bin/env python3
"""
Ninada v4.0 — Native Per-Language 24,000 Hz Studio VITS Training, ONNX Export & Scientific Benchmark.

Eliminates 100% of cross-language dataset & voice mismatch by training 6 dedicated native
single-speaker 24,000 Hz Studio VITS Male specialists (`onnx/{sa,kn,hi,ta,te,ml}.onnx`, ~28 MB each)
plus 6 dedicated native Female studio specialists (`onnx/{sa,kn,hi,ta,te,ml}_female.onnx`):
  - `kn`: Native Kannada (`SYSPIN/vits_Kannada_{Male,Female}` -> `ai4bharat/Rasa` Kannada 24 kHz)
  - `sa`: Classical Sanskrit (`SYSPIN/vits_Marathi_{Male,Female}` -> `ai4bharat/Rasa` Sanskrit 24 kHz; ZERO Hindi!)
  - `te`: Native Telugu (`SYSPIN/vits_Telugu_{Male,Female}` -> `ai4bharat/Rasa` Telugu 24 kHz)
  - `hi`: Native Hindi (`SYSPIN/vits_Hindi_{Male,Female}` -> `ai4bharat/Rasa` Hindi 24 kHz)
  - `ta`: Native Tamil (`SYSPIN/vits_Kannada_{Male,Female}` Brahmic +0x0100 -> `ai4bharat/Rasa` Tamil 24 kHz)
  - `ml`: Native Malayalam (`SYSPIN/vits_Kannada_{Male,Female}` Brahmic -0x0080 -> `ai4bharat/Rasa` Malayalam 24 kHz)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from onnx import TensorProto, helper, numpy_helper
from safetensors.torch import load_file, save_file
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "python" / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.run_bench import bootstrap_ci, compute_acoustic_metrics_24k, precompute_gt_features_24k
from ninada.engine import synthesize_klatt_pcm
from training.modeling_studio_vits import (
    HOP_LENGTH,
    SAMPLE_RATE_24K,
    SEGMENT_FRAMES,
    SUPPORTED_LANGS,
    MultiSpecExtractor24k,
    StudioCharTokenizer,
    StudioVitsModel,
    StudioVitsONNXWrapper,
    build_target_tokenizer_and_remap_embeddings,
    clean_native_text,
    discriminator_loss,
    feature_loss,
    generate_path,
    generator_loss,
    kl_loss,
    remap_embeddings_to_existing_tokenizer,
    sequence_mask,
    slice_segments,
)

NEUTRAL_RASA_STYLES = {
    "BOOK",
    "WIKI",
    "NEWS",
    "CONV",
    "INDIC",
    "PROPER NOUN",
    "ALEXA",
    "BB",
    "UMANG",
    "DIGI",
    "CHAT",
}

RASA_LANG_DIRS = {
    "sa": "Sanskrit",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
}

CANONICAL_PROBES: dict[str, str] = {
    "sa": "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥",
    "hi": "नमस्ते, आज का मौसम बहुत अच्छा है और हम सभी साथ मिलकर खुश हैं।",
    "kn": "ನಮಸ್ಕಾರ, ಇಂದು ಬೆಂಗಳೂರಿನಲ್ಲಿ ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.",
    "ta": "வணக்கம், இன்று சென்னையில் வானிலை மிகவும் இனிமையாக உள்ளது, நன்றி.",
    "te": "నమస్కారం, ఈరోజు హైదరాబాద్‌లో వాతావరణం చాలా బాగుంది.",
    "ml": "നമസ്കാരം, ഇന്ന് കേരളത്തിൽ വളരെ നല്ല കാലാവസ്ഥയാണ്, നന്ദി.",
    "en": "Hello and welcome to Bengaluru, where science, literature, and studio speech technology meet.",
    "mr": "नमस्कार, आज पुण्यातील हवामान खूप छान आणि आल्हाददायक आहे, धन्यवाद.",
    "bn": "নমস্কার, আজ কলকাতার আবহাওয়া খুব সুন্দর এবং মনোরম রয়েছে, ধন্যবাদ।",
    "gu": "નમસ્તે, આજે અમદાવાદમાં હવામાન ખૂબ જ સુંદર અને આનંદદાયક છે.",
    "pa": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, ਅੱਜ ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਮੌਸਮ ਬਹੁਤ ਸੁਹਾਵਣਾ ਅਤੇ ਵਧੀਆ ਹੈ।",
    "or": "ନମସ୍କାର, ଆଜି ଭୁବନେଶ୍ୱରରେ ପାଣିପାଗ ବହୁତ ସୁନ୍ଦର ଏବଂ ମନୋରମ ଅଛି।",
}

SPECIALIST_RECIPES: dict[str, dict[str, Any]] = {
    "kn": {
        "male_donor": "SYSPIN/vits_Kannada_Male",
        "female_donor": "SYSPIN/vits_Kannada_Female",
        "freeze_text_backbone": True,
        "male_steps": 900,
        "female_steps": 650,
        "lr_g": 6.5e-5,
        "lr_text": 1.5e-5,
        "lr_d": 6.5e-5,
        "target_male_utts": 1200,
        "target_female_utts": 800,
    },
    "te": {
        "male_donor": "SYSPIN/vits_Telugu_Male",
        "female_donor": "SYSPIN/vits_Telugu_Female",
        "male_steps": 2000,
        "female_steps": 1200,
        "lr_g": 1.4e-4,
        "lr_text": 1.4e-4,
        "lr_d": 1.4e-4,
        "target_male_utts": 1000,
        "target_female_utts": 700,
    },
    "hi": {
        "male_donor": "SYSPIN/vits_Hindi_Male",
        "female_donor": "SYSPIN/vits_Hindi_Female",
        "male_steps": 2000,
        "female_steps": 1200,
        "lr_g": 1.4e-4,
        "lr_text": 1.4e-4,
        "lr_d": 1.4e-4,
        "target_male_utts": 1000,
        "target_female_utts": 700,
    },
    "sa": {
        "male_donor": "SYSPIN/vits_Marathi_Male",
        "female_donor": "SYSPIN/vits_Marathi_Female",
        "male_steps": 3800,
        "female_steps": 2200,
        "lr_g": 1.8e-4,
        "lr_text": 2.8e-4,
        "lr_d": 1.6e-4,
        "target_male_utts": 1600,
        "target_female_utts": 1000,
    },
    "ta": {
        "male_donor": "SYSPIN/vits_Kannada_Male",
        "female_donor": "SYSPIN/vits_Kannada_Female",
        "male_steps": 4800,
        "female_steps": 2800,
        "lr_g": 2.0e-4,
        "lr_text": 3.4e-4,
        "lr_d": 1.8e-4,
        "target_male_utts": 1800,
        "target_female_utts": 1000,
    },
    "ml": {
        "male_donor": "SYSPIN/vits_Kannada_Male",
        "female_donor": "SYSPIN/vits_Kannada_Female",
        "male_steps": 4800,
        "female_steps": 2800,
        "lr_g": 2.0e-4,
        "lr_text": 3.4e-4,
        "lr_d": 1.8e-4,
        "target_male_utts": 1800,
        "target_female_utts": 1100,
    },
    "en": {
        "male_donor": "SYSPIN/vits_English_Male",
        "female_donor": "SYSPIN/vits_English_Female",
        "freeze_text_backbone": False,
        "male_steps": 8000,
        "female_steps": 8000,
        "lr_g": 1.6e-4,
        "lr_text": 1.6e-4,
        "lr_d": 1.5e-4,
        "target_male_utts": 3600,
        "target_female_utts": 3600,
    },
    "mr": {
        "male_donor": "SYSPIN/vits_Marathi_Male",
        "female_donor": "SYSPIN/vits_Marathi_Female",
        "male_steps": 1800,
        "female_steps": 1200,
        "lr_g": 1.3e-4,
        "lr_text": 1.0e-4,
        "lr_d": 1.3e-4,
        "target_male_utts": 1100,
        "target_female_utts": 800,
    },
    "bn": {
        "male_donor": "SYSPIN/vits_Bengali_Male",
        "female_donor": "SYSPIN/vits_Bengali_Female",
        "male_steps": 1800,
        "female_steps": 1200,
        "lr_g": 1.3e-4,
        "lr_text": 1.0e-4,
        "lr_d": 1.3e-4,
        "target_male_utts": 1100,
        "target_female_utts": 800,
    },
    "gu": {
        "male_donor": "SYSPIN/vits_Gujarati_Male",
        "female_donor": "SYSPIN/vits_Gujarati_Female",
        "male_steps": 1800,
        "female_steps": 1200,
        "lr_g": 1.3e-4,
        "lr_text": 1.0e-4,
        "lr_d": 1.3e-4,
        "target_male_utts": 1100,
        "target_female_utts": 800,
    },
    "pa": {
        "male_donor": "SYSPIN/vits_Hindi_Male",
        "female_donor": "SYSPIN/vits_Hindi_Female",
        "male_steps": 4200,
        "female_steps": 2500,
        "lr_g": 2.0e-4,
        "lr_text": 3.4e-4,
        "lr_d": 1.8e-4,
        "target_male_utts": 1600,
        "target_female_utts": 1000,
    },
    "or": {
        "male_donor": "SYSPIN/vits_Bengali_Male",
        "female_donor": "SYSPIN/vits_Bengali_Female",
        "male_steps": 4200,
        "female_steps": 2500,
        "lr_g": 2.0e-4,
        "lr_text": 3.4e-4,
        "lr_d": 1.8e-4,
        "target_male_utts": 1600,
        "target_female_utts": 1000,
    },
}


# ---------------------------------------------------------------------------
# GCS Incremental Sync Helpers
# ---------------------------------------------------------------------------


def gcs_upload_if_configured(bucket_name: str | None, local_path: Path, blob_path: str) -> None:
    if not bucket_name or not local_path.exists():
        return
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bucket.blob(blob_path).upload_from_filename(str(local_path), timeout=900)
        print(f"[GCS] Uploaded {local_path.name} -> gs://{bucket_name}/{blob_path}", flush=True)
    except Exception as exc:
        print(f"[WARN] GCS upload failed for {blob_path}: {exc}", flush=True)


def gcs_download_if_exists(bucket_name: str | None, blob_path: str, local_path: Path) -> bool:
    if not bucket_name:
        return False
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        if not blob.exists():
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        print(f"[GCS] Restored gs://{bucket_name}/{blob_path} -> {local_path}", flush=True)
        return True
    except Exception as exc:
        print(f"[WARN] GCS check/download failed for {blob_path}: {exc}", flush=True)
        return False


# ---------------------------------------------------------------------------
# W8A32 + FP16 ONNX Compression Preserving Spline & ConvPost Heads in FP16
# ---------------------------------------------------------------------------


def compress_studio_vits_onnx(
    fp32_onnx_path: Path,
    out_q8f16_path: Path,
    min_elems: int = 256,
    w8_conv_min_elems: int = 1024,
) -> dict[str, Any]:
    """
    Compress StudioVits FP32 ONNX (~109 MB) into ~28 MB W8A32 + FP16 ONNX with ZERO `ConvInteger`
    activation quantization. Keeps `conv_post` and `duration_predictor` spline projection heads in FP16
    so frame durations and harmonic phases match PyTorch FP32 (`< 0.5 dB` MCD).
    """
    model = onnx.load(str(fp32_onnx_path))
    consumers: dict[str, list[str]] = {}
    for n in model.graph.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n.name or "")

    new_inits = []
    prefix_nodes = []
    converted_w8 = 0
    converted_fp16 = 0
    saved_bytes = 0

    def _pack_float_array(arr: np.ndarray, out_name: str, tag: str, cnames: list[str]) -> bool:
        nonlocal converted_w8, converted_fp16, saved_bytes
        is_sensitive_head = (
            ("conv_post" in out_name)
            or ("duration_predictor.flows" in out_name and "proj" in out_name)
            or any(("conv_post" in c or ("duration_predictor" in c and "proj" in c)) for c in cnames)
        )
        if is_sensitive_head and arr.size >= 128:
            fp16_name = f"{tag}_fp16_packed"
            fp16_arr = arr.astype(np.float16)
            new_inits.append(numpy_helper.from_array(fp16_arr, name=fp16_name))
            prefix_nodes.append(
                helper.make_node("Cast", inputs=[fp16_name], outputs=[out_name], to=TensorProto.FLOAT)
            )
            converted_fp16 += 1
            saved_bytes += arr.nbytes - fp16_arr.nbytes
            return True

        if arr.ndim >= 2 and arr.size >= w8_conv_min_elems:
            if arr.shape[0] <= 2 and arr.shape[1] >= 16:
                s0, s1 = arr.shape[0], arr.shape[1]
                flat = arr.reshape(s0 * s1, -1).astype(np.float32)
                max_abs = np.max(np.abs(flat), axis=1)
                scale = np.where(max_abs > 1e-12, max_abs / 127.0, 1.0).astype(np.float32)
                scale_nd = scale.reshape([s0, s1] + [1] * (arr.ndim - 2))
            else:
                flat = arr.reshape(arr.shape[0], -1).astype(np.float32)
                max_abs = np.max(flat_abs := np.abs(flat), axis=1)
                scale = np.where(max_abs > 1e-12, max_abs / 127.0, 1.0).astype(np.float32)
                scale_nd = scale.reshape([arr.shape[0]] + [1] * (arr.ndim - 1))
            q_int8 = np.clip(np.round(arr / scale_nd), -127, 127).astype(np.int8)

            q_name = f"{tag}_w8_packed"
            s_name = f"{tag}_w8_scale"
            c_name = f"{tag}_w8_f32"
            new_inits.append(numpy_helper.from_array(q_int8, name=q_name))
            new_inits.append(numpy_helper.from_array(scale_nd, name=s_name))
            prefix_nodes.append(
                helper.make_node("Cast", inputs=[q_name], outputs=[c_name], to=TensorProto.FLOAT)
            )
            prefix_nodes.append(helper.make_node("Mul", inputs=[c_name, s_name], outputs=[out_name]))
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
            if _pack_float_array(arr, init.name, f"init_{i_idx}", consumers.get(init.name, [])):
                continue
        new_inits.append(init)

    kept_nodes = []
    for n_idx, node in enumerate(model.graph.node):
        if node.op_type == "Constant" and len(node.output) == 1:
            replaced = False
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t)
                    c_list = consumers.get(node.output[0], [node.name or ""])
                    if _pack_float_array(arr, node.output[0], f"const_{n_idx}", c_list):
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
    return {
        "converted_w8": converted_w8,
        "converted_fp16": converted_fp16,
        "saved_mb": round(saved_bytes / (1024 * 1024), 2),
        "size_mb": round(out_q8f16_path.stat().st_size / (1024 * 1024), 2),
    }


def export_specialist_to_onnx(
    model: StudioVitsModel,
    tok: StudioCharTokenizer,
    out_onnx_path: Path,
    probe_text: str,
    sample_wav_path: Path,
) -> dict[str, Any]:
    export_copy = copy.deepcopy(model).cpu().eval()
    export_copy.strip_weight_norm_for_export()
    wrapper = StudioVitsONNXWrapper(export_copy).eval()

    tmp_fp32 = out_onnx_path.parent / f"_tmp_{out_onnx_path.stem}_fp32.onnx"
    out_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    sample_wav_path.parent.mkdir(parents=True, exist_ok=True)

    probe_ids = tok.text_to_ids(probe_text)
    dummy_ids = torch.tensor([probe_ids[: min(len(probe_ids), 19)]], dtype=torch.long)
    dummy_scales = torch.tensor([1.0, 0.667, 0.8], dtype=torch.float32)

    kw: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_scales),
        str(tmp_fp32),
        input_names=["input_ids", "scales"],
        output_names=["waveform", "durations"],
        dynamic_axes={
            "input_ids": {1: "tokens"},
            "waveform": {1: "samples"},
            "durations": {1: "tokens"},
        },
        opset_version=17,
        do_constant_folding=True,
        **kw,
    )
    stats = compress_studio_vits_onnx(tmp_fp32, out_onnx_path)
    if tmp_fp32.exists():
        tmp_fp32.unlink()

    sess = ort.InferenceSession(str(out_onnx_path), providers=["CPUExecutionProvider"])
    ort_wav, ort_dur = sess.run(
        None,
        {
            "input_ids": np.asarray([probe_ids], dtype=np.int64),
            "scales": np.asarray([1.0, 0.40, 0.0], dtype=np.float32),
        },
    )
    wav_1d = np.asarray(ort_wav[0], dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(wav_1d))) if wav_1d.size else 0.0
    if peak > 0.95:
        wav_1d = wav_1d * (0.92 / peak)
    sf.write(str(sample_wav_path), wav_1d, SAMPLE_RATE_24K, subtype="PCM_16")
    stats["probe_duration_sec"] = round(wav_1d.size / SAMPLE_RATE_24K, 3)
    stats["probe_tokens"] = len(probe_ids)
    return stats


# ---------------------------------------------------------------------------
# Studio 24,000 Hz Audio Cleaning & `ai4bharat/Rasa` Native Split Curation
# ---------------------------------------------------------------------------


def resample_to_24k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if orig_sr == SAMPLE_RATE_24K:
        return wav
    if orig_sr == 48000:
        return resample_poly(wav, 1, 2).astype(np.float32)
    frac = Fraction(SAMPLE_RATE_24K, orig_sr).limit_denominator(200)
    return resample_poly(wav, frac.numerator, frac.denominator).astype(np.float32)


def clean_studio_vits_24k(wav: np.ndarray, pad_tail_sec: float = 0.145) -> np.ndarray | None:
    if wav.size < int(0.95 * SAMPLE_RATE_24K):
        return None
    wav = wav - float(np.mean(wav))
    peak = float(np.max(np.abs(wav)))
    if peak < 0.015 or not np.isfinite(peak):
        return None
    wav = wav / peak

    win = int(0.020 * SAMPLE_RATE_24K)
    env = np.convolve(np.abs(wav), np.ones(win, dtype=np.float32) / win, mode="same")
    active = np.where(env > 0.0025)[0]
    if active.size < int(0.65 * SAMPLE_RATE_24K):
        return None
    pad_lead = int(0.090 * SAMPLE_RATE_24K)
    pad_tail = int(pad_tail_sec * SAMPLE_RATE_24K)
    s = max(0, int(active[0]) - pad_lead)
    e = min(wav.size, int(active[-1]) + pad_tail)
    wav = wav[s:e]

    n_frames = wav.size // HOP_LENGTH
    if n_frames < 38 or n_frames > 860:  # ~0.97s .. ~9.17s @ 24 kHz
        return None
    wav = wav[: n_frames * HOP_LENGTH].copy()

    fade = int(0.006 * SAMPLE_RATE_24K)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    wav[:fade] *= ramp
    wav[-fade:] *= ramp[::-1]

    rms = float(np.sqrt(np.mean(wav * wav)))
    if rms < 0.012:
        return None
    wav = wav * (0.115 / rms)
    peak_after = float(np.max(np.abs(wav)))
    if peak_after > 0.92:
        wav = wav * (0.92 / peak_after)
    return wav.astype(np.float32)


def interleave_from_ends(shards: list[str]) -> list[str]:
    out: list[str] = []
    lo, hi = 0, len(shards) - 1
    while lo <= hi:
        out.append(shards[lo])
        if hi != lo:
            out.append(shards[hi])
        lo += 1
        hi -= 1
    return out


def gcs_blob_exists(bucket_name: str | None, blob_path: str) -> bool:
    if not bucket_name:
        return False
    try:
        from google.cloud import storage

        client = storage.Client()
        return bool(client.bucket(bucket_name).blob(blob_path).exists())
    except Exception:
        return False


def gcs_exists_v5_or_v4(bucket_name: str | None, v5_rel: str, v4_rel: str | None = None) -> bool:
    if not bucket_name:
        return False
    if gcs_blob_exists(bucket_name, f"ninada_v5/{v5_rel}"):
        return True
    fallback_rel = v4_rel if v4_rel is not None else v5_rel.replace("samples_v5/", "samples_v4/")
    return gcs_blob_exists(bucket_name, f"ninada_v4/{fallback_rel}")


def gcs_restore_v5_or_v4(
    bucket_name: str | None,
    v5_rel: str,
    local_path: Path,
    v4_rel: str | None = None,
) -> bool:
    if not bucket_name:
        return False
    if gcs_download_if_exists(bucket_name, f"ninada_v5/{v5_rel}", local_path):
        return True
    fallback_rel = v4_rel if v4_rel is not None else v5_rel.replace("samples_v5/", "samples_v4/")
    return gcs_download_if_exists(bucket_name, f"ninada_v4/{fallback_rel}", local_path)


def _clean_indictts_english_transcript(raw_text: str) -> str:
    """Repair legacy Windows-1252 apostrophes (`\\ufffd`) and drop corrupted lines."""
    txt = re.sub(r"(?<=[A-Za-z])\ufffd(?=[sStTdDmMrRvVlL]\b)", "'", str(raw_text or ""))
    txt = txt.replace("\ufffd", "")
    return re.sub(r"\s+", " ", txt).strip()


def stream_indictts_english_splits(
    lang_wav_dir: Path,
    t_male: int,
    t_fem: int,
    need_eval: int,
    eval_blacklist: set[str],
    seen_local: set[str],
    hf_tok: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Stream single-speaker 48,000 Hz -> 24,000 Hz full-band broadcast Studio Indian-English splits
    from `SPRINGLab/IndicTTS-English`:
      - Male: `train_hindifullmale_*.wav` (shards 08, 09, 10)
      - Female: `train_hindifullfemale_*.wav` (shards 07, 08)
    """
    fem_holdout = 40 if t_fem >= 100 else 0
    shard_indices: list[int] = []
    if t_fem > 0:
        shard_indices.extend([7, 8])
    if t_male + need_eval > 0:
        for idx in (9, 8, 10):
            if idx not in shard_indices:
                shard_indices.append(idx)
    ordered_shards = [f"data/train-{idx:05d}-of-00028.parquet" for idx in shard_indices]

    male_list: list[dict[str, Any]] = []
    fem_list: list[dict[str, Any]] = []

    for shard in ordered_shards:
        if len(male_list) >= t_male + need_eval and len(fem_list) >= t_fem + fem_holdout:
            break
        try:
            pq_path = hf_hub_download(
                repo_id="SPRINGLab/IndicTTS-English",
                filename=shard,
                repo_type="dataset",
                token=hf_tok,
            )
            pf = pq.ParquetFile(pq_path)
        except Exception as exc:
            print(f"[WARN] IndicTTS-English shard {shard} skipped: {exc}", flush=True)
            continue

        for batch in pf.iter_batches(batch_size=64):
            if len(male_list) >= t_male + need_eval and len(fem_list) >= t_fem + fem_holdout:
                break
            cols = batch.to_pydict()
            texts = cols.get("text", [])
            audio_col = cols.get("audio", [])

            for i in range(len(texts)):
                aud = audio_col[i] if i < len(audio_col) else None
                if not isinstance(aud, dict):
                    continue
                apath = str(aud.get("path") or "").strip().lower()
                if apath.startswith("train_hindifullmale_"):
                    g_str = "Male"
                elif apath.startswith("train_hindifullfemale_"):
                    g_str = "Female"
                else:
                    continue

                if g_str == "Male" and len(male_list) >= t_male + need_eval:
                    continue
                if g_str == "Female" and len(fem_list) >= t_fem + fem_holdout:
                    continue

                if "\ufffd" in str(texts[i] or "") and not re.search(
                    r"(?<=[A-Za-z])\ufffd(?=[sStTdDmMrRvVlL]\b)", str(texts[i] or "")
                ):
                    continue
                raw_text = _clean_indictts_english_transcript(str(texts[i] or ""))
                if len(raw_text) < 10 or len(raw_text) > 260:
                    continue
                norm_t = clean_native_text(raw_text, "en")
                if len(norm_t) < 8:
                    continue
                key = f"en:{norm_t}"
                if key in eval_blacklist or key in seen_local:
                    continue

                b_payload = aud.get("bytes")
                if not b_payload:
                    continue
                try:
                    raw_wav, orig_sr = sf.read(io.BytesIO(b_payload), dtype="float32")
                except Exception:
                    continue
                if raw_wav.ndim > 1:
                    raw_wav = raw_wav.mean(axis=1)
                if raw_wav.size == 0 or int(orig_sr) < 24000 or float(np.max(np.abs(raw_wav))) >= 0.998:
                    continue

                wav_24k = clean_studio_vits_24k(resample_to_24k(raw_wav, int(orig_sr)), pad_tail_sec=0.18)
                if wav_24k is None:
                    continue

                dur_s = wav_24k.size / SAMPLE_RATE_24K
                cps = len(norm_t) / max(dur_s, 0.1)
                if cps < 4.5 or cps > 25.0:
                    continue

                seen_local.add(key)
                uid = hashlib.sha1(f"{g_str}:{key}".encode("utf-8")).hexdigest()[:12]
                wpath = lang_wav_dir / g_str / f"en_{uid}.wav"
                sf.write(str(wpath), wav_24k, SAMPLE_RATE_24K, subtype="PCM_16")
                item = {
                    "utt_id": f"en_{uid}",
                    "lang": "en",
                    "gender": g_str,
                    "style": "STUDIO",
                    "text": raw_text,
                    "local_wav": str(wpath),
                    "n_frames": wav_24k.size // HOP_LENGTH,
                    "duration_sec": round(dur_s, 4),
                }
                if g_str == "Male":
                    male_list.append(item)
                else:
                    fem_list.append(item)

        try:
            if os.path.exists(pq_path) and os.path.islink(pq_path):
                real_p = os.path.realpath(pq_path)
                os.unlink(pq_path)
                if os.path.exists(real_p):
                    os.unlink(real_p)
        except Exception:
            pass

    if fem_holdout > 0 and len(fem_list) > fem_holdout:
        fem_list = fem_list[fem_holdout:]
    return male_list, fem_list


def prepare_heldout_and_native_splits(
    work_dir: Path,
    out_dir: Path,
    gcs_bucket: str | None,
    smoke_limit: int = 0,
    force_langs: set[str] | None = None,
    force_male_langs: set[str] | None = None,
    force_fem_langs: set[str] | None = None,
    only_langs: set[str] | None = None,
    skip_bench: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    """
    1. Restores leakage-free held-out `val` and `test` studio splits from
       `gs://training-gemina/ninada_v3/curated_24k_cache.tar.gz` + per-language `ninada_v5/eval_splits/{lang}_eval.tar.gz`.
    2. Streams native per-language Male & Female 24,000 Hz studio splits from `ai4bharat/Rasa`
       (and `SPRINGLab/IndicTTS-English` for `en`), strictly excluding all held-out `val`/`test` transcripts.
    """
    cache_dir = work_dir / "heldout_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    eval_blacklist: set[str] = set()
    seeded_male: dict[str, list[dict[str, Any]]] = {lc: [] for lc in SUPPORTED_LANGS}
    seeded_seen: dict[str, set[str]] = {lc: set() for lc in SUPPORTED_LANGS}

    f_male_set = force_male_langs if force_male_langs is not None else (force_langs or set())
    f_fem_set = force_fem_langs if force_fem_langs is not None else (force_langs or set())
    active_langs = only_langs if only_langs else set(SUPPORTED_LANGS)
    need_v3_cache = not (skip_bench and active_langs.isdisjoint({"sa", "hi", "kn", "te", "ta", "ml"}))

    tar_local = work_dir / "curated_24k_cache.tar.gz"
    if not smoke_limit and need_v3_cache and gcs_download_if_exists(gcs_bucket, "ninada_v3/curated_24k_cache.tar.gz", tar_local):
        with tarfile.open(tar_local, "r:gz") as tf:
            tf.extractall(path=cache_dir)
        tar_local.unlink(missing_ok=True)
        # Locate train_v3.jsonl, val_v3.jsonl, and test_v3.jsonl
        tr_files = list(cache_dir.rglob("train*.jsonl"))
        val_files = list(cache_dir.rglob("val*.jsonl"))
        test_files = list(cache_dir.rglob("test*.jsonl"))
        wav_lookup = {p.name: p for p in cache_dir.rglob("*.wav")}
        if val_files and test_files and wav_lookup:
            test_gt_dir = out_dir / "test_gt_wavs"
            test_gt_dir.mkdir(parents=True, exist_ok=True)
            for line in val_files[0].read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                r["utt_id"] = str(r.get("utt_id") or r.get("id") or Path(r["wav_path"]).stem)
                wn = Path(r["wav_path"]).name
                if wn in wav_lookup:
                    val_rows.append(r)
                    eval_blacklist.add(f"{r['lang']}:{clean_native_text(r['text'], r['lang'])}")
            for line in test_files[0].read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                r["utt_id"] = str(r.get("utt_id") or r.get("id") or Path(r["wav_path"]).stem)
                wn = Path(r["wav_path"]).name
                if wn in wav_lookup:
                    dst = test_gt_dir / wn
                    shutil.copy2(wav_lookup[wn], dst)
                    r["local_wav"] = str(dst)
                    r["wav_path"] = f"test_gt_wavs/{wn}"
                    test_rows.append(r)
                    eval_blacklist.add(f"{r['lang']}:{clean_native_text(r['text'], r['lang'])}")
            if tr_files:
                for line in tr_files[0].read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    lc = r.get("lang", "")
                    if lc not in seeded_male:
                        continue
                    wn = Path(r["wav_path"]).name
                    if wn not in wav_lookup:
                        continue
                    norm_t = clean_native_text(r["text"], lc)
                    key = f"{lc}:{norm_t}"
                    if len(norm_t) < 6 or key in eval_blacklist or key in seeded_seen[lc]:
                        continue
                    w_np, sr_c = sf.read(str(wav_lookup[wn]), dtype="float32")
                    if sr_c != SAMPLE_RATE_24K:
                        w_np = resample_to_24k(w_np, int(sr_c))
                    nf = w_np.shape[0] // HOP_LENGTH
                    if nf < 38 or nf > 860:
                        continue
                    w_np = w_np[: nf * HOP_LENGTH]
                    m_dir = work_dir / "rasa_24k" / lc / "Male"
                    m_dir.mkdir(parents=True, exist_ok=True)
                    uid = str(r.get("utt_id") or r.get("id") or Path(wn).stem)
                    dst_m = m_dir / f"{uid}.wav"
                    sf.write(str(dst_m), w_np, SAMPLE_RATE_24K, subtype="PCM_16")
                    seeded_seen[lc].add(key)
                    seeded_male[lc].append(
                        {
                            "utt_id": uid,
                            "lang": lc,
                            "gender": "Male",
                            "style": str(r.get("style", "BOOK")),
                            "text": r["text"],
                            "local_wav": str(dst_m),
                            "n_frames": nf,
                            "duration_sec": round(w_np.shape[0] / SAMPLE_RATE_24K, 4),
                        }
                    )
            print(
                f"[DATA] Loaded leakage-free held-out val={len(val_rows)}, test={len(test_rows)} + seeded Male={sum(len(v) for v in seeded_male.values())} studio utterances",
                flush=True,
            )
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Restore cached per-language held-out eval splits (`ninada_v5/eval_splits/{lc}_eval.tar.gz`) for new languages
    if not smoke_limit and gcs_bucket:
        test_gt_dir = out_dir / "test_gt_wavs"
        test_gt_dir.mkdir(parents=True, exist_ok=True)
        for lc in SUPPORTED_LANGS:
            if skip_bench and lc not in active_langs:
                continue
            if sum(1 for r in test_rows if r["lang"] == lc) >= 40:
                continue
            eval_tar = work_dir / f"{lc}_eval.tar.gz"
            eval_blob_name = "en_indictts_eval.tar.gz" if lc == "en" else f"{lc}_eval.tar.gz"
            if gcs_download_if_exists(gcs_bucket, f"ninada_v5/eval_splits/{eval_blob_name}", eval_tar):
                ex_dir = work_dir / f"eval_extract_{lc}"
                ex_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(eval_tar, "r:gz") as tf:
                    tf.extractall(path=ex_dir)
                eval_tar.unlink(missing_ok=True)
                w_map = {p.name: p for p in ex_dir.rglob("*.wav")}
                vf = ex_dir / "val.jsonl"
                tf_p = ex_dir / "test.jsonl"
                if vf.exists():
                    for line in vf.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        rv = json.loads(line)
                        val_rows.append(rv)
                        eval_blacklist.add(f"{lc}:{clean_native_text(rv['text'], lc)}")
                if tf_p.exists():
                    for line in tf_p.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        rt = json.loads(line)
                        wn = Path(rt["wav_path"]).name
                        if wn in w_map:
                            dst = test_gt_dir / wn
                            shutil.copy2(w_map[wn], dst)
                            rt["local_wav"] = str(dst)
                            rt["wav_path"] = f"test_gt_wavs/{wn}"
                            test_rows.append(rt)
                            eval_blacklist.add(f"{lc}:{clean_native_text(rt['text'], lc)}")
                shutil.rmtree(ex_dir, ignore_errors=True)
                print(
                    f"[DATA] Restored cached leakage-free eval split for {lc} (test_n={sum(1 for r in test_rows if r['lang'] == lc)})",
                    flush=True,
                )

    hf_tok = os.environ.get("HF_TOKEN") or None
    rasa_files: list[str] | None = None
    train_data: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for lang in SUPPORTED_LANGS:
        if lang not in active_langs:
            train_data[lang] = {"Male": list(seeded_male.get(lang, [])), "Female": []}
            continue
        recipe = SPECIALIST_RECIPES[lang]
        t_male = smoke_limit if smoke_limit > 0 else int(recipe["target_male_utts"])
        t_fem = smoke_limit if smoke_limit > 0 else int(recipe["target_female_utts"])
        skip_male_env = os.environ.get("SKIP_MALE", "0").strip() == "1"
        existing_lang_test = sum(1 for r in test_rows if r["lang"] == lang)
        target_test_n = 2 if smoke_limit > 0 else 40
        need_eval = 0 if (skip_male_env and skip_bench) else ((4 if smoke_limit > 0 else 80) if existing_lang_test < target_test_n else 0)

        male_done = (
            skip_male_env
            or (
                not smoke_limit
                and lang not in f_male_set
                and gcs_exists_v5_or_v4(gcs_bucket, f"checkpoints/{lang}_male.safetensors")
                and gcs_exists_v5_or_v4(gcs_bucket, f"onnx/{lang}.onnx")
            )
        )
        fem_done = (
            os.environ.get("SKIP_FEMALE", "0").strip() == "1"
            or (
                not smoke_limit
                and lang not in f_fem_set
                and gcs_exists_v5_or_v4(gcs_bucket, f"onnx/{lang}_female.onnx")
                and gcs_exists_v5_or_v4(gcs_bucket, f"samples_v5/{lang}_female.wav")
            )
        )
        if male_done and fem_done and need_eval == 0:
            train_data[lang] = {"Male": list(seeded_male.get(lang, [])), "Female": []}
            print(f"[DATA] {lang}: Both Male & Female specialists + eval split already in GCS; skipping shard stream", flush=True)
            continue
        if male_done and need_eval == 0:
            t_male = 0
        if fem_done:
            t_fem = 0

        lang_wav_dir = work_dir / "rasa_24k" / lang
        (lang_wav_dir / "Male").mkdir(parents=True, exist_ok=True)
        (lang_wav_dir / "Female").mkdir(parents=True, exist_ok=True)

        seen_local: set[str] = set(seeded_seen.get(lang, set()))
        if lang == "en":
            male_list, fem_list = stream_indictts_english_splits(
                lang_wav_dir=lang_wav_dir,
                t_male=t_male,
                t_fem=t_fem,
                need_eval=need_eval,
                eval_blacklist=eval_blacklist,
                seen_local=seen_local,
                hf_tok=hf_tok,
            )
        else:
            if rasa_files is None:
                api = HfApi(token=hf_tok)
                rasa_files = api.list_repo_files("ai4bharat/Rasa", repo_type="dataset")
            lang_dir = RASA_LANG_DIRS[lang]
            tr_shards = sorted(f for f in rasa_files if f.startswith(f"{lang_dir}/train-") and f.endswith(".parquet"))
            te_shards = sorted(f for f in rasa_files if f.startswith(f"{lang_dir}/test-") and f.endswith(".parquet"))
            ordered_shards = interleave_from_ends(tr_shards) + interleave_from_ends(te_shards)

            male_list = list(seeded_male.get(lang, []))[: t_male + need_eval]
            fem_list = []

            for shard in ordered_shards:
                if len(male_list) >= t_male + need_eval and len(fem_list) >= t_fem:
                    break
                try:
                    pq_path = hf_hub_download(repo_id="ai4bharat/Rasa", filename=shard, repo_type="dataset", token=hf_tok)
                    pf = pq.ParquetFile(pq_path)
                except Exception as exc:
                    print(f"[WARN] Shard {shard} skipped: {exc}", flush=True)
                    continue

                for batch in pf.iter_batches(batch_size=48):
                    if len(male_list) >= t_male + need_eval and len(fem_list) >= t_fem:
                        break
                    cols = batch.to_pydict()
                    texts = cols.get("text", [])
                    genders = cols.get("gender", [""] * len(texts))
                    styles = cols.get("style", ["BOOK"] * len(texts))
                    raw_bytes_col = cols.get("bytes")
                    audio_struct_col = cols.get("audio")

                    for i in range(len(texts)):
                        g_str = str(genders[i] or "").strip()
                        if g_str not in ("Male", "Female"):
                            continue
                        if g_str == "Male" and len(male_list) >= t_male + need_eval:
                            continue
                        if g_str == "Female" and len(fem_list) >= t_fem:
                            continue

                        st = str(styles[i] or "").strip().upper()
                        if st and st not in NEUTRAL_RASA_STYLES:
                            continue

                        raw_text = str(texts[i] or "").strip()
                        if len(raw_text) < 8 or len(raw_text) > 260:
                            continue
                        norm_t = clean_native_text(raw_text, lang)
                        if len(norm_t) < 6:
                            continue
                        key = f"{lang}:{norm_t}"
                        if key in eval_blacklist or key in seen_local:
                            continue

                        b_payload = None
                        if raw_bytes_col is not None and raw_bytes_col[i]:
                            b_payload = raw_bytes_col[i]
                        elif audio_struct_col is not None and isinstance(audio_struct_col[i], dict):
                            b_payload = audio_struct_col[i].get("bytes")
                        if not b_payload:
                            continue

                        try:
                            raw_wav, orig_sr = sf.read(io.BytesIO(b_payload), dtype="float32")
                        except Exception:
                            continue

                        wav_24k = clean_studio_vits_24k(resample_to_24k(raw_wav, int(orig_sr)))
                        if wav_24k is None:
                            continue

                        dur_s = wav_24k.size / SAMPLE_RATE_24K
                        cps = len(norm_t) / max(dur_s, 0.1)
                        if cps < 3.5 or cps > 28.0:
                            continue

                        seen_local.add(key)
                        uid = hashlib.sha1(f"{g_str}:{key}".encode("utf-8")).hexdigest()[:12]
                        wpath = lang_wav_dir / g_str / f"{lang}_{uid}.wav"
                        sf.write(str(wpath), wav_24k, SAMPLE_RATE_24K, subtype="PCM_16")
                        item = {
                            "utt_id": f"{lang}_{uid}",
                            "lang": lang,
                            "gender": g_str,
                            "style": st,
                            "text": raw_text,
                            "local_wav": str(wpath),
                            "n_frames": wav_24k.size // HOP_LENGTH,
                            "duration_sec": round(dur_s, 4),
                        }
                        if g_str == "Male":
                            male_list.append(item)
                        else:
                            fem_list.append(item)

                # Free raw parquet shard from HF cache if disk space tight
                try:
                    if os.path.exists(pq_path) and os.path.islink(pq_path):
                        real_p = os.path.realpath(pq_path)
                        os.unlink(pq_path)
                        if os.path.exists(real_p):
                            os.unlink(real_p)
                except Exception:
                    pass

        if need_eval > 0:
            half = need_eval // 2
            test_gt_dir = out_dir / "test_gt_wavs"
            test_gt_dir.mkdir(parents=True, exist_ok=True)
            carved_val = male_list[:half]
            carved_test: list[dict[str, Any]] = []
            for r in carved_val:
                val_rows.append(r)
                eval_blacklist.add(f"{lang}:{clean_native_text(r['text'], lang)}")
            for r in male_list[half : 2 * half]:
                dst = test_gt_dir / f"{r['utt_id']}.wav"
                shutil.copy2(r["local_wav"], dst)
                r_copy = dict(r)
                r_copy["local_wav"] = str(dst)
                r_copy["wav_path"] = f"test_gt_wavs/{dst.name}"
                test_rows.append(r_copy)
                carved_test.append(r_copy)
                eval_blacklist.add(f"{lang}:{clean_native_text(r['text'], lang)}")
            male_list = male_list[2 * half :]

            # Persist leakage-free held-out split for this language to GCS immediately
            if not smoke_limit and gcs_bucket and len(carved_test) >= 40:
                stage_d = work_dir / f"eval_stage_{lang}"
                (stage_d / "test_gt_wavs").mkdir(parents=True, exist_ok=True)
                with open(stage_d / "val.jsonl", "w", encoding="utf-8") as fv:
                    for rv in carved_val:
                        fv.write(
                            json.dumps(
                                {
                                    "utt_id": rv["utt_id"],
                                    "lang": lang,
                                    "text": rv["text"],
                                    "duration_sec": rv["duration_sec"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                with open(stage_d / "test.jsonl", "w", encoding="utf-8") as ft:
                    for rt in carved_test:
                        shutil.copy2(rt["local_wav"], stage_d / "test_gt_wavs" / f"{rt['utt_id']}.wav")
                        ft.write(
                            json.dumps(
                                {
                                    "utt_id": rt["utt_id"],
                                    "lang": lang,
                                    "text": rt["text"],
                                    "wav_path": f"test_gt_wavs/{rt['utt_id']}.wav",
                                    "duration_sec": rt["duration_sec"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                eval_tar_out = work_dir / f"{lang}_eval.tar.gz"
                with tarfile.open(eval_tar_out, "w:gz") as tf_out:
                    tf_out.add(stage_d / "val.jsonl", arcname="val.jsonl")
                    tf_out.add(stage_d / "test.jsonl", arcname="test.jsonl")
                    tf_out.add(stage_d / "test_gt_wavs", arcname="test_gt_wavs")
                eval_blob_out = "en_indictts_eval.tar.gz" if lang == "en" else f"{lang}_eval.tar.gz"
                gcs_upload_if_configured(gcs_bucket, eval_tar_out, f"ninada_v5/eval_splits/{eval_blob_out}")
                eval_tar_out.unlink(missing_ok=True)
                shutil.rmtree(stage_d, ignore_errors=True)

        min_m = 0 if t_male == 0 else (4 if smoke_limit > 0 else 150)
        min_f = 0 if t_fem == 0 else (4 if smoke_limit > 0 else 100)
        if len(male_list) < min_m or len(fem_list) < min_f:
            raise RuntimeError(
                f"Insufficient native studio data for {lang}: Male={len(male_list)}, Female={len(fem_list)} (check HF_TOKEN access)"
            )
        train_data[lang] = {"Male": male_list, "Female": fem_list}
        print(
            f"[DATA] {lang}: Male train={len(male_list)}, Female train={len(fem_list)} native 24 kHz studio utterances",
            flush=True,
        )

    return val_rows, test_rows, train_data


# ---------------------------------------------------------------------------
# Fast Length-Bucketed Collation & Adversarial VITS Fine-Tuning
# ---------------------------------------------------------------------------


def make_bucketed_batches(
    items: list[dict[str, Any]],
    tok: StudioCharTokenizer,
    batch_size: int,
    rng: random.Random,
) -> list[list[dict[str, Any]]]:
    enriched = []
    for it in items:
        ids = it.get("_cached_ids")
        if ids is None:
            ids = tok.text_to_ids(it["text"])
            it["_cached_ids"] = ids
        if 5 <= len(ids) <= 460:
            enriched.append(it)
    # Sort by audio frame length with small jitter so similar lengths batch together without padding waste
    enriched.sort(key=lambda r: r["n_frames"] + rng.randint(-12, 12))
    batches = [enriched[i : i + batch_size] for i in range(0, len(enriched), batch_size)]
    rng.shuffle(batches)
    return [b for b in batches if len(b) >= max(2, batch_size // 2)]


def collate_vits_batch(
    batch: list[dict[str, Any]],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b = len(batch)
    max_x = max(len(it["_cached_ids"]) for it in batch)
    max_wav = max(it["n_frames"] * HOP_LENGTH for it in batch)

    x = torch.full((b, max_x), pad_id, dtype=torch.long)
    x_lengths = torch.zeros(b, dtype=torch.long)
    wavs = torch.zeros((b, max_wav), dtype=torch.float32)
    spec_lengths = torch.zeros(b, dtype=torch.long)

    for i, it in enumerate(batch):
        ids = it["_cached_ids"]
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        x_lengths[i] = len(ids)
        w_i16 = it.get("_wav_i16")
        if w_i16 is not None:
            w_np = w_i16.astype(np.float32) * (1.0 / 32767.0)
        else:
            w_np, _ = sf.read(it["local_wav"], dtype="float32")
        n_s = min(w_np.shape[0], max_wav)
        wavs[i, :n_s] = torch.from_numpy(w_np[:n_s])
        spec_lengths[i] = n_s // HOP_LENGTH

    return (
        x.to(device, non_blocking=True),
        x_lengths.to(device, non_blocking=True),
        wavs.to(device, non_blocking=True),
        spec_lengths.to(device, non_blocking=True),
    )


def load_donor_checkpoint(repo_id: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cfg_p = hf_hub_download(repo_id=repo_id, filename="config.json")
    pth_p = hf_hub_download(repo_id=repo_id, filename="best_model.pth")
    cfg = json.loads(Path(cfg_p).read_text(encoding="utf-8"))
    ckpt = torch.load(pth_p, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return cfg, sd


EXACT_DONOR_EN_SENTENCES = [
    "Welcome to Ninada.",
    "Natural studio speech streams privately inside your browser.",
    "This text to speech studio runs completely inside your web browser using WebAssembly and AudioWorklet streaming.",
    "Once cached, voices work offline with full privacy.",
    "Hello and welcome to Bengaluru, where science, literature, and studio speech technology meet.",
    "Good morning, everyone. Today we are exploring neural speech synthesis on modern devices.",
    "The quick brown fox jumps over the lazy dog near the quiet river bank.",
    "Please pack my box with five dozen liquor jugs before seven thirty tonight.",
    "Sphinx of black quartz, judge my vow with clarity and grace.",
    "She sells seashells by the seashore, while waves gently wash over the warm sand.",
    "The birch canoe slid on the smooth planks near the clear water.",
    "Glue the sheet to the dark blue background before framing it.",
    "It is easy to tell the depth of a well when sunlight shines down.",
    "Rice is often served in round bowls alongside warm lentil soup.",
    "The juice of lemons makes fine punch on a sunny afternoon.",
    "Four hours of steady work faced us before the evening train.",
    "The source of the huge river is the clear spring high in the mountains.",
    "A pot of tea helps to pass the evening with old friends.",
    "The salt breeze came across from the sea at twilight.",
    "Hello from Bengaluru! How are you doing today?",
    "Could you please repeat the last address slowly so I can write it down?",
    "Notice how questions rise naturally at the end, do they not?",
    "Are we meeting at nine fifteen tomorrow morning, or did the schedule change?",
    "Dr. Rao arrived from Chennai at nine fifteen and joined Prof. Sharma in the auditorium.",
    "Let us test numbers like forty two, one hundred eight, and three thousand five hundred.",
    "In twenty twenty six, browser WebGPU and SIMD acceleration unlocked real time studio audio.",
    "Bengaluru, Mysuru, Mumbai, Delhi, Kolkata, Hyderabad, and Pune are vibrant cities.",
    "When the rain stopped, sunlight filtered through the green canopy across Cubbon Park.",
    "Artificial intelligence works best when models respect human privacy and run locally.",
    "Every syllable, vowel, and consonant should sound warm, clear, and effortless.",
    "Smooth breath pauses between sentences make long passages pleasant and easy to follow.",
    "Careful engineering eliminates metallic buzz and preserves natural vocal warmth.",
    "Deep neural vocoders synthesize twenty four kilohertz waveforms frame by frame.",
    "Warmth in the lower harmonics gives a studio voice its natural human presence.",
    "Crisp sibilants and gentle plosives prevent listener fatigue during long chapters.",
    "Keep your eyes on the stars, and keep your feet firmly on the ground.",
    "Patience, curiosity, and rigorous testing turn a rough prototype into a reliable tool.",
    "From classical literature to daily news, clear pronunciation makes every word shine.",
    "Everything works offline after the initial model download completes.",
    "Thank you for listening! Have a wonderful and peaceful evening ahead.",
]


def adapt_exact_donor_24k_vocoder(
    model: StudioVitsModel,
    donor_sd: dict[str, torch.Tensor],
    remapped_emb: torch.Tensor,
    tok: StudioCharTokenizer,
    spec_ext: MultiSpecExtractor24k,
    device: torch.device,
    total_steps: int,
    batch_size: int,
    lr_g: float,
    lr_d: float,
    tag: str,
) -> dict[str, float]:
    """
    Preserve the 500,000-step native SYSPIN English `text_encoder`, `duration_predictor`, and `flow`
    100% intact (eliminating cross-accent latent clash from external American English corpora):
      1. Apply exact closed-form `24000 / 22050` (`160 / 147`) frame-rate shift inside
         `duration_predictor.flows[0].translation` (zero extra ONNX nodes).
      2. Synthesize zero-drift `(z_24, wav_24)` pairs directly from the untouched 22,050 Hz donor.
      3. Adapt ONLY `waveform_decoder` + `disc` to crisp 24,000 Hz output.
    """
    from ninada.studio_g2p import clean_studio_text

    donor_22k = StudioVitsModel(num_chars=tok.num_chars, use_discriminator=False)
    psd = {k: v.clone() for k, v in donor_sd.items() if not k.startswith("disc.")}
    psd["text_encoder.emb.weight"] = remapped_emb.clone()
    donor_22k.load_state_dict(psd, strict=True)
    donor_22k.eval()

    with torch.no_grad():
        ea = model.duration_predictor.flows[0]
        pace_mult = 1.12 if tag.lower().startswith("en_m") else 1.0
        rate_shift = math.log((float(SAMPLE_RATE_24K) / 22050.0) * pace_mult)
        ea.translation.data[0, 0] -= rate_shift * float(torch.exp(ea.log_scale.data[0, 0]).item())

    donor_flow = donor_22k.flow.to(device).eval()
    donor_voc = donor_22k.waveform_decoder.to(device).eval()
    teacher_flow = model.flow.to(device).eval()
    teacher_enc = model.text_encoder.cpu().eval()
    teacher_dp = model.duration_predictor.cpu().eval()

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for s_idx, raw_s in enumerate(EXACT_DONOR_EN_SENTENCES):
            ids = tok.text_to_ids(clean_studio_text(raw_s, "en"))
            if len(ids) < 5:
                continue
            x = torch.tensor([ids], dtype=torch.long)
            xl = torch.tensor([len(ids)], dtype=torch.long)
            enc_x, m_p, logs_p, x_mask = teacher_enc(x, xl)
            for k in range(3):
                seed = 1000 + s_idx * 17 + k
                torch.manual_seed(seed)
                logw = teacher_dp(enc_x, x_mask, reverse=True, noise_scale=0.62 if k > 0 else 0.0)
                w = torch.clamp_max(torch.exp(logw), 50.0) * x_mask
                w_ceil = torch.ceil(w)
                w_ceil = torch.cat([w_ceil[:, :, :-1], torch.clamp_min(w_ceil[:, :, -1:], 8.0)], dim=2) * x_mask
                y_len_24 = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
                T_24 = int(y_len_24.item())
                if T_24 < SEGMENT_FRAMES + 2:
                    continue
                y_mask_24 = sequence_mask(y_len_24, None).unsqueeze(1).to(x_mask.dtype)
                attn_24 = generate_path(w_ceil.squeeze(1), x_mask.transpose(1, 2) * y_mask_24)
                m_p_24 = torch.matmul(m_p, attn_24)
                logs_p_24 = torch.matmul(logs_p, attn_24)
                torch.manual_seed(seed + 99999)
                z_p_24 = m_p_24 + torch.randn_like(m_p_24) * torch.exp(logs_p_24) * (0.58 if k > 0 else 0.50)
                z_24 = (teacher_flow(z_p_24.to(device), y_mask_24.to(device), reverse=True) * y_mask_24.to(device)).cpu()

                T_22 = int(math.ceil(T_24 * 147.0 / 160.0)) + 2
                idx_f = torch.arange(T_22, dtype=torch.float32) * (160.0 / 147.0)
                idx_0 = torch.clamp(idx_f.floor().long(), 0, T_24 - 1)
                idx_1 = torch.clamp(idx_0 + 1, 0, T_24 - 1)
                alpha = (idx_f - idx_0.float()).view(1, 1, T_22)
                z_p_22 = z_p_24[:, :, idx_0] * (1.0 - alpha) + z_p_24[:, :, idx_1] * alpha
                z_22 = donor_flow(z_p_22.to(device), torch.ones((1, 1, T_22), dtype=torch.float32, device=device), reverse=True)
                wav_22 = donor_voc(z_22)[0, 0].cpu().numpy()
                wav_24 = resample_poly(wav_22, 160, 147).astype(np.float32)
                target_len = T_24 * HOP_LENGTH
                wav_24 = wav_24[:target_len] if len(wav_24) >= target_len else np.pad(wav_24, (0, target_len - len(wav_24)))
                pairs.append((z_24[0].numpy().astype(np.float16), wav_24))

    del donor_22k
    voc = model.waveform_decoder.to(device).train()
    disc = model.disc.to(device).train()
    spec_ext.to(device)
    eff_bs = min(batch_size, 8)
    optim_g = torch.optim.AdamW(voc.parameters(), lr=lr_g, betas=(0.8, 0.99), weight_decay=0.01)
    optim_d = torch.optim.AdamW(disc.parameters(), lr=lr_d, betas=(0.8, 0.99), weight_decay=0.01)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=max(total_steps, 1), eta_min=2.5e-5)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=max(total_steps, 1), eta_min=2.5e-5)

    rng = random.Random(20261105)
    t_start = time.perf_counter()
    for step in range(1, total_steps + 1):
        zb = np.empty((eff_bs, 192, SEGMENT_FRAMES), dtype=np.float32)
        yb = np.empty((eff_bs, 1, SEGMENT_FRAMES * HOP_LENGTH), dtype=np.float32)
        for b in range(eff_bs):
            zf, wf = pairs[rng.randrange(len(pairs))]
            f0 = rng.randint(0, zf.shape[1] - SEGMENT_FRAMES)
            zb[b] = zf[:, f0 : f0 + SEGMENT_FRAMES].astype(np.float32)
            s0 = f0 * HOP_LENGTH
            yb[b, 0] = wf[s0 : s0 + SEGMENT_FRAMES * HOP_LENGTH]
        z_t = torch.from_numpy(zb).to(device)
        y_t = torch.from_numpy(yb).to(device)
        y_hat = voc(z_t)
        adv_on = step > min(60, total_steps // 4)
        if adv_on:
            dr, dg, _, _ = disc(y_t, y_hat.detach())
            ld = discriminator_loss(dr, dg)
            optim_d.zero_grad(set_to_none=True)
            ld.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 100.0)
            optim_d.step()
            sched_d.step()
        mel_r = spec_ext.wav_to_mel(y_t.squeeze(1))
        mel_g = spec_ext.wav_to_mel(y_hat.squeeze(1))
        loss_mel = F.l1_loss(mel_r, mel_g) * 45.0
        if adv_on:
            _, dg2, fmr, fmg = disc(y_t, y_hat)
            lg = loss_mel + feature_loss(fmr, fmg) + generator_loss(dg2)
        else:
            lg = loss_mel
        optim_g.zero_grad(set_to_none=True)
        lg.backward()
        torch.nn.utils.clip_grad_norm_(voc.parameters(), 100.0)
        optim_g.step()
        sched_g.step()
        if step % 150 == 0 or step == total_steps:
            print(f"[{tag}:ExactDonor24k] step {step}/{total_steps} | mel_l1={(loss_mel.item()/45.0):.4f}", flush=True)

    model.cpu().eval()
    return {"steps": total_steps, "wall_sec": round(time.perf_counter() - t_start, 2), "exact_donor_24k": True}


def train_single_specialist(
    model: StudioVitsModel,
    tok: StudioCharTokenizer,
    train_items: list[dict[str, Any]],
    spec_ext: MultiSpecExtractor24k,
    device: torch.device,
    total_steps: int,
    batch_size: int,
    lr_g: float,
    lr_text: float,
    lr_d: float,
    tag: str,
    freeze_text_backbone: bool = False,
) -> dict[str, float]:
    model.to(device)
    model.train()
    spec_ext.to(device)

    if freeze_text_backbone:
        # Preserve 500,000-step SYSPIN native conjunct representations and prior projection bit-for-bit;
        # adapt only `duration_predictor`, `flow`, `posterior_encoder`, and `waveform_decoder` to 24 kHz!
        model.text_encoder.requires_grad_(False)
        model.text_encoder.eval()
        text_params = list(model.duration_predictor.parameters())
    else:
        text_params = list(model.text_encoder.parameters()) + list(model.duration_predictor.parameters())

    all_text_ids = {id(p) for p in model.text_encoder.parameters()} | {
        id(p) for p in model.duration_predictor.parameters()
    }
    acoustic_params = [
        p for p in model.generator_parameters() if id(p) not in all_text_ids and p.requires_grad
    ]
    trainable_gen_params = [p for p in model.generator_parameters() if p.requires_grad]

    optim_g = torch.optim.AdamW(
        [
            {"params": text_params, "lr": lr_text},
            {"params": acoustic_params, "lr": lr_g},
        ],
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.01,
    )
    optim_d = torch.optim.AdamW(
        model.discriminator_parameters(),
        lr=lr_d,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.01,
    )
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=max(total_steps, 1), eta_min=lr_g * 0.25)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=max(total_steps, 1), eta_min=lr_d * 0.25)

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    rng = random.Random(20261101)
    step = 0
    t_start = time.perf_counter()
    running = {"loss_d": 0.0, "loss_mel": 0.0, "loss_kl": 0.0, "loss_dur": 0.0, "loss_fm": 0.0}
    n_logged = 0

    for it in train_items:
        if "_wav_i16" not in it:
            w_tmp, _ = sf.read(it["local_wav"], dtype="float32")
            it["_wav_i16"] = np.clip(np.round(w_tmp * 32767.0), -32767, 32767).astype(np.int16)

    try:
        while step < total_steps:
            batches = make_bucketed_batches(train_items, tok, batch_size=batch_size, rng=rng)
            if not batches:
                break
            for batch in batches:
                if step >= total_steps:
                    break
                x, x_lengths, wavs, spec_lengths = collate_vits_batch(batch, tok.pad_id, device)
                spec = spec_ext.linear_spec(wavs)
                # Align exact frame count if stft produced +/- 1 frame
                f_actual = spec.size(2)
                spec_lengths = torch.clamp_max(spec_lengths, f_actual)

                with torch.amp.autocast(device_type=device.type if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=use_bf16):
                    y_hat, l_length, ids_slice, _attn, (z_p, logs_q, m_p, logs_p, z_mask) = model.forward_train(
                        x, x_lengths, spec, spec_lengths
                    )
                    y_real_slice = slice_segments(
                        wavs.unsqueeze(1), ids_slice * HOP_LENGTH, SEGMENT_FRAMES * HOP_LENGTH
                    )
                    y_d_rs, y_d_gs, _, _ = model.disc(y_real_slice, y_hat.detach())
                    loss_d = discriminator_loss(y_d_rs, y_d_gs)

                if torch.isfinite(loss_d):
                    optim_d.zero_grad(set_to_none=True)
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(model.discriminator_parameters(), 100.0)
                    optim_d.step()

                with torch.amp.autocast(device_type=device.type if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=use_bf16):
                    _, y_d_gs_g, fmap_r, fmap_g = model.disc(y_real_slice, y_hat)
                    spec_slice = slice_segments(spec, ids_slice, SEGMENT_FRAMES)
                    mel_real = spec_ext.spec_to_mel(spec_slice)
                    mel_hat = spec_ext.wav_to_mel(y_hat.squeeze(1))
                    n_mf = min(mel_real.size(2), mel_hat.size(2))
                    loss_mel = F.l1_loss(mel_real[:, :, :n_mf], mel_hat[:, :, :n_mf]) * 45.0
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen = generator_loss(y_d_gs_g)
                    loss_g_tot = loss_gen + loss_fm + loss_mel + loss_kl + l_length

                if torch.isfinite(loss_g_tot):
                    optim_g.zero_grad(set_to_none=True)
                    loss_g_tot.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_gen_params, 100.0)
                    optim_g.step()

                sched_g.step()
                sched_d.step()
                step += 1

                running["loss_d"] += float(loss_d.detach().item())
                running["loss_mel"] += float((loss_mel.detach() / 45.0).item())
                running["loss_kl"] += float(loss_kl.detach().item())
                running["loss_dur"] += float(l_length.detach().item())
                running["loss_fm"] += float(loss_fm.detach().item())
                n_logged += 1

                if step % 200 == 0 or step == total_steps:
                    elapsed = time.perf_counter() - t_start
                    sps = step / max(elapsed, 1e-3)
                    print(
                        f"[{tag}] step {step}/{total_steps} ({sps:.1f} it/s) | "
                        f"mel_l1={running['loss_mel']/max(n_logged,1):.4f} "
                        f"kl={running['loss_kl']/max(n_logged,1):.3f} "
                        f"dur={running['loss_dur']/max(n_logged,1):.3f} "
                        f"fm={running['loss_fm']/max(n_logged,1):.3f} "
                        f"d={running['loss_d']/max(n_logged,1):.3f}",
                        flush=True,
                    )
                    running = {k: 0.0 for k in running}
                    n_logged = 0
    finally:
        for it in train_items:
            it.pop("_wav_i16", None)
        if freeze_text_backbone:
            model.text_encoder.requires_grad_(True)

    total_sec = time.perf_counter() - t_start
    return {"steps": step, "wall_sec": round(total_sec, 2)}


# ---------------------------------------------------------------------------
# Held-Out 240-Utterance Scientific Benchmark (Exact Parity with `bench/`)
# ---------------------------------------------------------------------------


def evaluate_all_systems_on_heldout(
    test_rows: list[dict[str, Any]],
    num_train_total: int,
    num_val_total: int,
    tokenizers: dict[str, StudioCharTokenizer],
    warmstart_wavs: dict[str, tuple[np.ndarray, int, float]],
    pt_models: dict[str, StudioVitsModel],
    onnx_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    from transformers import AutoTokenizer, VitsModel

    gt_audio_map: dict[str, np.ndarray] = {}
    gt_dur_map: dict[str, float] = {}
    for r in test_rows:
        uid = r["utt_id"]
        w, sr = sf.read(r["local_wav"], dtype="float32")
        if sr != SAMPLE_RATE_24K:
            w = resample_to_24k(w, sr)
        gt_audio_map[uid] = w
        gt_dur_map[uid] = len(w) / float(SAMPLE_RATE_24K)

    with ThreadPoolExecutor(max_workers=8) as pool:
        feats = list(pool.map(lambda r: precompute_gt_features_24k(gt_audio_map[r["utt_id"]]), test_rows))
    gt_feat_map = {r["utt_id"]: f for r, f in zip(test_rows, feats)}

    def _agg(
        per_utt: list[dict[str, Any]],
        size_mb: float,
        sr_hz: int,
        ttfa_list: list[float],
        rtf_list: list[float],
    ) -> dict[str, Any]:
        mcds = [u["mcd"] for u in per_utt]
        hfs = [u["hf_err"] for u in per_utt]
        f0s = [u["f0"] for u in per_utt]
        durs = [u["dur_err"] for u in per_utt]
        voiced = [1.0 if u["voiced"] else 0.0 for u in per_utt]
        per_lang: dict[str, Any] = {}
        for lc in SUPPORTED_LANGS:
            sub = [u for u in per_utt if u["lang"] == lc]
            if not sub:
                continue
            per_lang[lc] = {
                "n": len(sub),
                "voiced_rate_pct": round(100.0 * float(np.mean([1.0 if u["voiced"] else 0.0 for u in sub])), 1),
                "mcd_mean": round(float(np.mean([u["mcd"] for u in sub])), 3),
                "hf_8_12k_err_mean": round(float(np.mean([u["hf_err"] for u in sub])), 3),
                "f0_rmse_mean": round(float(np.mean([u["f0"] for u in sub])), 2),
                "dur_mae_mean": round(float(np.mean([u["dur_err"] for u in sub])), 3),
            }
        return {
            "sample_rate_hz": sr_hz,
            "size_mb": round(size_mb, 2),
            "voiced_rate_pct": round(100.0 * float(np.mean(voiced)), 2),
            "mcd_db": bootstrap_ci(mcds),
            "hf_8_12k_err_db": bootstrap_ci(hfs),
            "f0_rmse_cents": bootstrap_ci(f0s),
            "dur_mae_sec": bootstrap_ci(durs),
            "ttfa_ms": {
                "p50": round(float(np.percentile(ttfa_list, 50)), 2),
                "p95": round(float(np.percentile(ttfa_list, 95)), 2),
            },
            "rtf_mean": round(float(np.mean(rtf_list)), 4),
            "per_lang": per_lang,
        }

    systems_out: dict[str, Any] = {}

    # 1 & 2. Meta MMS-TTS Baselines (`mms_hin_raw` & `mms_5_fleet`)
    mms_repos = {
        "sa": "facebook/mms-tts-hin",
        "hi": "facebook/mms-tts-hin",
        "kn": "facebook/mms-tts-kan",
        "ta": "facebook/mms-tts-tam",
        "te": "facebook/mms-tts-tel",
        "ml": "facebook/mms-tts-mal",
        "en": "facebook/mms-tts-eng",
        "mr": "facebook/mms-tts-mar",
        "bn": "facebook/mms-tts-ben",
        "gu": "facebook/mms-tts-guj",
        "pa": "facebook/mms-tts-pan",
        "or": "facebook/mms-tts-ory",
    }
    print(f"[BENCH] Evaluating mms_hin_raw & mms_5_fleet on {len(test_rows)} held-out studio utterances...", flush=True)
    loaded_mms: dict[str, tuple[Any, Any]] = {}
    active_test_langs = {r["lang"] for r in test_rows}
    needed_mms_repos = sorted({mms_repos[lc] for lc in active_test_langs} | {"facebook/mms-tts-hin"})
    for repo in needed_mms_repos:
        tk_m = AutoTokenizer.from_pretrained(repo)
        md_m = VitsModel.from_pretrained(repo).to(device).eval()
        loaded_mms[repo] = (tk_m, md_m)

    fleet_sz_mb = round(len(set(mms_repos.values())) * 138.8, 1)
    for sys_key, use_fleet, sz_mb in [("mms_hin_raw", False, 138.8), ("mms_5_fleet", True, fleet_sz_mb)]:
        utts, ttfas, rtfs = [], [], []
        for r in test_rows:
            uid, lc = r["utt_id"], r["lang"]
            repo = mms_repos[lc] if use_fleet else "facebook/mms-tts-hin"
            tk_m, md_m = loaded_mms[repo]
            t0 = time.perf_counter()
            inps = tk_m(r["text"], return_tensors="pt")
            if inps["input_ids"].shape[1] < 3:
                syn = np.zeros(1600, dtype=np.float32)
            else:
                inps = {k: v.to(device) for k, v in inps.items()}
                with torch.no_grad():
                    torch.manual_seed(42)
                    syn = md_m(**inps).waveform[0].detach().cpu().numpy().astype(np.float32)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            dur_sy = max(0.05, len(syn) / 16000.0)
            ttfas.append(dt_ms)
            rtfs.append((dt_ms / 1000.0) / dur_sy)
            mcd, hf, f0, v_ok = compute_acoustic_metrics_24k(
                gt_audio_map[uid], syn, syn_sr=16000, gt_cache=gt_feat_map[uid]
            )
            utts.append(
                {"utt_id": uid, "lang": lc, "mcd": mcd, "hf_err": hf, "f0": f0, "dur_err": abs(gt_dur_map[uid] - len(syn) / 16000.0), "voiced": v_ok}
            )
        systems_out[sys_key] = _agg(utts, sz_mb, 16000, ttfas, rtfs)

    del loaded_mms
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3. Untuned Warm-Start Ablation (`ninada_v3_warmstart` key preserved for backwards CI schema compatibility + v4)
    print("[BENCH] Aggregating untuned step-0 warm-start baseline...", flush=True)
    ws_utts, ws_ttfas, ws_rtfs = [], [], []
    for r in test_rows:
        uid, lc = r["utt_id"], r["lang"]
        syn, sr_w, dt_ms = warmstart_wavs[uid]
        dur_sy = max(0.05, len(syn) / float(sr_w))
        ws_ttfas.append(dt_ms)
        ws_rtfs.append((dt_ms / 1000.0) / dur_sy)
        mcd, hf, f0, v_ok = compute_acoustic_metrics_24k(
            gt_audio_map[uid], syn, syn_sr=sr_w, gt_cache=gt_feat_map[uid]
        )
        ws_utts.append(
            {"utt_id": uid, "lang": lc, "mcd": mcd, "hf_err": hf, "f0": f0, "dur_err": abs(gt_dur_map[uid] - len(syn) / float(sr_w)), "voiced": v_ok}
        )
    systems_out["ninada_v3_warmstart"] = _agg(ws_utts, 112.0, SAMPLE_RATE_24K, ws_ttfas, ws_rtfs)

    # 4 & 5. Ninada v5.0 Native Studio VITS Specialists (PyTorch FP32 & Active Browser ONNX ~28 MB)
    print("[BENCH] Evaluating Ninada v5.0 Native Specialists (PyTorch FP32 + CPU ONNX W8A32/FP16)...", flush=True)
    ort_sessions: dict[str, ort.InferenceSession] = {}
    onnx_sizes: list[float] = []
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    eval_langs = [lc for lc in SUPPORTED_LANGS if lc in active_test_langs]
    for lc in eval_langs:
        op = onnx_dir / f"{lc}.onnx"
        onnx_sizes.append(op.stat().st_size / (1024 * 1024))
        ort_sessions[lc] = ort.InferenceSession(str(op), sess_options=sess_opts, providers=["CPUExecutionProvider"])
        # Warmup ORT session once
        ort_sessions[lc].run(
            None,
            {
                "input_ids": np.asarray([tokenizers[lc].text_to_ids(CANONICAL_PROBES[lc])[:15]], dtype=np.int64),
                "scales": np.asarray([1.0, 0.60, 0.0], dtype=np.float32),
            },
        )

    active_onnx_mb = float(np.mean(onnx_sizes))
    pt_utts, pt_ttfas, pt_rtfs = [], [], []
    ox_utts, ox_ttfas, ox_rtfs = [], [], []
    parity_mcds: list[float] = []
    pt_durs_det: list[float] = []
    ox_durs_det: list[float] = []

    for r in test_rows:
        uid, lc = r["utt_id"], r["lang"]
        tok = tokenizers[lc]
        ids = tok.text_to_ids(r["text"])
        pt_m = pt_models[lc].to(device).eval()

        # Clause TTFA chunk simulation (first clause <= 55 tokens for streaming first-chunk latency)
        clause_ids = ids[: min(len(ids), 55)]
        if clause_ids[-1] != tok.blank_id:
            clause_ids = clause_ids + [tok.blank_id]

        # PyTorch FP32
        t0 = time.perf_counter()
        with torch.no_grad():
            torch.manual_seed(42)
            _w_first, _ = pt_m.infer(
                torch.tensor([clause_ids], dtype=torch.long, device=device),
                length_scale=1.0,
                noise_scale=0.40,
                noise_scale_dp=0.0,
            )
        ttfa_pt = (time.perf_counter() - t0) * 1000.0 if len(ids) > 55 else 0.0

        t1 = time.perf_counter()
        with torch.no_grad():
            torch.manual_seed(42)
            wav_pt_t, _ = pt_m.infer(
                torch.tensor([ids], dtype=torch.long, device=device),
                length_scale=1.0,
                noise_scale=0.40,
                noise_scale_dp=0.0,
            )
            wav_pt = wav_pt_t[0].detach().cpu().numpy().astype(np.float32)
        tot_pt_ms = (time.perf_counter() - t1) * 1000.0
        pt_ttfas.append(ttfa_pt if ttfa_pt > 0.0 else tot_pt_ms)
        pt_rtfs.append((tot_pt_ms / 1000.0) / max(0.05, len(wav_pt) / float(SAMPLE_RATE_24K)))

        mcd_p, hf_p, f0_p, vok_p = compute_acoustic_metrics_24k(
            gt_audio_map[uid], wav_pt, syn_sr=SAMPLE_RATE_24K, gt_cache=gt_feat_map[uid]
        )
        pt_utts.append(
            {"utt_id": uid, "lang": lc, "mcd": mcd_p, "hf_err": hf_p, "f0": f0_p, "dur_err": abs(gt_dur_map[uid] - len(wav_pt) / float(SAMPLE_RATE_24K)), "voiced": vok_p}
        )

        # CPU ONNX W8A32+FP16
        sess = ort_sessions[lc]
        scales_np = np.asarray([1.0, 0.40, 0.0], dtype=np.float32)
        t2 = time.perf_counter()
        if len(ids) > 55:
            sess.run(None, {"input_ids": np.asarray([clause_ids], dtype=np.int64), "scales": scales_np})
            ttfa_ox = (time.perf_counter() - t2) * 1000.0
        else:
            ttfa_ox = 0.0

        t3 = time.perf_counter()
        ort_out, _ = sess.run(None, {"input_ids": np.asarray([ids], dtype=np.int64), "scales": scales_np})
        tot_ox_ms = (time.perf_counter() - t3) * 1000.0
        wav_ox = np.asarray(ort_out[0], dtype=np.float32).reshape(-1)
        ox_ttfas.append(ttfa_ox if ttfa_ox > 0.0 else tot_ox_ms)
        ox_rtfs.append((tot_ox_ms / 1000.0) / max(0.05, len(wav_ox) / float(SAMPLE_RATE_24K)))

        mcd_o, hf_o, f0_o, vok_o = compute_acoustic_metrics_24k(
            gt_audio_map[uid], wav_ox, syn_sr=SAMPLE_RATE_24K, gt_cache=gt_feat_map[uid]
        )
        ox_utts.append(
            {"utt_id": uid, "lang": lc, "mcd": mcd_o, "hf_err": hf_o, "f0": f0_o, "dur_err": abs(gt_dur_map[uid] - len(wav_ox) / float(SAMPLE_RATE_24K)), "voiced": vok_o}
        )

        # Deterministic parity (`noise_scale=0, noise_scale_dp=0`) between PyTorch FP32 and W8A32+FP16 ONNX
        with torch.no_grad():
            w_pt_det_t, d_pt_det = pt_m.infer(
                torch.tensor([ids], dtype=torch.long, device=device),
                length_scale=1.0,
                noise_scale=0.0,
                noise_scale_dp=0.0,
            )
            w_pt_det = w_pt_det_t[0].detach().cpu().numpy().astype(np.float32)
        w_ox_det_arr, d_ox_det = sess.run(
            None,
            {"input_ids": np.asarray([ids], dtype=np.int64), "scales": np.asarray([1.0, 0.0, 0.0], dtype=np.float32)},
        )
        w_ox_det = np.asarray(w_ox_det_arr[0], dtype=np.float32).reshape(-1)
        p_mcd, _, _, _ = compute_acoustic_metrics_24k(w_pt_det, w_ox_det, syn_sr=SAMPLE_RATE_24K)
        parity_mcds.append(p_mcd)
        pt_durs_det.append(float(d_pt_det.sum().item()))
        ox_durs_det.append(float(np.sum(d_ox_det)))

    systems_out["ninada_v3_pytorch"] = _agg(pt_utts, 112.0, SAMPLE_RATE_24K, pt_ttfas, pt_rtfs)
    systems_out["ninada_v3_onnx"] = _agg(ox_utts, active_onnx_mb, SAMPLE_RATE_24K, ox_ttfas, ox_rtfs)

    # 6. Zero-Dep Klatt-Akshara Tier-2 Fallback
    kl_utts, kl_ttfas, kl_rtfs = [], [], []
    for r in test_rows:
        uid, lc = r["utt_id"], r["lang"]
        t0 = time.perf_counter()
        pcm_kl, _ = synthesize_klatt_pcm(r["text"], lang=lc)
        syn_k = np.asarray(pcm_kl, dtype=np.float32)
        dt_k = (time.perf_counter() - t0) * 1000.0
        kl_ttfas.append(dt_k)
        kl_rtfs.append((dt_k / 1000.0) / max(0.05, len(syn_k) / float(SAMPLE_RATE_24K)))
        mcd_k, hf_k, f0_k, vok_k = compute_acoustic_metrics_24k(
            gt_audio_map[uid], syn_k, syn_sr=SAMPLE_RATE_24K, gt_cache=gt_feat_map[uid]
        )
        kl_utts.append(
            {"utt_id": uid, "lang": lc, "mcd": mcd_k, "hf_err": hf_k, "f0": f0_k, "dur_err": abs(gt_dur_map[uid] - len(syn_k) / float(SAMPLE_RATE_24K)), "voiced": vok_k}
        )
    systems_out["ninada_klatt_tier2"] = _agg(kl_utts, 0.0, SAMPLE_RATE_24K, kl_ttfas, kl_rtfs)

    dur_r = float(np.corrcoef(pt_durs_det, ox_durs_det)[0, 1]) if len(pt_durs_det) > 2 else 1.0
    return {
        "benchmark": "ninada_24khz_native_studio_12lang",
        "version": "5.0.1",
        "sample_rate_hz": SAMPLE_RATE_24K,
        "num_train_utterances": num_train_total,
        "num_val_utterances": num_val_total,
        "num_test_utterances": len(test_rows),
        "bootstrap_resamples": 1000,
        "systems": systems_out,
        "onnx_parity": {
            "active_specialist_size_mb": round(active_onnx_mb, 2),
            "per_lang_onnx_mb": {lc: round(sz, 2) for lc, sz in zip(eval_langs, onnx_sizes)},
            "pt_vs_onnx_mcd_db": round(float(np.mean(parity_mcds)), 3),
            "duration_corr_r": round(dur_r, 5),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/ninada_v5_work"))
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/ninada_v5_out"))
    parser.add_argument("--gcs-bucket", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-steps", type=int, default=0, help="If >0, run fast smoke test")
    parser.add_argument("--force-langs", type=str, default="", help="Comma-separated langs to force retrain (e.g. 'en' or 'en:male')")
    parser.add_argument("--only-langs", type=str, default="", help="Optional comma-separated subset of langs to train in this worker")
    parser.add_argument("--skip-bench", action="store_true", help="Exit after uploading trained specialists without running 12-lang benchmark")
    args = parser.parse_args()
    raw_force = [s.strip().lower() for s in args.force_langs.split(",") if s.strip()]
    force_male_langs = {s.split(":")[0] for s in raw_force if ":" not in s or s.endswith(":male")}
    force_fem_langs = {s.split(":")[0] for s in raw_force if ":" not in s or s.endswith(":female")}
    force_any_langs = force_male_langs | force_fem_langs
    only_langs_set = {s.strip().lower() for s in args.only_langs.split(",") if s.strip()}
    target_langs = [lc for lc in SUPPORTED_LANGS if not only_langs_set or lc in only_langs_set]

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = args.out_dir / "onnx"
    samples_dir = args.out_dir / "samples_v5"
    ckpt_dir = args.out_dir / "checkpoints"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Ninada v5.0 Native Studio Specialists (12 Languages) on device={device}", flush=True)

    gcs_bucket = args.gcs_bucket.strip() or None
    resume_male_cont = os.environ.get("RESUME_MALE", "0").strip() == "1"
    val_rows, test_rows, train_data = prepare_heldout_and_native_splits(
        args.work_dir,
        args.out_dir,
        gcs_bucket=gcs_bucket,
        smoke_limit=12 if args.smoke_steps > 0 else 0,
        force_langs=force_any_langs,
        force_male_langs=force_male_langs,
        force_fem_langs=force_fem_langs,
        only_langs=only_langs_set or None,
        skip_bench=args.skip_bench,
    )

    spec_ext = MultiSpecExtractor24k().to(device)
    tokenizers: dict[str, StudioCharTokenizer] = {}
    pt_models: dict[str, StudioVitsModel] = {}
    warmstart_wavs: dict[str, tuple[np.ndarray, int, float]] = {}
    training_summary: dict[str, Any] = {}

    # 1. Train & Export Primary Native Male 24,000 Hz Specialists
    skip_male = os.environ.get("SKIP_MALE", "0").strip() == "1"
    for lang in target_langs:
        recipe = SPECIALIST_RECIPES[lang]
        donor_cfg, donor_sd = load_donor_checkpoint(recipe["male_donor"])
        tok, remapped_emb = build_target_tokenizer_and_remap_embeddings(
            lang, donor_cfg, donor_sd["text_encoder.emb.weight"]
        )
        tokenizers[lang] = tok
        if skip_male and args.skip_bench:
            del donor_sd
            print(f"[SKIP_MALE] Built tokenizer for {lang}; skipping Male training.", flush=True)
            continue

        net = StudioVitsModel(num_chars=tok.num_chars, use_discriminator=True)
        patched_sd = dict(donor_sd)
        patched_sd["text_encoder.emb.weight"] = remapped_emb
        net.load_state_dict(patched_sd, strict=True)
        donor_sr_init = int(donor_cfg.get("audio", {}).get("sample_rate", 22050))
        if donor_sr_init != SAMPLE_RATE_24K and not recipe.get("exact_donor_24k_vocoder_adapt"):
            with torch.no_grad():
                net.duration_predictor.flows[0].translation.data[0, 0] += math.log(
                    float(SAMPLE_RATE_24K) / float(donor_sr_init)
                )

        # Record step-0 untuned warmstart baseline on held-out test split for this language
        if not args.skip_bench:
            net_ws = net.to(device).eval()
            donor_sr = int(donor_cfg.get("audio", {}).get("sample_rate", 22050))
            for r in test_rows:
                if r["lang"] != lang:
                    continue
                ids = tok.text_to_ids(r["text"])
                t0 = time.perf_counter()
                with torch.no_grad():
                    torch.manual_seed(42)
                    w_ws, _ = net_ws.infer(
                        torch.tensor([ids], dtype=torch.long, device=device),
                        length_scale=1.0,
                        noise_scale=0.667,
                        noise_scale_dp=0.8,
                    )
                dt_ms = (time.perf_counter() - t0) * 1000.0
                warmstart_wavs[r["utt_id"]] = (w_ws[0].detach().cpu().numpy().astype(np.float32), donor_sr, dt_ms)

        # Check GCS resume cache (`ninada_v5/` first, fallback `ninada_v4/`) for completed Male specialist
        ckpt_st_path = ckpt_dir / f"{lang}_male.safetensors"
        out_onnx_p = onnx_dir / f"{lang}.onnx"
        sample_p = samples_dir / f"{lang}.wav"
        if (
            not args.smoke_steps
            and lang not in force_male_langs
            and gcs_restore_v5_or_v4(gcs_bucket, f"checkpoints/{lang}_male.safetensors", ckpt_st_path)
            and gcs_restore_v5_or_v4(gcs_bucket, f"onnx/{lang}.onnx", out_onnx_p)
        ):
            net.load_state_dict(load_file(str(ckpt_st_path)), strict=False)
            gcs_restore_v5_or_v4(gcs_bucket, f"samples_v5/{lang}.wav", sample_p)
            print(f"[RESUME] Restored trained {lang} Male specialist from GCS!", flush=True)
            t_stats = {"restored_from_gcs": True}
        else:
            steps = args.smoke_steps if args.smoke_steps > 0 else int(recipe["male_steps"])
            lr_g_eff = float(recipe["lr_g"])
            lr_text_eff = float(recipe["lr_text"])
            lr_d_eff = float(recipe["lr_d"])
            if resume_male_cont and gcs_restore_v5_or_v4(
                gcs_bucket, f"checkpoints/{lang}_male.safetensors", ckpt_st_path
            ):
                net.load_state_dict(load_file(str(ckpt_st_path)), strict=False)
                steps = int(os.environ.get("RESUME_STEPS", "2200"))
                lr_g_eff *= 0.55
                lr_text_eff *= 0.55
                lr_d_eff *= 0.55
                print(
                    f"[CONT] Continuing {lang}:Male from GCS checkpoint for +{steps} steps (lr_g={lr_g_eff:.2e}, lr_text={lr_text_eff:.2e})",
                    flush=True,
                )
            if recipe.get("exact_donor_24k_vocoder_adapt"):
                t_stats = adapt_exact_donor_24k_vocoder(
                    net,
                    donor_sd,
                    remapped_emb,
                    tok,
                    spec_ext,
                    device=device,
                    total_steps=steps,
                    batch_size=args.batch_size,
                    lr_g=lr_g_eff,
                    lr_d=lr_d_eff,
                    tag=f"{lang}:Male",
                )
            else:
                t_stats = train_single_specialist(
                    net,
                    tok,
                    train_data[lang]["Male"],
                    spec_ext,
                    device=device,
                    total_steps=steps,
                    batch_size=args.batch_size,
                    lr_g=lr_g_eff,
                    lr_text=lr_text_eff,
                    lr_d=lr_d_eff,
                    tag=f"{lang}:Male",
                    freeze_text_backbone=bool(recipe.get("freeze_text_backbone", False)),
                )
            gen_sd = {
                k: v.detach().cpu().contiguous()
                for k, v in net.state_dict().items()
                if not k.startswith("disc.")
            }
            save_file(gen_sd, str(ckpt_st_path))
            ox_stats = export_specialist_to_onnx(
                net, tok, out_onnx_p, CANONICAL_PROBES[lang], sample_p
            )
            t_stats.update(ox_stats)
            gcs_upload_if_configured(gcs_bucket, ckpt_st_path, f"ninada_v5/checkpoints/{lang}_male.safetensors")
            gcs_upload_if_configured(gcs_bucket, out_onnx_p, f"ninada_v5/onnx/{lang}.onnx")
            gcs_upload_if_configured(gcs_bucket, sample_p, f"ninada_v5/samples_v5/{lang}.wav")

        if not args.skip_bench:
            pt_models[lang] = copy.deepcopy(net).cpu().eval()
        del net
        training_summary[f"{lang}_male"] = t_stats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save canonical `vocabs.json` when all 12 languages are loaded
    if len(target_langs) == len(SUPPORTED_LANGS):
        vocabs_payload = {
            "version": "5.0.1",
            "sample_rate": SAMPLE_RATE_24K,
            "hop_length": HOP_LENGTH,
            "languages": {lc: tokenizers[lc].to_dict() for lc in SUPPORTED_LANGS},
        }
        vocabs_path = args.out_dir / "vocabs.json"
        vocabs_path.write_text(json.dumps(vocabs_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        gcs_upload_if_configured(gcs_bucket, vocabs_path, "ninada_v5/vocabs.json")

    # 2. Train & Export Native Female 24,000 Hz Studio Specialists (`{lang}_female.onnx`)
    skip_female = os.environ.get("SKIP_FEMALE", "0").strip() == "1"
    for lang in ([] if skip_female else target_langs):
        recipe = SPECIALIST_RECIPES[lang]
        tok = tokenizers[lang]
        out_fem_onnx = onnx_dir / f"{lang}_female.onnx"
        sample_fem_p = samples_dir / f"{lang}_female.wav"
        ckpt_fem_path = ckpt_dir / f"{lang}_female.safetensors"

        if (
            not args.smoke_steps
            and lang not in force_fem_langs
            and gcs_restore_v5_or_v4(gcs_bucket, f"onnx/{lang}_female.onnx", out_fem_onnx)
            and gcs_restore_v5_or_v4(gcs_bucket, f"samples_v5/{lang}_female.wav", sample_fem_p)
        ):
            print(f"[RESUME] Restored trained {lang} Female specialist from GCS!", flush=True)
            training_summary[f"{lang}_female"] = {"restored_from_gcs": True}
            continue

        f_cfg, f_sd = load_donor_checkpoint(recipe["female_donor"])
        net_f = StudioVitsModel(num_chars=tok.num_chars, use_discriminator=True)
        patched_f_sd = dict(f_sd)
        patched_f_sd["text_encoder.emb.weight"] = remap_embeddings_to_existing_tokenizer(
            tok, f_cfg, f_sd["text_encoder.emb.weight"], lang
        )
        net_f.load_state_dict(patched_f_sd, strict=True)
        f_donor_sr = int(f_cfg.get("audio", {}).get("sample_rate", 22050))
        if f_donor_sr != SAMPLE_RATE_24K and not recipe.get("exact_donor_24k_vocoder_adapt"):
            with torch.no_grad():
                net_f.duration_predictor.flows[0].translation.data[0, 0] += math.log(
                    float(SAMPLE_RATE_24K) / float(f_donor_sr)
                )

        f_steps = args.smoke_steps if args.smoke_steps > 0 else int(recipe["female_steps"])
        if recipe.get("exact_donor_24k_vocoder_adapt"):
            f_stats = adapt_exact_donor_24k_vocoder(
                net_f,
                f_sd,
                patched_f_sd["text_encoder.emb.weight"],
                tok,
                spec_ext,
                device=device,
                total_steps=f_steps,
                batch_size=args.batch_size,
                lr_g=float(recipe["lr_g"]),
                lr_d=float(recipe["lr_d"]),
                tag=f"{lang}:Female",
            )
        else:
            f_stats = train_single_specialist(
                net_f,
                tok,
                train_data[lang]["Female"],
                spec_ext,
                device=device,
                total_steps=f_steps,
                batch_size=args.batch_size,
                lr_g=float(recipe["lr_g"]),
                lr_text=float(recipe["lr_text"]),
                lr_d=float(recipe["lr_d"]),
                tag=f"{lang}:Female",
                freeze_text_backbone=bool(recipe.get("freeze_text_backbone", False)),
            )
        gen_f_sd = {
            k: v.detach().cpu().contiguous()
            for k, v in net_f.state_dict().items()
            if not k.startswith("disc.")
        }
        save_file(gen_f_sd, str(ckpt_fem_path))
        ox_f_stats = export_specialist_to_onnx(
            net_f, tok, out_fem_onnx, CANONICAL_PROBES[lang], sample_fem_p
        )
        f_stats.update(ox_f_stats)
        training_summary[f"{lang}_female"] = f_stats
        gcs_upload_if_configured(gcs_bucket, ckpt_fem_path, f"ninada_v5/checkpoints/{lang}_female.safetensors")
        gcs_upload_if_configured(gcs_bucket, out_fem_onnx, f"ninada_v5/onnx/{lang}_female.onnx")
        gcs_upload_if_configured(gcs_bucket, sample_fem_p, f"ninada_v5/samples_v5/{lang}_female.wav")
        del net_f
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.skip_bench:
        print(f"[WORKER DONE] Finished training & uploading specialists for {target_langs}: {training_summary}", flush=True)
        return

    # 3. Save dataset split manifests & run full 480-utterance scientific benchmark
    num_train_total = sum(
        max(len(train_data[lc]["Male"]), 0 if args.smoke_steps else int(SPECIALIST_RECIPES[lc]["target_male_utts"]))
        + max(len(train_data[lc]["Female"]), 0 if args.smoke_steps else int(SPECIALIST_RECIPES[lc]["target_female_utts"]))
        for lc in SUPPORTED_LANGS
    )
    (args.out_dir / "training_summary_v5.json").write_text(
        json.dumps(training_summary, indent=2) + "\n", encoding="utf-8"
    )
    with open(args.out_dir / "test_v5.jsonl", "w", encoding="utf-8") as f_te:
        for r in test_rows:
            f_te.write(
                json.dumps(
                    {
                        "utt_id": r["utt_id"],
                        "lang": r["lang"],
                        "text": r["text"],
                        "wav_path": r.get("wav_path", ""),
                        "duration_sec": r.get("duration_sec", 0.0),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    bench_results = evaluate_all_systems_on_heldout(
        test_rows=test_rows,
        num_train_total=num_train_total,
        num_val_total=len(val_rows),
        tokenizers=tokenizers,
        warmstart_wavs=warmstart_wavs,
        pt_models=pt_models,
        onnx_dir=onnx_dir,
        device=device,
    )
    results_path = args.out_dir / "results.json"
    results_path.write_text(json.dumps(bench_results, indent=2) + "\n", encoding="utf-8")
    gcs_upload_if_configured(gcs_bucket, results_path, "ninada_v5/results.json")
    gcs_upload_if_configured(gcs_bucket, args.out_dir / "training_summary_v5.json", "ninada_v5/training_summary_v5.json")
    gcs_upload_if_configured(gcs_bucket, args.out_dir / "test_v5.jsonl", "ninada_v5/test_v5.jsonl")

    bundle_tar = args.out_dir / "ninada_v5_artifacts.tar.gz"
    with tarfile.open(bundle_tar, "w:gz") as tf:
        for rel in ("onnx", "samples_v5", "vocabs.json", "results.json", "training_summary_v5.json", "test_v5.jsonl", "test_gt_wavs"):
            target = args.out_dir / rel
            if target.exists():
                tf.add(target, arcname=rel)
    gcs_upload_if_configured(gcs_bucket, bundle_tar, "ninada_v5/ninada_v5_artifacts.tar.gz")

    print("\n=== NINADA v5.0 SCIENTIFIC SUMMARY ===", flush=True)
    for sk, sv in bench_results["systems"].items():
        print(
            f"  {sk:22s}: MCD={sv['mcd_db']['mean']:.2f} dB | HF={sv['hf_8_12k_err_db']['mean']:.2f} dB | "
            f"F0={sv['f0_rmse_cents']['mean']:.1f}c | Dur={sv['dur_mae_sec']['mean']:.2f}s | "
            f"Size={sv['size_mb']:.1f} MB | RTF={sv['rtf_mean']:.4f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
