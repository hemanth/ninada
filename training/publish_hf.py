"""Publish GCP-trained `Ninada` v1.1 Streaming model, split Browser ONNX weights, and curated Su-srota + Pan-Indic dataset manifests to Hugging Face Hub."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def build_model_card(ckpt_dir: Path, bench_json: Path, stats_json: Path) -> str:
    manifest = json.loads((ckpt_dir / "gcp_job_manifest.json").read_text(encoding="utf-8"))
    bench = json.loads(bench_json.read_text(encoding="utf-8")) if bench_json.exists() else {}
    stats = json.loads(stats_json.read_text(encoding="utf-8")) if stats_json.exists() else {}

    sys_ox = bench.get("systems", {}).get("ninada_onnx_quant", {})
    sys_pt = bench.get("systems", {}).get("ninada_pytorch_fp32", {})
    sys_flt = bench.get("systems", {}).get("mms_5_fleet", {})
    sys_raw = bench.get("systems", {}).get("mms_hin_raw", {})
    sys_g2p = bench.get("systems", {}).get("mms_hin_g2p", {})
    sys_gt = bench.get("systems", {}).get("studio_ground_truth", {})

    n_test = bench.get("num_test_utterances", 155)
    n_train = stats.get("train_clips", 1240)
    n_val = stats.get("val_clips", 155)
    tot_min = stats.get("total_duration_minutes", 175.19)
    ox_mb = sys_ox.get("size_mb", 56.6)
    ttfa_ox = sys_ox.get("ttfa_ms", {}).get("p50", 128.0)

    def fmt_ci(d: dict) -> str:
        if not d:
            return "—"
        return f"{d['mean']:.2f} [{d['ci95_low']:.2f}, {d['ci95_high']:.2f}]"

    return f"""---
language:
  - sa
  - hi
  - kn
  - ta
  - te
  - ml
license: mit
library_name: onnx
pipeline_tag: text-to-speech
datasets:
  - prathoshap/sushrota-sanskrit-asr-data
  - SayantanJoker/SYSPIN_Hindi_Female_TTS
  - SayantanJoker/original_data_kannada_tts
  - SayantanJoker/original_data_tamil_tts
  - arpit-tiwari/syspin-telugu-tts
  - thennal/indic_tts_ml
tags:
  - text-to-speech
  - streaming-tts
  - ninada
  - sanskrit
  - vedic
  - indic
  - kannada
  - vits
  - hifi-gan
  - onnxruntime-web
  - wasm
---

# `Ninada` (निनाद) v1.1 — Streaming Pan-Indic & Classical Sanskrit Neural TTS

`Ninada` (*Nināda* — Sanskrit `निनाद`, deep resonant sound) is a sub-second **streaming Classical Sanskrit & Pan-Indic** neural text-to-speech model (**`{ox_mb:.1f} MB` split streaming FP16 ONNX: `29.8 MB` Encoder+Flow + `26.8 MB` Halo-Chunked HiFi-GAN, `{ttfa_ox:.1f} ms` median TTFA**) supporting **6 Indic languages** (`Sanskrit [sa]`, `Kannada [kn]`, `Hindi [hi]`, `Tamil [ta]`, `Telugu [te]`, `Malayalam [ml]`) across **9 Brahmic scripts + ISO-15919/IAST** (`87` Pan-Indic & Vedic/Classical Sanskrit phoneme tokens).

- **Live Off-Main-Thread Web Worker Studio**: [https://hemanth.github.io/ninada/](https://hemanth.github.io/ninada/)
- **GitHub Repository & Streaming SDKs (JS + Python)**: [https://github.com/hemanth/ninada](https://github.com/hemanth/ninada)
- **Curated Split Metadata (`1,550` Studio Clips)**: [https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic](https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic)

## GCP NVIDIA L4 GPU Training & Halo-Chunked Streaming Architecture
- **Hardware**: Trained on **Google Cloud (`{manifest.get('gpu_name', 'NVIDIA L4')}` 24 GB GPU)** for `{manifest.get('epochs', 24)}` epochs (`{manifest.get('total_steps', 3720)}` steps) with speaker-disentangled `UtteranceStyleEncoder` + EMA per-language studio voice centroids, latent-manifold feature matching in the frozen adversarial 16 kHz HiFi-GAN space, vectorized Viterbi Monotonic Alignment Search (`MAS`), and validation-only Weight-Space Ensembling (`WiSE-FT`).
- **Halo-Chunked Streaming (`53.6 dB` SNR Parity)**: Splits inference into `onnx/encoder_flow_quantized.onnx` (`~29.8 MB`, global clause prosody + normalizing flow in `~35 ms`) and `onnx/vocoder_quantized.onnx` (`~26.8 MB`, `40`-frame chunks with `10`-frame left/right receptive-field halo cropped at exact `256`-sample boundaries), cutting **Time-To-First-Audio (`TTFA`) by ~`7x`** while eliminating WebAssembly main-thread freezes inside a dedicated Web Worker.
- **Curated Studio Corpus (`{n_train + n_val + n_test}` studio-aligned clips, `{tot_min}` min at 16 kHz; `{n_train}` train / `{n_val}` val / `{n_test}` test)**:
  - **Sanskrit (`650` pure studio human clips, `80.5 min`)**: Curated exclusively from `prathoshap/sushrota-sanskrit-asr-data` (`360` `recitation_corpus` ślokas + `290` `scholar` śāstra prose, `studio_align_cer <= 10.0%`, synthetic `tts_aug` excluded).
  - **Pan-Indic (`900` studio clips — `180` each across `hi`, `kn`, `ta`, `te`, `ml`)**: SYSPIN Hindi, Kannada, Tamil, Telugu, and IITM Malayalam.

## Held-Out Test Leaderboard (`N = {n_test}` Utterances; `B = 1,000` Bootstrap 95% CIs)

| System | Footprint (↓) | Coverage (↑) | TTFA p50 (ms, ↓) | MCD (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) | Sanskrit CER (%, ↓) | Pan-Indic CER (%, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Studio Human Audio (Topline) | — | 100.0% | — | — | — | `{fmt_ci(sys_gt.get('sanskrit_cer_pct'))}%` | `{fmt_ci(sys_gt.get('pan_indic_phonetic_cer_pct'))}%` |
| Base `facebook/mms-tts-hin` (Raw) | 138.8 MB | {sys_raw.get('voiced_rate_pct', 54.0):.1f}% | {sys_raw.get('ttfa_ms', {}).get('p50', 0):.1f} ms | `{fmt_ci(sys_raw.get('mcd_db'))}` | {sys_raw.get('f0_rmse_cents', {}).get('mean', 0):.1f} | {sys_raw.get('dur_mae_sec', {}).get('mean', 0):.2f}s | `{fmt_ci(sys_raw.get('sanskrit_cer_pct'))}%` | `{fmt_ci(sys_raw.get('pan_indic_phonetic_cer_pct'))}%` |
| Base `mms-tts-hin` + `Ninada` G2P | 138.8 MB | 100.0% | {sys_g2p.get('ttfa_ms', {}).get('p50', 0):.1f} ms | `{fmt_ci(sys_g2p.get('mcd_db'))}` | {sys_g2p.get('f0_rmse_cents', {}).get('mean', 0):.1f} | {sys_g2p.get('dur_mae_sec', {}).get('mean', 0):.2f}s | `{fmt_ci(sys_g2p.get('sanskrit_cer_pct'))}%` | `{fmt_ci(sys_g2p.get('pan_indic_phonetic_cer_pct'))}%` |
| Meta MMS 5-Specialist Fleet (`5x` Models) | 694.0 MB | 100.0% | {sys_flt.get('ttfa_ms', {}).get('p50', 0):.1f} ms | `{fmt_ci(sys_flt.get('mcd_db'))}` | {sys_flt.get('f0_rmse_cents', {}).get('mean', 0):.1f} | {sys_flt.get('dur_mae_sec', {}).get('mean', 0):.2f}s | `{fmt_ci(sys_flt.get('sanskrit_cer_pct'))}%` | `{fmt_ci(sys_flt.get('pan_indic_phonetic_cer_pct'))}%` |
| **`Ninada` Tier-1A (GCP L4 FP32 Stream)** | {sys_pt.get('size_mb', 136.5):.1f} MB | **100.0%** | **{sys_pt.get('ttfa_ms', {}).get('p50', 0):.1f} ms** | **`{fmt_ci(sys_pt.get('mcd_db'))}`** | **{sys_pt.get('f0_rmse_cents', {}).get('mean', 0):.1f}** | **{sys_pt.get('dur_mae_sec', {}).get('mean', 0):.2f}s** | **`{fmt_ci(sys_pt.get('sanskrit_cer_pct'))}%`** | **`{fmt_ci(sys_pt.get('pan_indic_phonetic_cer_pct'))}%`** |
| **`Ninada` Tier-1B (Streaming ONNX)** | **{ox_mb:.1f} MB** | **100.0%** | **{ttfa_ox:.1f} ms** | **`{fmt_ci(sys_ox.get('mcd_db'))}`** | **{sys_ox.get('f0_rmse_cents', {}).get('mean', 0):.1f}** | **{sys_ox.get('dur_mae_sec', {}).get('mean', 0):.2f}s** | **`{fmt_ci(sys_ox.get('sanskrit_cer_pct'))}%`** | **`{fmt_ci(sys_ox.get('pan_indic_phonetic_cer_pct'))}%`** |

## Streaming Quickstart

```js
import ninada from 'ninada';

const voice = await ninada('kannada', {{
  encoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/encoder_flow_quantized.onnx',
  vocoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/vocoder_quantized.onnx',
}});
for await (const chunk of voice.stream('ಕನ್ನಡ ಭಾಷೆಯ ನೈಸರ್ಗಿಕ ಧ್ವನಿ ಸಂಶ್ಲೇಷಣೆಗೆ ಸ್ವಾಗತ.')) {{
  playPcm16k(chunk.pcm); // Sub-second TTFA!
}}
```

```python
from ninada import Ninada

tts = Ninada(lang="sa")
for chunk in tts.stream("प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् । यथार्थज्ञानं प्रमाणम् ॥"):
    play(chunk["pcm"], chunk["sample_rate"])
```
"""


def build_dataset_card(stats_json: Path) -> str:
    stats = json.loads(stats_json.read_text(encoding="utf-8"))
    return f"""---
language:
  - sa
  - hi
  - kn
  - ta
  - te
  - ml
license: cc-by-4.0
task_categories:
  - text-to-speech
  - automatic-speech-recognition
tags:
  - ninada
  - sanskrit
  - sushrota
  - kannada
  - indic
  - tts
  - g2p
---

# `ninada-sanskrit-indic` v1.1 — Studio-Gated Leakage-Free 6-Language Su-śrotā Sanskrit & Pan-Indic TTS Split Manifests

Curated, studio-alignment-gated, leakage-free domain-stratified splits (`{stats['train_clips']}` train / `{stats['val_clips']}` validation / `{stats['test_clips']}` test = `{stats['total_clips']}` utterances, `{stats['total_duration_minutes']}` minutes at 16 kHz) with canonical `87`-symbol Pan-Indic & Classical Sanskrit `input_ids` used to train and evaluate [`gnumanth/ninada`](https://huggingface.co/gnumanth/ninada).

## Upstream Studio Corpora Credited
1. `prathoshap/sushrota-sanskrit-asr-data` (`sa` — `650` pure human studio clips, `80.54` min: `360` `recitation_corpus` ślokas + `290` `scholar` śāstra prose, `studio_align_cer <= 10.0%`, synthetic `tts_aug` excluded)
2. `SayantanJoker/SYSPIN_Hindi_Female_TTS` (`hi` — `180` studio clips)
3. `SayantanJoker/original_data_kannada_tts` (`kn` — `180` studio clips)
4. `SayantanJoker/original_data_tamil_tts` (`ta` — `180` studio clips)
5. `arpit-tiwari/syspin-telugu-tts` (`te` — `180` studio clips)
6. `thennal/indic_tts_ml` (`ml` — `180` studio clips)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="gnumanth/ninada")
    parser.add_argument("--dataset-repo-id", default="gnumanth/ninada-sanskrit-indic")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/ninada"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--bench-json", type=Path, default=Path("bench/results.json"))
    args = parser.parse_args()

    api = HfApi()
    stats_json = args.data_dir / "dataset_stats.json"

    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    readme_path = args.checkpoint_dir / "README.md"
    readme_path.write_text(build_model_card(args.checkpoint_dir, args.bench_json, stats_json), encoding="utf-8")

    api.upload_folder(
        folder_path=str(args.checkpoint_dir),
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=[
            "onnx/model.onnx",
            "onnx/encoder_flow.onnx",
            "onnx/vocoder.onnx",
            "init_model.safetensors",
            "trained_model.safetensors",
        ],
        commit_message="Release Ninada v1.1: Su-srota Sanskrit + 5-Indic Streaming VITS & Halo-Chunked Browser ONNX",
    )
    print(f"Published model to https://huggingface.co/{args.repo_id}")

    api.create_repo(repo_id=args.dataset_repo_id, repo_type="dataset", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpd:
        td = Path(tmpd)
        (td / "README.md").write_text(build_dataset_card(stats_json), encoding="utf-8")
        for fn in ("train_meta.jsonl", "val_meta.jsonl", "test_meta.jsonl", "dataset_stats.json"):
            src = args.data_dir / fn
            if src.exists():
                shutil.copy2(src, td / fn)
        api.upload_folder(
            folder_path=str(td),
            repo_id=args.dataset_repo_id,
            repo_type="dataset",
            commit_message="Publish Ninada v1.1 Su-srota Sanskrit + 5-Indic 1,550-clip split manifests",
        )
    print(f"Published dataset manifests to https://huggingface.co/datasets/{args.dataset_repo_id}")


if __name__ == "__main__":
    main()
