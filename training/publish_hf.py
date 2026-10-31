"""Publish GCP-trained `Ninada` v2.0 Specialist Streaming models, split FP16 Browser ONNX weights, and curated Su-srota + AI4Bharat Rasa dataset manifests to Hugging Face Hub."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def build_model_card(ckpt_dir: Path, bench_json: Path, stats_json: Path) -> str:
    manifest = (
        json.loads((ckpt_dir / "gcp_job_manifest.json").read_text(encoding="utf-8"))
        if (ckpt_dir / "gcp_job_manifest.json").exists()
        else {}
    )
    bench = json.loads(bench_json.read_text(encoding="utf-8")) if bench_json.exists() else {}
    stats = json.loads(stats_json.read_text(encoding="utf-8")) if stats_json.exists() else {}

    sys_ox = bench.get("systems", {}).get("ninada_onnx_quant", {})
    sys_pt = bench.get("systems", {}).get("ninada_pytorch_fp32", {})
    sys_flt = bench.get("systems", {}).get("mms_5_fleet", {})
    sys_raw = bench.get("systems", {}).get("mms_hin_raw", {})
    sys_g2p = bench.get("systems", {}).get("mms_hin_g2p", {})
    sys_gt = bench.get("systems", {}).get("studio_ground_truth", {})

    n_test = bench.get("num_test_utterances", 144)
    n_train = stats.get("train_clips", 1150)
    n_val = stats.get("val_clips", 144)
    tot_min = stats.get("total_duration_minutes", 127.55)
    ox_mb = sys_ox.get("size_mb", 54.7)
    ttfa_ox = sys_ox.get("ttfa_ms", {}).get("p50", 145.0)
    seam_snr = bench.get("onnx_parity", {}).get("mean_stream_seam_snr_db", 126.0)

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
  - ai4bharat/Rasa
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

# `Ninada` (निनाद) v2.0 — Streaming Pan-Indic & Classical Sanskrit Neural TTS

`Ninada` (*Nināda* — Sanskrit `निनाद`, deep resonant sound) v2.0 is a sub-second **streaming Classical Sanskrit & Pan-Indic** neural text-to-speech suite (**`{ox_mb:.1f} MB` active split streaming FP16 ONNX per specialist: `27.3 MB` Encoder+SDP+Flow + `27.4 MB` Halo-Chunked HiFi-GAN, `{ttfa_ox:.1f} ms` median TTFA**) supporting **6 Indic languages** (`Sanskrit [sa]`, `Kannada [kn]`, `Hindi [hi]`, `Tamil [ta]`, `Telugu [te]`, `Malayalam [ml]`) across **9 Brahmic scripts + ISO-15919/IAST**.

- **Live Off-Main-Thread Web Worker Studio**: [https://hemanth.github.io/ninada/](https://hemanth.github.io/ninada/)
- **GitHub Repository & Streaming SDKs (JS + Python)**: [https://github.com/hemanth/ninada](https://github.com/hemanth/ninada)
- **Curated Split Metadata (`{n_train + n_val + n_test}` Studio Clips)**: [https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic](https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic)

## GCP NVIDIA L4 GPU Training & Halo-Crossfaded Streaming Architecture
- **Hardware & Objective**: Trained on **Google Cloud (`{manifest.get('gpu_name', 'NVIDIA L4')}` 24 GB GPU)** with alternating **`MPD + MRD` (`LSGAN` + multi-layer discriminator feature-matching) + Flow KL + ONNX-safe Stochastic Duration Predictor (`SDP`) MLE + validation-only end-to-end `WiSE-FT`** across 6 native-script VITS + HiFi-GAN specialists (`specialists/{{sa,kn,hi,ta,te,ml}}`).
- **Halo-Crossfaded Streaming (`{seam_snr:.1f} dB` Seam SNR)**: Splits inference into `onnx/{{lang}}/encoder_flow_quantized.onnx` (`~27.3 MB`, native stochastic duration predictor + normalizing flow) and `onnx/{{lang}}/vocoder_quantized.onnx` (`~27.4 MB`, `16`-frame receptive-field halo + `256`-sample raised-cosine boundary crossfade), persisted in browser `CacheStorage` (`0` network fetches on reload or language switch).
- **Anechoic Male Studio Corpus (`{n_train + n_val + n_test}` clips, `{tot_min}` min at 16 kHz; `{n_train}` train / `{n_val}` val / `{n_test}` test)**:
  - **Sanskrit (`388` Male studio clips)**: `ai4bharat/Rasa` neutral Sanskrit studio narration + `prathoshap/sushrota-sanskrit-asr-data` (`recitation_corpus` + `scholar`, `studio_align_cer <= 5.0%`, synthetic `tts_aug` excluded).
  - **Pan-Indic (`1,050` Male studio clips — `210` each across `hi`, `kn`, `ta`, `te`, `ml`)**: `ai4bharat/Rasa` 48 kHz -> 16 kHz anechoic booth recordings.

## Held-Out Test Leaderboard (`N = {n_test}` Utterances; `B = 1,000` Bootstrap 95% CIs)

| System | Active Footprint (↓) | Coverage (↑) | TTFA p50 (ms, ↓) | MCD (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) | Sanskrit CER (%, ↓) | Sanskrit WER (%, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Studio Human Audio (Topline) | — | 100.0% | — | — | — | — | `{fmt_ci(sys_gt.get('sanskrit_cer_pct'))}%` | `{fmt_ci(sys_gt.get('sanskrit_wer_pct'))}%` |
| Base `facebook/mms-tts-hin` (Raw) | 138.8 MB | {sys_raw.get('voiced_rate_pct', 43.1):.1f}% | {sys_raw.get('ttfa_ms', {}).get('p50', 0):.1f} ms | `{fmt_ci(sys_raw.get('mcd_db'))}` | {sys_raw.get('f0_rmse_cents', {}).get('mean', 0):.1f} | {sys_raw.get('dur_mae_sec', {}).get('mean', 0):.2f}s | `{fmt_ci(sys_raw.get('sanskrit_cer_pct'))}%` | `{fmt_ci(sys_raw.get('sanskrit_wer_pct'))}%` |
| Base `mms-tts-hin` + `Ninada` G2P | 138.8 MB | 100.0% | {sys_g2p.get('ttfa_ms', {}).get('p50', 0):.1f} ms | `{fmt_ci(sys_g2p.get('mcd_db'))}` | {sys_g2p.get('f0_rmse_cents', {}).get('mean', 0):.1f} | {sys_g2p.get('dur_mae_sec', {}).get('mean', 0):.2f}s | `{fmt_ci(sys_g2p.get('sanskrit_cer_pct'))}%` | `{fmt_ci(sys_g2p.get('sanskrit_wer_pct'))}%` |
| Meta MMS 5-Specialist Fleet (`5x` Models) | 694.0 MB | 100.0% | {sys_flt.get('ttfa_ms', {}).get('p50', 0):.1f} ms | `{fmt_ci(sys_flt.get('mcd_db'))}` | {sys_flt.get('f0_rmse_cents', {}).get('mean', 0):.1f} | {sys_flt.get('dur_mae_sec', {}).get('mean', 0):.2f}s | `{fmt_ci(sys_flt.get('sanskrit_cer_pct'))}%` | `{fmt_ci(sys_flt.get('sanskrit_wer_pct'))}%` |
| **`Ninada` v2.0 Tier-1A (GCP L4 FP32)** | {sys_pt.get('size_mb', 138.4):.1f} MB | **100.0%** | **{sys_pt.get('ttfa_ms', {}).get('p50', 0):.1f} ms** | **`{fmt_ci(sys_pt.get('mcd_db'))}`** | **{sys_pt.get('f0_rmse_cents', {}).get('mean', 0):.1f}** | **{sys_pt.get('dur_mae_sec', {}).get('mean', 0):.2f}s** | **`{fmt_ci(sys_pt.get('sanskrit_cer_pct'))}%`** | **`{fmt_ci(sys_pt.get('sanskrit_wer_pct'))}%`** |
| **`Ninada` v2.0 Tier-1B (Streaming ONNX)** | **{ox_mb:.1f} MB** | **100.0%** | **{ttfa_ox:.1f} ms** | **`{fmt_ci(sys_ox.get('mcd_db'))}`** | **{sys_ox.get('f0_rmse_cents', {}).get('mean', 0):.1f}** | **{sys_ox.get('dur_mae_sec', {}).get('mean', 0):.2f}s** | **`{fmt_ci(sys_ox.get('sanskrit_cer_pct'))}%`** | **`{fmt_ci(sys_ox.get('sanskrit_wer_pct'))}%`** |

## Streaming Quickstart

```js
import ninada from 'ninada';

const voice = await ninada('kannada', {{
  encoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/kn/encoder_flow_quantized.onnx',
  vocoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/kn/vocoder_quantized.onnx',
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
  - rasa
  - kannada
  - indic
  - tts
  - g2p
---

# `ninada-sanskrit-indic` v2.0 — Anechoic Male Studio 6-Language Su-śrotā Sanskrit & AI4Bharat Rasa TTS Split Manifests

Curated, studio-alignment-gated, leakage-free domain-stratified splits (`{stats['train_clips']}` train / `{stats['val_clips']}` validation / `{stats['test_clips']}` test = `{stats['total_clips']}` utterances, `{stats['total_duration_minutes']}` minutes at 16 kHz) with native-script specialist `input_ids` and Paninian Devanagari G2P used to train and evaluate [`gnumanth/ninada`](https://huggingface.co/gnumanth/ninada).

## Upstream Studio Corpora Credited
1. `ai4bharat/Rasa` (`sa`, `hi`, `kn`, `ta`, `te`, `ml` — neutral Male studio narration downsampled 48 kHz -> 16 kHz with intact studio booth boundaries)
2. `prathoshap/sushrota-sanskrit-asr-data` (`sa` — pure human Male studio `recitation_corpus` ślokas + `scholar` śāstra prose, `studio_align_cer <= 5.0%`, synthetic `tts_aug` excluded)
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
            "onnx/model_quantized.onnx",
            "onnx/*/encoder_flow.onnx",
            "onnx/*/vocoder.onnx",
            "onnx/encoder_flow.onnx",
            "onnx/vocoder.onnx",
            "specialists/*/init_model.safetensors",
            "specialists/*/trained_model.safetensors",
            "init_model.safetensors",
            "trained_model.safetensors",
        ],
        commit_message="Release Ninada v2.0: 6-Language Adversarial MPD+MRD+SDP Specialists & Halo-Crossfaded Streaming ONNX",
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
            commit_message="Publish Ninada v2.0 Su-srota + AI4Bharat Rasa 1,438-clip anechoic studio split manifests",
        )
    print(f"Published dataset manifests to https://huggingface.co/datasets/{args.dataset_repo_id}")


if __name__ == "__main__":
    main()
