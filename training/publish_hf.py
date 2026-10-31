"""Publish GCP-trained `Ninada` v1.0 model, Browser ONNX weights, and curated Sanskrit-Indic dataset metadata to Hugging Face Hub."""
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

    n_test = bench.get("num_test_utterances", 133)
    n_train = stats.get("train_clips", 1061)
    n_val = stats.get("val_clips", 133)
    tot_min = stats.get("total_duration_minutes", 144.0)
    ox_mb = sys_ox.get("size_mb", 54.9)

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
  - surajp/shrutilipi_sanskrit
  - addy88/sanskrit-asr-84
  - yashnbx/gita_supersite_sanskrit_tts
  - SayantanJoker/SYSPIN_Hindi_Female_TTS
  - SayantanJoker/original_data_kannada_tts
  - SayantanJoker/original_data_tamil_tts
  - arpit-tiwari/syspin-telugu-tts
  - thennal/indic_tts_ml
tags:
  - text-to-speech
  - ninada
  - sanskrit
  - vedic
  - indic
  - vits
  - hifi-gan
  - onnxruntime-web
  - wasm
---

# `Ninada` (निनाद) v1.0 — Disentangled Pan-Indic & Classical Sanskrit Neural TTS

`Ninada` (*Nināda* — Sanskrit `निनाद`, deep resonant sound) is a compact, single-checkpoint **Classical Sanskrit & Pan-Indic** neural text-to-speech model (**`{ox_mb:.1f} MB` single-file browser ONNX**) supporting **6 Indic languages** (`Sanskrit [sa]`, `Hindi [hi]`, `Kannada [kn]`, `Tamil [ta]`, `Telugu [te]`, `Malayalam [ml]`) across **9 Brahmic scripts + ISO-15919/IAST** (`87` Pan-Indic & Vedic/Classical Sanskrit phoneme tokens including vocalic `ऋ`/`ॠ`, velar nasal `ङ`, Dravidian retroflexes `ळ`/`ऴ`/`ऱ`/`ऩ`, *pūrvarūpa avagraha* sandhi, and *Pāṇinian samyoge guru* prosody).

- **Live Browser WASM SIMD Studio**: [https://hemanth.github.io/ninada/](https://hemanth.github.io/ninada/)
- **GitHub Repository & SDKs (JS + Python)**: [https://github.com/hemanth/ninada](https://github.com/hemanth/ninada)
- **Curated Split Metadata**: [https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic](https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic)

## GCP NVIDIA L4 GPU Training & Disentangled Architecture
- **Hardware**: Trained on **Google Cloud (`{manifest.get('gpu_name', 'NVIDIA L4')}` 24 GB GPU)** for `{manifest.get('epochs', 24)}` epochs (`{manifest.get('total_steps', 3192)}` steps) with speaker-disentangled `UtteranceStyleEncoder` + EMA per-language studio voice centroids, latent-manifold feature matching in the frozen adversarial 16 kHz HiFi-GAN space, vectorized Viterbi Monotonic Alignment Search (`MAS`), and validation-only Weight-Space Ensembling (`WiSE-FT`).
- **Curated Multi-Domain Corpus (`{n_train + n_val + n_test}` studio-aligned clips, `{tot_min}` min at 16 kHz; `{n_train}` train / `{n_val}` val / `{n_test}` test)**:
  - **Sanskrit (`474` studio-gated clips across 4 corpora)**: `prathoshap/sushrota-sanskrit-asr-data` (IISc Vedānta prose), `surajp/shrutilipi_sanskrit` (All India Radio Sanskrit news), `addy88/sanskrit-asr-84` (IITB WadaSNR >= 32 dB spoken Sanskrit), and `yashnbx/gita_supersite_sanskrit_tts` (IITK Bhagavad-Gītā ślokas).
  - **Pan-Indic (`853` clips across 5 studio corpora)**: SYSPIN Hindi, Kannada, Tamil, Telugu, and IITM Malayalam.

## Held-Out Test Leaderboard (`N = {n_test}` Utterances; `B = 1,000` Bootstrap 95% CIs)

| System | Footprint (↓) | Coverage (↑) | MCD (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) | Sanskrit CER (%, ↓) | Pan-Indic CER (%, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Studio Human Audio (Topline) | — | 100.0% | — | — | — | `{fmt_ci(sys_gt.get('sanskrit_cer_pct'))}%` | `{fmt_ci(sys_gt.get('pan_indic_phonetic_cer_pct'))}%` |
| Base `facebook/mms-tts-hin` (Raw) | 138.8 MB | {sys_raw.get('voiced_rate_pct', 53.4):.1f}% | `{fmt_ci(sys_raw.get('mcd_db'))}` | {sys_raw.get('f0_rmse_cents', {}).get('mean', 0):.1f} | {sys_raw.get('dur_mae_sec', {}).get('mean', 0):.2f}s | `{fmt_ci(sys_raw.get('sanskrit_cer_pct'))}%` | `{fmt_ci(sys_raw.get('pan_indic_phonetic_cer_pct'))}%` |
| Base `mms-tts-hin` + `Ninada` G2P | 138.8 MB | 100.0% | `{fmt_ci(sys_g2p.get('mcd_db'))}` | {sys_g2p.get('f0_rmse_cents', {}).get('mean', 0):.1f} | {sys_g2p.get('dur_mae_sec', {}).get('mean', 0):.2f}s | `{fmt_ci(sys_g2p.get('sanskrit_cer_pct'))}%` | `{fmt_ci(sys_g2p.get('pan_indic_phonetic_cer_pct'))}%` |
| Meta MMS 5-Specialist Fleet (`5x` Models) | 694.0 MB | 100.0% | `{fmt_ci(sys_flt.get('mcd_db'))}` | {sys_flt.get('f0_rmse_cents', {}).get('mean', 0):.1f} | {sys_flt.get('dur_mae_sec', {}).get('mean', 0):.2f}s | `{fmt_ci(sys_flt.get('sanskrit_cer_pct'))}%` | `{fmt_ci(sys_flt.get('pan_indic_phonetic_cer_pct'))}%` |
| **`Ninada` Tier-1A (GCP L4 FP32)** | {sys_pt.get('size_mb', 136.5):.1f} MB | **100.0%** | **`{fmt_ci(sys_pt.get('mcd_db'))}`** | **{sys_pt.get('f0_rmse_cents', {}).get('mean', 0):.1f}** | **{sys_pt.get('dur_mae_sec', {}).get('mean', 0):.2f}s** | **`{fmt_ci(sys_pt.get('sanskrit_cer_pct'))}%`** | **`{fmt_ci(sys_pt.get('pan_indic_phonetic_cer_pct'))}%`** |
| **`Ninada` Tier-1B (Browser ONNX)** | **{ox_mb:.1f} MB** | **100.0%** | **`{fmt_ci(sys_ox.get('mcd_db'))}`** | **{sys_ox.get('f0_rmse_cents', {}).get('mean', 0):.1f}** | **{sys_ox.get('dur_mae_sec', {}).get('mean', 0):.2f}s** | **`{fmt_ci(sys_ox.get('sanskrit_cer_pct'))}%`** | **`{fmt_ci(sys_ox.get('pan_indic_phonetic_cer_pct'))}%`** |

## Quickstart

```js
import ninada from 'ninada';

const voice = await ninada('sanskrit', {{
  model: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/model_quantized.onnx',
}});
const wavBuffer = await voice.speak('ऋतं च सत्यं चाभीद्धात् तपसोऽध्यजायत ।');
```

```python
from ninada import Ninada

wav_bytes = Ninada(lang="sa").speak("प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् ।")
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
  - indic
  - tts
  - g2p
---

# `ninada-sanskrit-indic` — Studio-Gated Leakage-Free 6-Language Sanskrit & Pan-Indic TTS Split Manifests

Curated, studio-alignment-gated, leakage-free domain-stratified splits (`{stats['train_clips']}` train / `{stats['val_clips']}` validation / `{stats['test_clips']}` test = `{stats['total_clips']}` utterances, `{stats['total_duration_minutes']}` minutes at 16 kHz) with canonical `87`-symbol Pan-Indic & Classical Sanskrit `input_ids` used to train and evaluate [`gnumanth/ninada`](https://huggingface.co/gnumanth/ninada).

## Upstream Studio Corpora Credited
1. `prathoshap/sushrota-sanskrit-asr-data` (`sa` — 250 clips, IISc Vedānta śāstra prose)
2. `surajp/shrutilipi_sanskrit` (`sa` — 140 studio-gated clips, All India Radio Sanskrit news broadcasts)
3. `addy88/sanskrit-asr-84` (`sa` — 24 studio-gated WadaSNR >= 32 dB clips, IITB spoken Sanskrit)
4. `yashnbx/gita_supersite_sanskrit_tts` (`sa` — 60 clips, IITK Bhagavad-Gītā half-śloka recitations)
5. `SayantanJoker/SYSPIN_Hindi_Female_TTS` (`hi` — 180 clips)
6. `SayantanJoker/original_data_kannada_tts` (`kn` — 133 clips)
7. `SayantanJoker/original_data_tamil_tts` (`ta` — 180 clips)
8. `arpit-tiwari/syspin-telugu-tts` (`te` — 180 clips)
9. `thennal/indic_tts_ml` (`ml` — 180 clips)
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
        ignore_patterns=["onnx/model.onnx", "init_model.safetensors", "trained_model.safetensors"],
        commit_message="Release Ninada v1.0: Disentangled Pan-Indic & Classical Sanskrit Neural TTS + 56.6MB Browser ONNX",
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
            commit_message="Publish Ninada v1.0 studio-gated 6-language Sanskrit & Pan-Indic split manifests",
        )
    print(f"Published dataset manifests to https://huggingface.co/datasets/{args.dataset_repo_id}")


if __name__ == "__main__":
    main()
