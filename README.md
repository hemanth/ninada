# ninada (निनाद)

> Speak Classical & Vedic Sanskrit plus 6 Indic languages (`sa`, `hi`, `kn`, `ta`, `te`, `ml`) from a single **56.6 MB** Disentangled VITS + HiFi-GAN ONNX checkpoint in the browser (WASM SIMD), Node.js, and Python.

- **Live Browser WASM SIMD Studio:** <https://hemanth.github.io/ninada/>
- **Hugging Face Model (`gnumanth/ninada`):** <https://huggingface.co/gnumanth/ninada>
- **Hugging Face Dataset (`gnumanth/ninada-sanskrit-indic`):** <https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic>
- **Scientific Report:** [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md) · [`bench/results.json`](./bench/results.json)

![Ninada Architecture Blueprint](./docs/flow-diagram.svg)

## Why `ninada`?

Open-weight multilingual TTS ecosystems lack a native Classical/Vedic Sanskrit checkpoint (`facebook/mms-tts-san` does not exist on Hugging Face) and require **5 separate specialist models (`694.0 MB` combined)** to cover Hindi, Kannada, Tamil, Telugu, and Malayalam.

`ninada` (*Nināda* — Sanskrit `निनाद`, deep resonant sound) unifies **Sanskrit (`sa`, across 4 distinct speech domains)**, **Hindi (`hi`)**, **Kannada (`kn`)**, **Tamil (`ta`)**, **Telugu (`te`)**, and **Malayalam (`ml`)** across **9 Brahmic scripts + ISO-15919/IAST** into a single **`56.6 MB` browser-ready ONNX** model trained on **Google Cloud `NVIDIA L4` (`24 GB` VRAM) GPUs**:

1. **87-Symbol Pan-Indic & Pāṇinian Akṣara G2P (`0x80` ISCII Alignment):** Preserves Classical/Vedic Sanskrit vocalic `ऋ`/`ॠ` (`[85]`), homorganic velar nasal `ङ` (`[86]`), Dravidian retroflexes `ळ`/`ऴ`/`ऱ`/`ऩ`, *pūrvarūpa avagraha* sandhi (`ऽ`), Vedic *udātta/svarita* cantillation stripping, Tamil post-nasal stop voicing, and *samyoge guru* / *laghu* syllable weighting.
2. **Disentangled Multi-Speaker Training (`0` Inference Overhead):** A training-only `UtteranceStyleEncoder` (`[mel_mean_80, mel_std_80] -> R^192`) absorbs speaker pitch/timbre variance across 9 studio corpora while an EMA `lang_voice_embed` (`[6, 192]`) anchors clean per-language studio voices at inference — adding **zero extra ONNX ops or parameters**.
3. **Latent-Manifold HiFi-GAN Feature Matching + Validation `WiSE-FT`:** Optimizes text-encoder and normalizing-flow latents directly against the frozen adversarial 16 kHz HiFi-GAN manifold with vectorized Viterbi Monotonic Alignment Search (`MAS`), unbiased `floor(exp(log_dur) + 0.5)` duration rounding, and validation-only Weight-Space Ensembling (`WiSE-FT`, `α = 0.70`).
4. **Dual-Tier Graceful Degradation:** Runs Tier-1 Neural VITS+HiFi-GAN (`56.6 MB` single-file FP16 ONNX, `0` `ConvInteger` ops, `42.37 dB` waveform SNR / `r = 0.99996` vs PyTorch FP32) in `onnxruntime-web` WASM SIMD, `onnxruntime-node`, or Python `onnxruntime`, and falls back automatically to a **`0.0 MB` pure-math Cascade Klatt-Akṣara formant synthesizer** offline.

## Benchmark Leaderboard (`N = 133` Held-Out Studio Utterances, `B = 1,000` Bootstrap 95% CIs)

Evaluated strictly on the leakage-free held-out test split (`48` Sanskrit across IISc Vedānta prose, All India Radio Sanskrit news, IITB high-SNR spoken Sanskrit, and IITK Bhagavad-Gītā ślokas + `85` Hindi, Kannada, Tamil, Telugu, and Malayalam studio utterances) using Kubichek (1993) DTW Mel-Cepstral Distortion (`MCD`), voiced YIN Log-F0 RMSE, utterance Duration MAE, and an independent `115M`-parameter `Su-śrotā` Conformer-CTC round-trip ASR oracle:

| System | Footprint (↓) | Coverage (↑) | MCD (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) | Sanskrit CER (%, ↓) | Pan-Indic CER (%, ↓) | RTF (↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Studio Human Audio *(Topline)* | — | 100.0% | — | — | — | `4.16% [2.92, 5.42]` | `16.50%` | — |
| Base `facebook/mms-tts-hin` *(Raw)* | 138.8 MB | 55.6% | `7.60 [7.24, 7.97]` | 911.6 | 3.71s | `23.11% [20.07, 26.23]` | 60.50% | 0.066x* |
| Base `mms-tts-hin` + `ninada` G2P | 138.8 MB | 100.0% | `6.57 [6.18, 6.99]` | 738.5 | 1.16s | `23.65% [20.56, 26.73]` | 33.45% | 0.105x |
| Meta MMS 5-Specialist Fleet *(5x Models)* | 694.0 MB | 100.0% | `6.29 [5.89, 6.72]` | 720.2 | 1.19s | `23.11% [20.07, 26.23]` | 27.02% | 0.102x |
| **`ninada` Tier-1A (GCP L4 PyTorch FP32)** | 140.0 MB | **100.0%** | **`5.56 [5.11, 6.04]`** | **554.9** | **0.89s** | **`13.59% [11.45, 15.71]`** | **22.88%** | **0.095x** |
| **`ninada` Tier-1B (Browser ONNX)** | **56.6 MB** | **100.0%** | **`5.56 [5.12, 6.04]`** | **553.0** | **0.89s** | **`13.46% [11.31, 15.42]`** | **22.88%** | 0.188x |
| `ninada` Tier-2 *(Zero-Dep Klatt Fallback)* | **0.0 MB** | **100.0%** | `10.92 [10.68, 11.15]` | 673.4 | 2.08s | `86.65% [84.48, 88.54]` | 89.48% | **0.012x** |

*\*Raw `mms-tts-hin` emits empty/unvoiced audio on `44.4%` of held-out utterances (`kn`, `ta`, `te`, `ml` scripts OOV).*

### Per-Language Spectral Fidelity (`MCD dB`, ↓)

| Language | Held-Out `N` | `mms-tts-hin` (Raw) | `mms-tts-hin` + G2P | MMS 5-Specialist Fleet (`694 MB`) | **`ninada` Tier-1A (FP32)** | **`ninada` Tier-1B (Browser ONNX)** |
|---|---:|---:|---:|---:|---:|---:|
| Sanskrit (`sa`) | 48 | 8.41 | 8.44 | 8.41 | **7.89** | **7.89** |
| Hindi (`hi`) | 18 | 5.61 | 5.57 | 5.61 | **4.09** | **4.08** |
| Kannada (`kn`) | 13 | 8.93 | 5.73 | 5.29 | **4.25** | **4.25** |
| Tamil (`ta`) | 18 | 7.04 | 5.46 | 4.82 | **4.15** | **4.14** |
| Telugu (`te`) | 18 | 7.61 | 5.60 | 5.40 | **4.37** | **4.39** |
| Malayalam (`ml`) | 18 | 7.04 | 5.27 | 4.35 | 4.36 | 4.37 |

## Install

```sh
# JavaScript / TypeScript / Browser / Node.js
npm install ninada onnxruntime-web

# Python 3.10+
pip install ninada
```

## Usage

### Browser & Node.js (3 Lines)

```js
import ninada from 'ninada';

const voice = await ninada('sanskrit', {
  model: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/model_quantized.onnx',
});
const wavBuffer = await voice.speak('प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् ।');
```

Works seamlessly across native Brahmic scripts and Romanized ISO-15919/IAST:

```js
await voice.speak('prajñāpūrvakavyāpāravattvaṁ jīvatvam', { lang: 'sa' });
await voice.speak('ಅರಿವೇ ಗುರು, ನುಡಿಯೇ ಬೆಳಕು.', { lang: 'kn' });
await voice.speak('யாதும் ஊரே யாவரும் கேளிர்.', { lang: 'ta' });
```

### Python SDK

```python
from ninada import Ninada, text_to_ids

tts = Ninada(lang="sa")  # auto-downloads gnumanth/ninada Browser ONNX from HF Hub
wav_bytes = tts.speak("ऋतं च सत्यं चाभीद्धात् तपसोऽध्यजायत ।")
open("rigveda.wav", "wb").write(wav_bytes)

# Inspect Pan-Indic G2P & Paninian Guru/Laghu syllable weights
info = text_to_ids("प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् ।", lang="sa")
print([(a["orth"], a["weight"]) for a in info["aksharas"]])
```

### CLI (`npx` & Python)

```sh
npx ninada "सत्यमेव जयते नानृतम्" -l sa -m ./docs/model_quantized.onnx -o satyam.wav
ninada "ಅರಿವೇ ಗುರು" -l kn -o kannada.wav
```

## Reproducing GCP `NVIDIA L4` Training & Evaluation

```sh
# 1. Curate studio-gated 9-corpus Sanskrit + Pan-Indic dataset (1,327 clips, 80/10/10 zero-leakage split)
PYTHONPATH=python/src:. python training/prepare_dataset.py --out-dir data

# 2. Train Disentangled Ninada VITS on GPU (24 epochs, NVIDIA L4 24GB)
PYTHONPATH=python/src:. python training/train_gcp.py --data-dir data --output-dir artifacts/ninada --epochs 24 --batch-size 8

# 3. Validation-only WiSE-FT selection + Browser FP16 single-file ONNX export
PYTHONPATH=python/src:. python training/wise_ft_select.py --checkpoint-dir artifacts/ninada --val-meta data/val_meta.jsonl
PYTHONPATH=python/src:. python training/export_onnx.py --checkpoint-dir artifacts/ninada --copy-to-docs docs

# 4. Run 7-system held-out benchmark (B = 1,000 bootstrap CIs) + Playwright Chromium WASM SIMD check
PYTHONPATH=python/src:. python bench/run_bench.py
node verify-browser.mjs
```

## Upstream Corpora & Credits

Curated from 9 open Hugging Face speech corpora: `prathoshap/sushrota-sanskrit-asr-data` (IISc SPIRE Lab), `surajp/shrutilipi_sanskrit` (AI4Bharat), `addy88/sanskrit-asr-84` (IIT Bombay), `yashnbx/gita_supersite_sanskrit_tts` (IIT Kanpur), `SayantanJoker/SYSPIN_Hindi_Female_TTS`, `SayantanJoker/original_data_kannada_tts`, `SayantanJoker/original_data_tamil_tts` (IISc SYSPIN), `arpit-tiwari/syspin-telugu-tts`, and `thennal/indic_tts_ml` (IIT Madras), warm-started from `facebook/mms-tts-hin`.

## License

MIT © [Hemanth HM](https://h3manth.com)
