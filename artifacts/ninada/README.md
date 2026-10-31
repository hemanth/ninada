---
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

`Ninada` (*Nināda* — Sanskrit `निनाद`, deep resonant sound) is a compact, single-checkpoint **Classical Sanskrit & Pan-Indic** neural text-to-speech model (**`56.6 MB` single-file browser ONNX**) supporting **6 Indic languages** (`Sanskrit [sa]`, `Hindi [hi]`, `Kannada [kn]`, `Tamil [ta]`, `Telugu [te]`, `Malayalam [ml]`) across **9 Brahmic scripts + ISO-15919/IAST** (`87` Pan-Indic & Vedic/Classical Sanskrit phoneme tokens including vocalic `ऋ`/`ॠ`, velar nasal `ङ`, Dravidian retroflexes `ळ`/`ऴ`/`ऱ`/`ऩ`, *pūrvarūpa avagraha* sandhi, and *Pāṇinian samyoge guru* prosody).

- **Live Browser WASM SIMD Studio**: [https://hemanth.github.io/ninada/](https://hemanth.github.io/ninada/)
- **GitHub Repository & SDKs (JS + Python)**: [https://github.com/hemanth/ninada](https://github.com/hemanth/ninada)
- **Curated Split Metadata**: [https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic](https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic)

## GCP NVIDIA L4 GPU Training & Disentangled Architecture
- **Hardware**: Trained on **Google Cloud (`NVIDIA L4` 24 GB GPU)** for `24` epochs (`3192` steps) with speaker-disentangled `UtteranceStyleEncoder` + EMA per-language studio voice centroids, latent-manifold feature matching in the frozen adversarial 16 kHz HiFi-GAN space, vectorized Viterbi Monotonic Alignment Search (`MAS`), and validation-only Weight-Space Ensembling (`WiSE-FT`).
- **Curated Multi-Domain Corpus (`1327` studio-aligned clips, `144.0` min at 16 kHz; `1061` train / `133` val / `133` test)**:
  - **Sanskrit (`474` studio-gated clips across 4 corpora)**: `prathoshap/sushrota-sanskrit-asr-data` (IISc Vedānta prose), `surajp/shrutilipi_sanskrit` (All India Radio Sanskrit news), `addy88/sanskrit-asr-84` (IITB WadaSNR >= 32 dB spoken Sanskrit), and `yashnbx/gita_supersite_sanskrit_tts` (IITK Bhagavad-Gītā ślokas).
  - **Pan-Indic (`853` clips across 5 studio corpora)**: SYSPIN Hindi, Kannada, Tamil, Telugu, and IITM Malayalam.

## Held-Out Test Leaderboard (`N = 133` Utterances; `B = 1,000` Bootstrap 95% CIs)

| System | Footprint (↓) | Coverage (↑) | MCD (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) | Sanskrit CER (%, ↓) | Pan-Indic CER (%, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Studio Human Audio (Topline) | — | 100.0% | — | — | — | `4.16 [2.92, 5.42]%` | `16.50 [14.46, 18.76]%` |
| Base `facebook/mms-tts-hin` (Raw) | 138.8 MB | 55.6% | `7.60 [7.24, 7.97]` | 911.6 | 3.71s | `23.11 [20.07, 26.23]%` | `60.50 [54.01, 67.15]%` |
| Base `mms-tts-hin` + `Ninada` G2P | 138.8 MB | 100.0% | `6.57 [6.18, 6.99]` | 738.5 | 1.16s | `23.65 [20.56, 26.73]%` | `33.45 [30.91, 36.20]%` |
| Meta MMS 5-Specialist Fleet (`5x` Models) | 694.0 MB | 100.0% | `6.29 [5.89, 6.72]` | 720.2 | 1.19s | `23.11 [20.07, 26.23]%` | `27.02 [25.09, 28.84]%` |
| **`Ninada` Tier-1A (GCP L4 FP32)** | 140.0 MB | **100.0%** | **`5.56 [5.11, 6.04]`** | **554.9** | **0.89s** | **`13.59 [11.45, 15.71]%`** | **`22.88 [21.07, 24.77]%`** |
| **`Ninada` Tier-1B (Browser ONNX)** | **56.6 MB** | **100.0%** | **`5.56 [5.12, 6.04]`** | **553.0** | **0.89s** | **`13.46 [11.31, 15.42]%`** | **`22.88 [20.90, 24.98]%`** |

## Quickstart

```js
import ninada from 'ninada';

const voice = await ninada('sanskrit', {
  model: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/model_quantized.onnx',
});
const wavBuffer = await voice.speak('ऋतं च सत्यं चाभीद्धात् तपसोऽध्यजायत ।');
```

```python
from ninada import Ninada

wav_bytes = Ninada(lang="sa").speak("प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् ।")
```
