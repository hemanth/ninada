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

`Ninada` (*Nināda* — Sanskrit `निनाद`, deep resonant sound) v2.0 is a sub-second **streaming Classical Sanskrit & Pan-Indic** neural text-to-speech suite (**`54.7 MB` active split streaming FP16 ONNX per specialist: `27.3 MB` Encoder+SDP+Flow + `27.4 MB` Halo-Chunked HiFi-GAN, `180.1 ms` median TTFA**) supporting **6 Indic languages** (`Sanskrit [sa]`, `Kannada [kn]`, `Hindi [hi]`, `Tamil [ta]`, `Telugu [te]`, `Malayalam [ml]`) across **9 Brahmic scripts + ISO-15919/IAST**.

- **Live Off-Main-Thread Web Worker Studio**: [https://hemanth.github.io/ninada/](https://hemanth.github.io/ninada/)
- **GitHub Repository & Streaming SDKs (JS + Python)**: [https://github.com/hemanth/ninada](https://github.com/hemanth/ninada)
- **Curated Split Metadata (`1438` Studio Clips)**: [https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic](https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic)

## GCP NVIDIA L4 GPU Training & Halo-Crossfaded Streaming Architecture
- **Hardware & Objective**: Trained on **Google Cloud (`NVIDIA L4` 24 GB GPU)** with alternating **`MPD + MRD` (`LSGAN` + multi-layer discriminator feature-matching) + Flow KL + ONNX-safe Stochastic Duration Predictor (`SDP`) MLE + validation-only end-to-end `WiSE-FT`** across 6 native-script VITS + HiFi-GAN specialists (`specialists/{sa,kn,hi,ta,te,ml}`).
- **Halo-Crossfaded Streaming (`123.7 dB` Seam SNR)**: Splits inference into `onnx/{lang}/encoder_flow_quantized.onnx` (`~27.3 MB`, native stochastic duration predictor + normalizing flow) and `onnx/{lang}/vocoder_quantized.onnx` (`~27.4 MB`, `16`-frame receptive-field halo + `256`-sample raised-cosine boundary crossfade), persisted in browser `CacheStorage` (`0` network fetches on reload or language switch).
- **Anechoic Male Studio Corpus (`1438` clips, `127.55` min at 16 kHz; `1150` train / `144` val / `144` test)**:
  - **Sanskrit (`388` Male studio clips)**: `ai4bharat/Rasa` neutral Sanskrit studio narration + `prathoshap/sushrota-sanskrit-asr-data` (`recitation_corpus` + `scholar`, `studio_align_cer <= 5.0%`, synthetic `tts_aug` excluded).
  - **Pan-Indic (`1,050` Male studio clips — `210` each across `hi`, `kn`, `ta`, `te`, `ml`)**: `ai4bharat/Rasa` 48 kHz -> 16 kHz anechoic booth recordings.

## Held-Out Test Leaderboard (`N = 144` Utterances; `B = 1,000` Bootstrap 95% CIs)

| System | Active Footprint (↓) | Coverage (↑) | TTFA p50 (ms, ↓) | MCD (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) | Sanskrit CER (%, ↓) | Sanskrit WER (%, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Studio Human Audio (Topline) | — | 100.0% | — | — | — | — | `1.78 [1.14, 2.46]%` | `19.06 [12.81, 25.70]%` |
| Base `facebook/mms-tts-hin` (Raw) | 138.8 MB | 43.1% | 0.6 ms | `6.50 [6.30, 6.71]` | 961.1 | 3.10s | `19.48 [16.51, 22.60]%` | `70.30 [60.81, 79.41]%` |
| Base `mms-tts-hin` + `Ninada` G2P | 138.8 MB | 100.0% | 463.4 ms | `5.49 [5.40, 5.58]` | 654.8 | 0.89s | `20.27 [17.07, 23.62]%` | `70.83 [60.99, 80.35]%` |
| Meta MMS 5-Specialist Fleet (`5x` Models) | 694.0 MB | 100.0% | 507.5 ms | `5.09 [5.02, 5.17]` | 663.7 | 1.31s | `19.48 [16.51, 22.60]%` | `70.30 [60.81, 79.41]%` |
| **`Ninada` v2.0 Tier-1A (GCP L4 FP32)** | 138.4 MB | **100.0%** | **150.8 ms** | **`4.70 [4.63, 4.77]`** | **628.0** | **0.72s** | **`16.65 [13.17, 20.36]%`** | **`57.95 [49.77, 66.54]%`** |
| **`Ninada` v2.0 Tier-1B (Streaming ONNX)** | **54.7 MB** | **100.0%** | **180.1 ms** | **`4.69 [4.61, 4.76]`** | **626.4** | **0.73s** | **`16.95 [13.46, 20.28]%`** | **`58.79 [49.94, 67.29]%`** |

## Streaming Quickstart

```js
import ninada from 'ninada';

const voice = await ninada('kannada', {
  encoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/kn/encoder_flow_quantized.onnx',
  vocoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/kn/vocoder_quantized.onnx',
});
for await (const chunk of voice.stream('ಕನ್ನಡ ಭಾಷೆಯ ನೈಸರ್ಗಿಕ ಧ್ವನಿ ಸಂಶ್ಲೇಷಣೆಗೆ ಸ್ವಾಗತ.')) {
  playPcm16k(chunk.pcm); // Sub-second TTFA!
}
```

```python
from ninada import Ninada

tts = Ninada(lang="sa")
for chunk in tts.stream("प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् । यथार्थज्ञानं प्रमाणम् ॥"):
    play(chunk["pcm"], chunk["sample_rate"])
```
