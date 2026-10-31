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

`Ninada` (*Nināda* — Sanskrit `निनाद`, deep resonant sound) is a sub-second **streaming Classical Sanskrit & Pan-Indic** neural text-to-speech model (**`56.6 MB` split streaming FP16 ONNX: `29.8 MB` Encoder+Flow + `26.8 MB` Halo-Chunked HiFi-GAN, `178.6 ms` median TTFA**) supporting **6 Indic languages** (`Sanskrit [sa]`, `Kannada [kn]`, `Hindi [hi]`, `Tamil [ta]`, `Telugu [te]`, `Malayalam [ml]`) across **9 Brahmic scripts + ISO-15919/IAST** (`87` Pan-Indic & Vedic/Classical Sanskrit phoneme tokens).

- **Live Off-Main-Thread Web Worker Studio**: [https://hemanth.github.io/ninada/](https://hemanth.github.io/ninada/)
- **GitHub Repository & Streaming SDKs (JS + Python)**: [https://github.com/hemanth/ninada](https://github.com/hemanth/ninada)
- **Curated Split Metadata (`1,550` Studio Clips)**: [https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic](https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic)

## GCP NVIDIA L4 GPU Training & Halo-Chunked Streaming Architecture
- **Hardware**: Trained on **Google Cloud (`NVIDIA L4` 24 GB GPU)** for `24` epochs (`3720` steps) with speaker-disentangled `UtteranceStyleEncoder` + EMA per-language studio voice centroids, latent-manifold feature matching in the frozen adversarial 16 kHz HiFi-GAN space, vectorized Viterbi Monotonic Alignment Search (`MAS`), and validation-only Weight-Space Ensembling (`WiSE-FT`).
- **Halo-Chunked Streaming (`53.6 dB` SNR Parity)**: Splits inference into `onnx/encoder_flow_quantized.onnx` (`~29.8 MB`, global clause prosody + normalizing flow in `~35 ms`) and `onnx/vocoder_quantized.onnx` (`~26.8 MB`, `40`-frame chunks with `10`-frame left/right receptive-field halo cropped at exact `256`-sample boundaries), cutting **Time-To-First-Audio (`TTFA`) by ~`7x`** while eliminating WebAssembly main-thread freezes inside a dedicated Web Worker.
- **Curated Studio Corpus (`1550` studio-aligned clips, `175.19` min at 16 kHz; `1240` train / `155` val / `155` test)**:
  - **Sanskrit (`650` pure studio human clips, `80.5 min`)**: Curated exclusively from `prathoshap/sushrota-sanskrit-asr-data` (`360` `recitation_corpus` ślokas + `290` `scholar` śāstra prose, `studio_align_cer <= 10.0%`, synthetic `tts_aug` excluded).
  - **Pan-Indic (`900` studio clips — `180` each across `hi`, `kn`, `ta`, `te`, `ml`)**: SYSPIN Hindi, Kannada, Tamil, Telugu, and IITM Malayalam.

## Held-Out Test Leaderboard (`N = 155` Utterances; `B = 1,000` Bootstrap 95% CIs)

| System | Footprint (↓) | Coverage (↑) | TTFA p50 (ms, ↓) | MCD (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) | Sanskrit CER (%, ↓) | Pan-Indic CER (%, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Studio Human Audio (Topline) | — | 100.0% | — | — | — | `2.58 [1.84, 3.31]%` | `14.63 [12.60, 16.71]%` |
| Base `facebook/mms-tts-hin` (Raw) | 138.8 MB | 59.4% | 392.6 ms | `6.39 [6.20, 6.61]` | 861.6 | 3.54s | `24.65 [22.42, 26.87]%` | `57.99 [51.85, 64.36]%` |
| Base `mms-tts-hin` + `Ninada` G2P | 138.8 MB | 100.0% | 738.2 ms | `5.47 [5.38, 5.55]` | 714.4 | 1.26s | `24.89 [22.57, 27.20]%` | `32.60 [30.33, 34.95]%` |
| Meta MMS 5-Specialist Fleet (`5x` Models) | 694.0 MB | 100.0% | 741.7 ms | `5.16 [5.07, 5.26]` | 697.1 | 1.24s | `24.65 [22.42, 26.87]%` | `26.58 [25.01, 28.40]%` |
| **`Ninada` Tier-1A (GCP L4 FP32 Stream)** | 140.0 MB | **100.0%** | **127.8 ms** | **`4.33 [4.26, 4.42]`** | **539.1** | **0.82s** | **`12.77 [11.00, 14.68]%`** | **`20.35 [18.58, 22.07]%`** |
| **`Ninada` Tier-1B (Streaming ONNX)** | **56.6 MB** | **100.0%** | **178.6 ms** | **`4.33 [4.26, 4.41]`** | **536.1** | **0.82s** | **`13.32 [11.53, 15.24]%`** | **`20.50 [18.82, 22.11]%`** |

## Streaming Quickstart

```js
import ninada from 'ninada';

const voice = await ninada('kannada', {
  encoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/encoder_flow_quantized.onnx',
  vocoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/vocoder_quantized.onnx',
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
