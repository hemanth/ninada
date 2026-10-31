# ninada (निनाद) v2.1

> Natural, real-time streaming speech synthesis for Classical Sanskrit and 5 Indian languages (`sa`, `kn`, `hi`, `ta`, `te`, `ml`) — **`~260 ms` browser TTFA**, **`0.29x` multi-core WASM RTF**, **`0` audio underruns** via a dedicated `AudioWorklet` ring buffer, and **`54.7 MB`** split Halo-Streaming VITS + HiFi-GAN specialists cached offline in the browser, Node.js, and Python.

- **Live Streaming Voice Studio:** <https://hemanth.github.io/ninada/>
- **Hugging Face Model (`gnumanth/ninada`):** <https://huggingface.co/gnumanth/ninada>
- **Hugging Face Dataset (`gnumanth/ninada-sanskrit-indic`):** <https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic>
- **Scientific & Streaming Report:** [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md) · [`bench/results.json`](./bench/results.json)

![Ninada Streaming Architecture Blueprint](./docs/flow-diagram.svg)

## Architecture

Open-weight multilingual TTS models lack a native Classical/Vedic Sanskrit checkpoint (`facebook/mms-tts-san` does not exist), require **5 separate FP32 models (`694.0 MB` combined)** to cover Hindi, Kannada, Tamil, Telugu, and Malayalam, and synthesize utterances monolithically (blocking first audio until the entire waveform finishes).

`ninada` (*Nināda* — Sanskrit `निनाद`, deep resonant sound) introduces a custom **Two-Stage Split Halo-Streaming VITS + HiFi-GAN** architecture across **6 native-script voice specialists** (`sa`, `kn`, `hi`, `ta`, `te`, `ml`), each exported as a **`54.7 MB` split FP16 ONNX** pair (`27.3 MB` Stage-1 Encoder+SDP+Flow + `27.4 MB` Stage-2 Halo-Chunked Vocoder):

1. **Two-Stage Halo-Crossfaded Streaming (`180.1 ms` SDK / `262 ms` Browser TTFA, `122.1 dB` Seam SNR):** Splits synthesis into `onnx/{lang}/encoder_flow_quantized.onnx` (Akshara text encoder + Stochastic Duration Predictor + 4-step reversible latent flow) and `onnx/{lang}/vocoder_quantized.onnx` (progressive `44 -> 60 -> 78 -> 96` frame ladder with a `12`-frame receptive-field halo and `256`-sample raised-cosine boundary crossfade).
2. **Smooth Real-Time Browser `AudioWorklet` + Multi-Core WASM + Offline Cache (`0` Underruns, `0` Refetches):** Pairs a Cross-Origin Isolation Service Worker (`docs/coi-sw.js`, unlocking multi-core WASM SIMD at `~0.29x` RTF) with a dedicated real-time `AudioWorkletProcessor` (`docs/pcm-stream-worklet.js`, stateful $C^1$ Hermite upsampling + adaptive prebuffer watermark) and persistent `CacheStorage` / `IndexedDB` model caching.
3. **Studio-Gated Anechoic Pan-Indic Voice Specialists (`1,438` Clips, `127.55` Min):** Six dedicated male studio specialists refined with Multi-Period and Multi-Resolution Discriminators (`MPD + MRD`), Stochastic Duration Prediction, and validation-gated weight-space ensembling (`WiSE-FT`).
4. **Zero-Schwa-Deletion Sanskrit & Native Brahmic G2P + Zero-Dep Klatt Fallback:** Preserves Classical/Vedic Sanskrit vocalic `ṛ`/`ṝ`, velar `ṅ`, homorganic nasal sandhi, utterance-final vowels, and *pūrvarūpa avagraha* (`ऽ`), while routing `hi`, `kn`, `ta`, `te`, `ml` to native orthographies (including atomic Malayalam chillu letters `ൽ, ർ, ൺ, ൻ, ൾ`), degrading gracefully to a **`0.0 MB` pure-math Cascade Klatt-Akṣara stream** offline.

## Benchmark Leaderboard (`N = 144` Held-Out Studio Utterances, `B = 1,000` Bootstrap 95% CIs)

Evaluated strictly on the leakage-free held-out test split (`39` Sanskrit + `105` across `hi`, `kn`, `ta`, `te`, `ml` — `21` each) using Kubichek (1993) DTW Mel-Cepstral Distortion (`MCD`), voiced YIN Log-F0 RMSE, utterance Duration MAE, streaming `TTFA`, and an independent `115M`-parameter Conformer-CTC round-trip Sanskrit ASR oracle:

| System | Active Footprint (↓) | Coverage (↑) | TTFA p50 (ms, ↓) | MCD (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) | Sanskrit CER (%, ↓) | Sanskrit WER (%, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Studio Human Audio *(Topline)* | — | 100.0% | — | — | — | — | `1.78% [1.14, 2.46]` | `19.06%` |
| Base `facebook/mms-tts-hin` *(Raw)* | 138.8 MB | 43.1%* | 0.6 ms* | `6.50 [6.30, 6.71]` | 961.1 | 3.10s | `19.48% [16.51, 22.60]` | 70.30% |
| Base `mms-tts-hin` + `ninada` G2P | 138.8 MB | 100.0% | 463.4 ms | `5.49 [5.40, 5.58]` | 654.8 | 0.89s | `20.27% [17.07, 23.62]` | 70.83% |
| Meta MMS 5-Specialist Fleet *(5x Models)* | 694.0 MB | 100.0% | 507.5 ms | `5.09 [5.02, 5.17]` | 663.7 | 1.31s | `19.48% [16.51, 22.60]` | 70.30% |
| **`ninada` v2.1 Tier-1A (FP32 Checkpoint)** | 138.4 MB | **100.0%** | **150.8 ms** | **`4.70 [4.63, 4.77]`** | **628.0** | **0.72s** | **`16.65% [13.17, 20.36]`** | **57.95%** |
| **`ninada` v2.1 Tier-1B (Streaming ONNX)** | **54.7 MB** | **100.0%** | **180.1 ms** | **`4.69 [4.61, 4.76]`** | **626.4** | **0.73s** | **`16.95% [13.46, 20.28]`** | **58.79%** |
| `ninada` Tier-2 *(Zero-Dep Klatt Fallback)* | **0.0 MB** | **100.0%** | **34.6 ms** | `11.32 [11.12, 11.52]` | 533.5 | 1.33s | `87.78% [85.78, 89.80]` | 100.0% |

*\*Raw `mms-tts-hin` emits empty zero-length buffers (`0.6 ms`) on `56.9%` of held-out utterances (`kn`, `ta`, `te`, `ml` scripts OOV).*

### Per-Language Spectral Fidelity (`MCD dB`, ↓ — `ninada` v2.0 Ranks #1 on All 6 Languages)

| Language | Held-Out `N` | `mms-tts-hin` (Raw) | `mms-tts-hin` + G2P | MMS 5-Specialist Fleet (`694 MB`) | **`ninada` v2.0 Tier-1A (FP32)** | **`ninada` v2.0 Tier-1B (Streaming ONNX)** |
|---|---:|---:|---:|---:|---:|---:|
| Sanskrit (`sa`) | 39 | 5.35 | 5.34 | 5.35 | **4.90** | **4.87** |
| Hindi (`hi`) | 21 | 4.90 | 4.88 | 4.90 | **4.26** | **4.25** |
| Kannada (`kn`) | 21 | 7.70 | 5.51 | 4.86 | **4.51** | **4.59** |
| Tamil (`ta`) | 21 | 7.33 | 5.30 | 4.81 | **4.59** | **4.59** |
| Telugu (`te`) | 21 | 7.29 | 6.08 | 5.59 | **5.24** | **5.19** |
| Malayalam (`ml`) | 21 | 7.42 | 5.95 | 4.81 | **4.51** | **4.47** |

## Install

```sh
# JavaScript / TypeScript / Browser / Node.js
npm install ninada onnxruntime-web

# Python 3.10+
pip install ninada
```

## Streaming & One-Shot Usage

### Browser & Node.js (`voice.stream()`, `voice.speak()`, & Persistent Caching)

```js
import ninada from 'ninada';

const voice = await ninada('kannada', {
  encoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/kn/encoder_flow_quantized.onnx',
  vocoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/kn/vocoder_quantized.onnx',
  cacheModels: true, // persists in browser CacheStorage / IndexedDB (0 network fetches on reload)
});

// Sub-second 16-frame halo + 256-sample raised-cosine crossfaded streaming (Float32Array @ 16 kHz)
for await (const chunk of voice.stream('ಕನ್ನಡ ಭಾಷೆಯ ನೈಸರ್ಗಿಕ ಧ್ವನಿ ಸಂಶ್ಲೇಷಣೆಗೆ ಸ್ವಾಗತ.')) {
  playPcm16k(chunk.pcm); // chunk.ttfaMs ~ 180ms
}

// Or synthesize a complete 16 kHz RIFF WAV buffer
const wavBuffer = await voice.speak('ಕನ್ನಡವೇ ಸತ್ಯ ಕನ್ನಡವೇ ನಿತ್ಯ', { lang: 'kn' });
await voice.dispose(); // release WASM session heap when switching specialists
```

### Python SDK (`tts.stream()` & `tts.speak()`)

```python
from ninada import Ninada, text_to_ids

tts = Ninada(lang="sa")  # auto-downloads per-language gnumanth/ninada split streaming ONNX from HF Hub
for chunk in tts.stream("प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् । यथार्थज्ञानं प्रमाणम् ॥"):
    # chunk["pcm"] (float32 @ 16 kHz), chunk["ttfa_ms"] (~180 ms)
    pass

wav_bytes = tts.speak("ಕನ್ನಡವೇ ಸತ್ಯ ಕನ್ನಡವೇ ನಿತ್ಯ", lang="kn")
open("kannada.wav", "wb").write(wav_bytes)
```

### CLI (`npx` & Python)

```sh
npx ninada "ಕನ್ನಡವೇ ಸತ್ಯ ಕನ್ನಡವೇ ನಿತ್ಯ" -l kn -o kannada.wav
ninada "सत्यमेव जयते नानृतम्" -l sa -o satyam.wav
```

## Reproducing Training, Export & Verification

```sh
# 1. Prepare studio-gated dataset (1,438 clips, 80/10/10 zero-leakage split)
PYTHONPATH=python/src:. python training/prepare_dataset.py --out-dir data

# 2. Train 6 Adversarial MPD+MRD+SDP Specialists + Validation WiSE-FT
PYTHONPATH=python/src:. python training/train_gcp.py --data-dir data --output-dir artifacts/ninada --batch-size 4

# 3. Export Split Halo-Crossfaded Browser FP16 ONNX (6 Specialists, 54.7 MB each)
PYTHONPATH=python/src:. python training/export_onnx.py --checkpoint-dir artifacts/ninada --copy-to-docs docs

# 4. Run 7-system held-out benchmark (N = 144, B = 1,000 bootstrap CIs) + Playwright AudioWorklet & Cache check
PYTHONPATH=python/src:. python bench/run_bench.py
BROWSER_CHANNEL=chrome node verify-browser.mjs
```

## License

MIT © [Hemanth HM](https://h3manth.com)
