# ninada (निनाद) v2.0

> Stream Classical & Vedic Sanskrit plus 6 Indic languages (`sa`, `kn`, `hi`, `ta`, `te`, `ml`) with **`180.1 ms` median TTFA** and **`123.7 dB` crossfaded seam SNR** from **`54.7 MB`** native Halo-Chunked VITS + HiFi-GAN FP16 ONNX specialists in the browser (Web Worker WASM SIMD + persistent `CacheStorage`), Node.js, and Python.

- **Live Web Worker Streaming Studio:** <https://hemanth.github.io/ninada/>
- **Hugging Face Model (`gnumanth/ninada`):** <https://huggingface.co/gnumanth/ninada>
- **Hugging Face Dataset (`gnumanth/ninada-sanskrit-indic`):** <https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic>
- **Scientific & Streaming Report:** [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md) · [`bench/results.json`](./bench/results.json)

![Ninada Streaming Architecture Blueprint](./docs/flow-diagram.svg)

## Why `ninada`?

Open-weight multilingual TTS ecosystems lack a native Classical/Vedic Sanskrit checkpoint (`facebook/mms-tts-san` does not exist on Hugging Face), require **5 separate FP32 models (`694.0 MB` combined)** to cover Hindi, Kannada, Tamil, Telugu, and Malayalam, and synthesize utterances monolithically (blocking first audio until the entire HiFi-GAN waveform finishes).

`ninada` (*Nināda* — Sanskrit `निनाद`, deep resonant sound) v2.0 trains **6 native-script adversarial `MPD + MRD + SDP` studio Male VITS + HiFi-GAN specialists** (`sa`, `kn`, `hi`, `ta`, `te`, `ml`) across **9 Brahmic scripts + ISO-15919/IAST**, exporting each into a **`54.7 MB` split streaming FP16 ONNX** pair (`27.3 MB` Stage-1 Encoder+SDP+Flow + `27.4 MB` Stage-2 Halo-Chunked Vocoder) trained on **Google Cloud `NVIDIA L4` (`24 GB` VRAM) GPUs**:

1. **Halo-Crossfaded Neural Streaming (`180.1 ms` TTFA, `123.72 dB` Seam SNR):** Splits VITS inference into `onnx/{lang}/encoder_flow_quantized.onnx` (native ONNX-safe Stochastic Duration Predictor + 4-step normalizing flow) and `onnx/{lang}/vocoder_quantized.onnx` (`16`-frame receptive-field halo + `256`-sample raised-cosine boundary crossfade + `48 kHz` hardware Web Audio sinc upsampling), cutting **Time-To-First-Audio by `2.8x–3.4x`** with zero boundary clicks.
2. **Off-Main-Thread Browser Web Worker + Persistent `CacheStorage` (`0` Refetches):** Runs `onnxruntime-web` WASM SIMD inside a dedicated Web Worker (`docs/synth-worker.mjs`) with persistent browser `CacheStorage` (`ninada-onnx-v2.0.0`) + `IndexedDB` fallback (`0` network requests on reload or language revisit), explicit `voice.dispose()` WASM session release on language switch, strict UI control locking while any specialist loads, and gapless Web Audio scheduling (`16.8 ms` max main-thread frame gap across multi-sentence Kannada passages).
3. **Anechoic Male Studio `Su-śrotā` Sanskrit + `AI4Bharat Rasa` Corpus (`1,438` Clips, `127.55` Min):** Trained on studio-gated Male Sanskrit recordings from `prathoshap/sushrota-sanskrit-asr-data` (`recitation_corpus` ślokas + `scholar` śāstra prose, `studio_align_cer <= 5.0%`, synthetic `tts_aug` excluded) + `ai4bharat/Rasa` anechoic studio Male narration across `sa`, `hi`, `kn`, `ta`, `te`, `ml` (`1,150` train / `144` val / `144` test), refined with adversarial `MPD + MRD` feature matching and validation-only end-to-end `WiSE-FT`.
4. **Zero-Schwa-Deletion Sanskrit & Native Brahmic G2P + Zero-Dep Klatt Fallback:** Maps Classical/Vedic Sanskrit onto the zero-schwa-deletion Kannada phoneme inventory (`+3` donor-initialized extensions `ಙ`, `ೠ`, `ೄ`, homorganic nasal->anusvāra `ಂ`, utterance-final boundary protection, and *pūrvarūpa avagraha* sandhi `ऽ`), while routing `hi`, `kn`, `ta`, `te`, `ml` to native orthographies (including atomic Malayalam chillu letters `ൽ, ർ, ൺ, ൻ, ൾ`), degrading gracefully to a **`0.0 MB` pure-math Cascade Klatt-Akṣara stream** offline.

## Benchmark Leaderboard (`N = 144` Held-Out Studio Utterances, `B = 1,000` Bootstrap 95% CIs)

Evaluated strictly on the leakage-free held-out test split (`39` Sanskrit + `105` across `hi`, `kn`, `ta`, `te`, `ml` — `21` each) using Kubichek (1993) DTW Mel-Cepstral Distortion (`MCD`), voiced YIN Log-F0 RMSE, utterance Duration MAE, streaming `TTFA`, and an independent `115M`-parameter `Su-śrotā` Conformer-CTC round-trip Sanskrit ASR oracle:

| System | Active Footprint (↓) | Coverage (↑) | TTFA p50 (ms, ↓) | MCD (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) | Sanskrit CER (%, ↓) | Sanskrit WER (%, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Studio Human Audio *(Topline)* | — | 100.0% | — | — | — | — | `1.78% [1.14, 2.46]` | `19.06%` |
| Base `facebook/mms-tts-hin` *(Raw)* | 138.8 MB | 43.1%* | 0.6 ms* | `6.50 [6.30, 6.71]` | 961.1 | 3.10s | `19.48% [16.51, 22.60]` | 70.30% |
| Base `mms-tts-hin` + `ninada` G2P | 138.8 MB | 100.0% | 463.4 ms | `5.49 [5.40, 5.58]` | 654.8 | 0.89s | `20.27% [17.07, 23.62]` | 70.83% |
| Meta MMS 5-Specialist Fleet *(5x Models)* | 694.0 MB | 100.0% | 507.5 ms | `5.09 [5.02, 5.17]` | 663.7 | 1.31s | `19.48% [16.51, 22.60]` | 70.30% |
| **`ninada` v2.0 Tier-1A (GCP L4 FP32)** | 138.4 MB | **100.0%** | **150.8 ms** | **`4.70 [4.63, 4.77]`** | **628.0** | **0.72s** | **`16.65% [13.17, 20.36]`** | **57.95%** |
| **`ninada` v2.0 Tier-1B (Streaming ONNX)** | **54.7 MB** | **100.0%** | **180.1 ms** | **`4.69 [4.61, 4.76]`** | **626.4** | **0.73s** | **`16.95% [13.46, 20.28]`** | **58.79%** |
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

## Reproducing GCP `NVIDIA L4` Adversarial Training & Evaluation

```sh
# 1. Curate studio-gated Male Su-srota Sanskrit + AI4Bharat Rasa dataset (1,438 clips, 80/10/10 zero-leakage split)
PYTHONPATH=python/src:. python training/prepare_dataset.py --out-dir data

# 2. Train 6 Adversarial MPD+MRD+SDP Specialists + Validation WiSE-FT on GCP NVIDIA L4 GPU
PYTHONPATH=python/src:. python training/train_gcp.py --data-dir data --output-dir artifacts/ninada --batch-size 4

# 3. Export Split Halo-Crossfaded Browser FP16 ONNX (6 Specialists, 54.7 MB each, 123.7 dB Seam SNR)
PYTHONPATH=python/src:. python training/export_onnx.py --checkpoint-dir artifacts/ninada --copy-to-docs docs

# 4. Run 7-system held-out benchmark (N = 144, B = 1,000 bootstrap CIs) + Playwright Web Worker & Cache check
PYTHONPATH=python/src:. python bench/run_bench.py
BROWSER_CHANNEL=chrome node verify-browser.mjs
```

## Upstream Corpora & Credits

Curated from clean anechoic Male studio Hugging Face speech corpora: `prathoshap/sushrota-sanskrit-asr-data` (IISc SPIRE Lab — human `recitation_corpus` & `scholar` subsets, `tts_aug` excluded) and `ai4bharat/Rasa` (`sa`, `hi`, `kn`, `ta`, `te`, `ml` studio Male narration), warm-started from `facebook/mms-tts-{kan,hin,tam,tel,mal}`.

## License

MIT © [Hemanth HM](https://h3manth.com)
