# `Ninada` Scientific Multi-Benchmark & Ablation Report

## 1. Experimental Setup & Leakage-Free Studio Corpus
- **Held-Out Test Split**: `133` studio utterances (`16 kHz` mono PCM) across **6 Indic languages** (`sa` [48], `hi` [18], `kn` [13], `ta` [18], `te` [18], `ml` [18]), including **4 distinct Sanskrit speech domains** (IISc Vedanta prose, All India Radio Sanskrit news, IITB high-SNR spoken Sanskrit, and IITK Bhagavad-Gita shlokas), strictly disjoint (`0` shared audio hashes or normalized transcripts) from the training and validation splits.
- **Bootstrap Confidence Intervals**: `B = 1,000` non-parametric bootstrap resamples (`95% CI`).
- **Specialist Baseline Included**: In addition to single-checkpoint `facebook/mms-tts-hin` (`138.8 MB`), we benchmark against the **`694.0 MB` Meta MMS-TTS 5-Specialist Fleet** (`mms-tts-hin` + `mms-tts-kan` + `mms-tts-tam` + `mms-tts-tel` + `mms-tts-mal`, routing every language to its dedicated native-script checkpoint).
- **Independent Round-Trip Intelligibility Oracle**: `Su-srota` IndicConformer-CTC (`115M` params, `sushrota_sanskrit_ctc_int8.onnx`), completely independent of the VITS generator.

## 2. Overall Pan-Indic & Sanskrit Leaderboard (`N = 133` Test Utterances)

| System | Footprint (↓) | Coverage (↑) | MCD (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) | Sanskrit CER (%, ↓) | Pan-Indic Phonetic CER (%, ↓) | RTF (↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Studio Human Audio (Topline) | — | 100.0% | — | — | — | 4.16% [2.92, 5.42] | 16.50% | — |
| Base `mms-tts-hin` (Native Tokenizer) | 138.8 MB | 55.6% | 7.60 [7.24, 7.97] | 911.6 | 3.71s | 23.11% [20.07, 26.23] | 60.50% | 0.066x |
| Base `mms-tts-hin` + `Ninada` G2P | 138.8 MB | 100.0% | 6.57 [6.18, 6.99] | 738.5 | 1.16s | 23.65% [20.56, 26.73] | 33.45% | 0.105x |
| Meta MMS 5-Specialist Fleet (`5x` Models) | 694.0 MB | 100.0% | 6.29 [5.89, 6.72] | 720.2 | 1.19s | 23.11% [20.07, 26.23] | 27.02% | 0.102x |
| **`Ninada` Tier-1A (GCP L4 FP32)** | 140.0 MB | 100.0% | 5.56 [5.11, 6.04] | 554.9 | 0.89s | 13.59% [11.45, 15.71] | 22.88% | 0.095x |
| **`Ninada` Tier-1B (Browser ONNX)** | 56.6 MB | 100.0% | 5.56 [5.12, 6.04] | 553.0 | 0.89s | 13.46% [11.31, 15.42] | 22.88% | 0.188x |
| `Ninada` Tier-2 (Zero-Dep Klatt) | 0.0 MB | 100.0% | 10.92 [10.68, 11.15] | 673.4 | 2.08s | 86.65% [84.48, 88.54] | 89.48% | 0.012x |

## 3. Per-Language MCD (dB, ↓) Breakdown Across 6 Indic Languages

| Language | Test `N` | `mms-tts-hin` (Raw) | `mms-tts-hin` + G2P | MMS 5-Specialist Fleet (`694 MB`) | **`Ninada` Tier-1A (FP32)** | **`Ninada` Tier-1B (Browser ONNX)** |
|---|---:|---:|---:|---:|---:|---:|
| Sanskrit (`sa`) | 48 | 8.41 | 8.44 | 8.41 | **7.89** | **7.89** |
| Hindi (`hi`) | 18 | 5.61 | 5.57 | 5.61 | **4.09** | **4.08** |
| Kannada (`kn`) | 13 | 8.93 | 5.73 | 5.29 | **4.25** | **4.25** |
| Tamil (`ta`) | 18 | 7.04 | 5.46 | 4.82 | **4.15** | **4.14** |
| Telugu (`te`) | 18 | 7.61 | 5.60 | 5.40 | **4.37** | **4.39** |
| Malayalam (`ml`) | 18 | 7.04 | 5.27 | 4.35 | **4.36** | **4.37** |

## 4. Browser ONNX Compression & Numerical Parity
- **FP32 ONNX Size**: `112.61 MB` -> **Browser Single-File ONNX (`model_quantized.onnx`)**: `56.61 MB` (`2.0x` smaller than FP32, `12.3x` smaller than the `694.0 MB` 5-Specialist MMS Fleet, zero external `.data` shards, zero unsupported `ConvInteger` ops in `onnxruntime-web` WASM SIMD).
- **PyTorch FP32 <-> Browser ONNX Waveform Parity**: Mean SNR = **`42.37 dB`**, Mean Pearson `r` = **`0.999960`**.
