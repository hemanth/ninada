# `Ninada` v1.1 Scientific Multi-Benchmark & Streaming Report

## 1. Experimental Setup & Leakage-Free Studio Corpus
- **Held-Out Test Split**: `155` studio utterances (`16 kHz` mono PCM) across **6 Indic languages** (`sa` [65], `hi` [18], `kn` [18], `ta` [18], `te` [18], `ml` [18]), featuring **pure human studio Sanskrit** from `prathoshap/sushrota-sanskrit-asr-data` (`recitation_corpus` + `scholar`, synthetic `tts_aug` excluded) + 5 clean studio Indic corpora, strictly disjoint (`0` shared audio hashes or normalized transcripts) from the training and validation splits.
- **Bootstrap Confidence Intervals**: `B = 1,000` non-parametric bootstrap resamples (`95% CI`).
- **Specialist Baseline Included**: In addition to single-checkpoint `facebook/mms-tts-hin` (`138.8 MB`), we benchmark against the **`694.0 MB` Meta MMS-TTS 5-Specialist Fleet** (`mms-tts-hin` + `mms-tts-kan` + `mms-tts-tam` + `mms-tts-tel` + `mms-tts-mal`, routing every language to its dedicated native-script checkpoint).
- **Independent Round-Trip Intelligibility Oracle**: `Su-srota` IndicConformer-CTC (`115M` params, `sushrota_sanskrit_ctc_int8.onnx`), completely independent of the VITS generator.

## 2. Overall Pan-Indic & Sanskrit Leaderboard (`N = 155` Test Utterances)

| System | Footprint (↓) | Coverage (↑) | TTFA p50 (ms, ↓) | MCD (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) | Sanskrit CER (%, ↓) | Pan-Indic Phonetic CER (%, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Studio Human Audio (Topline) | — | 100.0% | — | — | — | — | 2.58% [1.84, 3.31] | 14.63% |
| Base `mms-tts-hin` (Native Tokenizer) | 138.8 MB | 59.4% | 392.6 ms | 6.39 [6.20, 6.61] | 861.6 | 3.54s | 24.65% [22.42, 26.87] | 57.99% |
| Base `mms-tts-hin` + `Ninada` G2P | 138.8 MB | 100.0% | 738.2 ms | 5.47 [5.38, 5.55] | 714.4 | 1.26s | 24.89% [22.57, 27.20] | 32.60% |
| Meta MMS 5-Specialist Fleet (`5x` Models) | 694.0 MB | 100.0% | 741.7 ms | 5.16 [5.07, 5.26] | 697.1 | 1.24s | 24.65% [22.42, 26.87] | 26.58% |
| **`Ninada` Tier-1A (GCP L4 FP32)** | 140.0 MB | 100.0% | 127.8 ms | 4.33 [4.26, 4.42] | 539.1 | 0.82s | 12.77% [11.00, 14.68] | 20.35% |
| **`Ninada` Tier-1B (Streaming ONNX)** | 56.6 MB | 100.0% | 178.6 ms | 4.33 [4.26, 4.41] | 536.1 | 0.82s | 13.32% [11.53, 15.24] | 20.50% |
| `Ninada` Tier-2 (Zero-Dep Klatt) | 0.0 MB | 100.0% | 57.8 ms | 10.61 [10.41, 10.83] | 640.9 | 2.32s | 85.95% [84.23, 87.63] | 89.44% |

## 3. Per-Language MCD (dB, ↓) Breakdown Across 6 Indic Languages

| Language | Test `N` | `mms-tts-hin` (Raw) | `mms-tts-hin` + G2P | MMS 5-Specialist Fleet (`694 MB`) | **`Ninada` Tier-1A (FP32)** | **`Ninada` Tier-1B (Browser ONNX)** |
|---|---:|---:|---:|---:|---:|---:|
| Sanskrit (`sa`) | 65 | 5.33 | 5.33 | 5.33 | **4.52** | **4.51** |
| Hindi (`hi`) | 18 | 5.61 | 5.61 | 5.61 | **4.08** | **4.09** |
| Kannada (`kn`) | 18 | 8.55 | 5.80 | 5.29 | **4.14** | **4.14** |
| Tamil (`ta`) | 18 | 6.94 | 5.49 | 4.70 | **4.07** | **4.06** |
| Telugu (`te`) | 18 | 7.80 | 5.60 | 5.15 | **4.39** | **4.39** |
| Malayalam (`ml`) | 18 | 6.92 | 5.33 | 4.44 | **4.29** | **4.28** |

## 4. Browser ONNX Compression & Numerical Parity
- **FP32 ONNX Size**: `112.61 MB` -> **Browser Single-File ONNX (`model_quantized.onnx`)**: `56.62 MB` (`2.0x` smaller than FP32, `12.6x` smaller than the `694.0 MB` 5-Specialist MMS Fleet, zero external `.data` shards, zero unsupported `ConvInteger` ops in `onnxruntime-web` WASM SIMD).
- **PyTorch FP32 <-> Browser ONNX Waveform Parity**: Mean SNR = **`41.09 dB`**, Mean Pearson `r` = **`0.999924`**.
