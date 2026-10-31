# `Ninada` v2.0 Scientific Multi-Benchmark & Halo-Streaming Report

## 1. Experimental Setup & Leakage-Free Anechoic Studio Corpus
- **Held-Out Test Split**: `144` anechoic studio Male utterances (`16 kHz` mono PCM) across **6 Indic languages** (`sa` [39], `hi` [21], `kn` [21], `ta` [21], `te` [21], `ml` [21]), curated from **`prathoshap/sushrota-sanskrit-asr-data`** (`recitation_corpus` + `scholar`, `studio_align_cer <= 5.0%`, synthetic `tts_aug` excluded) + **`ai4bharat/Rasa`** neutral studio Male narration (`48 kHz -> 16 kHz`), strictly disjoint (`0` shared audio hashes or normalized transcripts) from the training and validation splits.
- **Bootstrap Confidence Intervals**: `B = 1,000` non-parametric bootstrap resamples (`95% CI`).
- **Specialist Fleet Baseline Included**: Benchmarks against both single-checkpoint `facebook/mms-tts-hin` (`138.8 MB`) and the **`694.0 MB` Meta MMS-TTS 5-Specialist Fleet** (`mms-tts-{hin,kan,tam,tel,mal}`, routing each language to its dedicated checkpoint; `sa -> mms-tts-hin`).
- **Independent Round-Trip Intelligibility Oracle**: `Su-srota` IndicConformer-CTC (`115M` params, `sushrota_sanskrit_ctc_int8.onnx`), completely independent of the VITS + HiFi-GAN generator.

## 2. Overall Pan-Indic & Sanskrit Leaderboard (`N = 144` Test Utterances)

| System | Active Footprint (↓) | Coverage (↑) | TTFA p50 (ms, ↓) | MCD (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) | Sanskrit CER (%, ↓) | Pan-Indic Phonetic CER (%, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Studio Human Audio (Topline) | — | 100.0% | — | — | — | — | 1.78% [1.14, 2.46] | 18.86% |
| Base `mms-tts-hin` (Native Tokenizer) | 138.8 MB | 43.1% | 0.6 ms | 6.50 [6.30, 6.71] | 961.1 | 3.10s | 19.48% [16.51, 22.60] | 65.36% |
| Base `mms-tts-hin` + `Ninada` G2P | 138.8 MB | 100.0% | 463.4 ms | 5.49 [5.40, 5.58] | 654.8 | 0.89s | 20.27% [17.07, 23.62] | 34.44% |
| Meta MMS 5-Specialist Fleet (`5x` Models) | 694.0 MB | 100.0% | 507.5 ms | 5.09 [5.02, 5.17] | 663.7 | 1.31s | 19.48% [16.51, 22.60] | 26.73% |
| **`Ninada` v2.0 Tier-1A (GCP L4 FP32)** | 138.4 MB | 100.0% | 150.8 ms | 4.70 [4.63, 4.77] | 628.0 | 0.72s | 16.65% [13.17, 20.36] | 29.49% |
| **`Ninada` v2.0 Tier-1B (Streaming ONNX)** | 54.7 MB | 100.0% | 180.1 ms | 4.69 [4.61, 4.76] | 626.4 | 0.73s | 16.95% [13.46, 20.28] | 29.61% |
| `Ninada` Tier-2 (Zero-Dep Klatt) | 0.0 MB | 100.0% | 34.6 ms | 11.32 [11.12, 11.52] | 533.5 | 1.33s | 87.78% [85.78, 89.80] | 89.70% |

## 3. Per-Language MCD (dB, ↓) Breakdown Across 6 Indic Languages

| Language | Test `N` | `mms-tts-hin` (Raw) | `mms-tts-hin` + G2P | MMS 5-Specialist Fleet (`694 MB`) | **`Ninada` v2.0 Tier-1A (FP32)** | **`Ninada` v2.0 Tier-1B (Streaming ONNX)** |
|---|---:|---:|---:|---:|---:|---:|
| Sanskrit (`sa`) | 39 | 5.35 | 5.34 | 5.35 | **4.90** | **4.87** |
| Hindi (`hi`) | 21 | 4.90 | 4.88 | 4.90 | **4.26** | **4.25** |
| Kannada (`kn`) | 21 | 7.70 | 5.51 | 4.86 | **4.51** | **4.59** |
| Tamil (`ta`) | 21 | 7.33 | 5.30 | 4.81 | **4.59** | **4.59** |
| Telugu (`te`) | 21 | 7.29 | 6.08 | 5.59 | **5.24** | **5.19** |
| Malayalam (`ml`) | 21 | 7.42 | 5.95 | 4.81 | **4.51** | **4.47** |

## 4. Browser Streaming ONNX Compression & Seam Parity
- **Active Browser Specialist ONNX Size**: `54.7 MB` (`27.3 MB` Encoder+SDP+Flow + `27.4 MB` Halo-Windowed HiFi-GAN, `2.5x` smaller than a `138.8 MB` FP32 VITS checkpoint, `12.7x` smaller than the `694.0 MB` 5-Specialist MMS Fleet, zero `ConvInteger` or `NonZero` operators in `onnxruntime-web` WASM SIMD).
- **PyTorch FP32 <-> Streaming ONNX Waveform Parity**: Mean SNR = **`33.46 dB`**, Stream Seam SNR (`16`-frame halo + `256`-sample raised-cosine crossfade) = **`123.72 dB`**, Mean Pearson `r` = **`0.965623`**.
