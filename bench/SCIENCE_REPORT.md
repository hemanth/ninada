# `Ninada` v5.0 (`24,000 Hz` Native Per-Language Studio VITS Specialists) — 12-Language Scientific Multi-Benchmark Report

## 1. Experimental Setup & Leakage-Free `24 kHz` Anechoic Studio Corpus
- **Held-Out Test Split**: `480` single-speaker anechoic studio Male utterances (`24,000 Hz` mono PCM, downsampled cleanly from `48 kHz` studio masters in **`ai4bharat/Rasa`** and **`SPRINGLab/IndicTTS-English`**) across **12 Languages** (`sa` [40], `hi` [40], `kn` [40], `ta` [40], `te` [40], `ml` [40], `en` [40], `mr` [40], `bn` [40], `gu` [40], `pa` [40], `or` [40]), strictly disjoint from the `26700` native per-language studio training and `480` validation utterances.
- **Architecture**: **12 Dedicated Native Studio Male + 12 Dedicated Native Studio Female `24,000 Hz` VITS Specialists (`24` models total)** (`RelativePositionTransformer` + Conditional Rational-Quadratic Spline `StochasticDurationPredictor` + `ResidualCouplingBlock` Normalizing Flow + `HiFi-GAN v1` `[8, 8, 2, 2]` Upsampling Generator, `HOP_LENGTH = 256`), each trained exclusively on its own language's single-speaker studio corpus and native Unicode script tokens (`vocabs.json`) with zero cross-language voice bleed.
- **Bootstrap Confidence Intervals**: `B = 1,000` non-parametric bootstrap resamples (`95% CI`).
- **Fair Multi-Rate Comparison**: Kubichek (1993) DTW `MCD (dB)` is evaluated on the shared `0–8 kHz` vocal band so `16 kHz` VITS baselines are not penalized for Nyquist roll-off, while **Studio High-Frequency Error (`8–12 kHz` dB, ↓)** separately quantifies full-band `24 kHz` sibilance and studio air fidelity.

## 2. Overall 12-Language Studio Leaderboard (`N = 480` Held-Out Utterances)

| System | Sample Rate | Active Footprint (↓) | Coverage (↑) | TTFA p50 (ms, ↓) | MCD 0–8 kHz (dB, ↓) | Studio 8–12 kHz Err (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base `mms-tts-hin` (Single VITS) | 16 kHz | 138.8 MB | 29.4% | 0.9 ms | 7.98 [7.81, 8.16] | 65.16 dB | 1048.3 | 3.49s |
| Meta MMS 11-Specialist Fleet (`11x` VITS) | 16 kHz | 1526.8 MB | 100.0% | 204.2 ms | 5.17 [5.12, 5.22] | 64.97 dB | 679.4 | 1.04s |
| `Ninada` v5.0 Untuned Warm-Start (`step = 0`) | 24 kHz | 112.0 MB | 100.0% | 196.1 ms | 4.58 [4.54, 4.62] | 10.93 dB | 643.8 | 0.93s |
| **`Ninada` v5.0 Native Studio FP32 (Cloud GPU)** | 24 kHz | 112.0 MB | 100.0% | 45.0 ms | 3.27 [3.24, 3.30] | 6.07 dB | 505.6 | 0.60s |
| **`Ninada` v5.0 Native Browser ONNX (`W8A32+FP16`)** | 24 kHz | 27.9 MB | 100.0% | 382.4 ms | 3.27 [3.24, 3.29] | 6.03 dB | 506.3 | 0.60s |
| `Ninada` Tier-2 (Zero-Dep Klatt) | 24 kHz | 0.0 MB | 100.0% | 112.8 ms | 10.53 [10.43, 10.64] | 42.38 dB | 624.1 | 0.88s |

## 3. Per-Language MCD (dB, ↓) Across All 12 Languages

| Language | Test `N` | `mms-tts-hin` (`138.8 MB`) | MMS 11-Fleet (`1526.8 MB`) | Untuned Warm-Start (`step=0`) | **`Ninada` v5.0 FP32** | **`Ninada` v5.0 Native ONNX (`~27.9 MB`)** |
|---|---:|---:|---:|---:|---:|---:|
| Sanskrit (`sa`) | 40 | 5.32 | 5.32 | 4.79 | **3.52** | **3.51** |
| Hindi (`hi`) | 40 | 4.85 | 4.85 | 4.33 | **3.38** | **3.36** |
| Kannada (`kn`) | 40 | 9.01 | 5.04 | 4.17 | **3.35** | **3.36** |
| Tamil (`ta`) | 40 | 9.71 | 4.87 | 4.69 | **3.20** | **3.21** |
| Telugu (`te`) | 40 | 9.34 | 5.57 | 4.94 | **3.60** | **3.59** |
| Malayalam (`ml`) | 40 | 9.60 | 4.88 | 4.68 | **3.34** | **3.35** |
| Indian English (`en`) | 40 | 8.47 | 5.94 | 4.34 | **3.14** | **3.13** |
| Marathi (`mr`) | 40 | 4.92 | 4.62 | 4.55 | **3.10** | **3.10** |
| Bengali (`bn`) | 40 | 8.73 | 4.68 | 4.08 | **3.28** | **3.27** |
| Gujarati (`gu`) | 40 | 8.99 | 5.56 | 4.57 | **3.33** | **3.30** |
| Punjabi (`pa`) | 40 | 8.32 | 5.23 | 4.76 | **2.93** | **2.89** |
| Odia (`or`) | 40 | 8.44 | 5.49 | 5.07 | **3.09** | **3.11** |

## 4. `~28 MB` Per-Language Browser ONNX Compression & Mobile Split-Graph Streaming Fidelity
- **Active Browser Specialist Footprint**: **`27.92 MB`** per active language (`onnx/{sa,hi,kn,ta,te,ml,en,mr,bn,gu,pa,or}.onnx` + `*_female.onnx`, `5.0x` smaller than a single `138.8 MB` `MMS-TTS` VITS checkpoint and `54.7x` smaller than the `1526.8 MB` MMS 11-checkpoint fleet), persisted in `CacheStorage` + `IndexedDB` with `0` network refetches on reload or tab revisit.
- **PyTorch FP32 <-> Quantized ONNX Acoustic Parity**: Deterministic DTW Mel-Cepstral Deviation = **`0.58 dB`**, Duration Correlation `r` = **`0.9999`**, Native ORT CPU Real-Time Factor = **`0.202x`** (`5.0x` faster than real-time).
