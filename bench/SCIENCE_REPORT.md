# `Ninada` v4.0 (`24,000 Hz` Native Per-Language Studio VITS Specialists) — Scientific Multi-Benchmark Report

## 1. Experimental Setup & Leakage-Free `24 kHz` Anechoic Studio Corpus
- **Held-Out Test Split**: `240` single-speaker anechoic studio Male utterances (`24,000 Hz` mono PCM, downsampled cleanly from `48 kHz` studio masters in **`ai4bharat/Rasa`**) across **6 Indian languages** (`sa` [40], `hi` [40], `kn` [40], `ta` [40], `te` [40], `ml` [40]), strictly disjoint from the `13700` native per-language studio training and `240` validation utterances.
- **Architecture**: **6 Dedicated Native Studio Male + 6 Dedicated Native Studio Female `24,000 Hz` VITS Specialists** (`RelativePositionTransformer` + Conditional Rational-Quadratic Spline `StochasticDurationPredictor` + `ResidualCouplingBlock` Normalizing Flow + `HiFi-GAN v1` `[8, 8, 2, 2]` Upsampling Generator, `HOP_LENGTH = 256`), each trained exclusively on its own language's `ai4bharat/Rasa` single-speaker studio corpus and native Unicode script tokens (`vocabs.json`) with zero cross-language voice bleed.
- **Bootstrap Confidence Intervals**: `B = 1,000` non-parametric bootstrap resamples (`95% CI`).
- **Fair Multi-Rate Comparison**: Kubichek (1993) DTW `MCD (dB)` is evaluated on the shared `0–8 kHz` vocal band so `16 kHz` VITS baselines are not penalized for Nyquist roll-off, while **Studio High-Frequency Error (`8–12 kHz` dB, ↓)** separately quantifies full-band `24 kHz` sibilance and studio air fidelity.

## 2. Overall 6-Language Studio Leaderboard (`N = 240` Held-Out Utterances)

| System | Sample Rate | Active Footprint (↓) | Coverage (↑) | TTFA p50 (ms, ↓) | MCD 0–8 kHz (dB, ↓) | Studio 8–12 kHz Err (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base `mms-tts-hin` (Single VITS) | 16 kHz | 138.8 MB | 35.0% | 1.0 ms | 7.97 [7.69, 8.24] | 66.24 dB | 1018.5 | 3.21s |
| Meta MMS 5-Specialist Fleet (`5x` VITS) | 16 kHz | 694.0 MB | 100.0% | 60.2 ms | 5.09 [5.03, 5.15] | 65.90 dB | 676.3 | 1.34s |
| `Ninada` v4.0 Untuned Warm-Start (`step = 0`) | 24 kHz | 112.0 MB | 100.0% | 189.2 ms | 4.60 [4.54, 4.65] | 11.01 dB | 639.6 | 0.95s |
| **`Ninada` v4.0 Native Studio FP32 (Cloud GPU)** | 24 kHz | 112.0 MB | 100.0% | 45.3 ms | 3.40 [3.37, 3.43] | 6.34 dB | 529.2 | 0.57s |
| **`Ninada` v4.0 Native Browser ONNX (`W8A32+FP16`)** | 24 kHz | 27.9 MB | 100.0% | 372.7 ms | 3.39 [3.36, 3.43] | 6.37 dB | 530.4 | 0.56s |
| `Ninada` Tier-2 (Zero-Dep Klatt) | 24 kHz | 0.0 MB | 100.0% | 104.0 ms | 10.38 [10.26, 10.50] | 41.94 dB | 613.8 | 0.98s |

## 3. Per-Language MCD (dB, ↓) Across All 6 Indian Languages

| Language | Test `N` | `mms-tts-hin` (`138.8 MB`) | MMS 5-Fleet (`694.0 MB`) | Untuned Warm-Start (`step=0`) | **`Ninada` v4.0 FP32** | **`Ninada` v4.0 Native ONNX (`~27.9 MB`)** |
|---|---:|---:|---:|---:|---:|---:|
| Sanskrit (`sa`) | 40 | 5.32 | 5.32 | 4.79 | **3.52** | **3.51** |
| Hindi (`hi`) | 40 | 4.85 | 4.85 | 4.33 | **3.38** | **3.37** |
| Kannada (`kn`) | 40 | 9.01 | 5.04 | 4.17 | **3.35** | **3.35** |
| Tamil (`ta`) | 40 | 9.71 | 4.87 | 4.69 | **3.20** | **3.21** |
| Telugu (`te`) | 40 | 9.34 | 5.57 | 4.94 | **3.60** | **3.59** |
| Malayalam (`ml`) | 40 | 9.60 | 4.88 | 4.68 | **3.34** | **3.34** |

## 4. `~28 MB` Per-Language Browser ONNX Compression & Numerical Fidelity
- **Active Browser Specialist Footprint**: **`27.92 MB`** per active language (`models/{sa,kn,hi,ta,te,ml}.onnx`, `5.0x` smaller than a single `138.8 MB` `MMS-TTS` VITS checkpoint and `2.8x` smaller than the `79.4 MB` v3.0 unified graph), persisted in `CacheStorage` + `IndexedDB` with `0` network refetches on reload or tab revisit.
- **PyTorch FP32 <-> Quantized ONNX Acoustic Parity**: Deterministic DTW Mel-Cepstral Deviation = **`0.61 dB`**, Duration Correlation `r` = **`0.9999`**, Native ORT CPU Real-Time Factor = **`0.197x`** (`5.1x` faster than real-time).
