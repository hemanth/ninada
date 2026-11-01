# `Ninada` v3.0 (`24,000 Hz` Studio Neural TTS) — Scientific Multi-Benchmark Report

## 1. Experimental Setup & Leakage-Free `24 kHz` Anechoic Studio Corpus
- **Held-Out Test Split**: `240` single-speaker anechoic studio Male utterances (`24,000 Hz` mono PCM, downsampled cleanly from `48 kHz` studio masters in **`ai4bharat/Rasa`**) across **6 Indian languages** (`sa` [40], `hi` [40], `kn` [40], `ta` [40], `te` [40], `ml` [40]), strictly disjoint from the `1920` training and `240` validation utterances.
- **Architecture**: `24,000 Hz` `StyleTTS2 + PL-BERT + Explicit F0/Energy ProsodyPredictor + Harmonic-Source iSTFTNet` Fourier vocoder (`HOP_LENGTH = 600` [`40 Hz`], `F0_HOP_LENGTH = 300` [`80 Hz`], dedicated retroflex lateral phoneme slot `79 = 'ɭ'`, and length-indexed `[510, 1, 256]` per-language studio voice banks).
- **Bootstrap Confidence Intervals**: `B = 1,000` non-parametric bootstrap resamples (`95% CI`).
- **Fair Multi-Rate Comparison**: Kubichek (1993) DTW `MCD (dB)` is evaluated on the shared `0–8 kHz` vocal band so `16 kHz` VITS baselines are not penalized for Nyquist roll-off, while **Studio High-Frequency Error (`8–12 kHz` dB, ↓)** separately quantifies full-band `24 kHz` sibilance and studio air fidelity.

## 2. Overall 6-Language Studio Leaderboard (`N = 240` Held-Out Utterances)

| System | Sample Rate | Active Footprint (↓) | Coverage (↑) | TTFA p50 (ms, ↓) | MCD 0–8 kHz (dB, ↓) | Studio 8–12 kHz Err (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base `mms-tts-hin` (Single VITS) | 16 kHz | 138.8 MB | 35.0% | 0.3 ms | 7.98 [7.70, 8.24] | 66.39 dB | 1015.5 | 3.26s |
| Meta MMS 5-Specialist Fleet (`5x` VITS) | 16 kHz | 694.0 MB | 100.0% | 425.0 ms | 5.10 [5.04, 5.16] | 65.94 dB | 673.9 | 1.27s |
| `Ninada` v3.0 Untuned Prior (`alpha = 0.0`) | 24 kHz | 312.7 MB | 100.0% | 237.3 ms | 5.26 [5.21, 5.31] | 17.88 dB | 714.1 | 1.70s |
| **`Ninada` v3.0 Studio FP32 (Cloud GPU)** | 24 kHz | 312.7 MB | 100.0% | 174.7 ms | 4.80 [4.76, 4.85] | 9.45 dB | 825.5 | 0.58s |
| **`Ninada` v3.0 Unified Browser ONNX** | 24 kHz | 88.0 MB | 100.0% | 361.0 ms | 5.05 [5.00, 5.10] | 10.87 dB | 828.1 | 0.57s |
| `Ninada` Tier-2 (Zero-Dep Klatt) | 24 kHz | 0.0 MB | 100.0% | 37.9 ms | 10.38 [10.26, 10.50] | 41.94 dB | 613.8 | 0.98s |

## 3. Per-Language MCD (dB, ↓) Across All 6 Indian Languages

| Language | Test `N` | `mms-tts-hin` (`138.8 MB`) | MMS 5-Fleet (`694.0 MB`) | Untuned Prior (`alpha=0`) | **`Ninada` v3.0 FP32** | **`Ninada` v3.0 Unified ONNX (`87.4 MB`)** |
|---|---:|---:|---:|---:|---:|---:|
| Sanskrit (`sa`) | 40 | 5.33 | 5.33 | 5.36 | **4.99** | **5.21** |
| Hindi (`hi`) | 40 | 4.85 | 4.85 | 5.08 | **4.70** | **5.04** |
| Kannada (`kn`) | 40 | 9.01 | 5.06 | 5.37 | **4.93** | **5.18** |
| Tamil (`ta`) | 40 | 9.71 | 4.88 | 4.79 | **4.43** | **4.78** |
| Telugu (`te`) | 40 | 9.35 | 5.62 | 5.39 | **4.90** | **5.04** |
| Malayalam (`ml`) | 40 | 9.61 | 4.88 | 5.54 | **4.88** | **5.06** |

## 4. Unified `< 90 MB` Browser ONNX Compression & Numerical Fidelity
- **Unified 6-Language Browser Footprint**: **`87.48 MB`** (`ninada_24k_q8f16.onnx`) + **`0.50 MB`** per language voice bank (`voices/{sa,hi,kn,ta,te,ml}.bin`), replacing `694.0 MB` of separate `MMS-TTS` models (`7.9x` smaller) with `0` ONNX session recompilations on language switch.
- **PyTorch FP32 <-> Quantized ONNX Acoustic Parity**: DTW Mel-Cepstral Deviation = **`2.71 dB`**, Duration Exact Match / Correlation `r` = **`0.9687`**, Native ORT CPU Real-Time Factor = **`0.091x`** (`10.9x` faster than real-time).
