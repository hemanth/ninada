# `Ninada` v3.0 (`24,000 Hz` Studio Neural TTS) — Scientific Multi-Benchmark Report

## 1. Experimental Setup & Leakage-Free `24 kHz` Anechoic Studio Corpus
- **Held-Out Test Split**: `240` single-speaker anechoic studio Male utterances (`24,000 Hz` mono PCM, downsampled cleanly from `48 kHz` studio masters in **`ai4bharat/Rasa`**) across **6 Indian languages** (`sa` [40], `hi` [40], `kn` [40], `ta` [40], `te` [40], `ml` [40]), strictly disjoint from the `1920` training and `240` validation utterances.
- **Architecture**: `24,000 Hz` `StyleTTS2 + PL-BERT + Explicit F0/Energy ProsodyPredictor + Harmonic-Source iSTFTNet` Fourier vocoder (`HOP_LENGTH = 600` [`40 Hz`], `F0_HOP_LENGTH = 300` [`80 Hz`], dedicated retroflex lateral phoneme slot `79 = 'ɭ'`, and length-indexed `[510, 1, 256]` per-language studio voice banks).
- **Bootstrap Confidence Intervals**: `B = 1,000` non-parametric bootstrap resamples (`95% CI`).
- **Fair Multi-Rate Comparison**: Kubichek (1993) DTW `MCD (dB)` is evaluated on the shared `0–8 kHz` vocal band so `16 kHz` VITS baselines are not penalized for Nyquist roll-off, while **Studio High-Frequency Error (`8–12 kHz` dB, ↓)** separately quantifies full-band `24 kHz` sibilance and studio air fidelity.

## 2. Overall 6-Language Studio Leaderboard (`N = 240` Held-Out Utterances)

| System | Sample Rate | Active Footprint (↓) | Coverage (↑) | TTFA p50 (ms, ↓) | MCD 0–8 kHz (dB, ↓) | Studio 8–12 kHz Err (dB, ↓) | Log-F0 RMSE (c, ↓) | Dur MAE (s, ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base `mms-tts-hin` (Single VITS) | 16 kHz | 138.8 MB | 35.0% | 1.1 ms | 7.97 [7.69, 8.23] | 66.35 dB | 1016.3 | 3.26s |
| Meta MMS 5-Specialist Fleet (`5x` VITS) | 16 kHz | 694.0 MB | 100.0% | 184.9 ms | 5.09 [5.03, 5.16] | 65.90 dB | 675.0 | 1.27s |
| `Ninada` v3.0 Untuned Prior (`alpha = 0.0`) | 24 kHz | 312.7 MB | 100.0% | 84.6 ms | 5.24 [5.18, 5.29] | 17.69 dB | 709.4 | 1.77s |
| **`Ninada` v3.0 Studio FP32 (Cloud GPU)** | 24 kHz | 312.7 MB | 100.0% | 33.3 ms | 4.61 [4.56, 4.65] | 10.10 dB | 689.9 | 1.18s |
| **`Ninada` v3.0 Unified Browser ONNX** | 24 kHz | 79.4 MB | 100.0% | 1215.7 ms | 4.61 [4.57, 4.65] | 10.11 dB | 673.8 | 1.17s |
| `Ninada` Tier-2 (Zero-Dep Klatt) | 24 kHz | 0.0 MB | 100.0% | 104.9 ms | 10.38 [10.26, 10.50] | 41.94 dB | 613.8 | 0.98s |

## 3. Per-Language MCD (dB, ↓) Across All 6 Indian Languages

| Language | Test `N` | `mms-tts-hin` (`138.8 MB`) | MMS 5-Fleet (`694.0 MB`) | Untuned Prior (`alpha=0`) | **`Ninada` v3.0 FP32** | **`Ninada` v3.0 Unified ONNX (`79.4 MB`)** |
|---|---:|---:|---:|---:|---:|---:|
| Sanskrit (`sa`) | 40 | 5.31 | 5.31 | 5.32 | **4.88** | **4.88** |
| Hindi (`hi`) | 40 | 4.84 | 4.84 | 5.06 | **4.41** | **4.42** |
| Kannada (`kn`) | 40 | 9.01 | 5.07 | 5.34 | **4.59** | **4.59** |
| Tamil (`ta`) | 40 | 9.71 | 4.86 | 4.78 | **4.34** | **4.35** |
| Telugu (`te`) | 40 | 9.35 | 5.61 | 5.39 | **4.68** | **4.69** |
| Malayalam (`ml`) | 40 | 9.61 | 4.87 | 5.54 | **4.72** | **4.73** |

## 4. Unified `< 90 MB` Browser ONNX Compression & Numerical Fidelity
- **Unified 6-Language Browser Footprint**: **`78.91 MB`** (`ninada_24k_q8f16.onnx`) + **`0.50 MB`** per language voice bank (`voices/{sa,hi,kn,ta,te,ml,female}.bin`), replacing `694.0 MB` of separate `MMS-TTS` models (`8.7x` smaller) with `0` ONNX session recompilations on language switch.
- **PyTorch FP32 <-> Quantized ONNX Acoustic Parity**: DTW Mel-Cepstral Deviation = **`0.96 dB`**, Duration Exact Match / Correlation `r` = **`0.9967`**, Native ORT CPU Real-Time Factor = **`0.242x`** (`4.1x` faster than real-time).
