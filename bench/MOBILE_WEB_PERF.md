# Ninada Mobile Web Streaming Performance Ledger

Hardware host: Apple Silicon macOS (10 cores), Playwright Chromium mobile device emulation with CDP CPU throttling.
Benchmark harness: `ENFORCE_GATES=1 node bench/bench-mobile-web.mjs`.

## Profiles Measured
1. **`android_coi_3x`**: Pixel 7 (`412x915`), `3x` CPU throttling, COOP/COEP Service Worker (`coi-sw.js`) multi-threaded WASM SIMD (`4T`) — models Android Chrome and iOS 15.2+ Safari with Cross-Origin Isolation.
2. **`ios_1thread_2x`**: iPhone 14 (`390x844`), `2x` CPU throttling, `?coi=0` with Service Workers blocked (`1` WASM thread, no `SharedArrayBuffer`) — models worst-case restricted in-app mobile WebViews without COOP/COEP.

---

## Baseline (`8a0d561`, before optimizations)

| Profile & Scenario | Tap-to-Audible (ms) | Tap-to-First-Push (ms) | Worker TTFA (ms) | RTF | Underruns | Max rAF Gap (ms) |
|---|---:|---:|---:|---:|---:|---:|
| **Android COI (`3x` CPU, `4T`) — Cold FCP / Cold Ready Unlock** | `124` (FCP) | — | `2225.0` (boot) | — | — | — |
| **Android COI (`3x` CPU, `4T`) — Warm Reload Unlock** | `47` (FCP) | — | `814.0` (boot) | — | — | — |
| **Android COI (`3x` CPU, `4T`) — Immediate Tap After Unlock (SA)** | `2960.6` | `2944.0` | `1105.3` | `0.379x` | `0` | `16.7` |
| **Android COI (`3x` CPU, `4T`) — Warm Short (SA)** | `1172.4` | `1137.4` | `1128.3` | `0.384x` | `0` | `16.7` |
| **Android COI (`3x` CPU, `4T`) — Warm 3-Sentence (KN)** | `1669.2` | `1646.1` | `1632.9` | `0.395x` | `0` | `16.7` |
| **iOS 1-Thread (`2x` CPU, `1T`) — Cold FCP / Cold Ready Unlock** | `296` (FCP) | — | `1356.0` (boot) | — | — | — |
| **iOS 1-Thread (`2x` CPU, `1T`) — Immediate Tap After Unlock (SA)** | `9816.5` | `7758.4` | `2919.7` | `0.997x` | `0` | `16.8` |
| **iOS 1-Thread (`2x` CPU, `1T`) — Warm Short (SA)** | `5096.6` | `2992.5` | `2985.7` | `1.020x` | `0` | `16.8` |
| **iOS 1-Thread (`2x` CPU, `1T`) — Warm 3-Sentence (SA)** | `12161.3` | `3921.5` | `3912.8` | `1.091x` | `0` | `16.8` |

---

## Root Causes Identified via Profiling

1. **97.9% of VITS inference time resides in `/waveform_decoder` (HiFi-GAN `28 GMAC/s`)**, whereas `/text_encoder + /duration_predictor + /flow` consume **only 2.1%** (`~54ms` for a 4.4s sentence). Running `*.onnx` monolithically forced the worker to upsample 3.0s–5.5s of audio before emitting sample 0 (`1,128ms–3,913ms` worker TTFA).
2. **Over-conservative `adaptedWatermark` in `docs/index.html`** (`clauseDurSec + extraCushionSec`, without dividing by `effRtf` or subtracting zero-CPU pauses) forced `AudioWorklet` to hold back Clause 0 until subsequent clauses finished whenever `estRtf > 0.78`, adding `+2,104ms` to `+8,240ms` of silence wait on single-threaded mobile (`12,161.3ms` Tap-to-Audible).
3. **Unsolicited full-preset `runSynthesis(false)` immediately upon boot** locked the Web Worker in a multi-second synchronous WASM call right after unlocking `#synthBtn`, delaying an immediate mobile user tap by `+1,840ms` (Android COI) to `+7,668ms` (iOS 1-thread).
4. **Sequential `await ensureAudioContext()` (`audioWorklet.addModule`) inside user tap** + render-blocking Google Fonts CSS + `latencyHint: 'playback'` + fixed `280ms`/`650ms` initial watermark added avoidable FCP, pre-inference, and pre-playback latency on mobile.
5. **Mobile memory & main-thread overhead**: `MAX_WARM_SESSIONS = 3` retained 3 full WASM heaps on mobile, and O(N^2) waveform rescanning ran synchronously on every chunk.

---

## Optimization Log

### Step 1: In-Memory Zero-Dep VITS ONNX Protobuf Graph Splitter & Sub-Clause Latent-Window HiFi-GAN Streaming
- Implemented `splitStudioVitsOnnxBuffer` and `createSplitStudioVitsSession` in `js/src/index.js` (`< 8ms` zero-dependency `Uint8Array` protobuf split at `/Mul_8_output_0` `[1, 192, T_frames]` -> `/waveform_decoder/Tanh_output_0`, requiring **zero extra network downloads** and working transparently on existing cached 28 MB `.onnx` files across all 6 languages).
- Added `haloLatent = 10` frames (`106.7ms`, covering HiFi-GAN's exact `9.4`-frame receptive field) + `256`-sample (`10.67ms`) raised-cosine crossfade using the already-computed right halo (`55.5 dB` SNR parity with monolithic synthesis, `< 0.001` sample error, zero boundary clicks).
- Added `MessageChannel` `yieldMacrotask()` between vocoder windows so user re-taps / language switches cancel an active WASM stream within `< 200ms` instead of blocking for `3–4s`, and capped `MAX_WARM_SESSIONS = 1` on mobile / low-memory devices in `docs/synth-worker.mjs`.
- **Measured Impact**: Cut `WorkerTTFA` from `1,128–1,633ms` -> **`215–249ms`** (`6.5x` faster) on Android COI and from `2,985–3,913ms` -> **`491–507ms`** (`7.7x` faster) on iOS 1-thread.

### Step 2: Non-Blocking Mobile Critical Path, Parallel Dispatch & Immediate Worklet Start Notification
- Made Google Fonts stylesheet non-render-blocking (`media="print" onload="this.media='all'"`) and added `<link rel="modulepreload">` for `ninada.mjs`, `synth-worker.mjs`, and `ort/ort.wasm.min.mjs` (**cut iOS Cold FCP from `296ms` -> `60ms`, `4.9x` faster**).
- Switched `AudioContext` to `{ latencyHint: 'interactive' }`, pre-compiled `pcm-stream-worklet.js` in parallel with Web Worker ONNX initialization, added `pointerdown`/`touchstart` gesture warm-up, and dispatched `{ type: 'stream' }` to `synthWorker` at `t = 0ms` in parallel with `audioCtx.resume()`.
- Prioritized `pushLivePcmChunk` ahead of `requestAnimationFrame`-coalesced waveform updates and `setTimeout`-deferred DOM prosody inspection (`16.7ms` 60fps main-thread frame pacing), and emitted immediate `progress` from `pcm-stream-worklet.js` on `'buffering' -> 'playing'` transition.

### Step 3: Exact Closed-Form Queuing Watermark, 1-Thread Window Schedule Alignment & Boot Idle Grace
- Added a `180ms` post-unlock idle grace before unsolicited preview synthesis (`runSynthesis(false)`) so an immediate user tap starts with `0ms` worker queue contention.
- Derived the exact discrete queuing-theory watermark normalized by `effRtf` (`((effRtf - 0.98) * neuralTotalSec + lastChunkSec + marginSec - interPauseSec) / effRtf`), aligned 1-thread window sizes (`[28, 120, 98, ...]` with final-clause `~50`-frame tail tapering and `124`-frame middle clauses), and added natural `220ms` inter-sentence breath pauses.

---

## Final Verified Mobile Web Performance (vs. Baseline)

| Profile & Scenario | Baseline | Final | Speedup | Final Worker TTFA | Final RTF | Underruns |
|---|---:|---:|---:|---:|---:|---:|
| **Android COI (`3x` CPU, `4T`) — Cold FCP** | `124 ms` | **`114 ms`** | `1.09x` | — | — | — |
| **Android COI (`3x` CPU, `4T`) — Cold Ready Unlock** | `2,225 ms` | **`1,983 ms`** | `1.12x` | — | — | — |
| **Android COI (`3x` CPU, `4T`) — Warm Reload Unlock** | `814 ms` | **`738 ms`** | `1.10x` | — | — | — |
| **Android COI (`3x` CPU, `4T`) — Immediate Tap Audible (SA)** | `2,960.6 ms` | **`275.4 ms`** | **`10.75x`** | `263.3 ms` (`4.20x`) | `0.451x` | **`0`** |
| **Android COI (`3x` CPU, `4T`) — Warm Short Audible (SA)** | `1,172.4 ms` | **`255.7 ms`** | **`4.58x`** | `248.5 ms` (`4.54x`) | `0.444x` | **`0`** |
| **Android COI (`3x` CPU, `4T`) — Warm 3-Sentence Audible (KN)** | `1,669.2 ms` | **`278.8 ms`** | **`5.99x`** | `268.3 ms` (`6.09x`) | `0.494x` | **`0`** |
| **iOS 1-Thread (`2x` CPU, `1T`) — Cold FCP** | `296 ms` | **`60 ms`** | **`4.93x`** | — | — | — |
| **iOS 1-Thread (`2x` CPU, `1T`) — Cold Ready Unlock** | `1,356 ms` | **`1,071 ms`** | **`1.27x`** | — | — | — |
| **iOS 1-Thread (`2x` CPU, `1T`) — Immediate Tap Audible (SA)** | `9,816.5 ms` | **`2,148.4 ms`** | **`4.57x`** | `535.1 ms` (`5.46x`) | `1.142x` | **`0`** |
| **iOS 1-Thread (`2x` CPU, `1T`) — Warm Short Audible (SA)** | `5,096.6 ms` | **`2,138.4 ms`** | **`2.38x`** | `521.7 ms` (`5.72x`) | `1.149x` | **`0`** |
| **iOS 1-Thread (`2x` CPU, `1T`) — Warm 3-Sentence Audible (SA)** | `12,161.3 ms` | **`4,201.4 ms`** | **`2.89x`** | `550.4 ms` (`7.11x`) | `1.196x` | **`0`** |

> Note: Real iOS Safari `15.2+` and Android Chrome activate `coi-sw.js` (`window.crossOriginIsolated === true`), running the multi-threaded `4T` WASM path (`255–279 ms` Tap-to-Audible). The `ios_1thread_2x` (`?coi=0`) profile guards worst-case restricted in-app WebViews where `SharedArrayBuffer` is blocked.
