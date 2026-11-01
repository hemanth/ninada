# ninada

Stream 24,000 Hz studio speech for Classical Sanskrit and five Indian languages (`sa`, `kn`, `hi`, `ta`, `te`, `ml`) in the browser, Node.js, and Python from a single `87.5 MB` ONNX model.

```bash
npm install ninada onnxruntime-web
```

```bash
pip install ninada
```

## Quick start

```js
import ninada from 'ninada';

const voice = await ninada('kannada', {
  model: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/ninada_24k_q8f16.onnx',
  voiceUrl: 'https://huggingface.co/gnumanth/ninada/resolve/main/voices/kn.bin',
  cacheModels: true,
});

for await (const chunk of voice.stream('ನಮಸ್ಕಾರ, ಇಂದು ಬೆಂಗಳೂರಿನಲ್ಲಿ ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.')) {
  playPcm24k(chunk.pcm, chunk.sampleRate); // Float32Array @ 24,000 Hz
}

const wav = await voice.speak('ಕನ್ನಡವೇ ಸತ್ಯ ಕನ್ನಡವೇ ನಿತ್ಯ', { lang: 'kn' });
await voice.dispose();
```

`voice.stream()` yields `24,000 Hz` PCM clauses as they synthesize with `0` `AudioWorklet` underruns and `CacheStorage` persistence (`0` refetches on reload). `voice.speak()` returns a complete `24 kHz` RIFF WAV buffer.

## Python

```python
from ninada import Ninada

tts = Ninada(lang="sa")
for chunk in tts.stream("प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् । यथार्थज्ञानं प्रमाणम् ॥"):
    play(chunk["pcm"], chunk["sample_rate"])  # float32 @ 24,000 Hz (~0.09x CPU RTF)

wav_bytes = tts.speak("सत्यमेव जयते नानृतम्", lang="sa")
open("satyam.wav", "wb").write(wav_bytes)
```

`Ninada` auto-downloads the unified `87.5 MB` Q8F16 ONNX model (`ninada_24k_q8f16.onnx`) and `510 KB` per-language studio voice banks (`voices/{sa,hi,kn,ta,te,ml}.bin`), switching languages without recompiling the ONNX session.

## CLI

```bash
npx ninada "ಕನ್ನಡವೇ ಸತ್ಯ ಕನ್ನಡವೇ ನಿತ್ಯ" -l kn -o kannada.wav
ninada "सत्यमेव जयते नानृतम्" -l sa -o satyam.wav
```

Runs `24 kHz` neural synthesis when ONNX weights are available and falls back automatically to a `0.0 MB` zero-dependency Cascade Klatt-Akṣara synthesizer offline.

## Pan-Indic IPA G2P & 24 kHz Harmonic Vocoder

```js
import { textToIpa, textToIds } from 'ninada';

const ipa = textToIpa('धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।', 'sa');
// => "dʰˈəɾməkʂeːtɾeː kʊɾˈʊkʂeːtɾeː səməʋˈeːtɑː jʊjˈʊtsəʋəhᵊ ;"
const { aksharas } = textToIds('प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् ।', 'sa');
```

`textToIpa()` preserves Classical/Vedic Sanskrit vocalic `ṛ`/`ṝ`, homorganic nasal sandhi, visarga breathing, *pūrvarūpa avagraha* (`ऽ`), Hindi medial/final schwa deletion, Tamil intervocalic stop lenition, Malayalam atomic chillus (`ൽ, ർ, ൺ, ൻ, ൾ`), and dedicated phoneme slot `79` (`ɭ`) for Vedic and Dravidian retroflex laterals (`ळ / ಳ / ள / ళ / ള`) feeding a `24 kHz` `StyleTTS2 + PL-BERT + Harmonic-Source iSTFTNet` Fourier vocoder.

## Benchmarks (`N = 240` Held-Out 24 kHz Studio Utterances, `B = 1,000` Bootstrap 95% CIs)

| System | Rate | Footprint (↓) | Coverage (↑) | TTFA p50 (↓) | MCD 0–8 kHz (dB, ↓) | 8–12 kHz HF Err (dB, ↓) | Dur MAE (↓) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base `facebook/mms-tts-hin` | 16 kHz | 138.8 MB | 35.0% | 0.3 ms* | `7.98 [7.70, 8.24]` | 66.39 dB | 3.26s |
| Meta MMS 5-Specialist Fleet (`5x` VITS) | 16 kHz | 694.0 MB | 100.0% | 425.0 ms | `5.10 [5.04, 5.16]` | 65.94 dB | 1.27s |
| `ninada` v3.0 Untuned Prior (`alpha=0`) | 24 kHz | 312.7 MB | 100.0% | 237.3 ms | `5.26 [5.21, 5.31]` | 17.88 dB | 1.70s |
| **`ninada` v3.0 Studio FP32** | **24 kHz** | 312.7 MB | **100.0%** | **174.7 ms** | **`4.80 [4.76, 4.85]`** | **9.45 dB** | **0.58s** |
| **`ninada` v3.0 Unified Q8F16 ONNX** | **24 kHz** | **88.0 MB** | **100.0%** | **361.0 ms** | **`5.05 [5.00, 5.10]`** | **10.87 dB** | **0.57s** |
| `ninada` Tier-2 Klatt Fallback | 24 kHz | **0.0 MB** | **100.0%** | **37.9 ms** | `10.38 [10.26, 10.50]` | 41.94 dB | 0.98s |

Per-language held-out `MCD 0–8 kHz (dB, ↓)`: Sanskrit **`4.99`** (`5.21` ONNX vs `5.33` MMS), Hindi **`4.70`** (`5.04` ONNX), Kannada **`4.93`** (`5.18` ONNX), Tamil **`4.43`** (`4.78` ONNX vs `4.88` MMS), Telugu **`4.90`** (`5.04` ONNX vs `5.62` MMS), Malayalam **`4.88`** (`5.06` ONNX). Full bootstrap CIs and parity telemetry in [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md) and [`bench/results.json`](./bench/results.json).

## Reproduce & Verify

```bash
PYTHONPATH=python/src:. python training/train_gcp_24k.py --work-dir data/v3 --out-dir artifacts/ninada_v3 --epochs 2
PYTHONPATH=python/src:. python bench/run_bench.py --artifact-dir artifacts/ninada_v3
pytest -q && node verify-browser.mjs
```

Runs 24 kHz Viterbi studio alignment + Cloud GPU adaptation, evaluates all `240` held-out studio clips, and verifies `0` `AudioWorklet` underruns and `0` cached network refetches in headless Chromium.

## Links

- [Live Browser Voice Studio](https://hemanth.github.io/ninada/)
- [Hugging Face Model (`gnumanth/ninada`)](https://huggingface.co/gnumanth/ninada)
- [Hugging Face Dataset (`gnumanth/ninada-sanskrit-indic`)](https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic)

## License

MIT © [Hemanth.HM](https://h3manth.com)
