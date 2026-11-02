# ninada

Stream 24,000 Hz studio speech for Classical Sanskrit and five Indian languages (`sa`, `kn`, `hi`, `ta`, `te`, `ml`) in the browser, Node.js, and Python using `27.9 MB` native per-language VITS ONNX specialists (Male & Female) with zero cross-language accent bleed.

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
  models: {
    kn: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/kn.onnx',
    sa: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/sa.onnx',
  },
  cache: true,
});

for await (const chunk of voice.stream('ನಮಸ್ಕಾರ, ಇಂದು ಬೆಂಗಳೂರಿನಲ್ಲಿ ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.')) {
  playPcm24k(chunk.pcm, chunk.sampleRate); // Float32Array @ 24,000 Hz
}

const wav = await voice.speak('ಕನ್ನಡವೇ ಸತ್ಯ ಕನ್ನಡವೇ ನಿತ್ಯ', { lang: 'kn', voice: 'female' });
await voice.dispose();
```

`voice.stream()` yields `24,000 Hz` PCM clauses as they synthesize with `0` `AudioWorklet` underruns and `CacheStorage` + `IndexedDB` persistence (`0` refetches on reload). `voice.speak()` returns a complete `24 kHz` RIFF WAV buffer.

## Python

```python
from ninada import Ninada

tts = Ninada(lang="sa")
for chunk in tts.stream("प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् । यथार्थज्ञानं प्रमाणम् ॥"):
    play(chunk["pcm"], chunk["sample_rate"])  # float32 @ 24,000 Hz (~0.197x CPU RTF)

wav_bytes = tts.speak("सत्यमेव जयते नानृतम्", lang="sa", voice="female")
open("satyam.wav", "wb").write(wav_bytes)
```

`Ninada` loads dedicated `27.9 MB` W8A32+FP16 ONNX specialists (`onnx/{sa,kn,hi,ta,te,ml}.onnx` and `onnx/{sa,kn,hi,ta,te,ml}_female.onnx`), each trained exclusively on its own language's `ai4bharat/Rasa` single-speaker `24 kHz` studio recordings and native Unicode orthography.

## CLI

```bash
npx ninada "ಕನ್ನಡವೇ ಸತ್ಯ ಕನ್ನಡವೇ ನಿತ್ಯ" -l kn -o kannada.wav
ninada "सत्यमेव जयते नानृतम्" -l sa -o satyam.wav
```

Runs `24 kHz` neural synthesis when ONNX weights are available and falls back automatically to a `0.0 MB` zero-dependency Cascade Klatt-Akṣara synthesizer offline.

## Native Script Orthography & Prosody

```js
import { studioTextToIds, textToIds } from 'ninada';

const { studioIds, lang } = studioTextToIds('ವಾಙ್ಮಯ ತಪಸ್ಸು ಮತ್ತು ಜ್ಞಾನ ವಿಕಾಸ ಅತ್ಯಂತ ಮುಖ್ಯವಾದವು.', 'kn');
const { aksharas } = textToIds('धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।', 'sa');
```

`studioTextToIds()` tokenizes directly in each language's native Unicode character table (`sa` 144 tokens, `kn` 85, `hi` 120, `ta` 61, `te` 110, `ml` 101) — preserving Classical Sanskrit vocalic `ṛ`/`ṝ`/`ḷ`, visarga sandhi, Kannada `ಙ`/`ಃ` allophony, Tamil `ழ`/`ற`/`ன`, and Malayalam atomic chillus (`ൺ ൻ ർ ൽ ൾ ൿ`) feeding a `24 kHz` `RelativePositionTransformer + StochasticDurationPredictor + HiFi-GAN v1` studio generator.

## Benchmarks (`N = 240` Held-Out 24 kHz Studio Utterances, `B = 1,000` Bootstrap 95% CIs)

| System | Rate | Footprint (↓) | Coverage (↑) | TTFA p50 (↓) | MCD 0–8 kHz (dB, ↓) | 8–12 kHz HF Err (dB, ↓) | Dur MAE (↓) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base `facebook/mms-tts-hin` | 16 kHz | 138.8 MB | 35.0% | 1.0 ms* | `7.97 [7.69, 8.24]` | 66.24 dB | 3.21s |
| Meta MMS 5-Specialist Fleet (`5x` VITS) | 16 kHz | 694.0 MB | 100.0% | 60.2 ms | `5.09 [5.03, 5.15]` | 65.90 dB | 1.34s |
| `ninada` v4.0 Untuned Warm-Start (`step=0`) | 24 kHz | 112.0 MB | 100.0% | 189.2 ms | `4.60 [4.54, 4.65]` | 11.01 dB | 0.95s |
| **`ninada` v4.0 Native Studio FP32** | **24 kHz** | 112.0 MB | **100.0%** | **45.3 ms** | **`3.40 [3.37, 3.43]`** | **6.34 dB** | **0.57s** |
| **`ninada` v4.0 Native Browser ONNX** | **24 kHz** | **27.9 MB** | **100.0%** | **372.7 ms** | **`3.39 [3.36, 3.43]`** | **6.37 dB** | **0.56s** |
| `ninada` Tier-2 Klatt Fallback | 24 kHz | **0.0 MB** | **100.0%** | **104.0 ms** | `10.38 [10.26, 10.50]` | 41.94 dB | 0.98s |

Per-language held-out `MCD 0–8 kHz (dB, ↓)`: Sanskrit **`3.51`** (vs `5.32` MMS), Hindi **`3.37`** (vs `4.85` MMS), Kannada **`3.35`** (vs `5.04` MMS), Tamil **`3.21`** (vs `4.87` MMS), Telugu **`3.59`** (vs `5.57` MMS), Malayalam **`3.34`** (vs `4.88` MMS). Full bootstrap CIs and parity telemetry (`0.61 dB` parity MCD, `r = 0.9999`) in [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md) and [`bench/results.json`](./bench/results.json).

## Reproduce & Verify

```bash
PYTHONPATH=python/src:. python training/train_native_specialists_gcp.py --work-dir data/v4 --out-dir artifacts/ninada_v4
PYTHONPATH=python/src:. python bench/check_bench_thresholds.py
pytest -q && node verify-browser.mjs
```

Trains all 12 native `24 kHz` Male/Female studio VITS specialists, evaluates `240` held-out studio clips, and verifies `0` `AudioWorklet` underruns and `0` cached network refetches in headless Chromium.

## Links

- [Live Browser Voice Studio](https://hemanth.github.io/ninada/)
- [Hugging Face Model (`gnumanth/ninada`)](https://huggingface.co/gnumanth/ninada)
- [Hugging Face Dataset (`gnumanth/ninada-sanskrit-indic`)](https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic)

## License

MIT © [Hemanth.HM](https://h3manth.com)
