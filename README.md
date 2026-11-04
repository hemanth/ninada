# ninada

In-browser, Node.js, and Python 24,000 Hz text to speech for Indian English, Classical Sanskrit, and ten Indian languages (`sa`, `en`, `hi`, `kn`, `mr`, `bn`, `ta`, `te`, `ml`, `gu`, `pa`, `or`) powered by `27.9 MB` native per-language VITS ONNX specialists (`24` Male & Female studio models) with zero cross-language accent bleed.

```bash
npm install ninada webml-kit
```

```bash
pip install ninada
```

## Quick start

```js
import ninada from 'ninada';
import { detectDevice } from 'webml-kit';

const { backend } = await detectDevice();
const voice = await ninada('gnumanth/ninada', {
  lang: 'kn',
  voice: 'male',
  device: backend,
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

`Ninada` loads dedicated `27.9 MB` W8A32+FP16 ONNX specialists (`onnx/{sa,en,hi,kn,mr,bn,ta,te,ml,gu,pa,or}.onnx` and `onnx/*_female.onnx`), each trained exclusively on its own language's `ai4bharat/Rasa` or `SPRINGLab/IndicTTS-English` single-speaker `24 kHz` studio recordings and native Unicode orthography.

## CLI

```bash
npx ninada "Welcome to Ninada studio voice synthesis." -l en -v female -o english.wav
ninada "सत्यमेव जयते नानृतम्" -l sa -o satyam.wav
```

Runs `24 kHz` neural synthesis when ONNX weights are available and falls back automatically to a `0.0 MB` zero-dependency Cascade Klatt-Akṣara synthesizer offline.

## Native Script Orthography & Prosody

```js
import { studioTextToIds, textToIds } from 'ninada';

const { studioIds, lang } = studioTextToIds('मराठी भाषेची साहित्य परंपरा अत्यंत समृद्ध आहे.', 'mr');
const { aksharas } = textToIds('धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।', 'sa');
```

`studioTextToIds()` tokenizes directly in each language's native Unicode character table (`sa` 144 tokens, `en` 102, `hi` 120, `kn` 85, `mr` 144, `bn` 100, `ta` 61, `te` 110, `ml` 101, `gu` 124, `pa` 85, `or` 94) — preserving Classical Sanskrit vocalic `ṛ`/`ṝ`/`ḷ`, visarga sandhi, Indian English retroflex & stress patterns, Kannada `ಙ`/`ಃ` allophony, Tamil `ழ`/`ற`/`ன`, Malayalam atomic chillus (`ൺ ൻ ർ ൽ ൾ ൿ`), Gurmukhi tippi/addak (`ੰ`/`ੱ`), and Odia/Bengali conjuncts feeding a `24 kHz` `RelativePositionTransformer + StochasticDurationPredictor + HiFi-GAN v1` studio generator.

## Benchmarks (`N = 480` Held-Out 24 kHz Studio Utterances, `B = 1,000` Bootstrap 95% CIs)

| System | Rate | Footprint (↓) | Coverage (↑) | TTFA p50 (↓) | MCD 0–8 kHz (dB, ↓) | 8–12 kHz HF Err (dB, ↓) | Dur MAE (↓) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base `facebook/mms-tts-hin` | 16 kHz | 138.8 MB | 29.4% | 0.9 ms* | `7.98 [7.81, 8.16]` | 65.16 dB | 3.49s |
| Meta MMS 11-Specialist Fleet (`11x` VITS) | 16 kHz | 1526.8 MB | 100.0% | 204.2 ms | `5.17 [5.12, 5.22]` | 64.97 dB | 1.04s |
| `ninada` v5.0.1 Untuned Warm-Start (`step=0`) | 24 kHz | 112.0 MB | 100.0% | 191.0 ms | `4.58 [4.54, 4.62]` | 10.93 dB | 0.93s |
| **`ninada` v5.0.1 Native Studio FP32** | **24 kHz** | 112.0 MB | **100.0%** | **43.3 ms** | **`3.22 [3.19, 3.25]`** | **6.08 dB** | **0.58s** |
| **`ninada` v5.0.1 Native Browser ONNX** | **24 kHz** | **27.9 MB** | **100.0%** | **373.0 ms** | **`3.22 [3.19, 3.25]`** | **6.07 dB** | **0.58s** |
| `ninada` Tier-2 Klatt Fallback | 24 kHz | **0.0 MB** | **100.0%** | **110.2 ms** | `10.53 [10.43, 10.64]` | 42.38 dB | 0.88s |

Per-language held-out `MCD 0–8 kHz (dB, ↓)`: Punjabi **`2.89`** (vs `5.23` MMS), Indian English **`3.07`** (vs `5.94` MMS), Marathi **`3.10`** (vs `4.62` MMS), Odia **`3.11`** (vs `5.49` MMS), Tamil **`3.21`** (vs `4.87` MMS), Bengali **`3.27`** (vs `4.68` MMS), Gujarati **`3.30`** (vs `5.56` MMS), Malayalam **`3.35`** (vs `4.88` MMS), Hindi **`3.36`** (vs `4.85` MMS), Kannada **`3.36`** (vs `5.04` MMS), Sanskrit **`3.51`** (vs `5.32` MMS), Telugu **`3.59`** (vs `5.57` MMS). Full bootstrap CIs and parity telemetry (`0.58 dB` parity MCD, `r = 0.9999`) in [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md) and [`bench/results.json`](./bench/results.json).

## Reproduce & Verify

```bash
PYTHONPATH=python/src:. python training/train_native_specialists_gcp.py --work-dir data/v5 --out-dir artifacts/ninada_v5
PYTHONPATH=python/src:. python bench/check_bench_thresholds.py
pytest -q && node verify-browser.mjs
```

Trains all 24 native `24 kHz` Male/Female studio VITS specialists across 12 languages, evaluates `480` held-out studio clips, and verifies `0` `AudioWorklet` underruns and `0` cached network refetches in headless Chromium.

## Links

- [In-Browser Voice Studio](https://hemanth.github.io/ninada/)
- [Hugging Face Model (`gnumanth/ninada`)](https://huggingface.co/gnumanth/ninada)
- [Hugging Face Dataset (`gnumanth/ninada-sanskrit-indic`)](https://huggingface.co/datasets/gnumanth/ninada-sanskrit-indic)

## License

MIT © [Hemanth.HM](https://h3manth.com)
