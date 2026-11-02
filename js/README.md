# ninada

> Stream 24,000 Hz studio speech for Classical Sanskrit and 5 Indian languages (`sa`, `kn`, `hi`, `ta`, `te`, `ml`) in the browser and Node.js using dedicated `~28 MB` W8A32+FP16 Native Studio VITS ONNX specialists (`onnx/{sa,kn,hi,ta,te,ml}.onnx` + `onnx/{sa,kn,hi,ta,te,ml}_female.onnx`), with persistent `CacheStorage` caching (`0` reload network refetches) and zero-dependency Klatt Tier-2 fallback.

- **Live Web Worker WASM SIMD Studio:** <https://hemanth.github.io/ninada/>
- **Hugging Face Model:** <https://huggingface.co/gnumanth/ninada>
- **Source & Benchmarks:** <https://github.com/hemanth/ninada>

## Install

```sh
npm install ninada onnxruntime-web
```

## Streaming Usage

```js
import ninada from 'ninada';
import * as ort from 'onnxruntime-web';

const voice = await ninada('kannada', {
  models: {
    kn: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/kn.onnx',
    sa: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/sa.onnx',
  },
  cacheModels: true,
  ort,
});

for await (const chunk of voice.stream('ಕನ್ನಡ ಭಾಷೆಯ ನೈಸರ್ಗಿಕ ಧ್ವನಿ ಸಂಶ್ಲೇಷಣೆಗೆ ಸ್ವಾಗತ.')) {
  playPcm24k(chunk.pcm, chunk.sampleRate); // Float32Array @ 24,000 Hz
}

// Synthesize full 24 kHz RIFF WAV bytes (native Studio Male or Studio Female specialist)
const wav = await voice.speak('ಕನ್ನಡವೇ ಸತ್ಯ ಕನ್ನಡವೇ ನಿತ್ಯ', { lang: 'kn', voice: 'female' });
await voice.dispose();
```

## CLI

```sh
npx ninada "ऋतं च सत्यं च" -l sa -o out.wav
```

## License

MIT © [Hemanth HM](https://h3manth.com)
