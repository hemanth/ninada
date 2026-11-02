# ninada

Stream 24,000 Hz studio speech for Classical Sanskrit and 5 Indian languages (`sa`, `kn`, `hi`, `ta`, `te`, `ml`) in the browser and Node.js.

```bash
npm install ninada onnxruntime-web
```

## Quick start

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

const wav = await voice.speak('ಕನ್ನಡವೇ ಸತ್ಯ ಕನ್ನಡವೇ ನಿತ್ಯ', { lang: 'kn', voice: 'female' });
await voice.dispose();
```

`voice.stream()` yields 24 kHz `Float32Array` clauses as they synthesize. `voice.speak()` returns a complete 24 kHz RIFF WAV `Uint8Array` with automatic zero-dep Klatt fallback.

## CLI

```bash
npx ninada "ऋतं च सत्यं च" -l sa -o out.wav
```

Synthesizes any supported Brahmic script or IAST string directly to a 24 kHz WAV file.

## Related

- [Live Browser Studio](https://hemanth.github.io/ninada/) — Web Worker WASM SIMD playground
- [Hugging Face Weights](https://huggingface.co/gnumanth/ninada) — 12 native `~28 MB` W8A32+FP16 ONNX specialists
- [Source & Benchmarks](https://github.com/hemanth/ninada)

## License

MIT © [Hemanth.HM](https://h3manth.com)
