# ninada

In-browser and Node.js 24,000 Hz text to speech for English, Classical Sanskrit, and 10 Indian languages (`sa`, `en`, `hi`, `kn`, `mr`, `bn`, `ta`, `te`, `ml`, `gu`, `pa`, `or`).

```bash
npm install ninada webml-kit
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
  onProgress: ({ percent }) => console.log(`Loading specialist: ${percent}%`),
});

for await (const chunk of voice.stream('ಕನ್ನಡ ಭಾಷೆಯ ನೈಸರ್ಗಿಕ ಧ್ವನಿ ಸಂಶ್ಲೇಷಣೆಗೆ ಸ್ವಾಗತ.')) {
  playPcm24k(chunk.pcm, chunk.sampleRate); // Float32Array @ 24,000 Hz
}

const wav = await voice.speak('ಕನ್ನಡವೇ ಸತ್ಯ ಕನ್ನಡವೇ ನಿತ್ಯ', { lang: 'kn', voice: 'female' });
await voice.dispose();
```

`ninada` pairs directly with [`webml-kit`](https://github.com/hemanth/webml-kit) (`detectDevice`, `createOnnxPipeline`, persistent `Cache` API, and automatic WebGPU-to-WASM SIMD fallback) with zero `onnxruntime-web` boilerplate. You can also pass an explicit `createOnnxPipeline(...)` instance via `{ pipeline }` or local `.onnx` paths via `{ models }`. `voice.stream()` yields 24 kHz `Float32Array` clauses as they synthesize; `voice.speak()` returns a complete 24 kHz RIFF WAV `ArrayBuffer` with automatic zero-dep Klatt fallback.

## CLI

```bash
npx ninada "ऋतं च सत्यं च" -l sa -o out.wav
```

Synthesizes any supported Brahmic script or IAST string directly to a 24 kHz WAV file.

## Related

- [In-Browser Voice Studio](https://hemanth.github.io/ninada/) (Web Worker WebGPU & WASM SIMD playground)
- [Hugging Face Weights](https://huggingface.co/gnumanth/ninada) (48 dual-tier `27.9 MB` Studio `HiFi-GAN` + `21.7 MB` Mobile `iSTFTNet` ONNX specialists across `12` languages x Male & Female)
- [Source & Benchmarks](https://github.com/hemanth/ninada)

## License

MIT © [Hemanth.HM](https://h3manth.com)
