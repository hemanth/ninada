# ninada

> Stream Classical Sanskrit and 6 Indic languages (`sa`, `kn`, `hi`, `ta`, `te`, `ml`) in the browser and Node.js via 6 native Halo-Chunked VITS+HiFi-GAN FP16 ONNX specialists (`54.7 MB` per specialist, persistent `CacheStorage` + `IndexedDB` caching, `123.7 dB` crossfaded seam SNR) or zero-dependency Klatt synthesis.

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
  encoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/kn/encoder_flow_quantized.onnx',
  vocoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/kn/vocoder_quantized.onnx',
  cacheModels: true, // persists in browser CacheStorage / IndexedDB across reloads
  ort,
});

for await (const chunk of voice.stream('ಕನ್ನಡ ಭಾಷೆಯ ನೈಸರ್ಗಿಕ ಧ್ವನಿ ಸಂಶ್ಲೇಷಣೆಗೆ ಸ್ವಾಗತ.')) {
  playPcm16k(chunk.pcm); // Float32Array @ 16 kHz
}

// Or synthesize full 16 kHz RIFF WAV bytes
const wav = await voice.speak('ಕನ್ನಡವೇ ಸತ್ಯ ಕನ್ನಡವೇ ನಿತ್ಯ', { lang: 'kn' });
await voice.dispose(); // release WASM session heap when switching languages
```

## CLI

```sh
npx ninada "ऋतं च सत्यं च" -l sa -o out.wav
```

## License

MIT © [Hemanth HM](https://h3manth.com)
