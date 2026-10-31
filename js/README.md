# ninada

> Stream Classical Sanskrit and 6 Indic languages (`sa`, `kn`, `hi`, `ta`, `te`, `ml`) in the browser and Node.js via Halo-Chunked VITS+HiFi-GAN ONNX (`178.6 ms` TTFA) or zero-dependency Klatt synthesis.

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
  encoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/encoder_flow_quantized.onnx',
  vocoderModel: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/vocoder_quantized.onnx',
  ort,
});

for await (const chunk of voice.stream('ಕನ್ನಡ ಭಾಷೆಯ ನೈಸರ್ಗಿಕ ಧ್ವನಿ ಸಂಶ್ಲೇಷಣೆಗೆ ಸ್ವಾಗತ.')) {
  playPcm16k(chunk.pcm); // Float32Array @ 16 kHz
}

// Or synthesize full 16 kHz RIFF WAV bytes
const wav = await voice.speak('prajñāpūrvakavyāpāravattvaṁ jīvatvam', { lang: 'sa' });
```

## CLI

```sh
npx ninada "ऋतं च सत्यं च" -l sa -o out.wav
```

## License

MIT © [Hemanth HM](https://h3manth.com)
