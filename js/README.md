# ninada

> Speak Classical Sanskrit and 6 Indic languages (`sa`, `hi`, `kn`, `ta`, `te`, `ml`) in the browser and Node.js via a 56.6 MB Disentangled VITS+HiFi-GAN ONNX or zero-dependency Klatt synthesis.

- **Live Browser WASM SIMD Studio:** <https://hemanth.github.io/ninada/>
- **Hugging Face Model:** <https://huggingface.co/gnumanth/ninada>
- **Source & Benchmarks:** <https://github.com/hemanth/ninada>

## Install

```sh
npm install ninada onnxruntime-web
```

## Usage

```js
import ninada from 'ninada';
import * as ort from 'onnxruntime-web';

const voice = await ninada('sanskrit', {
  model: 'https://huggingface.co/gnumanth/ninada/resolve/main/onnx/model_quantized.onnx',
  ort,
});

const wav = await voice.speak('prajñāpūrvakavyāpāravattvaṁ jīvatvam');
```

## CLI

```sh
npx ninada "ऋतं च सत्यं च" -l sa -o out.wav
```

## License

MIT © [Hemanth HM](https://h3manth.com)
