#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ninada from '../src/index.js';

const args = process.argv.slice(2);
if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
  console.log('Usage: ninada <text> [-l sa|hi|kn|ta|te|ml] [-m model.onnx|dir] [-r 1.0] [-o output.wav] [--json]');
  process.exit(0);
}

let lang = 'sa';
let rate = 1.0;
let modelArg = null;
let outPath = 'ninada_out.wav';
let jsonOut = false;
const textParts = [];

for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if ((a === '-l' || a === '--lang') && args[i + 1]) {
    lang = args[++i];
  } else if ((a === '-m' || a === '--model') && args[i + 1]) {
    modelArg = args[++i];
  } else if ((a === '-r' || a === '--rate') && args[i + 1]) {
    rate = Number(args[++i]) || 1.0;
  } else if ((a === '-o' || a === '--output') && args[i + 1]) {
    outPath = args[++i];
  } else if (a === '--json') {
    jsonOut = true;
  } else {
    textParts.push(a);
  }
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const opts = {};
if (modelArg) {
  if (fs.existsSync(modelArg) && fs.statSync(modelArg).isDirectory()) {
    const ef = path.join(modelArg, 'encoder_flow_quantized.onnx');
    const vf = path.join(modelArg, 'vocoder_quantized.onnx');
    const mf = path.join(modelArg, 'model_quantized.onnx');
    if (fs.existsSync(ef) && fs.existsSync(vf)) {
      opts.encoderModel = ef;
      opts.vocoderModel = vf;
    } else if (fs.existsSync(mf)) {
      opts.model = mf;
    }
  } else {
    opts.model = modelArg;
  }
} else {
  for (const dir of [path.join(repoRoot, 'artifacts', 'ninada', 'onnx'), path.join(repoRoot, 'docs')]) {
    const ef = path.join(dir, 'encoder_flow_quantized.onnx');
    const vf = path.join(dir, 'vocoder_quantized.onnx');
    const mf = path.join(dir, 'model_quantized.onnx');
    if (fs.existsSync(ef) && fs.existsSync(vf)) {
      opts.encoderModel = ef;
      opts.vocoderModel = vf;
      break;
    }
    if (fs.existsSync(mf)) {
      opts.model = mf;
      break;
    }
  }
}

const text = textParts.join(' ');
const voice = await ninada(lang, opts);
const { wav, meta } = await voice.synthesize(text, { lang, rate });
fs.writeFileSync(outPath, Buffer.from(wav));

if (jsonOut) {
  console.log(JSON.stringify({ output: outPath, ...meta }, null, 2));
} else {
  console.log(`Wrote ${outPath} (${meta.durationSec.toFixed(2)}s, ttfa=${meta.ttfaMs.toFixed(1)}ms, tier=${meta.tier}, lang=${meta.lang})`);
}
