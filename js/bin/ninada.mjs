#!/usr/bin/env node
import fs from 'node:fs';
import ninada from '../src/index.js';

const args = process.argv.slice(2);
if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
  console.log('Usage: ninada <text> [-l sa|hi|kn|ta|te|ml] [-r 1.0] [-o output.wav] [--json]');
  process.exit(0);
}

let lang = 'sa';
let rate = 1.0;
let outPath = 'ninada_out.wav';
let jsonOut = false;
const textParts = [];

for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if ((a === '-l' || a === '--lang') && args[i + 1]) {
    lang = args[++i];
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

const text = textParts.join(' ');
const voice = await ninada(lang);
const { wav, meta } = await voice.synthesize(text, { lang, rate });
fs.writeFileSync(outPath, Buffer.from(wav));

if (jsonOut) {
  console.log(JSON.stringify({ output: outPath, ...meta }, null, 2));
} else {
  console.log(`Wrote ${outPath} (${meta.durationSec.toFixed(2)}s, tier=${meta.tier}, lang=${meta.lang})`);
}
