import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcDir = path.join(__dirname, 'src');
const distDir = path.join(__dirname, 'dist');
const docsDir = path.join(__dirname, '..', 'docs');
fs.mkdirSync(distDir, { recursive: true });
fs.mkdirSync(docsDir, { recursive: true });

// Zero-dep bundler: inline g2p.js + klatt.js + index.js into dist/index.mjs, dist/index.cjs, and docs/ninada.mjs
const g2pSrc = fs.readFileSync(path.join(srcDir, 'g2p.js'), 'utf8')
  .replace(/^export /gm, '');
const klattSrc = fs.readFileSync(path.join(srcDir, 'klatt.js'), 'utf8')
  .replace(/^import .*$/gm, '')
  .replace(/^export /gm, '');
const indexSrc = fs.readFileSync(path.join(srcDir, 'index.js'), 'utf8')
  .replace(/import \{[\s\S]*?\} from '\.\/g2p\.js';/m, '')
  .replace(/import \{[\s\S]*?\} from '\.\/klatt\.js';/m, '');

const esmBundle = `// Auto-generated ESM bundle for ninada\n${g2pSrc}\n${klattSrc}\n${indexSrc}`;
fs.writeFileSync(path.join(distDir, 'index.mjs'), esmBundle, 'utf8');
fs.writeFileSync(path.join(docsDir, 'ninada.mjs'), esmBundle, 'utf8');

const cjsBody = esmBundle
  .replace(/export default async function ninada/, 'async function ninada')
  .replace(/export \{[\s\S]*?\};/m, `module.exports = Object.assign(ninada, {
  default: ninada,
  ninada,
  SAMPLE_RATE,
  HOP_LENGTH,
  VOCAB_TOKENS,
  TOKEN_TO_ID,
  BLANK_ID,
  LANGUAGE_IDS,
  splitProsodicClauses,
  detectScriptAndLang,
  iastToDevanagari,
  brahmicToDevanagari,
  applyPhonologicalRules,
  segmentAksharas,
  textToIds,
  synthesizeKlattPcm,
  encodeWavBuffer,
});`);
fs.writeFileSync(path.join(distDir, 'index.cjs'), cjsBody, 'utf8');
console.log('Built js/dist/index.mjs, js/dist/index.cjs, and docs/ninada.mjs');
