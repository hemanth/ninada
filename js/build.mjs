import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcDir = path.join(__dirname, 'src');
const distDir = path.join(__dirname, 'dist');
const docsDir = path.join(__dirname, '..', 'docs');
fs.mkdirSync(distDir, { recursive: true });
fs.mkdirSync(docsDir, { recursive: true });

// Zero-dep bundler: isolate ipa_g2p.js + studio_g2p.js + inline g2p.js + klatt.js + index.js into dist/index.mjs, dist/index.cjs, and docs/ninada.mjs
const rawIpaSrc = fs.readFileSync(path.join(srcDir, 'ipa_g2p.js'), 'utf8')
  .replace(/^export /gm, '');
const ipaG2pSrc = `const { IPA_VOCAB, IPA_ID_TO_TOKEN, RETROFLEX_LATERAL_ID, textToIpa, ipaToIds } = (() => {\n${rawIpaSrc}\n  return { IPA_VOCAB, IPA_ID_TO_TOKEN, RETROFLEX_LATERAL_ID, textToIpa, ipaToIds };\n})();`;
const g2pSrc = fs.readFileSync(path.join(srcDir, 'g2p.js'), 'utf8')
  .replace(/import \{[\s\S]*?\} from '\.\/ipa_g2p\.js';/m, '')
  .replace(/^export \{ IPA_VOCAB[\s\S]*?\};/m, '')
  .replace(/^export /gm, '');
const rawStudioSrc = fs.readFileSync(path.join(srcDir, 'studio_g2p.js'), 'utf8')
  .replace(/import \{[\s\S]*?\} from '\.\/g2p\.js';/m, '')
  .replace(/^export /gm, '');
const studioG2pSrc = `const { STUDIO_HOP_LENGTH, STUDIO_VOCABS, cleanStudioText, studioTextToIds } = (() => {\n${rawStudioSrc}\n  return { STUDIO_HOP_LENGTH, STUDIO_VOCABS, cleanStudioText, studioTextToIds };\n})();`;
const klattSrc = fs.readFileSync(path.join(srcDir, 'klatt.js'), 'utf8')
  .replace(/^import .*$/gm, '')
  .replace(/^export /gm, '');
const indexSrc = fs.readFileSync(path.join(srcDir, 'index.js'), 'utf8')
  .replace(/import \{[\s\S]*?\} from '\.\/g2p\.js';/m, '')
  .replace(/import \{[\s\S]*?\} from '\.\/studio_g2p\.js';/m, '')
  .replace(/import \{[\s\S]*?\} from '\.\/klatt\.js';/m, '');

const esmBundle = `// Auto-generated ESM bundle for ninada v4.0\n${ipaG2pSrc}\n${g2pSrc}\n${studioG2pSrc}\n${klattSrc}\n${indexSrc}`;
fs.writeFileSync(path.join(distDir, 'index.mjs'), esmBundle, 'utf8');
fs.writeFileSync(path.join(docsDir, 'ninada.mjs'), esmBundle, 'utf8');

const cjsBody = esmBundle
  .replace(/export default async function ninada/, 'async function ninada')
  .replace(/export \{[\s\S]*?\};/m, `module.exports = Object.assign(ninada, {
  default: ninada,
  ninada,
  Ninada,
  DEFAULT_HF_REPO,
  SAMPLE_RATE,
  HOP_LENGTH,
  STUDIO_HOP_LENGTH,
  STUDIO_VOCABS,
  VOCAB_TOKENS,
  TOKEN_TO_ID,
  IPA_VOCAB,
  IPA_ID_TO_TOKEN,
  RETROFLEX_LATERAL_ID,
  BLANK_ID,
  LANGUAGE_IDS,
  NATIVE_VOCABS,
  NATIVE_TOKEN_TO_ID,
  splitProsodicClauses,
  detectScriptAndLang,
  iastToDevanagari,
  brahmicToDevanagari,
  applyPhonologicalRules,
  segmentAksharas,
  toNativeModelText,
  textToIpa,
  ipaToIds,
  textToIds,
  cleanStudioText,
  studioTextToIds,
  splitStudioVitsOnnxBuffer,
  createSplitStudioVitsSession,
  synthesizeKlattPcm,
  encodeWavBuffer,
  generateRoomTone,
});`);
fs.writeFileSync(path.join(distDir, 'index.cjs'), cjsBody, 'utf8');
console.log('Built js/dist/index.mjs, js/dist/index.cjs, and docs/ninada.mjs');
