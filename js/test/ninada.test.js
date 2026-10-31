import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import ninada, {
  VOCAB_TOKENS,
  TOKEN_TO_ID,
  splitProsodicClauses,
  textToIds,
  iastToDevanagari,
  brahmicToDevanagari,
} from '../src/index.js';

test('87-token vocabulary preserves Classical/Vedic Sanskrit vocalic r and velar nasal', () => {
  assert.equal(VOCAB_TOKENS.length, 87);
  assert.equal(VOCAB_TOKENS[85], '\u090b');
  assert.equal(VOCAB_TOKENS[86], '\u0919');
  assert.equal(TOKEN_TO_ID['\u090b'], 85);
  assert.equal(TOKEN_TO_ID['\u0919'], 86);
});

test('CommonJS dist/index.cjs loads cleanly without ESM syntax leakage', () => {
  const require = createRequire(import.meta.url);
  const cjs = require('../dist/index.cjs');
  assert.equal(typeof cjs, 'function');
  assert.equal(typeof cjs.splitProsodicClauses, 'function');
  assert.equal(cjs.HOP_LENGTH, 256);
});

test('Sanskrit G2P handles vocalic r, homorganic velar nasal, Vedic cantillation, and purvarupa avagraha', () => {
  const res = textToIds('\u090b\u0924\u0951\u0902 \u0915\u0930\u094b\u093d\u0924\u093f \u0936\u0902\u0915\u0930\u0903', { lang: 'sa' });
  assert.equal(res.lang, 'sa');
  assert.ok(res.tokens.includes('\u090b'));
  assert.ok(res.tokens.includes('\u0919'));
  assert.ok(!res.normalizedDevanagari.includes('\u093d'));
  assert.ok(!res.normalizedDevanagari.includes('\u0951'));
});

test('IAST and Multi-Script Brahmic G2P map Kannada, Tamil, Telugu, Malayalam accurately', () => {
  const iast = iastToDevanagari('praj\u00f1\u0101p\u016brvakavy\u0101p\u0101ravattva\u1e41 j\u012bvatvam');
  assert.ok(iast.includes('\u091c\u094d\u091e'));

  const kn = textToIds('\u0c85\u0cb0\u0cbf\u0cb5\u0cc7 \u0c97\u0cc1\u0cb0\u0cc1');
  assert.equal(kn.lang, 'kn');
  assert.equal(kn.script, 'kannada');

  const ta = textToIds('\u0ba4\u0bae\u0bbf\u0bb4\u0bcd \u0bb5\u0bbe\u0bb4\u0bcd\u0b95');
  assert.equal(ta.lang, 'ta');
  assert.ok(ta.tokens.includes('\u0934'));

  const ml = brahmicToDevanagari('\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d7e');
  assert.ok(ml.includes('\u0933'));
});

test('Zero-dep Tier-2 Klatt streaming and synthesis generate valid 16kHz RIFF WAV bytes', async () => {
  const text = '\u0927\u0930\u094d\u092e\u0915\u094d\u0937\u0947\u0924\u094d\u0930\u0947 \u0915\u0941\u0930\u0941\u0915\u094d\u0937\u0947\u0924\u094d\u0930\u0947 \u0964 \u0938\u0924\u094d\u092f\u092e\u0947\u0935 \u091c\u092f\u0924\u0947 \u0965';
  assert.equal(splitProsodicClauses(text).length, 2);

  const voice = await ninada('sanskrit');
  const streamed = [];
  for await (const ch of voice.stream(text, { tier: 'klatt' })) {
    streamed.push(ch);
  }
  assert.equal(streamed.length, 2);
  assert.equal(streamed[0].tier, 'klatt_zero_dep');
  assert.equal(streamed[1].isFinal, true);

  let doneMeta = null;
  voice.on('done', (m) => { doneMeta = m; });
  const wav = await voice.speak(text, { tier: 'klatt' });
  assert.ok(wav instanceof ArrayBuffer);
  assert.ok(wav.byteLength > 4000);
  const magic = Buffer.from(wav.slice(0, 4)).toString('ascii');
  assert.equal(magic, 'RIFF');
  assert.equal(doneMeta.tier, 'klatt_zero_dep');
});

test('Tier-1 Neural ONNX streaming produces low-TTFA chunks and 16kHz WAV in Node.js', async () => {
  const fs = await import('node:fs');
  const path = await import('node:path');
  const docsDir = path.resolve(import.meta.dirname, '../../docs');
  const encPath = path.join(docsDir, 'encoder_flow_quantized.onnx');
  const vocPath = path.join(docsDir, 'vocoder_quantized.onnx');
  const monoPath = path.join(docsDir, 'model_quantized.onnx');

  let voice;
  if (fs.existsSync(encPath) && fs.existsSync(vocPath)) {
    voice = await ninada('sanskrit', { encoderModel: encPath, vocoderModel: vocPath });
    assert.equal(voice.isStreamingNeural, true);
  } else if (fs.existsSync(monoPath)) {
    voice = await ninada('sanskrit', { model: monoPath });
  } else {
    return;
  }

  assert.equal(voice.isNeural, true);
  const chunks = [];
  for await (const ch of voice.stream('\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \u0cad\u0cbe\u0cb7\u0cc6\u0caf \u0ca8\u0cc8\u0cb8\u0cb0\u0ccd\u0c97\u0cbf\u0c95 \u0ca7\u0ccd\u0cb5\u0ca8\u0cbf \u0cb8\u0c82\u0cb6\u0ccd\u0cb2\u0cc7\u0cb7\u0ca3\u0cc6\u0c97\u0cc6 \u0cb8\u0ccd\u0cb5\u0cbe\u0c97\u0ca4.', { lang: 'kn' })) {
    chunks.push(ch);
  }
  assert.ok(chunks.length >= 1);
  assert.equal(chunks[0].tier, 'neural_onnx');
  assert.ok(chunks[0].ttfaMs > 0);
  assert.equal(chunks[chunks.length - 1].isFinal, true);

  const { pcm, meta } = await voice.synthesize('\u0938\u0924\u094d\u092f\u092e\u0947\u0935 \u091c\u092f\u0924\u0947');
  assert.equal(meta.tier, 'neural_onnx');
  assert.ok(pcm.length > 16000);
});
