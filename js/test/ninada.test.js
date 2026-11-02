import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import ninada, {
  SAMPLE_RATE,
  HOP_LENGTH,
  VOCAB_TOKENS,
  TOKEN_TO_ID,
  IPA_VOCAB,
  RETROFLEX_LATERAL_ID,
  splitProsodicClauses,
  textToIpa,
  textToIds,
  iastToDevanagari,
  brahmicToDevanagari,
  splitStudioVitsOnnxBuffer,
} from '../src/index.js';

test('87-token Brahmic vocabulary and 24 kHz IPA vocabulary with retroflex lateral slot 79', () => {
  assert.equal(SAMPLE_RATE, 24000);
  assert.equal(HOP_LENGTH, 600);
  assert.equal(VOCAB_TOKENS.length, 87);
  assert.equal(VOCAB_TOKENS[85], '\u090b');
  assert.equal(VOCAB_TOKENS[86], '\u0919');
  assert.equal(TOKEN_TO_ID['\u090b'], 85);
  assert.equal(TOKEN_TO_ID['\u0919'], 86);
  assert.equal(IPA_VOCAB['\u026d'], 79);
  assert.equal(RETROFLEX_LATERAL_ID, 79);
});

test('CommonJS dist/index.cjs loads cleanly without ESM syntax leakage', () => {
  const require = createRequire(import.meta.url);
  const cjs = require('../dist/index.cjs');
  assert.equal(typeof cjs, 'function');
  assert.equal(typeof cjs.splitProsodicClauses, 'function');
  assert.equal(typeof cjs.textToIpa, 'function');
  assert.equal(cjs.SAMPLE_RATE, 24000);
  assert.equal(cjs.HOP_LENGTH, 600);
});

test('Sanskrit G2P handles vocalic r, homorganic velar nasal, Vedic cantillation, and purvarupa avagraha', () => {
  const res = textToIds('\u090b\u0924\u0951\u0902 \u0915\u0930\u094b\u093d\u0924\u093f \u0936\u0902\u0915\u0930\u0903', { lang: 'sa' });
  assert.equal(res.lang, 'sa');
  assert.ok(res.tokens.includes('\u090b'));
  assert.ok(res.tokens.includes('\u0919'));
  assert.ok(!res.normalizedDevanagari.includes('\u093d'));
  assert.ok(!res.normalizedDevanagari.includes('\u0951'));
  assert.ok(Array.isArray(res.ipaIds) && res.ipaIds.length >= 10);
});

test('IAST and Multi-Script Brahmic G2P map Kannada, Tamil, Telugu, Malayalam accurately', () => {
  const iast = iastToDevanagari('praj\u00f1\u0101p\u016brvakavy\u0101p\u0101ravattva\u1e41 j\u012bvatvam');
  assert.ok(iast.includes('\u091c\u094d\u091e'));

  const kn = textToIds('\u0c85\u0cb0\u0cbf\u0cb5\u0cc7 \u0c97\u0cc1\u0cb0\u0cc1');
  assert.equal(kn.lang, 'kn');
  assert.equal(kn.script, 'kannada');
  assert.ok(textToIpa('\u0cb9\u0cb3\u0cc6\u0c97\u0ca8\u0ccd\u0ca8\u0ca1', 'kn').includes('\u026d'));

  const ta = textToIds('\u0ba4\u0bae\u0bbf\u0bb4\u0bcd \u0bb5\u0bbe\u0bb4\u0bcd\u0b95');
  assert.equal(ta.lang, 'ta');
  assert.ok(ta.tokens.includes('\u0934'));
  assert.ok(ta.ipa.includes('\u027b'));

  const ml = brahmicToDevanagari('\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d7e');
  assert.ok(ml.includes('\u0933'));
});

test('Zero-dep Tier-2 Klatt streaming and synthesis generate valid 24kHz RIFF WAV bytes', async () => {
  const text = '\u0927\u0930\u094d\u092e\u0915\u094d\u0937\u0947\u0924\u094d\u0930\u0947 \u0915\u0941\u0930\u0941\u0915\u094d\u0937\u0947\u0924\u094d\u0930\u0947 \u0964 \u0938\u0924\u094d\u092f\u092e\u0947\u0935 \u091c\u092f\u0924\u0947 \u0965';
  assert.equal(splitProsodicClauses(text).length, 2);

  const voice = await ninada('sanskrit');
  const streamed = [];
  for await (const ch of voice.stream(text, { tier: 'klatt' })) {
    streamed.push(ch);
  }
  assert.equal(streamed.length, 2);
  assert.equal(streamed[0].tier, 'klatt_zero_dep');
  assert.equal(streamed[0].sampleRate, 24000);
  assert.equal(streamed[1].isFinal, true);

  let doneMeta = null;
  voice.on('done', (m) => { doneMeta = m; });
  const wav = await voice.speak(text, { tier: 'klatt' });
  assert.ok(wav instanceof ArrayBuffer);
  assert.ok(wav.byteLength > 6000);
  const magic = Buffer.from(wav.slice(0, 4)).toString('ascii');
  assert.equal(magic, 'RIFF');
  assert.equal(doneMeta.tier, 'klatt_zero_dep');
});

test('Tier-1 24kHz Studio Neural ONNX streaming produces low-TTFA chunks and 24kHz WAV in Node.js', async () => {
  const fs = await import('node:fs');
  const path = await import('node:path');
  const docsDir = path.resolve(import.meta.dirname, '../../docs');
  const v4SaPath = path.join(docsDir, 'models', 'sa.onnx');
  const v4KnPath = path.join(docsDir, 'models', 'kn.onnx');
  const studioPath = path.join(docsDir, 'ninada_24k_q8f16.onnx');
  const encPath = path.join(docsDir, 'encoder_flow_quantized.onnx');
  const vocPath = path.join(docsDir, 'vocoder_quantized.onnx');

  let voice;
  if (fs.existsSync(v4SaPath) && fs.existsSync(v4KnPath)) {
    voice = await ninada('sanskrit', {
      models: { sa: v4SaPath, kn: v4KnPath },
    });
    assert.equal(voice.isStreamingNeural, true);
  } else if (fs.existsSync(studioPath)) {
    voice = await ninada('sanskrit', {
      model: studioPath,
      voice: path.join(docsDir, 'voices', 'sa.bin'),
    });
    assert.equal(voice.isStreamingNeural, true);
  } else if (fs.existsSync(encPath) && fs.existsSync(vocPath)) {
    voice = await ninada('sanskrit', { encoderModel: encPath, vocoderModel: vocPath });
    assert.equal(voice.isStreamingNeural, true);
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
  assert.equal(chunks[0].sampleRate, 24000);
  assert.ok(chunks[0].ttfaMs > 0);
  assert.equal(chunks[chunks.length - 1].isFinal, true);

  const { pcm, meta } = await voice.synthesize('\u0938\u0924\u094d\u092f\u092e\u0947\u0935 \u091c\u092f\u0924\u0947');
  assert.equal(meta.tier, 'neural_onnx');
  assert.equal(meta.sampleRate, 24000);
  assert.ok(pcm.length > 18000);

  const femRes = await voice.synthesize('\u0938\u0924\u094d\u092f\u092e\u0947\u0935 \u091c\u092f\u0924\u0947', { voice: 'female' });
  assert.equal(femRes.meta.tier, 'neural_onnx');
  assert.ok(femRes.pcm.length > 18000);
  await voice.dispose();
});

test('Pan-Indic IPA G2P enforces jna allophony, Dravidian close-mid short-e, medial ɐ, and zero ˈə stress', () => {
  const saIpa = textToIds('प्रज्ञा ज्ञानं', { lang: 'sa' }).ipa;
  const hiIpa = textToIds('प्रज्ञा विज्ञान', { lang: 'hi' }).ipa;
  const teIpa = textToIds('తెలుగు భాష', { lang: 'te' }).ipa;
  const knIpa = textToIds('ಕನ್ನಡ ಭಾಷೆಯ ನೈಸರ್ಗಿಕ ಧ್ವನಿ', { lang: 'kn' }).ipa;
  const mlIpa = textToIds('മലയാള ഭാഷയുടെ സ്വാഗതം', { lang: 'ml' }).ipa;

  assert.ok(saIpa.includes('ɡɲ'));
  assert.ok(hiIpa.includes('ɡj'));
  assert.ok(teIpa.includes('tˈɛlʊɡʊ'));
  assert.equal(saIpa.includes('ˈə'), false);
  assert.equal(knIpa.includes('ˈə'), false);
  assert.equal(mlIpa.includes('ˈə'), false);
  assert.ok(knIpa.includes('ɐ'));
  assert.ok(mlIpa.includes('jʊɖɛ'));
});

test('webml-kit createOnnxPipeline and Hub repo auto-loader work without importing onnxruntime-web', async () => {
  const calls = [];
  const fakeSession = {
    inputNames: ['input_ids', 'scales'],
    async run(feeds) {
      assert.ok(feeds.input_ids && feeds.scales);
      const wav = new Float32Array(4800);
      for (let i = 0; i < wav.length; i++) wav[i] = Math.sin((2 * Math.PI * 220 * i) / 24000) * 0.25;
      return {
        waveform: { data: wav, dispose() {} },
        durations: { data: new Float32Array([2, 3, 2, 3]), dispose() {} },
      };
    },
    async release() {},
  };
  const fakeWebml = {
    async createOnnxPipeline(config, onProgress) {
      calls.push(config);
      onProgress?.({ status: 'ready', file: config.modelFile, loaded: 100, total: 100, percent: 100 });
      return {
        sessions: [fakeSession],
        backend: 'wasm',
        dispose() {},
      };
    },
  };

  let progressEvents = 0;
  const voice = await ninada('gnumanth/ninada', {
    lang: 'kn',
    voice: 'female',
    webml: fakeWebml,
    onProgress: () => { progressEvents += 1; },
  });
  assert.equal(voice.isStreamingNeural, true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].modelId, 'gnumanth/ninada');
  assert.equal(calls[0].modelFile, 'onnx/kn_female.onnx');
  assert.equal(progressEvents, 1);

  const chunks = [];
  for await (const ch of voice.stream('ಮನುಜಮತ ವಿಶ್ವಪಥ')) {
    chunks.push(ch);
  }
  assert.ok(chunks.length >= 1);
  assert.equal(chunks[0].tier, 'neural_onnx');
  assert.equal(chunks[0].sampleRate, 24000);
  await voice.dispose();
});

test('In-memory Studio VITS ONNX splitter enables sub-clause latent-window streaming with >38 dB SNR parity', async () => {
  const fs = await import('node:fs');
  const path = await import('node:path');
  const saPath = path.resolve(import.meta.dirname, '../../docs/models/sa.onnx');
  if (!fs.existsSync(saPath)) return;

  const rawBuf = fs.readFileSync(saPath);
  const tSplit0 = performance.now();
  const split = splitStudioVitsOnnxBuffer(rawBuf);
  const splitMs = performance.now() - tSplit0;
  assert.ok(split, 'Expected splitStudioVitsOnnxBuffer to succeed on docs/models/sa.onnx');
  assert.ok(splitMs < 60, `Expected in-memory ONNX split < 60ms, got ${splitMs.toFixed(1)}ms`);
  assert.ok(split.encFlowBytes.byteLength > 5_000_000);
  assert.ok(split.vocoderBytes.byteLength > 5_000_000);

  const voice = await ninada('sanskrit', { models: { sa: saPath } });
  const singleClauseText = 'प्रज्ञापूर्वकव्यापारवत्त्वं जीवत्वम् यथार्थज्ञानं प्रमाणम् धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः';

  // Warmup JIT
  await voice.synthesize('सत्यम्', { noise: 0, noiseDuration: 0 });

  const streamChunks = [];
  for await (const ch of voice.stream(singleClauseText, { noise: 0, noiseDuration: 0 })) {
    streamChunks.push(ch);
  }
  const mono = await voice.synthesize(singleClauseText, { noise: 0, noiseDuration: 0 });

  assert.ok(
    streamChunks.length >= 3,
    `Expected >=3 sub-clause latent-window chunks for single unpunctuated clause, got ${streamChunks.length}`
  );
  assert.ok(
    streamChunks[0].ttfaMs * 2.2 < mono.meta.latencyMs,
    `Expected streamed Chunk 0 TTFA (${streamChunks[0].ttfaMs.toFixed(1)}ms) to be >2.2x faster than monolithic (${mono.meta.latencyMs.toFixed(1)}ms)`
  );

  // Verify SNR parity across speech samples (excluding trailing pause)
  const totalStreamLen = streamChunks.reduce((acc, c) => acc + c.pcm.length, 0);
  assert.equal(totalStreamLen, mono.pcm.length, 'Expected identical sample count between latent-streamed and monolithic');
  const cat = new Float32Array(totalStreamLen);
  let off = 0;
  for (const c of streamChunks) {
    cat.set(c.pcm, off);
    off += c.pcm.length;
  }
  let sigEnergy = 0;
  let errEnergy = 0;
  for (let i = 0; i < cat.length; i++) {
    sigEnergy += mono.pcm[i] * mono.pcm[i];
    const d = mono.pcm[i] - cat[i];
    errEnergy += d * d;
  }
  const snrDb = 10 * Math.log10(sigEnergy / Math.max(1e-12, errEnergy));
  assert.ok(snrDb > 35, `Expected latent-windowed HiFi-GAN SNR > 35 dB vs monolithic, got ${snrDb.toFixed(2)} dB`);
  await voice.dispose();
});


