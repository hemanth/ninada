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
  buildStudioSentenceBlocks,
  cleanStudioText,
  textToIpa,
  ipaToIds,
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
  splitStudioVitsOnnxBuffer(rawBuf); // Warmup V8 JIT
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
  for (const ch of streamChunks) {
    assert.equal(
      ch.vocoderBucketFrames,
      80,
      `Expected single static vocoder bucket shape (80), got ${ch.vocoderBucketFrames}`
    );
  }
  const snrDb = 10 * Math.log10(sigEnergy / Math.max(1e-12, errEnergy));
  assert.ok(snrDb > 45, `Expected latent-windowed HiFi-GAN SNR > 45 dB vs monolithic, got ${snrDb.toFixed(2)} dB`);
  await voice.dispose();
});

test('English CamelCase G2P and unchopped single-sentence prosody preserve breath-group continuity', () => {
  const preset1 =
    'Sub-clause latent streaming synthesizes the opening phrase in milliseconds while the AudioWorklet ring buffer maintains glitch-free twenty-four kilohertz playback.';
  assert.ok([1, 2].includes(splitProsodicClauses(preset1).length));

  const norm = cleanStudioText('AudioWorklet WebGPU 24 kHz 28 MB 120ms', 'en');
  assert.equal(norm, 'audio worklet web gpu twenty four kilohertz twenty eight megabytes one hundred twenty milliseconds');
});

test('12-language v5.0 roster supports Indian English, Marathi, Bengali, Gujarati, Punjabi, and Odia end-to-end', async () => {
  const samples = [
    ['en', 'Hello Dr. Rao, welcome to the 24,000 Hz Ninada studio in 2026!'],
    ['mr', 'नमस्कार, आज पुण्यातील हवामान खूप छान आहे.'],
    ['bn', 'নমস্কার, আজ কলকাতার আবহাওয়া খুব সুন্দর।'],
    ['gu', 'નમસ્તે, આજે અમદાવાદમાં હવામાન ખૂબ સુંદર છે.'],
    ['pa', 'ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, ਅੱਜ ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਮੌਸਮ ਵਧੀਆ ਹੈ।'],
    ['or', 'ନମସ୍କାର, ଆଜି ଭୁବନେଶ୍ୱରରେ ପାଣିପାଗ ବହୁତ ସୁନ୍ଦର ଅଛି।'],
  ];

  for (const [lc, text] of samples) {
    const parsed = textToIds(text, { lang: lc });
    assert.equal(parsed.lang, lc);
    assert.ok(parsed.inputIds.length >= 10, `Expected inputIds >= 10 for ${lc}`);
    assert.ok(parsed.aksharas.length >= 3, `Expected aksharas >= 3 for ${lc}`);

    const voice = await ninada(lc);
    const wav = await voice.speak(text, { tier: 'klatt' });
    assert.ok(wav.byteLength > 4000, `Expected non-empty 24 kHz WAV for ${lc}`);
    await voice.dispose();
  }

  const loadedFiles = [];
  const fakeSession = {
    inputNames: ['input_ids', 'scales'],
    async run() {
      return {
        waveform: { data: new Float32Array(2400).fill(0.05), dispose() {} },
        durations: { data: new Float32Array([2, 2, 2]), dispose() {} },
      };
    },
    async release() {},
  };
  const fakeWebml = {
    async createOnnxPipeline(cfg) {
      loadedFiles.push(cfg.modelFile);
      return { sessions: [fakeSession], backend: 'wasm', dispose() {} };
    },
  };
  const enVoice = await ninada('gnumanth/ninada', { lang: 'english', voice: 'female', webml: fakeWebml });
  await enVoice.speak('Welcome to Bengaluru');
  await enVoice.speak('ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ', { lang: 'punjabi', voice: 'male' });
  assert.deepEqual(loadedFiles, ['onnx/en_female.onnx', 'onnx/pa.onnx']);
  await enVoice.dispose();
});

test('Multi-clause Klatt streaming seam continuity, |Δy| < 0.008, and Seam SNR > 45 dB', async () => {
  const cases = [
    ['hindi', 'नमस्ते, आपकी समस्या का समाधान शीघ्र ही किया जाएगा । कृपया अपनी खाता संख्या बताएं, धन्यवाद ॥'],
    ['english', 'The AudioWorklet ring buffer processes audio chunks in real time, while the WebAssembly engine runs neural inference.'],
    ['kannada', 'ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಮತ್ತು ಅತ್ಯಂತ ಸುಂದರವಾದ ನಗರ । ಇದು ಭಾರತದ ಸಿಲಿಕಾನ್ ವ್ಯಾಲಿ ಎಂದು ಪ್ರಸಿದ್ಧವಾಗಿದೆ ॥'],
    ['telugu', 'హైదరాబాదు తెలంగాణ రాజధాని, అత్యంత సుందరమైన నగరం । ఇది భారతదేశ ఐటీ కేంద్రంగా ప్రసిద్ధి చెందింది, ధన్యవాదాలు ॥'],
  ];
  for (const [lang, text] of cases) {
    const voice = await ninada(lang);
    const chunks = [];
    for await (const ch of voice.stream(text, { tier: 'klatt' })) {
      chunks.push(ch);
    }
    assert.ok(chunks.length >= 2, `[${lang}] Expected >= 2 chunks, got ${chunks.length}`);

    const totalStreamLen = chunks.reduce((acc, c) => acc + c.pcm.length, 0);
    const stitched = new Float32Array(totalStreamLen);
    let off = 0;
    const boundaries = [];
    for (let i = 0; i < chunks.length; i++) {
      stitched.set(chunks[i].pcm, off);
      off += chunks[i].pcm.length;
      if (i < chunks.length - 1) boundaries.push(off);
    }

    // 1. Boundary transient step jumps |Δy| < 0.008
    for (let i = 0; i < boundaries.length; i++) {
      const b = boundaries[i];
      const jump = Math.abs(stitched[b] - stitched[b - 1]);
      assert.ok(
        jump < 0.008,
        `[${lang}] Boundary ${i} at sample ${b}: step jump |Δy| = ${jump.toFixed(6)} exceeded 0.008`
      );
    }

    // 2. Seam SNR > 45 dB vs monolithic synthesis
    const mono = await voice.synthesize(text, { tier: 'klatt' });
    assert.equal(mono.pcm.length, totalStreamLen, `[${lang}] Sample length parity required between stream and synthesize`);

    for (let i = 0; i < boundaries.length; i++) {
      const b = boundaries[i];
      const win = 512;
      const st = Math.max(0, b - 256);
      const ed = Math.min(mono.pcm.length, st + win);
      let sigPow = 0.0;
      let errPow = 0.0;
      for (let k = st; k < ed; k++) {
        const r = mono.pcm[k];
        const s = stitched[k];
        sigPow += r * r;
        errPow += (s - r) * (s - r);
      }
      const snr = errPow < 1e-12 ? 100.0 : 10.0 * Math.log10(sigPow / errPow);
      assert.ok(
        snr > 45.0,
        `[${lang}] Boundary ${i} at sample ${b}: Seam SNR ${snr.toFixed(2)} dB < 45.0 dB threshold`
      );
    }
    await voice.dispose();
  }
});

test('SemVer and multi-dot decimal normalization in cleanStudioText', () => {
  const cases = [
    ['version 2.0.1', 'version two point zero point one'],
    ['version 2.0.1.', 'version two point zero point one.'],
    ['Linux 6.1.102', 'linux six point one point one hundred two'],
    ['v1.2.3.4', 'v one point two point three point four'],
    ['IP 192.168.1.1', 'ip one hundred ninety two point one hundred sixty eight point one point one'],
    ['3.14159', 'three point one four one five nine'],
    ['0.001', 'zero point zero zero one'],
    ['100.50', 'one hundred point five zero'],
  ];
  for (const [raw, expected] of cases) {
    const res = cleanStudioText(raw, 'en');
    assert.equal(res, expected, `Mismatch for ${raw}`);
    if (!raw.endsWith('.')) {
      assert.ok(!res.includes('.'), `Found unexpected literal period in ${res}`);
    }
  }
});

test('Abbreviation protection for approx, Rs, and orphan-tail merging parity in splitProsodicClauses', () => {
  const cases = [
    ['Dr. Rao vs. Prof. Smith at St. Jude Inc. approx. 10 km/h.', 1],
    ['Prof. Smith vs. Dr. Watson: the benchmark completed in approx. 120ms with 99.9% accuracy; however, Mr. Brown from Univ. Dept. requested 3.14 iterations.', 3],
    ['Yes. No. Dr. Lee said etc. and approx. 5 MB was transferred.', 1],
    ['He paid Rs. 500 for the book.', 1],
    ['This is a reasonably long introductory clause that sets the stage, yes.', 2],
  ];
  for (const [text, expected] of cases) {
    const res = splitProsodicClauses(text, text.includes('stage, yes.') ? 60 : 100);
    assert.equal(res.length, expected, `Expected ${expected} clauses, got ${res.length}: ${JSON.stringify(res)}`);
  }
});

test('Punjabi numeral transliteration expands to phonetic aksharas for Klatt synthesis', () => {
  const text = 'ਇੱਕ ਕੁੱਤਾ ਅਤੇ ਸੱਤ ਬੱਚੇ, 500 ਰੁਪਏ।';
  const parsed = textToIds(text, { lang: 'pa' });
  const aksharas = parsed.aksharas.map((a) => a.akshara);
  assert.ok(aksharas.includes('पं'), `Expected numeral akshara 'पं' in ${JSON.stringify(aksharas)}`);
  assert.ok(parsed.nativeTokens.includes('੫'), `Expected Gurmukhi token '੫' in ${JSON.stringify(parsed.nativeTokens)}`);

  const pureParsed = textToIds('500', { lang: 'pa' });
  assert.equal(pureParsed.aksharas.length, 8, `Expected 8 aksharas for 500, got ${pureParsed.aksharas.length}`);
  assert.ok(pureParsed.nativeTokens.includes('੫'), `Expected '੫' in ${JSON.stringify(pureParsed.nativeTokens)}`);
  assert.ok(pureParsed.nativeTokens.includes('੦'), `Expected '੦' in ${JSON.stringify(pureParsed.nativeTokens)}`);

  const pure1234 = textToIds('1234', { lang: 'pa' });
  for (const digit of ['੧', '੨', '੩', '੪']) {
    assert.ok(pure1234.nativeTokens.includes(digit), `Expected '${digit}' in ${JSON.stringify(pure1234.nativeTokens)}`);
  }

  const danda500 = textToIds('500।', { lang: 'pa' });
  assert.ok(danda500.nativeTokens.includes('੫'), `Expected '੫' in ${JSON.stringify(danda500.nativeTokens)}`);
});

test('Kannada Dravidian streaming synthesis, vowel preservation, retroflex lateral ಳ, and offset monotonicity', async () => {
  // 1. Dravidian vowel preservation (ಎ, ಏ, ಒ, ಓ)
  const vowelText = 'ಎಲೆ ಏಣಿ ಒಂಟೆ ಓಟ';
  const vowelIpa = textToIpa(vowelText, 'kn');
  assert.ok(vowelIpa.includes('ɛ'), `Expected short e [ɛ] in ${vowelIpa}`);
  assert.ok(vowelIpa.includes('eː'), `Expected long e [eː] in ${vowelIpa}`);
  assert.ok(vowelIpa.includes('o'), `Expected short o [o] in ${vowelIpa}`);
  assert.ok(vowelIpa.includes('oː'), `Expected long o [oː] in ${vowelIpa}`);

  // 2. Retroflex lateral approximant ಳ (slot 79)
  const retroflexText = 'ಹಳೆಗನ್ನಡ ಮತ್ತು ಬೆಳ್ಳಿ ಬೆಳಕು';
  const retroflexIpa = textToIpa(retroflexText, 'kn');
  assert.ok(retroflexIpa.includes('ɭ'), `Expected retroflex lateral [ɭ] in ${retroflexIpa}`);
  const tokenIds = ipaToIds(retroflexIpa);
  assert.ok(tokenIds.includes(RETROFLEX_LATERAL_ID), `Expected slot ${RETROFLEX_LATERAL_ID} in ${tokenIds}`);
  assert.equal(IPA_VOCAB['ɭ'], 79);

  // 3. Geminate consonants (ಕ್ಕ, ತ್ತ)
  const geminateText = 'ಅಕ್ಕ ಮತ್ತು ಅತ್ತೆ';
  const geminateIpa = textToIpa(geminateText, 'kn');
  assert.ok(geminateIpa.includes('kk'), `Expected geminate kk in ${geminateIpa}`);
  assert.ok(geminateIpa.includes('tt'), `Expected geminate tt in ${geminateIpa}`);

  // 4. Streaming performance: TTFA < 150 ms (target < 50 ms), 0 overlap, offset monotonicity
  const voice = await ninada('kannada');
  const text = 'ಕನ್ನಡ ನಾಡು ಚಂದನದ ಬೀಡು . ಬೆಳ್ಳಿ ಬೆಳಕು ಹರಡಿತು ಮತ್ತು ಹಳೆಯ ಕನ್ನಡ ಕಾವ್ಯಗಳು ಜನರ ಮನಸ್ಸನ್ನು ತಣಿಸಿದವು .';
  const t0 = performance.now();
  const chunks = [];
  let firstChunkTime = null;
  for await (const ch of voice.stream(text, { tier: 'klatt' })) {
    if (firstChunkTime === null) firstChunkTime = performance.now() - t0;
    chunks.push(ch);
  }

  assert.ok(chunks.length >= 2, `Expected >= 2 chunks, got ${chunks.length}`);
  assert.ok(firstChunkTime !== null, 'Failed to capture first chunk time');
  assert.ok(firstChunkTime < 150.0, `TTFA ${firstChunkTime.toFixed(2)} ms exceeded 150 ms threshold`);

  // Sample offset monotonicity (Δsamples = 0) and zero chunk overlap
  let curOffset = 0;
  for (let idx = 0; idx < chunks.length; idx++) {
    const c = chunks[idx];
    assert.equal(c.sampleOffset, curOffset, `Chunk ${idx} sampleOffset ${c.sampleOffset} != ${curOffset}`);
    assert.equal(c.sample_offset, curOffset, `Chunk ${idx} sample_offset ${c.sample_offset} != ${curOffset}`);
    curOffset += c.pcm.length;
  }

  // 5. Acoustic seam continuity and Seam SNR > 45 dB vs monolithic
  const totalStreamLen = chunks.reduce((acc, c) => acc + c.pcm.length, 0);
  const stitched = new Float32Array(totalStreamLen);
  let off = 0;
  const boundaries = [];
  for (let i = 0; i < chunks.length; i++) {
    stitched.set(chunks[i].pcm, off);
    off += chunks[i].pcm.length;
    if (i < chunks.length - 1) boundaries.push(off);
  }

  const mono = await voice.synthesize(text, { tier: 'klatt' });
  assert.equal(mono.pcm.length, totalStreamLen, 'Sample length parity required between stream and synthesize');

  for (let i = 0; i < boundaries.length; i++) {
    const b = boundaries[i];
    const jump = Math.abs(stitched[b] - stitched[b - 1]);
    assert.ok(jump < 0.008, `Boundary ${i} at sample ${b}: step jump |Δy| = ${jump.toFixed(6)} exceeded 0.008`);
  }

  for (let i = 0; i < boundaries.length; i++) {
    const b = boundaries[i];
    const win = 512;
    const st = Math.max(0, b - 256);
    const ed = Math.min(mono.pcm.length, st + win);
    let sigPow = 0.0;
    let errPow = 0.0;
    for (let k = st; k < ed; k++) {
      const r = mono.pcm[k];
      const s = stitched[k];
      sigPow += r * r;
      errPow += (s - r) * (s - r);
    }
    const snr = errPow < 1e-12 ? 100.0 : 10.0 * Math.log10(sigPow / errPow);
    assert.ok(snr > 45.0, `Boundary ${i} at sample ${b}: Seam SNR ${snr.toFixed(2)} dB < 45.0 dB threshold`);
  }

  await voice.dispose();
});

test('buildStudioSentenceBlocks preserves commas and semicolons intact and splits strictly on sentences', () => {
  const passage =
    'Ninada runs privately in the browser, with zero server calls; audio streams smoothly. Second sentence follows cleanly! आप जैसे ही कोई वाक्य चुनते हैं, निनाद सुगम ऑडियो स्ट्रीमिंग शुरू करता है ।';
  const blocks = buildStudioSentenceBlocks(passage);
  assert.equal(blocks.length, 3, `Expected 3 full sentence blocks without comma splits, got ${blocks.length}`);
  assert.ok(blocks[0].text.includes(',') && blocks[0].text.includes(';'));
  assert.ok(blocks[2].text.includes(','));
});

