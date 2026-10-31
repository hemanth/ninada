import * as ortNs from './ort/ort.wasm.min.mjs';
import ninada from './ninada.mjs';

const ort = ortNs?.default?.InferenceSession ? ortNs.default : ortNs;

const HF_REPO = 'gnumanth/ninada';
const SUPPORTED_LANGS = ['sa', 'kn', 'hi', 'ta', 'te', 'ml'];

const DEFAULT_MODEL_CACHE_NAME = 'ninada-onnx-v2.0.0';
const MODEL_CACHE_PREFIX = 'ninada-onnx-';
const IDB_NAME = 'ninada-onnx-idb-v2';
const IDB_STORE = 'models';

const DEFAULT_ENC_BYTES = 28626120;
const DEFAULT_VOC_BYTES = 28730980;

let activeCacheName = DEFAULT_MODEL_CACHE_NAME;
let manifestPromise = null;
let cachesPruned = false;
let ortConfigured = false;

let voiceEngine = null;
let currentLoadedLang = null;
let lastReadyPayload = null;
let activeAbort = null;
let activeReqId = 0;
let runQueue = Promise.resolve();

function buildFallbackLangSpec(lc) {
  const isPrimary = lc === 'sa';
  return {
    lang: lc,
    encoderBytes: 0,
    vocoderBytes: 0,
    totalMB: 54.7,
    encoderCacheKey: `ninada_v2_${lc}_enc.onnx`,
    vocoderCacheKey: `ninada_v2_${lc}_voc.onnx`,
    encoderUrls: isPrimary
      ? [
          './encoder_flow_quantized.onnx',
          `https://huggingface.co/${HF_REPO}/resolve/main/onnx/sa/encoder_flow_quantized.onnx`,
        ]
      : [
          `./onnx/${lc}/encoder_flow_quantized.onnx`,
          `https://huggingface.co/${HF_REPO}/resolve/main/onnx/${lc}/encoder_flow_quantized.onnx`,
        ],
    vocoderUrls: isPrimary
      ? [
          './vocoder_quantized.onnx',
          `https://huggingface.co/${HF_REPO}/resolve/main/onnx/sa/vocoder_quantized.onnx`,
        ]
      : [
          `./onnx/${lc}/vocoder_quantized.onnx`,
          `https://huggingface.co/${HF_REPO}/resolve/main/onnx/${lc}/vocoder_quantized.onnx`,
        ],
  };
}

async function getModelsManifest() {
  if (manifestPromise) return manifestPromise;
  manifestPromise = (async () => {
    const fallbackLanguages = {};
    for (const lc of SUPPORTED_LANGS) {
      fallbackLanguages[lc] = buildFallbackLangSpec(lc);
    }
    try {
      const res = await fetch('./models_manifest.json', { cache: 'no-cache' });
      if (res.ok) {
        const json = await res.json();
        if (json?.cacheName) {
          activeCacheName = String(json.cacheName);
        }
        const merged = { ...fallbackLanguages };
        if (json?.languages && typeof json.languages === 'object') {
          for (const lc of SUPPORTED_LANGS) {
            if (json.languages[lc]) {
              merged[lc] = { ...fallbackLanguages[lc], ...json.languages[lc] };
            }
          }
        }
        return {
          version: json?.version || '2.0.0',
          cacheName: activeCacheName,
          languages: merged,
        };
      }
    } catch {
      // Fallback to built-in URL map
    }
    return {
      version: '2.0.0',
      cacheName: activeCacheName,
      languages: fallbackLanguages,
    };
  })();
  return manifestPromise;
}

function isValidOnnxBuffer(arr, expectedBytes) {
  if (!arr || arr.byteLength < 1000000 || arr[0] !== 0x08) return false;
  if (expectedBytes && expectedBytes > 0 && arr.byteLength !== expectedBytes) return false;
  return true;
}

async function pruneStaleModelCaches() {
  if (cachesPruned) return;
  cachesPruned = true;
  if (typeof caches === 'undefined' || !caches?.keys) return;
  try {
    const keys = await caches.keys();
    for (const k of keys) {
      if (k.startsWith(MODEL_CACHE_PREFIX) && k !== activeCacheName) {
        await caches.delete(k).catch(() => {});
      }
    }
  } catch {
    // Ignore storage enumeration restrictions
  }
}

function openModelIdb() {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined' || !indexedDB?.open) {
      reject(new Error('IndexedDB unavailable'));
      return;
    }
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('IndexedDB open failed'));
  });
}

async function readCachedModel(cacheKey, expectedBytes) {
  // Primary persistent cache: Browser CacheStorage API
  if (typeof caches !== 'undefined' && caches?.open) {
    try {
      const cache = await caches.open(activeCacheName);
      const cacheUrl = new URL(`./__ninada_cache__/${encodeURIComponent(cacheKey)}`, self.location.href).href;
      const hit = await cache.match(cacheUrl);
      if (hit) {
        const arr = new Uint8Array(await hit.arrayBuffer());
        if (isValidOnnxBuffer(arr, expectedBytes)) {
          return { bytes: arr, backend: 'cache_storage' };
        }
        await cache.delete(cacheUrl).catch(() => {});
      }
    } catch {
      // Fall through to IndexedDB
    }
  }

  // Fallback persistent cache: IndexedDB
  try {
    const db = await openModelIdb();
    const raw = await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readonly');
      const getReq = tx.objectStore(IDB_STORE).get(cacheKey);
      getReq.onsuccess = () => resolve(getReq.result || null);
      getReq.onerror = () => reject(getReq.error);
    });
    db.close();
    if (raw) {
      const ab = raw instanceof Blob ? await raw.arrayBuffer() : raw;
      const arr = ab instanceof Uint8Array ? ab : new Uint8Array(ab);
      if (isValidOnnxBuffer(arr, expectedBytes)) {
        return { bytes: arr, backend: 'indexeddb' };
      }
    }
  } catch {
    // Ignore IndexedDB unavailability
  }
  return null;
}

async function writeCachedModel(cacheKey, bytes) {
  if (typeof caches !== 'undefined' && caches?.open) {
    try {
      const cache = await caches.open(activeCacheName);
      const cacheUrl = new URL(`./__ninada_cache__/${encodeURIComponent(cacheKey)}`, self.location.href).href;
      await cache.put(
        cacheUrl,
        new Response(bytes, {
          headers: {
            'Content-Type': 'application/octet-stream',
            'Content-Length': String(bytes.byteLength),
            'Cache-Control': 'public, max-age=31536000, immutable',
          },
        })
      );
      return 'cache_storage';
    } catch {
      // Fall through to IndexedDB if CacheStorage fails
    }
  }

  try {
    const db = await openModelIdb();
    const blob = new Blob([bytes], { type: 'application/octet-stream' });
    await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readwrite');
      tx.objectStore(IDB_STORE).put(blob, cacheKey);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    db.close();
    return 'indexeddb';
  } catch {
    return null;
  }
}

async function fetchOrLoadModel(cacheKey, urls, expectedBytes, onDelta) {
  const cached = await readCachedModel(cacheKey, expectedBytes);
  if (cached) {
    onDelta(cached.bytes.byteLength, true);
    return {
      bytes: cached.bytes,
      fromCache: true,
      cacheBackend: cached.backend,
    };
  }

  for (const url of urls) {
    try {
      const res = await fetch(url);
      if (!res.ok) continue;
      // Note: CDNs like GitHub Pages / Fastly / Hugging Face send Content-Encoding: gzip
      // where Content-Length is the compressed wire size while reader.read() yields
      // decompressed ONNX protobuf bytes.
      const hasContentEncoding = Boolean(res.headers.get('content-encoding'));
      const headerLen = Number(res.headers.get('content-length')) || 0;
      const initialCap =
        !hasContentEncoding && headerLen > 0
          ? headerLen
          : Math.max(expectedBytes || 0, headerLen, 1048576);

      if (!res.body) {
        const ab = await res.arrayBuffer();
        const arr = new Uint8Array(ab);
        if (!isValidOnnxBuffer(arr, expectedBytes)) continue;
        onDelta(arr.byteLength, false);
        const cacheBackend = await writeCachedModel(cacheKey, arr);
        return { bytes: arr, fromCache: false, cacheBackend };
      }

      const reader = res.body.getReader();
      let out = new Uint8Array(initialCap);
      let offset = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const needed = offset + value.byteLength;
        if (needed > out.byteLength) {
          const nextCap = Math.max(needed, Math.ceil(out.byteLength * 1.5));
          const grown = new Uint8Array(nextCap);
          grown.set(out.subarray(0, offset), 0);
          out = grown;
        }
        out.set(value, offset);
        offset += value.byteLength;
        onDelta(value.byteLength, false);
      }

      const finalArr = offset === out.byteLength ? out : out.subarray(0, offset);
      if (!isValidOnnxBuffer(finalArr, expectedBytes)) continue;
      const cacheBackend = await writeCachedModel(cacheKey, finalArr);
      return { bytes: finalArr, fromCache: false, cacheBackend };
    } catch {
      // Try next mirror
    }
  }
  return null;
}

const SESSION_OPTS = {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
  enableCpuMemArena: false,
  enableMemPattern: false,
};

async function ensureLangReady(rawLang = 'sa', { emitReady = true } = {}) {
  const lc = SUPPORTED_LANGS.includes(rawLang) ? rawLang : 'sa';

  if (voiceEngine && currentLoadedLang === lc) {
    if (emitReady && lastReadyPayload) {
      self.postMessage({
        ...lastReadyPayload,
        lang: lc,
        fromCache: true,
      });
    }
    return voiceEngine;
  }

  // Release previous language specialist WASM sessions before allocating the new one
  if (voiceEngine?.dispose) {
    try {
      await voiceEngine.dispose();
    } catch {
      // Ignore release errors
    }
  }
  voiceEngine = null;
  currentLoadedLang = null;

  if (!ort || !ort.InferenceSession) {
    voiceEngine = await ninada(lc);
    currentLoadedLang = lc;
    lastReadyPayload = {
      type: 'ready',
      lang: lc,
      isNeural: false,
      isStreamingNeural: false,
      totalMB: 0,
      fromCache: false,
      cachedPersisted: false,
      cacheBackend: null,
      reason: 'ONNX Runtime WASM unavailable',
    };
    if (emitReady) self.postMessage(lastReadyPayload);
    return voiceEngine;
  }

  if (!ortConfigured) {
    ortConfigured = true;
    ort.env.wasm.wasmPaths = new URL('./ort/', import.meta.url).href;
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads =
      typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated
        ? Math.min(4, self.navigator?.hardwareConcurrency || 2)
        : 1;
  }

  const manifest = await getModelsManifest();
  await pruneStaleModelCaches();

  const spec = manifest.languages[lc] || buildFallbackLangSpec(lc);
  const encExpected = Number(spec.encoderBytes) || 0;
  const vocExpected = Number(spec.vocoderBytes) || 0;
  const expectedTotal =
    (encExpected || DEFAULT_ENC_BYTES) + (vocExpected || DEFAULT_VOC_BYTES);

  let loadedBytes = 0;
  const reportProgress = (delta, fromCache = false) => {
    loadedBytes += delta;
    const loadedMB = loadedBytes / 1048576;
    const totalMB = expectedTotal / 1048576;
    const pct = Math.min(97, Math.round((loadedBytes / expectedTotal) * 100));
    self.postMessage({
      type: 'progress',
      lang: lc,
      phase: fromCache ? 'cache' : 'download',
      loadedMB,
      totalMB,
      pct,
      fromCache,
    });
  };

  try {
    // Stage 1: Load from persistent cache (or stream & persist), compile Encoder+Flow, then free JS buffer
    let encRes = await fetchOrLoadModel(
      spec.encoderCacheKey,
      spec.encoderUrls,
      encExpected,
      reportProgress
    );
    if (encRes && encRes.bytes) {
      const encLen = encRes.bytes.byteLength;
      const encFromCache = encRes.fromCache;
      const encBackend = encRes.cacheBackend;
      self.postMessage({
        type: 'progress',
        lang: lc,
        phase: 'compile',
        loadedMB: encLen / 1048576,
        totalMB: expectedTotal / 1048576,
        pct: 52,
        fromCache: encFromCache,
      });
      const encSession = await ort.InferenceSession.create(encRes.bytes, SESSION_OPTS);
      encRes.bytes = null;
      encRes = null;
      await new Promise((r) => setTimeout(r, 8));

      // Stage 2: Load from persistent cache (or stream & persist), compile Halo-Chunked HiFi-GAN Vocoder
      let vocRes = await fetchOrLoadModel(
        spec.vocoderCacheKey,
        spec.vocoderUrls,
        vocExpected,
        reportProgress
      );
      if (vocRes && vocRes.bytes) {
        const totalMB = (encLen + vocRes.bytes.byteLength) / 1048576;
        const allFromCache = Boolean(encFromCache && vocRes.fromCache);
        const cachedPersisted = Boolean(
          (encFromCache || encBackend) && (vocRes.fromCache || vocRes.cacheBackend)
        );
        const cacheBackend = vocRes.cacheBackend || encBackend || null;
        self.postMessage({
          type: 'progress',
          lang: lc,
          phase: 'compile',
          loadedMB: totalMB,
          totalMB,
          pct: 99,
          fromCache: allFromCache,
        });
        let vocSession;
        try {
          vocSession = await ort.InferenceSession.create(vocRes.bytes, SESSION_OPTS);
        } catch (vocErr) {
          try { await encSession.release(); } catch {}
          throw vocErr;
        }
        vocRes.bytes = null;
        vocRes = null;

        voiceEngine = await ninada(lc, {
          encoderSession: encSession,
          vocoderSession: vocSession,
          ort,
        });
        currentLoadedLang = lc;
        lastReadyPayload = {
          type: 'ready',
          lang: lc,
          isNeural: voiceEngine.isNeural,
          isStreamingNeural: voiceEngine.isStreamingNeural,
          totalMB,
          fromCache: allFromCache,
          cachedPersisted,
          cacheBackend,
        };
        if (emitReady) self.postMessage(lastReadyPayload);
        return voiceEngine;
      }
      try { await encSession.release(); } catch {}
    }
  } catch (err) {
    voiceEngine = await ninada(lc);
    currentLoadedLang = lc;
    lastReadyPayload = {
      type: 'ready',
      lang: lc,
      isNeural: false,
      isStreamingNeural: false,
      totalMB: 0,
      fromCache: false,
      cachedPersisted: false,
      cacheBackend: null,
      reason: `WASM fallback (${err?.message || 'memory limit'}) — running Klatt Tier-2`,
    };
    if (emitReady) self.postMessage(lastReadyPayload);
    return voiceEngine;
  }

  voiceEngine = await ninada(lc);
  currentLoadedLang = lc;
  lastReadyPayload = {
    type: 'ready',
    lang: lc,
    isNeural: false,
    isStreamingNeural: false,
    totalMB: 0,
    fromCache: false,
    cachedPersisted: false,
    cacheBackend: null,
    reason: 'Model weights not found — running Klatt Tier-2',
  };
  if (emitReady) self.postMessage(lastReadyPayload);
  return voiceEngine;
}

self.addEventListener('message', (ev) => {
  const msg = ev.data || {};
  if (msg.type === 'init' || msg.type === 'load_lang') {
    const targetLang = msg.lang || 'sa';
    if (activeAbort) {
      activeAbort.abort();
    }
    runQueue = runQueue
      .then(() => ensureLangReady(targetLang, { emitReady: true }))
      .catch((err) => {
        self.postMessage({ type: 'error', reqId: 0, message: err?.message || String(err) });
      });
    return;
  }

  if (msg.type === 'cancel') {
    if (activeAbort && (!msg.reqId || msg.reqId === activeReqId)) {
      activeAbort.abort();
    }
    return;
  }

  if (msg.type === 'stream') {
    const reqId = Number(msg.reqId || Date.now());
    const targetLang = SUPPORTED_LANGS.includes(msg.lang) ? msg.lang : 'sa';
    const tier = msg.tier || 'auto';
    if (activeAbort) {
      activeAbort.abort();
    }
    const controller = new AbortController();
    activeAbort = controller;
    activeReqId = reqId;

    runQueue = runQueue
      .then(async () => {
        if (controller.signal.aborted) return;
        let engine;
        if (tier === 'klatt' && voiceEngine) {
          engine = voiceEngine;
        } else {
          const needsLoad = currentLoadedLang !== targetLang || !voiceEngine;
          engine = await ensureLangReady(targetLang, { emitReady: needsLoad });
        }
        if (controller.signal.aborted) return;

        const t0 = performance.now();
        let ttfaMs = null;
        let chunkCount = 0;
        let totalSamples = 0;
        let finalTier = 'klatt_zero_dep';
        let finalLang = targetLang;

        for await (const ch of engine.stream(msg.text || '', {
          lang: targetLang,
          rate: Number(msg.rate ?? 1.0),
          noise: Number(msg.noise ?? 0.667),
          noiseDuration: 0.8,
          tier,
          firstChunkFrames: 48,
          chunkFrames: Number(msg.chunkFrames || 96),
          haloFrames: 16,
          crossfadeSamples: 256,
          signal: controller.signal,
        })) {
          if (controller.signal.aborted) return;
          if (ttfaMs === null) ttfaMs = performance.now() - t0;
          chunkCount += 1;
          totalSamples += ch.pcm.length;
          finalTier = ch.tier;
          finalLang = ch.lang;

          const pcmTransfer =
            ch.pcm.byteOffset === 0 && ch.pcm.byteLength === ch.pcm.buffer.byteLength
              ? ch.pcm
              : ch.pcm.slice();

          self.postMessage(
            {
              type: 'chunk',
              reqId,
              pcm: pcmTransfer,
              sampleRate: ch.sampleRate,
              index: ch.index,
              clauseIndex: ch.clauseIndex,
              clauseCount: ch.clauseCount,
              isFinal: ch.isFinal,
              durationSec: ch.durationSec,
              latencyMs: ch.latencyMs,
              ttfaMs,
              lang: ch.lang,
              script: ch.script,
              tier: ch.tier,
              aksharas: ch.aksharas,
              durations: ch.durations,
            },
            [pcmTransfer.buffer]
          );
        }

        if (controller.signal.aborted) return;
        const totalMs = performance.now() - t0;
        self.postMessage({
          type: 'done',
          reqId,
          totalMs,
          ttfaMs: ttfaMs ?? totalMs,
          chunkCount,
          totalSamples,
          durationSec: totalSamples / 16000,
          tier: finalTier,
          isNeural: engine.isNeural,
          isStreamingNeural: engine.isStreamingNeural,
          lang: finalLang,
        });
      })
      .catch((err) => {
        if (!controller.signal.aborted) {
          self.postMessage({
            type: 'error',
            reqId,
            message: err?.message || String(err),
          });
        }
      });
  }
});
