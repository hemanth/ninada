import * as ortNs from './ort/ort.wasm.min.mjs';
import ninada, { SAMPLE_RATE } from './ninada.mjs';

const ort = ortNs?.default?.InferenceSession ? ortNs.default : ortNs;

const HF_REPO = 'gnumanth/ninada';
const SUPPORTED_LANGS = ['sa', 'kn', 'hi', 'ta', 'te', 'ml'];

const DEFAULT_MODEL_CACHE_NAME = 'ninada-onnx-v3.0.0';
const MODEL_CACHE_PREFIX = 'ninada-onnx-';
const IDB_NAME = 'ninada-onnx-idb-v3';
const IDB_STORE = 'models';

const DEFAULT_MODEL_BYTES = 96580000;
const VOICE_BANK_BYTES = 510 * 256 * 4; // 522,240 bytes

let activeCacheName = DEFAULT_MODEL_CACHE_NAME;
let manifestPromise = null;
let cachesPruned = false;
let ortConfigured = false;

let sharedSession = null;
let sharedModelMB = 0;
let sharedModelFromCache = false;
let sharedModelBackend = null;
let voiceEngine = null;
const loadedVoiceLangs = new Set();
let currentLoadedLang = null;
let lastReadyPayload = null;
let activeAbort = null;
let activeReqId = 0;
let runQueue = Promise.resolve();

function buildFallbackLangSpec(lc) {
  return {
    lang: lc,
    voiceBytes: VOICE_BANK_BYTES,
    voiceCacheKey: `ninada_v3_voice_${lc}_${VOICE_BANK_BYTES}.bin`,
    voiceUrls: [
      `./voices/${lc}.bin`,
      `https://huggingface.co/${HF_REPO}/resolve/main/voices/${lc}.bin`,
    ],
  };
}

function buildFallbackUnifiedSpec() {
  return {
    cacheKey: 'ninada_v3_24k_q8f16.onnx',
    expectedBytes: 0,
    totalMB: 92.1,
    urls: [
      './ninada_24k_q8f16.onnx',
      `https://huggingface.co/${HF_REPO}/resolve/main/onnx/ninada_24k_q8f16.onnx`,
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
    const fallbackUnified = buildFallbackUnifiedSpec();
    try {
      const res = await fetch('./models_manifest.json', { cache: 'no-cache' });
      if (res.ok) {
        const json = await res.json();
        if (json?.cacheName) {
          activeCacheName = String(json.cacheName);
        }
        const mergedLangs = { ...fallbackLanguages };
        if (json?.languages && typeof json.languages === 'object') {
          for (const lc of SUPPORTED_LANGS) {
            if (json.languages[lc]) {
              mergedLangs[lc] = { ...fallbackLanguages[lc], ...json.languages[lc] };
            }
          }
        }
        const unifiedModel = json?.unifiedModel
          ? { ...fallbackUnified, ...json.unifiedModel }
          : fallbackUnified;
        return {
          version: json?.version || '3.0.0',
          cacheName: activeCacheName,
          unifiedModel,
          languages: mergedLangs,
        };
      }
    } catch {
      // Fallback to built-in URL map
    }
    return {
      version: '3.0.0',
      cacheName: activeCacheName,
      unifiedModel: fallbackUnified,
      languages: fallbackLanguages,
    };
  })();
  return manifestPromise;
}

function isValidOnnxBuffer(arr, expectedBytes) {
  if (!arr || arr.byteLength < 10000000 || arr[0] !== 0x08) return false;
  if (expectedBytes && expectedBytes > 0 && arr.byteLength !== expectedBytes) return false;
  return true;
}

function isValidVoiceBuffer(arr, expectedBytes = VOICE_BANK_BYTES) {
  if (!arr) return false;
  return arr.byteLength === (expectedBytes || VOICE_BANK_BYTES);
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

async function readCachedAsset(cacheKey, validator) {
  if (typeof caches !== 'undefined' && caches?.open) {
    try {
      const cache = await caches.open(activeCacheName);
      const cacheUrl = new URL(`./__ninada_cache__/${encodeURIComponent(cacheKey)}`, self.location.href).href;
      const hit = await cache.match(cacheUrl);
      if (hit) {
        const arr = new Uint8Array(await hit.arrayBuffer());
        if (validator(arr)) {
          return { bytes: arr, backend: 'cache_storage' };
        }
        await cache.delete(cacheUrl).catch(() => {});
      }
    } catch {
      // Fall through to IndexedDB
    }
  }

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
      if (validator(arr)) {
        return { bytes: arr, backend: 'indexeddb' };
      }
    }
  } catch {
    // Ignore IndexedDB unavailability
  }
  return null;
}

async function writeCachedAsset(cacheKey, bytes) {
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

async function fetchOrLoadAsset(cacheKey, urls, expectedBytes, validator, onDelta) {
  const cached = await readCachedAsset(cacheKey, validator);
  if (cached) {
    onDelta?.(cached.bytes.byteLength, true);
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
      const hasContentEncoding = Boolean(res.headers.get('content-encoding'));
      const headerLen = Number(res.headers.get('content-length')) || 0;
      const initialCap =
        !hasContentEncoding && headerLen > 0
          ? headerLen
          : Math.max(expectedBytes || 0, headerLen, 524288);

      if (!res.body) {
        const ab = await res.arrayBuffer();
        const arr = new Uint8Array(ab);
        if (!validator(arr)) continue;
        onDelta?.(arr.byteLength, false);
        const cacheBackend = await writeCachedAsset(cacheKey, arr);
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
        onDelta?.(value.byteLength, false);
      }

      const finalArr = offset === out.byteLength ? out : out.subarray(0, offset);
      if (!validator(finalArr)) continue;
      const cacheBackend = await writeCachedAsset(cacheKey, finalArr);
      return { bytes: finalArr, fromCache: false, cacheBackend };
    } catch {
      // Try next mirror
    }
  }
  return null;
}

function buildSessionOpts() {
  const isCoi = typeof crossOriginIsolated !== 'undefined' && Boolean(crossOriginIsolated);
  return {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    enableCpuMemArena: isCoi,
    enableMemPattern: isCoi,
  };
}

async function warmupUnifiedSession(session, styleSlice = null) {
  try {
    const warmIds = new ort.Tensor('int64', BigInt64Array.from([0n, 86n, 4n, 0n]), [1, 4]);
    const warmStyle = new ort.Tensor(
      'float32',
      styleSlice ? new Float32Array(styleSlice.subarray(0, 256)) : new Float32Array(256),
      [1, 256]
    );
    const warmSpeed = new ort.Tensor('float32', Float32Array.from([1.0]), [1]);
    const out = await session.run({
      input_ids: warmIds,
      style: warmStyle,
      speed: warmSpeed,
    });
    warmIds.dispose?.();
    warmStyle.dispose?.();
    warmSpeed.dispose?.();
    out.waveform?.dispose?.();
    out.durations?.dispose?.();
  } catch {
    // Non-fatal JIT warmup
  }
}

async function ensureLangReady(rawLang = 'sa', { emitReady = true } = {}) {
  const lc = SUPPORTED_LANGS.includes(rawLang) ? rawLang : 'sa';

  if (voiceEngine && sharedSession && loadedVoiceLangs.has(lc)) {
    currentLoadedLang = lc;
    const isCoi = typeof crossOriginIsolated !== 'undefined' && Boolean(crossOriginIsolated);
    const activeThreads = ort?.env?.wasm?.numThreads || 1;
    lastReadyPayload = {
      type: 'ready',
      lang: lc,
      isNeural: true,
      isStreamingNeural: true,
      totalMB: sharedModelMB + VOICE_BANK_BYTES / 1048576,
      fromCache: true,
      cachedPersisted: true,
      cacheBackend: sharedModelBackend || 'cache_storage',
      threads: activeThreads,
      crossOriginIsolated: isCoi,
    };
    if (emitReady) self.postMessage(lastReadyPayload);
    return voiceEngine;
  }

  if (!ort || !ort.InferenceSession) {
    if (!voiceEngine) voiceEngine = await ninada(lc);
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

  const isCoi = typeof crossOriginIsolated !== 'undefined' && Boolean(crossOriginIsolated);
  if (!ortConfigured) {
    ortConfigured = true;
    ort.env.wasm.wasmPaths = new URL('./ort/', import.meta.url).href;
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = isCoi
      ? Math.min(4, Math.max(2, (self.navigator?.hardwareConcurrency || 4) - 1))
      : 1;
  }
  const activeThreads = ort.env.wasm.numThreads || 1;
  const sessionOpts = buildSessionOpts();

  const manifest = await getModelsManifest();
  await pruneStaleModelCaches();

  const unifiedSpec = manifest.unifiedModel;
  const langSpec = manifest.languages[lc] || buildFallbackLangSpec(lc);
  const needUnifiedLoad = !sharedSession;
  const expectedModelBytes = Number(unifiedSpec.expectedBytes) || DEFAULT_MODEL_BYTES;
  const expectedVoiceBytes = Number(langSpec.voiceBytes) || VOICE_BANK_BYTES;
  const expectedTotal = (needUnifiedLoad ? expectedModelBytes : 0) + expectedVoiceBytes;

  let loadedBytes = 0;
  const reportProgress = (delta, fromCache = false) => {
    loadedBytes += delta;
    const loadedMB = loadedBytes / 1048576;
    const totalMB = expectedTotal / 1048576;
    const pct = Math.min(96, Math.round((loadedBytes / Math.max(1, expectedTotal)) * 100));
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
    let modelJustLoadedFromCache = true;
    if (!sharedSession) {
      let modelRes = await fetchOrLoadAsset(
        unifiedSpec.cacheKey,
        unifiedSpec.urls,
        Number(unifiedSpec.expectedBytes) || 0,
        (arr) => isValidOnnxBuffer(arr, Number(unifiedSpec.expectedBytes) || 0),
        reportProgress
      );
      if (!modelRes || !modelRes.bytes) {
        throw new Error('24 kHz ONNX model asset not found');
      }
      sharedModelMB = modelRes.bytes.byteLength / 1048576;
      sharedModelFromCache = Boolean(modelRes.fromCache);
      sharedModelBackend = modelRes.cacheBackend;
      modelJustLoadedFromCache = Boolean(modelRes.fromCache);

      self.postMessage({
        type: 'progress',
        lang: lc,
        phase: 'compile',
        loadedMB: sharedModelMB,
        totalMB: sharedModelMB + expectedVoiceBytes / 1048576,
        pct: 94,
        fromCache: sharedModelFromCache,
      });

      sharedSession = await ort.InferenceSession.create(modelRes.bytes, sessionOpts);
      modelRes.bytes = null;
      modelRes = null;
      await new Promise((r) => setTimeout(r, 4));
    }

    let voiceRes = await fetchOrLoadAsset(
      langSpec.voiceCacheKey,
      langSpec.voiceUrls,
      expectedVoiceBytes,
      (arr) => isValidVoiceBuffer(arr, expectedVoiceBytes),
      reportProgress
    );
    if (!voiceRes || !voiceRes.bytes) {
      throw new Error(`Voice bank ${lc}.bin not found`);
    }

    const voiceFloats = new Float32Array(
      voiceRes.bytes.buffer,
      voiceRes.bytes.byteOffset,
      VOICE_BANK_BYTES / 4
    ).slice();
    const voiceFromCache = Boolean(voiceRes.fromCache);
    const voiceBackend = voiceRes.cacheBackend;
    voiceRes.bytes = null;
    voiceRes = null;

    if (!voiceEngine) {
      await warmupUnifiedSession(sharedSession, voiceFloats);
      voiceEngine = await ninada(lc, {
        session: sharedSession,
        voiceBank: voiceFloats,
        ort,
      });
    } else {
      voiceEngine.setVoiceBank(lc, voiceFloats);
    }

    loadedVoiceLangs.add(lc);
    currentLoadedLang = lc;

    const allFromCache = needUnifiedLoad
      ? Boolean(modelJustLoadedFromCache && voiceFromCache)
      : Boolean(voiceFromCache);
    const cachedPersisted = Boolean(
      (sharedModelFromCache || sharedModelBackend) && (voiceFromCache || voiceBackend)
    );

    lastReadyPayload = {
      type: 'ready',
      lang: lc,
      isNeural: voiceEngine.isNeural,
      isStreamingNeural: voiceEngine.isStreamingNeural,
      totalMB: sharedModelMB + expectedVoiceBytes / 1048576,
      fromCache: allFromCache,
      cachedPersisted,
      cacheBackend: voiceBackend || sharedModelBackend || null,
      threads: activeThreads,
      crossOriginIsolated: isCoi,
    };
    if (emitReady) self.postMessage(lastReadyPayload);
    return voiceEngine;
  } catch (err) {
    if (!voiceEngine) {
      voiceEngine = await ninada(lc);
    }
    currentLoadedLang = lc;
    lastReadyPayload = {
      type: 'ready',
      lang: lc,
      isNeural: Boolean(sharedSession),
      isStreamingNeural: Boolean(sharedSession),
      totalMB: 0,
      fromCache: false,
      cachedPersisted: false,
      cacheBackend: null,
      reason: `WASM fallback (${err?.message || 'memory limit'}) — running Klatt Tier-2`,
    };
    if (emitReady) self.postMessage(lastReadyPayload);
    return voiceEngine;
  }
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
          const needsLoad = !loadedVoiceLangs.has(targetLang) || !voiceEngine;
          engine = await ensureLangReady(targetLang, { emitReady: needsLoad });
        }
        if (controller.signal.aborted) return;

        const t0 = performance.now();
        let ttfaMs = null;
        let chunkCount = 0;
        let totalSamples = 0;
        let finalTier = 'klatt_zero_dep';
        let finalLang = targetLang;

        const workerIsCoi = typeof crossOriginIsolated !== 'undefined' && Boolean(crossOriginIsolated);
        const defaultChunkFrames = workerIsCoi ? 64 : 96;
        const defaultFirstFrames = workerIsCoi ? 40 : 56;

        for await (const ch of engine.stream(msg.text || '', {
          lang: targetLang,
          rate: Number(msg.rate ?? 1.0),
          noise: Number(msg.noise ?? 0.667),
          noiseDuration: Number(msg.noiseDuration ?? msg.noise_duration ?? 0.8),
          tier,
          firstChunkFrames: Number(msg.firstChunkFrames || defaultFirstFrames),
          chunkFrames: Number(msg.chunkFrames || defaultChunkFrames),
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
              sampleRate: ch.sampleRate || SAMPLE_RATE,
              index: ch.index,
              clauseIndex: ch.clauseIndex,
              clauseCount: ch.clauseCount,
              isFinal: ch.isFinal,
              durationSec: ch.durationSec,
              clauseDurationSec: ch.clauseDurationSec ?? ch.durationSec,
              latencyMs: ch.latencyMs,
              vocoderMs: ch.vocoderMs ?? ch.latencyMs,
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
          durationSec: totalSamples / SAMPLE_RATE,
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
