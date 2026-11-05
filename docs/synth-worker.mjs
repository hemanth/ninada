import ninada, { SAMPLE_RATE, createSplitStudioVitsSession } from './ninada.mjs?v=5.1.2';

const HF_REPO = 'gnumanth/ninada';
const SUPPORTED_LANGS = ['en', 'sa', 'kn', 'hi', 'ta', 'te', 'ml', 'mr', 'bn', 'gu', 'pa', 'or'];
const LOCAL_BUNDLED_LANGS = new Set(['en', 'sa', 'kn', 'hi', 'ta', 'te', 'ml']);

const DEFAULT_MODEL_CACHE_NAME = 'ninada-onnx-v5.1.1';
const MODEL_CACHE_PREFIX = 'ninada-onnx-';
const IDB_NAME = 'ninada-onnx-idb-v511';
const LEGACY_IDB_NAMES = [
  'ninada-onnx-idb-v3',
  'ninada-onnx-idb-v31',
  'ninada-onnx-idb-v32',
  'ninada-onnx-idb-v40',
  'ninada-onnx-idb-v402',
  'ninada-onnx-idb-v500',
  'ninada-onnx-idb-v501',
  'ninada-onnx-idb-v510',
];
const IDB_STORE = 'models';

const DEFAULT_STUDIO_SPECIALIST_BYTES = 29285000;
const DEFAULT_MOBILE_SPECIALIST_BYTES = 22740000;
const IS_IOS_WORKER = Boolean(
  /iPhone|iPad|iPod|CriOS|FxiOS|EdgiOS/i.test(self.navigator?.userAgent || '') ||
    (self.navigator?.platform === 'MacIntel' && (self.navigator?.maxTouchPoints || 0) > 1)
);
const IS_MOBILE_OR_LOW_MEM = Boolean(
  IS_IOS_WORKER ||
    (self.navigator?.deviceMemory && self.navigator.deviceMemory <= 4) ||
    /Android|iPhone|iPad|iPod|Mobile/i.test(self.navigator?.userAgent || '')
);
const MAX_WARM_SESSIONS = IS_MOBILE_OR_LOW_MEM ? 1 : 2;
const WORKER_PARAMS = (() => {
  try {
    return new URL(self.location.href).searchParams;
  } catch {
    return new URLSearchParams();
  }
})();
const FORCE_NO_GPU = WORKER_PARAMS.get('gpu') === '0';
const FORCE_GPU = WORKER_PARAMS.get('gpu') === '1';

let activeCacheName = DEFAULT_MODEL_CACHE_NAME;
let manifestPromise = null;
let cachesPruned = false;

let ort = null;
let ortBackendMode = 'none'; // 'webgpu' | 'wasm' | 'none'
// On Android Chrome & iOS Safari, WebGPU compiles ~600 HiFi-GAN WGSL/Vulkan shaders per shape ([44] + [112] = 6-10s stall!),
// whereas 4-thread ARM Neon WASM SIMD compiles in ~90ms and streams speech in ~350-650ms with 0 underruns.
let webGpuDisabled = FORCE_NO_GPU || (IS_MOBILE_OR_LOW_MEM && !FORCE_GPU);
let ortResolvePromise = null;
// Kick off ORT WASM runtime initialization immediately at Worker module load (t=0ms)
resolveWorkerOrt().catch(() => {});

// Active InferenceSession LRU keyed by `${lang}:${voiceKey}` (`sa:male`, `kn:male`, `kn:female`, ...)
const warmSessions = new Map();
let voiceEngine = null;
let currentLoadedLang = null;
let currentLoadedVoice = 'male';
let lastReadyPayload = null;
let activeAbort = null;
let activeReqId = 0;
let runQueue = Promise.resolve();

function unwrapOrtNamespace(ortNs) {
  return ortNs?.default?.InferenceSession ? ortNs.default : ortNs;
}

function configureOrtWasmEnv(ortInstance, { forWebGpuHybrid = false } = {}) {
  if (!ortInstance?.env?.wasm) return;
  const isCoi =
    typeof crossOriginIsolated !== 'undefined' &&
    Boolean(crossOriginIsolated) &&
    WORKER_PARAMS.get('coi') !== '0';
  ortInstance.env.wasm.wasmPaths = new URL('./ort/', import.meta.url).href;
  ortInstance.env.wasm.proxy = false;
  if (forWebGpuHybrid) {
    // Hybrid mode runs vocoder on WebGPU and compact prior/flow on <=2 WASM threads (avoids mobile little-core contention)
    ortInstance.env.wasm.numThreads = isCoi ? 2 : 1;
  } else {
    const cores = self.navigator?.hardwareConcurrency || 4;
    // Measured on a Pixel 10 Pro XL (8-core big.LITTLE): vocoder RTF 1 thr = 1.55, 2 thr = 3.6, 3 thr = 2.4,
    // 6 thr = 1.3. Small thread pools stall on per-op barriers waiting for little cores, so phones use either
    // a wide pool (8+ cores) or a single thread. iOS/iPadOS WebKit always uses 1 thread to avoid pthread Jetsam limits.
    const mobileThreads = IS_IOS_WORKER ? 1 : cores >= 8 ? Math.min(6, cores - 2) : 1;
    ortInstance.env.wasm.numThreads = isCoi
      ? IS_MOBILE_OR_LOW_MEM
        ? mobileThreads
        : Math.min(4, Math.max(2, cores - 1))
      : 1;
  }
}

async function resolveWorkerOrt({ forcePureWasm = false } = {}) {
  if (forcePureWasm) {
    webGpuDisabled = true;
    if (ortBackendMode === 'wasm' && ort?.InferenceSession) {
      return ort;
    }
    ortResolvePromise = null;
  }
  if (ort?.InferenceSession && (!forcePureWasm || ortBackendMode === 'wasm')) {
    return ort;
  }
  if (ortResolvePromise) return ortResolvePromise;

  ortResolvePromise = (async () => {
    if (!webGpuDisabled && self.navigator?.gpu?.requestAdapter) {
      try {
        const adapter = await Promise.race([
          self.navigator.gpu.requestAdapter(),
          new Promise((r) => setTimeout(() => r(null), 1500)),
        ]);
        if (adapter && !adapter.info?.isFallbackAdapter && !adapter.isFallbackAdapter) {
          const hasJspi = typeof WebAssembly !== 'undefined' && typeof WebAssembly.promising === 'function';
          const workerCoi = typeof crossOriginIsolated !== 'undefined' && Boolean(crossOriginIsolated);
          const isUltraLowRam = Boolean(self.navigator?.deviceMemory && self.navigator.deviceMemory <= 2);
          // On COI mobile without JSPI, 14 MB 4-thread WASM SIMD (<500ms) avoids 27 MB asyncify download;
          // on non-COI in-app WebViews (!workerCoi), enable WebGPU so speech streams in ~80-150ms with 0 underruns!
          if (!isUltraLowRam && (!IS_MOBILE_OR_LOW_MEM || hasJspi || !workerCoi)) {
            const gpuEntry = hasJspi ? './ort/ort.jspi.min.mjs' : './ort/ort.webgpu.min.mjs';
            const gpuNs = await import(gpuEntry);
            const gpuOrt = unwrapOrtNamespace(gpuNs);
            if (gpuOrt?.InferenceSession) {
              configureOrtWasmEnv(gpuOrt, { forWebGpuHybrid: true });
              ort = gpuOrt;
              ortBackendMode = 'webgpu';
              return ort;
            }
          }
        }
      } catch {
        // Fall through to pure WASM runtime
      }
      webGpuDisabled = true;
    }

    try {
      const wasmNs = await import('./ort/ort.wasm.min.mjs');
      const wasmOrt = unwrapOrtNamespace(wasmNs);
      if (wasmOrt?.InferenceSession) {
        configureOrtWasmEnv(wasmOrt, { forWebGpuHybrid: false });
        ort = wasmOrt;
        ortBackendMode = 'wasm';
        return ort;
      }
    } catch {
      ort = null;
      ortBackendMode = 'none';
    }
    return null;
  })();

  return ortResolvePromise;
}

async function switchToPureWasmOrt() {
  webGpuDisabled = true;
  for (const [, entry] of warmSessions) {
    try {
      await entry?.session?.release?.();
    } catch {}
  }
  warmSessions.clear();
  voiceEngine = null;
  ort = null;
  ortBackendMode = 'none';
  ortResolvePromise = null;
  return resolveWorkerOrt({ forcePureWasm: true });
}

function normalizeVoiceKey(voice) {
  return typeof voice === 'string' && voice.trim().toLowerCase() === 'female' ? 'female' : 'male';
}

function shouldUseMobileWeights() {
  const pref = (WORKER_PARAMS.get('weights') || '').trim().toLowerCase();
  if (pref === 'mobile' || pref === 'istft') return true;
  if (pref === 'studio' || pref === 'hifigan') return false;
  if (IS_MOBILE_OR_LOW_MEM) return true;
  const cores = self.navigator?.hardwareConcurrency || 4;
  return ortBackendMode !== 'webgpu' && cores <= 4;
}

function buildFallbackLangSpec(lc) {
  const verQ = lc === 'en' ? '?v=5.1.1' : '';
  const vTag = lc === 'en' ? 'v511' : 'v510';
  const hfMaleUrl = `https://huggingface.co/${HF_REPO}/resolve/main/onnx/${lc}.onnx`;
  const localMaleUrl = `./models/${lc}.onnx${verQ}`;
  const maleUrls = LOCAL_BUNDLED_LANGS.has(lc) ? [localMaleUrl, hfMaleUrl] : [hfMaleUrl, localMaleUrl];
  const hfMobMaleUrl = `https://huggingface.co/${HF_REPO}/resolve/main/onnx_mobile/${lc}.onnx`;
  const localMobMaleUrl = `./models_mobile/${lc}.onnx${verQ}`;
  return {
    lang: lc,
    cacheKey: `ninada_${vTag}_studio_${lc}_24k_q8f16.onnx`,
    expectedBytes: 0,
    totalMB: 27.93,
    urls: maleUrls,
    femaleCacheKey: `ninada_${vTag}_studio_${lc}_female_24k_q8f16.onnx`,
    femaleExpectedBytes: 0,
    femaleTotalMB: 27.93,
    femaleUrls: [
      `https://huggingface.co/${HF_REPO}/resolve/main/onnx/${lc}_female.onnx`,
      `./models/${lc}_female.onnx${verQ}`,
    ],
    mobileCacheKey: `ninada_${vTag}_mobile_${lc}_24k_q8f16.onnx`,
    mobileExpectedBytes: 0,
    mobileTotalMB: 21.68,
    mobileUrls: [localMobMaleUrl, hfMobMaleUrl, ...maleUrls],
    femaleMobileCacheKey: `ninada_${vTag}_mobile_${lc}_female_24k_q8f16.onnx`,
    femaleMobileExpectedBytes: 0,
    femaleMobileTotalMB: 21.68,
    femaleMobileUrls: [
      `./models_mobile/${lc}_female.onnx${verQ}`,
      `https://huggingface.co/${HF_REPO}/resolve/main/onnx_mobile/${lc}_female.onnx`,
      `https://huggingface.co/${HF_REPO}/resolve/main/onnx/${lc}_female.onnx`,
      `./models/${lc}_female.onnx${verQ}`,
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
      const res = await fetch('./models_manifest.json?v=5.1.2', { cache: 'no-cache' });
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
        return {
          version: json?.version || '5.1.1',
          cacheName: activeCacheName,
          languages: mergedLangs,
        };
      }
    } catch {
      // Fallback to built-in specialist URL map
    }
    return {
      version: '5.1.1',
      cacheName: activeCacheName,
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

async function pruneStaleModelCaches() {
  if (cachesPruned) return;
  cachesPruned = true;
  if (typeof caches !== 'undefined' && caches?.keys) {
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
  if (typeof indexedDB !== 'undefined' && indexedDB?.deleteDatabase) {
    for (const oldDb of LEGACY_IDB_NAMES) {
      try {
        indexedDB.deleteDatabase(oldDb);
      } catch {}
    }
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
      // On low-memory devices, skip duplicate 28 MB Blob allocation if CacheStorage is full or unavailable
      if (IS_MOBILE_OR_LOW_MEM) return null;
    }
  } else if (IS_MOBILE_OR_LOW_MEM) {
    return null;
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
  const useWebGpu = ortBackendMode === 'webgpu' && !webGpuDisabled;
  return {
    executionProviders: ['wasm'],
    vocoderExecutionProviders: useWebGpu ? ['webgpu'] : ['wasm'],
    strictWebGpu: useWebGpu,
    graphOptimizationLevel: 'disabled',
    enableCpuMemArena: false,
    enableMemPattern: false,
  };
}

async function warmupSpecialistSession(session) {
  // Pure WASM requires zero shader warmup; returning immediately eliminates 2-4s of mobile boot lock
  if (!session || session.vocoderBackend !== 'webgpu' || !session.vocoderSession) {
    return;
  }

  // WebGPU: pre-compile BOTH 44-frame first-chunk AND 112-frame steady-chunk WGSL shader pipelines at boot
  // so Sentence 2 / Chunk 1 never pauses mid-stream to compile [1, 192, 112] shaders!
  for (const bucketF of [44, 112]) {
    const zData = new Float32Array(192 * bucketF).fill(0.02);
    const warmLat = new ort.Tensor('float32', zData, [1, 192, bucketF]);
    const vocOut = await session.vocoderSession.run({ [session.latentName]: warmLat });
    warmLat.dispose?.();
    const wavT = vocOut[session.wav3dName];
    const wavData = wavT?.data;
    if (!wavData || wavData.length === 0) {
      wavT?.dispose?.();
      throw new Error('WebGPU vocoder returned empty output');
    }
    const checkN = Math.min(64, wavData.length);
    for (let i = 0; i < checkN; i++) {
      if (!Number.isFinite(wavData[i])) {
        wavT?.dispose?.();
        throw new Error('WebGPU vocoder returned non-finite output');
      }
    }
    wavT?.dispose?.();
  }

  if (IS_MOBILE_OR_LOW_MEM && typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
    const zData = new Float32Array(192 * 44).fill(0.02);
    const probeLat = new ort.Tensor('float32', zData, [1, 192, 44]);
    const t0 = performance.now();
    const probeOut = await session.vocoderSession.run({ [session.latentName]: probeLat });
    const dt = performance.now() - t0;
    probeLat.dispose?.();
    probeOut[session.wav3dName]?.dispose?.();
    if (dt > 260) {
      throw new Error(`Mobile WebGPU vocoder slower than multi-thread WASM (${Math.round(dt)}ms)`);
    }
  }
}

async function loadOrGetWarmSession(lc, voiceKey = 'male', { emitProgress = true } = {}) {
  const normVoice = normalizeVoiceKey(voiceKey);
  await resolveWorkerOrt();
  const useMobileWeights = shouldUseMobileWeights();
  const tierTag = useMobileWeights ? 'mobile' : 'studio';
  const slotKey = `${lc}:${normVoice}:${tierTag}`;

  if (warmSessions.has(slotKey)) {
    const hit = warmSessions.get(slotKey);
    warmSessions.delete(slotKey);
    warmSessions.set(slotKey, hit);
    return {
      ...hit,
      fromCache: true,
      cachedPersisted: true,
    };
  }

  // Pre-evict oldest warm session BEFORE allocating incoming ONNX buffer to minimize peak RSS
  while (warmSessions.size >= MAX_WARM_SESSIONS) {
    const oldestKey = warmSessions.keys().next().value;
    const evicted = warmSessions.get(oldestKey);
    warmSessions.delete(oldestKey);
    try {
      await evicted?.session?.release?.();
    } catch {}
  }

  const isFemale = normVoice === 'female';
  const manifest = await getModelsManifest();
  const langSpec = manifest.languages[lc] || buildFallbackLangSpec(lc);
  setTimeout(() => {
    pruneStaleModelCaches().catch(() => {});
  }, 1500);

  const hasMobileSpec = Boolean(
    useMobileWeights &&
      (isFemale
        ? Array.isArray(langSpec.femaleMobileUrls) && langSpec.femaleMobileUrls.length > 0
        : Array.isArray(langSpec.mobileUrls) && langSpec.mobileUrls.length > 0)
  );
  const cacheKey = hasMobileSpec
    ? isFemale
      ? langSpec.femaleMobileCacheKey || langSpec.femaleCacheKey
      : langSpec.mobileCacheKey || langSpec.cacheKey
    : isFemale
      ? langSpec.femaleCacheKey
      : langSpec.cacheKey;
  const urls = hasMobileSpec
    ? isFemale
      ? langSpec.femaleMobileUrls
      : langSpec.mobileUrls
    : isFemale
      ? langSpec.femaleUrls
      : langSpec.urls;
  const expectedBytes =
    Number(
      hasMobileSpec
        ? isFemale
          ? langSpec.femaleMobileExpectedBytes
          : langSpec.mobileExpectedBytes
        : isFemale
          ? langSpec.femaleExpectedBytes
          : langSpec.expectedBytes
    ) || 0;
  const expectedTotal =
    expectedBytes || (hasMobileSpec ? DEFAULT_MOBILE_SPECIALIST_BYTES : DEFAULT_STUDIO_SPECIALIST_BYTES);

  let loadedBytes = 0;
  const reportProgress = emitProgress
    ? (delta, fromCache = false) => {
        loadedBytes += delta;
        const loadedMB = loadedBytes / 1048576;
        const totalMB = expectedTotal / 1048576;
        const pct = Math.min(93, Math.round((loadedBytes / Math.max(1, expectedTotal)) * 100));
        self.postMessage({
          type: 'progress',
          lang: lc,
          voice: normVoice,
          phase: fromCache ? 'cache' : 'download',
          loadedMB,
          totalMB,
          pct,
          fromCache,
        });
      }
    : null;

  let modelRes = await fetchOrLoadAsset(
    cacheKey,
    urls,
    expectedBytes,
    (arr) => isValidOnnxBuffer(arr, expectedBytes),
    reportProgress
  );

  // If mobile weight isn't cached/published yet, fall back seamlessly to the Studio specialist for the same voice
  if ((!modelRes || !modelRes.bytes) && hasMobileSpec) {
    const studioKey = isFemale ? langSpec.femaleCacheKey : langSpec.cacheKey;
    const studioUrls = isFemale ? langSpec.femaleUrls : langSpec.urls;
    const studioBytes = Number(isFemale ? langSpec.femaleExpectedBytes : langSpec.expectedBytes) || 0;
    modelRes = await fetchOrLoadAsset(
      studioKey,
      studioUrls,
      studioBytes,
      (arr) => isValidOnnxBuffer(arr, studioBytes),
      reportProgress
    );
  }

  // If female specialist is offline/unreachable, fall back cleanly to the native Male specialist for this language
  if ((!modelRes || !modelRes.bytes) && isFemale) {
    return loadOrGetWarmSession(lc, 'male', { emitProgress });
  }
  if (!modelRes || !modelRes.bytes) {
    throw new Error(`24 kHz ${lc.toUpperCase()} Native Studio ONNX specialist not found`);
  }

  const totalMB = modelRes.bytes.byteLength / 1048576;
  const fromCache = Boolean(modelRes.fromCache);
  const cacheBackend = modelRes.cacheBackend;

  if (emitProgress) {
    self.postMessage({
      type: 'progress',
      lang: lc,
      voice: normVoice,
      phase: 'compile',
      loadedMB: totalMB,
      totalMB,
      pct: 95,
      fromCache,
    });
  }

  const sessionOpts = buildSessionOpts();
  const rawBox = { bytes: modelRes.bytes };
  modelRes.bytes = null;
  modelRes = null;

  let session = null;
  try {
    session = await createSplitStudioVitsSession(ort, rawBox, sessionOpts);
    await warmupSpecialistSession(session);
  } catch (compileOrWarmErr) {
    try {
      await session?.release?.();
    } catch {}
    session = null;

    // Automatic fallback 1: WebGPU driver/shader failure -> switch to multi-thread low-mem WASM
    if (ortBackendMode === 'webgpu') {
      const wasmOrt = await switchToPureWasmOrt();
      if (wasmOrt?.InferenceSession) {
        return loadOrGetWarmSession(lc, normVoice, { emitProgress: false });
      }
    }

    // Automatic fallback 2: Multi-thread SharedArrayBuffer OOM -> retry single-thread WASM
    if (ort?.env?.wasm && (ort.env.wasm.numThreads || 1) > 1) {
      ort.env.wasm.numThreads = 1;
      const retryRes = await fetchOrLoadAsset(
        cacheKey,
        urls,
        expectedBytes,
        (arr) => isValidOnnxBuffer(arr, expectedBytes),
        null
      );
      if (retryRes?.bytes) {
        const retryBox = { bytes: retryRes.bytes };
        retryRes.bytes = null;
        session = await createSplitStudioVitsSession(ort, retryBox, buildSessionOpts());
        await warmupSpecialistSession(session);
      } else {
        throw compileOrWarmErr;
      }
    } else {
      throw compileOrWarmErr;
    }
  }

  const entry = {
    session,
    totalMB,
    weightTier: totalMB < 25.0 ? 'mobile' : 'studio',
    fromCache,
    cachedPersisted: Boolean(fromCache || cacheBackend),
    cacheBackend: cacheBackend || 'cache_storage',
    vocoderBackend: session?.vocoderBackend || 'wasm',
  };
  warmSessions.set(slotKey, entry);
  return entry;
}

async function ensureLangReady(rawLang = 'en', { voice = 'male', emitReady = true } = {}) {
  const lc = SUPPORTED_LANGS.includes(rawLang) ? rawLang : 'en';
  const normVoice = normalizeVoiceKey(voice);
  await resolveWorkerOrt();
  const tierTag = shouldUseMobileWeights() ? 'mobile' : 'studio';
  const slotKey = `${lc}:${normVoice}:${tierTag}`;

  try {
    const entry = await loadOrGetWarmSession(lc, normVoice, { emitProgress: emitReady });
    if (!ort || !ort.InferenceSession) {
      throw new Error('ONNX Runtime WASM unavailable');
    }
    const isCoi =
    typeof crossOriginIsolated !== 'undefined' &&
    Boolean(crossOriginIsolated) &&
    WORKER_PARAMS.get('coi') !== '0';
    const activeThreads = ort?.env?.wasm?.numThreads || 1;
    const activeVocBackend = entry.session?.vocoderBackend || entry.vocoderBackend || 'wasm';

    if (!voiceEngine) {
      voiceEngine = await ninada(lc, {
        ort,
        getSession: async (reqLc, reqVoice) => {
          const targetLc = SUPPORTED_LANGS.includes(reqLc) ? reqLc : lc;
          const sEntry = await loadOrGetWarmSession(targetLc, reqVoice || 'male', { emitProgress: false });
          return sEntry.session;
        },
      });
    }

    currentLoadedLang = lc;
    currentLoadedVoice = normVoice;

    lastReadyPayload = {
      type: 'ready',
      lang: lc,
      voice: normVoice,
      isNeural: voiceEngine.isNeural,
      isStreamingNeural: voiceEngine.isStreamingNeural,
      vocoderBackend: activeVocBackend,
      weightTier: entry.weightTier || (entry.totalMB < 25.0 ? 'mobile' : 'studio'),
      totalMB: entry.totalMB,
      fromCache: Boolean(entry.fromCache),
      cachedPersisted: Boolean(entry.cachedPersisted),
      cacheBackend: entry.cacheBackend || null,
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
    currentLoadedVoice = normVoice;
    lastReadyPayload = {
      type: 'ready',
      lang: lc,
      voice: normVoice,
      isNeural: warmSessions.size > 0,
      isStreamingNeural: warmSessions.size > 0,
      vocoderBackend: 'none',
      totalMB: 0,
      fromCache: false,
      cachedPersisted: false,
      cacheBackend: null,
      reason: `WASM fallback (${err?.message || 'memory limit'}): running Klatt Tier-2`,
    };
    if (emitReady) self.postMessage(lastReadyPayload);
    return voiceEngine;
  }
}

self.addEventListener('message', (ev) => {
  const msg = ev.data || {};
  if (msg.type === 'init' || msg.type === 'load_lang') {
    const targetLang = msg.lang || 'en';
    const targetVoice = normalizeVoiceKey(msg.voice);
    if (activeAbort) {
      activeAbort.abort();
    }
    runQueue = runQueue
      .then(() => ensureLangReady(targetLang, { voice: targetVoice, emitReady: true }))
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
    const targetLang = SUPPORTED_LANGS.includes(msg.lang) ? msg.lang : 'en';
    const targetVoice = normalizeVoiceKey(msg.voice);
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
          const tierTag = shouldUseMobileWeights() ? 'mobile' : 'studio';
          const slotKey = `${targetLang}:${targetVoice}:${tierTag}`;
          const needsLoad = !warmSessions.has(slotKey) || !voiceEngine;
          engine = await ensureLangReady(targetLang, { voice: targetVoice, emitReady: needsLoad });
        }
        if (controller.signal.aborted) return;

        const t0 = performance.now();
        let ttfaMs = null;
        let chunkCount = 0;
        let totalSamples = 0;
        let finalTier = 'klatt_zero_dep';
        let finalLang = targetLang;
        const activeSlotKey = `${targetLang}:${targetVoice}:${shouldUseMobileWeights() ? 'mobile' : 'studio'}`;
        let finalVocoderBackend =
          warmSessions.get(activeSlotKey)?.vocoderBackend || 'wasm';

        const workerIsCoi = typeof crossOriginIsolated !== 'undefined' && Boolean(crossOriginIsolated);
        const workerThreads = ort?.env?.wasm?.numThreads || 1;
        const defaultChunkFrames = workerIsCoi ? 64 : 96;
        const defaultFirstFrames = workerIsCoi ? 40 : 56;

        for await (const ch of engine.stream(msg.text || '', {
          lang: targetLang,
          voice: targetVoice,
          rate: Number(msg.rate ?? 1.0),
          noise: msg.noise !== undefined ? Number(msg.noise) : undefined,
          noiseDuration:
            msg.noiseDuration !== undefined || msg.noise_duration !== undefined
              ? Number(msg.noiseDuration ?? msg.noise_duration)
              : undefined,
          tier,
          firstChunkFrames: Number(msg.firstChunkFrames || defaultFirstFrames),
          chunkFrames: Number(msg.chunkFrames || defaultChunkFrames),
          firstBucketFrames: finalVocoderBackend === 'webgpu' ? 44 : workerThreads <= 1 ? 52 : 68,
          mobileWasmStreaming: finalVocoderBackend !== 'webgpu',
          firstLatentFrames: Number(msg.firstLatentFrames || 68),
          singleThread: finalVocoderBackend !== 'webgpu' && (!workerIsCoi || workerThreads <= 1),
          yieldBetweenWindows: true,
          signal: controller.signal,
        })) {
          if (controller.signal.aborted) return;
          if (ttfaMs === null) ttfaMs = performance.now() - t0;
          chunkCount += 1;
          totalSamples += ch.pcm.length;
          finalTier = ch.tier;
          finalLang = ch.lang;
          if (ch.vocoderBackend) {
            finalVocoderBackend = ch.vocoderBackend;
          }

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
              estimatedTotalSec: ch.estimatedTotalSec ?? ch.clauseDurationSec ?? ch.durationSec,
              remainingClauseSec: ch.remainingClauseSec ?? 0,
              nextWindowSec: ch.nextWindowSec ?? 0,
              vocoderWindowAudioSec: ch.vocoderWindowAudioSec ?? ch.durationSec,
              vocoderBucketFrames: ch.vocoderBucketFrames ?? 0,
              vocoderBackend: ch.vocoderBackend || finalVocoderBackend,
              encFlowMs: ch.encFlowMs ?? 0,
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
          vocoderBackend: finalVocoderBackend,
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
