/* Ninada Cross-Origin Isolation Service Worker (enables multi-threaded WASM SIMD on static hosts like GitHub Pages) */
self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;

  let url;
  try {
    url = new URL(req.url);
  } catch {
    return;
  }

  // Let large ONNX model streams and CORS-enabled CDN assets stream directly without SW buffering
  if (url.origin !== self.location.origin || url.pathname.endsWith('.onnx')) {
    return;
  }

  event.respondWith(
    fetch(req)
      .then((res) => {
        if (!res || res.status === 0) return res;
        const headers = new Headers(res.headers);
        headers.set('Cross-Origin-Opener-Policy', 'same-origin');
        headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
        headers.set('Cross-Origin-Resource-Policy', 'same-origin');
        return new Response(res.body, {
          status: res.status,
          statusText: res.statusText,
          headers,
        });
      })
      .catch(() => fetch(req))
  );
});
