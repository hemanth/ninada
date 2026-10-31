#!/usr/bin/env node
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = path.join(__dirname, "docs");
const JS_DIST_DIR = path.join(__dirname, "js", "dist");
const PORT = Number(process.env.PORT || 4179);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".onnx": "application/octet-stream",
  ".wasm": "application/wasm",
  ".wav": "audio/wav",
};

const server = http.createServer((req, res) => {
  const urlObj = new URL(req.url || "/", `http://localhost:${PORT}`);
  let pathname = decodeURIComponent(urlObj.pathname);
  if (pathname === "/") pathname = "/index.html";

  let filePath = path.join(DOCS_DIR, pathname);
  if (pathname.startsWith("/dist/")) {
    filePath = path.join(JS_DIST_DIR, pathname.slice(6));
  }

  const resolved = path.resolve(filePath);
  if (!resolved.startsWith(DOCS_DIR) && !resolved.startsWith(JS_DIST_DIR)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
    res.writeHead(404);
    res.end("Not found");
    return;
  }

  const stat = fs.statSync(resolved);
  const ext = path.extname(resolved).toLowerCase();
  const contentType = MIME[ext] || "application/octet-stream";

  res.setHeader("Content-Type", contentType);
  res.setHeader("Accept-Ranges", "bytes");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "credentialless");

  const range = req.headers.range;
  if (range) {
    const match = /bytes=(\d*)-(\d*)/.exec(range);
    if (match) {
      const start = match[1] ? parseInt(match[1], 10) : 0;
      const end = match[2] ? parseInt(match[2], 10) : stat.size - 1;
      res.writeHead(206, {
        "Content-Range": `bytes ${start}-${end}/${stat.size}`,
        "Content-Length": end - start + 1,
      });
      fs.createReadStream(resolved, { start, end }).pipe(res);
      return;
    }
  }

  res.writeHead(200, { "Content-Length": stat.size });
  fs.createReadStream(resolved).pipe(res);
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Ninada browser studio listening on http://127.0.0.1:${PORT}`);
});
