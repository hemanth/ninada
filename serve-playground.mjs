#!/usr/bin/env node
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = path.join(__dirname, "docs");
const JS_DIST_DIR = path.join(__dirname, "js", "dist");
const ARTIFACTS_V5_ONNX_DIR = path.join(__dirname, "artifacts", "ninada_v5", "onnx");
const ARTIFACTS_V4_ONNX_DIR = path.join(__dirname, "artifacts", "ninada_v4", "onnx");
const PORT = Number(process.env.PORT || 4179);
const gzipCache = new Map();

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
  } else if (pathname.startsWith("/onnx/")) {
    const rel = pathname.slice(6);
    const c5 = path.join(ARTIFACTS_V5_ONNX_DIR, rel);
    filePath = fs.existsSync(c5) ? c5 : path.join(ARTIFACTS_V4_ONNX_DIR, rel);
  } else if (pathname.startsWith("/models/") && !fs.existsSync(filePath)) {
    const rel = pathname.slice(8);
    const c5 = path.join(ARTIFACTS_V5_ONNX_DIR, rel);
    const c4 = path.join(ARTIFACTS_V4_ONNX_DIR, rel);
    if (fs.existsSync(c5)) filePath = c5;
    else if (fs.existsSync(c4)) filePath = c4;
  }

  const resolved = path.resolve(filePath);
  if (
    !resolved.startsWith(DOCS_DIR) &&
    !resolved.startsWith(JS_DIST_DIR) &&
    !resolved.startsWith(ARTIFACTS_V5_ONNX_DIR) &&
    !resolved.startsWith(ARTIFACTS_V4_ONNX_DIR)
  ) {
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
  if (process.env.COOP_COEP !== "0") {
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader(
      "Cross-Origin-Embedder-Policy",
      process.env.COEP_CREDENTIALLESS === "1" ? "credentialless" : "require-corp"
    );
    res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
  }

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

  const acceptEnc = String(req.headers["accept-encoding"] || "");
  if (ext === ".onnx" && acceptEnc.includes("gzip")) {
    const gzKey = `${resolved}:${stat.mtimeMs}:${stat.size}`;
    let gz = gzipCache.get(gzKey);
    if (!gz) {
      for (const k of gzipCache.keys()) {
        if (k.startsWith(`${resolved}:`)) gzipCache.delete(k);
      }
      gz = zlib.gzipSync(fs.readFileSync(resolved), { level: 1 });
      gzipCache.set(gzKey, gz);
    }
    res.writeHead(200, {
      "Content-Encoding": "gzip",
      "Content-Length": gz.byteLength,
    });
    res.end(gz);
    return;
  }

  res.writeHead(200, { "Content-Length": stat.size });
  fs.createReadStream(resolved).pipe(res);
});

const HOST = process.env.HOST || "0.0.0.0";
server.listen(PORT, HOST, () => {
  console.log(`Ninada browser studio listening on http://127.0.0.1:${PORT}`);
});
