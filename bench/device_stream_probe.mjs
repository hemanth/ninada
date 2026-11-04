#!/usr/bin/env node
// Real-device streaming probe: drives Chrome (Android via `adb forward tcp:9225 localabstract:chrome_devtools_remote`)
// over CDP, types fresh text (bypasses boot pre-stream cache), taps Stream with a real touch gesture, and
// reconstructs the exact scheduled output timeline to measure audible-start latency, gaps, overlaps and seam clicks.
//
// Usage: CDP=http://127.0.0.1:9225 URL=http://localhost:4179/ node bench/device_stream_probe.mjs [label]
import fs from 'node:fs';
import path from 'node:path';

const CDP = process.env.CDP || 'http://127.0.0.1:9225';
const URL_ = process.env.URL || 'http://localhost:4179/';
const LABEL = process.argv[2] || 'run';
const RUNS = Number(process.env.RUNS || 2);
const OUT_DIR = process.env.OUT_DIR || '/tmp/ninada_probe';
const TEXTS = (process.env.TEXTS
  ? process.env.TEXTS.split('||')
  : [
      'The quick brown fox jumps over the lazy dog near the river bank. Streaming speech should sound smooth and continuous on every phone.',
      'Good morning everyone. Today we will talk about building fast, private speech synthesis that runs entirely inside your browser, without any servers at all.',
    ]);
fs.mkdirSync(OUT_DIR, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function connect(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    let id = 1;
    const pending = new Map();
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && pending.has(m.id)) {
        const p = pending.get(m.id);
        pending.delete(m.id);
        m.error ? p.rej(new Error(m.error.message)) : p.res(m.result);
      }
    };
    ws.onerror = reject;
    ws.onopen = () =>
      resolve({
        send: (method, params = {}) =>
          new Promise((res, rej) => {
            const i = id++;
            pending.set(i, { res, rej });
            ws.send(JSON.stringify({ id: i, method, params }));
          }),
        close: () => ws.close(),
      });
  });
}

const HOOK = `(() => {
  if (window.__probe) return;
  const P = (window.__probe = { sched: [], chunks: [], t0: performance.now() });
  const origStart = AudioBufferSourceNode.prototype.start;
  AudioBufferSourceNode.prototype.start = function (when = 0, ...rest) {
    try {
      if (this.buffer && this.buffer.length > 64) {
        const ch = this.buffer.getChannelData(0);
        P.sched.push({ when: +when || 0, now: this.context.currentTime, perf: performance.now(),
          sr: this.context.sampleRate, bsr: this.buffer.sampleRate, len: this.buffer.length,
          outLat: this.context.outputLatency || 0, baseLat: this.context.baseLatency || 0, pcm: new Float32Array(ch) });
        P.ctx = this.context;
      }
    } catch {}
    return origStart.call(this, when, ...rest);
  };
  const OrigWorker = window.Worker;
  window.Worker = function (...a) {
    const w = new OrigWorker(...a);
    const origPost = w.postMessage.bind(w);
    w.postMessage = (m, ...r) => {
      if (m && (m.type === 'stream' || m.type === 'cancel')) (P.reqs ||= []).push({ perf: performance.now(), type: m.type, reqId: m.reqId,
        text: (m.text || '').slice(0, 24) });
      return origPost(m, ...r);
    };
    w.addEventListener('message', (ev) => {
      const m = ev.data || {};
      if (m.type === 'chunk') P.chunks.push({ perf: performance.now(), req: m.reqId, idx: m.index, len: m.pcm?.length || 0,
        clause: m.clauseIndex, vocMs: m.vocoderMs, encMs: m.encFlowMs, bucket: m.vocoderBucketFrames, ttfa: m.ttfaMs });
      if (m.type === 'done') P.donePerf = performance.now();
      if (m.type === 'ready') P.ready = m;
    });
    return w;
  };
  window.Worker.prototype = OrigWorker.prototype;
})();`;

function b64f32(expr) {
  return `(() => { const a = ${expr}; const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); let s=''; for (let i=0;i<u.length;i+=0x8000) s += String.fromCharCode.apply(null, u.subarray(i,i+0x8000)); return btoa(s); })()`;
}

function writeWav(file, pcm, sr) {
  const buf = Buffer.alloc(44 + pcm.length * 2);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + pcm.length * 2, 4); buf.write('WAVEfmt ', 8);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22); buf.writeUInt32LE(sr, 24);
  buf.writeUInt32LE(sr * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34); buf.write('data', 36);
  buf.writeUInt32LE(pcm.length * 2, 40);
  for (let i = 0; i < pcm.length; i++) buf.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(pcm[i] * 32767))), 44 + i * 2);
  fs.writeFileSync(file, buf);
}

function analyze(sched, chunks, tapPerf, stats) {
  if (!sched.length) return { error: 'no audio scheduled' };
  const sr = sched[0].sr;
  const first = sched[0];
  const startFrame = Math.round(first.when * sr);
  let endFrame = startFrame;
  const rows = [];
  let gapsMs = 0, overlapMs = 0, lateStarts = 0;
  for (let i = 0; i < sched.length; i++) {
    const s = sched[i];
    const f = Math.round(s.when * sr);
    const gap = f - endFrame; // >0 silence, <0 overlap
    if (i > 0) { if (gap > 1) gapsMs += (gap / sr) * 1000; if (gap < -1) overlapMs += (-gap / sr) * 1000; }
    if (s.when + 1e-4 < s.now) lateStarts++;
    rows.push({ i, leadMs: +((s.when - s.now) * 1000).toFixed(1), durMs: +((s.len / s.bsr) * 1000).toFixed(0), gapFrames: i ? gap : 0 });
    endFrame = Math.max(endFrame, f + s.len);
  }
  const total = endFrame - startFrame;
  const out = new Float32Array(total);
  const boundaries = [];
  for (const s of sched) {
    const off = Math.round(s.when * sr) - startFrame;
    for (let k = 0; k < s.len; k++) if (off + k >= 0 && off + k < total) out[off + k] += s.pcm[k];
    boundaries.push(off);
  }
  // click metric: |2nd difference| near each boundary vs robust interior level
  const d2 = new Float32Array(total);
  for (let i = 1; i < total - 1; i++) d2[i] = Math.abs(out[i + 1] - 2 * out[i] + out[i - 1]);
  const sorted = Float32Array.from(d2).sort();
  const p999 = sorted[Math.floor(sorted.length * 0.999)] || 1e-9;
  const seamPeaks = boundaries.slice(1).map((b) => {
    let m = 0; for (let i = Math.max(1, b - 3); i < Math.min(total - 1, b + 3); i++) m = Math.max(m, d2[i]);
    return +(m / p999).toFixed(2);
  });
  // silent stretches inside speech (dropouts) > 30ms
  let run = 0, dropouts = 0;
  for (let i = 0; i < total; i++) { if (Math.abs(out[i]) < 1e-4) run++; else { if (run > 0.03 * sr && i > 0.2 * sr) dropouts++; run = 0; } }
  const audibleMs = first.perf - tapPerf + (first.when - first.now) * 1000 + first.outLat * 1000;
  // underrun headroom: min lead time of each schedule (how close each chunk came to arriving late)
  const leads = rows.slice(1).map((r) => r.leadMs);
  return {
    sr, nSched: sched.length, durSec: +(total / sr).toFixed(2),
    audibleStartMs: Math.round(audibleMs), firstChunkPerfMs: chunks.length ? Math.round(chunks[0].perf - tapPerf) : null,
    gapsMs: +gapsMs.toFixed(1), overlapMs: +overlapMs.toFixed(1), lateStarts, minLeadMs: leads.length ? Math.min(...leads) : null,
    dropouts30ms: dropouts, worstSeamClick: seamPeaks.length ? Math.max(...seamPeaks) : 0, seamPeaks,
    outLatMs: Math.round(first.outLat * 1000), baseLatMs: Math.round(first.baseLat * 1000),
    pageUnderruns: stats?.underrunCount, sentencePauses: stats?.sentencePauseCount, rows, out,
  };
}

async function main() {
  const ver = await (await fetch(`${CDP}/json/version`)).json();
  const bws = await connect(ver.webSocketDebuggerUrl.replace(/ws:\/\/[^/]+/, CDP.replace('http', 'ws')));
  const { targetId } = await bws.send('Target.createTarget', { url: 'about:blank' });
  let tab = null;
  for (let i = 0; i < 20 && !tab; i++) {
    tab = (await (await fetch(`${CDP}/json/list`)).json()).find((t) => t.id === targetId);
    if (!tab) await sleep(200);
  }
  const c = await connect(tab.webSocketDebuggerUrl.replace(/ws:\/\/[^/]+/, CDP.replace('http', 'ws')));
  const ev = async (expression, awaitPromise = false) => {
    const r = await c.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise });
    if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails).slice(0, 400));
    return r.result.value;
  };
  await c.send('Page.enable');
  await c.send('Runtime.enable');
  await c.send('Page.addScriptToEvaluateOnNewDocument', { source: HOOK });
  const tNav = Date.now();
  await c.send('Page.navigate', { url: URL_ + (URL_.includes('?') ? '&' : '?') + 'probe=' + Date.now() });
  await c.send('Page.bringToFront');

  for (let i = 0; i < 240; i++) {
    const ok = await ev(`Boolean(window.__ninadaReady) && !document.getElementById('synthBtn').disabled`).catch(() => false);
    if (ok) break;
    await sleep(250);
  }
  const env = await ev(`({ ua: navigator.userAgent, coi: crossOriginIsolated, cores: navigator.hardwareConcurrency, mem: navigator.deviceMemory,
    ready: window.__ninadaReadyInfo && { threads: window.__ninadaReadyInfo.threads, voc: window.__ninadaReadyInfo.vocoderBackend, cache: window.__ninadaReadyInfo.fromCache } })`);
  console.log(`[${LABEL}] ready in ${Date.now() - tNav}ms`, JSON.stringify(env));
  await sleep(4000); // let boot pre-stream finish

  const results = [];
  for (let r = 0; r < RUNS; r++) {
    const text = TEXTS[r % TEXTS.length] + (r >= TEXTS.length ? ` Run ${r}.` : '');
    await ev(`(() => { const t = document.getElementById('textInput'); t.value = ${JSON.stringify(text)}; t.dispatchEvent(new Event('input')); })()`);
    const { x, y } = await ev(`(() => { const el = document.getElementById('synthBtn'); el.scrollIntoView({block:'center'}); const b = el.getBoundingClientRect(); return { x: b.left + b.width/2, y: b.top + b.height/2 }; })()`);
    await sleep(400);
    await ev(HOOK);
    await ev(`(() => { window.__probe.reqs = window.__probe.reqs || []; window.__probe.sched = []; window.__probe.chunks = []; window.__probe.reqs = []; window.__probe.donePerf = 0; window.__probeTapPerf = 0;
      document.getElementById('synthBtn').addEventListener('touchstart', () => { window.__probeTapPerf = performance.now(); }, { once: true, capture: true }); })()`);
    await c.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] });
    await c.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
    const tTap = Date.now();
    await sleep(800);
    for (let i = 0; i < 160; i++) {
      const st = await ev(`({ playing: window.__ninadaStreamStats.isPlaying, done: window.__probe.donePerf > 0, n: window.__probe.sched.length })`);
      if (st.done && !st.playing && st.n > 0) break;
      await sleep(250);
    }
    const meta = await ev(`({ reqLog: window.__ninadaReqLog, loadPerf: performance.timeOrigin, reqs: window.__probe.reqs, tap: window.__probeTapPerf, chunks: window.__probe.chunks, stats: { ...window.__ninadaStreamStats }, last: window.__ninadaLastResult,
      sched: window.__probe.sched.map(s => ({ ...s, pcm: undefined })) })`);
    const sched = [];
    for (let i = 0; i < meta.sched.length; i++) {
      const b64 = await ev(b64f32(`window.__probe.sched[${i}].pcm`));
      const bytes = Buffer.from(b64, 'base64');
      sched.push({ ...meta.sched[i], pcm: new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4) });
    }
    const a = analyze(sched, meta.chunks, meta.tap, meta.stats);
    if (a.out) writeWav(path.join(OUT_DIR, `${LABEL}_r${r}.wav`), a.out, a.sr);
    const chunkArr = meta.chunks.map((ch, i) => ({ i, req: ch.req, atMs: Math.round(ch.perf - meta.tap), audMs: Math.round((ch.len / 24000) * 1000), vocMs: Math.round(ch.vocMs || 0), encMs: Math.round(ch.encMs || 0), bucket: ch.bucket }));
    const { out, rows, ...summary } = a;
    console.log(`[${LABEL} r${r}] wall ${Date.now() - tTap}ms`, JSON.stringify(summary));
    console.log('  chunks:', JSON.stringify(chunkArr));
    console.log('  reqLog:', JSON.stringify((meta.reqLog || []).map((q) => ({ ...q, t: Math.round(q.t - meta.tap) }))));
    console.log('  reqs  :', JSON.stringify((meta.reqs || []).map((q) => ({ ...q, atMs: Math.round(q.perf - meta.tap), perf: undefined }))));
    console.log('  sched :', JSON.stringify(rows));
    results.push({ summary, chunks: chunkArr, rows, env, last: meta.last });
  }
  fs.writeFileSync(path.join(OUT_DIR, `${LABEL}.json`), JSON.stringify(results, null, 2));
  c.close();
  await bws.send('Target.closeTarget', { targetId }).catch(() => {});
  bws.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
