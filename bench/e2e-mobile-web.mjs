#!/usr/bin/env node
/**
 * Ninada Mobile Web End-to-End User-Journey Suite (`e2e-mobile-web.mjs`)
 *
 * Exercises real mobile browser engines WITHOUT bypassing browser autoplay policies:
 *   1. Apple WebKit (`webkit` + `iPhone 14` 390x844, `hasTouch: true`)
 *      - Real `coi-sw.js` Cross-Origin Isolation boot & mobile viewport overflow (`<= 1px`)
 *      - Trusted `touchscreen.tap()` gesture unlocking `AudioContext` + `AudioWorklet`
 *      - Sub-clause 24 kHz Neural ONNX streaming (`chunkCount >= 4`, `0` underruns)
 *      - Mid-stream `#stopBtn` tap cancellation (`< 450ms` worker abort turnaround)
 *      - Tier-2 Zero-Dep Klatt formant streaming fallback (`tier === 'klatt_zero_dep'`)
 *      - Byte-level 24 kHz 16-bit Mono RIFF WAV download blob verification
 *
 *   2. Android Chrome (`chromium` + `Pixel 7` 412x915, NO `--autoplay-policy` flag!)
 *      - Strict gesture-gated `AudioContext` unlock via real `touchscreen.tap()`
 *      - Rapid mid-stream preset chip + language tab (`sa -> hi`) interruption with zero leaked audio
 *      - Studio Female specialist voice (`#voiceSelect = 'female'`) split-graph ONNX streaming
 *      - Rate (`1.15x`) & Noise (`0.55`) slider modulation
 *      - True Airplane-Mode Offline (`context.setOffline(true)`) streaming of unseen Sanskrit text from `CacheStorage`
 */
import { spawn } from "node:child_process";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, webkit, devices } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, "..");
const PORT = Number(process.env.PORT || 4198);
const TARGET_URL = process.env.TARGET_URL || `http://127.0.0.1:${PORT}/`;
const USE_LOCAL_SERVER = !process.env.TARGET_URL;

async function waitForServer(url, timeoutMs = 8000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(url);
      if (r.ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 120));
  }
  throw new Error(`Server at ${url} did not start within ${timeoutMs}ms`);
}

async function tapLocator(page, locator) {
  await locator.scrollIntoViewIfNeeded();
  const box = await locator.boundingBox();
  assert.ok(box, "Expected visible bounding box for mobile tap target");
  await page.touchscreen.tap(box.x + box.width / 2, box.y + box.height / 2);
}

async function waitForReadyUnlock(page, expectedLang = null, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const ok = await page.evaluate((lc) => {
        if (!window.__ninadaReady || window.__ninadaControlsLocked) return false;
        if (lc && window.__ninadaLoadedLang !== lc) return false;
        return true;
      }, expectedLang);
      if (ok) return;
    } catch {
      // Transient navigation while coi-sw.js activates on initial load
    }
    await new Promise((r) => setTimeout(r, 60));
  }
  throw new Error(`Timed out waiting for Ninada ready unlock (lang=${expectedLang || "any"})`);
}

async function runWebKitIosSafariE2E(baseUrl) {
  console.log("\n[1/2] Launching Apple WebKit (iPhone 14 Mobile Safari Engine, Strict Autoplay)...");
  const browser = await webkit.launch({ headless: true });
  const iphone = devices["iPhone 14"];
  const context = await browser.newContext({ ...iphone });
  const page = await context.newPage();

  const errors = [];
  page.on("pageerror", (e) => errors.push(`WEBKIT_PAGEERROR: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") {
      const txt = m.text();
      // Ignore headless Playwright macOS WebKit UA shadow-DOM ModernMediaControls icon placard messages
      if (/Button failed to load, iconName = .*-placard, layoutTraits = \[MacOSLayoutTraits/.test(txt)) {
        return;
      }
      errors.push(`WEBKIT_CONSOLE_ERROR: ${txt}`);
    }
  });

  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await waitForReadyUnlock(page, "sa");

  // Wait briefly for any initial silent preview synthesis to settle or stay idle
  await page.waitForTimeout(250);
  await page.evaluate(() => {
    if (typeof window.__ninadaCancel === "function") window.__ninadaCancel();
    window.__ninadaLastResult = null;
  });

  // 1a. Mobile layout overflow check on real WebKit engine
  const overflowPx = await page.evaluate(() => {
    const pre = document.querySelector("pre.code-block");
    const docOverflow = document.documentElement.scrollWidth - document.documentElement.clientWidth;
    const preOverflow = pre ? pre.scrollWidth - pre.clientWidth : 0;
    return Math.max(docOverflow, preOverflow);
  });
  assert.ok(overflowPx <= 1, `WebKit iPhone 14 horizontal overflow: ${overflowPx}px`);

  // 1b. Trusted touchscreen tap on #synthBtn -> unlock WebKit AudioContext + stream 24kHz Neural ONNX
  const synthBtn = page.locator("#synthBtn");
  await tapLocator(page, synthBtn);

  await page.waitForFunction(
    () => window.__ninadaLastResult && window.__ninadaLastResult.tier === "neural_onnx",
    null,
    { timeout: 90000 }
  );

  const webkitStream = await page.evaluate(async () => {
    const drained = await window.__ninadaWaitForPlayback();
    return {
      coi: Boolean(window.crossOriginIsolated),
      result: window.__ninadaLastResult,
      stats: { ...drained },
      btnLabel: document.getElementById("synthBtnLabel")?.textContent,
    };
  });
  console.log("  ✔ WebKit iPhone 14 Neural Stream:", {
    coi: webkitStream.coi,
    threads: webkitStream.result.threads,
    ttfaMs: Math.round(webkitStream.result.ttfaMs * 10) / 10,
    chunkCount: webkitStream.result.chunkCount,
    durationSec: Math.round(webkitStream.result.durationSec * 100) / 100,
    engine: webkitStream.stats.engine,
    playedSamples: webkitStream.stats.playedSamples,
    underrunCount: webkitStream.stats.underrunCount,
  });
  assert.equal(webkitStream.result.tier, "neural_onnx");
  assert.ok(webkitStream.result.chunkCount >= 4, `Expected sub-clause chunks >=4, got ${webkitStream.result.chunkCount}`);
  assert.ok(webkitStream.result.rms > 0.02);
  assert.equal(webkitStream.stats.underrunCount, 0, "Expected 0 underruns on WebKit iPhone 14");
  assert.ok(webkitStream.stats.playedSamples > 0, "Expected WebKit AudioContext to unlock and play samples on tap");

  // 1c. Byte-level 24 kHz RIFF WAV download blob verification
  const wavCheck = await page.evaluate(async () => {
    const href = document.getElementById("downloadLink")?.href || "";
    if (!href.startsWith("blob:")) return { ok: false, reason: `Invalid href: ${href}` };
    const buf = await (await fetch(href)).arrayBuffer();
    const dv = new DataView(buf);
    const riff = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
    const wave = String.fromCharCode(dv.getUint8(8), dv.getUint8(9), dv.getUint8(10), dv.getUint8(11));
    const channels = dv.getUint16(22, true);
    const sampleRate = dv.getUint32(24, true);
    const bitsPerSample = dv.getUint16(34, true);
    return {
      ok: true,
      byteLength: buf.byteLength,
      riff,
      wave,
      channels,
      sampleRate,
      bitsPerSample,
    };
  });
  console.log("  ✔ WebKit WAV Download Verification:", wavCheck);
  assert.equal(wavCheck.ok, true);
  assert.equal(wavCheck.riff, "RIFF");
  assert.equal(wavCheck.wave, "WAVE");
  assert.equal(wavCheck.channels, 1);
  assert.equal(wavCheck.sampleRate, 24000);
  assert.equal(wavCheck.bitsPerSample, 16);
  assert.ok(wavCheck.byteLength > 48000);

  // 1d. Mid-stream #stopBtn tap cancellation responsiveness
  await tapLocator(page, page.locator(".preset-chip", { hasText: "IAST Streaming" }));
  await page.evaluate(() => {
    window.__ninadaLastResult = null;
    if (window.__ninadaStreamStats) window.__ninadaStreamStats.totalPushedSamples = 0;
  });
  await tapLocator(page, synthBtn);
  await page.waitForFunction(
    () => (window.__ninadaStreamStats?.totalPushedSamples || 0) > 0,
    null,
    { timeout: 30000 }
  );
  const tCancel0 = Date.now();
  await tapLocator(page, page.locator("#stopBtn"));
  await page.waitForFunction(
    () =>
      document.getElementById("stopBtn")?.disabled === true &&
      document.getElementById("synthBtnLabel")?.textContent === "Stream Speech",
    null,
    { timeout: 2000 }
  );
  const cancelTurnaroundMs = Date.now() - tCancel0;
  console.log(`  ✔ WebKit Mid-Stream Tap Stop Turnaround: ${cancelTurnaroundMs}ms`);
  assert.ok(cancelTurnaroundMs < 600, `Stop cancellation took too long: ${cancelTurnaroundMs}ms`);

  // 1e. Tier-2 Zero-Dep Klatt formant streaming via UI select + tap
  await page.selectOption("#tierSelect", "klatt");
  await page.evaluate(() => {
    window.__ninadaLastResult = null;
  });
  await tapLocator(page, synthBtn);
  await page.waitForFunction(
    () => window.__ninadaLastResult && window.__ninadaLastResult.tier === "klatt_zero_dep",
    null,
    { timeout: 15000 }
  );
  const klattRes = await page.evaluate(async () => {
    await window.__ninadaWaitForPlayback();
    return window.__ninadaLastResult;
  });
  console.log("  ✔ WebKit Tier-2 Klatt Zero-Dep Stream:", {
    tier: klattRes.tier,
    ttfaMs: Math.round(klattRes.ttfaMs * 10) / 10,
    samples: klattRes.samples,
    rms: Math.round(klattRes.rms * 10000) / 10000,
  });
  assert.equal(klattRes.tier, "klatt_zero_dep");
  assert.ok(klattRes.samples > 24000);
  assert.ok(klattRes.rms > 0.02);

  // 1f. Switch back to Tier-1 Neural (`auto`) + tap Indian English (`en`) tab -> stream 24 kHz Indian English on WebKit!
  await page.selectOption("#tierSelect", "auto");
  await tapLocator(page, page.locator('.lang-btn[data-lang="en"]'));
  await waitForReadyUnlock(page, "en");
  await page.waitForTimeout(220);
  await page.evaluate(() => {
    if (typeof window.__ninadaCancel === "function") window.__ninadaCancel();
    window.__ninadaLastResult = null;
  });
  await tapLocator(page, synthBtn);
  await page.waitForFunction(
    () =>
      window.__ninadaLastResult &&
      window.__ninadaLastResult.lang === "en" &&
      window.__ninadaLastResult.tier === "neural_onnx",
    null,
    { timeout: 60000 }
  );
  const enWebkitStream = await page.evaluate(async () => {
    const drained = await window.__ninadaWaitForPlayback();
    return {
      result: window.__ninadaLastResult,
      stats: { ...drained },
    };
  });
  console.log("  ✔ WebKit Indian English (`en`) Neural Stream:", {
    lang: enWebkitStream.result.lang,
    ttfaMs: Math.round(enWebkitStream.result.ttfaMs * 10) / 10,
    chunkCount: enWebkitStream.result.chunkCount,
    underrunCount: enWebkitStream.stats.underrunCount,
  });
  assert.equal(enWebkitStream.result.lang, "en");
  assert.ok(enWebkitStream.result.rms > 0.02);
  assert.equal(enWebkitStream.stats.underrunCount, 0);

  assert.equal(errors.length, 0, `WebKit E2E errors: ${errors.join(" | ")}`);
  await context.close();
  await browser.close();
}

async function runAndroidGestureOfflineFemaleE2E(baseUrl) {
  console.log("\n[2/2] Launching Gesture-Gated Android Chrome (Pixel 7, Female Voice + Interruption + Airplane Offline)...");
  // Intentionally NO --autoplay-policy=no-user-gesture-required flag!
  const browser = await chromium.launch({ headless: true });
  const pixel = devices["Pixel 7"];
  const context = await browser.newContext({ ...pixel });
  const page = await context.newPage();

  const errors = [];
  page.on("pageerror", (e) => errors.push(`ANDROID_PAGEERROR: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`ANDROID_CONSOLE_ERROR: ${m.text()}`);
  });

  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await waitForReadyUnlock(page, "sa");

  // Verify AudioContext starts suspended before gesture (real mobile browser policy)
  await page.waitForTimeout(220);
  await page.evaluate(() => {
    if (typeof window.__ninadaCancel === "function") window.__ninadaCancel();
    window.__ninadaLastResult = null;
  });

  // 2a. Trusted touchscreen tap unlocks suspended AudioContext and plays full stream with 0 underruns
  const synthBtn = page.locator("#synthBtn");
  await tapLocator(page, synthBtn);
  await page.waitForFunction(
    () => window.__ninadaLastResult && window.__ninadaLastResult.tier === "neural_onnx",
    null,
    { timeout: 60000 }
  );
  const firstGestureStream = await page.evaluate(async () => {
    const drained = await window.__ninadaWaitForPlayback();
    return {
      result: window.__ninadaLastResult,
      stats: { ...drained },
    };
  });
  console.log("  ✔ Android Gesture-Gated First Tap Stream:", {
    ttfaMs: Math.round(firstGestureStream.result.ttfaMs * 10) / 10,
    playedSamples: firstGestureStream.stats.playedSamples,
    totalSamples: firstGestureStream.result.samples,
    underrunCount: firstGestureStream.stats.underrunCount,
  });
  assert.equal(firstGestureStream.stats.underrunCount, 0);
  assert.ok(
    firstGestureStream.stats.playedSamples >= Math.floor(firstGestureStream.result.samples * 0.99),
    `Expected full sample drain after gesture unlock (${firstGestureStream.result.samples}), got ${firstGestureStream.stats.playedSamples}`
  );

  // 2b. Rapid mid-stream interruption: start SA stream -> 120ms later tap Hindi (`hi`) tab -> stream Hindi!
  await tapLocator(page, page.locator(".preset-chip", { hasText: "IAST Streaming" }));
  await tapLocator(page, synthBtn);
  await page.waitForTimeout(120);
  await tapLocator(page, page.locator('.lang-btn[data-lang="hi"]'));
  await waitForReadyUnlock(page, "hi");
  await page.waitForTimeout(220);
  await page.evaluate(() => {
    if (typeof window.__ninadaCancel === "function") window.__ninadaCancel();
    window.__ninadaLastResult = null;
  });

  // Modulate rate & noise sliders before streaming Hindi
  await page.evaluate(() => {
    const r = document.getElementById("rateSlider");
    const n = document.getElementById("noiseSlider");
    r.value = "1.10";
    r.dispatchEvent(new Event("input", { bubbles: true }));
    n.value = "0.55";
    n.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await tapLocator(page, synthBtn);
  await page.waitForFunction(
    () =>
      window.__ninadaLastResult &&
      window.__ninadaLastResult.lang === "hi" &&
      window.__ninadaLastResult.tier === "neural_onnx",
    null,
    { timeout: 60000 }
  );
  const hiStream = await page.evaluate(async () => {
    const drained = await window.__ninadaWaitForPlayback();
    return {
      result: window.__ninadaLastResult,
      stats: { ...drained },
    };
  });
  console.log("  ✔ Android Mid-Stream Switch -> Hindi (`hi`) Stream:", {
    lang: hiStream.result.lang,
    ttfaMs: Math.round(hiStream.result.ttfaMs * 10) / 10,
    chunkCount: hiStream.result.chunkCount,
    underrunCount: hiStream.stats.underrunCount,
  });
  assert.equal(hiStream.result.lang, "hi");
  assert.equal(hiStream.stats.underrunCount, 0);

  // 2c. Switch to Studio Female voice (`#voiceSelect = 'female'`) on Hindi -> verify split-graph streaming
  await page.selectOption("#voiceSelect", "female");
  await waitForReadyUnlock(page, "hi");
  await page.waitForTimeout(220);
  await page.evaluate(() => {
    if (typeof window.__ninadaCancel === "function") window.__ninadaCancel();
    window.__ninadaLastResult = null;
  });
  await tapLocator(page, synthBtn);
  await page.waitForFunction(
    () =>
      window.__ninadaLastResult &&
      window.__ninadaLastResult.lang === "hi" &&
      window.__ninadaLastResult.tier === "neural_onnx",
    null,
    { timeout: 60000 }
  );
  const femaleStream = await page.evaluate(async () => {
    const drained = await window.__ninadaWaitForPlayback();
    return {
      voice: window.__ninadaReadyInfo?.voice,
      result: window.__ninadaLastResult,
      stats: { ...drained },
    };
  });
  console.log("  ✔ Android Studio Female (`hi:female`) Split-ONNX Stream:", {
    voice: femaleStream.voice,
    ttfaMs: Math.round(femaleStream.result.ttfaMs * 10) / 10,
    chunkCount: femaleStream.result.chunkCount,
    underrunCount: femaleStream.stats.underrunCount,
  });
  assert.equal(femaleStream.voice, "female");
  assert.ok(femaleStream.result.chunkCount >= 3);
  assert.ok(femaleStream.result.rms > 0.02);
  assert.equal(femaleStream.stats.underrunCount, 0);

  // 2d. True Airplane-Mode Offline E2E (`context.setOffline(true)`):
  // Switch back to cached `sa:male` while COMPLETELY OFFLINE (evicted from 1-slot mobile RAM LRU -> reloaded from CacheStorage!)
  // and synthesize brand-new custom Sanskrit text with ZERO network connectivity!
  await context.setOffline(true);
  await page.selectOption("#voiceSelect", "male");
  await waitForReadyUnlock(page, "hi");
  await tapLocator(page, page.locator('.lang-btn[data-lang="sa"]'));
  await waitForReadyUnlock(page, "sa");

  await page.fill("#textInput", "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः।");
  await page.evaluate(() => {
    if (typeof window.__ninadaCancel === "function") window.__ninadaCancel();
    window.__ninadaLastResult = null;
  });
  await tapLocator(page, synthBtn);
  await page.waitForFunction(
    () =>
      window.__ninadaLastResult &&
      window.__ninadaLastResult.lang === "sa" &&
      window.__ninadaLastResult.tier === "neural_onnx",
    null,
    { timeout: 60000 }
  );
  const offlineStream = await page.evaluate(async () => {
    const drained = await window.__ninadaWaitForPlayback();
    return {
      fromCache: window.__ninadaReadyInfo?.fromCache,
      result: window.__ninadaLastResult,
      stats: { ...drained },
    };
  });
  console.log("  ✔ Android Airplane-Mode 100% Offline CacheStorage Stream (`sa:male`):", {
    fromCache: offlineStream.fromCache,
    ttfaMs: Math.round(offlineStream.result.ttfaMs * 10) / 10,
    chunkCount: offlineStream.result.chunkCount,
    samples: offlineStream.result.samples,
    underrunCount: offlineStream.stats.underrunCount,
  });
  assert.equal(offlineStream.fromCache, true);
  assert.equal(offlineStream.result.tier, "neural_onnx");
  assert.ok(offlineStream.result.rms > 0.02);
  assert.equal(offlineStream.stats.underrunCount, 0);

  assert.equal(errors.length, 0, `Android E2E errors: ${errors.join(" | ")}`);
  await context.close();
  await browser.close();
}

async function main() {
  const server = USE_LOCAL_SERVER
    ? spawn(process.execPath, [path.join(ROOT_DIR, "serve-playground.mjs")], {
        cwd: ROOT_DIR,
        env: { ...process.env, PORT: String(PORT), COOP_COEP: "0" },
        stdio: ["ignore", "pipe", "pipe"],
      })
    : null;

  try {
    if (server) {
      await waitForServer(TARGET_URL);
    }
    await runWebKitIosSafariE2E(TARGET_URL);
    await runAndroidGestureOfflineFemaleE2E(TARGET_URL);
    console.log(`\nALL MOBILE WEB E2E USER-JOURNEY TESTS (WEBKIT iOS SAFARI + GESTURE-GATED ANDROID CHROME + FEMALE VOICE + OFFLINE AIRPLANE MODE) PASSED ON ${TARGET_URL}!\n`);
  } finally {
    if (server) {
      server.kill("SIGTERM");
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
