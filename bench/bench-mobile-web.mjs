#!/usr/bin/env node
/**
 * Ninada Mobile Web Streaming Benchmark & Regression Gate (`bench-mobile-web.mjs`)
 *
 * Measures end-to-end mobile web streaming performance across two realistic mobile profiles:
 *   1. `android_coi_3x`: Pixel 7 viewport (412x915), 3x CPU throttle, COOP/COEP multi-thread WASM
 *   2. `ios_1thread_2x`: iPhone 14 viewport (390x844), 2x CPU throttle, ?coi=0 single-thread WASM
 *      (models iOS Safari / in-app WebViews without SharedArrayBuffer)
 *
 * Captures true user-perceived metrics:
 *   - `fcpMs`             : First Contentful Paint
 *   - `readyUnlockMs`     : Navigation start -> `#synthBtn` unlocked (`window.__ninadaReady`)
 *   - `tapToFirstPushMs`  : User tap on `#synthBtn` -> first PCM chunk pushed to AudioWorklet
 *   - `tapToAudibleMs`    : User tap on `#synthBtn` -> first audio sample actually played (`playedSamples > 0`)
 *   - `workerTtfaMs`      : Worker internal synthesis TTFA (`__ninadaLastResult.ttfaMs`)
 *   - `rtf`               : End-to-end synthesis Real-Time Factor
 *   - `underrunCount`     : Hardware AudioWorklet buffer underruns during full real-time drain
 *   - `maxRafGapMs`       : Worst main-thread animation frame gap during streaming
 */
import { spawn } from "node:child_process";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, devices } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, "..");
const PORT = Number(process.env.PORT || 4196);
const TARGET_URL = process.env.TARGET_URL || `http://127.0.0.1:${PORT}/`;
const USE_LOCAL_SERVER = !process.env.TARGET_URL;
const ENFORCE_GATES = process.env.ENFORCE_GATES === "1";
const JSON_OUT = process.env.JSON_OUT || "";

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

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

function round1(n) {
  return Math.round(Number(n || 0) * 10) / 10;
}

function round3(n) {
  return Math.round(Number(n || 0) * 1000) / 1000;
}

/**
 * Clicks `#synthBtn` inside the page and polls at 4ms resolution inside the renderer
 * to capture exact wall-clock ms from tap -> first pushed chunk -> first audible played sample,
 * then waits for full synthesis + full real-time AudioWorklet drain.
 */
async function measureTapStream(page, { waitDrain = true, timeoutMs = 90000 } = {}) {
  return await page.evaluate(
    async ({ waitDrainInner, timeoutMsInner }) => {
      window.__ninadaLastResult = null;
      const stats = window.__ninadaStreamStats;
      if (stats) {
        stats.totalPushedSamples = 0;
        stats.playedSamples = 0;
        stats.underrunCount = 0;
        stats.maxRafGapMs = 0;
      }

      const tTap = performance.now();
      document.getElementById("synthBtn").click();

      let tapToFirstPushMs = null;
      let tapToAudibleMs = null;
      const deadline = performance.now() + timeoutMsInner;

      while (performance.now() < deadline) {
        const curStats = window.__ninadaStreamStats || {};
        if (tapToFirstPushMs === null && curStats.totalPushedSamples > 0) {
          tapToFirstPushMs = performance.now() - tTap;
        }
        if (tapToAudibleMs === null && curStats.playedSamples > 0) {
          tapToAudibleMs = performance.now() - tTap;
        }
        if (window.__ninadaLastResult && tapToAudibleMs !== null) {
          break;
        }
        // If playback already finished very quickly (<42ms worklet progress interval)
        if (
          window.__ninadaLastResult &&
          !curStats.isPlaying &&
          curStats.endedCount > 0 &&
          curStats.playedSamples > 0
        ) {
          if (tapToFirstPushMs === null) tapToFirstPushMs = performance.now() - tTap;
          if (tapToAudibleMs === null) tapToAudibleMs = performance.now() - tTap;
          break;
        }
        await new Promise((r) => setTimeout(r, 4));
      }

      if (!window.__ninadaLastResult) {
        throw new Error("Timed out waiting for __ninadaLastResult during mobile stream");
      }

      let finalStats = { ...(window.__ninadaStreamStats || {}) };
      if (waitDrainInner && typeof window.__ninadaWaitForPlayback === "function") {
        const drained = await window.__ninadaWaitForPlayback();
        finalStats = { ...drained };
        if (tapToAudibleMs === null && finalStats.playedSamples > 0) {
          tapToAudibleMs = performance.now() - tTap;
        }
      } else if (typeof document.getElementById("stopBtn")?.click === "function") {
        document.getElementById("stopBtn").click();
      }

      const res = window.__ninadaLastResult;
      const jsHeapMb = performance.memory
        ? Math.round((performance.memory.usedJSHeapSize / (1024 * 1024)) * 10) / 10
        : null;

      return {
        lang: res.lang,
        threads: res.threads,
        chunkCount: res.chunkCount,
        audioSec: Math.round((res.samples / 24000) * 100) / 100,
        rms: Math.round(res.rms * 10000) / 10000,
        tapToFirstPushMs: tapToFirstPushMs !== null ? Math.round(tapToFirstPushMs * 10) / 10 : -1,
        tapToAudibleMs: tapToAudibleMs !== null ? Math.round(tapToAudibleMs * 10) / 10 : -1,
        workerTtfaMs: Math.round(res.ttfaMs * 10) / 10,
        rtf: Math.round(res.rtf * 1000) / 1000,
        underrunCount: finalStats.underrunCount ?? -1,
        playedSamples: finalStats.playedSamples ?? 0,
        totalSamples: res.samples,
        watermarkMs: Math.round(((finalStats.watermarkSamples || 0) / 24000) * 1000),
        maxRafGapMs: Math.round((finalStats.maxRafGapMs || 0) * 10) / 10,
        engine: finalStats.engine || "unknown",
        jsHeapMb,
      };
    },
    { waitDrainInner: waitDrain, timeoutMsInner: timeoutMs }
  );
}

async function runAndroidCoiProfile(browser, baseUrl) {
  const pixel = devices["Pixel 7"];
  const context = await browser.newContext({
    ...pixel,
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(`PAGEERROR: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`CONSOLE_ERROR: ${m.text()}`);
  });

  const cdp = await context.newCDPSession(page);
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 3 });

  const navStart = Date.now();
  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });

  // Wait for COI reload (if needed) + worker model ready unlock
  await page.waitForFunction(
    () => window.crossOriginIsolated === true && window.__ninadaReady === true && !window.__ninadaControlsLocked,
    null,
    { timeout: 120000 }
  );
  const readyUnlockMs = Date.now() - navStart;

  const paintMetrics = await page.evaluate(() => {
    const fcp = performance.getEntriesByName("first-contentful-paint")[0];
    return {
      fcpMs: fcp ? Math.round(fcp.startTime) : null,
      threads: window.__ninadaReadyInfo?.threads || 1,
    };
  });

  // Scenario A1: Immediate user tap 80ms after button unlocks!
  await page.waitForTimeout(80);
  const immediateTap = await measureTapStream(page, { waitDrain: true });

  // Scenario A2: Warm short utterance (Subhashita, 3 runs -> median)
  await page.locator(".preset-chip").first().click();
  const shortRuns = [];
  for (let i = 0; i < 3; i++) {
    shortRuns.push(await measureTapStream(page, { waitDrain: true }));
  }

  // Scenario A3: Switch to Kannada (`kn`) + 3-sentence Streaming Passage (2 runs -> median)
  await page.locator('.lang-btn[data-lang="kn"]').click();
  await page.waitForFunction(
    () => window.__ninadaReady === true && window.__ninadaLoadedLang === "kn" && !window.__ninadaControlsLocked,
    null,
    { timeout: 120000 }
  );
  // Wait for any background synthesis on tab switch to settle before clean passage timing
  await page.waitForTimeout(100);
  await page.locator(".preset-chip", { hasText: "Streaming Passage" }).click();
  const knLongRuns = [];
  for (let i = 0; i < 2; i++) {
    knLongRuns.push(await measureTapStream(page, { waitDrain: true }));
  }

  // Scenario A4: Warm page reload (cached model boot speed)
  const reloadStart = Date.now();
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => window.crossOriginIsolated === true && window.__ninadaReady === true && !window.__ninadaControlsLocked,
    null,
    { timeout: 90000 }
  );
  const warmReloadUnlockMs = Date.now() - reloadStart;
  const warmPaint = await page.evaluate(() => {
    const fcp = performance.getEntriesByName("first-contentful-paint")[0];
    return {
      fcpMs: fcp ? Math.round(fcp.startTime) : null,
      cacheHit: Boolean(window.__ninadaCacheHit),
    };
  });

  await context.close();

  return {
    profile: "android_coi_3x",
    threads: paintMetrics.threads,
    coldFcpMs: paintMetrics.fcpMs,
    coldReadyUnlockMs: readyUnlockMs,
    warmFcpMs: warmPaint.fcpMs,
    warmReloadUnlockMs,
    warmCacheHit: warmPaint.cacheHit,
    immediateTapAfterUnlock: immediateTap,
    warmShortSaMedian: {
      tapToFirstPushMs: round1(median(shortRuns.map((r) => r.tapToFirstPushMs))),
      tapToAudibleMs: round1(median(shortRuns.map((r) => r.tapToAudibleMs))),
      workerTtfaMs: round1(median(shortRuns.map((r) => r.workerTtfaMs))),
      rtf: round3(median(shortRuns.map((r) => r.rtf))),
      audioSec: shortRuns[0].audioSec,
      chunkCount: shortRuns[0].chunkCount,
      watermarkMs: shortRuns[0].watermarkMs,
      underrunCount: Math.max(...shortRuns.map((r) => r.underrunCount)),
      maxRafGapMs: round1(median(shortRuns.map((r) => r.maxRafGapMs))),
      runs: shortRuns,
    },
    warmLongKnMedian: {
      tapToFirstPushMs: round1(median(knLongRuns.map((r) => r.tapToFirstPushMs))),
      tapToAudibleMs: round1(median(knLongRuns.map((r) => r.tapToAudibleMs))),
      workerTtfaMs: round1(median(knLongRuns.map((r) => r.workerTtfaMs))),
      rtf: round3(median(knLongRuns.map((r) => r.rtf))),
      audioSec: knLongRuns[0].audioSec,
      chunkCount: knLongRuns[0].chunkCount,
      watermarkMs: knLongRuns[0].watermarkMs,
      underrunCount: Math.max(...knLongRuns.map((r) => r.underrunCount)),
      maxRafGapMs: round1(median(knLongRuns.map((r) => r.maxRafGapMs))),
      runs: knLongRuns,
    },
    errors,
  };
}

async function runIosSingleThreadProfile(browser, baseUrl) {
  const iphone = devices["iPhone 14"];
  // Use Chromium mobile emulation with serviceWorkers blocked + ?coi=0 to test single-thread WASM SIMD + AudioWorklet
  const context = await browser.newContext({
    viewport: iphone.viewport,
    deviceScaleFactor: iphone.deviceScaleFactor,
    isMobile: true,
    hasTouch: true,
    userAgent: iphone.userAgent,
    serviceWorkers: "block",
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(`ST_PAGEERROR: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`ST_CONSOLE_ERROR: ${m.text()}`);
  });

  const cdp = await context.newCDPSession(page);
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 2 });

  const sep = baseUrl.includes("?") ? "&" : "?";
  const stUrl = `${baseUrl}${sep}coi=0`;
  const navStart = Date.now();
  await page.goto(stUrl, { waitUntil: "domcontentloaded" });

  await page.waitForFunction(
    () => window.__ninadaReady === true && !window.__ninadaControlsLocked,
    null,
    { timeout: 120000 }
  );
  const readyUnlockMs = Date.now() - navStart;
  const bootInfo = await page.evaluate(() => ({
    coi: Boolean(window.crossOriginIsolated),
    threads: window.__ninadaReadyInfo?.threads || 1,
    fcpMs: Math.round(performance.getEntriesByName("first-contentful-paint")[0]?.startTime || 0),
  }));

  // Scenario B1: Immediate user tap 80ms after button unlocks!
  await page.waitForTimeout(80);
  const immediateTap = await measureTapStream(page, { waitDrain: true });

  // Scenario B2: Warm short utterance (3 runs -> median)
  await page.locator(".preset-chip").first().click();
  const shortRuns = [];
  for (let i = 0; i < 3; i++) {
    shortRuns.push(await measureTapStream(page, { waitDrain: true }));
  }

  // Scenario B3: Warm 3-clause Streaming Passage in single-thread mode!
  await page.locator(".preset-chip", { hasText: "IAST Streaming" }).click();
  const longRuns = [];
  for (let i = 0; i < 2; i++) {
    longRuns.push(await measureTapStream(page, { waitDrain: true }));
  }

  await context.close();

  return {
    profile: "ios_1thread_2x",
    coi: bootInfo.coi,
    threads: bootInfo.threads,
    coldFcpMs: bootInfo.fcpMs,
    coldReadyUnlockMs: readyUnlockMs,
    immediateTapAfterUnlock: immediateTap,
    warmShortSaMedian: {
      tapToFirstPushMs: round1(median(shortRuns.map((r) => r.tapToFirstPushMs))),
      tapToAudibleMs: round1(median(shortRuns.map((r) => r.tapToAudibleMs))),
      workerTtfaMs: round1(median(shortRuns.map((r) => r.workerTtfaMs))),
      rtf: round3(median(shortRuns.map((r) => r.rtf))),
      audioSec: shortRuns[0].audioSec,
      chunkCount: shortRuns[0].chunkCount,
      watermarkMs: shortRuns[0].watermarkMs,
      underrunCount: Math.max(...shortRuns.map((r) => r.underrunCount)),
      maxRafGapMs: round1(median(shortRuns.map((r) => r.maxRafGapMs))),
      runs: shortRuns,
    },
    warmLongSaMedian: {
      tapToFirstPushMs: round1(median(longRuns.map((r) => r.tapToFirstPushMs))),
      tapToAudibleMs: round1(median(longRuns.map((r) => r.tapToAudibleMs))),
      workerTtfaMs: round1(median(longRuns.map((r) => r.workerTtfaMs))),
      rtf: round3(median(longRuns.map((r) => r.rtf))),
      audioSec: longRuns[0].audioSec,
      chunkCount: longRuns[0].chunkCount,
      watermarkMs: longRuns[0].watermarkMs,
      underrunCount: Math.max(...longRuns.map((r) => r.underrunCount)),
      maxRafGapMs: round1(median(longRuns.map((r) => r.maxRafGapMs))),
      runs: longRuns,
    },
    errors,
  };
}

function printSummaryTable(report) {
  const a = report.android_coi_3x;
  const b = report.ios_1thread_2x;
  console.log("\n=======================================================================================");
  console.log("  NINADA MOBILE WEB STREAMING BENCHMARK REPORT");
  console.log("=======================================================================================");
  console.log(
    `  Android COI (3x CPU, ${a.threads}T) | Cold FCP: ${a.coldFcpMs}ms | Cold Ready: ${a.coldReadyUnlockMs}ms | Warm Ready: ${a.warmReloadUnlockMs}ms`
  );
  console.log(
    `    - Immediate Tap (SA)  : Audible ${a.immediateTapAfterUnlock.tapToAudibleMs}ms | Push ${a.immediateTapAfterUnlock.tapToFirstPushMs}ms | WorkerTTFA ${a.immediateTapAfterUnlock.workerTtfaMs}ms | RTF ${a.immediateTapAfterUnlock.rtf}x | Underruns ${a.immediateTapAfterUnlock.underrunCount}`
  );
  console.log(
    `    - Warm Short (SA)     : Audible ${a.warmShortSaMedian.tapToAudibleMs}ms | Push ${a.warmShortSaMedian.tapToFirstPushMs}ms | WorkerTTFA ${a.warmShortSaMedian.workerTtfaMs}ms | RTF ${a.warmShortSaMedian.rtf}x | Underruns ${a.warmShortSaMedian.underrunCount} | rAF ${a.warmShortSaMedian.maxRafGapMs}ms`
  );
  console.log(
    `    - Warm 3-Sent (KN)    : Audible ${a.warmLongKnMedian.tapToAudibleMs}ms | Push ${a.warmLongKnMedian.tapToFirstPushMs}ms | WorkerTTFA ${a.warmLongKnMedian.workerTtfaMs}ms | RTF ${a.warmLongKnMedian.rtf}x | Underruns ${a.warmLongKnMedian.underrunCount} | rAF ${a.warmLongKnMedian.maxRafGapMs}ms`
  );
  console.log("---------------------------------------------------------------------------------------");
  console.log(
    `  iOS 1-Thread (2x CPU, ${b.threads}T, coi=${b.coi}) | Cold FCP: ${b.coldFcpMs}ms | Cold Ready: ${b.coldReadyUnlockMs}ms`
  );
  console.log(
    `    - Immediate Tap (SA)  : Audible ${b.immediateTapAfterUnlock.tapToAudibleMs}ms | Push ${b.immediateTapAfterUnlock.tapToFirstPushMs}ms | WorkerTTFA ${b.immediateTapAfterUnlock.workerTtfaMs}ms | RTF ${b.immediateTapAfterUnlock.rtf}x | Underruns ${b.immediateTapAfterUnlock.underrunCount}`
  );
  console.log(
    `    - Warm Short (SA)     : Audible ${b.warmShortSaMedian.tapToAudibleMs}ms | Push ${b.warmShortSaMedian.tapToFirstPushMs}ms | WorkerTTFA ${b.warmShortSaMedian.workerTtfaMs}ms | RTF ${b.warmShortSaMedian.rtf}x | Underruns ${b.warmShortSaMedian.underrunCount} | rAF ${b.warmShortSaMedian.maxRafGapMs}ms`
  );
  console.log(
    `    - Warm 3-Sent (SA)    : Audible ${b.warmLongSaMedian.tapToAudibleMs}ms | Push ${b.warmLongSaMedian.tapToFirstPushMs}ms | WorkerTTFA ${b.warmLongSaMedian.workerTtfaMs}ms | RTF ${b.warmLongSaMedian.rtf}x | Underruns ${b.warmLongSaMedian.underrunCount} | rAF ${b.warmLongSaMedian.maxRafGapMs}ms`
  );
  console.log("=======================================================================================\n");
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

    const browser = await chromium.launch({
      headless: true,
      args: [
        "--autoplay-policy=no-user-gesture-required",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
      ],
    });

    const androidReport = await runAndroidCoiProfile(browser, TARGET_URL);
    const iosReport = await runIosSingleThreadProfile(browser, TARGET_URL);
    await browser.close();

    const fullReport = {
      timestamp: new Date().toISOString(),
      targetUrl: TARGET_URL,
      android_coi_3x: androidReport,
      ios_1thread_2x: iosReport,
    };

    printSummaryTable(fullReport);

    if (JSON_OUT) {
      fs.mkdirSync(path.dirname(path.resolve(JSON_OUT)), { recursive: true });
      fs.writeFileSync(path.resolve(JSON_OUT), JSON.stringify(fullReport, null, 2));
    }

    const allErrors = [...androidReport.errors, ...iosReport.errors];
    assert.equal(allErrors.length, 0, `Errors during mobile benchmark: ${allErrors.join(" | ")}`);

    if (ENFORCE_GATES) {
      // Strict Mobile Web Streaming Regression Gates
      assert.equal(androidReport.immediateTapAfterUnlock.underrunCount, 0, "Android immediate-tap stream had underruns");
      assert.equal(androidReport.warmShortSaMedian.underrunCount, 0, "Android short stream had underruns");
      assert.equal(androidReport.warmLongKnMedian.underrunCount, 0, "Android KN 3-sentence stream had underruns");
      assert.equal(iosReport.immediateTapAfterUnlock.underrunCount, 0, "iOS 1-thread immediate-tap stream had underruns");
      assert.equal(iosReport.warmShortSaMedian.underrunCount, 0, "iOS 1-thread short stream had underruns");
      assert.equal(iosReport.warmLongSaMedian.underrunCount, 0, "iOS 1-thread 3-sentence stream had underruns");

      assert.ok(
        androidReport.immediateTapAfterUnlock.tapToAudibleMs < 550,
        `Android Immediate-Tap-to-Audible too slow: ${androidReport.immediateTapAfterUnlock.tapToAudibleMs}ms`
      );
      assert.ok(
        androidReport.warmShortSaMedian.tapToAudibleMs < 450,
        `Android Warm Short Tap-to-Audible too slow: ${androidReport.warmShortSaMedian.tapToAudibleMs}ms`
      );
      assert.ok(
        androidReport.warmLongKnMedian.tapToAudibleMs < 500,
        `Android Warm 3-Sentence Tap-to-Audible too slow: ${androidReport.warmLongKnMedian.tapToAudibleMs}ms`
      );
      assert.ok(
        iosReport.warmShortSaMedian.workerTtfaMs < 850,
        `iOS 1-Thread Worker TTFA too slow: ${iosReport.warmShortSaMedian.workerTtfaMs}ms`
      );
      assert.ok(
        iosReport.immediateTapAfterUnlock.tapToAudibleMs < 2600,
        `iOS 1-Thread Immediate-Tap-to-Audible too slow: ${iosReport.immediateTapAfterUnlock.tapToAudibleMs}ms`
      );
      assert.ok(
        iosReport.warmShortSaMedian.tapToAudibleMs < 2600,
        `iOS 1-Thread Warm Short Tap-to-Audible too slow: ${iosReport.warmShortSaMedian.tapToAudibleMs}ms`
      );
      assert.ok(
        iosReport.warmLongSaMedian.tapToAudibleMs < 5200,
        `iOS 1-Thread 3-Sentence Tap-to-Audible too slow: ${iosReport.warmLongSaMedian.tapToAudibleMs}ms`
      );
    }
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
