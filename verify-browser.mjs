#!/usr/bin/env node
import { spawn } from "node:child_process";
import assert from "node:assert/strict";
import path from "node:path";
import { chromium } from "playwright";

const PORT = Number(process.env.PORT || 4192);
const TARGET_URL = process.env.TARGET_URL || `http://127.0.0.1:${PORT}/`;
const USE_LOCAL_SERVER = !process.env.TARGET_URL;
const SCREENSHOT_DIR =
  process.env.SCREENSHOT_DIR ||
  "/Users/lika/.gemini/antigravity-cli/brain/83862443-6867-4d2e-b479-e41dc16a9673";

async function waitForServer(url, timeoutMs = 8000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(url);
      if (r.ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 150));
  }
  throw new Error(`Server at ${url} did not start within ${timeoutMs}ms`);
}

async function main() {
  const server = USE_LOCAL_SERVER
    ? spawn(process.execPath, ["serve-playground.mjs"], {
        env: { ...process.env, PORT: String(PORT) },
        stdio: ["ignore", "pipe", "pipe"],
      })
    : null;

  try {
    if (server) {
      await waitForServer(TARGET_URL);
    }

    const launchOpts = { headless: true };
    if (process.env.BROWSER_CHANNEL) {
      launchOpts.channel = process.env.BROWSER_CHANNEL;
    }
    const browser = await chromium.launch(launchOpts);
    const context = await browser.newContext({ viewport: { width: 1280, height: 920 } });
    const page = await context.newPage();

    const errors = [];
    page.on("pageerror", (err) => errors.push(`PAGEERROR: ${err.message}`));
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(`CONSOLE_ERROR: ${msg.text()}`);
    });
    page.on("response", (res) => {
      const u = res.url();
      // Ignore optional local probe when falling back or HF check, flag any 4xx/5xx on required page assets
      if (res.status() >= 400 && !u.endsWith(".onnx")) {
        errors.push(`HTTP_${res.status()}: ${u}`);
      }
    });

    await page.goto(TARGET_URL, { waitUntil: "domcontentloaded" });

    await page.waitForFunction(
      () => window.__ninadaReady === true && window.__ninadaLastResult && window.__ninadaLastResult.tier === "neural_onnx",
      null,
      { timeout: 90000 }
    );

    const saResult = await page.evaluate(() => window.__ninadaLastResult);
    console.log("Sanskrit Streaming Neural ONNX Result:", saResult);
    assert.equal(saResult.isNeural, true);
    assert.equal(saResult.tier, "neural_onnx");
    assert.equal(saResult.lang, "sa");
    assert.ok(saResult.samples > 16000, `Expected >1s audio, got ${saResult.samples} samples`);
    assert.ok(saResult.rms > 0.02, `Expected non-silent RMS > 0.02, got ${saResult.rms}`);
    assert.ok(saResult.ttfaMs > 0, `Expected positive TTFA ms, got ${saResult.ttfaMs}`);
    assert.ok(saResult.wavBytes > 32000);

    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, "ninada_studio_desktop.png"),
      fullPage: true,
    });

    // Verify EVERY language tab immediately updates presets AND populates native-script text!
    const expectedScriptRanges = [
      { lang: "kn", re: /[\u0C80-\u0CFF]/, minPresets: 4 },
      { lang: "hi", re: /[\u0900-\u097F]/, minPresets: 3 },
      { lang: "ta", re: /[\u0B80-\u0BFF]/, minPresets: 3 },
      { lang: "te", re: /[\u0C00-\u0C7F]/, minPresets: 3 },
      { lang: "ml", re: /[\u0D00-\u0D7F]/, minPresets: 3 },
      { lang: "sa", re: /[\u0900-\u097F]/, minPresets: 4 },
    ];

    for (const item of expectedScriptRanges) {
      await page.locator(`.lang-btn[data-lang="${item.lang}"]`).click();
      const tabState = await page.evaluate(() => ({
        val: document.getElementById("textInput").value,
        presets: Array.from(document.querySelectorAll(".preset-chip")).map((b) => b.textContent),
        badge: document.getElementById("detectedBadge").textContent,
      }));
      assert.ok(
        item.re.test(tabState.val),
        `Clicking ${item.lang} tab did not populate native script text! Got: ${tabState.val}`
      );
      assert.ok(
        tabState.presets.length >= item.minPresets,
        `Expected >=${item.minPresets} presets for ${item.lang}, got ${tabState.presets.length}`
      );
      assert.ok(tabState.badge.includes(item.lang.toUpperCase()));
    }

    // Test Kannada multi-sentence streaming AND measure main-thread UI responsiveness (must NOT freeze!)
    await page.locator('.lang-btn[data-lang="kn"]').click();
    await page.locator(".preset-chip", { hasText: "Streaming Passage" }).click();
    await page.evaluate(() => {
      window.__ninadaLastResult = null;
      window.__maxMainThreadGapMs = 0;
      window.__rafRunning = true;
      let last = performance.now();
      function loop(now) {
        const dt = now - last;
        if (dt > window.__maxMainThreadGapMs) window.__maxMainThreadGapMs = dt;
        last = now;
        if (window.__rafRunning) requestAnimationFrame(loop);
      }
      requestAnimationFrame(loop);
    });

    await page.locator("#synthBtn").click();
    await page.waitForFunction(
      () => window.__ninadaLastResult && window.__ninadaLastResult.lang === "kn" && window.__ninadaLastResult.tier === "neural_onnx",
      null,
      { timeout: 60000 }
    );
    const knStream = await page.evaluate(() => {
      window.__rafRunning = false;
      return {
        result: window.__ninadaLastResult,
        maxMainThreadGapMs: window.__maxMainThreadGapMs,
        crossOriginIsolated: window.crossOriginIsolated,
      };
    });
    console.log("Kannada Multi-Sentence Web Worker Stream:", knStream);
    assert.equal(knStream.result.lang, "kn");
    assert.equal(knStream.result.tier, "neural_onnx");
    assert.ok(knStream.result.chunkCount >= 3, `Expected >=3 streamed chunks, got ${knStream.result.chunkCount}`);
    assert.ok(knStream.result.rms > 0.02);
    assert.ok(
      knStream.maxMainThreadGapMs < 250,
      `Main thread blocked for ${knStream.maxMainThreadGapMs.toFixed(1)}ms (expected <250ms in Web Worker!)`
    );

    // Test Tamil streaming synthesis
    await page.locator('.lang-btn[data-lang="ta"]').click();
    await page.evaluate(() => { window.__ninadaLastResult = null; });
    await page.locator("#synthBtn").click();
    await page.waitForFunction(
      () => window.__ninadaLastResult && window.__ninadaLastResult.lang === "ta" && window.__ninadaLastResult.tier === "neural_onnx",
      null,
      { timeout: 45000 }
    );
    const taResult = await page.evaluate(() => window.__ninadaLastResult);
    console.log("Tamil Streaming Neural ONNX Result:", taResult);
    assert.equal(taResult.lang, "ta");
    assert.ok(taResult.rms > 0.02);

    // Check mobile viewport responsiveness
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('.lang-btn[data-lang="kn"]').click();
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, "ninada_studio_mobile.png"),
      fullPage: false,
    });

    assert.equal(errors.length, 0, `Browser errors detected: ${errors.join(" | ")}`);
    await browser.close();
    console.log(`ALL NINADA STREAMING BROWSER CHECKS PASSED ON ${TARGET_URL} WITH ZERO ERRORS!`);
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
