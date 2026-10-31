#!/usr/bin/env node
import { spawn } from "node:child_process";
import assert from "node:assert/strict";
import path from "node:path";
import { chromium } from "playwright";

const PORT = 4192;
const URL = `http://127.0.0.1:${PORT}/`;
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
  const server = spawn(process.execPath, ["serve-playground.mjs"], {
    env: { ...process.env, PORT: String(PORT) },
    stdio: ["ignore", "pipe", "pipe"],
  });

  try {
    await waitForServer(URL);
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({ viewport: { width: 1280, height: 920 } });
    const page = await context.newPage();

    const errors = [];
    page.on("pageerror", (err) => errors.push(`PAGEERROR: ${err.message}`));
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(`CONSOLE_ERROR: ${msg.text()}`);
    });

    await page.goto(URL, { waitUntil: "domcontentloaded" });

    await page.waitForFunction(
      () => window.__ninadaReady === true && window.__ninadaLastResult && window.__ninadaLastResult.tier === "neural_onnx",
      { timeout: 60000 }
    );

    const saResult = await page.evaluate(() => window.__ninadaLastResult);
    console.log("Sanskrit Neural ONNX Browser Result:", saResult);
    assert.equal(saResult.isNeural, true);
    assert.equal(saResult.tier, "neural_onnx");
    assert.equal(saResult.lang, "sa");
    assert.ok(saResult.samples > 16000, `Expected >1s audio, got ${saResult.samples} samples`);
    assert.ok(saResult.rms > 0.02, `Expected non-silent RMS > 0.02, got ${saResult.rms}`);
    assert.ok(saResult.wavBytes > 32000);

    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, "ninada_studio_desktop.png"),
      fullPage: true,
    });

    // Test Classical/Vedic Sanskrit Rigveda & Vedanta preset
    await page.locator(".preset-chip", { hasText: "Rigveda" }).click();
    await page.evaluate(() => { window.__ninadaLastResult = null; });
    await page.locator("#synthBtn").click();
    await page.waitForFunction(
      () => window.__ninadaLastResult && window.__ninadaLastResult.lang === "sa" && window.__ninadaLastResult.tier === "neural_onnx",
      { timeout: 30000 }
    );
    const vedicResult = await page.evaluate(() => window.__ninadaLastResult);
    console.log("Rigveda Sanskrit Neural ONNX Browser Result:", vedicResult);
    assert.equal(vedicResult.lang, "sa");
    assert.equal(vedicResult.tier, "neural_onnx");
    assert.ok(vedicResult.rms > 0.02);

    // Test Kannada preset
    await page.locator(".preset-chip", { hasText: "Kannada" }).click();
    await page.evaluate(() => { window.__ninadaLastResult = null; });
    await page.locator("#synthBtn").click();
    await page.waitForFunction(
      () => window.__ninadaLastResult && window.__ninadaLastResult.lang === "kn" && window.__ninadaLastResult.tier === "neural_onnx",
      { timeout: 30000 }
    );
    const knResult = await page.evaluate(() => window.__ninadaLastResult);
    console.log("Kannada Neural ONNX Browser Result:", knResult);
    assert.equal(knResult.lang, "kn");
    assert.equal(knResult.tier, "neural_onnx");
    assert.ok(knResult.rms > 0.02);

    // Test Tamil preset
    await page.locator(".preset-chip", { hasText: "Tamil" }).click();
    await page.evaluate(() => { window.__ninadaLastResult = null; });
    await page.locator("#synthBtn").click();
    await page.waitForFunction(
      () => window.__ninadaLastResult && window.__ninadaLastResult.lang === "ta" && window.__ninadaLastResult.tier === "neural_onnx",
      { timeout: 30000 }
    );
    const taResult = await page.evaluate(() => window.__ninadaLastResult);
    console.log("Tamil Neural ONNX Browser Result:", taResult);
    assert.equal(taResult.lang, "ta");
    assert.equal(taResult.tier, "neural_onnx");
    assert.ok(taResult.rms > 0.02);

    // Check mobile viewport responsiveness
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, "ninada_studio_mobile.png"),
      fullPage: false,
    });

    assert.equal(errors.length, 0, `Browser errors detected: ${errors.join(" | ")}`);
    await browser.close();
    console.log("ALL NINADA BROWSER ONNX WASM CHECKS PASSED WITH ZERO ERRORS!");
  } finally {
    server.kill("SIGTERM");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
