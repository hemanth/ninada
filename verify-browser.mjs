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

async function waitForIsolatedBoot(page, timeoutMs = 10000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const state = await page.evaluate(() => ({
        coi: Boolean(window.crossOriginIsolated),
        hasLockFlag: typeof window.__ninadaControlsLocked === "boolean",
      }));
      if (state.coi && state.hasLockFlag) {
        return state;
      }
    } catch {
      // Transient navigation while coi-sw.js activates and reloads once (<50ms)
    }
    await new Promise((r) => setTimeout(r, 40));
  }
  throw new Error("Timed out waiting for Cross-Origin Isolation (coi-sw.js) boot");
}

async function main() {
  // Run local server WITHOUT HTTP COOP/COEP headers (COOP_COEP=0) so coi-sw.js is tested identically to GitHub Pages!
  const server = USE_LOCAL_SERVER
    ? spawn(process.execPath, ["serve-playground.mjs"], {
        env: { ...process.env, PORT: String(PORT), COOP_COEP: "0" },
        stdio: ["ignore", "pipe", "pipe"],
      })
    : null;

  try {
    if (server) {
      await waitForServer(TARGET_URL);
    }

    const launchOpts = {
      headless: true,
      args: [
        "--autoplay-policy=no-user-gesture-required",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
      ],
    };
    if (process.env.BROWSER_CHANNEL) {
      launchOpts.channel = process.env.BROWSER_CHANNEL;
    }
    const browser = await chromium.launch(launchOpts);
    const context = await browser.newContext({ viewport: { width: 1280, height: 920 } });
    const page = await context.newPage();

    const errors = [];
    let holdNextAssetMs = 600;
    let activeBucket = "phase1";
    const onnxBuckets = {
      phase1: [],
      reload: [],
      switchLangs: [],
      revisitCached: [],
    };

    await context.route("**/*.onnx*", async (route) => {
      const u = route.request().url();
      onnxBuckets[activeBucket].push(u);
      if (holdNextAssetMs > 0) {
        const delay = holdNextAssetMs;
        holdNextAssetMs = 0;
        await new Promise((r) => setTimeout(r, delay));
      }
      await route.continue();
    });

    page.on("pageerror", (err) => errors.push(`PAGEERROR: ${err.message}`));
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(`CONSOLE_ERROR: ${msg.text()}`);
    });
    page.on("response", (res) => {
      const u = res.url();
      if (res.status() >= 400 && !u.endsWith(".onnx")) {
        errors.push(`HTTP_${res.status()}: ${u}`);
      }
    });

    await page.goto(TARGET_URL, { waitUntil: "domcontentloaded" });
    await waitForIsolatedBoot(page);

    // 1. Verify Cross-Origin Isolation + NO buttons or studio controls work before the model finishes loading!
    const preLoadLock = await page.evaluate(() => {
      const knBtn = document.querySelector('.lang-btn[data-lang="kn"]');
      const synthBtn = document.getElementById("synthBtn");
      const replayBtn = document.getElementById("replayBtn");
      const stopBtn = document.getElementById("stopBtn");
      const dl = document.getElementById("downloadLink");
      const langDisabled = Array.from(document.querySelectorAll(".lang-btn")).every((b) => b.disabled === true);
      const presetDisabled = Array.from(document.querySelectorAll(".preset-chip")).every((b) => b.disabled === true);

      // Attempt synthetic clicks while locked — must be completely ignored!
      if (knBtn) knBtn.click();
      if (synthBtn) synthBtn.click();

      return {
        crossOriginIsolated: Boolean(window.crossOriginIsolated),
        ready: Boolean(window.__ninadaReady),
        controlsLocked: window.__ninadaControlsLocked,
        synthDisabled: synthBtn?.disabled,
        synthLabel: document.getElementById("synthBtnLabel")?.textContent,
        replayDisabled: replayBtn?.disabled,
        stopDisabled: stopBtn?.disabled,
        textDisabled: document.getElementById("textInput")?.disabled,
        tierDisabled: document.getElementById("tierSelect")?.disabled,
        rateDisabled: document.getElementById("rateSlider")?.disabled,
        noiseDisabled: document.getElementById("noiseSlider")?.disabled,
        downloadDisabled: dl?.classList.contains("is-disabled") && dl?.getAttribute("aria-disabled") === "true",
        langDisabled,
        presetDisabled,
        badgeAfterBlockedClick: document.getElementById("detectedBadge")?.textContent,
      };
    });
    console.log("Pre-load COI & Button Lock Verification:", preLoadLock);
    assert.equal(preLoadLock.crossOriginIsolated, true, "Expected window.crossOriginIsolated === true via coi-sw.js");
    assert.equal(preLoadLock.ready, false, "Expected model to still be loading during 600ms ONNX hold");
    assert.equal(preLoadLock.controlsLocked, true);
    assert.equal(preLoadLock.synthDisabled, true);
    assert.equal(preLoadLock.synthLabel, "Loading Model...");
    assert.equal(preLoadLock.replayDisabled, true);
    assert.equal(preLoadLock.stopDisabled, true);
    assert.equal(preLoadLock.textDisabled, true);
    assert.equal(preLoadLock.tierDisabled, true);
    assert.equal(preLoadLock.rateDisabled, true);
    assert.equal(preLoadLock.noiseDisabled, true);
    assert.equal(preLoadLock.downloadDisabled, true);
    assert.equal(preLoadLock.langDisabled, true);
    assert.equal(preLoadLock.presetDisabled, true);
    assert.ok(
      preLoadLock.badgeAfterBlockedClick.includes("SA"),
      `Blocked click should not switch language; got ${preLoadLock.badgeAfterBlockedClick}`
    );

    await page.waitForFunction(
      () =>
        window.__ninadaReady === true &&
        window.__ninadaLoadedLang === "sa" &&
        window.__ninadaLastResult &&
        window.__ninadaLastResult.tier === "neural_onnx",
      null,
      { timeout: 90000 }
    );

    const postLoadState = await page.evaluate(() => ({
      crossOriginIsolated: Boolean(window.crossOriginIsolated),
      controlsLocked: window.__ninadaControlsLocked,
      loadedLang: window.__ninadaLoadedLang,
      modelCached: window.__ninadaModelCached,
      cacheHitFirstLoad: window.__ninadaCacheHit,
      readyInfo: window.__ninadaReadyInfo,
      synthDisabled: document.getElementById("synthBtn")?.disabled,
      langAnyDisabled: Array.from(document.querySelectorAll(".lang-btn")).some((b) => b.disabled),
      downloadDisabled: document.getElementById("downloadLink")?.classList.contains("is-disabled"),
    }));
    console.log("Post-load Multi-Thread WASM, Cache & Unlock State:", postLoadState);
    assert.equal(postLoadState.crossOriginIsolated, true);
    assert.ok(
      postLoadState.readyInfo.threads >= 2,
      `Expected >=2 WASM threads under COI, got ${postLoadState.readyInfo.threads}`
    );
    assert.equal(postLoadState.controlsLocked, false);
    assert.equal(postLoadState.loadedLang, "sa");
    assert.equal(postLoadState.modelCached, true, "Expected ONNX models to be persisted in browser cache");
    assert.equal(postLoadState.synthDisabled, false);
    assert.equal(postLoadState.langAnyDisabled, false);
    assert.equal(postLoadState.downloadDisabled, false);
    assert.equal(
      onnxBuckets.phase1.length,
      1,
      `Expected 1 native SA ONNX fetch on cold load, got ${onnxBuckets.phase1.length}: ${onnxBuckets.phase1.join(", ")}`
    );

    // 2. Reload page in same browser context — verify immediate COI + 0 network ONNX requests (100% browser cache hit!)
    activeBucket = "reload";
    await page.reload({ waitUntil: "domcontentloaded" });
    await page.waitForFunction(
      () =>
        window.__ninadaReady === true &&
        window.__ninadaLoadedLang === "sa" &&
        window.__ninadaLastResult &&
        window.__ninadaLastResult.tier === "neural_onnx",
      null,
      { timeout: 90000 }
    );
    const reloadCacheState = await page.evaluate(() => ({
      crossOriginIsolated: Boolean(window.crossOriginIsolated),
      cacheHit: window.__ninadaCacheHit,
      modelCached: window.__ninadaModelCached,
      controlsLocked: window.__ninadaControlsLocked,
      loaderText: document.getElementById("loaderLabel")?.textContent,
    }));
    console.log("Reload Browser Cache Verification:", {
      ...reloadCacheState,
      onnxNetworkRequestsOnReload: onnxBuckets.reload.length,
    });
    assert.equal(reloadCacheState.crossOriginIsolated, true);
    assert.equal(
      onnxBuckets.reload.length,
      0,
      `Expected 0 ONNX network requests on reload (cached), got ${onnxBuckets.reload.length}: ${onnxBuckets.reload.join(", ")}`
    );
    assert.equal(reloadCacheState.cacheHit, true, "Expected window.__ninadaCacheHit === true on reload");
    assert.equal(reloadCacheState.controlsLocked, false);
    assert.ok(reloadCacheState.loaderText.includes("Loaded from Cache"));

    const saResult = await page.evaluate(() => window.__ninadaLastResult);
    console.log("Sanskrit 24 kHz Streaming Neural ONNX Result:", saResult);
    assert.equal(saResult.isNeural, true);
    assert.equal(saResult.tier, "neural_onnx");
    assert.equal(saResult.lang, "sa");
    assert.ok(saResult.samples > 24000, `Expected >1s 24kHz audio, got ${saResult.samples} samples`);
    assert.ok(saResult.rms > 0.02, `Expected non-silent RMS > 0.02, got ${saResult.rms}`);
    assert.ok(saResult.ttfaMs > 0 && saResult.ttfaMs < 1500, `Expected TTFA < 1500ms, got ${saResult.ttfaMs}`);
    assert.ok(saResult.rtf < 0.95, `Expected multi-threaded RTF < 0.95 (<1.0x real-time), got ${saResult.rtf}`);
    assert.ok(saResult.wavBytes > 48000);

    // Verify highlighted SDK code showcase, install pills, and interactive code tabs
    const codeCheckInitial = await page.evaluate(() => ({
      installCount: document.querySelectorAll(".install-pill").length,
      tabCount: document.querySelectorAll(".code-tab-btn").length,
      kwCount: document.querySelectorAll("#sdkSnippet .tok-kw").length,
      strCount: document.querySelectorAll("#sdkSnippet .tok-str").length,
      fnCount: document.querySelectorAll("#sdkSnippet .tok-fn").length,
      badge: document.getElementById("sdkLangBadge")?.textContent,
    }));
    assert.equal(codeCheckInitial.installCount, 2, "Expected npm and pip install pills in masthead");
    assert.equal(codeCheckInitial.tabCount, 4, "Expected 4 SDK code showcase tabs");
    assert.ok(codeCheckInitial.kwCount >= 4 && codeCheckInitial.strCount >= 3 && codeCheckInitial.fnCount >= 2);
    assert.ok(codeCheckInitial.badge.includes("SA"));

    await page.locator('.code-tab-btn[data-code-tab="python"]').click();
    const pyCodeText = await page.locator("#sdkSnippet").textContent();
    assert.ok(pyCodeText.includes("from ninada import Ninada"), "Expected Python tab snippet");
    await page.locator('.code-tab-btn[data-code-tab="js"]').click();

    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, "ninada_studio_desktop.png"),
      fullPage: true,
    });

    // 3. Verify language-switch button lock + 0 ONNX recompilations across all 6 languages!
    activeBucket = "switchLangs";
    holdNextAssetMs = 450;
    await page.locator('.lang-btn[data-lang="kn"]').click();

    const switchLock = await page.evaluate(() => ({
      ready: Boolean(window.__ninadaReady),
      controlsLocked: window.__ninadaControlsLocked,
      synthDisabled: document.getElementById("synthBtn")?.disabled,
      synthLabel: document.getElementById("synthBtnLabel")?.textContent,
      langDisabled: Array.from(document.querySelectorAll(".lang-btn")).every((b) => b.disabled === true),
    }));
    console.log("Language-Switch Button Lock Verification (SA -> KN):", switchLock);
    assert.equal(switchLock.ready, false, "Expected model ready=false while KN specialist loads");
    assert.equal(switchLock.controlsLocked, true, "Expected all buttons locked while KN specialist loads");
    assert.equal(switchLock.synthDisabled, true);
    assert.equal(switchLock.synthLabel, "Loading Model...");
    assert.equal(switchLock.langDisabled, true);

    await page.waitForFunction(
      () => window.__ninadaReady === true && window.__ninadaLoadedLang === "kn" && !window.__ninadaControlsLocked,
      null,
      { timeout: 90000 }
    );

    const expectedScriptRanges = [
      { lang: "kn", re: /[\u0C80-\u0CFF]/, minPresets: 4 },
      { lang: "hi", re: /[\u0900-\u097F]/, minPresets: 3 },
      { lang: "ta", re: /[\u0B80-\u0BFF]/, minPresets: 3 },
      { lang: "te", re: /[\u0C00-\u0C7F]/, minPresets: 3 },
      { lang: "ml", re: /[\u0D00-\u0D7F]/, minPresets: 3 },
      { lang: "sa", re: /[\u0900-\u097F]/, minPresets: 4 },
    ];

    for (const item of expectedScriptRanges) {
      const curLang = await page.evaluate(() => window.__ninadaLoadedLang);
      if (curLang !== item.lang) {
        await page.locator(`.lang-btn[data-lang="${item.lang}"]`).click();
        await page.waitForFunction(
          (lc) => window.__ninadaReady === true && window.__ninadaLoadedLang === lc && !window.__ninadaControlsLocked,
          item.lang,
          { timeout: 90000 }
        );
      }
      const tabState = await page.evaluate(() => ({
        val: document.getElementById("textInput").value,
        presets: Array.from(document.querySelectorAll(".preset-chip")).map((b) => b.textContent),
        badge: document.getElementById("detectedBadge").textContent,
        modelCached: window.__ninadaModelCached,
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
      assert.equal(tabState.modelCached, true, `Expected ${item.lang} ONNX specialist to be cached`);
    }

    assert.equal(
      onnxBuckets.switchLangs.length,
      5,
      `Expected exactly 5 native specialist ONNX downloads (kn, hi, ta, te, ml) when visiting all 6 tabs, got ${onnxBuckets.switchLangs.length}: ${onnxBuckets.switchLangs.join(", ")}`
    );

    // 4. Revisit cached Kannada (`kn`) — verify 0 network ONNX requests + Real-Time AudioWorklet Multi-Sentence Smoothness!
    activeBucket = "revisitCached";
    await page.locator('.lang-btn[data-lang="kn"]').click();
    await page.waitForFunction(
      () => window.__ninadaReady === true && window.__ninadaLoadedLang === "kn" && !window.__ninadaControlsLocked,
      null,
      { timeout: 90000 }
    );
    assert.equal(
      onnxBuckets.revisitCached.length,
      0,
      `Expected 0 ONNX network requests when switching back to cached KN, got ${onnxBuckets.revisitCached.length}`
    );

    await page.locator(".preset-chip", { hasText: "Streaming Passage" }).click();
    await page.evaluate(() => {
      window.__ninadaLastResult = null;
    });

    await page.locator("#synthBtn").click();
    await page.waitForFunction(
      () =>
        window.__ninadaLastResult &&
        window.__ninadaLastResult.lang === "kn" &&
        window.__ninadaLastResult.tier === "neural_onnx",
      null,
      { timeout: 60000 }
    );

    // Wait for full real-time hardware AudioWorklet playback drain!
    const knStream = await page.evaluate(async () => {
      const midStreamLabel = document.getElementById("synthBtnLabel")?.textContent;
      const statsAfterDrain = await window.__ninadaWaitForPlayback();
      return {
        result: window.__ninadaLastResult,
        midStreamLabel,
        stats: { ...statsAfterDrain },
        finalBtnLabel: document.getElementById("synthBtnLabel")?.textContent,
        stopDisabledAfterEnd: document.getElementById("stopBtn")?.disabled,
        audioPlayerHasBlob: (document.getElementById("audioPlayer")?.src || "").startsWith("blob:"),
      };
    });
    console.log("Kannada 3-Sentence Real-Time AudioWorklet Stream:", knStream);
    assert.equal(knStream.result.lang, "kn");
    assert.equal(knStream.result.tier, "neural_onnx");
    assert.ok(knStream.result.chunkCount >= 3, `Expected >=3 streamed chunks, got ${knStream.result.chunkCount}`);
    assert.ok(knStream.result.rms > 0.02);
    assert.ok(knStream.result.ttfaMs < 1500, `Expected KN TTFA < 1500ms, got ${knStream.result.ttfaMs}`);
    assert.ok(knStream.result.rtf < 0.95, `Expected KN RTF < 0.95 (<1.0x real-time), got ${knStream.result.rtf}`);
    assert.equal(knStream.stats.engine, "audioworklet", `Expected AudioWorklet engine, got ${knStream.stats.engine}`);
    assert.equal(
      knStream.stats.underrunCount,
      0,
      `Expected ZERO audio buffer underruns during 3-sentence KN stream, got ${knStream.stats.underrunCount}!`
    );
    assert.ok(
      knStream.stats.playedSamples >= Math.floor(knStream.result.samples * 0.99),
      `Expected full sample drain (${knStream.result.samples}), played ${knStream.stats.playedSamples}`
    );
    assert.ok(
      knStream.stats.maxRafGapMs < 120,
      `Expected smooth 60fps UI (<120ms max rAF gap), got ${knStream.stats.maxRafGapMs.toFixed(1)}ms`
    );
    assert.equal(knStream.finalBtnLabel, "Stream Speech");
    assert.equal(knStream.stopDisabledAfterEnd, true);
    assert.equal(knStream.audioPlayerHasBlob, true);

    // 5. Test cached Tamil (`ta`) multi-sentence real-time AudioWorklet streaming (0 network ONNX requests, 0 underruns!)
    await page.locator('.lang-btn[data-lang="ta"]').click();
    await page.waitForFunction(
      () => window.__ninadaReady === true && window.__ninadaLoadedLang === "ta" && !window.__ninadaControlsLocked,
      null,
      { timeout: 90000 }
    );
    assert.equal(
      onnxBuckets.revisitCached.length,
      0,
      `Expected 0 ONNX network requests when switching back to cached TA, got ${onnxBuckets.revisitCached.length}`
    );
    await page.locator(".preset-chip", { hasText: "Streaming Passage" }).click();
    await page.evaluate(() => {
      window.__ninadaLastResult = null;
    });
    await page.locator("#synthBtn").click();
    await page.waitForFunction(
      () =>
        window.__ninadaLastResult &&
        window.__ninadaLastResult.lang === "ta" &&
        window.__ninadaLastResult.tier === "neural_onnx",
      null,
      { timeout: 45000 }
    );
    const taStream = await page.evaluate(async () => {
      const statsAfterDrain = await window.__ninadaWaitForPlayback();
      return {
        result: window.__ninadaLastResult,
        stats: { ...statsAfterDrain },
      };
    });
    console.log("Tamil Multi-Sentence Real-Time AudioWorklet Stream:", taStream);
    assert.equal(taStream.result.lang, "ta");
    assert.ok(taStream.result.rms > 0.02);
    assert.equal(
      taStream.stats.underrunCount,
      0,
      `Expected ZERO audio buffer underruns during TA stream, got ${taStream.stats.underrunCount}!`
    );

    // 5b. Test cached Malayalam (`ml`) multi-sentence real-time AudioWorklet streaming (0 network ONNX requests, 0 underruns!)
    await page.locator('.lang-btn[data-lang="ml"]').click();
    await page.waitForFunction(
      () => window.__ninadaReady === true && window.__ninadaLoadedLang === "ml" && !window.__ninadaControlsLocked,
      null,
      { timeout: 90000 }
    );
    assert.equal(
      onnxBuckets.revisitCached.length,
      0,
      `Expected 0 ONNX network requests when switching back to cached ML, got ${onnxBuckets.revisitCached.length}`
    );
    await page.locator(".preset-chip", { hasText: "Streaming Passage" }).click();
    await page.evaluate(() => {
      window.__ninadaLastResult = null;
    });
    await page.locator("#synthBtn").click();
    await page.waitForFunction(
      () =>
        window.__ninadaLastResult &&
        window.__ninadaLastResult.lang === "ml" &&
        window.__ninadaLastResult.tier === "neural_onnx",
      null,
      { timeout: 45000 }
    );
    const mlStream = await page.evaluate(async () => {
      const statsAfterDrain = await window.__ninadaWaitForPlayback();
      return {
        result: window.__ninadaLastResult,
        stats: { ...statsAfterDrain },
      };
    });
    console.log("Malayalam Multi-Sentence Real-Time AudioWorklet Stream:", mlStream);
    assert.equal(mlStream.result.lang, "ml");
    assert.ok(mlStream.result.rms > 0.02);
    assert.equal(
      mlStream.stats.underrunCount,
      0,
      `Expected ZERO audio buffer underruns during ML stream, got ${mlStream.stats.underrunCount}!`
    );

    // 6. Check mobile viewport responsiveness
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('.lang-btn[data-lang="kn"]').click();
    await page.waitForFunction(
      () => window.__ninadaReady === true && window.__ninadaLoadedLang === "kn" && !window.__ninadaControlsLocked,
      null,
      { timeout: 90000 }
    );
    await page.waitForTimeout(220);
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, "ninada_studio_mobile.png"),
      fullPage: false,
    });

    // 7. Single-thread fallback check (?coi=0 in isolated context without SW) when running local verification
    if (USE_LOCAL_SERVER) {
      const singleThreadCtx = await browser.newContext({
        viewport: { width: 1280, height: 800 },
        serviceWorkers: "block",
      });
      const stPage = await singleThreadCtx.newPage();
      stPage.on("pageerror", (err) => errors.push(`ST_PAGEERROR: ${err.message}`));
      await stPage.goto(`${TARGET_URL}?coi=0`, { waitUntil: "domcontentloaded" });
      await stPage.waitForFunction(
        () =>
          window.__ninadaReady === true &&
          window.__ninadaLoadedLang === "sa" &&
          window.__ninadaLastResult &&
          window.__ninadaLastResult.tier === "neural_onnx",
        null,
        { timeout: 90000 }
      );
      await stPage.evaluate(() => {
        window.__ninadaLastResult = null;
      });
      await stPage.locator("#synthBtn").click();
      await stPage.waitForFunction(
        () => window.__ninadaLastResult && window.__ninadaLastResult.tier === "neural_onnx",
        null,
        { timeout: 60000 }
      );
      const stStream = await stPage.evaluate(async () => {
        const stats = await window.__ninadaWaitForPlayback();
        return {
          coi: window.crossOriginIsolated,
          result: window.__ninadaLastResult,
          stats: { ...stats },
        };
      });
      console.log("Single-Thread Non-COI Fallback Stream Check:", stStream);
      assert.equal(stStream.coi, false);
      assert.equal(stStream.result.threads, 1);
      assert.equal(
        stStream.stats.underrunCount,
        0,
        `Expected ZERO underruns even in 1-thread fallback mode, got ${stStream.stats.underrunCount}!`
      );
      await singleThreadCtx.close();
    }

    assert.equal(errors.length, 0, `Browser errors detected: ${errors.join(" | ")}`);
    await browser.close();
    console.log(`ALL NINADA v4.0 24KHZ NATIVE STUDIO SPECIALIST STREAMING & CACHE BROWSER CHECKS PASSED ON ${TARGET_URL} WITH ZERO UNDERRUNS & ZERO ERRORS!`);
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
