// @portal-browser hydrated operator-console UX contracts.
//
// The shared mock FastAPI fixture intentionally serves inert JavaScript. This
// spec opts into the current production bundles so browser assertions exercise
// the real navigation, keyboard, cancellation, validation, and overlay logic.

import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { expectNoWcagViolations } from "./helpers/accessibility.mjs";

const SPEC_DIR = path.dirname(fileURLToPath(import.meta.url));
const PORTAL_ASSET_DIR = path.resolve(SPEC_DIR, "../../../../public/portal-assets");
const PORTAL_SCRIPT_NAME = /^portal(?:-[a-z0-9-]+)?\.js$/;

const EMPTY_PREVIEW = Object.freeze({
  field_errors: [],
  field_warnings: [],
  inactive_fields: [],
});

function jsonHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": "no-store",
    "Content-Type": "application/json; charset=utf-8",
  };
}

async function fulfillJson(route, payload, status = 200) {
  await route.fulfill({
    status,
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
}

async function installHydratedPortalRoutes(page, options = {}) {
  const jobs = Array.isArray(options.jobs) ? options.jobs : [];
  const preview = {
    ...EMPTY_PREVIEW,
    ...(options.preview && typeof options.preview === "object" ? options.preview : {}),
  };
  const runtime = {
    actor: options.actor || { username: "smoke-admin", role: "admin" },
    cancelRequests: 0,
    eventStreamRequests: 0,
    jobSubmissions: 0,
  };

  await page.addInitScript(() => {
    const initializationKey = "__tp_ux_storage_initialized";
    if (sessionStorage.getItem(initializationKey) === "true") return;
    window.sessionStorage.clear();
    for (let index = window.localStorage.length - 1; index >= 0; index -= 1) {
      const key = window.localStorage.key(index) || "";
      if (key === "tp_orchestrator_profiles_final" || key.startsWith("tp_orchestrator_profiles_final:")) {
        window.localStorage.removeItem(key);
      }
    }
    window.localStorage.removeItem("tp_portal_transient_draft");
    sessionStorage.setItem(initializationKey, "true");
  });

  if (options.keepEventStreamOpen) {
    await page.addInitScript(() => {
      class HydratedUxEventSource {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSED = 2;

        constructor(url) {
          this.url = String(url || "");
          this.readyState = HydratedUxEventSource.OPEN;
          this.listeners = new Map();
          queueMicrotask(() => this.onopen?.());
        }

        addEventListener(name, callback) {
          this.listeners.set(String(name), callback);
        }

        close() {
          this.readyState = HydratedUxEventSource.CLOSED;
        }
      }
      window.EventSource = HydratedUxEventSource;
    });
  }

  await page.route("**/__mock-artifact-preview.png", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "image/png",
      body: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
        "base64",
      ),
    });
  });

  await page.route("**/__mock-portal/*.js", async (route) => {
    const scriptName = path.posix.basename(new URL(route.request().url()).pathname);
    if (!PORTAL_SCRIPT_NAME.test(scriptName)) {
      await route.abort("failed");
      return;
    }
    if (scriptName === "portal-profile.js" && options.profileAssetDelayMs) {
      await new Promise((resolve) => setTimeout(resolve, options.profileAssetDelayMs));
    }
    await route.fulfill({
      status: 200,
      contentType: "text/javascript; charset=utf-8",
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-store",
      },
      body: await readFile(path.join(PORTAL_ASSET_DIR, scriptName)),
    });
  });

  await page.route("**/portal/bootstrap", async (route) => {
    await fulfillJson(route, {
      authMode: "managed",
      csrfToken: "hydrated-ux-csrf",
      actor: runtime.actor,
      features: {
        apiKeyInput: false,
        directDebug: false,
        artifactViewerModal: true,
        reviewSurfaceDeferred: true,
        stagedUploads: false,
        rumTelemetry: false,
        fastVlmCaptioning: false,
      },
    });
  });

  await page.route("**/healthz", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/plain; charset=utf-8",
      headers: { "Cache-Control": "no-store" },
      body: "ok",
    });
  });

  await page.route("**/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const pathname = url.pathname;
    const method = request.method();

    if (/^\/v1\/jobs\/[^/]+\/events$/.test(pathname)) {
      runtime.eventStreamRequests += 1;
      if (options.completeStreamOnCancel) {
        const deadline = Date.now() + 3000;
        while (runtime.cancelRequests === 0 && Date.now() < deadline) {
          await new Promise((resolve) => setTimeout(resolve, 20));
        }
      }
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream; charset=utf-8",
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Cache-Control": "no-store",
        },
        body: options.completeStreamOnCancel && runtime.cancelRequests > 0
          ? 'event: done\ndata: {"state":"canceled","exit_code":0}\n\n'
          : "retry: 60000\n: hydrated browser test stream\n\n",
      });
      return;
    }

    if (method === "POST" && /^\/v1\/jobs\/[^/]+\/cancel$/.test(pathname)) {
      runtime.cancelRequests += 1;
      if (options.cancelFailure) {
        await new Promise((resolve) => setTimeout(resolve, options.cancelDelayMs || 300));
        try {
          await fulfillJson(
            route,
            {
              error: {
                code: "cancel_not_confirmed",
                message: "Worker rejected cancellation while the job remained active.",
              },
            },
            503,
          );
        } catch {
          // A client-side timeout intentionally aborts this route in the
          // bounded-cancellation contract below.
        }
        return;
      }
      const canceledJobId = decodeURIComponent(pathname.split("/").at(-2) || "");
      const canceledJob = jobs.find((item) => String(item.id) === canceledJobId);
      if (canceledJob) {
        canceledJob.state = "canceled";
        canceledJob.finished_at = "2026-08-09T20:00:06Z";
      }
      await fulfillJson(route, { success: true, data: { state: "canceled" } });
      return;
    }

    if (method === "GET" && /^\/v1\/jobs\/[^/]+\/artifacts\//.test(pathname)) {
      await route.fulfill({
        status: 200,
        contentType: "image/png",
        headers: { "Cache-Control": "no-store" },
        body: Buffer.from(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
          "base64",
        ),
      });
      return;
    }

    const jobDetailMatch = pathname.match(/^\/v1\/jobs\/([^/]+)$/);
    if (method === "GET" && jobDetailMatch) {
      if (options.jobDetailDelayMs) {
        await new Promise((resolve) => setTimeout(resolve, options.jobDetailDelayMs));
      }
      const jobId = decodeURIComponent(jobDetailMatch[1]);
      const job = jobs.find((item) => String(item.id) === jobId);
      try {
        await fulfillJson(
          route,
          job
            ? { success: true, data: job }
            : { error: { code: "not_found", message: "Job not found." } },
          job ? 200 : 404,
        );
      } catch {
        // The bounded detail refresh may abort before a delayed fixture reply.
      }
      return;
    }

    if (pathname === "/v1/jobs" && method === "GET") {
      await fulfillJson(route, { success: true, data: { jobs } });
      return;
    }

    if (pathname === "/v1/jobs" && method === "POST") {
      runtime.jobSubmissions += 1;
      await fulfillJson(route, {
        success: true,
        data: { id: "job-dispatched", events_url: "/v1/jobs/job-dispatched/events" },
      });
      return;
    }

    if (pathname === "/v1/config-metadata") {
      await fulfillJson(route, {
        success: true,
        data: {
          pipeline: "lux-depth-v3",
          fields: {},
          estimate_bands: {},
          debug_bundle_policy: {},
          advanced_sections: [],
          backend_catalog: {},
          model_catalog: {},
        },
      });
      return;
    }

    if (pathname === "/v1/config-preview") {
      const requestPayload = request.postDataJSON();
      const args = requestPayload?.args || {};
      await fulfillJson(route, {
        success: true,
        data: {
          pipeline: requestPayload?.pipeline || "lux-depth-v3",
          normalized_args: args,
          execution_args: args,
          field_errors: preview.field_errors,
          field_warnings: preview.field_warnings,
          inactive_fields: preview.inactive_fields,
          readiness: { status: "ready", missing_prerequisites: [] },
          estimate_summary: { runtime_band: "low", gpu_pressure: "low", research_risk: "none" },
          argv_preview: "lux-depth-v3 --input-dir ./input_images --output-dir ./output/lux_depth_v3_apex",
        },
      });
      return;
    }

    if (pathname === "/v1/presets") {
      await fulfillJson(route, {
        success: true,
        data: {
          presets: [
            {
              name: "premium",
              label: "Premium",
              stability: "stable",
              recommended_args: {},
              advanced_sections: [],
            },
          ],
        },
      });
      return;
    }

    if (pathname === "/v1/readiness") {
      await fulfillJson(route, {
        success: true,
        data: {
          server: { status: "ready" },
          pipelines: {
            "lux-depth-v3": { status: "ready", missing_prerequisites: [] },
          },
        },
      });
      return;
    }

    if (pathname === "/v1/portal/events" || pathname === "/v1/portal/rum") {
      await route.fulfill({ status: 204, body: "" });
      return;
    }

    await fulfillJson(route, { success: true, data: {} });
  });

  return runtime;
}

async function gotoHydratedPortal(page, target = "/portal") {
  const response = await page.goto(target);
  expect(response?.status()).toBe(200);
  await expect(page.locator("body")).toHaveAttribute("data-bootstrap-status", "ready");
  await expect(page.locator("#healthText")).toHaveText("Backend Online");
  await expect(page.locator('[data-ui="portal-topbar"]')).toBeVisible();
}

test.describe("hydrated portal UX runtime", { tag: "@portal-browser" }, () => {
  test("navigation keeps page semantics and opens empty Review by mouse and keyboard", async ({ page }) => {
    await installHydratedPortalRoutes(page);
    await gotoHydratedPortal(page);

    const overviewLink = page.locator('[data-view-link="overview"]');
    const reviewLink = page.locator('[data-view-link="review"]');
    const viewTitle = page.locator("#consoleViewTitle");
    const overviewTitle = page.locator("#overviewViewTitle");
    const reviewTitle = page.locator("#reviewViewTitle");
    const statusRegion = page.locator("#portalA11yStatus");

    await expect(overviewLink).toHaveAttribute("aria-current", "page");
    await expect(page.locator("[data-view-link][aria-current='page']")).toHaveCount(1);
    await expect(page.locator("[data-view-link][aria-selected]")).toHaveCount(0);

    await reviewLink.click();
    await expect(page).toHaveURL(/\/portal\?view=review$/);
    await expect(page.locator("body")).toHaveAttribute("data-console-view", "review");
    await expect(reviewLink).toHaveAttribute("aria-current", "page");
    await expect(overviewLink).not.toHaveAttribute("aria-current", /.+/);
    await expect(viewTitle).toHaveText("Review");
    await expect(reviewTitle).toBeFocused();
    await expect(statusRegion).toContainText("Review view opened.");
    await expect(page.locator("#emptyArtifactState")).toBeVisible();

    await overviewLink.click();
    await expect(viewTitle).toHaveText("Overview");
    await expect(overviewTitle).toBeFocused();
    await page.keyboard.press("4");
    await expect(page).toHaveURL(/\/portal\?view=review$/);
    await expect(reviewLink).toHaveAttribute("aria-current", "page");
    await expect(viewTitle).toHaveText("Review");
    await expect(reviewTitle).toBeFocused();
    await expect(statusRegion).toContainText("Review view opened.");
    await expect(page.locator("[data-view-link][aria-selected]")).toHaveCount(0);
  });

  test("mobile workspace navigation keeps the focused surface heading in view", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await installHydratedPortalRoutes(page);
    await gotoHydratedPortal(page);

    for (const [view, headingId] of [["operate", "#operateViewTitle"], ["review", "#reviewViewTitle"]]) {
      await page.locator(`[data-view-link="${view}"]`).click();
      const heading = page.locator(headingId);
      await expect(heading).toBeFocused();
      const visibility = await heading.evaluate((element) => {
        const rect = element.getBoundingClientRect();
        const topbarBottom = document.querySelector('[data-ui="portal-topbar"]')?.getBoundingClientRect().bottom || 0;
        return {
          bottom: rect.bottom,
          top: rect.top,
          topbarBottom,
          viewportHeight: window.innerHeight,
        };
      });
      expect(visibility.top).toBeGreaterThanOrEqual(visibility.topbarBottom - 1);
      expect(visibility.bottom).toBeLessThanOrEqual(visibility.viewportHeight + 1);
    }
  });

  test("deferred Operate assets stay idle until an active dispatch or Operate workspace", async ({ page }) => {
    const requestedScripts = [];
    page.on("request", (request) => {
      const pathname = new URL(request.url()).pathname;
      if (pathname.startsWith("/__mock-portal/") && pathname.endsWith(".js")) {
        requestedScripts.push(path.posix.basename(pathname));
      }
    });
    await installHydratedPortalRoutes(page);
    await gotoHydratedPortal(page);

    await page.waitForTimeout(200);
    expect(requestedScripts).not.toContain("portal-operate.js");

    await page.locator('[data-view-link="build"]').click();
    await expect(page).toHaveURL(/\/portal\?view=build$/);
    await page.waitForTimeout(200);
    expect(requestedScripts).not.toContain("portal-operate.js");

    await page.locator("#buildStepTab4").click();
    await page.locator("#runJobBtn").click();
    await expect(page.locator("#runJobBtn")).toHaveText("Open Live Job");
    await expect.poll(() => requestedScripts.includes("portal-operate.js")).toBe(true);
    await expect(page.locator('[data-ui="queue-row"][data-job-id="job-dispatched"]')).toHaveCount(1);

    await page.locator('[data-view-link="operate"]').click();
    await expect(page).toHaveURL(/view=operate/);
  });

  test("unchanged health polling does not mutate visible status or empty Review announcements", async ({ page }) => {
    await installHydratedPortalRoutes(page);
    await gotoHydratedPortal(page, "/portal?view=review");
    await expect(page.locator("#artifactMeta")).toHaveText("No job selected");
    await expect(page.locator(".topbar-status")).toHaveAttribute("role", "group");
    await expect(page.locator(".topbar-status")).not.toHaveAttribute("aria-live", /.+/);

    await page.evaluate(() => {
      window.__uxStatusMutationCounts = { health: 0, empty: 0 };
      const observe = (element, key) => {
        const observer = new MutationObserver((records) => {
          window.__uxStatusMutationCounts[key] += records.length;
        });
        observer.observe(element, { childList: true, characterData: true, subtree: true });
      };
      observe(document.querySelector(".topbar-status"), "health");
      observe(document.getElementById("emptyArtifactState"), "empty");
    });

    const healthResponse = page.waitForResponse((response) => new URL(response.url()).pathname === "/healthz");
    await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));
    await healthResponse;
    await page.waitForTimeout(100);
    await expect.poll(() => page.evaluate(() => window.__uxStatusMutationCounts)).toEqual({ health: 0, empty: 0 });
  });

  test("delayed profile loading cannot open after leaving Build", async ({ page }) => {
    await installHydratedPortalRoutes(page, { profileAssetDelayMs: 700 });
    await gotoHydratedPortal(page, "/portal?view=build");

    await page.locator("#saveProfileBtn").click();
    await page.locator('[data-view-link="overview"]').click();
    await expect(page).toHaveURL(/\/portal\?view=overview$/);
    await expect(page.locator("#overviewViewTitle")).toBeFocused();
    await page.waitForTimeout(850);
    await expect(page.locator("#profileManagerDialog")).toBeHidden();
    await expect(page.locator("#overviewViewTitle")).toBeFocused();
  });

  test("another modal cancels a pending profile-open intent", async ({ page }) => {
    await installHydratedPortalRoutes(page, { profileAssetDelayMs: 700 });
    await gotoHydratedPortal(page, "/portal?view=build");

    await page.locator("#saveProfileBtn").click();
    await page.locator("#shortcutsBtn").click();
    await expect(page.locator("#shortcutsModal")).toBeVisible();
    await page.waitForTimeout(850);
    await expect(page.locator("#profileManagerDialog")).toBeHidden();
    await expect(page.locator("#closeShortcutsBtn")).toBeFocused();

    await page.locator("#closeShortcutsBtn").click();
    await expect(page.locator("#shortcutsModal")).toBeHidden();
    await expect(page.locator("#shortcutsBtn")).toBeFocused();
    await expect(page.locator("#profileManagerDialog")).toBeHidden();
  });

  test("dispatch and copy shortcuts are scoped to ready Build step 4 outside typing targets", async ({ page }) => {
    await installHydratedPortalRoutes(page);
    await gotoHydratedPortal(page);

    await page.evaluate(() => {
      window.__uxRuntimeShortcutCalls = { dispatch: 0, copy: 0 };
      document.getElementById("runJobBtn").click = () => {
        window.__uxRuntimeShortcutCalls.dispatch += 1;
      };
      document.getElementById("copyCliBtn").click = () => {
        window.__uxRuntimeShortcutCalls.copy += 1;
      };
    });

    await page.keyboard.press("Control+Enter");
    await page.keyboard.press("Control+Alt+C");
    await expect.poll(() => page.evaluate(() => window.__uxRuntimeShortcutCalls)).toEqual({ dispatch: 0, copy: 0 });

    await page.locator('[data-view-link="build"]').click();
    await expect(page).toHaveURL(/\/portal\?view=build$/);
    await expect(page.locator("#consoleViewTitle")).toHaveText("Build");
    await expect(page.locator("#buildStepTitle")).toBeFocused();
    await page.keyboard.press("Control+Enter");
    await expect.poll(() => page.evaluate(() => window.__uxRuntimeShortcutCalls)).toEqual({ dispatch: 0, copy: 0 });
    await page.keyboard.press("Control+Alt+C");
    await expect.poll(() => page.evaluate(() => window.__uxRuntimeShortcutCalls)).toEqual({ dispatch: 0, copy: 0 });

    await page.locator("#buildStepTab4").click();
    const dispatchButton = page.locator("#runJobBtn");
    await expect(dispatchButton).toBeEnabled();

    await page.evaluate(() => {
      document.getElementById("runJobBtn").disabled = true;
    });
    await page.keyboard.press("Control+Enter");
    await expect.poll(() => page.evaluate(() => window.__uxRuntimeShortcutCalls.dispatch)).toBe(0);

    await page.evaluate(() => {
      document.getElementById("runJobBtn").disabled = false;
      document.getElementById("buildStepTab4").focus();
    });
    await page.keyboard.press("Control+Enter");
    await page.keyboard.press("Control+Alt+C");
    await expect.poll(() => page.evaluate(() => window.__uxRuntimeShortcutCalls)).toEqual({ dispatch: 1, copy: 1 });

    await page.evaluate(() => {
      const typingTarget = document.createElement("input");
      typingTarget.type = "text";
      typingTarget.dataset.ui = "hydrated-shortcut-typing-target";
      typingTarget.value = "keep keyboard input isolated";
      document.getElementById("cli-shell").appendChild(typingTarget);
      typingTarget.focus();
    });
    await expect(page.locator('[data-ui="hydrated-shortcut-typing-target"]')).toBeFocused();
    await page.keyboard.press("Control+Enter");
    await page.keyboard.press("Control+Alt+C");
    await expect.poll(() => page.evaluate(() => window.__uxRuntimeShortcutCalls)).toEqual({ dispatch: 1, copy: 1 });
  });

  test("failed cancellation remains pending until response then restores Retry Cancel and focus", async ({ page }) => {
    const runningJob = {
      id: "job-running",
      pipeline: "lux-depth-v3",
      state: "running",
      progress: 42,
      logs_tail: ["[INFO] Processing source images."],
      artifacts: [],
      created_at: "2026-08-09T20:00:00Z",
      started_at: "2026-08-09T20:00:01Z",
      updated_at: "2026-08-09T20:00:05Z",
      events_url: "/v1/jobs/job-running/events",
    };
    const runtime = await installHydratedPortalRoutes(page, {
      jobs: [runningJob],
      cancelFailure: true,
      cancelDelayMs: 400,
    });
    await gotoHydratedPortal(page, "/portal?view=operate");

    const cancelButton = page.locator('[data-action="cancel-job"][data-job-id="job-running"]');
    const queueRow = page.locator('[data-ui="queue-row"][data-job-id="job-running"]');
    await expect(cancelButton).toBeVisible();
    await expect(cancelButton).toHaveText("Cancel");
    await expect(page.locator("#portalA11yStatus")).toHaveCount(1);
    await page.evaluate(() => {
      const statusRegion = document.getElementById("portalA11yStatus");
      const failureMessage = "Cancellation was not confirmed for job job-running. The job remains active. Retry Cancel.";
      window.__cancelFailureAnnouncementCount = 0;
      new MutationObserver(() => {
        if (statusRegion.textContent.trim() === failureMessage) {
          window.__cancelFailureAnnouncementCount += 1;
        }
      }).observe(statusRegion, { childList: true, characterData: true, subtree: true });
    });

    await cancelButton.click();
    await expect(cancelButton).toHaveAttribute("aria-disabled", "true");
    await expect(cancelButton).toHaveAttribute("aria-busy", "true");
    await expect(cancelButton).toHaveText("Canceling…");
    await expect(cancelButton).toBeFocused();
    await expect(page.locator("#portalA11yStatus")).toContainText("Canceling job job-running.");

    await expect(cancelButton).toBeEnabled();
    await expect(cancelButton).toHaveText("Retry Cancel");
    await expect(cancelButton).toBeFocused();
    await expect(queueRow).toContainText("running");
    await expect(queueRow).toContainText("Cancellation was not confirmed");
    await expect(page.locator("#portalA11yStatus")).toContainText(
      "Cancellation was not confirmed for job job-running. The job remains active. Retry Cancel.",
    );
    await expect.poll(() => page.evaluate(() => window.__cancelFailureAnnouncementCount)).toBe(1);
    expect(runtime.cancelRequests).toBe(1);
  });

  test("Operate exposes semantic queue controls, inspector keyboarding, and successful cancel focus", async ({ page }) => {
    const runningJob = {
      id: "job-cancel-success",
      pipeline: "lux-depth-v3",
      state: "running",
      progress: 61,
      logs_tail: ["[INFO] Rendering governed output."],
      artifacts: { items: [] },
      created_at: "2026-08-09T20:00:00Z",
      started_at: "2026-08-09T20:00:01Z",
      updated_at: "2026-08-09T20:00:05Z",
      events_url: "",
    };
    const runtime = await installHydratedPortalRoutes(page, {
      jobs: [runningJob],
      keepEventStreamOpen: true,
    });
    await gotoHydratedPortal(page, "/portal?view=operate");
    expect(await page.evaluate(() => window.EventSource.name)).toBe("HydratedUxEventSource");

    const queue = page.locator("#jobList");
    const row = page.locator('[data-ui="queue-row"][data-job-id="job-cancel-success"]');
    const inspectButton = row.locator('[data-action="inspect-job"]');
    const cancelButton = row.locator('[data-action="cancel-job"]');
    await expect(queue).toHaveAttribute("role", "list");
    await expect(row).toHaveCount(1);
    await expect.poll(() => row.evaluate((element) => element.tagName)).toBe("LI");
    await expect(inspectButton).toHaveAttribute("aria-pressed", "true");
    await expect(cancelButton).toHaveText("Cancel");
    await expect(page.locator("#queueDeltaStatus")).toContainText("Queue has 1 job; 1 active.");

    const timelineTab = page.locator("#inspectorTimelineTab");
    const logsTab = page.locator("#inspectorLogsTab");
    const overviewTab = page.locator("#inspectorOverviewTab");
    await expect(timelineTab).toHaveAttribute("aria-selected", "true");
    await timelineTab.focus();
    await page.keyboard.press("End");
    await expect(logsTab).toBeFocused();
    await expect(logsTab).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("Home");
    await expect(overviewTab).toBeFocused();
    await expect(overviewTab).toHaveAttribute("aria-selected", "true");

    await cancelButton.click();
    await expect(row).toContainText("canceled");
    await expect(cancelButton).toBeHidden();
    await expect(inspectButton).toBeFocused();
    await expect(page.locator("#portalA11yStatus")).toContainText("Job job-cancel-success canceled.");
    await expect(page.locator("#queueDeltaStatus")).toContainText("Queue has 1 job; 1 active.");
    await expect(page.locator("#queueDeltaStatus")).not.toContainText("Job job-cancel-success is now canceled.");
    expect(runtime.cancelRequests).toBe(1);
    expect(runtime.eventStreamRequests).toBe(0);
    await expectNoWcagViolations(page, "Portal Operate surface");
  });

  test("hung cancellation is bounded and restores Retry Cancel before detail refresh settles", async ({ page }) => {
    const runningJob = {
      id: "job-cancel-timeout",
      pipeline: "lux-depth-v3",
      state: "running",
      progress: 27,
      logs_tail: [],
      artifacts: { items: [] },
      created_at: "2026-08-09T20:00:00Z",
      updated_at: "2026-08-09T20:00:05Z",
      events_url: "/v1/jobs/job-cancel-timeout/events",
    };
    const runtime = await installHydratedPortalRoutes(page, {
      jobs: [runningJob],
      cancelFailure: true,
      cancelDelayMs: 7000,
      jobDetailDelayMs: 7000,
    });
    await gotoHydratedPortal(page, "/portal?view=operate");

    const cancelButton = page.locator('[data-action="cancel-job"][data-job-id="job-cancel-timeout"]');
    await cancelButton.click();
    await expect(cancelButton).toHaveText("Canceling…");
    await expect(cancelButton).toHaveAttribute("aria-busy", "true");
    await expect(cancelButton).toHaveText("Retry Cancel", { timeout: 6500 });
    await expect(cancelButton).not.toHaveAttribute("aria-busy", /.+/);
    await expect(cancelButton).toBeFocused();
    await expect(page.locator("#portalA11yStatus")).toContainText("Retry Cancel");
    expect(runtime.cancelRequests).toBe(1);
  });

  test("dispatch exposes an Open Live Job handoff into Operate", async ({ page }) => {
    const runtime = await installHydratedPortalRoutes(page);
    await gotoHydratedPortal(page, "/portal?view=build");
    await page.locator("#buildStepTab4").click();
    const dispatchButton = page.locator("#runJobBtn");
    await expect(dispatchButton).toBeEnabled();
    await dispatchButton.click();
    await expect(dispatchButton).toHaveText("Open Live Job");
    await expect(dispatchButton).toBeFocused();
    expect(runtime.jobSubmissions).toBe(1);

    await dispatchButton.click();
    await expect(page).toHaveURL(/view=operate.*job=job-dispatched|job=job-dispatched.*view=operate/);
    await expect(page.locator("#operateViewTitle")).toBeFocused();
    await expect(page.locator('[data-ui="queue-row"][data-job-id="job-dispatched"]')).toBeVisible();
  });

  test("Review exposes dynamic artifact text and remains axe-clean", async ({ page }) => {
    const completedJob = {
      id: "job-reviewable",
      pipeline: "lux-depth-v3",
      state: "succeeded",
      progress: 100,
      logs_tail: ["[INFO] Output indexed."],
      artifacts: {
        items: [
          {
            path: "outputs/depth-primary.png",
            relative_path: "outputs/depth-primary.png",
            artifact_type: "image",
            media_kind: "image",
            previewable: true,
            browser_previewable: true,
            preview_url: "/v1/jobs/job-reviewable/artifacts/outputs%2Fdepth-primary.png",
            content_type: "image/png",
            size_bytes: 68,
            sha256: "a".repeat(64),
            display_hint: { role: "primary", label: "Primary depth", priority: 1 },
          },
        ],
      },
      created_at: "2026-08-09T20:00:00Z",
      finished_at: "2026-08-09T20:00:05Z",
      updated_at: "2026-08-09T20:00:05Z",
    };
    await installHydratedPortalRoutes(page, { jobs: [completedJob] });
    await gotoHydratedPortal(page, "/portal?view=review");

    const preview = page.locator("#artifactPreviewSoloImage");
    await expect(preview).toBeVisible();
    await expect(preview).toHaveAttribute("alt", "Primary depth preview: outputs/depth-primary.png");
    await expect(page.locator("#artifactThumbnailRail")).toHaveAttribute("role", "group");
    await expect(page.locator("#reviewStatusBanner")).toContainText(/Review|artifact/i);
    await expectNoWcagViolations(page, "Portal Review surface");
  });

  test("Build and profile management execute with valid semantics and protected names", async ({ page }) => {
    const runtime = await installHydratedPortalRoutes(page);
    await gotoHydratedPortal(page, "/portal?view=build");
    await expect(page.locator("#buildStepTitle")).toBeVisible();
    await expectNoWcagViolations(page, "Portal Build surface");

    const manageButton = page.locator("#saveProfileBtn");
    await manageButton.click();
    const dialog = page.locator("#profileManagerDialog");
    const nameInput = page.locator("#profileManagerName");
    const saveButton = page.locator('[data-profile-action="save"]');
    const renameButton = page.locator('[data-profile-action="rename"]');
    const deleteButton = page.locator('[data-profile-action="delete"]');
    const readProfileStore = () => page.evaluate(() => {
      const keys = [];
      for (let index = 0; index < localStorage.length; index += 1) {
        const key = localStorage.key(index) || "";
        if (key.startsWith("tp_orchestrator_profiles_final:")) keys.push(key);
      }
      const key = keys.sort()[0] || "";
      return { key, profiles: key ? JSON.parse(localStorage.getItem(key) || "{}") : {} };
    });
    await expect(dialog).toBeVisible();

    await page.locator('[data-profile-action="close"]').focus();
    await page.keyboard.type("?");
    await expect(page.locator("#shortcutsModal")).toBeHidden();
    await expect(dialog).toBeVisible();

    await nameInput.fill("__proto__");
    await expect(nameInput).toHaveAttribute("aria-invalid", "true");
    await expect(page.locator("#profileManagerMessage")).toContainText("reserved object key");
    await expect(saveButton).toBeDisabled();

    await nameInput.fill("toString");
    await saveButton.click();
    await expect(page.locator("#profileSelect")).toHaveValue("toString");
    await expect(dialog.locator(":focus")).toHaveCount(1);
    await deleteButton.click();
    await deleteButton.click();
    await expect(page.locator("#profileSelect")).toHaveValue("");
    await expect(dialog.locator(":focus")).toHaveCount(1);

    await nameInput.fill("Morning Run");
    await saveButton.click();
    await expect(page.locator("#profileManagerMessage")).toContainText("Morning Run” saved");
    await expect(page.locator("#profileSelect")).toHaveValue("Morning Run");
    await expect
      .poll(async () => Object.keys((await readProfileStore()).profiles))
      .toEqual(["Morning Run"]);
    await expect.poll(async () => (await readProfileStore()).key).toContain("managed%3Asmoke-admin%3Aadmin");

    await page.locator('[data-profile-action="close"]').click();
    await page.locator("#qualityTier").selectOption("standard");
    await expect(manageButton).toHaveText("Save Changes");
    await expect(manageButton).toHaveAttribute("data-profile-state", "dirty");
    await manageButton.click();
    await expect(nameInput).toHaveValue("Morning Run");
    await expect(saveButton).toHaveText("Overwrite Profile");
    await saveButton.click();
    await expect(saveButton).toHaveText("Confirm Overwrite");
    await saveButton.click();
    await expect(page.locator("#profileManagerMessage")).toContainText("Morning Run” saved");
    await expect(manageButton).toHaveText("Saved");

    await nameInput.fill("Alternate Run");
    await expect(saveButton).toHaveText("Save New Profile");
    await saveButton.click();
    await expect(page.locator("#profileSelect")).toHaveValue("Alternate Run");
    await page.locator('[data-profile-action="close"]').click();

    await page.locator("#qualityTier").selectOption("premium");
    await expect(manageButton).toHaveAttribute("data-profile-state", "dirty");
    await manageButton.click();
    await nameInput.fill("Morning Run");
    await saveButton.click();
    await expect(saveButton).toHaveText("Confirm Overwrite");
    await deleteButton.click();
    await expect(deleteButton).toHaveText("Confirm Delete");
    await expect(saveButton).toHaveText("Overwrite Profile");
    await page.locator('[data-profile-action="close"]').click();
    await page.locator("#profileSelect").selectOption("");
    await expect(dialog).toBeVisible();
    await expect(page.locator("#profileSelect")).toHaveValue("Alternate Run");
    await expect(page.locator("#profileManagerMessage")).toContainText("before clearing the selection");
    await page.locator('[data-profile-action="close"]').click();
    await expect(manageButton).toHaveAttribute("data-profile-state", "dirty");
    await page.locator("#profileSelect").selectOption("Morning Run");
    const pendingLoadButton = page.locator('[data-profile-action="load-pending"]');
    await expect(dialog).toBeVisible();
    await expect(page.locator("#profileSelect")).toHaveValue("Alternate Run");
    await expect(pendingLoadButton).toContainText("Morning Run");
    await page.locator('[data-profile-action="close"]').click();
    await expect(page.locator("#profileSelect")).toHaveValue("Alternate Run");
    await expect(page.locator("#qualityTier")).toHaveValue("premium");

    await page.locator("#profileSelect").selectOption("Morning Run");
    await pendingLoadButton.click();
    await expect(dialog).toBeHidden();
    await expect(page.locator("#profileSelect")).toHaveValue("Morning Run");
    await expect(page.locator("#qualityTier")).toHaveValue("standard");
    await manageButton.click();
    await nameInput.fill("Afternoon Run");
    await renameButton.click();
    await expect(page.locator("#profileSelect")).toHaveValue("Afternoon Run");
    await expect(page.locator("#profileManagerMessage")).toContainText("renamed to “Afternoon Run”");

    await deleteButton.click();
    await expect(deleteButton).toHaveText("Confirm Delete");
    await deleteButton.click();
    await expect(page.locator("#profileSelect")).toHaveValue("");
    await expect(page.locator("#profileManagerMessage")).toContainText("deleted");
    await expect
      .poll(async () => Object.keys((await readProfileStore()).profiles))
      .toEqual(["Alternate Run"]);
    await expect(dialog.locator(":focus")).toHaveCount(1);
    await expectNoWcagViolations(page, "Portal profile manager");

    runtime.actor = { username: "other-admin", role: "admin" };
    await page.goto("/portal?view=build");
    await expect(page.locator("body")).toHaveAttribute("data-bootstrap-status", "ready");
    await expect(page.locator("#profileSelect option")).toHaveCount(1);
    await expect(page.locator("#profileSelect")).toHaveValue("");
  });

  test("restored unprofiled drafts survive delayed profile loading and require discard confirmation", async ({ page }) => {
    await installHydratedPortalRoutes(page, { profileAssetDelayMs: 700 });
    await gotoHydratedPortal(page, "/portal?view=build");

    const manageButton = page.locator("#saveProfileBtn");
    const profileSelect = page.locator("#profileSelect");
    await manageButton.click();
    await page.locator("#profileManagerName").fill("Baseline Run");
    await page.locator('[data-profile-action="save"]').click();
    await page.locator('[data-profile-action="close"]').click();

    await page.locator("#qualityTier").selectOption("standard");
    await expect(manageButton).toHaveText("Save Changes");
    await expect
      .poll(() =>
        page.evaluate(() => {
          const raw = sessionStorage.getItem("tp_portal_transient_draft") || "";
          return raw.includes("standard");
        }),
      )
      .toBe(true);

    await page.reload();
    await expect(page.locator("body")).toHaveAttribute("data-bootstrap-status", "ready");
    await expect(page.locator("#qualityTier")).toHaveValue("standard");
    await expect(profileSelect.locator('option[value="Baseline Run"]')).toHaveCount(1);
    await expect(manageButton).toHaveText("Save Profile");
    await expect(manageButton).toHaveAttribute("data-profile-state", "unsaved");

    await profileSelect.selectOption("Baseline Run");
    const dialog = page.locator("#profileManagerDialog");
    await expect(dialog).toBeVisible();
    await expect(profileSelect).toHaveValue("");
    await expect(page.locator("#qualityTier")).toHaveValue("standard");
    await expect(page.locator('[data-profile-action="load-pending"]')).toContainText("Baseline Run");
    await expect(page.locator("#profileManagerMessage")).toContainText("Current draft has unsaved changes");

    await page.locator('[data-profile-action="close"]').click();
    await expect(dialog).toBeHidden();
    await expect(profileSelect).toHaveValue("");
    await expect(page.locator("#qualityTier")).toHaveValue("standard");

    await profileSelect.selectOption("Baseline Run");
    await expect(dialog).toBeVisible();
    await page.evaluate(() => {
      for (let index = localStorage.length - 1; index >= 0; index -= 1) {
        const key = localStorage.key(index) || "";
        if (key.startsWith("tp_orchestrator_profiles_final:")) localStorage.removeItem(key);
      }
    });
    await page.locator('[data-profile-action="load-pending"]').click();
    await expect(dialog).toBeVisible();
    await expect(page.locator("#profileManagerMessage")).toContainText("no longer available");
    await expect(profileSelect.locator('option[value="Baseline Run"]')).toHaveCount(0);
    await expect(profileSelect).toHaveValue("");
    await expect(page.locator("#qualityTier")).toHaveValue("standard");
    await expect(manageButton).toHaveText("Save Profile");
  });

  test("managed actors claim legacy browser profiles explicitly", async ({ page }) => {
    await installHydratedPortalRoutes(page);
    await gotoHydratedPortal(page, "/portal?view=build");

    const manageButton = page.locator("#saveProfileBtn");
    const profileSelect = page.locator("#profileSelect");
    await manageButton.click();
    await page.locator("#profileManagerName").fill("Legacy Run");
    await page.locator('[data-profile-action="save"]').click();
    await page.locator('[data-profile-action="close"]').click();

    await page.evaluate(() => {
      const scopedKey = Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index) || "")
        .find((key) => key.startsWith("tp_orchestrator_profiles_final:"));
      if (!scopedKey) throw new Error("Expected an actor-scoped profile store");
      localStorage.setItem("tp_orchestrator_profiles_final", localStorage.getItem(scopedKey) || "{}");
      localStorage.removeItem(scopedKey);
    });

    await page.reload();
    await expect(page.locator("body")).toHaveAttribute("data-bootstrap-status", "ready");
    await expect(profileSelect.locator("option")).toHaveCount(1);
    await manageButton.click();
    const importButton = page.locator('[data-profile-action="import-legacy"]');
    await expect(importButton).toBeVisible();
    await expect(importButton).toHaveText("Import 1 Legacy Profile");

    await importButton.click();
    await expect(page.locator("#profileManagerMessage")).toContainText("shared by every portal user");
    await expect(importButton).toHaveText("Confirm Claim & Import");
    await expect
      .poll(() => page.evaluate(() => localStorage.getItem("tp_orchestrator_profiles_final") !== null))
      .toBe(true);

    await importButton.click();
    await expect(page.locator("#profileManagerMessage")).toContainText("1 legacy profile imported");
    await expect(importButton).toBeHidden();
    await expect(profileSelect.locator('option[value="Legacy Run"]')).toHaveCount(1);
    await expect
      .poll(() =>
        page.evaluate(() => ({
          legacy: localStorage.getItem("tp_orchestrator_profiles_final"),
          scoped: Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index) || "")
            .find((key) => key.startsWith("tp_orchestrator_profiles_final:")) || "",
        })),
      )
      .toEqual({ legacy: null, scoped: "tp_orchestrator_profiles_final:managed%3Asmoke-admin%3Aadmin" });
  });

  test("drafts remain protected when another tab removes the active profile", async ({ page }) => {
    await installHydratedPortalRoutes(page);
    await gotoHydratedPortal(page, "/portal?view=build");

    const manageButton = page.locator("#saveProfileBtn");
    const profileSelect = page.locator("#profileSelect");
    const qualityTier = page.locator("#qualityTier");
    const initialTier = await qualityTier.inputValue();
    const editedTier = initialTier === "standard" ? "premium" : "standard";
    await manageButton.click();
    const nameInput = page.locator("#profileManagerName");
    const saveButton = page.locator('[data-profile-action="save"]');
    await nameInput.fill("Primary Run");
    await saveButton.click();
    await nameInput.fill("Removed Elsewhere");
    await saveButton.click();
    await page.locator('[data-profile-action="close"]').click();

    await qualityTier.selectOption(editedTier);
    await expect(manageButton).toHaveText("Save Changes");
    await page.evaluate(() => {
      const scopedKey = Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index) || "")
        .find((key) => key.startsWith("tp_orchestrator_profiles_final:"));
      if (!scopedKey) throw new Error("Expected an actor-scoped profile store");
      const profiles = JSON.parse(localStorage.getItem(scopedKey) || "{}");
      delete profiles["Removed Elsewhere"];
      localStorage.setItem(scopedKey, JSON.stringify(profiles));
    });

    await profileSelect.selectOption("Primary Run");
    const dialog = page.locator("#profileManagerDialog");
    const pendingLoadButton = page.locator('[data-profile-action="load-pending"]');
    await expect(dialog).toBeVisible();
    await expect(profileSelect).toHaveValue("Removed Elsewhere");
    await expect(qualityTier).toHaveValue(editedTier);
    await expect(pendingLoadButton).toContainText("Primary Run");

    await page.locator('[data-profile-action="close"]').click();
    await expect(qualityTier).toHaveValue(editedTier);
    await profileSelect.selectOption("Primary Run");
    await pendingLoadButton.click();
    await expect(dialog).toBeHidden();
    await expect(profileSelect).toHaveValue("Primary Run");
    await expect(qualityTier).toHaveValue(initialTier);
  });

  test("preview field errors expose relationships and reveal the first invalid control", async ({ page }) => {
    await installHydratedPortalRoutes(page, {
      preview: {
        field_errors: [
          {
            field: "input_dir",
            code: "input_directory_unavailable",
            message: "Choose an input directory that the managed worker can read.",
          },
        ],
      },
    });
    await gotoHydratedPortal(page, "/portal?view=build");

    const input = page.locator("#inputDir");
    await expect(input).toHaveAttribute("aria-invalid", "true");
    await expect(input).toHaveAttribute("aria-errormessage", "inputDirStatus");
    await expect(input).toHaveAttribute("aria-describedby", /\binputDirStatus\b/);
    await expect(page.locator("#inputDirStatus")).toContainText("managed worker can read");

    await page.locator("#buildStepTab4").click();
    await expect(input).toBeHidden();
    await page.evaluate(() => {
      const dispatchButton = document.getElementById("runJobBtn");
      dispatchButton.disabled = false;
      dispatchButton.click();
    });

    await expect(page.locator("#buildStepTab2")).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator("#buildStepTab2")).toHaveAttribute("aria-current", "step");
    await expect(input).toBeVisible();
    await expect(input).toBeFocused();
  });

  test("modal isolation covers the topbar and main content and restores both focus and state", async ({ page }) => {
    await installHydratedPortalRoutes(page);
    await gotoHydratedPortal(page);

    const trigger = page.locator("#shortcutsBtn");
    const modal = page.locator("#shortcutsModal");
    await trigger.click();
    await expect(modal).toBeVisible();
    await expect(modal).toHaveAttribute("aria-hidden", "false");
    await expect(page.locator("#closeShortcutsBtn")).toBeFocused();
    await expect
      .poll(() =>
        page.evaluate(() => {
          const topbar = document.querySelector('[data-ui="portal-topbar"]');
          const main = document.getElementById("main-content");
          return [topbar, main].map((element) => ({
            inert: element.inert,
            ariaHidden: element.getAttribute("aria-hidden"),
          }));
        }),
      )
      .toEqual([
        { inert: true, ariaHidden: "true" },
        { inert: true, ariaHidden: "true" },
      ]);

    await page.keyboard.press("Escape");
    await expect(modal).toBeHidden();
    await expect(trigger).toBeFocused();
    await expect
      .poll(() =>
        page.evaluate(() => {
          const topbar = document.querySelector('[data-ui="portal-topbar"]');
          const main = document.getElementById("main-content");
          return [topbar, main].map((element) => ({
            inert: element.inert,
            ariaHidden: element.getAttribute("aria-hidden"),
          }));
        }),
      )
      .toEqual([
        { inert: false, ariaHidden: null },
        { inert: false, ariaHidden: null },
      ]);
  });
});
