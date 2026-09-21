import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

import { isManagedAuthMode } from "../portal-src/internal/bootstrap-auth.js";

const source = readFileSync(new URL("../portal-src/portal.template.js", import.meta.url), "utf8");
const helperNames = [
  "_defaultPortalBootstrap", "_isBootstrapReady", "_isManagedAuthMode",
  "_syncApiKeyInputState", "_applyPortalBootstrap", "_normalizeApiToken",
  "_currentApiToken", "_buildAuthHeaders", "_handleDirectDebugApiKeyUpdate",
];
const helpers = helperNames.map((name) => {
  const start = source.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `missing ${name}`);
  const end = source.indexOf("\nfunction ", start + 1);
  assert.notEqual(end, -1, `missing end of ${name}`);
  return source.slice(start, end);
}).join("\n");

function credentialHarness({ deniedStore = "", denyGetter = false } = {}) {
  const stores = {
    localStorage: new Map([["tp_api_key", "legacy-local"], ["unrelated", "preserve"]]),
    sessionStorage: new Map([["tp_api_key", "legacy-session"], ["unrelated", "preserve"]]),
  };
  const removalAttempts = [];
  const state = { bootstrap: { status: "pending" }, auth: {}, rum: { queuedSamples: [] } };
  const context = vm.createContext({
    state,
    els: { apiKeyInput: { value: "" }, logoutBtn: { hidden: true } },
    API_KEY_STORAGE_KEY: "tp_api_key",
    SAFE_HTTP_METHODS: new Set(["GET", "HEAD", "OPTIONS"]),
    portalInternals: { isManagedAuthMode, normalizePortalRumTraceparent: () => "" },
    _setBootstrapStatus: (status) => { state.bootstrap.status = status; },
    _rumTelemetryEnabled: () => false,
    _syncBootstrapUi: () => {},
    _resetProtectedFamilySuppression: () => {},
  });
  for (const [name, store] of Object.entries(stores)) {
    Object.defineProperty(context, name, {
      get() {
        if (name === deniedStore && denyGetter) {
          removalAttempts.push(name);
          throw new Error("Storage access denied");
        }
        return {
          getItem() { throw new Error("Credential storage reads are forbidden"); },
          setItem() { throw new Error("Credential storage writes are forbidden"); },
          removeItem(key) {
            removalAttempts.push(name);
            if (name === deniedStore) throw new Error("Storage removal denied");
            store.delete(key);
          },
        };
      },
    });
  }
  vm.runInContext(helpers, context);
  return { context, stores, removalAttempts };
}

function headers(context, method = "GET") {
  // Convert the VM object's prototype into the test realm for strict comparison.
  return { ...context._buildAuthHeaders({}, method) };
}

test("bootstrap discards both legacy credentials and explicit entry authenticates only this page", () => {
  const { context, stores } = credentialHarness();
  context._applyPortalBootstrap({ authMode: "direct_debug" });
  assert.equal(context.els.apiKeyInput.value, "");
  assert.deepEqual(headers(context), {});
  for (const store of Object.values(stores)) {
    assert.equal(store.has("tp_api_key"), false);
    assert.equal(store.get("unrelated"), "preserve");
  }

  context.els.apiKeyInput.value = " Bearer page-token ";
  // Normal input listeners call this helper with no options.
  context._handleDirectDebugApiKeyUpdate();
  assert.deepEqual(headers(context), { Authorization: "Bearer page-token", "x-api-key": "page-token" });
  for (const store of Object.values(stores)) assert.equal(store.has("tp_api_key"), false);
  context.els.apiKeyInput.value = "";
  assert.deepEqual(headers(context), {});
});

test("managed transitions clear page credentials and retain unsafe-request CSRF behavior", () => {
  const { context } = credentialHarness();
  context._applyPortalBootstrap({ authMode: "direct_debug" });
  context.els.apiKeyInput.value = "page-token";
  context._applyPortalBootstrap({ authMode: "managed", csrfToken: "managed-csrf" });
  assert.equal(context.els.apiKeyInput.value, "");
  assert.deepEqual(headers(context), {});
  assert.deepEqual(headers(context, "POST"), { "X-CSRF-Token": "managed-csrf" });
  context._applyPortalBootstrap({ authMode: "direct_debug" });
  assert.deepEqual(headers(context, "POST"), {});
});

for (const status of ["pending", "degraded", "unavailable"]) {
  test(`${status} bootstrap clears page credentials and fails closed`, () => {
    const { context } = credentialHarness();
    context._applyPortalBootstrap({ authMode: "direct_debug" });
    context.els.apiKeyInput.value = "page-token";
    context._applyPortalBootstrap({ authMode: "direct_debug" }, { status });
    assert.equal(context.els.apiKeyInput.value, "");
    context.els.apiKeyInput.value = "injected-token";
    assert.deepEqual(headers(context, "POST"), {});
    context._handleDirectDebugApiKeyUpdate();
    assert.equal(context.els.apiKeyInput.value, "");
  });
}

for (const deniedStore of ["localStorage", "sessionStorage"]) {
  for (const denyGetter of [false, true]) {
    test(`denied ${deniedStore} ${denyGetter ? "access" : "removal"} cannot block cleanup or managed bootstrap`, () => {
      const { context, stores, removalAttempts } = credentialHarness({ deniedStore, denyGetter });
      context.els.apiKeyInput.value = "page-token";
      assert.doesNotThrow(() => context._applyPortalBootstrap({ authMode: "managed", csrfToken: "csrf" }));
      assert.equal(context.els.apiKeyInput.value, "");
      assert.deepEqual(removalAttempts, ["localStorage", "sessionStorage"]);
      const accessibleStore = deniedStore === "localStorage" ? "sessionStorage" : "localStorage";
      assert.equal(stores[accessibleStore].has("tp_api_key"), false);
      assert.deepEqual(headers(context, "POST"), { "X-CSRF-Token": "csrf" });
      context._applyPortalBootstrap({ authMode: "direct_debug" });
      context.els.apiKeyInput.value = "fresh-page-token";
      assert.doesNotThrow(() => context._handleDirectDebugApiKeyUpdate());
      assert.deepEqual(headers(context), { Authorization: "Bearer fresh-page-token", "x-api-key": "fresh-page-token" });
    });
  }
}
