import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../portal-src/portal.template.js", import.meta.url), "utf8");
function section(start, end) {
  const a = source.indexOf(start), b = source.indexOf(end, a);
  assert.ok(a >= 0 && b > a); return source.slice(a, b);
}
test("an established backend owner freezes during context outage instead of falling back to actor names", () => {
  const owner = { schema: "tp.portal.recovery_context.v1", tenantId: "tenant-a", ownerId: "a".repeat(64) };
  const state = { portalUi: {}, auth: { mode: "managed", actor: { accessEmail: "a@example.test", username: "alice" }, features: {} } };
  let authenticated = true;
  const h = vm.createContext({ state, _portalPrivilegesReady: () => authenticated, generatePayload: () => ({ args: { input_dir: "/draft" } }) });
  vm.runInContext(section("function _validRecoveryContext(", "function _acceptRecoveredJob("), h);
  assert.equal(h._clientRecoveryContext().ready, true, "legacy managed bootstrap remains usable without new features");
  state.auth.recoveryContext = owner;
  assert.equal(h._clientRecoveryContext().scope, owner.ownerId);
  state.auth.recoveryContext = null;
  assert.equal(h._clientRecoveryContext().ready, false);
  assert.equal(h._clientRecoveryContext().scope, owner.ownerId);
  state.auth.actor = { accessEmail: "new@example.test", username: "new" };
  assert.equal(h._clientRecoveryContext().ready, false);
  state.auth.recoveryContext = owner; authenticated = false;
  assert.equal(h._clientRecoveryContext().ready, false);
  authenticated = true; assert.equal(h._clientRecoveryContext().ready, true);
  state.auth.logoutPending = true; assert.equal(h._clientRecoveryContext().ready, false);
});

test("dispatch delegates only after readiness and never invents a local job for unknown transport", async () => {
  const calls = [], state = { backendOk: true, jobs: [], portalUi: {}, config: {} };
  const h = vm.createContext({ state, configImportGeneration: 0, els: {}, generatePayload: () => ({ pipeline: "archive-gate-b" }),
    _blockManagedUnavailableAction: () => false, _firstInvalidBuildInput: () => null,
    currentPipelineDispatchStatus: () => "ready", isLuxPipeline: () => false,
    _loadClientRecoverySurface: async () => ({ submit: async (...args) => { calls.push(args); return null; } }),
    renderWorkspaceGuidance() {}, _syncBootstrapGuardedControls() {}, createToast() {}, DISPATCH_BACKEND_OFFLINE_MESSAGE: "offline",
  });
  vm.runInContext(section("async function submitJob(", "function startSseWatchdog("), h);
  await h.submitJob();
  assert.equal(calls.length, 1); assert.equal(state.jobs.length, 0); assert.equal(state.portalUi.dispatchPending, false);
  assert.equal(calls[0][1].separate, false);
  await h.submitJob({ separate: true }); assert.equal(calls[1][1].separate, true);
  state.backendOk = false; await h.submitJob(); assert.equal(calls.length, 2);
});

test("bootstrap remembers verified ownership before deferred recovery loads and preserves it through outage", () => {
  const state = { portalUi: {}, rum: {}, auth: {} };
  const h = vm.createContext({ state, configImportGeneration: 0, els: {}, _portalPrivilegesReady: () => true, generatePayload: () => ({}),
    _rumTelemetryEnabled: () => false, _setBootstrapStatus() {}, _syncApiKeyInputState() {}, _syncBootstrapUi() {},
    portalInternals: { normalizePortalRumTraceparent: () => "" },
  });
  vm.runInContext(section("function _validRecoveryContext(", "function _acceptRecoveredJob(")
    + section("function _applyPortalBootstrap(", "function _normalizeFetchFailureReason("), h);
  const owner = { schema: "tp.portal.recovery_context.v1", tenantId: "tenant-a", ownerId: "a".repeat(64) };
  h._applyPortalBootstrap({ authMode: "managed", recoveryContextStatus: "ready", recoveryContext: owner });
  assert.equal(state.portalUi.recoveryOwner, owner);
  h._applyPortalBootstrap({ authMode: "managed", recoveryContextStatus: "unavailable", recoveryContext: null });
  assert.equal(state.portalUi.recoveryOwner, owner);
  assert.equal(h._clientRecoveryContext().ready, false);
  assert.equal(h._clientRecoveryContext().scope, owner.ownerId);
});


test("Review uses the verified backend owner and freezes instead of falling back during outage", () => {
  const owner = { schema: "tp.portal.recovery_context.v1", tenantId: "tenant-a", ownerId: "a".repeat(64) };
  const state = { portalUi: { recoveryOwner: owner }, auth: { recoveryContext: owner, recoveryContextStatus: "ready" } };
  const h = vm.createContext({ state, _transientDraftOwnerKey: () => "verified-actor-pair" });
  vm.runInContext(section("function _validRecoveryContext(", "function _clientRecoveryContext(")
    + section("function _reviewAuthorization(", "function _requestReviewNavigation("), h);
  assert.equal(h._reviewAuthorization().ownerKey, owner.ownerId);
  assert.equal(h._reviewAuthorization().status, "ready");
  state.auth.recoveryContextStatus = "unavailable"; state.auth.recoveryContext = null;
  assert.equal(h._reviewAuthorization().status, "unavailable");
  assert.equal(h._reviewAuthorization().ownerKey, owner.ownerId);
  state.auth.logoutPending = true; assert.equal(h._reviewAuthorization().status, "revoked");
});
