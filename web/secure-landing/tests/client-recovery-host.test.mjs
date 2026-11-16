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
  const h = vm.createContext({ state, configImportGeneration: 0, _portalPrivilegesReady: () => authenticated, generatePayload: () => ({ args: { input_dir: "/draft" } }) });
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
    _clientRecoveryOperationStamp: () => "stable-operation",
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


function deferredOperationHarness() {
  const owner = { schema: 'tp.portal.recovery_context.v1', tenantId: 'tenant-a', ownerId: 'a'.repeat(64) };
  const state = { backendOk: true, portalUi: {}, config: {}, jobs: [],
    auth: { mode: 'managed', actor: { accessEmail: 'a@example.test', username: 'alice' }, recoveryContextStatus: 'ready', recoveryContext: owner, features: {} } };
  let ready = true, payload = { pipeline: 'archive-gate-b', args: { input_dir: '/alice/source' } }, release;
  const pending = new Promise(resolve => { release = resolve; });
  const submitted = [], uploaded = [], notices = [];
  const selected = { entries: [{ file: { owner: 'alice' }, relativePath: 'alice.jpg', name: 'alice.jpg', sizeBytes: 4 }], totalBytes: 4 };
  const h = vm.createContext({ state, configImportGeneration: 0, els: {},
    _portalPrivilegesReady: () => ready, generatePayload: () => structuredClone(payload),
    _blockManagedUnavailableAction: () => false, _firstInvalidBuildInput: () => null,
    currentPipelineDispatchStatus: () => 'ready', isLuxPipeline: () => false,
    _stagedUploadsEnabledForState: () => true, _isProtectedFamilySuppressed: () => false,
    _collectStagedUploadSelection: () => selected,
    _setStagedUploadState: patch => { state.portalUi.stagedUpload = { ...state.portalUi.stagedUpload, ...patch }; },
    _syncStagedUploadUi() {}, _loadClientRecoverySurface: () => pending,
    renderWorkspaceGuidance() {}, _syncBootstrapGuardedControls() {},
    createToast: (...args) => notices.push(args), DISPATCH_BACKEND_OFFLINE_MESSAGE: 'offline',
  });
  vm.runInContext(section('function _validRecoveryContext(', 'function _acceptRecoveredJob(')
    + section('function _applyStagedUploadResult(', 'function resumeBlockedJobStreamsAfterAuthUpdate(')
    + section('async function submitJob(', 'function startSseWatchdog('), h);
  return { state, h, submitted, uploaded, notices,
    release: () => release({ submit: data => submitted.push({ owner: state.auth.recoveryContext.ownerId, data }),
      upload: data => uploaded.push({ owner: state.auth.recoveryContext.ownerId, data }) }),
    change(scenario) {
      if (scenario === 'owner') {
        state.auth.actor.username = 'bob'; state.auth.recoveryContext = { ...owner, tenantId: 'tenant-b', ownerId: 'b'.repeat(64) };
        h.configImportGeneration += 1;
      } else if (scenario === 'owner-aba') {
        state.auth.recoveryContext = { ...owner, ownerId: 'b'.repeat(64) }; h.configImportGeneration += 1;
        state.auth.recoveryContext = owner; h.configImportGeneration += 1;
      } else if (scenario === 'unready') { ready = false; h.configImportGeneration += 1; }
      else if (scenario === 'unready-aba') {
        ready = false; h.configImportGeneration += 1;
        ready = true; h.configImportGeneration += 1;
      }
      else if (scenario === 'edit') { payload.args.input_dir = '/alice/new-source'; h.configImportGeneration += 1; }
      else if (scenario === 'aba') {
        const original = structuredClone(payload);
        payload.args.input_dir = '/alice/temporary'; h.configImportGeneration += 1;
        payload = original; h.configImportGeneration += 1;
      }
    },
  };
}

for (const type of ['job', 'upload']) {
  for (const scenario of ['owner', 'owner-aba', 'unready', 'unready-aba', 'edit', 'aba']) {
    test(`${type} first-load continuation sends nothing after ${scenario} context change`, async () => {
      const h = deferredOperationHarness();
      const operation = type === 'job' ? h.h.submitJob() : h.h._submitStagedUploadSelection([]);
      h.change(scenario); h.release();
      await operation;
      for (let index = 0; index < 8; index += 1) await Promise.resolve();
      assert.equal(h.submitted.length, 0); assert.equal(h.uploaded.length, 0);
      const messages = [...h.notices.map(([message]) => message), h.state.portalUi.stagedUpload?.error || ''];
      assert.ok(messages.some(message => /No (?:request|upload) was sent/.test(message)));
    });
  }
  test(`${type} first-load continuation delegates once when its source context stays current`, async () => {
    const h = deferredOperationHarness();
    const operation = type === 'job' ? h.h.submitJob() : h.h._submitStagedUploadSelection([]);
    h.release(); await operation;
    for (let index = 0; index < 8; index += 1) await Promise.resolve();
    assert.equal(h.submitted.length, type === 'job' ? 1 : 0);
    assert.equal(h.uploaded.length, type === 'upload' ? 1 : 0);
  });
}

function recoveredJobHostHarness() {
  const state = { currentView: 'build', jobs: [], selectedJobId: 'existing-job', portalUi: {} };
  const notices = [], scheduled = [], events = [];
  let ready = true, dirty = false;
  const h = vm.createContext({ state, els: { logPane: {}, runJobBtn: { dataset: {} } },
    _clientRecoveryContext: () => ({ ready }),
    deferredReviewSurfaceApi: { hasUnsavedChanges: () => dirty },
    _findJobById: id => state.jobs.find(row => row.id === id),
    hydrateJobFromServer: data => ({ ...data, logs: [], pipeline: 'archive-gate-b' }),
    scheduleRenderJobQueue: () => scheduled.push(state.selectedJobId),
    createToast: (...args) => notices.push(args), _announcePortalStatus() {},
    emitPortalEvent: (...args) => events.push(args),
    _isJobStreamRecoverable: () => false, _jobHasActiveStream: () => false,
    startJobEventStream() {}, refreshJobStatus() {},
  });
  vm.runInContext(section('function _acceptRecoveredJob(', 'function _loadClientRecoverySurface('), h);
  return { state, notices, scheduled, events, h,
    ready: value => { ready = value; }, dirty: value => { dirty = value; } };
}

test('automatic recovered-run attachment waits while the operator is away from Build', () => {
  const h = recoveredJobHostHarness(), job = { id: 'accepted-job' };
  for (const view of ['overview', 'operate', 'review']) {
    h.state.currentView = view;
    assert.equal(h.h._acceptRecoveredJob(job, { automatic: true }), false);
    assert.equal(h.state.selectedJobId, 'existing-job');
    assert.equal(h.state.jobs.length, 0);
    assert.equal(h.scheduled.length, 0);
  }
  h.state.currentView = 'build';
  assert.equal(h.h._acceptRecoveredJob(job, { automatic: true }), true);
  assert.equal(h.state.selectedJobId, job.id);
  assert.equal(h.state.jobs.length, 1);
});

test('explicit recovered-run Open preserves dirty Review until Save or discard has completed', () => {
  const h = recoveredJobHostHarness(), job = { id: 'accepted-job' };
  h.state.currentView = 'review'; h.dirty(true);
  assert.equal(h.h._acceptRecoveredJob(job, { automatic: false }), false);
  assert.equal(h.state.selectedJobId, 'existing-job');
  assert.equal(h.state.jobs.length, 0);
  assert.equal(h.events.length, 0);
  assert.match(h.notices[0][0], /Save or discard Review edits/);
  h.dirty(false);
  assert.equal(h.h._acceptRecoveredJob(job, { automatic: false }), true);
  assert.equal(h.state.selectedJobId, job.id);
  assert.deepEqual(h.scheduled, [job.id]);
});

test('recovered-run attachment fails closed during unavailable access or a retained Review draft', () => {
  const h = recoveredJobHostHarness(), job = { id: 'accepted-job' };
  h.ready(false);
  assert.equal(h.h._acceptRecoveredJob(job), false);
  h.ready(true); h.dirty(true);
  assert.equal(h.h._acceptRecoveredJob(job, { automatic: true }), false);
  assert.equal(h.state.selectedJobId, 'existing-job');
  assert.equal(h.state.jobs.length, 0);
});
