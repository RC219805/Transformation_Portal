import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { buildPathGuidance, createDeferredBuildSurfaceApi } from "../portal-src/build-surface-deferred.js";

const portal = readFileSync(new URL("../portal-src/portal.template.js", import.meta.url), "utf8");
const html = readFileSync(new URL("../../../portal.html", import.meta.url), "utf8");
const section = (start, end) => portal.slice(portal.indexOf(start), portal.indexOf(end, portal.indexOf(start)));

test("managed guidance describes all pipelines and Unified workflows without inventing approved paths", () => {
  for (const [pipeline, workflow, input, output] of [
    ['lux-depth-v3', undefined, /source-image folder for Lux V3/, /enhanced images and depth/],
    ['lux-depth-v5', undefined, /original photographs for V5/, /enhanced photographs/],
    ['lux-depth-v6', undefined, /original photographs for V6/, /TIFF photographs/],
    ['lux-depth', 'process', /original photographs for Unified grading/, /TIFF photographs/],
    ['lux-depth', 'infer', /original photographs for Unified inference/, /enhanced photographs/],
    ['archive-gate-a', undefined, /archive root whose files match the Archive Index/, /fixity hash manifest/],
    ['archive-gate-b', undefined, /archive source folder containing the relative file paths/, /BagIt package and its build report/],
    ['archive-gate-c', undefined, /workspace folder for dispatch validation.*METS export reads its records from the Rights Manifest/s, /METS XML/],
  ]) {
    const copy = buildPathGuidance({ pipeline, workflow, authMode: 'managed' });
    assert.match(copy.input, input, pipeline);
    assert.match(copy.output, output, pipeline);
    assert.match(copy.summary, /Example defaults do not grant access/);
    assert.match(copy.summary, /current preview validates access/);
    assert.doesNotMatch(JSON.stringify(copy), /Local example|\.\/input_images|\.\/output|\/tenant\//);
    assert.match(copy.archiveIndex, /CSV or CSV.gz.*approved input location/);
    assert.match(copy.rightsManifest, /prior archive stage, in an approved output location/);
    if (['lux-depth-v5', 'lux-depth-v6', 'lux-depth'].includes(pipeline)) {
      assert.match(copy.output, /approved writable destination separate from the source folder and any manifest directories/);
      assert.match(copy.output, /Prefer a fresh folder for each run/);
    }
    assert.doesNotMatch(copy.output, /must not already exist|parent must exist|must be absent or empty/);
  }
});

test("upload guidance follows availability and never promises authorization or changes output", () => {
  const options = { pipeline: 'lux-depth', workflow: 'process', authMode: 'managed' };
  const enabled = buildPathGuidance({ ...options, uploadsAvailable: true });
  const disabled = buildPathGuidance({ ...options, uploadsAvailable: false });
  assert.match(enabled.input, /Choose files or Choose folder stages local files/);
  assert.match(enabled.input, /Uploads leave the output path unchanged/);
  assert.doesNotMatch(enabled.input, /authorized|approved|Uploads are unavailable/);
  assert.match(disabled.input, /Uploads are unavailable here/);
  assert.doesNotMatch(disabled.input, /Choose files or Choose folder/);
  assert.equal(enabled.output, disabled.output);
  for (const pipeline of ['archive-gate-b', 'archive-gate-c']) {
    assert.doesNotMatch(buildPathGuidance({ ...options, pipeline, uploadsAvailable: true }).input, /Choose files|Choose folder|Uploads/);
  }
});

test("only explicit direct-debug mode offers advisory server-relative examples", () => {
  for (const pipeline of ['lux-depth-v3', 'lux-depth-v5', 'lux-depth-v6', 'lux-depth', 'archive-gate-a', 'archive-gate-b', 'archive-gate-c']) {
    const copy = buildPathGuidance({ pipeline, workflow: 'infer', authMode: 'direct_debug' });
    assert.match(copy.summary, /repository root on the processing server.*server path checks/);
    assert.match(copy.input, /Local example only: /);
    assert.match(copy.output, /Local example only: /);
  }
  for (const authMode of [undefined, '', 'unavailable', 'managed']) {
    const copy = buildPathGuidance({ pipeline: 'archive-gate-a', authMode });
    assert.doesNotMatch(JSON.stringify(copy), /Local example|tests\/fixtures/);
    assert.match(copy.summary, /approved for this workspace/);
  }
});

test("persistent guidance survives field errors and rendering never edits draft paths", () => {
  const pathGuidance = Object.fromEntries(['summary', 'input', 'output', 'archiveIndex', 'rightsManifest'].map((key) => [key, { textContent: '' }]));
  const inputDir = { value: '/approved/source' };
  const outputDir = { value: '/approved/runs/custom' };
  const inputDirStatus = { textContent: '' };
  const state = { pipeline: 'lux-depth', auth: { mode: 'managed' }, config: { inputDir: inputDir.value, outputDir: outputDir.value } };
  const before = JSON.stringify(state);
  const payload = { pipeline: 'lux-depth', args: { workflow: 'process', input_dir: inputDir.value, output_dir: outputDir.value } };
  const api = createDeferredBuildSurfaceApi({
    state, els: { pathGuidance, inputDir, outputDir, inputDirStatus },
    generatePayload: () => payload,
    _stagedUploadsEnabledForState: () => false,
    _previewIssueForField: (field) => field === 'input_dir' ? { detail: { message: 'Input is outside this workspace.' } } : null,
    _renderIssueStatus: (el, helper, issue) => { if (el) el.textContent = issue?.detail?.message || helper; }
  });
  api.renderFieldPreviewStatuses();
  assert.equal(inputDirStatus.textContent, 'Input is outside this workspace.');
  assert.match(pathGuidance.input.textContent, /Unified grading/);
  assert.match(pathGuidance.summary.textContent, /approved input and output locations/);
  payload.args.workflow = 'infer';
  api.renderFieldPreviewStatuses();
  assert.match(pathGuidance.input.textContent, /Unified inference/);
  assert.equal(inputDir.value, '/approved/source');
  assert.equal(outputDir.value, '/approved/runs/custom');
  assert.equal(JSON.stringify(state), before);
});

test("bootstrap transitions refresh loaded guidance without loading an inactive Build bundle", () => {
  const state = { pipeline: 'lux-depth', bootstrap: { status: 'ready' }, auth: { mode: 'direct_debug', features: { stagedUploads: true } } };
  const pathGuidance = Object.fromEntries(['summary', 'input', 'output', 'archiveIndex', 'rightsManifest'].map((key) => [key, { textContent: '' }]));
  const els = { pathGuidance };
  const ready = () => state.bootstrap.status === 'ready';
  const buildApi = createDeferredBuildSurfaceApi({
    state, els,
    generatePayload: () => ({ pipeline: state.pipeline, args: { workflow: 'process' } }),
    _stagedUploadsEnabledForState: () => ready() && state.auth.features.stagedUploads,
    _previewIssueForField: () => null,
    _renderIssueStatus() {}
  });
  let loadedApi = null;
  const context = {
    state, els, document: { body: { dataset: {} } },
    _isBootstrapReady: ready,
    _bootstrapSurfaceSummary: () => ({}),
    _deferredBuildSurfaceApi: () => loadedApi,
    _loadDeferredBuildSurface: () => assert.fail('Bootstrap UI must not load the Build bundle')
  };
  for (const name of ['_announcePortalStatus', '_syncBootstrapGuardedControls', 'syncBuildSurfaceApplicability', '_syncOverviewBuildLoadingState', 'renderOperatorActionRail', '_findJobById', 'renderSelectedJobRecoveryActions', 'renderReviewStatusActions', 'renderArtifactViewer', '_syncStagedUploadUi']) {
    context[name] = () => {};
  }
  const sync = new Function(...Object.keys(context), `${section('function _syncBootstrapUi(', 'function _applyPortalBootstrap(')}; return _syncBootstrapUi;`)(...Object.values(context));
  sync();
  assert.equal(pathGuidance.input.textContent, '');
  loadedApi = buildApi;
  sync();
  assert.match(pathGuidance.input.textContent, /Choose files or Choose folder.*Local example only/);
  state.bootstrap.status = 'unavailable';
  state.auth = { mode: 'managed_unavailable', features: {} };
  sync();
  assert.match(pathGuidance.input.textContent, /Uploads are unavailable here/);
  assert.doesNotMatch(pathGuidance.input.textContent, /Choose files|Local example/);
  assert.match(pathGuidance.summary.textContent, /approved for this workspace/);
  state.bootstrap.status = 'ready';
  state.auth = { mode: 'managed', features: { stagedUploads: false } };
  sync();
  assert.match(pathGuidance.input.textContent, /Uploads are unavailable here/);
  state.auth.features.stagedUploads = true;
  sync();
  assert.match(pathGuidance.input.textContent, /Choose files or Choose folder/);
  assert.doesNotMatch(pathGuidance.input.textContent, /Uploads are unavailable|Local example/);
});

test("path controls retain status relationships alongside persistent guidance", () => {
  for (const [id, name] of [['inputDir', 'inputDir'], ['outputDir', 'outputDir'], ['archiveIndexPath', 'archiveIndex'], ['rightsManifestPath', 'rightsManifest']]) {
    const tag = html.match(new RegExp(`<input[^>]*id="${id}"[^>]*>`))?.[0] || '';
    assert.match(tag, new RegExp(`aria-describedby="[^"]*${name}Guidance ${name}Status pathGuidanceSummary"`));
    assert.match(tag, new RegExp(`aria-errormessage="${name}Status"`));
    assert.match(html, new RegExp(`id="${name}Guidance" data-ui="path-guidance-${name}"`));
  }
  for (const id of ['v5MaterialsManifest', 'v5CompanionsManifest', 'v6CompanionsManifest']) {
    assert.match(html, new RegExp(`id="${id}"[^>]*aria-describedby="${id}Guidance"`));
    assert.match(html, new RegExp(`id="${id}Guidance"[^>]*>Existing .* approved input location; leave blank when unused`));
  }
});

test("archive-root preview errors and focus target only alias the Gate A input control", () => {
  const state = { pipeline: 'archive-gate-a' };
  const preview = { status: 'ready', field_errors: [{ field: 'archive_root', message: 'Archive root is missing.' }], field_warnings: [] };
  const issueForField = new Function('state', '_currentPreviewForPayload', `${section('function _previewIssueForField(', 'function _syncIssueAccessibility(')}; return _previewIssueForField;`)(state, () => preview);
  assert.equal(issueForField('input_dir').detail.message, 'Archive root is missing.');
  assert.equal(issueForField('output_dir'), null);
  assert.equal(issueForField('input_dir', { pipeline: 'archive-gate-b' }), null);
  const els = { inputDir: {}, photography: {}, photographyV6: {}, reconstruction: {}, runtime: {}, raw: {}, flags: {} };
  const controlForField = new Function('state', 'els', '_isCompositePhotography', 'portalInternals', `${section('function _buildControlForPreviewField(', 'function _buildStepForControl(')}; return _buildControlForPreviewField;`)(state, els, () => false, { PHOTOGRAPHY_V6_FIELDS: {} });
  assert.equal(controlForField('archive_root'), els.inputDir);
  state.pipeline = 'archive-gate-b';
  assert.equal(issueForField('input_dir'), null);
  assert.equal(controlForField('archive_root'), null);
});
