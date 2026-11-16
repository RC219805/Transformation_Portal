import test from 'node:test';
import assert from 'node:assert/strict';
import { createDeferredConfigTransferApi, MAX_CONFIG_IMPORT_BYTES } from '../portal-src/config-transfer-deferred.js';
import { createPortalConfigState } from '../portal-src/internal/state.js';

function fixture(options = {}) {
  const state = { pipeline: 'lux-depth-v3', config: createPortalConfigState() };
  let owner = 'managed:alice', generation = 0;
  const notices = [], commits = [];
  const host = { state, portalInternals: {}, supportedPipelines: new Set(['lux-depth-v3', 'archive-gate-a']),
    snapshot: () => JSON.stringify([owner, generation, state]),
    migrate: () => {}, notify: (...args) => notices.push(args),
    confirmReplacement: async () => true,
    commit: (next) => { commits.push(next); Object.assign(state, next); },
    _normalizeVerboseQuietFlags: () => {},
    _parsePositiveIntOrNull: (value) => Number(value) > 0 ? Number(value) : null,
    _parseNonNegativeIntOrNull: (value) => Number.isFinite(Number(value)) && Number(value) >= 0 ? Number(value) : null,
    _parseProbabilityOrNull: () => null,
    _textOrFallback: (value, fallback) => value || fallback,
    parseBoolLike: (value, fallback) => typeof value === 'boolean' ? value : fallback,
    ...Object.fromEntries(['Da3ModelKey','DepthBackend','GroupingMode','QualityTier','RunCardVersion','Sam2ModelSize','SegmentationBackend','VlmCaptioningProxyFormat'].map(name => [`_resolve${name}`, value => value])),
    ...options,
  };
  return { state, host, notices, commits, api: createDeferredConfigTransferApi(host),
    edit: () => { generation += 1; }, actor: value => { owner = value; } };
}
const file = (data) => { const text = JSON.stringify(data); return { size: Buffer.byteLength(text), text: async () => text }; };
const payload = { schema: 'tp.portal.export.v1', pipeline: 'archive-gate-a', args: { archive_root: '/authorized/input', output_dir: '/authorized/output', overwrite: false } };

for (const data of [null, [], { pipeline: 'archive-gate-a', args: 'bad' }, { pipeline: 'bad', args: {} }, { schema: 'tp.portal.export.v99', args: {} }, { args: [] }]) {
  test(`invalid configuration leaves the complete draft unchanged: ${JSON.stringify(data)}`, async () => {
    const h = fixture(), before = structuredClone(h.state);
    assert.equal(await h.api.importFile(file(data)), false);
    assert.deepEqual(h.state, before); assert.equal(h.commits.length, 0);
    assert.deepEqual(h.notices, [['Invalid JSON file.', 'error']]);
  });
}

test('size is bounded before reading the file', async () => {
  const h = fixture();
  assert.equal(await h.api.importFile({ size: MAX_CONFIG_IMPORT_BYTES + 1, text: () => assert.fail('must not read') }), false);
  assert.equal(h.commits.length, 0);
});

test('current export and unversioned legacy arguments commit a detached candidate once', async () => {
  for (const schema of ['tp.portal.export.v1', undefined]) {
    const h = fixture(), original = h.state.config;
    assert.equal(await h.api.importFile(file({ ...payload, schema })), true);
    assert.equal(h.commits.length, 1); assert.notEqual(h.state.config, original);
    assert.equal(h.state.pipeline, 'archive-gate-a');
    assert.equal(h.state.config.inputDir, '/authorized/input');
    assert.equal(h.state.config.outputDir, '/authorized/output');
    assert.equal(h.state.config.flags.overwrite, false);
    assert.notEqual(original.inputDir, '/authorized/input');
  }
});

for (const change of ['edit', 'actor']) {
  test(`a delayed read cannot replace a newer ${change}`, async () => {
    let finish; const h = fixture(), before = structuredClone(h.state);
    const result = h.api.importFile({ size: 100, text: () => new Promise(resolve => { finish = resolve; }) });
    if (change === 'edit') h.edit(); else h.actor('managed:bob');
    finish(JSON.stringify(payload));
    assert.equal(await result, false); assert.deepEqual(h.state, before);
    assert.equal(h.commits.length, 0);
  });
}

test('cancelled replacement and edits made while confirming preserve the existing draft', async () => {
  for (const edit of [false, true]) {
    let finish; const h = fixture({ confirmReplacement: () => new Promise(resolve => { finish = resolve; }) });
    const before = structuredClone(h.state), result = h.api.importFile(file(payload));
    await new Promise(resolve => setImmediate(resolve));
    if (edit) h.edit();
    finish(edit);
    assert.equal(await result, false); assert.deepEqual(h.state, before); assert.equal(h.commits.length, 0);
  }
});

test('a second selection supersedes an older read', async () => {
  let finish; const h = fixture();
  const old = h.api.importFile({ size: 100, text: () => new Promise(resolve => { finish = resolve; }) });
  assert.equal(await h.api.importFile(file(payload)), true);
  finish(JSON.stringify({ args: {}, pipeline: 'lux-depth-v3' }));
  assert.equal(await old, false); assert.equal(h.commits.length, 1);
});

test('normalization failure only changes the candidate', async () => {
  const h = fixture({ migrate: args => { args.archive_root = 'changed'; throw new Error('migration failed'); } });
  const before = structuredClone(h.state);
  assert.equal(await h.api.importFile(file(payload)), false); assert.deepEqual(h.state, before);
});
