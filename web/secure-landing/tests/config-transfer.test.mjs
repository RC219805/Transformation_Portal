import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createDeferredConfigTransferApi, MAX_CONFIG_IMPORT_BYTES } from '../portal-src/config-transfer-deferred.js';
import { createPortalConfigState } from '../portal-src/internal/state.js';

const resolverNames = ['Da3ModelKey', 'DepthBackend', 'GroupingMode', 'QualityTier', 'RunCardVersion', 'Sam2ModelSize', 'SegmentationBackend', 'VlmCaptioningProxyFormat'];
const source = readFileSync(new URL('../portal-src/portal.template.js', import.meta.url), 'utf8');
const firstNormalizer = source.indexOf('const TRUTHY_VALUES =');
const afterResolvers = source.indexOf('function _parsePositiveIntOrNull(', firstNormalizer);
assert.ok(firstNormalizer >= 0 && afterResolvers > firstNormalizer);
const productionNormalizers = vm.runInNewContext(source.slice(firstNormalizer, afterResolvers)
  + `\n({ _textOrFallback, parseBoolLike, ${resolverNames.map(name => `_resolve${name}`).join(', ')} })`);

function fixture(options = {}) {
  const state = { pipeline: 'lux-depth-v3', config: createPortalConfigState() };
  let owner = 'managed:alice', generation = 0;
  const notices = [], commits = [];
  const host = { state, portalInternals: {}, supportedPipelines: new Set(['lux-depth-v3', 'archive-gate-a']),
    configTemplate: createPortalConfigState,
    snapshot: () => JSON.stringify([owner, generation, state]),
    migrate: () => {}, notify: (...args) => notices.push(args),
    confirmReplacement: async () => true,
    commit: (next) => { commits.push(next); Object.assign(state, next); },
    _normalizeVerboseQuietFlags: () => {},
    _parsePositiveIntOrNull: (value) => Number(value) > 0 ? Number(value) : null,
    _parseNonNegativeIntOrNull: (value) => Number.isFinite(Number(value)) && Number(value) >= 0 ? Number(value) : null,
    _parseProbabilityOrNull: () => null,
    ...productionNormalizers,
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

for (const size of [NaN, Infinity, -Infinity, '12', null, undefined, -1]) {
  test(`non-finite or nonnumeric file size ${String(size)} is rejected before reading`, async () => {
    const h = fixture(), before = structuredClone(h.state);
    assert.equal(await h.api.importFile({ size, text: () => assert.fail('must not read') }), false);
    assert.deepEqual(h.state, before); assert.equal(h.commits.length, 0);
  });
}

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

test('raw text imports preserve existing whitespace semantics', async () => {
  for (const [field, path] of [
    ['input_dir', ['inputDir']], ['archive_root', ['inputDir']], ['output_dir', ['outputDir']],
    ['preset', ['preset']], ['depth_device', ['depthDevice']], ['v2_preset', ['v2Preset']],
  ]) {
    const h = fixture(), value = '  retained import text  ';
    assert.equal(await h.api.importFile(file({ args: { [field]: value } })), true);
    assert.equal(path.reduce((current, key) => current[key], h.state.config), value, field);
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

for (const field of ['input_dir', 'archive_root', 'output_dir', 'preset', 'depth_device', 'v2_input_color', 'sam2_checkpoint_path']) {
  test(`object-valued ${field} is rejected before replacing any draft state`, async () => {
    const h = fixture(), before = structuredClone(h.state);
    assert.equal(await h.api.importFile(file({ ...payload, args: { [field]: { toString: null } } })), false);
    assert.deepEqual(h.state, before); assert.equal(h.commits.length, 0);
    assert.deepEqual(h.notices, [['Invalid JSON file.', 'error']]);
  });
}

for (const field of ['input_dir', 'archive_root', 'output_dir', 'preset', 'depth_device', 'v2_input_color']) {
  for (const value of [0, 7, false, true, [], { toString: null }]) {
    test(`invalid primitive/container type for ${field} is rejected atomically: ${JSON.stringify(value)}`, async () => {
      const h = fixture(), before = structuredClone(h.state);
      assert.equal(await h.api.importFile(file({ args: { [field]: value } })), false);
      assert.deepEqual(h.state, before); assert.equal(h.commits.length, 0);
    });
  }
}

const normalizedTextFields = [
  ['sam2_checkpoint_path', ['segmentation', 'sam2CheckpointPath'], '/models/sam2.pt'],
  ['archive_index', ['gate', 'archiveIndex'], '/input/index.json'],
  ['manifest_jsonl', ['gate', 'manifestJsonl'], '/input/rights.jsonl'],
  ['cameras_sidecar_path', ['reconstruction', 'camerasSidecarPath'], '/input/cameras.json'],
  ['reconstruction_tier', ['reconstruction', 'tier'], 'apex_research'],
  ['raw_ingest_mode', ['raw', 'ingestMode'], 'auto'],
  ['raw_wb_mode', ['raw', 'wbMode'], 'camera'],
  ['raw_demosaic', ['raw', 'demosaic'], 'AHD'],
  ['vlm_captioning_model', ['captioning', 'model'], 'default'],
  ['fastvlm_python_executable', ['captioning', 'pythonExecutable'], '/runtime/python'],
  ['fastvlm_mlx_vlm_dir', ['captioning', 'mlxVlmDir'], '/runtime/mlx-vlm'],
  ['log_level', ['runtime', 'logLevel'], 'INFO'],
];

const selectorFields = [
  ['quality_tier', ['qualityTier'], ' APEX ', 'apex', 'standard'],
  ['depth_backend', ['depthBackend'], ' DEPTH_PRO ', 'depth_pro', 'da3'],
  ['model_key', ['modelKey'], ' DA3_RESEARCH ', 'da3-research', 'da3-metric'],
  ['segmentation_backend', ['segmentation', 'backend'], ' SAM2 ', 'sam2', 'stub'],
  ['sam2_model_size', ['segmentation', 'sam2ModelSize'], ' LARGE ', 'large', 'base'],
  ['run_card_version', ['emits', 'runCardVersion'], ' V2 ', 'v2', 'v1'],
  ['grouping_mode', ['reconstruction', 'groupingMode'], ' PARENT_DIR ', 'parent_dir', 'single'],
  ['vlm_captioning_proxy_format', ['captioning', 'proxyFormat'], ' JPEG ', 'jpeg', 'png'],
];

for (const [field, path, legacy, normalized, closedDefault] of selectorFields) {
  for (const value of [{}, [], false, true, 0, 7]) {
    test(`raw selector ${field} rejects ${JSON.stringify(value)} before production normalization`, async () => {
      let confirmations = 0;
      const h = fixture({ confirmReplacement: async () => { confirmations += 1; return true; } });
      const before = structuredClone(h.state);
      assert.equal(await h.api.importFile(file({ pipeline: 'archive-gate-a', args: { [field]: value } })), false);
      assert.deepEqual(h.state, before); assert.equal(h.commits.length, 0); assert.equal(confirmations, 0);
      assert.deepEqual(h.notices, [['Invalid JSON file.', 'error']]);
    });
  }
  test(`${field} preserves production string normalization and distinct empty/whitespace fallbacks`, async () => {
    const h = fixture(), read = () => path.reduce((current, key) => current[key], h.state.config);
    assert.equal(await h.api.importFile(file({ args: { [field]: legacy } })), true);
    assert.equal(read(), normalized);
    for (const value of [undefined, null, '']) {
      assert.equal(await h.api.importFile(file({ args: { [field]: value } })), true);
      assert.equal(read(), normalized);
    }
    for (const value of ['   ', 'unsupported-selector']) {
      assert.equal(await h.api.importFile(file({ args: { [field]: value } })), true);
      assert.equal(read(), closedDefault);
    }
  });
}

test('production depth and model aliases retain their existing selector meaning', async () => {
  const h = fixture();
  assert.equal(await h.api.importFile(file({ args: { depth_backend: ' depth-anything-v3 ', model_key: ' DA3 ' } })), true);
  assert.equal(h.state.config.depthBackend, 'da3'); assert.equal(h.state.config.modelKey, 'da3-research');
});

for (const field of ['output_bit_depth', 'outputBitDepth']) {
  for (const value of [{}, [], false, true]) {
    test(`raw output selector ${field} rejects ${JSON.stringify(value)} atomically`, async () => {
      let confirmations = 0;
      const h = fixture({ confirmReplacement: async () => { confirmations += 1; return true; } });
      const before = structuredClone(h.state);
      assert.equal(await h.api.importFile(file({ pipeline: 'archive-gate-a', args: { [field]: value } })), false);
      assert.deepEqual(h.state, before); assert.equal(h.commits.length, 0); assert.equal(confirmations, 0);
    });
  }
  test(`${field} preserves existing numeric/string resolution and missing/null fallback`, async () => {
    for (const value of [undefined, null, '', 0, -1, 8, 16, 12, '8', ' 8 ', '16', 'unsupported']) {
      const h = fixture(); h.state.config.outputBitDepth = 8;
      assert.equal(await h.api.importFile(file({ args: { [field]: value } })), true);
      assert.equal(h.state.config.outputBitDepth, Number(value ?? 8) === 8 ? 8 : 16);
    }
  });
}

test('both output aliases are validated even when the preferred alias is valid', async () => {
  const h = fixture(), before = structuredClone(h.state);
  assert.equal(await h.api.importFile(file({ args: { output_bit_depth: 8, outputBitDepth: {} } })), false);
  assert.deepEqual(h.state, before); assert.equal(h.commits.length, 0);
  assert.equal(await h.api.importFile(file({ args: { output_bit_depth: '8', outputBitDepth: 16 } })), true);
  assert.equal(h.state.config.outputBitDepth, 8);
});

for (const [field, path, legacy] of normalizedTextFields) {
  for (const value of [0, 7, false, true, [], {}]) {
    test(`raw ${field} rejects ${JSON.stringify(value)} before text coercion`, async () => {
      const h = fixture(), before = structuredClone(h.state);
      assert.equal(await h.api.importFile(file({ args: { [field]: value } })), false);
      assert.deepEqual(h.state, before); assert.equal(h.commits.length, 0);
      assert.deepEqual(h.notices, [['Invalid JSON file.', 'error']]);
    });
  }
  test(`${field} preserves trimmed legacy strings and optional text fallbacks`, async () => {
    const h = fixture();
    assert.equal(await h.api.importFile(file({ args: { [field]: `  ${legacy}  ` } })), true);
    const read = () => path.reduce((value, key) => value[key], h.state.config);
    assert.equal(read(), legacy);
    for (const value of [undefined, null, '', '   ']) {
      assert.equal(await h.api.importFile(file({ args: { [field]: value } })), true);
      assert.equal(read(), legacy);
    }
  });
}

test('canonical field types preserve numeric imports after an editable numeric control held text', async () => {
  const h = fixture(); h.state.config.reconstruction.iterations = '1200';
  assert.equal(await h.api.importFile(file({ args: { reconstruction_iterations: '1000', max_workers: 2 } })), true);
  assert.equal(h.state.config.reconstruction.iterations, 1000);
  assert.equal(h.state.config.runtime.maxWorkers, 2);
});

test('the detached import preserves all existing boolean flag mappings', async () => {
  const h = fixture();
  assert.equal(await h.api.importFile(file({ args: { materials_v3: false, pbr: false, cache_depth: false, overwrite: true,
    enable_v2: true, save_float_depth: true, force_depth: true, strict_inputs: true, verify_images: true,
    allow_semantic_fallback: true, verbose: true, quiet: false } })), true);
  assert.deepEqual(h.state.config.flags, { materials: false, pbr: false, cache: false, overwrite: true, enableV2: true,
    saveFloatDepth: true, forceDepth: true, strictInputs: true, verifyImages: true, allowSemanticFallback: true,
    verbose: true, quiet: false });
});

test('final configuration normalization finishes on the detached candidate', async () => {
  let calls = 0;
  const h = fixture({ migrate: value => { calls += 1; if (calls === 2) { value.inputDir = '/candidate-only'; throw new Error('normalization failed'); } } });
  const before = structuredClone(h.state);
  assert.equal(await h.api.importFile(file(payload)), false);
  assert.equal(calls, 2); assert.deepEqual(h.state, before); assert.equal(h.commits.length, 0);
});
