// Build surface deferred bundle.
//
// Carved out of portal-src/portal.template.js. This module owns Build-only
// form-render UI: pipeline metadata projection onto controls, archive vs.
// lux field visibility, runtime worker-mode controls, and per-field
// config-preview status rendering. All shared state, DOM, callbacks,
// formatters, and parsers are passed via the host so the bundle never
// reads from globals.
//
// What stays in portal.template.js (and rides the host):
//   - state, els (single object refs each)
//   - submitJob, _firstInvalidBuildInput (dispatch core; not exposed here)
//   - generatePayload (30+ call sites across surfaces; payload synthesis)
//   - fetchConfigPreview, scheduleConfigPreview (transport layer)
//   - _renderIssueStatus, _previewIssueForField (also used by
//     renderReconstructionRuntimeSummary which stays in main)
//   - _metadataField, _normalizeWorkerMode (shared with status-text
//     helpers in main)
//   - _resolveDa3ModelKey, canonicalArchiveCommand (cross-surface
//     resolvers that this carve does not own)

export function createDeferredBuildSurfaceApi(host) {
    const {
        state,
        els,
        _metadataField,
        _normalizeWorkerMode,
        _previewIssueForField,
        _renderIssueStatus,
        _resolveDa3ModelKey,
        canonicalArchiveCommand,
        generatePayload,
    } = host;

    function _setSelectOptions(selectEl, options, selectedValue) {
        if (!selectEl || !Array.isArray(options) || options.length === 0) return;
        const normalizedOptions = options
            .map((option) => ({
                value: String(option?.value ?? '').trim(),
                label: String(option?.label ?? option?.value ?? '').trim()
            }))
            .filter((option) => option.value);
        if (normalizedOptions.length === 0) return;
        const preferred = String(selectedValue ?? selectEl.value ?? '').trim();
        selectEl.innerHTML = '';
        normalizedOptions.forEach((option) => {
            const el = document.createElement('option');
            el.value = option.value;
            el.textContent = option.label || option.value;
            selectEl.appendChild(el);
        });
        const hasPreferred = normalizedOptions.some((option) => option.value === preferred);
        selectEl.value = hasPreferred ? preferred : normalizedOptions[0].value;
    }

    function _applyNumericConstraints(input, field) {
        if (!input || !field) return;
        if (field.min !== undefined && field.min !== null) input.min = String(field.min);
        if (field.max !== undefined && field.max !== null) input.max = String(field.max);
        if (field.step !== undefined && field.step !== null) input.step = String(field.step);
    }

    function _buildFieldStatusCopy(fieldName, payload = null) {
        const currentPayload = payload || generatePayload();
        if (fieldName === 'input_dir') {
            return currentPayload.pipeline === 'archive-gate-a'
                ? 'Choose the archive root that fixity-scan should inspect.'
                : 'Set the source folder for the next run.';
        }
        if (fieldName === 'output_dir') {
            return currentPayload.pipeline === 'lux-depth-v3'
                ? 'Choose the governed destination for generated outputs.'
                : 'Choose the governed destination for stage outputs.';
        }
        if (fieldName === 'archive_index') {
            return 'Required for fixity-scan. Supply an existing normalized archive index that is safe to read from the local allowlist.';
        }
        if (fieldName === 'manifest_jsonl') {
            return 'Required for bag-build and mets-export. Point to a rights-manifest artifact produced by an earlier archive stage.';
        }
        return '';
    }

    function _setInputRequired(input, required) {
        if (!input) return;
        if (required) {
            input.setAttribute('required', '');
            input.setAttribute('aria-required', 'true');
        } else {
            input.removeAttribute('required');
            input.removeAttribute('aria-required');
        }
    }

    function applyLuxMetadataToControls() {
        if (state.pipeline !== 'lux-depth-v3') return;

        const groupingField = _metadataField('grouping_mode');
        const tierField = _metadataField('reconstruction_tier');
        const rawIngestField = _metadataField('raw_ingest_mode');
        const rawWbField = _metadataField('raw_wb_mode');
        const rawDemosaicField = _metadataField('raw_demosaic');
        const maxWorkersField = _metadataField('max_workers');
        const maxGpuWorkersField = _metadataField('max_gpu_workers');
        const iterationsField = _metadataField('reconstruction_iterations');
        const logLevelField = _metadataField('log_level');
        const modelKeyField = _metadataField('model_key');

        if (els.modelKey && modelKeyField?.options) {
            _setSelectOptions(els.modelKey, modelKeyField.options, state.config.modelKey);
            state.config.modelKey = _resolveDa3ModelKey(els.modelKey.value || state.config.modelKey);
        }

        if (els.reconstruction.groupingMode && groupingField?.options) {
            _setSelectOptions(els.reconstruction.groupingMode, groupingField.options, state.config.reconstruction?.groupingMode);
            state.config.reconstruction.groupingMode = String(els.reconstruction.groupingMode.value || state.config.reconstruction?.groupingMode || 'single');
        }
        if (els.reconstruction.tier && tierField?.options) {
            _setSelectOptions(els.reconstruction.tier, tierField.options, state.config.reconstruction?.tier);
            state.config.reconstruction.tier = String(els.reconstruction.tier.value || state.config.reconstruction?.tier || 'apex_research');
        }
        if (els.raw.ingestMode && rawIngestField?.options) {
            _setSelectOptions(els.raw.ingestMode, rawIngestField.options, state.config.raw?.ingestMode);
            state.config.raw.ingestMode = String(els.raw.ingestMode.value || state.config.raw?.ingestMode || 'auto');
        }
        if (els.runtime.logLevel && logLevelField?.options) {
            _setSelectOptions(els.runtime.logLevel, logLevelField.options, state.config.runtime?.logLevel);
            state.config.runtime.logLevel = String(els.runtime.logLevel.value || state.config.runtime?.logLevel || '');
        }
        if (els.raw.wbModeBadge && rawWbField) {
            els.raw.wbModeBadge.textContent = String(rawWbField.display_value || rawWbField.default || 'camera');
        }
        if (els.raw.wbModeHint && rawWbField?.helper_text) {
            els.raw.wbModeHint.textContent = String(rawWbField.helper_text);
        }
        if (els.raw.wbMode) {
            els.raw.wbMode.value = String(rawWbField?.display_value || rawWbField?.default || state.config.raw?.wbMode || 'camera');
        }
        if (els.raw.demosaicBadge && rawDemosaicField) {
            els.raw.demosaicBadge.textContent = String(rawDemosaicField.display_value || rawDemosaicField.default || 'AHD');
        }
        if (els.raw.demosaicHint && rawDemosaicField?.helper_text) {
            els.raw.demosaicHint.textContent = String(rawDemosaicField.helper_text);
        }
        if (els.raw.demosaic) {
            els.raw.demosaic.value = String(rawDemosaicField?.display_value || rawDemosaicField?.default || state.config.raw?.demosaic || 'AHD');
        }
        _applyNumericConstraints(els.reconstruction.iterations, iterationsField);
        _applyNumericConstraints(els.runtime.maxWorkers, maxWorkersField);
        _applyNumericConstraints(els.runtime.maxGpuWorkers, maxGpuWorkersField);
    }

    function renderFieldPreviewStatuses(payload = null) {
        const currentPayload = payload || generatePayload();
        _renderIssueStatus(
            els.inputDirStatus,
            _buildFieldStatusCopy('input_dir', currentPayload),
            _previewIssueForField('input_dir', currentPayload),
            els.inputDir
        );
        _renderIssueStatus(
            els.outputDirStatus,
            _buildFieldStatusCopy('output_dir', currentPayload),
            _previewIssueForField('output_dir', currentPayload),
            els.outputDir
        );
        _renderIssueStatus(
            els.archiveIndexStatus,
            _buildFieldStatusCopy('archive_index', currentPayload),
            _previewIssueForField('archive_index', currentPayload),
            els.archiveIndexPath
        );
        _renderIssueStatus(
            els.rightsManifestStatus,
            _buildFieldStatusCopy('manifest_jsonl', currentPayload),
            _previewIssueForField('manifest_jsonl', currentPayload),
            els.rightsManifestPath
        );
    }

    function syncRuntimeWorkerModeControls() {
        const runtime = state.config.runtime || {};
        runtime.maxWorkersMode = _normalizeWorkerMode(runtime.maxWorkersMode || (runtime.maxWorkers ? 'fixed' : 'auto'));
        runtime.maxGpuWorkersMode = _normalizeWorkerMode(runtime.maxGpuWorkersMode || (runtime.maxGpuWorkers ? 'fixed' : 'auto'));
        state.config.runtime = runtime;

        if (els.runtime.maxWorkersMode) {
            els.runtime.maxWorkersMode.value = runtime.maxWorkersMode;
        }
        if (els.runtime.maxWorkersValueField) {
            els.runtime.maxWorkersValueField.classList.toggle('hidden', runtime.maxWorkersMode !== 'fixed');
        }
        if (els.runtime.maxWorkers) {
            els.runtime.maxWorkers.disabled = runtime.maxWorkersMode !== 'fixed';
        }
        if (els.runtime.maxGpuWorkersMode) {
            els.runtime.maxGpuWorkersMode.value = runtime.maxGpuWorkersMode;
        }
        if (els.runtime.maxGpuWorkersValueField) {
            els.runtime.maxGpuWorkersValueField.classList.toggle('hidden', runtime.maxGpuWorkersMode !== 'fixed');
        }
        if (els.runtime.maxGpuWorkers) {
            els.runtime.maxGpuWorkers.disabled = runtime.maxGpuWorkersMode !== 'fixed';
        }
    }

    function refreshArchiveFieldVisibility() {
        const pipelineName = state.pipeline;
        const canonicalCommand = canonicalArchiveCommand(pipelineName);
        if (els.archiveCanonicalCommand) els.archiveCanonicalCommand.textContent = canonicalCommand || 'archive';
        if (els.archiveCanonicalCommandHint) {
            els.archiveCanonicalCommandHint.textContent = pipelineName === 'archive-gate-a'
                ? 'The portal build flow uses fixity-scan for archive-gate-a. For a safe local smoke run, pair ./tests/fixtures/archive_small/archive_root with /tmp/gate-a-smoke-portal and ./tests/fixtures/archive_small/archive_index_normalized.csv.gz.'
                : 'The portal build flow uses a prior rights-manifest artifact for downstream archive stages.';
        }
        if (els.archiveIndexField) {
            els.archiveIndexField.classList.toggle('hidden', pipelineName !== 'archive-gate-a');
        }
        if (els.rightsManifestField) {
            els.rightsManifestField.classList.toggle('hidden', !(pipelineName === 'archive-gate-b' || pipelineName === 'archive-gate-c'));
        }
        _setInputRequired(els.archiveIndexPath, pipelineName === 'archive-gate-a');
        _setInputRequired(els.rightsManifestPath, pipelineName === 'archive-gate-b' || pipelineName === 'archive-gate-c');
    }

    return {
        applyLuxMetadataToControls,
        renderFieldPreviewStatuses,
        syncRuntimeWorkerModeControls,
        refreshArchiveFieldVisibility,
    };
}
