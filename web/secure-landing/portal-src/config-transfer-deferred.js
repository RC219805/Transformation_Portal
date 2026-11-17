// Configuration transfer is loaded only after an explicit import action.
export const MAX_CONFIG_IMPORT_BYTES = 1024 * 1024;

export function createDeferredConfigTransferApi(host) {
    const { state, portalInternals, _normalizeVerboseQuietFlags, _parseNonNegativeIntOrNull, _parsePositiveIntOrNull, _parseProbabilityOrNull, _resolveDa3ModelKey, _resolveDepthBackend, _resolveGroupingMode, _resolveQualityTier, _resolveRunCardVersion, _resolveSam2ModelSize, _resolveSegmentationBackend, _resolveVlmCaptioningProxyFormat, _textOrFallback, parseBoolLike } = host;
    let readIntent = 0;
    function text(value, fallback, normalize) {
        if (value != null && typeof value !== 'string') throw new Error();
        return normalize ? _textOrFallback(value, fallback) : value || fallback;
    }
    function validateFields(next, previous) {
        for (const [key, value] of Object.entries(next)) {
            const expected = previous?.[key] ?? (key === 'emit_master16' ? false : '');
            let valid = value !== null && typeof value === typeof expected;
            if (value !== null && typeof value === 'object') {
                valid = !Array.isArray(value) && expected !== null && typeof expected === 'object';
                if (valid) validateFields(value, expected);
            } else if (typeof expected === 'number') {
                valid = (typeof value === 'number' || typeof value === 'string') && String(value).trim() !== '' && Number.isFinite(Number(value));
            } else if (key === 'maxWorkers' || key === 'maxGpuWorkers') {
                valid = value === '' || Number.isSafeInteger(value) && value > 0;
            }
            if (!valid) throw new Error();
        }
    }
    function candidate(data) {
        if (!data || typeof data !== 'object' || Array.isArray(data)
            || (data.schema !== undefined && data.schema !== 'tp.portal.export.v1')
            || !data.args || typeof data.args !== 'object' || Array.isArray(data.args)) {
            throw new Error();
        }
        const { pipeline = state.pipeline, args } = data;
        if (typeof pipeline !== 'string' || !host.supportedPipelines.has(pipeline)) throw new Error();
        // importFile already parsed detached JSON; only the live draft needs copying.
        const c = JSON.parse(JSON.stringify(state.config));
        if (typeof args.output_bit_depth !== 'number') text(args.output_bit_depth);
        if (typeof args.outputBitDepth !== 'number') text(args.outputBitDepth);
        host.migrate(args);
            if (pipeline === 'lux-depth') {
                c.photographyUnified = portalInternals.unifiedPhotographyConfigFromArgs(args, c.photographyUnified);
            } else if (pipeline === 'lux-depth-v6') {
                c.photographyV6 = portalInternals.photographyV6ConfigFromArgs(args, c.photographyV6);
            } else if (pipeline === 'lux-depth-v5') {
                c.photography = portalInternals.photographyConfigFromArgs(args, c.photography);
            }
            if ('emit_master16' in args) c.emit_master16 = args.emit_master16;
            c.inputDir = text(args.archive_root, text(args.input_dir, c.inputDir));
            c.outputDir = text(args.output_dir, c.outputDir);
            c.preset = text(args.preset, c.preset);
            c.qualityTier = _resolveQualityTier(text(args.quality_tier, c.qualityTier));
            c.depthBackend = _resolveDepthBackend(text(args.depth_backend, c.depthBackend));
            c.modelKey = _resolveDa3ModelKey(text(args.model_key, c.modelKey));
            c.depthDevice = text(args.depth_device, c.depthDevice);
            c.segmentation = c.segmentation || {};
            c.segmentation.enable = parseBoolLike(args.enable_segmentation, c.segmentation.enable);
            c.segmentation.backend = _resolveSegmentationBackend(text(args.segmentation_backend, c.segmentation.backend));
            c.segmentation.sam2ModelSize = _resolveSam2ModelSize(text(args.sam2_model_size, c.segmentation.sam2ModelSize));
            c.segmentation.sam2CheckpointPath = text(
                args.sam2_checkpoint_path,
                c.segmentation.sam2CheckpointPath, true
            );
            c.segmentation.sam2TilingEnabled = parseBoolLike(
                args.sam2_tiling_enabled,
                c.segmentation.sam2TilingEnabled
            );
            c.segmentation.sam2TileSizePx = _parsePositiveIntOrNull(args.sam2_tile_size_px) || c.segmentation.sam2TileSizePx;
            c.segmentation.sam2OverlapPx =
                _parseNonNegativeIntOrNull(args.sam2_overlap_px) ?? c.segmentation.sam2OverlapPx;
            c.segmentation.sam2GlobalPassLongestSide =
                _parsePositiveIntOrNull(args.sam2_global_pass_longest_side) || c.segmentation.sam2GlobalPassLongestSide;
            c.segmentation.sam2MaxConcurrency =
                _parsePositiveIntOrNull(args.sam2_max_concurrency) || c.segmentation.sam2MaxConcurrency;
            c.segmentation.sam2PointsPerSide =
                _parsePositiveIntOrNull(args.sam2_points_per_side) || c.segmentation.sam2PointsPerSide;
            c.segmentation.sam2PointsPerBatch =
                _parsePositiveIntOrNull(args.sam2_points_per_batch) || c.segmentation.sam2PointsPerBatch;
            c.segmentation.sam2PredIouThresh =
                _parseProbabilityOrNull(args.sam2_pred_iou_thresh) ?? c.segmentation.sam2PredIouThresh;
            c.segmentation.sam2StabilityScoreThresh =
                _parseProbabilityOrNull(args.sam2_stability_score_thresh) ?? c.segmentation.sam2StabilityScoreThresh;
            c.segmentation.sam2CropNLayers =
                _parseNonNegativeIntOrNull(args.sam2_crop_n_layers) ?? c.segmentation.sam2CropNLayers;
            c.segmentation.strict = parseBoolLike(args.strict_segmentation, c.segmentation.strict);

            for (const [field, argument] of [
                ['materials', 'materials_v3'], ['pbr', 'pbr'], ['cache', 'cache_depth'], ['overwrite', 'overwrite'],
                ['enableV2', 'enable_v2'], ['saveFloatDepth', 'save_float_depth'], ['forceDepth', 'force_depth'],
                ['strictInputs', 'strict_inputs'], ['verifyImages', 'verify_images'],
                ['allowSemanticFallback', 'allow_semantic_fallback'], ['verbose', 'verbose'], ['quiet', 'quiet']
            ]) c.flags[field] = parseBoolLike(args[argument], c.flags[field]);
            _normalizeVerboseQuietFlags(c.flags, true);
            c.v2Preset = text(args.v2_preset, c.v2Preset);
            if ('v2_input_color' in args) c.v2InputColor = args.v2_input_color;

            c.outputBitDepth = Number(
                args.output_bit_depth ?? args.outputBitDepth ?? c.outputBitDepth
            ) === 8 ? 8 : 16;
            c.emits.runCard = parseBoolLike(args.emit_run_card, c.emits.runCard);
            c.emits.runCardVersion = _resolveRunCardVersion(text(args.run_card_version, c.emits.runCardVersion));
            c.emits.runCardIncludeProofs = parseBoolLike(
                args.run_card_include_proofs,
                c.emits.runCardIncludeProofs
            );

            c.gate = c.gate || {};
            c.gate.archiveIndex = text(
                args.archive_index,
                c.gate.archiveIndex, true
            );
            c.gate.manifestJsonl = text(
                args.manifest_jsonl,
                c.gate.manifestJsonl, true
            );
            c.licenses.nonCommercialOk = parseBoolLike(
                args.non_commercial_ok,
                c.licenses.nonCommercialOk
            );
            c.licenses.acceptApple = parseBoolLike(
                args.accept_apple_depth_pro_research_license,
                c.licenses.acceptApple
            );
            c.licenses.acceptResearchTools = parseBoolLike(
                args.accept_research_tools_license,
                c.licenses.acceptResearchTools
            );

            c.reconstruction = c.reconstruction || {};
            c.reconstruction.enable = parseBoolLike(
                args.enable_reconstruction,
                c.reconstruction.enable
            );
            c.reconstruction.groupingMode = _resolveGroupingMode(
                text(args.grouping_mode, c.reconstruction.groupingMode)
            );
            c.reconstruction.camerasSidecarPath = text(
                args.cameras_sidecar_path,
                c.reconstruction.camerasSidecarPath, true
            );
            c.reconstruction.iterations = _parsePositiveIntOrNull(
                args.reconstruction_iterations
            ) || c.reconstruction.iterations;
            c.reconstruction.tier = text(
                args.reconstruction_tier,
                c.reconstruction.tier, true
            );
            c.reconstruction.emitSceneDebugBundle = parseBoolLike(
                args.emit_scene_debug_bundle,
                c.reconstruction.emitSceneDebugBundle
            );

            c.raw = c.raw || {};
            c.raw.ingestMode = text(args.raw_ingest_mode, c.raw.ingestMode, true);
            c.raw.wbMode = text(args.raw_wb_mode, c.raw.wbMode, true);
            c.raw.demosaic = text(args.raw_demosaic, c.raw.demosaic, true);

            c.captioning = c.captioning || {};
            c.captioning.enableFastVlm = parseBoolLike(
                args.vlm_captioning_enabled,
                c.captioning.enableFastVlm
            );
            c.captioning.model = text(args.vlm_captioning_model, c.captioning.model, true);
            c.captioning.proxyFormat = _resolveVlmCaptioningProxyFormat(
                text(args.vlm_captioning_proxy_format, c.captioning.proxyFormat)
            );
            c.captioning.maxSidePx =
                _parsePositiveIntOrNull(args.vlm_captioning_max_side_px) || c.captioning.maxSidePx;
            c.captioning.timeoutSeconds =
                _parsePositiveIntOrNull(args.fastvlm_timeout_seconds) || c.captioning.timeoutSeconds;
            c.captioning.pythonExecutable = text(
                args.fastvlm_python_executable,
                c.captioning.pythonExecutable, true
            );
            c.captioning.mlxVlmDir = text(
                args.fastvlm_mlx_vlm_dir,
                c.captioning.mlxVlmDir, true
            );

            c.runtime = c.runtime || {};
            c.runtime.maxWorkers = _parsePositiveIntOrNull(args.max_workers) || '';
            c.runtime.maxWorkersMode = c.runtime.maxWorkers === '' ? 'auto' : 'fixed';
            c.runtime.maxGpuWorkers = _parsePositiveIntOrNull(args.max_gpu_workers) || '';
            c.runtime.maxGpuWorkersMode = c.runtime.maxGpuWorkers === '' ? 'auto' : 'fixed';
            c.runtime.logLevel = text(args.log_level, c.runtime.logLevel, true);
        host.migrate(c);
        validateFields(c, host.configTemplate?.() || state.config);
        return { pipeline, config: c };
    }
    async function importFile(file, expected = host.snapshot()) {
        const intent = ++readIntent;
        if (!file) return false;
        let next;
        try {
            if (!Number.isFinite(file.size) || file.size > MAX_CONFIG_IMPORT_BYTES || file.size < 0) throw new Error();
            next = candidate(JSON.parse(await file.text()));
        } catch {
            host.notify('Invalid JSON file.', 'error');
            return false;
        }
        const current = () => intent === readIntent && expected && expected === host.snapshot();
        if (!current()) return false;
        if (!(await host.confirmReplacement()) || !current()) return false;
        host.commit(next);
        return true;
    }
    return { importFile };
}
