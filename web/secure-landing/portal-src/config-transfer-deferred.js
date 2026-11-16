// Configuration transfer is loaded only after an explicit import action.
export const MAX_CONFIG_IMPORT_BYTES = 1024 * 1024;

export function createDeferredConfigTransferApi(host) {
    const { state, portalInternals, _normalizeVerboseQuietFlags, _parseNonNegativeIntOrNull, _parsePositiveIntOrNull, _parseProbabilityOrNull, _resolveDa3ModelKey, _resolveDepthBackend, _resolveGroupingMode, _resolveQualityTier, _resolveRunCardVersion, _resolveSam2ModelSize, _resolveSegmentationBackend, _resolveVlmCaptioningProxyFormat, _textOrFallback, parseBoolLike } = host;
    let readIntent = 0;
    function candidate(data) {
        if (!data || typeof data !== 'object' || Array.isArray(data)
            || (data.schema !== undefined && data.schema !== 'tp.portal.export.v1')
            || !data.args || typeof data.args !== 'object' || Array.isArray(data.args)) {
            throw new Error('Invalid configuration shape.');
        }
        const pipeline = data.pipeline === undefined ? state.pipeline : data.pipeline;
        if (typeof pipeline !== 'string' || !host.supportedPipelines.has(pipeline)) throw new Error('Unsupported pipeline.');
        const c = JSON.parse(JSON.stringify(state.config));
        data = JSON.parse(JSON.stringify(data));
        host.migrate(data.args);
            if (pipeline === 'lux-depth') {
                c.photographyUnified = portalInternals.unifiedPhotographyConfigFromArgs(data.args, c.photographyUnified);
            } else if (pipeline === 'lux-depth-v6') {
                c.photographyV6 = portalInternals.photographyV6ConfigFromArgs(data.args, c.photographyV6);
            } else if (pipeline === 'lux-depth-v5') {
                c.photography = portalInternals.photographyConfigFromArgs(data.args, c.photography);
            }
            if ('emit_master16' in data.args) c.emit_master16 = data.args.emit_master16;
            c.inputDir = data.args.archive_root || data.args.input_dir || c.inputDir;
            c.outputDir = data.args.output_dir || c.outputDir;
            c.preset = data.args.preset || c.preset;
            c.qualityTier = _resolveQualityTier(data.args.quality_tier || c.qualityTier);
            c.depthBackend = _resolveDepthBackend(data.args.depth_backend || c.depthBackend);
            c.modelKey = _resolveDa3ModelKey(data.args.model_key || c.modelKey);
            c.depthDevice = data.args.depth_device || c.depthDevice;
            c.segmentation = c.segmentation || {};
            c.segmentation.enable = parseBoolLike(data.args.enable_segmentation, c.segmentation.enable);
            c.segmentation.backend = _resolveSegmentationBackend(data.args.segmentation_backend || c.segmentation.backend);
            c.segmentation.sam2ModelSize = _resolveSam2ModelSize(data.args.sam2_model_size || c.segmentation.sam2ModelSize);
            c.segmentation.sam2CheckpointPath = _textOrFallback(
                data.args.sam2_checkpoint_path,
                c.segmentation.sam2CheckpointPath
            );
            c.segmentation.sam2TilingEnabled = parseBoolLike(
                data.args.sam2_tiling_enabled,
                c.segmentation.sam2TilingEnabled
            );
            c.segmentation.sam2TileSizePx = _parsePositiveIntOrNull(data.args.sam2_tile_size_px) || c.segmentation.sam2TileSizePx;
            c.segmentation.sam2OverlapPx =
                _parseNonNegativeIntOrNull(data.args.sam2_overlap_px) ?? c.segmentation.sam2OverlapPx;
            c.segmentation.sam2GlobalPassLongestSide =
                _parsePositiveIntOrNull(data.args.sam2_global_pass_longest_side) || c.segmentation.sam2GlobalPassLongestSide;
            c.segmentation.sam2MaxConcurrency =
                _parsePositiveIntOrNull(data.args.sam2_max_concurrency) || c.segmentation.sam2MaxConcurrency;
            c.segmentation.sam2PointsPerSide =
                _parsePositiveIntOrNull(data.args.sam2_points_per_side) || c.segmentation.sam2PointsPerSide;
            c.segmentation.sam2PointsPerBatch =
                _parsePositiveIntOrNull(data.args.sam2_points_per_batch) || c.segmentation.sam2PointsPerBatch;
            c.segmentation.sam2PredIouThresh =
                _parseProbabilityOrNull(data.args.sam2_pred_iou_thresh) ?? c.segmentation.sam2PredIouThresh;
            c.segmentation.sam2StabilityScoreThresh =
                _parseProbabilityOrNull(data.args.sam2_stability_score_thresh) ?? c.segmentation.sam2StabilityScoreThresh;
            c.segmentation.sam2CropNLayers =
                _parseNonNegativeIntOrNull(data.args.sam2_crop_n_layers) ?? c.segmentation.sam2CropNLayers;
            c.segmentation.strict = parseBoolLike(data.args.strict_segmentation, c.segmentation.strict);

            c.flags.materials = parseBoolLike(data.args.materials_v3, c.flags.materials);
            c.flags.pbr = parseBoolLike(data.args.pbr, c.flags.pbr);
            c.flags.cache = parseBoolLike(data.args.cache_depth, c.flags.cache);
            c.flags.overwrite = parseBoolLike(data.args.overwrite, c.flags.overwrite);
            c.flags.enableV2 = parseBoolLike(data.args.enable_v2, c.flags.enableV2);
            c.flags.saveFloatDepth = parseBoolLike(data.args.save_float_depth, c.flags.saveFloatDepth);
            c.flags.forceDepth = parseBoolLike(data.args.force_depth, c.flags.forceDepth);
            c.flags.strictInputs = parseBoolLike(data.args.strict_inputs, c.flags.strictInputs);
            c.flags.verifyImages = parseBoolLike(data.args.verify_images, c.flags.verifyImages);
            c.flags.allowSemanticFallback = parseBoolLike(
                data.args.allow_semantic_fallback,
                c.flags.allowSemanticFallback
            );
            c.flags.verbose = parseBoolLike(data.args.verbose, c.flags.verbose);
            c.flags.quiet = parseBoolLike(data.args.quiet, c.flags.quiet);
            _normalizeVerboseQuietFlags(c.flags, true);
            c.v2Preset = data.args.v2_preset || c.v2Preset;
            if ('v2_input_color' in data.args) c.v2InputColor = data.args.v2_input_color;

            c.outputBitDepth = Number(
                data.args.output_bit_depth ?? data.args.outputBitDepth ?? c.outputBitDepth
            ) === 8 ? 8 : 16;
            c.emits.runCard = parseBoolLike(data.args.emit_run_card, c.emits.runCard);
            c.emits.runCardVersion = _resolveRunCardVersion(data.args.run_card_version || c.emits.runCardVersion);
            c.emits.runCardIncludeProofs = parseBoolLike(
                data.args.run_card_include_proofs,
                c.emits.runCardIncludeProofs
            );

            c.gate = c.gate || {};
            c.gate.archiveIndex = _textOrFallback(
                data.args.archive_index,
                c.gate.archiveIndex
            );
            c.gate.manifestJsonl = _textOrFallback(
                data.args.manifest_jsonl,
                c.gate.manifestJsonl
            );
            c.licenses.nonCommercialOk = parseBoolLike(
                data.args.non_commercial_ok,
                c.licenses.nonCommercialOk
            );
            c.licenses.acceptApple = parseBoolLike(
                data.args.accept_apple_depth_pro_research_license,
                c.licenses.acceptApple
            );
            c.licenses.acceptResearchTools = parseBoolLike(
                data.args.accept_research_tools_license,
                c.licenses.acceptResearchTools
            );

            c.reconstruction = c.reconstruction || {};
            c.reconstruction.enable = parseBoolLike(
                data.args.enable_reconstruction,
                c.reconstruction.enable
            );
            c.reconstruction.groupingMode = _resolveGroupingMode(
                data.args.grouping_mode || c.reconstruction.groupingMode
            );
            c.reconstruction.camerasSidecarPath = _textOrFallback(
                data.args.cameras_sidecar_path,
                c.reconstruction.camerasSidecarPath
            );
            c.reconstruction.iterations = _parsePositiveIntOrNull(
                data.args.reconstruction_iterations
            ) || c.reconstruction.iterations;
            c.reconstruction.tier = _textOrFallback(
                data.args.reconstruction_tier,
                c.reconstruction.tier
            );
            c.reconstruction.emitSceneDebugBundle = parseBoolLike(
                data.args.emit_scene_debug_bundle,
                c.reconstruction.emitSceneDebugBundle
            );

            c.raw = c.raw || {};
            c.raw.ingestMode = _textOrFallback(data.args.raw_ingest_mode, c.raw.ingestMode);
            c.raw.wbMode = _textOrFallback(data.args.raw_wb_mode, c.raw.wbMode);
            c.raw.demosaic = _textOrFallback(data.args.raw_demosaic, c.raw.demosaic);

            c.captioning = c.captioning || {};
            c.captioning.enableFastVlm = parseBoolLike(
                data.args.vlm_captioning_enabled,
                c.captioning.enableFastVlm
            );
            c.captioning.model = _textOrFallback(data.args.vlm_captioning_model, c.captioning.model);
            c.captioning.proxyFormat = _resolveVlmCaptioningProxyFormat(
                data.args.vlm_captioning_proxy_format || c.captioning.proxyFormat
            );
            c.captioning.maxSidePx =
                _parsePositiveIntOrNull(data.args.vlm_captioning_max_side_px) || c.captioning.maxSidePx;
            c.captioning.timeoutSeconds =
                _parsePositiveIntOrNull(data.args.fastvlm_timeout_seconds) || c.captioning.timeoutSeconds;
            c.captioning.pythonExecutable = _textOrFallback(
                data.args.fastvlm_python_executable,
                c.captioning.pythonExecutable
            );
            c.captioning.mlxVlmDir = _textOrFallback(
                data.args.fastvlm_mlx_vlm_dir,
                c.captioning.mlxVlmDir
            );

            c.runtime = c.runtime || {};
            c.runtime.maxWorkers = _parsePositiveIntOrNull(data.args.max_workers) || '';
            c.runtime.maxWorkersMode = c.runtime.maxWorkers === '' ? 'auto' : 'fixed';
            c.runtime.maxGpuWorkers = _parsePositiveIntOrNull(data.args.max_gpu_workers) || '';
            c.runtime.maxGpuWorkersMode = c.runtime.maxGpuWorkers === '' ? 'auto' : 'fixed';
            c.runtime.logLevel = _textOrFallback(data.args.log_level, c.runtime.logLevel);
        return { pipeline, config: c };
    }
    async function importFile(file, expected = host.snapshot()) {
        const intent = ++readIntent;
        if (!file) return false;
        let next;
        try {
            if (!Number.isFinite(file.size) || file.size > MAX_CONFIG_IMPORT_BYTES || file.size < 0) throw new Error('Configuration is too large.');
            next = candidate(JSON.parse(await file.text()));
        } catch {
            host.notify('Invalid JSON file.', 'error');
            return false;
        }
        const current = () => intent === readIntent && expected && expected === host.snapshot();
        if (!current()) return false;
        const allowed = await host.confirmReplacement();
        if (!allowed || !current()) return false;
        host.commit(next);
        return true;
    }
    return { importFile };
}
