export function createPortalConfigState() {
  return {
    preset: "premium",
    inputDir: "./input_images",
    outputDir: "./output/lux_depth_v3_apex",
    qualityTier: "premium",
    depthBackend: "da3",
    modelKey: "da3-metric",
    depthDevice: "cpu",
    segmentation: {
      enable: true,
      backend: "efficientsam",
      sam2ModelSize: "base",
      sam2CheckpointPath: "",
      sam2TilingEnabled: false,
      sam2TileSizePx: 1536,
      sam2OverlapPx: 256,
      sam2GlobalPassLongestSide: 1280,
      sam2MaxConcurrency: 1,
      sam2PointsPerSide: 32,
      sam2PointsPerBatch: 64,
      sam2PredIouThresh: 0.88,
      sam2StabilityScoreThresh: 0.85,
      sam2CropNLayers: 1,
      strict: true
    },
    flags: {
      materials: true,
      pbr: true,
      cache: true,
      overwrite: false,
      enableV2: false,
      saveFloatDepth: false,
      forceDepth: false,
      strictInputs: false,
      verifyImages: false,
      allowSemanticFallback: false,
      verbose: false,
      quiet: false
    },
    v2Preset: "default",
    outputBitDepth: 16,
    emits: {
      runCard: true,
      runCardVersion: "v1",
      runCardIncludeProofs: false
    },
    gate: { archiveIndex: "", manifestJsonl: "" },
    licenses: { nonCommercialOk: false, acceptApple: false, acceptResearchTools: false },
    reconstruction: {
      enable: false,
      groupingMode: "single",
      camerasSidecarPath: "",
      iterations: 1000,
      tier: "apex_research",
      emitSceneDebugBundle: false
    },
    raw: {
      ingestMode: "auto",
      wbMode: "camera",
      demosaic: "AHD"
    },
    captioning: {
      enableFastVlm: false,
      model: "default",
      proxyFormat: "png",
      maxSidePx: 1600,
      timeoutSeconds: 180,
      pythonExecutable: "",
      mlxVlmDir: ""
    },
    runtime: {
      maxWorkersMode: "auto",
      maxWorkers: "",
      maxGpuWorkersMode: "auto",
      maxGpuWorkers: "",
      logLevel: ""
    }
  };
}

export function createPortalMetadataState() {
  return {
    pipeline: "",
    fields: {},
    estimate_bands: {},
    debug_bundle_policy: {},
    advanced_sections: [],
    backend_catalog: {},
    model_catalog: {}
  };
}

export function createPortalPreviewState() {
  return {
    pipeline: "",
    requestKey: "",
    status: "idle",
    field_errors: [],
    field_warnings: [],
    inactive_fields: [],
    normalized_args: {},
    execution_args: {},
    submitted_args: {},
    readiness: null,
    estimate_summary: null,
    debug_bundle_summary: null,
    captioning_summary: null,
    next_best_action: null,
    argv_preview: "",
    error: "",
    error_reason: "",
    error_status: 0
  };
}

export function createPortalUiState() {
  return {
    debugBundleAcknowledged: false,
    effectiveConfigOpen: false,
    stagedUpload: {
      busy: false,
      progressPercent: 0,
      status: "idle",
      summary: "",
      error: "",
      lastBatchId: "",
      fileCount: 0,
      totalBytes: 0,
      receipt: null
    },
    artifactViewer: {
      open: false,
      jobId: "",
      artifactPath: "",
      zoomPercent: 100
    },
    debugBundleGuardrailSeen: false,
    buildStep: 1,
    lastOverlayTrigger: null,
    lastSelectedJobId: "",
    disclosurePrefs: {
      advanced: null,
      governance: null,
      reconstruction: null,
      captioning: null,
      dispatchTools: false
    }
  };
}

export function createPortalReadinessState() {
  return {
    server: {},
    pipelines: {}
  };
}

export function createPortalAuthState() {
  return {
    mode: "managed_unavailable",
    csrfToken: "",
    actor: null,
    logoutPending: false,
    features: {
      apiKeyInput: false,
      directDebug: false,
      artifactViewerModal: false,
      reviewSurfaceDeferred: false,
      stagedUploads: false,
      rumTelemetry: false,
      fastVlmCaptioning: false
    }
  };
}

export function createPortalBootstrapState(now = Date.now()) {
  const lastTransitionAt = Number.isFinite(Number(now)) ? Number(now) : Date.now();
  return {
    status: "pending",
    lastErrorReason: "",
    lastHttpStatus: 0,
    lastRateLimitHint: null,
    activeController: null,
    activeTimeoutId: null,
    lastTransitionAt,
    lastHealthEndpointPath: "",
    pendingOnlineFollowup: false,
    onlineFollowupComplete: false,
    retry: {
      timer: null,
      attempt: 0,
      deadlineAt: 0,
      lastDelayMs: 0,
      lastReason: "",
      lastHttpStatus: 0,
      lastAttemptAt: 0,
      lastOutcome: "",
      lastEventAt: 0
    }
  };
}

export function createPortalLastDiagnosticsState() {
  return {
    warnings: [],
    expectedOutputs: [],
    healthState: "good",
    healthLabel: "good"
  };
}
