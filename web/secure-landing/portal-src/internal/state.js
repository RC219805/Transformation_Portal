export function createPortalConfigState() {
  return {
    preset: "premium",
    inputDir: "./input_images",
    outputDir: "./output/lux_depth_v3_apex",
    qualityTier: "premium",
    depthBackend: "da3",
    depthDevice: "cpu",
    segmentation: {
      enable: false,
      backend: "stub",
      sam2ModelSize: "base",
      sam2CheckpointPath: "",
      strict: false
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
    emits: {
      master16: true,
      upscaled16: true,
      marketing: false,
      report: true,
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
    backend_catalog: {}
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
    debugBundleGuardrailSeen: false,
    buildStep: 1,
    lastOverlayTrigger: null,
    lastSelectedJobId: "",
    disclosurePrefs: {
      advanced: null,
      governance: null,
      reconstruction: null,
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
    features: {
      apiKeyInput: false,
      directDebug: false
    }
  };
}

export function createPortalBootstrapState(now = Date.now()) {
  const lastTransitionAt = Number.isFinite(Number(now)) ? Number(now) : Date.now();
  return {
    status: "pending",
    lastErrorReason: "",
    lastHttpStatus: 0,
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
