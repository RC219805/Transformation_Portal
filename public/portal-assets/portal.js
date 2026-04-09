const API_BASE = '';
const STORAGE_KEY = 'tp_orchestrator_profiles_final';
const API_KEY_STORAGE_KEY = 'tp_api_key';
const THEME_STORAGE_KEY = 'tp_theme';
const THEME_STORAGE_VERSION_KEY = 'tp_theme_version';
const THEME_STORAGE_VERSION = '2';
const THEME_PREFERENCES = Object.freeze(['system', 'dark', 'light']);
const MAX_JOB_LOG_LINES = 1200;
const BOOTSTRAP_TIMEOUT_MS = 3500;
const BOOTSTRAP_RETRY_BASE_DELAY_MS = 1000;
const BOOTSTRAP_RETRY_MAX_DELAY_MS = 12000;
const BOOTSTRAP_RETRY_MAX_EXPONENT = 4;
const BOOTSTRAP_RETRY_JITTER_MS = 400;
const BOOTSTRAP_RETRY_MAX_ATTEMPTS = 4;
const BOOTSTRAP_RETRY_MAX_WINDOW_MS = 60000;
const BOOTSTRAP_RETRIABLE_HTTP_STATUSES = new Set([500, 502, 503, 504]);
const HEALTH_CHECK_INTERVAL_MS = 15000;
const HEALTH_CHECK_TIMEOUT_MS = 2000;
const HEALTH_CHECK_MIN_GAP_MS = 10000;
const SSE_RECONNECT_BASE_DELAY_MS = 1000;
const SSE_RECONNECT_MAX_DELAY_MS = 30000;
const SSE_RECONNECT_MAX_EXPONENT = 5;
const SSE_RECONNECT_JITTER_MS = 250;
const SSE_STALL_CHECK_INTERVAL_MS = 10000;
const SSE_STALL_THRESHOLD_MS = 45000;
const CONFIG_PREVIEW_DEBOUNCE_MS = 250;
const CONFIG_PREVIEW_SUPPORTED_PIPELINES = new Set([
    'lux-depth-v3',
    'archive-gate-a',
    'archive-gate-b',
    'archive-gate-c'
]);
const EVENT_SOURCE_READY_STATE_CONNECTING = 0;
const EVENT_SOURCE_READY_STATE_OPEN = 1;
const EVENT_SOURCE_READY_STATE_CLOSED = 2;
const TIMELINE_PROGRESS_CHECKPOINTS = [5, 25, 50, 75, 100];
const SAFE_JOB_STATES = new Set(['queued', 'running', 'succeeded', 'partial', 'failed', 'canceled', 'ready', 'offline']);
const SAFE_HTTP_METHODS = new Set(['GET', 'HEAD', 'OPTIONS']);

let queueRenderScheduled = false;
let queuedReviewSurfaceRefresh = false;
let healthPollIntervalId = null;
let sseWatchdogIntervalId = null;
let healthCheckInFlight = false;
let lastHealthCheckAt = 0;
let configPreviewTimerId = null;

const state = {
    pipeline: 'lux-depth-v3',
    config: {
        preset: 'premium',
        inputDir: './input_images',
        outputDir: './output/lux_depth_v3_apex',
        qualityTier: 'premium',
        depthBackend: 'da3',
        depthDevice: 'cpu',
        segmentation: {
            enable: false,
            backend: 'stub',
            sam2ModelSize: 'base',
            sam2CheckpointPath: '',
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
        v2Preset: 'default',
        emits: {
            master16: true, upscaled16: true, marketing: false, report: true, runCard: true
        },
        gate: { archiveIndex: '', manifestJsonl: '' },
        licenses: { nonCommercialOk: false, acceptApple: false, acceptResearchTools: false },
        reconstruction: {
            enable: false,
            groupingMode: 'single',
            camerasSidecarPath: '',
            iterations: 1000,
            tier: 'apex_research',
            emitSceneDebugBundle: false
        },
        raw: {
            ingestMode: 'auto',
            wbMode: 'camera',
            demosaic: 'AHD'
        },
        runtime: {
            maxWorkersMode: 'auto',
            maxWorkers: '',
            maxGpuWorkersMode: 'auto',
            maxGpuWorkers: '',
            logLevel: ''
        }
    },
    jobs: [],
    jobsLoadStatus: 'pending',
    selectedJobId: null,
    currentView: 'overview',
    inspectorTab: 'overview',
    artifactUi: {
        selectedByJob: {},
        compareByJob: {}
    },
    backendOk: false,
    metadata: {
        pipeline: '',
        fields: {},
        estimate_bands: {},
        debug_bundle_policy: {},
        advanced_sections: []
    },
    preview: {
        pipeline: '',
        requestKey: '',
        status: 'idle',
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
        argv_preview: '',
        error: '',
        error_reason: '',
        error_status: 0
    },
    portalUi: {
        debugBundleAcknowledged: false,
        effectiveConfigOpen: false,
        debugBundleGuardrailSeen: false,
        buildStep: 1,
        lastOverlayTrigger: null,
        lastSelectedJobId: '',
        disclosurePrefs: {
            advanced: null,
            governance: null,
            reconstruction: null,
            dispatchTools: false,
        }
    },
    readiness: {
        server: {},
        pipelines: {}
    },
    themePreference: 'system',
    theme: 'light',
    presetsByPipeline: {},
    auth: {
        mode: 'managed_unavailable',
        csrfToken: '',
        actor: null,
        features: {
            apiKeyInput: false,
            directDebug: false
        }
    },
    bootstrap: {
        status: 'pending',
        lastErrorReason: '',
        lastHttpStatus: 0,
        activeController: null,
        activeTimeoutId: null,
        lastTransitionAt: Date.now(),
        lastHealthEndpointPath: '',
        pendingOnlineFollowup: false,
        onlineFollowupComplete: false,
        retry: {
            timer: null,
            attempt: 0,
            deadlineAt: 0,
            lastDelayMs: 0,
            lastReason: '',
            lastHttpStatus: 0,
            lastAttemptAt: 0,
            lastOutcome: '',
            lastEventAt: 0
        }
    },
    lastDiagnostics: {
        warnings: [],
        expectedOutputs: [],
        healthState: 'good',
        healthLabel: 'good'
    }
};

const els = {
    overviewShell: document.getElementById('overview-shell'),
    missionShell: document.getElementById('mission-shell'),
    missionShellContent: document.getElementById('missionShellContent'),
    missionShellSkeletonState: document.getElementById('missionShellSkeletonState'),
    intelligenceShell: document.getElementById('intelligence-shell'),
    intelligenceShellContent: document.getElementById('intelligenceShellContent'),
    intelligenceShellSkeletonState: document.getElementById('intelligenceShellSkeletonState'),
    consoleGrid: document.getElementById('console-grid'),
    consoleContextShell: document.getElementById('console-context-shell'),
    consoleViewTitle: document.getElementById('consoleViewTitle'),
    consoleViewSummary: document.getElementById('consoleViewSummary'),
    consoleViewMeta: document.getElementById('consoleViewMeta'),
    consoleContextRibbon: document.getElementById('consoleContextRibbon'),
    contextRibbonJob: document.getElementById('contextRibbonJob'),
    contextRibbonJobMeta: document.getElementById('contextRibbonJobMeta'),
    contextRibbonState: document.getElementById('contextRibbonState'),
    contextRibbonFreshness: document.getElementById('contextRibbonFreshness'),
    contextRibbonArtifact: document.getElementById('contextRibbonArtifact'),
    contextRibbonArtifactMeta: document.getElementById('contextRibbonArtifactMeta'),
    contextRibbonCompare: document.getElementById('contextRibbonCompare'),
    contextRibbonCompareMeta: document.getElementById('contextRibbonCompareMeta'),
    heroRunBtn: document.getElementById('heroRunBtn'),
    resumeDraftBtn: document.getElementById('resumeDraftBtn'),
    heroExportBtn: document.getElementById('heroExportBtn'),
    refreshHealthBtn: document.getElementById('refreshHealthBtn'),
    heroPipelineValue: document.getElementById('heroPipelineValue'),
    heroPresetValue: document.getElementById('heroPresetValue'),
    heroModeValue: document.getElementById('heroModeValue'),
    heroQueueValue: document.getElementById('heroQueueValue'),
    capabilityChips: document.getElementById('capabilityChips'),
    presetHeadline: document.getElementById('presetHeadline'),
    presetStabilityBadge: document.getElementById('presetStabilityBadge'),
    backendModeBadge: document.getElementById('backendModeBadge'),
    presetDescription: document.getElementById('presetDescription'),
    presetBuilderHint: document.getElementById('presetBuilderHint'),
    presetBuilderShell: document.getElementById('presetBuilderShell'),
    flagsShell: document.getElementById('flags-shell'),
    heroInputDir: document.getElementById('heroInputDir'),
    heroOutputDir: document.getElementById('heroOutputDir'),
    heroReadinessLabel: document.getElementById('heroReadinessLabel'),
    heroWarningCount: document.getElementById('heroWarningCount'),
    governanceBannerTitle: document.getElementById('governanceBannerTitle'),
    governanceBannerBody: document.getElementById('governanceBannerBody'),
    governanceChecklist: document.getElementById('governanceChecklist'),
    buildStepTitle: document.getElementById('buildStepTitle'),
    buildStepSummary: document.getElementById('buildStepSummary'),
    buildStepTabs: document.getElementById('buildStepTabs'),
    buildStepBackBtn: document.getElementById('buildStepBackBtn'),
    buildStepNextBtn: document.getElementById('buildStepNextBtn'),
    buildStepTab1: document.getElementById('buildStepTab1'),
    buildStepTab2: document.getElementById('buildStepTab2'),
    buildStepTab3: document.getElementById('buildStepTab3'),
    buildStepTab4: document.getElementById('buildStepTab4'),

    pipelineSelect: document.getElementById('pipelineSelect'),
    presetSelect: document.getElementById('presetSelect'),
    profileSelect: document.getElementById('profileSelect'),
    saveProfileBtn: document.getElementById('saveProfileBtn'),
    apiKeySection: document.getElementById('apiKeySection'),
    apiKeyInput: document.getElementById('apiKeyInput'),
    authModeBadge: document.getElementById('authModeBadge'),
    apiKeyManagedHint: document.getElementById('apiKeyManagedHint'),

    inputDir: document.getElementById('inputDir'),
    outputDir: document.getElementById('outputDir'),
    inputDirStatus: document.getElementById('inputDirStatus'),
    outputDirStatus: document.getElementById('outputDirStatus'),
    archiveCanonicalCommand: document.getElementById('archiveCanonicalCommand'),
    archiveCanonicalCommandHint: document.getElementById('archiveCanonicalCommandHint'),
    archiveIndexField: document.getElementById('archiveIndexField'),
    archiveIndexPath: document.getElementById('archiveIndexPath'),
    archiveIndexStatus: document.getElementById('archiveIndexStatus'),
    rightsManifestField: document.getElementById('rightsManifestField'),
    rightsManifestPath: document.getElementById('rightsManifestPath'),
    rightsManifestStatus: document.getElementById('rightsManifestStatus'),
    qualityTier: document.getElementById('qualityTier'),
    depthBackend: document.getElementById('depthBackend'),
    depthDevice: document.getElementById('depthDevice'),
    segmentationBackendField: document.getElementById('segmentationBackendField'),
    sam2ModelSizeField: document.getElementById('sam2ModelSizeField'),
    strictSegmentationField: document.getElementById('strictSegmentationField'),
    sam2CheckpointField: document.getElementById('sam2CheckpointField'),
    segmentationApplicabilityHint: document.getElementById('segmentationApplicabilityHint'),
    segmentation: {
        enable: document.getElementById('enableSegmentation'),
        backend: document.getElementById('segmentationBackend'),
        sam2ModelSize: document.getElementById('sam2ModelSize'),
        sam2CheckpointPath: document.getElementById('sam2CheckpointPath'),
        strict: document.getElementById('strictSegmentation')
    },

    flags: {
        materials: document.getElementById('flagMaterials'),
        pbr: document.getElementById('flagPBR'),
        cache: document.getElementById('flagCache'),
        overwrite: document.getElementById('flagOverwrite'),
        enableV2: document.getElementById('flagEnableV2'),
        saveFloatDepth: document.getElementById('saveFloatDepth'),
        forceDepth: document.getElementById('forceDepth'),
        strictInputs: document.getElementById('strictInputs'),
        verifyImages: document.getElementById('verifyImages'),
        allowSemanticFallback: document.getElementById('allowSemanticFallback'),
        verbose: document.getElementById('verboseFlag'),
        quiet: document.getElementById('quietFlag')
    },
    v2Preset: document.getElementById('v2Preset'),
    v2PresetField: document.getElementById('v2PresetField'),

    emits: {
        master16: document.getElementById('emitMaster16'),
        upscaled16: document.getElementById('emitUpscaled16'),
        marketing: document.getElementById('emitMarketing'),
        report: document.getElementById('emitReport'),
        runCard: document.getElementById('emitRunCard')
    },

    licenses: {
        nonCommercialOk: document.getElementById('licenseNonCommercial'),
        acceptApple: document.getElementById('licenseApple'),
        acceptResearchTools: document.getElementById('licenseResearchTools')
    },
    licenseNonCommercialField: document.getElementById('licenseNonCommercialField'),
    licenseAppleField: document.getElementById('licenseAppleField'),
    licenseResearchToolsField: document.getElementById('licenseResearchToolsField'),
    governanceDetailsHint: document.getElementById('governanceDetailsHint'),
    reconstruction: {
        enable: document.getElementById('enableReconstruction'),
        groupingMode: document.getElementById('groupingMode'),
        camerasSidecarPath: document.getElementById('camerasSidecarPath'),
        iterations: document.getElementById('reconstructionIterations'),
        tier: document.getElementById('reconstructionTier'),
        emitSceneDebugBundle: document.getElementById('emitSceneDebugBundle'),
        groupingModeStatus: document.getElementById('groupingModeStatus'),
        iterationsStatus: document.getElementById('reconstructionIterationsStatus'),
        camerasSidecarStatus: document.getElementById('camerasSidecarStatus'),
        tierStatus: document.getElementById('reconstructionTierStatus')
    },
    reconstructionConfigFields: document.getElementById('reconstructionConfigFields'),
    runtimeTuningFields: document.getElementById('runtimeTuningFields'),
    reconstructionDetailsHint: document.getElementById('reconstructionDetailsHint'),
    reconstructionSummaryHint: document.getElementById('reconstructionSummaryHint'),
    openEffectiveConfigBtn: document.getElementById('openEffectiveConfigBtn'),
    effectiveConfigBtn: document.getElementById('effectiveConfigBtn'),
    summaryReconstructionState: document.getElementById('summaryReconstructionState'),
    summaryRuntimeWorkers: document.getElementById('summaryRuntimeWorkers'),
    summaryRawIngest: document.getElementById('summaryRawIngest'),
    summaryDebugBundle: document.getElementById('summaryDebugBundle'),
    summaryPreviewState: document.getElementById('summaryPreviewState'),
    estimateRuntimeBand: document.getElementById('estimateRuntimeBand'),
    estimateGpuBand: document.getElementById('estimateGpuBand'),
    estimateResearchRisk: document.getElementById('estimateResearchRisk'),
    estimateSummaryLabel: document.getElementById('estimateSummaryLabel'),
    debugBundleGuardrail: document.getElementById('debugBundleGuardrail'),
    debugBundleDestination: document.getElementById('debugBundleDestination'),
    debugBundleSensitivity: document.getElementById('debugBundleSensitivity'),
    debugBundleAcknowledge: document.getElementById('debugBundleAcknowledge'),
    debugBundleAcknowledgeHint: document.getElementById('debugBundleAcknowledgeHint'),
    raw: {
        ingestMode: document.getElementById('rawIngestMode'),
        wbMode: document.getElementById('rawWbMode'),
        demosaic: document.getElementById('rawDemosaic'),
        wbModeBadge: document.getElementById('rawWbModeBadge'),
        wbModeHint: document.getElementById('rawWbModeHint'),
        demosaicBadge: document.getElementById('rawDemosaicBadge'),
        demosaicHint: document.getElementById('rawDemosaicHint'),
        ingestModeStatus: document.getElementById('rawIngestModeStatus')
    },
    runtime: {
        maxWorkersMode: document.getElementById('maxWorkersMode'),
        maxWorkers: document.getElementById('maxWorkers'),
        maxWorkersValueField: document.getElementById('maxWorkersValueField'),
        maxWorkersStatus: document.getElementById('maxWorkersStatus'),
        maxGpuWorkersMode: document.getElementById('maxGpuWorkersMode'),
        maxGpuWorkers: document.getElementById('maxGpuWorkers'),
        maxGpuWorkersValueField: document.getElementById('maxGpuWorkersValueField'),
        maxGpuWorkersStatus: document.getElementById('maxGpuWorkersStatus'),
        logLevel: document.getElementById('logLevel'),
        logLevelStatus: document.getElementById('logLevelStatus')
    },

    fieldsLuxDepth: document.getElementById('fieldsLuxDepth'),
    fieldsArchiveGate: document.getElementById('fieldsArchiveGate'),
    advancedFlagsDetails: document.getElementById('advancedFlagsDetails'),
    governanceDetails: document.getElementById('governanceDetails'),
    reconstructionDetails: document.getElementById('reconstructionDetails'),

    cliPreview: document.getElementById('cliPreview'),
    copyCliBtn: document.getElementById('copyCliBtn'),
    importBtn: document.getElementById('importBtn'),
    exportBtn: document.getElementById('exportBtn'),
    fileInput: document.getElementById('fileInput'),
    runJobBtn: document.getElementById('runJobBtn'),
    dispatchToolsDetails: document.getElementById('dispatchToolsDetails'),
    preRunWarnings: document.getElementById('preRunWarnings'),
    preRunWarningsEmpty: document.getElementById('preRunWarningsEmpty'),
    expectedOutputsList: document.getElementById('expectedOutputsList'),
    datasetHealthIndicator: document.getElementById('datasetHealthIndicator'),
    datasetHealthText: document.getElementById('datasetHealthText'),
    nextBestActionLabel: document.getElementById('nextBestActionLabel'),
    nextBestActionDetail: document.getElementById('nextBestActionDetail'),
    nextBestActionTone: document.getElementById('nextBestActionTone'),

    buildShell: document.getElementById('build-shell'),
    profileShell: document.getElementById('profile-shell'),
    profileShellContent: document.getElementById('profileShellContent'),
    profileShellSkeletonState: document.getElementById('profileShellSkeletonState'),
    buildStepperShell: document.getElementById('buildStepperShell'),
    buildStepperShellContent: document.getElementById('buildStepperShellContent'),
    buildStepperSkeletonState: document.getElementById('buildStepperSkeletonState'),
    governanceShell: document.getElementById('governance-shell'),
    parametersShell: document.getElementById('parameters-shell'),
    parametersShellContent: document.getElementById('parametersShellContent'),
    parametersShellSkeletonState: document.getElementById('parametersShellSkeletonState'),
    jobsShell: document.getElementById('jobs-shell'),
    selectedJobShell: document.getElementById('selected-job-shell'),
    selectedJobShellContent: document.getElementById('selectedJobShellContent'),
    selectedJobSkeletonState: document.getElementById('selectedJobSkeletonState'),
    queueShell: document.getElementById('queue-shell'),
    queueSkeletonState: document.getElementById('queueSkeletonState'),
    jobList: document.getElementById('jobList'),
    emptyQueueState: document.getElementById('emptyQueueState'),
    queueCount: document.getElementById('queueCount'),
    selectedJobStateBadge: document.getElementById('selectedJobStateBadge'),
    selectedJobIdLabel: document.getElementById('selectedJobIdLabel'),
    selectedJobPipelineLabel: document.getElementById('selectedJobPipelineLabel'),
    selectedJobArtifactCount: document.getElementById('selectedJobArtifactCount'),
    selectedJobStreamStatus: document.getElementById('selectedJobStreamStatus'),
    selectedJobProgressText: document.getElementById('selectedJobProgressText'),
    selectedJobProgressBar: document.getElementById('selectedJobProgressBar'),
    selectedJobMetaLine: document.getElementById('selectedJobMetaLine'),
    selectedJobFreshness: document.getElementById('selectedJobFreshness'),
    selectedJobSummary: document.getElementById('selectedJobSummary'),
    selectedJobTransportAlert: document.getElementById('selectedJobTransportAlert'),
    openRunDetailsBtn: document.getElementById('openRunDetailsBtn'),
    inspectorOverviewTab: document.getElementById('inspectorOverviewTab'),
    inspectorTimelineTab: document.getElementById('inspectorTimelineTab'),
    inspectorLogsTab: document.getElementById('inspectorLogsTab'),
    selectedJobOverviewPanel: document.getElementById('selectedJobOverviewPanel'),
    selectedJobTimelinePanel: document.getElementById('selectedJobTimelinePanel'),
    selectedJobLogsPanel: document.getElementById('selectedJobLogsPanel'),
    selectedJobTimelineList: document.getElementById('selectedJobTimelineList'),
    selectedJobTimelineEmpty: document.getElementById('selectedJobTimelineEmpty'),
    selectedJobLogPreview: document.getElementById('selectedJobLogPreview'),
    artifactsShell: document.getElementById('artifacts-shell'),
    artifactShellContent: document.getElementById('artifactShellContent'),
    artifactSkeletonState: document.getElementById('artifactSkeletonState'),
    artifactMeta: document.getElementById('artifactMeta'),
    emptyArtifactState: document.getElementById('emptyArtifactState'),
    artifactCompareBtn: document.getElementById('artifactCompareBtn'),
    artifactPreviewStage: document.getElementById('artifactPreviewStage'),
    artifactCompareStage: document.getElementById('artifactCompareStage'),
    artifactPreviewImage: document.getElementById('artifactPreviewImage'),
    artifactPreviewSoloImage: document.getElementById('artifactPreviewSoloImage'),
    artifactCompareImage: document.getElementById('artifactCompareImage'),
    artifactPreviewPrimaryCaption: document.getElementById('artifactPreviewPrimaryCaption'),
    artifactCompareCaption: document.getElementById('artifactCompareCaption'),
    artifactMetadataCard: document.getElementById('artifactMetadataCard'),
    artifactMetadataBar: document.getElementById('artifactMetadataBar'),
    artifactSelectionTitle: document.getElementById('artifactSelectionTitle'),
    artifactSelectionMeta: document.getElementById('artifactSelectionMeta'),
    reviewStatusBanner: document.getElementById('reviewStatusBanner'),
    reviewStatusTitle: document.getElementById('reviewStatusTitle'),
    reviewStatusDetail: document.getElementById('reviewStatusDetail'),
    reviewProvenanceGrid: document.getElementById('reviewProvenanceGrid'),
    reviewProvenanceArtifactRole: document.getElementById('reviewProvenanceArtifactRole'),
    reviewProvenanceRunState: document.getElementById('reviewProvenanceRunState'),
    reviewProvenancePath: document.getElementById('reviewProvenancePath'),
    reviewProvenanceFreshness: document.getElementById('reviewProvenanceFreshness'),
    reviewProvenanceSource: document.getElementById('reviewProvenanceSource'),
    reviewProvenanceBatch: document.getElementById('reviewProvenanceBatch'),
    reviewCompareSummary: document.getElementById('reviewCompareSummary'),
    reviewCompareTitle: document.getElementById('reviewCompareTitle'),
    reviewCompareDetail: document.getElementById('reviewCompareDetail'),
    openArtifactBtn: document.getElementById('openArtifactBtn'),
    downloadArtifactBtn: document.getElementById('downloadArtifactBtn'),
    copyArtifactPathBtn: document.getElementById('copyArtifactPathBtn'),
    artifactThumbnailRail: document.getElementById('artifactThumbnailRail'),
    runCardActions: document.getElementById('runCardActions'),
    viewRunCardBtn: document.getElementById('viewRunCardBtn'),
    copyRunCardPathBtn: document.getElementById('copyRunCardPathBtn'),
    copyRunCardFingerprintBtn: document.getElementById('copyRunCardFingerprintBtn'),
    logsShell: document.getElementById('logs-shell'),
    logPane: document.getElementById('logPane'),
    logMetaLabel: document.getElementById('logMetaLabel'),
    logStatusIndicator: document.getElementById('logStatusIndicator'),
    clearLogsBtn: document.getElementById('clearLogsBtn'),
    queueStatusSummary: document.getElementById('queueStatusSummary'),

    themeBtn: document.getElementById('themeBtn'),
    shortcutsBtn: document.getElementById('shortcutsBtn'),
    shortcutsModal: document.getElementById('shortcutsModal'),
    shortcutsPanel: document.getElementById('shortcutsPanel'),
    closeShortcutsBtn: document.getElementById('closeShortcutsBtn'),
    advancedFlagsSummary: document.getElementById('advancedFlagsSummary'),
    governanceDetailsSummary: document.getElementById('governanceDetailsSummary'),
    reconstructionDetailsSummary: document.getElementById('reconstructionDetailsSummary'),
    dispatchToolsSummary: document.getElementById('dispatchToolsSummary'),
    effectiveConfigDrawer: document.getElementById('effectiveConfigDrawer'),
    closeEffectiveConfigBtn: document.getElementById('closeEffectiveConfigBtn'),
    effectiveConfigMeta: document.getElementById('effectiveConfigMeta'),
    requestedConfigJson: document.getElementById('requestedConfigJson'),
    effectiveConfigJson: document.getElementById('effectiveConfigJson'),
    inactiveConfigJson: document.getElementById('inactiveConfigJson'),
    effectiveEstimateLabel: document.getElementById('effectiveEstimateLabel'),
    effectiveReadinessSummary: document.getElementById('effectiveReadinessSummary'),
    effectiveArgvPreview: document.getElementById('effectiveArgvPreview'),

    healthIndicator: document.getElementById('healthIndicator'),
    healthText: document.getElementById('healthText'),
    toastContainer: document.getElementById('toastContainer')
};

const ambientMotion = {
    rafId: null,
    recomputeId: null,
    currentShiftX: 0,
    currentShiftY: 0,
    targetShiftX: 0,
    targetShiftY: 0,
    currentFocusX: 52,
    currentFocusY: 26,
    targetFocusX: 52,
    targetFocusY: 26,
    currentStageX: 0,
    currentStageY: 0,
    targetStageX: 0,
    targetStageY: 0,
    currentStageScale: 1,
    targetStageScale: 1,
    currentStageRotate: 0,
    targetStageRotate: 0,
    currentColorA: [45, 212, 191],
    currentColorB: [245, 158, 11],
    currentColorC: [56, 189, 248],
    currentFocusColor: [125, 211, 252],
    targetColorA: [45, 212, 191],
    targetColorB: [245, 158, 11],
    targetColorC: [56, 189, 248],
    targetFocusColor: [125, 211, 252],
    pointerX: 0,
    pointerY: 0,
    pointerActive: false,
    activeZoneEl: null,
    activeZoneConfig: null,
    motionQuery: null,
    performanceLite: false,
    enabled: true
};

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function parseZoneNumber(value, fallback) {
    const parsed = Number.parseFloat(String(value ?? '').trim());
    return Number.isFinite(parsed) ? parsed : fallback;
}

const DEFAULT_AMBIENT_PALETTE = {
    colorA: [45, 212, 191],
    colorB: [245, 158, 11],
    colorC: [56, 189, 248],
    focusColor: [125, 211, 252]
};

function cloneAmbientColor(color) {
    return [color[0], color[1], color[2]];
}

function parseAmbientColor(value, fallback) {
    const raw = String(value ?? '').trim();
    if (!raw) return cloneAmbientColor(fallback);

    const normalizedHex = raw.startsWith('#') ? raw.slice(1) : raw;
    if (/^[0-9a-fA-F]{6}$/.test(normalizedHex)) {
        return [
            Number.parseInt(normalizedHex.slice(0, 2), 16),
            Number.parseInt(normalizedHex.slice(2, 4), 16),
            Number.parseInt(normalizedHex.slice(4, 6), 16)
        ];
    }

    const channels = raw
        .split(/[,\s]+/)
        .map((part) => Number.parseFloat(part))
        .filter((part) => Number.isFinite(part));
    if (channels.length >= 3) {
        return channels.slice(0, 3).map((channel) => clamp(Math.round(channel), 0, 255));
    }

    return cloneAmbientColor(fallback);
}

function stepAmbientColor(current, target, easing) {
    for (let index = 0; index < 3; index += 1) {
        current[index] += (target[index] - current[index]) * easing;
    }
}

function ambientColorDistance(current, target) {
    return Math.abs(target[0] - current[0])
        + Math.abs(target[1] - current[1])
        + Math.abs(target[2] - current[2]);
}

function readAmbientZoneConfig(element) {
    if (!element || !element.dataset) return null;
    return {
        focusX: clamp(parseZoneNumber(element.dataset.ambientFocusX, 52), 8, 92),
        focusY: clamp(parseZoneNumber(element.dataset.ambientFocusY, 26), 8, 92),
        shiftX: parseZoneNumber(element.dataset.ambientShiftX, 0),
        shiftY: parseZoneNumber(element.dataset.ambientShiftY, 0),
        stageScale: clamp(parseZoneNumber(element.dataset.ambientStageScale, 1), 1, 1.08),
        stageRotate: clamp(parseZoneNumber(element.dataset.ambientStageRotate, 0), -1.5, 1.5),
        colorA: parseAmbientColor(element.dataset.ambientColorA, DEFAULT_AMBIENT_PALETTE.colorA),
        colorB: parseAmbientColor(element.dataset.ambientColorB, DEFAULT_AMBIENT_PALETTE.colorB),
        colorC: parseAmbientColor(element.dataset.ambientColorC, DEFAULT_AMBIENT_PALETTE.colorC),
        focusColor: parseAmbientColor(element.dataset.ambientFocusColor, DEFAULT_AMBIENT_PALETTE.focusColor)
    };
}

function _writeAmbientVariables() {
    const root = document.documentElement;
    root.style.setProperty('--ambient-shift-x', `${ambientMotion.currentShiftX.toFixed(2)}px`);
    root.style.setProperty('--ambient-shift-y', `${ambientMotion.currentShiftY.toFixed(2)}px`);
    root.style.setProperty('--ambient-focus-x', `${ambientMotion.currentFocusX.toFixed(2)}%`);
    root.style.setProperty('--ambient-focus-y', `${ambientMotion.currentFocusY.toFixed(2)}%`);
    root.style.setProperty('--ambient-stage-x', `${ambientMotion.currentStageX.toFixed(2)}px`);
    root.style.setProperty('--ambient-stage-y', `${ambientMotion.currentStageY.toFixed(2)}px`);
    root.style.setProperty('--ambient-stage-scale', ambientMotion.currentStageScale.toFixed(4));
    root.style.setProperty('--ambient-stage-rotate', `${ambientMotion.currentStageRotate.toFixed(3)}deg`);
    root.style.setProperty('--ambient-color-a', ambientMotion.currentColorA.map((value) => Math.round(value)).join(' '));
    root.style.setProperty('--ambient-color-b', ambientMotion.currentColorB.map((value) => Math.round(value)).join(' '));
    root.style.setProperty('--ambient-color-c', ambientMotion.currentColorC.map((value) => Math.round(value)).join(' '));
    root.style.setProperty('--ambient-focus-color', ambientMotion.currentFocusColor.map((value) => Math.round(value)).join(' '));
}

function _queueAmbientFrame() {
    if (ambientMotion.rafId !== null) return;
    ambientMotion.rafId = window.requestAnimationFrame(_stepAmbientMotion);
}

function shouldUsePerformanceLiteMode() {
    const saveData = Boolean(navigator.connection && navigator.connection.saveData);
    const coarsePointer = window.matchMedia('(pointer: coarse)').matches;
    const noHover = window.matchMedia('(hover: none)').matches;
    const narrowViewport = window.innerWidth < 960;
    const limitedCpu = Number(navigator.hardwareConcurrency || 8) <= 4;
    return saveData || coarsePointer || (noHover && narrowViewport) || (noHover && limitedCpu);
}

function applyAmbientCapabilityMode() {
    ambientMotion.performanceLite = shouldUsePerformanceLiteMode();
    document.documentElement.classList.toggle('performance-lite', ambientMotion.performanceLite);
    return ambientMotion.performanceLite;
}

function _scheduleAmbientTargetRecompute() {
    if (ambientMotion.recomputeId !== null) return;
    ambientMotion.recomputeId = window.requestAnimationFrame(() => {
        ambientMotion.recomputeId = null;
        _recomputeAmbientTargets();
    });
}

function _recomputeAmbientTargets() {
    const zone = ambientMotion.activeZoneConfig;
    const width = Math.max(window.innerWidth || 1, 1);
    const height = Math.max(window.innerHeight || 1, 1);
    const pointerX = ambientMotion.pointerActive ? ambientMotion.pointerX : width / 2;
    const pointerY = ambientMotion.pointerActive ? ambientMotion.pointerY : height / 2;
    const xRatio = (pointerX / width) - 0.5;
    const yRatio = (pointerY / height) - 0.5;
    const pointerWeight = zone ? 0.42 : 1;

    const baseFocusX = zone ? zone.focusX : 52;
    const baseFocusY = zone ? zone.focusY : 26;
    const baseShiftX = zone ? zone.shiftX : 0;
    const baseShiftY = zone ? zone.shiftY : 0;
    const baseStageScale = zone ? zone.stageScale : 1;
    const baseStageRotate = zone ? zone.stageRotate : 0;
    const baseColorA = zone ? zone.colorA : DEFAULT_AMBIENT_PALETTE.colorA;
    const baseColorB = zone ? zone.colorB : DEFAULT_AMBIENT_PALETTE.colorB;
    const baseColorC = zone ? zone.colorC : DEFAULT_AMBIENT_PALETTE.colorC;
    const baseFocusColor = zone ? zone.focusColor : DEFAULT_AMBIENT_PALETTE.focusColor;

    ambientMotion.targetFocusX = clamp(baseFocusX + (xRatio * 18 * pointerWeight), 8, 92);
    ambientMotion.targetFocusY = clamp(baseFocusY + (yRatio * 16 * pointerWeight), 8, 92);
    ambientMotion.targetShiftX = clamp(baseShiftX + (xRatio * 52 * pointerWeight), -72, 72);
    ambientMotion.targetShiftY = clamp(baseShiftY + (yRatio * 44 * pointerWeight), -64, 64);
    ambientMotion.targetStageX = clamp((zone ? zone.shiftX * 0.8 : 0) + (xRatio * 14 * pointerWeight), -30, 30);
    ambientMotion.targetStageY = clamp((zone ? zone.shiftY * 0.8 : 0) + (yRatio * 12 * pointerWeight), -28, 28);
    ambientMotion.targetStageScale = baseStageScale;
    ambientMotion.targetStageRotate = baseStageRotate + (xRatio * 0.22 * pointerWeight);
    ambientMotion.targetColorA = cloneAmbientColor(baseColorA);
    ambientMotion.targetColorB = cloneAmbientColor(baseColorB);
    ambientMotion.targetColorC = cloneAmbientColor(baseColorC);
    ambientMotion.targetFocusColor = cloneAmbientColor(baseFocusColor);
    _queueAmbientFrame();
}

function _resetAmbientTargets() {
    ambientMotion.pointerActive = false;
    _scheduleAmbientTargetRecompute();
}

function _clearAmbientActiveZone() {
    if (ambientMotion.activeZoneEl) {
        ambientMotion.activeZoneEl.removeAttribute('data-ambient-active');
    }
    ambientMotion.activeZoneEl = null;
    ambientMotion.activeZoneConfig = null;
    _scheduleAmbientTargetRecompute();
}

function _activateAmbientZone(element) {
    if (!element) return;
    if (ambientMotion.activeZoneEl === element) return;
    if (ambientMotion.activeZoneEl) {
        ambientMotion.activeZoneEl.removeAttribute('data-ambient-active');
    }
    ambientMotion.activeZoneEl = element;
    ambientMotion.activeZoneConfig = readAmbientZoneConfig(element);
    element.setAttribute('data-ambient-active', 'true');
    _scheduleAmbientTargetRecompute();
}

function _stepAmbientMotion() {
    ambientMotion.rafId = null;
    const easing = ambientMotion.enabled ? 0.085 : 0.16;
    ambientMotion.currentShiftX += (ambientMotion.targetShiftX - ambientMotion.currentShiftX) * easing;
    ambientMotion.currentShiftY += (ambientMotion.targetShiftY - ambientMotion.currentShiftY) * easing;
    ambientMotion.currentFocusX += (ambientMotion.targetFocusX - ambientMotion.currentFocusX) * easing;
    ambientMotion.currentFocusY += (ambientMotion.targetFocusY - ambientMotion.currentFocusY) * easing;
    ambientMotion.currentStageX += (ambientMotion.targetStageX - ambientMotion.currentStageX) * easing;
    ambientMotion.currentStageY += (ambientMotion.targetStageY - ambientMotion.currentStageY) * easing;
    ambientMotion.currentStageScale += (ambientMotion.targetStageScale - ambientMotion.currentStageScale) * easing;
    ambientMotion.currentStageRotate += (ambientMotion.targetStageRotate - ambientMotion.currentStageRotate) * easing;
    stepAmbientColor(ambientMotion.currentColorA, ambientMotion.targetColorA, easing);
    stepAmbientColor(ambientMotion.currentColorB, ambientMotion.targetColorB, easing);
    stepAmbientColor(ambientMotion.currentColorC, ambientMotion.targetColorC, easing);
    stepAmbientColor(ambientMotion.currentFocusColor, ambientMotion.targetFocusColor, easing);
    _writeAmbientVariables();

    const remaining = Math.abs(ambientMotion.targetShiftX - ambientMotion.currentShiftX)
        + Math.abs(ambientMotion.targetShiftY - ambientMotion.currentShiftY)
        + Math.abs(ambientMotion.targetFocusX - ambientMotion.currentFocusX)
        + Math.abs(ambientMotion.targetFocusY - ambientMotion.currentFocusY)
        + Math.abs(ambientMotion.targetStageX - ambientMotion.currentStageX)
        + Math.abs(ambientMotion.targetStageY - ambientMotion.currentStageY)
        + Math.abs(ambientMotion.targetStageScale - ambientMotion.currentStageScale)
        + Math.abs(ambientMotion.targetStageRotate - ambientMotion.currentStageRotate)
        + ambientColorDistance(ambientMotion.currentColorA, ambientMotion.targetColorA)
        + ambientColorDistance(ambientMotion.currentColorB, ambientMotion.targetColorB)
        + ambientColorDistance(ambientMotion.currentColorC, ambientMotion.targetColorC)
        + ambientColorDistance(ambientMotion.currentFocusColor, ambientMotion.targetFocusColor);
    if (remaining > 0.2) {
        _queueAmbientFrame();
    }
}

function _updateAmbientTargetsFromPointer(clientX, clientY) {
    if (!ambientMotion.enabled) return;
    ambientMotion.pointerActive = true;
    ambientMotion.pointerX = clientX;
    ambientMotion.pointerY = clientY;
    _scheduleAmbientTargetRecompute();
}

function setupAmbientMotion() {
    ambientMotion.motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const applyPreference = () => {
        const performanceLite = applyAmbientCapabilityMode();
        ambientMotion.enabled = !ambientMotion.motionQuery.matches && !performanceLite;
        if (!ambientMotion.enabled) {
            if (ambientMotion.activeZoneEl) {
                ambientMotion.activeZoneEl.removeAttribute('data-ambient-active');
            }
            ambientMotion.activeZoneEl = null;
            ambientMotion.activeZoneConfig = null;
            _resetAmbientTargets();
        } else {
            _scheduleAmbientTargetRecompute();
        }
    };

    document.querySelectorAll('[data-ambient-zone]').forEach((zone) => {
        zone.addEventListener('mouseenter', () => _activateAmbientZone(zone));
        zone.addEventListener('mouseleave', () => {
            if (!zone.matches(':focus-within')) {
                _clearAmbientActiveZone();
            }
        });
        zone.addEventListener('focusin', () => _activateAmbientZone(zone));
        zone.addEventListener('focusout', (event) => {
            const next = event.relatedTarget;
            if (next && zone.contains(next)) return;
            if (!zone.matches(':hover')) {
                _clearAmbientActiveZone();
            }
        });
    });

    window.addEventListener('pointermove', (event) => {
        if (!ambientMotion.enabled) return;
        _updateAmbientTargetsFromPointer(event.clientX, event.clientY);
    }, { passive: true });
    window.addEventListener('pointerout', (event) => {
        if (event.relatedTarget === null) {
            _resetAmbientTargets();
        }
    });
    window.addEventListener('scroll', () => {
        if (ambientMotion.activeZoneEl) {
            _scheduleAmbientTargetRecompute();
        }
    }, { passive: true });
    window.addEventListener('blur', _resetAmbientTargets);
    window.addEventListener('resize', () => {
        applyPreference();
        _resetAmbientTargets();
    });
    ambientMotion.motionQuery.addEventListener('change', applyPreference);
    applyPreference();
    _writeAmbientVariables();
}

const CONSOLE_VIEW_META = {
    overview: {
        title: 'Overview',
        summary: 'Status, recent jobs, and the clearest next operator action.',
        meta: 'Use this surface to understand connection mode, current draft, and where the next useful click should go.'
    },
    build: {
        title: 'Build',
        summary: 'Move from pipeline choice to dispatch through one preview-backed step at a time.',
        meta: 'This view keeps connection mode, paths, contextual options, and launch in one guided build flow.'
    },
    operate: {
        title: 'Operate',
        summary: 'Queue, selected run, artifact arrival, and logs in one live runtime workspace.',
        meta: 'Operators should identify state, progress, and next action in a single pass.'
    },
    review: {
        title: 'Review',
        summary: 'Review governed outputs, provenance, and downloadable artifacts from a dedicated audit surface.',
        meta: 'Completed runs become reviewable products, not just finished jobs.'
    }
};

function resolveConsoleView(value) {
    const candidate = String(value || '').trim().toLowerCase();
    return Object.prototype.hasOwnProperty.call(CONSOLE_VIEW_META, candidate) ? candidate : 'overview';
}

function _normalizeSelectedJobId(jobId) {
    return String(jobId || '').trim();
}

function _normalizeArtifactRoutePath(value) {
    return String(value || '').trim();
}

function _normalizeCompareQueryValue(value) {
    return String(value || '').trim() === '1';
}

function _rememberSelectedJob(jobId) {
    const normalized = _normalizeSelectedJobId(jobId);
    if (normalized) {
        state.portalUi.lastSelectedJobId = normalized;
    }
    return normalized;
}

function _rememberArtifactSelection(jobId, artifactPath) {
    const normalizedJobId = _normalizeSelectedJobId(jobId);
    if (!normalizedJobId) return '';
    const normalizedArtifactPath = _normalizeArtifactRoutePath(artifactPath);
    if (normalizedArtifactPath) {
        state.artifactUi.selectedByJob[normalizedJobId] = normalizedArtifactPath;
    } else {
        delete state.artifactUi.selectedByJob[normalizedJobId];
    }
    return normalizedArtifactPath;
}

function _rememberComparePreference(jobId, enabled) {
    const normalizedJobId = _normalizeSelectedJobId(jobId);
    if (!normalizedJobId) return false;
    const normalizedEnabled = Boolean(enabled);
    state.artifactUi.compareByJob[normalizedJobId] = normalizedEnabled;
    return normalizedEnabled;
}

function _preferredSelectedJobId() {
    return _normalizeSelectedJobId(state.selectedJobId || state.portalUi.lastSelectedJobId);
}

function _activeRouteContext(jobId = '') {
    const normalizedJobId = _normalizeSelectedJobId(jobId);
    if (!normalizedJobId) {
        return { artifactPath: '', compareEnabled: false };
    }
    const selected = state.jobs.find((job) => job.id === normalizedJobId) || null;
    const fallbackArtifactPath = _normalizeArtifactRoutePath(state.artifactUi.selectedByJob[normalizedJobId]);
    if (!selected) {
        return {
            artifactPath: fallbackArtifactPath,
            compareEnabled: Boolean(state.artifactUi.compareByJob[normalizedJobId])
        };
    }
    const selectedArtifact = _selectedArtifactForJob(selected);
    const artifactPath = _normalizeArtifactRoutePath(_artifactRouteKey(selectedArtifact));
    const compareCandidate = findCompareArtifact(
        selectedArtifact,
        rankArtifactsForDisplay(Array.isArray(selected.artifacts) ? selected.artifacts : [])
    );
    return {
        artifactPath,
        compareEnabled: Boolean(compareCandidate) && Boolean(state.artifactUi.compareByJob[normalizedJobId])
    };
}

function setActiveWorkspaceLink(viewName) {
    document.querySelectorAll('[data-view-link]').forEach((link) => {
        const active = String(link.dataset.viewLink || '') === String(viewName || '');
        link.classList.toggle('is-active', active);
        link.setAttribute('aria-selected', active ? 'true' : 'false');
        if (active) {
            link.setAttribute('aria-current', 'location');
        } else {
            link.removeAttribute('aria-current');
        }
    });
}

function _routeUrlForView(viewName, jobId = '', artifactPath = '', compareEnabled = null) {
    const url = new URL(window.location.href);
    const resolvedView = resolveConsoleView(viewName);
    const resolvedJobId = _normalizeSelectedJobId(jobId);
    url.searchParams.set('view', resolvedView);
    if ((resolvedView === 'operate' || resolvedView === 'review') && resolvedJobId) {
        const activeContext = _activeRouteContext(resolvedJobId);
        const resolvedArtifactPath = _normalizeArtifactRoutePath(artifactPath) || activeContext.artifactPath;
        const resolvedCompareEnabled = compareEnabled === null ? activeContext.compareEnabled : Boolean(compareEnabled);
        url.searchParams.set('job', resolvedJobId);
        if (resolvedArtifactPath) {
            url.searchParams.set('artifact', resolvedArtifactPath);
        } else {
            url.searchParams.delete('artifact');
        }
        if (resolvedCompareEnabled) {
            url.searchParams.set('compare', '1');
        } else {
            url.searchParams.delete('compare');
        }
    } else {
        url.searchParams.delete('job');
        url.searchParams.delete('artifact');
        url.searchParams.delete('compare');
    }
    return url;
}

function _syncConsoleRoute(replace = false) {
    const url = _routeUrlForView(state.currentView, state.selectedJobId);
    const nextHref = `${url.pathname}${url.search}${url.hash}`;
    const currentHref = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextHref === currentHref) return;
    const method = replace ? 'replaceState' : 'pushState';
    window.history[method]({ view: state.currentView, jobId: state.selectedJobId || '' }, '', nextHref);
}

function updateConsoleViewContext() {
    const viewMeta = CONSOLE_VIEW_META[state.currentView] || CONSOLE_VIEW_META.overview;
    if (els.consoleViewTitle) els.consoleViewTitle.textContent = viewMeta.title;
    if (els.consoleViewSummary) els.consoleViewSummary.textContent = viewMeta.summary;
    if (els.consoleViewMeta) {
        if ((state.currentView === 'operate' || state.currentView === 'review') && state.selectedJobId) {
            const selected = state.jobs.find((job) => job.id === state.selectedJobId) || null;
            const stateLabel = titleCaseToken(selected?.state, 'Pending');
            const surfaceLabel = state.currentView === 'review' ? 'review surface' : 'live workspace';
            els.consoleViewMeta.textContent = `${state.selectedJobId} • ${stateLabel} ${surfaceLabel}`;
        } else {
            els.consoleViewMeta.textContent = viewMeta.meta;
        }
    }
    document.title = state.currentView === 'overview'
        ? 'Transformation Portal — Orchestrator'
        : `Transformation Portal — ${viewMeta.title}`;
}

function renderConsoleContextRibbon() {
    if (!els.consoleContextRibbon) return;
    const ribbonVisible = state.currentView === 'operate' || state.currentView === 'review';
    els.consoleContextRibbon.classList.toggle('hidden', !ribbonVisible);
    if (!ribbonVisible) return;

    const selected = state.jobs.find((job) => job.id === state.selectedJobId) || null;
    const artifacts = Array.isArray(selected?.artifacts) ? rankArtifactsForDisplay(selected.artifacts) : [];
    const selectedArtifact = selected ? _selectedArtifactForJob(selected) : null;
    const compareCandidate = selected ? findCompareArtifact(selectedArtifact, artifacts) : null;
    const compareEnabled = Boolean(selected && compareCandidate && state.artifactUi.compareByJob[String(selected.id || '')]);
    const lastActivityAt = Number(selected?.lastEventAt || selected?.updatedAt || selected?.createdAt || 0);
    const freshnessLabel = selected ? `Updated ${formatRelativeTime(lastActivityAt)}` : 'No live telemetry';
    const artifactCount = artifacts.length;

    if (els.contextRibbonJob) {
        els.contextRibbonJob.textContent = selected ? String(selected.id || 'unknown') : 'No job selected';
    }
    if (els.contextRibbonJobMeta) {
        els.contextRibbonJobMeta.textContent = selected
            ? `${String(selected.pipeline || 'unknown')} • ${artifactCount} artifact${artifactCount === 1 ? '' : 's'} indexed`
            : 'Choose a run in operate or review to pin context here.';
    }
    if (els.contextRibbonState) {
        els.contextRibbonState.textContent = selected ? titleCaseToken(selected.state, 'Unknown') : 'Idle';
    }
    if (els.contextRibbonFreshness) {
        els.contextRibbonFreshness.textContent = freshnessLabel;
    }
    if (els.contextRibbonArtifact) {
        els.contextRibbonArtifact.textContent = selectedArtifact ? artifactLabel(selectedArtifact) : 'Awaiting selection';
    }
    if (els.contextRibbonArtifactMeta) {
        els.contextRibbonArtifactMeta.textContent = selectedArtifact
            ? `${artifactDisplayLabel(selectedArtifact)}${compareCandidate ? ' • compare pair available' : ''}`
            : 'Review context will show the active artifact path here.';
    }
    if (els.contextRibbonCompare) {
        els.contextRibbonCompare.textContent = compareEnabled
            ? 'Compare on'
            : compareCandidate
                ? 'Single view'
                : 'No compare pair';
    }
    if (els.contextRibbonCompareMeta) {
        els.contextRibbonCompareMeta.textContent = compareEnabled
            ? 'URL-backed review context includes compare=1 for this selection.'
            : compareCandidate
                ? 'Toggle compare to inspect the paired artifact side by side.'
                : 'Deep-linkable review context stays aligned with the URL.';
    }
}

function applyConsoleViewLayout() {
    state.currentView = resolveConsoleView(state.currentView);
    if (document.body) {
        document.body.dataset.consoleView = state.currentView;
    }
    if (els.overviewShell) els.overviewShell.classList.toggle('hidden', state.currentView !== 'overview');
    if (els.consoleGrid) els.consoleGrid.classList.toggle('hidden', state.currentView === 'overview');
    if (els.buildShell) {
        const buildActive = state.currentView === 'build';
        els.buildShell.classList.toggle('hidden', !buildActive);
        els.buildShell.style.gridColumn = buildActive ? 'span 12 / span 12' : '';
    }
    if (els.jobsShell) {
        const jobsActive = state.currentView === 'operate' || state.currentView === 'review';
        els.jobsShell.classList.toggle('hidden', !jobsActive);
        els.jobsShell.style.gridColumn = jobsActive ? 'span 12 / span 12' : '';
    }
    if (els.queueShell) {
        els.queueShell.classList.toggle('hidden', state.currentView === 'review');
    }
    if (els.openRunDetailsBtn) {
        els.openRunDetailsBtn.textContent = state.currentView === 'review' ? 'Review Open' : 'Open Review';
    }
    if (state.currentView === 'operate' && state.selectedJobId) {
        setInspectorTab('timeline');
    } else if (state.currentView === 'review') {
        setInspectorTab('overview');
    }
    updateConsoleViewContext();
    setActiveWorkspaceLink(state.currentView);
    renderConsoleContextRibbon();
}

function navigateConsoleView(viewName, options = {}) {
    const replace = Boolean(options.replace);
    state.currentView = resolveConsoleView(viewName);
    const explicitJobId = _normalizeSelectedJobId(options.jobId);
    const hasArtifactOption = Object.prototype.hasOwnProperty.call(options, 'artifactPath');
    const hasCompareOption = Object.prototype.hasOwnProperty.call(options, 'compareEnabled');
    if (explicitJobId) {
        state.selectedJobId = explicitJobId;
        _rememberSelectedJob(explicitJobId);
        if (hasArtifactOption) {
            _rememberArtifactSelection(explicitJobId, options.artifactPath);
        }
        if (hasCompareOption) {
            _rememberComparePreference(explicitJobId, options.compareEnabled);
        }
    } else if (state.currentView === 'operate' || state.currentView === 'review') {
        const preferredJobId = _preferredSelectedJobId();
        if (preferredJobId) {
            state.selectedJobId = preferredJobId;
            _rememberSelectedJob(preferredJobId);
        }
    } else if (state.selectedJobId) {
        _rememberSelectedJob(state.selectedJobId);
    }
    applyConsoleViewLayout();
    _syncConsoleRoute(replace);
    renderJobQueue();
}

function applyConsoleRouteFromLocation(replace = false) {
    const url = new URL(window.location.href);
    state.currentView = resolveConsoleView(url.searchParams.get('view'));
    const routeJobId = _normalizeSelectedJobId(url.searchParams.get('job'));
    const routeArtifactPath = _normalizeArtifactRoutePath(url.searchParams.get('artifact'));
    const routeCompareEnabled = _normalizeCompareQueryValue(url.searchParams.get('compare'));
    const routeHasArtifact = url.searchParams.has('artifact');
    const routeHasCompare = url.searchParams.has('compare');
    if (routeJobId) {
        state.selectedJobId = routeJobId;
        _rememberSelectedJob(routeJobId);
        if (routeHasArtifact) {
            _rememberArtifactSelection(routeJobId, routeArtifactPath);
        } else {
            _rememberArtifactSelection(routeJobId, '');
        }
        if (routeHasCompare) {
            _rememberComparePreference(routeJobId, routeCompareEnabled);
        } else {
            _rememberComparePreference(routeJobId, false);
        }
    } else if (state.currentView === 'operate' || state.currentView === 'review') {
        const preferredJobId = _preferredSelectedJobId();
        if (preferredJobId) {
            state.selectedJobId = preferredJobId;
            _rememberSelectedJob(preferredJobId);
        }
    } else if (state.selectedJobId) {
        _rememberSelectedJob(state.selectedJobId);
    }
    applyConsoleViewLayout();
    _syncConsoleRoute(replace);
}

function setupSectionRail() {
    const links = Array.from(document.querySelectorAll('[data-view-link]'));
    if (!links.length) return;

    links.forEach((link) => {
        link.addEventListener('click', (event) => {
            const isPlainPrimaryClick = event.button === 0
                && !event.metaKey
                && !event.ctrlKey
                && !event.shiftKey
                && !event.altKey;

            if (event.defaultPrevented || !isPlainPrimaryClick) {
                return;
            }
            event.preventDefault();
            const nextView = resolveConsoleView(link.dataset.viewLink);
            if (nextView === 'review' && !state.selectedJobId) {
                createToast('Select a run first, then open its review surface.', 'info');
                return;
            }
            navigateConsoleView(nextView);
        });
    });
    setActiveWorkspaceLink(state.currentView);
}

const BUILD_STEP_CONTENT = Object.freeze({
    lux: [
        {
            label: 'Configure',
            meta: 'Pipeline and preset posture.',
            title: '1. Configure the draft',
            summary: 'Choose the pipeline and preset that should drive the next preview-backed run.'
        },
        {
            label: 'Paths',
            meta: 'Inputs, outputs, and roots.',
            title: '2. Set paths',
            summary: 'Supply input and output roots before opening anything advanced.'
        },
        {
            label: 'Outputs',
            meta: 'Deliverables, posture, and readiness.',
            title: '3. Shape deliverables and confirm output posture',
            summary: 'Keep deliverables, primary output posture, and immediate readiness readable first.'
        },
        {
            label: 'Dispatch',
            meta: 'Primary review, launch, and parity tools.',
            title: '4. Review and dispatch',
            summary: 'Use preview-backed warnings, effective argv, and readiness to launch with confidence.'
        }
    ],
    archive: [
        {
            label: 'Stage',
            meta: 'Select the archive stage.',
            title: '1. Select the archive stage',
            summary: 'Pick the archive stage from the connection card, then continue into stage-specific inputs.'
        },
        {
            label: 'Paths',
            meta: 'Required stage files.',
            title: '2. Provide stage inputs',
            summary: 'Supply the archive root, archive index, or rights manifest required for the selected stage.'
        },
        {
            label: 'Readiness',
            meta: 'Review governance signals.',
            title: '3. Review readiness',
            summary: 'Confirm the stage-specific readiness checks before moving to dispatch.'
        },
        {
            label: 'Dispatch',
            meta: 'Canonical command and launch.',
            title: '4. Dispatch archive job',
            summary: 'Review the canonical command, warnings, and launch the archive job.'
        }
    ]
});

function _currentBuildStepContent() {
    return state.pipeline === 'lux-depth-v3' ? BUILD_STEP_CONTENT.lux : BUILD_STEP_CONTENT.archive;
}

function _minimumBuildStep() {
    return state.pipeline === 'lux-depth-v3' ? 1 : 2;
}

function resolveBuildStep(value) {
    const parsed = Number.parseInt(String(value || ''), 10);
    const minimum = _minimumBuildStep();
    if (!Number.isFinite(parsed)) return minimum;
    return Math.max(minimum, Math.min(4, parsed));
}

function _buildStepButtons() {
    return [els.buildStepTab1, els.buildStepTab2, els.buildStepTab3, els.buildStepTab4].filter(Boolean);
}

function _setBuildStepButtonCopy(button, content) {
    if (!button || !content) return;
    const label = button.querySelector('.build-step-label');
    const meta = button.querySelector('.build-step-meta');
    if (label) label.textContent = content.label;
    if (meta) meta.textContent = content.meta;
}

function syncBuildStepUi() {
    state.portalUi.buildStep = resolveBuildStep(state.portalUi.buildStep);
    const activeStep = state.portalUi.buildStep;
    const minimum = _minimumBuildStep();
    const stepContent = _currentBuildStepContent();

    if (els.buildStepTitle) els.buildStepTitle.textContent = stepContent[activeStep - 1]?.title || 'Build';
    if (els.buildStepSummary) els.buildStepSummary.textContent = stepContent[activeStep - 1]?.summary || '';

    _buildStepButtons().forEach((button, index) => {
        const step = index + 1;
        const content = stepContent[index] || BUILD_STEP_CONTENT.lux[index];
        const unavailable = step < minimum;
        const active = step === activeStep;
        _setBuildStepButtonCopy(button, content);
        button.classList.toggle('is-active', active);
        button.classList.toggle('is-disabled', unavailable);
        button.disabled = unavailable;
        button.tabIndex = active ? 0 : -1;
        button.setAttribute('aria-selected', active ? 'true' : 'false');
        button.setAttribute('aria-disabled', unavailable ? 'true' : 'false');
    });

    document.querySelectorAll('[data-build-step-panel]').forEach((panel) => {
        const step = Number.parseInt(String(panel.getAttribute('data-build-step-panel') || ''), 10);
        const active = step === activeStep;
        panel.hidden = !active;
        panel.setAttribute('data-step-active', active ? 'true' : 'false');
        panel.setAttribute('data-step-hidden', active ? 'false' : 'true');
    });

    if (els.buildStepBackBtn) {
        els.buildStepBackBtn.disabled = activeStep <= minimum;
    }
    if (els.buildStepNextBtn) {
        els.buildStepNextBtn.disabled = activeStep >= 4;
        els.buildStepNextBtn.textContent = activeStep >= 4 ? 'Dispatch Ready' : 'Next';
    }
}

function setBuildStep(nextStep, options = {}) {
    const previous = resolveBuildStep(state.portalUi.buildStep);
    const resolved = resolveBuildStep(nextStep);
    state.portalUi.buildStep = resolved;
    syncBuildStepUi();
    if (!options.silent && resolved > previous) {
        void emitPortalEvent('step_completed', {
            surface: 'build_stepper',
            metadata: { step: previous, next_step: resolved }
        });
    }
}

function setupBuildStepper() {
    _buildStepButtons().forEach((button) => {
        button.addEventListener('click', () => {
            setBuildStep(button.dataset.buildStepTarget);
        });
        button.addEventListener('keydown', (event) => {
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
            event.preventDefault();
            const buttons = _buildStepButtons().filter((candidate) => !candidate.disabled);
            const currentIndex = buttons.indexOf(button);
            if (currentIndex === -1) return;
            let nextIndex = currentIndex;
            if (event.key === 'ArrowRight') nextIndex = Math.min(buttons.length - 1, currentIndex + 1);
            if (event.key === 'ArrowLeft') nextIndex = Math.max(0, currentIndex - 1);
            if (event.key === 'Home') nextIndex = 0;
            if (event.key === 'End') nextIndex = buttons.length - 1;
            const nextButton = buttons[nextIndex];
            if (!nextButton) return;
            nextButton.focus();
            setBuildStep(nextButton.dataset.buildStepTarget);
        });
    });

    if (els.buildStepBackBtn) {
        els.buildStepBackBtn.addEventListener('click', () => setBuildStep(state.portalUi.buildStep - 1, { silent: true }));
    }
    if (els.buildStepNextBtn) {
        els.buildStepNextBtn.addEventListener('click', () => {
            if (state.portalUi.buildStep >= 4) {
                if (els.runJobBtn) els.runJobBtn.focus();
                return;
            }
            setBuildStep(state.portalUi.buildStep + 1);
        });
    }

    syncBuildStepUi();
}

function truncateMiddle(value, maxLength = 44) {
    const text = String(value || '').trim();
    if (text.length <= maxLength) return text || '—';
    const edge = Math.max(8, Math.floor((maxLength - 1) / 2));
    return `${text.slice(0, edge)}…${text.slice(-edge)}`;
}

function titleCaseToken(value, fallback = 'Unknown') {
    const text = String(value || '').trim();
    if (!text) return fallback;
    return text
        .split(/[_\-\s]+/)
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' ');
}

function parseTimestamp(value, fallback = Date.now()) {
    if (typeof value === 'number' && Number.isFinite(value)) {
        return value > 0 && value < 1e12 ? value * 1000 : value;
    }
    if (typeof value === 'string' && value.trim()) {
        const numeric = Number(value);
        if (Number.isFinite(numeric)) {
            return numeric > 0 && numeric < 1e12 ? numeric * 1000 : numeric;
        }
        const parsed = Date.parse(value);
        if (Number.isFinite(parsed)) return parsed;
    }
    return fallback;
}

function formatRelativeTime(timestamp) {
    if (!Number.isFinite(timestamp) || timestamp <= 0) return 'just now';
    const diffMs = Math.max(0, Date.now() - timestamp);
    const diffSeconds = Math.round(diffMs / 1000);
    if (diffSeconds < 10) return 'just now';
    if (diffSeconds < 60) return `${diffSeconds}s ago`;
    const diffMinutes = Math.round(diffSeconds / 60);
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    const diffHours = Math.round(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.round(diffHours / 24);
    return `${diffDays}d ago`;
}

function formatTransportLabel(job) {
    if (!job) return 'idle';
    if (job.reconnectBlocked) return 'auth blocked';
    const nativeReadyState = _nativeEventSourceReadyState(job.eventSource);
    if (job.usesFetchSse && _jobHasActiveStream(job)) return 'fetch stream';
    if (nativeReadyState === EVENT_SOURCE_READY_STATE_CONNECTING) return 'event reconnecting';
    if (nativeReadyState === EVENT_SOURCE_READY_STATE_OPEN) return 'event stream';
    if (nativeReadyState === EVENT_SOURCE_READY_STATE_CLOSED) return 'event closed';
    if (_jobHasActiveStream(job)) return 'event stream';
    if (job.state === 'running' || job.state === 'queued') return 'awaiting stream';
    return 'closed';
}

function formatBytes(sizeBytes) {
    if (!Number.isFinite(sizeBytes) || sizeBytes < 0) return 'unknown size';
    if (sizeBytes < 1024) return `${sizeBytes} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let value = sizeBytes / 1024;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024;
        unitIndex += 1;
    }
    return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatDuration(startMs, endMs = Date.now()) {
    if (!Number.isFinite(startMs) || startMs <= 0) return 'elapsed unknown';
    const safeEnd = Number.isFinite(endMs) && endMs > startMs ? endMs : Date.now();
    const totalSeconds = Math.max(0, Math.round((safeEnd - startMs) / 1000));
    if (totalSeconds < 60) return `${totalSeconds}s elapsed`;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    if (minutes < 60) return `${minutes}m ${seconds}s elapsed`;
    const hours = Math.floor(minutes / 60);
    const remMinutes = minutes % 60;
    return `${hours}h ${remMinutes}m elapsed`;
}

function formatTimelineTimestamp(timestamp) {
    if (!Number.isFinite(timestamp) || timestamp <= 0) return 'just now';
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function artifactMediaKind(artifact) {
    return String(artifact?.media_kind || artifact?.artifact_type || 'file').trim().toLowerCase();
}

function artifactContentType(artifact) {
    return String(artifact?.content_type || '').trim();
}

function artifactIsPreviewable(artifact) {
    return Boolean(artifact?.previewable) && artifactMediaKind(artifact) === 'image';
}

function artifactLabel(artifact) {
    return String(artifact?.relative_path || artifact?.path || 'artifact').trim();
}

function _artifactRouteKey(artifact) {
    return _normalizeArtifactRoutePath(artifactLabel(artifact));
}

function _normalizeArtifactDisplayHint(rawHint) {
    if (!rawHint || typeof rawHint !== 'object') return null;
    const role = String(rawHint.role || '').trim();
    const label = String(rawHint.label || '').trim();
    const compareGroup = String(rawHint.compare_group || '').trim();
    const priorityValue = Number(rawHint.priority);
    return {
        role,
        label,
        priority: Number.isFinite(priorityValue) ? priorityValue : null,
        compare_group: compareGroup
    };
}

function artifactDisplayHint(artifact) {
    return _normalizeArtifactDisplayHint(artifact?.display_hint);
}

function artifactDisplayPriority(artifact) {
    const priority = artifactDisplayHint(artifact)?.priority;
    return Number.isFinite(priority) ? priority : null;
}

function artifactDisplayLabel(artifact) {
    const label = String(artifactDisplayHint(artifact)?.label || '').trim();
    return label || titleCaseToken(artifactMediaKind(artifact), 'File');
}

function artifactCompareGroup(artifact) {
    return String(artifactDisplayHint(artifact)?.compare_group || '').trim();
}

function artifactNameParts(artifact) {
    const relativePath = artifactLabel(artifact);
    const segments = relativePath.split('/');
    const fileName = segments[segments.length - 1] || relativePath;
    const stem = fileName.replace(/\.[^.]+$/, '').toLowerCase();
    return {
        relativePath,
        fileName,
        stem,
        parent: segments.slice(0, -1).join('/').toLowerCase(),
    };
}

function sanitizeManagedAssetUrl(rawUrl) {
    const text = String(rawUrl || '').trim();
    if (!text) return '';
    try {
        const parsed = new URL(text, window.location.origin);
        if (!['http:', 'https:'].includes(parsed.protocol)) return '';
        if (parsed.origin !== window.location.origin) return '';
        if (!parsed.pathname.startsWith('/v1/jobs/')) return '';
        return `${parsed.pathname}${parsed.search}`;
    } catch (_err) {
        return '';
    }
}

function buildArtifactUrl(job, artifact) {
    const directUrl = String(artifact?.url || '').trim();
    if (directUrl) return sanitizeManagedAssetUrl(`${API_BASE}${directUrl}`);
    if (!job || !artifact) return '';
    const relativePath = artifactLabel(artifact);
    if (!relativePath) return '';
    const encodedSegments = relativePath.split('/').map((segment) => encodeURIComponent(segment)).join('/');
    return sanitizeManagedAssetUrl(`${API_BASE}/v1/jobs/${encodeURIComponent(String(job.id || ''))}/artifacts/${encodedSegments}`);
}

function artifactHeroScore(artifact) {
    if (!artifact || typeof artifact !== 'object') return -1;
    const info = artifactNameParts(artifact);
    const name = `${info.parent}/${info.fileName}`.toLowerCase();
    let score = 0;

    if (artifactIsPreviewable(artifact)) score += 500;
    if (artifactMediaKind(artifact) === 'image') score += 180;
    if (/(master16|upscaled16|final|result|render|beauty|marketing|depth)/.test(name)) score += 80;
    if (/(mask|thumb|preview|debug)/.test(name)) score -= 15;
    if (/(run[_-]?card|manifest|report|metadata|json|yaml|yml|txt|csv|log)/.test(name)) score -= 55;
    if (Number.isFinite(artifact.size_bytes)) score += Math.min(40, Math.round(artifact.size_bytes / 500000));

    return score;
}

function rankArtifactsForDisplay(artifacts) {
    return [...artifacts].sort((left, right) => {
        const priorityDelta = (artifactDisplayPriority(right) ?? -1) - (artifactDisplayPriority(left) ?? -1);
        if (priorityDelta !== 0) return priorityDelta;
        const scoreDelta = artifactHeroScore(right) - artifactHeroScore(left);
        if (scoreDelta !== 0) return scoreDelta;
        return artifactLabel(left).localeCompare(artifactLabel(right));
    });
}

function findCompareArtifact(primaryArtifact, artifacts) {
    if (!primaryArtifact || !artifactIsPreviewable(primaryArtifact)) return null;
    const primaryGroup = artifactCompareGroup(primaryArtifact);
    if (primaryGroup) {
        const hintedCandidate = rankArtifactsForDisplay(
            artifacts.filter((candidate) => (
                candidate
                && candidate.path !== primaryArtifact.path
                && artifactIsPreviewable(candidate)
                && artifactCompareGroup(candidate) === primaryGroup
            ))
        )[0] || null;
        if (hintedCandidate) return hintedCandidate;
    }
    const primary = artifactNameParts(primaryArtifact);
    const primaryExt = primary.fileName.includes('.') ? primary.fileName.split('.').pop().toLowerCase() : '';
    let best = null;
    let bestScore = -1;

    artifacts.forEach((candidate) => {
        if (!candidate || candidate.path === primaryArtifact.path || !artifactIsPreviewable(candidate)) return;
        const info = artifactNameParts(candidate);
        const ext = info.fileName.includes('.') ? info.fileName.split('.').pop().toLowerCase() : '';
        let score = 0;
        if (info.parent && info.parent === primary.parent) score += 30;
        if (ext && ext === primaryExt) score += 20;
        if (info.stem && primary.stem && (info.stem.includes(primary.stem) || primary.stem.includes(info.stem))) score += 35;
        if (/(master16|upscaled16|final|result|render|beauty|marketing|depth)/.test(info.fileName.toLowerCase())) score += 10;
        if (score > bestScore) {
            bestScore = score;
            best = candidate;
        }
    });

    return bestScore > 0 ? best : null;
}

function _timelineEntry(kind, label, detail, timestamp, tone = 'info', key = '') {
    return {
        kind,
        label,
        detail,
        tone,
        timestamp: Number.isFinite(timestamp) && timestamp > 0 ? timestamp : Date.now(),
        key: key || `${kind}|${label}|${detail}`
    };
}

function _ensureJobTimeline(job) {
    if (!job) return;
    if (!Array.isArray(job.timeline)) job.timeline = [];
    if (!Array.isArray(job.transportWarnings)) job.transportWarnings = [];
    if (!Array.isArray(job.progressMilestones)) job.progressMilestones = [];
}

function _pushTimelineEntry(job, entry) {
    if (!job || !entry) return;
    _ensureJobTimeline(job);
    if (job.timeline.some((existing) => existing && existing.key === entry.key)) return;
    job.timeline.push(entry);
    job.timeline.sort((left, right) => (Number(left.timestamp) || 0) - (Number(right.timestamp) || 0));
    if (job.timeline.length > 40) {
        job.timeline.splice(0, job.timeline.length - 40);
    }
}

function _noteTransportWarning(job, code, detail, tone = 'warn') {
    if (!job || !code || !detail) return;
    _ensureJobTimeline(job);
    if (job.transportWarnings.some((warning) => warning && warning.code === code && warning.detail === detail)) return;
    const warning = {
        code,
        detail,
        tone,
        timestamp: Date.now()
    };
    job.transportWarnings.push(warning);
    _pushTimelineEntry(job, _timelineEntry('transport', 'Transport warning', detail, warning.timestamp, tone, `transport|${code}|${detail}`));
}

function _recordProgressTimeline(job, progress, timestamp = Date.now()) {
    if (!job) return;
    _ensureJobTimeline(job);
    TIMELINE_PROGRESS_CHECKPOINTS.forEach((checkpoint) => {
        if (progress < checkpoint || job.progressMilestones.includes(checkpoint)) return;
        job.progressMilestones.push(checkpoint);
        _pushTimelineEntry(
            job,
            _timelineEntry(
                'progress',
                `Progress reached ${checkpoint}%`,
                checkpoint === 100 ? 'Run reached the final progress checkpoint.' : `Checkpoint ${checkpoint}% crossed.`,
                timestamp,
                checkpoint >= 100 ? 'success' : 'info',
                `progress|${checkpoint}`
            )
        );
    });
}

function _reconcileJobTimeline(job) {
    if (!job) return;
    _ensureJobTimeline(job);
    _pushTimelineEntry(job, _timelineEntry('dispatch', 'Dispatch created', 'Run request created in the cockpit.', job.createdAt, 'info', 'dispatch|created'));

    if (job.startedAt > 0) {
        _pushTimelineEntry(job, _timelineEntry('state', 'Run started', 'Worker admitted the run and execution began.', job.startedAt, 'info', 'state|started'));
    } else if (job.state === 'queued') {
        _pushTimelineEntry(job, _timelineEntry('state', 'Queued', 'Awaiting worker admission.', job.createdAt, 'info', 'state|queued'));
    }

    if (Number(job.progress) > 0) {
        _recordProgressTimeline(job, Number(job.progress), Number(job.updatedAt || job.startedAt || job.createdAt || Date.now()));
    }

    const artifactCount = Array.isArray(job.artifacts) ? job.artifacts.length : 0;
    if (artifactCount > 0) {
        _pushTimelineEntry(
            job,
            _timelineEntry(
                'artifact',
                'Artifacts indexed',
                `${artifactCount} artifact${artifactCount === 1 ? '' : 's'} available for visual review.`,
                Number(job.finishedAt || job.updatedAt || Date.now()),
                'success',
                `artifacts|${artifactCount}`
            )
        );
    }

    if (job.state === 'running') {
        _pushTimelineEntry(
            job,
            _timelineEntry(
                'state',
                'Run active',
                'Streaming progress and output updates.',
                Number(job.updatedAt || job.startedAt || Date.now()),
                'info',
                'state|running'
            )
        );
    }

    if (job.reconnectBlocked) {
        _noteTransportWarning(job, 'auth_blocked', 'Transport is blocked until authentication is restored.', 'warn');
    }

    if (job.state === 'partial') {
        _pushTimelineEntry(
            job,
            _timelineEntry(
                'done',
                'Run partially completed',
                jobOutcomeSummary(job) || 'Some inputs failed, but outputs remain reviewable.',
                Number(job.finishedAt || job.updatedAt || Date.now()),
                'warn',
                'done|partial'
            )
        );
    } else if (job.state === 'succeeded') {
        _pushTimelineEntry(job, _timelineEntry('done', 'Run completed', 'Outputs are ready for review.', Number(job.finishedAt || job.updatedAt || Date.now()), 'success', 'done|succeeded'));
    } else if (job.state === 'failed') {
        _pushTimelineEntry(job, _timelineEntry('done', 'Run failed', getReadableError(job.error) || 'Runner reported a terminal failure.', Number(job.finishedAt || job.updatedAt || Date.now()), 'error', 'done|failed'));
    } else if (job.state === 'canceled') {
        _pushTimelineEntry(job, _timelineEntry('done', 'Run canceled', 'Execution was canceled before completion.', Number(job.finishedAt || job.updatedAt || Date.now()), 'warn', 'done|canceled'));
    }
}

function setInspectorTab(tabName) {
    const nextTab = ['overview', 'timeline', 'logs'].includes(tabName) ? tabName : 'overview';
    state.inspectorTab = nextTab;

    const tabConfig = [
        { name: 'overview', button: els.inspectorOverviewTab, panel: els.selectedJobOverviewPanel },
        { name: 'timeline', button: els.inspectorTimelineTab, panel: els.selectedJobTimelinePanel },
        { name: 'logs', button: els.inspectorLogsTab, panel: els.selectedJobLogsPanel },
    ];

    tabConfig.forEach(({ name, button, panel }) => {
        const active = name === nextTab;
        if (button) {
            button.setAttribute('role', 'tab');
            button.setAttribute('aria-selected', active ? 'true' : 'false');
            button.tabIndex = active ? 0 : -1;
            button.className = active
                ? 'rounded-full border border-cyan-200 dark:border-cyan-900/60 bg-cyan-50 dark:bg-cyan-900/20 px-3 py-1.5 text-[10px] font-bold uppercase tracking-[0.18em] text-cyan-700 dark:text-cyan-300 transition-colors'
                : 'rounded-full border border-slate-200 dark:border-slate-800 px-3 py-1.5 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400 transition-colors';
        }
        if (panel) {
            panel.classList.toggle('hidden', !active);
            panel.setAttribute('role', 'tabpanel');
            panel.setAttribute('aria-hidden', active ? 'false' : 'true');
        }
    });

    if (els.logsShell) {
        const showLogsShell = nextTab === 'logs' || state.currentView === 'review';
        els.logsShell.classList.toggle('hidden', !showLogsShell);
        els.logsShell.classList.toggle('flex', showLogsShell);
    }
}

function _derivePresetResearchFlag(preset, fallbackName = '') {
    if (preset && Object.prototype.hasOwnProperty.call(preset, 'is_research')) {
        return Boolean(preset.is_research);
    }
    return String(preset?.name || preset?.label || fallbackName || '')
        .trim()
        .toLowerCase()
        .includes('research');
}

function currentPresetDescriptor() {
    if (state.pipeline !== 'lux-depth-v3') {
        return {
            name: 'default',
            label: 'default',
            stability: 'stable',
            description: 'Archive governance workflow preset.',
            is_research: false,
            recommended_args: {},
            advanced_sections: []
        };
    }
    const presets = Array.isArray(state.presetsByPipeline[state.pipeline]) ? state.presetsByPipeline[state.pipeline] : [];
    const selected = presets.find((preset) => String(preset.name) === String(state.config.preset));
    if (selected) return selected;
    return {
        name: state.config.preset || 'custom',
        label: state.config.preset || 'custom',
        stability: 'custom',
        description: state.pipeline === 'lux-depth-v3'
            ? 'Custom or offline preset selection.'
            : 'Archive governance workflow preset.',
        is_research: _derivePresetResearchFlag({
            name: state.config.preset || 'custom',
            label: state.config.preset || 'custom'
        }),
        recommended_args: {},
        advanced_sections: []
    };
}

function canonicalArchiveCommand(pipelineName) {
    if (pipelineName === 'archive-gate-a') return 'fixity-scan';
    if (pipelineName === 'archive-gate-b') return 'bag-build';
    if (pipelineName === 'archive-gate-c') return 'mets-export';
    return '';
}

function _setContextVisibility(el, visible) {
    if (!el) return;
    el.classList.toggle('hidden', !visible);
}

function _presetRequiresResearchAcknowledgments(preset, args = {}) {
    return _derivePresetResearchFlag(preset, args.preset);
}

function _currentPreviewReadiness(payload = null) {
    const preview = _currentPreviewForPayload(payload);
    return preview && preview.readiness && typeof preview.readiness === 'object'
        ? preview.readiness
        : null;
}

function currentPipelineReadiness(payload = null) {
    const previewReadiness = _currentPreviewReadiness(payload);
    if (previewReadiness) return previewReadiness;
    return state.readiness && state.readiness.pipelines
        ? state.readiness.pipelines[state.pipeline] || null
        : null;
}

function _currentPipelineLocalPrereqs() {
    return {
        archiveIndex: _textOrFallback(
            els.archiveIndexPath ? els.archiveIndexPath.value : state.config?.gate?.archiveIndex,
            state.config?.gate?.archiveIndex || ''
        ),
        manifestJsonl: _textOrFallback(
            els.rightsManifestPath ? els.rightsManifestPath.value : state.config?.gate?.manifestJsonl,
            state.config?.gate?.manifestJsonl || ''
        )
    };
}

function _readinessIssueSatisfiedLocally(issue) {
    if (!issue || typeof issue !== 'object') return false;
    const reason = String(issue.reason || '').trim().toLowerCase();
    const prereqs = _currentPipelineLocalPrereqs();
    if (reason === 'archive_index_required') return Boolean(prereqs.archiveIndex);
    if (reason === 'rights_manifest_required' || reason === 'manifest_jsonl_required') {
        return Boolean(prereqs.manifestJsonl);
    }
    return false;
}

function currentPipelineReadinessIssues(payload = null) {
    const previewReadiness = _currentPreviewReadiness(payload);
    const readiness = previewReadiness || currentPipelineReadiness(payload);
    const issues = Array.isArray(readiness?.missing_prerequisites) ? readiness.missing_prerequisites : [];
    if (previewReadiness) return issues;
    return issues.filter((issue) => !_readinessIssueSatisfiedLocally(issue));
}

function currentPipelineDispatchStatus(payload = null) {
    const readiness = currentPipelineReadiness(payload);
    const rawStatus = String(readiness?.status || '').trim().toLowerCase();
    if (!rawStatus) return '';
    const issues = currentPipelineReadinessIssues(payload);
    if (issues.some((issue) => String(issue?.severity || '').trim().toLowerCase() === 'blocked')) {
        return 'blocked';
    }
    if (issues.some((issue) => String(issue?.reason || '').trim().toLowerCase() === 'archive_index_required')) {
        return 'degraded';
    }
    if (issues.some((issue) => String(issue?.severity || '').trim().toLowerCase() === 'degraded')) {
        return 'degraded';
    }
    if (rawStatus === 'blocked' || rawStatus === 'degraded') return 'ready';
    return rawStatus;
}

function renderCapabilityChips(payload) {
    if (!els.capabilityChips) return;
    const args = payload?.args || {};
    const chips = [];
    const readiness = currentPipelineReadiness();
    const dispatchStatus = currentPipelineDispatchStatus();
    if (state.pipeline === 'lux-depth-v3') {
        chips.push(`Preset ${String(args.preset || state.config.preset || 'custom')}`);
        if (readiness?.canary_status) chips.push(`Canary ${titleCaseToken(readiness.canary_status, 'Unknown')}`);
    } else {
        chips.push(`Command ${canonicalArchiveCommand(state.pipeline)}`);
    }
    chips.push(state.backendOk ? 'Live backend connected' : 'Backend offline');
    if (dispatchStatus || readiness?.status) chips.push(`Readiness ${titleCaseToken(dispatchStatus || readiness.status, 'Unknown')}`);
    if (parseBoolLike(args.enable_segmentation, false)) {
        chips.push(`Segmentation ${String(args.segmentation_backend || 'enabled')}`);
    }
    if (parseBoolLike(args.materials_v3, false)) chips.push('Materials V3');
    if (parseBoolLike(args.pbr, false)) chips.push('PBR generation');
    if (parseBoolLike(args.enable_reconstruction, false)) chips.push('Scene reconstruction');
    if (parseBoolLike(args.emit_run_card, false)) chips.push('Run card emission');

    els.capabilityChips.innerHTML = '';
    chips.forEach((chip) => {
        const tag = document.createElement('span');
        tag.className = 'accent-chip';
        tag.textContent = chip;
        els.capabilityChips.appendChild(tag);
    });
}

function syncBuildSurfaceApplicability(payload = null) {
    const args = payload?.args || generatePayload().args || {};
    const preset = currentPresetDescriptor();
    const isLuxPipeline = state.pipeline === 'lux-depth-v3';
    const segmentationEnabled = parseBoolLike(args.enable_segmentation, false);
    const segmentationBackend = _resolveSegmentationBackend(args.segmentation_backend || state.config.segmentation?.backend);
    const showSam2Controls = segmentationEnabled && segmentationBackend === 'sam2';
    const enableV2 = parseBoolLike(args.enable_v2, false);
    const reconstructionEnabled = parseBoolLike(args.enable_reconstruction, false);
    const researchPreset = _presetRequiresResearchAcknowledgments(preset, args);
    const depthBackend = String(args.depth_backend || '').trim().toLowerCase();
    const nonCommercialChecked = parseBoolLike(args.non_commercial_ok, false);
    const appleChecked = parseBoolLike(args.accept_apple_depth_pro_research_license, false);
    const researchToolsChecked = parseBoolLike(args.accept_research_tools_license, false);
    const nonCommercialRequired = researchPreset || depthBackend === 'depth_pro' || reconstructionEnabled;
    const appleRequired = depthBackend === 'depth_pro';
    const researchToolsRequired = reconstructionEnabled;
    const governanceVisible = isLuxPipeline && (
        nonCommercialRequired
        || appleRequired
        || researchToolsRequired
        || nonCommercialChecked
        || appleChecked
        || researchToolsChecked
    );

    _setContextVisibility(els.segmentationBackendField, isLuxPipeline && segmentationEnabled);
    _setContextVisibility(els.strictSegmentationField, isLuxPipeline && segmentationEnabled);
    _setContextVisibility(els.sam2ModelSizeField, isLuxPipeline && showSam2Controls);
    _setContextVisibility(els.sam2CheckpointField, isLuxPipeline && showSam2Controls);
    if (els.segmentationApplicabilityHint) {
        els.segmentationApplicabilityHint.textContent = !isLuxPipeline
            ? ''
            : !segmentationEnabled
                ? 'Turn segmentation on to choose a backend and strictness policy. SAM2-only controls appear when that backend is selected.'
                : showSam2Controls
                    ? 'SAM2 is active, so model size and checkpoint controls are now available.'
                    : `Segmentation is active via ${segmentationBackend}. SAM2-only controls stay hidden until you switch back to sam2.`;
    }

    _setContextVisibility(els.v2PresetField, isLuxPipeline && enableV2);
    if (els.v2Preset) {
        els.v2Preset.disabled = !enableV2;
    }

    _setContextVisibility(els.governanceDetails, governanceVisible);
    _setContextVisibility(
        els.licenseNonCommercialField,
        governanceVisible && (nonCommercialRequired || nonCommercialChecked)
    );
    _setContextVisibility(
        els.licenseAppleField,
        governanceVisible && (appleRequired || appleChecked)
    );
    _setContextVisibility(
        els.licenseResearchToolsField,
        governanceVisible && (researchToolsRequired || researchToolsChecked)
    );
    if (els.governanceDetailsHint) {
        if (!governanceVisible) {
            els.governanceDetailsHint.textContent = 'Only shown when the current preset or backend requires explicit acknowledgments.';
        } else if (appleRequired && researchToolsRequired) {
            els.governanceDetailsHint.textContent = 'This run needs both Depth Pro and reconstruction acknowledgments before dispatch.';
        } else if (appleRequired) {
            els.governanceDetailsHint.textContent = 'This run needs Depth Pro research acknowledgments before dispatch.';
        } else if (researchToolsRequired) {
            els.governanceDetailsHint.textContent = 'This run needs reconstruction acknowledgments before dispatch.';
        } else {
            els.governanceDetailsHint.textContent = 'This research preset needs a non-commercial acknowledgment before dispatch.';
        }
    }

    _setContextVisibility(els.reconstructionConfigFields, reconstructionEnabled);
    _setContextVisibility(els.runtimeTuningFields, isLuxPipeline);
    if (els.reconstructionDetailsHint) {
        els.reconstructionDetailsHint.textContent = reconstructionEnabled
            ? 'Scene reconstruction is active. Grouping, sidecar, tier, and debug controls are now available.'
            : 'Runtime tuning stays available here. Reconstruction-only settings stay preserved and inactive until you enable the feature.';
    }
}

function _setDisclosureSummaryBadge(element, text) {
    if (!element) return;
    element.textContent = String(text || '').trim() || 'Optional';
}

function syncDisclosurePanels(payload = null) {
    const args = payload?.args || generatePayload().args || {};
    const preset = currentPresetDescriptor();
    const advancedSections = Array.isArray(preset.advanced_sections) ? preset.advanced_sections : [];
    const researchPreset = _presetRequiresResearchAcknowledgments(preset, args);
    const reconstructionEnabled = parseBoolLike(args.enable_reconstruction, false);
    const depthBackend = String(args.depth_backend || '').trim().toLowerCase();
    const previewFieldGroups = {
        advanced: [
            'save_float_depth',
            'force_depth',
            'strict_inputs',
            'verify_images',
            'allow_semantic_fallback',
            'verbose',
            'quiet',
            'raw_ingest_mode',
            'raw_wb_mode',
            'raw_demosaic',
            'max_workers',
            'max_gpu_workers',
            'log_level',
        ],
        governance: [
            'preset',
            'depth_backend',
            'non_commercial_ok',
            'accept_apple_depth_pro_research_license',
            'accept_research_tools_license',
        ],
        reconstruction: [
            'enable_reconstruction',
            'grouping_mode',
            'cameras_sidecar_path',
            'reconstruction_iterations',
            'reconstruction_tier',
            'emit_scene_debug_bundle',
            'raw_ingest_mode',
            'max_workers',
            'max_gpu_workers',
            'log_level',
        ],
    };
    const advancedActive = parseBoolLike(args.save_float_depth, false)
        || parseBoolLike(args.force_depth, false)
        || parseBoolLike(args.strict_inputs, false)
        || parseBoolLike(args.verify_images, false)
        || parseBoolLike(args.allow_semantic_fallback, false)
        || parseBoolLike(args.verbose, false)
        || parseBoolLike(args.quiet, false)
        || String(args.log_level || '').trim() !== ''
        || String(args.max_workers || '').trim() !== ''
        || String(args.max_gpu_workers || '').trim() !== '';
    const governanceActive = researchPreset
        || depthBackend === 'depth_pro'
        || reconstructionEnabled
        || parseBoolLike(args.non_commercial_ok, false)
        || parseBoolLike(args.accept_apple_depth_pro_research_license, false)
        || parseBoolLike(args.accept_research_tools_license, false)
        || advancedSections.includes('governance');
    const reconstructionActive = reconstructionEnabled
        || String(args.cameras_sidecar_path || '').trim() !== ''
        || String(args.grouping_mode || 'single').trim().toLowerCase() !== 'single'
        || String(args.raw_ingest_mode || 'auto').trim().toLowerCase() !== 'auto'
        || String(args.max_workers || '').trim() !== ''
        || String(args.max_gpu_workers || '').trim() !== ''
        || String(args.log_level || '').trim() !== '';
    const hasPreviewIssueForGroup = (groupName) => previewFieldGroups[groupName].some((fieldName) => Boolean(_previewIssueForField(fieldName, payload)));
    const disclosurePrefs = state.portalUi.disclosurePrefs || {};
    const autoOpenState = {
        advanced: advancedActive || advancedSections.includes('advanced') || hasPreviewIssueForGroup('advanced'),
        governance: governanceActive || advancedSections.includes('governance') || hasPreviewIssueForGroup('governance'),
        reconstruction: reconstructionActive || advancedSections.includes('reconstruction') || hasPreviewIssueForGroup('reconstruction'),
        dispatchTools: false,
    };
    const syncPanel = (name, element) => {
        if (!element) return;
        const shouldOpen = name === 'dispatchTools'
            ? disclosurePrefs.dispatchTools === true
            : autoOpenState[name] || disclosurePrefs[name] === true;
        element.dataset.autoOpen = autoOpenState[name] ? 'true' : 'false';
        if (element.open !== shouldOpen) {
            element.open = shouldOpen;
        }
    };

    syncPanel('advanced', els.advancedFlagsDetails);
    syncPanel('governance', els.governanceDetails);
    syncPanel('reconstruction', els.reconstructionDetails);
    syncPanel('dispatchTools', els.dispatchToolsDetails);

    const currentPreview = _currentPreviewForPayload(payload || generatePayload());
    _setDisclosureSummaryBadge(
        els.advancedFlagsSummary,
        hasPreviewIssueForGroup('advanced') ? 'Needs review' : advancedActive ? 'Active' : 'Optional'
    );
    _setDisclosureSummaryBadge(
        els.governanceDetailsSummary,
        hasPreviewIssueForGroup('governance') || researchPreset || depthBackend === 'depth_pro'
            ? 'Required'
            : governanceActive
                ? 'Active'
                : 'Conditional'
    );
    _setDisclosureSummaryBadge(
        els.reconstructionDetailsSummary,
        hasPreviewIssueForGroup('reconstruction')
            ? 'Needs review'
            : reconstructionEnabled
                ? 'Enabled'
                : reconstructionActive
                    ? 'Configured'
                    : 'Runtime baseline'
    );
    _setDisclosureSummaryBadge(
        els.dispatchToolsSummary,
        els.dispatchToolsDetails?.open
            ? 'Open'
            : currentPreview?.status === 'loading'
                ? 'Preview loading'
                : currentPreview?.status === 'error'
                    ? 'Preview error'
                    : 'Collapsed'
    );
}

function renderGovernanceBanner(payload) {
    if (!els.governanceChecklist) return;
    const args = payload?.args || {};
    const dispatchStatus = currentPipelineDispatchStatus();
    const readinessIssues = currentPipelineReadinessIssues();
    const items = [];
    let title = 'Run posture is clear.';
    let body = 'The current configuration is valid for dispatch. Any license acknowledgments or runtime caveats will surface here before execution.';

    if (state.pipeline === 'lux-depth-v3') {
        if (String(args.preset || '').toLowerCase().includes('v3.1')) {
            items.push({
                tone: parseBoolLike(args.non_commercial_ok, false) ? 'ok' : 'warn',
                label: 'Research preset',
                detail: parseBoolLike(args.non_commercial_ok, false)
                    ? 'Non-commercial acknowledgment is already set.'
                    : 'This preset requires non-commercial acknowledgment before dispatch.'
            });
        }
        if (String(args.depth_backend || '').toLowerCase() === 'depth_pro') {
            const ok = parseBoolLike(args.non_commercial_ok, false) && parseBoolLike(args.accept_apple_depth_pro_research_license, false);
            items.push({
                tone: ok ? 'ok' : 'warn',
                label: 'Depth Pro license',
                detail: ok
                    ? 'Depth Pro research acknowledgments are complete.'
                    : 'Depth Pro requires non-commercial and Apple research-license acknowledgment.'
            });
        }
        if (parseBoolLike(args.enable_reconstruction, false)) {
            const ok = parseBoolLike(args.non_commercial_ok, false) && parseBoolLike(args.accept_research_tools_license, false);
            items.push({
                tone: ok ? 'ok' : 'warn',
                label: 'Reconstruction governance',
                detail: ok
                    ? 'Scene reconstruction acknowledgments are complete.'
                    : 'Scene reconstruction requires non-commercial and research-tools acknowledgment.'
            });
        }
        if (parseBoolLike(args.materials_v3, false) && String(args.quality_tier || '').toLowerCase() === 'apex') {
            const segmentationReady = parseBoolLike(args.enable_segmentation, false)
                && String(args.segmentation_backend || '').toLowerCase() !== 'stub'
                && parseBoolLike(args.strict_segmentation, false);
            items.push({
                tone: segmentationReady ? 'ok' : 'warn',
                label: 'APEX materials guardrail',
                detail: segmentationReady
                    ? 'Segmentation prerequisites for APEX materials are satisfied.'
                    : 'APEX + Materials V3 expects enabled strict segmentation with a non-stub backend.'
            });
        }
    } else {
        items.push({
            tone: dispatchStatus === 'blocked' ? 'warn' : dispatchStatus === 'degraded' ? 'info' : 'ok',
            label: 'Archive governance',
            detail: dispatchStatus === 'blocked'
                ? 'Dispatch is blocked until the canonical archive prerequisites are supplied.'
                : dispatchStatus === 'degraded'
                    ? 'Archive stage is available, but it still requires operator-supplied prerequisites at dispatch time.'
                    : 'This workflow stays within deterministic archive and provenance outputs.'
        });
    }

    items.push({
        tone: state.backendOk ? 'ok' : 'info',
        label: state.backendOk ? 'Backend live' : 'Backend offline',
        detail: state.backendOk
            ? 'Jobs will be submitted to the live orchestrator API.'
            : 'Execution dispatch stays disabled until the live orchestrator API is reachable.'
    });

    readinessIssues.forEach((issue) => {
        items.push({
            tone: issue.severity === 'blocked' ? 'warn' : 'info',
            label: titleCaseToken(issue.reason || 'readiness issue', 'Readiness issue'),
            detail: String(issue.message || 'Pipeline readiness reported an operator-facing prerequisite.')
        });
    });

    const hasWarnings = items.some((item) => item.tone === 'warn');
    if (hasWarnings) {
        title = 'Execution is blocked.';
        body = 'Before dispatch, satisfy the blocked readiness prerequisites listed below.';
    } else if (items.some((item) => item.tone === 'info')) {
        title = 'Configuration is valid with readiness caveats.';
        body = 'The run contract is coherent, but at least one prerequisite still needs operator attention before dispatch can be treated as fully ready.';
    }

    if (els.governanceBannerTitle) els.governanceBannerTitle.textContent = title;
    if (els.governanceBannerBody) els.governanceBannerBody.textContent = body;

    els.governanceChecklist.innerHTML = '';
    items.forEach((item) => {
        const card = document.createElement('div');
        const toneClass = item.tone === 'warn'
            ? 'border-amber-200 bg-amber-50/90 dark:border-amber-900/60 dark:bg-amber-900/20'
            : item.tone === 'ok'
                ? 'border-emerald-200 bg-emerald-50/90 dark:border-emerald-900/60 dark:bg-emerald-900/20'
                : 'border-slate-200 bg-white/80 dark:border-slate-700 dark:bg-slate-900/60';
        card.className = `governance-item ${toneClass}`;

        const heading = document.createElement('p');
        heading.className = 'text-[10px] font-extrabold uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400';
        heading.textContent = item.label;
        card.appendChild(heading);

        const detail = document.createElement('p');
        detail.className = 'mt-2 text-[12px] leading-6 text-slate-700 dark:text-slate-200';
        detail.textContent = item.detail;
        card.appendChild(detail);

        els.governanceChecklist.appendChild(card);
    });
}

function renderPresetIntelligence(payload) {
    const preset = currentPresetDescriptor();
    const diagnostics = state.lastDiagnostics || {};
    const warnings = Array.isArray(diagnostics.warnings) ? diagnostics.warnings : [];
    const readiness = currentPipelineReadiness();
    const dispatchStatus = currentPipelineDispatchStatus();
    const readinessIssues = currentPipelineReadinessIssues();
    const advancedSections = Array.isArray(preset.advanced_sections) ? preset.advanced_sections : [];

    if (els.presetHeadline) els.presetHeadline.textContent = String(preset.label || preset.name || 'custom');
    if (els.presetDescription) {
        els.presetDescription.textContent = String(
            preset.description ||
            (state.pipeline === 'lux-depth-v3'
                ? 'Preset metadata unavailable while offline.'
                : 'Archive governance preset.')
        );
    }
    if (els.presetStabilityBadge) {
        els.presetStabilityBadge.textContent = titleCaseToken(preset.stability, 'Custom');
    }
    if (els.backendModeBadge) {
        const readinessLabel = dispatchStatus || readiness?.status ? titleCaseToken(dispatchStatus || readiness?.status, 'Unknown') : 'Unknown';
        els.backendModeBadge.textContent = state.backendOk ? `Backend Live • ${readinessLabel}` : 'Backend Offline';
    }
    if (els.heroInputDir) {
        els.heroInputDir.textContent = truncateMiddle(payload?.args?.input_dir || state.config.inputDir || './input_images', 52);
        els.heroInputDir.title = String(payload?.args?.input_dir || state.config.inputDir || './input_images');
    }
    if (els.heroOutputDir) {
        els.heroOutputDir.textContent = truncateMiddle(payload?.args?.output_dir || state.config.outputDir || './output', 52);
        els.heroOutputDir.title = String(payload?.args?.output_dir || state.config.outputDir || './output');
    }
    if (els.heroReadinessLabel) {
        if (!state.backendOk) {
            els.heroReadinessLabel.textContent = 'Backend offline';
        } else if (dispatchStatus === 'blocked') {
            els.heroReadinessLabel.textContent = 'Dispatch blocked';
        } else if (dispatchStatus === 'degraded') {
            els.heroReadinessLabel.textContent = 'Prerequisites still required';
        } else if (warnings.length === 0) {
            els.heroReadinessLabel.textContent = 'Ready for dispatch';
        } else {
            els.heroReadinessLabel.textContent = titleCaseToken(diagnostics.healthLabel || 'warnings detected', 'Warnings detected');
        }
    }
    if (els.heroWarningCount) {
        els.heroWarningCount.textContent = String(warnings.length + readinessIssues.length);
    }
    if (els.presetBuilderHint) {
        const sectionLabel = advancedSections.length > 0
            ? `Advanced focus: ${advancedSections.join(', ')}.`
            : 'Preset keeps advanced sections collapsed until the run needs them.';
        els.presetBuilderHint.textContent = `${String(preset.description || 'Preset metadata unavailable.')} ${sectionLabel}`;
    }
}

function renderMissionControl(payload = null) {
    const currentPayload = payload || generatePayload();
    const activeJob = _latestActiveJob();
    const reviewJob = _latestReviewableJob();
    const activeJobs = state.jobs.filter((job) => job && (job.state === 'running' || job.state === 'queued')).length;
    syncBuildSurfaceApplicability(currentPayload);
    if (els.heroPipelineValue) els.heroPipelineValue.textContent = String(state.pipeline || 'lux-depth-v3');
    if (els.heroPresetValue) {
        els.heroPresetValue.textContent = state.pipeline === 'lux-depth-v3'
            ? String(currentPayload?.args?.preset || state.config.preset || 'custom')
            : canonicalArchiveCommand(state.pipeline) || 'archive';
    }
    if (els.heroModeValue) {
        const readiness = currentPipelineReadiness();
        const dispatchStatus = currentPipelineDispatchStatus();
        els.heroModeValue.textContent = state.backendOk
            ? `Live backend • ${titleCaseToken(dispatchStatus || readiness?.status || 'unknown', 'Unknown')}`
            : 'Backend offline';
    }
    if (els.heroQueueValue) {
        els.heroQueueValue.textContent = activeJobs > 0 ? `${activeJobs} live job${activeJobs === 1 ? '' : 's'}` : '0 live jobs';
    }
    if (els.heroRunBtn) {
        els.heroRunBtn.disabled = false;
    }
    if (els.resumeDraftBtn) {
        els.resumeDraftBtn.disabled = false;
    }
    if (els.heroExportBtn) {
        els.heroExportBtn.disabled = !activeJob;
        els.heroExportBtn.dataset.jobId = activeJob ? String(activeJob.id || '') : '';
    }
    if (els.refreshHealthBtn) {
        els.refreshHealthBtn.disabled = !reviewJob;
        els.refreshHealthBtn.dataset.jobId = reviewJob ? String(reviewJob.id || '') : '';
    }

    renderCapabilityChips(currentPayload);
    renderPresetIntelligence(currentPayload);
    renderGovernanceBanner(currentPayload);
    syncDisclosurePanels(currentPayload);
    _syncOverviewBuildLoadingState(currentPayload);
}

function _isJobsHydrationPending() {
    if (state.jobs.length > 0) return false;
    if (state.jobsLoadStatus === 'loading') return true;
    return state.jobsLoadStatus === 'pending'
        && (state.bootstrap.status === 'pending' || state.bootstrap.status === 'degraded');
}

function _toggleSurfaceSkeleton(container, content, skeleton, isLoading) {
    if (container) {
        container.setAttribute('aria-busy', isLoading ? 'true' : 'false');
    }
    if (content) {
        content.classList.toggle('hidden', isLoading);
    }
    if (skeleton) {
        skeleton.classList.toggle('hidden', !isLoading);
        skeleton.setAttribute('aria-hidden', 'true');
    }
}

function _setSurfaceLoadingState(container, isLoading) {
    if (!container) return;
    container.classList.toggle('surface-loading', isLoading);
    container.setAttribute('data-surface-loading', isLoading ? 'true' : 'false');
}

function _isBootstrapSurfaceLoading() {
    return state.bootstrap.status === 'pending';
}

function _isBuildPreviewRefreshing(payload = null) {
    if (_isBootstrapSurfaceLoading() || !_isBootstrapReady()) return false;
    const currentPayload = payload || generatePayload();
    const preview = _currentPreviewForPayload(currentPayload);
    return Boolean(preview && preview.status === 'loading');
}

function _syncOverviewBuildLoadingState(payload = null) {
    const currentPayload = payload || generatePayload();
    const bootstrapLoading = _isBootstrapSurfaceLoading();
    const previewRefreshing = _isBuildPreviewRefreshing(currentPayload);
    const shellBusy = bootstrapLoading || previewRefreshing;

    _toggleSurfaceSkeleton(els.missionShell, els.missionShellContent, els.missionShellSkeletonState, bootstrapLoading);
    _toggleSurfaceSkeleton(els.intelligenceShell, els.intelligenceShellContent, els.intelligenceShellSkeletonState, bootstrapLoading);
    _toggleSurfaceSkeleton(els.profileShell, els.profileShellContent, els.profileShellSkeletonState, bootstrapLoading);
    _toggleSurfaceSkeleton(els.buildStepperShell, els.buildStepperShellContent, els.buildStepperSkeletonState, bootstrapLoading);
    _toggleSurfaceSkeleton(els.parametersShell, els.parametersShellContent, els.parametersShellSkeletonState, bootstrapLoading);

    [
        els.missionShell,
        els.intelligenceShell,
        els.profileShell,
        els.buildStepperShell,
        els.governanceShell,
        els.parametersShell
    ].forEach((container) => {
        if (!container) return;
        _setSurfaceLoadingState(container, previewRefreshing && !bootstrapLoading);
        container.setAttribute('aria-busy', shellBusy ? 'true' : 'false');
    });

    [els.overviewShell, els.consoleContextShell, els.buildShell].forEach((container) => {
        if (!container) return;
        container.setAttribute('aria-busy', shellBusy ? 'true' : 'false');
    });

    if (document.body) {
        document.body.dataset.bootstrapLoading = bootstrapLoading ? 'true' : 'false';
        document.body.dataset.buildPreviewLoading = previewRefreshing ? 'true' : 'false';
    }
}

function renderSelectedJobTimeline(job) {
    if (!els.selectedJobTimelineList || !els.selectedJobTimelineEmpty) return;
    if (!job) {
        els.selectedJobTimelineList.innerHTML = '';
        els.selectedJobTimelineEmpty.classList.remove('hidden');
        return;
    }

    _reconcileJobTimeline(job);
    const entries = Array.isArray(job.timeline) ? [...job.timeline] : [];
    if (entries.length === 0) {
        els.selectedJobTimelineList.innerHTML = '';
        els.selectedJobTimelineEmpty.classList.remove('hidden');
        return;
    }

    els.selectedJobTimelineEmpty.classList.add('hidden');
    const fragment = document.createDocumentFragment();
    entries.forEach((entry) => {
        const toneClass = entry.tone === 'error'
            ? 'border-red-200 bg-red-50/90 dark:border-red-900/50 dark:bg-red-900/10'
            : entry.tone === 'warn'
                ? 'border-amber-200 bg-amber-50/90 dark:border-amber-900/50 dark:bg-amber-900/10'
                : entry.tone === 'success'
                    ? 'border-emerald-200 bg-emerald-50/90 dark:border-emerald-900/50 dark:bg-emerald-900/10'
                    : 'border-slate-200 bg-white/80 dark:border-slate-800 dark:bg-slate-900/60';
        const item = document.createElement('li');
        item.className = `rounded-2xl border px-4 py-3 ${toneClass}`;

        const header = document.createElement('div');
        header.className = 'flex items-start justify-between gap-3';

        const label = document.createElement('p');
        label.className = 'text-[11px] font-bold uppercase tracking-[0.18em] text-slate-600 dark:text-slate-300';
        label.textContent = String(entry.label || 'Timeline entry');
        header.appendChild(label);

        const timestamp = document.createElement('span');
        timestamp.className = 'text-[10px] font-mono text-slate-500 dark:text-slate-400';
        timestamp.textContent = formatTimelineTimestamp(Number(entry.timestamp || 0));
        header.appendChild(timestamp);
        item.appendChild(header);

        const detail = document.createElement('p');
        detail.className = 'mt-2 text-[12px] leading-6 text-slate-600 dark:text-slate-300';
        detail.textContent = String(entry.detail || '');
        item.appendChild(detail);

        fragment.appendChild(item);
    });

    els.selectedJobTimelineList.innerHTML = '';
    els.selectedJobTimelineList.appendChild(fragment);
}

function renderSelectedJobInspector() {
    const jobsLoading = _isJobsHydrationPending();
    _toggleSurfaceSkeleton(els.selectedJobShell, els.selectedJobShellContent, els.selectedJobSkeletonState, jobsLoading);
    if (jobsLoading) {
        if (els.selectedJobStateBadge) els.selectedJobStateBadge.textContent = 'Syncing';
        if (els.selectedJobFreshness) els.selectedJobFreshness.textContent = 'Hydrating queue';
        if (els.selectedJobMetaLine) {
            els.selectedJobMetaLine.textContent = 'Recovering recent runs, transport state, and previewable outputs.';
        }
        if (els.openRunDetailsBtn) els.openRunDetailsBtn.disabled = true;
        renderConsoleContextRibbon();
        return;
    }

    const selected = state.jobs.find((job) => job.id === state.selectedJobId) || null;
    if (!selected) {
        if (els.selectedJobStateBadge) els.selectedJobStateBadge.textContent = 'Idle';
        if (els.selectedJobIdLabel) els.selectedJobIdLabel.textContent = 'No job selected';
        if (els.selectedJobPipelineLabel) els.selectedJobPipelineLabel.textContent = 'Awaiting dispatch';
        if (els.selectedJobArtifactCount) els.selectedJobArtifactCount.textContent = '0 indexed';
        if (els.selectedJobStreamStatus) els.selectedJobStreamStatus.textContent = 'Inactive';
        if (els.selectedJobProgressText) els.selectedJobProgressText.textContent = '0%';
        if (els.selectedJobProgressBar) els.selectedJobProgressBar.value = 0;
        if (els.selectedJobMetaLine) els.selectedJobMetaLine.textContent = 'Queue idle. Select a run to inspect transport, recency, and output state.';
        if (els.selectedJobFreshness) els.selectedJobFreshness.textContent = 'No live telemetry';
        if (els.logMetaLabel) els.logMetaLabel.textContent = 'Select a job to stream or inspect its log output.';
        if (els.selectedJobSummary) els.selectedJobSummary.textContent = 'Choose a job from the queue or dispatch a new run to inspect progress, artifacts, and live stream state.';
        if (els.selectedJobTransportAlert) {
            els.selectedJobTransportAlert.classList.add('hidden');
            els.selectedJobTransportAlert.textContent = '';
        }
        if (els.openRunDetailsBtn) els.openRunDetailsBtn.disabled = true;
        if (els.selectedJobLogPreview) {
            els.selectedJobLogPreview.textContent = 'No live logs yet.';
        }
        renderSelectedJobTimeline(null);
        setInspectorTab(state.inspectorTab);
        renderConsoleContextRibbon();
        return;
    }

    _reconcileJobTimeline(selected);
    const artifactCount = Array.isArray(selected.artifacts) ? selected.artifacts.length : 0;
    const readableError = getReadableError(selected.error);
    const outcomeSummary = jobOutcomeSummary(selected);
    const nativeReadyState = _nativeEventSourceReadyState(selected.eventSource);
    const streamStatus = selected.reconnectBlocked
        ? 'Blocked by auth'
        : selected.usesFetchSse && _jobHasActiveStream(selected)
            ? 'Fetch stream active'
            : nativeReadyState === EVENT_SOURCE_READY_STATE_CONNECTING
                ? 'SSE stream reconnecting'
                : nativeReadyState === EVENT_SOURCE_READY_STATE_OPEN
                    ? 'SSE stream active'
                    : nativeReadyState === EVENT_SOURCE_READY_STATE_CLOSED
                        ? 'SSE stream closed'
                        : _jobHasActiveStream(selected)
                            ? 'SSE stream active'
                            : (selected.state === 'running' || selected.state === 'queued' ? 'Waiting for stream' : 'Closed');
    const lastActivityAt = Number(selected.lastEventAt || selected.updatedAt || selected.createdAt || 0);
    const activityLabel = formatRelativeTime(lastActivityAt);
    const transportLabel = formatTransportLabel(selected);
    const elapsedLabel = formatDuration(Number(selected.createdAt || 0), Number(selected.finishedAt || Date.now()));
    const latestWarning = Array.isArray(selected.transportWarnings) && selected.transportWarnings.length > 0
        ? selected.transportWarnings[selected.transportWarnings.length - 1]
        : null;
    const visibleAlert = latestWarning && latestWarning.tone !== 'info' ? latestWarning : null;

    if (els.selectedJobStateBadge) els.selectedJobStateBadge.textContent = titleCaseToken(selected.state, 'Unknown');
    if (els.selectedJobIdLabel) {
        els.selectedJobIdLabel.textContent = String(selected.id || 'unknown');
        els.selectedJobIdLabel.title = String(selected.id || 'unknown');
    }
    if (els.selectedJobPipelineLabel) els.selectedJobPipelineLabel.textContent = String(selected.pipeline || 'unknown');
    if (els.selectedJobArtifactCount) els.selectedJobArtifactCount.textContent = `${artifactCount} indexed`;
    if (els.selectedJobStreamStatus) els.selectedJobStreamStatus.textContent = `${streamStatus} • ${elapsedLabel}`;
    if (els.selectedJobProgressText) els.selectedJobProgressText.textContent = `${Math.max(0, Math.min(100, Number(selected.progress) || 0))}%`;
    if (els.selectedJobProgressBar) {
        els.selectedJobProgressBar.max = 100;
        els.selectedJobProgressBar.value = Math.max(0, Math.min(100, Number(selected.progress) || 0));
    }
    if (els.selectedJobMetaLine) {
        els.selectedJobMetaLine.textContent = `${titleCaseToken(selected.pipeline, 'Unknown')} • ${transportLabel} • ${elapsedLabel}`;
    }
    if (els.selectedJobFreshness) {
        els.selectedJobFreshness.textContent = `Updated ${activityLabel}`;
    }
    if (els.logMetaLabel) {
        els.logMetaLabel.textContent = `${String(selected.pipeline || 'unknown')} • ${transportLabel} • ${artifactCount} artifact${artifactCount === 1 ? '' : 's'}`;
    }
    if (els.selectedJobSummary) {
        els.selectedJobSummary.textContent = selected.state === 'partial' && outcomeSummary
            ? `${outcomeSummary}.`
            : readableError
                ? readableError
                : outcomeSummary
                    ? `${outcomeSummary}.`
                    : `Operators can now read ${titleCaseToken(selected.state, 'job')} state at a glance: ${artifactCount} artifact${artifactCount === 1 ? '' : 's'} indexed, ${transportLabel} transport, ${elapsedLabel}.`;
    }
    if (els.selectedJobTransportAlert) {
        if (visibleAlert) {
            els.selectedJobTransportAlert.textContent = String(visibleAlert.detail || '');
            els.selectedJobTransportAlert.classList.remove('hidden');
        } else {
            els.selectedJobTransportAlert.classList.add('hidden');
            els.selectedJobTransportAlert.textContent = '';
        }
    }
    if (els.openRunDetailsBtn) els.openRunDetailsBtn.disabled = false;
    if (els.selectedJobLogPreview) {
        const previewLines = Array.isArray(selected.logs) ? selected.logs.slice(-12) : [];
        els.selectedJobLogPreview.textContent = previewLines.length > 0 ? previewLines.join('\n') : 'No live logs yet.';
    }

    renderSelectedJobTimeline(selected);
    setInspectorTab(state.inspectorTab);
    renderConsoleContextRibbon();
}

function createToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast-enter pointer-events-auto rounded-xl border px-4 py-3.5 shadow-lg flex items-start gap-3 ${
        type === 'error' ? 'bg-red-50 dark:bg-red-950/50 border-red-200 dark:border-red-900/50 text-red-800 dark:text-red-300' :
        type === 'success' ? 'bg-emerald-50 dark:bg-emerald-950/50 border-emerald-200 dark:border-emerald-900/50 text-emerald-800 dark:text-emerald-300' :
        'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-800 dark:text-slate-200'
    }`;

    const messageEl = document.createElement('p');
    messageEl.className = 'text-[12px] font-medium leading-relaxed';
    messageEl.textContent = String(message);
    toast.appendChild(messageEl);
    els.toastContainer.appendChild(toast);

    requestAnimationFrame(() => {
        toast.classList.remove('toast-enter');
        toast.classList.add('toast-active');
    });

    setTimeout(() => {
        toast.classList.remove('toast-active');
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function _defaultPortalBootstrap() {
    return {
        authMode: 'managed_unavailable',
        csrfToken: null,
        actor: null,
        features: {
            apiKeyInput: false,
            directDebug: false
        }
    };
}

function _isBootstrapReady() {
    return state.bootstrap && state.bootstrap.status === 'ready';
}

function _clearBootstrapRetryTimer() {
    if (state.bootstrap.retry.timer !== null) {
        clearTimeout(state.bootstrap.retry.timer);
        state.bootstrap.retry.timer = null;
        return true;
    }
    return false;
}

function _recordBootstrapRetryEvent(outcome, details = null) {
    const eventDetails = details && typeof details === 'object' ? details : {};
    const retry = state.bootstrap.retry;
    const parsedAttempt = Number(eventDetails.attempt);
    const parsedHttpStatus = Number(eventDetails.httpStatus);
    const parsedDelayMs = Number(eventDetails.delayMs);

    if (Number.isFinite(parsedAttempt)) {
        retry.attempt = Math.max(0, Math.trunc(parsedAttempt));
    }
    if (eventDetails.reason !== undefined) {
        retry.lastReason = String(eventDetails.reason || '').trim();
    }
    if (Number.isFinite(parsedHttpStatus)) {
        retry.lastHttpStatus = Math.max(0, Math.trunc(parsedHttpStatus));
    }
    if (Number.isFinite(parsedDelayMs)) {
        retry.lastDelayMs = Math.max(0, Math.trunc(parsedDelayMs));
    }
    retry.lastOutcome = String(outcome || '').trim();
    retry.lastEventAt = Date.now();

    const payload = {
        outcome: retry.lastOutcome,
        attempt: retry.attempt,
        reason: retry.lastReason,
        httpStatus: retry.lastHttpStatus,
        delayMs: retry.lastDelayMs
    };
    if (retry.lastOutcome.startsWith('terminal')) {
        console.warn('[portal bootstrap retry]', payload);
    } else {
        console.info('[portal bootstrap retry]', payload);
    }
}

function _hasActiveBootstrapRetryState() {
    return Boolean(
        state.bootstrap.retry.timer !== null
        || state.bootstrap.retry.deadlineAt > 0
    );
}

function _hasBootstrapRetryHistory() {
    return Boolean(
        state.bootstrap.retry.attempt > 0
        || state.bootstrap.retry.lastOutcome
    );
}

function _finalizeBootstrapRetry(outcome, details = null) {
    _clearBootstrapRetryTimer();
    state.bootstrap.retry.deadlineAt = 0;
    _recordBootstrapRetryEvent(outcome, details);
}

function _bootstrapFailureDetails(reason = '', httpStatus = 0, message = '') {
    const normalizedReason = String(reason || '').trim().toLowerCase();
    const normalizedStatus = Number.isFinite(Number(httpStatus)) ? Number(httpStatus) : 0;
    const overrideMessage = String(message || '').trim();

    if (normalizedReason === 'auth_failure' || normalizedReason === 'auth' || normalizedStatus === 401 || normalizedStatus === 403) {
        return {
            reason: 'auth_failure',
            retryable: false,
            toastMessage: overrideMessage || 'Managed authentication expired. Sign in again to restore privileged actions.',
            actionMessage: 'Managed authentication is required. Sign in again before continuing.'
        };
    }
    if (normalizedReason === 'access_outage') {
        return {
            reason: 'access_outage',
            retryable: true,
            toastMessage: overrideMessage || 'Managed access verification is temporarily unavailable. Privileged actions stay blocked until Access validation recovers.',
            actionMessage: 'Managed access verification is temporarily unavailable. Wait for Access validation to recover before continuing.'
        };
    }
    if (normalizedReason === 'config_failure') {
        return {
            reason: 'config_failure',
            retryable: false,
            toastMessage: overrideMessage || 'Managed front door configuration is incomplete. Privileged actions stay blocked until configuration is fixed.',
            actionMessage: 'Managed front door configuration is incomplete. Resolve the front door configuration before continuing.'
        };
    }
    if (normalizedReason === 'invalid_json') {
        return {
            reason: 'invalid_json',
            retryable: false,
            toastMessage: overrideMessage || 'Portal bootstrap returned an invalid response. Privileged actions remain blocked.',
            actionMessage: 'Portal bootstrap returned an invalid response. Wait for the front door response contract to recover before continuing.'
        };
    }
    if (
        normalizedReason === 'upstream_unavailable'
        || normalizedReason === 'timeout'
        || normalizedReason === 'network'
        || BOOTSTRAP_RETRIABLE_HTTP_STATUSES.has(normalizedStatus)
    ) {
        return {
            reason: normalizedReason === 'timeout' || normalizedReason === 'network' ? normalizedReason : 'upstream_unavailable',
            retryable: true,
            toastMessage: overrideMessage || 'Managed front door is waiting on upstream recovery. Privileged actions remain blocked until the portal service responds again.',
            actionMessage: 'Managed front door is waiting on upstream recovery. Retry once the portal service becomes available again.'
        };
    }
    return {
        reason: normalizedReason || 'config_failure',
        retryable: false,
        toastMessage: overrideMessage || 'Managed front door is unavailable. Privileged actions remain blocked until configuration is restored.',
        actionMessage: 'Managed front door is unavailable. Resolve the front door configuration before continuing.'
    };
}

function _isBootstrapRetryableFailure(reason = '', httpStatus = 0) {
    return _bootstrapFailureDetails(reason, httpStatus).retryable;
}

function _nextBootstrapRetryDelayMs(attempt) {
    const retryAttempt = Math.max(1, Number.isFinite(Number(attempt)) ? Number(attempt) : 1);
    const exponent = Math.min(retryAttempt - 1, BOOTSTRAP_RETRY_MAX_EXPONENT);
    const backoff = BOOTSTRAP_RETRY_BASE_DELAY_MS * (2 ** exponent);
    const jitter = Math.floor(Math.random() * (BOOTSTRAP_RETRY_JITTER_MS + 1));
    return Math.min(BOOTSTRAP_RETRY_MAX_DELAY_MS, backoff + jitter);
}

function _scheduleBootstrapRetry(reason = '', httpStatus = 0) {
    if (!_isBootstrapRetryableFailure(reason, httpStatus)) {
        _finalizeBootstrapRetry('terminal_not_retryable', { reason, httpStatus });
        return false;
    }
    if (_isBootstrapReady() || state.bootstrap.status !== 'degraded') {
        _finalizeBootstrapRetry('terminal_bootstrap_resolved', { reason, httpStatus });
        return false;
    }
    if (state.bootstrap.retry.timer !== null) {
        _recordBootstrapRetryEvent('skipped_already_scheduled', { reason, httpStatus });
        return false;
    }

    const now = Date.now();
    if (!state.bootstrap.retry.deadlineAt || state.bootstrap.retry.deadlineAt <= now) {
        state.bootstrap.retry.deadlineAt = now + BOOTSTRAP_RETRY_MAX_WINDOW_MS;
    }

    const attempt = state.bootstrap.retry.attempt + 1;
    if (attempt > BOOTSTRAP_RETRY_MAX_ATTEMPTS) {
        _finalizeBootstrapRetry('terminal_retry_limit', {
            attempt: state.bootstrap.retry.attempt,
            reason,
            httpStatus
        });
        return false;
    }

    const delayMs = _nextBootstrapRetryDelayMs(attempt);
    if ((now + delayMs) > state.bootstrap.retry.deadlineAt) {
        _finalizeBootstrapRetry('terminal_retry_window_elapsed', {
            attempt: state.bootstrap.retry.attempt,
            reason,
            httpStatus
        });
        return false;
    }

    state.bootstrap.retry.timer = window.setTimeout(() => {
        state.bootstrap.retry.timer = null;
        state.bootstrap.retry.attempt = attempt;
        state.bootstrap.retry.lastAttemptAt = Date.now();
        _recordBootstrapRetryEvent('attempt_started', { attempt, reason, httpStatus, delayMs: 0 });
        void loadPortalBootstrap({ isRetryAttempt: true, attempt, retryReason: reason });
    }, delayMs);

    _recordBootstrapRetryEvent('scheduled', { attempt, reason, httpStatus, delayMs });
    return true;
}

function _setBootstrapStatus(status, reason = '', httpStatus = 0) {
    const allowedStatuses = new Set(['pending', 'ready', 'degraded', 'unavailable']);
    const nextStatus = allowedStatuses.has(status) ? status : 'pending';
    state.bootstrap.status = nextStatus;
    state.bootstrap.lastErrorReason = String(reason || '').trim();
    state.bootstrap.lastHttpStatus = Number.isFinite(Number(httpStatus)) ? Number(httpStatus) : 0;
    state.bootstrap.lastTransitionAt = Date.now();
    if (document.body) {
        document.body.dataset.bootstrapStatus = nextStatus;
    }
}

function _trackBootstrapRequest(controller, timeoutId) {
    state.bootstrap.activeController = controller;
    state.bootstrap.activeTimeoutId = timeoutId;
}

function _clearTrackedBootstrapRequest(controller, timeoutId) {
    if (state.bootstrap.activeController === controller) {
        state.bootstrap.activeController = null;
    }
    if (state.bootstrap.activeTimeoutId === timeoutId) {
        state.bootstrap.activeTimeoutId = null;
    }
}

function _cancelPendingBootstrapRequest(reason = 'request_aborted') {
    const controller = state.bootstrap.activeController;
    const timeoutId = state.bootstrap.activeTimeoutId;
    state.bootstrap.activeController = null;
    state.bootstrap.activeTimeoutId = null;
    if (timeoutId !== null) {
        clearTimeout(timeoutId);
    }
    if (controller && !controller.signal.aborted) {
        controller.abort(reason);
    }
}

function _portalPrivilegesReady() {
    return _isBootstrapReady() && !_isManagedUnavailableMode();
}

function _queueBootstrapOnlineFollowup() {
    state.bootstrap.pendingOnlineFollowup = true;
    state.bootstrap.onlineFollowupComplete = false;
}

function _flushBootstrapOnlineFollowup(force = false) {
    if (!state.backendOk) return false;
    if (!_isBootstrapReady()) {
        _queueBootstrapOnlineFollowup();
        return false;
    }
    if (state.bootstrap.onlineFollowupComplete && !force) {
        return false;
    }
    state.bootstrap.pendingOnlineFollowup = false;
    state.bootstrap.onlineFollowupComplete = true;
    void fetchReadiness(true);
    void fetchPresetsForPipeline(state.pipeline, true);
    void fetchConfigMetadata(state.pipeline, true);
    scheduleConfigPreview(true);
    if (state.jobs.length === 0) {
        state.jobsLoadStatus = 'loading';
        renderJobQueue();
    }
    void recoverJobs();
    return true;
}

function _syncBootstrapGuardedControls() {
    const readinessStatus = currentPipelineDispatchStatus();
    const preview = _currentPreviewForPayload();
    const luxPreviewBlocked = state.pipeline === 'lux-depth-v3'
        && state.backendOk
        && _isBootstrapReady()
        && (
            !preview
            || preview.status === 'loading'
            || preview.status === 'error'
            || (preview.status === 'ready' && Array.isArray(preview.field_errors) && preview.field_errors.length > 0)
        );
    const debugBundleBlocked = state.pipeline === 'lux-depth-v3'
        && _effectiveDebugBundleEnabled(preview)
        && !state.portalUi.debugBundleAcknowledged;
    const canRunJobs = _portalPrivilegesReady()
        && state.backendOk
        && readinessStatus === 'ready'
        && !luxPreviewBlocked
        && !debugBundleBlocked;
    if (els.runJobBtn && els.runJobBtn.textContent !== 'Dispatching...') {
        els.runJobBtn.disabled = !canRunJobs;
    }
}

function _isManagedAuthMode() {
    return state.auth && state.auth.mode !== 'direct_debug';
}

function _isManagedUnavailableMode() {
    return state.auth && state.auth.mode === 'managed_unavailable';
}

function _clearStoredApiKeyState(clearPersisted = true) {
    if (clearPersisted) {
        localStorage.removeItem(API_KEY_STORAGE_KEY);
        sessionStorage.removeItem(API_KEY_STORAGE_KEY);
    }
    if (els.apiKeyInput) els.apiKeyInput.value = '';
}

function _syncBootstrapUi() {
    const bootstrapReady = _isBootstrapReady();
    const showApiKeyInput = bootstrapReady && state.auth.features.apiKeyInput;
    const badgeLabel = bootstrapReady ? state.auth.mode : 'unknown';
    if (els.apiKeySection) {
        els.apiKeySection.classList.toggle('hidden', !showApiKeyInput);
    }
    if (els.authModeBadge) {
        els.authModeBadge.textContent = badgeLabel;
    }
    if (els.apiKeyManagedHint) {
        if (bootstrapReady && !_isManagedAuthMode()) {
            els.apiKeyManagedHint.classList.add('hidden');
        } else {
            els.apiKeyManagedHint.classList.remove('hidden');
            els.apiKeyManagedHint.textContent = bootstrapReady
                ? 'Managed mode is active. Backend credentials stay server-side and are never stored in the browser.'
                : 'Bootstrap is still being confirmed. Privileged actions remain disabled until portal authentication is resolved.';
        }
    }
    if (els.apiKeyInput) {
        els.apiKeyInput.disabled = !showApiKeyInput;
    }
    _syncBootstrapGuardedControls();
    _syncOverviewBuildLoadingState();
}

function _applyPortalBootstrap(rawBootstrap, options = {}) {
    const bootstrap = rawBootstrap && typeof rawBootstrap === 'object' ? rawBootstrap : _defaultPortalBootstrap();
    const nextStatus = ['pending', 'ready', 'degraded', 'unavailable'].includes(String(options.status || ''))
        ? String(options.status)
        : 'ready';
    const mode = bootstrap.authMode === 'managed'
        ? 'managed'
        : bootstrap.authMode === 'direct_debug'
            ? 'direct_debug'
            : 'managed_unavailable';
    const isManagedMode = mode !== 'direct_debug';
    state.auth = {
        mode,
        csrfToken: mode === 'managed' ? String(bootstrap.csrfToken || '') : '',
        actor: mode === 'managed' && bootstrap.actor && typeof bootstrap.actor === 'object' ? bootstrap.actor : null,
        features: {
            apiKeyInput: mode === 'direct_debug' && bootstrap.features?.apiKeyInput !== false,
            directDebug: mode === 'direct_debug' && bootstrap.features?.directDebug !== false
        }
    };
    _setBootstrapStatus(nextStatus, options.reason || '', options.httpStatus || 0);
    if (_isBootstrapReady() && isManagedMode) {
        _clearStoredApiKeyState(true);
    }
    _loadApiKeyIntoInputs();
    _syncBootstrapUi();
}

function _normalizeFetchFailureReason(error, timeoutReason = 'request_timeout') {
    const reason = String(error && typeof error === 'object' && 'reason' in error ? error.reason : '').trim().toLowerCase();
    const name = String(error && typeof error === 'object' && 'name' in error ? error.name : '').trim().toLowerCase();
    const message = String(error instanceof Error ? error.message : error || '').trim().toLowerCase();
    const combined = `${reason} ${name} ${message}`.trim();
    if (
        reason === String(timeoutReason || '').trim().toLowerCase()
        || name === 'timeouterror'
        || combined.includes('timed out')
        || combined.includes('timeout')
    ) {
        return 'timeout';
    }
    if (reason === 'request_aborted' || reason === 'navigation_abort' || name === 'aborterror') {
        return 'aborted';
    }
    if (combined.includes('failed to fetch') || combined.includes('networkerror') || combined.includes('load failed')) {
        return 'network';
    }
    return 'error';
}

async function loadPortalBootstrap(options = null) {
    const fallback = _defaultPortalBootstrap();
    const bootstrapOptions = options && typeof options === 'object' ? options : null;
    const retryAttempt = Number.isInteger(bootstrapOptions && bootstrapOptions.attempt)
        && bootstrapOptions.attempt > 0
        ? bootstrapOptions.attempt
        : 0;
    const retryReason = String(bootstrapOptions && bootstrapOptions.retryReason || '').trim().toLowerCase();
    const isRetryAttempt = Boolean((bootstrapOptions && bootstrapOptions.isRetryAttempt) || retryAttempt > 0);
    const bootstrapCancelReason = isRetryAttempt
        ? (retryReason ? `bootstrap_retry_replaced:${retryReason}` : 'bootstrap_retry_replaced')
        : 'bootstrap_replaced';
    _cancelPendingBootstrapRequest(bootstrapCancelReason);
    if (!isRetryAttempt) {
        _applyPortalBootstrap(fallback, { status: 'pending' });
    }
    try {
        const res = await fetchWithTimeout(
            `${API_BASE}/portal/bootstrap`,
            {
                headers: { 'Accept': 'application/json' },
                cache: 'no-store'
            },
            BOOTSTRAP_TIMEOUT_MS,
            'bootstrap_timeout',
            {
                onStart: _trackBootstrapRequest,
                onFinally: _clearTrackedBootstrapRequest
            }
        );
        let payload = null;
        let payloadParsed = false;
        try {
            payload = await res.json();
            payloadParsed = true;
        } catch {
            payloadParsed = false;
        }
        if (res.status === 401 || res.status === 403) {
            const failure = _bootstrapFailureDetails(
                payloadParsed && payload && typeof payload === 'object' ? payload.reason : 'auth_failure',
                res.status,
                payloadParsed && payload && typeof payload === 'object' ? payload.message : ''
            );
            _finalizeBootstrapRetry('terminal_auth_redirect', { reason: failure.reason, httpStatus: res.status });
            _applyPortalBootstrap(fallback, { status: 'unavailable', reason: failure.reason, httpStatus: res.status });
            createToast(failure.toastMessage, 'error');
            window.location.assign('/login');
            return;
        }
        if (!res.ok) {
            const failure = _bootstrapFailureDetails(
                payloadParsed && payload && typeof payload === 'object' ? payload.reason : `http_${res.status}`,
                res.status,
                payloadParsed && payload && typeof payload === 'object' ? payload.message : ''
            );
            const status = failure.retryable ? 'degraded' : 'unavailable';
            _applyPortalBootstrap(fallback, { status, reason: failure.reason, httpStatus: res.status });
            const retryScheduled = failure.retryable && _scheduleBootstrapRetry(failure.reason, res.status);
            if (!isRetryAttempt || !retryScheduled) {
                createToast(failure.toastMessage, 'error');
            }
            return;
        }
        if (!payloadParsed || !payload || typeof payload !== 'object') {
            const failure = _bootstrapFailureDetails('invalid_json');
            _finalizeBootstrapRetry('terminal_invalid_json', { reason: 'invalid_json' });
            _applyPortalBootstrap(fallback, { status: 'unavailable', reason: 'invalid_json' });
            createToast(failure.toastMessage, 'error');
            return;
        }
        if (_hasBootstrapRetryHistory()) {
            _finalizeBootstrapRetry('succeeded', {
                attempt: state.bootstrap.retry.attempt,
                reason: 'bootstrap_ready',
                httpStatus: res.status,
                delayMs: 0
            });
        }
        const previousHealthEndpointPath = String(state.bootstrap.lastHealthEndpointPath || '');
        _applyPortalBootstrap(payload, { status: 'ready' });
        const nextHealthEndpointPath = _healthEndpointPath();
        if (state.backendOk && previousHealthEndpointPath && previousHealthEndpointPath !== nextHealthEndpointPath) {
            _queueBootstrapOnlineFollowup();
            void checkBackend(true);
            return;
        }
        _flushBootstrapOnlineFollowup();
    } catch (error) {
        const normalizedReason = _normalizeFetchFailureReason(error, 'bootstrap_timeout');
        if (normalizedReason === 'aborted') {
            return;
        }
        const failure = _bootstrapFailureDetails(normalizedReason, 0);
        const status = failure.retryable ? 'degraded' : 'unavailable';
        _applyPortalBootstrap(fallback, { status, reason: failure.reason });
        const retryScheduled = failure.retryable && _scheduleBootstrapRetry(failure.reason, 0);
        if (!isRetryAttempt || !retryScheduled) {
            createToast(failure.toastMessage, 'error');
        }
    }
}

function _blockManagedUnavailableAction(actionLabel) {
    if (!_isManagedUnavailableMode()) return false;
    const failure = _bootstrapFailureDetails(state.bootstrap.lastErrorReason, state.bootstrap.lastHttpStatus);
    createToast(`${failure.actionMessage} Unable to ${actionLabel} until recovery completes.`, 'error');
    return true;
}

function _healthEndpointPath() {
    if (!_isBootstrapReady()) return '/healthz';
    return _isManagedAuthMode() ? '/healthz' : '/ready';
}

function _normalizeApiToken(raw) {
    const value = String(raw || '').trim();
    if (!value) return '';
    if (value.toLowerCase().startsWith('bearer ')) return value.slice(7).trim();
    return value;
}

function _persistApiKeyFromInputs() {
    if (!_isBootstrapReady()) {
        _clearStoredApiKeyState(false);
        return;
    }
    if (_isManagedAuthMode()) {
        _clearStoredApiKeyState(true);
        return;
    }
    const token = _normalizeApiToken(els.apiKeyInput ? els.apiKeyInput.value : '');
    localStorage.removeItem(API_KEY_STORAGE_KEY);
    if (!token) {
        sessionStorage.removeItem(API_KEY_STORAGE_KEY);
        return;
    }
    sessionStorage.setItem(API_KEY_STORAGE_KEY, token);
}

function _loadApiKeyIntoInputs() {
    if (!_isBootstrapReady()) {
        _clearStoredApiKeyState(false);
        return;
    }
    if (_isManagedAuthMode()) {
        _clearStoredApiKeyState(true);
        return;
    }
    const localValue = localStorage.getItem(API_KEY_STORAGE_KEY) || '';
    const sessionValue = sessionStorage.getItem(API_KEY_STORAGE_KEY) || '';
    const stored = sessionValue || localValue;
    if (localValue && !sessionValue) {
        sessionStorage.setItem(API_KEY_STORAGE_KEY, localValue);
    }
    localStorage.removeItem(API_KEY_STORAGE_KEY);
    if (els.apiKeyInput) els.apiKeyInput.value = stored;
}

function _currentApiToken() {
    if (!_isBootstrapReady()) return '';
    if (_isManagedAuthMode()) return '';
    return _normalizeApiToken(
        (els.apiKeyInput && els.apiKeyInput.value) ||
        sessionStorage.getItem(API_KEY_STORAGE_KEY) ||
        ''
    );
}

function _buildAuthHeaders(base = {}, method = 'GET') {
    const headers = { ...base };
    const normalizedMethod = String(method || 'GET').toUpperCase();
    if (!_isBootstrapReady()) {
        return headers;
    }
    if (_isManagedAuthMode()) {
        if (!SAFE_HTTP_METHODS.has(normalizedMethod) && state.auth.csrfToken) {
            headers['X-CSRF-Token'] = state.auth.csrfToken;
        }
        return headers;
    }
    const token = _currentApiToken();
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
        headers['x-api-key'] = token;
    }
    return headers;
}

function resumeBlockedJobStreamsAfterAuthUpdate() {
    if (_isManagedAuthMode()) return;
    const token = _currentApiToken();
    if (!token) return;
    state.jobs.forEach((job) => {
        if (!job) return;
        _ensureJobStreamState(job);
        if (!job.reconnectBlocked) return;
        if (!_isJobStreamRecoverable(job)) return;
        if (_jobHasActiveStream(job)) return;
        appendJobLog(job, '[INFO] API key updated. Reconnecting event stream...');
        logToPane(job.id, '[INFO] API key updated. Reconnecting event stream...');
        startJobEventStream(job, job.eventStreamUrl);
    });
}

async function copyToClipboard(text) {
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
        } else {
            const ta = document.createElement("textarea");
            ta.value = text;
            ta.style.position = "absolute";
            ta.style.left = "-9999px";
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
        }
        createToast('Copied to clipboard.', 'success');
    } catch (err) {
        createToast('Failed to copy text.', 'error');
    }
}

function shellQuote(str) {
    const v = String(str).replace(/[\r\n]+/g, " ").trim().replace(/\\/g, "\\\\").replace(/"/g, '\\"');
    return `"${v}"`;
}

function appendJobLog(job, line) {
    if (!job || typeof line !== 'string' || !line.trim()) return;
    if (!Array.isArray(job.logs)) job.logs = [];
    job.logs.push(line);
    if (job.logs.length > MAX_JOB_LOG_LINES) {
        job.logs.splice(0, job.logs.length - MAX_JOB_LOG_LINES);
    }
    job.updatedAt = Date.now();
}

function getReadableError(errorObj) {
    if (!errorObj || typeof errorObj !== 'object') return '';
    const code = typeof errorObj.code === 'string' ? errorObj.code.trim() : '';
    const message = typeof errorObj.message === 'string' ? errorObj.message.trim() : '';
    if (code && message) return `[${code}] ${message}`;
    return message || code || '';
}

function normalizeRunSummary(rawSummary) {
    if (!rawSummary || typeof rawSummary !== 'object') return null;

    const toNonNegativeInt = (value) => {
        if (typeof value === 'boolean') return null;
        const parsed = Number(value);
        if (!Number.isFinite(parsed) || parsed < 0) return null;
        return Math.round(parsed);
    };

    const normalized = {
        source: String(rawSummary.source || '').trim(),
        batch_id: String(rawSummary.batch_id || '').trim(),
        total_images: toNonNegativeInt(rawSummary.total_images),
        success_count: toNonNegativeInt(rawSummary.success_count),
        error_count: toNonNegativeInt(rawSummary.error_count),
        artifact_index_count: toNonNegativeInt(rawSummary.artifact_index_count),
        reviewable_outputs: Boolean(rawSummary.reviewable_outputs),
        partial: Boolean(rawSummary.partial)
    };

    if (
        normalized.total_images === null
        && normalized.success_count !== null
        && normalized.error_count !== null
    ) {
        normalized.total_images = normalized.success_count + normalized.error_count;
    }

    const hasMeaningfulContent = normalized.batch_id
        || normalized.total_images !== null
        || normalized.success_count !== null
        || normalized.error_count !== null
        || normalized.artifact_index_count !== null
        || normalized.reviewable_outputs
        || normalized.partial;

    return hasMeaningfulContent ? normalized : null;
}

function describeRunSummary(summary) {
    if (!summary) return '';
    const segments = [];
    const total = Number.isFinite(summary.total_images) ? summary.total_images : null;
    const successCount = Number.isFinite(summary.success_count) ? summary.success_count : null;
    const errorCount = Number.isFinite(summary.error_count) ? summary.error_count : null;

    if (successCount !== null && total !== null && total > 0) {
        segments.push(`${successCount}/${total} images succeeded`);
    } else if (successCount !== null) {
        segments.push(`${successCount} image${successCount === 1 ? '' : 's'} succeeded`);
    }

    if (errorCount !== null && errorCount > 0) {
        segments.push(`${errorCount} failed`);
    }

    if (summary.partial) {
        segments.push('outputs remain reviewable');
    }

    return segments.join(' • ');
}

function jobOutcomeSummary(job) {
    return describeRunSummary(normalizeRunSummary(job?.run_summary));
}

function normalizeArtifactItems(artifactsContainer) {
    if (!artifactsContainer || typeof artifactsContainer !== 'object') return [];
    const items = Array.isArray(artifactsContainer.items) ? artifactsContainer.items : [];
    return items
        .filter((item) => item && typeof item.path === 'string')
        .map((item) => ({
            artifact_type: String(item.artifact_type || 'file'),
            media_kind: String(item.media_kind || item.artifact_type || 'file'),
            previewable: Boolean(item.previewable),
            content_type: typeof item.content_type === 'string' ? item.content_type : '',
            url: typeof item.url === 'string' ? item.url : '',
            path: String(item.path),
            relative_path: String(item.relative_path || item.path),
            size_bytes: typeof item.size_bytes === 'number' ? item.size_bytes : null,
            sha256: typeof item.sha256 === 'string' ? item.sha256 : '',
            display_hint: _normalizeArtifactDisplayHint(item.display_hint)
        }));
}

function upsertArtifact(job, artifact) {
    if (!job || !artifact || !artifact.path) return;
    if (!Array.isArray(job.artifacts)) job.artifacts = [];
    const normalizedArtifact = {
        ...artifact,
        display_hint: _normalizeArtifactDisplayHint(artifact.display_hint)
    };
    const existing = job.artifacts.find((entry) => entry.path === artifact.path);
    if (existing) {
        existing.artifact_type = normalizedArtifact.artifact_type || existing.artifact_type;
        existing.media_kind = normalizedArtifact.media_kind || existing.media_kind;
        existing.previewable = typeof normalizedArtifact.previewable === 'boolean' ? normalizedArtifact.previewable : existing.previewable;
        existing.content_type = normalizedArtifact.content_type || existing.content_type;
        existing.url = normalizedArtifact.url || existing.url;
        existing.relative_path = normalizedArtifact.relative_path || existing.relative_path;
        existing.size_bytes = normalizedArtifact.size_bytes ?? existing.size_bytes;
        existing.sha256 = normalizedArtifact.sha256 || existing.sha256;
        existing.display_hint = normalizedArtifact.display_hint || existing.display_hint || null;
    } else {
        job.artifacts.push(normalizedArtifact);
    }
    _reconcileJobTimeline(job);
}

function getRunCardArtifact(job) {
    if (!job || !Array.isArray(job.artifacts)) return null;
    return job.artifacts.find((artifact) => {
        const displayRole = String(artifactDisplayHint(artifact)?.role || '').trim().toLowerCase();
        const type = String(artifact.artifact_type || '').toLowerCase();
        const relPath = String(artifact.relative_path || artifact.path || '').toLowerCase();
        return displayRole === 'run_card' || type === 'run_card' || relPath.includes('run_card');
    }) || null;
}

function updateRunCardActions(job) {
    if (!els.runCardActions) return;
    const runCard = getRunCardArtifact(job);
    if (!runCard) {
        els.runCardActions.classList.add('hidden');
        els.runCardActions.classList.remove('flex');
        return;
    }
    const runCardPath = String(runCard.relative_path || runCard.path || '');
    const runCardUrl = buildArtifactUrl(job, runCard);
    const runCardSha = String(runCard.sha256 || '');
    if (els.viewRunCardBtn) {
        els.viewRunCardBtn.dataset.url = runCardUrl;
    }
    if (els.copyRunCardPathBtn) els.copyRunCardPathBtn.dataset.path = runCardPath;
    if (els.copyRunCardFingerprintBtn) {
        els.copyRunCardFingerprintBtn.dataset.fingerprint = runCardSha;
        els.copyRunCardFingerprintBtn.disabled = !runCardSha;
        els.copyRunCardFingerprintBtn.classList.toggle('opacity-50', !runCardSha);
        els.copyRunCardFingerprintBtn.classList.toggle('cursor-not-allowed', !runCardSha);
    }
    els.runCardActions.classList.remove('hidden');
    els.runCardActions.classList.add('flex');
}

function _selectedArtifactForJob(job) {
    if (!job || !Array.isArray(job.artifacts) || job.artifacts.length === 0) return null;
    const normalizedJobId = _normalizeSelectedJobId(job.id);
    const ranked = rankArtifactsForDisplay(job.artifacts);
    const selectedPath = _normalizeArtifactRoutePath(state.artifactUi.selectedByJob[normalizedJobId]);
    const selected = ranked.find((artifact) => _artifactRouteKey(artifact) === selectedPath);
    if (selectedPath && !selected) {
        delete state.artifactUi.compareByJob[normalizedJobId];
    }
    const hero = selected || ranked[0] || null;
    if (hero) {
        state.artifactUi.selectedByJob[normalizedJobId] = _artifactRouteKey(hero);
    }
    return hero;
}

function _renderArtifactMetadataCard(job, artifact) {
    if (!els.artifactMetadataCard) return;
    els.artifactMetadataCard.innerHTML = '';

    const title = document.createElement('p');
    title.className = 'text-[12px] font-semibold text-slate-800 dark:text-slate-100';
    title.textContent = artifact
        ? artifactLabel(artifact)
        : 'Select a completed job to review the highest-value output here.';
    els.artifactMetadataCard.appendChild(title);

    const detail = document.createElement('p');
    detail.className = 'mt-2 text-[12px] leading-6 text-slate-600 dark:text-slate-300';
    if (!job) {
        detail.textContent = 'No job selected.';
    } else if (!artifact) {
        detail.textContent = 'Artifacts will appear here when the selected run indexes outputs.';
    } else {
        detail.textContent = `${artifactDisplayLabel(artifact)} • ${artifactContentType(artifact) || 'binary'} • ${formatBytes(artifact.size_bytes)}.`;
    }
    els.artifactMetadataCard.appendChild(detail);
}

function _latestVisibleTransportWarning(job) {
    const warnings = Array.isArray(job?.transportWarnings) ? job.transportWarnings : [];
    for (let index = warnings.length - 1; index >= 0; index -= 1) {
        const warning = warnings[index];
        if (warning && warning.tone !== 'info') return warning;
    }
    return null;
}

function _reviewStatusSnapshot(job, artifact) {
    if (!job) {
        return {
            visible: false,
            tone: 'info',
            title: 'Awaiting completed run',
            detail: 'Select a job to review related warnings, completion state, and output readiness.'
        };
    }

    const artifactCount = Array.isArray(job.artifacts) ? job.artifacts.length : 0;
    const summary = normalizeRunSummary(job.run_summary);
    const readableError = getReadableError(job.error);
    const outcomeSummary = jobOutcomeSummary(job);
    const freshestActivityAt = Number(job.lastEventAt || job.updatedAt || job.finishedAt || job.createdAt || 0);
    const freshnessLabel = formatRelativeTime(freshestActivityAt);
    const visibleWarning = _latestVisibleTransportWarning(job);
    const reviewableOutputs = Boolean(summary?.reviewable_outputs) || artifactCount > 0;

    if (job.state === 'partial') {
        return {
            visible: true,
            tone: 'warning',
            title: 'Run partially completed',
            detail: outcomeSummary
                ? `${outcomeSummary}. Updated ${freshnessLabel}.`
                : 'Some inputs failed, but outputs remain reviewable.'
        };
    }

    if (job.state === 'failed') {
        return {
            visible: true,
            tone: reviewableOutputs ? 'warning' : 'error',
            title: reviewableOutputs ? 'Run failed after indexing reviewable outputs' : 'Run failed before outputs were ready',
            detail: readableError
                || (reviewableOutputs
                    ? `${artifactCount} artifact${artifactCount === 1 ? '' : 's'} remain available for review. Updated ${freshnessLabel}.`
                    : 'No reviewable outputs were indexed before the failure was reported.')
        };
    }

    if (job.state === 'canceled') {
        return {
            visible: true,
            tone: reviewableOutputs ? 'warning' : 'error',
            title: reviewableOutputs ? 'Run canceled after partial output capture' : 'Run canceled before review outputs were ready',
            detail: reviewableOutputs
                ? `${artifactCount} artifact${artifactCount === 1 ? '' : 's'} remain available for review despite cancellation. Updated ${freshnessLabel}.`
                : 'Execution was canceled before reviewable outputs were indexed.'
        };
    }

    if (job.state === 'offline') {
        return {
            visible: true,
            tone: 'warning',
            title: reviewableOutputs ? 'Run is offline with reviewable outputs' : 'Run is offline',
            detail: reviewableOutputs
                ? `${artifactCount} artifact${artifactCount === 1 ? '' : 's'} remain available, but live backend status is stale until connectivity is restored.`
                : 'Live backend status is stale until connectivity is restored.'
        };
    }

    if (job.reconnectBlocked) {
        return {
            visible: true,
            tone: 'warning',
            title: 'Transport warning recorded',
            detail: 'Authentication must be restored before live event transport can reconnect.'
        };
    }

    if (visibleWarning) {
        return {
            visible: true,
            tone: visibleWarning.tone === 'error' ? 'error' : 'warning',
            title: 'Transport warning recorded',
            detail: String(visibleWarning.detail || 'Live transport reported an operator-visible warning.')
        };
    }

    if (job.state === 'running' || job.state === 'queued') {
        return {
            visible: true,
            tone: 'info',
            title: 'Run still in progress',
            detail: artifactCount > 0
                ? `${artifactCount} artifact${artifactCount === 1 ? '' : 's'} already indexed. Updated ${freshnessLabel}.`
                : 'Artifacts and provenance will populate here as outputs arrive.'
        };
    }

    return {
        visible: true,
        tone: 'ready',
        title: artifact ? 'Outputs ready for review' : 'Run ready for review',
        detail: outcomeSummary
            ? `${outcomeSummary}. Updated ${freshnessLabel}.`
            : `${artifactCount} artifact${artifactCount === 1 ? '' : 's'} indexed and ready for operator review.`
    };
}

function _renderReviewStatusBanner(job, artifact) {
    if (!els.reviewStatusBanner || !els.reviewStatusTitle || !els.reviewStatusDetail) return;
    const snapshot = _reviewStatusSnapshot(job, artifact);
    els.reviewStatusBanner.dataset.tone = snapshot.tone;
    if (!snapshot.visible) {
        els.reviewStatusBanner.classList.add('hidden');
        els.reviewStatusTitle.textContent = snapshot.title;
        els.reviewStatusDetail.textContent = snapshot.detail;
        return;
    }
    els.reviewStatusTitle.textContent = snapshot.title;
    els.reviewStatusDetail.textContent = snapshot.detail;
    els.reviewStatusBanner.classList.remove('hidden');
}

function _renderArtifactProvenance(job, artifact) {
    if (!els.reviewProvenanceGrid) return;

    if (!job) {
        els.reviewProvenanceGrid.classList.add('hidden');
        if (els.reviewProvenanceArtifactRole) els.reviewProvenanceArtifactRole.textContent = 'Awaiting indexed output';
        if (els.reviewProvenanceRunState) els.reviewProvenanceRunState.textContent = 'No job selected';
        if (els.reviewProvenancePath) {
            els.reviewProvenancePath.textContent = 'Preview, metadata, and actions will appear here when outputs are indexed.';
            els.reviewProvenancePath.removeAttribute('title');
        }
        if (els.reviewProvenanceFreshness) els.reviewProvenanceFreshness.textContent = 'No live telemetry';
        if (els.reviewProvenanceSource) els.reviewProvenanceSource.textContent = 'Not reported';
        if (els.reviewProvenanceBatch) els.reviewProvenanceBatch.textContent = 'Not reported';
        return;
    }

    const summary = normalizeRunSummary(job.run_summary);
    const artifactDescriptor = artifact
        ? `${artifactDisplayLabel(artifact)} • ${artifactContentType(artifact) || 'binary'} • ${formatBytes(artifact.size_bytes)}`
        : 'Awaiting indexed artifact';
    const relativePath = artifact ? artifactLabel(artifact) : 'Artifacts will appear here when the selected run indexes outputs.';
    const freshnessLabel = `Updated ${formatRelativeTime(Number(job.lastEventAt || job.updatedAt || job.finishedAt || job.createdAt || 0))}`;
    const runStateLabel = `${titleCaseToken(job.state, 'Unknown')} • ${titleCaseToken(job.pipeline, 'Unknown')}`;
    const sourceLabel = summary?.source || titleCaseToken(job.pipeline, 'Not reported');
    const batchLabel = summary?.batch_id || 'Not reported';

    if (els.reviewProvenanceArtifactRole) els.reviewProvenanceArtifactRole.textContent = artifactDescriptor;
    if (els.reviewProvenanceRunState) els.reviewProvenanceRunState.textContent = runStateLabel;
    if (els.reviewProvenancePath) {
        els.reviewProvenancePath.textContent = relativePath;
        els.reviewProvenancePath.title = relativePath;
    }
    if (els.reviewProvenanceFreshness) els.reviewProvenanceFreshness.textContent = freshnessLabel;
    if (els.reviewProvenanceSource) els.reviewProvenanceSource.textContent = sourceLabel;
    if (els.reviewProvenanceBatch) els.reviewProvenanceBatch.textContent = batchLabel;
    els.reviewProvenanceGrid.classList.remove('hidden');
}

function _renderReviewCompareSummary(primaryArtifact, compareArtifact, compareEnabled) {
    if (!els.reviewCompareSummary || !els.reviewCompareTitle || !els.reviewCompareDetail) return;
    if (!primaryArtifact || !compareArtifact) {
        els.reviewCompareSummary.classList.add('hidden');
        els.reviewCompareTitle.textContent = 'Compare pair available';
        els.reviewCompareDetail.textContent = 'Enable compare mode to inspect paired outputs side by side.';
        return;
    }

    const primaryLabel = artifactLabel(primaryArtifact);
    const compareLabel = artifactLabel(compareArtifact);
    els.reviewCompareTitle.textContent = compareEnabled ? 'Comparing paired outputs' : 'Compare pair available';
    els.reviewCompareDetail.textContent = compareEnabled
        ? `${primaryLabel} is shown against ${compareLabel}.`
        : `${compareLabel} is available as a side-by-side comparison for ${primaryLabel}.`;
    els.reviewCompareSummary.classList.remove('hidden');
}

function _resetArtifactActionButtons() {
    if (els.openArtifactBtn) {
        els.openArtifactBtn.disabled = true;
        delete els.openArtifactBtn.dataset.url;
    }
    if (els.downloadArtifactBtn) {
        els.downloadArtifactBtn.disabled = true;
        delete els.downloadArtifactBtn.dataset.url;
        delete els.downloadArtifactBtn.dataset.filename;
    }
    if (els.copyArtifactPathBtn) {
        els.copyArtifactPathBtn.disabled = true;
        delete els.copyArtifactPathBtn.dataset.path;
    }
}

function renderReviewSurfaces(payload = null) {
    const currentPayload = payload || generatePayload();
    renderArtifactPanel();
    renderSelectedJobInspector();
    renderReconstructionRuntimeSummary(currentPayload);
    renderMissionControl(payload);
}

function renderArtifactPanel() {
    const jobsLoading = _isJobsHydrationPending();
    _toggleSurfaceSkeleton(els.artifactsShell, els.artifactShellContent, els.artifactSkeletonState, jobsLoading);
    if (jobsLoading) {
        _resetArtifactActionButtons();
        _renderReviewStatusBanner(null, null);
        _renderArtifactProvenance(null, null);
        _renderReviewCompareSummary(null, null, false);
        if (els.artifactMeta) els.artifactMeta.textContent = 'Hydrating artifacts';
        renderConsoleContextRibbon();
        return;
    }

    if (!els.artifactMeta || !els.artifactThumbnailRail) return;
    els.artifactThumbnailRail.setAttribute('role', 'listbox');
    els.artifactThumbnailRail.setAttribute('aria-label', 'Artifact thumbnails');
    const selected = state.jobs.find((item) => item.id === state.selectedJobId);
    const artifacts = Array.isArray(selected?.artifacts) ? rankArtifactsForDisplay(selected.artifacts) : [];

    if (!selected) {
        _resetArtifactActionButtons();
        els.artifactMeta.textContent = 'No job selected';
        els.artifactThumbnailRail.innerHTML = '';
        if (els.artifactSelectionTitle) els.artifactSelectionTitle.textContent = 'No artifact selected';
        if (els.artifactSelectionMeta) els.artifactSelectionMeta.textContent = 'Preview, metadata, and actions will appear here when outputs are indexed.';
        if (els.artifactCompareBtn) {
            els.artifactCompareBtn.classList.add('hidden');
            els.artifactCompareBtn.setAttribute('aria-pressed', 'false');
            els.artifactCompareBtn.removeAttribute('aria-controls');
        }
        if (els.artifactPreviewSoloImage) {
            els.artifactPreviewSoloImage.classList.add('hidden');
            els.artifactPreviewSoloImage.removeAttribute('src');
        }
        if (els.artifactPreviewImage) {
            els.artifactPreviewImage.classList.add('hidden');
            els.artifactPreviewImage.removeAttribute('src');
        }
        if (els.artifactCompareImage) {
            els.artifactCompareImage.classList.add('hidden');
            els.artifactCompareImage.removeAttribute('src');
        }
        if (els.artifactCompareStage) {
            els.artifactCompareStage.classList.add('hidden');
            els.artifactCompareStage.setAttribute('aria-hidden', 'true');
        }
        _renderArtifactMetadataCard(null, null);
        _renderReviewStatusBanner(null, null);
        _renderArtifactProvenance(null, null);
        _renderReviewCompareSummary(null, null, false);
        updateRunCardActions(null);
        if (els.emptyArtifactState) els.emptyArtifactState.style.display = 'block';
        renderConsoleContextRibbon();
        _syncConsoleRoute(true);
        return;
    }

    const errorText = getReadableError(selected.error);
    els.artifactMeta.textContent = `${artifacts.length} artifacts${errorText ? ` • ${errorText}` : ''}`;

    if (artifacts.length === 0) {
        _resetArtifactActionButtons();
        els.artifactThumbnailRail.innerHTML = '';
        if (els.artifactSelectionTitle) els.artifactSelectionTitle.textContent = 'No artifact selected';
        if (els.artifactSelectionMeta) els.artifactSelectionMeta.textContent = 'Artifacts will appear here when the selected run indexes outputs.';
        if (els.artifactCompareBtn) {
            els.artifactCompareBtn.classList.add('hidden');
            els.artifactCompareBtn.setAttribute('aria-pressed', 'false');
            els.artifactCompareBtn.removeAttribute('aria-controls');
        }
        if (els.artifactPreviewSoloImage) {
            els.artifactPreviewSoloImage.classList.add('hidden');
            els.artifactPreviewSoloImage.removeAttribute('src');
        }
        if (els.artifactPreviewImage) {
            els.artifactPreviewImage.classList.add('hidden');
            els.artifactPreviewImage.removeAttribute('src');
        }
        if (els.artifactCompareImage) {
            els.artifactCompareImage.classList.add('hidden');
            els.artifactCompareImage.removeAttribute('src');
        }
        if (els.artifactCompareStage) {
            els.artifactCompareStage.classList.add('hidden');
            els.artifactCompareStage.setAttribute('aria-hidden', 'true');
        }
        _renderArtifactMetadataCard(selected, null);
        _renderReviewStatusBanner(selected, null);
        _renderArtifactProvenance(selected, null);
        _renderReviewCompareSummary(null, null, false);
        updateRunCardActions(selected);
        if (els.emptyArtifactState) els.emptyArtifactState.style.display = 'block';
        renderConsoleContextRibbon();
        _syncConsoleRoute(true);
        return;
    }

    if (els.emptyArtifactState) els.emptyArtifactState.style.display = 'none';
    const selectedArtifact = _selectedArtifactForJob(selected);
    const compareCandidate = findCompareArtifact(selectedArtifact, artifacts);
    const compareEnabled = Boolean(compareCandidate) && Boolean(state.artifactUi.compareByJob[String(selected.id || '')]);

    if (els.artifactCompareBtn) {
        if (compareCandidate) {
            els.artifactCompareBtn.classList.remove('hidden');
            els.artifactCompareBtn.textContent = compareEnabled ? 'Single View' : 'Compare';
            els.artifactCompareBtn.setAttribute('aria-pressed', compareEnabled ? 'true' : 'false');
            els.artifactCompareBtn.setAttribute('aria-controls', 'artifactCompareStage');
        } else {
            els.artifactCompareBtn.classList.add('hidden');
            els.artifactCompareBtn.setAttribute('aria-pressed', 'false');
            els.artifactCompareBtn.removeAttribute('aria-controls');
        }
    }

    if (els.artifactSelectionTitle) {
        els.artifactSelectionTitle.textContent = selectedArtifact ? artifactLabel(selectedArtifact) : 'No artifact selected';
    }
    if (els.artifactSelectionMeta) {
        els.artifactSelectionMeta.textContent = selectedArtifact
            ? `${artifactDisplayLabel(selectedArtifact)} • ${artifactContentType(selectedArtifact) || 'binary'} • ${formatBytes(selectedArtifact.size_bytes)}`
            : 'Preview, metadata, and actions will appear here when outputs are indexed.';
    }
    _renderReviewStatusBanner(selected, selectedArtifact);
    _renderArtifactProvenance(selected, selectedArtifact);
    _renderReviewCompareSummary(selectedArtifact, compareCandidate, compareEnabled);

    if (els.openArtifactBtn) {
        const openUrl = selectedArtifact ? buildArtifactUrl(selected, selectedArtifact) : '';
        els.openArtifactBtn.disabled = !openUrl;
        els.openArtifactBtn.dataset.url = openUrl;
    }
    if (els.downloadArtifactBtn) {
        const downloadUrl = selectedArtifact ? buildArtifactUrl(selected, selectedArtifact) : '';
        els.downloadArtifactBtn.disabled = !downloadUrl;
        els.downloadArtifactBtn.dataset.url = downloadUrl;
        els.downloadArtifactBtn.dataset.filename = selectedArtifact ? artifactNameParts(selectedArtifact).fileName : '';
    }
    if (els.copyArtifactPathBtn) {
        els.copyArtifactPathBtn.disabled = !selectedArtifact;
        els.copyArtifactPathBtn.dataset.path = selectedArtifact ? artifactLabel(selectedArtifact) : '';
    }

    if (els.artifactCompareStage) {
        els.artifactCompareStage.classList.toggle('hidden', !compareEnabled);
        els.artifactCompareStage.setAttribute('aria-hidden', compareEnabled ? 'false' : 'true');
    }
    if (els.artifactPreviewSoloImage) els.artifactPreviewSoloImage.classList.toggle('hidden', compareEnabled || !artifactIsPreviewable(selectedArtifact));
    if (els.artifactMetadataCard) els.artifactMetadataCard.classList.toggle('hidden', artifactIsPreviewable(selectedArtifact));
    if (compareEnabled && selectedArtifact && compareCandidate) {
        if (els.artifactPreviewImage) {
            els.artifactPreviewImage.src = buildArtifactUrl(selected, selectedArtifact);
            els.artifactPreviewImage.classList.remove('hidden');
        }
        if (els.artifactPreviewPrimaryCaption) els.artifactPreviewPrimaryCaption.textContent = artifactLabel(selectedArtifact);
        if (els.artifactCompareImage) {
            els.artifactCompareImage.src = buildArtifactUrl(selected, compareCandidate);
            els.artifactCompareImage.classList.remove('hidden');
        }
        if (els.artifactCompareCaption) els.artifactCompareCaption.textContent = artifactLabel(compareCandidate);
    } else if (artifactIsPreviewable(selectedArtifact)) {
        if (els.artifactPreviewSoloImage) {
            els.artifactPreviewSoloImage.src = buildArtifactUrl(selected, selectedArtifact);
            els.artifactPreviewSoloImage.classList.remove('hidden');
        }
        if (els.artifactPreviewImage) {
            els.artifactPreviewImage.classList.add('hidden');
            els.artifactPreviewImage.removeAttribute('src');
        }
        if (els.artifactCompareImage) {
            els.artifactCompareImage.classList.add('hidden');
            els.artifactCompareImage.removeAttribute('src');
        }
    } else {
        if (els.artifactPreviewSoloImage) {
            els.artifactPreviewSoloImage.classList.add('hidden');
            els.artifactPreviewSoloImage.removeAttribute('src');
        }
        if (els.artifactPreviewImage) {
            els.artifactPreviewImage.classList.add('hidden');
            els.artifactPreviewImage.removeAttribute('src');
        }
        if (els.artifactCompareImage) {
            els.artifactCompareImage.classList.add('hidden');
            els.artifactCompareImage.removeAttribute('src');
        }
        _renderArtifactMetadataCard(selected, selectedArtifact);
    }

    const fragment = document.createDocumentFragment();
    artifacts.forEach((artifact) => {
        const button = document.createElement('button');
        const active = selectedArtifact && artifact.path === selectedArtifact.path;
        button.type = 'button';
        button.dataset.artifactPath = _artifactRouteKey(artifact);
        button.setAttribute('role', 'option');
        button.setAttribute('aria-selected', active ? 'true' : 'false');
        button.tabIndex = active ? 0 : -1;
        button.className = active
            ? 'rounded-2xl border border-cyan-300 dark:border-cyan-900/60 bg-cyan-50/90 dark:bg-cyan-900/20 p-3 text-left shadow-sm transition-colors'
            : 'rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-900/50 p-3 text-left hover:bg-white/90 dark:hover:bg-slate-800/80 transition-colors';

        if (artifactIsPreviewable(artifact)) {
            const thumb = document.createElement('img');
            thumb.alt = artifactLabel(artifact);
            thumb.src = buildArtifactUrl(selected, artifact);
            thumb.className = 'h-24 w-full rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-900/60 object-cover';
            button.appendChild(thumb);
        } else {
            const placeholder = document.createElement('div');
            placeholder.className = 'flex h-24 items-center justify-center rounded-xl border border-dashed border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/70 text-[11px] font-bold uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400';
            placeholder.textContent = artifactDisplayLabel(artifact);
            button.appendChild(placeholder);
        }

        const title = document.createElement('p');
        title.className = 'mt-3 text-[11px] font-semibold text-slate-800 dark:text-slate-100 truncate';
        title.textContent = artifactNameParts(artifact).fileName;
        button.appendChild(title);

        const meta = document.createElement('p');
        meta.className = 'mt-1 text-[10px] font-mono text-slate-500 dark:text-slate-400 truncate';
        meta.textContent = `${artifactDisplayLabel(artifact)} • ${formatBytes(artifact.size_bytes)}`;
        button.appendChild(meta);

        fragment.appendChild(button);
    });
    els.artifactThumbnailRail.innerHTML = '';
    els.artifactThumbnailRail.appendChild(fragment);
    updateRunCardActions(selected);
    renderConsoleContextRibbon();
    _syncConsoleRoute(true);
}

function selectJob(jobId) {
    const previousJobId = state.selectedJobId;
    state.selectedJobId = jobId;
    _rememberSelectedJob(jobId);
    const job = state.jobs.find((item) => item.id === jobId);
    if (job && els.logPane) {
        els.logPane.textContent = job.logs.join('\n') + (job.logs.length ? '\n' : '');
        els.logPane.scrollTop = els.logPane.scrollHeight;
    }
    renderReviewSurfaces();
    if (state.currentView === 'operate') {
        setInspectorTab('timeline');
    } else if (state.currentView === 'review') {
        setInspectorTab('overview');
    }
    if (state.currentView === 'operate' || state.currentView === 'review') {
        _syncConsoleRoute(true);
        updateConsoleViewContext();
    }
    if (job && String(jobId || '') !== String(previousJobId || '')) {
        void emitPortalEvent('job_selected', {
            surface: 'job_queue',
            metadata: {
                job_id: String(job.id || ''),
                pipeline: String(job.pipeline || '')
            }
        });
    }
    scheduleRenderJobQueue(false);
}

function hydrateJobFromServer(rawJob) {
    if (!rawJob || !rawJob.id) return null;
    const id = String(rawJob.id);
    const createdAt = parseTimestamp(rawJob.created_at || rawJob.started_at || rawJob.updated_at, Date.now());
    const startedAt = parseTimestamp(rawJob.started_at, 0);
    const finishedAt = parseTimestamp(rawJob.finished_at, 0);
    const lastEventAt = parseTimestamp(rawJob.last_event_at, 0);
    const updatedAt = parseTimestamp(rawJob.last_event_at || rawJob.updated_at || rawJob.finished_at || rawJob.started_at || rawJob.created_at, createdAt);
    const hydrated = {
        id,
        pipeline: String(rawJob.pipeline || 'unknown'),
        state: String(rawJob.state || 'queued'),
        progress: Math.max(0, Math.min(100, Number(rawJob.progress) || 0)),
        logs: Array.isArray(rawJob.logs_tail) ? rawJob.logs_tail.slice(-MAX_JOB_LOG_LINES) : [],
        artifacts: normalizeArtifactItems(rawJob.artifacts),
        run_summary: normalizeRunSummary(rawJob.run_summary),
        error: rawJob.error || null,
        eventSource: null,
        fetchAbortController: null,
        eventStreamUrl: String(rawJob.events_url || `/v1/jobs/${id}/events`),
        usesFetchSse: false,
        sseRetry: { attempt: 0, timer: null },
        reconnectBlocked: false,
        lastEventAt,
        mockInterval: null,
        startedAt,
        finishedAt,
        createdAt,
        updatedAt,
        timeline: [],
        transportWarnings: [],
        progressMilestones: []
    };
    _reconcileJobTimeline(hydrated);
    return hydrated;
}

function _isJobStreamRecoverable(job) {
    if (!job) return false;
    return job.state === 'running' || job.state === 'queued';
}

function _ensureJobStreamState(job) {
    if (!job) return;
    if (!job.sseRetry || typeof job.sseRetry !== 'object') {
        job.sseRetry = { attempt: 0, timer: null };
    } else {
        const attempt = Number(job.sseRetry.attempt) || 0;
        job.sseRetry.attempt = Math.max(0, attempt);
        if (!job.sseRetry.timer) job.sseRetry.timer = null;
    }
    if (typeof job.lastEventAt !== 'number') job.lastEventAt = 0;
    if (typeof job.usesFetchSse !== 'boolean') job.usesFetchSse = false;
    if (typeof job.reconnectBlocked !== 'boolean') job.reconnectBlocked = false;
    if (typeof job.eventStreamUrl !== 'string' || !job.eventStreamUrl.trim()) {
        const fallbackId = encodeURIComponent(String(job.id || ''));
        job.eventStreamUrl = `/v1/jobs/${fallbackId}/events`;
    }
    if (!Object.prototype.hasOwnProperty.call(job, 'fetchAbortController')) {
        job.fetchAbortController = null;
    }
}

function _isNativeEventSourceHandle(handle) {
    return Boolean(handle && typeof handle.readyState === 'number' && typeof handle.addEventListener === 'function');
}

function _nativeEventSourceReadyState(handle) {
    if (!_isNativeEventSourceHandle(handle)) return null;
    const readyState = Number(handle.readyState);
    return Number.isInteger(readyState) ? readyState : null;
}

function _jobHasActiveStream(job) {
    if (!job) return false;
    if (job.fetchAbortController) return true;
    if (!job.eventSource) return false;
    const nativeReadyState = _nativeEventSourceReadyState(job.eventSource);
    if (nativeReadyState === EVENT_SOURCE_READY_STATE_CLOSED) return false;
    return true;
}

function _clearSseRetry(job, resetAttempt = true) {
    if (!job || !job.sseRetry) return;
    if (job.sseRetry.timer) {
        clearTimeout(job.sseRetry.timer);
        job.sseRetry.timer = null;
    }
    if (resetAttempt) job.sseRetry.attempt = 0;
}

function _teardownJobEventStream(job) {
    if (!job) return;
    if (job.eventSource) {
        job.eventSource.close();
        job.eventSource = null;
    }
    if (job.fetchAbortController) {
        job.fetchAbortController.abort();
        job.fetchAbortController = null;
    }
}

function _markJobEventActivity(job) {
    if (!job) return;
    _ensureJobStreamState(job);
    job.lastEventAt = Date.now();
    job.updatedAt = job.lastEventAt;
    if (job.sseRetry && job.sseRetry.attempt > 0) {
        job.sseRetry.attempt = 0;
    }
}

function _syncHydratedJob(existing, hydrated, rawJob = null) {
    if (!existing || !hydrated) return;
    existing.pipeline = hydrated.pipeline;
    existing.state = hydrated.state;
    existing.progress = hydrated.progress;
    existing.logs = hydrated.logs;
    existing.error = hydrated.error;
    existing.artifacts = hydrated.artifacts;
    existing.run_summary = hydrated.run_summary;
    existing.startedAt = hydrated.startedAt || existing.startedAt || 0;
    existing.finishedAt = hydrated.finishedAt || existing.finishedAt || 0;
    existing.createdAt = hydrated.createdAt || existing.createdAt || Date.now();
    existing.updatedAt = hydrated.updatedAt || existing.updatedAt || existing.createdAt;
    existing.lastEventAt = hydrated.lastEventAt || existing.lastEventAt || 0;
    existing.timeline = Array.isArray(existing.timeline) && existing.timeline.length > 0 ? existing.timeline : hydrated.timeline;
    existing.transportWarnings = Array.isArray(existing.transportWarnings) ? existing.transportWarnings : hydrated.transportWarnings;
    existing.progressMilestones = Array.isArray(existing.progressMilestones) ? existing.progressMilestones : hydrated.progressMilestones;
    if (rawJob && typeof rawJob.events_url === 'string' && rawJob.events_url.trim()) {
        existing.eventStreamUrl = rawJob.events_url.trim();
    } else if (!existing.eventStreamUrl) {
        existing.eventStreamUrl = hydrated.eventStreamUrl;
    }
    _ensureJobStreamState(existing);
    _reconcileJobTimeline(existing);
    if (!_isJobStreamRecoverable(existing)) {
        existing.reconnectBlocked = false;
        _clearSseRetry(existing, true);
        _teardownJobEventStream(existing);
    }
}

async function refreshJobStatus(job) {
    if (!job || !job.id) return;
    try {
        const headers = _buildAuthHeaders({ 'Accept': 'application/json' });
        const encodedId = encodeURIComponent(String(job.id));
        const res = await fetch(`${API_BASE}/v1/jobs/${encodedId}`, { headers, cache: 'no-store' });
        if (!res.ok) return;
        const payload = await res.json();
        const rawJob = payload?.data;
        if (!rawJob || String(rawJob.id || '') !== String(job.id)) return;
        const hydrated = hydrateJobFromServer(rawJob);
        if (!hydrated) return;
        _syncHydratedJob(job, hydrated, rawJob);
        if (state.selectedJobId === job.id && els.logPane) {
            els.logPane.textContent = job.logs.join('\n') + (job.logs.length ? '\n' : '');
            els.logPane.scrollTop = els.logPane.scrollHeight;
            if (!_isJobStreamRecoverable(job) && els.logStatusIndicator) {
                els.logStatusIndicator.classList.add('hidden');
            }
        }
        scheduleRenderJobQueue();
    } catch {
        // best-effort refresh for reconnect convergence
    }
}

function scheduleSseReconnect(job) {
    if (!_isJobStreamRecoverable(job)) return;
    _ensureJobStreamState(job);
    if (job.reconnectBlocked) return;
    if (job.sseRetry.timer || _jobHasActiveStream(job)) return;

    const attempt = job.sseRetry.attempt + 1;
    const exponent = Math.min(attempt, SSE_RECONNECT_MAX_EXPONENT);
    const backoff = SSE_RECONNECT_BASE_DELAY_MS * (2 ** exponent);
    const delayMs = Math.min(SSE_RECONNECT_MAX_DELAY_MS, backoff) + Math.floor(Math.random() * SSE_RECONNECT_JITTER_MS);

    job.sseRetry.attempt = attempt;
    _noteTransportWarning(job, 'reconnect_pending', `Event stream disconnected. Reconnecting in ${Math.ceil(delayMs / 1000)}s (attempt ${attempt}).`, 'warn');
    appendJobLog(job, `[WARN] Event stream disconnected. Reconnecting in ${Math.ceil(delayMs / 1000)}s (attempt ${attempt}).`);
    logToPane(job.id, `[WARN] Event stream disconnected. Reconnecting in ${Math.ceil(delayMs / 1000)}s (attempt ${attempt}).`);
    scheduleRenderJobQueue();

    job.sseRetry.timer = setTimeout(() => {
        job.sseRetry.timer = null;
        void (async () => {
            if (!_isJobStreamRecoverable(job)) {
                _clearSseRetry(job, true);
                return;
            }
            _teardownJobEventStream(job);
            await refreshJobStatus(job);
            if (!_isJobStreamRecoverable(job)) {
                _clearSseRetry(job, true);
                return;
            }
            startJobEventStream(job, job.eventStreamUrl);
        })();
    }, delayMs);
}

function stopJobActivity(job) {
    if (!job) return;
    _ensureJobStreamState(job);
    _clearSseRetry(job, true);
    _teardownJobEventStream(job);
    job.lastEventAt = 0;
    job.usesFetchSse = false;
    job.reconnectBlocked = false;
    if (job.mockInterval) {
        clearInterval(job.mockInterval);
        job.mockInterval = null;
    }
}

function cleanupActiveJobHandles() {
    state.jobs.forEach(stopJobActivity);
    if (healthPollIntervalId !== null) stopHealthPolling();
    if (sseWatchdogIntervalId !== null) stopSseWatchdog();
    if (_hasActiveBootstrapRetryState()) {
        _finalizeBootstrapRetry('terminal_navigation_abort', { reason: 'navigation_abort' });
    }
    _cancelPendingBootstrapRequest('navigation_abort');
}

function scheduleRenderJobQueue(includeReviewSurfaces = true) {
    queuedReviewSurfaceRefresh = queuedReviewSurfaceRefresh || includeReviewSurfaces;
    if (queueRenderScheduled) return;
    queueRenderScheduled = true;
    requestAnimationFrame(() => {
        queueRenderScheduled = false;
        const shouldRenderReviewSurfaces = queuedReviewSurfaceRefresh;
        queuedReviewSurfaceRefresh = false;
        renderJobQueue(shouldRenderReviewSurfaces);
    });
}

async function fetchWithTimeout(url, options = {}, timeoutMs = HEALTH_CHECK_TIMEOUT_MS, timeoutReason = 'request_timeout', lifecycle = null) {
    const controller = new AbortController();
    let didTimeout = false;
    const timeoutId = setTimeout(() => {
        didTimeout = true;
        controller.abort(timeoutReason);
    }, timeoutMs);
    let abortListener = null;
    if (options && options.signal && options.signal !== controller.signal) {
        if (options.signal.aborted) {
            controller.abort(options.signal.reason || 'request_aborted');
        } else {
            abortListener = () => controller.abort(options.signal.reason || 'request_aborted');
            options.signal.addEventListener('abort', abortListener, { once: true });
        }
    }
    if (lifecycle && typeof lifecycle.onStart === 'function') {
        lifecycle.onStart(controller, timeoutId);
    }
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } catch (error) {
        if (didTimeout) {
            const timeoutError = new Error(timeoutReason);
            timeoutError.name = 'AppTimeoutError';
            timeoutError.reason = timeoutReason;
            throw timeoutError;
        }
        throw error;
    } finally {
        clearTimeout(timeoutId);
        if (abortListener && options && options.signal) {
            options.signal.removeEventListener('abort', abortListener);
        }
        if (lifecycle && typeof lifecycle.onFinally === 'function') {
            lifecycle.onFinally(controller, timeoutId);
        }
    }
}

const TRUTHY_VALUES = new Set(['1', 'true', 'yes', 'on']);
const FALSY_VALUES = new Set(['0', 'false', 'no', 'off']);
const LUX_QUALITY_TIERS = new Set(['standard', 'premium', 'apex']);
const LUX_DEPTH_BACKENDS = new Set(['da3', 'depth_pro']);
const LUX_SEGMENTATION_BACKENDS = new Set(['stub', 'efficientsam', 'sam2']);
const SAM2_MODEL_SIZES = new Set(['base', 'large']);
const LUX_GROUPING_MODES = new Set(['single', 'parent_dir']);

function parseBoolLike(value, defaultValue = false) {
    if (value === null || value === undefined) return defaultValue;
    if (typeof value === 'boolean') return value;
    if (typeof value === 'number') return value !== 0;
    const normalized = String(value).trim().toLowerCase();
    if (!normalized) return defaultValue;
    if (TRUTHY_VALUES.has(normalized)) return true;
    if (FALSY_VALUES.has(normalized)) return false;
    return defaultValue;
}

function canonicalDepthBackend(value) {
    const backend = String(value || '').trim().toLowerCase();
    if (!backend) return 'da3';
    if (backend === 'depth_anything_v3' || backend === 'depth-anything-v3') return 'da3';
    return backend;
}

function _textOrFallback(value, fallback = '') {
    const text = String(value ?? '').trim();
    if (text) return text;
    return String(fallback ?? '').trim();
}

function _resolveQualityTier(value) {
    const normalized = String(value || '').trim().toLowerCase();
    if (LUX_QUALITY_TIERS.has(normalized)) return normalized;
    return 'standard';
}

function _resolveDepthBackend(value) {
    const normalized = canonicalDepthBackend(value);
    if (LUX_DEPTH_BACKENDS.has(normalized)) return normalized;
    return 'da3';
}

function _resolveSegmentationBackend(value) {
    const normalized = String(value || '').trim().toLowerCase();
    if (LUX_SEGMENTATION_BACKENDS.has(normalized)) return normalized;
    return 'stub';
}

function _resolveSam2ModelSize(value) {
    const normalized = String(value || '').trim().toLowerCase();
    if (SAM2_MODEL_SIZES.has(normalized)) return normalized;
    return 'base';
}

function _resolveGroupingMode(value) {
    const normalized = String(value || '').trim().toLowerCase();
    if (LUX_GROUPING_MODES.has(normalized)) return normalized;
    return 'single';
}

function _parsePositiveIntOrNull(value) {
    if (value === null || value === undefined) return null;
    const text = String(value).trim();
    if (!text) return null;
    const parsed = Number.parseInt(text, 10);
    if (!Number.isFinite(parsed) || parsed < 1) return null;
    return parsed;
}

function _normalizeVerboseQuietFlags(flags, notify = false) {
    if (!flags) return false;
    flags.verbose = parseBoolLike(flags.verbose, false);
    flags.quiet = parseBoolLike(flags.quiet, false);
    if (!flags.verbose || !flags.quiet) return false;
    flags.quiet = false;
    if (notify) {
        createToast('verbose and quiet are mutually exclusive; disabled quiet.', 'info');
    }
    return true;
}

function syncSegmentationControlState(config) {
    if (!els.segmentation.sam2ModelSize || !els.segmentation.backend) return;
    const backend = _resolveSegmentationBackend(config.segmentation?.backend);
    els.segmentation.sam2ModelSize.disabled = backend !== 'sam2';
    if (els.segmentation.sam2CheckpointPath) {
        els.segmentation.sam2CheckpointPath.disabled = backend !== 'sam2';
    }
}

const RECON_RUNTIME_DEFAULT_STATUS = Object.freeze({
    grouping_mode: 'Preserved even when reconstruction is toggled off.',
    reconstruction_iterations: 'Balanced default: 1000 iterations.',
    cameras_sidecar_path: 'Recommended for multi-view reconstruction runs.',
    reconstruction_tier: 'Metadata-backed tier options update when the backend is online.',
    raw_ingest_mode: 'Auto balances safety and runtime for the current ingest contract.',
    max_workers: 'Auto lets the backend choose a safe CPU worker cap.',
    max_gpu_workers: 'Auto keeps GPU parallelism conservative to reduce VRAM contention.',
    log_level: 'Use the default log level unless you need deeper runtime detail.'
});

function _normalizeWorkerMode(value) {
    return String(value || '').trim().toLowerCase() === 'fixed' ? 'fixed' : 'auto';
}

function _portalEstimateBand(score) {
    if (score <= 1) return 'low';
    if (score === 2) return 'medium';
    return 'high';
}

function _titleizeEstimateToken(value, fallback = 'Unknown') {
    const token = String(value || '').trim();
    if (!token) return fallback;
    return token
        .split(/[_\s-]+/)
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' ');
}

function _buildLocalEstimateSummary(args = {}) {
    const normalizedArgs = args && typeof args === 'object' ? args : {};
    let runtimeScore = 0;
    let gpuScore = 0;
    let researchRisk = 'none';
    const reasons = [];

    if (String(normalizedArgs.quality_tier || '').trim().toLowerCase() === 'apex') {
        runtimeScore += 1;
    }
    if (String(normalizedArgs.depth_backend || '').trim().toLowerCase() === 'depth_pro') {
        runtimeScore += 1;
        gpuScore += 1;
        researchRisk = 'research_only';
        reasons.push('depth_pro_research_backend');
    }
    if (
        parseBoolLike(normalizedArgs.enable_segmentation, false)
        && String(normalizedArgs.segmentation_backend || '').trim().toLowerCase() === 'sam2'
    ) {
        runtimeScore += 1;
        gpuScore += 1;
        reasons.push('sam2_segmentation');
    }
    if (parseBoolLike(normalizedArgs.enable_reconstruction, false)) {
        runtimeScore += 1;
        gpuScore += 1;
        researchRisk = 'research_only';
        reasons.push('scene_reconstruction');
        const iterations = Number.parseInt(String(normalizedArgs.reconstruction_iterations || '1000'), 10) || 1000;
        if (iterations >= 2000) {
            runtimeScore += 1;
            reasons.push('high_iteration_count');
        }
        if (iterations >= 3000) {
            gpuScore += 1;
        }
        const tier = String(normalizedArgs.reconstruction_tier || 'apex_research').trim().toLowerCase();
        if (tier === 'apex_research_ultra') {
            runtimeScore += 1;
            gpuScore += 1;
            reasons.push('ultra_reconstruction_tier');
        } else if (tier === 'experimental') {
            runtimeScore += 1;
            gpuScore += 1;
            researchRisk = 'experimental';
            reasons.push('experimental_reconstruction_tier');
        }
    }
    if (String(normalizedArgs.raw_ingest_mode || '').trim().toLowerCase() === 'force_rawpy') {
        runtimeScore += 1;
        reasons.push('forced_rawpy_ingest');
    }
    if (parseBoolLike(normalizedArgs.emit_scene_debug_bundle, false)) {
        runtimeScore += 1;
        reasons.push('debug_bundle_emission');
    }

    const maxWorkers = Number.parseInt(String(normalizedArgs.max_workers || ''), 10);
    if (Number.isFinite(maxWorkers) && maxWorkers > 4) {
        runtimeScore += 1;
        reasons.push('cpu_worker_override');
    }
    const maxGpuWorkers = Number.parseInt(String(normalizedArgs.max_gpu_workers || ''), 10);
    if (Number.isFinite(maxGpuWorkers) && maxGpuWorkers >= 2) {
        gpuScore += 1;
        reasons.push('gpu_worker_override');
    }
    if (researchRisk === 'none' && String(normalizedArgs.preset || '').toLowerCase().includes('research')) {
        researchRisk = 'research_only';
    }

    const runtimeBand = _portalEstimateBand(runtimeScore);
    const gpuBand = _portalEstimateBand(gpuScore);
    return {
        runtime_band: runtimeBand,
        gpu_pressure: gpuBand,
        research_risk: researchRisk,
        reasons,
        summary_label: `${_titleizeEstimateToken(runtimeBand)} runtime · ${_titleizeEstimateToken(gpuBand)} GPU pressure · ${_titleizeEstimateToken(researchRisk)} posture`
    };
}

function _buildLocalDebugBundleSummary(args = {}) {
    const outputDir = String(args.output_dir || '').trim();
    return {
        enabled: parseBoolLike(args.emit_scene_debug_bundle, false),
        requires_acknowledgement: parseBoolLike(args.emit_scene_debug_bundle, false),
        output_root: outputDir,
        destination: 'reconstruction/<scene-fingerprint>/debug',
        includes: ['scene_manifest', 'camera_payload', 'input_image_copies', 'segmentation_overlays', 'reprojection_preview'],
        sensitivity: 'camera_metadata_and_source_images',
        notes: [
            'Debug bundles may copy source imagery and camera metadata into the output tree.',
            'Portal dispatch requires an explicit acknowledgement before enabling debug bundle emission.'
        ]
    };
}

function _emptyPreviewState(status = 'idle', pipeline = '') {
    return {
        pipeline,
        requestKey: '',
        status,
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
        argv_preview: '',
        error: '',
        error_reason: '',
        error_status: 0
    };
}

function _setPreviewState(nextState) {
    state.preview = {
        ..._emptyPreviewState(state.preview?.status || 'idle', state.pipeline),
        ...(nextState && typeof nextState === 'object' ? nextState : {})
    };
}

function _currentBuildSurfacePathFieldValue(fieldName) {
    switch (String(fieldName || '').trim()) {
        case 'input_dir':
            return _textOrFallback(els.inputDir ? els.inputDir.value : state.config?.inputDir, state.config?.inputDir || '');
        case 'output_dir':
            return _textOrFallback(els.outputDir ? els.outputDir.value : state.config?.outputDir, state.config?.outputDir || '');
        case 'archive_index':
            return _textOrFallback(
                els.archiveIndexPath ? els.archiveIndexPath.value : state.config?.gate?.archiveIndex,
                state.config?.gate?.archiveIndex || ''
            );
        case 'manifest_jsonl':
            return _textOrFallback(
                els.rightsManifestPath ? els.rightsManifestPath.value : state.config?.gate?.manifestJsonl,
                state.config?.gate?.manifestJsonl || ''
            );
        case 'sam2_checkpoint_path':
            return _textOrFallback(
                els.segmentation.sam2CheckpointPath ? els.segmentation.sam2CheckpointPath.value : state.config?.segmentation?.sam2CheckpointPath,
                state.config?.segmentation?.sam2CheckpointPath || ''
            );
        case 'cameras_sidecar_path':
            return _textOrFallback(
                els.reconstruction.camerasSidecarPath ? els.reconstruction.camerasSidecarPath.value : state.config?.reconstruction?.camerasSidecarPath,
                state.config?.reconstruction?.camerasSidecarPath || ''
            );
        default:
            return '';
    }
}

function _previewFailureDetails(preview = null) {
    const matchedPreview = preview && typeof preview === 'object' ? preview : _currentPreviewForPayload();
    const reason = String(matchedPreview?.error_reason || matchedPreview?.error || '').trim().toLowerCase();

    if (reason === 'auth_failure' || reason === 'preview_auth_failed') {
        return {
            reason: 'auth_failure',
            summaryLabel: 'Preview auth failed',
            healthLabel: 'preview auth failed',
            luxBlockedMessage: 'BLOCKED: Preview-backed validation could not authenticate. Ensure the portal API key matches TP_API_KEY before dispatch.',
            archiveWarningMessage: 'WARNING: Preview-backed validation could not authenticate. Local rendering is shown until preview auth is restored.',
            toastMessage: 'Preview-backed validation could not authenticate. Ensure the portal API key matches TP_API_KEY.',
            telemetryReason: 'preview_auth_failed'
        };
    }
    if (reason === 'validation_error' || reason === 'preview_validation_error') {
        return {
            reason: 'validation_error',
            summaryLabel: 'Preview invalid',
            healthLabel: 'preview invalid',
            luxBlockedMessage: 'BLOCKED: Preview-backed validation rejected the current Lux configuration. Review the active inputs and retry.',
            archiveWarningMessage: 'WARNING: Preview-backed validation rejected the current configuration. Local rendering is shown until the request is corrected.',
            toastMessage: 'Preview-backed validation rejected the current configuration. Review the active inputs and retry.',
            telemetryReason: 'preview_validation_error'
        };
    }
    return {
        reason: 'service_failure',
        summaryLabel: 'Preview unavailable',
        healthLabel: 'preview unavailable',
        luxBlockedMessage: 'BLOCKED: Preview-backed validation is unavailable. Wait for validation to recover before dispatch.',
        archiveWarningMessage: 'WARNING: Preview-backed validation is unavailable. Local rendering is shown until validation recovers.',
        toastMessage: 'Preview-backed validation is unavailable. Dispatch stays paused until validation recovers.',
        telemetryReason: 'preview_service_unavailable'
    };
}

function _nextBestActionLabel(fieldName, fallback) {
    const normalizedField = String(fieldName || '').trim().replace(/_/g, ' ');
    if (!normalizedField || normalizedField === 'payload') return fallback;
    return `Resolve ${normalizedField}`;
}

function _nextBestActionDetail(issue, fallback) {
    if (!issue || typeof issue !== 'object') return fallback;
    const message = String(issue.message || '').trim().replace(/^(BLOCKED|WARNING):\s*/i, '');
    const suggestion = String(issue.suggestion || '').trim();
    const combined = suggestion && suggestion !== message
        ? `${message || fallback} ${suggestion}`.trim()
        : (message || fallback);
    return combined || fallback;
}

function _normalizeNextBestAction(rawAction) {
    if (!rawAction || typeof rawAction !== 'object') return null;
    const action = String(rawAction.action || '').trim();
    const label = String(rawAction.label || '').trim();
    const detail = String(rawAction.detail || '').trim();
    const field = String(rawAction.field || '').trim();
    const tone = String(rawAction.tone || '').trim().toLowerCase();
    return {
        action,
        field,
        label: label || 'Review dispatch posture',
        detail: detail || 'Preview guidance will summarize the clearest next step for this draft.',
        tone: ['blocked', 'warning', 'ready', 'info'].includes(tone) ? tone : 'info'
    };
}

function _buildLocalNextBestAction(payload = null, preview = null) {
    const currentPayload = payload || generatePayload();
    const matchedPreview = preview && typeof preview === 'object' ? preview : _currentPreviewForPayload(currentPayload);
    const previewErrors = matchedPreview && Array.isArray(matchedPreview.field_errors) ? matchedPreview.field_errors : [];
    const previewWarnings = matchedPreview && Array.isArray(matchedPreview.field_warnings) ? matchedPreview.field_warnings : [];
    const readinessIssues = currentPipelineReadinessIssues(currentPayload);
    const blockedReadinessIssue = readinessIssues.find((issue) => String(issue?.severity || '').trim().toLowerCase() === 'blocked');
    const readinessWarning = readinessIssues.find((issue) => String(issue?.severity || '').trim().toLowerCase() !== 'blocked');

    if (matchedPreview?.status === 'loading') {
        return {
            action: 'wait_for_preview',
            field: 'config_preview',
            label: 'Wait for preview to refresh',
            detail: 'Preview-backed validation is refreshing the current draft. Review readiness again when it completes.',
            tone: 'info'
        };
    }

    if (previewErrors.length > 0) {
        const issue = previewErrors[0];
        const field = String(issue?.field || '').trim();
        return {
            action: 'resolve_validation_error',
            field,
            label: _nextBestActionLabel(field, 'Resolve configuration issue'),
            detail: _nextBestActionDetail(issue, 'Resolve the current configuration issue before dispatch.'),
            tone: 'blocked'
        };
    }

    if (!state.backendOk) {
        return {
            action: 'restore_backend_connection',
            field: 'backend_connection',
            label: 'Restore backend connection',
            detail: 'Preview-backed validation and dispatch resume when the orchestrator backend is reachable again.',
            tone: 'blocked'
        };
    }

    if (matchedPreview?.status === 'error') {
        const previewFailure = _previewFailureDetails(matchedPreview);
        const detail = currentPayload.pipeline === 'lux-depth-v3'
            ? String(previewFailure.luxBlockedMessage || '').replace(/^BLOCKED:\s*/i, '')
            : String(previewFailure.archiveWarningMessage || '').replace(/^WARNING:\s*/i, '');
        return {
            action: 'resolve_preview_error',
            field: 'config_preview',
            label: currentPayload.pipeline === 'lux-depth-v3' ? 'Resolve preview validation' : 'Review preview status',
            detail: detail || 'Preview-backed validation needs attention before dispatch.',
            tone: currentPayload.pipeline === 'lux-depth-v3' ? 'blocked' : 'warning'
        };
    }

    if (blockedReadinessIssue) {
        const field = String(blockedReadinessIssue?.field || '').trim();
        return {
            action: 'resolve_readiness',
            field,
            label: _nextBestActionLabel(field, 'Resolve readiness prerequisite'),
            detail: _nextBestActionDetail(blockedReadinessIssue, 'A dispatch prerequisite is still missing.'),
            tone: 'blocked'
        };
    }

    if (previewWarnings.length > 0) {
        const issue = previewWarnings[0];
        const field = String(issue?.field || '').trim();
        return {
            action: 'review_warning',
            field,
            label: _nextBestActionLabel(field, 'Review warning before dispatch'),
            detail: _nextBestActionDetail(issue, 'Review the current warning before dispatch.'),
            tone: 'warning'
        };
    }

    if (readinessWarning) {
        const field = String(readinessWarning?.field || '').trim();
        return {
            action: 'review_readiness_warning',
            field,
            label: _nextBestActionLabel(field, 'Review readiness warning'),
            detail: _nextBestActionDetail(readinessWarning, 'Review the current readiness warning before dispatch.'),
            tone: 'warning'
        };
    }

    const readiness = currentPipelineReadiness(currentPayload);
    const canonicalCommand = String(readiness?.canonical_command || canonicalArchiveCommand(currentPayload.pipeline) || '').trim();
    if (currentPayload.pipeline === 'lux-depth-v3') {
        return {
            action: 'dispatch_ready',
            field: 'run_job',
            label: 'Execute the Lux run',
            detail: 'Preview-backed validation is ready. Review the expected outputs and dispatch when satisfied.',
            tone: 'ready'
        };
    }
    return {
        action: 'dispatch_ready',
        field: 'run_job',
        label: 'Dispatch the archive stage',
        detail: canonicalCommand
            ? `Canonical command ${canonicalCommand} is ready. Review the expected outputs and dispatch when satisfied.`
            : 'Archive readiness is clear. Review the expected outputs and dispatch when satisfied.',
        tone: 'ready'
    };
}

function _effectiveNextBestAction(payload = null, preview = null) {
    const currentPayload = payload || generatePayload();
    const currentPreview = preview && typeof preview === 'object' ? preview : _currentPreviewForPayload(currentPayload);
    return _normalizeNextBestAction(currentPreview?.next_best_action) || _buildLocalNextBestAction(currentPayload, currentPreview);
}

function renderNextBestAction(payload = null, preview = null) {
    if (!els.nextBestActionLabel || !els.nextBestActionDetail || !els.nextBestActionTone) return;
    const action = _effectiveNextBestAction(payload, preview);
    els.nextBestActionLabel.textContent = String(action?.label || 'Review dispatch posture');
    els.nextBestActionDetail.textContent = String(
        action?.detail || 'Preview guidance will summarize the clearest next step for this draft.'
    );
    const tone = String(action?.tone || 'info').trim().toLowerCase();
    els.nextBestActionTone.dataset.tone = tone || 'info';
    els.nextBestActionTone.textContent = titleCaseToken(tone || 'info', 'Info');
}

function _setBuildSurfacePathFieldValue(fieldName, nextValue) {
    const value = _textOrFallback(nextValue, '');
    state.config = state.config || {};
    switch (String(fieldName || '').trim()) {
        case 'input_dir':
            state.config.inputDir = value;
            if (els.inputDir) els.inputDir.value = value;
            return true;
        case 'output_dir':
            state.config.outputDir = value;
            if (els.outputDir) els.outputDir.value = value;
            return true;
        case 'archive_index':
            state.config.gate = state.config.gate || {};
            state.config.gate.archiveIndex = value;
            if (els.archiveIndexPath) els.archiveIndexPath.value = value;
            return true;
        case 'manifest_jsonl':
            state.config.gate = state.config.gate || {};
            state.config.gate.manifestJsonl = value;
            if (els.rightsManifestPath) els.rightsManifestPath.value = value;
            return true;
        case 'sam2_checkpoint_path':
            state.config.segmentation = state.config.segmentation || {};
            state.config.segmentation.sam2CheckpointPath = value;
            if (els.segmentation.sam2CheckpointPath) els.segmentation.sam2CheckpointPath.value = value;
            return true;
        case 'cameras_sidecar_path':
            state.config.reconstruction = state.config.reconstruction || {};
            state.config.reconstruction.camerasSidecarPath = value;
            if (els.reconstruction.camerasSidecarPath) els.reconstruction.camerasSidecarPath.value = value;
            return true;
        default:
            return false;
    }
}

function _reconcilePreviewRepairedPaths(previewState) {
    const preview = previewState && typeof previewState === 'object' ? previewState : _emptyPreviewState();
    const warnings = Array.isArray(preview.field_warnings) ? preview.field_warnings : [];
    const normalizedArgs = preview.normalized_args && typeof preview.normalized_args === 'object' ? preview.normalized_args : {};
    const submittedArgs = preview.submitted_args && typeof preview.submitted_args === 'object' ? preview.submitted_args : {};
    let changed = false;

    warnings.forEach((issue) => {
        if (String(issue?.code || '') !== 'repo_local_path_repaired') return;
        const fieldName = String(issue?.field || '').trim();
        if (!fieldName) return;
        const submittedValue = _textOrFallback(submittedArgs[fieldName], '');
        const normalizedValue = _textOrFallback(normalizedArgs[fieldName], '');
        if (!submittedValue || !normalizedValue || submittedValue === normalizedValue) return;
        if (_currentBuildSurfacePathFieldValue(fieldName) !== submittedValue) return;
        changed = _setBuildSurfacePathFieldValue(fieldName, normalizedValue) || changed;
    });

    if (!changed) return preview;
    const currentPayload = generatePayload();
    return {
        ...preview,
        requestKey: _configPreviewRequestKey(currentPayload),
        submitted_args: currentPayload.args && typeof currentPayload.args === 'object' ? { ...currentPayload.args } : {}
    };
}

function _configPreviewRequestKey(payload) {
    if (!payload || typeof payload !== 'object') return '';
    try {
        return JSON.stringify({ pipeline: payload.pipeline || '', args: payload.args || {} });
    } catch {
        return `${String(payload.pipeline || '')}:${Date.now()}`;
    }
}

function _currentPreviewForPayload(payload = null) {
    const currentPayload = payload || generatePayload();
    const requestKey = _configPreviewRequestKey(currentPayload);
    if (
        state.preview
        && state.preview.pipeline === currentPayload.pipeline
        && state.preview.requestKey === requestKey
    ) {
        return state.preview;
    }
    return null;
}

function _effectivePreviewSnapshot(payload = null) {
    const currentPayload = payload || generatePayload();
    const preview = _currentPreviewForPayload(currentPayload);
    if (preview && preview.status === 'ready') return preview;
    return {
        pipeline: currentPayload.pipeline,
        requestKey: _configPreviewRequestKey(currentPayload),
        status: state.backendOk ? 'local_fallback' : 'offline',
        field_errors: [],
        field_warnings: [],
        inactive_fields: [],
        normalized_args: { ...(currentPayload.args || {}) },
        execution_args: { ...(currentPayload.args || {}) },
        submitted_args: { ...(currentPayload.args || {}) },
        readiness: null,
        estimate_summary: _buildLocalEstimateSummary(currentPayload.args || {}),
        debug_bundle_summary: _buildLocalDebugBundleSummary(currentPayload.args || {}),
        next_best_action: _buildLocalNextBestAction(currentPayload, preview),
        argv_preview: '',
        error: ''
    };
}

function _effectiveDebugBundleEnabled(preview = null, payload = null) {
    const currentPayload = payload || generatePayload();
    const matchedPreview = preview || _currentPreviewForPayload(currentPayload);
    if (matchedPreview && matchedPreview.status === 'ready') {
        return parseBoolLike(matchedPreview.normalized_args?.emit_scene_debug_bundle, false);
    }
    return parseBoolLike(currentPayload.args?.enable_reconstruction, false)
        && parseBoolLike(currentPayload.args?.emit_scene_debug_bundle, false);
}

function _configPreviewEnabledForPipeline(pipelineName = state.pipeline) {
    return CONFIG_PREVIEW_SUPPORTED_PIPELINES.has(String(pipelineName || '').trim()) && state.backendOk && _isBootstrapReady();
}

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

function _metadataField(name) {
    const fields = state.metadata && state.metadata.fields && typeof state.metadata.fields === 'object'
        ? state.metadata.fields
        : {};
    const value = fields[name];
    return value && typeof value === 'object' ? value : null;
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
    if (els.reconstruction.iterations && iterationsField?.min) {
        els.reconstruction.iterations.min = String(iterationsField.min);
    }
    if (els.runtime.maxWorkers && maxWorkersField?.min) {
        els.runtime.maxWorkers.min = String(maxWorkersField.min);
    }
    if (els.runtime.maxGpuWorkers && maxGpuWorkersField?.min) {
        els.runtime.maxGpuWorkers.min = String(maxGpuWorkersField.min);
    }
}

function _previewIssueForField(fieldName, payload = null) {
    const preview = _currentPreviewForPayload(payload);
    if (!preview || preview.status !== 'ready') return null;
    const errors = Array.isArray(preview.field_errors) ? preview.field_errors : [];
    const warnings = Array.isArray(preview.field_warnings) ? preview.field_warnings : [];
    const error = errors.find((item) => String(item?.field || '') === fieldName);
    if (error) return { tone: 'error', detail: error };
    const warning = warnings.find((item) => String(item?.field || '') === fieldName);
    if (warning) return { tone: 'warning', detail: warning };
    return null;
}

function _renderIssueStatus(el, helperText, issue) {
    if (!el) return;
    el.textContent = issue?.detail?.message || helperText;
    el.classList.remove('text-slate-500', 'dark:text-slate-400', 'text-amber-700', 'dark:text-amber-300', 'text-red-600', 'dark:text-red-300');
    if (issue?.tone === 'error') {
        el.classList.add('text-red-600', 'dark:text-red-300');
    } else if (issue?.tone === 'warning') {
        el.classList.add('text-amber-700', 'dark:text-amber-300');
    } else {
        el.classList.add('text-slate-500', 'dark:text-slate-400');
    }
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

function renderFieldPreviewStatuses(payload = null) {
    const currentPayload = payload || generatePayload();
    _renderIssueStatus(
        els.inputDirStatus,
        _buildFieldStatusCopy('input_dir', currentPayload),
        _previewIssueForField('input_dir', currentPayload)
    );
    _renderIssueStatus(
        els.outputDirStatus,
        _buildFieldStatusCopy('output_dir', currentPayload),
        _previewIssueForField('output_dir', currentPayload)
    );
    _renderIssueStatus(
        els.archiveIndexStatus,
        _buildFieldStatusCopy('archive_index', currentPayload),
        _previewIssueForField('archive_index', currentPayload)
    );
    _renderIssueStatus(
        els.rightsManifestStatus,
        _buildFieldStatusCopy('manifest_jsonl', currentPayload),
        _previewIssueForField('manifest_jsonl', currentPayload)
    );
}

function _jobRecencyValue(job) {
    return Number(job?.lastEventAt || job?.updatedAt || job?.finishedAt || job?.startedAt || job?.createdAt || 0);
}

function _latestJobBy(predicate) {
    return state.jobs
        .filter((job) => Boolean(job) && predicate(job))
        .sort((left, right) => _jobRecencyValue(right) - _jobRecencyValue(left))[0] || null;
}

function _latestActiveJob() {
    return _latestJobBy((job) => job.state === 'running' || job.state === 'queued');
}

function _latestReviewableJob() {
    return _latestJobBy((job) => {
        const summary = normalizeRunSummary(job.run_summary);
        const artifactCount = Array.isArray(job.artifacts) ? job.artifacts.length : 0;
        return artifactCount > 0
            || Boolean(summary.reviewable_outputs)
            || job.state === 'succeeded'
            || job.state === 'partial';
    });
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

async function emitPortalEvent(eventType, options = {}) {
    if (!state.backendOk || !_isBootstrapReady()) return;
    const eventOptions = options && typeof options === 'object' ? options : {};
    const payload = {
        event_type: String(eventType || '').trim().toLowerCase(),
        pipeline: state.pipeline,
        surface: String(eventOptions.surface || '').trim().toLowerCase(),
        field: String(eventOptions.field || '').trim(),
        metadata: eventOptions.metadata && typeof eventOptions.metadata === 'object' ? eventOptions.metadata : {},
        reasons: Array.isArray(eventOptions.reasons) ? eventOptions.reasons : []
    };
    try {
        const headers = _buildAuthHeaders({ 'Content-Type': 'application/json' }, 'POST');
        await fetch(`${API_BASE}/v1/portal/events`, {
            method: 'POST',
            headers,
            body: JSON.stringify(payload)
        });
    } catch {
        // best-effort telemetry only
    }
}

async function fetchConfigMetadata(pipelineName = state.pipeline, silent = false) {
    if (String(pipelineName || '').trim() !== 'lux-depth-v3' || !state.backendOk || !_isBootstrapReady()) {
        state.metadata = {
            pipeline: '',
            fields: {},
            estimate_bands: {},
            debug_bundle_policy: {},
            advanced_sections: []
        };
        renderReviewSurfaces();
        return;
    }
    try {
        const headers = _buildAuthHeaders({ 'Accept': 'application/json' });
        const res = await fetch(`${API_BASE}/v1/config-metadata?pipeline=${encodeURIComponent(pipelineName)}`, {
            headers,
            cache: 'no-store'
        });
        if (!res.ok) throw new Error(`config metadata fetch failed (${res.status})`);
        const payload = await res.json();
        const data = payload?.success === true && payload?.data && typeof payload.data === 'object'
            ? payload.data
            : null;
        if (!data) throw new Error('invalid config metadata payload');
        state.metadata = {
            pipeline: String(data.pipeline || pipelineName),
            fields: data.fields && typeof data.fields === 'object' ? data.fields : {},
            estimate_bands: data.estimate_bands && typeof data.estimate_bands === 'object' ? data.estimate_bands : {},
            debug_bundle_policy: data.debug_bundle_policy && typeof data.debug_bundle_policy === 'object' ? data.debug_bundle_policy : {},
            advanced_sections: Array.isArray(data.advanced_sections) ? data.advanced_sections.map((item) => String(item || '')) : []
        };
        applyLuxMetadataToControls();
        renderReviewSurfaces();
        scheduleConfigPreview(true);
    } catch {
        if (!silent) createToast('Failed to refresh config metadata from backend.', 'error');
    }
}

async function fetchConfigPreview(payload) {
    const currentPayload = payload && typeof payload === 'object' ? payload : generatePayload();
    const requestKey = _configPreviewRequestKey(currentPayload);
    const refreshPreviewDrivenSurfaces = (nextPayload = currentPayload) => {
        renderCLI();
        renderPreRunDiagnostics(nextPayload);
        _syncBootstrapGuardedControls();
    };

    if (!_configPreviewEnabledForPipeline(currentPayload.pipeline)) {
        _setPreviewState({
            ..._emptyPreviewState(state.backendOk ? 'local_fallback' : 'offline', currentPayload.pipeline),
            requestKey
        });
        refreshPreviewDrivenSurfaces(currentPayload);
        return;
    }

    _setPreviewState({
        ..._emptyPreviewState('loading', currentPayload.pipeline),
        requestKey,
        submitted_args: currentPayload.args && typeof currentPayload.args === 'object' ? { ...currentPayload.args } : {}
    });
    refreshPreviewDrivenSurfaces(currentPayload);

    try {
        const headers = _buildAuthHeaders({ 'Content-Type': 'application/json', 'Accept': 'application/json' }, 'POST');
        const res = await fetch(`${API_BASE}/v1/config-preview`, {
            method: 'POST',
            headers,
            body: JSON.stringify(currentPayload)
        });
        const response = await res.json();
        if (!res.ok) {
            const errorPayload = response?.error && typeof response.error === 'object' ? response.error : {};
            const errorDetails = errorPayload.details && typeof errorPayload.details === 'object' ? errorPayload.details : {};
            const classifiedFailure = _previewFailureDetails({
                error_reason: res.status === 401 || res.status === 403
                    ? 'auth_failure'
                    : res.status >= 400 && res.status < 500
                        ? 'validation_error'
                        : 'service_failure'
            });
            if (_configPreviewRequestKey(generatePayload()) !== requestKey) {
                return;
            }
            _setPreviewState({
                ..._emptyPreviewState('error', currentPayload.pipeline),
                requestKey,
                error: String(errorPayload.code || '').trim().toLowerCase() || 'preview_unavailable',
                error_reason: classifiedFailure.reason,
                error_status: res.status
            });
            if (classifiedFailure.reason === 'validation_error') {
                void emitPortalEvent('preview_error_seen', {
                    surface: 'reconstruction_runtime',
                    reasons: [
                        String(errorDetails.reason || '').trim().toLowerCase() || classifiedFailure.telemetryReason
                    ],
                    metadata: { status: res.status }
                });
            }
            return;
        }
        const data = response?.success === true && response?.data && typeof response.data === 'object'
            ? response.data
            : null;
        if (!data) throw new Error('invalid config preview payload');
        if (_configPreviewRequestKey(generatePayload()) !== requestKey) {
            return;
        }
        const previewFieldErrors = Array.isArray(data.field_errors) ? data.field_errors : [];
        const nextPreviewState = {
            pipeline: String(data.pipeline || currentPayload.pipeline),
            requestKey,
            status: 'ready',
            field_errors: previewFieldErrors,
            field_warnings: Array.isArray(data.field_warnings) ? data.field_warnings : [],
            inactive_fields: Array.isArray(data.inactive_fields) ? data.inactive_fields : [],
            normalized_args: data.normalized_args && typeof data.normalized_args === 'object' ? data.normalized_args : {},
            execution_args: data.execution_args && typeof data.execution_args === 'object' ? data.execution_args : {},
            submitted_args: currentPayload.args && typeof currentPayload.args === 'object' ? { ...currentPayload.args } : {},
            readiness: data.readiness && typeof data.readiness === 'object' ? data.readiness : null,
            estimate_summary: data.estimate_summary && typeof data.estimate_summary === 'object' ? data.estimate_summary : null,
            debug_bundle_summary: data.debug_bundle_summary && typeof data.debug_bundle_summary === 'object' ? data.debug_bundle_summary : null,
            next_best_action: _normalizeNextBestAction(data.next_best_action),
            argv_preview: String(data.argv_preview || ''),
            error: '',
            error_reason: '',
            error_status: 0
        };
        _setPreviewState(_reconcilePreviewRepairedPaths(nextPreviewState));
        if (previewFieldErrors.length > 0) {
            void emitPortalEvent('preview_error_seen', {
                surface: 'reconstruction_runtime',
                reasons: previewFieldErrors.map((item) => String(item?.code || '')).filter(Boolean).slice(0, 8),
                metadata: { count: previewFieldErrors.length }
            });
        }
    } catch {
        const classifiedFailure = _previewFailureDetails({ error_reason: 'service_failure' });
        _setPreviewState({
            ..._emptyPreviewState('error', currentPayload.pipeline),
            requestKey,
            error: 'preview_unavailable',
            error_reason: classifiedFailure.reason,
            error_status: 0
        });
    } finally {
        refreshPreviewDrivenSurfaces(generatePayload());
    }
}

function scheduleConfigPreview(immediate = false) {
    if (configPreviewTimerId !== null) {
        clearTimeout(configPreviewTimerId);
        configPreviewTimerId = null;
    }
    const payload = generatePayload();
    if (!CONFIG_PREVIEW_SUPPORTED_PIPELINES.has(String(payload.pipeline || '').trim())) {
        _setPreviewState(_emptyPreviewState('idle', payload.pipeline));
        renderCLI();
        return;
    }
    if (immediate) {
        void fetchConfigPreview(payload);
        return;
    }
    configPreviewTimerId = window.setTimeout(() => {
        configPreviewTimerId = null;
        void fetchConfigPreview(generatePayload());
    }, CONFIG_PREVIEW_DEBOUNCE_MS);
}

function _formatWorkerSummary(mode, value, label) {
    if (_normalizeWorkerMode(mode) !== 'fixed') return `${label} Auto`;
    const parsed = Number.parseInt(String(value || ''), 10);
    return Number.isFinite(parsed) && parsed > 0 ? `${label} ${parsed}` : `${label} Fixed`;
}

function _formatPreviewStateLabel(payload = null) {
    const preview = _currentPreviewForPayload(payload);
    if (!preview) {
        return state.backendOk ? 'Refreshing' : 'Local fallback';
    }
    if (preview.status === 'ready') return 'Preview ready';
    if (preview.status === 'loading') return 'Refreshing';
    if (preview.status === 'error') return _previewFailureDetails(preview).summaryLabel;
    if (preview.status === 'offline') return 'Local fallback';
    if (preview.status === 'local_fallback') return 'Local fallback';
    return 'Local fallback';
}

function _effectiveReadinessSummary(readiness) {
    if (!readiness || typeof readiness !== 'object') {
        const status = currentPipelineDispatchStatus();
        return status ? `Current readiness: ${_titleizeEstimateToken(status)}` : 'Readiness details load with the backend preview.';
    }
    const status = _titleizeEstimateToken(String(readiness.status || '').trim() || 'unknown');
    const issues = Array.isArray(readiness.missing_prerequisites) ? readiness.missing_prerequisites : [];
    if (issues.length === 0) return `Readiness: ${status}`;
    const firstIssue = issues[0];
    return `Readiness: ${status} · ${String(firstIssue?.message || 'Additional prerequisites reported.')}`;
}

function renderReconstructionRuntimeSummary(payload = null) {
    if (state.pipeline !== 'lux-depth-v3') return;
    const currentPayload = payload || generatePayload();
    const matchedPreview = _currentPreviewForPayload(currentPayload);
    const preview = matchedPreview || _effectivePreviewSnapshot(currentPayload);
    const previewStatus = matchedPreview ? matchedPreview.status : preview.status;
    const runtime = state.config.runtime || {};
    const estimate = preview.estimate_summary || _buildLocalEstimateSummary(currentPayload.args || {});
    const debugSummary = preview.debug_bundle_summary || _buildLocalDebugBundleSummary(currentPayload.args || {});
    const reconstructionEnabled = parseBoolLike(currentPayload.args.enable_reconstruction, false);
    const debugEnabled = _effectiveDebugBundleEnabled(matchedPreview, currentPayload);

    if (els.summaryReconstructionState) {
        els.summaryReconstructionState.textContent = reconstructionEnabled ? 'On' : 'Off';
    }
    if (els.summaryRuntimeWorkers) {
        els.summaryRuntimeWorkers.textContent = [
            _formatWorkerSummary(runtime.maxWorkersMode, currentPayload.args.max_workers, 'CPU'),
            _formatWorkerSummary(runtime.maxGpuWorkersMode, currentPayload.args.max_gpu_workers, 'GPU')
        ].join(' · ');
    }
    if (els.summaryRawIngest) {
        els.summaryRawIngest.textContent = _titleizeEstimateToken(currentPayload.args.raw_ingest_mode || 'auto', 'Auto');
    }
    if (els.summaryDebugBundle) {
        els.summaryDebugBundle.textContent = debugEnabled
            ? state.portalUi.debugBundleAcknowledged
                ? 'On · acknowledged'
                : 'On · acknowledgement needed'
            : 'Off';
    }
    if (els.summaryPreviewState) {
        els.summaryPreviewState.textContent = _formatPreviewStateLabel(currentPayload);
    }
    if (els.estimateRuntimeBand) {
        els.estimateRuntimeBand.textContent = _titleizeEstimateToken(estimate.runtime_band || 'low', 'Low');
    }
    if (els.estimateGpuBand) {
        els.estimateGpuBand.textContent = _titleizeEstimateToken(estimate.gpu_pressure || 'low', 'Low');
    }
    if (els.estimateResearchRisk) {
        els.estimateResearchRisk.textContent = _titleizeEstimateToken(estimate.research_risk || 'none', 'None');
    }
    if (els.estimateSummaryLabel) {
        els.estimateSummaryLabel.textContent = String(estimate.summary_label || 'No estimate available.');
    }
    if (els.reconstructionSummaryHint) {
        if (previewStatus === 'ready') {
            els.reconstructionSummaryHint.textContent = 'Applies to the next run. Preview-backed validation and normalization are live.';
        } else if (previewStatus === 'error') {
            els.reconstructionSummaryHint.textContent = 'Preview-backed validation is unavailable right now, so dispatch stays paused until it recovers.';
        } else if (!state.backendOk) {
            els.reconstructionSummaryHint.textContent = 'Applies to the next run. Backend preview is unavailable while the orchestrator is offline.';
        } else {
            els.reconstructionSummaryHint.textContent = 'Applies to the next run. Reconstruction-specific values are preserved when the feature is off.';
        }
    }

    _renderIssueStatus(
        els.reconstruction.groupingModeStatus,
        groupingFieldStatusText(),
        _previewIssueForField('grouping_mode', currentPayload)
    );
    _renderIssueStatus(
        els.reconstruction.iterationsStatus,
        iterationFieldStatusText(),
        _previewIssueForField('reconstruction_iterations', currentPayload)
    );
    _renderIssueStatus(
        els.reconstruction.camerasSidecarStatus,
        RECON_RUNTIME_DEFAULT_STATUS.cameras_sidecar_path,
        _previewIssueForField('cameras_sidecar_path', currentPayload)
    );
    _renderIssueStatus(
        els.reconstruction.tierStatus,
        reconstructionTierStatusText(),
        _previewIssueForField('reconstruction_tier', currentPayload)
    );
    _renderIssueStatus(
        els.raw.ingestModeStatus,
        rawIngestModeStatusText(),
        _previewIssueForField('raw_ingest_mode', currentPayload)
    );
    _renderIssueStatus(
        els.runtime.maxWorkersStatus,
        maxWorkersStatusText(),
        _previewIssueForField('max_workers', currentPayload)
    );
    _renderIssueStatus(
        els.runtime.maxGpuWorkersStatus,
        maxGpuWorkersStatusText(),
        _previewIssueForField('max_gpu_workers', currentPayload)
    );
    _renderIssueStatus(
        els.runtime.logLevelStatus,
        logLevelStatusText(),
        _previewIssueForField('log_level', currentPayload)
    );

    renderDebugBundleGuardrail(currentPayload, debugSummary, preview);
    renderEffectiveConfigDrawer(currentPayload, preview);
}

function renderDebugBundleGuardrail(payload = null, debugSummary = null, preview = null) {
    const currentPayload = payload || generatePayload();
    const currentPreview = preview || _effectivePreviewSnapshot(currentPayload);
    const summary = debugSummary || currentPreview.debug_bundle_summary || _buildLocalDebugBundleSummary(currentPayload.args || {});
    const enabled = _effectiveDebugBundleEnabled(currentPreview, currentPayload);
    const outputRoot = String(summary.output_root || '').trim();
    const destinationSuffix = String(summary.destination || '').trim();
    const destinationText = outputRoot && destinationSuffix
        ? `${outputRoot.replace(/[\\/]$/, '')}/${destinationSuffix.replace(/^[\\/]+/, '')}`
        : destinationSuffix || 'pending preview';

    if (els.debugBundleGuardrail) {
        els.debugBundleGuardrail.classList.toggle('hidden', !enabled);
    }
    if (!enabled) {
        state.portalUi.debugBundleAcknowledged = false;
        state.portalUi.debugBundleGuardrailSeen = false;
        if (els.debugBundleAcknowledge) {
            els.debugBundleAcknowledge.checked = false;
        }
        return;
    }

    if (els.debugBundleDestination) {
        els.debugBundleDestination.textContent = `Destination: ${destinationText}`;
    }
    if (els.debugBundleSensitivity) {
        els.debugBundleSensitivity.textContent = 'May include source images, camera metadata, segmentation overlays, and reprojection previews.';
    }
    if (els.debugBundleAcknowledge) {
        els.debugBundleAcknowledge.checked = Boolean(state.portalUi.debugBundleAcknowledged);
    }
    if (els.debugBundleAcknowledgeHint) {
        const previewIssue = _previewIssueForField('emit_scene_debug_bundle', currentPayload);
        if (previewIssue?.detail?.message) {
            els.debugBundleAcknowledgeHint.textContent = previewIssue.detail.message;
            els.debugBundleAcknowledgeHint.classList.remove('text-amber-700', 'dark:text-amber-300', 'text-red-600', 'dark:text-red-300');
            els.debugBundleAcknowledgeHint.classList.add(previewIssue.tone === 'error' ? 'text-red-600' : 'text-amber-700');
            if (previewIssue.tone === 'error') {
                els.debugBundleAcknowledgeHint.classList.add('dark:text-red-300');
            } else {
                els.debugBundleAcknowledgeHint.classList.add('dark:text-amber-300');
            }
        } else {
            els.debugBundleAcknowledgeHint.textContent = state.portalUi.debugBundleAcknowledged
                ? 'Acknowledged for the next dispatch.'
                : 'Required before dispatch when debug bundle emission is enabled.';
            els.debugBundleAcknowledgeHint.classList.remove('text-red-600', 'dark:text-red-300');
            els.debugBundleAcknowledgeHint.classList.add('text-amber-700', 'dark:text-amber-300');
        }
    }
    if (!state.portalUi.debugBundleGuardrailSeen) {
        state.portalUi.debugBundleGuardrailSeen = true;
        void emitPortalEvent('debug_bundle_guardrail_seen', {
            surface: 'reconstruction_runtime',
            field: 'emit_scene_debug_bundle',
            metadata: { enabled: true },
            reasons: Array.isArray(summary.includes) ? summary.includes.slice(0, 3) : []
        });
    }
}

function renderEffectiveConfigDrawer(payload = null, preview = null) {
    const currentPayload = payload || generatePayload();
    const effectivePreview = preview || _effectivePreviewSnapshot(currentPayload);
    if (els.requestedConfigJson) {
        els.requestedConfigJson.textContent = JSON.stringify(currentPayload.args || {}, null, 2);
    }
    if (els.effectiveConfigJson) {
        els.effectiveConfigJson.textContent = JSON.stringify(effectivePreview.normalized_args || currentPayload.args || {}, null, 2);
    }
    if (els.inactiveConfigJson) {
        els.inactiveConfigJson.textContent = JSON.stringify(effectivePreview.inactive_fields || [], null, 2);
    }
    if (els.effectiveEstimateLabel) {
        els.effectiveEstimateLabel.textContent = String(
            effectivePreview.estimate_summary?.summary_label
            || _buildLocalEstimateSummary(currentPayload.args || {}).summary_label
            || 'No preview estimate yet.'
        );
    }
    if (els.effectiveReadinessSummary) {
        els.effectiveReadinessSummary.textContent = _effectiveReadinessSummary(effectivePreview.readiness);
    }
    if (els.effectiveArgvPreview) {
        els.effectiveArgvPreview.textContent = String(effectivePreview.argv_preview || els.cliPreview?.textContent || '');
    }
    if (els.effectiveConfigMeta) {
        const inactiveCount = Array.isArray(effectivePreview.inactive_fields) ? effectivePreview.inactive_fields.length : 0;
        els.effectiveConfigMeta.textContent = effectivePreview.status === 'ready'
            ? `Preview-backed normalization is live. ${inactiveCount} inactive preserved field${inactiveCount === 1 ? '' : 's'} are tracked for the next run.`
            : 'Preview-backed normalization is unavailable, so this drawer is showing the local requested configuration and fallback posture.';
    }
}

function groupingFieldStatusText() {
    const field = _metadataField('grouping_mode');
    return String(field?.helper_text || RECON_RUNTIME_DEFAULT_STATUS.grouping_mode);
}

function iterationFieldStatusText() {
    const field = _metadataField('reconstruction_iterations');
    const recommended = field?.recommended && typeof field.recommended === 'object'
        ? field.recommended
        : null;
    if (recommended) {
        return `Recommended: fast ${recommended.fast}, balanced ${recommended.balanced}, high quality ${recommended.high_quality}.`;
    }
    return RECON_RUNTIME_DEFAULT_STATUS.reconstruction_iterations;
}

function reconstructionTierStatusText() {
    const field = _metadataField('reconstruction_tier');
    return String(field?.helper_text || RECON_RUNTIME_DEFAULT_STATUS.reconstruction_tier);
}

function rawIngestModeStatusText() {
    const field = _metadataField('raw_ingest_mode');
    return String(field?.helper_text || RECON_RUNTIME_DEFAULT_STATUS.raw_ingest_mode);
}

function maxWorkersStatusText() {
    const field = _metadataField('max_workers');
    const softMax = Number.parseInt(String(field?.soft_max || ''), 10);
    if (Number.isFinite(softMax) && softMax > 0) {
        return `Auto lets the backend choose a safe CPU worker cap. Recommended manual ceiling: ${softMax}.`;
    }
    return RECON_RUNTIME_DEFAULT_STATUS.max_workers;
}

function maxGpuWorkersStatusText() {
    const field = _metadataField('max_gpu_workers');
    const softMax = Number.parseInt(String(field?.soft_max || ''), 10);
    if (Number.isFinite(softMax) && softMax > 0) {
        return `Auto keeps GPU parallelism conservative. Recommended manual ceiling: ${softMax}.`;
    }
    return RECON_RUNTIME_DEFAULT_STATUS.max_gpu_workers;
}

function logLevelStatusText() {
    const field = _metadataField('log_level');
    return String(field?.helper_text || RECON_RUNTIME_DEFAULT_STATUS.log_level);
}

function buildCanonicalLuxDepthArgs(config) {
    const preset = _textOrFallback(
        els.presetSelect ? els.presetSelect.value : config.preset,
        config.preset || 'premium'
    ) || 'premium';
    const qualityTier = _resolveQualityTier(
        _textOrFallback(els.qualityTier ? els.qualityTier.value : config.qualityTier, config.qualityTier || 'standard')
    );
    const depthBackend = _resolveDepthBackend(
        _textOrFallback(els.depthBackend ? els.depthBackend.value : config.depthBackend, config.depthBackend || 'da3')
    );
    const depthDevice = _textOrFallback(
        els.depthDevice ? els.depthDevice.value : config.depthDevice,
        config.depthDevice || ''
    );
    const segmentationEnable = els.segmentation.enable
        ? Boolean(els.segmentation.enable.checked)
        : parseBoolLike(config.segmentation?.enable, false);
    const segmentationBackend = _resolveSegmentationBackend(
        _textOrFallback(
            els.segmentation.backend ? els.segmentation.backend.value : config.segmentation?.backend,
            config.segmentation?.backend || 'stub'
        )
    );
    const sam2ModelSize = _resolveSam2ModelSize(
        _textOrFallback(
            els.segmentation.sam2ModelSize ? els.segmentation.sam2ModelSize.value : config.segmentation?.sam2ModelSize,
            config.segmentation?.sam2ModelSize || 'base'
        )
    );
    const sam2CheckpointPath = _textOrFallback(
        els.segmentation.sam2CheckpointPath ? els.segmentation.sam2CheckpointPath.value : config.segmentation?.sam2CheckpointPath,
        config.segmentation?.sam2CheckpointPath || ''
    );
    const strictSegmentation = els.segmentation.strict
        ? Boolean(els.segmentation.strict.checked)
        : parseBoolLike(config.segmentation?.strict, false);
    const v2Preset = _textOrFallback(els.v2Preset ? els.v2Preset.value : config.v2Preset, config.v2Preset || 'default');

    const materialsV3 = els.flags.materials
        ? Boolean(els.flags.materials.checked)
        : parseBoolLike(config.flags?.materials, false);
    const pbr = els.flags.pbr
        ? Boolean(els.flags.pbr.checked)
        : parseBoolLike(config.flags?.pbr, false);
    const cacheDepth = els.flags.cache
        ? Boolean(els.flags.cache.checked)
        : parseBoolLike(config.flags?.cache, false);
    const enableV2 = els.flags.enableV2
        ? Boolean(els.flags.enableV2.checked)
        : parseBoolLike(config.flags?.enableV2, false);
    const saveFloatDepth = els.flags.saveFloatDepth
        ? Boolean(els.flags.saveFloatDepth.checked)
        : parseBoolLike(config.flags?.saveFloatDepth, false);
    const forceDepth = els.flags.forceDepth
        ? Boolean(els.flags.forceDepth.checked)
        : parseBoolLike(config.flags?.forceDepth, false);
    const strictInputs = els.flags.strictInputs
        ? Boolean(els.flags.strictInputs.checked)
        : parseBoolLike(config.flags?.strictInputs, false);
    const verifyImages = els.flags.verifyImages
        ? Boolean(els.flags.verifyImages.checked)
        : parseBoolLike(config.flags?.verifyImages, false);
    const allowSemanticFallback = els.flags.allowSemanticFallback
        ? Boolean(els.flags.allowSemanticFallback.checked)
        : parseBoolLike(config.flags?.allowSemanticFallback, false);
    const verbose = els.flags.verbose
        ? Boolean(els.flags.verbose.checked)
        : parseBoolLike(config.flags?.verbose, false);
    const quiet = els.flags.quiet
        ? Boolean(els.flags.quiet.checked)
        : parseBoolLike(config.flags?.quiet, false);

    const emitMaster16 = els.emits.master16
        ? Boolean(els.emits.master16.checked)
        : parseBoolLike(config.emits?.master16, true);
    const emitUpscaled16 = els.emits.upscaled16
        ? Boolean(els.emits.upscaled16.checked)
        : parseBoolLike(config.emits?.upscaled16, true);
    const emitMarketing = els.emits.marketing
        ? Boolean(els.emits.marketing.checked)
        : parseBoolLike(config.emits?.marketing, false);
    const emitReport = els.emits.report
        ? Boolean(els.emits.report.checked)
        : parseBoolLike(config.emits?.report, true);
    const emitRunCard = els.emits.runCard
        ? Boolean(els.emits.runCard.checked)
        : parseBoolLike(config.emits?.runCard, true);

    const nonCommercialOk = els.licenses.nonCommercialOk
        ? Boolean(els.licenses.nonCommercialOk.checked)
        : parseBoolLike(config.licenses?.nonCommercialOk, false);
    const acceptApple = els.licenses.acceptApple
        ? Boolean(els.licenses.acceptApple.checked)
        : parseBoolLike(config.licenses?.acceptApple, false);
    const acceptResearchTools = els.licenses.acceptResearchTools
        ? Boolean(els.licenses.acceptResearchTools.checked)
        : parseBoolLike(config.licenses?.acceptResearchTools, false);

    const enableReconstruction = els.reconstruction.enable
        ? Boolean(els.reconstruction.enable.checked)
        : parseBoolLike(config.reconstruction?.enable, false);
    const groupingMode = _resolveGroupingMode(
        _textOrFallback(
            els.reconstruction.groupingMode ? els.reconstruction.groupingMode.value : config.reconstruction?.groupingMode,
            config.reconstruction?.groupingMode || 'single'
        )
    );
    const camerasSidecarPath = _textOrFallback(
        els.reconstruction.camerasSidecarPath ? els.reconstruction.camerasSidecarPath.value : config.reconstruction?.camerasSidecarPath,
        config.reconstruction?.camerasSidecarPath || ''
    );
    const reconstructionIterations = _parsePositiveIntOrNull(
        els.reconstruction.iterations ? els.reconstruction.iterations.value : config.reconstruction?.iterations
    );
    const reconstructionTier = _textOrFallback(
        els.reconstruction.tier ? els.reconstruction.tier.value : config.reconstruction?.tier,
        config.reconstruction?.tier || 'apex_research'
    );
    const emitSceneDebugBundle = els.reconstruction.emitSceneDebugBundle
        ? Boolean(els.reconstruction.emitSceneDebugBundle.checked)
        : parseBoolLike(config.reconstruction?.emitSceneDebugBundle, false);

    const rawIngestMode = _textOrFallback(
        els.raw.ingestMode ? els.raw.ingestMode.value : config.raw?.ingestMode,
        config.raw?.ingestMode || 'auto'
    );
    const rawWbMode = _textOrFallback(
        els.raw.wbMode ? els.raw.wbMode.value : config.raw?.wbMode,
        config.raw?.wbMode || 'camera'
    );
    const rawDemosaic = _textOrFallback(
        els.raw.demosaic ? els.raw.demosaic.value : config.raw?.demosaic,
        config.raw?.demosaic || 'AHD'
    );

    const maxWorkersMode = _normalizeWorkerMode(
        els.runtime.maxWorkersMode ? els.runtime.maxWorkersMode.value : config.runtime?.maxWorkersMode
    );
    const maxWorkers = maxWorkersMode === 'fixed'
        ? _parsePositiveIntOrNull(
            els.runtime.maxWorkers ? els.runtime.maxWorkers.value : config.runtime?.maxWorkers
        )
        : null;
    const maxGpuWorkersMode = _normalizeWorkerMode(
        els.runtime.maxGpuWorkersMode ? els.runtime.maxGpuWorkersMode.value : config.runtime?.maxGpuWorkersMode
    );
    const maxGpuWorkers = maxGpuWorkersMode === 'fixed'
        ? _parsePositiveIntOrNull(
            els.runtime.maxGpuWorkers ? els.runtime.maxGpuWorkers.value : config.runtime?.maxGpuWorkers
        )
        : null;
    const logLevel = _textOrFallback(
        els.runtime.logLevel ? els.runtime.logLevel.value : config.runtime?.logLevel,
        config.runtime?.logLevel || ''
    );

    const args = {
        preset,
        quality_tier: qualityTier,
        depth_backend: depthBackend,
        enable_segmentation: segmentationEnable,
        segmentation_backend: segmentationBackend,
        sam2_model_size: sam2ModelSize,
        strict_segmentation: strictSegmentation,
        materials_v3: materialsV3,
        pbr,
        save_float_depth: saveFloatDepth,
        cache_depth: cacheDepth,
        enable_v2: enableV2,
        v2_preset: v2Preset,
        emit_master16: emitMaster16,
        emit_upscaled16: emitUpscaled16,
        emit_marketing: emitMarketing,
        emit_report: emitReport,
        emit_run_card: emitRunCard,
        non_commercial_ok: nonCommercialOk,
        accept_apple_depth_pro_research_license: acceptApple,
        accept_research_tools_license: acceptResearchTools,
        enable_reconstruction: enableReconstruction,
        grouping_mode: groupingMode,
        reconstruction_tier: reconstructionTier,
        emit_scene_debug_bundle: emitSceneDebugBundle,
        force_depth: forceDepth,
        strict_inputs: strictInputs,
        verify_images: verifyImages,
        allow_semantic_fallback: allowSemanticFallback,
        raw_ingest_mode: rawIngestMode,
        raw_wb_mode: rawWbMode,
        raw_demosaic: rawDemosaic,
        verbose,
        quiet
    };
    if (depthDevice) args.depth_device = depthDevice;
    if (sam2CheckpointPath) args.sam2_checkpoint_path = sam2CheckpointPath;
    if (camerasSidecarPath) args.cameras_sidecar_path = camerasSidecarPath;
    if (reconstructionIterations !== null) args.reconstruction_iterations = reconstructionIterations;
    if (maxWorkers !== null) args.max_workers = maxWorkers;
    if (maxGpuWorkers !== null) args.max_gpu_workers = maxGpuWorkers;
    if (logLevel) args.log_level = logLevel;
    return args;
}

function seedPresetFallbacks() {
    if (!els.presetSelect) return;
    const fallback = [...els.presetSelect.options]
        .map((opt) => ({
            name: String(opt.value || '').trim(),
            label: String(opt.textContent || opt.value || '').trim(),
            stability: 'unknown',
            description: '',
            is_research: _derivePresetResearchFlag({
                name: String(opt.value || '').trim(),
                label: String(opt.textContent || opt.value || '').trim()
            }),
            recommended_args: {},
            advanced_sections: []
        }))
        .filter((item) => item.name);
    if (fallback.length > 0) {
        state.presetsByPipeline['lux-depth-v3'] = fallback;
    }
}

function applyPresetRecommendedArgs(presetName) {
    const presets = Array.isArray(state.presetsByPipeline[state.pipeline]) ? state.presetsByPipeline[state.pipeline] : [];
    const preset = presets.find((item) => String(item?.name || '') === String(presetName || ''));
    const recommended = preset && preset.recommended_args && typeof preset.recommended_args === 'object'
        ? preset.recommended_args
        : null;
    if (!recommended) return;

    const c = state.config;
    c.preset = String(preset.name || presetName || c.preset || 'premium');
    if (Object.prototype.hasOwnProperty.call(recommended, 'quality_tier')) c.qualityTier = _resolveQualityTier(recommended.quality_tier);
    if (Object.prototype.hasOwnProperty.call(recommended, 'depth_backend')) c.depthBackend = _resolveDepthBackend(recommended.depth_backend);
    if (Object.prototype.hasOwnProperty.call(recommended, 'depth_device')) c.depthDevice = _textOrFallback(recommended.depth_device, c.depthDevice);

    c.segmentation = c.segmentation || {};
    if (Object.prototype.hasOwnProperty.call(recommended, 'enable_segmentation')) c.segmentation.enable = parseBoolLike(recommended.enable_segmentation, c.segmentation.enable);
    if (Object.prototype.hasOwnProperty.call(recommended, 'segmentation_backend')) c.segmentation.backend = _resolveSegmentationBackend(recommended.segmentation_backend);
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_model_size')) c.segmentation.sam2ModelSize = _resolveSam2ModelSize(recommended.sam2_model_size);
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_checkpoint_path')) c.segmentation.sam2CheckpointPath = _textOrFallback(recommended.sam2_checkpoint_path, '');
    if (Object.prototype.hasOwnProperty.call(recommended, 'strict_segmentation')) c.segmentation.strict = parseBoolLike(recommended.strict_segmentation, c.segmentation.strict);

    c.flags = c.flags || {};
    if (Object.prototype.hasOwnProperty.call(recommended, 'materials_v3')) c.flags.materials = parseBoolLike(recommended.materials_v3, c.flags.materials);
    if (Object.prototype.hasOwnProperty.call(recommended, 'pbr')) c.flags.pbr = parseBoolLike(recommended.pbr, c.flags.pbr);
    if (Object.prototype.hasOwnProperty.call(recommended, 'cache_depth')) c.flags.cache = parseBoolLike(recommended.cache_depth, c.flags.cache);
    if (Object.prototype.hasOwnProperty.call(recommended, 'enable_v2')) c.flags.enableV2 = parseBoolLike(recommended.enable_v2, c.flags.enableV2);
    if (Object.prototype.hasOwnProperty.call(recommended, 'save_float_depth')) c.flags.saveFloatDepth = parseBoolLike(recommended.save_float_depth, c.flags.saveFloatDepth);
    if (Object.prototype.hasOwnProperty.call(recommended, 'force_depth')) c.flags.forceDepth = parseBoolLike(recommended.force_depth, c.flags.forceDepth);
    if (Object.prototype.hasOwnProperty.call(recommended, 'strict_inputs')) c.flags.strictInputs = parseBoolLike(recommended.strict_inputs, c.flags.strictInputs);
    if (Object.prototype.hasOwnProperty.call(recommended, 'verify_images')) c.flags.verifyImages = parseBoolLike(recommended.verify_images, c.flags.verifyImages);
    if (Object.prototype.hasOwnProperty.call(recommended, 'allow_semantic_fallback')) c.flags.allowSemanticFallback = parseBoolLike(recommended.allow_semantic_fallback, c.flags.allowSemanticFallback);
    if (Object.prototype.hasOwnProperty.call(recommended, 'verbose')) c.flags.verbose = parseBoolLike(recommended.verbose, c.flags.verbose);
    if (Object.prototype.hasOwnProperty.call(recommended, 'quiet')) c.flags.quiet = parseBoolLike(recommended.quiet, c.flags.quiet);
    _normalizeVerboseQuietFlags(c.flags, false);

    if (Object.prototype.hasOwnProperty.call(recommended, 'v2_preset')) c.v2Preset = _textOrFallback(recommended.v2_preset, c.v2Preset || 'default');

    c.emits = c.emits || {};
    if (Object.prototype.hasOwnProperty.call(recommended, 'emit_master16')) c.emits.master16 = parseBoolLike(recommended.emit_master16, c.emits.master16);
    if (Object.prototype.hasOwnProperty.call(recommended, 'emit_upscaled16')) c.emits.upscaled16 = parseBoolLike(recommended.emit_upscaled16, c.emits.upscaled16);
    if (Object.prototype.hasOwnProperty.call(recommended, 'emit_marketing')) c.emits.marketing = parseBoolLike(recommended.emit_marketing, c.emits.marketing);
    if (Object.prototype.hasOwnProperty.call(recommended, 'emit_report')) c.emits.report = parseBoolLike(recommended.emit_report, c.emits.report);
    if (Object.prototype.hasOwnProperty.call(recommended, 'emit_run_card')) c.emits.runCard = parseBoolLike(recommended.emit_run_card, c.emits.runCard);

    c.licenses = c.licenses || {};
    if (Object.prototype.hasOwnProperty.call(recommended, 'non_commercial_ok')) c.licenses.nonCommercialOk = parseBoolLike(recommended.non_commercial_ok, c.licenses.nonCommercialOk);
    if (Object.prototype.hasOwnProperty.call(recommended, 'accept_apple_depth_pro_research_license')) c.licenses.acceptApple = parseBoolLike(recommended.accept_apple_depth_pro_research_license, c.licenses.acceptApple);
    if (Object.prototype.hasOwnProperty.call(recommended, 'accept_research_tools_license')) c.licenses.acceptResearchTools = parseBoolLike(recommended.accept_research_tools_license, c.licenses.acceptResearchTools);

    c.reconstruction = c.reconstruction || {};
    if (Object.prototype.hasOwnProperty.call(recommended, 'enable_reconstruction')) c.reconstruction.enable = parseBoolLike(recommended.enable_reconstruction, c.reconstruction.enable);
    if (Object.prototype.hasOwnProperty.call(recommended, 'grouping_mode')) c.reconstruction.groupingMode = _resolveGroupingMode(recommended.grouping_mode);
    if (Object.prototype.hasOwnProperty.call(recommended, 'reconstruction_tier')) c.reconstruction.tier = _textOrFallback(recommended.reconstruction_tier, c.reconstruction.tier);
    if (Object.prototype.hasOwnProperty.call(recommended, 'emit_scene_debug_bundle')) c.reconstruction.emitSceneDebugBundle = parseBoolLike(recommended.emit_scene_debug_bundle, c.reconstruction.emitSceneDebugBundle);

    c.runtime = c.runtime || {};
    if (Object.prototype.hasOwnProperty.call(recommended, 'max_workers')) {
        c.runtime.maxWorkers = _parsePositiveIntOrNull(recommended.max_workers) || '';
        c.runtime.maxWorkersMode = c.runtime.maxWorkers === '' ? 'auto' : 'fixed';
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'max_gpu_workers')) {
        c.runtime.maxGpuWorkers = _parsePositiveIntOrNull(recommended.max_gpu_workers) || '';
        c.runtime.maxGpuWorkersMode = c.runtime.maxGpuWorkers === '' ? 'auto' : 'fixed';
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'log_level')) {
        c.runtime.logLevel = _textOrFallback(recommended.log_level, c.runtime.logLevel || '');
    }
}

function applyPipelinePresetOptions(pipelineName) {
    if (!els.presetSelect || pipelineName !== 'lux-depth-v3') return;
    const presets = state.presetsByPipeline[pipelineName];
    if (!Array.isArray(presets) || presets.length === 0) return;

    const selectedBefore = state.config.preset;
    els.presetSelect.innerHTML = '';
    presets.forEach((preset) => {
        if (!preset || !preset.name) return;
        const option = document.createElement('option');
        option.value = String(preset.name);
        option.textContent = String(preset.label || preset.name);
        els.presetSelect.appendChild(option);
    });

    const names = presets.map((preset) => String(preset.name));
    if (names.includes(selectedBefore)) {
        els.presetSelect.value = selectedBefore;
    } else {
        const first = names[0] || 'premium';
        els.presetSelect.value = first;
        state.config.preset = first;
    }
}

function parsePresetsResponse(payload, pipelineName) {
    if (!payload || typeof payload !== 'object' || payload.success !== true || !payload.data) return [];
    if (Array.isArray(payload.data.presets)) return payload.data.presets;
    if (Array.isArray(payload.data.pipelines)) {
        const found = payload.data.pipelines.find((item) => item && item.pipeline === pipelineName);
        if (found && Array.isArray(found.presets)) return found.presets;
    }
    return [];
}

async function fetchPresetsForPipeline(pipelineName, silent = false) {
    if (!pipelineName || pipelineName !== 'lux-depth-v3' || !state.backendOk) return;
    try {
        const headers = _buildAuthHeaders({ 'Accept': 'application/json' });
        const res = await fetch(`${API_BASE}/v1/presets?pipeline=${encodeURIComponent(pipelineName)}`, { headers });
        if (!res.ok) throw new Error(`preset fetch failed (${res.status})`);
        const payload = await res.json();
        const presets = parsePresetsResponse(payload, pipelineName)
            .filter((preset) => preset && typeof preset.name === 'string')
            .map((preset) => ({
                name: String(preset.name),
                label: String(preset.label || preset.name),
                stability: String(preset.stability || 'unknown'),
                description: String(preset.description || ''),
                is_research: _derivePresetResearchFlag(preset),
                recommended_args: preset.recommended_args && typeof preset.recommended_args === 'object'
                    ? { ...preset.recommended_args }
                    : {},
                advanced_sections: Array.isArray(preset.advanced_sections)
                    ? preset.advanced_sections.map((item) => String(item))
                    : []
            }));
        if (presets.length > 0) {
            state.presetsByPipeline[pipelineName] = presets;
            applyPipelinePresetOptions(pipelineName);
            renderCLI();
            renderReviewSurfaces();
            scheduleConfigPreview(true);
        }
    } catch (err) {
        if (!silent) createToast('Failed to refresh presets from backend.', 'error');
    }
}

async function fetchReadiness(silent = false) {
    if (!state.backendOk || !_isBootstrapReady()) return;
    try {
        const headers = _buildAuthHeaders({ 'Accept': 'application/json' });
        const res = await fetch(`${API_BASE}/v1/readiness`, { headers });
        if (!res.ok) throw new Error(`readiness fetch failed (${res.status})`);
        const payload = await res.json();
        if (!payload || payload.success !== true || !payload.data) throw new Error('invalid readiness payload');
        state.readiness = {
            server: payload.data.server || {},
            pipelines: payload.data.pipelines || {}
        };
        renderReviewSurfaces();
        _syncBootstrapGuardedControls();
    } catch (err) {
        state.readiness = { server: {}, pipelines: {} };
        renderReviewSurfaces();
        _syncBootstrapGuardedControls();
        if (!silent) createToast('Failed to refresh execution readiness from backend.', 'error');
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
}

function updateUIFromState() {
    if (!els.pipelineSelect) return;
    els.pipelineSelect.value = state.pipeline;

    if (state.pipeline === 'lux-depth-v3') {
        els.fieldsLuxDepth.classList.remove('hidden');
        els.fieldsArchiveGate.classList.add('hidden');
        if (els.presetBuilderShell) els.presetBuilderShell.classList.remove('hidden');
        if (els.flagsShell) els.flagsShell.classList.remove('hidden');
    } else {
        els.fieldsLuxDepth.classList.add('hidden');
        els.fieldsArchiveGate.classList.remove('hidden');
        if (els.presetBuilderShell) els.presetBuilderShell.classList.add('hidden');
        if (els.flagsShell) els.flagsShell.classList.add('hidden');
    }
    refreshArchiveFieldVisibility();
    if (state.pipeline !== 'lux-depth-v3' && state.portalUi.buildStep < 2) {
        state.portalUi.buildStep = 2;
    } else {
        state.portalUi.buildStep = resolveBuildStep(state.portalUi.buildStep);
    }
    syncBuildStepUi();

    const c = state.config;
    c.depthBackend = _resolveDepthBackend(c.depthBackend);
    c.segmentation = c.segmentation || {};
    c.segmentation.enable = parseBoolLike(c.segmentation.enable, false);
    c.segmentation.backend = _resolveSegmentationBackend(c.segmentation.backend);
    c.segmentation.sam2ModelSize = _resolveSam2ModelSize(c.segmentation.sam2ModelSize);
    c.segmentation.sam2CheckpointPath = _textOrFallback(c.segmentation.sam2CheckpointPath, '');
    c.segmentation.strict = parseBoolLike(c.segmentation.strict, false);
    c.flags = c.flags || {};
    c.flags.materials = parseBoolLike(c.flags.materials, false);
    c.flags.pbr = parseBoolLike(c.flags.pbr, false);
    c.flags.cache = parseBoolLike(c.flags.cache, false);
    c.flags.overwrite = parseBoolLike(c.flags.overwrite, false);
    c.flags.enableV2 = parseBoolLike(c.flags.enableV2, false);
    c.flags.saveFloatDepth = parseBoolLike(c.flags.saveFloatDepth, false);
    c.flags.forceDepth = parseBoolLike(c.flags.forceDepth, false);
    c.flags.strictInputs = parseBoolLike(c.flags.strictInputs, false);
    c.flags.verifyImages = parseBoolLike(c.flags.verifyImages, false);
    c.flags.allowSemanticFallback = parseBoolLike(c.flags.allowSemanticFallback, false);
    c.flags.verbose = parseBoolLike(c.flags.verbose, false);
    c.flags.quiet = parseBoolLike(c.flags.quiet, false);
    _normalizeVerboseQuietFlags(c.flags, false);
    c.emits = c.emits || {};
    c.emits.master16 = parseBoolLike(c.emits.master16, true);
    c.emits.upscaled16 = parseBoolLike(c.emits.upscaled16, true);
    c.emits.marketing = parseBoolLike(c.emits.marketing, false);
    c.emits.report = parseBoolLike(c.emits.report, true);
    c.emits.runCard = parseBoolLike(c.emits.runCard, true);
    c.gate = c.gate || {};
    c.gate.archiveIndex = _textOrFallback(c.gate.archiveIndex, '');
    c.gate.manifestJsonl = _textOrFallback(c.gate.manifestJsonl, '');
    c.licenses = c.licenses || {};
    c.licenses.nonCommercialOk = parseBoolLike(c.licenses.nonCommercialOk, false);
    c.licenses.acceptApple = parseBoolLike(c.licenses.acceptApple, false);
    c.licenses.acceptResearchTools = parseBoolLike(c.licenses.acceptResearchTools, false);
    c.reconstruction = c.reconstruction || {};
    c.reconstruction.enable = parseBoolLike(c.reconstruction.enable, false);
    c.reconstruction.groupingMode = _resolveGroupingMode(c.reconstruction.groupingMode);
    c.reconstruction.camerasSidecarPath = _textOrFallback(c.reconstruction.camerasSidecarPath, '');
    c.reconstruction.iterations = _parsePositiveIntOrNull(c.reconstruction.iterations) || 1000;
    c.reconstruction.tier = _textOrFallback(c.reconstruction.tier, 'apex_research');
    c.reconstruction.emitSceneDebugBundle = parseBoolLike(c.reconstruction.emitSceneDebugBundle, false);
    c.raw = c.raw || {};
    c.raw.ingestMode = _textOrFallback(c.raw.ingestMode, 'auto');
    c.raw.wbMode = _textOrFallback(c.raw.wbMode, 'camera');
    c.raw.demosaic = _textOrFallback(c.raw.demosaic, 'AHD');
    c.runtime = c.runtime || {};
    c.runtime.maxWorkersMode = _normalizeWorkerMode(c.runtime.maxWorkersMode || (c.runtime.maxWorkers ? 'fixed' : 'auto'));
    c.runtime.maxWorkers = _parsePositiveIntOrNull(c.runtime.maxWorkers) || '';
    c.runtime.maxGpuWorkersMode = _normalizeWorkerMode(c.runtime.maxGpuWorkersMode || (c.runtime.maxGpuWorkers ? 'fixed' : 'auto'));
    c.runtime.maxGpuWorkers = _parsePositiveIntOrNull(c.runtime.maxGpuWorkers) || '';
    c.runtime.logLevel = _textOrFallback(c.runtime.logLevel, '');
    applyLuxMetadataToControls();
    applyPipelinePresetOptions(state.pipeline);
    if (els.presetSelect) els.presetSelect.value = c.preset;
    if (els.inputDir) els.inputDir.value = c.inputDir;
    if (els.outputDir) els.outputDir.value = c.outputDir;
    if (els.archiveIndexPath) els.archiveIndexPath.value = c.gate.archiveIndex;
    if (els.rightsManifestPath) els.rightsManifestPath.value = c.gate.manifestJsonl;
    if (els.qualityTier) els.qualityTier.value = c.qualityTier;
    if (els.depthBackend) els.depthBackend.value = c.depthBackend;
    if (els.depthDevice) els.depthDevice.value = c.depthDevice;
    if (els.segmentation.backend) els.segmentation.backend.value = c.segmentation.backend;
    if (els.segmentation.sam2ModelSize) els.segmentation.sam2ModelSize.value = c.segmentation.sam2ModelSize;
    if (els.segmentation.sam2CheckpointPath) els.segmentation.sam2CheckpointPath.value = c.segmentation.sam2CheckpointPath;
    if (els.v2Preset) {
        els.v2Preset.value = c.v2Preset;
        els.v2Preset.disabled = !c.flags.enableV2;
    }
    if (els.reconstruction.groupingMode) els.reconstruction.groupingMode.value = c.reconstruction.groupingMode;
    if (els.reconstruction.camerasSidecarPath) els.reconstruction.camerasSidecarPath.value = c.reconstruction.camerasSidecarPath;
    if (els.reconstruction.iterations) els.reconstruction.iterations.value = String(c.reconstruction.iterations);
    if (els.reconstruction.tier) els.reconstruction.tier.value = c.reconstruction.tier;
    if (els.raw.ingestMode) els.raw.ingestMode.value = c.raw.ingestMode;
    if (els.raw.wbMode) els.raw.wbMode.value = c.raw.wbMode;
    if (els.raw.demosaic) els.raw.demosaic.value = c.raw.demosaic;
    if (els.runtime.maxWorkersMode) els.runtime.maxWorkersMode.value = c.runtime.maxWorkersMode;
    if (els.runtime.maxWorkers) els.runtime.maxWorkers.value = c.runtime.maxWorkers === '' ? '' : String(c.runtime.maxWorkers);
    if (els.runtime.maxGpuWorkersMode) els.runtime.maxGpuWorkersMode.value = c.runtime.maxGpuWorkersMode;
    if (els.runtime.maxGpuWorkers) {
        els.runtime.maxGpuWorkers.value = c.runtime.maxGpuWorkers === '' ? '' : String(c.runtime.maxGpuWorkers);
    }
    if (els.runtime.logLevel) els.runtime.logLevel.value = c.runtime.logLevel;

    const safeSyncCheck = (el, val) => {
        if (el) {
            el.checked = val;
            if (el.hasAttribute('role') && el.getAttribute('role') === 'switch') {
                el.setAttribute('aria-checked', val);
            }
        }
    };

    safeSyncCheck(els.flags.materials, c.flags.materials);
    safeSyncCheck(els.flags.pbr, c.flags.pbr);
    safeSyncCheck(els.flags.cache, c.flags.cache);
    safeSyncCheck(els.flags.overwrite, c.flags.overwrite);
    safeSyncCheck(els.flags.enableV2, c.flags.enableV2);
    safeSyncCheck(els.flags.saveFloatDepth, c.flags.saveFloatDepth);
    safeSyncCheck(els.flags.forceDepth, c.flags.forceDepth);
    safeSyncCheck(els.flags.strictInputs, c.flags.strictInputs);
    safeSyncCheck(els.flags.verifyImages, c.flags.verifyImages);
    safeSyncCheck(els.flags.allowSemanticFallback, c.flags.allowSemanticFallback);
    safeSyncCheck(els.flags.verbose, c.flags.verbose);
    safeSyncCheck(els.flags.quiet, c.flags.quiet);
    safeSyncCheck(els.segmentation.enable, c.segmentation.enable);
    safeSyncCheck(els.segmentation.strict, c.segmentation.strict);

    safeSyncCheck(els.emits.master16, c.emits.master16);
    safeSyncCheck(els.emits.upscaled16, c.emits.upscaled16);
    safeSyncCheck(els.emits.marketing, c.emits.marketing);
    safeSyncCheck(els.emits.report, c.emits.report);
    safeSyncCheck(els.emits.runCard, c.emits.runCard);

    safeSyncCheck(els.licenses.nonCommercialOk, c.licenses.nonCommercialOk);
    safeSyncCheck(els.licenses.acceptApple, c.licenses.acceptApple);
    safeSyncCheck(els.licenses.acceptResearchTools, c.licenses.acceptResearchTools);
    safeSyncCheck(els.reconstruction.enable, c.reconstruction.enable);
    safeSyncCheck(els.reconstruction.emitSceneDebugBundle, c.reconstruction.emitSceneDebugBundle);
    state.portalUi.debugBundleAcknowledged = c.reconstruction.emitSceneDebugBundle
        ? Boolean(state.portalUi.debugBundleAcknowledged)
        : false;
    if (els.debugBundleAcknowledge) {
        els.debugBundleAcknowledge.checked = Boolean(state.portalUi.debugBundleAcknowledged);
    }
    syncSegmentationControlState(c);
    syncRuntimeWorkerModeControls();
    renderFieldPreviewStatuses();

    renderCLI();
    scheduleConfigPreview(true);
}

function reconcileBuildSurfaceFromDom() {
    const nextPipeline = _textOrFallback(els.pipelineSelect ? els.pipelineSelect.value : state.pipeline, state.pipeline);
    const nextInputDir = _textOrFallback(els.inputDir ? els.inputDir.value : state.config.inputDir, state.config.inputDir);
    const nextOutputDir = _textOrFallback(els.outputDir ? els.outputDir.value : state.config.outputDir, state.config.outputDir);
    const nextArchiveIndex = _textOrFallback(
        els.archiveIndexPath ? els.archiveIndexPath.value : state.config?.gate?.archiveIndex,
        state.config?.gate?.archiveIndex || ''
    );
    const nextManifestJsonl = _textOrFallback(
        els.rightsManifestPath ? els.rightsManifestPath.value : state.config?.gate?.manifestJsonl,
        state.config?.gate?.manifestJsonl || ''
    );

    let changed = false;
    let pipelineChanged = false;

    if (nextPipeline && nextPipeline !== state.pipeline) {
        state.pipeline = nextPipeline;
        changed = true;
        pipelineChanged = true;
    }
    if (nextInputDir !== state.config.inputDir) {
        state.config.inputDir = nextInputDir;
        changed = true;
    }
    if (nextOutputDir !== state.config.outputDir) {
        state.config.outputDir = nextOutputDir;
        changed = true;
    }
    state.config.gate = state.config.gate || {};
    if (nextArchiveIndex !== state.config.gate.archiveIndex) {
        state.config.gate.archiveIndex = nextArchiveIndex;
        changed = true;
    }
    if (nextManifestJsonl !== state.config.gate.manifestJsonl) {
        state.config.gate.manifestJsonl = nextManifestJsonl;
        changed = true;
    }

    if (!changed) return false;

    if (pipelineChanged) {
        updateUIFromState();
        void fetchPresetsForPipeline(state.pipeline, true);
        void fetchReadiness(true);
        void fetchConfigMetadata(state.pipeline, true);
        return true;
    }

    renderCLI();
    renderReviewSurfaces();
    _syncBootstrapGuardedControls();
    scheduleConfigPreview(true);
    return true;
}

function generatePayload() {
    const p = state.pipeline;
    const c = state.config;
    const inputDirValue = _textOrFallback(els.inputDir ? els.inputDir.value : c.inputDir, c.inputDir);
    const outputDirValue = _textOrFallback(els.outputDir ? els.outputDir.value : c.outputDir, c.outputDir);

    let args = {
        input_dir: inputDirValue,
        output_dir: outputDirValue
    };

    if (p === 'lux-depth-v3') {
        args = {
            ...args,
            overwrite: els.flags.overwrite
                ? Boolean(els.flags.overwrite.checked)
                : parseBoolLike(c.flags.overwrite, false),
            ...buildCanonicalLuxDepthArgs(c)
        };
    } else {
        const archiveCommand = canonicalArchiveCommand(p);
        args = {
            ...args,
            archive_command: archiveCommand
        };
        if (p === 'archive-gate-a') {
            args.archive_root = inputDirValue;
            args.archive_index = _textOrFallback(
                els.archiveIndexPath ? els.archiveIndexPath.value : c.gate.archiveIndex,
                c.gate.archiveIndex
            );
        } else {
            args.manifest_jsonl = _textOrFallback(
                els.rightsManifestPath ? els.rightsManifestPath.value : c.gate.manifestJsonl,
                c.gate.manifestJsonl
            );
        }
    }

    return { pipeline: p, args };
}

function renderPreRunDiagnostics(payload) {
    if (!payload) return;

    const warnings = [];
    const expectedOutputs = [];
    let healthState = 'good';
    let healthLabel = 'good';

    if (payload.pipeline === 'lux-depth-v3') {
        const args = payload.args || {};
        const preview = _currentPreviewForPayload(payload);
        const reconstructionEnabled = parseBoolLike(args.enable_reconstruction, false);
        const sidecarPath = String(args.cameras_sidecar_path || '').trim();
        const groupingMode = String(args.grouping_mode || 'single').trim().toLowerCase();
        const rawIngestMode = String(args.raw_ingest_mode || '').trim().toLowerCase();
        const strictInputs = parseBoolLike(args.strict_inputs, false);
        const segmentationEnabled = parseBoolLike(args.enable_segmentation, false);

        expectedOutputs.push('Depth maps');
        if (parseBoolLike(args.pbr, false)) expectedOutputs.push('PBR maps');
        if (segmentationEnabled) expectedOutputs.push('Segmentation masks');
        if (parseBoolLike(args.enable_v2, false)) expectedOutputs.push('V2 enhanced outputs');
        if (reconstructionEnabled) expectedOutputs.push('Reconstruction report bundle');
        if (parseBoolLike(args.emit_run_card, false)) expectedOutputs.push('Run card JSON');
        if (parseBoolLike(args.emit_scene_debug_bundle, false)) expectedOutputs.push('Reconstruction debug bundle');

        if (preview && preview.status === 'ready') {
            const previewErrors = Array.isArray(preview.field_errors) ? preview.field_errors : [];
            const previewWarnings = Array.isArray(preview.field_warnings) ? preview.field_warnings : [];
            previewErrors.forEach((issue) => warnings.push(`BLOCKED: ${String(issue?.message || 'Preview validation error.')}`));
            previewWarnings.forEach((issue) => warnings.push(`WARNING: ${String(issue?.message || 'Preview warning.')}`));
            if (previewErrors.length > 0) {
                healthState = 'risk';
                healthLabel = 'preview blocked';
            } else if (previewWarnings.length > 0 && healthState === 'good') {
                healthState = 'warn';
                healthLabel = 'preview cautions';
            }
        } else if (state.backendOk && preview && preview.status === 'error') {
            const previewFailure = _previewFailureDetails(preview);
            warnings.push(previewFailure.luxBlockedMessage);
            healthState = 'risk';
            healthLabel = previewFailure.healthLabel;
        } else if (!state.backendOk) {
            warnings.push('WARNING: Backend preview is offline. Local fallback rendering is available, but dispatch remains blocked until connectivity is restored.');
            if (healthState === 'good') {
                healthState = 'warn';
                healthLabel = 'backend offline';
            }
        } else if (state.backendOk && (!preview || preview.status === 'loading')) {
            warnings.push('WARNING: Preview-backed validation is refreshing.');
            if (healthState === 'good') {
                healthState = 'warn';
                healthLabel = 'preview refreshing';
            }
        }

        if (!preview || preview.status !== 'ready') {
            if (reconstructionEnabled && !sidecarPath) {
                warnings.push('WARNING: Camera sidecar path is missing; reconstruction may fail.');
                healthState = 'risk';
                healthLabel = 'reconstruction risk';
            }
            if (reconstructionEnabled && groupingMode === 'single') {
                warnings.push('WARNING: Reconstruction enabled with grouping mode "single"; overlap may be weak.');
                if (healthState !== 'risk') {
                    healthState = 'warn';
                    healthLabel = 'weak overlap';
                }
            }
            if (rawIngestMode === 'force_rawpy') {
                warnings.push('WARNING: RAW ingest mode force_rawpy is experimental and may increase runtime.');
                if (healthState === 'good') {
                    healthState = 'warn';
                    healthLabel = 'raw ingest stress';
                }
            }
            if (strictInputs) {
                warnings.push('WARNING: strict_inputs is enabled; malformed inputs will fail fast.');
            }
            if (parseBoolLike(args.materials_v3, false) && !segmentationEnabled) {
                warnings.push('WARNING: Materials V3 is enabled while segmentation is off.');
                if (healthState === 'good') {
                    healthState = 'warn';
                    healthLabel = 'config mismatch';
                }
            }
        } else if (strictInputs) {
            warnings.push('WARNING: strict_inputs is enabled; malformed inputs will fail fast.');
        }
    } else {
        const args = payload.args || {};
        const preview = _currentPreviewForPayload(payload);
        const readinessIssues = currentPipelineReadinessIssues(payload);
        const dispatchStatus = currentPipelineDispatchStatus(payload);
        const archiveCommand = String(args.archive_command || canonicalArchiveCommand(payload.pipeline) || 'archive').trim();
        expectedOutputs.push(`${archiveCommand} outputs`);
        expectedOutputs.push('Pipeline execution logs');

        if (preview && preview.status === 'ready') {
            const previewErrors = Array.isArray(preview.field_errors) ? preview.field_errors : [];
            const previewWarnings = Array.isArray(preview.field_warnings) ? preview.field_warnings : [];
            previewErrors.forEach((issue) => warnings.push(`BLOCKED: ${String(issue?.message || 'Preview validation error.')}`));
            previewWarnings.forEach((issue) => warnings.push(`WARNING: ${String(issue?.message || 'Preview warning.')}`));
            if (previewErrors.length > 0) {
                healthState = 'risk';
                healthLabel = 'preview blocked';
            } else if (previewWarnings.length > 0 && healthState === 'good') {
                healthState = 'warn';
                healthLabel = 'preview cautions';
            }
        } else if (state.backendOk && preview && preview.status === 'error') {
            const previewFailure = _previewFailureDetails(preview);
            warnings.push(previewFailure.archiveWarningMessage);
            if (healthState === 'good') {
                healthState = 'warn';
                healthLabel = previewFailure.healthLabel;
            }
        } else if (!state.backendOk) {
            warnings.push('WARNING: Backend preview is offline. Local fallback rendering is available, but dispatch remains blocked until connectivity is restored.');
            if (healthState === 'good') {
                healthState = 'warn';
                healthLabel = 'backend offline';
            }
        } else if (state.backendOk && (!preview || preview.status === 'loading')) {
            warnings.push('WARNING: Preview-backed validation is refreshing.');
            if (healthState === 'good') {
                healthState = 'warn';
                healthLabel = 'preview refreshing';
            }
        }

        readinessIssues.forEach((issue) => {
            const severity = String(issue?.severity || '').trim().toLowerCase();
            warnings.push(
                `${severity === 'blocked' ? 'BLOCKED' : 'WARNING'}: ${String(issue?.message || 'Pipeline readiness reported an operator-facing prerequisite.')}`
            );
            if (severity === 'blocked') {
                healthState = 'risk';
                healthLabel = 'dispatch blocked';
            } else if (healthState === 'good') {
                healthState = 'warn';
                healthLabel = 'prerequisites still required';
            }
        });

        if (healthState === 'good') {
            if (dispatchStatus === 'blocked') {
                healthState = 'risk';
                healthLabel = 'dispatch blocked';
            } else if (dispatchStatus === 'degraded') {
                healthState = 'warn';
                healthLabel = 'prerequisites still required';
            } else {
                healthLabel = 'archive pipeline ready';
            }
        }
    }

    if (els.preRunWarnings) {
        els.preRunWarnings.innerHTML = '';
        warnings.forEach((warning) => {
            const li = document.createElement('li');
            li.className = 'leading-relaxed';
            li.textContent = warning.startsWith('WARNING:') || warning.startsWith('BLOCKED:')
                ? warning
                : `WARNING: ${warning}`;
            els.preRunWarnings.appendChild(li);
        });
    }
    if (els.preRunWarningsEmpty) {
        els.preRunWarningsEmpty.style.display = warnings.length === 0 ? 'block' : 'none';
    }

    if (els.expectedOutputsList) {
        els.expectedOutputsList.innerHTML = '';
        expectedOutputs.forEach((item) => {
            const li = document.createElement('li');
            li.className = 'leading-relaxed';
            li.textContent = `- ${item}`;
            els.expectedOutputsList.appendChild(li);
        });
    }

    if (els.datasetHealthIndicator) {
        const healthClass = healthState === 'good' ? 'ready' : healthState === 'warn' ? 'running' : 'offline';
        els.datasetHealthIndicator.className = `status-dot ${healthClass}`;
    }
    if (els.datasetHealthText) {
        els.datasetHealthText.textContent = `Dataset health: ${healthLabel}`;
    }
    renderNextBestAction(payload, _currentPreviewForPayload(payload) || _effectivePreviewSnapshot(payload));

    state.lastDiagnostics = {
        warnings,
        expectedOutputs,
        healthState,
        healthLabel
    };
    renderReviewSurfaces(payload);
}

function renderCLI() {
    if (!els.cliPreview) return;
    const payload = generatePayload();
    const preview = _currentPreviewForPayload(payload);
    const q = (v) => `"${String(v).replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
    const onoff = (v) => (parseBoolLike(v, false) ? '"on"' : '"off"');

    const cliLines = [
        `${payload.pipeline}`,
        `  --input-dir ${q(payload.args.input_dir)}`,
        `  --output-dir ${q(payload.args.output_dir)}`
    ];

    if (payload.pipeline === 'lux-depth-v3') {
        cliLines.push(`  --preset ${q(payload.args.preset)}`);
        cliLines.push(`  --quality-tier ${q(payload.args.quality_tier)}`);
        cliLines.push(`  --depth-backend ${q(payload.args.depth_backend)}`);
        if (payload.args.depth_device) {
            cliLines.push(`  --depth-device ${q(payload.args.depth_device)}`);
        }
        cliLines.push(`  --materials-v3 ${onoff(payload.args.materials_v3)}`);
        cliLines.push(`  --enable-segmentation ${onoff(payload.args.enable_segmentation)}`);
        cliLines.push(`  --segmentation-backend ${q(payload.args.segmentation_backend)}`);
        if (payload.args.segmentation_backend === 'sam2') {
            cliLines.push(`  --sam2-model-size ${q(payload.args.sam2_model_size)}`);
        }
        if (payload.args.sam2_checkpoint_path) {
            cliLines.push(`  --sam2-checkpoint-path ${q(payload.args.sam2_checkpoint_path)}`);
        }
        if (parseBoolLike(payload.args.strict_segmentation, false)) {
            cliLines.push(`  --strict-segmentation`);
        }
        cliLines.push(`  --pbr ${onoff(payload.args.pbr)}`);
        cliLines.push(`  --save-float-depth ${onoff(payload.args.save_float_depth)}`);
        cliLines.push(`  --cache-depth ${onoff(payload.args.cache_depth)}`);
        cliLines.push(`  --emit-master16 ${onoff(payload.args.emit_master16)}`);
        cliLines.push(`  --emit-upscaled16 ${onoff(payload.args.emit_upscaled16)}`);
        cliLines.push(`  --emit-marketing ${onoff(payload.args.emit_marketing)}`);
        cliLines.push(`  --emit-report ${onoff(payload.args.emit_report)}`);
        cliLines.push(`  --emit-run-card ${onoff(payload.args.emit_run_card)}`);
        cliLines.push(`  --enable-v2 ${onoff(payload.args.enable_v2)}`);

        if (parseBoolLike(payload.args.enable_v2, false) && payload.args.v2_preset) {
            cliLines.push(`  --v2-preset ${q(payload.args.v2_preset)}`);
        }
        if (parseBoolLike(payload.args.non_commercial_ok, false)) cliLines.push(`  --non-commercial-ok "true"`);
        if (parseBoolLike(payload.args.accept_apple_depth_pro_research_license, false)) {
            cliLines.push(`  --accept-apple-depth-pro-research-license "true"`);
        }
        if (parseBoolLike(payload.args.accept_research_tools_license, false)) {
            cliLines.push(`  --accept-research-tools-license "true"`);
        }

        cliLines.push(`  --enable-reconstruction ${onoff(payload.args.enable_reconstruction)}`);
        cliLines.push(`  --grouping-mode ${q(payload.args.grouping_mode)}`);
        if (payload.args.cameras_sidecar_path) {
            cliLines.push(`  --cameras-sidecar-path ${q(payload.args.cameras_sidecar_path)}`);
        }
        if (payload.args.reconstruction_iterations) {
            cliLines.push(`  --reconstruction-iterations ${q(payload.args.reconstruction_iterations)}`);
        }
        if (payload.args.reconstruction_tier) {
            cliLines.push(`  --reconstruction-tier ${q(payload.args.reconstruction_tier)}`);
        }
        cliLines.push(`  --emit-scene-debug-bundle ${onoff(payload.args.emit_scene_debug_bundle)}`);

        if (parseBoolLike(payload.args.force_depth, false)) cliLines.push(`  --force-depth`);
        if (parseBoolLike(payload.args.strict_inputs, false)) cliLines.push(`  --strict-inputs`);
        if (payload.args.raw_ingest_mode) cliLines.push(`  --raw-ingest-mode ${q(payload.args.raw_ingest_mode)}`);
        if (payload.args.raw_wb_mode) cliLines.push(`  --raw-wb-mode ${q(payload.args.raw_wb_mode)}`);
        if (payload.args.raw_demosaic) cliLines.push(`  --raw-demosaic ${q(payload.args.raw_demosaic)}`);
        if (payload.args.max_workers) cliLines.push(`  --max-workers ${q(payload.args.max_workers)}`);
        if (payload.args.max_gpu_workers) cliLines.push(`  --max-gpu-workers ${q(payload.args.max_gpu_workers)}`);
        if (parseBoolLike(payload.args.verify_images, false)) cliLines.push(`  --verify-images`);
        if (parseBoolLike(payload.args.allow_semantic_fallback, false)) cliLines.push(`  --allow-semantic-fallback`);
        if (parseBoolLike(payload.args.verbose, false)) cliLines.push(`  --verbose`);
        if (parseBoolLike(payload.args.quiet, false)) cliLines.push(`  --quiet`);
        if (payload.args.log_level) cliLines.push(`  --log-level ${q(payload.args.log_level)}`);
        if (parseBoolLike(payload.args.overwrite, false)) cliLines.push(`  --overwrite`);
    } else {
        if (payload.args.archive_command) {
            cliLines.push(`  --archive-command ${q(payload.args.archive_command)}`);
        }
        if (payload.args.archive_root) {
            cliLines.push(`  --archive-root ${q(payload.args.archive_root)}`);
        }
        if (payload.args.archive_index) {
            cliLines.push(`  --archive-index ${q(payload.args.archive_index)}`);
        }
        if (payload.args.manifest_jsonl) {
            cliLines.push(`  --manifest-jsonl ${q(payload.args.manifest_jsonl)}`);
        }
    }

    const cli = cliLines.map((line, idx) =>
        idx === cliLines.length - 1 ? line : `${line} \\`
    );

    const previewErrors = preview && preview.status === 'ready' && Array.isArray(preview.field_errors)
        ? preview.field_errors
        : [];

    if (preview && preview.status === 'ready' && previewErrors.length === 0 && String(preview.argv_preview || '').trim()) {
        els.cliPreview.textContent = String(preview.argv_preview || '');
    } else if (preview && preview.status === 'ready' && previewErrors.length > 0) {
        const firstError = previewErrors[0];
        els.cliPreview.textContent = `# Preview blocked\n# ${String(firstError?.message || 'Resolve preview validation issues to view the effective argv.')}`;
    } else {
        els.cliPreview.textContent = cli.join('\n');
    }
    renderFieldPreviewStatuses(payload);
    renderPreRunDiagnostics(payload);
    _syncBootstrapGuardedControls();
    _syncOverviewBuildLoadingState(payload);
}

function bindInputs() {
    const trackedTelemetryField = (category, key) => {
        const normalizedCategory = String(category || '').trim();
        const normalizedKey = String(key || '').trim();
        const lookup = {
            ':qualityTier': 'quality_tier',
            ':depthBackend': 'depth_backend',
            'segmentation:backend': 'segmentation_backend',
            'segmentation:strict': 'strict_segmentation',
            'licenses:nonCommercialOk': 'non_commercial_ok',
            'licenses:acceptApple': 'accept_apple_depth_pro_research_license',
            'licenses:acceptResearchTools': 'accept_research_tools_license',
            'reconstruction:enable': 'enable_reconstruction',
            'reconstruction:groupingMode': 'grouping_mode',
            'reconstruction:iterations': 'reconstruction_iterations',
            'reconstruction:tier': 'reconstruction_tier',
            'reconstruction:emitSceneDebugBundle': 'emit_scene_debug_bundle',
            'raw:ingestMode': 'raw_ingest_mode',
            'runtime:maxWorkersMode': 'max_workers_mode',
            'runtime:maxWorkers': 'max_workers',
            'runtime:maxGpuWorkersMode': 'max_gpu_workers_mode',
            'runtime:maxGpuWorkers': 'max_gpu_workers',
            'runtime:logLevel': 'log_level'
        };
        return lookup[`${normalizedCategory}:${normalizedKey}`] || null;
    };

    const telemetrySurfaceFor = (category) => {
        if (category === 'reconstruction' || category === 'raw' || category === 'runtime') {
            return 'reconstruction_runtime';
        }
        return 'dispatch';
    };

    const safeBindText = (el, category, key) => {
        if (!el) return;
        el.addEventListener('change', (e) => {
            if (category) state.config[category][key] = e.target.value;
            else if (key in state.config) state.config[key] = e.target.value;
            else state[key] = e.target.value;
            if (key === 'pipeline') {
                updateUIFromState();
                void fetchPresetsForPipeline(state.pipeline, true);
                void fetchReadiness(true);
                void fetchConfigMetadata(state.pipeline, true);
            }
            else {
                const field = trackedTelemetryField(category, key);
                if (field) {
                    void emitPortalEvent('field_commit', {
                        surface: telemetrySurfaceFor(category),
                        field,
                        metadata: { value_set: Boolean(String(e.target.value || '').trim()) }
                    });
                }
                renderCLI();
                scheduleConfigPreview();
            }
        });
    };
    const safeBindInput = (el, category, key) => {
        if (!el) return;
        el.addEventListener('input', (e) => {
            if (category) state.config[category][key] = e.target.value;
            else state.config[key] = e.target.value;
            renderCLI();
            if (trackedTelemetryField(category, key)) {
                scheduleConfigPreview();
            }
        });
        el.addEventListener('change', (e) => {
            const field = trackedTelemetryField(category, key);
            if (field) {
                void emitPortalEvent('field_commit', {
                    surface: telemetrySurfaceFor(category),
                    field,
                    metadata: { value_set: Boolean(String(e.target.value || '').trim()) }
                });
            }
            scheduleConfigPreview();
        });
    };
    const safeBindCheck = (el, category, key) => {
        if (!el) return;
        el.addEventListener('change', (e) => {
            state.config[category][key] = e.target.checked;
            if (el.hasAttribute('role') && el.getAttribute('role') === 'switch') {
                el.setAttribute('aria-checked', e.target.checked);
            }
            if (key === 'enableV2' && els.v2Preset) els.v2Preset.disabled = !e.target.checked;
            if (category === 'flags' && (key === 'verbose' || key === 'quiet') && e.target.checked) {
                const otherKey = key === 'verbose' ? 'quiet' : 'verbose';
                if (parseBoolLike(state.config.flags[otherKey], false)) {
                    state.config.flags[otherKey] = false;
                    const otherEl = els.flags[otherKey];
                    if (otherEl) {
                        otherEl.checked = false;
                        if (otherEl.hasAttribute('role') && otherEl.getAttribute('role') === 'switch') {
                            otherEl.setAttribute('aria-checked', false);
                        }
                    }
                    createToast(`verbose and quiet are mutually exclusive; disabled ${otherKey}.`, 'info');
                }
            }
            const field = trackedTelemetryField(category, key);
            if (field) {
                void emitPortalEvent('toggle_change', {
                    surface: telemetrySurfaceFor(category),
                    field,
                    metadata: { enabled: Boolean(e.target.checked) }
                });
            }
            if (category === 'reconstruction' && key === 'emitSceneDebugBundle' && !e.target.checked) {
                state.portalUi.debugBundleAcknowledged = false;
                state.portalUi.debugBundleGuardrailSeen = false;
                if (els.debugBundleAcknowledge) els.debugBundleAcknowledge.checked = false;
            }
            renderCLI();
            scheduleConfigPreview();
        });
    };

    safeBindText(els.pipelineSelect, null, 'pipeline');
    if (els.presetSelect) {
        els.presetSelect.addEventListener('change', (e) => {
            const nextPreset = String(e.target.value || '').trim();
            state.config.preset = nextPreset;
            applyPresetRecommendedArgs(nextPreset);
            updateUIFromState();
        });
    }
    safeBindInput(els.inputDir, null, 'inputDir');
    safeBindInput(els.outputDir, null, 'outputDir');
    safeBindInput(els.archiveIndexPath, 'gate', 'archiveIndex');
    safeBindInput(els.rightsManifestPath, 'gate', 'manifestJsonl');
    safeBindText(els.qualityTier, null, 'qualityTier');
    safeBindText(els.depthBackend, null, 'depthBackend');
    safeBindText(els.depthDevice, null, 'depthDevice');
    safeBindText(els.segmentation.backend, 'segmentation', 'backend');
    safeBindText(els.segmentation.sam2ModelSize, 'segmentation', 'sam2ModelSize');
    safeBindInput(els.segmentation.sam2CheckpointPath, 'segmentation', 'sam2CheckpointPath');
    safeBindInput(els.v2Preset, null, 'v2Preset');
    safeBindText(els.reconstruction.groupingMode, 'reconstruction', 'groupingMode');
    safeBindInput(els.reconstruction.camerasSidecarPath, 'reconstruction', 'camerasSidecarPath');
    safeBindInput(els.reconstruction.iterations, 'reconstruction', 'iterations');
    safeBindInput(els.reconstruction.tier, 'reconstruction', 'tier');
    safeBindText(els.raw.ingestMode, 'raw', 'ingestMode');
    safeBindText(els.raw.wbMode, 'raw', 'wbMode');
    safeBindText(els.raw.demosaic, 'raw', 'demosaic');
    safeBindInput(els.runtime.maxWorkers, 'runtime', 'maxWorkers');
    safeBindInput(els.runtime.maxGpuWorkers, 'runtime', 'maxGpuWorkers');
    safeBindText(els.runtime.logLevel, 'runtime', 'logLevel');

    safeBindCheck(els.flags.materials, 'flags', 'materials');
    safeBindCheck(els.flags.pbr, 'flags', 'pbr');
    safeBindCheck(els.flags.cache, 'flags', 'cache');
    safeBindCheck(els.flags.overwrite, 'flags', 'overwrite');
    safeBindCheck(els.flags.enableV2, 'flags', 'enableV2');
    safeBindCheck(els.flags.saveFloatDepth, 'flags', 'saveFloatDepth');
    safeBindCheck(els.flags.forceDepth, 'flags', 'forceDepth');
    safeBindCheck(els.flags.strictInputs, 'flags', 'strictInputs');
    safeBindCheck(els.flags.verifyImages, 'flags', 'verifyImages');
    safeBindCheck(els.flags.allowSemanticFallback, 'flags', 'allowSemanticFallback');
    safeBindCheck(els.flags.verbose, 'flags', 'verbose');
    safeBindCheck(els.flags.quiet, 'flags', 'quiet');
    safeBindCheck(els.segmentation.enable, 'segmentation', 'enable');
    safeBindCheck(els.segmentation.strict, 'segmentation', 'strict');
    safeBindCheck(els.reconstruction.enable, 'reconstruction', 'enable');
    safeBindCheck(els.reconstruction.emitSceneDebugBundle, 'reconstruction', 'emitSceneDebugBundle');

    safeBindCheck(els.emits.master16, 'emits', 'master16');
    safeBindCheck(els.emits.upscaled16, 'emits', 'upscaled16');
    safeBindCheck(els.emits.marketing, 'emits', 'marketing');
    safeBindCheck(els.emits.report, 'emits', 'report');
    safeBindCheck(els.emits.runCard, 'emits', 'runCard');

    safeBindCheck(els.licenses.nonCommercialOk, 'licenses', 'nonCommercialOk');
    safeBindCheck(els.licenses.acceptApple, 'licenses', 'acceptApple');
    safeBindCheck(els.licenses.acceptResearchTools, 'licenses', 'acceptResearchTools');

    if (els.segmentation.backend) {
        els.segmentation.backend.addEventListener('change', () => {
            syncSegmentationControlState(state.config);
            renderCLI();
            scheduleConfigPreview();
        });
    }

    if (els.runtime.maxWorkersMode) {
        els.runtime.maxWorkersMode.addEventListener('change', (e) => {
            state.config.runtime.maxWorkersMode = _normalizeWorkerMode(e.target.value);
            syncRuntimeWorkerModeControls();
            void emitPortalEvent('field_commit', {
                surface: 'reconstruction_runtime',
                field: 'max_workers_mode',
                metadata: { mode: state.config.runtime.maxWorkersMode }
            });
            renderCLI();
            scheduleConfigPreview();
        });
    }

    if (els.runtime.maxGpuWorkersMode) {
        els.runtime.maxGpuWorkersMode.addEventListener('change', (e) => {
            state.config.runtime.maxGpuWorkersMode = _normalizeWorkerMode(e.target.value);
            syncRuntimeWorkerModeControls();
            void emitPortalEvent('field_commit', {
                surface: 'reconstruction_runtime',
                field: 'max_gpu_workers_mode',
                metadata: { mode: state.config.runtime.maxGpuWorkersMode }
            });
            renderCLI();
            scheduleConfigPreview();
        });
    }

    if (els.debugBundleAcknowledge) {
        els.debugBundleAcknowledge.addEventListener('change', (e) => {
            state.portalUi.debugBundleAcknowledged = Boolean(e.target.checked);
            renderCLI();
            _syncBootstrapGuardedControls();
            void emitPortalEvent('field_commit', {
                surface: 'reconstruction_runtime',
                field: 'debug_bundle_acknowledged',
                metadata: { enabled: Boolean(e.target.checked) }
            });
        });
    }

    if (els.apiKeyInput) {
        els.apiKeyInput.addEventListener('input', _persistApiKeyFromInputs);
        els.apiKeyInput.addEventListener('change', () => {
            _persistApiKeyFromInputs();
            resumeBlockedJobStreamsAfterAuthUpdate();
        });
    }
}

function getProfiles() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch { return {}; }
}

function refreshProfileDropdown() {
    if (!els.profileSelect) return;
    const profiles = getProfiles();
    els.profileSelect.innerHTML = '<option value="">Select Profile...</option>';
    Object.keys(profiles).sort().forEach((k) => {
        const opt = document.createElement('option');
        opt.value = k;
        opt.textContent = k;
        els.profileSelect.appendChild(opt);
    });
}

function renderJobQueue(includeReviewSurfaces = true) {
    if (!els.jobList) return;
    els.jobList.setAttribute('role', 'listbox');
    const queueLoading = _isJobsHydrationPending();
    if (els.queueShell) {
        els.queueShell.setAttribute('aria-busy', queueLoading ? 'true' : 'false');
    }
    if (els.queueSkeletonState) {
        els.queueSkeletonState.classList.toggle('hidden', !queueLoading);
        els.queueSkeletonState.setAttribute('aria-hidden', 'true');
    }
    els.jobList.classList.toggle('hidden', queueLoading);
    if (els.queueCount) els.queueCount.textContent = `${state.jobs.length} jobs`;
    if (queueLoading) {
        if (els.queueCount) els.queueCount.textContent = 'Syncing...';
        if (els.queueStatusSummary) els.queueStatusSummary.textContent = 'Recovering recent runs and live transport state.';
        if (els.emptyQueueState) els.emptyQueueState.style.display = 'none';
        els.jobList.innerHTML = '';
        if (includeReviewSurfaces) renderReviewSurfaces();
        return;
    }
    if (state.jobs.length === 0) {
        if (els.emptyQueueState) els.emptyQueueState.style.display = 'flex';
        if (els.queueStatusSummary) els.queueStatusSummary.textContent = 'Newest jobs stay pinned to the top.';
        els.jobList.innerHTML = '';
        if (includeReviewSurfaces) renderReviewSurfaces();
        return;
    }
    if (els.emptyQueueState) els.emptyQueueState.style.display = 'none';

    const fragment = document.createDocumentFragment();
    [...state.jobs].reverse().forEach((job) => {
        const li = document.createElement('li');
        li.dataset.jobId = job.id;
        li.dataset.ui = 'queue-row';
        const isSelected = state.selectedJobId === job.id;
        li.className = `rounded-2xl border p-4 cursor-pointer transition-all ${isSelected ? 'border-cyan-500 bg-cyan-50/80 dark:bg-cyan-900/20 shadow-md ring-1 ring-cyan-500/15' : 'border-slate-200 dark:border-slate-800 bg-slate-50/75 dark:bg-slate-900/45 hover:bg-white/90 dark:hover:bg-slate-800/80'}`;
        li.tabIndex = 0;
        li.setAttribute('role', 'option');
        li.setAttribute('aria-selected', isSelected ? 'true' : 'false');

        let badgeColor = 'bg-slate-100 text-slate-600 border-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:border-slate-700';
        if (job.state === 'running' || job.state === 'partial') badgeColor = 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-900/30 dark:text-amber-400 dark:border-amber-800';
        if (job.state === 'succeeded') badgeColor = 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-400 dark:border-emerald-800';
        if (job.state === 'failed' || job.state === 'canceled') badgeColor = 'bg-red-50 text-red-700 border-red-200 dark:bg-red-900/30 dark:text-red-400 dark:border-red-800';

        const safeState = SAFE_JOB_STATES.has(job.state) ? job.state : 'offline';
        const safePipeline = String(job.pipeline || 'unknown');
        const safeId = String(job.id || 'job_unknown');
        const safeProgress = Math.max(0, Math.min(100, Number(job.progress) || 0));
        const canCancel = _portalPrivilegesReady() && (job.state === 'running' || job.state === 'queued');
        const artifactCount = Array.isArray(job.artifacts) ? job.artifacts.length : 0;
        const errorLine = getReadableError(job.error);
        const outcomeSummary = jobOutcomeSummary(job);
        const transportLabel = formatTransportLabel(job);
        const freshnessLabel = formatRelativeTime(Number(job.lastEventAt || job.updatedAt || job.createdAt || 0));

        const header = document.createElement('div');
        header.className = 'flex items-center justify-between mb-3';

        const headerLeft = document.createElement('div');
        headerLeft.className = 'flex items-center gap-2';

        const statusDot = document.createElement('span');
        statusDot.className = `status-dot ${safeState}`;
        headerLeft.appendChild(statusDot);

        const pipelineSpan = document.createElement('span');
        pipelineSpan.className = 'text-[12px] font-semibold text-slate-900 dark:text-slate-100 truncate max-w-[150px]';
        pipelineSpan.textContent = safePipeline;
        headerLeft.appendChild(pipelineSpan);

        header.appendChild(headerLeft);

        const cancelButton = document.createElement('button');
        cancelButton.dataset.action = 'cancel-job';
        cancelButton.dataset.jobId = safeId;
        cancelButton.className = `text-[10px] uppercase font-bold text-red-500 hover:text-red-700 focus:outline-none ${canCancel ? '' : 'hidden'}`;
        cancelButton.textContent = 'Cancel';
        header.appendChild(cancelButton);
        li.appendChild(header);

        const meta = document.createElement('div');
        meta.className = 'flex items-center justify-between gap-2 text-[10px] text-slate-500 dark:text-slate-400 font-mono mb-2';

        const idSpan = document.createElement('span');
        idSpan.textContent = `${safeId.substring(0, 8)}...`;
        meta.appendChild(idSpan);

        const metaRight = document.createElement('div');
        metaRight.className = 'flex items-center gap-1.5';

        const transportChip = document.createElement('span');
        transportChip.className = 'job-chip';
        transportChip.textContent = transportLabel;
        metaRight.appendChild(transportChip);

        const stateBadge = document.createElement('span');
        stateBadge.className = `px-2 py-0.5 rounded-full border ${badgeColor}`;
        stateBadge.textContent = safeState;
        metaRight.appendChild(stateBadge);

        meta.appendChild(metaRight);
        li.appendChild(meta);

        const subMeta = document.createElement('div');
        subMeta.className = 'mb-3 flex items-center justify-between gap-2 text-[10px] text-slate-500 dark:text-slate-400';

        const freshness = document.createElement('span');
        freshness.className = 'micro-status';
        freshness.textContent = `Updated ${freshnessLabel}`;
        subMeta.appendChild(freshness);

        const artifacts = document.createElement('span');
        artifacts.className = 'job-chip';
        artifacts.textContent = `${artifactCount} artifact${artifactCount === 1 ? '' : 's'}`;
        subMeta.appendChild(artifacts);

        li.appendChild(subMeta);

        const progressRow = document.createElement('div');
        progressRow.className = 'flex items-center gap-2 mt-1';

        const progressEl = document.createElement('progress');
        progressEl.max = 100;
        progressEl.value = safeProgress;
        progressEl.className = 'flex-1';
        progressRow.appendChild(progressEl);

        const progressText = document.createElement('span');
        progressText.className = 'text-[10px] font-medium text-slate-500 dark:text-slate-400 w-8 text-right';
        progressText.textContent = `${safeProgress}%`;
        progressRow.appendChild(progressText);
        li.appendChild(progressRow);

        const summary = document.createElement('p');
        summary.className = 'mt-3 text-[11px] leading-5 text-slate-500 dark:text-slate-400';
        summary.textContent = job.state === 'partial' && outcomeSummary
            ? outcomeSummary
            : errorLine
                ? errorLine
                : outcomeSummary || `${artifactCount} artifact${artifactCount === 1 ? '' : 's'} indexed`;
        li.appendChild(summary);

        fragment.appendChild(li);
    });

    els.jobList.innerHTML = '';
    els.jobList.appendChild(fragment);
    if (els.queueStatusSummary) {
        const selected = state.jobs.find((job) => job.id === state.selectedJobId) || null;
        els.queueStatusSummary.textContent = selected
            ? `Inspector focus: ${String(selected.pipeline || 'job')} • updated ${formatRelativeTime(Number(selected.updatedAt || selected.createdAt || 0))}`
            : 'Newest jobs stay pinned to the top.';
    }
    if (includeReviewSurfaces) renderReviewSurfaces();
}

function handleJobListKeydown(event) {
    const row = event.target.closest('li[data-job-id]');
    if (!row || !els.jobList) return;
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        selectJob(row.dataset.jobId);
        return;
    }
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const rows = Array.from(els.jobList.querySelectorAll('li[data-job-id]'));
    const currentIndex = rows.indexOf(row);
    if (currentIndex === -1) return;
    let nextIndex = currentIndex;
    if (event.key === 'ArrowDown') nextIndex = Math.min(rows.length - 1, currentIndex + 1);
    if (event.key === 'ArrowUp') nextIndex = Math.max(0, currentIndex - 1);
    if (event.key === 'Home') nextIndex = 0;
    if (event.key === 'End') nextIndex = rows.length - 1;
    const nextRow = rows[nextIndex];
    if (nextRow) nextRow.focus();
}

function _focusArtifactRailButton(path) {
    if (!els.artifactThumbnailRail) return;
    const targetPath = String(path || '').trim();
    if (!targetPath) return;
    const buttons = Array.from(els.artifactThumbnailRail.querySelectorAll('button[data-artifact-path]'));
    const nextButton = buttons.find((candidate) => String(candidate.dataset.artifactPath || '').trim() === targetPath);
    if (nextButton) nextButton.focus();
}

function handleArtifactRailKeydown(event) {
    const button = event.target.closest('button[data-artifact-path]');
    if (!button || !els.artifactThumbnailRail) return;
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        button.click();
        return;
    }
    if (!['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const buttons = Array.from(els.artifactThumbnailRail.querySelectorAll('button[data-artifact-path]'));
    const currentIndex = buttons.indexOf(button);
    if (currentIndex === -1) return;
    let nextIndex = currentIndex;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex = Math.min(buttons.length - 1, currentIndex + 1);
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex = Math.max(0, currentIndex - 1);
    if (event.key === 'Home') nextIndex = 0;
    if (event.key === 'End') nextIndex = buttons.length - 1;
    const nextButton = buttons[nextIndex];
    if (nextButton) nextButton.focus();
}

function logToPane(jobId, line) {
    if (state.selectedJobId === jobId || !state.selectedJobId) {
        if (els.logPane) {
            els.logPane.textContent += line + '\n';
            els.logPane.scrollTop = els.logPane.scrollHeight;
        }
    }
}

async function checkBackend(force = false) {
    const now = Date.now();
    if (!force && (healthCheckInFlight || (now - lastHealthCheckAt) < HEALTH_CHECK_MIN_GAP_MS)) {
        return;
    }
    const wasOnline = state.backendOk;
    healthCheckInFlight = true;
    lastHealthCheckAt = now;
    try {
        const healthEndpointPath = _healthEndpointPath();
        state.bootstrap.lastHealthEndpointPath = healthEndpointPath;
        const res = await fetchWithTimeout(`${API_BASE}${healthEndpointPath}`, {}, HEALTH_CHECK_TIMEOUT_MS);
        if (res.ok) {
            state.backendOk = true;
            if (els.healthIndicator) els.healthIndicator.className = 'status-dot ready';
            if (els.healthText) els.healthText.textContent = 'Backend Online';
            if (!wasOnline) {
                state.jobs.forEach((job) => {
                    if (_isJobStreamRecoverable(job)) {
                        _noteTransportWarning(job, 'backend_recovered', 'Backend connectivity restored. Live telemetry can resume.', 'info');
                    }
                });
                _queueBootstrapOnlineFollowup();
            }
            _flushBootstrapOnlineFollowup(force);
        } else {
            throw new Error('backend not ready');
        }
    } catch {
        state.backendOk = false;
        state.readiness = { server: {}, pipelines: {} };
        state.bootstrap.pendingOnlineFollowup = false;
        state.bootstrap.onlineFollowupComplete = false;
        if (
            state.jobs.length === 0 &&
            (state.jobsLoadStatus === 'pending' || state.jobsLoadStatus === 'loading')
        ) {
            state.jobsLoadStatus = 'offline';
        }
        state.jobs.forEach((job) => {
            if (_isJobStreamRecoverable(job)) {
                _noteTransportWarning(job, 'backend_offline', 'Backend health check failed. Live telemetry may be stale until connectivity is restored.', 'warn');
            }
        });
        if (els.healthIndicator) els.healthIndicator.className = 'status-dot offline';
        if (els.healthText) els.healthText.textContent = 'Backend Offline';
    } finally {
        renderJobQueue(false);
        renderReviewSurfaces();
        _syncBootstrapGuardedControls();
        healthCheckInFlight = false;
    }
}

async function extractApiError(res) {
    const raw = await res.text();
    let payload = null;
    try {
        payload = raw ? JSON.parse(raw) : null;
    } catch {
        payload = null;
    }

    if (payload && payload.error && payload.error.message) {
        const code = payload.error.code ? `[${payload.error.code}] ` : '';
        return { message: `${code}${payload.error.message}`, error: payload.error };
    }
    if (payload && typeof payload.detail === 'string') {
        return { message: payload.detail, error: null };
    }
    return { message: raw || `Request failed (${res.status})`, error: null };
}

function _applyJobStreamEvent(job, eventName, parsed) {
    if (!job) return;
    _markJobEventActivity(job);
    if (eventName === 'log') {
        const line = String(parsed.line || '');
        appendJobLog(job, line);
        logToPane(job.id, line);
        return;
    }
    if (eventName === 'progress') {
        job.progress = Math.max(0, Math.min(100, Number(parsed.progress) || 0));
         _recordProgressTimeline(job, job.progress, Date.now());
        scheduleRenderJobQueue();
        return;
    }
    if (eventName === 'state') {
        const nextState = String(parsed.state || job.state || 'running');
        job.state = nextState;
        if (nextState === 'running' && !Number.isFinite(job.startedAt)) {
            job.startedAt = Date.now();
        } else if (nextState === 'running' && (!job.startedAt || job.startedAt <= 0)) {
            job.startedAt = Date.now();
        }
        _reconcileJobTimeline(job);
        scheduleRenderJobQueue();
        return;
    }
    if (eventName === 'artifact') {
        upsertArtifact(job, {
            artifact_type: String(parsed.artifact_type || 'file'),
            media_kind: String(parsed.media_kind || parsed.artifact_type || 'file'),
            previewable: Boolean(parsed.previewable),
            content_type: typeof parsed.content_type === 'string' ? parsed.content_type : '',
            url: typeof parsed.url === 'string' ? parsed.url : '',
            path: String(parsed.path || ''),
            relative_path: String(parsed.relative_path || parsed.path || ''),
            size_bytes: typeof parsed.size_bytes === 'number' ? parsed.size_bytes : null,
            sha256: typeof parsed.sha256 === 'string' ? parsed.sha256 : '',
            display_hint: _normalizeArtifactDisplayHint(parsed.display_hint)
        });
        _pushTimelineEntry(
            job,
            _timelineEntry(
                'artifact',
                'Artifact indexed',
                `${String(parsed.relative_path || parsed.path || 'artifact')} is ready for review.`,
                Date.now(),
                'success',
                `artifact|${String(parsed.path || '')}`
            )
        );
        scheduleRenderJobQueue();
        return;
    }
    if (eventName === 'done') {
        job.state = String(parsed.state || job.state || 'failed');
        job.progress = (job.state === 'succeeded' || job.state === 'partial') ? 100 : job.progress;
        job.finishedAt = Date.now();
        _recordProgressTimeline(job, job.progress, job.finishedAt);
        if (parsed.error) {
            job.error = parsed.error;
            const readable = getReadableError(parsed.error);
            if (readable) {
                const line = `[ERROR] ${readable}`;
                appendJobLog(job, line);
                logToPane(job.id, line);
            }
        }
        if (parsed.artifacts && parsed.artifacts.items) {
            job.artifacts = normalizeArtifactItems(parsed.artifacts);
        }
        if (parsed.run_summary) {
            job.run_summary = normalizeRunSummary(parsed.run_summary);
        }
        _reconcileJobTimeline(job);
        const endLine = `[SYSTEM] Stream closed. Exit code: ${parsed.exit_code}`;
        appendJobLog(job, endLine);
        logToPane(job.id, endLine);
        stopJobActivity(job);
        scheduleRenderJobQueue();
        createToast(
            job.state === 'partial' ? 'Job partially completed. Reviewable outputs are available.' : `Job ${job.state}.`,
            job.state === 'succeeded' ? 'success' : job.state === 'partial' ? 'info' : 'error'
        );
        if (state.selectedJobId === job.id && els.logStatusIndicator) els.logStatusIndicator.classList.add('hidden');
    }
}

async function _startAuthorizedFetchSse(job, eventsUrl) {
    if (!job || !eventsUrl || !_isJobStreamRecoverable(job)) return;
    _ensureJobStreamState(job);
    const controller = new AbortController();
    const handle = { close: () => controller.abort() };
    job.fetchAbortController = controller;
    job.eventSource = handle;
    const headers = _buildAuthHeaders({ 'Accept': 'text/event-stream' });
    const streamUrl = `${API_BASE}${eventsUrl}`;
    let sawDoneEvent = false;
    let shouldReconnect = true;
    try {
        const res = await fetch(streamUrl, {
            method: 'GET',
            headers,
            signal: controller.signal,
            cache: 'no-store'
        });
        if (!res.ok) {
            const parsedError = await extractApiError(res);
            if (parsedError.error) job.error = parsedError.error;
            const status = Number(res.status) || 0;
            const isAuthError = status === 401 || status === 403;
            const isRetryableStatus = status === 429 || status >= 500;
            shouldReconnect = isRetryableStatus;
            if (!isRetryableStatus) {
                job.reconnectBlocked = true;
                _clearSseRetry(job, true);
            }
            if (isAuthError) {
                _noteTransportWarning(
                    job,
                    'auth_blocked',
                    _isManagedAuthMode()
                        ? 'Managed authentication expired or was denied. Restore the managed session to resume transport.'
                        : 'API-key backed transport is blocked. Update credentials to resume live events.',
                    'warn'
                );
                const line = _isManagedAuthMode()
                    ? `[ERROR] Event stream authorization failed (${status}). Restore the managed session to resume live job events.`
                    : `[ERROR] Event stream authorization failed (${status}). Update API key to resume live job events.`;
                appendJobLog(job, line);
                logToPane(job.id, line);
                createToast(
                    _isManagedAuthMode()
                        ? 'Event stream authorization failed. Restore the managed session to resume live logs.'
                        : 'Event stream authorization failed. Update API key to resume live logs.',
                    'error'
                );
            } else {
                _noteTransportWarning(job, `stream_status_${status}`, `Event stream subscribe failed with status ${status}. ${parsedError.message}`, isRetryableStatus ? 'warn' : 'error');
                const line = `[WARN] Event stream subscribe failed (${status}). ${parsedError.message}`;
                appendJobLog(job, line);
                logToPane(job.id, line);
            }
            scheduleRenderJobQueue();
            return;
        }
        if (!res.body) return;
        const reconnectAttempt = Number(job.sseRetry?.attempt) || 0;
        job.reconnectBlocked = false;
        _markJobEventActivity(job);
        if (reconnectAttempt > 0) {
            void emitPortalEvent('stream_reconnected', {
                surface: 'stream_transport',
                metadata: {
                    attempt: reconnectAttempt,
                    job_id: String(job.id || ''),
                    transport: 'fetch'
                }
            });
        }
        scheduleRenderJobQueue();

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let currentEvent = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            let newlineIndex = buffer.indexOf('\n');
            while (newlineIndex !== -1) {
                const rawLine = buffer.slice(0, newlineIndex);
                buffer = buffer.slice(newlineIndex + 1);
                const line = rawLine.replace(/\r$/, '');

                if (line.startsWith(':')) {
                    _markJobEventActivity(job);
                    newlineIndex = buffer.indexOf('\n');
                    continue;
                }
                if (line.startsWith('event: ')) {
                    currentEvent = line.slice(7).trim();
                    newlineIndex = buffer.indexOf('\n');
                    continue;
                }
                if (line.startsWith('data: ')) {
                    const dataText = line.slice(6).trim();
                    if (!dataText) {
                        newlineIndex = buffer.indexOf('\n');
                        continue;
                    }
                    try {
                        const parsed = JSON.parse(dataText);
                        _applyJobStreamEvent(job, currentEvent, parsed);
                        if (currentEvent === 'done') {
                            sawDoneEvent = true;
                            return;
                        }
                    } catch {
                        // Ignore malformed stream lines and continue processing.
                    }
                    newlineIndex = buffer.indexOf('\n');
                    continue;
                }
                if (!line) currentEvent = '';
                newlineIndex = buffer.indexOf('\n');
            }
        }
    } catch (err) {
        if (!controller.signal.aborted) {
            _noteTransportWarning(job, 'stream_error', `Event stream error: ${err instanceof Error ? err.message : String(err)}`, 'warn');
            const line = `[WARN] Event stream error: ${err instanceof Error ? err.message : String(err)}`;
            appendJobLog(job, line);
            logToPane(job.id, line);
            scheduleRenderJobQueue();
        }
    } finally {
        if (job.eventSource === handle) {
            job.eventSource = null;
        }
        if (job.fetchAbortController === controller) {
            job.fetchAbortController = null;
        }
        if (shouldReconnect && !sawDoneEvent && !controller.signal.aborted && _isJobStreamRecoverable(job)) {
            scheduleSseReconnect(job);
        }
    }
}

function startJobEventStream(job, eventsUrl) {
    if (!job) return;
    _ensureJobStreamState(job);
    if (typeof eventsUrl === 'string' && eventsUrl.trim()) {
        job.eventStreamUrl = eventsUrl.trim();
    }
    if (!_isJobStreamRecoverable(job)) {
        stopJobActivity(job);
        return;
    }
    job.reconnectBlocked = false;
    _clearSseRetry(job, false);
    if (_jobHasActiveStream(job)) return;

    const token = _currentApiToken();
    const streamPath = job.eventStreamUrl;
    const streamUrl = `${API_BASE}${streamPath}`;
    job.lastEventAt = Date.now();
    if (token) {
        job.usesFetchSse = true;
        void _startAuthorizedFetchSse(job, streamPath);
        return;
    }

    job.usesFetchSse = false;
    const es = new EventSource(streamUrl);
    job.eventSource = es;
    es.onopen = () => {
        if (job.eventSource !== es) return;
        const reconnectAttempt = Number(job.sseRetry?.attempt) || 0;
        job.reconnectBlocked = false;
        _markJobEventActivity(job);
        if (reconnectAttempt > 0) {
            void emitPortalEvent('stream_reconnected', {
                surface: 'stream_transport',
                metadata: {
                    attempt: reconnectAttempt,
                    job_id: String(job.id || ''),
                    transport: 'native'
                }
            });
        }
        scheduleRenderJobQueue();
    };
    const safeParseSseEvent = (eventName, e) => {
        try {
            const parsed = JSON.parse(e.data);
            _applyJobStreamEvent(job, eventName, parsed);
        } catch {
            // Ignore malformed SSE data payloads; continue streaming.
        }
    };
    es.addEventListener('log', (e) => safeParseSseEvent('log', e));
    es.addEventListener('progress', (e) => safeParseSseEvent('progress', e));
    es.addEventListener('state', (e) => safeParseSseEvent('state', e));
    es.addEventListener('artifact', (e) => safeParseSseEvent('artifact', e));
    es.addEventListener('done', (e) => safeParseSseEvent('done', e));
    es.onerror = () => {
        if (job.eventSource !== es) return;
        const readyState = _nativeEventSourceReadyState(es);
        if (readyState === EVENT_SOURCE_READY_STATE_CONNECTING) {
            _noteTransportWarning(job, 'eventsource_reconnecting', 'Native SSE connection dropped. Browser is retrying in the background.', 'warn');
            scheduleRenderJobQueue();
            return;
        }
        const warningCode = readyState === EVENT_SOURCE_READY_STATE_CLOSED ? 'eventsource_closed' : 'eventsource_error';
        const warningDetail = readyState === EVENT_SOURCE_READY_STATE_CLOSED
            ? 'Native SSE connection closed. Reconnecting to restore live telemetry.'
            : 'Native SSE transport failed. Reconnecting to restore live telemetry.';
        const logLine = readyState === EVENT_SOURCE_READY_STATE_CLOSED
            ? '[WARN] Native SSE connection closed. Reconnecting...'
            : '[WARN] Native SSE transport failed. Reconnecting...';
        _noteTransportWarning(job, warningCode, warningDetail, 'warn');
        appendJobLog(job, logLine);
        logToPane(job.id, logLine);
        _teardownJobEventStream(job);
        scheduleRenderJobQueue();
        scheduleSseReconnect(job);
    };
}

async function recoverJobs() {
    if (!state.backendOk) return;
    if (state.jobs.length === 0) {
        state.jobsLoadStatus = 'loading';
        renderJobQueue();
    }
    try {
        const headers = _buildAuthHeaders({ 'Accept': 'application/json' });
        const res = await fetch(`${API_BASE}/v1/jobs`, { headers });
        if (!res.ok) {
            if (state.jobs.length === 0) {
                state.jobsLoadStatus = 'error';
                renderJobQueue();
            }
            return;
        }
        const payload = await res.json();
        const jobsFromServer = Array.isArray(payload?.data?.jobs) ? payload.data.jobs : [];
        if (jobsFromServer.length === 0) {
            state.jobsLoadStatus = 'ready';
            renderJobQueue();
            return;
        }

        const byId = new Map(state.jobs.map((job) => [job.id, job]));
        [...jobsFromServer].reverse().forEach((rawJob) => {
            const id = String(rawJob.id || '');
            if (!id) return;
            const hydrated = hydrateJobFromServer(rawJob);
            if (!hydrated) return;
            const existing = byId.get(id);
            if (existing) {
                _syncHydratedJob(existing, hydrated, rawJob);
            } else {
                state.jobs.push(hydrated);
                byId.set(id, hydrated);
            }
        });

        state.jobsLoadStatus = 'ready';

        if (state.selectedJobId && !state.jobs.some((job) => job.id === state.selectedJobId)) {
            state.selectedJobId = '';
        }
        if (!state.selectedJobId && state.jobs.length > 0) {
            const retained = state.jobs.find((job) => job.id === state.portalUi.lastSelectedJobId) || null;
            const nextSelectedJob = retained || state.jobs[state.jobs.length - 1];
            state.selectedJobId = nextSelectedJob.id;
            _rememberSelectedJob(nextSelectedJob.id);
        }

        state.jobs.forEach((job) => {
            if (_isJobStreamRecoverable(job) && !_jobHasActiveStream(job)) {
                startJobEventStream(job, job.eventStreamUrl);
            }
        });
        if (state.selectedJobId) selectJob(state.selectedJobId);
        else {
            scheduleRenderJobQueue();
        }
    } catch (err) {
        if (state.jobs.length === 0) {
            state.jobsLoadStatus = 'error';
            renderJobQueue();
        }
    }
}

async function cancelJob(id) {
    const job = state.jobs.find((item) => item.id === id);
    if (!job || (job.state !== 'running' && job.state !== 'queued')) return;
    if (_blockManagedUnavailableAction('change job state')) return;

    void emitPortalEvent('cancel_requested', {
        surface: 'job_queue',
        metadata: {
            job_id: String(job.id || ''),
            pipeline: String(job.pipeline || '')
        }
    });
    stopJobActivity(job);
    job.state = 'canceled';
    appendJobLog(job, `[WARN] Cancelled by user.`);
    logToPane(id, `[WARN] Cancelled by user.`);

    const headers = _buildAuthHeaders({}, 'POST');
    fetch(`${API_BASE}/v1/jobs/${id}/cancel`, { method: 'POST', headers }).catch(() => {});

    scheduleRenderJobQueue();
    if (state.selectedJobId === id && els.logStatusIndicator) els.logStatusIndicator.classList.add('hidden');
    createToast("Job canceled.", "error");
}

function handleJobListClick(event) {
    const cancelBtn = event.target.closest('[data-action="cancel-job"]');
    if (cancelBtn) {
        event.stopPropagation();
        cancelJob(cancelBtn.dataset.jobId);
        return;
    }

    const row = event.target.closest('li[data-job-id]');
    if (!row) return;
    selectJob(row.dataset.jobId);
}

async function submitJob() {
    if (_blockManagedUnavailableAction('dispatch jobs')) return;
    const payload = generatePayload();
    const readinessStatus = currentPipelineDispatchStatus();

    if (payload.pipeline === 'lux-depth-v3') {
        const preview = _currentPreviewForPayload(payload);
        if (!preview || preview.status === 'loading') {
            createToast('Configuration preview is still refreshing. Try dispatch again in a moment.', 'info');
            void emitPortalEvent('dispatch_blocked', {
                surface: 'dispatch',
                reasons: ['preview_loading']
            });
            return;
        }
        if (preview.status === 'error') {
            const previewFailure = _previewFailureDetails(preview);
            createToast(previewFailure.toastMessage, 'error');
            void emitPortalEvent('dispatch_blocked', {
                surface: 'dispatch',
                reasons: [previewFailure.telemetryReason]
            });
            return;
        }
        if (Array.isArray(preview.field_errors) && preview.field_errors.length > 0) {
            const firstError = preview.field_errors[0];
            const conflictError = preview.field_errors.find(
                (item) => String(item?.code || '').trim() === 'conflicting_log_verbosity_flags'
            );
            createToast(
                conflictError
                    ? 'verbose and quiet are mutually exclusive; disable one flag.'
                    : String(firstError?.message || 'Preview validation blocked dispatch.'),
                'error'
            );
            void emitPortalEvent('dispatch_blocked', {
                surface: 'dispatch',
                field: String(firstError?.field || '').trim(),
                reasons: preview.field_errors.map((item) => String(item?.code || '')).filter(Boolean).slice(0, 8)
            });
            return;
        }
        if (_effectiveDebugBundleEnabled(preview, payload) && !state.portalUi.debugBundleAcknowledged) {
            createToast('Acknowledge the reconstruction debug-bundle guardrail before dispatch.', 'error');
            void emitPortalEvent('dispatch_blocked', {
                surface: 'dispatch',
                field: 'debug_bundle_acknowledged',
                reasons: ['debug_bundle_acknowledgement_required']
            });
            return;
        }
    }

    if (!state.backendOk) {
        createToast('Backend is offline. Dispatch is disabled until connectivity is restored.', 'error');
        return;
    }
    if (!readinessStatus) {
        createToast('Execution readiness is still loading. Try again in a moment.', 'info');
        return;
    }
    if (readinessStatus !== 'ready') {
        const firstIssue = currentPipelineReadinessIssues()[0];
        createToast(firstIssue?.message || 'Pipeline is blocked by missing prerequisites.', 'error');
        void emitPortalEvent('dispatch_blocked', {
            surface: 'dispatch',
            reasons: [String(firstIssue?.reason || 'readiness_blocked').trim().toLowerCase() || 'readiness_blocked']
        });
        return;
    }

    if (els.runJobBtn) {
        els.runJobBtn.disabled = true;
        els.runJobBtn.textContent = "Dispatching...";
    }

    const randomId = window.crypto?.randomUUID ? window.crypto.randomUUID().replace(/-/g, '').slice(0, 8) : Math.random().toString(36).slice(2, 8);
    const localId = `job_${randomId}`;

    const job = {
        id: localId,
        pipeline: payload.pipeline,
        state: 'queued',
        progress: 0,
        logs: [],
        artifacts: [],
        error: null,
        eventSource: null,
        fetchAbortController: null,
        eventStreamUrl: null,
        usesFetchSse: false,
        sseRetry: { attempt: 0, timer: null },
        reconnectBlocked: false,
        lastEventAt: 0,
        mockInterval: null,
        startedAt: 0,
        finishedAt: 0,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        timeline: [],
        transportWarnings: [],
        progressMilestones: []
    };
    _reconcileJobTimeline(job);
    state.jobs.push(job);
    state.selectedJobId = job.id;
    if (els.logPane) els.logPane.textContent = '';
    if (els.logStatusIndicator) els.logStatusIndicator.classList.remove('hidden');
    scheduleRenderJobQueue();
    const initLine = `[INFO] Initializing ${payload.pipeline}...`;
    appendJobLog(job, initLine);
    logToPane(job.id, initLine);

    try {
        const headers = _buildAuthHeaders({ 'Content-Type': 'application/json' }, 'POST');
        const res = await fetch(`${API_BASE}/v1/jobs`, {
            method: 'POST',
            headers,
            body: JSON.stringify(payload)
        });
        if (!res.ok) {
            const parsedError = await extractApiError(res);
            if (parsedError.error) job.error = parsedError.error;
            throw new Error(parsedError.message);
        }
        const data = await res.json();

        job.id = data.data.id;
        state.selectedJobId = job.id;
        state.portalUi.lastSelectedJobId = String(job.id || '');
        scheduleRenderJobQueue();
        createToast(`Job dispatched: ${job.id}`, 'success');
        void emitPortalEvent('job_submitted', {
            surface: 'dispatch',
            metadata: {
                job_id: String(job.id || ''),
                pipeline: String(job.pipeline || payload.pipeline || '')
            }
        });

        startJobEventStream(job, data.data.events_url);
    } catch (err) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        job.state = 'failed';
        job.finishedAt = Date.now();
        appendJobLog(job, `[ERROR] ${errorMessage}`);
        _reconcileJobTimeline(job);
        scheduleRenderJobQueue();
        createToast("Backend submission failed.", "error");
        if (state.selectedJobId === job.id && els.logStatusIndicator) els.logStatusIndicator.classList.add('hidden');
    } finally {
        if (els.runJobBtn) {
            els.runJobBtn.textContent = "Execute Job";
        }
        _syncBootstrapGuardedControls();
    }
}

function startSseWatchdog() {
    if (sseWatchdogIntervalId !== null) return;
    sseWatchdogIntervalId = setInterval(() => {
        if (document.hidden) return;
        const now = Date.now();
        state.jobs.forEach((job) => {
            if (!job) return;
            _ensureJobStreamState(job);
            if (job.reconnectBlocked) return;

            if (!_isJobStreamRecoverable(job)) {
                _clearSseRetry(job, true);
                return;
            }
            if (job.sseRetry.timer) return;

            if (!_jobHasActiveStream(job)) {
                const reconnectDetail = job.usesFetchSse
                    ? 'Fetch event stream is not active. Reconnecting to restore live telemetry.'
                    : 'Native SSE stream is not active. Reconnecting to restore live telemetry.';
                const reconnectCode = job.usesFetchSse ? 'fetch_stream_inactive' : 'eventsource_inactive';
                _noteTransportWarning(job, reconnectCode, reconnectDetail, 'warn');
                _teardownJobEventStream(job);
                scheduleSseReconnect(job);
                return;
            }

            if (job.lastEventAt > 0 && (now - job.lastEventAt) <= SSE_STALL_THRESHOLD_MS) return;
            const stalledDetail = job.usesFetchSse
                ? 'Fetch event stream stalled. Reconnecting to restore live telemetry.'
                : 'Native SSE stream stalled. Reconnecting to restore live telemetry.';
            const stalledLine = job.usesFetchSse
                ? '[WARN] Fetch event stream stalled. Reconnecting...'
                : '[WARN] Native SSE stream stalled. Reconnecting...';
            _noteTransportWarning(job, 'stream_stalled', stalledDetail, 'warn');
            appendJobLog(job, stalledLine);
            logToPane(job.id, stalledLine);
            _teardownJobEventStream(job);
            scheduleSseReconnect(job);
        });
    }, SSE_STALL_CHECK_INTERVAL_MS);
}

function stopSseWatchdog() {
    if (sseWatchdogIntervalId === null) return;
    clearInterval(sseWatchdogIntervalId);
    sseWatchdogIntervalId = null;
}

function startHealthPolling() {
    if (healthPollIntervalId !== null) return;
    startSseWatchdog();
    healthPollIntervalId = setInterval(() => {
        if (!document.hidden) checkBackend();
    }, HEALTH_CHECK_INTERVAL_MS);
}

function stopHealthPolling() {
    if (healthPollIntervalId === null) return;
    clearInterval(healthPollIntervalId);
    healthPollIntervalId = null;
    stopSseWatchdog();
}

if (els.saveProfileBtn) els.saveProfileBtn.addEventListener('click', () => {
    const name = prompt("Profile name:");
    if (!name || !name.trim()) return;
    const profiles = getProfiles();
    profiles[name.trim()] = { pipeline: state.pipeline, config: JSON.parse(JSON.stringify(state.config)) };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(profiles));
    refreshProfileDropdown();
    els.profileSelect.value = name.trim();
    createToast(`Profile "${name.trim()}" saved.`, "success");
});

if (els.profileSelect) els.profileSelect.addEventListener('change', (e) => {
    const name = e.target.value;
    if (!name) return;
    const profiles = getProfiles();
    if (profiles[name]) {
        state.pipeline = profiles[name].pipeline;
        state.config = JSON.parse(JSON.stringify(profiles[name].config));
        updateUIFromState();
        void fetchPresetsForPipeline(state.pipeline, true);
        createToast(`Profile ${name} loaded.`);
    }
});

if (els.exportBtn) els.exportBtn.addEventListener('click', () => {
    const payload = generatePayload();
    const effectivePreview = _effectivePreviewSnapshot(payload);
    const exportPayload = {
        schema: 'tp.portal.export.v1',
        pipeline: payload.pipeline,
        args: payload.args,
        effective_args: effectivePreview.normalized_args || payload.args,
        execution_args: effectivePreview.execution_args || payload.args,
        inactive_fields: effectivePreview.inactive_fields || [],
        estimate_summary: effectivePreview.estimate_summary || _buildLocalEstimateSummary(payload.args || {}),
        argv_preview: String(effectivePreview.argv_preview || els.cliPreview?.textContent || '')
    };
    const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `job_${state.pipeline}_config.json`;
    a.click();
    URL.revokeObjectURL(url);
    void emitPortalEvent('config_exported', {
        surface: 'effective_config',
        metadata: { has_preview: effectivePreview.status === 'ready' }
    });
});

if (els.heroRunBtn) {
    els.heroRunBtn.addEventListener('click', () => {
        navigateConsoleView('build');
        setBuildStep(1, { silent: true });
        if (els.pipelineSelect) els.pipelineSelect.focus();
    });
}

if (els.resumeDraftBtn) {
    els.resumeDraftBtn.addEventListener('click', () => {
        navigateConsoleView('build');
        syncBuildStepUi();
        const activeStep = document.querySelector('.build-step-tab.is-active');
        if (activeStep && typeof activeStep.focus === 'function') activeStep.focus();
    });
}

if (els.heroExportBtn) {
    els.heroExportBtn.addEventListener('click', () => {
        const jobId = String(els.heroExportBtn.dataset.jobId || '').trim();
        if (!jobId) {
            createToast('No active job is available right now.', 'info');
            return;
        }
        navigateConsoleView('operate', { jobId });
    });
}

if (els.importBtn) els.importBtn.addEventListener('click', () => els.fileInput.click());
if (els.fileInput) els.fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
        const data = JSON.parse(await file.text());
        if (data.pipeline) state.pipeline = data.pipeline;
        if (data.args) {
            const c = state.config;
            c.inputDir = data.args.archive_root || data.args.input_dir || c.inputDir;
            c.outputDir = data.args.output_dir || c.outputDir;
            c.preset = data.args.preset || c.preset;
            c.qualityTier = _resolveQualityTier(data.args.quality_tier || c.qualityTier);
            c.depthBackend = _resolveDepthBackend(data.args.depth_backend || c.depthBackend);
            c.depthDevice = data.args.depth_device || c.depthDevice;
            c.segmentation = c.segmentation || {};
            c.segmentation.enable = parseBoolLike(data.args.enable_segmentation, c.segmentation.enable);
            c.segmentation.backend = _resolveSegmentationBackend(data.args.segmentation_backend || c.segmentation.backend);
            c.segmentation.sam2ModelSize = _resolveSam2ModelSize(data.args.sam2_model_size || c.segmentation.sam2ModelSize);
            c.segmentation.sam2CheckpointPath = _textOrFallback(
                data.args.sam2_checkpoint_path,
                c.segmentation.sam2CheckpointPath
            );
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

            c.emits.master16 = parseBoolLike(data.args.emit_master16, c.emits.master16);
            c.emits.upscaled16 = parseBoolLike(data.args.emit_upscaled16, c.emits.upscaled16);
            c.emits.marketing = parseBoolLike(data.args.emit_marketing, c.emits.marketing);
            c.emits.report = parseBoolLike(data.args.emit_report, c.emits.report);
            c.emits.runCard = parseBoolLike(data.args.emit_run_card, c.emits.runCard);

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

            c.runtime = c.runtime || {};
            c.runtime.maxWorkers = _parsePositiveIntOrNull(data.args.max_workers) || '';
            c.runtime.maxWorkersMode = c.runtime.maxWorkers === '' ? 'auto' : 'fixed';
            c.runtime.maxGpuWorkers = _parsePositiveIntOrNull(data.args.max_gpu_workers) || '';
            c.runtime.maxGpuWorkersMode = c.runtime.maxGpuWorkers === '' ? 'auto' : 'fixed';
            c.runtime.logLevel = _textOrFallback(data.args.log_level, c.runtime.logLevel);
        }
        state.portalUi.debugBundleAcknowledged = false;
        state.portalUi.debugBundleGuardrailSeen = false;
        updateUIFromState();
        void fetchPresetsForPipeline(state.pipeline, true);
        void fetchConfigMetadata(state.pipeline, true);
        createToast("Configuration imported.", "success");
    } catch (err) {
        createToast("Invalid JSON file.", "error");
    }
    els.fileInput.value = "";
});

if (els.copyCliBtn) els.copyCliBtn.addEventListener('click', async () => {
    await copyToClipboard(els.cliPreview.textContent);
});

if (els.inspectorOverviewTab) els.inspectorOverviewTab.addEventListener('click', () => setInspectorTab('overview'));
if (els.inspectorTimelineTab) els.inspectorTimelineTab.addEventListener('click', () => setInspectorTab('timeline'));
if (els.inspectorLogsTab) els.inspectorLogsTab.addEventListener('click', () => setInspectorTab('logs'));
if (els.openRunDetailsBtn) {
    els.openRunDetailsBtn.addEventListener('click', () => {
        if (!state.selectedJobId) {
            createToast('Select a run first, then open its review surface.', 'info');
            return;
        }
        const selectedJob = state.jobs.find((item) => item.id === state.selectedJobId);
        void emitPortalEvent('run_details_opened', {
            surface: 'job_inspector',
            metadata: {
                job_id: String(state.selectedJobId || ''),
                pipeline: String(selectedJob?.pipeline || '')
            }
        });
        navigateConsoleView('review', { jobId: state.selectedJobId });
    });
}

if (els.artifactThumbnailRail) {
    els.artifactThumbnailRail.addEventListener('click', (event) => {
        const button = event.target.closest('button[data-artifact-path]');
        if (!button) return;
        const selectedJob = state.jobs.find((item) => item.id === state.selectedJobId);
        if (!selectedJob) return;
        const path = String(button.dataset.artifactPath || '').trim();
        if (!path) return;
        const shouldRestoreFocus = event.detail === 0;
        _rememberArtifactSelection(String(selectedJob.id || ''), path);
        renderReviewSurfaces();
        if (shouldRestoreFocus) {
            requestAnimationFrame(() => {
                _focusArtifactRailButton(path);
            });
        }
    });
    els.artifactThumbnailRail.addEventListener('keydown', handleArtifactRailKeydown);
}

if (els.artifactCompareBtn) {
    els.artifactCompareBtn.addEventListener('click', () => {
        const selectedJob = state.jobs.find((item) => item.id === state.selectedJobId);
        if (!selectedJob) return;
        const key = String(selectedJob.id || '');
        _rememberComparePreference(key, !Boolean(state.artifactUi.compareByJob[key]));
        void emitPortalEvent('artifact_compared', {
            surface: 'artifact_review',
            metadata: {
                enabled: Boolean(state.artifactUi.compareByJob[key]),
                job_id: key,
                pipeline: String(selectedJob.pipeline || '')
            }
        });
        renderReviewSurfaces();
    });
}

if (els.openArtifactBtn) {
    els.openArtifactBtn.addEventListener('click', () => {
        const url = sanitizeManagedAssetUrl(els.openArtifactBtn.dataset.url);
        if (!url) {
            createToast('No artifact URL is available for this selection.', 'error');
            return;
        }
        const selectedJob = state.jobs.find((item) => item.id === state.selectedJobId);
        void emitPortalEvent('artifact_opened', {
            surface: 'artifact_review',
            metadata: {
                job_id: String(selectedJob?.id || ''),
                media_kind: String(_selectedArtifactForJob(selectedJob)?.media_kind || 'file'),
                pipeline: String(selectedJob?.pipeline || '')
            }
        });
        window.open(url, '_blank', 'noopener,noreferrer');
    });
}

if (els.downloadArtifactBtn) {
    els.downloadArtifactBtn.addEventListener('click', () => {
        const url = sanitizeManagedAssetUrl(els.downloadArtifactBtn.dataset.url);
        if (!url) {
            createToast('No artifact is available to download.', 'error');
            return;
        }
        const filename = String(els.downloadArtifactBtn.dataset.filename || 'artifact');
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = filename;
        anchor.rel = 'noopener noreferrer';
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
    });
}

if (els.copyArtifactPathBtn) {
    els.copyArtifactPathBtn.addEventListener('click', async () => {
        const path = String(els.copyArtifactPathBtn.dataset.path || '').trim();
        if (!path) {
            createToast('No artifact path is available for this selection.', 'error');
            return;
        }
        await copyToClipboard(path);
    });
}

if (els.viewRunCardBtn) els.viewRunCardBtn.addEventListener('click', async () => {
    const runCardUrl = sanitizeManagedAssetUrl(els.viewRunCardBtn.dataset.url);
    if (!runCardUrl) {
        createToast('Run card URL is not available for this job.', 'error');
        return;
    }
    window.open(runCardUrl, '_blank', 'noopener,noreferrer');
});

if (els.copyRunCardPathBtn) els.copyRunCardPathBtn.addEventListener('click', async () => {
    const runCardPath = String(els.copyRunCardPathBtn.dataset.path || '').trim();
    if (!runCardPath) {
        createToast('Run card path is not available for this job.', 'error');
        return;
    }
    await copyToClipboard(runCardPath);
});

if (els.copyRunCardFingerprintBtn) {
    els.copyRunCardFingerprintBtn.addEventListener('click', async () => {
        const fingerprint = String(els.copyRunCardFingerprintBtn.dataset.fingerprint || '').trim();
        if (!fingerprint) {
            createToast('No run card fingerprint available for this job.', 'error');
            return;
        }
        await copyToClipboard(fingerprint);
    });
}

if (els.clearLogsBtn) els.clearLogsBtn.addEventListener('click', () => {
    if (els.logPane) els.logPane.textContent = '';
});
if (els.runJobBtn) els.runJobBtn.addEventListener('click', submitJob);
if (els.refreshHealthBtn) {
    els.refreshHealthBtn.addEventListener('click', () => {
        const jobId = String(els.refreshHealthBtn.dataset.jobId || '').trim();
        if (!jobId) {
            createToast('No reviewable output is available yet.', 'info');
            return;
        }
        navigateConsoleView('review', { jobId });
    });
}
if (els.jobList) els.jobList.addEventListener('click', handleJobListClick);
if (els.jobList) els.jobList.addEventListener('keydown', handleJobListKeydown);

function _normalizeThemePreference(value) {
    const normalized = String(value || '').trim().toLowerCase();
    return THEME_PREFERENCES.includes(normalized) ? normalized : '';
}

function _systemThemeMode(themeQuery = null) {
    const query = themeQuery || window.matchMedia('(prefers-color-scheme: dark)');
    return query.matches ? 'dark' : 'light';
}

function _effectiveThemeFromPreference(preference, themeQuery = null) {
    return preference === 'system' ? _systemThemeMode(themeQuery) : preference;
}

function _migrateThemePreferenceStorage() {
    const storageVersion = localStorage.getItem(THEME_STORAGE_VERSION_KEY);
    if (storageVersion === THEME_STORAGE_VERSION) return;

    if (localStorage.getItem(THEME_STORAGE_KEY) !== null) {
        localStorage.removeItem(THEME_STORAGE_KEY);
    }

    localStorage.setItem(THEME_STORAGE_VERSION_KEY, THEME_STORAGE_VERSION);
}

function _nextThemePreference(preference) {
    const normalizedPreference = _normalizeThemePreference(preference) || 'system';
    const currentIndex = THEME_PREFERENCES.indexOf(normalizedPreference);
    const nextIndex = (currentIndex + 1) % THEME_PREFERENCES.length;
    return THEME_PREFERENCES[nextIndex];
}

function _syncThemeButton() {
    if (!els.themeBtn) return;
    const preference = _normalizeThemePreference(state.themePreference) || 'system';
    const nextPreference = _nextThemePreference(preference);
    const preferenceLabel = preference.charAt(0).toUpperCase() + preference.slice(1);
    const nextLabel = nextPreference.charAt(0).toUpperCase() + nextPreference.slice(1);
    const effectiveLabel = state.theme === 'dark' ? 'Dark' : 'Light';

    els.themeBtn.textContent = preference === 'system' ? `Theme: System (${effectiveLabel})` : `Theme: ${preferenceLabel}`;
    els.themeBtn.setAttribute('aria-label', `Theme preference: ${preference}. Click to switch to ${nextPreference}.`);
    els.themeBtn.title = preference === 'system' ? `Following system (${effectiveLabel}). Next: ${nextLabel}.` : `Theme locked to ${preferenceLabel}. Next: ${nextLabel}.`;
}

function applyThemePreference(preference, options) {
    const normalizedPreference = _normalizeThemePreference(preference) || 'system';
    const themeOptions = options && typeof options === 'object' ? options : null;
    const persist = !themeOptions || themeOptions.persist !== false;
    const mode = _effectiveThemeFromPreference(normalizedPreference, themeOptions && themeOptions.themeQuery ? themeOptions.themeQuery : null);

    state.themePreference = normalizedPreference;
    state.theme = mode;
    document.documentElement.classList.toggle('dark', mode === 'dark');

    if (persist) {
        localStorage.setItem(THEME_STORAGE_VERSION_KEY, THEME_STORAGE_VERSION);
        if (normalizedPreference === 'system') localStorage.removeItem(THEME_STORAGE_KEY);
        else localStorage.setItem(THEME_STORAGE_KEY, normalizedPreference);
    }

    _syncThemeButton();
}

if (els.themeBtn) els.themeBtn.addEventListener('click', () => {
    applyThemePreference(_nextThemePreference(state.themePreference));
});

function _rememberOverlayTrigger(trigger = document.activeElement) {
    state.portalUi.lastOverlayTrigger = trigger && typeof trigger.focus === 'function' ? trigger : null;
}

function _restoreOverlayFocus() {
    const trigger = state.portalUi.lastOverlayTrigger;
    if (trigger && document.contains(trigger) && typeof trigger.focus === 'function') {
        trigger.focus();
    }
    state.portalUi.lastOverlayTrigger = null;
}

function _overlayFocusableElements(root) {
    if (!root) return [];
    return Array.from(
        root.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')
    ).filter((element) => {
        if (!(element instanceof HTMLElement)) return false;
        if (element.hasAttribute('disabled')) return false;
        if (element.getAttribute('aria-hidden') === 'true') return false;
        return window.getComputedStyle(element).display !== 'none';
    });
}

function _activeOverlayPanel() {
    if (els.shortcutsModal && !els.shortcutsModal.classList.contains('hidden')) {
        return els.shortcutsPanel;
    }
    if (els.effectiveConfigDrawer && !els.effectiveConfigDrawer.classList.contains('hidden')) {
        return els.effectiveConfigDrawer.querySelector('[role="dialog"]');
    }
    return null;
}

function _trapOverlayFocus(event) {
    if (event.key !== 'Tab') return false;
    const panel = _activeOverlayPanel();
    if (!panel) return false;
    const focusable = _overlayFocusableElements(panel);
    if (focusable.length === 0) return false;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const activeElement = document.activeElement;
    if (event.shiftKey && (activeElement === first || !panel.contains(activeElement))) {
        event.preventDefault();
        last.focus();
        return true;
    }
    if (!event.shiftKey && (activeElement === last || !panel.contains(activeElement))) {
        event.preventDefault();
        first.focus();
        return true;
    }
    return false;
}

const toggleModal = (show, trigger = document.activeElement) => {
    if (show) {
        _rememberOverlayTrigger(trigger);
        els.shortcutsModal.classList.remove('hidden');
        els.shortcutsModal.classList.add('flex');
        void els.shortcutsModal.offsetWidth;
        els.shortcutsModal.classList.remove('opacity-0');
        els.shortcutsPanel.classList.remove('scale-95', 'opacity-0');
        els.shortcutsPanel.classList.add('scale-100', 'opacity-100');
        els.shortcutsModal.setAttribute("aria-hidden", "false");
        els.shortcutsModal.dataset.overlayOpen = 'true';
        els.closeShortcutsBtn.focus();
    } else {
        els.shortcutsModal.classList.add('opacity-0');
        els.shortcutsPanel.classList.remove('scale-100', 'opacity-100');
        els.shortcutsPanel.classList.add('scale-95', 'opacity-0');
        setTimeout(() => {
            els.shortcutsModal.classList.add('hidden');
            els.shortcutsModal.classList.remove('flex');
            els.shortcutsModal.setAttribute("aria-hidden", "true");
            els.shortcutsModal.dataset.overlayOpen = 'false';
            _restoreOverlayFocus();
        }, 200);
    }
};

const toggleEffectiveConfigDrawer = (show, trigger = document.activeElement) => {
    if (!els.effectiveConfigDrawer) return;
    state.portalUi.effectiveConfigOpen = Boolean(show);
    if (show) {
        _rememberOverlayTrigger(trigger);
        renderEffectiveConfigDrawer(generatePayload());
        els.effectiveConfigDrawer.classList.remove('hidden');
        els.effectiveConfigDrawer.classList.add('flex');
        els.effectiveConfigDrawer.setAttribute('aria-hidden', 'false');
        els.effectiveConfigDrawer.dataset.overlayOpen = 'true';
        if (els.closeEffectiveConfigBtn) els.closeEffectiveConfigBtn.focus();
        void emitPortalEvent('effective_config_opened', {
            surface: 'effective_config',
            metadata: { preview_ready: _currentPreviewForPayload(generatePayload())?.status === 'ready' }
        });
        return;
    }
    els.effectiveConfigDrawer.classList.add('hidden');
    els.effectiveConfigDrawer.classList.remove('flex');
    els.effectiveConfigDrawer.setAttribute('aria-hidden', 'true');
    els.effectiveConfigDrawer.dataset.overlayOpen = 'false';
    _restoreOverlayFocus();
};

if (els.shortcutsBtn) els.shortcutsBtn.addEventListener('click', (event) => toggleModal(true, event.currentTarget));
if (els.closeShortcutsBtn) els.closeShortcutsBtn.addEventListener('click', () => toggleModal(false));
if (els.shortcutsModal) els.shortcutsModal.addEventListener('click', (e) => {
    if (e.target === els.shortcutsModal) toggleModal(false);
});
if (els.openEffectiveConfigBtn) {
    els.openEffectiveConfigBtn.addEventListener('click', (event) => toggleEffectiveConfigDrawer(true, event.currentTarget));
}
if (els.effectiveConfigBtn) {
    els.effectiveConfigBtn.addEventListener('click', (event) => toggleEffectiveConfigDrawer(true, event.currentTarget));
}
if (els.closeEffectiveConfigBtn) {
    els.closeEffectiveConfigBtn.addEventListener('click', () => toggleEffectiveConfigDrawer(false));
}
if (els.effectiveConfigDrawer) {
    els.effectiveConfigDrawer.addEventListener('click', (e) => {
        if (e.target === els.effectiveConfigDrawer) toggleEffectiveConfigDrawer(false);
    });
}

document.addEventListener('keydown', (e) => {
    if (_trapOverlayFocus(e)) {
        return;
    }
    if (e.key === "Escape" && els.effectiveConfigDrawer && !els.effectiveConfigDrawer.classList.contains("hidden")) {
        e.preventDefault();
        toggleEffectiveConfigDrawer(false);
        return;
    }
    if (e.key === "Escape" && els.shortcutsModal && !els.shortcutsModal.classList.contains("hidden")) {
        e.preventDefault();
        toggleModal(false);
        return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        submitJob();
    }
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'c') {
        e.preventDefault();
        if (els.copyCliBtn) els.copyCliBtn.click();
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'j') {
        e.preventDefault();
        if (els.themeBtn) els.themeBtn.click();
    }
});

document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        _resetAmbientTargets();
        stopHealthPolling();
        return;
    }
    reconcileBuildSurfaceFromDom();
    void checkBackend(true);
    startHealthPolling();
});

function setupDisclosurePanels() {
    const ensureDisclosurePrefs = () => {
        if (!state.portalUi.disclosurePrefs) {
            state.portalUi.disclosurePrefs = {
                advanced: null,
                governance: null,
                reconstruction: null,
                dispatchTools: false,
            };
        }
        return state.portalUi.disclosurePrefs;
    };

    const registerDisclosurePanel = (name, element) => {
        if (!element) return;
        element.addEventListener('toggle', () => {
            const disclosurePrefs = ensureDisclosurePrefs();
            if (name === 'dispatchTools') {
                disclosurePrefs.dispatchTools = Boolean(element.open);
                syncDisclosurePanels(generatePayload());
                return;
            }

            const autoOpen = element.dataset.autoOpen === 'true';
            if (autoOpen) {
                disclosurePrefs[name] = null;
                if (!element.open) {
                    window.requestAnimationFrame(() => {
                        syncDisclosurePanels(generatePayload());
                    });
                }
                return;
            }

            disclosurePrefs[name] = Boolean(element.open);
            syncDisclosurePanels(generatePayload());
        });
    };

    registerDisclosurePanel('advanced', els.advancedFlagsDetails);
    registerDisclosurePanel('governance', els.governanceDetails);
    registerDisclosurePanel('reconstruction', els.reconstructionDetails);
    registerDisclosurePanel('dispatchTools', els.dispatchToolsDetails);
}

window.addEventListener('beforeunload', cleanupActiveJobHandles);
window.addEventListener('pagehide', cleanupActiveJobHandles);
window.addEventListener('pageshow', () => {
    reconcileBuildSurfaceFromDom();
});
window.addEventListener('focus', () => {
    reconcileBuildSurfaceFromDom();
});
window.addEventListener('popstate', () => {
    applyConsoleRouteFromLocation(true);
    renderJobQueue();
});

async function init() {
    const themeQuery = window.matchMedia('(prefers-color-scheme: dark)');
    _migrateThemePreferenceStorage();
    const savedThemePreference = _normalizeThemePreference(localStorage.getItem(THEME_STORAGE_KEY)) || 'system';
    applyThemePreference(savedThemePreference, { persist: false, themeQuery });
    setupAmbientMotion();

    themeQuery.addEventListener('change', () => {
        if (state.themePreference === 'system') {
            applyThemePreference('system', { persist: false, themeQuery });
        }
    });

    const bootstrapPromise = loadPortalBootstrap();
    seedPresetFallbacks();
    refreshProfileDropdown();
    applyConsoleRouteFromLocation(true);
    updateUIFromState();
    setupBuildStepper();
    bindInputs();
    setupDisclosurePanels();
    if (window.requestAnimationFrame) {
        window.requestAnimationFrame(() => {
            reconcileBuildSurfaceFromDom();
        });
    } else {
        window.setTimeout(() => {
            reconcileBuildSurfaceFromDom();
        }, 0);
    }
    setupSectionRail();
    _syncBootstrapUi();
    renderJobQueue();
    void checkBackend(true);
    void fetchConfigMetadata(state.pipeline, true);
    startHealthPolling();
    await bootstrapPromise;
    renderJobQueue();
}

document.addEventListener('DOMContentLoaded', () => {
    void init();
});
