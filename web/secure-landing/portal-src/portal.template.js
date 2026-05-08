// ============================================================================
// TRANSFORMATION PORTAL — OPERATOR CONSOLE
// ============================================================================
//
// This file contains the client-side logic for the portal operator console.
// It is organized into the following sections:
//
//   1. CONSTANTS          - Configuration constants and magic values
//   2. STATE              - Application state container
//   3. DOM REFERENCES     - Element references (els)
//   4. AMBIENT MOTION     - Background animation system
//   5. ROUTING            - View routing (?view=overview|build|operate|review)
//   6. BUILD STEPPER      - Multi-step build flow UI
//   7. UTILITIES          - Format helpers, string manipulation
//   8. ARTIFACT HELPERS   - Artifact classification and labeling
//   9. JOB RENDERING      - Job queue and inspector rendering
//  10. PREVIEW & CONFIG   - Config preview and validation
//  11. API LAYER          - Fetch, SSE, health checks
//  12. THEME              - Dark/light mode toggle
//  13. OVERLAYS           - Shortcuts, effective config drawers
//  14. INITIALIZATION     - Startup and event binding
//
// Contract notes:
//   - ?view= query param routes: overview, build, operate, review
//   - job=, artifact=, compare=1 additive params
//   - data-ui attributes used by browser smoke tests
//   - /portal/assets/* served by FastAPI
//
// ============================================================================

// ============================================================================
// 1. CONSTANTS
// ============================================================================

const API_BASE = '';
const STORAGE_KEY = 'tp_orchestrator_profiles_final';
const API_KEY_STORAGE_KEY = 'tp_api_key';
const TRANSIENT_DRAFT_STORAGE_KEY = 'tp_portal_transient_draft';
const TRANSIENT_DRAFT_SCHEMA = 'tp.portal.transient_draft.v1';
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
const BOOTSTRAP_RETRIABLE_HTTP_STATUSES = new Set([429, 500, 502, 503, 504]);
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
const CONFIG_PREVIEW_SERVICE_RETRY_BASE_MS = 2500;
const CONFIG_PREVIEW_SERVICE_RETRY_MAX_ATTEMPTS = 3;
const TRANSIENT_DRAFT_PERSIST_DEBOUNCE_MS = 200;
const DEFERRED_REVIEW_SURFACE_RETRY_WINDOW_MS = 30000;
const DEFERRED_SURFACE_RETRY_WINDOW_MS = 30000;
const DEFERRED_SURFACE_REGISTRY = Object.freeze({
    operate: { datasetKey: 'operateSurfaceJsUrl', factoryName: 'createDeferredOperateSurfaceApi' },
    build: { datasetKey: 'buildSurfaceJsUrl', factoryName: 'createDeferredBuildSurfaceApi' },
    overview: { datasetKey: 'overviewSurfaceJsUrl', factoryName: 'createDeferredOverviewSurfaceApi' },
});
const _deferredSurfaceState = new Map();
const DISPATCH_BACKEND_OFFLINE_MESSAGE = 'Backend is offline. Dispatch is disabled until connectivity is restored.';
const CONFIG_PREVIEW_SUPPORTED_PIPELINES = new Set([
    'lux-depth-v3',
    'archive-gate-a',
    'archive-gate-b',
    'archive-gate-c'
]);
const STAGED_UPLOAD_SUPPORTED_PIPELINES = new Set(['lux-depth-v3', 'archive-gate-a']);
const EVENT_SOURCE_READY_STATE_CONNECTING = 0;
const EVENT_SOURCE_READY_STATE_OPEN = 1;
const EVENT_SOURCE_READY_STATE_CLOSED = 2;
const TIMELINE_PROGRESS_CHECKPOINTS = [5, 25, 50, 75, 100];
const SAFE_JOB_STATES = new Set(['queued', 'running', 'succeeded', 'partial', 'failed', 'canceled', 'ready', 'offline']);
const SAFE_HTTP_METHODS = new Set(['GET', 'HEAD', 'OPTIONS']);

// Module-level mutable state for scheduling
let _operatePendingScheduleRender = false;
let _operatePendingIncludeReview = false;
let healthPollIntervalId = null;
let sseWatchdogIntervalId = null;
let healthCheckInFlight = false;
let lastHealthCheckAt = 0;
let configPreviewTimerId = null;
let configPreviewServiceRetryTimerId = null;
let configPreviewServiceRetryAttempts = 0;
// Latest Retry-After / X-RateLimit-Reset hint observed for the config-preview
// fetch path, parked here so the dispatch readiness banner can surface a live
// countdown to the user. Cleared on success and on _clearConfigPreviewServiceRetry.
let configPreviewLastRateLimitHint = null;
let transientDraftPersistTimerId = null;
let transientDraftPersistIdleId = null;
let deferredReviewSurfaceApi = null;
let deferredReviewSurfaceLoadPromise = null;
let deferredReviewSurfaceCssLoaded = false;
let deferredReviewSurfaceLoadFailedAt = 0;
let deferredReviewSurfaceLoadLastToastAt = 0;

/* __PORTAL_INTERNALS__ */

const portalInternals = __PortalInternal;
const portalRoute = portalInternals.createPortalRouteHelpers(window);
const portalDom = portalInternals.createDomContract(document, {
    devAssertions: portalInternals.shouldEnableDomAssertions(window)
});
const portalRenderScheduler = portalInternals.createRenderScheduler(window);
const portalRenderSurfaces = portalInternals.createRenderSurfaceRegistry();
const _domId = (id, required = false) => portalDom.id(id, { required });

// ============================================================================
// 2. STATE
// ============================================================================

const state = {
    pipeline: 'lux-depth-v3',
    config: portalInternals.createPortalConfigState(),
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
    metadata: portalInternals.createPortalMetadataState(),
    preview: portalInternals.createPortalPreviewState(),
    portalUi: portalInternals.createPortalUiState(),
    readiness: portalInternals.createPortalReadinessState(),
    themePreference: 'system',
    theme: 'light',
    presetsByPipeline: {},
    auth: portalInternals.createPortalAuthState(),
    bootstrap: portalInternals.createPortalBootstrapState(Date.now()),
    lastDiagnostics: portalInternals.createPortalLastDiagnosticsState(),
    rum: portalInternals.createPortalRumState()
};

// ============================================================================
// 3. DOM REFERENCES
// ============================================================================

const els = {
    overviewShell: _domId('overview-shell'),
    overviewRuntimeClarityShell: _domId('overviewRuntimeClarityShell'),
    overviewRuntimeBriefing: _domId('overviewRuntimeBriefing'),
    missionShell: _domId('mission-shell'),
    missionShellContent: _domId('missionShellContent'),
    missionShellSkeletonState: _domId('missionShellSkeletonState'),
    intelligenceShell: _domId('intelligence-shell'),
    intelligenceShellContent: _domId('intelligenceShellContent'),
    intelligenceShellSkeletonState: _domId('intelligenceShellSkeletonState'),
    consoleGrid: _domId('console-grid'),
    consoleContextShell: _domId('console-context-shell'),
    consoleViewTitle: _domId('consoleViewTitle'),
    consoleViewSummary: _domId('consoleViewSummary'),
    consoleViewMeta: _domId('consoleViewMeta'),
    consoleContextRibbon: _domId('consoleContextRibbon'),
    consoleActionRail: _domId('consoleActionRail'),
    consoleActionRailTitle: _domId('consoleActionRailTitle'),
    consoleActionRailDetail: _domId('consoleActionRailDetail'),
    consoleActionRailHint: _domId('consoleActionRailHint'),
    consoleActionRailActions: _domId('consoleActionRailActions'),
    consoleActionPrimaryBtn: _domId('consoleActionPrimaryBtn'),
    consoleActionSecondaryBtn1: _domId('consoleActionSecondaryBtn1'),
    consoleActionSecondaryBtn2: _domId('consoleActionSecondaryBtn2'),
    contextRibbonCard1: _domId('contextRibbonCard1'),
    contextRibbonCard1Label: _domId('contextRibbonCard1Label'),
    contextRibbonJob: _domId('contextRibbonJob'),
    contextRibbonJobMeta: _domId('contextRibbonJobMeta'),
    contextRibbonCard2: _domId('contextRibbonCard2'),
    contextRibbonCard2Label: _domId('contextRibbonCard2Label'),
    contextRibbonState: _domId('contextRibbonState'),
    contextRibbonFreshness: _domId('contextRibbonFreshness'),
    contextRibbonCard3: _domId('contextRibbonCard3'),
    contextRibbonCard3Label: _domId('contextRibbonCard3Label'),
    contextRibbonArtifact: _domId('contextRibbonArtifact'),
    contextRibbonArtifactMeta: _domId('contextRibbonArtifactMeta'),
    contextRibbonCard4: _domId('contextRibbonCard4'),
    contextRibbonCard4Label: _domId('contextRibbonCard4Label'),
    contextRibbonCompare: _domId('contextRibbonCompare'),
    contextRibbonCompareMeta: _domId('contextRibbonCompareMeta'),
    heroRunBtn: _domId('heroRunBtn'),
    resumeDraftBtn: _domId('resumeDraftBtn'),
    heroExportBtn: _domId('heroExportBtn'),
    refreshHealthBtn: _domId('refreshHealthBtn'),
    heroPipelineValue: _domId('heroPipelineValue'),
    heroPresetValue: _domId('heroPresetValue'),
    heroModeValue: _domId('heroModeValue'),
    heroQueueValue: _domId('heroQueueValue'),
    overviewStatsRow: _domId('overviewStatsRow'),
    overviewStatsSkeletonState: _domId('overviewStatsSkeletonState'),
    overviewCapabilityRow: _domId('overviewCapabilityRow'),
    overviewCapabilitySkeletonState: _domId('overviewCapabilitySkeletonState'),
    capabilityChips: _domId('capabilityChips'),
    presetHeadline: _domId('presetHeadline'),
    presetStabilityBadge: _domId('presetStabilityBadge'),
    backendModeBadge: _domId('backendModeBadge'),
    presetDescription: _domId('presetDescription'),
    presetBuilderHint: _domId('presetBuilderHint'),
    presetBuilderShell: _domId('presetBuilderShell'),
    flagsShell: _domId('flags-shell'),
    heroInputDir: _domId('heroInputDir'),
    heroOutputDir: _domId('heroOutputDir'),
    heroReadinessLabel: _domId('heroReadinessLabel'),
    heroWarningCount: _domId('heroWarningCount'),
    governanceBannerTitle: _domId('governanceBannerTitle'),
    governanceBannerBody: _domId('governanceBannerBody'),
    governanceChecklist: _domId('governanceChecklist'),
    buildStepTitle: _domId('buildStepTitle'),
    buildStepSummary: _domId('buildStepSummary'),
    buildStepTabs: _domId('buildStepTabs'),
    buildStepBackBtn: _domId('buildStepBackBtn'),
    buildStepNextBtn: _domId('buildStepNextBtn'),
    buildStepTab1: _domId('buildStepTab1'),
    buildStepTab2: _domId('buildStepTab2'),
    buildStepTab3: _domId('buildStepTab3'),
    buildStepTab4: _domId('buildStepTab4'),
    buildPulseDraftCard: _domId('buildPulseDraftCard'),
    buildPulseDraft: _domId('buildPulseDraft'),
    buildPulseDraftMeta: _domId('buildPulseDraftMeta'),
    buildPulseStepCard: _domId('buildPulseStepCard'),
    buildPulseStep: _domId('buildPulseStep'),
    buildPulseStepMeta: _domId('buildPulseStepMeta'),
    buildPulsePreviewCard: _domId('buildPulsePreviewCard'),
    buildPulsePreview: _domId('buildPulsePreview'),
    buildPulsePreviewMeta: _domId('buildPulsePreviewMeta'),
    buildPulseDispatchCard: _domId('buildPulseDispatchCard'),
    buildPulseDispatch: _domId('buildPulseDispatch'),
    buildPulseDispatchMeta: _domId('buildPulseDispatchMeta'),

    pipelineSelect: _domId('pipelineSelect'),
    presetSelect: _domId('presetSelect'),
    profileSelect: _domId('profileSelect'),
    saveProfileBtn: _domId('saveProfileBtn'),
    apiKeySection: _domId('apiKeySection'),
    apiKeyInput: _domId('apiKeyInput'),
    authModeBadge: _domId('authModeBadge'),
    apiKeyManagedHint: _domId('apiKeyManagedHint'),
    portalAccessState: _domId('portalAccessState'),
    bootstrapStatusBadge: _domId('bootstrapStatusBadge'),
    bootstrapRecoveryHint: _domId('bootstrapRecoveryHint'),
    governancePostureHint: _domId('governancePostureHint'),

    inputDir: _domId('inputDir'),
    outputDir: _domId('outputDir'),
    inputDirStatus: _domId('inputDirStatus'),
    stagedUploadShell: _domId('stagedUploadShell'),
    stagedUploadStatus: _domId('stagedUploadStatus'),
    stagedUploadDropzone: _domId('stagedUploadDropzone'),
    stagedUploadPickFilesBtn: _domId('stagedUploadPickFilesBtn'),
    stagedUploadPickFolderBtn: _domId('stagedUploadPickFolderBtn'),
    stagedUploadFilesInput: _domId('stagedUploadFilesInput'),
    stagedUploadFolderInput: _domId('stagedUploadFolderInput'),
    stagedUploadProgressBar: _domId('stagedUploadProgressBar'),
    stagedUploadProgressLabel: _domId('stagedUploadProgressLabel'),
    stagedUploadSummary: _domId('stagedUploadSummary'),
    stagedUploadError: _domId('stagedUploadError'),
    outputDirStatus: _domId('outputDirStatus'),
    archiveCanonicalCommand: _domId('archiveCanonicalCommand'),
    archiveCanonicalCommandHint: _domId('archiveCanonicalCommandHint'),
    archiveIndexField: _domId('archiveIndexField'),
    archiveIndexPath: _domId('archiveIndexPath'),
    archiveIndexStatus: _domId('archiveIndexStatus'),
    rightsManifestField: _domId('rightsManifestField'),
    rightsManifestPath: _domId('rightsManifestPath'),
    rightsManifestStatus: _domId('rightsManifestStatus'),
    qualityTier: _domId('qualityTier'),
    depthBackend: _domId('depthBackend'),
    modelKey: _domId('modelKey'),
    depthDevice: _domId('depthDevice'),
    segmentationBackendField: _domId('segmentationBackendField'),
    sam2ModelSizeField: _domId('sam2ModelSizeField'),
    strictSegmentationField: _domId('strictSegmentationField'),
    sam2CheckpointField: _domId('sam2CheckpointField'),
    sam2TuningPanel: _domId('sam2TuningPanel'),
    sam2TilingToggleField: _domId('sam2TilingToggleField'),
    sam2TilingConfigFields: _domId('sam2TilingConfigFields'),
    sam2GeneratorConfigFields: _domId('sam2GeneratorConfigFields'),
    sam2TuningHint: _domId('sam2TuningHint'),
    segmentationApplicabilityHint: _domId('segmentationApplicabilityHint'),
    segmentation: {
        enable: _domId('enableSegmentation'),
        backend: _domId('segmentationBackend'),
        sam2ModelSize: _domId('sam2ModelSize'),
        sam2CheckpointPath: _domId('sam2CheckpointPath'),
        sam2TilingEnabled: _domId('sam2TilingEnabled'),
        sam2TileSizePx: _domId('sam2TileSizePx'),
        sam2OverlapPx: _domId('sam2OverlapPx'),
        sam2GlobalPassLongestSide: _domId('sam2GlobalPassLongestSide'),
        sam2MaxConcurrency: _domId('sam2MaxConcurrency'),
        sam2PointsPerSide: _domId('sam2PointsPerSide'),
        sam2PointsPerBatch: _domId('sam2PointsPerBatch'),
        sam2PredIouThresh: _domId('sam2PredIouThresh'),
        sam2StabilityScoreThresh: _domId('sam2StabilityScoreThresh'),
        sam2CropNLayers: _domId('sam2CropNLayers'),
        strict: _domId('strictSegmentation')
    },

    flags: {
        materials: _domId('flagMaterials'),
        pbr: _domId('flagPBR'),
        cache: _domId('flagCache'),
        overwrite: _domId('flagOverwrite'),
        enableV2: _domId('flagEnableV2'),
        saveFloatDepth: _domId('saveFloatDepth'),
        forceDepth: _domId('forceDepth'),
        strictInputs: _domId('strictInputs'),
        verifyImages: _domId('verifyImages'),
        allowSemanticFallback: _domId('allowSemanticFallback'),
        verbose: _domId('verboseFlag'),
        quiet: _domId('quietFlag')
    },
    v2Preset: _domId('v2Preset'),
    v2PresetField: _domId('v2PresetField'),

    emits: {
        master16: _domId('emitMaster16'),
        upscaled16: _domId('emitUpscaled16'),
        marketing: _domId('emitMarketing'),
        report: _domId('emitReport'),
        runCard: _domId('emitRunCard'),
        runCardVersion: _domId('runCardVersion'),
        runCardIncludeProofs: _domId('emitRunCardIncludeProofs')
    },
    runCardVersionField: _domId('runCardVersionField'),

    licenses: {
        nonCommercialOk: _domId('licenseNonCommercial'),
        acceptApple: _domId('licenseApple'),
        acceptResearchTools: _domId('licenseResearchTools')
    },
    licenseNonCommercialField: _domId('licenseNonCommercialField'),
    licenseAppleField: _domId('licenseAppleField'),
    licenseResearchToolsField: _domId('licenseResearchToolsField'),
    governanceDetailsHint: _domId('governanceDetailsHint'),
    reconstruction: {
        enable: _domId('enableReconstruction'),
        groupingMode: _domId('groupingMode'),
        camerasSidecarPath: _domId('camerasSidecarPath'),
        iterations: _domId('reconstructionIterations'),
        tier: _domId('reconstructionTier'),
        emitSceneDebugBundle: _domId('emitSceneDebugBundle'),
        groupingModeStatus: _domId('groupingModeStatus'),
        iterationsStatus: _domId('reconstructionIterationsStatus'),
        camerasSidecarStatus: _domId('camerasSidecarStatus'),
        tierStatus: _domId('reconstructionTierStatus')
    },
    reconstructionConfigFields: _domId('reconstructionConfigFields'),
    runtimeTuningFields: _domId('runtimeTuningFields'),
    reconstructionDetailsHint: _domId('reconstructionDetailsHint'),
    reconstructionSummaryHint: _domId('reconstructionSummaryHint'),
    openEffectiveConfigBtn: _domId('openEffectiveConfigBtn'),
    effectiveConfigBtn: _domId('effectiveConfigBtn'),
    summaryReconstructionState: _domId('summaryReconstructionState'),
    summaryRuntimeWorkers: _domId('summaryRuntimeWorkers'),
    summaryRawIngest: _domId('summaryRawIngest'),
    summaryDebugBundle: _domId('summaryDebugBundle'),
    summaryPreviewState: _domId('summaryPreviewState'),
    estimateRuntimeBand: _domId('estimateRuntimeBand'),
    estimateGpuBand: _domId('estimateGpuBand'),
    estimateResearchRisk: _domId('estimateResearchRisk'),
    estimateSummaryLabel: _domId('estimateSummaryLabel'),
    debugBundleGuardrail: _domId('debugBundleGuardrail'),
    debugBundleDestination: _domId('debugBundleDestination'),
    debugBundleSensitivity: _domId('debugBundleSensitivity'),
    debugBundleAcknowledge: _domId('debugBundleAcknowledge'),
    debugBundleAcknowledgeHint: _domId('debugBundleAcknowledgeHint'),
    raw: {
        ingestMode: _domId('rawIngestMode'),
        wbMode: _domId('rawWbMode'),
        demosaic: _domId('rawDemosaic'),
        wbModeBadge: _domId('rawWbModeBadge'),
        wbModeHint: _domId('rawWbModeHint'),
        demosaicBadge: _domId('rawDemosaicBadge'),
        demosaicHint: _domId('rawDemosaicHint'),
        ingestModeStatus: _domId('rawIngestModeStatus')
    },
    captioning: {
        details: _domId('captioningDetails'),
        fields: _domId('captioningConfigFields'),
        enabledFields: _domId('fastVlmCaptioningFields'),
        enableFastVlm: _domId('enableFastVlmCaptioning'),
        model: _domId('fastVlmCaptioningModel'),
        proxyFormat: _domId('fastVlmProxyFormat'),
        maxSidePx: _domId('fastVlmMaxSidePx'),
        timeoutSeconds: _domId('fastVlmTimeoutSeconds'),
        pythonExecutable: _domId('fastVlmPythonExecutable'),
        mlxVlmDir: _domId('fastVlmMlxVlmDir'),
        status: _domId('captioningStatus'),
        readinessList: _domId('captioningReadinessList'),
        readinessScope: _domId('captioningReadinessScope')
    },
    runtime: {
        maxWorkersMode: _domId('maxWorkersMode'),
        maxWorkers: _domId('maxWorkers'),
        maxWorkersValueField: _domId('maxWorkersValueField'),
        maxWorkersStatus: _domId('maxWorkersStatus'),
        maxGpuWorkersMode: _domId('maxGpuWorkersMode'),
        maxGpuWorkers: _domId('maxGpuWorkers'),
        maxGpuWorkersValueField: _domId('maxGpuWorkersValueField'),
        maxGpuWorkersStatus: _domId('maxGpuWorkersStatus'),
        logLevel: _domId('logLevel'),
        logLevelStatus: _domId('logLevelStatus')
    },

    fieldsLuxDepth: _domId('fieldsLuxDepth'),
    fieldsArchiveGate: _domId('fieldsArchiveGate'),
    advancedFlagsDetails: _domId('advancedFlagsDetails'),
    captioningDetails: _domId('captioningDetails'),
    captioningDetailsHint: _domId('captioningDetailsHint'),
    captioningDetailsSummary: _domId('captioningDetailsSummary'),
    governanceDetails: _domId('governanceDetails'),
    reconstructionDetails: _domId('reconstructionDetails'),

    cliPreview: _domId('cliPreview'),
    copyCliBtn: _domId('copyCliBtn'),
    importBtn: _domId('importBtn'),
    exportBtn: _domId('exportBtn'),
    fileInput: _domId('fileInput'),
    runJobBtn: _domId('runJobBtn'),
    dispatchReadinessReason: _domId('dispatchReadinessReason'),
    dispatchToolsDetails: _domId('dispatchToolsDetails'),
    preRunWarnings: _domId('preRunWarnings'),
    preRunWarningsEmpty: _domId('preRunWarningsEmpty'),
    expectedOutputsList: _domId('expectedOutputsList'),
    datasetHealthIndicator: _domId('datasetHealthIndicator'),
    datasetHealthText: _domId('datasetHealthText'),
    nextBestActionLabel: _domId('nextBestActionLabel'),
    nextBestActionDetail: _domId('nextBestActionDetail'),
    nextBestActionTone: _domId('nextBestActionTone'),

    buildShell: _domId('build-shell'),
    profileShell: _domId('profile-shell'),
    profileShellContent: _domId('profileShellContent'),
    profileShellSkeletonState: _domId('profileShellSkeletonState'),
    buildRuntimeClarityShell: _domId('buildRuntimeClarityShell'),
    buildRuntimeBriefing: _domId('buildRuntimeBriefing'),
    buildStepperShell: _domId('buildStepperShell'),
    buildStepperShellContent: _domId('buildStepperShellContent'),
    buildStepperSkeletonState: _domId('buildStepperSkeletonState'),
    governanceShell: _domId('governance-shell'),
    parametersShell: _domId('parameters-shell'),
    parametersShellContent: _domId('parametersShellContent'),
    parametersShellSkeletonState: _domId('parametersShellSkeletonState'),
    jobsShell: _domId('jobs-shell'),
    selectedJobShell: _domId('selected-job-shell'),
    selectedJobShellContent: _domId('selectedJobShellContent'),
    selectedJobSkeletonState: _domId('selectedJobSkeletonState'),
    queueShell: _domId('queue-shell'),
    queueSkeletonState: _domId('queueSkeletonState'),
    jobList: _domId('jobList'),
    emptyQueueState: _domId('emptyQueueState'),
    emptyQueueTitle: _domId('emptyQueueTitle'),
    emptyQueueDetail: _domId('emptyQueueDetail'),
    emptyQueueAction: _domId('emptyQueueAction'),
    queueCount: _domId('queueCount'),
    selectedJobStateBadge: _domId('selectedJobStateBadge'),
    selectedJobIdLabel: _domId('selectedJobIdLabel'),
    selectedJobPipelineLabel: _domId('selectedJobPipelineLabel'),
    selectedJobArtifactCount: _domId('selectedJobArtifactCount'),
    selectedJobStreamStatus: _domId('selectedJobStreamStatus'),
    selectedJobProgressText: _domId('selectedJobProgressText'),
    selectedJobProgressBar: _domId('selectedJobProgressBar'),
    selectedJobMetaLine: _domId('selectedJobMetaLine'),
    selectedJobFreshness: _domId('selectedJobFreshness'),
    selectedJobSummary: _domId('selectedJobSummary'),
    selectedJobRecoveryTitle: _domId('selectedJobRecoveryTitle'),
    selectedJobRecoveryDetail: _domId('selectedJobRecoveryDetail'),
    selectedJobRecoveryActions: _domId('selectedJobRecoveryActions'),
    selectedJobRecoveryPrimaryBtn: _domId('selectedJobRecoveryPrimaryBtn'),
    selectedJobRecoverySecondaryBtn: _domId('selectedJobRecoverySecondaryBtn'),
    selectedJobTransportAlert: _domId('selectedJobTransportAlert'),
    openRunDetailsBtn: _domId('openRunDetailsBtn'),
    inspectorOverviewTab: _domId('inspectorOverviewTab'),
    inspectorTimelineTab: _domId('inspectorTimelineTab'),
    inspectorLogsTab: _domId('inspectorLogsTab'),
    selectedJobOverviewPanel: _domId('selectedJobOverviewPanel'),
    selectedJobTimelinePanel: _domId('selectedJobTimelinePanel'),
    selectedJobLogsPanel: _domId('selectedJobLogsPanel'),
    selectedJobTimelineList: _domId('selectedJobTimelineList'),
    selectedJobTimelineEmpty: _domId('selectedJobTimelineEmpty'),
    selectedJobLogPreview: _domId('selectedJobLogPreview'),
    artifactsShell: _domId('artifacts-shell'),
    artifactShellContent: _domId('artifactShellContent'),
    artifactSkeletonState: _domId('artifactSkeletonState'),
    artifactMeta: _domId('artifactMeta'),
    emptyArtifactState: _domId('emptyArtifactState'),
    emptyArtifactTitle: _domId('emptyArtifactTitle'),
    emptyArtifactDetail: _domId('emptyArtifactDetail'),
    emptyArtifactAction: _domId('emptyArtifactAction'),
    artifactCompareBtn: _domId('artifactCompareBtn'),
    artifactPreviewStage: _domId('artifactPreviewStage'),
    artifactCompareStage: _domId('artifactCompareStage'),
    artifactPreviewImage: _domId('artifactPreviewImage'),
    artifactPreviewSoloImage: _domId('artifactPreviewSoloImage'),
    artifactCompareImage: _domId('artifactCompareImage'),
    artifactPreviewPrimaryCaption: _domId('artifactPreviewPrimaryCaption'),
    artifactCompareCaption: _domId('artifactCompareCaption'),
    artifactMetadataCard: _domId('artifactMetadataCard'),
    artifactMetadataBar: _domId('artifactMetadataBar'),
    artifactSelectionTitle: _domId('artifactSelectionTitle'),
    artifactSelectionMeta: _domId('artifactSelectionMeta'),
    reviewStatusBanner: _domId('reviewStatusBanner'),
    reviewStatusTitle: _domId('reviewStatusTitle'),
    reviewStatusDetail: _domId('reviewStatusDetail'),
    reviewStatusAction: _domId('reviewStatusAction'),
    reviewStatusActions: _domId('reviewStatusActions'),
    reviewStatusPrimaryBtn: _domId('reviewStatusPrimaryBtn'),
    reviewStatusSecondaryBtn: _domId('reviewStatusSecondaryBtn'),
    reviewProvenanceGrid: _domId('reviewProvenanceGrid'),
    reviewProvenanceArtifactRole: _domId('reviewProvenanceArtifactRole'),
    reviewProvenanceRunState: _domId('reviewProvenanceRunState'),
    reviewProvenancePath: _domId('reviewProvenancePath'),
    reviewProvenanceFingerprint: _domId('reviewProvenanceFingerprint'),
    reviewProvenanceFreshness: _domId('reviewProvenanceFreshness'),
    reviewProvenanceSource: _domId('reviewProvenanceSource'),
    reviewProvenanceBatch: _domId('reviewProvenanceBatch'),
    reviewProvenanceCaptioning: _domId('reviewProvenanceCaptioning'),
    reviewCompareSummary: _domId('reviewCompareSummary'),
    reviewCompareTitle: _domId('reviewCompareTitle'),
    reviewCompareDetail: _domId('reviewCompareDetail'),
    openArtifactBtn: _domId('openArtifactBtn'),
    downloadArtifactBtn: _domId('downloadArtifactBtn'),
    copyArtifactPathBtn: _domId('copyArtifactPathBtn'),
    copyArtifactFingerprintBtn: _domId('copyArtifactFingerprintBtn'),
    artifactThumbnailRail: _domId('artifactThumbnailRail'),
    runCardActions: _domId('runCardActions'),
    viewRunCardBtn: _domId('viewRunCardBtn'),
    copyRunCardPathBtn: _domId('copyRunCardPathBtn'),
    copyRunCardFingerprintBtn: _domId('copyRunCardFingerprintBtn'),
    logsShell: _domId('logs-shell'),
    logPane: _domId('logPane'),
    logMetaLabel: _domId('logMetaLabel'),
    logStatusIndicator: _domId('logStatusIndicator'),
    clearLogsBtn: _domId('clearLogsBtn'),
    queueStatusSummary: _domId('queueStatusSummary'),

    themeBtn: _domId('themeBtn'),
    shortcutsBtn: _domId('shortcutsBtn'),
    shortcutsModal: _domId('shortcutsModal'),
    shortcutsPanel: _domId('shortcutsPanel'),
    closeShortcutsBtn: _domId('closeShortcutsBtn'),
    advancedFlagsSummary: _domId('advancedFlagsSummary'),
    governanceDetailsSummary: _domId('governanceDetailsSummary'),
    reconstructionDetailsSummary: _domId('reconstructionDetailsSummary'),
    dispatchToolsSummary: _domId('dispatchToolsSummary'),
    effectiveConfigDrawer: _domId('effectiveConfigDrawer'),
    closeEffectiveConfigBtn: _domId('closeEffectiveConfigBtn'),
    effectiveConfigMeta: _domId('effectiveConfigMeta'),
    requestedConfigJson: _domId('requestedConfigJson'),
    effectiveConfigJson: _domId('effectiveConfigJson'),
    inactiveConfigJson: _domId('inactiveConfigJson'),
    effectiveEstimateLabel: _domId('effectiveEstimateLabel'),
    effectiveReadinessSummary: _domId('effectiveReadinessSummary'),
    effectiveArgvPreview: _domId('effectiveArgvPreview'),
    artifactViewerModal: _domId('artifactViewerModal'),
    artifactViewerPanel: _domId('artifactViewerPanel'),
    artifactViewerTitle: _domId('artifactViewerTitle'),
    artifactViewerMeta: _domId('artifactViewerMeta'),
    artifactViewerStage: _domId('artifactViewerStage'),
    artifactViewerImage: _domId('artifactViewerImage'),
    artifactViewerFallback: _domId('artifactViewerFallback'),
    artifactViewerFallbackTitle: _domId('artifactViewerFallbackTitle'),
    artifactViewerFallbackDetail: _domId('artifactViewerFallbackDetail'),
    artifactViewerPath: _domId('artifactViewerPath'),
    artifactViewerFingerprint: _domId('artifactViewerFingerprint'),
    artifactViewerZoomValue: _domId('artifactViewerZoomValue'),
    artifactViewerPrevBtn: _domId('artifactViewerPrevBtn'),
    artifactViewerNextBtn: _domId('artifactViewerNextBtn'),
    artifactViewerZoomOutBtn: _domId('artifactViewerZoomOutBtn'),
    artifactViewerZoomInBtn: _domId('artifactViewerZoomInBtn'),
    artifactViewerResetZoomBtn: _domId('artifactViewerResetZoomBtn'),
    artifactViewerOpenRawBtn: _domId('artifactViewerOpenRawBtn'),
    artifactViewerCopyPathBtn: _domId('artifactViewerCopyPathBtn'),
    artifactViewerCopyFingerprintBtn: _domId('artifactViewerCopyFingerprintBtn'),
    artifactViewerStatus: _domId('artifactViewerStatus'),
    closeArtifactViewerBtn: _domId('closeArtifactViewerBtn'),

    healthIndicator: _domId('healthIndicator'),
    healthText: _domId('healthText'),
    toastContainer: _domId('toastContainer')
};

portalDom.assertPresent(els, [
    'overviewShell',
    'consoleGrid',
    'pipelineSelect',
    'presetSelect',
    'portalAccessState',
    'bootstrapStatusBadge',
    'jobList',
    'artifactPreviewStage',
    'artifactThumbnailRail',
    'toastContainer'
]);

// ============================================================================
// 4. AMBIENT MOTION
// ============================================================================

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

// ============================================================================
// 5. ROUTING
// ============================================================================

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

const WORKSPACE_VIEW_SHORTCUTS = Object.freeze({
    '1': 'overview',
    '2': 'build',
    '3': 'operate',
    '4': 'review'
});

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
    return portalRoute.build({
        viewName,
        jobId,
        artifactPath,
        compareEnabled,
        resolveView: resolveConsoleView,
        normalizeSelectedJobId: _normalizeSelectedJobId,
        normalizeArtifactRoutePath: _normalizeArtifactRoutePath,
        activeContext: _activeRouteContext
    });
}

function _syncConsoleRoute(replace = false) {
    const url = _routeUrlForView(state.currentView, state.selectedJobId);
    const nextHref = `${url.pathname}${url.search}${url.hash}`;
    const currentHref = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextHref === currentHref) return;
    const method = replace ? 'replaceState' : 'pushState';
    window.history[method]({ view: state.currentView, jobId: state.selectedJobId || '' }, '', nextHref);
}

function _managedReturnToPath() {
    const url = new URL(window.location.href);
    if (url.pathname !== '/portal') return '/portal';
    return `${url.pathname}${url.search}`;
}

function _managedLoginUrlForCurrentRoute() {
    return `/login?returnTo=${encodeURIComponent(_managedReturnToPath())}`;
}

function _copyTransientDraftConfig(config = state.config) {
    return JSON.parse(JSON.stringify(config && typeof config === 'object' ? config : portalInternals.createPortalConfigState()));
}

function _managedDraftOwnerKey() {
    const actor = state.auth && state.auth.actor && typeof state.auth.actor === 'object' ? state.auth.actor : null;
    const accessEmail = String(actor?.accessEmail || '').trim().toLowerCase();
    if (accessEmail) return `managed:${accessEmail}`;
    const username = String(actor?.username || '').trim().toLowerCase();
    const role = String(actor?.role || '').trim().toLowerCase();
    if (!username && !role) return '';
    return `managed:${[username, role].filter(Boolean).join(':')}`;
}

function _transientDraftOwnerKey() {
    if (!_isBootstrapReady()) return '';
    if (_isManagedAuthMode()) return _managedDraftOwnerKey();
    return 'direct_debug';
}

function _clearTransientPortalDraft() {
    try {
        sessionStorage.removeItem(TRANSIENT_DRAFT_STORAGE_KEY);
    } catch {
        // Ignore storage access failures during teardown or quota exhaustion.
    }
}

function _readTransientPortalDraft() {
    let raw = '';
    try {
        raw = sessionStorage.getItem(TRANSIENT_DRAFT_STORAGE_KEY) || '';
    } catch {
        return null;
    }
    if (!raw) return null;

    let parsed = null;
    try {
        parsed = JSON.parse(raw);
    } catch {
        _clearTransientPortalDraft();
        return null;
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        _clearTransientPortalDraft();
        return null;
    }

    const schema = String(parsed.schema || '').trim();
    const ownerKey = String(parsed.ownerKey || '').trim();
    const savedAt = Number(parsed.savedAt || 0);
    const pipeline = String(parsed.pipeline || '').trim();
    const config = parsed.config;
    if (
        schema !== TRANSIENT_DRAFT_SCHEMA
        || !ownerKey
        || !Number.isFinite(savedAt)
        || savedAt <= 0
        || !pipeline
        || !config
        || typeof config !== 'object'
        || Array.isArray(config)
    ) {
        _clearTransientPortalDraft();
        return null;
    }

    return {
        schema,
        ownerKey,
        savedAt,
        pipeline,
        config: _copyTransientDraftConfig(config),
        buildStep: resolveBuildStep(parsed.buildStep)
    };
}

function _persistTransientPortalDraft() {
    const ownerKey = _transientDraftOwnerKey();
    if (!ownerKey) return false;
    const snapshot = {
        schema: TRANSIENT_DRAFT_SCHEMA,
        pipeline: String(state.pipeline || '').trim() || 'lux-depth-v3',
        config: _copyTransientDraftConfig(),
        buildStep: resolveBuildStep(state.portalUi.buildStep),
        savedAt: Date.now(),
        ownerKey
    };
    try {
        sessionStorage.setItem(TRANSIENT_DRAFT_STORAGE_KEY, JSON.stringify(snapshot));
        return true;
    } catch {
        return false;
    }
}

function _cancelScheduledTransientPortalDraftPersist() {
    if (transientDraftPersistTimerId !== null) {
        window.clearTimeout(transientDraftPersistTimerId);
        transientDraftPersistTimerId = null;
    }
    if (transientDraftPersistIdleId !== null && typeof window.cancelIdleCallback === 'function') {
        window.cancelIdleCallback(transientDraftPersistIdleId);
    }
    transientDraftPersistIdleId = null;
}

function _flushPendingTransientPortalDraftPersist() {
    _cancelScheduledTransientPortalDraftPersist();
    return _persistTransientPortalDraft();
}

function _scheduleTransientPortalDraftPersist(options) {
    const settings = options && typeof options === 'object' ? options : {};
    if (settings.immediate) return _flushPendingTransientPortalDraftPersist();
    _cancelScheduledTransientPortalDraftPersist();

    const commitSnapshot = () => {
        transientDraftPersistTimerId = null;
        transientDraftPersistIdleId = null;
        _persistTransientPortalDraft();
    };
    const scheduleCommit = () => {
        transientDraftPersistTimerId = null;
        if (typeof window.requestIdleCallback === 'function') {
            transientDraftPersistIdleId = window.requestIdleCallback(() => {
                commitSnapshot();
            }, { timeout: TRANSIENT_DRAFT_PERSIST_DEBOUNCE_MS });
            return;
        }
        commitSnapshot();
    };

    transientDraftPersistTimerId = window.setTimeout(scheduleCommit, TRANSIENT_DRAFT_PERSIST_DEBOUNCE_MS);
    return true;
}

function _restoreTransientPortalDraft() {
    const snapshot = _readTransientPortalDraft();
    if (!snapshot) return false;
    const ownerKey = _transientDraftOwnerKey();
    if (!ownerKey || snapshot.ownerKey !== ownerKey) {
        _clearTransientPortalDraft();
        return false;
    }
    state.pipeline = snapshot.pipeline;
    state.config = _copyTransientDraftConfig(snapshot.config);
    state.portalUi.buildStep = resolveBuildStep(snapshot.buildStep);
    return true;
}

function _jobFreshnessLabel(job) {
    if (!job) return 'No live telemetry';
    const lastActivityAt = Number(job.lastEventAt || job.updatedAt || job.finishedAt || job.createdAt || 0);
    return `Updated ${formatRelativeTime(lastActivityAt)}`;
}

function _compareSurfaceCopy(selectedArtifact, compareArtifact, compareEnabled) {
    if (!selectedArtifact || !compareArtifact) {
        return {
            ribbonValue: 'No compare pair',
            ribbonMeta: 'No paired comparison is available for the current artifact.',
            summaryTitle: 'No compare pair',
            summaryDetail: 'No paired comparison is available for the current artifact.',
        };
    }

    const primaryLabel = artifactLabel(selectedArtifact);
    const compareLabel = artifactLabel(compareArtifact);
    if (compareEnabled) {
        return {
            ribbonValue: 'Compare on',
            ribbonMeta: 'Paired comparison is pinned in the URL-backed review context.',
            summaryTitle: 'Comparing paired outputs',
            summaryDetail: `${primaryLabel} is pinned against ${compareLabel} in this review context.`,
        };
    }

    return {
        ribbonValue: 'Pair available',
        ribbonMeta: 'Paired comparison is available for the current artifact selection.',
        summaryTitle: 'Paired comparison available',
        summaryDetail: `${compareLabel} is available as a side-by-side comparison for ${primaryLabel}.`,
    };
}

function _findJobById(jobId) {
    const normalizedJobId = _normalizeSelectedJobId(jobId);
    if (!normalizedJobId) return null;
    return state.jobs.find((job) => _normalizeSelectedJobId(job?.id) === normalizedJobId) || null;
}

function _jobHasReviewableOutputs(job) {
    if (!job) return false;
    const summary = normalizeRunSummary(job.run_summary);
    const artifactCount = Array.isArray(job.artifacts) ? job.artifacts.length : 0;
    return artifactCount > 0 || Boolean(summary?.reviewable_outputs);
}

function _operatorAction(key, label, options = {}) {
    const normalizedKey = String(key || '').trim();
    const normalizedLabel = String(label || '').trim();
    if (!normalizedKey || !normalizedLabel) return null;
    return {
        key: normalizedKey,
        label: normalizedLabel,
        tone: String(options.tone || 'info'),
        jobId: _normalizeSelectedJobId(options.jobId || ''),
        artifactPath: _normalizeArtifactRoutePath(options.artifactPath || ''),
        detail: String(options.detail || '').trim(),
        disabled: Boolean(options.disabled)
    };
}

function _compactOperatorActions(actions, maxCount = 2) {
    const seen = new Set();
    const compact = [];
    (Array.isArray(actions) ? actions : []).forEach((action) => {
        if (!action || !action.key || seen.has(action.key)) return;
        seen.add(action.key);
        compact.push(action);
    });
    return compact.slice(0, maxCount);
}

function _operatorActionContext(job) {
    const normalizedJobId = _normalizeSelectedJobId(job?.id);
    const artifacts = Array.isArray(job?.artifacts) ? rankArtifactsForDisplay(job.artifacts) : [];
    const selectedArtifact = job ? _selectedArtifactForJob(job) : null;
    const heroArtifact = artifacts[0] || selectedArtifact || null;
    const activeArtifact = selectedArtifact || heroArtifact;
    const compareCandidate = job ? findCompareArtifact(activeArtifact, artifacts) : null;
    const compareEnabled = Boolean(
        normalizedJobId
        && compareCandidate
        && state.artifactUi.compareByJob[normalizedJobId]
    );
    return {
        job,
        jobId: normalizedJobId,
        artifacts,
        artifactCount: artifacts.length,
        selectedArtifact: activeArtifact,
        heroArtifact,
        compareCandidate,
        compareEnabled,
        reviewableOutputs: _jobHasReviewableOutputs(job)
    };
}

function _preferredOperatorActionJob() {
    const preferredJobId = _preferredSelectedJobId();
    return _findJobById(preferredJobId) || _latestActiveJob() || _latestReviewableJob() || null;
}

function _operatorActionHintHtml() {
    const base = [
        '<span class="kbd">1</span> overview',
        '<span class="kbd">2</span> build',
        '<span class="kbd">3</span> operate',
        '<span class="kbd">4</span> review',
        '<span class="kbd">?</span> shortcuts'
    ];
    if (state.currentView === 'build') {
        base.push('<span class="kbd">Ctrl/⌘ + Enter</span> dispatch');
        base.push('<span class="kbd">Ctrl/⌘ + Shift + C</span> copy CLI');
    }
    return `Keyboard: ${base.join(', ')}.`;
}

function _operatorRecoveryActionSnapshot(context) {
    const job = context.job;
    const bootstrapStatus = String(state.bootstrap?.status || 'pending').trim().toLowerCase();
    const reconnectBlocked = Boolean(job?.reconnectBlocked);
    const hasBootstrapFailure = bootstrapStatus === 'degraded' || bootstrapStatus === 'unavailable';
    if (!reconnectBlocked && !hasBootstrapFailure) return null;

    const failure = reconnectBlocked
        ? _bootstrapFailureDetails('auth_failure', 401)
        : _bootstrapFailureDetails(
            state.bootstrap.lastErrorReason,
            state.bootstrap.lastHttpStatus,
            '',
            state.bootstrap.lastRateLimitHint
        );
    const tone = failure.retryable ? 'warning' : 'blocked';
    const retryAtMs = Number.isFinite(Number(failure.retryAtMs)) && Number(failure.retryAtMs) > 0
        ? Number(failure.retryAtMs)
        : null;

    if (failure.reason === 'auth_failure' || reconnectBlocked) {
        return {
            title: 'Restore access before live actions continue',
            detail: failure.actionMessage,
            tone,
            primary: _operatorAction('restore_access', 'Restore Access', {
                jobId: context.jobId,
                tone,
                detail: failure.actionMessage
            }),
            secondary: _compactOperatorActions([
                _operatorAction('retry_status_check', 'Retry Status Check', {
                    jobId: context.jobId,
                    tone: 'info',
                    detail: 'Retry bootstrap and backend status checks without expanding the route contract.'
                })
            ])
        };
    }

    return {
        title: failure.reason === 'access_outage' ? 'Managed access is degraded' : 'Portal recovery is required',
        detail: failure.actionMessage,
        retryCountdownAtMs: retryAtMs,
        tone,
        primary: _operatorAction('retry_status_check', 'Retry Status Check', {
            jobId: context.jobId,
            tone,
            detail: failure.actionMessage
        }),
        secondary: _compactOperatorActions([
            context.reviewableOutputs
                ? _operatorAction('review_retained_outputs', 'Review Retained Outputs', {
                    jobId: context.jobId,
                    tone: 'warning',
                    detail: 'Open retained outputs while live status is recovering.'
                })
                : _operatorAction('return_to_build', 'Return to Build', {
                    tone: 'info',
                    detail: 'Return to the build surface while access recovery completes.'
                })
        ])
    };
}

function _operatorActionRailSnapshot(jobOverride = undefined) {
    const job = arguments.length > 0 ? jobOverride : _preferredOperatorActionJob();
    const context = _operatorActionContext(job);
    const recoverySnapshot = _operatorRecoveryActionSnapshot(context);
    if (recoverySnapshot) return recoverySnapshot;

    if (!job) {
        return {
            title: state.backendOk ? 'Prepare the next governed dispatch' : 'Restore backend connectivity',
            detail: state.backendOk
                ? 'Open Build to continue the active draft. The last selected run will stay actionable when one exists.'
                : 'Connectivity must recover before preview-backed dispatch and live review can continue.',
            tone: state.backendOk ? 'info' : 'warning',
            primary: _operatorAction('open_build', 'Open Build', {
                tone: 'info',
                detail: 'Open Build without changing any route or API contract.'
            }),
            secondary: _compactOperatorActions([
                _operatorAction('resume_draft', 'Resume Draft', {
                    tone: 'info',
                    detail: 'Resume the current draft and keep the active step focused.'
                }),
                !state.backendOk
                    ? _operatorAction('retry_status_check', 'Retry Status Check', {
                        tone: 'warning',
                        detail: 'Retry bootstrap and backend checks while the portal is offline.'
                    })
                    : null
            ])
        };
    }

    if (job.state === 'running' || job.state === 'queued') {
        return {
            title: context.artifactCount > 0 ? 'Live run already has early outputs' : 'Stay with the live run',
            detail: context.artifactCount > 0
                ? 'Operate stays primary while early artifacts index. Review remains one click away when you need the retained outputs.'
                : 'Use Operate to watch progress, warnings, and transport freshness until the first artifacts arrive.',
            tone: _jobSurfaceTone(job),
            primary: _operatorAction('stay_in_operate', 'Stay in Operate', {
                jobId: context.jobId,
                tone: _jobSurfaceTone(job),
                detail: 'Keep the selected run pinned in Operate.'
            }),
            secondary: _compactOperatorActions([
                context.reviewableOutputs
                    ? _operatorAction('open_early_artifacts', 'Open Early Artifacts', {
                        jobId: context.jobId,
                        tone: 'warning',
                        detail: 'Open Review using the current selected run and artifact route state.'
                    })
                    : null,
                context.heroArtifact
                    ? _operatorAction('open_latest_artifact', 'Open Latest Artifact', {
                        jobId: context.jobId,
                        artifactPath: _artifactRouteKey(context.heroArtifact),
                        tone: 'info',
                        detail: 'Open the highest-ranked indexed artifact for the selected run.'
                    })
                    : null
            ])
        };
    }

    if (job.state === 'partial' || job.state === 'failed' || job.state === 'canceled') {
        return {
            title: context.reviewableOutputs ? 'Retained outputs are ready for triage' : 'Return to Build after triage',
            detail: context.reviewableOutputs
                ? 'Open the retained outputs before rerunning failed inputs or rebuilding the next dispatch.'
                : 'No reviewable outputs were retained. Return to Build after confirming the latest run context.',
            tone: context.reviewableOutputs ? 'warning' : 'blocked',
            primary: context.reviewableOutputs
                ? _operatorAction('review_retained_outputs', 'Review Retained Outputs', {
                    jobId: context.jobId,
                    tone: 'warning',
                    detail: 'Open Review for the retained outputs of the selected run.'
                })
                : _operatorAction('return_to_build', 'Return to Build', {
                    tone: 'info',
                    detail: 'Return to the build surface to prepare the next dispatch.'
                }),
            secondary: _compactOperatorActions([
                _operatorAction('return_to_build', 'Return to Build', {
                    tone: 'info',
                    detail: 'Return to the build surface to prepare the next dispatch.'
                }),
                context.reviewableOutputs && context.heroArtifact
                    ? _operatorAction('open_latest_artifact', 'Open Latest Artifact', {
                        jobId: context.jobId,
                        artifactPath: _artifactRouteKey(context.heroArtifact),
                        tone: 'warning',
                        detail: 'Open the highest-ranked retained artifact without changing the current route.'
                    })
                    : null
            ])
        };
    }

    if (job.state === 'offline') {
        return {
            title: context.reviewableOutputs ? 'Cached outputs remain reviewable' : 'Restore backend connectivity',
            detail: context.reviewableOutputs
                ? 'Live status is stale until connectivity returns, but retained artifacts stay available for operator review.'
                : 'Live status is stale until connectivity returns. Retry the portal status check before trusting this run state.',
            tone: 'warning',
            primary: context.reviewableOutputs
                ? _operatorAction('review_retained_outputs', 'Review Retained Outputs', {
                    jobId: context.jobId,
                    tone: 'warning',
                    detail: 'Review retained outputs while backend connectivity recovers.'
                })
                : _operatorAction('retry_status_check', 'Retry Status Check', {
                    jobId: context.jobId,
                    tone: 'warning',
                    detail: 'Retry bootstrap and backend status checks for the selected run.'
                }),
            secondary: _compactOperatorActions([
                _operatorAction('retry_status_check', 'Retry Status Check', {
                    jobId: context.jobId,
                    tone: 'warning',
                    detail: 'Retry bootstrap and backend status checks for the selected run.'
                }),
                _operatorAction('return_to_build', 'Return to Build', {
                    tone: 'info',
                    detail: 'Return to the build surface while connectivity recovers.'
                })
            ])
        };
    }

    return {
        title: context.reviewableOutputs ? 'Review context is ready' : 'Awaiting indexed outputs',
        detail: context.reviewableOutputs
            ? 'Open Review, pop the latest indexed artifact, or toggle compare without expanding the current route contract.'
            : 'This run is selected, but no indexed outputs are available yet. Stay with the selected job context until they arrive.',
        tone: context.reviewableOutputs ? 'ready' : 'info',
        primary: context.reviewableOutputs
            ? _operatorAction('open_review', 'Open Review', {
                jobId: context.jobId,
                tone: 'ready',
                detail: 'Open Review using the selected run and current route-backed compare preference.'
            })
            : _operatorAction('stay_in_operate', 'Stay in Operate', {
                jobId: context.jobId,
                tone: 'info',
                detail: 'Keep the selected run pinned in Operate until outputs arrive.'
            }),
        secondary: _compactOperatorActions([
            context.heroArtifact
                ? _operatorAction('open_latest_artifact', 'Open Latest Artifact', {
                    jobId: context.jobId,
                    artifactPath: _artifactRouteKey(context.heroArtifact),
                    tone: 'ready',
                    detail: 'Open the highest-ranked indexed artifact for the selected run.'
                })
                : null,
            context.compareCandidate
                ? _operatorAction('toggle_compare', 'Toggle Compare', {
                    jobId: context.jobId,
                    tone: context.compareEnabled ? 'ready' : 'info',
                    detail: 'Toggle compare using the current artifact route and compare=1 contract.'
                })
                : null
        ])
    };
}

function _renderOperatorActionButton(button, action) {
    if (!button) return;
    if (!action) {
        button.classList.add('hidden');
        button.disabled = true;
        button.textContent = '';
        delete button.dataset.actionKey;
        delete button.dataset.jobId;
        delete button.dataset.artifactPath;
        delete button.dataset.tone;
        delete button.dataset.actionLabel;
        button.removeAttribute('title');
        return;
    }
    button.textContent = action.label;
    button.disabled = Boolean(action.disabled);
    button.dataset.actionKey = action.key;
    button.dataset.jobId = action.jobId || '';
    button.dataset.artifactPath = action.artifactPath || '';
    button.dataset.tone = action.tone || 'info';
    button.dataset.actionLabel = action.label;
    if (action.detail) {
        button.title = action.detail;
    } else {
        button.removeAttribute('title');
    }
    button.classList.remove('hidden');
}

function _renderContextualActionRow(container, primaryButton, secondaryButton, primaryAction, secondaryAction = null) {
    if (!container) return;
    const hasActions = Boolean(primaryAction || secondaryAction);
    container.classList.toggle('hidden', !hasActions);
    _renderOperatorActionButton(primaryButton, primaryAction);
    _renderOperatorActionButton(secondaryButton, secondaryAction);
}

// WeakMap keyed by a banner's detail DOM element to its active countdown's
// cancel function. Per-element keying lets each banner surface (operator
// action rail, dispatch readiness reason) track its own countdown so a
// re-render of one cannot cancel the other.
const _activeRetryCountdownCancellers = new WeakMap();

// Render a banner's detail text and, when a rate-limit Retry-After hint is
// present, append a live countdown span built via createElement (never inline
// HTML). The static detail text stays unchanged for screen-reader semantics;
// the countdown carries aria-hidden="true" so it is purely visual. Replacing
// the banner cancels any previously active countdown for this DOM element.
function _renderBannerDetailWithRetryCountdown(detailEl, detailText, retryCountdownAtMs) {
    if (!detailEl) return;
    const priorCancel = _activeRetryCountdownCancellers.get(detailEl);
    if (typeof priorCancel === 'function') {
        try { priorCancel(); } catch (_err) { /* best-effort */ }
    }
    _activeRetryCountdownCancellers.delete(detailEl);
    detailEl.textContent = String(detailText || '');
    const targetMs = Number(retryCountdownAtMs);
    if (!Number.isFinite(targetMs) || targetMs <= Date.now()) return;
    if (typeof document === 'undefined' || !document?.createElement) return;
    detailEl.appendChild(document.createTextNode(' '));
    const span = document.createElement('span');
    span.setAttribute('data-retry-countdown', '');
    span.setAttribute('aria-hidden', 'true');
    span.textContent = portalInternals.formatRetryCountdown(
        Math.ceil((targetMs - Date.now()) / 1000)
    );
    detailEl.appendChild(span);
    _activeRetryCountdownCancellers.set(detailEl, portalInternals.startRetryCountdown({
        retryAtMs: targetMs,
        onTick: (secondsRemaining) => {
            span.textContent = portalInternals.formatRetryCountdown(secondsRemaining);
        },
        onComplete: () => {
            span.textContent = portalInternals.formatRetryCountdown(0);
            _activeRetryCountdownCancellers.delete(detailEl);
        }
    }));
}

function renderOperatorActionRail() {
    if (!els.consoleActionRail) return;
    const visible = ['overview', 'build', 'operate', 'review'].includes(state.currentView);
    els.consoleActionRail.classList.toggle('hidden', !visible);
    if (!visible) {
        // Cancel any active countdown when the rail is hidden so the timer
        // doesn't keep ticking against a detached element.
        if (els.consoleActionRailDetail) {
            _renderBannerDetailWithRetryCountdown(els.consoleActionRailDetail, '', null);
        }
        return;
    }

    const snapshot = _operatorActionRailSnapshot();
    els.consoleActionRail.dataset.tone = snapshot.tone || 'info';
    if (els.consoleActionRailTitle) els.consoleActionRailTitle.textContent = snapshot.title;
    if (els.consoleActionRailDetail) {
        _renderBannerDetailWithRetryCountdown(
            els.consoleActionRailDetail,
            snapshot.detail,
            snapshot.retryCountdownAtMs
        );
    }
    if (els.consoleActionRailHint) {
        els.consoleActionRailHint.innerHTML = _operatorActionHintHtml();
    }
    if (els.consoleActionRailActions) {
        const hasActions = Boolean(snapshot.primary || (Array.isArray(snapshot.secondary) && snapshot.secondary.length > 0));
        els.consoleActionRailActions.classList.toggle('hidden', !hasActions);
    }
    _renderOperatorActionButton(els.consoleActionPrimaryBtn, snapshot.primary);
    _renderOperatorActionButton(els.consoleActionSecondaryBtn1, snapshot.secondary?.[0] || null);
    _renderOperatorActionButton(els.consoleActionSecondaryBtn2, snapshot.secondary?.[1] || null);
}

function renderSelectedJobRecoveryActions(job) {
    const snapshot = _operatorActionRailSnapshot(job);
    _renderContextualActionRow(
        els.selectedJobRecoveryActions,
        els.selectedJobRecoveryPrimaryBtn,
        els.selectedJobRecoverySecondaryBtn,
        snapshot.primary || null,
        snapshot.secondary?.[0] || null
    );
}

function renderReviewStatusActions(job, artifact) {
    const snapshot = _operatorActionRailSnapshot(job);
    _renderContextualActionRow(
        els.reviewStatusActions,
        els.reviewStatusPrimaryBtn,
        els.reviewStatusSecondaryBtn,
        snapshot.primary || null,
        snapshot.secondary?.[0] || null
    );
}

function _openReviewSurfaceForJob(job, surface = 'job_inspector') {
    const normalizedJobId = _normalizeSelectedJobId(job?.id);
    if (!normalizedJobId || !job) {
        createToast('Select a run first, then open its review surface.', 'info');
        return false;
    }
    void emitPortalEvent('run_details_opened', {
        surface,
        metadata: {
            job_id: normalizedJobId,
            pipeline: String(job.pipeline || '')
        }
    });
    navigateConsoleView('review', { jobId: normalizedJobId });
    return true;
}

function _openManagedArtifactWindow(job, artifact, surface = 'artifact_review') {
    const url = sanitizeManagedAssetUrl(buildArtifactUrl(job, artifact));
    if (!url) {
        createToast('No artifact URL is available for this selection.', 'error');
        return false;
    }
    void emitPortalEvent('artifact_opened', {
        surface,
        metadata: {
            job_id: String(job.id || ''),
            media_kind: String(artifact.media_kind || 'file'),
            pipeline: String(job.pipeline || '')
        }
    });
    window.open(url, '_blank', 'noopener,noreferrer');
    return true;
}

function _openArtifactForSelection(job, artifact, surface = 'artifact_review') {
    if (!job || !artifact) {
        createToast('No artifact is available for this selection.', 'info');
        return false;
    }
    if (_artifactViewerEnabled()) {
        const openedInViewer = _openArtifactViewer(job, artifact, document.activeElement, surface);
        if (openedInViewer) {
            return true;
        }
    }
    return _openManagedArtifactWindow(job, artifact, surface);
}

function _toggleCompareSurface(job, surface = 'artifact_review') {
    if (!job) return false;
    const key = _normalizeSelectedJobId(job.id);
    const selectedArtifact = _selectedArtifactForJob(job);
    const compareCandidate = findCompareArtifact(
        selectedArtifact,
        rankArtifactsForDisplay(Array.isArray(job.artifacts) ? job.artifacts : [])
    );
    if (!key || !compareCandidate) {
        createToast('No paired comparison is available for this artifact.', 'info');
        return false;
    }
    _rememberComparePreference(key, !Boolean(state.artifactUi.compareByJob[key]));
    void emitPortalEvent('artifact_compared', {
        surface,
        metadata: {
            enabled: Boolean(state.artifactUi.compareByJob[key]),
            job_id: key,
            pipeline: String(job.pipeline || '')
        }
    });
    renderReviewSurfaces();
    return true;
}

function _retryPortalStatus(job = null) {
    void loadPortalBootstrap();
    void checkBackend(true);
    if (job) {
        void refreshJobStatus(job);
    }
    return true;
}

function handleOperatorActionClick(event) {
    const button = event.target.closest('button[data-action-key]');
    if (!button || button.disabled) return;
    const actionKey = String(button.dataset.actionKey || '').trim();
    if (!actionKey) return;
    const job = _findJobById(button.dataset.jobId || state.selectedJobId || _preferredSelectedJobId());
    const context = _operatorActionContext(job);
    switch (actionKey) {
        case 'open_build':
            navigateConsoleView('build');
            setBuildStep(1, { silent: true });
            if (els.pipelineSelect) els.pipelineSelect.focus();
            break;
        case 'resume_draft':
            navigateConsoleView('build');
            syncBuildStepUi();
            {
                const activeStep = document.querySelector('.build-step-tab.is-active');
                if (activeStep && typeof activeStep.focus === 'function') activeStep.focus();
            }
            break;
        case 'return_to_build':
            navigateConsoleView('build');
            break;
        case 'stay_in_operate':
            if (!job) {
                createToast('No active run is available right now.', 'info');
                return;
            }
            navigateConsoleView('operate', { jobId: context.jobId });
            break;
        case 'open_review':
        case 'review_retained_outputs':
        case 'open_early_artifacts':
            if (state.currentView === 'review' && context.jobId) {
                navigateConsoleView('review', {
                    jobId: context.jobId,
                    artifactPath: _artifactRouteKey(context.selectedArtifact),
                    compareEnabled: context.compareEnabled
                });
            } else {
                _openReviewSurfaceForJob(job, 'action_rail');
            }
            break;
        case 'open_latest_artifact':
            _openArtifactForSelection(job, context.heroArtifact || context.selectedArtifact, 'action_rail');
            break;
        case 'toggle_compare':
            _toggleCompareSurface(job, 'action_rail');
            break;
        case 'retry_status_check':
            _retryPortalStatus(job);
            break;
        case 'restore_access':
            _flushPendingTransientPortalDraftPersist();
            window.location.assign(_managedLoginUrlForCurrentRoute());
            break;
        default:
            break;
    }
}

function _dispatchReadinessSnapshot(payload = null) {
    const currentPayload = payload || generatePayload();
    const readinessStatus = currentPipelineDispatchStatus(currentPayload);

    if (!_portalPrivilegesReady()) {
        return {
            canRun: false,
            tone: 'blocked',
            detail: !_isBootstrapReady()
                ? 'Portal bootstrap is still being confirmed before privileged actions can run.'
                : 'Managed portal access is unavailable, so dispatch remains disabled.',
        };
    }

    if (!state.backendOk) {
        return {
            canRun: false,
            tone: 'blocked',
            detail: DISPATCH_BACKEND_OFFLINE_MESSAGE,
        };
    }

    if (currentPayload.pipeline === 'lux-depth-v3') {
        const preview = _currentPreviewForPayload(currentPayload);
        if (!preview || preview.status === 'loading') {
            return {
                canRun: false,
                tone: 'info',
                detail: 'Preview-backed validation is refreshing. Dispatch unlocks when the current draft settles.',
            };
        }
        if (preview.status === 'error') {
            const previewFailure = _previewFailureDetails(preview);
            // Surface a live countdown only when the failure is a service
            // failure (mapped from 429) AND we have a usable Retry-After hint.
            // Other failure modes (auth, validation) do not carry a cooldown.
            const retryAtMs = previewFailure.reason === 'service_failure'
                && configPreviewLastRateLimitHint
                && Number.isFinite(Number(configPreviewLastRateLimitHint.retryAtMs))
                && Number(configPreviewLastRateLimitHint.retryAtMs) > Date.now()
                ? Number(configPreviewLastRateLimitHint.retryAtMs)
                : null;
            return {
                canRun: false,
                tone: 'blocked',
                detail: String(previewFailure.luxBlockedMessage || 'Preview-backed validation needs attention before dispatch.')
                    .replace(/^(BLOCKED|WARNING):\s*/i, ''),
                retryCountdownAtMs: retryAtMs,
            };
        }
        if (Array.isArray(preview.field_errors) && preview.field_errors.length > 0) {
            const firstError = preview.field_errors[0];
            const conflictError = preview.field_errors.find(
                (item) => String(item?.code || '').trim() === 'conflicting_log_verbosity_flags'
            );
            return {
                canRun: false,
                tone: 'blocked',
                detail: conflictError
                    ? 'verbose and quiet are mutually exclusive; disable one flag before dispatch.'
                    : String(firstError?.message || 'Preview validation blocked dispatch.'),
            };
        }
        if (_effectiveDebugBundleEnabled(preview, currentPayload) && !state.portalUi.debugBundleAcknowledged) {
            return {
                canRun: false,
                tone: 'blocked',
                detail: 'Debug bundle acknowledgement is required before dispatch.',
            };
        }
    }

    if (!readinessStatus) {
        return {
            canRun: false,
            tone: 'info',
            detail: 'Execution readiness is still loading. Dispatch unlocks when readiness finishes.',
        };
    }
    if (readinessStatus !== 'ready') {
        const firstIssue = currentPipelineReadinessIssues(currentPayload)[0];
        return {
            canRun: false,
            tone: String(firstIssue?.severity || '').trim().toLowerCase() === 'blocked' ? 'blocked' : 'warning',
            detail: String(firstIssue?.message || 'Pipeline prerequisites still need operator attention before dispatch.'),
        };
    }

    return {
        canRun: true,
        tone: 'ready',
        detail: currentPayload.pipeline === 'lux-depth-v3'
            ? 'Preview-backed validation, readiness, and acknowledgments are clear for dispatch.'
            : 'Readiness checks are clear for the selected archive stage.',
    };
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

function _summaryTone(value) {
    const tone = String(value || '').trim().toLowerCase();
    return ['ready', 'warning', 'blocked', 'info'].includes(tone) ? tone : 'info';
}

function _setSummaryCard(card, labelElement, valueElement, metaElement, summary) {
    if (!valueElement || !metaElement) return;
    const nextSummary = summary && typeof summary === 'object' ? summary : {};
    if (labelElement) labelElement.textContent = String(nextSummary.label || '');
    valueElement.textContent = String(nextSummary.value || '');
    metaElement.textContent = String(nextSummary.meta || '');
    if (card) {
        card.dataset.tone = _summaryTone(nextSummary.tone);
    }
}

function _jobSurfaceTone(job) {
    const stateLabel = String(job?.state || '').trim().toLowerCase();
    if (['succeeded', 'ready', 'partial'].includes(stateLabel)) return 'ready';
    if (['failed', 'canceled', 'offline'].includes(stateLabel)) return 'blocked';
    return 'info';
}

function _previewSurfaceSummary(payload = null) {
    const currentPayload = payload || generatePayload();
    const preview = _currentPreviewForPayload(currentPayload);
    if (!state.backendOk) {
        return {
            value: 'Backend offline',
            meta: 'Preview-backed validation resumes when the orchestrator backend becomes reachable again.',
            tone: 'blocked'
        };
    }
    if (!preview) {
        return {
            value: 'Refreshing',
            meta: 'Preview-backed validation is hydrating the current draft.',
            tone: 'info'
        };
    }
    if (preview.status === 'loading') {
        return {
            value: 'Refreshing',
            meta: 'Preview-backed validation is recalculating the active draft.',
            tone: 'info'
        };
    }
    if (preview.status === 'error') {
        const details = _previewFailureDetails(preview);
        const message = currentPayload.pipeline === 'lux-depth-v3'
            ? String(details.luxBlockedMessage || '').replace(/^BLOCKED:\s*/i, '')
            : String(details.archiveWarningMessage || '').replace(/^WARNING:\s*/i, '');
        return {
            value: details.summaryLabel,
            meta: message || 'Preview-backed validation needs operator attention.',
            tone: currentPayload.pipeline === 'lux-depth-v3' ? 'blocked' : 'warning'
        };
    }
    if (preview.status === 'ready') {
        const errors = Array.isArray(preview.field_errors) ? preview.field_errors.length : 0;
        const warnings = Array.isArray(preview.field_warnings) ? preview.field_warnings.length : 0;
        if (errors > 0) {
            return {
                value: 'Preview invalid',
                meta: `${errors} blocking issue${errors === 1 ? '' : 's'} need resolution before dispatch.`,
                tone: 'blocked'
            };
        }
        if (warnings > 0) {
            return {
                value: 'Preview ready with warnings',
                meta: `${warnings} warning${warnings === 1 ? '' : 's'} remain visible before dispatch.`,
                tone: 'warning'
            };
        }
        return {
            value: 'Preview ready',
            meta: 'Preview-backed validation and normalized arguments are aligned with the current draft.',
            tone: 'ready'
        };
    }
    if (preview.status === 'offline' || preview.status === 'local_fallback') {
        return {
            value: 'Local fallback',
            meta: 'Local rendering is standing in while preview-backed validation is unavailable.',
            tone: 'warning'
        };
    }
    return {
        value: _formatPreviewStateLabel(currentPayload),
        meta: 'Preview-backed validation will summarize the active draft here.',
        tone: 'info'
    };
}

function _metadataBackendEntry(name) {
    const catalog = state.metadata?.backend_catalog;
    const key = String(name || '').trim().toLowerCase();
    if (!key || !catalog || typeof catalog !== 'object') return null;
    const entry = catalog[key];
    return entry && typeof entry === 'object' ? entry : null;
}

function _runtimeBriefingPolicyTone(entry) {
    const code = String(entry?.policy_posture?.code || '').trim().toLowerCase();
    if (code === 'governed_default') return 'ready';
    if (code === 'managed_optional' || code === 'deterministic_fallback') return 'info';
    if (code === 'research_only' || code === 'experimental_segmentation') return 'warning';
    return 'info';
}

function _runtimeBriefingArgs(payload = null) {
    const currentPayload = payload || generatePayload();
    const preview = _currentPreviewForPayload(currentPayload);
    const normalizedArgs = preview?.normalized_args && typeof preview.normalized_args === 'object'
        ? preview.normalized_args
        : null;
    return {
        payload: currentPayload,
        preview,
        args: normalizedArgs ? { ...(currentPayload.args || {}), ...normalizedArgs } : (currentPayload.args || {})
    };
}

function _runtimeAcknowledgmentSnapshot(entry, args = {}) {
    const required = Array.isArray(entry?.required_acknowledgments) ? entry.required_acknowledgments : [];
    const pending = [];
    const completed = [];
    required.forEach((item) => {
        const field = String(item?.field || '').trim();
        if (!field) return;
        const label = String(item?.label || titleCaseToken(field, field)).trim();
        if (parseBoolLike(args[field], false)) {
            completed.push(label);
            return;
        }
        pending.push(label);
    });
    return { pending, completed };
}

function _runtimeBriefingCardSummary(summary = {}) {
    return {
        label: String(summary.label || 'Runtime summary'),
        value: String(summary.value || 'Unavailable'),
        detail: String(summary.detail || ''),
        meta: String(summary.meta || ''),
        tone: String(summary.tone || 'info')
    };
}

function _runtimeDepthBackendSummary(args = {}) {
    const backendName = _resolveDepthBackend(args.depth_backend || state.config.depthBackend || 'da3');
    const entry = _metadataBackendEntry(backendName);
    const acknowledgments = _runtimeAcknowledgmentSnapshot(entry, args);
    const meta = [
        entry?.model_provider_label,
        entry?.model_display_label,
        acknowledgments.pending.length > 0
            ? `Pending: ${acknowledgments.pending.join(', ')}`
            : entry?.policy_posture?.label
    ].filter(Boolean).join(' • ');
    return _runtimeBriefingCardSummary({
        label: 'Depth backend',
        value: entry?.label || titleCaseToken(backendName, 'Unknown'),
        detail: entry?.operator_summary || 'Backend-owned metadata is unavailable, so keep using readiness and preview as the source of truth.',
        meta: meta || 'Backend-owned policy',
        tone: acknowledgments.pending.length > 0 ? 'blocked' : _runtimeBriefingPolicyTone(entry)
    });
}

function _runtimeSegmentationBackendSummary(args = {}) {
    const segmentationEnabled = parseBoolLike(args.enable_segmentation, false);
    if (!segmentationEnabled) {
        return _runtimeBriefingCardSummary({
            label: 'Segmentation backend',
            value: 'Segmentation off',
            detail: 'This draft stays on the deterministic path until segmentation is enabled.',
            meta: 'Enable segmentation to choose EfficientSAM or SAM2.',
            tone: 'info'
        });
    }

    const backendName = _resolveSegmentationBackend(args.segmentation_backend || state.config.segmentation?.backend || 'stub');
    const entry = _metadataBackendEntry(backendName);
    const modelSize = backendName === 'sam2'
        ? _resolveSam2ModelSize(args.sam2_model_size || state.config.segmentation?.sam2ModelSize || 'base')
        : '';
    const meta = [
        entry?.model_provider_label,
        entry?.model_display_label,
        modelSize ? `Model ${_titleizeEstimateToken(modelSize)}` : '',
        parseBoolLike(args.strict_segmentation, false) ? 'Strict masks on' : 'Strict masks off'
    ].filter(Boolean).join(' • ');
    return _runtimeBriefingCardSummary({
        label: 'Segmentation backend',
        value: entry?.label || titleCaseToken(backendName, 'Unknown'),
        detail: entry?.operator_summary || 'Segmentation metadata is unavailable, so keep the configured backend visible in the draft.',
        meta: meta || 'Backend-owned policy',
        tone: _runtimeBriefingPolicyTone(entry)
    });
}

function _runtimeStateSummary(payload = null) {
    const { preview } = _runtimeBriefingArgs(payload);
    const readiness = currentPipelineReadiness(payload);
    const issues = currentPipelineReadinessIssues(payload);
    const dispatchStatus = titleCaseToken(currentPipelineDispatchStatus(payload) || readiness?.status || 'unknown', 'Unknown');

    if (!state.backendOk) {
        return _runtimeBriefingCardSummary({
            label: 'Canary/runtime state',
            value: 'Backend offline',
            detail: 'Live readiness, canary posture, and recent-run recovery resume when backend connectivity returns.',
            meta: 'Preview-backed dispatch remains paused while the orchestrator backend is offline.',
            tone: 'blocked'
        });
    }

    if (preview?.status === 'loading' && !readiness) {
        return _runtimeBriefingCardSummary({
            label: 'Canary/runtime state',
            value: 'Refreshing',
            detail: 'Preview-backed readiness is recalculating the active draft.',
            meta: 'Canary posture appears here when backend readiness returns.',
            tone: 'info'
        });
    }

    if (!readiness || typeof readiness !== 'object') {
        return _runtimeBriefingCardSummary({
            label: 'Canary/runtime state',
            value: 'Readiness loading',
            detail: 'Base readiness and canary posture are still hydrating from the backend.',
            meta: 'Dispatch state becomes authoritative when readiness finishes.',
            tone: 'info'
        });
    }

    const canaryLabel = `Canary ${titleCaseToken(readiness.canary_status || 'unknown', 'Unknown')}`;
    const firstIssue = issues[0];
    if (firstIssue) {
        return _runtimeBriefingCardSummary({
            label: 'Canary/runtime state',
            value: `${dispatchStatus} • ${canaryLabel}`,
            detail: String(firstIssue.message || 'Pipeline readiness reported an operator-facing prerequisite.'),
            meta: 'Resolve the current readiness issue before dispatching this draft.',
            tone: String(firstIssue.severity || '').trim().toLowerCase() === 'blocked' ? 'blocked' : 'warning'
        });
    }

    const notes = Array.isArray(readiness.notes)
        ? readiness.notes.map((item) => String(item || '').trim()).filter(Boolean)
        : [];
    return _runtimeBriefingCardSummary({
        label: 'Canary/runtime state',
        value: canaryLabel,
        detail: notes[notes.length - 1] || notes[0] || 'Base readiness and canary posture are aligned with the current draft.',
        meta: `Dispatch ${dispatchStatus}`,
        tone: readiness.canary_status === 'ready'
            ? 'ready'
            : readiness.canary_status === 'degraded' || readiness.canary_status === 'unavailable'
                ? 'warning'
                : 'info'
    });
}

function _runtimeCheckpointExpectationSummary(args = {}) {
    const depthEntry = _metadataBackendEntry(_resolveDepthBackend(args.depth_backend || state.config.depthBackend || 'da3'));
    const entries = [{ entry: depthEntry }];
    if (parseBoolLike(args.enable_segmentation, false)) {
        entries.push({
            entry: _metadataBackendEntry(
                _resolveSegmentationBackend(args.segmentation_backend || state.config.segmentation?.backend || 'stub')
            )
        });
    }

    const statements = [];
    const details = [];
    let hasRequiredFieldMissing = false;
    let hasRuntimeManagedRequirement = false;
    let hasOptionalPath = false;
    let hasOptionalUnset = false;
    let hasNoPathRequirement = false;

    entries.forEach(({ entry }) => {
        if (!entry) return;
        const expectation = entry.checkpoint_expectation && typeof entry.checkpoint_expectation === 'object'
            ? entry.checkpoint_expectation
            : null;
        if (!expectation) return;
        const entryLabel = String(entry.label || 'Backend').trim();
        const field = String(expectation.field || '').trim();
        const providedValue = field ? String(args[field] || '').trim() : '';

        if (expectation.required && field && !providedValue) {
            hasRequiredFieldMissing = true;
            statements.push(`${entryLabel}: required checkpoint path missing.`);
        } else if (expectation.required && field && providedValue) {
            statements.push(`${entryLabel}: required checkpoint path supplied.`);
        } else if (expectation.required) {
            hasRuntimeManagedRequirement = true;
            statements.push(`${entryLabel}: runtime-managed checkpoint required.`);
        } else if (field && providedValue) {
            hasOptionalPath = true;
            statements.push(`${entryLabel}: optional checkpoint path supplied.`);
        } else if (field) {
            hasOptionalUnset = true;
            statements.push(`${entryLabel}: optional checkpoint path not set.`);
        } else {
            hasNoPathRequirement = true;
            statements.push(`${entryLabel}: no explicit checkpoint path required.`);
        }

        const detail = String(expectation.detail || '').trim();
        if (detail) details.push(detail);
    });

    let value = 'No explicit path required';
    let tone = 'ready';
    if (hasRequiredFieldMissing) {
        value = 'Checkpoint path missing';
        tone = 'blocked';
    } else if (hasRuntimeManagedRequirement) {
        value = 'Runtime checkpoint required';
        tone = 'warning';
    } else if (hasOptionalPath) {
        value = 'Checkpoint path supplied';
        tone = 'ready';
    } else if (hasOptionalUnset) {
        value = 'Checkpoint path optional';
        tone = 'info';
    } else if (hasNoPathRequirement) {
        value = 'No explicit path required';
        tone = 'ready';
    }

    return _runtimeBriefingCardSummary({
        label: 'Checkpoint expectation',
        value,
        detail: statements.join(' ') || 'Checkpoint expectations load from backend metadata.',
        meta: details.join(' ') || 'Backend-owned runtime expectation',
        tone
    });
}

function _runtimeLicensePostureSummary(args = {}) {
    const selectedEntries = [
        _metadataBackendEntry(_resolveDepthBackend(args.depth_backend || state.config.depthBackend || 'da3'))
    ];
    if (parseBoolLike(args.enable_segmentation, false)) {
        selectedEntries.push(
            _metadataBackendEntry(
                _resolveSegmentationBackend(args.segmentation_backend || state.config.segmentation?.backend || 'stub')
            )
        );
    }

    const postures = [];
    const pending = new Set();
    const completed = new Set();
    const details = [];
    let tone = 'ready';

    selectedEntries.filter(Boolean).forEach((entry) => {
        const acknowledgments = _runtimeAcknowledgmentSnapshot(entry, args);
        const postureLabel = String(entry?.policy_posture?.label || entry?.label || 'Policy').trim();
        postures.push(`${String(entry?.label || 'Backend').trim()}: ${postureLabel}`);
        acknowledgments.pending.forEach((label) => pending.add(label));
        acknowledgments.completed.forEach((label) => completed.add(label));
        const detail = String(entry?.policy_posture?.detail || entry?.operator_summary || '').trim();
        if (detail) details.push(detail);
        if (acknowledgments.pending.length > 0) {
            tone = 'blocked';
        } else if (tone !== 'blocked') {
            const entryTone = _runtimeBriefingPolicyTone(entry);
            tone = entryTone === 'ready' ? tone : entryTone;
        }
    });

    const meta = [];
    if (pending.size > 0) {
        meta.push(`Pending: ${Array.from(pending).join(', ')}`);
    } else if (completed.size > 0) {
        meta.push(`Acknowledged: ${Array.from(completed).join(', ')}`);
    } else {
        meta.push('No backend acknowledgments are currently required.');
    }
    if (parseBoolLike(args.enable_reconstruction, false)) {
        meta.push('Scene reconstruction governance stays in the primary governance panel.');
    }

    return _runtimeBriefingCardSummary({
        label: 'License posture',
        value: pending.size > 0 ? 'Acknowledgment required' : (postures.join(' • ') || 'Backend-owned policy'),
        detail: details.join(' ') || 'Backend-owned policy remains the operator source of truth for selected backends.',
        meta: meta.join(' • '),
        tone
    });
}

function _renderRuntimeBriefingCard(summary = {}) {
    const card = document.createElement('article');
    card.className = 'runtime-briefing-card';
    card.dataset.tone = String(summary.tone || 'info');

    const label = document.createElement('p');
    label.className = 'runtime-briefing-label';
    label.textContent = String(summary.label || 'Runtime summary');

    const value = document.createElement('p');
    value.className = 'runtime-briefing-value';
    value.textContent = String(summary.value || 'Unavailable');

    const detail = document.createElement('p');
    detail.className = 'runtime-briefing-detail';
    detail.textContent = String(summary.detail || '');

    const meta = document.createElement('p');
    meta.className = 'runtime-briefing-meta';
    meta.textContent = String(summary.meta || '');

    card.appendChild(label);
    card.appendChild(value);
    if (detail.textContent) card.appendChild(detail);
    if (meta.textContent) card.appendChild(meta);
    return card;
}

function renderRuntimeBriefing(payload = null) {
    const currentPayload = payload || generatePayload();
    const shells = [els.overviewRuntimeClarityShell, els.buildRuntimeClarityShell].filter(Boolean);
    const containers = [els.overviewRuntimeBriefing, els.buildRuntimeBriefing].filter(Boolean);
    const isLuxPipeline = String(currentPayload.pipeline || '').trim() === 'lux-depth-v3';

    shells.forEach((shell) => {
        shell.classList.toggle('hidden', !isLuxPipeline);
    });
    if (!isLuxPipeline) {
        containers.forEach((container) => {
            container.innerHTML = '';
        });
        return;
    }

    const { args } = _runtimeBriefingArgs(currentPayload);
    const summaries = [
        _runtimeDepthBackendSummary(args),
        _runtimeSegmentationBackendSummary(args),
        _runtimeStateSummary(currentPayload),
        _runtimeCheckpointExpectationSummary(args),
        _runtimeLicensePostureSummary(args)
    ];

    containers.forEach((container) => {
        container.innerHTML = '';
        const fragment = document.createDocumentFragment();
        summaries.forEach((summary) => {
            fragment.appendChild(_renderRuntimeBriefingCard(summary));
        });
        container.appendChild(fragment);
    });
}

function renderBuildStepPulse(payload = null) {
    const currentPayload = payload || generatePayload();
    const activeStep = resolveBuildStep(state.portalUi.buildStep);
    const stepContent = _currentBuildStepContent();
    const activeStepContent = stepContent[activeStep - 1] || BUILD_STEP_CONTENT.lux[activeStep - 1];
    const nextAction = _effectiveNextBestAction(currentPayload);
    const previewSummary = _previewSurfaceSummary(currentPayload);
    const draftValue = state.pipeline === 'lux-depth-v3'
        ? String(currentPayload?.args?.preset || state.config.preset || 'custom')
        : canonicalArchiveCommand(state.pipeline) || 'archive';
    const draftMeta = state.pipeline === 'lux-depth-v3'
        ? `${String(state.pipeline || 'lux-depth-v3')} • ${titleCaseToken(currentPayload?.args?.quality_tier || state.config.qualityTier || 'premium', 'Premium')} posture`
        : `${String(state.pipeline || 'archive')} • deterministic archive stage`;

    _setSummaryCard(els.buildPulseDraftCard, null, els.buildPulseDraft, els.buildPulseDraftMeta, {
        value: draftValue,
        meta: draftMeta,
        tone: state.backendOk ? 'ready' : 'info'
    });
    _setSummaryCard(els.buildPulseStepCard, null, els.buildPulseStep, els.buildPulseStepMeta, {
        value: `${activeStep} of 4 · ${String(activeStepContent?.label || 'Focus')}`,
        meta: String(activeStepContent?.summary || activeStepContent?.meta || 'Build focus is pinned here.'),
        tone: 'info'
    });
    _setSummaryCard(els.buildPulsePreviewCard, null, els.buildPulsePreview, els.buildPulsePreviewMeta, previewSummary);
    _setSummaryCard(els.buildPulseDispatchCard, null, els.buildPulseDispatch, els.buildPulseDispatchMeta, {
        value: String(nextAction?.label || 'Review dispatch posture'),
        meta: String(nextAction?.detail || 'The next operator action stays pinned here while the draft changes.'),
        tone: String(nextAction?.tone || 'info')
    });
}

function renderConsoleContextRibbon() {
    if (!els.consoleContextRibbon) {
        renderOperatorActionRail();
        return;
    }
    const ribbonVisible = ['overview', 'build', 'operate', 'review'].includes(state.currentView);
    els.consoleContextRibbon.classList.toggle('hidden', !ribbonVisible);
    if (!ribbonVisible) {
        renderOperatorActionRail();
        return;
    }

    if (state.currentView === 'operate' || state.currentView === 'review') {
        const selected = state.jobs.find((job) => job.id === state.selectedJobId) || null;
        const artifacts = Array.isArray(selected?.artifacts) ? rankArtifactsForDisplay(selected.artifacts) : [];
        const selectedArtifact = selected ? _selectedArtifactForJob(selected) : null;
        const compareCandidate = selected ? findCompareArtifact(selectedArtifact, artifacts) : null;
        const compareEnabled = Boolean(selected && compareCandidate && state.artifactUi.compareByJob[String(selected.id || '')]);
        const artifactCount = artifacts.length;
        const compareCopy = _compareSurfaceCopy(selectedArtifact, compareCandidate, compareEnabled);

        _setSummaryCard(els.contextRibbonCard1, els.contextRibbonCard1Label, els.contextRibbonJob, els.contextRibbonJobMeta, {
            label: 'Job',
            value: selected ? String(selected.id || 'unknown') : 'No job selected',
            meta: selected
                ? `${String(selected.pipeline || 'unknown')} • ${artifactCount} artifact${artifactCount === 1 ? '' : 's'} indexed`
                : 'Choose a run in operate or review to pin context here.',
            tone: selected ? _jobSurfaceTone(selected) : 'info'
        });
        _setSummaryCard(els.contextRibbonCard2, els.contextRibbonCard2Label, els.contextRibbonState, els.contextRibbonFreshness, {
            label: 'State',
            value: selected ? titleCaseToken(selected.state, 'Unknown') : 'Idle',
            meta: _jobFreshnessLabel(selected),
            tone: selected ? _jobSurfaceTone(selected) : 'info'
        });
        _setSummaryCard(els.contextRibbonCard3, els.contextRibbonCard3Label, els.contextRibbonArtifact, els.contextRibbonArtifactMeta, {
            label: 'Artifact',
            value: selectedArtifact ? artifactLabel(selectedArtifact) : 'Awaiting selection',
            meta: selectedArtifact
                ? `${artifactDisplayLabel(selectedArtifact)}${compareCandidate ? ' • paired comparison available' : ' • single artifact context'}`
                : 'Review context will show the active artifact path here.',
            tone: selectedArtifact ? 'ready' : 'info'
        });
        _setSummaryCard(els.contextRibbonCard4, els.contextRibbonCard4Label, els.contextRibbonCompare, els.contextRibbonCompareMeta, {
            label: 'Compare',
            value: selected ? compareCopy.ribbonValue : 'No compare pair',
            meta: selected ? compareCopy.ribbonMeta : 'Deep-linkable review context stays aligned with the URL.',
            tone: selected && compareEnabled ? 'ready' : 'info'
        });
        renderOperatorActionRail();
        return;
    }

    const currentPayload = generatePayload();
    const activeJob = _latestActiveJob();
    const reviewJob = _latestReviewableJob();
    const activeJobTone = activeJob ? _jobSurfaceTone(activeJob) : (state.backendOk ? 'info' : 'blocked');
    const reviewJobTone = reviewJob ? _jobSurfaceTone(reviewJob) : 'info';
    const nextAction = _effectiveNextBestAction(currentPayload);
    const stepContent = _currentBuildStepContent();
    const activeStep = resolveBuildStep(state.portalUi.buildStep);
    const activeStepContent = stepContent[activeStep - 1] || BUILD_STEP_CONTENT.lux[activeStep - 1];
    const draftValue = state.pipeline === 'lux-depth-v3'
        ? String(currentPayload?.args?.preset || state.config.preset || 'custom')
        : canonicalArchiveCommand(state.pipeline) || 'archive';

    _setSummaryCard(els.contextRibbonCard1, els.contextRibbonCard1Label, els.contextRibbonJob, els.contextRibbonJobMeta, {
        label: 'Live lane',
        value: activeJob
            ? `${titleCaseToken(activeJob.state, 'Unknown')} • ${Math.max(0, Math.min(100, Number(activeJob.progress) || 0))}%`
            : state.backendOk
                ? 'No live run'
                : 'Backend offline',
        meta: activeJob
            ? `${String(activeJob.pipeline || 'unknown')} • ${_jobFreshnessLabel(activeJob)}`
            : state.backendOk
                ? 'Dispatch from Build to watch live progress here.'
                : 'Restore the orchestrator connection to unlock preview-backed dispatch.',
        tone: activeJobTone
    });
    _setSummaryCard(els.contextRibbonCard2, els.contextRibbonCard2Label, els.contextRibbonState, els.contextRibbonFreshness, {
        label: 'Review lane',
        value: reviewJob
            ? (jobOutcomeSummary(reviewJob) || 'Review available')
            : 'No review target',
        meta: reviewJob
            ? `${Array.isArray(reviewJob.artifacts) ? reviewJob.artifacts.length : 0} artifact${Array.isArray(reviewJob.artifacts) && reviewJob.artifacts.length === 1 ? '' : 's'} indexed • open Review to inspect the latest output.`
            : 'Completed or partial outputs will appear here when the current draft finishes.',
        tone: reviewJobTone
    });
    _setSummaryCard(els.contextRibbonCard3, els.contextRibbonCard3Label, els.contextRibbonArtifact, els.contextRibbonArtifactMeta, {
        label: 'Dispatch lane',
        value: String(nextAction?.label || 'Review dispatch posture'),
        meta: String(nextAction?.detail || 'Preview guidance will summarize the clearest next step for this draft.'),
        tone: String(nextAction?.tone || 'info')
    });
    _setSummaryCard(els.contextRibbonCard4, els.contextRibbonCard4Label, els.contextRibbonCompare, els.contextRibbonCompareMeta, {
        label: state.currentView === 'build' ? 'Current focus' : 'Draft',
        value: state.currentView === 'build'
            ? `${activeStep} of 4 · ${String(activeStepContent?.label || 'Focus')}`
            : draftValue,
        meta: state.currentView === 'build'
            ? String(activeStepContent?.summary || activeStepContent?.meta || 'Build focus is pinned here.')
            : `${String(state.pipeline || 'lux-depth-v3')} • ${state.backendOk ? 'live backend connected' : 'dispatch paused until the backend recovers'}.`,
        tone: 'info'
    });
    renderOperatorActionRail();
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
        } else if (state.currentView === 'review') {
            _rememberArtifactSelection(explicitJobId, '');
        }
        if (hasCompareOption) {
            _rememberComparePreference(explicitJobId, options.compareEnabled);
        }
    } else if (state.currentView === 'operate' || state.currentView === 'review') {
        const preferredJobId = _preferredSelectedJobId();
        if (preferredJobId) {
            state.selectedJobId = preferredJobId;
            _rememberSelectedJob(preferredJobId);
            if (state.currentView === 'review' && !hasArtifactOption) {
                _rememberArtifactSelection(preferredJobId, '');
            }
        }
    } else if (state.selectedJobId) {
        _rememberSelectedJob(state.selectedJobId);
    }
    _primeDeferredReviewSurface('route');
    _primeDeferredOperateSurface();
    _primeDeferredBuildSurface();
    applyConsoleViewLayout();
    _syncConsoleRoute(replace);
    renderJobQueue();
}

function applyConsoleRouteFromLocation(replace = false) {
    const routeState = portalRoute.read({
        resolveView: resolveConsoleView,
        normalizeSelectedJobId: _normalizeSelectedJobId,
        normalizeArtifactRoutePath: _normalizeArtifactRoutePath,
        normalizeCompareQueryValue: _normalizeCompareQueryValue
    });
    state.currentView = routeState.view;
    const routeJobId = routeState.jobId;
    const routeArtifactPath = routeState.artifactPath;
    const routeCompareEnabled = routeState.compareEnabled;
    const routeHasArtifact = routeState.hasArtifact;
    const routeHasCompare = routeState.hasCompare;
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
    _primeDeferredReviewSurface('route');
    _primeDeferredOperateSurface();
    _primeDeferredBuildSurface();
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

// ============================================================================
// 6. BUILD STEPPER
// ============================================================================

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
            summary: 'Keep deliverables, the posture band, and immediate readiness readable before opening contextual controls.'
        },
        {
            label: 'Dispatch',
            meta: 'Primary review, launch, and parity tools.',
            title: '4. Review and dispatch',
            summary: 'Use the primary dispatch lane first, then open evidence and CLI parity only when needed.'
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

    renderBuildStepPulse(generatePayload());
}

function setBuildStep(nextStep, options) {
    const settings = options && typeof options === 'object' ? options : {};
    const previous = resolveBuildStep(state.portalUi.buildStep);
    const resolved = resolveBuildStep(nextStep);
    state.portalUi.buildStep = resolved;
    syncBuildStepUi();
    _persistTransientPortalDraft();
    if (!settings.silent && resolved > previous) {
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

// ============================================================================
// 7. UTILITIES
// ============================================================================

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

function _displayJobState(job) {
    if (!job) return 'idle';
    const rawState = String(job.state || '').trim().toLowerCase();
    const artifactCount = Array.isArray(job.artifacts) ? job.artifacts.length : 0;
    const reviewableOutputs = _jobHasReviewableOutputs(job);
    if (rawState === 'queued') return 'queued';
    if (rawState === 'running' && artifactCount > 0) return 'indexing';
    if (rawState === 'running') return 'running';
    // Terminal runs stay reviewable once outputs are retained, even when the active artifact is metadata-only.
    if ((rawState === 'succeeded' || rawState === 'ready') && reviewableOutputs) return 'reviewable';
    if (rawState === 'partial') return 'partial-failure';
    if (rawState === 'failed' || rawState === 'canceled') return 'failed';
    return rawState || 'idle';
}

function _displayJobStateTone(job) {
    const displayState = _displayJobState(job);
    if (displayState === 'reviewable') return 'ready';
    if (displayState === 'running' || displayState === 'queued' || displayState === 'indexing' || displayState === 'partial-failure') {
        return 'running';
    }
    if (displayState === 'failed' || displayState === 'offline') return 'offline';
    return 'ready';
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

function _syncSwitchStateLabels() {
    if (typeof document === 'undefined') return;
    Array.from(document.querySelectorAll('input[role="switch"]')).forEach((input) => {
        const label = input.closest('label');
        if (!label) return;
        const toggleWrap = input.parentElement;
        if (!toggleWrap) return;
        let controlsWrap = label.querySelector('[data-switch-controls-wrap="true"]');
        if (!controlsWrap && toggleWrap.parentElement === label) {
            controlsWrap = document.createElement('span');
            controlsWrap.dataset.switchControlsWrap = 'true';
            controlsWrap.className = 'ml-3 inline-flex items-center gap-3';
            label.insertBefore(controlsWrap, toggleWrap);
            controlsWrap.appendChild(toggleWrap);
        } else if (!controlsWrap && toggleWrap.parentElement instanceof HTMLElement) {
            controlsWrap = toggleWrap.parentElement;
            controlsWrap.dataset.switchControlsWrap = 'true';
            controlsWrap.classList.add('ml-3', 'inline-flex', 'items-center', 'gap-3');
        }
        let stateLabel = label.querySelector('[data-switch-state-label="true"]');
        if (!stateLabel) {
            stateLabel = document.createElement('span');
            stateLabel.dataset.switchStateLabel = 'true';
            stateLabel.className = 'inline-flex min-w-[3rem] items-center justify-center rounded-full border px-2 py-1 text-[10px] font-bold uppercase tracking-[0.16em]';
            if (controlsWrap) {
                controlsWrap.insertBefore(stateLabel, controlsWrap.firstChild);
            } else {
                label.appendChild(stateLabel);
            }
        } else if (controlsWrap && stateLabel.parentElement !== controlsWrap) {
            controlsWrap.insertBefore(stateLabel, controlsWrap.firstChild);
        }
        const checked = Boolean(input.checked);
        stateLabel.textContent = checked ? 'On' : 'Off';
        stateLabel.className = checked
            ? 'inline-flex min-w-[3rem] items-center justify-center rounded-full border border-emerald-300 bg-emerald-50 px-2 py-1 text-[10px] font-bold uppercase tracking-[0.16em] text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/30 dark:text-emerald-300'
            : 'inline-flex min-w-[3rem] items-center justify-center rounded-full border border-slate-300 bg-slate-100 px-2 py-1 text-[10px] font-bold uppercase tracking-[0.16em] text-slate-600 dark:border-slate-700 dark:bg-slate-900/80 dark:text-slate-300';
        input.setAttribute('aria-checked', checked ? 'true' : 'false');
        const baseLabel = String(input.getAttribute('aria-label') || input.id || 'toggle').replace(/\s+\((on|off)\)$/i, '');
        input.setAttribute('aria-label', `${baseLabel} (${checked ? 'on' : 'off'})`);
    });
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

// ============================================================================
// 8. ARTIFACT HELPERS
// ============================================================================

function artifactMediaKind(artifact) {
    return String(artifact?.media_kind || artifact?.artifact_type || 'file').trim().toLowerCase();
}

function artifactContentType(artifact) {
    return String(artifact?.content_type || '').trim();
}

function artifactIsPreviewable(artifact) {
    return Boolean(artifact?.previewable) && artifactMediaKind(artifact) === 'image';
}

// Strict subset of `artifactIsPreviewable`: artifacts the browser can render
// directly via <img> (PNG/JPEG/WebP/GIF/AVIF/SVG) or that have a sibling PNG
// proxy (preview_url). TIFF/EXR without a proxy are excluded so the portal
// never asks the browser to decode a format it cannot, avoiding the failed-
// load → retry loop that produced the 503/.tif churn in the diagnosis logs.
function artifactIsBrowserPreviewable(artifact) {
    if (!artifact) return false;
    if (Boolean(artifact.browser_previewable)) return true;
    if (typeof artifact.preview_url === 'string' && artifact.preview_url.trim()) return true;
    return false;
}

// Resolve the URL that should be used as <img src> for inline preview. Returns
// the empty string when the artifact is not browser-previewable so callers can
// skip rendering an <img> entirely.
function artifactPreviewSrc(job, artifact) {
    if (!artifactIsBrowserPreviewable(artifact)) return '';
    const previewUrl = String(artifact?.preview_url || '').trim();
    if (previewUrl) {
        return sanitizeManagedAssetUrl(previewUrl);
    }
    return buildArtifactUrl(job, artifact);
}

// Page-lifetime cache of artifact URLs that previously returned 404. The viewer
// uses this to skip retries for artifacts that have already been confirmed
// missing, preventing the repeated-fetch loop seen in the diagnosis logs.
const _artifactNotFoundUrls = new Set();

function _markArtifactUrlNotFound(url) {
    if (typeof url !== 'string' || !url) return;
    _artifactNotFoundUrls.add(url);
}

function _clearArtifactUrlNotFoundCache() {
    _artifactNotFoundUrls.clear();
}

function _isArtifactUrlKnownMissing(url) {
    if (typeof url !== 'string' || !url) return false;
    return _artifactNotFoundUrls.has(url);
}

function artifactLabel(artifact) {
    return String(artifact?.relative_path || artifact?.path || 'artifact').trim();
}

function _artifactRouteKey(artifact) {
    if (!artifact || typeof artifact !== 'object') return '';
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
        return portalInternals.finalizeManagedAssetUrl(parsed);
    } catch (_err) {
        return '';
    }
}

function buildArtifactUrl(job, artifact) {
    return portalInternals.buildManagedArtifactUrl({
        job,
        artifact,
        apiBase: API_BASE,
        artifactLabel,
        sanitizeManagedAssetUrl
    });
}

function artifactFingerprint(artifact) {
    return String(artifact?.sha256 || '').trim();
}

function _artifactFingerprintLabel(artifact) {
    return artifactFingerprint(artifact) || 'Not reported';
}

function _artifactViewerEnabled() {
    return Boolean(state.auth?.features?.artifactViewerModal);
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
    if (!primaryArtifact || !artifactIsBrowserPreviewable(primaryArtifact)) return null;
    const primaryGroup = artifactCompareGroup(primaryArtifact);
    if (primaryGroup) {
        const hintedCandidate = rankArtifactsForDisplay(
            artifacts.filter((candidate) => (
                candidate
                && candidate.path !== primaryArtifact.path
                && artifactIsBrowserPreviewable(candidate)
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
        if (!candidate || candidate.path === primaryArtifact.path || !artifactIsBrowserPreviewable(candidate)) return;
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
    const api = _deferredOperateSurfaceApi();
    if (api?.setInspectorTab) {
        api.setInspectorTab(tabName);
        return;
    }
    state.inspectorTab = ['overview', 'timeline', 'logs'].includes(tabName) ? tabName : 'overview';
    if (!_shouldLoadDeferredOperateSurface()) return;
    void _loadDeferredOperateSurface().then((loaded) => {
        if (loaded?.setInspectorTab) loaded.setInspectorTab(state.inspectorTab);
    });
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
    if (String(state.config.preset || '').trim() === 'custom') {
        return {
            name: 'custom',
            label: 'custom',
            stability: 'custom',
            description: 'Manual configuration mode. Current draft values are preserved and curated presets stop backfilling the Lux contract.',
            is_research: false,
            recommended_args: {},
            advanced_sections: ['advanced']
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

function _fastVlmCaptioningFeatureEnabled() {
    return Boolean(
        _isBootstrapReady()
        && state.auth?.features?.fastVlmCaptioning
    );
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
    const previewReadiness = _currentPreviewReadiness(payload);
    const readiness = currentPipelineReadiness(payload);
    const rawStatus = String(readiness?.status || '').trim().toLowerCase();
    if (!rawStatus) return '';
    if (previewReadiness && rawStatus === 'blocked') return 'blocked';
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
    if (parseBoolLike(args.emit_run_card, false) && parseBoolLike(args.run_card_include_proofs, false)) {
        chips.push('Run card proofs');
    }
    if (parseBoolLike(args.vlm_captioning_enabled, false)) {
        chips.push('FastVLM advisory captions');
    }

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
    const sam2TilingEnabled = parseBoolLike(args.sam2_tiling_enabled, false);
    const enableV2 = parseBoolLike(args.enable_v2, false);
    const reconstructionEnabled = parseBoolLike(args.enable_reconstruction, false);
    const captioningFeatureVisible = isLuxPipeline && _fastVlmCaptioningFeatureEnabled();
    const captioningEnabled = captioningFeatureVisible && parseBoolLike(args.vlm_captioning_enabled, false);
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
    _setContextVisibility(els.sam2TuningPanel, isLuxPipeline && showSam2Controls);
    _setContextVisibility(els.sam2TilingConfigFields, isLuxPipeline && showSam2Controls && sam2TilingEnabled);
    _setContextVisibility(els.sam2GeneratorConfigFields, isLuxPipeline && showSam2Controls);
    if (els.segmentationApplicabilityHint) {
        els.segmentationApplicabilityHint.textContent = !isLuxPipeline
            ? ''
            : !segmentationEnabled
                ? 'Turn segmentation on to choose a backend and strictness policy. SAM2-only controls appear when that backend is selected.'
                : showSam2Controls
                    ? 'SAM2 is active, so generator controls are live now. Tiling values matter only when tiling is enabled.'
                    : `Segmentation is active via ${segmentationBackend}. SAM2-only controls stay hidden until you switch back to sam2.`;
    }
    if (els.sam2TuningHint) {
        els.sam2TuningHint.textContent = sam2TilingEnabled
            ? 'Generator controls and tiling controls are both active for this SAM2 run.'
            : 'Generator controls always apply while SAM2 is active. Tiling values matter only when tiling is enabled.';
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
            els.governanceDetailsHint.textContent = 'Open only when the current preset or backend requires explicit acknowledgments.';
        } else if (appleRequired && researchToolsRequired) {
            els.governanceDetailsHint.textContent = 'Needs attention before dispatch: both Depth Pro and reconstruction acknowledgments are required.';
        } else if (appleRequired) {
            els.governanceDetailsHint.textContent = 'Needs attention before dispatch: Depth Pro research acknowledgments are required.';
        } else if (researchToolsRequired) {
            els.governanceDetailsHint.textContent = 'Needs attention before dispatch: reconstruction acknowledgments are required.';
        } else {
            els.governanceDetailsHint.textContent = 'Needs attention before dispatch: this research preset requires a non-commercial acknowledgment.';
        }
    }

    _setContextVisibility(els.reconstructionConfigFields, reconstructionEnabled);
    _setContextVisibility(els.runtimeTuningFields, isLuxPipeline);
    _setContextVisibility(els.captioningDetails, captioningFeatureVisible);
    _setContextVisibility(els.captioning.enabledFields, captioningEnabled);
    const captioningControls = [
        els.captioning.enableFastVlm,
        els.captioning.model,
        els.captioning.proxyFormat,
        els.captioning.maxSidePx,
        els.captioning.timeoutSeconds,
        els.captioning.pythonExecutable,
        els.captioning.mlxVlmDir,
    ].filter(Boolean);
    captioningControls.forEach((control) => {
        control.disabled = !captioningFeatureVisible || (control !== els.captioning.enableFastVlm && !captioningEnabled);
    });
    if (!captioningFeatureVisible && els.captioning.enableFastVlm) {
        els.captioning.enableFastVlm.checked = false;
        els.captioning.enableFastVlm.setAttribute('aria-checked', 'false');
        state.config.captioning = state.config.captioning || {};
        state.config.captioning.enableFastVlm = false;
    }
    if (els.captioningDetailsHint) {
        els.captioningDetailsHint.textContent = !captioningFeatureVisible
            ? 'FastVLM captioning is disabled for this portal cohort.'
            : captioningEnabled
                ? 'FastVLM captions will emit advisory review sidecars only.'
                : 'Open to enable optional FastVLM caption sidecars for review context.';
    }
    if (els.captioning.status) {
        const preview = _currentPreviewForPayload(payload) || _effectivePreviewSnapshot(payload);
        const summary = preview.captioning_summary || {};
        const readiness = _captioningRuntimeReadiness(summary);
        const runtimeStatus = String(readiness.status || '').trim();
        let captioningStatusText = 'FastVLM captions are off and no captioning args will be emitted.';
        if (!captioningFeatureVisible) {
            captioningStatusText = 'FastVLM caption controls are feature gated for this cohort.';
        } else if (captioningEnabled && runtimeStatus === 'invalid_config') {
            captioningStatusText = 'FastVLM captioning config has invalid runtime paths; preview validation must be repaired before dispatch.';
        } else if (captioningEnabled && runtimeStatus === 'missing_runtime') {
            captioningStatusText = 'FastVLM runtime paths are not fully present; captioning remains advisory and may be skipped.';
        } else if (captioningEnabled && runtimeStatus === 'ready') {
            captioningStatusText = 'FastVLM runtime paths are present for advisory captioning.';
        } else if (captioningEnabled) {
            captioningStatusText = 'FastVLM captions are advisory sidecar metadata and never satisfy quality gates.';
        }
        els.captioning.status.textContent = captioningStatusText;
        _renderCaptioningReadiness(summary, {
            visible: captioningFeatureVisible,
            enabled: captioningEnabled
        });
    }
    if (els.reconstructionDetailsHint) {
        els.reconstructionDetailsHint.textContent = reconstructionEnabled
            ? 'Contextual reconstruction controls are active. Grouping, sidecar, tier, and debug settings are now available.'
            : 'Open only when the posture band or preview calls for deeper runtime tuning.';
    }
}

function _setDisclosureSummaryBadge(element, text, tone = 'info') {
    if (!element) return;
    element.textContent = String(text || '').trim() || 'Optional';
    element.dataset.tone = String(tone || 'info').trim().toLowerCase() || 'info';
}

function syncDisclosurePanels(payload = null) {
    const currentPayload = payload || generatePayload();
    const args = currentPayload.args || {};
    const preset = currentPresetDescriptor();
    const advancedSections = Array.isArray(preset.advanced_sections) ? preset.advanced_sections : [];
    const researchPreset = _presetRequiresResearchAcknowledgments(preset, args);
    const reconstructionEnabled = parseBoolLike(args.enable_reconstruction, false);
    const depthBackend = String(args.depth_backend || '').trim().toLowerCase();
    const nonCommercialRequired = researchPreset || depthBackend === 'depth_pro' || reconstructionEnabled;
    const appleRequired = depthBackend === 'depth_pro';
    const researchToolsRequired = reconstructionEnabled;
    const nonCommercialChecked = parseBoolLike(args.non_commercial_ok, false);
    const appleChecked = parseBoolLike(args.accept_apple_depth_pro_research_license, false);
    const researchToolsChecked = parseBoolLike(args.accept_research_tools_license, false);
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
        captioning: [
            'vlm_captioning_enabled',
            'vlm_captioning_backend',
            'vlm_captioning_model',
            'vlm_captioning_proxy_format',
            'vlm_captioning_max_side_px',
            'fastvlm_python_executable',
            'fastvlm_mlx_vlm_dir',
            'fastvlm_timeout_seconds',
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
    const captioningActive = _fastVlmCaptioningFeatureEnabled() && parseBoolLike(args.vlm_captioning_enabled, false);
    const hasPreviewIssueForGroup = (groupName) => previewFieldGroups[groupName].some((fieldName) => Boolean(_previewIssueForField(fieldName, currentPayload)));
    const currentPreview = _currentPreviewForPayload(currentPayload);
    const advancedNeedsAttention = hasPreviewIssueForGroup('advanced');
    const governanceNeedsAttention = hasPreviewIssueForGroup('governance')
        || (nonCommercialRequired && !nonCommercialChecked)
        || (appleRequired && !appleChecked)
        || (researchToolsRequired && !researchToolsChecked);
    const reconstructionNeedsAttention = hasPreviewIssueForGroup('reconstruction')
        || (_effectiveDebugBundleEnabled(currentPreview, currentPayload) && !state.portalUi.debugBundleAcknowledged);
    const captioningNeedsAttention = hasPreviewIssueForGroup('captioning');
    const disclosurePrefs = state.portalUi.disclosurePrefs || {};
    const autoOpenState = {
        advanced: advancedActive || advancedSections.includes('advanced') || hasPreviewIssueForGroup('advanced'),
        governance: governanceActive || advancedSections.includes('governance') || hasPreviewIssueForGroup('governance'),
        reconstruction: reconstructionActive || advancedSections.includes('reconstruction') || hasPreviewIssueForGroup('reconstruction'),
        captioning: captioningActive || hasPreviewIssueForGroup('captioning'),
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
    syncPanel('captioning', els.captioningDetails);
    syncPanel('dispatchTools', els.dispatchToolsDetails);

    _setDisclosureSummaryBadge(
        els.advancedFlagsSummary,
        advancedNeedsAttention ? 'Needs attention' : advancedActive ? 'Contextual' : 'Secondary',
        advancedNeedsAttention ? 'attention' : 'contextual'
    );
    _setDisclosureSummaryBadge(
        els.governanceDetailsSummary,
        governanceNeedsAttention ? 'Needs attention' : governanceActive ? 'Contextual' : 'Contextual',
        governanceNeedsAttention ? 'attention' : 'contextual'
    );
    _setDisclosureSummaryBadge(
        els.reconstructionDetailsSummary,
        reconstructionNeedsAttention ? 'Needs attention' : reconstructionActive ? 'Contextual' : 'Contextual',
        reconstructionNeedsAttention ? 'attention' : 'contextual'
    );
    _setDisclosureSummaryBadge(
        els.captioningDetailsSummary,
        captioningNeedsAttention ? 'Needs attention' : captioningActive ? 'Advisory on' : 'Advisory off',
        captioningNeedsAttention ? 'attention' : captioningActive ? 'ready' : 'contextual'
    );
    _setDisclosureSummaryBadge(
        els.dispatchToolsSummary,
        els.dispatchToolsDetails?.open
            ? 'Open'
            : currentPreview?.status === 'loading'
                ? 'Preview loading'
                : currentPreview?.status === 'error'
                    ? 'Preview error'
                    : 'Collapsed',
        els.dispatchToolsDetails?.open
            ? 'ready'
            : currentPreview?.status === 'error'
                ? 'attention'
                : currentPreview?.status === 'loading'
                    ? 'info'
                    : 'contextual'
    );
}

function _dispatchChecklistTone(tone = 'info') {
    if (tone === 'block') {
        return {
            badge: 'BLOCK',
            cardClass: 'border-red-200 bg-red-50/90 dark:border-red-900/60 dark:bg-red-950/25',
            badgeClass: 'border-red-300 bg-red-100 text-red-700 dark:border-red-800 dark:bg-red-950/60 dark:text-red-200',
            detailClass: 'text-red-900 dark:text-red-100'
        };
    }
    if (tone === 'warn') {
        return {
            badge: 'WARN',
            cardClass: 'border-amber-200 bg-amber-50/90 dark:border-amber-900/60 dark:bg-amber-950/25',
            badgeClass: 'border-amber-300 bg-amber-100 text-amber-700 dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-200',
            detailClass: 'text-amber-900 dark:text-amber-100'
        };
    }
    return {
        badge: 'PASS',
        cardClass: 'border-emerald-200 bg-emerald-50/90 dark:border-emerald-900/60 dark:bg-emerald-950/25',
        badgeClass: 'border-emerald-300 bg-emerald-100 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-200',
        detailClass: 'text-emerald-900 dark:text-emerald-100'
    };
}

function _dispatchChecklistItems(payload) {
    const args = payload?.args || {};
    const preview = _currentPreviewForPayload(payload) || _effectivePreviewSnapshot(payload);
    const readinessIssues = currentPipelineReadinessIssues();
    const preset = currentPresetDescriptor();
    const previewErrors = Array.isArray(preview?.field_errors) ? preview.field_errors : [];
    const previewWarnings = Array.isArray(preview?.field_warnings) ? preview.field_warnings : [];
    const dispatchStatus = currentPipelineDispatchStatus();
    const isLuxPipeline = state.pipeline === 'lux-depth-v3';
    const researchPreset = _presetRequiresResearchAcknowledgments(preset, args);
    const reconstructionEnabled = parseBoolLike(args.enable_reconstruction, false);
    const depthProEnabled = String(args.depth_backend || '').trim().toLowerCase() === 'depth_pro';
    const nonCommercialChecked = parseBoolLike(args.non_commercial_ok, false);
    const appleChecked = parseBoolLike(args.accept_apple_depth_pro_research_license, false);
    const researchToolsChecked = parseBoolLike(args.accept_research_tools_license, false);
    const debugBundleEnabled = parseBoolLike(args.emit_scene_debug_bundle, false);
    const captioningEnabled = parseBoolLike(args.vlm_captioning_enabled, false);
    const materialsApexGuardrail = parseBoolLike(args.materials_v3, false) && String(args.quality_tier || '').toLowerCase() === 'apex';
    const segmentationReady = parseBoolLike(args.enable_segmentation, false)
        && String(args.segmentation_backend || '').toLowerCase() !== 'stub'
        && parseBoolLike(args.strict_segmentation, false);
    const items = [];

    if (preview?.status === 'error' || previewErrors.length > 0) {
        items.push({
            tone: 'block',
            label: 'Preview normalization',
            detail: String(previewErrors[0]?.message || preview?.error || 'Preview validation is blocking dispatch.')
        });
    } else if (preview?.status === 'loading') {
        items.push({
            tone: 'warn',
            label: 'Preview normalization',
            detail: 'Preview-backed normalization is still refreshing. Dispatch remains cautious until the current draft settles.'
        });
    } else if (previewWarnings.length > 0) {
        items.push({
            tone: 'warn',
            label: 'Preview normalization',
            detail: String(previewWarnings[0]?.message || 'Preview surfaced operator-facing warnings.')
        });
    } else {
        items.push({
            tone: 'pass',
            label: 'Preview normalization',
            detail: 'The normalized config is coherent and ready to drive dispatch parity views.'
        });
    }

    if (!state.backendOk) {
        items.push({
            tone: 'block',
            label: 'Backend connectivity',
            detail: 'The orchestrator API is offline. Dispatch stays disabled until connectivity returns.'
        });
    } else {
        items.push({
            tone: 'pass',
            label: 'Backend connectivity',
            detail: 'The live orchestrator API is reachable for managed dispatch.'
        });
    }

    const blockedReadinessIssue = readinessIssues.find((issue) => String(issue?.severity || '').trim().toLowerCase() === 'blocked');
    const warnedReadinessIssue = readinessIssues.find((issue) => String(issue?.severity || '').trim().toLowerCase() !== 'blocked');
    if (blockedReadinessIssue || dispatchStatus === 'blocked') {
        items.push({
            tone: 'block',
            label: 'Dispatch readiness',
            detail: String(
                blockedReadinessIssue?.message
                || 'A pipeline prerequisite is still blocking dispatch.'
            )
        });
    } else if (warnedReadinessIssue || dispatchStatus === 'degraded') {
        items.push({
            tone: 'warn',
            label: 'Dispatch readiness',
            detail: String(
                warnedReadinessIssue?.message
                || 'The run is valid, but at least one operator prerequisite still needs attention.'
            )
        });
    } else {
        items.push({
            tone: 'pass',
            label: 'Dispatch readiness',
            detail: 'No blocking readiness prerequisites remain for the current draft.'
        });
    }

    if (isLuxPipeline) {
        const missingAcknowledgments = [];
        if (researchPreset && !nonCommercialChecked) missingAcknowledgments.push('non-commercial acknowledgment');
        if (depthProEnabled && !appleChecked) missingAcknowledgments.push('Apple Depth Pro research license');
        if (reconstructionEnabled && !researchToolsChecked) missingAcknowledgments.push('research tools license');
        if (missingAcknowledgments.length > 0) {
            items.push({
                tone: 'block',
                label: 'Governance acknowledgments',
                detail: `Dispatch requires: ${missingAcknowledgments.join(', ')}.`
            });
        } else {
            items.push({
                tone: 'pass',
                label: 'Governance acknowledgments',
                detail: 'Required research, license, and non-commercial acknowledgments are complete for this draft.'
            });
        }

        if (materialsApexGuardrail && !segmentationReady) {
            items.push({
                tone: 'warn',
                label: 'APEX materials guardrail',
                detail: 'APEX + Materials V3 expects strict segmentation with a non-stub backend before dispatch.'
            });
        } else if (materialsApexGuardrail) {
            items.push({
                tone: 'pass',
                label: 'APEX materials guardrail',
                detail: 'Strict segmentation and backend posture are aligned for APEX materials.'
            });
        }

        if (debugBundleEnabled && !state.portalUi.debugBundleAcknowledged) {
            items.push({
                tone: 'block',
                label: 'Debug bundle acknowledgment',
                detail: 'Scene debug bundle capture is enabled. Operator acknowledgment is required before dispatch.'
            });
        } else {
            items.push({
                tone: 'pass',
                label: 'Debug bundle acknowledgment',
                detail: debugBundleEnabled
                    ? 'Debug bundle capture is enabled and acknowledged.'
                    : 'No additional debug-bundle acknowledgment is required.'
            });
        }

        if (captioningEnabled) {
            items.push({
                tone: 'warn',
                label: 'FastVLM captioning',
                detail: 'FastVLM caption sidecars are advisory review metadata and are not used for quality gates.'
            });
        }
    } else {
        items.push({
            tone: dispatchStatus === 'blocked' ? 'block' : dispatchStatus === 'degraded' ? 'warn' : 'pass',
            label: 'Archive governance',
            detail: dispatchStatus === 'blocked'
                ? 'Canonical archive prerequisites are still missing for this stage.'
                : dispatchStatus === 'degraded'
                    ? 'The archive stage is valid, but operator-supplied prerequisites are still required.'
                    : 'The archive command, input posture, and deterministic outputs are aligned.'
        });
    }

    return items;
}

function renderGovernanceBanner(payload) {
    if (!els.governanceChecklist) return;
    const items = _dispatchChecklistItems(payload);
    const hasBlocked = items.some((item) => item.tone === 'block');
    const hasWarnings = items.some((item) => item.tone === 'warn');

    if (els.governanceBannerTitle) {
        els.governanceBannerTitle.textContent = hasBlocked
            ? 'Dispatch is blocked.'
            : hasWarnings
                ? 'Dispatch needs operator attention.'
                : 'Dispatch is ready.';
    }
    if (els.governanceBannerBody) {
        els.governanceBannerBody.textContent = hasBlocked
            ? 'Clear the blocked checklist rows below before treating Step 4 as launch-ready.'
            : hasWarnings
                ? 'The draft is coherent, but at least one checklist row still needs operator attention before launch.'
                : 'The current draft, governance posture, and runtime checks are aligned for dispatch.';
    }
    if (els.governancePostureHint) {
        els.governancePostureHint.textContent = hasBlocked
            ? 'Step 4 mirrors this checklist so the launch lane stays explicit about what is still blocked.'
            : hasWarnings
                ? 'Use this checklist to clear warnings before treating the launch lane as final.'
                : 'Step 4 stays aligned to this checklist so launch remains trustworthy and reviewable.';
    }

    els.governanceChecklist.innerHTML = '';
    items.forEach((item) => {
        const tone = _dispatchChecklistTone(item.tone);
        const card = document.createElement('div');
        card.className = `governance-item ${tone.cardClass}`;
        card.dataset.tone = item.tone;

        const topRow = document.createElement('div');
        topRow.className = 'flex items-start justify-between gap-3';

        const heading = document.createElement('p');
        heading.className = 'text-[10px] font-extrabold uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400';
        heading.textContent = item.label;
        topRow.appendChild(heading);

        const badge = document.createElement('span');
        badge.className = `inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.16em] ${tone.badgeClass}`;
        badge.textContent = tone.badge;
        topRow.appendChild(badge);

        card.appendChild(topRow);

        const detail = document.createElement('p');
        detail.className = `mt-2 text-[12px] leading-6 ${tone.detailClass}`;
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
    renderBuildStepPulse(currentPayload);
    renderRuntimeBriefing(currentPayload);
    renderConsoleContextRibbon();
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

function _setButtonBusy(btn, busy) {
    if (!btn) return;
    btn.setAttribute('aria-busy', busy ? 'true' : 'false');
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
    _toggleSurfaceSkeleton(els.overviewStatsRow, els.overviewStatsRow, els.overviewStatsSkeletonState, bootstrapLoading);
    _toggleSurfaceSkeleton(els.overviewCapabilityRow, els.overviewCapabilityRow, els.overviewCapabilitySkeletonState, bootstrapLoading);
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

function renderSelectedJobInspector() {
    const api = _deferredOperateSurfaceApi();
    if (api?.renderSelectedJobInspector) {
        api.renderSelectedJobInspector();
        return;
    }
    if (!_shouldLoadDeferredOperateSurface()) return;
    void _loadDeferredOperateSurface().then((loaded) => {
        if (loaded?.renderSelectedJobInspector) loaded.renderSelectedJobInspector();
    });
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
            directDebug: false,
            artifactViewerModal: false,
            reviewSurfaceDeferred: false,
            stagedUploads: false,
            rumTelemetry: false,
            fastVlmCaptioning: false
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
    state.bootstrap.lastRateLimitHint = null;
    _recordBootstrapRetryEvent(outcome, details);
}

function _bootstrapFailureDetails(reason = '', httpStatus = 0, message = '', rateLimitHint = null) {
    const normalizedReason = String(reason || '').trim().toLowerCase();
    const normalizedStatus = Number.isFinite(Number(httpStatus)) ? Number(httpStatus) : 0;
    const overrideMessage = String(message || '').trim();
    const retryAtMs = rateLimitHint && Number.isFinite(Number(rateLimitHint.retryAtMs))
        && Number(rateLimitHint.retryAtMs) > 0
        ? Number(rateLimitHint.retryAtMs)
        : null;

    if (normalizedReason === 'auth_failure' || normalizedReason === 'auth' || normalizedStatus === 401 || normalizedStatus === 403) {
        return {
            reason: 'auth_failure',
            retryable: false,
            toastMessage: overrideMessage || 'Managed authentication expired. Sign in again to restore privileged actions.',
            actionMessage: 'Managed authentication is required. Sign in again before continuing.'
        };
    }
    if (normalizedReason === 'rate_limited' || normalizedStatus === 429) {
        return {
            reason: 'rate_limited',
            retryable: true,
            toastMessage: overrideMessage || 'Portal is rate limited. Bootstrap will retry automatically when the cooldown expires.',
            actionMessage: 'Portal is rate limited. Wait for the cooldown to expire before retrying.',
            retryAtMs
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

function _scheduleBootstrapRetry(reason = '', httpStatus = 0, rateLimitHint = null) {
    // Persist the parsed hint so the recovery banner renderer can surface a
    // live countdown to the user. The slot is cleared by
    // _finalizeBootstrapRetry on success / terminal outcomes so a stale hint
    // never lingers past the cooldown window.
    state.bootstrap.lastRateLimitHint = rateLimitHint && Number(rateLimitHint?.retryAtMs) > 0
        ? rateLimitHint
        : null;
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

    // Prefer the upstream Retry-After / X-RateLimit-Reset hint when present
    // (e.g. on HTTP 429). The parser already clamped the value to a sane
    // upper bound; we additionally clamp to BOOTSTRAP_RETRY_MAX_DELAY_MS so
    // the existing fallback's max remains the single source of truth, and
    // an absurd or buggy upstream cannot stretch a single retry beyond the
    // bootstrap window.
    let delayMs = _nextBootstrapRetryDelayMs(attempt);
    let delaySource = 'exponential_backoff';
    if (
        rateLimitHint
        && Number.isFinite(Number(rateLimitHint.retryAfterMs))
        && Number(rateLimitHint.retryAfterMs) > 0
    ) {
        delayMs = Math.min(Number(rateLimitHint.retryAfterMs), BOOTSTRAP_RETRY_MAX_DELAY_MS);
        delaySource = String(rateLimitHint.source || 'rate_limit_hint');
    }
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
        _recordBootstrapRetryEvent('attempt_started', { attempt, reason, httpStatus, delayMs: 0, delaySource });
        void loadPortalBootstrap({ isRetryAttempt: true, attempt, retryReason: reason });
    }, delayMs);

    _recordBootstrapRetryEvent('scheduled', { attempt, reason, httpStatus, delayMs, delaySource });
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

// Generalized retry discipline for protected /v1/* fetches.
//
// The managed front door already classifies upstream 401/403 as
// CONFIG_FAILURE with retryable:false (web/secure-landing/lib/managed-failure.js).
// Bootstrap honors that flag; everything else (refreshJobStatus, RUM, portal
// events, config-preview/metadata, readiness, presets) used to swallow it and
// keep polling, which is the 503 storm visible in the diagnosis logs.
//
// Behaviour:
//   - When a protected fetch returns a body with details.retryable === false,
//     the corresponding endpoint family is suppressed for the page lifetime.
//   - Subsequent calls to _isProtectedFamilySuppressed(family) return true so
//     callers can skip the fetch entirely.
//   - One console.warn per family makes the diagnosis visible in DevTools.
//   - _resetProtectedFamilySuppression() clears the map when the operator
//     updates the direct-debug API key or an explicit recovery action retries.
const _protectedFamilySuppression = new Map();

function _classifyProtectedEndpointFamily(url) {
    const path = String(url || '').split('?')[0];
    if (!path) return '';
    if (path.includes('/v1/jobs/') && path.endsWith('/events')) return 'jobs_events';
    if (path.includes('/v1/jobs/') && path.endsWith('/cancel')) return 'jobs_cancel';
    if (/\/v1\/jobs\/[^/]+$/.test(path)) return 'jobs_detail';
    if (path.endsWith('/v1/jobs')) return 'jobs_list';
    if (path.endsWith('/v1/config-metadata')) return 'config_metadata';
    if (path.endsWith('/v1/config-preview')) return 'config_preview';
    if (path.endsWith('/v1/portal/rum')) return 'portal_rum';
    if (path.endsWith('/v1/portal/events')) return 'portal_events';
    if (path.endsWith('/v1/readiness')) return 'readiness';
    if (path.endsWith('/v1/presets')) return 'presets';
    if (path.endsWith('/v1/uploads/staging')) return 'uploads_staging';
    return '';
}

function _isProtectedFamilySuppressed(family) {
    return Boolean(family) && _protectedFamilySuppression.has(family);
}

function _recordProtectedFamilySuppression(family, payload) {
    if (!family || _protectedFamilySuppression.has(family)) return;
    const detail = payload && typeof payload === 'object' ? payload : {};
    _protectedFamilySuppression.set(family, {
        recordedAt: Date.now(),
        reason: String(detail.reason || 'config_failure'),
        upstreamStatus: Number.isFinite(Number(detail.upstreamStatus)) ? Number(detail.upstreamStatus) : 0
    });
    try {
        // eslint-disable-next-line no-console
        console.warn(
            `[portal] suppressing further requests to ${family}: reason=${detail.reason || 'config_failure'} ` +
            `upstreamStatus=${detail.upstreamStatus || 'unknown'}. ` +
            `Resolve frontdoor configuration (TP_BACKEND_API_KEY / backend TP_API_KEY).`
        );
    } catch {
        // logging is best-effort
    }
}

function _protectedErrorDetailsFromPayload(payload) {
    if (!payload || typeof payload !== 'object') return null;
    const nestedDetails = payload.error && typeof payload.error === 'object'
        ? payload.error.details
        : null;
    if (nestedDetails && typeof nestedDetails === 'object') return nestedDetails;
    const topLevelDetails = payload.details;
    return topLevelDetails && typeof topLevelDetails === 'object' ? topLevelDetails : null;
}

function _nonRetryableProtectedDetails(payload) {
    const details = _protectedErrorDetailsFromPayload(payload);
    return details && details.retryable === false ? details : null;
}

// Inspect a non-OK protected response. If the body carries details.retryable === false,
// suppress the family and return true so the caller can short-circuit.
async function _maybeSuppressOnProtectedResponse(family, response) {
    if (!family || !response || response.ok) return false;
    let body = null;
    try {
        body = await response.clone().json();
    } catch {
        return false;
    }
    const details = _nonRetryableProtectedDetails(body);
    if (!details) return false;
    _recordProtectedFamilySuppression(family, {
        reason: details.reason,
        upstreamStatus: details.upstreamStatus
    });
    return true;
}

function _resetProtectedFamilySuppression(family) {
    if (typeof family === 'string' && family) {
        _protectedFamilySuppression.delete(family);
    } else {
        _protectedFamilySuppression.clear();
    }
}

function _handleDirectDebugApiKeyUpdate(options = null) {
    const resumeStreams = Boolean(options.resumeStreams);
    _persistApiKeyFromInputs();
    _resetProtectedFamilySuppression();
    if (!resumeStreams) return;
    resumeBlockedJobStreamsAfterAuthUpdate();
    void checkBackend(true);
    void fetchConfigMetadata(state.pipeline, true);
    void fetchPresetsForPipeline(state.pipeline, true);
    void fetchReadiness(true);
    void fetchConfigPreview(generatePayload());
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
    const readiness = _dispatchReadinessSnapshot();
    if (els.runJobBtn && els.runJobBtn.textContent !== 'Dispatching...') {
        els.runJobBtn.disabled = !readiness.canRun;
    }
    if (els.dispatchReadinessReason) {
        _renderBannerDetailWithRetryCountdown(
            els.dispatchReadinessReason,
            readiness.detail,
            readiness.retryCountdownAtMs
        );
        els.dispatchReadinessReason.dataset.tone = readiness.tone;
    }
}

function _isManagedAuthMode() {
    return portalInternals.isManagedAuthMode(state.auth);
}

function _isManagedUnavailableMode() {
    return portalInternals.isManagedUnavailableMode(state.auth);
}

function _bootstrapSurfaceSummary() {
    const bootstrapStatus = String(state.bootstrap.status || 'pending').trim().toLowerCase();
    if (bootstrapStatus === 'ready') {
        if (_isManagedAuthMode()) {
            return {
                tone: 'ready',
                badge: 'Managed access',
                detail: 'Managed access is verified. Backend credentials stay server-side and browser-side API key entry remains hidden.',
                apiHint: 'Managed mode is active. Backend credentials stay server-side and are never stored in the browser.'
            };
        }
        return {
            tone: 'warning',
            badge: 'Direct debug',
            detail: 'Direct debug is active. Browser-side API key entry is available only for local troubleshooting.',
            apiHint: 'Direct debug is active. Browser-side API key entry is available only for local troubleshooting.'
        };
    }
    if (bootstrapStatus === 'degraded' || bootstrapStatus === 'unavailable') {
        const failure = _bootstrapFailureDetails(state.bootstrap.lastErrorReason, state.bootstrap.lastHttpStatus);
        const tone = failure.retryable ? 'warning' : 'blocked';
        return {
            tone,
            badge: failure.retryable ? 'Recovery pending' : 'Recovery required',
            detail: failure.actionMessage,
            apiHint: failure.actionMessage
        };
    }
    return {
        tone: 'info',
        badge: 'Confirming access',
        detail: 'Bootstrap is still being confirmed. Privileged actions remain disabled until portal authentication is resolved.',
        apiHint: 'Bootstrap is still being confirmed. Privileged actions remain disabled until portal authentication is resolved.'
    };
}

function _clearStoredApiKeyState(clearPersisted = true) {
    if (clearPersisted) {
        localStorage.removeItem(API_KEY_STORAGE_KEY);
        sessionStorage.removeItem(API_KEY_STORAGE_KEY);
    }
    if (els.apiKeyInput) els.apiKeyInput.value = '';
}

function _stagedUploadsVisibleForState() {
    return Boolean(
        _isBootstrapReady()
        && state.auth?.features?.stagedUploads
        && STAGED_UPLOAD_SUPPORTED_PIPELINES.has(String(state.pipeline || '').trim())
    );
}

function _stagedUploadErrorMessage(payload, fallback = 'Failed to stage uploads.') {
    const message = String(payload?.error?.message || '').trim();
    return message || fallback;
}

function _setStagedUploadState(patch = {}) {
    state.portalUi.stagedUpload = {
        ...(state.portalUi.stagedUpload || {}),
        ...(patch && typeof patch === 'object' ? patch : {})
    };
}

function _syncStagedUploadUi() {
    const visible = _stagedUploadsVisibleForState();
    const uploadState = state.portalUi.stagedUpload || {};
    const busy = Boolean(uploadState.busy);
    const progressPercent = Math.max(0, Math.min(100, Number(uploadState.progressPercent) || 0));
    if (els.stagedUploadShell) {
        els.stagedUploadShell.classList.toggle('hidden', !visible);
        els.stagedUploadShell.dataset.busy = busy ? 'true' : 'false';
    }
    if (els.stagedUploadStatus) {
        if (!visible) {
            els.stagedUploadStatus.textContent = 'Upload a file set and the staged input directory will replace the current source path.';
        } else if (uploadState.error) {
            els.stagedUploadStatus.textContent = 'The last staged upload attempt failed. Fix the payload and try again.';
        } else if (busy) {
            els.stagedUploadStatus.textContent = String(uploadState.status || 'Uploading files to the staged input directory...');
        } else if (uploadState.summary) {
            els.stagedUploadStatus.textContent = 'Staged uploads are ready. Review the receipt below before dispatching the next run.';
        } else {
            els.stagedUploadStatus.textContent = 'Upload a file set and the staged input directory will replace the current source path.';
        }
    }
    if (els.stagedUploadDropzone) {
        els.stagedUploadDropzone.dataset.disabled = !visible || busy ? 'true' : 'false';
        els.stagedUploadDropzone.classList.toggle('opacity-60', !visible || busy);
        els.stagedUploadDropzone.classList.toggle('cursor-not-allowed', !visible || busy);
        els.stagedUploadDropzone.setAttribute('aria-disabled', !visible || busy ? 'true' : 'false');
    }
    if (els.stagedUploadPickFilesBtn) {
        els.stagedUploadPickFilesBtn.disabled = !visible || busy;
    }
    if (els.stagedUploadPickFolderBtn) {
        els.stagedUploadPickFolderBtn.disabled = !visible || busy;
    }
    if (els.stagedUploadFilesInput) {
        els.stagedUploadFilesInput.disabled = !visible || busy;
    }
    if (els.stagedUploadFolderInput) {
        els.stagedUploadFolderInput.disabled = !visible || busy;
    }
    if (els.stagedUploadProgressBar) {
        els.stagedUploadProgressBar.style.width = `${progressPercent}%`;
    }
    if (els.stagedUploadProgressLabel) {
        if (busy) {
            els.stagedUploadProgressLabel.textContent = `${progressPercent}% uploaded`;
        } else if (uploadState.summary) {
            els.stagedUploadProgressLabel.textContent = 'Upload complete.';
        } else {
            els.stagedUploadProgressLabel.textContent = 'No upload in progress.';
        }
    }
    if (els.stagedUploadSummary) {
        _renderStagedUploadSummary(els.stagedUploadSummary, uploadState);
    }
    if (els.stagedUploadError) {
        els.stagedUploadError.textContent = String(uploadState.error || '');
    }
}

function _syncBootstrapUi() {
    const bootstrapReady = _isBootstrapReady();
    const showApiKeyInput = bootstrapReady && state.auth.features.apiKeyInput;
    const badgeLabel = bootstrapReady ? state.auth.mode : 'unknown';
    const summary = _bootstrapSurfaceSummary();
    if (els.apiKeySection) {
        els.apiKeySection.classList.toggle('hidden', !showApiKeyInput);
    }
    if (els.authModeBadge) {
        els.authModeBadge.textContent = badgeLabel;
    }
    if (document.body) {
        document.body.dataset.bootstrapReason = String(state.bootstrap.lastErrorReason || '');
        document.body.dataset.authMode = String(state.auth.mode || 'managed_unavailable');
    }
    if (els.portalAccessState) {
        els.portalAccessState.dataset.tone = summary.tone;
        els.portalAccessState.dataset.bootstrapStatus = String(state.bootstrap.status || 'pending');
        els.portalAccessState.dataset.bootstrapReason = String(state.bootstrap.lastErrorReason || '');
    }
    if (els.bootstrapStatusBadge) {
        els.bootstrapStatusBadge.dataset.tone = summary.tone;
        els.bootstrapStatusBadge.textContent = summary.badge;
    }
    if (els.bootstrapRecoveryHint) {
        els.bootstrapRecoveryHint.textContent = summary.detail;
    }
    if (els.apiKeyManagedHint) {
        if (bootstrapReady && !_isManagedAuthMode()) {
            els.apiKeyManagedHint.classList.add('hidden');
        } else {
            els.apiKeyManagedHint.classList.remove('hidden');
            els.apiKeyManagedHint.textContent = summary.apiHint;
        }
    }
    if (els.apiKeyInput) {
        els.apiKeyInput.disabled = !showApiKeyInput;
    }
    _syncBootstrapGuardedControls();
    syncBuildSurfaceApplicability();
    _syncOverviewBuildLoadingState();
    renderOperatorActionRail();
    const selectedJob = _findJobById(state.selectedJobId);
    renderSelectedJobRecoveryActions(selectedJob);
    renderReviewStatusActions(selectedJob);
    renderArtifactViewer();
    _syncStagedUploadUi();
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
            directDebug: mode === 'direct_debug' && bootstrap.features?.directDebug !== false,
            artifactViewerModal: Boolean(bootstrap.features?.artifactViewerModal),
            reviewSurfaceDeferred: Boolean(bootstrap.features?.reviewSurfaceDeferred),
            stagedUploads: Boolean(bootstrap.features?.stagedUploads),
            rumTelemetry: Boolean(bootstrap.features?.rumTelemetry),
            fastVlmCaptioning: Boolean(bootstrap.features?.fastVlmCaptioning)
        }
    };
    const bootstrapTraceparent = portalInternals.normalizePortalRumTraceparent(
        options.traceparent,
        state.rum.pageTraceparent
    );
    state.rum.pageTraceparent = bootstrapTraceparent;
    state.rum.bootstrapTraceparent = bootstrapTraceparent;
    state.rum.enabled = Boolean(state.auth.features.rumTelemetry);
    _setBootstrapStatus(nextStatus, options.reason || '', options.httpStatus || 0);
    if (_isBootstrapReady() && isManagedMode) {
        _clearStoredApiKeyState(true);
    }
    if (!_rumTelemetryEnabled()) {
        state.rum.queuedSamples = [];
    }
    _loadApiKeyIntoInputs();
    _syncBootstrapUi();
    if (_rumTelemetryEnabled()) {
        void _flushQueuedPortalRumSamples();
    }
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
    const fallback = portalInternals.defaultPortalBootstrapPayload();
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
                headers: {
                    'Accept': 'application/json',
                    traceparent: state.rum.pageTraceparent
                },
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
            _flushPendingTransientPortalDraftPersist();
            window.location.assign(_managedLoginUrlForCurrentRoute());
            return;
        }
        const bootstrapTraceparent = portalInternals.normalizePortalRumTraceparent(
            res.headers.get('traceparent'),
            state.rum.pageTraceparent
        );
        if (!res.ok) {
            // Honour the orchestrator's 429 retry contract (Retry-After +
            // X-RateLimit-Reset). The hint is captured from the response
            // object only on HTTP 429; auth (401/403) and other failures
            // fall through to existing exponential backoff or terminal
            // handling.
            const rateLimitHint = res.status === 429
                ? portalInternals.parseRateLimitRetryHint(res)
                : null;
            const failure = _bootstrapFailureDetails(
                payloadParsed && payload && typeof payload === 'object'
                    ? payload.reason
                    : (res.status === 429 ? 'rate_limited' : `http_${res.status}`),
                res.status,
                payloadParsed && payload && typeof payload === 'object' ? payload.message : ''
            );
            const status = failure.retryable ? 'degraded' : 'unavailable';
            _applyPortalBootstrap(fallback, { status, reason: failure.reason, httpStatus: res.status });
            const retryScheduled = failure.retryable
                && _scheduleBootstrapRetry(failure.reason, res.status, rateLimitHint);
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
        _applyPortalBootstrap(payload, { status: 'ready', traceparent: bootstrapTraceparent });
        _recordPortalRumMilestone('bootstrap_ready', _portalRumNow(), {
            traceparent: bootstrapTraceparent
        });
        _scheduleFirstViewInteractiveRum();
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

function _buildAuthHeaders(base = {}, method = 'GET', options = null) {
    const headers = { ...base };
    const normalizedMethod = String(method || 'GET').toUpperCase();
    const authOptions = options && typeof options === 'object' ? options : null;
    const traceparent = authOptions && typeof authOptions.traceparent === 'string'
        ? authOptions.traceparent.trim()
        : '';
    if (traceparent) {
        headers.traceparent = traceparent;
    }
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

function _normalizeStagedUploadRelativePath(file) {
    if (!file || typeof file !== 'object') return '';
    const rawRelativePath = typeof file.webkitRelativePath === 'string' && file.webkitRelativePath.trim()
        ? file.webkitRelativePath
        : String(file.name || '');
    return rawRelativePath.replace(/\\/g, '/').replace(/^\/+/, '').trim();
}

function _collectStagedUploadSelection(fileList) {
    const entries = [];
    let totalBytes = 0;
    Array.from(fileList || []).forEach((file) => {
        const relativePath = _normalizeStagedUploadRelativePath(file);
        if (!relativePath) return;
        const sizeBytes = Number(file && typeof file === 'object' ? file.size : 0) || 0;
        totalBytes += Math.max(0, sizeBytes);
        entries.push({
            file,
            relativePath,
            sizeBytes,
            name: String(file?.name || ''),
        });
    });
    return { entries, totalBytes };
}

function _stagedUploadToneClasses(tone = 'info') {
    if (tone === 'blocked') {
        return 'border-red-200 bg-red-50/90 text-red-800 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-200';
    }
    if (tone === 'warning') {
        return 'border-amber-200 bg-amber-50/90 text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200';
    }
    if (tone === 'success') {
        return 'border-emerald-200 bg-emerald-50/90 text-emerald-800 dark:border-emerald-900/60 dark:bg-emerald-950/30 dark:text-emerald-200';
    }
    return 'border-slate-200 bg-white/90 text-slate-800 dark:border-slate-700 dark:bg-slate-900/70 dark:text-slate-200';
}

function _appendStagedUploadFact(container, label, value, tone = 'neutral') {
    if (!container || !String(value || '').trim()) return;
    const card = document.createElement('div');
    card.className = `rounded-lg border px-3 py-3 ${_stagedUploadToneClasses(
        tone === 'blocked' ? 'blocked' : tone === 'warning' ? 'warning' : tone === 'success' ? 'success' : 'info'
    )}`;

    const heading = document.createElement('p');
    heading.className = 'text-[10px] font-bold uppercase tracking-[0.18em] opacity-75';
    heading.textContent = label;
    card.appendChild(heading);

    const detail = document.createElement('p');
    detail.className = 'mt-2 text-[12px] font-semibold leading-6';
    detail.textContent = value;
    card.appendChild(detail);

    container.appendChild(card);
}

function _renderStagedUploadSummary(container, uploadState) {
    if (!container) return;
    container.replaceChildren();

    const summaryText = String(uploadState?.summary || '').trim();
    if (!summaryText) return;

    const receipt = uploadState?.receipt && typeof uploadState.receipt === 'object' ? uploadState.receipt : null;
    const receiptSummary = receipt?.summary && typeof receipt.summary === 'object' ? receipt.summary : {};
    const topLevelRoots = Array.isArray(receiptSummary.top_level_roots)
        ? receiptSummary.top_level_roots.map((item) => String(item || '').trim()).filter(Boolean)
        : [];
    const warnings = Array.isArray(receiptSummary.warnings)
        ? receiptSummary.warnings.map((item) => String(item || '').trim()).filter(Boolean)
        : [];
    const receivedAtMs = Number.isFinite(Number(receipt?.received_at_epoch_seconds))
        ? Number(receipt.received_at_epoch_seconds) * 1000
        : 0;

    const shell = document.createElement('section');
    shell.className = 'rounded-xl border border-slate-200 bg-slate-50/90 p-4 dark:border-slate-800 dark:bg-slate-950/40';

    const header = document.createElement('div');
    header.className = 'flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between';

    const headerCopy = document.createElement('div');
    const title = document.createElement('p');
    title.className = 'text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400';
    title.textContent = 'Upload receipt';
    headerCopy.appendChild(title);

    const summary = document.createElement('p');
    summary.className = 'mt-2 text-[12px] font-semibold leading-6 text-slate-900 dark:text-white';
    summary.textContent = summaryText;
    headerCopy.appendChild(summary);
    header.appendChild(headerCopy);

    if (receipt?.batch_id) {
        const badge = document.createElement('span');
        badge.className = 'inline-flex items-center rounded-full border border-slate-300 bg-white px-3 py-1 text-[10px] font-mono uppercase tracking-[0.16em] text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300';
        badge.textContent = receipt.batch_id;
        header.appendChild(badge);
    }

    shell.appendChild(header);

    const grid = document.createElement('div');
    grid.className = 'mt-4 grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4';
    _appendStagedUploadFact(
        grid,
        'Files received',
        `${Math.max(0, Number(receiptSummary.file_count) || 0)} file${Number(receiptSummary.file_count) === 1 ? '' : 's'}`,
        'success'
    );
    _appendStagedUploadFact(grid, 'Payload size', formatBytes(Math.max(0, Number(receiptSummary.total_bytes) || 0)), 'info');
    _appendStagedUploadFact(
        grid,
        'Preserved roots',
        topLevelRoots.length > 0 ? topLevelRoots.join(', ') : 'Flat file set',
        topLevelRoots.length > 0 ? 'success' : 'info'
    );
    _appendStagedUploadFact(
        grid,
        'Completed',
        receivedAtMs > 0 ? formatRelativeTime(receivedAtMs) : 'just now',
        'success'
    );
    shell.appendChild(grid);

    const artifacts = receipt?.artifacts && typeof receipt.artifacts === 'object' ? receipt.artifacts : {};
    const artifactEntries = [
        ['Baseline manifest', artifacts.baseline_manifest_path],
        ['Capture metadata', artifacts.capture_metadata_path],
        ['Upload receipt', artifacts.upload_receipt_path],
    ].filter(([, value]) => String(value || '').trim());
    if (artifactEntries.length > 0) {
        const artifactList = document.createElement('div');
        artifactList.className = 'mt-4 rounded-lg border border-slate-200 bg-white/90 p-3 dark:border-slate-800 dark:bg-slate-900/70';

        const artifactTitle = document.createElement('p');
        artifactTitle.className = 'text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400';
        artifactTitle.textContent = 'Artifacts';
        artifactList.appendChild(artifactTitle);

        artifactEntries.forEach(([label, value]) => {
            const row = document.createElement('div');
            row.className = 'mt-3';

            const rowLabel = document.createElement('p');
            rowLabel.className = 'text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500 dark:text-slate-400';
            rowLabel.textContent = label;
            row.appendChild(rowLabel);

            const rowValue = document.createElement('p');
            rowValue.className = 'mt-1 break-all rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] font-mono leading-5 text-slate-700 dark:border-slate-700 dark:bg-slate-950/60 dark:text-slate-200';
            rowValue.textContent = String(value || '');
            row.appendChild(rowValue);

            artifactList.appendChild(row);
        });

        shell.appendChild(artifactList);
    }

    if (warnings.length > 0) {
        const warningList = document.createElement('div');
        warningList.className = 'mt-4 rounded-lg border border-amber-200 bg-amber-50/90 p-3 dark:border-amber-900/60 dark:bg-amber-950/30';

        const warningTitle = document.createElement('p');
        warningTitle.className = 'text-[10px] font-bold uppercase tracking-[0.18em] text-amber-700 dark:text-amber-300';
        warningTitle.textContent = 'Inline failures';
        warningList.appendChild(warningTitle);

        warnings.forEach((warning) => {
            const item = document.createElement('p');
            item.className = 'mt-2 text-[12px] leading-6 text-amber-800 dark:text-amber-100';
            item.textContent = warning;
            warningList.appendChild(item);
        });

        shell.appendChild(warningList);
    }

    container.appendChild(shell);
}

function _applyStagedUploadResult(result) {
    const inputDir = String(result?.input_dir || '').trim();
    const summary = result?.summary && typeof result.summary === 'object' ? result.summary : {};
    const fileCount = Math.max(0, Number(summary.file_count) || 0);
    const totalBytes = Math.max(0, Number(summary.total_bytes) || 0);
    if (!inputDir) {
        throw new Error('staged upload response did not include input_dir');
    }

    _setStagedUploadState({
        busy: false,
        progressPercent: 100,
        status: 'ready',
        summary: `Staged ${fileCount} file${fileCount === 1 ? '' : 's'} (${formatBytes(totalBytes)}). Input directory updated with receipt-backed operator detail.`,
        error: '',
        lastBatchId: String(result?.batch_id || ''),
        fileCount,
        totalBytes,
        receipt: result && typeof result === 'object' ? result : null,
    });

    state.config.inputDir = inputDir;
    if (els.inputDir) {
        els.inputDir.value = inputDir;
        els.inputDir.dispatchEvent(new Event('input', { bubbles: true }));
        els.inputDir.dispatchEvent(new Event('change', { bubbles: true }));
    } else {
        renderCLI();
        scheduleConfigPreview(true);
    }
    renderFieldPreviewStatuses();
    _syncStagedUploadUi();
    createToast('Upload staged. Input directory updated.', 'success');
}

function _submitStagedUploadSelection(fileList) {
    if (_blockManagedUnavailableAction('stage uploads')) return;
    if (!_stagedUploadsVisibleForState()) return;
    if (_isProtectedFamilySuppressed('uploads_staging')) {
        createToast('Staged uploads are paused until frontdoor configuration is repaired.', 'error');
        return;
    }

    const selection = _collectStagedUploadSelection(fileList);
    if (selection.entries.length === 0) {
        _setStagedUploadState({
            busy: false,
            progressPercent: 0,
            status: 'idle',
            summary: '',
            error: 'Select at least one file before staging uploads.',
            receipt: null,
        });
        _syncStagedUploadUi();
        createToast('Select at least one file before staging uploads.', 'info');
        return;
    }

    _setStagedUploadState({
        busy: true,
        progressPercent: 0,
        status: `Uploading ${selection.entries.length} file${selection.entries.length === 1 ? '' : 's'} to staged storage...`,
        summary: '',
        error: '',
        fileCount: selection.entries.length,
        totalBytes: selection.totalBytes,
        receipt: null,
    });
    _syncStagedUploadUi();

    const formData = new FormData();
    selection.entries.forEach(({ file, relativePath }) => {
        formData.append('files', file, relativePath);
    });
    formData.append(
        'client_manifest',
        JSON.stringify({
            schema: 'tp.portal.upload_manifest.v1',
            files: selection.entries.map((entry) => ({
                relative_path: entry.relativePath,
                size_bytes: entry.sizeBytes,
                name: entry.name,
            })),
        })
    );

    const requestTraceparent = portalInternals.createChildTraceparent(_portalRumTraceparent());
    const headers = _buildAuthHeaders({}, 'POST', { traceparent: requestTraceparent });
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/v1/uploads/staging`);
    xhr.responseType = 'json';
    Object.entries(headers).forEach(([key, value]) => {
        if (typeof value === 'string' && value) {
            xhr.setRequestHeader(key, value);
        }
    });

    xhr.upload.onprogress = (event) => {
        if (!event.lengthComputable) return;
        const progressPercent = Math.max(0, Math.min(100, Math.round((event.loaded / event.total) * 100)));
        _setStagedUploadState({
            progressPercent,
            status: `Uploading ${selection.entries.length} file${selection.entries.length === 1 ? '' : 's'} to staged storage...`,
        });
        _syncStagedUploadUi();
    };

    xhr.onerror = () => {
        _setStagedUploadState({
            busy: false,
            progressPercent: 0,
            status: 'idle',
            error: 'Network failure interrupted the staged upload.',
            receipt: null,
        });
        _syncStagedUploadUi();
        createToast('Network failure interrupted the staged upload.', 'error');
    };

    xhr.onabort = () => {
        _setStagedUploadState({
            busy: false,
            progressPercent: 0,
            status: 'idle',
            error: 'Staged upload canceled before completion.',
            receipt: null,
        });
        _syncStagedUploadUi();
        createToast('Staged upload canceled before completion.', 'error');
    };

    xhr.onload = () => {
        const payload = xhr.response && typeof xhr.response === 'object'
            ? xhr.response
            : (() => {
                try {
                    return JSON.parse(xhr.responseText || '{}');
                } catch (_err) {
                    return {};
                }
            })();
        if (xhr.status >= 200 && xhr.status < 300 && payload?.success && payload.data) {
            _applyStagedUploadResult(payload.data);
            return;
        }

        const nonRetryableDetails = _nonRetryableProtectedDetails(payload);
        if (nonRetryableDetails) {
            _recordProtectedFamilySuppression('uploads_staging', nonRetryableDetails);
        }
        const errorMessage = _stagedUploadErrorMessage(payload);
        _setStagedUploadState({
            busy: false,
            progressPercent: 0,
            status: 'idle',
            error: errorMessage,
            receipt: null,
        });
        _syncStagedUploadUi();
        createToast(errorMessage, 'error');
    };

    xhr.send(formData);
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

const CAPTIONING_RUN_STATUS_VALUES = new Set([
    'off',
    'requested',
    'succeeded',
    'failed',
    'skipped',
    'missing_runtime',
    'invalid_config',
    'unsupported_backend'
]);

function toNonNegativeCaptioningRunStatusInt(value) {
    if (typeof value === 'boolean') return 0;
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed < 0) return 0;
    return Math.round(parsed);
}

function normalizeCaptioningRunStatus(rawStatus) {
    if (!rawStatus || typeof rawStatus !== 'object') return null;
    let status = String(rawStatus.status || '').trim().toLowerCase();
    const sidecarCount = toNonNegativeCaptioningRunStatusInt(rawStatus.sidecar_count);
    const failedCount = toNonNegativeCaptioningRunStatusInt(rawStatus.failed_count);
    const enabled = parseBoolLike(rawStatus.enabled, status !== 'off');
    if (!CAPTIONING_RUN_STATUS_VALUES.has(status)) {
        if (!enabled) {
            status = 'off';
        } else if (failedCount > 0) {
            status = 'failed';
        } else if (sidecarCount > 0) {
            status = 'succeeded';
        } else {
            status = 'skipped';
        }
    }
    return {
        status,
        enabled: status === 'off' ? false : enabled,
        backend: String(rawStatus.backend || 'fastvlm').trim().toLowerCase() || 'fastvlm',
        model_role: String(rawStatus.model_role || '').trim(),
        model_id: rawStatus.model_id === null || rawStatus.model_id === undefined ? null : String(rawStatus.model_id),
        model_path: rawStatus.model_path === null || rawStatus.model_path === undefined ? null : String(rawStatus.model_path),
        role: 'advisory',
        sidecar_count: sidecarCount,
        raw_count: toNonNegativeCaptioningRunStatusInt(rawStatus.raw_count),
        proxy_count: toNonNegativeCaptioningRunStatusInt(rawStatus.proxy_count),
        failed_count: failedCount,
        used_for_quality_gate: false,
        policy_violation: Boolean(rawStatus.policy_violation)
    };
}

function captioningRunStatusLabel(status) {
    const normalized = String(status?.status || '').trim();
    if (normalized === 'off') return 'Off';
    if (normalized === 'requested') return 'Requested';
    if (normalized === 'succeeded') return 'Succeeded';
    if (normalized === 'failed') return 'Failed';
    if (normalized === 'skipped') return 'Skipped';
    if (normalized === 'missing_runtime') return 'Missing runtime';
    if (normalized === 'invalid_config') return 'Invalid config';
    if (normalized === 'unsupported_backend') return 'Unsupported backend';
    return 'Not requested';
}

function captioningRunStatusSummary(status) {
    if (!status) return 'FastVLM: Not requested';
    const label = captioningRunStatusLabel(status);
    const sidecars = Number(status.sidecar_count) || 0;
    const suffix = sidecars > 0 ? ` (${sidecars} sidecar${sidecars === 1 ? '' : 's'})` : '';
    return `FastVLM: ${label}${suffix}`;
}

function captioningRunStatusDetail(status) {
    if (!status) return 'FastVLM: Not requested';
    const parts = [
        captioningRunStatusSummary(status),
        `${Number(status.raw_count) || 0} raw`,
        `${Number(status.proxy_count) || 0} proxy`,
    ];
    if (Number(status.failed_count) > 0) parts.push(`${Number(status.failed_count)} failed`);
    if (status.model_role) parts.push(`role ${status.model_role}`);
    if (status.model_id) parts.push(String(status.model_id));
    return parts.join(' • ');
}

function createCaptioningRunStatusChip(status) {
    if (!status) return null;
    const chip = document.createElement('span');
    chip.className = 'job-chip';
    chip.dataset.ui = 'captioning-run-status';
    chip.dataset.status = String(status.status || 'off');
    chip.dataset.sidecarCount = String(Number(status.sidecar_count) || 0);
    chip.dataset.rawCount = String(Number(status.raw_count) || 0);
    chip.dataset.proxyCount = String(Number(status.proxy_count) || 0);
    chip.dataset.failedCount = String(Number(status.failed_count) || 0);
    chip.textContent = captioningRunStatusSummary(status);
    chip.title = captioningRunStatusDetail(status);
    chip.setAttribute('aria-label', captioningRunStatusDetail(status));
    return chip;
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
        partial: Boolean(rawSummary.partial),
        captioning_status: normalizeCaptioningRunStatus(rawSummary.captioning_status)
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
        || normalized.partial
        || normalized.captioning_status;

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
    if (summary.captioning_status) {
        segments.push(captioningRunStatusSummary(summary.captioning_status));
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
            browser_previewable: Boolean(item.browser_previewable),
            content_type: typeof item.content_type === 'string' ? item.content_type : '',
            url: typeof item.url === 'string' ? item.url : '',
            download_url: typeof item.download_url === 'string' ? item.download_url : '',
            preview_url: typeof item.preview_url === 'string' ? item.preview_url : '',
            preview_mime_type: typeof item.preview_mime_type === 'string' ? item.preview_mime_type : '',
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
        existing.browser_previewable = typeof normalizedArtifact.browser_previewable === 'boolean' ? normalizedArtifact.browser_previewable : existing.browser_previewable;
        existing.content_type = normalizedArtifact.content_type || existing.content_type;
        existing.url = normalizedArtifact.url || existing.url;
        existing.download_url = normalizedArtifact.download_url || existing.download_url;
        existing.preview_url = normalizedArtifact.preview_url || existing.preview_url;
        existing.preview_mime_type = normalizedArtifact.preview_mime_type || existing.preview_mime_type;
        existing.relative_path = normalizedArtifact.relative_path || existing.relative_path;
        existing.size_bytes = normalizedArtifact.size_bytes ?? existing.size_bytes;
        existing.sha256 = normalizedArtifact.sha256 || existing.sha256;
        existing.display_hint = normalizedArtifact.display_hint || existing.display_hint || null;
    } else {
        job.artifacts.push(normalizedArtifact);
    }
    _clearArtifactUrlNotFoundCache();
    _reconcileJobTimeline(job);
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

function _latestVisibleTransportWarning(job) {
    const warnings = Array.isArray(job?.transportWarnings) ? job.transportWarnings : [];
    for (let index = warnings.length - 1; index >= 0; index -= 1) {
        const warning = warnings[index];
        if (warning && warning.tone !== 'info') return warning;
    }
    return null;
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
    if (els.copyArtifactFingerprintBtn) {
        els.copyArtifactFingerprintBtn.disabled = true;
        delete els.copyArtifactFingerprintBtn.dataset.fingerprint;
    }
}

function _reviewSurfaceDeferredEnabled() {
    return Boolean(_isBootstrapReady() && state.auth?.features?.reviewSurfaceDeferred);
}

function _reviewSurfaceAssetUrl(datasetKey) {
    const body = document.body;
    return body ? String(body.dataset?.[datasetKey] || '').trim() : '';
}

function _ensureDeferredReviewSurfaceCss() {
    if (deferredReviewSurfaceCssLoaded) return true;
    const href = _reviewSurfaceAssetUrl('reviewSurfaceCssUrl');
    if (!href) return false;
    const existing = document.querySelector('link[data-ui="portal-review-surface-css"]');
    if (existing) {
        deferredReviewSurfaceCssLoaded = true;
        return true;
    }
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    link.dataset.ui = 'portal-review-surface-css';
    document.head.appendChild(link);
    deferredReviewSurfaceCssLoaded = true;
    return true;
}

function _createDeferredReviewSurfaceHost() {
    return {
        state,
        els,
        clamp,
        normalizeRunSummary,
        captioningRunStatusSummary,
        titleCaseToken,
        formatRelativeTime,
        getReadableError,
        createToast,
        emitPortalEvent,
        _isJobsHydrationPending,
        _toggleSurfaceSkeleton,
        _resetArtifactActionButtons,
        _setSurfaceEmptyState,
        renderConsoleContextRibbon,
        renderReviewStatusActions,
        _syncConsoleRoute,
        _findJobById,
        _selectedArtifactForJob,
        rankArtifactsForDisplay,
        findCompareArtifact,
        buildArtifactUrl,
        sanitizeManagedAssetUrl,
        artifactIsPreviewable,
        artifactIsBrowserPreviewable,
        artifactPreviewSrc,
        _markArtifactUrlNotFound,
        _isArtifactUrlKnownMissing,
        artifactLabel,
        artifactDisplayHint,
        artifactDisplayLabel,
        artifactContentType,
        artifactFingerprint,
        artifactNameParts,
        formatBytes,
        _buildAuthHeaders,
        _artifactFingerprintLabel,
        _artifactRouteKey,
        _artifactViewerEnabled,
        _normalizeSelectedJobId,
        _normalizeArtifactRoutePath,
        _rememberArtifactSelection,
        _rememberOverlayTrigger,
        _restoreOverlayFocus,
        renderReviewSurfaces,
        _compareSurfaceCopy,
        _jobFreshnessLabel,
        _jobHasReviewableOutputs,
        _latestVisibleTransportWarning,
    };
}

function _deferredReviewSurfaceLoadRetryBlocked(now = Date.now()) {
    return deferredReviewSurfaceLoadFailedAt > 0 && (now - deferredReviewSurfaceLoadFailedAt) < DEFERRED_REVIEW_SURFACE_RETRY_WINDOW_MS;
}

function _clearDeferredReviewSurfaceLoadFailure() {
    deferredReviewSurfaceLoadFailedAt = 0;
    deferredReviewSurfaceLoadLastToastAt = 0;
}

function _noteDeferredReviewSurfaceLoadFailure() {
    const now = Date.now();
    deferredReviewSurfaceLoadFailedAt = now;
    if ((now - deferredReviewSurfaceLoadLastToastAt) < DEFERRED_REVIEW_SURFACE_RETRY_WINDOW_MS) return;
    deferredReviewSurfaceLoadLastToastAt = now;
    createToast('Review surfaces failed to load. Reload the portal and retry the review action.', 'error');
}

async function _loadDeferredReviewSurface() {
    if (deferredReviewSurfaceApi) return deferredReviewSurfaceApi;
    if (deferredReviewSurfaceLoadPromise) return deferredReviewSurfaceLoadPromise;
    if (_deferredReviewSurfaceLoadRetryBlocked()) return null;
    const moduleUrl = _reviewSurfaceAssetUrl('reviewSurfaceJsUrl');
    if (!moduleUrl) return null;
    _ensureDeferredReviewSurfaceCss();
    deferredReviewSurfaceLoadPromise = import(moduleUrl)
        .then((module) => {
            if (!module || typeof module.createDeferredReviewSurfaceApi !== 'function') {
                throw new Error('Deferred review surface module missing createDeferredReviewSurfaceApi');
            }
            _clearDeferredReviewSurfaceLoadFailure();
            deferredReviewSurfaceApi = module.createDeferredReviewSurfaceApi(_createDeferredReviewSurfaceHost());
            return deferredReviewSurfaceApi;
        })
        .catch((error) => {
            console.error('Failed to load deferred review surface', error);
            _noteDeferredReviewSurfaceLoadFailure();
            deferredReviewSurfaceApi = null;
            return null;
        })
        .finally(() => {
            deferredReviewSurfaceLoadPromise = null;
        });
    return deferredReviewSurfaceLoadPromise;
}

function _selectedOperateArtifactContext() {
    if (state.currentView !== 'operate') return false;
    const selectedJob = _findJobById(state.selectedJobId);
    return Boolean(selectedJob && Array.isArray(selectedJob.artifacts) && selectedJob.artifacts.length > 0);
}

function _shouldLoadDeferredReviewSurface(reason = '') {
    if (!_isBootstrapReady()) return false;
    if (!_reviewSurfaceDeferredEnabled()) return true;
    if (state.portalUi?.artifactViewer?.open) return true;
    if (state.currentView === 'review') return true;
    if (_selectedOperateArtifactContext()) return true;
    return String(reason || '').trim() === 'force';
}

function _primeDeferredReviewSurface(reason = '') {
    if (!_shouldLoadDeferredReviewSurface(reason) || _deferredReviewSurfaceLoadRetryBlocked()) return;
    void _loadDeferredReviewSurface().then((api) => {
        if (!api) return;
        api.renderArtifactPanel();
        api.renderArtifactViewer();
    });
}

function _deferredSurfaceEntry(name) {
    let entry = _deferredSurfaceState.get(name);
    if (!entry) {
        entry = { api: null, loadPromise: null, lastFailureAt: 0, lastToastAt: 0 };
        _deferredSurfaceState.set(name, entry);
    }
    return entry;
}

function _deferredSurfaceRetryBlocked(entry, now = Date.now()) {
    return entry.lastFailureAt > 0 && (now - entry.lastFailureAt) < DEFERRED_SURFACE_RETRY_WINDOW_MS;
}

async function loadDeferredSurface(name, hostFactory) {
    const descriptor = DEFERRED_SURFACE_REGISTRY[name];
    if (!descriptor) return null;
    const entry = _deferredSurfaceEntry(name);
    if (entry.api) return entry.api;
    if (entry.loadPromise) return entry.loadPromise;
    if (_deferredSurfaceRetryBlocked(entry)) return null;
    const moduleUrl = String(document.body?.dataset?.[descriptor.datasetKey] || '').trim();
    if (!moduleUrl) return null;
    entry.loadPromise = import(moduleUrl)
        .then((module) => {
            const factory = module?.[descriptor.factoryName];
            if (typeof factory !== 'function') {
                throw new Error(`Deferred surface "${name}" missing factory ${descriptor.factoryName}`);
            }
            entry.lastFailureAt = 0;
            entry.api = factory(typeof hostFactory === 'function' ? hostFactory() : {});
            return entry.api;
        })
        .catch((error) => {
            console.error(`Failed to load deferred surface "${name}"`, error);
            const now = Date.now();
            entry.lastFailureAt = now;
            if ((now - entry.lastToastAt) >= DEFERRED_SURFACE_RETRY_WINDOW_MS) {
                entry.lastToastAt = now;
                createToast(`Failed to load the ${name} surface. Reload the portal to retry.`, 'error');
            }
            entry.api = null;
            return null;
        })
        .finally(() => {
            entry.loadPromise = null;
        });
    return entry.loadPromise;
}

function _createDeferredOperateSurfaceHost() {
    return {
        state,
        els,
        portalRenderScheduler,
        EVENT_SOURCE_READY_STATE_CONNECTING,
        EVENT_SOURCE_READY_STATE_OPEN,
        EVENT_SOURCE_READY_STATE_CLOSED,
        _displayJobState,
        _displayJobStateTone,
        _isJobsHydrationPending,
        _jobFreshnessLabel,
        _jobHasActiveStream,
        _latestVisibleTransportWarning,
        _nativeEventSourceReadyState,
        _portalPrivilegesReady,
        _reconcileJobTimeline,
        _setSurfaceEmptyState,
        _toggleSurfaceSkeleton,
        createCaptioningRunStatusChip,
        formatDuration,
        formatRelativeTime,
        formatTimelineTimestamp,
        formatTransportLabel,
        getReadableError,
        jobOutcomeSummary,
        normalizeRunSummary,
        renderConsoleContextRibbon,
        renderReviewSurfaces,
        renderSelectedJobRecoveryActions,
        titleCaseToken,
    };
}

function _deferredOperateSurfaceApi() {
    return _deferredSurfaceState.get('operate')?.api || null;
}

function _loadDeferredOperateSurface() {
    return loadDeferredSurface('operate', _createDeferredOperateSurfaceHost);
}

function _isOperateQueuePanelVisible() {
    return Boolean(
        els.queueShell
        && !els.queueShell.classList.contains('hidden')
        && els.queueShell.getAttribute('aria-hidden') !== 'true'
    );
}

function _shouldLoadDeferredOperateSurface() {
    if (!_isBootstrapReady()) return false;
    return state.currentView === 'operate' || state.currentView === 'review' || _isOperateQueuePanelVisible();
}

function _primeDeferredOperateSurface() {
    if (!_shouldLoadDeferredOperateSurface()) return;
    void _loadDeferredOperateSurface();
}

function _createDeferredBuildSurfaceHost() {
    return {
        state,
        els,
        _metadataField,
        _normalizeWorkerMode,
        _previewIssueForField,
        _renderIssueStatus,
        _resolveDa3ModelKey,
        canonicalArchiveCommand,
        generatePayload,
    };
}

function _deferredBuildSurfaceApi() {
    return _deferredSurfaceState.get('build')?.api || null;
}

function _loadDeferredBuildSurface() {
    return loadDeferredSurface('build', _createDeferredBuildSurfaceHost);
}

function _shouldLoadDeferredBuildSurface() {
    if (!_isBootstrapReady()) return false;
    return state.currentView === 'build';
}

function _reconcileDeferredBuildSurface(api = _deferredBuildSurfaceApi()) {
    if (!api) return;
    if (api.applyLuxMetadataToControls) api.applyLuxMetadataToControls();
    if (api.renderFieldPreviewStatuses) api.renderFieldPreviewStatuses();
    if (api.syncRuntimeWorkerModeControls) api.syncRuntimeWorkerModeControls();
    if (api.refreshArchiveFieldVisibility) api.refreshArchiveFieldVisibility();
}

function _primeDeferredBuildSurface() {
    if (!_shouldLoadDeferredBuildSurface()) return;
    void _loadDeferredBuildSurface().then((loaded) => {
        _reconcileDeferredBuildSurface(loaded);
    });
}

function _renderDeferredReviewSurfaceFallback(jobsLoading = false) {
    _toggleSurfaceSkeleton(els.artifactsShell, els.artifactShellContent, els.artifactSkeletonState, jobsLoading);
    if (jobsLoading) {
        _resetArtifactActionButtons();
        if (els.artifactMeta) els.artifactMeta.textContent = 'Hydrating artifacts';
        renderConsoleContextRibbon();
        return;
    }
    const reviewSurfaceLoadBlocked = _deferredReviewSurfaceLoadRetryBlocked();
    if (els.artifactMeta) {
        if (reviewSurfaceLoadBlocked) {
            els.artifactMeta.textContent = 'Review surface unavailable';
        } else {
            els.artifactMeta.textContent = _shouldLoadDeferredReviewSurface() ? 'Loading review surface' : 'Review surface deferred';
        }
    }
    if (els.artifactThumbnailRail) {
        els.artifactThumbnailRail.setAttribute('role', 'listbox');
        els.artifactThumbnailRail.setAttribute('aria-label', 'Artifact thumbnails');
        els.artifactThumbnailRail.innerHTML = '';
    }
    if (els.emptyArtifactState) els.emptyArtifactState.style.display = 'block';
    if (reviewSurfaceLoadBlocked) {
        _setSurfaceEmptyState(els.emptyArtifactState, els.emptyArtifactTitle, els.emptyArtifactDetail, {
            tone: 'warning',
            title: 'Review surface unavailable',
            detail: 'Reload the portal to retry loading the review surface assets for this artifact context.',
        });
        if (els.emptyArtifactAction) {
            els.emptyArtifactAction.textContent = 'Next action: reload the portal before reopening review.';
        }
    }
    if (els.reviewStatusBanner) els.reviewStatusBanner.classList.add('hidden');
    if (els.reviewProvenanceGrid) els.reviewProvenanceGrid.classList.add('hidden');
    if (els.reviewCompareSummary) els.reviewCompareSummary.classList.add('hidden');
    _resetArtifactActionButtons();
    renderConsoleContextRibbon();
}

function renderReviewSurfaces(payload = null) {
    const currentPayload = payload || generatePayload();
    renderArtifactPanel();
    renderArtifactViewer();
    renderSelectedJobInspector();
    renderReconstructionRuntimeSummary(currentPayload);
    renderMissionControl(payload);
}

function renderArtifactPanel() {
    const jobsLoading = _isJobsHydrationPending();
    if (deferredReviewSurfaceApi) {
        deferredReviewSurfaceApi.renderArtifactPanel();
        return;
    }
    _primeDeferredReviewSurface('render');
    _primeDeferredOperateSurface();
    _primeDeferredBuildSurface();
    _renderDeferredReviewSurfaceFallback(jobsLoading);
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
    _primeDeferredReviewSurface('route');
    _primeDeferredOperateSurface();
    _primeDeferredBuildSurface();
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
    return portalInternals.isNativeEventSourceHandle(handle);
}

function _nativeEventSourceReadyState(handle) {
    return portalInternals.nativeEventSourceReadyState(handle);
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
    if (_isProtectedFamilySuppressed('jobs_detail')) return;
    try {
        const headers = _buildAuthHeaders({ 'Accept': 'application/json' });
        const encodedId = encodeURIComponent(String(job.id));
        const res = await fetch(`${API_BASE}/v1/jobs/${encodedId}`, { headers, cache: 'no-store' });
        if (!res.ok) {
            await _maybeSuppressOnProtectedResponse('jobs_detail', res);
            return;
        }
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
    if (_isProtectedFamilySuppressed('jobs_events')) {
        job.reconnectBlocked = true;
        _clearSseRetry(job, true);
        return;
    }
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
    const api = _deferredOperateSurfaceApi();
    if (api?.scheduleRenderJobQueue) {
        api.scheduleRenderJobQueue(includeReviewSurfaces);
        return;
    }
    if (!_shouldLoadDeferredOperateSurface()) return;
    _operatePendingIncludeReview = _operatePendingIncludeReview || includeReviewSurfaces;
    if (_operatePendingScheduleRender) return;
    _operatePendingScheduleRender = true;
    void _loadDeferredOperateSurface().then((loaded) => {
        const flag = _operatePendingIncludeReview;
        _operatePendingScheduleRender = false;
        _operatePendingIncludeReview = false;
        if (loaded?.scheduleRenderJobQueue) loaded.scheduleRenderJobQueue(flag);
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

function _portalRumNow() {
    if (window.performance && typeof window.performance.now === 'function') {
        return window.performance.now();
    }
    return Date.now();
}

function _portalRumTraceparent(fallback = '') {
    return portalInternals.normalizePortalRumTraceparent(
        state.rum.bootstrapTraceparent || state.rum.pageTraceparent,
        fallback
    );
}

function _rumTelemetryEnabled() {
    return Boolean(_isBootstrapReady() && state.auth?.features?.rumTelemetry);
}

function _portalRumBasePayload(sample = {}) {
    const sampleOptions = sample && typeof sample === 'object' ? sample : {};
    return {
        event_type: String(sampleOptions.eventType || '').trim().toLowerCase(),
        route: '/portal',
        view: portalInternals.normalizePortalRumView(sampleOptions.view || state.currentView),
        value: Number(sampleOptions.value),
        unit: String(sampleOptions.unit || '').trim().toLowerCase(),
        metric: String(sampleOptions.metric || '').trim().toLowerCase(),
        metadata: sampleOptions.metadata && typeof sampleOptions.metadata === 'object' ? sampleOptions.metadata : {},
        traceparent: portalInternals.normalizePortalRumTraceparent(
            sampleOptions.traceparent,
            _portalRumTraceparent()
        ),
        keepalive: Boolean(sampleOptions.keepalive)
    };
}

function _queuePortalRumSample(sample = {}) {
    if (_isBootstrapReady() && !state.auth?.features?.rumTelemetry) return;
    const payload = _portalRumBasePayload(sample);
    if (!payload.event_type || !Number.isFinite(payload.value) || !payload.unit) return;
    state.rum.queuedSamples.push(payload);
    if (_rumTelemetryEnabled()) {
        void _flushQueuedPortalRumSamples();
    }
}

async function _flushQueuedPortalRumSamples(options = {}) {
    if (!_rumTelemetryEnabled()) {
        state.rum.queuedSamples = [];
        return;
    }
    if (state.rum.queuedSamples.length === 0) {
        return;
    }
    if (_isProtectedFamilySuppressed('portal_rum')) {
        // Drop the queued samples; they will not be accepted while the
        // frontdoor is in non-retryable config_failure state.
        state.rum.queuedSamples.splice(0, state.rum.queuedSamples.length);
        return;
    }
    const flushOptions = options && typeof options === 'object' ? options : {};
    const keepalive = Boolean(flushOptions.keepalive);
    const queued = state.rum.queuedSamples.splice(0, state.rum.queuedSamples.length);
    for (const sample of queued) {
        try {
            const headers = _buildAuthHeaders({ 'Content-Type': 'application/json' }, 'POST', {
                traceparent: sample.traceparent
            });
            const res = await fetch(`${API_BASE}/v1/portal/rum`, {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    event_type: sample.event_type,
                    route: sample.route,
                    view: sample.view,
                    value: sample.value,
                    unit: sample.unit,
                    metric: sample.metric,
                    metadata: sample.metadata
                }),
                keepalive: keepalive || sample.keepalive
            });
            if (res && !res.ok) {
                await _maybeSuppressOnProtectedResponse('portal_rum', res);
            }
        } catch {
            // best-effort telemetry only
        }
    }
}

function _recordPortalRumMilestone(eventType, value, options = {}) {
    if (state.rum.emittedMilestones[eventType]) return;
    state.rum.emittedMilestones[eventType] = true;
    _queuePortalRumSample({
        eventType,
        value,
        unit: 'ms',
        metric: 'duration',
        traceparent: options.traceparent || _portalRumTraceparent(),
        view: options.view,
        metadata: options.metadata
    });
}

function _scheduleFirstViewInteractiveRum() {
    if (state.rum.firstInteractiveScheduled || state.rum.emittedMilestones.first_view_interactive) {
        return;
    }
    state.rum.firstInteractiveScheduled = true;
    const emit = () => {
        state.rum.firstInteractiveScheduled = false;
        _recordPortalRumMilestone('first_view_interactive', _portalRumNow(), {
            traceparent: _portalRumTraceparent()
        });
    };
    if (window.requestAnimationFrame) {
        window.requestAnimationFrame(emit);
        return;
    }
    window.setTimeout(emit, 0);
}

function _finalizePortalRumVitals() {
    if (state.rum.vitals.finalized) return;
    state.rum.vitals.finalized = true;
    if (Number.isFinite(state.rum.vitals.lcpMs)) {
        _queuePortalRumSample({
            eventType: 'core_web_vital',
            metric: 'lcp',
            value: state.rum.vitals.lcpMs,
            unit: 'ms',
            traceparent: _portalRumTraceparent(),
            keepalive: true
        });
    }
    if (Number.isFinite(state.rum.vitals.inpMs)) {
        _queuePortalRumSample({
            eventType: 'core_web_vital',
            metric: 'inp',
            value: state.rum.vitals.inpMs,
            unit: 'ms',
            traceparent: _portalRumTraceparent(),
            keepalive: true
        });
    }
    _queuePortalRumSample({
        eventType: 'core_web_vital',
        metric: 'cls',
        value: state.rum.vitals.clsScore,
        unit: 'score',
        traceparent: _portalRumTraceparent(),
        keepalive: true
    });
}

function _flushPortalRumOnPagehide() {
    _finalizePortalRumVitals();
    if (_rumTelemetryEnabled()) {
        void _flushQueuedPortalRumSamples({ keepalive: true });
    }
}

function _startPortalRumObservers() {
    if (state.rum.observersStarted || typeof window.PerformanceObserver !== 'function') return;
    state.rum.observersStarted = true;
    const supportedTypes = Array.isArray(window.PerformanceObserver.supportedEntryTypes)
        ? new Set(window.PerformanceObserver.supportedEntryTypes)
        : new Set();
    if (supportedTypes.has('largest-contentful-paint')) {
        const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const latest = entries[entries.length - 1];
            if (latest) {
                state.rum.vitals.lcpMs = latest.startTime;
            }
        });
        observer.observe({ type: 'largest-contentful-paint', buffered: true });
        window.addEventListener('pagehide', () => observer.disconnect(), { once: true });
    }
    if (supportedTypes.has('layout-shift')) {
        const observer = new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
                if (!entry.hadRecentInput) {
                    state.rum.vitals.clsScore = Number((state.rum.vitals.clsScore + entry.value).toFixed(4));
                }
            });
        });
        observer.observe({ type: 'layout-shift', buffered: true });
        window.addEventListener('pagehide', () => observer.disconnect(), { once: true });
    }
    if (supportedTypes.has('event')) {
        const observer = new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
                if (Number(entry.interactionId) > 0) {
                    state.rum.vitals.inpMs = Math.max(Number(state.rum.vitals.inpMs) || 0, entry.duration || 0);
                }
            });
        });
        observer.observe({ type: 'event', buffered: true, durationThreshold: 16 });
        window.addEventListener('pagehide', () => observer.disconnect(), { once: true });
    }
}

const TRUTHY_VALUES = new Set(['1', 'true', 'yes', 'on']);
const FALSY_VALUES = new Set(['0', 'false', 'no', 'off']);
const LUX_QUALITY_TIERS = new Set(['standard', 'premium', 'apex']);
const LUX_DEPTH_BACKENDS = new Set(['da3', 'depth_pro']);
const LUX_DA3_MODEL_KEYS = new Set(['da3-metric', 'da3-research']);
const LUX_SEGMENTATION_BACKENDS = new Set(['stub', 'efficientsam', 'sam2']);
const SAM2_MODEL_SIZES = new Set(['base', 'large']);
const LUX_GROUPING_MODES = new Set(['single', 'parent_dir']);
const VLM_CAPTIONING_PROXY_FORMATS = new Set(['png', 'jpeg']);

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

function _resolveDa3ModelKey(value) {
    const normalized = String(value || '').trim().toLowerCase().replace(/_/g, '-');
    if (LUX_DA3_MODEL_KEYS.has(normalized)) return normalized;
    if (normalized === 'da3') return 'da3-research';
    return 'da3-metric';
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

function _resolveVlmCaptioningProxyFormat(value) {
    const normalized = String(value || '').trim().toLowerCase();
    if (VLM_CAPTIONING_PROXY_FORMATS.has(normalized)) return normalized;
    return 'png';
}

function _resolveRunCardVersion(value) {
    const normalized = String(value || '').trim().toLowerCase();
    return normalized === 'v2' ? 'v2' : 'v1';
}

function _parsePositiveIntOrNull(value) {
    if (value === null || value === undefined) return null;
    const text = String(value).trim();
    if (!text) return null;
    const parsed = Number.parseInt(text, 10);
    if (!Number.isFinite(parsed) || parsed < 1) return null;
    return parsed;
}

function _parseNonNegativeIntOrNull(value) {
    if (value === null || value === undefined) return null;
    const text = String(value).trim();
    if (!text) return null;
    const parsed = Number.parseInt(text, 10);
    if (!Number.isFinite(parsed) || parsed < 0) return null;
    return parsed;
}

function _parseProbabilityOrNull(value) {
    if (value === null || value === undefined) return null;
    const text = String(value).trim();
    if (!text) return null;
    const parsed = Number.parseFloat(text);
    if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) return null;
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
    if (!els.segmentation.backend) return;
    const segmentation = config?.segmentation || {};
    const segmentationEnabled = parseBoolLike(segmentation.enable, false);
    const backend = _resolveSegmentationBackend(segmentation.backend);
    const sam2Active = segmentationEnabled && backend === 'sam2';
    const sam2TilingEnabled = parseBoolLike(segmentation.sam2TilingEnabled, false);

    if (els.segmentation.sam2ModelSize) {
        els.segmentation.sam2ModelSize.disabled = !sam2Active;
    }
    if (els.segmentation.sam2CheckpointPath) {
        els.segmentation.sam2CheckpointPath.disabled = !sam2Active;
    }
    if (els.segmentation.sam2TilingEnabled) {
        els.segmentation.sam2TilingEnabled.disabled = !sam2Active;
    }
    if (els.sam2TuningPanel) {
        els.sam2TuningPanel.classList.toggle('hidden', !sam2Active);
    }
    if (els.sam2TilingConfigFields) {
        els.sam2TilingConfigFields.classList.toggle('hidden', !sam2Active || !sam2TilingEnabled);
    }
    if (els.sam2GeneratorConfigFields) {
        els.sam2GeneratorConfigFields.classList.toggle('hidden', !sam2Active);
    }
    [
        els.segmentation.sam2TileSizePx,
        els.segmentation.sam2OverlapPx,
        els.segmentation.sam2GlobalPassLongestSide,
        els.segmentation.sam2MaxConcurrency,
    ].forEach((field) => {
        if (field) field.disabled = !sam2Active || !sam2TilingEnabled;
    });
    [
        els.segmentation.sam2PointsPerSide,
        els.segmentation.sam2PointsPerBatch,
        els.segmentation.sam2PredIouThresh,
        els.segmentation.sam2StabilityScoreThresh,
        els.segmentation.sam2CropNLayers,
    ].forEach((field) => {
        if (field) field.disabled = !sam2Active;
    });
}

function syncRunCardControlState(config) {
    const emits = config?.emits || {};
    const runCardEnabled = parseBoolLike(emits.runCard, false);
    if (els.runCardVersionField) {
        els.runCardVersionField.classList.toggle('opacity-60', !runCardEnabled);
    }
    if (els.emits.runCardVersion) {
        els.emits.runCardVersion.disabled = !runCardEnabled;
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
    if (parseBoolLike(normalizedArgs.vlm_captioning_enabled, false)) {
        runtimeScore += 1;
        reasons.push('fastvlm_captioning');
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

function _buildLocalCaptioningSummary(args = {}) {
    const enabled = parseBoolLike(args.vlm_captioning_enabled, false);
    const runtimeStatus = enabled ? 'unknown' : 'off';
    return {
        feature_enabled: _fastVlmCaptioningFeatureEnabled(),
        enabled,
        backend: String(args.vlm_captioning_backend || 'fastvlm'),
        model: String(args.vlm_captioning_model || 'default'),
        proxy_format: String(args.vlm_captioning_proxy_format || 'png'),
        max_side_px: Number.parseInt(String(args.vlm_captioning_max_side_px || '1600'), 10) || 1600,
        fastvlm_python_executable: String(args.fastvlm_python_executable || ''),
        fastvlm_mlx_vlm_dir: String(args.fastvlm_mlx_vlm_dir || ''),
        timeout_seconds: Number.parseInt(String(args.fastvlm_timeout_seconds || '180'), 10) || 180,
        runtime_path_status: {},
        runtime_readiness: {
            status: runtimeStatus,
            checks: {},
            verification_scope: 'path-existence'
        },
        runtime_status: runtimeStatus,
        role: 'advisory',
        used_for_quality_gate: false
    };
}

function _captioningRuntimeReadiness(summary = {}) {
    const readiness = summary && typeof summary.runtime_readiness === 'object' && summary.runtime_readiness
        ? summary.runtime_readiness
        : {};
    const fallbackStatus = String(summary?.runtime_status || (parseBoolLike(summary?.enabled, false) ? 'unknown' : 'off')).trim();
    return {
        status: String(readiness.status || fallbackStatus || 'off').trim(),
        checks: readiness.checks && typeof readiness.checks === 'object' ? readiness.checks : {},
        verification_scope: String(readiness.verification_scope || 'path-existence').trim()
    };
}

function _captioningRuntimeLabel(status) {
    const normalized = String(status || '').trim();
    if (normalized === 'ready') return 'Ready';
    if (normalized === 'missing_runtime') return 'Missing runtime';
    if (normalized === 'invalid_config') return 'Invalid config';
    if (normalized === 'off') return 'Off';
    if (normalized === 'unknown') return 'Preview pending';
    return normalized || 'Off';
}

function _captioningCheckLabel(name) {
    const normalized = String(name || '').trim();
    if (normalized === 'python_executable') return 'Python';
    if (normalized === 'mlx_vlm_dir') return 'mlx-vlm';
    if (normalized === 'model_path') return 'Model';
    return normalized.replace(/_/g, ' ');
}

function _renderCaptioningReadiness(summary = {}, options = {}) {
    const visible = Boolean(options.visible);
    const enabled = Boolean(options.enabled);
    const readiness = _captioningRuntimeReadiness(summary);
    if (els.captioning.readinessScope) {
        els.captioning.readinessScope.textContent = visible
            ? `FastVLM readiness: ${_captioningRuntimeLabel(readiness.status)} · ${readiness.verification_scope}`
            : 'FastVLM readiness is hidden for this portal cohort.';
    }
    if (!els.captioning.readinessList) return;
    els.captioning.readinessList.dataset.status = String(readiness.status || 'off');
    els.captioning.readinessList.replaceChildren();
    if (!visible || !enabled) {
        return;
    }
    const entries = Object.entries(readiness.checks || {});
    if (entries.length === 0) {
        const item = document.createElement('li');
        item.textContent = 'Preview-backed readiness details are pending.';
        els.captioning.readinessList.appendChild(item);
        return;
    }
    entries.forEach(([name, check]) => {
        const item = document.createElement('li');
        item.dataset.status = String(check?.status || 'unknown');
        const label = _captioningCheckLabel(name);
        const status = _captioningRuntimeLabel(check?.status || 'unknown');
        const path = String(check?.path || '').trim();
        const remediation = String(check?.remediation || '').trim();
        item.textContent = `${label}: ${status}${path ? ` · ${path}` : ''}${remediation ? ` · ${remediation}` : ''}`;
        els.captioning.readinessList.appendChild(item);
    });
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
        captioning_summary: null,
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
        captioning_summary: _buildLocalCaptioningSummary(currentPayload.args || {}),
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

function _metadataField(name) {
    const fields = state.metadata && state.metadata.fields && typeof state.metadata.fields === 'object'
        ? state.metadata.fields
        : {};
    const value = fields[name];
    return value && typeof value === 'object' ? value : null;
}

function applyLuxMetadataToControls() {
    const api = _deferredBuildSurfaceApi();
    if (api?.applyLuxMetadataToControls) {
        api.applyLuxMetadataToControls();
        return;
    }
    if (!_shouldLoadDeferredBuildSurface()) return;
    void _loadDeferredBuildSurface().then((loaded) => {
        if (loaded?.applyLuxMetadataToControls) loaded.applyLuxMetadataToControls();
    });
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

function renderFieldPreviewStatuses(payload = null) {
    const api = _deferredBuildSurfaceApi();
    if (api?.renderFieldPreviewStatuses) {
        api.renderFieldPreviewStatuses(payload);
        return;
    }
    if (!_shouldLoadDeferredBuildSurface()) return;
    void _loadDeferredBuildSurface().then((loaded) => {
        if (loaded?.renderFieldPreviewStatuses) loaded.renderFieldPreviewStatuses(payload);
    });
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
            || Boolean(summary?.reviewable_outputs)
            || job.state === 'succeeded'
            || job.state === 'partial';
    });
}

function syncRuntimeWorkerModeControls() {
    const api = _deferredBuildSurfaceApi();
    if (api?.syncRuntimeWorkerModeControls) {
        api.syncRuntimeWorkerModeControls();
        return;
    }
    if (!_shouldLoadDeferredBuildSurface()) return;
    void _loadDeferredBuildSurface().then((loaded) => {
        if (loaded?.syncRuntimeWorkerModeControls) loaded.syncRuntimeWorkerModeControls();
    });
}

async function emitPortalEvent(eventType, options = {}) {
    if (!state.backendOk || !_isBootstrapReady()) return;
    if (_isProtectedFamilySuppressed('portal_events')) return;
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
        const res = await fetch(`${API_BASE}/v1/portal/events`, {
            method: 'POST',
            headers,
            body: JSON.stringify(payload)
        });
        if (res && !res.ok) {
            await _maybeSuppressOnProtectedResponse('portal_events', res);
        }
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
            advanced_sections: [],
            backend_catalog: {},
            model_catalog: {},
        };
        renderReviewSurfaces();
        return;
    }
    if (_isProtectedFamilySuppressed('config_metadata')) return;
    try {
        const headers = _buildAuthHeaders({ 'Accept': 'application/json' });
        const res = await fetch(`${API_BASE}/v1/config-metadata?pipeline=${encodeURIComponent(pipelineName)}`, {
            headers,
            cache: 'no-store'
        });
        if (!res.ok) {
            await _maybeSuppressOnProtectedResponse('config_metadata', res);
            throw new Error(`config metadata fetch failed (${res.status})`);
        }
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
            advanced_sections: Array.isArray(data.advanced_sections) ? data.advanced_sections.map((item) => String(item || '')) : [],
            backend_catalog: data.backend_catalog && typeof data.backend_catalog === 'object' ? data.backend_catalog : {},
            model_catalog: data.model_catalog && typeof data.model_catalog === 'object' ? data.model_catalog : {},
        };
        applyLuxMetadataToControls();
        renderReviewSurfaces();
        scheduleConfigPreview(true);
    } catch {
        if (!silent) createToast('Failed to refresh config metadata from backend.', 'error');
    }
}

function _clearConfigPreviewServiceRetry() {
    if (configPreviewServiceRetryTimerId !== null) {
        clearTimeout(configPreviewServiceRetryTimerId);
        configPreviewServiceRetryTimerId = null;
    }
    configPreviewServiceRetryAttempts = 0;
    configPreviewLastRateLimitHint = null;
}

function _scheduleConfigPreviewServiceRetry(rateLimitHint = null) {
    configPreviewLastRateLimitHint = rateLimitHint && Number(rateLimitHint?.retryAtMs) > 0
        ? rateLimitHint
        : configPreviewLastRateLimitHint;
    if (configPreviewServiceRetryAttempts >= CONFIG_PREVIEW_SERVICE_RETRY_MAX_ATTEMPTS) {
        return;
    }
    if (configPreviewServiceRetryTimerId !== null) {
        return;
    }
    configPreviewServiceRetryAttempts += 1;
    let delay = CONFIG_PREVIEW_SERVICE_RETRY_BASE_MS * configPreviewServiceRetryAttempts;
    if (
        rateLimitHint
        && Number.isFinite(Number(rateLimitHint.retryAfterMs))
        && Number(rateLimitHint.retryAfterMs) > 0
    ) {
        // Clamp to the existing scheduled-window upper bound so a buggy
        // upstream cannot stretch a single preview retry indefinitely.
        const maxScheduledDelay = CONFIG_PREVIEW_SERVICE_RETRY_BASE_MS * CONFIG_PREVIEW_SERVICE_RETRY_MAX_ATTEMPTS;
        delay = Math.min(Number(rateLimitHint.retryAfterMs), maxScheduledDelay);
    }
    configPreviewServiceRetryTimerId = window.setTimeout(() => {
        configPreviewServiceRetryTimerId = null;
        // Always retry against the *current* form state, not a stale
        // snapshot — the cooldown may outlive the user's edit so the next
        // attempt should reflect what is on screen now.
        const payload = generatePayload();
        if (!_configPreviewEnabledForPipeline(payload.pipeline)) {
            return;
        }
        void fetchConfigPreview(payload);
    }, delay);
}

async function fetchConfigPreview(payload) {
    if (_isProtectedFamilySuppressed('config_preview')) {
        _clearConfigPreviewServiceRetry();
        return;
    }
    const currentPayload = payload && typeof payload === 'object' ? payload : generatePayload();
    const requestKey = _configPreviewRequestKey(currentPayload);
    const refreshPreviewDrivenSurfaces = (nextPayload = currentPayload) => {
        renderCLI();
        renderPreRunDiagnostics(nextPayload);
        _syncBootstrapGuardedControls();
    };

    if (!_configPreviewEnabledForPipeline(currentPayload.pipeline)) {
        _clearConfigPreviewServiceRetry();
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
        if (!res.ok) {
            await _maybeSuppressOnProtectedResponse('config_preview', res);
        }
        const response = await res.json();
        if (!res.ok) {
            const errorPayload = response?.error && typeof response.error === 'object' ? response.error : {};
            const errorDetails = errorPayload.details && typeof errorPayload.details === 'object' ? errorPayload.details : {};
            // 429 is a transient rate-limit event, not a 4xx validation
            // failure of the user's draft. Route it to the service-retry
            // path so the existing scheduled retry runs against the
            // current form state, with the upstream Retry-After hint.
            const isRateLimited = res.status === 429;
            const rateLimitHint = isRateLimited
                ? portalInternals.parseRateLimitRetryHint(res)
                : null;
            const classifiedFailure = _previewFailureDetails({
                error_reason: res.status === 401 || res.status === 403
                    ? 'auth_failure'
                    : isRateLimited
                        ? 'service_failure'
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
            if (classifiedFailure.reason === 'service_failure') {
                _scheduleConfigPreviewServiceRetry(rateLimitHint);
            } else {
                _clearConfigPreviewServiceRetry();
            }
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
            captioning_summary: data.captioning_summary && typeof data.captioning_summary === 'object' ? data.captioning_summary : null,
            next_best_action: _normalizeNextBestAction(data.next_best_action),
            argv_preview: String(data.argv_preview || ''),
            error: '',
            error_reason: '',
            error_status: 0
        };
        _setPreviewState(_reconcilePreviewRepairedPaths(nextPreviewState));
        _clearConfigPreviewServiceRetry();
        if (previewFieldErrors.length > 0) {
            void emitPortalEvent('preview_error_seen', {
                surface: 'reconstruction_runtime',
                reasons: previewFieldErrors.map((item) => String(item?.code || '')).filter(Boolean).slice(0, 8),
                metadata: { count: previewFieldErrors.length }
            });
        }
    } catch {
        if (_configPreviewRequestKey(generatePayload()) !== requestKey) {
            return;
        }
        const classifiedFailure = _previewFailureDetails({ error_reason: 'service_failure' });
        _setPreviewState({
            ..._emptyPreviewState('error', currentPayload.pipeline),
            requestKey,
            error: 'preview_unavailable',
            error_reason: classifiedFailure.reason,
            error_status: 0
        });
        _scheduleConfigPreviewServiceRetry();
    } finally {
        refreshPreviewDrivenSurfaces(generatePayload());
    }
}

function scheduleConfigPreview(immediate = false) {
    if (configPreviewTimerId !== null) {
        clearTimeout(configPreviewTimerId);
        configPreviewTimerId = null;
    }
    _clearConfigPreviewServiceRetry();
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
            els.reconstructionSummaryHint.textContent = 'Preview-backed validation, normalization, and runtime estimates reflect the next dispatch.';
        } else if (previewStatus === 'error') {
            els.reconstructionSummaryHint.textContent = 'Preview-backed validation is unavailable right now, so posture is shown from the current draft while dispatch stays paused.';
        } else if (!state.backendOk) {
            els.reconstructionSummaryHint.textContent = 'Backend preview is unavailable while the orchestrator is offline, so posture is shown from the current local draft.';
        } else {
            els.reconstructionSummaryHint.textContent = 'Primary run posture updates here before you open contextual runtime or research controls.';
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
        const captioningSummary = effectivePreview.captioning_summary || _buildLocalCaptioningSummary(currentPayload.args || {});
        const captioningReadiness = _captioningRuntimeReadiness(captioningSummary);
        const captioningReadinessLabel = _captioningRuntimeLabel(captioningReadiness.status).toLowerCase();
        const captioningNote = parseBoolLike(captioningSummary.enabled, false)
            ? ` FastVLM advisory captioning is enabled and remains outside quality gates. FastVLM readiness is ${captioningReadinessLabel} (${captioningReadiness.verification_scope}).`
            : '';
        els.effectiveConfigMeta.textContent = effectivePreview.status === 'ready'
            ? `Preview-backed normalization is live. ${inactiveCount} inactive preserved field${inactiveCount === 1 ? '' : 's'} are tracked for the next run.${captioningNote}`
            : `Preview-backed normalization is unavailable, so this drawer is showing the local requested configuration and fallback posture.${captioningNote}`;
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
    const modelKey = _resolveDa3ModelKey(
        _textOrFallback(els.modelKey ? els.modelKey.value : config.modelKey, config.modelKey || 'da3-metric')
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
    const sam2Active = segmentationEnable && segmentationBackend === 'sam2';
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
    const sam2TilingEnabled = els.segmentation.sam2TilingEnabled
        ? Boolean(els.segmentation.sam2TilingEnabled.checked)
        : parseBoolLike(config.segmentation?.sam2TilingEnabled, false);
    const sam2TileSizePx = _parsePositiveIntOrNull(
        els.segmentation.sam2TileSizePx ? els.segmentation.sam2TileSizePx.value : config.segmentation?.sam2TileSizePx
    );
    const sam2OverlapPx = _parseNonNegativeIntOrNull(
        els.segmentation.sam2OverlapPx ? els.segmentation.sam2OverlapPx.value : config.segmentation?.sam2OverlapPx
    );
    const sam2GlobalPassLongestSide = _parsePositiveIntOrNull(
        els.segmentation.sam2GlobalPassLongestSide
            ? els.segmentation.sam2GlobalPassLongestSide.value
            : config.segmentation?.sam2GlobalPassLongestSide
    );
    const sam2MaxConcurrency = _parsePositiveIntOrNull(
        els.segmentation.sam2MaxConcurrency ? els.segmentation.sam2MaxConcurrency.value : config.segmentation?.sam2MaxConcurrency
    );
    const sam2PointsPerSide = _parsePositiveIntOrNull(
        els.segmentation.sam2PointsPerSide ? els.segmentation.sam2PointsPerSide.value : config.segmentation?.sam2PointsPerSide
    );
    const sam2PointsPerBatch = _parsePositiveIntOrNull(
        els.segmentation.sam2PointsPerBatch ? els.segmentation.sam2PointsPerBatch.value : config.segmentation?.sam2PointsPerBatch
    );
    const sam2PredIouThresh = _parseProbabilityOrNull(
        els.segmentation.sam2PredIouThresh ? els.segmentation.sam2PredIouThresh.value : config.segmentation?.sam2PredIouThresh
    );
    const sam2StabilityScoreThresh = _parseProbabilityOrNull(
        els.segmentation.sam2StabilityScoreThresh
            ? els.segmentation.sam2StabilityScoreThresh.value
            : config.segmentation?.sam2StabilityScoreThresh
    );
    const sam2CropNLayers = _parseNonNegativeIntOrNull(
        els.segmentation.sam2CropNLayers ? els.segmentation.sam2CropNLayers.value : config.segmentation?.sam2CropNLayers
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
    const emitRunCardIncludeProofs = els.emits.runCardIncludeProofs
        ? Boolean(els.emits.runCardIncludeProofs.checked)
        : parseBoolLike(config.emits?.runCardIncludeProofs, false);
    const runCardVersion = _resolveRunCardVersion(
        _textOrFallback(
            els.emits.runCardVersion ? els.emits.runCardVersion.value : config.emits?.runCardVersion,
            config.emits?.runCardVersion || 'v1'
        )
    );

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
    const captioningFeatureEnabled = _fastVlmCaptioningFeatureEnabled();
    const captioningConfig = config.captioning || {};
    const enableFastVlmCaptioning = captioningFeatureEnabled && (
        els.captioning.enableFastVlm
            ? Boolean(els.captioning.enableFastVlm.checked)
            : parseBoolLike(captioningConfig.enableFastVlm, false)
    );
    const fastVlmCaptioningModel = _textOrFallback(
        els.captioning.model ? els.captioning.model.value : captioningConfig.model,
        captioningConfig.model || 'default'
    ) || 'default';
    const fastVlmProxyFormat = _resolveVlmCaptioningProxyFormat(
        _textOrFallback(
            els.captioning.proxyFormat ? els.captioning.proxyFormat.value : captioningConfig.proxyFormat,
            captioningConfig.proxyFormat || 'png'
        )
    );
    const fastVlmMaxSidePx = _parsePositiveIntOrNull(
        els.captioning.maxSidePx ? els.captioning.maxSidePx.value : captioningConfig.maxSidePx
    ) || 1600;
    const fastVlmTimeoutSeconds = _parsePositiveIntOrNull(
        els.captioning.timeoutSeconds ? els.captioning.timeoutSeconds.value : captioningConfig.timeoutSeconds
    ) || 180;
    const fastVlmPythonExecutable = _textOrFallback(
        els.captioning.pythonExecutable ? els.captioning.pythonExecutable.value : captioningConfig.pythonExecutable,
        captioningConfig.pythonExecutable || ''
    );
    const fastVlmMlxVlmDir = _textOrFallback(
        els.captioning.mlxVlmDir ? els.captioning.mlxVlmDir.value : captioningConfig.mlxVlmDir,
        captioningConfig.mlxVlmDir || ''
    );

    const args = {
        preset,
        quality_tier: qualityTier,
        depth_backend: depthBackend,
        model_key: modelKey,
        enable_segmentation: segmentationEnable,
        segmentation_backend: segmentationBackend,
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
        run_card_version: runCardVersion,
        run_card_include_proofs: emitRunCardIncludeProofs,
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
    if (sam2Active) {
        args.sam2_model_size = sam2ModelSize;
        if (sam2CheckpointPath) args.sam2_checkpoint_path = sam2CheckpointPath;
        if (sam2PointsPerSide !== null) args.sam2_points_per_side = sam2PointsPerSide;
        if (sam2PointsPerBatch !== null) args.sam2_points_per_batch = sam2PointsPerBatch;
        if (sam2PredIouThresh !== null) args.sam2_pred_iou_thresh = sam2PredIouThresh;
        if (sam2StabilityScoreThresh !== null) args.sam2_stability_score_thresh = sam2StabilityScoreThresh;
        if (sam2CropNLayers !== null) args.sam2_crop_n_layers = sam2CropNLayers;
        if (sam2TilingEnabled) {
            args.sam2_tiling_enabled = true;
            if (sam2TileSizePx !== null) args.sam2_tile_size_px = sam2TileSizePx;
            if (sam2OverlapPx !== null) args.sam2_overlap_px = sam2OverlapPx;
            if (sam2GlobalPassLongestSide !== null) args.sam2_global_pass_longest_side = sam2GlobalPassLongestSide;
            if (sam2MaxConcurrency !== null) args.sam2_max_concurrency = sam2MaxConcurrency;
        }
    }
    if (camerasSidecarPath) args.cameras_sidecar_path = camerasSidecarPath;
    if (reconstructionIterations !== null) args.reconstruction_iterations = reconstructionIterations;
    if (maxWorkers !== null) args.max_workers = maxWorkers;
    if (maxGpuWorkers !== null) args.max_gpu_workers = maxGpuWorkers;
    if (logLevel) args.log_level = logLevel;
    if (captioningFeatureEnabled) {
        args.vlm_captioning_enabled = enableFastVlmCaptioning;
        if (enableFastVlmCaptioning) {
            args.vlm_captioning_backend = 'fastvlm';
            args.vlm_captioning_model = fastVlmCaptioningModel;
            args.vlm_captioning_proxy_format = fastVlmProxyFormat;
            args.vlm_captioning_max_side_px = fastVlmMaxSidePx;
            args.fastvlm_timeout_seconds = fastVlmTimeoutSeconds;
            if (fastVlmPythonExecutable) args.fastvlm_python_executable = fastVlmPythonExecutable;
            if (fastVlmMlxVlmDir) args.fastvlm_mlx_vlm_dir = fastVlmMlxVlmDir;
        }
    }
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
    if (String(presetName || '').trim() === 'custom') {
        state.config.preset = 'custom';
        return;
    }
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
    if (Object.prototype.hasOwnProperty.call(recommended, 'model_key')) c.modelKey = _resolveDa3ModelKey(recommended.model_key);
    if (Object.prototype.hasOwnProperty.call(recommended, 'depth_device')) c.depthDevice = _textOrFallback(recommended.depth_device, c.depthDevice);

    c.segmentation = c.segmentation || {};
    if (Object.prototype.hasOwnProperty.call(recommended, 'enable_segmentation')) c.segmentation.enable = parseBoolLike(recommended.enable_segmentation, c.segmentation.enable);
    if (Object.prototype.hasOwnProperty.call(recommended, 'segmentation_backend')) c.segmentation.backend = _resolveSegmentationBackend(recommended.segmentation_backend);
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_model_size')) c.segmentation.sam2ModelSize = _resolveSam2ModelSize(recommended.sam2_model_size);
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_checkpoint_path')) c.segmentation.sam2CheckpointPath = _textOrFallback(recommended.sam2_checkpoint_path, '');
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_tiling_enabled')) {
        c.segmentation.sam2TilingEnabled = parseBoolLike(recommended.sam2_tiling_enabled, c.segmentation.sam2TilingEnabled);
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_tile_size_px')) {
        c.segmentation.sam2TileSizePx = _parsePositiveIntOrNull(recommended.sam2_tile_size_px) || c.segmentation.sam2TileSizePx;
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_overlap_px')) {
        c.segmentation.sam2OverlapPx = _parseNonNegativeIntOrNull(recommended.sam2_overlap_px) ?? c.segmentation.sam2OverlapPx;
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_global_pass_longest_side')) {
        c.segmentation.sam2GlobalPassLongestSide =
            _parsePositiveIntOrNull(recommended.sam2_global_pass_longest_side) || c.segmentation.sam2GlobalPassLongestSide;
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_max_concurrency')) {
        c.segmentation.sam2MaxConcurrency = _parsePositiveIntOrNull(recommended.sam2_max_concurrency) || c.segmentation.sam2MaxConcurrency;
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_points_per_side')) {
        c.segmentation.sam2PointsPerSide = _parsePositiveIntOrNull(recommended.sam2_points_per_side) || c.segmentation.sam2PointsPerSide;
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_points_per_batch')) {
        c.segmentation.sam2PointsPerBatch = _parsePositiveIntOrNull(recommended.sam2_points_per_batch) || c.segmentation.sam2PointsPerBatch;
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_pred_iou_thresh')) {
        c.segmentation.sam2PredIouThresh = _parseProbabilityOrNull(recommended.sam2_pred_iou_thresh) ?? c.segmentation.sam2PredIouThresh;
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_stability_score_thresh')) {
        c.segmentation.sam2StabilityScoreThresh =
            _parseProbabilityOrNull(recommended.sam2_stability_score_thresh) ?? c.segmentation.sam2StabilityScoreThresh;
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'sam2_crop_n_layers')) {
        c.segmentation.sam2CropNLayers = _parseNonNegativeIntOrNull(recommended.sam2_crop_n_layers) ?? c.segmentation.sam2CropNLayers;
    }
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
    if (Object.prototype.hasOwnProperty.call(recommended, 'run_card_version')) {
        c.emits.runCardVersion = _resolveRunCardVersion(recommended.run_card_version);
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'run_card_include_proofs')) {
        c.emits.runCardIncludeProofs = parseBoolLike(recommended.run_card_include_proofs, c.emits.runCardIncludeProofs);
    }

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

    c.captioning = c.captioning || {};
    if (Object.prototype.hasOwnProperty.call(recommended, 'vlm_captioning_enabled')) {
        c.captioning.enableFastVlm = parseBoolLike(recommended.vlm_captioning_enabled, c.captioning.enableFastVlm);
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'vlm_captioning_model')) {
        c.captioning.model = _textOrFallback(recommended.vlm_captioning_model, c.captioning.model || 'default');
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'vlm_captioning_proxy_format')) {
        c.captioning.proxyFormat = _resolveVlmCaptioningProxyFormat(recommended.vlm_captioning_proxy_format);
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'vlm_captioning_max_side_px')) {
        c.captioning.maxSidePx = _parsePositiveIntOrNull(recommended.vlm_captioning_max_side_px) || c.captioning.maxSidePx || 1600;
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'fastvlm_timeout_seconds')) {
        c.captioning.timeoutSeconds = _parsePositiveIntOrNull(recommended.fastvlm_timeout_seconds) || c.captioning.timeoutSeconds || 180;
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'fastvlm_python_executable')) {
        c.captioning.pythonExecutable = _textOrFallback(recommended.fastvlm_python_executable, c.captioning.pythonExecutable || '');
    }
    if (Object.prototype.hasOwnProperty.call(recommended, 'fastvlm_mlx_vlm_dir')) {
        c.captioning.mlxVlmDir = _textOrFallback(recommended.fastvlm_mlx_vlm_dir, c.captioning.mlxVlmDir || '');
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
    if (![...els.presetSelect.options].some((option) => String(option.value || '') === 'custom')) {
        const customOption = document.createElement('option');
        customOption.value = 'custom';
        customOption.textContent = 'custom (Manual)';
        els.presetSelect.appendChild(customOption);
    }

    const names = [...els.presetSelect.options].map((option) => String(option.value || ''));
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
    if (_isProtectedFamilySuppressed('presets')) return;
    try {
        const headers = _buildAuthHeaders({ 'Accept': 'application/json' });
        const res = await fetch(`${API_BASE}/v1/presets?pipeline=${encodeURIComponent(pipelineName)}`, { headers });
        if (!res.ok) {
            await _maybeSuppressOnProtectedResponse('presets', res);
            throw new Error(`preset fetch failed (${res.status})`);
        }
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
    if (_isProtectedFamilySuppressed('readiness')) return;
    try {
        const headers = _buildAuthHeaders({ 'Accept': 'application/json' });
        const res = await fetch(`${API_BASE}/v1/readiness`, { headers });
        if (!res.ok) {
            await _maybeSuppressOnProtectedResponse('readiness', res);
            throw new Error(`readiness fetch failed (${res.status})`);
        }
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
    const api = _deferredBuildSurfaceApi();
    if (api?.refreshArchiveFieldVisibility) {
        api.refreshArchiveFieldVisibility();
        return;
    }
    if (!_shouldLoadDeferredBuildSurface()) return;
    void _loadDeferredBuildSurface().then((loaded) => {
        if (loaded?.refreshArchiveFieldVisibility) loaded.refreshArchiveFieldVisibility();
    });
}

function _firstInvalidBuildInput() {
    const candidates = [
        els.inputDir,
        els.outputDir,
        els.archiveIndexPath,
        els.rightsManifestPath
    ];
    for (const input of candidates) {
        if (!input || typeof input.checkValidity !== 'function') continue;
        if (input.offsetParent === null && input.getClientRects().length === 0) continue;
        if (!input.checkValidity()) return input;
    }
    return null;
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
    c.modelKey = _resolveDa3ModelKey(c.modelKey);
    c.segmentation = c.segmentation || {};
    c.segmentation.enable = parseBoolLike(c.segmentation.enable, false);
    c.segmentation.backend = _resolveSegmentationBackend(c.segmentation.backend);
    c.segmentation.sam2ModelSize = _resolveSam2ModelSize(c.segmentation.sam2ModelSize);
    c.segmentation.sam2CheckpointPath = _textOrFallback(c.segmentation.sam2CheckpointPath, '');
    c.segmentation.sam2TilingEnabled = parseBoolLike(c.segmentation.sam2TilingEnabled, false);
    c.segmentation.sam2TileSizePx = _parsePositiveIntOrNull(c.segmentation.sam2TileSizePx) || 1536;
    c.segmentation.sam2OverlapPx = _parseNonNegativeIntOrNull(c.segmentation.sam2OverlapPx) ?? 256;
    c.segmentation.sam2GlobalPassLongestSide = _parsePositiveIntOrNull(c.segmentation.sam2GlobalPassLongestSide) || 1280;
    c.segmentation.sam2MaxConcurrency = _parsePositiveIntOrNull(c.segmentation.sam2MaxConcurrency) || 1;
    c.segmentation.sam2PointsPerSide = _parsePositiveIntOrNull(c.segmentation.sam2PointsPerSide) || 32;
    c.segmentation.sam2PointsPerBatch = _parsePositiveIntOrNull(c.segmentation.sam2PointsPerBatch) || 64;
    c.segmentation.sam2PredIouThresh = _parseProbabilityOrNull(c.segmentation.sam2PredIouThresh) ?? 0.88;
    c.segmentation.sam2StabilityScoreThresh = _parseProbabilityOrNull(c.segmentation.sam2StabilityScoreThresh) ?? 0.85;
    c.segmentation.sam2CropNLayers = _parseNonNegativeIntOrNull(c.segmentation.sam2CropNLayers) ?? 1;
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
    c.emits.runCardVersion = _resolveRunCardVersion(c.emits.runCardVersion);
    c.emits.runCardIncludeProofs = parseBoolLike(c.emits.runCardIncludeProofs, false);
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
    c.captioning = c.captioning || {};
    c.captioning.enableFastVlm = parseBoolLike(c.captioning.enableFastVlm, false);
    c.captioning.model = _textOrFallback(c.captioning.model, 'default');
    c.captioning.proxyFormat = _resolveVlmCaptioningProxyFormat(c.captioning.proxyFormat);
    c.captioning.maxSidePx = _parsePositiveIntOrNull(c.captioning.maxSidePx) || 1600;
    c.captioning.timeoutSeconds = _parsePositiveIntOrNull(c.captioning.timeoutSeconds) || 180;
    c.captioning.pythonExecutable = _textOrFallback(c.captioning.pythonExecutable, '');
    c.captioning.mlxVlmDir = _textOrFallback(c.captioning.mlxVlmDir, '');
    applyLuxMetadataToControls();
    applyPipelinePresetOptions(state.pipeline);
    if (els.presetSelect) els.presetSelect.value = c.preset;
    if (els.inputDir) els.inputDir.value = c.inputDir;
    if (els.outputDir) els.outputDir.value = c.outputDir;
    if (els.archiveIndexPath) els.archiveIndexPath.value = c.gate.archiveIndex;
    if (els.rightsManifestPath) els.rightsManifestPath.value = c.gate.manifestJsonl;
    if (els.qualityTier) els.qualityTier.value = c.qualityTier;
    if (els.depthBackend) els.depthBackend.value = c.depthBackend;
    if (els.modelKey) els.modelKey.value = c.modelKey;
    if (els.depthDevice) els.depthDevice.value = c.depthDevice;
    if (els.segmentation.backend) els.segmentation.backend.value = c.segmentation.backend;
    if (els.segmentation.sam2ModelSize) els.segmentation.sam2ModelSize.value = c.segmentation.sam2ModelSize;
    if (els.segmentation.sam2CheckpointPath) els.segmentation.sam2CheckpointPath.value = c.segmentation.sam2CheckpointPath;
    if (els.segmentation.sam2TileSizePx) els.segmentation.sam2TileSizePx.value = String(c.segmentation.sam2TileSizePx);
    if (els.segmentation.sam2OverlapPx) els.segmentation.sam2OverlapPx.value = String(c.segmentation.sam2OverlapPx);
    if (els.segmentation.sam2GlobalPassLongestSide) {
        els.segmentation.sam2GlobalPassLongestSide.value = String(c.segmentation.sam2GlobalPassLongestSide);
    }
    if (els.segmentation.sam2MaxConcurrency) els.segmentation.sam2MaxConcurrency.value = String(c.segmentation.sam2MaxConcurrency);
    if (els.segmentation.sam2PointsPerSide) els.segmentation.sam2PointsPerSide.value = String(c.segmentation.sam2PointsPerSide);
    if (els.segmentation.sam2PointsPerBatch) els.segmentation.sam2PointsPerBatch.value = String(c.segmentation.sam2PointsPerBatch);
    if (els.segmentation.sam2PredIouThresh) {
        els.segmentation.sam2PredIouThresh.value = String(c.segmentation.sam2PredIouThresh);
    }
    if (els.segmentation.sam2StabilityScoreThresh) {
        els.segmentation.sam2StabilityScoreThresh.value = String(c.segmentation.sam2StabilityScoreThresh);
    }
    if (els.segmentation.sam2CropNLayers) els.segmentation.sam2CropNLayers.value = String(c.segmentation.sam2CropNLayers);
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
    if (els.captioning.model) els.captioning.model.value = c.captioning.model;
    if (els.captioning.proxyFormat) els.captioning.proxyFormat.value = c.captioning.proxyFormat;
    if (els.captioning.maxSidePx) els.captioning.maxSidePx.value = String(c.captioning.maxSidePx);
    if (els.captioning.timeoutSeconds) els.captioning.timeoutSeconds.value = String(c.captioning.timeoutSeconds);
    if (els.captioning.pythonExecutable) els.captioning.pythonExecutable.value = c.captioning.pythonExecutable;
    if (els.captioning.mlxVlmDir) els.captioning.mlxVlmDir.value = c.captioning.mlxVlmDir;

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
    safeSyncCheck(els.segmentation.sam2TilingEnabled, c.segmentation.sam2TilingEnabled);
    safeSyncCheck(els.segmentation.strict, c.segmentation.strict);

    safeSyncCheck(els.emits.master16, c.emits.master16);
    safeSyncCheck(els.emits.upscaled16, c.emits.upscaled16);
    safeSyncCheck(els.emits.marketing, c.emits.marketing);
    safeSyncCheck(els.emits.report, c.emits.report);
    safeSyncCheck(els.emits.runCard, c.emits.runCard);
    if (els.emits.runCardVersion) els.emits.runCardVersion.value = c.emits.runCardVersion;
    safeSyncCheck(els.emits.runCardIncludeProofs, c.emits.runCardIncludeProofs);

    safeSyncCheck(els.licenses.nonCommercialOk, c.licenses.nonCommercialOk);
    safeSyncCheck(els.licenses.acceptApple, c.licenses.acceptApple);
    safeSyncCheck(els.licenses.acceptResearchTools, c.licenses.acceptResearchTools);
    safeSyncCheck(els.reconstruction.enable, c.reconstruction.enable);
    safeSyncCheck(els.reconstruction.emitSceneDebugBundle, c.reconstruction.emitSceneDebugBundle);
    safeSyncCheck(els.captioning.enableFastVlm, c.captioning.enableFastVlm);
    state.portalUi.debugBundleAcknowledged = c.reconstruction.emitSceneDebugBundle
        ? Boolean(state.portalUi.debugBundleAcknowledged)
        : false;
    if (els.debugBundleAcknowledge) {
        els.debugBundleAcknowledge.checked = Boolean(state.portalUi.debugBundleAcknowledged);
    }
    _syncSwitchStateLabels();
    syncSegmentationControlState(c);
    syncRunCardControlState(c);
    syncRuntimeWorkerModeControls();
    renderFieldPreviewStatuses();
    _syncStagedUploadUi();

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
        const canonicalLuxArgs = buildCanonicalLuxDepthArgs(c);
        args = {
            ...canonicalLuxArgs,
            input_dir: inputDirValue,
            output_dir: outputDirValue,
            overwrite: els.flags.overwrite
                ? Boolean(els.flags.overwrite.checked)
                : parseBoolLike(c.flags.overwrite, false)
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
        const captioningEnabled = parseBoolLike(args.vlm_captioning_enabled, false);

        expectedOutputs.push('Depth maps');
        if (parseBoolLike(args.pbr, false)) expectedOutputs.push('PBR maps');
        if (segmentationEnabled) expectedOutputs.push('Segmentation masks');
        if (parseBoolLike(args.enable_v2, false)) expectedOutputs.push('V2 enhanced outputs');
        if (reconstructionEnabled) expectedOutputs.push('Reconstruction report bundle');
        if (parseBoolLike(args.emit_run_card, false)) expectedOutputs.push('Run card JSON');
        if (parseBoolLike(args.emit_scene_debug_bundle, false)) expectedOutputs.push('Reconstruction debug bundle');
        if (captioningEnabled) expectedOutputs.push('Advisory FastVLM caption sidecars');

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
            if (captioningEnabled) {
                warnings.push('WARNING: FastVLM captions are advisory sidecar metadata and do not satisfy quality gates.');
                if (healthState === 'good') {
                    healthState = 'warn';
                    healthLabel = 'advisory captioning';
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

    const checklistItems = _dispatchChecklistItems(payload);
    if (els.preRunWarnings) {
        els.preRunWarnings.innerHTML = '';
        checklistItems.forEach((item) => {
            const li = document.createElement('li');
            const tone = _dispatchChecklistTone(item.tone);
            li.className = `rounded-lg border px-3 py-2 leading-relaxed ${tone.cardClass}`;
            li.dataset.tone = item.tone;
            li.textContent = `${tone.badge}: ${item.label} - ${item.detail}`;
            els.preRunWarnings.appendChild(li);
        });
    }
    if (els.preRunWarningsEmpty) {
        els.preRunWarningsEmpty.style.display = checklistItems.length === 0 ? 'block' : 'none';
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
    _syncBootstrapGuardedControls();
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
        if (payload.args.depth_backend === 'da3') {
            cliLines.push(`  --model-key ${q(payload.args.model_key || 'da3-metric')}`);
        }
        if (payload.args.depth_device) {
            cliLines.push(`  --depth-device ${q(payload.args.depth_device)}`);
        }
        cliLines.push(`  --materials-v3 ${onoff(payload.args.materials_v3)}`);
        cliLines.push(`  --enable-segmentation ${onoff(payload.args.enable_segmentation)}`);
        cliLines.push(`  --segmentation-backend ${q(payload.args.segmentation_backend)}`);
        if (payload.args.segmentation_backend === 'sam2') {
            cliLines.push(`  --sam2-model-size ${q(payload.args.sam2_model_size)}`);
            if (parseBoolLike(payload.args.sam2_tiling_enabled, false)) {
                cliLines.push(`  --sam2-tiling-enabled`);
            }
            if (parseBoolLike(payload.args.sam2_tiling_enabled, false) && payload.args.sam2_tile_size_px) {
                cliLines.push(`  --sam2-tile-size-px ${q(payload.args.sam2_tile_size_px)}`);
            }
            if (parseBoolLike(payload.args.sam2_tiling_enabled, false) && payload.args.sam2_overlap_px !== undefined) {
                cliLines.push(`  --sam2-overlap-px ${q(payload.args.sam2_overlap_px)}`);
            }
            if (parseBoolLike(payload.args.sam2_tiling_enabled, false) && payload.args.sam2_global_pass_longest_side) {
                cliLines.push(`  --sam2-global-pass-longest-side ${q(payload.args.sam2_global_pass_longest_side)}`);
            }
            if (parseBoolLike(payload.args.sam2_tiling_enabled, false) && payload.args.sam2_max_concurrency) {
                cliLines.push(`  --sam2-max-concurrency ${q(payload.args.sam2_max_concurrency)}`);
            }
            if (payload.args.sam2_points_per_side) {
                cliLines.push(`  --sam2-points-per-side ${q(payload.args.sam2_points_per_side)}`);
            }
            if (payload.args.sam2_points_per_batch) {
                cliLines.push(`  --sam2-points-per-batch ${q(payload.args.sam2_points_per_batch)}`);
            }
            if (payload.args.sam2_pred_iou_thresh !== undefined) {
                cliLines.push(`  --sam2-pred-iou-thresh ${q(payload.args.sam2_pred_iou_thresh)}`);
            }
            if (payload.args.sam2_stability_score_thresh !== undefined) {
                cliLines.push(`  --sam2-stability-score-thresh ${q(payload.args.sam2_stability_score_thresh)}`);
            }
            if (payload.args.sam2_crop_n_layers !== undefined) {
                cliLines.push(`  --sam2-crop-n-layers ${q(payload.args.sam2_crop_n_layers)}`);
            }
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
        cliLines.push(`  --run-card-version ${q(payload.args.run_card_version || 'v1')}`);
        cliLines.push(`  --run-card-include-proofs ${onoff(payload.args.run_card_include_proofs)}`);
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
        if (parseBoolLike(payload.args.vlm_captioning_enabled, false)) {
            cliLines.push(`  --vlm-captioning "on"`);
            cliLines.push(`  --vlm-captioning-backend ${q(payload.args.vlm_captioning_backend || 'fastvlm')}`);
            cliLines.push(`  --vlm-captioning-model ${q(payload.args.vlm_captioning_model || 'default')}`);
            cliLines.push(`  --vlm-captioning-proxy-format ${q(payload.args.vlm_captioning_proxy_format || 'png')}`);
            cliLines.push(`  --vlm-captioning-max-side-px ${q(payload.args.vlm_captioning_max_side_px || 1600)}`);
            if (payload.args.fastvlm_python_executable) {
                cliLines.push(`  --fastvlm-python ${q(payload.args.fastvlm_python_executable)}`);
            }
            if (payload.args.fastvlm_mlx_vlm_dir) {
                cliLines.push(`  --fastvlm-mlx-vlm-dir ${q(payload.args.fastvlm_mlx_vlm_dir)}`);
            }
            cliLines.push(`  --fastvlm-timeout-seconds ${q(payload.args.fastvlm_timeout_seconds || 180)}`);
        }
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
            ':modelKey': 'model_key',
            'segmentation:backend': 'segmentation_backend',
            'segmentation:sam2TilingEnabled': 'sam2_tiling_enabled',
            'segmentation:sam2TileSizePx': 'sam2_tile_size_px',
            'segmentation:sam2OverlapPx': 'sam2_overlap_px',
            'segmentation:sam2GlobalPassLongestSide': 'sam2_global_pass_longest_side',
            'segmentation:sam2MaxConcurrency': 'sam2_max_concurrency',
            'segmentation:sam2PointsPerSide': 'sam2_points_per_side',
            'segmentation:sam2PointsPerBatch': 'sam2_points_per_batch',
            'segmentation:sam2PredIouThresh': 'sam2_pred_iou_thresh',
            'segmentation:sam2StabilityScoreThresh': 'sam2_stability_score_thresh',
            'segmentation:sam2CropNLayers': 'sam2_crop_n_layers',
            'segmentation:strict': 'strict_segmentation',
            'emits:runCardVersion': 'run_card_version',
            'licenses:nonCommercialOk': 'non_commercial_ok',
            'licenses:acceptApple': 'accept_apple_depth_pro_research_license',
            'licenses:acceptResearchTools': 'accept_research_tools_license',
            'reconstruction:enable': 'enable_reconstruction',
            'reconstruction:groupingMode': 'grouping_mode',
            'reconstruction:iterations': 'reconstruction_iterations',
            'reconstruction:tier': 'reconstruction_tier',
            'reconstruction:emitSceneDebugBundle': 'emit_scene_debug_bundle',
            'raw:ingestMode': 'raw_ingest_mode',
            'captioning:enableFastVlm': 'vlm_captioning_enabled',
            'captioning:model': 'vlm_captioning_model',
            'captioning:proxyFormat': 'vlm_captioning_proxy_format',
            'captioning:maxSidePx': 'vlm_captioning_max_side_px',
            'captioning:timeoutSeconds': 'fastvlm_timeout_seconds',
            'captioning:pythonExecutable': 'fastvlm_python_executable',
            'captioning:mlxVlmDir': 'fastvlm_mlx_vlm_dir',
            'runtime:maxWorkersMode': 'max_workers_mode',
            'runtime:maxWorkers': 'max_workers',
            'runtime:maxGpuWorkersMode': 'max_gpu_workers_mode',
            'runtime:maxGpuWorkers': 'max_gpu_workers',
            'runtime:logLevel': 'log_level'
        };
        return lookup[`${normalizedCategory}:${normalizedKey}`] || null;
    };

    const telemetrySurfaceFor = (category) => {
        if (category === 'reconstruction' || category === 'raw' || category === 'runtime' || category === 'captioning') {
            return 'reconstruction_runtime';
        }
        return 'dispatch';
    };
    const syncDependentControlState = (category, key) => {
        if (category === 'segmentation' && (key === 'enable' || key === 'backend' || key === 'sam2TilingEnabled')) {
            syncSegmentationControlState(state.config);
        }
        if (category === 'emits' && key === 'runCard') {
            syncRunCardControlState(state.config);
        }
        if (category === 'captioning' && key === 'enableFastVlm') {
            syncBuildSurfaceApplicability();
        }
    };

    const safeBindText = (el, category, key) => {
        if (!el) return;
        el.addEventListener('change', (e) => {
            if (category) state.config[category][key] = e.target.value;
            else if (key in state.config) state.config[key] = e.target.value;
            else state[key] = e.target.value;
            if (key === 'pipeline') {
                updateUIFromState();
                _persistTransientPortalDraft();
                void fetchPresetsForPipeline(state.pipeline, true);
                void fetchReadiness(true);
                void fetchConfigMetadata(state.pipeline, true);
            }
            else {
                syncDependentControlState(category, key);
                _persistTransientPortalDraft();
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
            _scheduleTransientPortalDraftPersist();
            renderCLI();
            if (trackedTelemetryField(category, key)) {
                scheduleConfigPreview();
            }
        });
        el.addEventListener('change', (e) => {
            const field = trackedTelemetryField(category, key);
            _scheduleTransientPortalDraftPersist({ immediate: true });
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
            syncDependentControlState(category, key);
            _persistTransientPortalDraft();
            _syncSwitchStateLabels();
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
            _persistTransientPortalDraft();
        });
    }
    safeBindInput(els.inputDir, null, 'inputDir');
    safeBindInput(els.outputDir, null, 'outputDir');
    safeBindInput(els.archiveIndexPath, 'gate', 'archiveIndex');
    safeBindInput(els.rightsManifestPath, 'gate', 'manifestJsonl');
    safeBindText(els.qualityTier, null, 'qualityTier');
    safeBindText(els.depthBackend, null, 'depthBackend');
    safeBindText(els.modelKey, null, 'modelKey');
    safeBindText(els.depthDevice, null, 'depthDevice');
    safeBindText(els.segmentation.backend, 'segmentation', 'backend');
    safeBindText(els.segmentation.sam2ModelSize, 'segmentation', 'sam2ModelSize');
    safeBindInput(els.segmentation.sam2CheckpointPath, 'segmentation', 'sam2CheckpointPath');
    safeBindInput(els.segmentation.sam2TileSizePx, 'segmentation', 'sam2TileSizePx');
    safeBindInput(els.segmentation.sam2OverlapPx, 'segmentation', 'sam2OverlapPx');
    safeBindInput(els.segmentation.sam2GlobalPassLongestSide, 'segmentation', 'sam2GlobalPassLongestSide');
    safeBindInput(els.segmentation.sam2MaxConcurrency, 'segmentation', 'sam2MaxConcurrency');
    safeBindInput(els.segmentation.sam2PointsPerSide, 'segmentation', 'sam2PointsPerSide');
    safeBindInput(els.segmentation.sam2PointsPerBatch, 'segmentation', 'sam2PointsPerBatch');
    safeBindInput(els.segmentation.sam2PredIouThresh, 'segmentation', 'sam2PredIouThresh');
    safeBindInput(els.segmentation.sam2StabilityScoreThresh, 'segmentation', 'sam2StabilityScoreThresh');
    safeBindInput(els.segmentation.sam2CropNLayers, 'segmentation', 'sam2CropNLayers');
    safeBindInput(els.v2Preset, null, 'v2Preset');
    safeBindText(els.reconstruction.groupingMode, 'reconstruction', 'groupingMode');
    safeBindInput(els.reconstruction.camerasSidecarPath, 'reconstruction', 'camerasSidecarPath');
    safeBindInput(els.reconstruction.iterations, 'reconstruction', 'iterations');
    safeBindInput(els.reconstruction.tier, 'reconstruction', 'tier');
    safeBindText(els.raw.ingestMode, 'raw', 'ingestMode');
    safeBindText(els.raw.wbMode, 'raw', 'wbMode');
    safeBindText(els.raw.demosaic, 'raw', 'demosaic');
    safeBindText(els.captioning.proxyFormat, 'captioning', 'proxyFormat');
    safeBindInput(els.captioning.model, 'captioning', 'model');
    safeBindInput(els.captioning.maxSidePx, 'captioning', 'maxSidePx');
    safeBindInput(els.captioning.timeoutSeconds, 'captioning', 'timeoutSeconds');
    safeBindInput(els.captioning.pythonExecutable, 'captioning', 'pythonExecutable');
    safeBindInput(els.captioning.mlxVlmDir, 'captioning', 'mlxVlmDir');
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
    safeBindCheck(els.segmentation.sam2TilingEnabled, 'segmentation', 'sam2TilingEnabled');
    safeBindCheck(els.segmentation.strict, 'segmentation', 'strict');
    safeBindCheck(els.reconstruction.enable, 'reconstruction', 'enable');
    safeBindCheck(els.reconstruction.emitSceneDebugBundle, 'reconstruction', 'emitSceneDebugBundle');
    safeBindCheck(els.captioning.enableFastVlm, 'captioning', 'enableFastVlm');

    safeBindCheck(els.emits.master16, 'emits', 'master16');
    safeBindCheck(els.emits.upscaled16, 'emits', 'upscaled16');
    safeBindCheck(els.emits.marketing, 'emits', 'marketing');
    safeBindCheck(els.emits.report, 'emits', 'report');
    safeBindCheck(els.emits.runCard, 'emits', 'runCard');
    safeBindText(els.emits.runCardVersion, 'emits', 'runCardVersion');
    safeBindCheck(els.emits.runCardIncludeProofs, 'emits', 'runCardIncludeProofs');

    safeBindCheck(els.licenses.nonCommercialOk, 'licenses', 'nonCommercialOk');
    safeBindCheck(els.licenses.acceptApple, 'licenses', 'acceptApple');
    safeBindCheck(els.licenses.acceptResearchTools, 'licenses', 'acceptResearchTools');

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
        els.apiKeyInput.addEventListener('input', () => {
            _handleDirectDebugApiKeyUpdate();
        });
        els.apiKeyInput.addEventListener('change', () => {
            _handleDirectDebugApiKeyUpdate({ resumeStreams: true });
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

function _setSurfaceEmptyState(container, titleEl, detailEl, copy) {
    if (!container) return;
    container.dataset.tone = String(copy?.tone || 'neutral');
    if (titleEl) titleEl.textContent = String(copy?.title || '');
    if (detailEl) detailEl.textContent = String(copy?.detail || '');
}

// ============================================================================
// 9. JOB RENDERING (Operate surface — main bundle keeps thin shims that
// delegate to operate-surface-deferred.js on first call.)
// ============================================================================

function renderJobQueue(includeReviewSurfaces = true) {
    const api = _deferredOperateSurfaceApi();
    if (api?.renderJobQueue) {
        api.renderJobQueue(includeReviewSurfaces);
        return;
    }
    if (!_shouldLoadDeferredOperateSurface()) return;
    void _loadDeferredOperateSurface().then((loaded) => {
        if (loaded?.renderJobQueue) loaded.renderJobQueue(includeReviewSurfaces);
    });
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

// ============================================================================
// 10. API LAYER
// ============================================================================

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
            browser_previewable: Boolean(parsed.browser_previewable),
            content_type: typeof parsed.content_type === 'string' ? parsed.content_type : '',
            url: typeof parsed.url === 'string' ? parsed.url : '',
            download_url: typeof parsed.download_url === 'string' ? parsed.download_url : '',
            preview_url: typeof parsed.preview_url === 'string' ? parsed.preview_url : '',
            preview_mime_type: typeof parsed.preview_mime_type === 'string' ? parsed.preview_mime_type : '',
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
            _clearArtifactUrlNotFoundCache();
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
    if (_isProtectedFamilySuppressed('jobs_events')) {
        job.reconnectBlocked = true;
        _clearSseRetry(job, true);
        return;
    }
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
            const suppressed = await _maybeSuppressOnProtectedResponse('jobs_events', res);
            const parsedError = await extractApiError(res);
            if (parsedError.error) job.error = parsedError.error;
            const status = Number(res.status) || 0;
            const isAuthError = status === 401 || status === 403;
            const isRetryableStatus = status === 429 || status >= 500;
            shouldReconnect = isRetryableStatus && !suppressed;
            if (suppressed || !isRetryableStatus) {
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
            _queuePortalRumSample({
                eventType: 'sse_reconnect',
                value: 1,
                unit: 'count',
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
    if (_isProtectedFamilySuppressed('jobs_events')) {
        job.reconnectBlocked = true;
        _clearSseRetry(job, true);
        return;
    }
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
            _queuePortalRumSample({
                eventType: 'sse_reconnect',
                value: 1,
                unit: 'count',
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
    if (_isProtectedFamilySuppressed('jobs_list')) return;
    if (state.jobs.length === 0) {
        state.jobsLoadStatus = 'loading';
        renderJobQueue();
    }
    try {
        const headers = _buildAuthHeaders({ 'Accept': 'application/json' });
        const res = await fetch(`${API_BASE}/v1/jobs`, { headers });
        if (!res.ok) {
            await _maybeSuppressOnProtectedResponse('jobs_list', res);
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
    if (_isProtectedFamilySuppressed('jobs_cancel')) return;

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

    const requestTraceparent = portalInternals.createChildTraceparent(_portalRumTraceparent());
    const requestStartedAt = _portalRumNow();
    const headers = _buildAuthHeaders({}, 'POST', { traceparent: requestTraceparent });
    fetch(`${API_BASE}/v1/jobs/${id}/cancel`, { method: 'POST', headers })
        .then((response) => {
            if (!response.ok) {
                void _maybeSuppressOnProtectedResponse('jobs_cancel', response);
            }
            _queuePortalRumSample({
                eventType: 'queue_request',
                metric: 'cancel',
                value: _portalRumNow() - requestStartedAt,
                unit: 'ms',
                traceparent: requestTraceparent,
                metadata: {
                    outcome: response.ok ? 'ok' : 'error',
                    status: response.status
                }
            });
        })
        .catch(() => {
            _queuePortalRumSample({
                eventType: 'queue_request',
                metric: 'cancel',
                value: _portalRumNow() - requestStartedAt,
                unit: 'ms',
                traceparent: requestTraceparent,
                metadata: {
                    outcome: 'error'
                }
            });
        });

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
    const invalidField = _firstInvalidBuildInput();
    if (invalidField) {
        invalidField.reportValidity();
        void emitPortalEvent('dispatch_blocked', {
            surface: 'dispatch',
            field: invalidField.id || '',
            reasons: ['client_constraint_invalid']
        });
        return;
    }
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
        createToast(DISPATCH_BACKEND_OFFLINE_MESSAGE, 'error');
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
        _setButtonBusy(els.runJobBtn, true);
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

    const requestTraceparent = portalInternals.createChildTraceparent(_portalRumTraceparent());
    const requestStartedAt = _portalRumNow();
    let queueRumRecorded = false;
    try {
        const headers = _buildAuthHeaders({ 'Content-Type': 'application/json' }, 'POST', {
            traceparent: requestTraceparent
        });
        const res = await fetch(`${API_BASE}/v1/jobs`, {
            method: 'POST',
            headers,
            body: JSON.stringify(payload)
        });
        _queuePortalRumSample({
            eventType: 'queue_request',
            metric: 'submit',
            value: _portalRumNow() - requestStartedAt,
            unit: 'ms',
            traceparent: requestTraceparent,
            metadata: {
                outcome: res.ok ? 'ok' : 'error',
                status: res.status
            }
        });
        queueRumRecorded = true;
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
        if (!queueRumRecorded) {
            _queuePortalRumSample({
                eventType: 'queue_request',
                metric: 'submit',
                value: _portalRumNow() - requestStartedAt,
                unit: 'ms',
                traceparent: requestTraceparent,
                metadata: {
                    outcome: 'error'
                }
            });
        }
        const errorMessage = err instanceof Error ? err.message : String(err);
        job.state = 'failed';
        job.finishedAt = Date.now();
        appendJobLog(job, `[ERROR] ${errorMessage}`);
        _reconcileJobTimeline(job);
        scheduleRenderJobQueue();
        const toastMessage = errorMessage
            ? `Backend submission failed: ${truncateMiddle(errorMessage, 180)}`
            : 'Backend submission failed.';
        createToast(toastMessage, "error");
        if (state.selectedJobId === job.id && els.logStatusIndicator) els.logStatusIndicator.classList.add('hidden');
    } finally {
        if (els.runJobBtn) {
            els.runJobBtn.textContent = "Execute Job";
            _setButtonBusy(els.runJobBtn, false);
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
        _persistTransientPortalDraft();
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
        captioning_summary: effectivePreview.captioning_summary || _buildLocalCaptioningSummary(payload.args || {}),
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

if (els.stagedUploadPickFilesBtn && els.stagedUploadFilesInput) {
    els.stagedUploadPickFilesBtn.addEventListener('click', () => {
        if (_stagedUploadsVisibleForState() && !state.portalUi?.stagedUpload?.busy) {
            els.stagedUploadFilesInput.click();
        }
    });
}
if (els.stagedUploadPickFolderBtn && els.stagedUploadFolderInput) {
    els.stagedUploadPickFolderBtn.addEventListener('click', () => {
        if (_stagedUploadsVisibleForState() && !state.portalUi?.stagedUpload?.busy) {
            els.stagedUploadFolderInput.click();
        }
    });
}
if (els.stagedUploadFilesInput) {
    els.stagedUploadFilesInput.addEventListener('change', (event) => {
        _submitStagedUploadSelection(event.target.files);
        event.target.value = '';
    });
}
if (els.stagedUploadFolderInput) {
    els.stagedUploadFolderInput.addEventListener('change', (event) => {
        _submitStagedUploadSelection(event.target.files);
        event.target.value = '';
    });
}
if (els.stagedUploadDropzone) {
    els.stagedUploadDropzone.addEventListener('dragover', (event) => {
        if (!_stagedUploadsVisibleForState() || state.portalUi?.stagedUpload?.busy) return;
        event.preventDefault();
    });
    els.stagedUploadDropzone.addEventListener('drop', (event) => {
        if (!_stagedUploadsVisibleForState() || state.portalUi?.stagedUpload?.busy) return;
        event.preventDefault();
        _submitStagedUploadSelection(event.dataTransfer?.files);
    });
    els.stagedUploadDropzone.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        if (!_stagedUploadsVisibleForState() || state.portalUi?.stagedUpload?.busy) return;
        event.preventDefault();
        if (els.stagedUploadFilesInput) {
            els.stagedUploadFilesInput.click();
        }
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

            c.emits.master16 = parseBoolLike(data.args.emit_master16, c.emits.master16);
            c.emits.upscaled16 = parseBoolLike(data.args.emit_upscaled16, c.emits.upscaled16);
            c.emits.marketing = parseBoolLike(data.args.emit_marketing, c.emits.marketing);
            c.emits.report = parseBoolLike(data.args.emit_report, c.emits.report);
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
        }
        state.portalUi.debugBundleAcknowledged = false;
        state.portalUi.debugBundleGuardrailSeen = false;
        updateUIFromState();
        _persistTransientPortalDraft();
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
        const selectedJob = state.jobs.find((item) => item.id === state.selectedJobId);
        _openReviewSurfaceForJob(selectedJob, 'job_inspector');
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
        _toggleCompareSurface(selectedJob, 'artifact_review');
    });
}

if (els.openArtifactBtn) {
    els.openArtifactBtn.addEventListener('click', () => {
        const selectedJob = state.jobs.find((item) => item.id === state.selectedJobId);
        _openArtifactForSelection(selectedJob, _selectedArtifactForJob(selectedJob), 'artifact_review');
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

if (els.copyArtifactFingerprintBtn) {
    els.copyArtifactFingerprintBtn.addEventListener('click', async () => {
        const fingerprint = String(els.copyArtifactFingerprintBtn.dataset.fingerprint || '').trim();
        if (!fingerprint) {
            createToast('No artifact fingerprint is available for this selection.', 'error');
            return;
        }
        await copyToClipboard(fingerprint);
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
if (els.consoleActionRailActions) {
    els.consoleActionRailActions.addEventListener('click', handleOperatorActionClick);
}
if (els.selectedJobRecoveryActions) {
    els.selectedJobRecoveryActions.addEventListener('click', handleOperatorActionClick);
}
if (els.reviewStatusActions) {
    els.reviewStatusActions.addEventListener('click', handleOperatorActionClick);
}
if (els.jobList) els.jobList.addEventListener('click', handleJobListClick);
if (els.jobList) els.jobList.addEventListener('keydown', handleJobListKeydown);

// ============================================================================
// 11. THEME
// ============================================================================

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
    document.documentElement.classList.toggle('light', mode === 'light');

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

// ============================================================================
// 12. OVERLAYS & PANELS
// ============================================================================

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
    if (els.artifactViewerModal && !els.artifactViewerModal.classList.contains('hidden')) {
        return els.artifactViewerPanel;
    }
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

function _isTypingTarget(target) {
    if (!(target instanceof HTMLElement)) return false;
    const tagName = String(target.tagName || '').toLowerCase();
    if (target.isContentEditable || target.closest('[contenteditable="true"]')) return true;
    if (tagName === 'textarea' || tagName === 'select') return true;
    if (tagName !== 'input') return false;
    const inputType = String(target.getAttribute('type') || 'text').trim().toLowerCase();
    return !['button', 'checkbox', 'color', 'file', 'hidden', 'radio', 'range', 'reset', 'submit'].includes(inputType);
}

function _artifactViewerContext() {
    if (deferredReviewSurfaceApi && typeof deferredReviewSurfaceApi._artifactViewerContext === 'function') {
        return deferredReviewSurfaceApi._artifactViewerContext();
    }
    return {
        job: null,
        artifacts: [],
        artifact: null,
        index: -1,
        url: '',
        inlinePreview: false,
        zoomPercent: 100
    };
}

function _setArtifactViewerZoom(nextZoom) {
    if (deferredReviewSurfaceApi) {
        deferredReviewSurfaceApi._setArtifactViewerZoom(nextZoom);
        return;
    }
    _primeDeferredReviewSurface('viewer');
}

function _navigateArtifactViewerSelection(direction) {
    if (deferredReviewSurfaceApi) {
        return deferredReviewSurfaceApi._navigateArtifactViewerSelection(direction);
    }
    _primeDeferredReviewSurface('viewer');
    return false;
}

function renderArtifactViewer() {
    if (deferredReviewSurfaceApi) {
        deferredReviewSurfaceApi.renderArtifactViewer();
        return;
    }
    if (!els.artifactViewerModal || !els.artifactViewerPanel) return;
    const shouldShow = false;
    els.artifactViewerModal.classList.toggle('hidden', !shouldShow);
    els.artifactViewerModal.classList.toggle('flex', shouldShow);
    els.artifactViewerModal.setAttribute('aria-hidden', shouldShow ? 'false' : 'true');
    els.artifactViewerModal.dataset.overlayOpen = shouldShow ? 'true' : 'false';
    if (els.artifactViewerStatus) {
        els.artifactViewerStatus.textContent = 'Artifact viewer is closed.';
    }
    if (Boolean(state.portalUi?.artifactViewer?.open)) {
        _primeDeferredReviewSurface('viewer');
    }
}

function _closeArtifactViewer(restoreFocus = true) {
    if (deferredReviewSurfaceApi) {
        deferredReviewSurfaceApi._closeArtifactViewer(restoreFocus);
        return;
    }
    state.portalUi.artifactViewer.open = false;
    renderArtifactViewer();
    if (restoreFocus) _restoreOverlayFocus();
}

function _openArtifactViewer(job, artifact, trigger = document.activeElement, surface = 'artifact_review') {
    if (!_artifactViewerEnabled() || !els.artifactViewerModal) return false;
    if (!job || !artifact) {
        createToast('No artifact is available for this selection.', 'info');
        return false;
    }
    if (deferredReviewSurfaceApi) {
        return deferredReviewSurfaceApi._openArtifactViewer(job, artifact, trigger);
    }
    _primeDeferredReviewSurface('viewer');
    void _loadDeferredReviewSurface().then((api) => {
        if (api) {
            api._openArtifactViewer(job, artifact, trigger);
            return;
        }
        _openManagedArtifactWindow(job, artifact, surface);
    });
    return true;
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
if (els.closeArtifactViewerBtn) {
    els.closeArtifactViewerBtn.addEventListener('click', () => _closeArtifactViewer());
}
if (els.artifactViewerModal) {
    els.artifactViewerModal.addEventListener('click', (event) => {
        if (event.target === els.artifactViewerModal) _closeArtifactViewer();
    });
}
if (els.artifactViewerPrevBtn) {
    els.artifactViewerPrevBtn.addEventListener('click', () => {
        _navigateArtifactViewerSelection(-1);
    });
}
if (els.artifactViewerNextBtn) {
    els.artifactViewerNextBtn.addEventListener('click', () => {
        _navigateArtifactViewerSelection(1);
    });
}
if (els.artifactViewerZoomOutBtn) {
    els.artifactViewerZoomOutBtn.addEventListener('click', () => {
        _setArtifactViewerZoom((state.portalUi?.artifactViewer?.zoomPercent || 100) - 25);
    });
}
if (els.artifactViewerZoomInBtn) {
    els.artifactViewerZoomInBtn.addEventListener('click', () => {
        _setArtifactViewerZoom((state.portalUi?.artifactViewer?.zoomPercent || 100) + 25);
    });
}
if (els.artifactViewerResetZoomBtn) {
    els.artifactViewerResetZoomBtn.addEventListener('click', () => {
        _setArtifactViewerZoom(100);
    });
}
if (els.artifactViewerOpenRawBtn) {
    els.artifactViewerOpenRawBtn.addEventListener('click', () => {
        const url = sanitizeManagedAssetUrl(els.artifactViewerOpenRawBtn.dataset.url);
        if (!url) {
            createToast('No artifact URL is available for this selection.', 'error');
            return;
        }
        window.open(url, '_blank', 'noopener,noreferrer');
    });
}
if (els.artifactViewerCopyPathBtn) {
    els.artifactViewerCopyPathBtn.addEventListener('click', async () => {
        const path = String(els.artifactViewerCopyPathBtn.dataset.path || '').trim();
        if (!path) {
            createToast('No artifact path is available for this selection.', 'error');
            return;
        }
        await copyToClipboard(path);
    });
}
if (els.artifactViewerCopyFingerprintBtn) {
    els.artifactViewerCopyFingerprintBtn.addEventListener('click', async () => {
        const fingerprint = String(els.artifactViewerCopyFingerprintBtn.dataset.fingerprint || '').trim();
        if (!fingerprint) {
            createToast('No artifact fingerprint is available for this selection.', 'error');
            return;
        }
        await copyToClipboard(fingerprint);
    });
}

document.addEventListener('keydown', (e) => {
    if (_trapOverlayFocus(e)) {
        return;
    }
    const key = String(e.key || '');
    const isPlainShortcut = !e.ctrlKey && !e.metaKey && !e.altKey && !_isTypingTarget(e.target);
    if (e.key === "Escape" && els.artifactViewerModal && !els.artifactViewerModal.classList.contains("hidden")) {
        e.preventDefault();
        _closeArtifactViewer();
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
    if (isPlainShortcut && (key === '?' || (key === '/' && e.shiftKey))) {
        if (els.shortcutsModal && els.shortcutsModal.classList.contains('hidden')) {
            e.preventDefault();
            toggleModal(true);
        }
        return;
    }
    if (isPlainShortcut && Object.prototype.hasOwnProperty.call(WORKSPACE_VIEW_SHORTCUTS, key)) {
        const nextView = WORKSPACE_VIEW_SHORTCUTS[key];
        if (nextView === 'review' && !state.selectedJobId) {
            e.preventDefault();
            createToast('Select a run first, then open its review surface.', 'info');
            return;
        }
        e.preventDefault();
        navigateConsoleView(nextView);
        return;
    }
    if (isPlainShortcut && els.artifactViewerModal && !els.artifactViewerModal.classList.contains('hidden')) {
        if (key === 'ArrowLeft') {
            e.preventDefault();
            _navigateArtifactViewerSelection(-1);
            return;
        }
        if (key === 'ArrowRight') {
            e.preventDefault();
            _navigateArtifactViewerSelection(1);
            return;
        }
        if (key === '+' || key === '=') {
            e.preventDefault();
            _setArtifactViewerZoom((state.portalUi?.artifactViewer?.zoomPercent || 100) + 25);
            return;
        }
        if (key === '-') {
            e.preventDefault();
            _setArtifactViewerZoom((state.portalUi?.artifactViewer?.zoomPercent || 100) - 25);
            return;
        }
        if (key === '0') {
            e.preventDefault();
            _setArtifactViewerZoom(100);
            return;
        }
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
        _finalizePortalRumVitals();
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
                captioning: null,
                dispatchTools: false,
            };
        }
        if (!Object.prototype.hasOwnProperty.call(state.portalUi.disclosurePrefs, 'captioning')) {
            state.portalUi.disclosurePrefs.captioning = null;
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
    registerDisclosurePanel('captioning', els.captioningDetails);
    registerDisclosurePanel('dispatchTools', els.dispatchToolsDetails);
}

// ============================================================================
// 13. INITIALIZATION
// ============================================================================

portalRenderSurfaces.register('jobQueue', {
    init(currentState) {
        if (!currentState || !currentState.jobs) return;
    },
    render() {
        renderJobQueue();
    }
});

window.addEventListener('beforeunload', _flushPendingTransientPortalDraftPersist);
window.addEventListener('beforeunload', cleanupActiveJobHandles);
window.addEventListener('beforeunload', _flushPortalRumOnPagehide);
window.addEventListener('pagehide', _flushPendingTransientPortalDraftPersist);
window.addEventListener('pagehide', cleanupActiveJobHandles);
window.addEventListener('pagehide', _flushPortalRumOnPagehide);
window.addEventListener('pageshow', () => {
    reconcileBuildSurfaceFromDom();
});
window.addEventListener('focus', () => {
    reconcileBuildSurfaceFromDom();
});
window.addEventListener('popstate', () => {
    applyConsoleRouteFromLocation(true);
    portalRenderSurfaces.render('jobQueue', state);
});

async function init() {
    const themeQuery = window.matchMedia('(prefers-color-scheme: dark)');
    _migrateThemePreferenceStorage();
    const savedThemePreference = _normalizeThemePreference(localStorage.getItem(THEME_STORAGE_KEY)) || 'system';
    applyThemePreference(savedThemePreference, { persist: false, themeQuery });
    setupAmbientMotion();
    _startPortalRumObservers();

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
    portalRenderSurfaces.init(state);
    if (window.requestAnimationFrame) {
        window.requestAnimationFrame(() => {
            reconcileBuildSurfaceFromDom();
            _recordPortalRumMilestone('portal_shell_rendered', _portalRumNow(), {
                traceparent: state.rum.pageTraceparent
            });
        });
    } else {
        window.setTimeout(() => {
            reconcileBuildSurfaceFromDom();
            _recordPortalRumMilestone('portal_shell_rendered', _portalRumNow(), {
                traceparent: state.rum.pageTraceparent
            });
        }, 0);
    }
    setupSectionRail();
    _syncBootstrapUi();
    renderJobQueue();
    startHealthPolling();
    await bootstrapPromise;
    _primeDeferredReviewSurface('bootstrap');
    _primeDeferredOperateSurface();
    _primeDeferredBuildSurface();
    _restoreTransientPortalDraft();
    updateUIFromState();
    _persistTransientPortalDraft();
    portalRenderSurfaces.render('jobQueue', state);
    void checkBackend(true);
    void fetchConfigMetadata(state.pipeline, true);
}

document.addEventListener('DOMContentLoaded', () => {
    void init();
});
