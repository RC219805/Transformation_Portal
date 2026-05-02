import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import crypto from "node:crypto";
import zlib from "node:zlib";

import postcss from "postcss";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTDOOR_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(FRONTDOOR_ROOT, "..", "..");
const PORTAL_CSS_SOURCE_DIR = path.resolve(FRONTDOOR_ROOT, "portal-src", "styles");
const PORTAL_CSS_INDEX_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "index.css");
const PORTAL_CSS_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal.css");
const PORTAL_ASSETS_DIR = path.dirname(PORTAL_CSS_ASSET_PATH);
const PORTAL_REVIEW_CSS_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-review.css");
const PORTAL_HTML_PATH = path.resolve(REPO_ROOT, "portal.html");
const PORTAL_SOURCE_DIR = path.resolve(FRONTDOOR_ROOT, "portal-src");
const PORTAL_ASSET_JS_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal.js");
const PORTAL_REVIEW_JS_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-review.js");
const BASELINE_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "architecture-baseline.json");
const OWNERSHIP_DRAIN_REPORT_PATH = path.resolve(FRONTDOOR_ROOT, "reports", "portal-css-ownership-drain.json");
const COMPAT_HOLD_UTILITIES_SOURCE = "./utilities.compat-hold.css";
const COMPAT_HOLD_UTILITIES_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "utilities.compat-hold.css");
const OVERRIDES_COMPAT_SOURCE = "./overrides.compat.css";
const OVERRIDES_COMPAT_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "overrides.compat.css");
const PHASE9_DUPLICATE_PHASE = "phase-9-duplicate-ownership-closure";
const PHASE9_DUPLICATE_CONTEXT_COUNT_BEFORE = 87;
const PHASE10_ADDITIVE_PHASE = "phase-10-css-additive-duplicate-consolidation";
const PHASE10_ADDITIVE_CONTEXT_COUNT_BEFORE = 30;
const PHASE10_SAFE_CANDIDATE_COUNT_BEFORE = 1;
const PHASE10_GENERATED_PORTAL_CSS_HASH_BEFORE =
  "c4288e656c797f547ca6d89c1096e79fa34435782f0aaaf3be80eb1bb0561471";
const PHASE10_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE = "847cbddecbf7";
const PHASE10_GENERATED_PORTAL_CSS_BYTES_BEFORE = 80547;
const PHASE10_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE = 15720;
const PHASE11_SURFACE_PHASE = "phase-11-css-surface-list-consolidation";
const PHASE11_DUPLICATE_CONTEXT_COUNT_BEFORE = 85;
const PHASE11_DUPLICATE_CONTEXT_COUNT_AFTER = 76;
const PHASE11_ADDITIVE_CONTEXT_COUNT_BEFORE = 29;
const PHASE11_ADDITIVE_CONTEXT_COUNT_AFTER = 20;
const PHASE11_CONFLICTING_PERMANENT_CONTEXT_COUNT = 56;
const PHASE11_EXPECTED_CONSOLIDATED_CONTEXT_COUNT = 9;
const PHASE11_TARGET_FILE = "web/secure-landing/portal-src/styles/components/surface-normalization.css";
const PHASE11_GENERATED_PORTAL_CSS_HASH_BEFORE =
  "fce12e29f1800375b5c34e1f0e1ebc9d3981ab1a6f731bea6a3e0e0d2212151e";
const PHASE11_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE = "d72696ab972c";
const PHASE11_GENERATED_PORTAL_CSS_BYTES_BEFORE = 80599;
const PHASE11_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE = 15721;
const PHASE11_BASELINE_SHA256 =
  "ab9f4a92eb699d811a85d737ef3f5cb2b285e6c809afcf34bbf179e1e229d17a";
const PHASE11_TARGET_DUPLICATE_KEYS = new Set([
  ".review-compare-summary|||components|||",
  "#artifactMetadataBar|||components|||",
  "#artifactMetadataCard|||components|||",
  "#artifactPreviewStage|||components|||",
  "#reconstructionRuntimeSummary|||components|||",
  ".review-status-banner[data-tone=\"ready\"]|||components|||",
  ".review-status-banner[data-tone=\"warning\"]|||components|||",
  ".review-status-banner[data-tone=\"error\"]|||components|||",
  ".review-status-banner[data-tone=\"info\"]|||components|||"
]);
const PHASE12_COMPONENT_PHASE = "phase-12-component-singleton-consolidation";
const PHASE12_DUPLICATE_CONTEXT_COUNT_BEFORE = 76;
const PHASE12_DUPLICATE_CONTEXT_COUNT_AFTER = 73;
const PHASE12_ADDITIVE_CONTEXT_COUNT_BEFORE = 20;
const PHASE12_ADDITIVE_CONTEXT_COUNT_AFTER = 17;
const PHASE12_CONFLICTING_PERMANENT_CONTEXT_COUNT = 56;
const PHASE12_EXPECTED_CONSOLIDATED_CONTEXT_COUNT = 3;
const PHASE12_GENERATED_PORTAL_CSS_HASH_BEFORE =
  "67a600aafdfc066a8adf373e1323b287939977ebc306d4e1c0635069fddbbd87";
const PHASE12_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE = "8f492bf35cbe";
const PHASE12_GENERATED_PORTAL_CSS_BYTES_BEFORE = 80353;
const PHASE12_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE = 15712;
const PHASE12_SHELL_FOUNDATION_FILE =
  "web/secure-landing/portal-src/styles/components/shell-foundation.css";
const PHASE12_OPERATOR_CONSOLE_FILE =
  "web/secure-landing/portal-src/styles/components/operator-console.css";
const PHASE12_TARGET_SELECTORS = [
  ".ambient-orb",
  ".ambient-ring",
  ".overview-actions @media (max-width: 767px)"
];
const PHASE12_TARGET_DUPLICATE_KEYS = new Set([
  ".ambient-orb|||components|||",
  ".ambient-ring|||components|||",
  ".overview-actions|||components|||@media (max-width: 767px)"
]);
const PHASE13_INTERACTION_PHASE = "phase-13-interaction-outline-consolidation";
const PHASE13_DUPLICATE_CONTEXT_COUNT_BEFORE = 73;
const PHASE13_DUPLICATE_CONTEXT_COUNT_AFTER = 70;
const PHASE13_ADDITIVE_CONTEXT_COUNT_BEFORE = 17;
const PHASE13_ADDITIVE_CONTEXT_COUNT_AFTER = 14;
const PHASE13_CONFLICTING_PERMANENT_CONTEXT_COUNT = 56;
const PHASE13_EXPECTED_CONSOLIDATED_CONTEXT_COUNT = 3;
const PHASE13_GENERATED_PORTAL_CSS_HASH_BEFORE =
  "5c82d054b928ac320c932cd9eb133fb2f39cfdc9e865c648cc9987de68c0e37b";
const PHASE13_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE = "61c134a0012d";
const PHASE13_GENERATED_PORTAL_CSS_BYTES_BEFORE = 80390;
const PHASE13_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE = 15713;
const PHASE13_OPERATOR_CONSOLE_FILE =
  "web/secure-landing/portal-src/styles/components/operator-console.css";
const PHASE13_DISPATCH_SURFACES_FILE =
  "web/secure-landing/portal-src/styles/components/dispatch-surfaces.css";
const PHASE13_WORKSPACE_SURFACES_FILE =
  "web/secure-landing/portal-src/styles/components/workspace-surfaces.css";
const PHASE13_TARGET_SELECTORS = [
  ".build-step-tab:hover",
  ".dispatch-tool-btn:hover",
  ".workspace-link:hover"
];
const PHASE13_TARGET_DUPLICATE_KEYS = new Set([
  ".build-step-tab:hover|||components|||",
  ".dispatch-tool-btn:hover|||components|||",
  ".workspace-link:hover|||components|||"
]);
const PHASE13_TARGET_FILES_BY_KEY = new Map([
  [".build-step-tab:hover|||components|||", PHASE13_OPERATOR_CONSOLE_FILE],
  [".dispatch-tool-btn:hover|||components|||", PHASE13_DISPATCH_SURFACES_FILE],
  [".workspace-link:hover|||components|||", PHASE13_WORKSPACE_SURFACES_FILE]
]);
const PHASE13_BASELINE_SHA256 =
  "0cd2906f93cdc44817461fe60c7d0cda970ac8b5aa36715fce27dadf058c2ed9";
const PHASE14_SKELETON_PHASE = "phase-14-skeleton-primitive-consolidation";
const PHASE14_DUPLICATE_CONTEXT_COUNT_BEFORE = 70;
const PHASE14_DUPLICATE_CONTEXT_COUNT_AFTER = 67;
const PHASE14_ADDITIVE_CONTEXT_COUNT_BEFORE = 14;
const PHASE14_ADDITIVE_CONTEXT_COUNT_AFTER = 11;
const PHASE14_CONFLICTING_PERMANENT_CONTEXT_COUNT = 56;
const PHASE14_EXPECTED_CONSOLIDATED_CONTEXT_COUNT = 3;
const PHASE14_GENERATED_PORTAL_CSS_HASH_BEFORE =
  "3e6719511b1536ebe2e4c067b693930aa77ab242115368ae4ba3aa8e2aa6f3b3";
const PHASE14_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE = "c06fbcdf1f37";
const PHASE14_GENERATED_PORTAL_CSS_BYTES_BEFORE = 80569;
const PHASE14_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE = 15743;
const PHASE14_BASELINE_SHA256 =
  "f3c0c14b22eb72a8834edd5936458c9fef651804397fbbaf969bdf141bc7dea2";
const PHASE14_TARGET_FILE = PHASE13_DISPATCH_SURFACES_FILE;
const PHASE14_TARGET_SELECTORS = [".skeleton-line", ".skeleton-block", ".skeleton-pill"];
const PHASE14_TARGET_DUPLICATE_KEYS = new Set(PHASE14_TARGET_SELECTORS.map((selector) => `${selector}|||components|||`));
const PHASE14_SHARED_PROPERTIES = ["background", "display", "overflow", "position"];
const PHASE14_SINGLETON_PROPERTIES_BY_SELECTOR = new Map([
  [".skeleton-line", ["border-radius", "height", "width"]],
  [".skeleton-block", ["border-radius", "min-height"]],
  [".skeleton-pill", ["border-radius", "height", "width"]]
]);
const PHASE14_VARIANTS_BY_SELECTOR = new Map([
  [".skeleton-line", [".skeleton-line-short", ".skeleton-line-medium", ".skeleton-line-tiny"]],
  [".skeleton-block", [".skeleton-block-compact"]],
  [".skeleton-pill", [".skeleton-pill-short"]]
]);
const PHASE15_SURFACE_LOADING_PHASE = "phase-15-surface-loading-consolidation";
const PHASE15_DUPLICATE_CONTEXT_COUNT_BEFORE = 67;
const PHASE15_DUPLICATE_CONTEXT_COUNT_AFTER = 64;
const PHASE15_ADDITIVE_CONTEXT_COUNT_BEFORE = 11;
const PHASE15_ADDITIVE_CONTEXT_COUNT_AFTER = 8;
const PHASE15_CONFLICTING_PERMANENT_CONTEXT_COUNT = 56;
const PHASE15_EXPECTED_CONSOLIDATED_CONTEXT_COUNT = 3;
const PHASE15_GENERATED_PORTAL_CSS_HASH_BEFORE =
  "4598d98fd19ccaf0728403838d7dfdc395cc1904b1cb147af559ecda6bce726a";
const PHASE15_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE = "c798785d38cf";
const PHASE15_GENERATED_PORTAL_CSS_BYTES_BEFORE = 80661;
const PHASE15_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE = 15738;
const PHASE15_TARGET_FILE = PHASE13_DISPATCH_SURFACES_FILE;
const PHASE15_OPERATOR_CONSOLE_FILE = PHASE13_OPERATOR_CONSOLE_FILE;
const PHASE15_TARGET_SELECTORS = [".surface-loading", ".surface-loading::after", ".dark .surface-loading"];
const PHASE15_TARGET_DUPLICATE_KEYS = new Set(PHASE15_TARGET_SELECTORS.map((selector) => `${selector}|||components|||`));
const PHASE15_DISPATCH_PROPERTIES_BY_SELECTOR = new Map([
  [".surface-loading", ["box-shadow", "position", "transition"]],
  [
    ".surface-loading::after",
    ["border-radius", "content", "height", "left", "opacity", "pointer-events", "position", "right", "top"]
  ],
  [".dark .surface-loading", ["box-shadow"]]
]);
const PHASE15_OPERATOR_PROPERTIES_BY_SELECTOR = new Map([
  [".surface-loading", ["background", "border"]],
  [".surface-loading::after", ["background"]],
  [".dark .surface-loading", ["background", "border-color"]]
]);
const PHASE16_REVIEW_SURFACE_PHASE = "phase-16-review-surface-consolidation";
const PHASE16_DUPLICATE_CONTEXT_COUNT_BEFORE = 64;
const PHASE16_DUPLICATE_CONTEXT_COUNT_AFTER = 61;
const PHASE16_ADDITIVE_CONTEXT_COUNT_BEFORE = 8;
const PHASE16_ADDITIVE_CONTEXT_COUNT_AFTER = 5;
const PHASE16_CONFLICTING_PERMANENT_CONTEXT_COUNT = 56;
const PHASE16_EXPECTED_CONSOLIDATED_CONTEXT_COUNT = 3;
const PHASE16_GENERATED_PORTAL_CSS_HASH_BEFORE =
  "eb245c4f90079fb63803aa1ccbfc40e771a6bf53f85a6b48994fabdea8c62281";
const PHASE16_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE = "ba13115d7572";
const PHASE16_GENERATED_PORTAL_CSS_BYTES_BEFORE = 80598;
const PHASE16_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE = 15725;
const PHASE16_SURFACE_NORMALIZATION_FILE =
  "web/secure-landing/portal-src/styles/components/surface-normalization.css";
const PHASE16_OPERATOR_CONSOLE_FILE = PHASE13_OPERATOR_CONSOLE_FILE;
const PHASE16_TARGET_SELECTORS = [".review-status-banner", ".dark .review-status-banner", ".review-provenance-item"];
const PHASE16_TARGET_DUPLICATE_KEYS = new Set(PHASE16_TARGET_SELECTORS.map((selector) => `${selector}|||components|||`));
const PHASE16_SURFACE_PROPERTIES_BY_SELECTOR = new Map([
  [".review-status-banner", ["background", "border-color", "border-radius"]],
  [".dark .review-status-banner", ["background"]],
  [".review-provenance-item", ["background", "border-color", "border-radius"]]
]);
const PHASE16_SECONDARY_PROPERTIES_BY_SELECTOR = new Map([
  [".review-status-banner", ["box-shadow"]],
  [".dark .review-status-banner", ["box-shadow"]],
  [".review-provenance-item", ["box-shadow"]]
]);
const PHASE17_SURFACE_FINAL_PASS_PHASE = "phase-17-surface-final-pass-consolidation";
const PHASE17_DUPLICATE_CONTEXT_COUNT_BEFORE = 61;
const PHASE17_DUPLICATE_CONTEXT_COUNT_AFTER = 58;
const PHASE17_ADDITIVE_CONTEXT_COUNT_BEFORE = 5;
const PHASE17_ADDITIVE_CONTEXT_COUNT_AFTER = 2;
const PHASE17_CONFLICTING_PERMANENT_CONTEXT_COUNT = 56;
const PHASE17_EXPECTED_CONSOLIDATED_CONTEXT_COUNT = 3;
const PHASE17_GENERATED_PORTAL_CSS_HASH_BEFORE =
  "ad357cc9c93fe976e349ddcbcac33a62e8014eaa7003a36d561b5c3c7672b429";
const PHASE17_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE = "ea5def42614e";
const PHASE17_GENERATED_PORTAL_CSS_BYTES_BEFORE = 80813;
const PHASE17_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE = 15731;
const PHASE17_SURFACE_NORMALIZATION_FILE = PHASE16_SURFACE_NORMALIZATION_FILE;
const PHASE17_OPERATOR_CONSOLE_FILE = PHASE13_OPERATOR_CONSOLE_FILE;
const PHASE17_SHELL_FOUNDATION_FILE = PHASE12_SHELL_FOUNDATION_FILE;
const PHASE17_TARGET_SELECTORS = [".workspace-shell", ".shell-noise", ".review-provenance-label"];
const PHASE17_TARGET_DUPLICATE_KEYS = new Set(PHASE17_TARGET_SELECTORS.map((selector) => `${selector}|||components|||`));
const PHASE17_FINAL_PROPERTIES_BY_SELECTOR = new Map([
  [".workspace-shell", ["background", "border-color", "border-radius", "box-shadow", "position", "transition"]],
  [
    ".shell-noise",
    ["background-image", "background-position", "background-size", "display", "inset", "opacity", "pointer-events", "position", "z-index"]
  ],
  [".review-provenance-label", ["font-size", "letter-spacing", "line-height"]]
]);
const PHASE17_PRIMARY_PROPERTIES_BY_SELECTOR = new Map([
  [".workspace-shell", ["background", "border-color", "border-radius"]],
  [
    ".shell-noise",
    ["background-image", "background-position", "background-size", "inset", "opacity", "pointer-events", "position", "z-index"]
  ],
  [".review-provenance-label", ["font-size", "line-height"]]
]);
const PHASE17_SOURCE_PROPERTIES_BY_SELECTOR = new Map([
  [".workspace-shell", ["position", "transition"]],
  [".shell-noise", ["display"]],
  [".review-provenance-label", ["letter-spacing"]]
]);
const PHASE17_WORKSPACE_SHADOW_PROPERTIES = ["box-shadow"];
const VALID_PHASE9_CONTEXT_TYPES = new Set([
  "responsive-context",
  "theme-context",
  "accessibility-context",
  "runtime-state",
  "component-family-shared-state",
  "utility-owned",
  "document-root-normalization",
  "surface-normalization-final-pass"
]);
const VALID_PHASE9_DISPOSITIONS = new Set(["keep-owned", "defer-owned", "report-only"]);
const VALID_PHASE9_REMOVAL_STATUSES = new Set(["permanent", "removable-later"]);
const VALID_PHASE9_DECLARATION_CONFLICTS = new Set(["identical", "additive", "conflicting", "contextual"]);
const PHASE10_UNSAFE_REASONS = new Set([
  "source-order-sensitive",
  "shorthand-longhand-overlap",
  "custom-property-order-sensitive",
  "selector-list-ambiguous",
  "cross-layer",
  "cross-context",
  "hotspot",
  "conflicting",
  "intervening-overlap",
  "coverage-ambiguous"
]);
const PHASE11_UNSAFE_REASONS = new Set([
  "selector-not-phase11-target",
  "conflicting-permanent",
  "hotspot",
  "cross-layer",
  "cross-context",
  "selector-list-coverage-ambiguous",
  "background-shorthand-order-sensitive",
  "shorthand-longhand-overlap",
  "intervening-overlap",
  "dark-pair-missing",
  "source-order-sensitive",
  "specificity-changing-grouping"
]);
const PHASE12_UNSAFE_REASONS = new Set([
  "selector-not-phase12-target",
  "conflicting-permanent",
  "hotspot",
  "cross-layer",
  "cross-context",
  "selector-list-coverage-ambiguous",
  "shorthand-longhand-overlap",
  "custom-property-order-sensitive",
  "intervening-overlap",
  "source-order-sensitive",
  "specificity-changing-grouping"
]);
const PHASE13_UNSAFE_REASONS = new Set([
  "selector-not-phase13-target",
  "conflicting-permanent",
  "hotspot",
  "cross-layer",
  "cross-context",
  "selector-list-coverage-ambiguous",
  "focus-visible-coverage-missing",
  "missing-hover-outline",
  "intervening-overlap",
  "source-order-sensitive",
  "specificity-changing-grouping"
]);
const PHASE14_UNSAFE_REASONS = new Set([
  "selector-not-phase14-target",
  "conflicting-permanent",
  "hotspot",
  "cross-layer",
  "cross-context",
  "selector-list-coverage-ambiguous",
  "shared-selector-list-drift",
  "singleton-declaration-missing",
  "variant-order-sensitive",
  "specificity-changing-grouping"
]);
const PHASE15_UNSAFE_REASONS = new Set([
  "selector-not-phase15-target",
  "conflicting-permanent",
  "hotspot",
  "cross-layer",
  "cross-context",
  "selector-list-coverage-ambiguous",
  "source-file-drift",
  "dispatch-declaration-missing",
  "operator-declaration-drift",
  "source-order-sensitive",
  "specificity-changing-grouping"
]);
const PHASE16_UNSAFE_REASONS = new Set([
  "selector-not-phase16-target",
  "conflicting-permanent",
  "hotspot",
  "cross-layer",
  "cross-context",
  "selector-list-coverage-ambiguous",
  "source-file-drift",
  "surface-declaration-missing",
  "operator-declaration-drift",
  "source-order-sensitive",
  "tone-order-sensitive",
  "dark-provenance-override-missing",
  "specificity-changing-grouping"
]);
const PHASE17_UNSAFE_REASONS = new Set([
  "selector-not-phase17-target",
  "conflicting-permanent",
  "hotspot",
  "cross-layer",
  "cross-context",
  "selector-list-coverage-ambiguous",
  "source-file-drift",
  "primary-declaration-missing",
  "source-declaration-drift",
  "source-order-sensitive",
  "ambient-override-missing",
  "dark-override-missing",
  "specificity-changing-grouping"
]);
const PORTAL_CSS_RENDER_TOKENS = {
  "__PORTAL_FONT_SANS_URL__": "fonts/portal-sans.woff2",
  "__PORTAL_FONT_MONO_URL__": "fonts/portal-mono.woff2"
};

const args = new Set(process.argv.slice(2));
const WRITE_BASELINE = args.has("--write-baseline");
const WRITE_OWNERSHIP_REPORT = args.has("--write-ownership-report");
const REPORT = args.has("--report") || WRITE_BASELINE;
const SENTINEL_FIXTURE_ARG = "--check-sentinel-fixture";
const sentinelFixtureIndex = process.argv.indexOf(SENTINEL_FIXTURE_ARG);
const SENTINEL_FIXTURE_PATH = sentinelFixtureIndex >= 0 ? process.argv[sentinelFixtureIndex + 1] : "";
const DUPLICATE_BASELINE_FIXTURE_ARG = "--check-duplicate-baseline-fixture";
const duplicateBaselineFixtureIndex = process.argv.indexOf(DUPLICATE_BASELINE_FIXTURE_ARG);
const DUPLICATE_BASELINE_FIXTURE_PATH =
  duplicateBaselineFixtureIndex >= 0 ? process.argv[duplicateBaselineFixtureIndex + 1] : "";
const DUPLICATE_BASELINE_DUPLICATES_PATH =
  duplicateBaselineFixtureIndex >= 0 ? process.argv[duplicateBaselineFixtureIndex + 2] : "";
const PHASE10_ADDITIVE_FIXTURE_ARG = "--check-phase10-additive-fixture";
const phase10AdditiveFixtureIndex = process.argv.indexOf(PHASE10_ADDITIVE_FIXTURE_ARG);
const PHASE10_ADDITIVE_FIXTURE_PATH =
  phase10AdditiveFixtureIndex >= 0 ? process.argv[phase10AdditiveFixtureIndex + 1] : "";
const PHASE11_SURFACE_FIXTURE_ARG = "--check-phase11-surface-fixture";
const phase11SurfaceFixtureIndex = process.argv.indexOf(PHASE11_SURFACE_FIXTURE_ARG);
const PHASE11_SURFACE_FIXTURE_PATH =
  phase11SurfaceFixtureIndex >= 0 ? process.argv[phase11SurfaceFixtureIndex + 1] : "";
const PHASE12_COMPONENT_FIXTURE_ARG = "--check-phase12-component-fixture";
const phase12ComponentFixtureIndex = process.argv.indexOf(PHASE12_COMPONENT_FIXTURE_ARG);
const PHASE12_COMPONENT_FIXTURE_PATH =
  phase12ComponentFixtureIndex >= 0 ? process.argv[phase12ComponentFixtureIndex + 1] : "";
const PHASE13_INTERACTION_FIXTURE_ARG = "--check-phase13-interaction-fixture";
const phase13InteractionFixtureIndex = process.argv.indexOf(PHASE13_INTERACTION_FIXTURE_ARG);
const PHASE13_INTERACTION_FIXTURE_PATH =
  phase13InteractionFixtureIndex >= 0 ? process.argv[phase13InteractionFixtureIndex + 1] : "";
const PHASE14_SKELETON_FIXTURE_ARG = "--check-phase14-skeleton-fixture";
const phase14SkeletonFixtureIndex = process.argv.indexOf(PHASE14_SKELETON_FIXTURE_ARG);
const PHASE14_SKELETON_FIXTURE_PATH =
  phase14SkeletonFixtureIndex >= 0 ? process.argv[phase14SkeletonFixtureIndex + 1] : "";
const PHASE15_SURFACE_LOADING_FIXTURE_ARG = "--check-phase15-surface-loading-fixture";
const phase15SurfaceLoadingFixtureIndex = process.argv.indexOf(PHASE15_SURFACE_LOADING_FIXTURE_ARG);
const PHASE15_SURFACE_LOADING_FIXTURE_PATH =
  phase15SurfaceLoadingFixtureIndex >= 0 ? process.argv[phase15SurfaceLoadingFixtureIndex + 1] : "";
const PHASE16_REVIEW_SURFACE_FIXTURE_ARG = "--check-phase16-review-surface-fixture";
const phase16ReviewSurfaceFixtureIndex = process.argv.indexOf(PHASE16_REVIEW_SURFACE_FIXTURE_ARG);
const PHASE16_REVIEW_SURFACE_FIXTURE_PATH =
  phase16ReviewSurfaceFixtureIndex >= 0 ? process.argv[phase16ReviewSurfaceFixtureIndex + 1] : "";
const PHASE17_SURFACE_FINAL_PASS_FIXTURE_ARG = "--check-phase17-surface-final-pass-fixture";
const phase17SurfaceFinalPassFixtureIndex = process.argv.indexOf(PHASE17_SURFACE_FINAL_PASS_FIXTURE_ARG);
const PHASE17_SURFACE_FINAL_PASS_FIXTURE_PATH =
  phase17SurfaceFinalPassFixtureIndex >= 0 ? process.argv[phase17SurfaceFinalPassFixtureIndex + 1] : "";

const EXPECTED_LAYER_ORDER = ["tokens", "base", "components", "utilities", "overrides"];
const EXPECTED_LAYER_IMPORTS = [
  ["../../../../web/shared/shared-ui-tokens.css", "tokens"],
  ["./tokens.css", "tokens"],
  ["./base.css", "base"],
  ["./components/status-feedback.css", "components"],
  ["./components/shell-foundation.css", "components"],
  ["./components/console-context.css", "components"],
  ["./components/dispatch-surfaces.css", "components"],
  ["./components/responsive-layout.css", "components"],
  ["./components/workspace-surfaces.css", "components"],
  ["./components/operator-console.css", "components"],
  ["./components/surface-normalization.css", "components"],
  ["./utilities.required.css", "utilities"],
  ["./utilities.dynamic.css", "utilities"],
  [COMPAT_HOLD_UTILITIES_SOURCE, "utilities"],
  ["./overrides.performance.css", "overrides"],
  ["./overrides.accessibility.css", "overrides"]
];
const HOTSPOT_SELECTORS = new Set([
  ".shell-bg",
  ".workspace-rail",
  ".workspace-link",
  ".hero-action",
  ".operator-action-btn",
  ".stepper-nav-btn",
  ".dispatch-tool-btn",
  ".console-context-ribbon",
  ".topbar-status"
]);

const UTILITY_EXACT_CLASSES = new Set([
  "absolute",
  "antialiased",
  "block",
  "border",
  "flex",
  "fixed",
  "grid",
  "group",
  "hidden",
  "inline",
  "inline-block",
  "inline-flex",
  "peer",
  "relative",
  "rounded",
  "sr-only",
  "sticky",
  "transform",
  "truncate",
  "uppercase"
]);
const UTILITY_OWNER_ALLOWLIST = new Map([
  ["dark", "theme owner class"],
  ["light", "theme owner class"],
  ["performance-lite", "performance mode owner class"]
]);
const UTILITY_PREFIX_PATTERN =
  /^(?:-?m[trblxy]?-.+|p[trblxy]?-.+|space-[xy]-.+|gap-.+|grid-cols-.+|col-span-.+|row-span-.+|flex-.+|items-.+|justify-.+|self-.+|place-.+|w-.+|h-.+|min-w-.+|min-h-.+|max-w-.+|max-h-.+|rounded-.+|border-.+|bg-.+|from-.+|via-.+|to-.+|text-.+|font-.+|tracking-.+|leading-.+|shadow-.+|ring-.+|opacity-.+|overflow-.+|object-.+|inset-.+|top-.+|right-.+|bottom-.+|left-.+|-?z-.+|cursor-.+|pointer-events-.+|select-.+|resize-.+|whitespace-.+|break-.+|duration-.+|ease-.+|transition-.+|translate-.+|scale-.+|backdrop-.+|animate-.+|outline-.+|fill-.+|stroke-.+|order-.+|basis-.+|shrink-.+|grow-.+|underline|no-underline)$/;
const UTILITY_VARIANTS = new Set([
  "active",
  "dark",
  "disabled",
  "focus",
  "focus-visible",
  "group-hover",
  "hover",
  "lg",
  "md",
  "peer-checked",
  "peer-focus-visible",
  "selection",
  "sm",
  "xl"
]);

function relativePath(filePath) {
  return path.relative(REPO_ROOT, filePath);
}

function readText(filePath) {
  return readFileSync(filePath, "utf-8");
}

function listFiles(directory, predicate) {
  const results = [];
  for (const entry of readdirSync(directory)) {
    const entryPath = path.join(directory, entry);
    const stats = statSync(entryPath);
    if (stats.isDirectory()) {
      results.push(...listFiles(entryPath, predicate));
    } else if (predicate(entryPath)) {
      results.push(entryPath);
    }
  }
  return results.sort();
}

function parseCss(filePath, content = readText(filePath)) {
  try {
    return postcss.parse(content, { from: filePath });
  } catch (error) {
    throw new Error(`${relativePath(filePath)} failed to parse as CSS: ${error.message}`);
  }
}

function splitSelectorList(selectorText) {
  const selectors = [];
  let current = "";
  let bracketDepth = 0;
  let parenDepth = 0;
  let quote = "";
  for (let index = 0; index < selectorText.length; index += 1) {
    const character = selectorText[index];
    const previous = selectorText[index - 1] || "";
    if (quote) {
      current += character;
      if (character === quote && previous !== "\\") {
        quote = "";
      }
      continue;
    }
    if (character === "\"" || character === "'") {
      quote = character;
      current += character;
      continue;
    }
    if (character === "[") {
      bracketDepth += 1;
    } else if (character === "]") {
      bracketDepth = Math.max(0, bracketDepth - 1);
    } else if (character === "(") {
      parenDepth += 1;
    } else if (character === ")") {
      parenDepth = Math.max(0, parenDepth - 1);
    } else if (character === "," && bracketDepth === 0 && parenDepth === 0) {
      const selector = current.trim().replace(/\s+/g, " ");
      if (selector) {
        selectors.push(selector);
      }
      current = "";
      continue;
    }
    current += character;
  }
  const trailing = current.trim().replace(/\s+/g, " ");
  if (trailing) {
    selectors.push(trailing);
  }
  return selectors;
}

function atRuleContext(node) {
  const contexts = [];
  let current = node.parent;
  while (current) {
    if (current.type === "atrule") {
      contexts.unshift(`@${current.name} ${current.params || ""}`.trim());
    }
    current = current.parent;
  }
  return contexts;
}

function layerContext(node) {
  return atRuleContext(node).filter((context) => context.startsWith("@layer "));
}

function selectorStateContext(selector) {
  const states = [];
  if (selector.includes(".dark")) {
    states.push("dark");
  }
  for (const state of [":hover", ":focus", ":focus-visible", ":disabled", ".is-active", ".hidden"]) {
    if (selector.includes(state)) {
      states.push(state.replace(/^[.:]/, ""));
    }
  }
  if (selector.includes("[data-")) {
    states.push("data-state");
  }
  return states.sort();
}

function declarationEntries(rule) {
  const entries = [];
  rule.walkDecls((decl) => {
    entries.push([decl.prop.trim(), decl.value.trim(), Boolean(decl.important)]);
  });
  return entries.sort((left, right) => left[0].localeCompare(right[0]) || left[1].localeCompare(right[1]));
}

function declarationSignature(entries) {
  return crypto.createHash("sha1").update(JSON.stringify(entries)).digest("hex");
}

function declarationMap(entries) {
  const map = new Map();
  for (const [property, value, important] of entries) {
    const next = `${important ? "!" : ""}${value}`;
    const values = map.get(property) || new Set();
    values.add(next);
    map.set(property, values);
  }
  return map;
}

function classifyDuplicate(records) {
  const signatures = new Set(records.map((record) => record.declarationSignature));
  if (signatures.size === 1) {
    return "identical";
  }
  const seen = new Map();
  let conflict = false;
  for (const record of records) {
    const map = declarationMap(record.declarations);
    for (const [property, values] of map) {
      const current = seen.get(property) || new Set();
      for (const value of values) {
        current.add(value);
      }
      if (current.size > 1) {
        conflict = true;
      }
      seen.set(property, current);
    }
  }
  return conflict ? "conflicting" : "additive";
}

function collectRuleRecords(cssFiles, layerByPath = new Map()) {
  const records = [];
  for (const filePath of cssFiles) {
    const root = parseCss(filePath);
    const layer = layerByPath.get(path.resolve(filePath)) || "unlayered";
    root.walkRules((rule) => {
      const context = atRuleContext(rule);
      const selectorList = splitSelectorList(rule.selector);
      for (const selector of selectorList) {
        const declarations = declarationEntries(rule);
        records.push({
          selector,
          layer,
          context,
          layerContext: layerContext(rule),
          stateContext: selectorStateContext(selector),
          source: relativePath(filePath),
          line: rule.source?.start?.line || 0,
          column: rule.source?.start?.column || 0,
          ruleSelector: rule.selector.trim().replace(/\s+/g, " "),
          selectorList,
          declarations,
          declarationSignature: declarationSignature(declarations)
        });
      }
    });
  }
  return records;
}

function duplicateKey(record) {
  return `${record.selector}|||${record.layer}|||${record.context.join(" > ")}`;
}

function collectDuplicateReport(records) {
  const byKey = new Map();
  for (const record of records) {
    const key = duplicateKey(record);
    const existing = byKey.get(key) || [];
    existing.push(record);
    byKey.set(key, existing);
  }
  return Array.from(byKey.entries())
    .filter(([, recordsForKey]) => recordsForKey.length > 1)
    .map(([key, recordsForKey]) => ({
      key,
      selector: recordsForKey[0].selector,
      layer: recordsForKey[0].layer,
      context: recordsForKey[0].context,
      stateContext: Array.from(new Set(recordsForKey.flatMap((record) => record.stateContext))).sort(),
      category: classifyDuplicate(recordsForKey),
      hotspot: HOTSPOT_SELECTORS.has(recordsForKey[0].selector),
      records: recordsForKey.map((record) => ({
        source: record.source,
        line: record.line,
        column: record.column,
        layer: record.layer,
        declarationSignature: record.declarationSignature,
        ruleSelector: record.ruleSelector,
        selectorList: record.selectorList,
        declarations: record.declarations,
        properties: Array.from(new Set(record.declarations.map(([property]) => property))).sort()
      }))
    }))
    .sort((left, right) => left.key.localeCompare(right.key));
}

function loadBaseline() {
  if (!existsSync(BASELINE_PATH)) {
    return null;
  }
  return JSON.parse(readText(BASELINE_PATH));
}

function productionLayerBySourcePath() {
  const imports = collectCssImportGraph(PORTAL_CSS_INDEX_PATH, []);
  const layerByPath = new Map();
  for (const record of imports) {
    if (record.resolvedPath) {
      layerByPath.set(path.resolve(record.resolvedPath), record.layerName || "unlayered");
    }
  }
  return layerByPath;
}

function uniqueSorted(values) {
  return Array.from(new Set(values)).sort();
}

function duplicateSources(duplicate) {
  return uniqueSorted(duplicate.records.map((record) => path.basename(record.source)));
}

function contextTypeForDuplicate(duplicate) {
  const selector = duplicate.selector || "";
  const contextText = (duplicate.context || []).join(" ");
  if (/prefers-reduced-motion|forced-colors|prefers-contrast/.test(contextText)) {
    return "accessibility-context";
  }
  if (contextText.includes("@media")) {
    return "responsive-context";
  }
  if (selector.startsWith(".dark ") || selector.includes("html:not(.light)")) {
    return "theme-context";
  }
  if (selector === "html" || selector.startsWith("#")) {
    return "document-root-normalization";
  }
  if (/\[data-|:hover|:focus|:focus-visible|:disabled|\.is-active/.test(selector)) {
    return "runtime-state";
  }
  if (duplicate.records.some((record) => path.basename(record.source).startsWith("utilities."))) {
    return "utility-owned";
  }
  if (duplicate.records.some((record) => record.source.endsWith("components/surface-normalization.css"))) {
    return "surface-normalization-final-pass";
  }
  return "component-family-shared-state";
}

function ownerReasonForDuplicate(duplicate) {
  const sourceList = duplicateSources(duplicate).join(", ");
  if (duplicate.category === "conflicting") {
    return `Intentional cascade: ${duplicate.selector} keeps the reviewed source order across ${sourceList}; the later production rule provides the final declaration set.`;
  }
  if (duplicate.category === "additive") {
    return `Additive ownership: ${duplicate.selector} splits non-overlapping declarations across ${sourceList} while preserving the current computed-style parity.`;
  }
  if (duplicate.category === "identical") {
    return `Identical duplicate retained as report-only evidence until a mechanically safe consolidation removes one source without selector-list coverage loss.`;
  }
  return `Contextual duplicate retained with explicit ownership until a later parity-backed consolidation proves removal safe.`;
}

function duplicateBaselineMetadata(duplicate) {
  return {
    key: duplicate.key,
    selector: duplicate.selector,
    layer: duplicate.layer,
    atRuleContext: duplicate.context,
    stateContext: duplicate.stateContext,
    category: duplicate.category,
    hotspot: duplicate.hotspot,
    records: duplicate.records.map((record) => ({
      source: record.source,
      line: record.line,
      column: record.column,
      layer: record.layer,
      declarationSignature: record.declarationSignature,
      properties: record.properties
    }))
  };
}

function baselineFromReport(duplicates) {
  return {
    version: 1,
    phase: PHASE9_DUPLICATE_PHASE,
    duplicateKeys: duplicates.map((duplicate) => ({
      ...duplicateBaselineMetadata(duplicate),
      owners: ["portal-css-architecture"],
      ownerReason: ownerReasonForDuplicate(duplicate),
      phase: PHASE9_DUPLICATE_PHASE,
      contextType: contextTypeForDuplicate(duplicate),
      disposition: duplicate.category === "additive" ? "report-only" : "keep-owned",
      removalStatus: duplicate.category === "additive" ? "removable-later" : "permanent",
      declarationConflict: duplicate.category,
      parity: "green"
    }))
  };
}

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function entryHasOwners(entry) {
  return Array.isArray(entry.owners) && entry.owners.some((owner) => isNonEmptyString(owner));
}

function checkDuplicateBaselineMetadata(entry, duplicate, failures, label) {
  const expected = duplicateBaselineMetadata(duplicate);
  for (const field of ["selector", "layer", "category", "hotspot"]) {
    if (JSON.stringify(entry[field]) !== JSON.stringify(expected[field])) {
      failures.push(`${label} duplicate ${duplicate.key} has stale ${field}`);
    }
  }
  for (const field of ["atRuleContext", "stateContext", "records"]) {
    if (JSON.stringify(entry[field]) !== JSON.stringify(expected[field])) {
      failures.push(`${label} duplicate ${duplicate.key} has stale ${field}`);
    }
  }
}

function checkDuplicateBaselineEntry(entry, duplicate, failures, label) {
  checkDuplicateBaselineMetadata(entry, duplicate, failures, label);
  if (!entryHasOwners(entry)) {
    failures.push(`${label} duplicate ${duplicate.key} missing owners`);
  }
  if (!isNonEmptyString(entry.ownerReason)) {
    failures.push(`${label} duplicate ${duplicate.key} missing ownerReason`);
  }
  if (entry.phase !== PHASE9_DUPLICATE_PHASE) {
    failures.push(`${label} duplicate ${duplicate.key} must declare phase ${PHASE9_DUPLICATE_PHASE}`);
  }
  if (!VALID_PHASE9_CONTEXT_TYPES.has(entry.contextType)) {
    failures.push(`${label} duplicate ${duplicate.key} has invalid contextType ${entry.contextType || "missing"}`);
  }
  if (!VALID_PHASE9_DISPOSITIONS.has(entry.disposition)) {
    failures.push(`${label} duplicate ${duplicate.key} has invalid disposition ${entry.disposition || "missing"}`);
  }
  if (!VALID_PHASE9_REMOVAL_STATUSES.has(entry.removalStatus)) {
    failures.push(`${label} duplicate ${duplicate.key} has invalid removalStatus ${entry.removalStatus || "missing"}`);
  }
  if (!VALID_PHASE9_DECLARATION_CONFLICTS.has(entry.declarationConflict)) {
    failures.push(`${label} duplicate ${duplicate.key} has invalid declarationConflict ${entry.declarationConflict || "missing"}`);
  } else if (entry.declarationConflict !== duplicate.category) {
    failures.push(`${label} duplicate ${duplicate.key} declarationConflict must match current ${duplicate.category} classification`);
  }
  if (entry.parity !== "green") {
    failures.push(`${label} duplicate ${duplicate.key} must record parity green`);
  }
  if (duplicate.hotspot || entry.hotspot) {
    failures.push(`${label} hotspot duplicate context ${duplicate.key} is forbidden even when owned`);
  }
  if (
    duplicate.category === "conflicting" &&
    isNonEmptyString(entry.ownerReason) &&
    !/(cascade|source order|final declaration|final-order|winning|later production rule)/i.test(entry.ownerReason)
  ) {
    failures.push(`${label} conflicting duplicate ${duplicate.key} ownerReason must explain intended cascade/source-order behavior`);
  }
}

function checkDuplicateBaselineEntries(duplicates, baseline, failures, label = relativePath(BASELINE_PATH)) {
  if (!baseline || !Array.isArray(baseline.duplicateKeys)) {
    failures.push(`${label} is missing duplicateKeys`);
    return;
  }
  const knownByKey = new Map((baseline.duplicateKeys || []).map((entry) => [entry.key, entry]));
  const currentKeys = new Set(duplicates.map((duplicate) => duplicate.key));
  for (const duplicate of duplicates) {
    const entry = knownByKey.get(duplicate.key);
    if (!entry) {
      failures.push(`new unclassified duplicate selector ${duplicate.key}`);
      continue;
    }
    checkDuplicateBaselineEntry(entry, duplicate, failures, label);
  }
  for (const entry of baseline.duplicateKeys || []) {
    if (!currentKeys.has(entry.key)) {
      failures.push(`stale duplicate baseline entry ${entry.key}; refresh ${label}`);
    }
  }
}

function checkDuplicateBaseline(duplicates, failures) {
  if (WRITE_BASELINE) {
    writeFileSync(BASELINE_PATH, `${JSON.stringify(baselineFromReport(duplicates), null, 2)}\n`, "utf-8");
    return;
  }
  const baseline = loadBaseline();
  if (!baseline) {
    failures.push(`${relativePath(BASELINE_PATH)} is missing. Run node ./scripts/check-portal-css-architecture.mjs --write-baseline after reviewing duplicate ownership.`);
    return;
  }
  checkDuplicateBaselineEntries(duplicates, baseline, failures);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function escapeCssClassToken(token) {
  return token.replace(/[^A-Za-z0-9_-]/g, (character) => `\\${character}`);
}

function classTokenBase(token) {
  const parts = token.split(":");
  while (parts.length > 1 && UTILITY_VARIANTS.has(parts[0])) {
    parts.shift();
  }
  return parts.join(":");
}

function isUtilityLikeClassToken(token) {
  if (!token || UTILITY_OWNER_ALLOWLIST.has(token)) {
    return false;
  }
  if (token.startsWith(":") || token.endsWith(":") || /[$"'`;=<>?{}()]/.test(token)) {
    return false;
  }
  const base = classTokenBase(token);
  return UTILITY_EXACT_CLASSES.has(base) || UTILITY_PREFIX_PATTERN.test(base);
}

function unescapeCssClassToken(token) {
  return token
    .replace(/\\([:/.[\]()%#,])/g, "$1")
    .replace(/\\([0-9a-fA-F]{1,6})\s?/g, (_match, value) => String.fromCodePoint(parseInt(value, 16)))
    .replace(/\\/g, "");
}

function classTokensFromSelector(selector) {
  const tokens = [];
  const pattern = /\.((?:\\.|[A-Za-z0-9_-])+)/g;
  for (const match of selector.matchAll(pattern)) {
    tokens.push(unescapeCssClassToken(match[1]));
  }
  return Array.from(new Set(tokens));
}

function recordClassToken(token, sourceLabel, classTokens) {
  const normalized = token.trim();
  if (!isUtilityLikeClassToken(normalized)) {
    return;
  }
  const sources = classTokens.get(normalized) || new Set();
  sources.add(sourceLabel);
  classTokens.set(normalized, sources);
}

function recordClassTokenList(rawClassText, sourceLabel, classTokens) {
  for (const token of rawClassText.split(/\s+/)) {
    recordClassToken(token, sourceLabel, classTokens);
  }
}

function collectUtilityClassTokens() {
  const classTokens = new Map();
  const sourceFiles = [
    PORTAL_HTML_PATH,
    ...listFiles(PORTAL_SOURCE_DIR, (entryPath) => entryPath.endsWith(".js")),
    PORTAL_ASSET_JS_PATH,
    PORTAL_REVIEW_JS_PATH
  ].filter((entryPath) => existsSync(entryPath));

  for (const sourcePath of sourceFiles) {
    const sourceLabel = relativePath(sourcePath);
    const content = readText(sourcePath);
    for (const match of content.matchAll(/\bclass=(["'])(.*?)\1/gs)) {
      recordClassTokenList(match[2], sourceLabel, classTokens);
    }
    for (const match of content.matchAll(/(["'`])((?:\\.|(?!\1)[\s\S])*?)\1/g)) {
      recordClassTokenList(match[2], sourceLabel, classTokens);
    }
  }

  return classTokens;
}

function cssContainsClassToken(css, token) {
  const escapedClass = escapeCssClassToken(token);
  const pattern = new RegExp(`\\.${escapeRegExp(escapedClass)}(?=[\\s.#:{,>+~\\[]|$)`);
  return pattern.test(css);
}

function checkUtilityCoverage(failures) {
  const searchableCss = `${readText(PORTAL_CSS_ASSET_PATH)}\n${readText(PORTAL_REVIEW_CSS_ASSET_PATH)}`;
  const missing = [];
  for (const [token, sources] of collectUtilityClassTokens()) {
    if (!cssContainsClassToken(searchableCss, token)) {
      missing.push(`${token} (${Array.from(sources).sort().join(", ")})`);
    }
  }
  if (missing.length > 0) {
    failures.push(`missing utility compatibility coverage: ${missing.sort().join("; ")}`);
  }
}

function parseLayeredImport(params) {
  const match = params.match(/^url\((["']?)(.+?)\1\)|^(["'])(.+?)\3/);
  const source = match ? (match[2] || match[4] || "").trim() : "";
  const layerMatch = params.match(/\blayer\(\s*([^)]+?)\s*\)/);
  return [source, layerMatch ? layerMatch[1].trim() : ""];
}

function resolveImportPath(fromPath, source) {
  if (!source.startsWith(".")) {
    return null;
  }
  return path.resolve(path.dirname(fromPath), source);
}

function collectCssImportGraph(entryPath, failures, seen = new Set(), records = []) {
  const resolvedEntry = path.resolve(entryPath);
  if (seen.has(resolvedEntry)) {
    return records;
  }
  seen.add(resolvedEntry);

  const root = parseCss(resolvedEntry);
  root.walkAtRules("import", (atRule) => {
    const [source, layerName] = parseLayeredImport(atRule.params);
    const resolvedPath = resolveImportPath(resolvedEntry, source);
    const record = {
      fromPath: resolvedEntry,
      source,
      layerName,
      line: atRule.source?.start?.line || 0,
      resolvedPath
    };
    records.push(record);

    if (!resolvedPath) {
      return;
    }
    if (!existsSync(resolvedPath)) {
      failures.push(`${relativePath(resolvedEntry)}:${record.line} imports missing CSS source ${source}`);
      return;
    }
    collectCssImportGraph(resolvedPath, failures, seen, records);
  });

  return records;
}

function checkProductionLayerImports(failures) {
  const root = parseCss(PORTAL_CSS_INDEX_PATH);
  const imports = [];
  let declaredLayerOrder = null;
  root.each((node) => {
    if (node.type === "comment") {
      return;
    }
    if (node.type === "atrule" && node.name === "layer" && !node.nodes) {
      declaredLayerOrder = layerDeclarationNames(node.params);
    } else if (node.type === "atrule" && node.name === "import") {
      imports.push(parseLayeredImport(node.params));
    } else if (node.type !== "comment") {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} should only contain the layer declaration and layered imports`);
    }
  });
  if (JSON.stringify(declaredLayerOrder) !== JSON.stringify(EXPECTED_LAYER_ORDER)) {
    failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} declares unexpected layer order ${(declaredLayerOrder || []).join(", ")}`);
  }
  if (JSON.stringify(imports) !== JSON.stringify(EXPECTED_LAYER_IMPORTS)) {
    failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} production layer import order drifted`);
  }
  for (const [source, layerName] of imports) {
    if (layerName === "utilities" && source.startsWith("./overrides.")) {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} imports override source ${source} into layer(utilities)`);
    }
    if (layerName === "utilities" && source.startsWith("./components/")) {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} imports component source ${source} into layer(utilities)`);
    }
    if (layerName !== "utilities" && /^\.\/utilities\./.test(source)) {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} imports utility source ${source} into layer(${layerName})`);
    }
    if (
      layerName === "utilities" &&
      /(?:reset|compat)/.test(source) &&
      source !== "./utilities.compat-hold.css"
    ) {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} imports transitional compatibility source ${source} into layer(utilities)`);
    }
    if (source === "./components/workspace-performance.css" && layerName === "utilities") {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} imports workspace performance source through layer(utilities)`);
    }
    if (source === "./overrides.operator-console-reset.css") {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} still imports transitional operator console reset source`);
    }
  }
}

function describeCssNode(node) {
  if (node.type === "rule") {
    return `rule "${node.selector}"`;
  }
  if (node.type === "atrule") {
    const params = node.params ? ` ${node.params}` : "";
    return `at-rule @${node.name}${params}`;
  }
  if (node.type === "decl") {
    return `declaration "${node.prop}"`;
  }
  return `${node.type} node`;
}

function sentinelGuidance(filePath) {
  const basename = path.basename(filePath);
  if (basename === "utilities.compat-hold.css") {
    return "Move real utility rules to utilities.required.css or utilities.dynamic.css; do not add rules to the sentinel.";
  }
  if (basename === "overrides.compat.css") {
    return "Move real override rules to overrides.performance.css, overrides.accessibility.css, or an owned component stylesheet; do not add rules to the sentinel.";
  }
  return "Move real CSS to the appropriate owned source file; do not add rules to the sentinel.";
}

function sentinelCssViolations(filePath) {
  const root = parseCss(filePath);
  return (root.nodes || [])
    .filter((node) => node.type !== "comment")
    .map((node) => ({
      description: describeCssNode(node),
      line: node.source?.start?.line || 0
    }));
}

function checkSentinelOnlyFile(filePath, failures, label = relativePath(filePath)) {
  if (!existsSync(filePath)) {
    failures.push(`${label} sentinel file is missing`);
    return;
  }
  for (const violation of sentinelCssViolations(filePath)) {
    failures.push(
      `${label} must remain sentinel-only. Found ${violation.description} at line ${violation.line}. ${sentinelGuidance(filePath)}`
    );
  }
}

function checkPhase8SentinelImportGraph(failures) {
  const imports = collectCssImportGraph(PORTAL_CSS_INDEX_PATH, failures);
  const compatHoldImports = imports.filter((record) => record.resolvedPath === COMPAT_HOLD_UTILITIES_PATH);
  if (compatHoldImports.length !== 1) {
    failures.push(
      `${relativePath(PORTAL_CSS_INDEX_PATH)} import graph must import ${COMPAT_HOLD_UTILITIES_SOURCE} exactly once in layer(utilities); found ${compatHoldImports.length}`
    );
  } else if (compatHoldImports[0].layerName !== "utilities") {
    failures.push(
      `${relativePath(compatHoldImports[0].fromPath)}:${compatHoldImports[0].line} imports ${COMPAT_HOLD_UTILITIES_SOURCE} into layer(${compatHoldImports[0].layerName || "none"})`
    );
  }

  const overridesCompatImports = imports.filter((record) => record.resolvedPath === OVERRIDES_COMPAT_PATH);
  for (const record of overridesCompatImports) {
    failures.push(
      `${relativePath(record.fromPath)}:${record.line} must not import ${OVERRIDES_COMPAT_SOURCE}; overrides.compat.css is a Phase 8 unshipped sentinel`
    );
  }
}

function checkPhase8Sentinels(failures) {
  checkSentinelOnlyFile(COMPAT_HOLD_UTILITIES_PATH, failures);
  checkSentinelOnlyFile(OVERRIDES_COMPAT_PATH, failures);
  checkPhase8SentinelImportGraph(failures);
}

function layerDeclarationNames(params) {
  return params.split(",").map((entry) => entry.trim()).filter(Boolean);
}

function checkSourceCssGovernance(cssFiles, failures) {
  for (const filePath of cssFiles) {
    const root = parseCss(filePath);
    const source = relativePath(filePath);
    root.walkAtRules((atRule) => {
      if (atRule.name === "import" && filePath !== PORTAL_CSS_INDEX_PATH) {
        failures.push(`${source} must not use @import; production imports belong in index.css`);
      }
      if (atRule.name === "layer" && filePath !== PORTAL_CSS_INDEX_PATH) {
        failures.push(`${source} must not declare @layer; production layer ownership belongs in index.css`);
      }
    });
  }
}

function previousOwnerComment(node) {
  let current = node.prev();
  while (current && current.type === "comment") {
    const text = String(current.text || "");
    if (text.includes("portal-override-owner:")) {
      return text;
    }
    current = current.prev();
  }
  return "";
}

function ownerCommentForNode(node) {
  let current = node;
  while (current) {
    const owner = previousOwnerComment(current);
    if (owner) {
      return owner;
    }
    current = current.parent;
  }
  return "";
}

function fileOwnerComment(content) {
  const match = content.match(/\/\*\s*portal-override-owner:\s*([\s\S]*?)\*\//);
  return match ? match[0] : "";
}

function importantOwnerFor(decl, sourceContent) {
  const contexts = atRuleContext(decl);
  if (contexts.some((context) => context.includes("prefers-reduced-motion"))) {
    return "reduced-motion";
  }
  const ruleOwner = previousOwnerComment(decl.parent);
  if (ruleOwner) {
    return ruleOwner;
  }
  return fileOwnerComment(sourceContent);
}

function checkImportantGovernance(cssFiles, failures) {
  for (const filePath of cssFiles) {
    const content = readText(filePath);
    const root = parseCss(filePath, content);
    const source = relativePath(filePath);
    root.walkDecls((decl) => {
      if (!decl.important) {
        return;
      }
      const owner = importantOwnerFor(decl, content);
      if (
        !owner.includes("portal-override-owner:") &&
        !owner.includes("reduced-motion") &&
        !owner.includes("performance-lite") &&
        !owner.includes("forced-colors") &&
        !owner.includes("compatibility-final-order")
      ) {
        failures.push(`${source}:${decl.source?.start?.line || 0} has unowned !important declaration ${decl.prop}`);
      }
    });
  }
}

function checkOverrideOwners(cssFiles, failures) {
  for (const filePath of cssFiles.filter((entryPath) => path.basename(entryPath).startsWith("overrides."))) {
    const content = readText(filePath);
    const source = relativePath(filePath);
    if (!content.includes("portal-override-owner:")) {
      failures.push(`${source} missing portal-override-owner comment`);
    }
    for (const comment of content.matchAll(/\/\*\s*portal-override-owner:\s*compatibility-final-order[\s\S]*?\*\//g)) {
      if (!comment[0].includes("reason:")) {
        failures.push(`${source} compatibility-final-order owner comment missing reason`);
      }
      if (!comment[0].includes("removal-phase:")) {
        failures.push(`${source} compatibility-final-order owner comment missing removal-phase`);
      }
    }
    if (path.basename(filePath) === "overrides.compat.css") {
      const root = parseCss(filePath, content);
      root.walkRules((rule) => {
        const owner = ownerCommentForNode(rule);
        if (!owner.includes("portal-override-owner: compatibility-final-order")) {
          failures.push(`${source}:${rule.source?.start?.line || 0} compatibility rule missing compatibility-final-order owner`);
        }
        if (!owner.includes("reason:")) {
          failures.push(`${source}:${rule.source?.start?.line || 0} compatibility rule missing reason`);
        }
        if (!owner.includes("removal-phase:")) {
          failures.push(`${source}:${rule.source?.start?.line || 0} compatibility rule missing removal-phase`);
        }
      });
    }
  }
}

function checkSelectorFileBoundaries(cssFiles, failures) {
  for (const filePath of cssFiles) {
    const source = relativePath(filePath);
    const basename = path.basename(filePath);
    const inUtilityFile = basename.startsWith("utilities.");
    const inComponentFile = source.includes("portal-src/styles/components/");
    if (!inUtilityFile && !inComponentFile) {
      continue;
    }

    const root = parseCss(filePath);
    root.walkRules((rule) => {
      for (const selector of splitSelectorList(rule.selector)) {
        const classTokens = classTokensFromSelector(selector)
          .filter((token) => !UTILITY_OWNER_ALLOWLIST.has(token));
        if (classTokens.length === 0) {
          continue;
        }
        const utilityTokens = classTokens.filter(isUtilityLikeClassToken);
        const componentTokens = classTokens.filter((token) => !isUtilityLikeClassToken(token));

        if (inUtilityFile && componentTokens.length > 0) {
          failures.push(`${source}:${rule.source?.start?.line || 0} has component-shaped selector in utilities file ${selector}`);
        }
        if (
          inComponentFile &&
          utilityTokens.length > 0 &&
          componentTokens.length === 0 &&
          !/[#[]/.test(selector)
        ) {
          failures.push(`${source}:${rule.source?.start?.line || 0} has utility-shaped selector in component file ${selector}`);
        }
      }
    });
  }
}

function checkLayerContract(failures) {
  const sourceIndex = parseCss(PORTAL_CSS_INDEX_PATH);
  let sourceDeclaresLayers = false;
  sourceIndex.each((node) => {
    if (node.type === "atrule" && node.name === "layer" && !node.nodes) {
      sourceDeclaresLayers = true;
      const names = layerDeclarationNames(node.params);
      if (JSON.stringify(names) !== JSON.stringify(EXPECTED_LAYER_ORDER)) {
        failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} declares unexpected layer order ${names.join(", ")}`);
      }
    }
  });

  if (!sourceDeclaresLayers) {
    return;
  }

  const generated = parseCss(PORTAL_CSS_ASSET_PATH);
  generated.walkRules((rule) => {
    const contexts = atRuleContext(rule);
    if (contexts.some((context) => context.startsWith("@keyframes "))) {
      return;
    }
    if (contexts.filter((context) => context.startsWith("@layer ")).length === 0) {
      failures.push(`generated CSS has unlayered ordinary selector rule ${rule.selector}`);
    }
  });
}

function selectorSetForFile(filePath) {
  const selectors = new Set();
  const root = parseCss(filePath);
  root.walkRules((rule) => {
    for (const selector of splitSelectorList(rule.selector)) {
      selectors.add(selector);
    }
  });
  return selectors;
}

function ruleCountForFile(filePath) {
  let count = 0;
  parseCss(filePath).walkRules(() => {
    count += 1;
  });
  return count;
}

function selectorContainsIdToken(selector) {
  // Strip attribute selector contents (`[href^="#"]`) and quoted string
  // bodies. PostCSS already strips comments before surfacing rule.selector,
  // so once those two sources of `#` are removed every remaining `#` in
  // the selector starts an ID token (CSS selector grammar allows ID tokens
  // adjacent to type/class selectors with no separator, e.g. `a#anchor`).
  let stripped = selector;
  let previous;
  do {
    previous = stripped;
    stripped = stripped.replace(/\[[^\[\]]*\]/g, "");
  } while (stripped !== previous);
  stripped = stripped.replace(/"[^"]*"/g, "").replace(/'[^']*'/g, "");
  return /#(?:\\.|[A-Za-z_-])/.test(stripped);
}

function checkOverridesCompatNoNewIds(failures) {
  const compatPath = path.resolve(PORTAL_CSS_SOURCE_DIR, "overrides.compat.css");
  if (!existsSync(compatPath)) {
    return;
  }
  const source = relativePath(compatPath);
  const root = parseCss(compatPath);
  root.walkRules((rule) => {
    for (const selector of splitSelectorList(rule.selector)) {
      if (selectorContainsIdToken(selector)) {
        failures.push(
          `${source}:${rule.source?.start?.line || 0} ID-specific selector ${selector} is forbidden in overrides.compat.css after Phase 6 semantic-ownership migration`
        );
      }
    }
  });
}

function checkOverridesCompatNoImportant(failures) {
  const compatPath = path.resolve(PORTAL_CSS_SOURCE_DIR, "overrides.compat.css");
  if (!existsSync(compatPath)) {
    return;
  }
  const source = relativePath(compatPath);
  const root = parseCss(compatPath);
  root.walkDecls((decl) => {
    if (decl.important) {
      failures.push(
        `${source}:${decl.source?.start?.line || 0} !important on ${decl.prop} is forbidden in overrides.compat.css; raise the issue in the Phase 7 markup migration instead`
      );
    }
  });
}

function checkPhase6SemanticHooksPresent(failures) {
  if (!existsSync(PORTAL_HTML_PATH)) {
    failures.push(`portal.html missing at ${relativePath(PORTAL_HTML_PATH)}; cannot validate Phase 6 semantic hooks`);
    return;
  }
  const compatPath = path.resolve(PORTAL_CSS_SOURCE_DIR, "overrides.compat.css");
  if (!existsSync(compatPath)) {
    return;
  }
  const compatRoot = parseCss(compatPath);
  const semanticTokens = new Set();
  compatRoot.walkRules((rule) => {
    for (const selector of splitSelectorList(rule.selector)) {
      for (const token of classTokensFromSelector(selector)) {
        if (token.startsWith("portal-") || token.startsWith("artifact-")) {
          semanticTokens.add(token);
        }
      }
    }
  });
  if (semanticTokens.size === 0) {
    return;
  }
  const html = readText(PORTAL_HTML_PATH);
  for (const token of semanticTokens) {
    const wordBoundary = new RegExp(`(^|[\\s"'])${escapeRegExp(token)}([\\s"']|$)`, "m");
    if (!wordBoundary.test(html)) {
      failures.push(
        `Phase 6 semantic hook .${token} used in overrides.compat.css does not match any class on portal.html; fix the typo or restore the markup hook`
      );
    }
  }
}

function sha256File(filePath) {
  return crypto.createHash("sha256").update(readText(filePath)).digest("hex");
}

function phase9EntryIsOwned(entry, duplicate) {
  if (!entry) {
    return false;
  }
  const failures = [];
  checkDuplicateBaselineEntry(entry, duplicate, failures, relativePath(BASELINE_PATH));
  return failures.length === 0;
}

function countDuplicatesByCategory(duplicates, category) {
  return duplicates.filter((duplicate) => duplicate.category === category).length;
}

function fingerprintBytes(payload) {
  return crypto.createHash("sha256").update(payload).digest("hex").slice(0, 12);
}

function portalAssetVersionedUrl(assetName) {
  const assetPath = path.resolve(PORTAL_ASSETS_DIR, assetName);
  const encodedPath = assetName.split("/").map((part) => encodeURIComponent(part)).join("/");
  return `/portal/assets/${encodedPath}?v=${fingerprintBytes(readFileSync(assetPath))}`;
}

function renderedPortalCssText() {
  let rendered = readText(PORTAL_CSS_ASSET_PATH);
  for (const [token, assetName] of Object.entries(PORTAL_CSS_RENDER_TOKENS)) {
    rendered = rendered.replaceAll(token, portalAssetVersionedUrl(assetName));
  }
  return rendered;
}

function renderedPortalCssFingerprint() {
  return fingerprintBytes(Buffer.from(renderedPortalCssText(), "utf8"));
}

function gzipByteLength(text) {
  return zlib.gzipSync(text, { level: 9 }).length;
}

function declarationRecords(record) {
  if (Array.isArray(record.declarations)) {
    return record.declarations.map(([property, value = "", important = false]) => [
      String(property || "").trim(),
      String(value || "").trim(),
      Boolean(important)
    ]);
  }
  return (record.properties || []).map((property) => [String(property || "").trim(), "", false]);
}

function propertyFamily(property) {
  const prop = String(property || "").trim().toLowerCase();
  if (!prop) {
    return "";
  }
  if (prop.startsWith("--")) {
    return prop;
  }
  if (prop === "all") {
    return "all";
  }
  if (prop === "font" || prop.startsWith("font-") || prop === "line-height") {
    return "font";
  }
  if (prop === "background" || prop.startsWith("background-")) {
    return "background";
  }
  if (prop === "border" || /^border-(?:top|right|bottom|left)$/.test(prop)) {
    return "border-line";
  }
  if (/^border-(?:top|right|bottom|left)-/.test(prop)) {
    return prop.replace(/^border-(top|right|bottom|left)-(width|style|color)$/, "border-$1-line");
  }
  if (prop === "border-color" || prop === "border-width" || prop === "border-style") {
    return "border-line";
  }
  if (prop === "border-radius" || prop.startsWith("border-") && prop.endsWith("-radius")) {
    return "border-radius";
  }
  for (const family of ["margin", "padding", "inset", "transition", "animation", "transform", "box-shadow"]) {
    if (prop === family || prop.startsWith(`${family}-`)) {
      return family;
    }
  }
  return prop;
}

function declarationOverlapReason(records) {
  const properties = new Map();
  const families = new Map();
  for (const [recordIndex, record] of records.entries()) {
    for (const [property] of declarationRecords(record)) {
      const normalizedProperty = property.toLowerCase();
      if (!normalizedProperty) {
        continue;
      }
      const propertyOwners = properties.get(normalizedProperty) || new Set();
      propertyOwners.add(recordIndex);
      if (propertyOwners.size > 1) {
        return "shorthand-longhand-overlap";
      }
      properties.set(normalizedProperty, propertyOwners);

      const family = propertyFamily(normalizedProperty);
      if (!family) {
        continue;
      }
      const familyProperties = families.get(family) || new Map();
      const familyOwners = familyProperties.get(normalizedProperty) || new Set();
      familyOwners.add(recordIndex);
      familyProperties.set(normalizedProperty, familyOwners);
      families.set(family, familyProperties);
    }
  }

  for (const familyProperties of families.values()) {
    const familyOwners = new Set();
    for (const owners of familyProperties.values()) {
      for (const owner of owners) {
        familyOwners.add(owner);
      }
    }
    if (familyOwners.size > 1 && familyProperties.size > 1) {
      return "shorthand-longhand-overlap";
    }
  }
  return "";
}

function customPropertyDependencyReason(records) {
  const assigned = new Set();
  const values = [];
  for (const record of records) {
    for (const [property, value] of declarationRecords(record)) {
      if (property.startsWith("--")) {
        assigned.add(property);
      }
      values.push(String(value || ""));
    }
  }
  for (const token of assigned) {
    const tokenPattern = new RegExp(`var\\(\\s*${escapeRegExp(token)}(?:\\s*[,\\)])`);
    if (values.some((value) => tokenPattern.test(value))) {
      return "custom-property-order-sensitive";
    }
  }
  return "";
}

function baselineEntryForPhase10(duplicate, baselineEntries) {
  return baselineEntries.get(duplicate.key) || duplicate.baselineEntry || duplicate;
}

function selectorListForRecord(record) {
  if (Array.isArray(record.selectorList) && record.selectorList.length > 0) {
    return record.selectorList;
  }
  if (record.ruleSelector) {
    return splitSelectorList(record.ruleSelector);
  }
  return [record.selector].filter(Boolean);
}

function analyzePhase10AdditiveCandidate(duplicate, baselineEntry) {
  const entry = baselineEntry || {};
  const candidate = {
    key: duplicate.key,
    selector: duplicate.selector,
    layer: duplicate.layer,
    atRuleContext: duplicate.context || [],
    declarationConflict: duplicate.category,
    removalStatus: entry.removalStatus || duplicate.removalStatus || null,
    candidateStatus: "deferred"
  };

  if (duplicate.hotspot || entry.hotspot) {
    return { ...candidate, unsafeReason: "hotspot" };
  }
  if (duplicate.category !== "additive") {
    return { ...candidate, unsafeReason: "conflicting" };
  }
  if ((entry.removalStatus || duplicate.removalStatus) !== "removable-later") {
    return { ...candidate, unsafeReason: "coverage-ambiguous" };
  }

  const records = duplicate.records || [];
  if (records.length !== 2) {
    return { ...candidate, unsafeReason: "coverage-ambiguous" };
  }
  if (new Set(records.map((record) => record.layer)).size > 1) {
    return { ...candidate, unsafeReason: "cross-layer" };
  }
  if (new Set(records.map((record) => JSON.stringify(record.context || duplicate.context || []))).size > 1) {
    return { ...candidate, unsafeReason: "cross-context" };
  }

  const customDependencyReason = customPropertyDependencyReason(records);
  if (customDependencyReason) {
    return { ...candidate, unsafeReason: customDependencyReason };
  }
  const overlapReason = declarationOverlapReason(records);
  if (overlapReason) {
    return { ...candidate, unsafeReason: overlapReason };
  }

  const [left, right] = records;
  if (!left.source || left.source !== right.source) {
    return { ...candidate, unsafeReason: "source-order-sensitive" };
  }

  const listRecords = records.map((record) => ({ record, selectorList: selectorListForRecord(record) }));
  const singleton = listRecords.find((item) => item.selectorList.length === 1 && item.selectorList[0] === duplicate.selector);
  const selectorList = listRecords.find((item) => item.selectorList.length > 1 && item.selectorList.includes(duplicate.selector));
  if (!singleton || !selectorList) {
    return { ...candidate, unsafeReason: "selector-list-ambiguous" };
  }
  if (selectorList.selectorList.filter((selector) => selector !== duplicate.selector).length === 0) {
    return { ...candidate, unsafeReason: "coverage-ambiguous" };
  }
  if ((singleton.record.line || 0) >= (selectorList.record.line || 0)) {
    return { ...candidate, unsafeReason: "source-order-sensitive" };
  }
  if ((selectorList.record.line || 0) - (singleton.record.line || 0) > 8) {
    return { ...candidate, unsafeReason: "intervening-overlap" };
  }

  return {
    ...candidate,
    candidateStatus: "safe",
    targetLocation: {
      file: singleton.record.source,
      line: singleton.record.line,
      column: singleton.record.column
    },
    sourceLocation: {
      file: selectorList.record.source,
      line: selectorList.record.line,
      column: selectorList.record.column
    },
    safetyChecks: {
      sameLayer: true,
      sameAtRuleStack: true,
      noPropertyOverlap: true,
      noShorthandOverlap: true,
      noInterveningOverlap: true,
      selectorListCoveragePreserved: true
    }
  };
}

function analyzePhase10AdditiveCandidates(duplicates, baseline = loadBaseline()) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  return duplicates
    .filter((duplicate) => {
      const entry = baselineEntryForPhase10(duplicate, baselineEntries);
      return duplicate.category === "additive" && (entry.removalStatus || duplicate.removalStatus) === "removable-later";
    })
    .map((duplicate) => analyzePhase10AdditiveCandidate(duplicate, baselineEntryForPhase10(duplicate, baselineEntries)))
    .sort((left, right) => left.key.localeCompare(right.key));
}

function hasSpecificityChangingGrouping(record) {
  return /:(?:is|where)\s*\(/.test(String(record.ruleSelector || record.selector || ""));
}

function hasBackgroundFamilyProperty(record) {
  return declarationRecords(record).some(([property]) => propertyFamily(property) === "background");
}

function phase11BackgroundHazard(records) {
  if (records.some((record) => Boolean(record.backgroundOverlap || record.interveningBackgroundWrite))) {
    return true;
  }
  const backgroundRecords = records.filter(hasBackgroundFamilyProperty);
  if (backgroundRecords.length < 2) {
    return false;
  }
  const backgroundProperties = new Set();
  for (const record of backgroundRecords) {
    for (const [property] of declarationRecords(record)) {
      if (propertyFamily(property) === "background") {
        backgroundProperties.add(property.toLowerCase());
      }
    }
  }
  return backgroundProperties.has("background") && backgroundProperties.size > 1;
}

function phase11SourceLocations(duplicate) {
  return (duplicate.records || []).map((record) => ({
    file: record.source,
    line: record.line,
    column: record.column
  }));
}

function analyzePhase11SurfaceCandidate(duplicate, baselineEntry = {}) {
  const entry = baselineEntry || {};
  const records = duplicate.records || [];
  const candidate = {
    key: duplicate.key,
    selector: duplicate.selector,
    layer: duplicate.layer,
    atRuleContext: duplicate.context || [],
    declarationConflict: duplicate.category,
    removalStatus: entry.removalStatus || duplicate.removalStatus || null,
    targetFile: PHASE11_TARGET_FILE,
    sourceLocations: phase11SourceLocations(duplicate),
    candidateStatus: "deferred"
  };

  if (duplicate.hotspot || entry.hotspot) {
    return { ...candidate, unsafeReason: "hotspot" };
  }
  if (duplicate.category !== "additive" || (entry.removalStatus || duplicate.removalStatus) !== "removable-later") {
    return { ...candidate, unsafeReason: "conflicting-permanent" };
  }
  if (!PHASE11_TARGET_DUPLICATE_KEYS.has(duplicate.key)) {
    return { ...candidate, unsafeReason: "selector-not-phase11-target" };
  }
  if (records.some(hasSpecificityChangingGrouping)) {
    return { ...candidate, unsafeReason: "specificity-changing-grouping" };
  }
  if (records.some((record) => record.missingDarkPair) || duplicate.missingDarkPair) {
    return { ...candidate, unsafeReason: "dark-pair-missing" };
  }
  if (new Set(records.map((record) => record.layer || duplicate.layer)).size > 1) {
    return { ...candidate, unsafeReason: "cross-layer" };
  }
  if (new Set(records.map((record) => JSON.stringify(record.context || duplicate.context || []))).size > 1) {
    return { ...candidate, unsafeReason: "cross-context" };
  }
  if (!records.every((record) => record.source === PHASE11_TARGET_FILE)) {
    return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
  }
  if (phase11BackgroundHazard(records)) {
    return { ...candidate, unsafeReason: "background-shorthand-order-sensitive" };
  }
  const overlapReason = declarationOverlapReason(records);
  if (overlapReason) {
    return { ...candidate, unsafeReason: overlapReason };
  }
  if (records.some((record) => record.interveningOverlap)) {
    return { ...candidate, unsafeReason: "intervening-overlap" };
  }
  if (records.some((record) => record.sourceOrderSensitive)) {
    return { ...candidate, unsafeReason: "source-order-sensitive" };
  }

  return {
    ...candidate,
    candidateStatus: "safe",
    approvedPhase11Target: true,
    safetyChecks: {
      sameLayer: true,
      sameAtRuleStack: true,
      sameThemeContext: true,
      noHotspot: true,
      noConflictingPermanentEntry: true,
      selectorListCoveragePreserved: true,
      noSpecificityChangingGrouping: true,
      noBackgroundFamilySourceOrderHazard: true,
      noInterveningOverlap: true
    }
  };
}

function analyzePhase11SurfaceCandidates(duplicates, baseline = loadBaseline()) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  return duplicates
    .map((duplicate) => analyzePhase11SurfaceCandidate(duplicate, baselineEntries.get(duplicate.key) || duplicate.baselineEntry || duplicate))
    .sort((left, right) => left.key.localeCompare(right.key));
}

function recordProperties(record) {
  if (Array.isArray(record.properties) && record.properties.length > 0) {
    return record.properties.map((property) => String(property).toLowerCase()).sort();
  }
  return declarationRecords(record).map(([property]) => property.toLowerCase()).sort();
}

function isPhase12AmbientTarget(duplicate) {
  return duplicate.key === ".ambient-orb|||components|||" || duplicate.key === ".ambient-ring|||components|||";
}

function isPhase12OverviewTarget(duplicate) {
  return duplicate.key === ".overview-actions|||components|||@media (max-width: 767px)";
}

function analyzePhase12ComponentCandidate(duplicate, baselineEntry = {}) {
  const entry = baselineEntry || {};
  const records = duplicate.records || [];
  const candidate = {
    key: duplicate.key,
    selector: duplicate.selector,
    layer: duplicate.layer,
    atRuleContext: duplicate.context || [],
    declarationConflict: duplicate.category,
    removalStatus: entry.removalStatus || duplicate.removalStatus || null,
    sourceLocations: phase11SourceLocations(duplicate),
    candidateStatus: "deferred"
  };

  if (duplicate.hotspot || entry.hotspot) {
    return { ...candidate, unsafeReason: "hotspot" };
  }
  if (duplicate.category !== "additive" || (entry.removalStatus || duplicate.removalStatus) !== "removable-later") {
    return { ...candidate, unsafeReason: "conflicting-permanent" };
  }
  if (!PHASE12_TARGET_DUPLICATE_KEYS.has(duplicate.key)) {
    return { ...candidate, unsafeReason: "selector-not-phase12-target" };
  }
  if (records.some(hasSpecificityChangingGrouping)) {
    return { ...candidate, unsafeReason: "specificity-changing-grouping" };
  }
  if (new Set(records.map((record) => record.layer || duplicate.layer)).size > 1) {
    return { ...candidate, unsafeReason: "cross-layer" };
  }
  if (new Set(records.map((record) => JSON.stringify(record.context || duplicate.context || []))).size > 1) {
    return { ...candidate, unsafeReason: "cross-context" };
  }
  const customDependencyReason = customPropertyDependencyReason(records);
  if (customDependencyReason) {
    return { ...candidate, unsafeReason: customDependencyReason };
  }
  const overlapReason = declarationOverlapReason(records);
  if (overlapReason) {
    return { ...candidate, unsafeReason: overlapReason };
  }
  if (records.some((record) => record.interveningOverlap)) {
    return { ...candidate, unsafeReason: "intervening-overlap" };
  }
  if (records.some((record) => record.sourceOrderSensitive)) {
    return { ...candidate, unsafeReason: "source-order-sensitive" };
  }

  if (isPhase12AmbientTarget(duplicate)) {
    if (!records.every((record) => record.source === PHASE12_SHELL_FOUNDATION_FILE)) {
      return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
    }
    const shared = records.find((record) => selectorListForRecord(record).length > 1);
    const singleton = records.find((record) => {
      const selectorList = selectorListForRecord(record);
      return selectorList.length === 1 && selectorList[0] === duplicate.selector;
    });
    if (!shared || !singleton || !selectorListForRecord(shared).includes(duplicate.selector)) {
      return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
    }
    if (JSON.stringify(recordProperties(shared)) !== JSON.stringify(["inset", "position"])) {
      return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
    }
    if ((shared.line || 0) >= (singleton.line || 0)) {
      return { ...candidate, unsafeReason: "source-order-sensitive" };
    }
    return {
      ...candidate,
      candidateStatus: "safe",
      approvedPhase12Target: true,
      targetLocation: {
        file: singleton.source,
        line: singleton.line,
        column: singleton.column
      },
      sourceLocation: {
        file: shared.source,
        line: shared.line,
        column: shared.column
      },
      safetyChecks: {
        sameLayer: true,
        sameAtRuleStack: true,
        selectorListCoveragePreserved: true,
        noPropertyOverlap: true,
        noShorthandOverlap: true,
        noInterveningOverlap: true,
        singletonBeforeModifiers: true
      }
    };
  }

  if (isPhase12OverviewTarget(duplicate)) {
    if (
      !records.every(
        (record) =>
          record.source === PHASE12_OPERATOR_CONSOLE_FILE &&
          JSON.stringify(record.context || duplicate.context || []) === JSON.stringify(["@media (max-width: 767px)"])
      )
    ) {
      return { ...candidate, unsafeReason: "cross-context" };
    }
    const shared = records.find((record) => selectorListForRecord(record).includes(".build-stepper-actions-inline"));
    const singleton = records.find((record) => {
      const selectorList = selectorListForRecord(record);
      return selectorList.length === 1 && selectorList[0] === duplicate.selector;
    });
    if (!shared || !singleton || !selectorListForRecord(shared).includes(duplicate.selector)) {
      return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
    }
    if (JSON.stringify(recordProperties(shared)) !== JSON.stringify(["justify-content"])) {
      return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
    }
    if ((shared.line || 0) >= (singleton.line || 0)) {
      return { ...candidate, unsafeReason: "source-order-sensitive" };
    }
    if ((singleton.line || 0) - (shared.line || 0) > 8) {
      return { ...candidate, unsafeReason: "intervening-overlap" };
    }
    return {
      ...candidate,
      candidateStatus: "safe",
      approvedPhase12Target: true,
      targetLocation: {
        file: singleton.source,
        line: singleton.line,
        column: singleton.column
      },
      sourceLocation: {
        file: shared.source,
        line: shared.line,
        column: shared.column
      },
      safetyChecks: {
        sameLayer: true,
        sameAtRuleStack: true,
        selectorListCoveragePreserved: true,
        buildStepperCoveragePreserved: true,
        noPropertyOverlap: true,
        noShorthandOverlap: true,
        noInterveningOverlap: true
      }
    };
  }

  return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
}

function analyzePhase12ComponentCandidates(duplicates, baseline = loadBaseline()) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  return duplicates
    .map((duplicate) => analyzePhase12ComponentCandidate(duplicate, baselineEntries.get(duplicate.key) || duplicate.baselineEntry || duplicate))
    .sort((left, right) => left.key.localeCompare(right.key));
}

function phase13FocusVisibleSelector(selector) {
  return String(selector || "").replace(/:hover$/, ":focus-visible");
}

function phase13RecordHasHoverOutline(record) {
  return declarationRecords(record).some(
    ([property, value]) => property.toLowerCase() === "outline" && String(value).trim().toLowerCase() === "none"
  );
}

function phase13RecordHasOnlyOutline(record) {
  return JSON.stringify(recordProperties(record)) === JSON.stringify(["outline"]);
}

function analyzePhase13InteractionCandidate(duplicate, baselineEntry = {}) {
  const entry = baselineEntry || {};
  const records = duplicate.records || [];
  const candidate = {
    key: duplicate.key,
    selector: duplicate.selector,
    layer: duplicate.layer,
    atRuleContext: duplicate.context || [],
    declarationConflict: duplicate.category,
    removalStatus: entry.removalStatus || duplicate.removalStatus || null,
    sourceLocations: phase11SourceLocations(duplicate),
    candidateStatus: "deferred"
  };

  if (duplicate.hotspot || entry.hotspot) {
    return { ...candidate, unsafeReason: "hotspot" };
  }
  if (duplicate.category !== "additive" || (entry.removalStatus || duplicate.removalStatus) !== "removable-later") {
    return { ...candidate, unsafeReason: "conflicting-permanent" };
  }
  if (!PHASE13_TARGET_DUPLICATE_KEYS.has(duplicate.key)) {
    return { ...candidate, unsafeReason: "selector-not-phase13-target" };
  }
  if (duplicate.layer !== "components" || records.some((record) => (record.layer || duplicate.layer) !== "components")) {
    return { ...candidate, unsafeReason: "cross-layer" };
  }
  if (records.some(hasSpecificityChangingGrouping)) {
    return { ...candidate, unsafeReason: "specificity-changing-grouping" };
  }
  if (new Set(records.map((record) => JSON.stringify(record.context || duplicate.context || []))).size > 1) {
    return { ...candidate, unsafeReason: "cross-context" };
  }
  if (JSON.stringify(duplicate.context || []) !== JSON.stringify([])) {
    return { ...candidate, unsafeReason: "cross-context" };
  }
  if (records.length !== 2) {
    return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
  }

  const expectedFile = PHASE13_TARGET_FILES_BY_KEY.get(duplicate.key);
  if (!expectedFile || !records.every((record) => record.source === expectedFile)) {
    return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
  }

  const focusVisibleSelector = phase13FocusVisibleSelector(duplicate.selector);
  const shared = records.find((record) => {
    const selectorList = selectorListForRecord(record);
    return selectorList.includes(duplicate.selector) && selectorList.includes(focusVisibleSelector);
  });
  const singleton = records.find((record) => {
    const selectorList = selectorListForRecord(record);
    return selectorList.length === 1 && selectorList[0] === duplicate.selector;
  });
  if (!shared) {
    return { ...candidate, unsafeReason: "focus-visible-coverage-missing" };
  }
  if (!singleton || !phase13RecordHasOnlyOutline(singleton) || !phase13RecordHasHoverOutline(singleton)) {
    return { ...candidate, unsafeReason: "missing-hover-outline" };
  }
  if (recordProperties(shared).includes("outline")) {
    return { ...candidate, unsafeReason: "missing-hover-outline" };
  }
  if ((shared.line || 0) >= (singleton.line || 0)) {
    return { ...candidate, unsafeReason: "source-order-sensitive" };
  }
  if ((singleton.line || 0) - (shared.line || 0) > 8) {
    return { ...candidate, unsafeReason: "intervening-overlap" };
  }

  return {
    ...candidate,
    candidateStatus: "safe",
    approvedPhase13Target: true,
    targetLocation: {
      file: shared.source,
      line: shared.line,
      column: shared.column
    },
    sourceLocation: {
      file: singleton.source,
      line: singleton.line,
      column: singleton.column
    },
    safetyChecks: {
      sameFile: true,
      sameLayer: true,
      emptyAtRuleContext: true,
      selectorListCoveragePreserved: true,
      hoverOutlinePreserved: true,
      focusVisibleCoveragePreserved: true,
      noSpecificityChangingGrouping: true,
      noInterveningOverlap: true
    }
  };
}

function analyzePhase13InteractionCandidates(duplicates, baseline = loadBaseline()) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  return duplicates
    .map((duplicate) =>
      analyzePhase13InteractionCandidate(duplicate, baselineEntries.get(duplicate.key) || duplicate.baselineEntry || duplicate)
    )
    .sort((left, right) => left.key.localeCompare(right.key));
}

function selectorListEquals(left, right) {
  return JSON.stringify([...left].sort()) === JSON.stringify([...right].sort());
}

function phase14HasExpectedSingletonDeclarations(record, selector) {
  const expected = PHASE14_SINGLETON_PROPERTIES_BY_SELECTOR.get(selector) || [];
  return JSON.stringify(recordProperties(record)) === JSON.stringify(expected);
}

function analyzePhase14SkeletonCandidate(duplicate, baselineEntry = {}) {
  const entry = baselineEntry || {};
  const records = duplicate.records || [];
  const candidate = {
    key: duplicate.key,
    selector: duplicate.selector,
    layer: duplicate.layer,
    atRuleContext: duplicate.context || [],
    declarationConflict: duplicate.category,
    removalStatus: entry.removalStatus || duplicate.removalStatus || null,
    sourceLocations: phase11SourceLocations(duplicate),
    candidateStatus: "deferred"
  };

  if (duplicate.hotspot || entry.hotspot) {
    return { ...candidate, unsafeReason: "hotspot" };
  }
  if (duplicate.category !== "additive" || (entry.removalStatus || duplicate.removalStatus) !== "removable-later") {
    return { ...candidate, unsafeReason: "conflicting-permanent" };
  }
  if (!PHASE14_TARGET_DUPLICATE_KEYS.has(duplicate.key)) {
    return { ...candidate, unsafeReason: "selector-not-phase14-target" };
  }
  if (duplicate.layer !== "components" || records.some((record) => (record.layer || duplicate.layer) !== "components")) {
    return { ...candidate, unsafeReason: "cross-layer" };
  }
  if (records.some(hasSpecificityChangingGrouping)) {
    return { ...candidate, unsafeReason: "specificity-changing-grouping" };
  }
  if (new Set(records.map((record) => JSON.stringify(record.context || duplicate.context || []))).size > 1) {
    return { ...candidate, unsafeReason: "cross-context" };
  }
  if (JSON.stringify(duplicate.context || []) !== JSON.stringify([])) {
    return { ...candidate, unsafeReason: "cross-context" };
  }
  if (records.length !== 2) {
    return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
  }
  if (!records.every((record) => record.source === PHASE14_TARGET_FILE)) {
    return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
  }

  const shared = records.find((record) => selectorListEquals(selectorListForRecord(record), PHASE14_TARGET_SELECTORS));
  const singleton = records.find((record) => {
    const selectorList = selectorListForRecord(record);
    return selectorList.length === 1 && selectorList[0] === duplicate.selector;
  });
  if (!shared) {
    return { ...candidate, unsafeReason: "shared-selector-list-drift" };
  }
  if (JSON.stringify(recordProperties(shared)) !== JSON.stringify(PHASE14_SHARED_PROPERTIES)) {
    return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
  }
  if (!singleton || !phase14HasExpectedSingletonDeclarations(singleton, duplicate.selector)) {
    return { ...candidate, unsafeReason: "singleton-declaration-missing" };
  }
  if ((shared.line || 0) >= (singleton.line || 0)) {
    return { ...candidate, unsafeReason: "variant-order-sensitive" };
  }

  return {
    ...candidate,
    candidateStatus: "safe",
    approvedPhase14Target: true,
    targetLocation: {
      file: singleton.source,
      line: singleton.line,
      column: singleton.column
    },
    sourceLocation: {
      file: shared.source,
      line: shared.line,
      column: shared.column
    },
    safetyChecks: {
      sameFile: true,
      sameLayer: true,
      emptyAtRuleContext: true,
      exactSharedSelectorList: true,
      singletonDeclarationsPreserved: true,
      darkBackgroundOverridePreserved: true,
      reducedMotionCoveragePreserved: true,
      noSpecificityChangingGrouping: true
    }
  };
}

function analyzePhase14SkeletonCandidates(duplicates, baseline = loadBaseline()) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  return duplicates
    .map((duplicate) =>
      analyzePhase14SkeletonCandidate(duplicate, baselineEntries.get(duplicate.key) || duplicate.baselineEntry || duplicate)
    )
    .sort((left, right) => left.key.localeCompare(right.key));
}

function phase15HasExpectedProperties(record, expected) {
  return JSON.stringify(recordProperties(record)) === JSON.stringify(expected || []);
}

function analyzePhase15SurfaceLoadingCandidate(duplicate, baselineEntry = {}) {
  const entry = baselineEntry || {};
  const records = duplicate.records || [];
  const candidate = {
    key: duplicate.key,
    selector: duplicate.selector,
    layer: duplicate.layer,
    atRuleContext: duplicate.context || [],
    declarationConflict: duplicate.category,
    removalStatus: entry.removalStatus || duplicate.removalStatus || null,
    sourceLocations: phase11SourceLocations(duplicate),
    candidateStatus: "deferred"
  };

  if (duplicate.hotspot || entry.hotspot) {
    return { ...candidate, unsafeReason: "hotspot" };
  }
  if (duplicate.category !== "additive" || (entry.removalStatus || duplicate.removalStatus) !== "removable-later") {
    return { ...candidate, unsafeReason: "conflicting-permanent" };
  }
  if (!PHASE15_TARGET_DUPLICATE_KEYS.has(duplicate.key)) {
    return { ...candidate, unsafeReason: "selector-not-phase15-target" };
  }
  if (duplicate.layer !== "components" || records.some((record) => (record.layer || duplicate.layer) !== "components")) {
    return { ...candidate, unsafeReason: "cross-layer" };
  }
  if (records.some(hasSpecificityChangingGrouping)) {
    return { ...candidate, unsafeReason: "specificity-changing-grouping" };
  }
  if (new Set(records.map((record) => JSON.stringify(record.context || duplicate.context || []))).size > 1) {
    return { ...candidate, unsafeReason: "cross-context" };
  }
  if (JSON.stringify(duplicate.context || []) !== JSON.stringify([])) {
    return { ...candidate, unsafeReason: "cross-context" };
  }
  if (records.length !== 2) {
    return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
  }

  const dispatchRecord = records.find((record) => record.source === PHASE15_TARGET_FILE);
  const operatorRecord = records.find((record) => record.source === PHASE15_OPERATOR_CONSOLE_FILE);
  if (!dispatchRecord || !operatorRecord) {
    return { ...candidate, unsafeReason: "source-file-drift" };
  }
  if (!phase15HasExpectedProperties(dispatchRecord, PHASE15_DISPATCH_PROPERTIES_BY_SELECTOR.get(duplicate.selector))) {
    return { ...candidate, unsafeReason: "dispatch-declaration-missing" };
  }
  if (!phase15HasExpectedProperties(operatorRecord, PHASE15_OPERATOR_PROPERTIES_BY_SELECTOR.get(duplicate.selector))) {
    return { ...candidate, unsafeReason: "operator-declaration-drift" };
  }
  if ((dispatchRecord.line || 0) >= (operatorRecord.line || 0)) {
    return { ...candidate, unsafeReason: "source-order-sensitive" };
  }

  return {
    ...candidate,
    candidateStatus: "safe",
    approvedPhase15Target: true,
    targetLocation: {
      file: dispatchRecord.source,
      line: dispatchRecord.line,
      column: dispatchRecord.column
    },
    sourceLocation: {
      file: operatorRecord.source,
      line: operatorRecord.line,
      column: operatorRecord.column
    },
    safetyChecks: {
      sameLayer: true,
      emptyAtRuleContext: true,
      targetFilePreserved: true,
      operatorDeclarationsRemoved: true,
      reducedMotionCoveragePreserved: true,
      noSpecificityChangingGrouping: true
    }
  };
}

function analyzePhase15SurfaceLoadingCandidates(duplicates, baseline = loadBaseline()) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  return duplicates
    .map((duplicate) =>
      analyzePhase15SurfaceLoadingCandidate(duplicate, baselineEntries.get(duplicate.key) || duplicate.baselineEntry || duplicate)
    )
    .sort((left, right) => left.key.localeCompare(right.key));
}

function analyzePhase16ReviewSurfaceCandidate(duplicate, baselineEntry = {}) {
  const entry = baselineEntry || {};
  const records = duplicate.records || [];
  const candidate = {
    key: duplicate.key,
    selector: duplicate.selector,
    layer: duplicate.layer,
    atRuleContext: duplicate.context || [],
    declarationConflict: duplicate.category,
    removalStatus: entry.removalStatus || duplicate.removalStatus || null,
    sourceLocations: phase11SourceLocations(duplicate),
    candidateStatus: "deferred"
  };

  if (duplicate.hotspot || entry.hotspot) {
    return { ...candidate, unsafeReason: "hotspot" };
  }
  if (duplicate.category !== "additive" || (entry.removalStatus || duplicate.removalStatus) !== "removable-later") {
    return { ...candidate, unsafeReason: "conflicting-permanent" };
  }
  if (!PHASE16_TARGET_DUPLICATE_KEYS.has(duplicate.key)) {
    return { ...candidate, unsafeReason: "selector-not-phase16-target" };
  }
  if (duplicate.layer !== "components" || records.some((record) => (record.layer || duplicate.layer) !== "components")) {
    return { ...candidate, unsafeReason: "cross-layer" };
  }
  if (records.some(hasSpecificityChangingGrouping)) {
    return { ...candidate, unsafeReason: "specificity-changing-grouping" };
  }
  if (new Set(records.map((record) => JSON.stringify(record.context || duplicate.context || []))).size > 1) {
    return { ...candidate, unsafeReason: "cross-context" };
  }
  if (JSON.stringify(duplicate.context || []) !== JSON.stringify([])) {
    return { ...candidate, unsafeReason: "cross-context" };
  }
  if (records.length !== 2) {
    return { ...candidate, unsafeReason: "selector-list-coverage-ambiguous" };
  }

  const surfaceRecords = records.filter((record) => record.source === PHASE16_SURFACE_NORMALIZATION_FILE);
  const operatorRecord = records.find((record) => record.source === PHASE16_OPERATOR_CONSOLE_FILE);
  const surfacePrimary = surfaceRecords.find((record) =>
    phase15HasExpectedProperties(record, PHASE16_SURFACE_PROPERTIES_BY_SELECTOR.get(duplicate.selector))
  );
  const secondary = records.find((record) =>
    phase15HasExpectedProperties(record, PHASE16_SECONDARY_PROPERTIES_BY_SELECTOR.get(duplicate.selector))
  );
  if (!surfacePrimary) {
    return { ...candidate, unsafeReason: "surface-declaration-missing" };
  }
  if (
    (duplicate.selector === ".review-status-banner" || duplicate.selector === ".dark .review-status-banner") &&
    !operatorRecord
  ) {
    return { ...candidate, unsafeReason: "source-file-drift" };
  }
  if (
    (duplicate.selector === ".review-status-banner" || duplicate.selector === ".dark .review-status-banner") &&
    !phase15HasExpectedProperties(operatorRecord, PHASE16_SECONDARY_PROPERTIES_BY_SELECTOR.get(duplicate.selector))
  ) {
    return { ...candidate, unsafeReason: "operator-declaration-drift" };
  }
  if (!secondary) {
    return { ...candidate, unsafeReason: "surface-declaration-missing" };
  }
  if (duplicate.selector === ".review-provenance-item" && secondary.source !== PHASE16_SURFACE_NORMALIZATION_FILE) {
    return { ...candidate, unsafeReason: "source-file-drift" };
  }
  if (
    duplicate.selector === ".review-provenance-item" &&
    (surfacePrimary.line || 0) >= (secondary.line || 0)
  ) {
    return { ...candidate, unsafeReason: "source-order-sensitive" };
  }

  return {
    ...candidate,
    candidateStatus: "safe",
    approvedPhase16Target: true,
    targetLocation: {
      file: surfacePrimary.source,
      line: surfacePrimary.line,
      column: surfacePrimary.column
    },
    sourceLocation: {
      file: secondary.source,
      line: secondary.line,
      column: secondary.column
    },
    safetyChecks: {
      sameLayer: true,
      emptyAtRuleContext: true,
      selectorListCoveragePreserved: true,
      exactDeclarationMovement: true,
      toneRuleOrderPreserved: true,
      darkProvenanceOverridePreserved: true,
      noSpecificityChangingGrouping: true
    }
  };
}

function analyzePhase16ReviewSurfaceCandidates(duplicates, baseline = loadBaseline()) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  return duplicates
    .map((duplicate) =>
      analyzePhase16ReviewSurfaceCandidate(duplicate, baselineEntries.get(duplicate.key) || duplicate.baselineEntry || duplicate)
    )
    .sort((left, right) => left.key.localeCompare(right.key));
}

function phase17ExpectedPrimarySource(selector) {
  if (selector === ".workspace-shell") {
    return PHASE17_SURFACE_NORMALIZATION_FILE;
  }
  if (selector === ".shell-noise" || selector === ".review-provenance-label") {
    return PHASE17_OPERATOR_CONSOLE_FILE;
  }
  return null;
}

function phase17ExpectedSourceFile(selector) {
  if (selector === ".workspace-shell") {
    return PHASE17_SHELL_FOUNDATION_FILE;
  }
  if (selector === ".shell-noise" || selector === ".review-provenance-label") {
    return PHASE17_SURFACE_NORMALIZATION_FILE;
  }
  return null;
}

function analyzePhase17SurfaceFinalPassCandidate(duplicate, baselineEntry = {}) {
  const entry = baselineEntry || {};
  const records = duplicate.records || [];
  const candidate = {
    key: duplicate.key,
    selector: duplicate.selector,
    layer: duplicate.layer,
    atRuleContext: duplicate.context || [],
    declarationConflict: duplicate.category,
    removalStatus: entry.removalStatus || duplicate.removalStatus || null,
    sourceLocations: phase11SourceLocations(duplicate),
    candidateStatus: "deferred"
  };

  if (duplicate.hotspot || entry.hotspot) {
    return { ...candidate, unsafeReason: "hotspot" };
  }
  if (duplicate.category !== "additive" || (entry.removalStatus || duplicate.removalStatus) !== "removable-later") {
    return { ...candidate, unsafeReason: "conflicting-permanent" };
  }
  if (!PHASE17_TARGET_DUPLICATE_KEYS.has(duplicate.key)) {
    return { ...candidate, unsafeReason: "selector-not-phase17-target" };
  }
  if (duplicate.layer !== "components" || records.some((record) => (record.layer || duplicate.layer) !== "components")) {
    return { ...candidate, unsafeReason: "cross-layer" };
  }
  if (records.some(hasSpecificityChangingGrouping)) {
    return { ...candidate, unsafeReason: "specificity-changing-grouping" };
  }
  if (new Set(records.map((record) => JSON.stringify(record.context || duplicate.context || []))).size > 1) {
    return { ...candidate, unsafeReason: "cross-context" };
  }
  if (JSON.stringify(duplicate.context || []) !== JSON.stringify([])) {
    return { ...candidate, unsafeReason: "cross-context" };
  }

  const expectedPrimarySource = phase17ExpectedPrimarySource(duplicate.selector);
  const expectedSourceFile = phase17ExpectedSourceFile(duplicate.selector);
  const primary = records.find(
    (record) =>
      record.source === expectedPrimarySource &&
      phase15HasExpectedProperties(record, PHASE17_PRIMARY_PROPERTIES_BY_SELECTOR.get(duplicate.selector))
  );
  const source = records.find(
    (record) =>
      record.source === expectedSourceFile &&
      phase15HasExpectedProperties(record, PHASE17_SOURCE_PROPERTIES_BY_SELECTOR.get(duplicate.selector))
  );

  if (!primary) {
    return { ...candidate, unsafeReason: "primary-declaration-missing" };
  }
  if (!source) {
    return { ...candidate, unsafeReason: "source-declaration-drift" };
  }

  let sourceLocations = [source];
  if (duplicate.selector === ".workspace-shell") {
    const shadow = records.find(
      (record) =>
        record.source === PHASE17_SURFACE_NORMALIZATION_FILE &&
        phase15HasExpectedProperties(record, PHASE17_WORKSPACE_SHADOW_PROPERTIES)
    );
    if (!shadow) {
      return { ...candidate, unsafeReason: "primary-declaration-missing" };
    }
    sourceLocations = [source, shadow];
    if ((primary.line || 0) >= (shadow.line || 0)) {
      return { ...candidate, unsafeReason: "source-order-sensitive" };
    }
  }

  return {
    ...candidate,
    candidateStatus: "safe",
    approvedPhase17Target: true,
    targetLocation: {
      file: primary.source,
      line: primary.line,
      column: primary.column
    },
    sourceLocations: sourceLocations.map((record) => ({
      file: record.source,
      line: record.line,
      column: record.column
    })),
    safetyChecks: {
      sameLayer: true,
      emptyAtRuleContext: true,
      selectorListCoveragePreserved: true,
      exactDeclarationMovement: true,
      darkOverridePreserved: true,
      ambientOverridePreserved: true,
      noSpecificityChangingGrouping: true
    }
  };
}

function analyzePhase17SurfaceFinalPassCandidates(duplicates, baseline = loadBaseline()) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  return duplicates
    .map((duplicate) =>
      analyzePhase17SurfaceFinalPassCandidate(duplicate, baselineEntries.get(duplicate.key) || duplicate.baselineEntry || duplicate)
    )
    .sort((left, right) => left.key.localeCompare(right.key));
}

function phase14FindRule(root, selectors) {
  let match = null;
  root.walkRules((rule) => {
    if (selectorListEquals(splitSelectorList(rule.selector), selectors)) {
      match = rule;
    }
  });
  return match;
}

function findRuleContainingSelector(root, selector) {
  let match = null;
  root.walkRules((rule) => {
    if (splitSelectorList(rule.selector).includes(selector)) {
      match = rule;
    }
  });
  return match;
}

function declarationsByProperty(rule) {
  const declarations = new Map();
  if (!rule) {
    return declarations;
  }
  rule.walkDecls((decl) => {
    declarations.set(decl.prop.toLowerCase(), {
      value: String(decl.value || "").trim(),
      important: Boolean(decl.important)
    });
  });
  return declarations;
}

function normalizedDeclarationValue(value) {
  return String(value || "")
    .replace(/\s+/g, " ")
    .replace(/\(\s+/g, "(")
    .replace(/\s+\)/g, ")")
    .trim();
}

function checkRuleDeclarations(rule, expected, failures, label) {
  if (!rule) {
    failures.push(`${label} rule is missing`);
    return;
  }
  const declarations = declarationsByProperty(rule);
  for (const [property, expectedValue] of Object.entries(expected)) {
    const actual = declarations.get(property);
    if (!actual || normalizedDeclarationValue(actual.value) !== normalizedDeclarationValue(expectedValue)) {
      failures.push(`${label} must preserve ${property}: ${expectedValue}`);
    }
  }
}

function checkRuleExactDeclarations(rule, expected, failures, label) {
  checkRuleDeclarations(rule, expected, failures, label);
  if (!rule) {
    return;
  }
  const actualProperties = Array.from(declarationsByProperty(rule).keys()).sort();
  const expectedProperties = Object.keys(expected).sort();
  if (JSON.stringify(actualProperties) !== JSON.stringify(expectedProperties)) {
    failures.push(`${label} declaration set drifted: expected ${expectedProperties.join(", ")}`);
  }
}

function checkPhase14SkeletonSourceShape(failures) {
  const dispatchRoot = parseCss(path.resolve(REPO_ROOT, PHASE14_TARGET_FILE));
  const sharedPrimitiveRule = phase14FindRule(dispatchRoot, PHASE14_TARGET_SELECTORS);
  if (sharedPrimitiveRule) {
    failures.push(`${PHASE14_TARGET_FILE} shared skeleton primitive selector list must be deleted after Phase 14`);
  }

  const expectedSingletons = new Map([
    [
      ".skeleton-line",
      {
        display: "block",
        overflow: "hidden",
        position: "relative",
        background: "rgba(226, 232, 240, 0.72)",
        height: "0.75rem",
        width: "100%",
        "border-radius": "999px"
      }
    ],
    [
      ".skeleton-block",
      {
        display: "block",
        overflow: "hidden",
        position: "relative",
        background: "rgba(226, 232, 240, 0.72)",
        "min-height": "13.5rem",
        "border-radius": "1.25rem"
      }
    ],
    [
      ".skeleton-pill",
      {
        display: "block",
        overflow: "hidden",
        position: "relative",
        background: "rgba(226, 232, 240, 0.72)",
        width: "4.75rem",
        height: "1.6rem",
        "border-radius": "999px"
      }
    ]
  ]);

  for (const [selector, declarations] of expectedSingletons.entries()) {
    const singleton = phase14FindRule(dispatchRoot, [selector]);
    checkRuleDeclarations(singleton, declarations, failures, `${PHASE14_TARGET_FILE} ${selector}`);
    const singletonLine = singleton?.source?.start?.line || 0;
    for (const variantSelector of PHASE14_VARIANTS_BY_SELECTOR.get(selector) || []) {
      const variant = phase14FindRule(dispatchRoot, [variantSelector]);
      if (!variant) {
        failures.push(`${PHASE14_TARGET_FILE} ${variantSelector} variant rule is missing`);
        continue;
      }
      const variantLine = variant.source?.start?.line || 0;
      if (singletonLine <= 0 || variantLine <= singletonLine) {
        failures.push(`${PHASE14_TARGET_FILE} ${selector} base rule must stay before ${variantSelector}`);
      }
    }
  }

  checkRuleDeclarations(
    phase14FindRule(dispatchRoot, PHASE14_TARGET_SELECTORS.map((selector) => `${selector}::before`)),
    {
      content: "\"\"",
      position: "absolute",
      inset: "0",
      background: "linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.85) 50%, transparent 100%)",
      transform: "translateX(-100%)",
      animation: "skeletonShimmer 1.4s ease-in-out infinite",
      "will-change": "transform",
      "pointer-events": "none"
    },
    failures,
    `${PHASE14_TARGET_FILE} skeleton shimmer`
  );
  checkRuleDeclarations(
    phase14FindRule(dispatchRoot, PHASE14_TARGET_SELECTORS.map((selector) => `.dark ${selector}`)),
    { background: "rgba(51, 65, 85, 0.72)" },
    failures,
    `${PHASE14_TARGET_FILE} dark skeleton background`
  );

  const accessibilityRoot = parseCss(path.resolve(PORTAL_CSS_SOURCE_DIR, "overrides.accessibility.css"));
  let reducedMotionRule = null;
  accessibilityRoot.walkRules((rule) => {
    const selectors = splitSelectorList(rule.selector);
    const context = atRuleContext(rule).join(" ");
    if (
      PHASE14_TARGET_SELECTORS.every((selector) => selectors.includes(selector)) &&
      /prefers-reduced-motion:\s*reduce/.test(context)
    ) {
      reducedMotionRule = rule;
    }
  });
  if (!reducedMotionRule) {
    failures.push("overrides.accessibility.css reduced-motion skeleton coverage is missing");
  } else {
    const declarations = declarationsByProperty(reducedMotionRule);
    for (const property of ["animation", "transition", "transform"]) {
      const declaration = declarations.get(property);
      if (!declaration || declaration.value !== "none" || !declaration.important) {
        failures.push(`overrides.accessibility.css reduced-motion skeleton ${property} must remain none !important`);
      }
    }
  }
}

function checkPhase15SurfaceLoadingSourceShape(failures) {
  const dispatchRoot = parseCss(path.resolve(REPO_ROOT, PHASE15_TARGET_FILE));
  const operatorRoot = parseCss(path.resolve(REPO_ROOT, PHASE15_OPERATOR_CONSOLE_FILE));

  const expectedDispatchRules = new Map([
    [
      ".surface-loading",
      {
        position: "relative",
        border: "1px solid rgba(148, 163, 184, 0.16)",
        background:
          "linear-gradient(180deg, rgba(255, 255, 255, 0.84), rgba(248, 250, 252, 0.76)), rgba(255, 255, 255, 0.8)",
        "box-shadow": "inset 0 0 0 1px rgba(15, 118, 110, 0.12), 0 18px 40px rgba(15, 23, 42, 0.08)",
        transition: "box-shadow 0.2s ease, border-color 0.2s ease"
      }
    ],
    [
      ".surface-loading::after",
      {
        content: "\"\"",
        position: "absolute",
        left: "1rem",
        right: "1rem",
        top: "0.75rem",
        height: "2px",
        "border-radius": "999px",
        "pointer-events": "none",
        opacity: "0.82",
        background: "linear-gradient(90deg, transparent, rgba(34, 211, 238, 0.18), transparent)"
      }
    ],
    [
      ".dark .surface-loading",
      {
        "border-color": "rgba(71, 85, 105, 0.48)",
        background:
          "linear-gradient(180deg, rgba(15, 23, 42, 0.84), var(--shell-veil-strong)), rgba(15, 23, 42, 0.78)",
        "box-shadow": "inset 0 0 0 1px rgba(45, 212, 191, 0.18), 0 22px 48px rgba(2, 6, 23, 0.3)"
      }
    ]
  ]);

  for (const [selector, declarations] of expectedDispatchRules.entries()) {
    checkRuleDeclarations(
      phase14FindRule(dispatchRoot, [selector]),
      declarations,
      failures,
      `${PHASE15_TARGET_FILE} ${selector}`
    );
    if (phase14FindRule(operatorRoot, [selector])) {
      failures.push(`${PHASE15_OPERATOR_CONSOLE_FILE} ${selector} duplicate rule must be removed after Phase 15`);
    }
  }

  checkRuleDeclarations(
    phase14FindRule(dispatchRoot, [".dark .surface-loading::after"]),
    {
      background:
        "linear-gradient(90deg, rgba(45, 212, 191, 0.18), rgba(103, 232, 249, 0.78), rgba(45, 212, 191, 0.18))"
    },
    failures,
    `${PHASE15_TARGET_FILE} .dark .surface-loading::after`
  );

  const accessibilityRoot = parseCss(path.resolve(PORTAL_CSS_SOURCE_DIR, "overrides.accessibility.css"));
  let reducedMotionRule = null;
  accessibilityRoot.walkRules((rule) => {
    const selectors = splitSelectorList(rule.selector);
    const context = atRuleContext(rule).join(" ");
    if (
      selectors.includes(".surface-loading") &&
      selectors.includes(".surface-loading::after") &&
      /prefers-reduced-motion:\s*reduce/.test(context)
    ) {
      reducedMotionRule = rule;
    }
  });
  if (!reducedMotionRule) {
    failures.push("overrides.accessibility.css reduced-motion surface-loading coverage is missing");
  } else {
    const declarations = declarationsByProperty(reducedMotionRule);
    for (const property of ["animation", "transition", "transform"]) {
      const declaration = declarations.get(property);
      if (!declaration || declaration.value !== "none" || !declaration.important) {
        failures.push(`overrides.accessibility.css reduced-motion surface-loading ${property} must remain none !important`);
      }
    }
  }
}

function checkPhase16ReviewSurfaceSourceShape(failures, options = {}) {
  const surfaceLabel = options.surfaceLabel || PHASE16_SURFACE_NORMALIZATION_FILE;
  const operatorLabel = options.operatorLabel || PHASE16_OPERATOR_CONSOLE_FILE;
  const surfaceRoot = options.surfaceRoot || parseCss(path.resolve(REPO_ROOT, PHASE16_SURFACE_NORMALIZATION_FILE));
  const operatorRoot = options.operatorRoot || parseCss(path.resolve(REPO_ROOT, PHASE16_OPERATOR_CONSOLE_FILE));

  const phase16SharedSurfaceSelectors = [
    ".workspace-shell",
    ".shell-panel",
    ".shell-panel-strong",
    ".panel-subtle",
    ".stat-tile",
    ".console-context-card",
    ".console-action-rail",
    ".build-pulse-card",
    ".runtime-briefing-card",
    ".review-compare-summary",
    "#artifactPreviewStage",
    "#artifactMetadataBar",
    "#artifactMetadataCard",
    "#reconstructionRuntimeSummary",
    ".build-step-tab"
  ];
  const sharedSurfaceRule =
    phase14FindRule(surfaceRoot, phase16SharedSurfaceSelectors) ||
    phase14FindRule(surfaceRoot, phase16SharedSurfaceSelectors.filter((selector) => selector !== ".workspace-shell"));
  if (!sharedSurfaceRule) {
    failures.push(`${surfaceLabel} shared surface chrome selector list drifted`);
  }
  if (findRuleContainingSelector(operatorRoot, ".review-status-banner")) {
    failures.push(`${operatorLabel} .review-status-banner duplicate rule must be removed after Phase 16`);
  }
  if (findRuleContainingSelector(operatorRoot, ".dark .review-status-banner")) {
    failures.push(`${operatorLabel} .dark .review-status-banner duplicate rule must be removed after Phase 16`);
  }

  const reviewStatusRule = phase14FindRule(surfaceRoot, [".review-status-banner"]);
  const darkReviewStatusRule = phase14FindRule(surfaceRoot, [".dark .review-status-banner"]);
  const reviewProvenanceRule = phase14FindRule(surfaceRoot, [".review-provenance-item"]);
  checkRuleDeclarations(
    reviewStatusRule,
    {
      "border-radius": "var(--ux-radius-lg)",
      "border-color": "var(--ux-border-subtle)",
      background: "var(--ux-surface-elevated)",
      "box-shadow": "inset 0 1px 0 rgba(255, 255, 255, 0.2)"
    },
    failures,
    `${surfaceLabel} .review-status-banner`
  );
  checkRuleDeclarations(
    darkReviewStatusRule,
    {
      background: "var(--ux-surface-muted)",
      "box-shadow": "inset 0 1px 0 var(--shell-tint-faint)"
    },
    failures,
    `${surfaceLabel} .dark .review-status-banner`
  );
  checkRuleDeclarations(
    reviewProvenanceRule,
    {
      "border-radius": "var(--ux-radius-lg)",
      "border-color": "var(--ux-border-subtle)",
      background: "var(--ux-surface-elevated)",
      "box-shadow": "var(--ux-shadow-surface)"
    },
    failures,
    `${surfaceLabel} .review-provenance-item`
  );

  const readyToneRule = phase14FindRule(surfaceRoot, [".review-status-banner[data-tone=\"ready\"]"]);
  if (
    darkReviewStatusRule?.source?.start?.line &&
    readyToneRule?.source?.start?.line &&
    darkReviewStatusRule.source.start.line >= readyToneRule.source.start.line
  ) {
    failures.push(`${surfaceLabel} .dark .review-status-banner must stay before tone rules`);
  }

  const darkProvenanceRule = findRuleContainingSelector(surfaceRoot, ".dark .review-provenance-item");
  if (!darkProvenanceRule) {
    failures.push(`${surfaceLabel} .dark .review-provenance-item rule is missing`);
  } else {
    checkRuleDeclarations(
      darkProvenanceRule,
      { background: "var(--ux-surface-muted)" },
      failures,
      `${surfaceLabel} .dark .review-provenance-item`
    );
    if (
      reviewProvenanceRule?.source?.start?.line &&
      darkProvenanceRule.source?.start?.line &&
      darkProvenanceRule.source.start.line <= reviewProvenanceRule.source.start.line
    ) {
      failures.push(`${surfaceLabel} .dark .review-provenance-item must stay after light singleton`);
    }
  }

  const shadowRule = phase14FindRule(surfaceRoot, [".console-context-card", ".build-pulse-card", ".runtime-briefing-card"]);
  if (!shadowRule) {
    failures.push(`${surfaceLabel} shared shadow selector list drifted`);
  }
}

function checkPhase17SurfaceFinalPassSourceShape(failures, options = {}) {
  const surfaceLabel = options.surfaceLabel || PHASE17_SURFACE_NORMALIZATION_FILE;
  const operatorLabel = options.operatorLabel || PHASE17_OPERATOR_CONSOLE_FILE;
  const shellLabel = options.shellLabel || PHASE17_SHELL_FOUNDATION_FILE;
  const surfaceRoot = options.surfaceRoot || parseCss(path.resolve(REPO_ROOT, PHASE17_SURFACE_NORMALIZATION_FILE));
  const operatorRoot = options.operatorRoot || parseCss(path.resolve(REPO_ROOT, PHASE17_OPERATOR_CONSOLE_FILE));
  const shellRoot = options.shellRoot || parseCss(path.resolve(REPO_ROOT, PHASE17_SHELL_FOUNDATION_FILE));

  const workspaceRule = phase14FindRule(surfaceRoot, [".workspace-shell"]);
  checkRuleExactDeclarations(
    workspaceRule,
    {
      position: "relative",
      "border-radius": "var(--ux-radius-lg)",
      "border-color": "var(--ux-border-subtle)",
      background: "var(--ux-surface-elevated)",
      transition:
        "transform 0.35s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.35s ease, box-shadow 0.35s ease, background-color 0.35s ease",
      "box-shadow": "var(--ux-shadow-surface)"
    },
    failures,
    `${surfaceLabel} .workspace-shell`
  );
  if (phase14FindRule(shellRoot, [".workspace-shell"])) {
    failures.push(`${shellLabel} .workspace-shell duplicate rule must be removed after Phase 17`);
  }
  const workspaceAmbientRule = phase14FindRule(shellRoot, [".workspace-shell[data-ambient-active=\"true\"]"]);
  const darkWorkspaceAmbientRule = phase14FindRule(shellRoot, [".dark .workspace-shell[data-ambient-active=\"true\"]"]);
  if (!workspaceAmbientRule || !darkWorkspaceAmbientRule) {
    failures.push(`${shellLabel} workspace ambient-active overrides must remain after Phase 17`);
  }
  const darkWorkspaceRule = findRuleContainingSelector(surfaceRoot, ".dark .workspace-shell");
  if (!darkWorkspaceRule) {
    failures.push(`${surfaceLabel} .dark .workspace-shell override is missing`);
  } else {
    checkRuleDeclarations(
      darkWorkspaceRule,
      { background: "var(--ux-surface-muted)" },
      failures,
      `${surfaceLabel} .dark .workspace-shell`
    );
    if (workspaceRule?.source?.start?.line && darkWorkspaceRule.source?.start?.line <= workspaceRule.source.start.line) {
      failures.push(`${surfaceLabel} .dark .workspace-shell must stay after light singleton`);
    }
  }

  const shellNoiseRule = phase14FindRule(operatorRoot, [".shell-noise"]);
  checkRuleExactDeclarations(
    shellNoiseRule,
    {
      display: "none",
      position: "fixed",
      inset: "0",
      "z-index": "-20",
      "pointer-events": "none",
      opacity: "0.2",
      "background-image":
        "radial-gradient(rgba(15, 23, 42, 0.08) 0.45px, transparent 0.45px), radial-gradient(rgba(8, 145, 178, 0.06) 0.45px, transparent 0.45px)",
      "background-position": "0 0, 16px 16px",
      "background-size": "22px 22px, 28px 28px"
    },
    failures,
    `${operatorLabel} .shell-noise`
  );
  if (phase14FindRule(surfaceRoot, [".shell-noise"])) {
    failures.push(`${surfaceLabel} .shell-noise duplicate rule must be removed after Phase 17`);
  }
  checkRuleDeclarations(
    phase14FindRule(operatorRoot, [".dark .shell-noise"]),
    {
      opacity: "0.18",
      "background-image":
        "radial-gradient(rgba(255, 255, 255, 0.08) 0.45px, transparent 0.45px), radial-gradient(rgba(34, 211, 238, 0.08) 0.45px, transparent 0.45px)"
    },
    failures,
    `${operatorLabel} .dark .shell-noise`
  );

  const provenanceLabelRule = findRuleContainingSelector(operatorRoot, ".review-provenance-label");
  checkRuleExactDeclarations(
    provenanceLabelRule,
    {
      "font-size": "13px",
      "line-height": "1.45",
      "letter-spacing": "var(--ux-meta-tracking)"
    },
    failures,
    `${operatorLabel} .review-provenance-label`
  );
  if (findRuleContainingSelector(surfaceRoot, ".review-provenance-label")) {
    failures.push(`${surfaceLabel} .review-provenance-label duplicate rule must be removed after Phase 17`);
  }

  const sharedSurfaceRule = phase14FindRule(surfaceRoot, [
    ".shell-panel",
    ".shell-panel-strong",
    ".panel-subtle",
    ".stat-tile",
    ".console-context-card",
    ".console-action-rail",
    ".build-pulse-card",
    ".runtime-briefing-card",
    ".review-compare-summary",
    "#artifactPreviewStage",
    "#artifactMetadataBar",
    "#artifactMetadataCard",
    "#reconstructionRuntimeSummary",
    ".build-step-tab"
  ]);
  if (!sharedSurfaceRule) {
    failures.push(`${surfaceLabel} shared surface chrome selector list drifted after Phase 17`);
  }
  const sharedShadowRule = phase14FindRule(surfaceRoot, [".portal-topbar", ".portal-context-shell", ".shell-panel", ".shell-panel-strong"]);
  if (!sharedShadowRule) {
    failures.push(`${surfaceLabel} shared shadow selector list drifted after Phase 17`);
  }
}

function deferredReasonCounts(candidates) {
  const counts = {};
  for (const candidate of candidates) {
    if (candidate.candidateStatus !== "deferred") {
      continue;
    }
    const reason = candidate.unsafeReason || "coverage-ambiguous";
    counts[reason] = (counts[reason] || 0) + 1;
  }
  return Object.fromEntries(Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)));
}

function buildPhase9DuplicateState(duplicates, baseline, phase8State) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  const ownedDuplicateContextCount = duplicates.filter((duplicate) =>
    phase9EntryIsOwned(baselineEntries.get(duplicate.key), duplicate)
  ).length;
  const duplicateContextCountAfter = duplicates.length;
  const reclassifiedDuplicateContextCount = Math.max(0, PHASE9_DUPLICATE_CONTEXT_COUNT_BEFORE - duplicateContextCountAfter);
  return {
    phase: PHASE9_DUPLICATE_PHASE,
    baselinePath: relativePath(BASELINE_PATH),
    baselineSha256: existsSync(BASELINE_PATH) ? sha256File(BASELINE_PATH) : null,
    duplicateContextCountBefore: PHASE9_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter,
    duplicateContextCount: duplicateContextCountAfter,
    ownedDuplicateContextCount,
    unownedDuplicateContextCount: duplicateContextCountAfter - ownedDuplicateContextCount,
    hotspotDuplicateContextCount: duplicates.filter((duplicate) => duplicate.hotspot).length,
    consolidatedDuplicateContextCount: 0,
    reclassifiedDuplicateContextCount,
    conflictingDuplicateContextCount: countDuplicatesByCategory(duplicates, "conflicting"),
    additiveDuplicateContextCount: countDuplicatesByCategory(duplicates, "additive"),
    identicalDuplicateContextCount: countDuplicatesByCategory(duplicates, "identical"),
    contextualDuplicateContextCount: countDuplicatesByCategory(duplicates, "contextual"),
    removedRawBytes: 0,
    removedGzipBytes: 0,
    sentinelStatePreserved:
      Boolean(phase8State?.utilitiesCompatHold?.imported) &&
      phase8State.utilitiesCompatHold.layer === "utilities" &&
      Boolean(phase8State.utilitiesCompatHold.sentinelOnly) &&
      phase8State.utilitiesCompatHold.sourceRuleCount === 0 &&
      !phase8State?.overridesCompat?.imported &&
      Boolean(phase8State?.overridesCompat?.sentinelOnly) &&
      phase8State.overridesCompat.sourceRuleCount === 0,
    parityBaselineChanged: false
  };
}

function buildPhase10AdditiveConsolidationState(duplicates, baseline, phase8State) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  const ownedDuplicateContextCount = duplicates.filter((duplicate) =>
    phase9EntryIsOwned(baselineEntries.get(duplicate.key), duplicate)
  ).length;
  const cssText = readText(PORTAL_CSS_ASSET_PATH);
  const generatedRawBytesAfter = Buffer.byteLength(cssText, "utf8");
  const generatedGzipBytesAfter = gzipByteLength(cssText);
  const candidates = analyzePhase10AdditiveCandidates(duplicates, baseline);
  const safeCandidates = candidates.filter((candidate) => candidate.candidateStatus === "safe");
  const deferredCandidates = candidates.filter((candidate) => candidate.candidateStatus === "deferred");
  const additiveDuplicateContextCountAfter = countDuplicatesByCategory(duplicates, "additive");
  const generatedRawByteDelta = generatedRawBytesAfter - PHASE10_GENERATED_PORTAL_CSS_BYTES_BEFORE;
  const generatedGzipByteDelta = generatedGzipBytesAfter - PHASE10_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE;

  return {
    phase: PHASE10_ADDITIVE_PHASE,
    baselinePath: relativePath(BASELINE_PATH),
    baselineSha256: existsSync(BASELINE_PATH) ? sha256File(BASELINE_PATH) : null,
    additiveDuplicateContextCountBefore: PHASE10_ADDITIVE_CONTEXT_COUNT_BEFORE,
    additiveDuplicateContextCountAfter,
    safeCandidateCountBefore: PHASE10_SAFE_CANDIDATE_COUNT_BEFORE,
    safeCandidateCountAfter: safeCandidates.length,
    consolidatedCandidateCount: Math.max(0, PHASE10_ADDITIVE_CONTEXT_COUNT_BEFORE - additiveDuplicateContextCountAfter),
    deferredCandidateCount: deferredCandidates.length,
    deferredReasonCounts: deferredReasonCounts(candidates),
    deferredCandidates: deferredCandidates.map((candidate) => ({
      key: candidate.key,
      selector: candidate.selector,
      unsafeReason: candidate.unsafeReason
    })),
    conflictingDuplicateContextCount: countDuplicatesByCategory(duplicates, "conflicting"),
    ownedDuplicateContextCount,
    unownedDuplicateContextCount: duplicates.length - ownedDuplicateContextCount,
    hotspotDuplicateContextCount: duplicates.filter((duplicate) => duplicate.hotspot).length,
    removedRawBytes: Math.max(0, -generatedRawByteDelta),
    removedGzipBytes: Math.max(0, -generatedGzipByteDelta),
    generatedRawBytesBefore: PHASE10_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawBytesAfter,
    generatedRawByteDelta,
    generatedGzipBytesBefore: PHASE10_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipBytesAfter,
    generatedGzipByteDelta,
    generatedPortalCssHashBefore: PHASE10_GENERATED_PORTAL_CSS_HASH_BEFORE,
    generatedPortalCssHashAfter: sha256File(PORTAL_CSS_ASSET_PATH),
    renderedPortalCssFingerprintBefore: PHASE10_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    renderedPortalCssFingerprintAfter: renderedPortalCssFingerprint(),
    sentinelStatePreserved:
      Boolean(phase8State?.utilitiesCompatHold?.imported) &&
      phase8State.utilitiesCompatHold.layer === "utilities" &&
      Boolean(phase8State.utilitiesCompatHold.sentinelOnly) &&
      phase8State.utilitiesCompatHold.sourceRuleCount === 0 &&
      !phase8State?.overridesCompat?.imported &&
      Boolean(phase8State?.overridesCompat?.sentinelOnly) &&
      phase8State.overridesCompat.sourceRuleCount === 0,
    parityBaselineChanged: false
  };
}

function sentinelStateIsPreserved(phase8State) {
  return (
    Boolean(phase8State?.utilitiesCompatHold?.imported) &&
    phase8State.utilitiesCompatHold.layer === "utilities" &&
    Boolean(phase8State.utilitiesCompatHold.sentinelOnly) &&
    phase8State.utilitiesCompatHold.sourceRuleCount === 0 &&
    !phase8State?.overridesCompat?.imported &&
    Boolean(phase8State?.overridesCompat?.sentinelOnly) &&
    phase8State.overridesCompat.sourceRuleCount === 0
  );
}

function expectedHistoricalPhase11SurfaceListConsolidationState() {
  return {
    phase: PHASE11_SURFACE_PHASE,
    targetFile: PHASE11_TARGET_FILE,
    baselinePath: relativePath(BASELINE_PATH),
    baselineSha256: PHASE11_BASELINE_SHA256,
    duplicateContextCountBefore: PHASE11_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter: PHASE11_DUPLICATE_CONTEXT_COUNT_AFTER,
    additiveDuplicateContextCountBefore: PHASE11_ADDITIVE_CONTEXT_COUNT_BEFORE,
    additiveDuplicateContextCountAfter: PHASE11_ADDITIVE_CONTEXT_COUNT_AFTER,
    conflictingPermanentContextCountBefore: PHASE11_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    conflictingPermanentContextCountAfter: PHASE11_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    unownedDuplicateContextCountAfter: 0,
    hotspotDuplicateContextCountAfter: 0,
    consolidatedContextCount: PHASE11_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    expectedConsolidatedContextCount: PHASE11_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    consolidatedContexts: Array.from(PHASE11_TARGET_DUPLICATE_KEYS).sort(),
    remainingTargetContexts: [],
    unexpectedResolvedContextCount: 0,
    deferredOutOfScopeCandidateCount: PHASE11_ADDITIVE_CONTEXT_COUNT_AFTER,
    deferredReasonCounts: {
      "selector-not-phase11-target": PHASE11_ADDITIVE_CONTEXT_COUNT_AFTER
    },
    removedRawBytes: 246,
    removedGzipBytes: 9,
    generatedRawBytesBefore: PHASE11_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawBytesAfter: PHASE12_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawByteDelta: PHASE12_GENERATED_PORTAL_CSS_BYTES_BEFORE - PHASE11_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedGzipBytesBefore: PHASE11_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipBytesAfter: PHASE12_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipByteDelta: PHASE12_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE - PHASE11_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedPortalCssHashBefore: PHASE11_GENERATED_PORTAL_CSS_HASH_BEFORE,
    generatedPortalCssHashAfter: PHASE12_GENERATED_PORTAL_CSS_HASH_BEFORE,
    renderedPortalCssFingerprintBefore: PHASE11_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    renderedPortalCssFingerprintAfter: PHASE12_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    sentinelStatePreserved: true,
    parityBaselineChanged: false
  };
}

function buildPhase12ComponentSingletonConsolidationState(duplicates, baseline, phase8State) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  const ownedDuplicateContextCount = duplicates.filter((duplicate) =>
    phase9EntryIsOwned(baselineEntries.get(duplicate.key), duplicate)
  ).length;
  const duplicateKeys = new Set(duplicates.map((duplicate) => duplicate.key));
  const consolidatedContexts = Array.from(PHASE12_TARGET_DUPLICATE_KEYS)
    .filter((key) => !duplicateKeys.has(key))
    .sort();
  const remainingTargetContexts = Array.from(PHASE12_TARGET_DUPLICATE_KEYS)
    .filter((key) => duplicateKeys.has(key))
    .sort();
  const cssText = readText(PORTAL_CSS_ASSET_PATH);
  const generatedRawBytesAfter = Buffer.byteLength(cssText, "utf8");
  const generatedGzipBytesAfter = gzipByteLength(cssText);
  const generatedRawByteDelta = generatedRawBytesAfter - PHASE12_GENERATED_PORTAL_CSS_BYTES_BEFORE;
  const generatedGzipByteDelta = generatedGzipBytesAfter - PHASE12_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE;
  const candidates = analyzePhase12ComponentCandidates(duplicates, baseline).filter(
    (candidate) => candidate.declarationConflict === "additive" && candidate.removalStatus === "removable-later"
  );

  return {
    phase: PHASE12_COMPONENT_PHASE,
    targetSelectors: PHASE12_TARGET_SELECTORS,
    baselinePath: relativePath(BASELINE_PATH),
    baselineSha256: existsSync(BASELINE_PATH) ? sha256File(BASELINE_PATH) : null,
    duplicateContextCountBefore: PHASE12_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter: duplicates.length,
    additiveDuplicateContextCountBefore: PHASE12_ADDITIVE_CONTEXT_COUNT_BEFORE,
    additiveDuplicateContextCountAfter: countDuplicatesByCategory(duplicates, "additive"),
    conflictingPermanentContextCountBefore: PHASE12_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    conflictingPermanentContextCountAfter: countDuplicatesByCategory(duplicates, "conflicting"),
    unownedDuplicateContextCountAfter: duplicates.length - ownedDuplicateContextCount,
    hotspotDuplicateContextCountAfter: duplicates.filter((duplicate) => duplicate.hotspot).length,
    consolidatedContextCount: consolidatedContexts.length,
    expectedConsolidatedContextCount: PHASE12_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    consolidatedContexts,
    remainingTargetContexts,
    unexpectedResolvedContextCount: Math.max(
      0,
      PHASE12_DUPLICATE_CONTEXT_COUNT_BEFORE - duplicates.length - consolidatedContexts.length
    ),
    nonTargetAdditiveCandidatesDeferredReason: "selector-not-phase12-target",
    deferredOutOfScopeCandidateCount: candidates.filter(
      (candidate) => candidate.unsafeReason === "selector-not-phase12-target"
    ).length,
    deferredReasonCounts: deferredReasonCounts(candidates),
    removedRawBytes: Math.max(0, -generatedRawByteDelta),
    removedGzipBytes: Math.max(0, -generatedGzipByteDelta),
    generatedRawBytesBefore: PHASE12_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawBytesAfter,
    generatedRawByteDelta,
    generatedGzipBytesBefore: PHASE12_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipBytesAfter,
    generatedGzipByteDelta,
    generatedPortalCssHashBefore: PHASE12_GENERATED_PORTAL_CSS_HASH_BEFORE,
    generatedPortalCssHashAfter: sha256File(PORTAL_CSS_ASSET_PATH),
    renderedPortalCssFingerprintBefore: PHASE12_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    renderedPortalCssFingerprintAfter: renderedPortalCssFingerprint(),
    sentinelStatePreserved: sentinelStateIsPreserved(phase8State),
    parityBaselineChanged: false
  };
}

function expectedHistoricalPhase12ComponentSingletonConsolidationState() {
  return {
    phase: PHASE12_COMPONENT_PHASE,
    targetSelectors: PHASE12_TARGET_SELECTORS,
    baselinePath: relativePath(BASELINE_PATH),
    baselineSha256: "a76052f0ecddc719c0182610d58f85b8f8ebe4b87463f1d8632401f36e3635bf",
    duplicateContextCountBefore: PHASE12_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter: PHASE12_DUPLICATE_CONTEXT_COUNT_AFTER,
    additiveDuplicateContextCountBefore: PHASE12_ADDITIVE_CONTEXT_COUNT_BEFORE,
    additiveDuplicateContextCountAfter: PHASE12_ADDITIVE_CONTEXT_COUNT_AFTER,
    conflictingPermanentContextCountBefore: PHASE12_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    conflictingPermanentContextCountAfter: PHASE12_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    unownedDuplicateContextCountAfter: 0,
    hotspotDuplicateContextCountAfter: 0,
    consolidatedContextCount: PHASE12_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    expectedConsolidatedContextCount: PHASE12_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    consolidatedContexts: Array.from(PHASE12_TARGET_DUPLICATE_KEYS).sort(),
    remainingTargetContexts: [],
    unexpectedResolvedContextCount: 0,
    nonTargetAdditiveCandidatesDeferredReason: "selector-not-phase12-target",
    deferredOutOfScopeCandidateCount: PHASE12_ADDITIVE_CONTEXT_COUNT_AFTER,
    deferredReasonCounts: {
      "selector-not-phase12-target": PHASE12_ADDITIVE_CONTEXT_COUNT_AFTER
    },
    removedRawBytes: 0,
    removedGzipBytes: 0,
    generatedRawBytesBefore: PHASE12_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawBytesAfter: PHASE13_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawByteDelta: PHASE13_GENERATED_PORTAL_CSS_BYTES_BEFORE - PHASE12_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedGzipBytesBefore: PHASE12_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipBytesAfter: PHASE13_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipByteDelta: PHASE13_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE - PHASE12_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedPortalCssHashBefore: PHASE12_GENERATED_PORTAL_CSS_HASH_BEFORE,
    generatedPortalCssHashAfter: PHASE13_GENERATED_PORTAL_CSS_HASH_BEFORE,
    renderedPortalCssFingerprintBefore: PHASE12_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    renderedPortalCssFingerprintAfter: PHASE13_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    sentinelStatePreserved: true,
    parityBaselineChanged: false
  };
}

function buildPhase13InteractionOutlineConsolidationState(duplicates, baseline, phase8State) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  const ownedDuplicateContextCount = duplicates.filter((duplicate) =>
    phase9EntryIsOwned(baselineEntries.get(duplicate.key), duplicate)
  ).length;
  const duplicateKeys = new Set(duplicates.map((duplicate) => duplicate.key));
  const consolidatedContexts = Array.from(PHASE13_TARGET_DUPLICATE_KEYS)
    .filter((key) => !duplicateKeys.has(key))
    .sort();
  const remainingTargetContexts = Array.from(PHASE13_TARGET_DUPLICATE_KEYS)
    .filter((key) => duplicateKeys.has(key))
    .sort();
  const cssText = readText(PORTAL_CSS_ASSET_PATH);
  const generatedRawBytesAfter = Buffer.byteLength(cssText, "utf8");
  const generatedGzipBytesAfter = gzipByteLength(cssText);
  const generatedRawByteDelta = generatedRawBytesAfter - PHASE13_GENERATED_PORTAL_CSS_BYTES_BEFORE;
  const generatedGzipByteDelta = generatedGzipBytesAfter - PHASE13_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE;
  const candidates = analyzePhase13InteractionCandidates(duplicates, baseline).filter(
    (candidate) => candidate.declarationConflict === "additive" && candidate.removalStatus === "removable-later"
  );

  return {
    phase: PHASE13_INTERACTION_PHASE,
    targetSelectors: PHASE13_TARGET_SELECTORS,
    baselinePath: relativePath(BASELINE_PATH),
    baselineSha256: existsSync(BASELINE_PATH) ? sha256File(BASELINE_PATH) : null,
    duplicateContextCountBefore: PHASE13_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter: duplicates.length,
    additiveDuplicateContextCountBefore: PHASE13_ADDITIVE_CONTEXT_COUNT_BEFORE,
    additiveDuplicateContextCountAfter: countDuplicatesByCategory(duplicates, "additive"),
    conflictingPermanentContextCountBefore: PHASE13_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    conflictingPermanentContextCountAfter: countDuplicatesByCategory(duplicates, "conflicting"),
    unownedDuplicateContextCountAfter: duplicates.length - ownedDuplicateContextCount,
    hotspotDuplicateContextCountAfter: duplicates.filter((duplicate) => duplicate.hotspot).length,
    consolidatedContextCount: consolidatedContexts.length,
    expectedConsolidatedContextCount: PHASE13_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    consolidatedContexts,
    remainingTargetContexts,
    unexpectedResolvedContextCount: Math.max(
      0,
      PHASE13_DUPLICATE_CONTEXT_COUNT_BEFORE - duplicates.length - consolidatedContexts.length
    ),
    nonTargetAdditiveCandidatesDeferredReason: "selector-not-phase13-target",
    deferredOutOfScopeCandidateCount: candidates.filter(
      (candidate) => candidate.unsafeReason === "selector-not-phase13-target"
    ).length,
    deferredReasonCounts: deferredReasonCounts(candidates),
    phase12HistoricalEvidencePreserved: true,
    removedRawBytes: Math.max(0, -generatedRawByteDelta),
    removedGzipBytes: Math.max(0, -generatedGzipByteDelta),
    generatedRawBytesBefore: PHASE13_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawBytesAfter,
    generatedRawByteDelta,
    generatedGzipBytesBefore: PHASE13_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipBytesAfter,
    generatedGzipByteDelta,
    generatedPortalCssHashBefore: PHASE13_GENERATED_PORTAL_CSS_HASH_BEFORE,
    generatedPortalCssHashAfter: sha256File(PORTAL_CSS_ASSET_PATH),
    renderedPortalCssFingerprintBefore: PHASE13_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    renderedPortalCssFingerprintAfter: renderedPortalCssFingerprint(),
    sentinelStatePreserved: sentinelStateIsPreserved(phase8State),
    parityBaselineChanged: false
  };
}

function expectedHistoricalPhase13InteractionOutlineConsolidationState() {
  return {
    phase: PHASE13_INTERACTION_PHASE,
    targetSelectors: PHASE13_TARGET_SELECTORS,
    baselinePath: relativePath(BASELINE_PATH),
    baselineSha256: PHASE13_BASELINE_SHA256,
    duplicateContextCountBefore: PHASE13_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter: PHASE13_DUPLICATE_CONTEXT_COUNT_AFTER,
    additiveDuplicateContextCountBefore: PHASE13_ADDITIVE_CONTEXT_COUNT_BEFORE,
    additiveDuplicateContextCountAfter: PHASE13_ADDITIVE_CONTEXT_COUNT_AFTER,
    conflictingPermanentContextCountBefore: PHASE13_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    conflictingPermanentContextCountAfter: PHASE13_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    unownedDuplicateContextCountAfter: 0,
    hotspotDuplicateContextCountAfter: 0,
    consolidatedContextCount: PHASE13_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    expectedConsolidatedContextCount: PHASE13_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    consolidatedContexts: Array.from(PHASE13_TARGET_DUPLICATE_KEYS).sort(),
    remainingTargetContexts: [],
    unexpectedResolvedContextCount: 0,
    nonTargetAdditiveCandidatesDeferredReason: "selector-not-phase13-target",
    deferredOutOfScopeCandidateCount: PHASE13_ADDITIVE_CONTEXT_COUNT_AFTER,
    deferredReasonCounts: {
      "selector-not-phase13-target": PHASE13_ADDITIVE_CONTEXT_COUNT_AFTER
    },
    phase12HistoricalEvidencePreserved: true,
    removedRawBytes: 0,
    removedGzipBytes: 0,
    generatedRawBytesBefore: PHASE13_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawBytesAfter: PHASE14_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawByteDelta: PHASE14_GENERATED_PORTAL_CSS_BYTES_BEFORE - PHASE13_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedGzipBytesBefore: PHASE13_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipBytesAfter: PHASE14_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipByteDelta: PHASE14_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE - PHASE13_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedPortalCssHashBefore: PHASE13_GENERATED_PORTAL_CSS_HASH_BEFORE,
    generatedPortalCssHashAfter: PHASE14_GENERATED_PORTAL_CSS_HASH_BEFORE,
    renderedPortalCssFingerprintBefore: PHASE13_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    renderedPortalCssFingerprintAfter: PHASE14_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    sentinelStatePreserved: true,
    parityBaselineChanged: false
  };
}

function expectedHistoricalPhase14SkeletonPrimitiveConsolidationState() {
  return {
    phase: PHASE14_SKELETON_PHASE,
    targetSelectors: PHASE14_TARGET_SELECTORS,
    targetFile: PHASE14_TARGET_FILE,
    baselinePath: relativePath(BASELINE_PATH),
    baselineSha256: PHASE14_BASELINE_SHA256,
    duplicateContextCountBefore: PHASE14_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter: PHASE14_DUPLICATE_CONTEXT_COUNT_AFTER,
    additiveDuplicateContextCountBefore: PHASE14_ADDITIVE_CONTEXT_COUNT_BEFORE,
    additiveDuplicateContextCountAfter: PHASE14_ADDITIVE_CONTEXT_COUNT_AFTER,
    conflictingPermanentContextCountBefore: PHASE14_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    conflictingPermanentContextCountAfter: PHASE14_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    unownedDuplicateContextCountAfter: 0,
    hotspotDuplicateContextCountAfter: 0,
    consolidatedContextCount: PHASE14_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    expectedConsolidatedContextCount: PHASE14_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    consolidatedContexts: Array.from(PHASE14_TARGET_DUPLICATE_KEYS).sort(),
    remainingTargetContexts: [],
    unexpectedResolvedContextCount: 0,
    nonTargetAdditiveCandidatesDeferredReason: "selector-not-phase14-target",
    deferredOutOfScopeCandidateCount: PHASE14_ADDITIVE_CONTEXT_COUNT_AFTER,
    deferredReasonCounts: {
      "selector-not-phase14-target": PHASE14_ADDITIVE_CONTEXT_COUNT_AFTER
    },
    phase13HistoricalEvidencePreserved: true,
    removedRawBytes: 0,
    removedGzipBytes: 5,
    generatedRawBytesBefore: PHASE14_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawBytesAfter: PHASE15_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawByteDelta: PHASE15_GENERATED_PORTAL_CSS_BYTES_BEFORE - PHASE14_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedGzipBytesBefore: PHASE14_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipBytesAfter: PHASE15_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipByteDelta: PHASE15_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE - PHASE14_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedPortalCssHashBefore: PHASE14_GENERATED_PORTAL_CSS_HASH_BEFORE,
    generatedPortalCssHashAfter: PHASE15_GENERATED_PORTAL_CSS_HASH_BEFORE,
    renderedPortalCssFingerprintBefore: PHASE14_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    renderedPortalCssFingerprintAfter: PHASE15_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    sentinelStatePreserved: true,
    parityBaselineChanged: false
  };
}

function buildPhase15SurfaceLoadingConsolidationState(duplicates, baseline, phase8State) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  const ownedDuplicateContextCount = duplicates.filter((duplicate) =>
    phase9EntryIsOwned(baselineEntries.get(duplicate.key), duplicate)
  ).length;
  const duplicateKeys = new Set(duplicates.map((duplicate) => duplicate.key));
  const consolidatedContexts = Array.from(PHASE15_TARGET_DUPLICATE_KEYS)
    .filter((key) => !duplicateKeys.has(key))
    .sort();
  const remainingTargetContexts = Array.from(PHASE15_TARGET_DUPLICATE_KEYS)
    .filter((key) => duplicateKeys.has(key))
    .sort();
  const cssText = readText(PORTAL_CSS_ASSET_PATH);
  const generatedRawBytesAfter = Buffer.byteLength(cssText, "utf8");
  const generatedGzipBytesAfter = gzipByteLength(cssText);
  const generatedRawByteDelta = generatedRawBytesAfter - PHASE15_GENERATED_PORTAL_CSS_BYTES_BEFORE;
  const generatedGzipByteDelta = generatedGzipBytesAfter - PHASE15_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE;
  const candidates = analyzePhase15SurfaceLoadingCandidates(duplicates, baseline).filter(
    (candidate) => candidate.declarationConflict === "additive" && candidate.removalStatus === "removable-later"
  );

  return {
    phase: PHASE15_SURFACE_LOADING_PHASE,
    targetSelectors: PHASE15_TARGET_SELECTORS,
    targetFile: PHASE15_TARGET_FILE,
    sourceFile: PHASE15_OPERATOR_CONSOLE_FILE,
    baselinePath: relativePath(BASELINE_PATH),
    baselineSha256: existsSync(BASELINE_PATH) ? sha256File(BASELINE_PATH) : null,
    duplicateContextCountBefore: PHASE15_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter: duplicates.length,
    additiveDuplicateContextCountBefore: PHASE15_ADDITIVE_CONTEXT_COUNT_BEFORE,
    additiveDuplicateContextCountAfter: countDuplicatesByCategory(duplicates, "additive"),
    conflictingPermanentContextCountBefore: PHASE15_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    conflictingPermanentContextCountAfter: countDuplicatesByCategory(duplicates, "conflicting"),
    unownedDuplicateContextCountAfter: duplicates.length - ownedDuplicateContextCount,
    hotspotDuplicateContextCountAfter: duplicates.filter((duplicate) => duplicate.hotspot).length,
    consolidatedContextCount: consolidatedContexts.length,
    expectedConsolidatedContextCount: PHASE15_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    consolidatedContexts,
    remainingTargetContexts,
    unexpectedResolvedContextCount: Math.max(
      0,
      PHASE15_DUPLICATE_CONTEXT_COUNT_BEFORE - duplicates.length - consolidatedContexts.length
    ),
    nonTargetAdditiveCandidatesDeferredReason: "selector-not-phase15-target",
    deferredOutOfScopeCandidateCount: candidates.filter(
      (candidate) => candidate.unsafeReason === "selector-not-phase15-target"
    ).length,
    deferredReasonCounts: deferredReasonCounts(candidates),
    phase14HistoricalEvidencePreserved: true,
    removedRawBytes: Math.max(0, -generatedRawByteDelta),
    removedGzipBytes: Math.max(0, -generatedGzipByteDelta),
    generatedRawBytesBefore: PHASE15_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawBytesAfter,
    generatedRawByteDelta,
    generatedGzipBytesBefore: PHASE15_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipBytesAfter,
    generatedGzipByteDelta,
    generatedPortalCssHashBefore: PHASE15_GENERATED_PORTAL_CSS_HASH_BEFORE,
    generatedPortalCssHashAfter: sha256File(PORTAL_CSS_ASSET_PATH),
    renderedPortalCssFingerprintBefore: PHASE15_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    renderedPortalCssFingerprintAfter: renderedPortalCssFingerprint(),
    sentinelStatePreserved: sentinelStateIsPreserved(phase8State),
    parityBaselineChanged: false
  };
}

function buildPhase16ReviewSurfaceConsolidationState(duplicates, baseline, phase8State) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  const ownedDuplicateContextCount = duplicates.filter((duplicate) =>
    phase9EntryIsOwned(baselineEntries.get(duplicate.key), duplicate)
  ).length;
  const duplicateKeys = new Set(duplicates.map((duplicate) => duplicate.key));
  const consolidatedContexts = Array.from(PHASE16_TARGET_DUPLICATE_KEYS)
    .filter((key) => !duplicateKeys.has(key))
    .sort();
  const remainingTargetContexts = Array.from(PHASE16_TARGET_DUPLICATE_KEYS)
    .filter((key) => duplicateKeys.has(key))
    .sort();
  const cssText = readText(PORTAL_CSS_ASSET_PATH);
  const generatedRawBytesAfter = Buffer.byteLength(cssText, "utf8");
  const generatedGzipBytesAfter = gzipByteLength(cssText);
  const generatedRawByteDelta = generatedRawBytesAfter - PHASE16_GENERATED_PORTAL_CSS_BYTES_BEFORE;
  const generatedGzipByteDelta = generatedGzipBytesAfter - PHASE16_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE;
  const candidates = analyzePhase16ReviewSurfaceCandidates(duplicates, baseline).filter(
    (candidate) => candidate.declarationConflict === "additive" && candidate.removalStatus === "removable-later"
  );

  return {
    phase: PHASE16_REVIEW_SURFACE_PHASE,
    targetSelectors: PHASE16_TARGET_SELECTORS,
    targetFile: PHASE16_SURFACE_NORMALIZATION_FILE,
    sourceFile: PHASE16_OPERATOR_CONSOLE_FILE,
    baselinePath: relativePath(BASELINE_PATH),
    baselineSha256: existsSync(BASELINE_PATH) ? sha256File(BASELINE_PATH) : null,
    duplicateContextCountBefore: PHASE16_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter: duplicates.length,
    additiveDuplicateContextCountBefore: PHASE16_ADDITIVE_CONTEXT_COUNT_BEFORE,
    additiveDuplicateContextCountAfter: countDuplicatesByCategory(duplicates, "additive"),
    conflictingPermanentContextCountBefore: PHASE16_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    conflictingPermanentContextCountAfter: countDuplicatesByCategory(duplicates, "conflicting"),
    unownedDuplicateContextCountAfter: duplicates.length - ownedDuplicateContextCount,
    hotspotDuplicateContextCountAfter: duplicates.filter((duplicate) => duplicate.hotspot).length,
    consolidatedContextCount: consolidatedContexts.length,
    expectedConsolidatedContextCount: PHASE16_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    consolidatedContexts,
    remainingTargetContexts,
    unexpectedResolvedContextCount: Math.max(
      0,
      PHASE16_DUPLICATE_CONTEXT_COUNT_BEFORE - duplicates.length - consolidatedContexts.length
    ),
    nonTargetAdditiveCandidatesDeferredReason: "selector-not-phase16-target",
    deferredOutOfScopeCandidateCount: candidates.filter(
      (candidate) => candidate.unsafeReason === "selector-not-phase16-target"
    ).length,
    deferredReasonCounts: deferredReasonCounts(candidates),
    phase15HistoricalEvidencePreserved: true,
    removedRawBytes: Math.max(0, -generatedRawByteDelta),
    removedGzipBytes: Math.max(0, -generatedGzipByteDelta),
    generatedRawBytesBefore: PHASE16_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawBytesAfter,
    generatedRawByteDelta,
    generatedGzipBytesBefore: PHASE16_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipBytesAfter,
    generatedGzipByteDelta,
    generatedPortalCssHashBefore: PHASE16_GENERATED_PORTAL_CSS_HASH_BEFORE,
    generatedPortalCssHashAfter: sha256File(PORTAL_CSS_ASSET_PATH),
    renderedPortalCssFingerprintBefore: PHASE16_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    renderedPortalCssFingerprintAfter: renderedPortalCssFingerprint(),
    sentinelStatePreserved: sentinelStateIsPreserved(phase8State),
    parityBaselineChanged: false
  };
}

function buildPhase17SurfaceFinalPassConsolidationState(duplicates, baseline, phase8State) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  const ownedDuplicateContextCount = duplicates.filter((duplicate) =>
    phase9EntryIsOwned(baselineEntries.get(duplicate.key), duplicate)
  ).length;
  const duplicateKeys = new Set(duplicates.map((duplicate) => duplicate.key));
  const consolidatedContexts = Array.from(PHASE17_TARGET_DUPLICATE_KEYS)
    .filter((key) => !duplicateKeys.has(key))
    .sort();
  const remainingTargetContexts = Array.from(PHASE17_TARGET_DUPLICATE_KEYS)
    .filter((key) => duplicateKeys.has(key))
    .sort();
  const cssText = readText(PORTAL_CSS_ASSET_PATH);
  const generatedRawBytesAfter = Buffer.byteLength(cssText, "utf8");
  const generatedGzipBytesAfter = gzipByteLength(cssText);
  const generatedRawByteDelta = generatedRawBytesAfter - PHASE17_GENERATED_PORTAL_CSS_BYTES_BEFORE;
  const generatedGzipByteDelta = generatedGzipBytesAfter - PHASE17_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE;
  const candidates = analyzePhase17SurfaceFinalPassCandidates(duplicates, baseline).filter(
    (candidate) => candidate.declarationConflict === "additive" && candidate.removalStatus === "removable-later"
  );

  return {
    phase: PHASE17_SURFACE_FINAL_PASS_PHASE,
    targetSelectors: PHASE17_TARGET_SELECTORS,
    targetFile: PHASE17_SURFACE_NORMALIZATION_FILE,
    sourceFiles: [PHASE17_OPERATOR_CONSOLE_FILE, PHASE17_SHELL_FOUNDATION_FILE],
    baselinePath: relativePath(BASELINE_PATH),
    baselineSha256: existsSync(BASELINE_PATH) ? sha256File(BASELINE_PATH) : null,
    duplicateContextCountBefore: PHASE17_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter: duplicates.length,
    additiveDuplicateContextCountBefore: PHASE17_ADDITIVE_CONTEXT_COUNT_BEFORE,
    additiveDuplicateContextCountAfter: countDuplicatesByCategory(duplicates, "additive"),
    conflictingPermanentContextCountBefore: PHASE17_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    conflictingPermanentContextCountAfter: countDuplicatesByCategory(duplicates, "conflicting"),
    unownedDuplicateContextCountAfter: duplicates.length - ownedDuplicateContextCount,
    hotspotDuplicateContextCountAfter: duplicates.filter((duplicate) => duplicate.hotspot).length,
    consolidatedContextCount: consolidatedContexts.length,
    expectedConsolidatedContextCount: PHASE17_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    consolidatedContexts,
    remainingTargetContexts,
    unexpectedResolvedContextCount: Math.max(
      0,
      PHASE17_DUPLICATE_CONTEXT_COUNT_BEFORE - duplicates.length - consolidatedContexts.length
    ),
    nonTargetAdditiveCandidatesDeferredReason: "selector-not-phase17-target",
    deferredOutOfScopeCandidateCount: candidates.filter(
      (candidate) => candidate.unsafeReason === "selector-not-phase17-target"
    ).length,
    deferredReasonCounts: deferredReasonCounts(candidates),
    phase16HistoricalEvidencePreserved: true,
    removedRawBytes: Math.max(0, -generatedRawByteDelta),
    removedGzipBytes: Math.max(0, -generatedGzipByteDelta),
    generatedRawBytesBefore: PHASE17_GENERATED_PORTAL_CSS_BYTES_BEFORE,
    generatedRawBytesAfter,
    generatedRawByteDelta,
    generatedGzipBytesBefore: PHASE17_GENERATED_PORTAL_CSS_GZIP_BYTES_BEFORE,
    generatedGzipBytesAfter,
    generatedGzipByteDelta,
    generatedPortalCssHashBefore: PHASE17_GENERATED_PORTAL_CSS_HASH_BEFORE,
    generatedPortalCssHashAfter: sha256File(PORTAL_CSS_ASSET_PATH),
    renderedPortalCssFingerprintBefore: PHASE17_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    renderedPortalCssFingerprintAfter: renderedPortalCssFingerprint(),
    sentinelStatePreserved: sentinelStateIsPreserved(phase8State),
    parityBaselineChanged: false
  };
}

function checkHistoricalPhase9State(report, failures) {
  const state = report.phase9DuplicateState || {};
  const expected = {
    phase: PHASE9_DUPLICATE_PHASE,
    duplicateContextCountBefore: 87,
    duplicateContextCountAfter: 85,
    duplicateContextCount: 85,
    ownedDuplicateContextCount: 85,
    unownedDuplicateContextCount: 0,
    hotspotDuplicateContextCount: 0,
    conflictingDuplicateContextCount: 56,
    additiveDuplicateContextCount: 29,
    sentinelStatePreserved: true,
    parityBaselineChanged: false
  };
  for (const [field, value] of Object.entries(expected)) {
    if (state[field] !== value) {
      failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase9DuplicateState ${field} must remain historical ${value}`);
    }
  }
}

function checkHistoricalPhase10State(report, failures) {
  const state = report.phase10AdditiveConsolidationState || {};
  const expected = {
    phase: PHASE10_ADDITIVE_PHASE,
    additiveDuplicateContextCountBefore: 30,
    additiveDuplicateContextCountAfter: 29,
    safeCandidateCountBefore: 1,
    safeCandidateCountAfter: 0,
    consolidatedCandidateCount: 1,
    deferredCandidateCount: 29,
    conflictingDuplicateContextCount: 56,
    unownedDuplicateContextCount: 0,
    hotspotDuplicateContextCount: 0,
    generatedPortalCssHashAfter: PHASE11_GENERATED_PORTAL_CSS_HASH_BEFORE,
    renderedPortalCssFingerprintAfter: PHASE11_RENDERED_PORTAL_CSS_FINGERPRINT_BEFORE,
    sentinelStatePreserved: true,
    parityBaselineChanged: false
  };
  for (const [field, value] of Object.entries(expected)) {
    if (state[field] !== value) {
      failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase10AdditiveConsolidationState ${field} must remain historical ${value}`);
    }
  }
}

function checkHistoricalPhase11State(report, failures) {
  const expected = expectedHistoricalPhase11SurfaceListConsolidationState();
  if (JSON.stringify(report.phase11SurfaceListConsolidationState || null) !== JSON.stringify(expected)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase11SurfaceListConsolidationState immutable historical evidence drifted`);
  }
}

function checkHistoricalPhase12State(report, failures) {
  const expected = expectedHistoricalPhase12ComponentSingletonConsolidationState();
  if (JSON.stringify(report.phase12ComponentSingletonConsolidationState || null) !== JSON.stringify(expected)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase12ComponentSingletonConsolidationState immutable historical evidence drifted`);
  }
}

function checkHistoricalPhase13State(report, failures) {
  const expected = expectedHistoricalPhase13InteractionOutlineConsolidationState();
  if (JSON.stringify(report.phase13InteractionOutlineConsolidationState || null) !== JSON.stringify(expected)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase13InteractionOutlineConsolidationState immutable historical evidence drifted`);
  }
}

function checkHistoricalPhase14State(report, failures) {
  const expected = expectedHistoricalPhase14SkeletonPrimitiveConsolidationState();
  if (JSON.stringify(report.phase14SkeletonPrimitiveConsolidationState || null) !== JSON.stringify(expected)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase14SkeletonPrimitiveConsolidationState immutable historical evidence drifted`);
  }
}

function checkHistoricalPhase15State(report, failures) {
  checkPhase15SurfaceLoadingConsolidationState(report.phase15SurfaceLoadingConsolidationState || {}, failures);
}

function checkHistoricalPhase16State(report, failures) {
  checkPhase16ReviewSurfaceConsolidationState(report.phase16ReviewSurfaceConsolidationState || {}, failures);
}

function checkPhase15SurfaceLoadingConsolidationState(state, failures) {
  const expectedFields = {
    phase: PHASE15_SURFACE_LOADING_PHASE,
    duplicateContextCountBefore: PHASE15_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter: PHASE15_DUPLICATE_CONTEXT_COUNT_AFTER,
    additiveDuplicateContextCountBefore: PHASE15_ADDITIVE_CONTEXT_COUNT_BEFORE,
    additiveDuplicateContextCountAfter: PHASE15_ADDITIVE_CONTEXT_COUNT_AFTER,
    conflictingPermanentContextCountBefore: PHASE15_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    conflictingPermanentContextCountAfter: PHASE15_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    unownedDuplicateContextCountAfter: 0,
    hotspotDuplicateContextCountAfter: 0,
    consolidatedContextCount: PHASE15_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    expectedConsolidatedContextCount: PHASE15_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    unexpectedResolvedContextCount: 0,
    nonTargetAdditiveCandidatesDeferredReason: "selector-not-phase15-target",
    phase14HistoricalEvidencePreserved: true,
    sentinelStatePreserved: true,
    parityBaselineChanged: false
  };
  for (const [field, value] of Object.entries(expectedFields)) {
    if (state[field] !== value) {
      failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase15SurfaceLoadingConsolidationState ${field} must be ${value}`);
    }
  }
  if (JSON.stringify(state.targetSelectors || []) !== JSON.stringify(PHASE15_TARGET_SELECTORS)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase15SurfaceLoadingConsolidationState targetSelectors drifted`);
  }
  if (JSON.stringify(state.remainingTargetContexts || []) !== JSON.stringify([])) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase15SurfaceLoadingConsolidationState still has unresolved target contexts`);
  }
  if (JSON.stringify(state.consolidatedContexts || []) !== JSON.stringify(Array.from(PHASE15_TARGET_DUPLICATE_KEYS).sort())) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase15SurfaceLoadingConsolidationState consolidated context allowlist drifted`);
  }
}

function checkPhase16ReviewSurfaceConsolidationState(state, failures) {
  const expectedFields = {
    phase: PHASE16_REVIEW_SURFACE_PHASE,
    duplicateContextCountBefore: PHASE16_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter: PHASE16_DUPLICATE_CONTEXT_COUNT_AFTER,
    additiveDuplicateContextCountBefore: PHASE16_ADDITIVE_CONTEXT_COUNT_BEFORE,
    additiveDuplicateContextCountAfter: PHASE16_ADDITIVE_CONTEXT_COUNT_AFTER,
    conflictingPermanentContextCountBefore: PHASE16_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    conflictingPermanentContextCountAfter: PHASE16_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    unownedDuplicateContextCountAfter: 0,
    hotspotDuplicateContextCountAfter: 0,
    consolidatedContextCount: PHASE16_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    expectedConsolidatedContextCount: PHASE16_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    unexpectedResolvedContextCount: 0,
    nonTargetAdditiveCandidatesDeferredReason: "selector-not-phase16-target",
    phase15HistoricalEvidencePreserved: true,
    sentinelStatePreserved: true,
    parityBaselineChanged: false
  };
  for (const [field, value] of Object.entries(expectedFields)) {
    if (state[field] !== value) {
      failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase16ReviewSurfaceConsolidationState ${field} must be ${value}`);
    }
  }
  if (JSON.stringify(state.targetSelectors || []) !== JSON.stringify(PHASE16_TARGET_SELECTORS)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase16ReviewSurfaceConsolidationState targetSelectors drifted`);
  }
  if (JSON.stringify(state.remainingTargetContexts || []) !== JSON.stringify([])) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase16ReviewSurfaceConsolidationState still has unresolved target contexts`);
  }
  if (JSON.stringify(state.consolidatedContexts || []) !== JSON.stringify(Array.from(PHASE16_TARGET_DUPLICATE_KEYS).sort())) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase16ReviewSurfaceConsolidationState consolidated context allowlist drifted`);
  }
}

function checkPhase17SurfaceFinalPassConsolidationState(state, failures) {
  const expectedFields = {
    phase: PHASE17_SURFACE_FINAL_PASS_PHASE,
    duplicateContextCountBefore: PHASE17_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter: PHASE17_DUPLICATE_CONTEXT_COUNT_AFTER,
    additiveDuplicateContextCountBefore: PHASE17_ADDITIVE_CONTEXT_COUNT_BEFORE,
    additiveDuplicateContextCountAfter: PHASE17_ADDITIVE_CONTEXT_COUNT_AFTER,
    conflictingPermanentContextCountBefore: PHASE17_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    conflictingPermanentContextCountAfter: PHASE17_CONFLICTING_PERMANENT_CONTEXT_COUNT,
    unownedDuplicateContextCountAfter: 0,
    hotspotDuplicateContextCountAfter: 0,
    consolidatedContextCount: PHASE17_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    expectedConsolidatedContextCount: PHASE17_EXPECTED_CONSOLIDATED_CONTEXT_COUNT,
    unexpectedResolvedContextCount: 0,
    nonTargetAdditiveCandidatesDeferredReason: "selector-not-phase17-target",
    phase16HistoricalEvidencePreserved: true,
    sentinelStatePreserved: true,
    parityBaselineChanged: false
  };
  for (const [field, value] of Object.entries(expectedFields)) {
    if (state[field] !== value) {
      failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase17SurfaceFinalPassConsolidationState ${field} must be ${value}`);
    }
  }
  if (JSON.stringify(state.targetSelectors || []) !== JSON.stringify(PHASE17_TARGET_SELECTORS)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase17SurfaceFinalPassConsolidationState targetSelectors drifted`);
  }
  if (JSON.stringify(state.remainingTargetContexts || []) !== JSON.stringify([])) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase17SurfaceFinalPassConsolidationState still has unresolved target contexts`);
  }
  if (JSON.stringify(state.consolidatedContexts || []) !== JSON.stringify(Array.from(PHASE17_TARGET_DUPLICATE_KEYS).sort())) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase17SurfaceFinalPassConsolidationState consolidated context allowlist drifted`);
  }
}

function checkOwnershipDrainReport(failures, duplicates) {
  if (!existsSync(OWNERSHIP_DRAIN_REPORT_PATH)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} is missing`);
    return;
  }
  const report = JSON.parse(readText(OWNERSHIP_DRAIN_REPORT_PATH));
  const moves = Array.isArray(report.moves) ? report.moves : [];
  const summary = report.summary || {};
  if (summary.utilityLayerImportsAfter !== 3) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} must record 3 utility-layer imports after Phase 5`);
  }
  if (summary.compatHoldCount !== 0) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} must record 0 compat-hold utilities`);
  }
  for (const [index, move] of moves.entries()) {
    for (const field of ["selector", "from", "fromLayer", "to", "toLayer", "classification", "parity"]) {
      if (!move[field]) {
        failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} move ${index} missing ${field}`);
      }
    }
    if (move.parity !== "green") {
      failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} move ${index} is not parity green`);
    }
    if (move.classification === "compatibility-final-order" && !move.reason) {
      failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} compatibility move ${index} missing reason`);
    }
  }

  const compatPath = path.resolve(PORTAL_CSS_SOURCE_DIR, "overrides.compat.css");
  const compatSelectors = selectorSetForFile(compatPath);
  const reportedCompatSelectors = new Set(
    moves
      .filter((move) => move.to === "overrides.compat.css")
      .map((move) => move.selector)
  );
  for (const selector of compatSelectors) {
    if (!reportedCompatSelectors.has(selector)) {
      failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} missing overrides.compat.css selector ${selector}`);
    }
  }
  if (summary.overridesCompatRuleCount !== ruleCountForFile(compatPath)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} overrides.compat.css rule count is stale`);
  }
  if (summary.overridesCompatBytes !== 0) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} overrides.compat.css shipped byte debt must be 0`);
  }

  const expectedPhase8State = buildPhase8SentinelState(collectCssImportGraph(PORTAL_CSS_INDEX_PATH, []));
  const expectedPhase11State = expectedHistoricalPhase11SurfaceListConsolidationState();
  const expectedPhase12State = expectedHistoricalPhase12ComponentSingletonConsolidationState();
  const expectedPhase13State = expectedHistoricalPhase13InteractionOutlineConsolidationState();
  const expectedPhase14State = expectedHistoricalPhase14SkeletonPrimitiveConsolidationState();
  const expectedPhase17State = buildPhase17SurfaceFinalPassConsolidationState(duplicates, loadBaseline(), expectedPhase8State);

  if (WRITE_OWNERSHIP_REPORT) {
    report.phase8SentinelState = expectedPhase8State;
    report.phase11SurfaceListConsolidationState = expectedPhase11State;
    report.phase12ComponentSingletonConsolidationState = expectedPhase12State;
    report.phase13InteractionOutlineConsolidationState = expectedPhase13State;
    report.phase14SkeletonPrimitiveConsolidationState = expectedPhase14State;
    report.phase17SurfaceFinalPassConsolidationState = expectedPhase17State;
    writeFileSync(OWNERSHIP_DRAIN_REPORT_PATH, `${JSON.stringify(report, null, 2)}\n`, "utf-8");
  }

  if (!WRITE_OWNERSHIP_REPORT && JSON.stringify(report.phase8SentinelState || null) !== JSON.stringify(expectedPhase8State)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase8SentinelState is stale`);
  }

  checkHistoricalPhase9State(report, failures);
  checkHistoricalPhase10State(report, failures);
  checkHistoricalPhase11State(report, failures);
  checkHistoricalPhase12State(report, failures);
  checkHistoricalPhase13State(report, failures);
  checkHistoricalPhase14State(report, failures);
  checkHistoricalPhase15State(report, failures);
  checkHistoricalPhase16State(report, failures);

  checkPhase17SurfaceFinalPassConsolidationState(expectedPhase17State, failures);
  if (
    !WRITE_OWNERSHIP_REPORT &&
    JSON.stringify(report.phase17SurfaceFinalPassConsolidationState || null) !== JSON.stringify(expectedPhase17State)
  ) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase17SurfaceFinalPassConsolidationState is stale`);
  }
}

function sentinelSourceRuleCount(filePath) {
  return existsSync(filePath) ? ruleCountForFile(filePath) : 0;
}

function sentinelStateFor(filePath, imports) {
  const importRecords = imports.filter((record) => record.resolvedPath === filePath);
  const sourceRuleCount = sentinelSourceRuleCount(filePath);
  const imported = importRecords.length > 0;
  const shippedRuleCount = imported ? sourceRuleCount : 0;
  return {
    path: relativePath(filePath),
    imported,
    layer: imported ? importRecords[0].layerName : null,
    sourceRuleCount,
    shippedRuleCount,
    shippedByteDebt: shippedRuleCount > 0 ? Buffer.byteLength(readText(filePath), "utf8") : 0,
    sentinelOnly: existsSync(filePath) && sentinelCssViolations(filePath).length === 0
  };
}

function buildPhase8SentinelState(imports) {
  return {
    utilitiesCompatHold: sentinelStateFor(COMPAT_HOLD_UTILITIES_PATH, imports),
    overridesCompat: sentinelStateFor(OVERRIDES_COMPAT_PATH, imports)
  };
}

if (sentinelFixtureIndex >= 0) {
  if (!SENTINEL_FIXTURE_PATH) {
    console.error(`ERROR: ${SENTINEL_FIXTURE_ARG} requires a CSS fixture path`);
    process.exit(1);
  }
  const fixtureFailures = [];
  checkSentinelOnlyFile(path.resolve(FRONTDOOR_ROOT, SENTINEL_FIXTURE_PATH), fixtureFailures, SENTINEL_FIXTURE_PATH);
  if (fixtureFailures.length > 0) {
    for (const failure of fixtureFailures) {
      console.error(`ERROR: ${failure}`);
    }
    process.exit(1);
  }
  console.log("portal css sentinel fixture: OK");
  process.exit(0);
}

function resolveFixturePath(filePath) {
  return path.isAbsolute(filePath) ? filePath : path.resolve(FRONTDOOR_ROOT, filePath);
}

if (duplicateBaselineFixtureIndex >= 0) {
  if (!DUPLICATE_BASELINE_FIXTURE_PATH || !DUPLICATE_BASELINE_DUPLICATES_PATH) {
    console.error(`ERROR: ${DUPLICATE_BASELINE_FIXTURE_ARG} requires baseline and duplicate JSON fixture paths`);
    process.exit(1);
  }
  const fixtureFailures = [];
  const baseline = JSON.parse(readText(resolveFixturePath(DUPLICATE_BASELINE_FIXTURE_PATH)));
  const duplicateFixture = JSON.parse(readText(resolveFixturePath(DUPLICATE_BASELINE_DUPLICATES_PATH)));
  const fixtureDuplicates = Array.isArray(duplicateFixture) ? duplicateFixture : duplicateFixture.duplicates || [];
  checkDuplicateBaselineEntries(
    fixtureDuplicates,
    baseline,
    fixtureFailures,
    path.basename(DUPLICATE_BASELINE_FIXTURE_PATH)
  );
  if (fixtureFailures.length > 0) {
    for (const failure of fixtureFailures) {
      console.error(`ERROR: ${failure}`);
    }
    process.exit(1);
  }
  console.log("portal css duplicate baseline fixture: OK");
  process.exit(0);
}

if (phase10AdditiveFixtureIndex >= 0) {
  if (!PHASE10_ADDITIVE_FIXTURE_PATH) {
    console.error(`ERROR: ${PHASE10_ADDITIVE_FIXTURE_ARG} requires a duplicate candidate JSON fixture path`);
    process.exit(1);
  }
  const fixtureFailures = [];
  const fixture = JSON.parse(readText(resolveFixturePath(PHASE10_ADDITIVE_FIXTURE_PATH)));
  const fixtureDuplicates = Array.isArray(fixture) ? fixture : fixture.duplicates || [];
  const candidates = analyzePhase10AdditiveCandidates(fixtureDuplicates, { duplicateKeys: fixture.baselineEntries || [] });
  const candidateByKey = new Map(candidates.map((candidate) => [candidate.key, candidate]));
  for (const expected of fixture.expectedCandidates || []) {
    const candidate = candidateByKey.get(expected.key);
    if (!candidate) {
      fixtureFailures.push(`missing Phase 10 candidate ${expected.key}`);
      continue;
    }
    if (expected.candidateStatus && candidate.candidateStatus !== expected.candidateStatus) {
      fixtureFailures.push(
        `Phase 10 candidate ${expected.key} expected ${expected.candidateStatus} but found ${candidate.candidateStatus}`
      );
    }
    if (expected.unsafeReason && candidate.unsafeReason !== expected.unsafeReason) {
      fixtureFailures.push(
        `Phase 10 candidate ${expected.key} expected unsafeReason ${expected.unsafeReason} but found ${candidate.unsafeReason || "none"}`
      );
    }
  }
  if (fixture.expectedState && JSON.stringify(fixture.phase10AdditiveConsolidationState || null) !== JSON.stringify(fixture.expectedState)) {
    fixtureFailures.push("phase10AdditiveConsolidationState is stale");
  }
  if (fixture.historicalPhase10State) {
    checkHistoricalPhase10State({ phase10AdditiveConsolidationState: fixture.historicalPhase10State }, fixtureFailures);
  }
  if (fixtureFailures.length > 0) {
    for (const failure of fixtureFailures) {
      console.error(`ERROR: ${failure}`);
    }
    process.exit(1);
  }
  console.log(
    `portal css phase10 additive fixture: OK (${candidates.filter((candidate) => candidate.candidateStatus === "safe").length} safe, ${candidates.filter((candidate) => candidate.candidateStatus === "deferred").length} deferred)`
  );
  process.exit(0);
}

if (phase11SurfaceFixtureIndex >= 0) {
  if (!PHASE11_SURFACE_FIXTURE_PATH) {
    console.error(`ERROR: ${PHASE11_SURFACE_FIXTURE_ARG} requires a duplicate candidate JSON fixture path`);
    process.exit(1);
  }
  const fixtureFailures = [];
  const fixture = JSON.parse(readText(resolveFixturePath(PHASE11_SURFACE_FIXTURE_PATH)));
  const fixtureDuplicates = Array.isArray(fixture) ? fixture : fixture.duplicates || [];
  const candidates = analyzePhase11SurfaceCandidates(fixtureDuplicates, { duplicateKeys: fixture.baselineEntries || [] });
  const candidateByKey = new Map(candidates.map((candidate) => [candidate.key, candidate]));
  for (const expected of fixture.expectedCandidates || []) {
    const candidate = candidateByKey.get(expected.key);
    if (!candidate) {
      fixtureFailures.push(`missing Phase 11 candidate ${expected.key}`);
      continue;
    }
    if (expected.candidateStatus && candidate.candidateStatus !== expected.candidateStatus) {
      fixtureFailures.push(
        `Phase 11 candidate ${expected.key} expected ${expected.candidateStatus} but found ${candidate.candidateStatus}`
      );
    }
    if (expected.unsafeReason && candidate.unsafeReason !== expected.unsafeReason) {
      fixtureFailures.push(
        `Phase 11 candidate ${expected.key} expected unsafeReason ${expected.unsafeReason} but found ${candidate.unsafeReason || "none"}`
      );
    }
    if (expected.unsafeReason && !PHASE11_UNSAFE_REASONS.has(expected.unsafeReason)) {
      fixtureFailures.push(`Phase 11 fixture expected unsupported unsafeReason ${expected.unsafeReason}`);
    }
  }
  if (
    fixture.expectedState &&
    JSON.stringify(fixture.phase11SurfaceListConsolidationState || null) !== JSON.stringify(fixture.expectedState)
  ) {
    fixtureFailures.push("phase11SurfaceListConsolidationState is stale");
  }
  if (fixtureFailures.length > 0) {
    for (const failure of fixtureFailures) {
      console.error(`ERROR: ${failure}`);
    }
    process.exit(1);
  }
  console.log(
    `portal css phase11 surface fixture: OK (${candidates.filter((candidate) => candidate.candidateStatus === "safe").length} safe, ${candidates.filter((candidate) => candidate.candidateStatus === "deferred").length} deferred)`
  );
  process.exit(0);
}

if (phase12ComponentFixtureIndex >= 0) {
  if (!PHASE12_COMPONENT_FIXTURE_PATH) {
    console.error(`ERROR: ${PHASE12_COMPONENT_FIXTURE_ARG} requires a duplicate candidate JSON fixture path`);
    process.exit(1);
  }
  const fixtureFailures = [];
  const fixture = JSON.parse(readText(resolveFixturePath(PHASE12_COMPONENT_FIXTURE_PATH)));
  const fixtureDuplicates = Array.isArray(fixture) ? fixture : fixture.duplicates || [];
  const candidates = analyzePhase12ComponentCandidates(fixtureDuplicates, { duplicateKeys: fixture.baselineEntries || [] });
  const candidateByKey = new Map(candidates.map((candidate) => [candidate.key, candidate]));
  for (const expected of fixture.expectedCandidates || []) {
    const candidate = candidateByKey.get(expected.key);
    if (!candidate) {
      fixtureFailures.push(`missing Phase 12 candidate ${expected.key}`);
      continue;
    }
    if (expected.candidateStatus && candidate.candidateStatus !== expected.candidateStatus) {
      fixtureFailures.push(
        `Phase 12 candidate ${expected.key} expected ${expected.candidateStatus} but found ${candidate.candidateStatus}`
      );
    }
    if (expected.unsafeReason && candidate.unsafeReason !== expected.unsafeReason) {
      fixtureFailures.push(
        `Phase 12 candidate ${expected.key} expected unsafeReason ${expected.unsafeReason} but found ${candidate.unsafeReason || "none"}`
      );
    }
    if (expected.unsafeReason && !PHASE12_UNSAFE_REASONS.has(expected.unsafeReason)) {
      fixtureFailures.push(`Phase 12 fixture expected unsupported unsafeReason ${expected.unsafeReason}`);
    }
  }
  if (
    fixture.expectedState &&
    JSON.stringify(fixture.phase12ComponentSingletonConsolidationState || null) !== JSON.stringify(fixture.expectedState)
  ) {
    fixtureFailures.push("phase12ComponentSingletonConsolidationState is stale");
  }
  if (fixtureFailures.length > 0) {
    for (const failure of fixtureFailures) {
      console.error(`ERROR: ${failure}`);
    }
    process.exit(1);
  }
  console.log(
    `portal css phase12 component fixture: OK (${candidates.filter((candidate) => candidate.candidateStatus === "safe").length} safe, ${candidates.filter((candidate) => candidate.candidateStatus === "deferred").length} deferred)`
  );
  process.exit(0);
}

if (phase13InteractionFixtureIndex >= 0) {
  if (!PHASE13_INTERACTION_FIXTURE_PATH) {
    console.error(`ERROR: ${PHASE13_INTERACTION_FIXTURE_ARG} requires a duplicate candidate JSON fixture path`);
    process.exit(1);
  }
  const fixtureFailures = [];
  const fixture = JSON.parse(readText(resolveFixturePath(PHASE13_INTERACTION_FIXTURE_PATH)));
  const fixtureDuplicates = Array.isArray(fixture) ? fixture : fixture.duplicates || [];
  const candidates = analyzePhase13InteractionCandidates(fixtureDuplicates, { duplicateKeys: fixture.baselineEntries || [] });
  const candidateByKey = new Map(candidates.map((candidate) => [candidate.key, candidate]));
  for (const expected of fixture.expectedCandidates || []) {
    const candidate = candidateByKey.get(expected.key);
    if (!candidate) {
      fixtureFailures.push(`missing Phase 13 candidate ${expected.key}`);
      continue;
    }
    if (expected.candidateStatus && candidate.candidateStatus !== expected.candidateStatus) {
      fixtureFailures.push(
        `Phase 13 candidate ${expected.key} expected ${expected.candidateStatus} but found ${candidate.candidateStatus}`
      );
    }
    if (expected.unsafeReason && candidate.unsafeReason !== expected.unsafeReason) {
      fixtureFailures.push(
        `Phase 13 candidate ${expected.key} expected unsafeReason ${expected.unsafeReason} but found ${candidate.unsafeReason || "none"}`
      );
    }
    if (expected.unsafeReason && !PHASE13_UNSAFE_REASONS.has(expected.unsafeReason)) {
      fixtureFailures.push(`Phase 13 fixture expected unsupported unsafeReason ${expected.unsafeReason}`);
    }
  }
  if (
    fixture.expectedState &&
    JSON.stringify(fixture.phase13InteractionOutlineConsolidationState || null) !== JSON.stringify(fixture.expectedState)
  ) {
    fixtureFailures.push("phase13InteractionOutlineConsolidationState is stale");
  }
  if (
    fixture.expectedPhase12State &&
    JSON.stringify(fixture.phase12ComponentSingletonConsolidationState || null) !== JSON.stringify(fixture.expectedPhase12State)
  ) {
    fixtureFailures.push("phase12ComponentSingletonConsolidationState immutable historical evidence drifted");
  }
  if (fixture.baselineEntries) {
    for (const entry of fixture.baselineEntries) {
      if (/selector-not-phase13-target/.test(String(entry.ownerReason || ""))) {
        fixtureFailures.push("selector-not-phase13-target must not overwrite live baseline ownerReason");
      }
    }
  }
  if (fixtureFailures.length > 0) {
    for (const failure of fixtureFailures) {
      console.error(`ERROR: ${failure}`);
    }
    process.exit(1);
  }
  console.log(
    `portal css phase13 interaction fixture: OK (${candidates.filter((candidate) => candidate.candidateStatus === "safe").length} safe, ${candidates.filter((candidate) => candidate.candidateStatus === "deferred").length} deferred)`
  );
  process.exit(0);
}

if (phase14SkeletonFixtureIndex >= 0) {
  if (!PHASE14_SKELETON_FIXTURE_PATH) {
    console.error(`ERROR: ${PHASE14_SKELETON_FIXTURE_ARG} requires a duplicate candidate JSON fixture path`);
    process.exit(1);
  }
  const fixtureFailures = [];
  const fixture = JSON.parse(readText(resolveFixturePath(PHASE14_SKELETON_FIXTURE_PATH)));
  const fixtureDuplicates = Array.isArray(fixture) ? fixture : fixture.duplicates || [];
  const candidates = analyzePhase14SkeletonCandidates(fixtureDuplicates, { duplicateKeys: fixture.baselineEntries || [] });
  const candidateByKey = new Map(candidates.map((candidate) => [candidate.key, candidate]));
  for (const expected of fixture.expectedCandidates || []) {
    const candidate = candidateByKey.get(expected.key);
    if (!candidate) {
      fixtureFailures.push(`missing Phase 14 candidate ${expected.key}`);
      continue;
    }
    if (expected.candidateStatus && candidate.candidateStatus !== expected.candidateStatus) {
      fixtureFailures.push(
        `Phase 14 candidate ${expected.key} expected ${expected.candidateStatus} but found ${candidate.candidateStatus}`
      );
    }
    if (expected.unsafeReason && candidate.unsafeReason !== expected.unsafeReason) {
      fixtureFailures.push(
        `Phase 14 candidate ${expected.key} expected unsafeReason ${expected.unsafeReason} but found ${candidate.unsafeReason || "none"}`
      );
    }
    if (expected.unsafeReason && !PHASE14_UNSAFE_REASONS.has(expected.unsafeReason)) {
      fixtureFailures.push(`Phase 14 fixture expected unsupported unsafeReason ${expected.unsafeReason}`);
    }
  }
  if (
    fixture.expectedState &&
    JSON.stringify(fixture.phase14SkeletonPrimitiveConsolidationState || null) !== JSON.stringify(fixture.expectedState)
  ) {
    fixtureFailures.push("phase14SkeletonPrimitiveConsolidationState is stale");
  }
  if (
    fixture.expectedPhase13State &&
    JSON.stringify(fixture.phase13InteractionOutlineConsolidationState || null) !== JSON.stringify(fixture.expectedPhase13State)
  ) {
    fixtureFailures.push("phase13InteractionOutlineConsolidationState immutable historical evidence drifted");
  }
  if (fixture.baselineEntries) {
    for (const entry of fixture.baselineEntries) {
      if (/selector-not-phase14-target/.test(String(entry.ownerReason || ""))) {
        fixtureFailures.push("selector-not-phase14-target must not overwrite live baseline ownerReason");
      }
    }
  }
  if (fixtureFailures.length > 0) {
    for (const failure of fixtureFailures) {
      console.error(`ERROR: ${failure}`);
    }
    process.exit(1);
  }
  console.log(
    `portal css phase14 skeleton fixture: OK (${candidates.filter((candidate) => candidate.candidateStatus === "safe").length} safe, ${candidates.filter((candidate) => candidate.candidateStatus === "deferred").length} deferred)`
  );
  process.exit(0);
}

if (phase15SurfaceLoadingFixtureIndex >= 0) {
  if (!PHASE15_SURFACE_LOADING_FIXTURE_PATH) {
    console.error(`ERROR: ${PHASE15_SURFACE_LOADING_FIXTURE_ARG} requires a duplicate candidate JSON fixture path`);
    process.exit(1);
  }
  const fixtureFailures = [];
  const fixture = JSON.parse(readText(resolveFixturePath(PHASE15_SURFACE_LOADING_FIXTURE_PATH)));
  const fixtureDuplicates = Array.isArray(fixture) ? fixture : fixture.duplicates || [];
  const candidates = analyzePhase15SurfaceLoadingCandidates(fixtureDuplicates, { duplicateKeys: fixture.baselineEntries || [] });
  const candidateByKey = new Map(candidates.map((candidate) => [candidate.key, candidate]));
  for (const expected of fixture.expectedCandidates || []) {
    const candidate = candidateByKey.get(expected.key);
    if (!candidate) {
      fixtureFailures.push(`missing Phase 15 candidate ${expected.key}`);
      continue;
    }
    if (expected.candidateStatus && candidate.candidateStatus !== expected.candidateStatus) {
      fixtureFailures.push(
        `Phase 15 candidate ${expected.key} expected ${expected.candidateStatus} but found ${candidate.candidateStatus}`
      );
    }
    if (expected.unsafeReason && candidate.unsafeReason !== expected.unsafeReason) {
      fixtureFailures.push(
        `Phase 15 candidate ${expected.key} expected unsafeReason ${expected.unsafeReason} but found ${candidate.unsafeReason || "none"}`
      );
    }
    if (expected.unsafeReason && !PHASE15_UNSAFE_REASONS.has(expected.unsafeReason)) {
      fixtureFailures.push(`Phase 15 fixture expected unsupported unsafeReason ${expected.unsafeReason}`);
    }
  }
  if (
    fixture.expectedState &&
    JSON.stringify(fixture.phase15SurfaceLoadingConsolidationState || null) !== JSON.stringify(fixture.expectedState)
  ) {
    fixtureFailures.push("phase15SurfaceLoadingConsolidationState is stale");
  }
  if (
    fixture.expectedPhase14State &&
    JSON.stringify(fixture.phase14SkeletonPrimitiveConsolidationState || null) !== JSON.stringify(fixture.expectedPhase14State)
  ) {
    fixtureFailures.push("phase14SkeletonPrimitiveConsolidationState immutable historical evidence drifted");
  }
  if (fixture.baselineEntries) {
    for (const entry of fixture.baselineEntries) {
      if (/selector-not-phase15-target/.test(String(entry.ownerReason || ""))) {
        fixtureFailures.push("selector-not-phase15-target must not overwrite live baseline ownerReason");
      }
    }
  }
  if (fixtureFailures.length > 0) {
    for (const failure of fixtureFailures) {
      console.error(`ERROR: ${failure}`);
    }
    process.exit(1);
  }
  console.log(
    `portal css phase15 surface-loading fixture: OK (${candidates.filter((candidate) => candidate.candidateStatus === "safe").length} safe, ${candidates.filter((candidate) => candidate.candidateStatus === "deferred").length} deferred)`
  );
  process.exit(0);
}

if (phase16ReviewSurfaceFixtureIndex >= 0) {
  if (!PHASE16_REVIEW_SURFACE_FIXTURE_PATH) {
    console.error(`ERROR: ${PHASE16_REVIEW_SURFACE_FIXTURE_ARG} requires a duplicate candidate JSON fixture path`);
    process.exit(1);
  }
  const fixtureFailures = [];
  const fixture = JSON.parse(readText(resolveFixturePath(PHASE16_REVIEW_SURFACE_FIXTURE_PATH)));
  const fixtureDuplicates = Array.isArray(fixture) ? fixture : fixture.duplicates || [];
  const candidates = analyzePhase16ReviewSurfaceCandidates(fixtureDuplicates, { duplicateKeys: fixture.baselineEntries || [] });
  const candidateByKey = new Map(candidates.map((candidate) => [candidate.key, candidate]));
  for (const expected of fixture.expectedCandidates || []) {
    const candidate = candidateByKey.get(expected.key);
    if (!candidate) {
      fixtureFailures.push(`missing Phase 16 candidate ${expected.key}`);
      continue;
    }
    if (expected.candidateStatus && candidate.candidateStatus !== expected.candidateStatus) {
      fixtureFailures.push(
        `Phase 16 candidate ${expected.key} expected ${expected.candidateStatus} but found ${candidate.candidateStatus}`
      );
    }
    if (expected.unsafeReason && candidate.unsafeReason !== expected.unsafeReason) {
      fixtureFailures.push(
        `Phase 16 candidate ${expected.key} expected unsafeReason ${expected.unsafeReason} but found ${candidate.unsafeReason || "none"}`
      );
    }
    if (expected.unsafeReason && !PHASE16_UNSAFE_REASONS.has(expected.unsafeReason)) {
      fixtureFailures.push(`Phase 16 fixture expected unsupported unsafeReason ${expected.unsafeReason}`);
    }
  }
  if (
    fixture.expectedState &&
    JSON.stringify(fixture.phase16ReviewSurfaceConsolidationState || null) !== JSON.stringify(fixture.expectedState)
  ) {
    fixtureFailures.push("phase16ReviewSurfaceConsolidationState is stale");
  }
  if (
    fixture.expectedPhase15State &&
    JSON.stringify(fixture.phase15SurfaceLoadingConsolidationState || null) !== JSON.stringify(fixture.expectedPhase15State)
  ) {
    fixtureFailures.push("phase15SurfaceLoadingConsolidationState immutable historical evidence drifted");
  }
  if (fixture.baselineEntries) {
    for (const entry of fixture.baselineEntries) {
      if (/selector-not-phase16-target/.test(String(entry.ownerReason || ""))) {
        fixtureFailures.push("selector-not-phase16-target must not overwrite live baseline ownerReason");
      }
    }
  }
  if (fixture.sourceShape) {
    const sourceShapeFailures = [];
    checkPhase16ReviewSurfaceSourceShape(sourceShapeFailures, {
      surfaceRoot: parseCss("phase16-review-surface.fixture.surface.css", fixture.sourceShape.surfaceCss || ""),
      operatorRoot: parseCss("phase16-review-surface.fixture.operator.css", fixture.sourceShape.operatorCss || ""),
      surfaceLabel: "phase16-review-surface fixture surface",
      operatorLabel: "phase16-review-surface fixture operator"
    });
    const expectedSourceShapeFailures = fixture.expectedSourceShapeFailures || [];
    for (const expected of expectedSourceShapeFailures) {
      if (!sourceShapeFailures.some((failure) => failure.includes(expected))) {
        fixtureFailures.push(`missing Phase 16 source-shape failure ${expected}`);
      }
    }
    for (const failure of sourceShapeFailures) {
      if (!expectedSourceShapeFailures.some((expected) => failure.includes(expected))) {
        fixtureFailures.push(`unexpected Phase 16 source-shape failure ${failure}`);
      }
    }
  }
  if (fixtureFailures.length > 0) {
    for (const failure of fixtureFailures) {
      console.error(`ERROR: ${failure}`);
    }
    process.exit(1);
  }
  console.log(
    `portal css phase16 review-surface fixture: OK (${candidates.filter((candidate) => candidate.candidateStatus === "safe").length} safe, ${candidates.filter((candidate) => candidate.candidateStatus === "deferred").length} deferred)`
  );
  process.exit(0);
}

if (phase17SurfaceFinalPassFixtureIndex >= 0) {
  if (!PHASE17_SURFACE_FINAL_PASS_FIXTURE_PATH) {
    console.error(`ERROR: ${PHASE17_SURFACE_FINAL_PASS_FIXTURE_ARG} requires a duplicate candidate JSON fixture path`);
    process.exit(1);
  }
  const fixtureFailures = [];
  const fixture = JSON.parse(readText(resolveFixturePath(PHASE17_SURFACE_FINAL_PASS_FIXTURE_PATH)));
  const fixtureDuplicates = Array.isArray(fixture) ? fixture : fixture.duplicates || [];
  const candidates = analyzePhase17SurfaceFinalPassCandidates(fixtureDuplicates, { duplicateKeys: fixture.baselineEntries || [] });
  const candidateByKey = new Map(candidates.map((candidate) => [candidate.key, candidate]));
  for (const expected of fixture.expectedCandidates || []) {
    const candidate = candidateByKey.get(expected.key);
    if (!candidate) {
      fixtureFailures.push(`missing Phase 17 candidate ${expected.key}`);
      continue;
    }
    if (expected.candidateStatus && candidate.candidateStatus !== expected.candidateStatus) {
      fixtureFailures.push(
        `Phase 17 candidate ${expected.key} expected ${expected.candidateStatus} but found ${candidate.candidateStatus}`
      );
    }
    if (expected.unsafeReason && candidate.unsafeReason !== expected.unsafeReason) {
      fixtureFailures.push(
        `Phase 17 candidate ${expected.key} expected unsafeReason ${expected.unsafeReason} but found ${candidate.unsafeReason || "none"}`
      );
    }
    if (expected.unsafeReason && !PHASE17_UNSAFE_REASONS.has(expected.unsafeReason)) {
      fixtureFailures.push(`Phase 17 fixture expected unsupported unsafeReason ${expected.unsafeReason}`);
    }
  }
  if (
    fixture.expectedState &&
    JSON.stringify(fixture.phase17SurfaceFinalPassConsolidationState || null) !== JSON.stringify(fixture.expectedState)
  ) {
    fixtureFailures.push("phase17SurfaceFinalPassConsolidationState is stale");
  }
  if (
    fixture.expectedPhase16State &&
    JSON.stringify(fixture.phase16ReviewSurfaceConsolidationState || null) !== JSON.stringify(fixture.expectedPhase16State)
  ) {
    fixtureFailures.push("phase16ReviewSurfaceConsolidationState immutable historical evidence drifted");
  }
  if (fixture.baselineEntries) {
    for (const entry of fixture.baselineEntries) {
      if (/selector-not-phase17-target/.test(String(entry.ownerReason || ""))) {
        fixtureFailures.push("selector-not-phase17-target must not overwrite live baseline ownerReason");
      }
    }
  }
  if (fixture.sourceShape) {
    const sourceShapeFailures = [];
    checkPhase17SurfaceFinalPassSourceShape(sourceShapeFailures, {
      surfaceRoot: parseCss("phase17-surface-final-pass.fixture.surface.css", fixture.sourceShape.surfaceCss || ""),
      operatorRoot: parseCss("phase17-surface-final-pass.fixture.operator.css", fixture.sourceShape.operatorCss || ""),
      shellRoot: parseCss("phase17-surface-final-pass.fixture.shell.css", fixture.sourceShape.shellCss || ""),
      surfaceLabel: "phase17-surface-final-pass fixture surface",
      operatorLabel: "phase17-surface-final-pass fixture operator",
      shellLabel: "phase17-surface-final-pass fixture shell"
    });
    const expectedSourceShapeFailures = fixture.expectedSourceShapeFailures || [];
    for (const expected of expectedSourceShapeFailures) {
      if (!sourceShapeFailures.some((failure) => failure.includes(expected))) {
        fixtureFailures.push(`missing Phase 17 source-shape failure ${expected}`);
      }
    }
    for (const failure of sourceShapeFailures) {
      if (!expectedSourceShapeFailures.some((expected) => failure.includes(expected))) {
        fixtureFailures.push(`unexpected Phase 17 source-shape failure ${failure}`);
      }
    }
  }
  if (fixtureFailures.length > 0) {
    for (const failure of fixtureFailures) {
      console.error(`ERROR: ${failure}`);
    }
    process.exit(1);
  }
  console.log(
    `portal css phase17 surface-final-pass fixture: OK (${candidates.filter((candidate) => candidate.candidateStatus === "safe").length} safe, ${candidates.filter((candidate) => candidate.candidateStatus === "deferred").length} deferred)`
  );
  process.exit(0);
}

const failures = [];
const cssFiles = listFiles(PORTAL_CSS_SOURCE_DIR, (entryPath) => entryPath.endsWith(".css"));
const records = collectRuleRecords(cssFiles, productionLayerBySourcePath());
const duplicates = collectDuplicateReport(records);

checkProductionLayerImports(failures);
checkPhase8Sentinels(failures);
checkSourceCssGovernance(cssFiles, failures);
checkImportantGovernance(cssFiles, failures);
checkOverrideOwners(cssFiles, failures);
checkSelectorFileBoundaries(cssFiles, failures);
checkDuplicateBaseline(duplicates, failures);
checkUtilityCoverage(failures);
checkLayerContract(failures);
checkPhase14SkeletonSourceShape(failures);
checkPhase15SurfaceLoadingSourceShape(failures);
checkPhase16ReviewSurfaceSourceShape(failures);
checkPhase17SurfaceFinalPassSourceShape(failures);
checkOwnershipDrainReport(failures, duplicates);
checkOverridesCompatNoNewIds(failures);
checkOverridesCompatNoImportant(failures);
checkPhase6SemanticHooksPresent(failures);

if (REPORT) {
  const hotspotCount = duplicates.filter((duplicate) => duplicate.hotspot).length;
  console.log(
    `portal css architecture report: ${records.length} selector records, ${duplicates.length} duplicate contexts, ${hotspotCount} hotspot duplicate contexts`
  );
  for (const duplicate of duplicates.filter((entry) => entry.hotspot).slice(0, 20)) {
    console.log(`hotspot ${duplicate.category}: ${duplicate.key}`);
  }
}

if (failures.length > 0) {
  for (const failure of failures) {
    console.error(`ERROR: ${failure}`);
  }
  process.exit(1);
}

console.log("portal css architecture: OK");
