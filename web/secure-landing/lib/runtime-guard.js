const SUPPORTED_NODE_RANGE = ">=22.23.2 <23";
const RECOVERY_GUIDANCE =
  "Install a patched Node 22 release (>=22.23.2), reinstall or rebuild front-door dependencies under that runtime, then retry.";

const NATIVE_DEPENDENCIES = Object.freeze([
  "better-sqlite3",
  "argon2"
]);

export function supportedNodeRange() {
  return SUPPORTED_NODE_RANGE;
}

export function recoveryGuidance() {
  return RECOVERY_GUIDANCE;
}

export function isSupportedNodeVersion(version = process.versions.node) {
  const match = /^v?(\d+)\.(\d+)\.(\d+)$/.exec(String(version || ""));
  if (!match) return false;
  const [major, minor, patch] = match.slice(1).map(Number);
  return [major, minor, patch].every(Number.isSafeInteger) &&
    major === 22 && (minor > 23 || (minor === 23 && patch >= 2));
}

export function formatUnsupportedNodeMessage(version = process.versions.node) {
  return `secure-landing-frontdoor requires Node ${SUPPORTED_NODE_RANGE}. Current runtime: ${version}. ${RECOVERY_GUIDANCE}`;
}

export function ensureSupportedNodeVersion(version = process.versions.node) {
  if (!isSupportedNodeVersion(version)) {
    throw new Error(formatUnsupportedNodeMessage(version));
  }
}

function classifyNativeDependencyFailure(error) {
  const message = error instanceof Error ? error.message : String(error);
  if (/compiled against a different Node\.js version/i.test(message) || /NODE_MODULE_VERSION/i.test(message)) {
    return "native addon ABI mismatch";
  }
  return message.replace(/\s+/g, " ").trim() || "native dependency failed to load";
}

export function formatNativeDependencyFailureMessage(
  dependencyName,
  error,
  version = process.versions.node
) {
  const reason = classifyNativeDependencyFailure(error);
  return `secure-landing-frontdoor could not load native dependency "${dependencyName}" under Node ${version}: ${reason}. ${RECOVERY_GUIDANCE}`;
}

export async function ensureNativeDependenciesLoaded(version = process.versions.node) {
  for (const dependencyName of NATIVE_DEPENDENCIES) {
    try {
      await import(dependencyName);
    } catch (error) {
      throw new Error(formatNativeDependencyFailureMessage(dependencyName, error, version), {
        cause: error
      });
    }
  }
}

export async function ensureSupportedRuntime(version = process.versions.node) {
  ensureSupportedNodeVersion(version);
  await ensureNativeDependenciesLoaded(version);
}
