export function defaultPortalBootstrapPayload() {
  return {
    authMode: "managed_unavailable",
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

export function isManagedAuthMode(authState) {
  return Boolean(authState && authState.mode !== "direct_debug");
}

export function isManagedUnavailableMode(authState) {
  return Boolean(authState && authState.mode === "managed_unavailable");
}
