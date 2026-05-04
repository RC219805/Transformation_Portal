export function createDeferredReviewSurfaceApi(host) {
  const {
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
  } = host;
  let artifactViewerObjectUrl = "";
  let artifactViewerPreviewPath = "";
  let artifactViewerPreviewSource = "";
  let artifactViewerPreviewRequestId = 0;
  let artifactViewerFallbackEventKey = "";
  let artifactViewerAbortController = null;
  let artifactViewerKeydownHandler = null;
  const ARTIFACT_VIEWER_FETCH_TIMEOUT_MS = 15000;
  const ADVISORY_CAPTION_CACHE_MAX_ENTRIES = 24;
  const advisoryCaptionPayloadCache = new Map();
  let advisoryCaptionCacheScope = "";
  let advisoryCaptionRenderRequestId = 0;

  function _artifactViewerEventMetadata(job, artifact, extra = {}) {
    const metadata = {
      job_id: _normalizeSelectedJobId(job?.id),
      media_kind: String(artifact?.media_kind || "file"),
      artifact_fingerprint: artifactFingerprint(artifact).toLowerCase(),
      viewer_mode: "modal",
    };
    Object.entries(extra).forEach(([key, value]) => {
      if (value === undefined || value === null || value === "") return;
      metadata[key] = value;
    });
    return metadata;
  }

  function _emitArtifactViewerFallback(context, fallbackReason) {
    if (!context?.job || !context?.artifact || !fallbackReason) return;
    const fallbackKey = [
      _normalizeSelectedJobId(context.job.id),
      _artifactRouteKey(context.artifact),
      fallbackReason,
    ]
      .filter(Boolean)
      .join(":");
    if (fallbackKey && fallbackKey === artifactViewerFallbackEventKey) return;
    artifactViewerFallbackEventKey = fallbackKey;
    void emitPortalEvent("artifact_viewer_fallback", {
      surface: "artifact_review",
      metadata: _artifactViewerEventMetadata(context.job, context.artifact, {
        fallback_reason: fallbackReason,
      }),
    });
  }

  function _revokeArtifactViewerObjectUrl() {
    if (!artifactViewerObjectUrl) return;
    URL.revokeObjectURL(artifactViewerObjectUrl);
    artifactViewerObjectUrl = "";
  }

  function _clearArtifactViewerPreviewCache() {
    artifactViewerPreviewRequestId += 1;
    artifactViewerPreviewPath = "";
    artifactViewerPreviewSource = "";
    _revokeArtifactViewerObjectUrl();
  }

  function _abortArtifactViewerPreview(reason = "superseded", controller = artifactViewerAbortController) {
    if (!controller) return;
    controller.__tpAbortReason = String(reason || "superseded");
    try {
      controller.abort();
    } catch (_abortErr) {}
    if (artifactViewerAbortController === controller) {
      artifactViewerAbortController = null;
    }
  }

  function _artifactViewerAbortReason(controller) {
    if (!controller) return "";
    if (controller.__tpAbortReason) {
      return String(controller.__tpAbortReason);
    }
    const signalReason = controller.signal?.reason;
    if (signalReason instanceof Error) {
      return signalReason.name || signalReason.message || "";
    }
    return signalReason ? String(signalReason) : "";
  }

  function _showArtifactViewerFallback(context, artifactName, fallbackOptions) {
    const url = context?.url || "";
    const isRetryable = Boolean(fallbackOptions?.retryable && url);
    const fallbackReason = isRetryable
      ? "inline_preview_failed"
      : url ? "inline_preview_unavailable" : "asset_url_unavailable";
    if (els.artifactViewerImage) {
      els.artifactViewerImage.classList.add("hidden");
      els.artifactViewerImage.removeAttribute("src");
      els.artifactViewerImage.style.transform = "scale(1)";
    }
    if (els.artifactViewerFallback) {
      els.artifactViewerFallback.classList.remove("hidden");
    }
    if (els.artifactViewerFallbackTitle) {
      els.artifactViewerFallbackTitle.textContent = isRetryable
        ? "Inline preview failed to load"
        : url ? "Inline preview unavailable" : "Artifact URL unavailable";
    }
    if (els.artifactViewerFallbackDetail) {
      els.artifactViewerFallbackDetail.textContent = isRetryable
        ? "The managed asset request did not complete. Retry the preview or continue reviewing the retained metadata and fingerprints."
        : url
          ? "This artifact stays reviewable through retained metadata, integrity fingerprints, and the managed raw asset link."
          : "The browser cannot resolve a managed asset URL for this artifact, so review stays pinned to the retained metadata above.";
    }
    _renderArtifactViewerRetry(isRetryable ? { context, artifactName } : null);
    _setArtifactViewerStatus(
      isRetryable
        ? `${artifactName} inline preview failed to load; retry available.`
        : url
          ? `${artifactName} is open with metadata fallback because an inline preview is unavailable.`
          : `${artifactName} is open with metadata fallback because the managed asset URL is unavailable.`
    );
    _emitArtifactViewerFallback(context, fallbackReason);
  }

  function _renderArtifactViewerRetry(target) {
    if (!els.artifactViewerFallback) return;
    const existing = els.artifactViewerFallback.querySelector("[data-ui='artifact-viewer-retry']");
    if (!target) {
      if (existing) existing.remove();
      return;
    }
    const button = existing || document.createElement("button");
    if (!existing) {
      button.type = "button";
      button.className = "artifact-viewer-retry-btn";
      button.dataset.ui = "artifact-viewer-retry";
      button.textContent = "Retry preview";
      els.artifactViewerFallback.appendChild(button);
    }
    button.onclick = () => {
      button.disabled = true;
      button.textContent = "Retrying…";
      Promise.resolve(_loadArtifactViewerInlinePreview(target.context, target.artifactName)).finally(() => {
        button.disabled = false;
        button.textContent = "Retry preview";
      });
    };
  }

  function _renderArtifactViewerInlineImage(src, zoomPercent) {
    if (!els.artifactViewerImage) return;
    els.artifactViewerImage.src = src;
    els.artifactViewerImage.classList.remove("hidden");
    els.artifactViewerImage.style.transform = `scale(${zoomPercent / 100})`;
    _renderArtifactViewerRetry(null);
    if (els.artifactViewerFallback) els.artifactViewerFallback.classList.add("hidden");
  }

  async function _loadArtifactViewerInlinePreview(context, artifactName) {
    const artifactPath = _artifactRouteKey(context.artifact);
    if (!context.url || !artifactPath) {
      _showArtifactViewerFallback(context, artifactName);
      return;
    }
    if (artifactViewerObjectUrl && artifactViewerPreviewPath === artifactPath && artifactViewerPreviewSource === context.url) {
      _renderArtifactViewerInlineImage(artifactViewerObjectUrl, context.zoomPercent);
      _setArtifactViewerStatus(`${artifactName} preview open at ${context.zoomPercent}% zoom.`);
      return;
    }

    const requestId = ++artifactViewerPreviewRequestId;
    artifactViewerPreviewPath = artifactPath;
    artifactViewerPreviewSource = context.url;
    _abortArtifactViewerPreview("superseded");
    const controller = typeof AbortController === "function" ? new AbortController() : null;
    if (controller) controller.__tpAbortReason = "";
    artifactViewerAbortController = controller;
    const timeoutId = controller
      ? setTimeout(() => {
          _abortArtifactViewerPreview("timeout", controller);
        }, ARTIFACT_VIEWER_FETCH_TIMEOUT_MS)
      : null;

    try {
      if (state.auth?.mode === "direct_debug" && typeof _buildAuthHeaders === "function") {
        const fetchOptions = {
          headers: _buildAuthHeaders({ Accept: artifactContentType(context.artifact) || "*/*" }, "GET"),
        };
        if (controller) fetchOptions.signal = controller.signal;
        const response = await fetch(context.url, fetchOptions);
        if (!response.ok) {
          throw new Error(`artifact_preview_${response.status}`);
        }
        const objectUrl = URL.createObjectURL(await response.blob());
        if (
          requestId !== artifactViewerPreviewRequestId ||
          !state.portalUi?.artifactViewer?.open ||
          state.portalUi.artifactViewer.artifactPath !== artifactPath
        ) {
          URL.revokeObjectURL(objectUrl);
          return;
        }
        _revokeArtifactViewerObjectUrl();
        artifactViewerObjectUrl = objectUrl;
        _renderArtifactViewerInlineImage(objectUrl, context.zoomPercent);
        _setArtifactViewerStatus(`${artifactName} preview open at ${context.zoomPercent}% zoom.`);
        return;
      }

      _renderArtifactViewerInlineImage(context.url, context.zoomPercent);
      _setArtifactViewerStatus(`${artifactName} preview open at ${context.zoomPercent}% zoom.`);
    } catch (err) {
      if (requestId !== artifactViewerPreviewRequestId) return;
      const abortReason = err?.name === "AbortError" ? _artifactViewerAbortReason(controller) : "";
      const retryable = err?.name !== "AbortError" || abortReason === "timeout";
      if (!retryable) return;
      const failureMessage =
        abortReason === "timeout"
          ? `request timed out after ${ARTIFACT_VIEWER_FETCH_TIMEOUT_MS / 1000}s`
          : err?.message || "network error";
      try {
        createToast(`Preview unavailable: ${failureMessage}`, "error");
      } catch (_toastErr) {}
      _showArtifactViewerFallback(context, artifactName, { retryable });
    } finally {
      if (timeoutId !== null) clearTimeout(timeoutId);
      if (artifactViewerAbortController === controller) {
        artifactViewerAbortController = null;
      }
    }
  }

  function getRunCardArtifact(job) {
    if (!job || !Array.isArray(job.artifacts)) return null;
    return (
      job.artifacts.find((artifact) => {
        const displayRole = String(artifactDisplayHint(artifact)?.role || "").trim().toLowerCase();
        const type = String(artifact.artifact_type || "").toLowerCase();
        const relPath = String(artifact.relative_path || artifact.path || "").toLowerCase();
        return displayRole === "run_card" || type === "run_card" || relPath.includes("run_card");
      }) || null
    );
  }

  function updateRunCardActions(job) {
    if (!els.runCardActions) return;
    const runCard = getRunCardArtifact(job);
    if (!runCard) {
      els.runCardActions.classList.add("hidden");
      els.runCardActions.classList.remove("flex");
      return;
    }
    const runCardPath = String(runCard.relative_path || runCard.path || "");
    const runCardUrl = buildArtifactUrl(job, runCard);
    const runCardSha = String(runCard.sha256 || "");
    if (els.viewRunCardBtn) {
      els.viewRunCardBtn.dataset.url = runCardUrl;
    }
    if (els.copyRunCardPathBtn) els.copyRunCardPathBtn.dataset.path = runCardPath;
    if (els.copyRunCardFingerprintBtn) {
      els.copyRunCardFingerprintBtn.dataset.fingerprint = runCardSha;
      els.copyRunCardFingerprintBtn.disabled = !runCardSha;
      els.copyRunCardFingerprintBtn.classList.toggle("opacity-50", !runCardSha);
      els.copyRunCardFingerprintBtn.classList.toggle("cursor-not-allowed", !runCardSha);
    }
    els.runCardActions.classList.remove("hidden");
    els.runCardActions.classList.add("flex");
  }

  function _isVlmCaptionSidecarArtifact(artifact) {
    if (!artifact) return false;
    const relPath = String(artifact.relative_path || artifact.path || "").toLowerCase();
    const type = String(artifact.artifact_type || "").toLowerCase();
    return type === "vlm_caption_sidecar" || relPath.endsWith(".vlm_captioning.sidecar.json");
  }

  function _isVlmCaptionRawArtifact(artifact) {
    if (!artifact) return false;
    const relPath = String(artifact.relative_path || artifact.path || "").toLowerCase();
    const type = String(artifact.artifact_type || "").toLowerCase();
    return type === "vlm_caption_raw" || relPath.endsWith(".vlm_captioning.raw.txt");
  }

  function _isVlmCaptionProxyArtifact(artifact) {
    if (!artifact) return false;
    const relPath = String(artifact.relative_path || artifact.path || "").toLowerCase();
    const type = String(artifact.artifact_type || "").toLowerCase();
    return type === "vlm_caption_proxy" || (relPath.includes("/captioning/") && /_proxy\.(png|jpe?g)$/i.test(relPath));
  }

  function _captionArtifactKind(artifact) {
    if (_isVlmCaptionSidecarArtifact(artifact)) return "sidecar";
    if (_isVlmCaptionRawArtifact(artifact)) return "raw";
    if (_isVlmCaptionProxyArtifact(artifact)) return "proxy";
    return "";
  }

  function _captionStemFromArtifact(artifact) {
    const relPath = String(artifact?.relative_path || artifact?.path || "");
    const fileName = relPath.split("/").pop() || "";
    return fileName
      .replace(/\.vlm_captioning\.sidecar\.json$/i, "")
      .replace(/\.vlm_captioning\.raw\.txt$/i, "")
      .replace(/_proxy\.(png|jpe?g)$/i, "");
  }

  function _captioningEvidenceArtifacts(job, artifact) {
    const artifacts = Array.isArray(job?.artifacts) ? job.artifacts : [];
    const selectedStem = _captionStemFromArtifact(artifact);
    const group = { sidecar: null, raw: null, proxy: null };
    const captionArtifacts = artifacts.filter((candidate) => Boolean(_captionArtifactKind(candidate)));
    const candidates = selectedStem
      ? captionArtifacts.filter((candidate) => _captionStemFromArtifact(candidate) === selectedStem)
      : captionArtifacts;
    candidates.forEach((candidate) => {
      const kind = _captionArtifactKind(candidate);
      if (kind && !group[kind]) group[kind] = candidate;
    });
    return group;
  }

  function _findVlmCaptionSidecar(job, artifact) {
    if (!artifact) return null;
    if (_isVlmCaptionSidecarArtifact(artifact)) return artifact;
    return _captioningEvidenceArtifacts(job, artifact).sidecar;
  }

  function _hasCaptioningEvidence(job, artifact) {
    const summary = normalizeRunSummary(job?.run_summary);
    const artifacts = _captioningEvidenceArtifacts(job, artifact);
    return Boolean(summary?.captioning_status || artifacts.sidecar || artifacts.raw || artifacts.proxy);
  }

  function _rememberAdvisoryCaptionCacheEntry(cacheKey, entry) {
    if (!cacheKey) return;
    if (advisoryCaptionPayloadCache.has(cacheKey)) {
      advisoryCaptionPayloadCache.delete(cacheKey);
    }
    advisoryCaptionPayloadCache.set(cacheKey, entry);
    while (advisoryCaptionPayloadCache.size > ADVISORY_CAPTION_CACHE_MAX_ENTRIES) {
      const oldestKey = advisoryCaptionPayloadCache.keys().next().value;
      if (!oldestKey) break;
      advisoryCaptionPayloadCache.delete(oldestKey);
    }
  }

  function _advisoryCaptionCredentialSignature(value) {
    const text = String(value || "");
    let hash = 0;
    for (let index = 0; index < text.length; index += 1) {
      hash = ((hash * 33) ^ text.charCodeAt(index)) >>> 0;
    }
    return `${text.length}:${hash.toString(36)}`;
  }

  function _resetAdvisoryCaptionCacheForAuth(requestHeaders) {
    const nextScope = _advisoryCaptionCredentialSignature([
      state.auth?.mode,
      requestHeaders?.Authorization ||
        requestHeaders?.authorization ||
        requestHeaders?.["x-api-key"] ||
        requestHeaders?.["X-API-Key"],
    ]
      .map((value) => String(value || "").trim())
      .filter(Boolean)
      .join(":"));
    if (nextScope === advisoryCaptionCacheScope) return;
    advisoryCaptionPayloadCache.clear();
    advisoryCaptionCacheScope = nextScope;
  }

  function _loadAdvisoryCaptionPayload(url) {
    const fetchUrl = String(url || "").trim();
    const requestHeaders = _buildAuthHeaders();
    _resetAdvisoryCaptionCacheForAuth(requestHeaders);
    const cacheKey = fetchUrl;
    if (!cacheKey) return Promise.reject(new Error("missing advisory caption URL"));
    const cached = advisoryCaptionPayloadCache.get(cacheKey);
    if (cached?.status === "fulfilled") return Promise.resolve(cached.payload);
    if (cached?.status === "pending") return cached.promise;

    const promise = fetch(fetchUrl, {
      headers: requestHeaders,
      credentials: "same-origin",
    })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((payload) => {
        _rememberAdvisoryCaptionCacheEntry(cacheKey, { status: "fulfilled", payload });
        return payload;
      })
      .catch((error) => {
        advisoryCaptionPayloadCache.delete(cacheKey);
        throw error;
      });

    _rememberAdvisoryCaptionCacheEntry(cacheKey, { status: "pending", promise });
    return promise;
  }

  function _captionListText(value) {
    if (Array.isArray(value)) return value.filter(Boolean).join(", ");
    return String(value || "").trim();
  }

  function _appendCaptionRow(container, label, value) {
    const text = _captionListText(value);
    if (!text) return;
    const row = document.createElement("div");
    row.className = "grid grid-cols-[88px_minmax(0,1fr)] gap-3 text-[12px] leading-5";
    const labelNode = document.createElement("dt");
    labelNode.className = "font-semibold text-slate-700 dark:text-slate-200";
    labelNode.textContent = label;
    const valueNode = document.createElement("dd");
    valueNode.className = "min-w-0 text-slate-600 dark:text-slate-300";
    valueNode.textContent = text;
    row.append(labelNode, valueNode);
    container.appendChild(row);
  }

  function _captionBooleanLabel(value) {
    if (value === true) return "Yes";
    if (value === false) return "No";
    return "";
  }

  function _appendCaptioningEvidenceMetric(container, label, value) {
    const text = value === 0 ? "0" : String(value || "").trim();
    if (!text) return;
    const item = document.createElement("span");
    item.className = "rounded border border-slate-200 px-2 py-1 text-[11px] text-slate-600 dark:border-slate-700 dark:text-slate-300";
    item.textContent = `${label}: ${text}`;
    container.appendChild(item);
  }

  function _appendCaptioningEvidenceLink(container, job, artifact, dataUi, label) {
    if (!artifact) return;
    const url = sanitizeManagedAssetUrl(buildArtifactUrl(job, artifact));
    if (!url) return;
    const link = document.createElement("a");
    link.className = "inline-flex items-center rounded border border-slate-200 px-2 py-1 text-[11px] font-semibold text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800";
    link.dataset.ui = dataUi;
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = label;
    link.title = artifactLabel(artifact);
    link.setAttribute("aria-label", `${label}: ${artifactLabel(artifact)}`);
    container.appendChild(link);
  }

  function _renderCaptioningEvidenceStrip(job, artifact) {
    if (!job || !els.artifactMetadataCard || !_hasCaptioningEvidence(job, artifact)) return;
    const summary = normalizeRunSummary(job.run_summary);
    const status = summary?.captioning_status || null;
    const artifacts = _captioningEvidenceArtifacts(job, artifact);
    const sidecarCount = Math.max(Number(status?.sidecar_count) || 0, artifacts.sidecar ? 1 : 0);
    const rawCount = Math.max(Number(status?.raw_count) || 0, artifacts.raw ? 1 : 0);
    const proxyCount = Math.max(Number(status?.proxy_count) || 0, artifacts.proxy ? 1 : 0);
    const modelLabel = [status?.model_role, status?.model_id || status?.model_path].filter(Boolean).join(" • ");

    const strip = document.createElement("section");
    strip.className = "mt-4 rounded border border-slate-200 bg-white/70 p-3 dark:border-slate-700 dark:bg-slate-900";
    strip.dataset.ui = "captioning-evidence-strip";
    strip.dataset.status = String(status?.status || "off");

    const metrics = document.createElement("div");
    metrics.className = "flex flex-wrap gap-2";
    _appendCaptioningEvidenceMetric(metrics, "Status", captioningRunStatusSummary(status));
    _appendCaptioningEvidenceMetric(metrics, "Sidecars", sidecarCount);
    _appendCaptioningEvidenceMetric(metrics, "Raw", rawCount);
    _appendCaptioningEvidenceMetric(metrics, "Proxy", proxyCount);
    _appendCaptioningEvidenceMetric(metrics, "Model", modelLabel);
    strip.appendChild(metrics);

    const links = document.createElement("div");
    links.className = "mt-3 flex flex-wrap gap-2";
    _appendCaptioningEvidenceLink(links, job, artifacts.sidecar, "captioning-sidecar-link", "Sidecar");
    _appendCaptioningEvidenceLink(links, job, artifacts.raw, "captioning-raw-link", "Raw");
    _appendCaptioningEvidenceLink(links, job, artifacts.proxy, "captioning-proxy-link", "Proxy");
    if (links.childNodes.length > 0) {
      links.dataset.ui = "captioning-evidence-link";
      strip.appendChild(links);
    }
    els.artifactMetadataCard.appendChild(strip);
  }

  function _renderAdvisoryCaptionUnavailable(panel) {
    panel.innerHTML = "";
    panel.dataset.status = "unavailable";
    const message = document.createElement("p");
    message.className = "text-[12px] font-semibold text-slate-800 dark:text-slate-100";
    message.textContent = "Advisory caption unavailable.";
    panel.appendChild(message);
  }

  function _renderAdvisoryCaptionPayload(panel, payload, sidecarArtifact) {
    panel.innerHTML = "";
    const root = payload?.vlm_captioning || {};
    if (!root || typeof root !== "object" || Array.isArray(root) || Object.keys(root).length === 0) {
      _renderAdvisoryCaptionUnavailable(panel);
      return;
    }
    panel.dataset.status = String(root.runtime_diagnostics?.status || (root.validated ? "validated" : "available"));
    const caption = root.caption || {};
    const title = document.createElement("p");
    title.className = "text-[12px] font-semibold text-slate-800 dark:text-slate-100";
    title.textContent = "Advisory VLM caption. Not used for quality gates.";
    panel.appendChild(title);

    const fields = document.createElement("dl");
    fields.className = "mt-3 space-y-1";
    _appendCaptionRow(fields, "Scene", caption.scene);
    _appendCaptionRow(fields, "Materials", caption.materials);
    _appendCaptionRow(fields, "Features", caption.features);
    _appendCaptionRow(fields, "Natural", caption.natural);
    _appendCaptionRow(fields, "Lighting", caption.lighting);
    _appendCaptionRow(fields, "Issues", caption.issues);
    _appendCaptionRow(fields, "Uncertain", caption.uncertain);
    _appendCaptionRow(fields, "Validated", _captionBooleanLabel(root.validated));
    _appendCaptionRow(fields, "Warnings", root.warnings);
    _appendCaptionRow(fields, "Model", root.model);
    _appendCaptionRow(fields, "Model role", root.model_role);
    _appendCaptionRow(fields, "Runtime status", root.runtime_diagnostics?.status);
    panel.appendChild(fields);

    const proxyPath = String(root.image_proxy?.proxy_path || "").trim();
    if (proxyPath) {
      const proxy = document.createElement("p");
      proxy.className = "mt-3 break-all text-[11px] text-slate-500 dark:text-slate-400";
      proxy.textContent = `Proxy: ${proxyPath}`;
      panel.appendChild(proxy);
    }

    const rawText = String(root.raw_model_text || "").trim();
    if (rawText) {
      const details = document.createElement("details");
      details.className = "mt-3 text-[12px] text-slate-600 dark:text-slate-300";
      const summary = document.createElement("summary");
      summary.className = "cursor-pointer font-semibold text-slate-700 dark:text-slate-200";
      summary.textContent = "Raw output";
      const pre = document.createElement("pre");
      pre.className = "mt-2 max-h-[220px] overflow-y-auto whitespace-pre-wrap break-words rounded border border-slate-200 bg-slate-50 p-3 text-[11px] leading-5 dark:border-slate-700 dark:bg-slate-900";
      pre.textContent = rawText;
      details.append(summary, pre);
      panel.appendChild(details);
    }

    const path = artifactLabel(sidecarArtifact);
    if (path) {
      const sidecarPath = document.createElement("p");
      sidecarPath.className = "mt-3 break-all text-[11px] text-slate-500 dark:text-slate-400";
      sidecarPath.textContent = path;
      panel.appendChild(sidecarPath);
    }
  }

  function _renderAdvisoryCaptionPanel(job, artifact) {
    const sidecar = _findVlmCaptionSidecar(job, artifact);
    if (!sidecar || !els.artifactMetadataCard) return;
    const requestId = ++advisoryCaptionRenderRequestId;
    const panel = document.createElement("section");
    panel.className = "mt-4 border-t border-slate-200 pt-4 dark:border-slate-700";
    panel.dataset.ui = "advisory-caption-panel";
    panel.textContent = "Loading advisory caption.";
    els.artifactMetadataCard.appendChild(panel);

    const url = sanitizeManagedAssetUrl(buildArtifactUrl(job, sidecar));
    if (!url) {
      _renderAdvisoryCaptionUnavailable(panel);
      return;
    }
    _loadAdvisoryCaptionPayload(url)
      .then((payload) => {
        if (requestId !== advisoryCaptionRenderRequestId || !panel.isConnected) return;
        _renderAdvisoryCaptionPayload(panel, payload, sidecar);
      })
      .catch(() => {
        if (requestId !== advisoryCaptionRenderRequestId || !panel.isConnected) return;
        _renderAdvisoryCaptionUnavailable(panel);
      });
  }

  function _renderArtifactMetadataCard(job, artifact) {
    if (!els.artifactMetadataCard) return;
    els.artifactMetadataCard.innerHTML = "";

    const title = document.createElement("p");
    title.className = "text-[12px] font-semibold text-slate-800 dark:text-slate-100";
    title.textContent = artifact
      ? artifactLabel(artifact)
      : "Select a completed run to bring the primary review artifact into focus here.";
    els.artifactMetadataCard.appendChild(title);

    const detail = document.createElement("p");
    detail.className = "mt-2 text-[12px] leading-6 text-slate-600 dark:text-slate-300";
    if (!job) {
      detail.textContent = "Preview, provenance, and next review actions will appear here after you choose a reviewable job.";
    } else if (!artifact) {
      detail.textContent = "This run has not indexed a reviewable artifact yet. Stay with the inspector for progress, transport, and freshness context.";
    } else {
      detail.textContent = `${artifactDisplayLabel(artifact)} • ${artifactContentType(artifact) || "binary"} • ${formatBytes(artifact.size_bytes)}.`;
    }
    els.artifactMetadataCard.appendChild(detail);
    _renderCaptioningEvidenceStrip(job, artifact);
    _renderAdvisoryCaptionPanel(job, artifact);
  }

  const REVIEW_STATUS_BUILDERS = Object.freeze({
    awaiting_job: () => ({
      visible: false,
      tone: "info",
      title: "Awaiting completed run",
      detail: "Select a job to review related warnings, completion state, and output readiness.",
      action: "Next action: use the selected run state, warning context, and freshness above to decide whether to recover or open review."
    }),
    partial_reviewable: ({ outcomeSummary, freshnessLabel }) => ({
      visible: true,
      tone: "warning",
      title: "Run partially completed",
      detail: outcomeSummary ? `${outcomeSummary}. Updated ${freshnessLabel}.` : "Some inputs failed, but outputs remain reviewable.",
      action: "Next action: open review for the retained outputs before rerunning failed inputs."
    }),
    failed_reviewable: ({ readableError, artifactCount, freshnessLabel }) => ({
      visible: true,
      tone: "warning",
      title: "Run failed after indexing reviewable outputs",
      detail: readableError || `${artifactCount} artifact${artifactCount === 1 ? "" : "s"} remain available for review. Updated ${freshnessLabel}.`,
      action: "Next action: open review for the retained outputs, then decide whether this run needs a retry."
    }),
    failed_unreviewable: ({ readableError }) => ({
      visible: true,
      tone: "error",
      title: "Run failed before outputs were ready",
      detail: readableError || "No reviewable outputs were indexed before the failure was reported.",
      action: "Next action: inspect the latest warning and failure context in Operate before retrying the run."
    }),
    canceled_reviewable: ({ artifactCount, freshnessLabel }) => ({
      visible: true,
      tone: "warning",
      title: "Run canceled after partial output capture",
      detail: `${artifactCount} artifact${artifactCount === 1 ? "" : "s"} remain available for review despite cancellation. Updated ${freshnessLabel}.`,
      action: "Next action: review the retained outputs before deciding whether to rerun the canceled work."
    }),
    canceled_unreviewable: () => ({
      visible: true,
      tone: "error",
      title: "Run canceled before review outputs were ready",
      detail: "Execution was canceled before reviewable outputs were indexed.",
      action: "Next action: reopen Build or restore the run context before dispatching again."
    }),
    offline_reviewable: ({ artifactCount }) => ({
      visible: true,
      tone: "warning",
      title: "Run is offline with reviewable outputs",
      detail: `${artifactCount} artifact${artifactCount === 1 ? "" : "s"} remain available, but live backend status is stale until connectivity is restored.`,
      action: "Next action: review the cached outputs while backend connectivity recovers."
    }),
    offline_unreviewable: () => ({
      visible: true,
      tone: "warning",
      title: "Run is offline",
      detail: "Live backend status is stale until connectivity is restored.",
      action: "Next action: restore backend connectivity before depending on this run state."
    }),
    transport_blocked: () => ({
      visible: true,
      tone: "warning",
      title: "Transport warning recorded",
      detail: "Authentication must be restored before live event transport can reconnect.",
      action: "Next action: restore authentication so live transport and freshness can recover."
    }),
    transport_warning: ({ visibleWarning }) => ({
      visible: true,
      tone: visibleWarning.tone === "error" ? "error" : "warning",
      title: "Transport warning recorded",
      detail: String(visibleWarning.detail || "Live transport reported an operator-visible warning."),
      action: "Next action: inspect the latest transport warning in Operate before continuing into review."
    }),
    in_progress: ({ artifactCount, freshnessLabel }) => ({
      visible: true,
      tone: "info",
      title: "Run still in progress",
      detail: artifactCount > 0
        ? `${artifactCount} artifact${artifactCount === 1 ? "" : "s"} already indexed. Updated ${freshnessLabel}.`
        : "Artifacts and provenance will populate here as outputs arrive.",
      action: artifactCount > 0
        ? "Next action: keep review open only if you need the early artifacts; Operate remains the primary live surface."
        : "Next action: stay in Operate until indexed outputs or a blocking warning arrives."
    }),
    ready: ({ artifact, artifactCount, outcomeSummary, freshnessLabel }) => ({
      visible: true,
      tone: "ready",
      title: artifact ? "Outputs ready for review" : "Run ready for review",
      detail: outcomeSummary
        ? `${outcomeSummary}. Updated ${freshnessLabel}.`
        : `${artifactCount} artifact${artifactCount === 1 ? "" : "s"} indexed and ready for operator review.`,
      action: "Next action: use the selected run state, warning context, and freshness above to decide whether to recover or open review."
    })
  });

  function _reviewStatusState(job, reviewableOutputs, visibleWarning) {
    if (!job) return "awaiting_job";
    if (job.state === "partial") return "partial_reviewable";
    if (job.state === "failed") return reviewableOutputs ? "failed_reviewable" : "failed_unreviewable";
    if (job.state === "canceled") return reviewableOutputs ? "canceled_reviewable" : "canceled_unreviewable";
    if (job.state === "offline") return reviewableOutputs ? "offline_reviewable" : "offline_unreviewable";
    if (job.reconnectBlocked) return "transport_blocked";
    if (visibleWarning) return "transport_warning";
    if (job.state === "running" || job.state === "queued") return "in_progress";
    return "ready";
  }

  function _reviewStatusSnapshot(job, artifact) {
    const summary = job ? normalizeRunSummary(job.run_summary) : {};
    const reviewableOutputs = _jobHasReviewableOutputs(job);
    const artifactCount = job && Array.isArray(job.artifacts) ? job.artifacts.length : 0;
    const readableError = job ? getReadableError(job.error) : "";
    const visibleWarning = _latestVisibleTransportWarning(job);
    const freshnessLabel = _jobFreshnessLabel(job);
    const outcomeSummary = typeof summary?.outcome_summary === "string" ? summary.outcome_summary.trim() : "";
    const stateToken = _reviewStatusState(job, reviewableOutputs, visibleWarning);
    const builder = REVIEW_STATUS_BUILDERS[stateToken] || REVIEW_STATUS_BUILDERS.ready;
    return {
      state: stateToken,
      ...builder({
        artifact,
        artifactCount,
        freshnessLabel,
        outcomeSummary,
        readableError,
        reviewableOutputs,
        summary,
        visibleWarning
      })
    };
  }

  function _renderReviewStatusBanner(job, artifact) {
    if (!els.reviewStatusBanner || !els.reviewStatusTitle || !els.reviewStatusDetail) return;
    const snapshot = _reviewStatusSnapshot(job, artifact);
    els.reviewStatusBanner.dataset.tone = snapshot.tone;
    els.reviewStatusBanner.dataset.reviewState = snapshot.state;
    if (!snapshot.visible) {
      els.reviewStatusBanner.classList.add("hidden");
      els.reviewStatusTitle.textContent = snapshot.title;
      els.reviewStatusDetail.textContent = snapshot.detail;
      if (els.reviewStatusAction) els.reviewStatusAction.textContent = snapshot.action;
      renderReviewStatusActions(job, artifact);
      return;
    }
    els.reviewStatusTitle.textContent = snapshot.title;
    els.reviewStatusDetail.textContent = snapshot.detail;
    if (els.reviewStatusAction) els.reviewStatusAction.textContent = snapshot.action;
    els.reviewStatusBanner.classList.remove("hidden");
    renderReviewStatusActions(job, artifact);
  }

  function _renderArtifactProvenance(job, artifact) {
    if (!els.reviewProvenanceGrid) return;

    if (!job) {
      els.reviewProvenanceGrid.classList.add("hidden");
      if (els.reviewProvenanceArtifactRole) els.reviewProvenanceArtifactRole.textContent = "Awaiting indexed output";
      if (els.reviewProvenanceRunState) els.reviewProvenanceRunState.textContent = "No job selected";
      if (els.reviewProvenancePath) {
        els.reviewProvenancePath.textContent = "Preview, metadata, and actions will appear here when outputs are indexed.";
        els.reviewProvenancePath.removeAttribute("title");
      }
      if (els.reviewProvenanceFingerprint) els.reviewProvenanceFingerprint.textContent = "Not reported";
      if (els.reviewProvenanceFreshness) els.reviewProvenanceFreshness.textContent = "No live telemetry";
      if (els.reviewProvenanceSource) els.reviewProvenanceSource.textContent = "Not reported";
      if (els.reviewProvenanceBatch) els.reviewProvenanceBatch.textContent = "Not reported";
      if (els.reviewProvenanceCaptioning) {
        els.reviewProvenanceCaptioning.textContent = "FastVLM: Not requested";
        els.reviewProvenanceCaptioning.dataset.status = "off";
      }
      return;
    }

    const summary = normalizeRunSummary(job.run_summary);
    const artifactDescriptor = artifact
      ? `${artifactDisplayLabel(artifact)} • ${artifactContentType(artifact) || "binary"} • ${formatBytes(artifact.size_bytes)}`
      : "Awaiting indexed artifact";
    const relativePath = artifact ? artifactLabel(artifact) : "Artifacts will appear here when the selected run indexes outputs.";
    const freshnessLabel = _jobFreshnessLabel(job);
    const runStateLabel = `${titleCaseToken(job.state, "Unknown")} • ${titleCaseToken(job.pipeline, "Unknown")}`;
    const sourceLabel = summary?.source || titleCaseToken(job.pipeline, "Not reported");
    const batchLabel = summary?.batch_id || "Not reported";
    const captioningStatus = summary?.captioning_status || null;
    const captioningLabel = captioningRunStatusSummary(captioningStatus);

    if (els.reviewProvenanceArtifactRole) els.reviewProvenanceArtifactRole.textContent = artifactDescriptor;
    if (els.reviewProvenanceRunState) els.reviewProvenanceRunState.textContent = runStateLabel;
    if (els.reviewProvenancePath) {
      els.reviewProvenancePath.textContent = relativePath;
      els.reviewProvenancePath.title = relativePath;
    }
    if (els.reviewProvenanceFingerprint) els.reviewProvenanceFingerprint.textContent = _artifactFingerprintLabel(artifact);
    if (els.reviewProvenanceFreshness) els.reviewProvenanceFreshness.textContent = freshnessLabel;
    if (els.reviewProvenanceSource) els.reviewProvenanceSource.textContent = sourceLabel;
    if (els.reviewProvenanceBatch) els.reviewProvenanceBatch.textContent = batchLabel;
    if (els.reviewProvenanceCaptioning) {
      els.reviewProvenanceCaptioning.textContent = captioningLabel;
      els.reviewProvenanceCaptioning.dataset.status = String(captioningStatus?.status || "off");
    }
    els.reviewProvenanceGrid.classList.remove("hidden");
  }

  function _renderReviewCompareSummary(primaryArtifact, compareArtifact, compareEnabled) {
    if (!els.reviewCompareSummary || !els.reviewCompareTitle || !els.reviewCompareDetail) return;
    const compareCopy = _compareSurfaceCopy(primaryArtifact, compareArtifact, compareEnabled);
    if (!primaryArtifact || !compareArtifact) {
      els.reviewCompareSummary.classList.add("hidden");
      els.reviewCompareTitle.textContent = compareCopy.summaryTitle;
      els.reviewCompareDetail.textContent = compareCopy.summaryDetail;
      return;
    }

    els.reviewCompareTitle.textContent = compareCopy.summaryTitle;
    els.reviewCompareDetail.textContent = compareCopy.summaryDetail;
    els.reviewCompareSummary.classList.remove("hidden");
  }

  function _artifactEmptyStateCopy(job) {
    if (!job) {
      return {
        tone: "neutral",
        title: "Select a completed run",
        detail: "Choose a reviewable job to load preview, provenance, and compare context here.",
        action: "Next action: inspect the selected run in Operate or wait for indexed outputs before reopening review."
      };
    }
    if (job.state === "running" || job.state === "queued") {
      return {
        tone: "info",
        title: "Outputs are still arriving",
        detail: "This run has not indexed reviewable artifacts yet. Stay on the inspector for live progress and freshness updates.",
        action: "Next action: keep the run in Operate until indexed outputs appear or a blocking warning arrives."
      };
    }
    if (job.state === "failed" || job.state === "canceled") {
      return {
        tone: "warning",
        title: "No reviewable outputs indexed",
        detail: "This run ended before artifacts were available. Inspect the run status and transport warnings above for recovery context.",
        action: "Next action: inspect the selected run in Operate or decide whether the failed run should be retried."
      };
    }
    return {
      tone: "neutral",
      title: "No indexed artifacts yet",
      detail: "Artifacts will appear here when the selected run finishes indexing its review outputs.",
      action: "Next action: inspect the selected run in Operate or wait for indexed outputs before reopening review."
    };
  }

  function renderArtifactPanel() {
    const jobsLoading = _isJobsHydrationPending();
    _toggleSurfaceSkeleton(els.artifactsShell, els.artifactShellContent, els.artifactSkeletonState, jobsLoading);
    if (jobsLoading) {
      _resetArtifactActionButtons();
      _renderReviewStatusBanner(null, null);
      _renderArtifactProvenance(null, null);
      _renderReviewCompareSummary(null, null, false);
      if (els.artifactMeta) els.artifactMeta.textContent = "Hydrating artifacts";
      renderConsoleContextRibbon();
      return;
    }

    if (!els.artifactMeta || !els.artifactThumbnailRail) return;
    els.artifactThumbnailRail.setAttribute("role", "listbox");
    els.artifactThumbnailRail.setAttribute("aria-label", "Artifact thumbnails");
    const selected = state.jobs.find((item) => item.id === state.selectedJobId);
    const artifacts = Array.isArray(selected?.artifacts) ? rankArtifactsForDisplay(selected.artifacts) : [];

    if (!selected) {
      _resetArtifactActionButtons();
      const emptyCopy = _artifactEmptyStateCopy(null);
      _setSurfaceEmptyState(els.emptyArtifactState, els.emptyArtifactTitle, els.emptyArtifactDetail, emptyCopy);
      if (els.emptyArtifactAction) els.emptyArtifactAction.textContent = emptyCopy.action || "";
      els.artifactMeta.textContent = "No job selected";
      els.artifactThumbnailRail.innerHTML = "";
      if (els.artifactSelectionTitle) els.artifactSelectionTitle.textContent = "No artifact selected";
      if (els.artifactSelectionMeta) els.artifactSelectionMeta.textContent = "Preview, provenance, and actions will appear here after you choose a reviewable run.";
      if (els.artifactCompareBtn) {
        els.artifactCompareBtn.classList.add("hidden");
        els.artifactCompareBtn.setAttribute("aria-pressed", "false");
        els.artifactCompareBtn.removeAttribute("aria-controls");
      }
      if (els.artifactPreviewSoloImage) {
        els.artifactPreviewSoloImage.classList.add("hidden");
        els.artifactPreviewSoloImage.removeAttribute("src");
      }
      if (els.artifactPreviewImage) {
        els.artifactPreviewImage.classList.add("hidden");
        els.artifactPreviewImage.removeAttribute("src");
      }
      if (els.artifactCompareImage) {
        els.artifactCompareImage.classList.add("hidden");
        els.artifactCompareImage.removeAttribute("src");
      }
      if (els.artifactCompareStage) {
        els.artifactCompareStage.classList.add("hidden");
        els.artifactCompareStage.setAttribute("aria-hidden", "true");
      }
      _renderArtifactMetadataCard(null, null);
      _renderReviewStatusBanner(null, null);
      _renderArtifactProvenance(null, null);
      _renderReviewCompareSummary(null, null, false);
      updateRunCardActions(null);
      if (els.emptyArtifactState) els.emptyArtifactState.style.display = "block";
      renderConsoleContextRibbon();
      _syncConsoleRoute(true);
      return;
    }

    const errorText = getReadableError(selected.error);
    els.artifactMeta.textContent = `${artifacts.length} artifacts${errorText ? ` • ${errorText}` : ""}`;

    if (artifacts.length === 0) {
      _resetArtifactActionButtons();
      const emptyCopy = _artifactEmptyStateCopy(selected);
      _setSurfaceEmptyState(els.emptyArtifactState, els.emptyArtifactTitle, els.emptyArtifactDetail, emptyCopy);
      if (els.emptyArtifactAction) els.emptyArtifactAction.textContent = emptyCopy.action || "";
      els.artifactThumbnailRail.innerHTML = "";
      if (els.artifactSelectionTitle) els.artifactSelectionTitle.textContent = "No artifact selected";
      if (els.artifactSelectionMeta) els.artifactSelectionMeta.textContent = "Review surfaces will populate here when the selected run indexes outputs.";
      if (els.artifactCompareBtn) {
        els.artifactCompareBtn.classList.add("hidden");
        els.artifactCompareBtn.setAttribute("aria-pressed", "false");
        els.artifactCompareBtn.removeAttribute("aria-controls");
      }
      if (els.artifactPreviewSoloImage) {
        els.artifactPreviewSoloImage.classList.add("hidden");
        els.artifactPreviewSoloImage.removeAttribute("src");
      }
      if (els.artifactPreviewImage) {
        els.artifactPreviewImage.classList.add("hidden");
        els.artifactPreviewImage.removeAttribute("src");
      }
      if (els.artifactCompareImage) {
        els.artifactCompareImage.classList.add("hidden");
        els.artifactCompareImage.removeAttribute("src");
      }
      if (els.artifactCompareStage) {
        els.artifactCompareStage.classList.add("hidden");
        els.artifactCompareStage.setAttribute("aria-hidden", "true");
      }
      _renderArtifactMetadataCard(selected, null);
      _renderReviewStatusBanner(selected, null);
      _renderArtifactProvenance(selected, null);
      _renderReviewCompareSummary(null, null, false);
      updateRunCardActions(selected);
      if (els.emptyArtifactState) els.emptyArtifactState.style.display = "block";
      renderConsoleContextRibbon();
      _syncConsoleRoute(true);
      return;
    }

    if (els.emptyArtifactState) els.emptyArtifactState.style.display = "none";
    const selectedArtifact = _selectedArtifactForJob(selected);
    const compareCandidate = findCompareArtifact(selectedArtifact, artifacts);
    const captioningEvidenceVisible = _hasCaptioningEvidence(selected, selectedArtifact);
    const compareEnabled = Boolean(compareCandidate) && Boolean(state.artifactUi.compareByJob[String(selected.id || "")]);

    if (els.artifactCompareBtn) {
      if (compareCandidate) {
        els.artifactCompareBtn.classList.remove("hidden");
        els.artifactCompareBtn.textContent = compareEnabled ? "Single View" : "Compare";
        els.artifactCompareBtn.setAttribute("aria-pressed", compareEnabled ? "true" : "false");
        els.artifactCompareBtn.setAttribute("aria-controls", "artifactCompareStage");
      } else {
        els.artifactCompareBtn.classList.add("hidden");
        els.artifactCompareBtn.setAttribute("aria-pressed", "false");
        els.artifactCompareBtn.removeAttribute("aria-controls");
      }
    }

    if (els.artifactSelectionTitle) {
      els.artifactSelectionTitle.textContent = selectedArtifact ? artifactLabel(selectedArtifact) : "No artifact selected";
    }
    if (els.artifactSelectionMeta) {
      els.artifactSelectionMeta.textContent = selectedArtifact
        ? `${artifactDisplayLabel(selectedArtifact)} • ${artifactContentType(selectedArtifact) || "binary"} • ${formatBytes(selectedArtifact.size_bytes)}`
        : "Preview, metadata, and actions will appear here when outputs are indexed.";
    }
    _renderReviewStatusBanner(selected, selectedArtifact);
    _renderArtifactProvenance(selected, selectedArtifact);
    _renderReviewCompareSummary(selectedArtifact, compareCandidate, compareEnabled);

    if (els.openArtifactBtn) {
      const openUrl = selectedArtifact ? buildArtifactUrl(selected, selectedArtifact) : "";
      els.openArtifactBtn.textContent = _artifactViewerEnabled() ? "Inspect" : "Open";
      els.openArtifactBtn.disabled = !openUrl;
      els.openArtifactBtn.dataset.url = openUrl;
    }
    if (els.downloadArtifactBtn) {
      const downloadUrl = selectedArtifact ? buildArtifactUrl(selected, selectedArtifact) : "";
      els.downloadArtifactBtn.disabled = !downloadUrl;
      els.downloadArtifactBtn.dataset.url = downloadUrl;
      els.downloadArtifactBtn.dataset.filename = selectedArtifact ? artifactNameParts(selectedArtifact).fileName : "";
    }
    if (els.copyArtifactPathBtn) {
      els.copyArtifactPathBtn.disabled = !selectedArtifact;
      els.copyArtifactPathBtn.dataset.path = selectedArtifact ? artifactLabel(selectedArtifact) : "";
    }
    if (els.copyArtifactFingerprintBtn) {
      const fingerprint = selectedArtifact ? artifactFingerprint(selectedArtifact) : "";
      els.copyArtifactFingerprintBtn.disabled = !fingerprint;
      els.copyArtifactFingerprintBtn.dataset.fingerprint = fingerprint;
    }

    if (els.artifactCompareStage) {
      els.artifactCompareStage.classList.toggle("hidden", !compareEnabled);
      els.artifactCompareStage.setAttribute("aria-hidden", compareEnabled ? "false" : "true");
    }
    if (els.artifactPreviewSoloImage) {
      els.artifactPreviewSoloImage.classList.toggle("hidden", compareEnabled || !artifactIsPreviewable(selectedArtifact));
    }
    if (els.artifactMetadataCard) {
      els.artifactMetadataCard.classList.toggle("hidden", artifactIsPreviewable(selectedArtifact) && !captioningEvidenceVisible);
    }
    if (compareEnabled && selectedArtifact && compareCandidate) {
      if (els.artifactPreviewImage) {
        els.artifactPreviewImage.src = buildArtifactUrl(selected, selectedArtifact);
        els.artifactPreviewImage.classList.remove("hidden");
      }
      if (els.artifactPreviewPrimaryCaption) els.artifactPreviewPrimaryCaption.textContent = artifactLabel(selectedArtifact);
      if (els.artifactCompareImage) {
        els.artifactCompareImage.src = buildArtifactUrl(selected, compareCandidate);
        els.artifactCompareImage.classList.remove("hidden");
      }
      if (els.artifactCompareCaption) els.artifactCompareCaption.textContent = artifactLabel(compareCandidate);
      if (captioningEvidenceVisible) _renderArtifactMetadataCard(selected, selectedArtifact);
    } else if (artifactIsPreviewable(selectedArtifact)) {
      if (els.artifactPreviewSoloImage) {
        els.artifactPreviewSoloImage.src = buildArtifactUrl(selected, selectedArtifact);
        els.artifactPreviewSoloImage.classList.remove("hidden");
      }
      if (els.artifactPreviewImage) {
        els.artifactPreviewImage.classList.add("hidden");
        els.artifactPreviewImage.removeAttribute("src");
      }
      if (els.artifactCompareImage) {
        els.artifactCompareImage.classList.add("hidden");
        els.artifactCompareImage.removeAttribute("src");
      }
      if (captioningEvidenceVisible) _renderArtifactMetadataCard(selected, selectedArtifact);
    } else {
      if (els.artifactPreviewSoloImage) {
        els.artifactPreviewSoloImage.classList.add("hidden");
        els.artifactPreviewSoloImage.removeAttribute("src");
      }
      if (els.artifactPreviewImage) {
        els.artifactPreviewImage.classList.add("hidden");
        els.artifactPreviewImage.removeAttribute("src");
      }
      if (els.artifactCompareImage) {
        els.artifactCompareImage.classList.add("hidden");
        els.artifactCompareImage.removeAttribute("src");
      }
      _renderArtifactMetadataCard(selected, selectedArtifact);
    }

    const fragment = document.createDocumentFragment();
    artifacts.forEach((artifact) => {
      const button = document.createElement("button");
      const active = selectedArtifact && artifact.path === selectedArtifact.path;
      button.type = "button";
      button.dataset.artifactPath = _artifactRouteKey(artifact);
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", active ? "true" : "false");
      button.tabIndex = active ? 0 : -1;
      button.className = active
        ? "rounded-2xl border border-cyan-300 dark:border-cyan-900/60 bg-cyan-50/90 dark:bg-cyan-900/20 p-3 text-left shadow-sm transition-colors"
        : "rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-900/50 p-3 text-left hover:bg-white/90 dark:hover:bg-slate-800/80 transition-colors";

      if (artifactIsPreviewable(artifact)) {
        const thumb = document.createElement("img");
        thumb.alt = artifactLabel(artifact);
        thumb.src = buildArtifactUrl(selected, artifact);
        thumb.className = "h-24 w-full rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-900/60 object-cover";
        button.appendChild(thumb);
      } else {
        const placeholder = document.createElement("div");
        placeholder.className = "flex h-24 items-center justify-center rounded-xl border border-dashed border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/70 text-[11px] font-bold uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400";
        placeholder.textContent = artifactDisplayLabel(artifact);
        button.appendChild(placeholder);
      }

      const title = document.createElement("p");
      title.className = "mt-3 text-[11px] font-semibold text-slate-800 dark:text-slate-100 truncate";
      title.textContent = artifactNameParts(artifact).fileName;
      button.appendChild(title);

      const meta = document.createElement("p");
      meta.className = "mt-1 text-[10px] font-mono text-slate-500 dark:text-slate-400 truncate";
      meta.textContent = `${artifactDisplayLabel(artifact)} • ${formatBytes(artifact.size_bytes)}`;
      button.appendChild(meta);

      fragment.appendChild(button);
    });
    els.artifactThumbnailRail.innerHTML = "";
    els.artifactThumbnailRail.appendChild(fragment);
    updateRunCardActions(selected);
    renderConsoleContextRibbon();
    _syncConsoleRoute(true);
  }

  function _artifactViewerContext() {
    const viewerState = state.portalUi?.artifactViewer || {};
    const requestedJobId = _normalizeSelectedJobId(viewerState.jobId || state.selectedJobId);
    const job = requestedJobId ? _findJobById(requestedJobId) : null;
    const artifacts = Array.isArray(job?.artifacts) ? rankArtifactsForDisplay(job.artifacts) : [];
    if (!job || artifacts.length === 0) {
      return {
        job,
        artifacts,
        artifact: null,
        index: -1,
        url: "",
        inlinePreview: false,
        zoomPercent: 100
      };
    }
    const requestedPath = _normalizeArtifactRoutePath(viewerState.artifactPath || "");
    const artifact =
      artifacts.find((candidate) => _artifactRouteKey(candidate) === requestedPath) ||
      _selectedArtifactForJob(job) ||
      artifacts[0] ||
      null;
    const index = artifact ? artifacts.findIndex((candidate) => candidate.path === artifact.path) : -1;
    const url = artifact ? sanitizeManagedAssetUrl(buildArtifactUrl(job, artifact)) : "";
    return {
      job,
      artifacts,
      artifact,
      index,
      url,
      inlinePreview: Boolean(artifact && artifactIsPreviewable(artifact) && url),
      zoomPercent: clamp(Number(viewerState.zoomPercent || 100), 50, 250)
    };
  }

  function _setArtifactViewerStatus(message) {
    if (!els.artifactViewerStatus) return;
    els.artifactViewerStatus.textContent = String(message || "").trim() || "Artifact viewer is closed.";
  }

  function _setArtifactViewerZoom(nextZoom) {
    state.portalUi.artifactViewer.zoomPercent = clamp(Number(nextZoom || 100), 50, 250);
    renderArtifactViewer();
  }

  function _navigateArtifactViewerSelection(direction) {
    const context = _artifactViewerContext();
    if (!context.job || !context.artifact) return false;
    const nextIndex = context.index + Number(direction || 0);
    if (nextIndex < 0 || nextIndex >= context.artifacts.length) return false;
    const nextArtifact = context.artifacts[nextIndex];
    const nextPath = _artifactRouteKey(nextArtifact);
    state.portalUi.artifactViewer.jobId = _normalizeSelectedJobId(context.job.id);
    state.portalUi.artifactViewer.artifactPath = nextPath;
    state.portalUi.artifactViewer.zoomPercent = 100;
    _rememberArtifactSelection(context.job.id, nextPath);
    renderReviewSurfaces();
    return true;
  }

  function renderArtifactViewer() {
    if (!els.artifactViewerModal || !els.artifactViewerPanel) return;
    const shouldShow = Boolean(state.portalUi?.artifactViewer?.open) && _artifactViewerEnabled();
    els.artifactViewerModal.classList.toggle("hidden", !shouldShow);
    els.artifactViewerModal.classList.toggle("flex", shouldShow);
    els.artifactViewerModal.setAttribute("aria-hidden", shouldShow ? "false" : "true");
    els.artifactViewerModal.dataset.overlayOpen = shouldShow ? "true" : "false";
    if (!shouldShow) {
      if (!_artifactViewerEnabled() && Boolean(state.portalUi?.artifactViewer?.open)) {
        _closeArtifactViewer(false);
        return;
      }
      _clearArtifactViewerPreviewCache();
      if (els.artifactViewerImage) {
        els.artifactViewerImage.classList.add("hidden");
        els.artifactViewerImage.removeAttribute("src");
        els.artifactViewerImage.style.transform = "scale(1)";
      }
      if (els.artifactViewerFallback) els.artifactViewerFallback.classList.add("hidden");
      _setArtifactViewerStatus("Artifact viewer is closed.");
      return;
    }

    const context = _artifactViewerContext();
    if (!context.job || !context.artifact) {
      _closeArtifactViewer(false);
      return;
    }

    const { artifact, index, artifacts, inlinePreview, job, url, zoomPercent } = context;
    const relPath = artifactLabel(artifact);
    const fingerprint = artifactFingerprint(artifact);
    const artifactName = artifactNameParts(artifact).fileName;
    if (els.artifactViewerTitle) els.artifactViewerTitle.textContent = artifactName;
    if (els.artifactViewerMeta) {
      els.artifactViewerMeta.textContent = `${artifactDisplayLabel(artifact)} • ${artifactContentType(artifact) || "binary"} • ${formatBytes(artifact.size_bytes)}`;
    }
    if (els.artifactViewerPath) els.artifactViewerPath.textContent = relPath;
    if (els.artifactViewerFingerprint) els.artifactViewerFingerprint.textContent = _artifactFingerprintLabel(artifact);
    if (els.artifactViewerZoomValue) {
      els.artifactViewerZoomValue.textContent = inlinePreview ? `${zoomPercent}% zoom` : "Inline preview unavailable";
    }
    if (els.artifactViewerPrevBtn) els.artifactViewerPrevBtn.disabled = index <= 0;
    if (els.artifactViewerNextBtn) els.artifactViewerNextBtn.disabled = index >= artifacts.length - 1;
    if (els.artifactViewerZoomOutBtn) els.artifactViewerZoomOutBtn.disabled = !inlinePreview;
    if (els.artifactViewerZoomInBtn) els.artifactViewerZoomInBtn.disabled = !inlinePreview;
    if (els.artifactViewerResetZoomBtn) els.artifactViewerResetZoomBtn.disabled = !inlinePreview;
    if (els.artifactViewerOpenRawBtn) {
      els.artifactViewerOpenRawBtn.disabled = !url;
      els.artifactViewerOpenRawBtn.dataset.url = url;
    }
    if (els.artifactViewerCopyPathBtn) {
      els.artifactViewerCopyPathBtn.disabled = !relPath;
      els.artifactViewerCopyPathBtn.dataset.path = relPath;
    }
    if (els.artifactViewerCopyFingerprintBtn) {
      els.artifactViewerCopyFingerprintBtn.disabled = !fingerprint;
      els.artifactViewerCopyFingerprintBtn.dataset.fingerprint = fingerprint;
    }

    if (inlinePreview && els.artifactViewerImage) {
      const artifactPath = _artifactRouteKey(artifact);
      if (artifactViewerPreviewPath && artifactViewerPreviewPath !== artifactPath) {
        _clearArtifactViewerPreviewCache();
      }
      if (els.artifactViewerImage) {
        // Reset the previous handler so closures over stale artifactPath don't stack and leak.
        els.artifactViewerImage.onerror = null;
        els.artifactViewerImage.onerror = () => {
          const activeContext = _artifactViewerContext();
          if (!activeContext.artifact) return;
          if (_artifactRouteKey(activeContext.artifact) !== artifactPath) return;
          _showArtifactViewerFallback(activeContext, artifactName, { retryable: true });
        };
      }
      if (state.auth?.mode !== "direct_debug") {
        _renderArtifactViewerInlineImage(url, zoomPercent);
        _setArtifactViewerStatus(`${artifactName} preview open at ${zoomPercent}% zoom.`);
      } else {
        if (els.artifactViewerImage) {
          els.artifactViewerImage.classList.add("hidden");
          els.artifactViewerImage.removeAttribute("src");
          els.artifactViewerImage.style.transform = `scale(${zoomPercent / 100})`;
        }
        if (els.artifactViewerFallback) els.artifactViewerFallback.classList.add("hidden");
        _setArtifactViewerStatus(`${artifactName} preview loading.`);
        void _loadArtifactViewerInlinePreview(context, artifactName);
      }
      return;
    }

    _clearArtifactViewerPreviewCache();
    _showArtifactViewerFallback(context, artifactName);
    void job;
  }

  function _setArtifactViewerBackgroundInert(inert) {
    const main = document.getElementById("main-content");
    if (!main) return;
    if (inert) main.setAttribute("aria-hidden", "true");
    else main.removeAttribute("aria-hidden");
    try {
      main.inert = Boolean(inert);
    } catch (_) {}
  }

  function _handleArtifactViewerKeydown(event) {
    if (!state.portalUi?.artifactViewer?.open) return;
    if (event.key === "Escape") {
      event.preventDefault();
      _closeArtifactViewer(true);
      return;
    }
    if (event.key !== "Tab" || !els.artifactViewerPanel) return;
    const nodes = els.artifactViewerPanel.querySelectorAll(
      'button:not([disabled]),[href],input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])'
    );
    const visible = Array.prototype.filter.call(nodes, (node) => node.offsetParent !== null);
    if (!visible.length) return;
    const first = visible[0];
    const last = visible[visible.length - 1];
    const active = document.activeElement;
    const outside = !els.artifactViewerPanel.contains(active);
    if (event.shiftKey && (active === first || outside)) {
      last.focus();
      event.preventDefault();
    } else if (!event.shiftKey && (active === last || outside)) {
      first.focus();
      event.preventDefault();
    }
  }

  function _closeArtifactViewer(restoreFocus = true) {
    state.portalUi.artifactViewer.open = false;
    _abortArtifactViewerPreview("close");
    if (artifactViewerKeydownHandler && typeof document !== "undefined") {
      document.removeEventListener("keydown", artifactViewerKeydownHandler, true);
      artifactViewerKeydownHandler = null;
    }
    _setArtifactViewerBackgroundInert(false);
    _renderArtifactViewerRetry(null);
    renderArtifactViewer();
    if (restoreFocus) {
      _restoreOverlayFocus();
    }
  }

  function _openArtifactViewer(job, artifact, trigger = document.activeElement) {
    if (!_artifactViewerEnabled() || !els.artifactViewerModal) return false;
    if (!job || !artifact) {
      createToast("No artifact is available for this selection.", "info");
      return false;
    }
    state.portalUi.artifactViewer.open = true;
    state.portalUi.artifactViewer.jobId = _normalizeSelectedJobId(job.id);
    state.portalUi.artifactViewer.artifactPath = _artifactRouteKey(artifact);
    state.portalUi.artifactViewer.zoomPercent = 100;
    artifactViewerFallbackEventKey = "";
    _rememberOverlayTrigger(trigger);
    _setArtifactViewerBackgroundInert(true);
    if (!artifactViewerKeydownHandler && typeof document !== "undefined") {
      artifactViewerKeydownHandler = _handleArtifactViewerKeydown;
      document.addEventListener("keydown", artifactViewerKeydownHandler, true);
    }
    renderArtifactViewer();
    if (els.closeArtifactViewerBtn) els.closeArtifactViewerBtn.focus();
    void emitPortalEvent("artifact_viewer_opened", {
      surface: "artifact_review",
      metadata: _artifactViewerEventMetadata(job, artifact),
    });
    return true;
  }

  return {
    renderArtifactPanel,
    renderArtifactViewer,
    _artifactViewerContext,
    _setArtifactViewerZoom,
    _navigateArtifactViewerSelection,
    _closeArtifactViewer,
    _openArtifactViewer
  };
}
