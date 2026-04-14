export function createDeferredReviewSurfaceApi(host) {
  const {
    state,
    els,
    clamp,
    normalizeRunSummary,
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

  function _showArtifactViewerFallback(url, artifactName) {
    if (els.artifactViewerImage) {
      els.artifactViewerImage.classList.add("hidden");
      els.artifactViewerImage.removeAttribute("src");
      els.artifactViewerImage.style.transform = "scale(1)";
    }
    if (els.artifactViewerFallback) {
      els.artifactViewerFallback.classList.remove("hidden");
    }
    if (els.artifactViewerFallbackTitle) {
      els.artifactViewerFallbackTitle.textContent = url ? "Inline preview unavailable" : "Artifact URL unavailable";
    }
    if (els.artifactViewerFallbackDetail) {
      els.artifactViewerFallbackDetail.textContent = url
        ? "This artifact stays reviewable through retained metadata, integrity fingerprints, and the managed raw asset link."
        : "The browser cannot resolve a managed asset URL for this artifact, so review stays pinned to the retained metadata above.";
    }
    _setArtifactViewerStatus(
      url
        ? `${artifactName} is open with metadata fallback because an inline preview is unavailable.`
        : `${artifactName} is open with metadata fallback because the managed asset URL is unavailable.`
    );
  }

  function _renderArtifactViewerInlineImage(src, zoomPercent) {
    if (!els.artifactViewerImage) return;
    els.artifactViewerImage.src = src;
    els.artifactViewerImage.classList.remove("hidden");
    els.artifactViewerImage.style.transform = `scale(${zoomPercent / 100})`;
    if (els.artifactViewerFallback) els.artifactViewerFallback.classList.add("hidden");
  }

  async function _loadArtifactViewerInlinePreview(context, artifactName) {
    const artifactPath = _artifactRouteKey(context.artifact);
    if (!context.url || !artifactPath) {
      _showArtifactViewerFallback(context.url, artifactName);
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

    try {
      if (state.auth?.mode === "direct_debug" && typeof _buildAuthHeaders === "function") {
        const response = await fetch(context.url, {
          headers: _buildAuthHeaders({ Accept: artifactContentType(context.artifact) || "*/*" }, "GET"),
        });
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
    } catch {
      if (requestId !== artifactViewerPreviewRequestId) return;
      _showArtifactViewerFallback(context.url, artifactName);
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
  }

  function _reviewStatusSnapshot(job, artifact) {
    if (!job) {
      return {
        visible: false,
        tone: "info",
        title: "Awaiting completed run",
        detail: "Select a job to review related warnings, completion state, and output readiness.",
        action: "Next action: use the selected run state, warning context, and freshness above to decide whether to recover or open review."
      };
    }

    const summary = normalizeRunSummary(job.run_summary);
    const reviewableOutputs = _jobHasReviewableOutputs(job);
    const artifactCount = Array.isArray(job.artifacts) ? job.artifacts.length : 0;
    const readableError = getReadableError(job.error);
    const visibleWarning = _latestVisibleTransportWarning(job);
    const freshnessLabel = _jobFreshnessLabel(job);
    const outcomeSummary = typeof summary?.outcome_summary === "string" ? summary.outcome_summary.trim() : "";

    if (job.state === "partial") {
      return {
        visible: true,
        tone: "warning",
        title: "Run partially completed",
        detail: outcomeSummary ? `${outcomeSummary}. Updated ${freshnessLabel}.` : "Some inputs failed, but outputs remain reviewable.",
        action: "Next action: open review for the retained outputs before rerunning failed inputs."
      };
    }

    if (job.state === "failed") {
      return {
        visible: true,
        tone: reviewableOutputs ? "warning" : "error",
        title: reviewableOutputs ? "Run failed after indexing reviewable outputs" : "Run failed before outputs were ready",
        detail:
          readableError ||
          (reviewableOutputs
            ? `${artifactCount} artifact${artifactCount === 1 ? "" : "s"} remain available for review. Updated ${freshnessLabel}.`
            : "No reviewable outputs were indexed before the failure was reported."),
        action: reviewableOutputs
          ? "Next action: open review for the retained outputs, then decide whether this run needs a retry."
          : "Next action: inspect the latest warning and failure context in Operate before retrying the run."
      };
    }

    if (job.state === "canceled") {
      return {
        visible: true,
        tone: reviewableOutputs ? "warning" : "error",
        title: reviewableOutputs ? "Run canceled after partial output capture" : "Run canceled before review outputs were ready",
        detail: reviewableOutputs
          ? `${artifactCount} artifact${artifactCount === 1 ? "" : "s"} remain available for review despite cancellation. Updated ${freshnessLabel}.`
          : "Execution was canceled before reviewable outputs were indexed.",
        action: reviewableOutputs
          ? "Next action: review the retained outputs before deciding whether to rerun the canceled work."
          : "Next action: reopen Build or restore the run context before dispatching again."
      };
    }

    if (job.state === "offline") {
      return {
        visible: true,
        tone: "warning",
        title: reviewableOutputs ? "Run is offline with reviewable outputs" : "Run is offline",
        detail: reviewableOutputs
          ? `${artifactCount} artifact${artifactCount === 1 ? "" : "s"} remain available, but live backend status is stale until connectivity is restored.`
          : "Live backend status is stale until connectivity is restored.",
        action: reviewableOutputs
          ? "Next action: review the cached outputs while backend connectivity recovers."
          : "Next action: restore backend connectivity before depending on this run state."
      };
    }

    if (job.reconnectBlocked) {
      return {
        visible: true,
        tone: "warning",
        title: "Transport warning recorded",
        detail: "Authentication must be restored before live event transport can reconnect.",
        action: "Next action: restore authentication so live transport and freshness can recover."
      };
    }

    if (visibleWarning) {
      return {
        visible: true,
        tone: visibleWarning.tone === "error" ? "error" : "warning",
        title: "Transport warning recorded",
        detail: String(visibleWarning.detail || "Live transport reported an operator-visible warning."),
        action: "Next action: inspect the latest transport warning in Operate before continuing into review."
      };
    }

    if (job.state === "running" || job.state === "queued") {
      return {
        visible: true,
        tone: "info",
        title: "Run still in progress",
        detail: artifactCount > 0
          ? `${artifactCount} artifact${artifactCount === 1 ? "" : "s"} already indexed. Updated ${freshnessLabel}.`
          : "Artifacts and provenance will populate here as outputs arrive.",
        action: artifactCount > 0
          ? "Next action: keep review open only if you need the early artifacts; Operate remains the primary live surface."
          : "Next action: stay in Operate until indexed outputs or a blocking warning arrives."
      };
    }

    return {
      visible: true,
      tone: "ready",
      title: artifact ? "Outputs ready for review" : "Run ready for review",
      detail: outcomeSummary
        ? `${outcomeSummary}. Updated ${freshnessLabel}.`
        : `${artifactCount} artifact${artifactCount === 1 ? "" : "s"} indexed and ready for operator review.`,
      action: "Next action: use the selected run state, warning context, and freshness above to decide whether to recover or open review."
    };
  }

  function _renderReviewStatusBanner(job, artifact) {
    if (!els.reviewStatusBanner || !els.reviewStatusTitle || !els.reviewStatusDetail) return;
    const snapshot = _reviewStatusSnapshot(job, artifact);
    els.reviewStatusBanner.dataset.tone = snapshot.tone;
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
      els.artifactMetadataCard.classList.toggle("hidden", artifactIsPreviewable(selectedArtifact));
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
      if (!_artifactViewerEnabled()) {
        state.portalUi.artifactViewer.open = false;
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
      state.portalUi.artifactViewer.open = false;
      renderArtifactViewer();
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
        els.artifactViewerImage.onerror = () => {
          const activeContext = _artifactViewerContext();
          if (!activeContext.artifact) return;
          if (_artifactRouteKey(activeContext.artifact) !== artifactPath) return;
          _showArtifactViewerFallback(url, artifactName);
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
    _showArtifactViewerFallback(url, artifactName);
    void job;
  }

  function _closeArtifactViewer(restoreFocus = true) {
    state.portalUi.artifactViewer.open = false;
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
    _rememberOverlayTrigger(trigger);
    renderArtifactViewer();
    if (els.closeArtifactViewerBtn) els.closeArtifactViewerBtn.focus();
    void emitPortalEvent("artifact_viewer_opened", {
      surface: "artifact_viewer_modal",
      metadata: {
        job_id: String(job.id || ""),
        media_kind: String(artifact.media_kind || "file"),
        pipeline: String(job.pipeline || "")
      }
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
