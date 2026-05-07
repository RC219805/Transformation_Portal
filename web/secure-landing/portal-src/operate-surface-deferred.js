// Operate surface deferred bundle.
//
// Carved out of portal-src/portal.template.js. This module owns the
// pure render path for the Operate surface (job queue + selected-job
// inspector). All state, DOM references, telemetry, formatting, and
// shared helpers are passed in via the host argument so the bundle
// never reads from globals.
//
// The data plane (SSE handlers, hydration, dispatch, selection, timeline
// reconciliation) intentionally stays in portal.template.js because
// those callers fire from many places and several are also consumed by
// the review surface.

export function createDeferredOperateSurfaceApi(host) {
    const {
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
    } = host;

    let queueRenderScheduled = false;
    let queuedReviewSurfaceRefresh = false;

    function _queueEmptyStateCopy() {
        if (state.jobsLoadStatus === 'offline' || (!state.backendOk && state.jobs.length === 0)) {
            return {
                tone: 'warning',
                title: 'Queue unavailable',
                detail: 'Backend connectivity is offline. Restore the managed backend to recover recent runs and live transport state.',
                action: 'Next action: restore backend connectivity so recent runs and live transport can recover.'
            };
        }
        if (state.jobsLoadStatus === 'error') {
            return {
                tone: 'error',
                title: 'Queue recovery needs attention',
                detail: 'Recent jobs could not be recovered. Refresh the workspace after backend health returns to continue.',
                action: 'Next action: confirm backend health, then refresh the workspace to rehydrate recent runs.'
            };
        }
        return {
            tone: 'neutral',
            title: 'No runs yet',
            detail: 'Dispatch a run from Build or wait for recovery to repopulate recent operator activity.',
            action: 'Next action: open Build to prepare the next run or restore backend connectivity to recover recent history.'
        };
    }

    function _selectedJobRecoverySnapshot(job) {
        if (!job) {
            return {
                title: 'Select or dispatch a run',
                detail: 'Use Queue to inspect a recent run or open Build to create the next governed dispatch.'
            };
        }

        const artifactCount = Array.isArray(job.artifacts) ? job.artifacts.length : 0;
        const visibleWarning = _latestVisibleTransportWarning(job);

        if (job.reconnectBlocked) {
            return {
                title: 'Restore authentication',
                detail: 'Authentication must be restored before live transport can reconnect and freshness can recover.'
            };
        }

        if (visibleWarning) {
            return {
                title: 'Review the latest warning',
                detail: String(visibleWarning.detail || 'Live transport reported an operator-visible warning.')
            };
        }

        if (job.state === 'failed' || job.state === 'canceled') {
            return artifactCount > 0
                ? {
                    title: 'Open review for retained outputs',
                    detail: 'Review the indexed outputs before deciding whether this run needs a retry.'
                }
                : {
                    title: 'Inspect failure before rerun',
                    detail: 'No reviewable outputs were indexed. Use the run state and warning context above before retrying.'
                };
        }

        if (job.state === 'offline') {
            return artifactCount > 0
                ? {
                    title: 'Review cached outputs while backend recovers',
                    detail: 'Outputs remain available, but live backend state is stale until connectivity returns.'
                }
                : {
                    title: 'Restore backend connectivity',
                    detail: 'Backend connectivity must recover before this run state can be trusted again.'
                };
        }

        if (job.state === 'running' || job.state === 'queued') {
            return artifactCount > 0
                ? {
                    title: 'Stay with the live run',
                    detail: 'Fresh artifacts are already indexing. Keep Operate open until review context stabilizes.'
                }
                : {
                    title: 'Wait for indexed outputs',
                    detail: 'Use the selected run state, warning context, and freshness above to decide whether to recover or open review.'
                };
        }

        if (job.state === 'partial') {
            return {
                title: 'Open review for partial outputs',
                detail: 'Review the indexed artifacts and warning context before deciding whether to rerun the failed inputs.'
            };
        }

        return artifactCount > 0
            ? {
                title: 'Open review',
                detail: 'Outputs and provenance are ready. Move to Review when you want compare and artifact actions.'
            }
            : {
                title: 'Wait for indexed outputs',
                detail: 'Use the selected run state, warning context, and freshness above to decide whether to recover or open review.'
            };
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
            if (els.selectedJobRecoveryTitle) els.selectedJobRecoveryTitle.textContent = 'Recovering selected run context';
            if (els.selectedJobRecoveryDetail) {
                els.selectedJobRecoveryDetail.textContent = 'The latest warning, artifact freshness, and recovery action will repopulate here when queue hydration finishes.';
            }
            renderSelectedJobRecoveryActions(null);
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
            if (els.selectedJobRecoveryTitle) els.selectedJobRecoveryTitle.textContent = 'Select or dispatch a run';
            if (els.selectedJobRecoveryDetail) {
                els.selectedJobRecoveryDetail.textContent = 'Use Queue to inspect a recent run or open Build to create the next governed dispatch.';
            }
            renderSelectedJobRecoveryActions(null);
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
        const transportLabel = formatTransportLabel(selected);
        const elapsedLabel = formatDuration(Number(selected.createdAt || 0), Number(selected.finishedAt || Date.now()));
        const displayState = _displayJobState(selected);
        const latestWarning = Array.isArray(selected.transportWarnings) && selected.transportWarnings.length > 0
            ? selected.transportWarnings[selected.transportWarnings.length - 1]
            : null;
        const visibleAlert = latestWarning && latestWarning.tone !== 'info' ? latestWarning : null;

        if (els.selectedJobStateBadge) els.selectedJobStateBadge.textContent = titleCaseToken(displayState, 'Unknown');
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
            els.selectedJobMetaLine.textContent = `${titleCaseToken(displayState, 'Unknown')} • ${transportLabel} • ${elapsedLabel}`;
        }
        if (els.selectedJobFreshness) {
            els.selectedJobFreshness.textContent = _jobFreshnessLabel(selected);
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
                        : `Operators can now read ${titleCaseToken(displayState, 'job')} state at a glance: ${artifactCount} artifact${artifactCount === 1 ? '' : 's'} indexed, ${transportLabel} transport, ${elapsedLabel}.`;
        }
        const recovery = _selectedJobRecoverySnapshot(selected);
        if (els.selectedJobRecoveryTitle) els.selectedJobRecoveryTitle.textContent = recovery.title;
        if (els.selectedJobRecoveryDetail) els.selectedJobRecoveryDetail.textContent = recovery.detail;
        renderSelectedJobRecoveryActions(selected);
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
            const emptyCopy = _queueEmptyStateCopy();
            _setSurfaceEmptyState(els.emptyQueueState, els.emptyQueueTitle, els.emptyQueueDetail, emptyCopy);
            if (els.emptyQueueAction) els.emptyQueueAction.textContent = emptyCopy.action || '';
            if (els.emptyQueueState) els.emptyQueueState.style.display = 'flex';
            if (els.queueStatusSummary) {
                els.queueStatusSummary.textContent = state.jobsLoadStatus === 'ready'
                    ? 'Dispatch a run to populate live queue and inspector context.'
                    : state.jobsLoadStatus === 'offline'
                        ? 'Queue is paused while backend connectivity is offline.'
                        : 'Queue recovery needs operator attention before live history can refresh.';
            }
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
            const displayState = _displayJobState(job);
            if (displayState === 'running' || displayState === 'indexing' || displayState === 'partial-failure') {
                badgeColor = 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-900/30 dark:text-amber-400 dark:border-amber-800';
            }
            if (displayState === 'reviewable') {
                badgeColor = 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-400 dark:border-emerald-800';
            }
            if (displayState === 'failed') {
                badgeColor = 'bg-red-50 text-red-700 border-red-200 dark:bg-red-900/30 dark:text-red-400 dark:border-red-800';
            }

            const safePipeline = String(job.pipeline || 'unknown');
            const safeId = String(job.id || 'job_unknown');
            const safeProgress = Math.max(0, Math.min(100, Number(job.progress) || 0));
            const canCancel = _portalPrivilegesReady() && (job.state === 'running' || job.state === 'queued');
            const artifactCount = Array.isArray(job.artifacts) ? job.artifacts.length : 0;
            const errorLine = getReadableError(job.error);
            const outcomeSummary = jobOutcomeSummary(job);
            const captioningRunStatus = normalizeRunSummary(job.run_summary)?.captioning_status || null;
            const transportLabel = formatTransportLabel(job);
            const freshnessLabel = formatRelativeTime(Number(job.lastEventAt || job.updatedAt || job.createdAt || 0));

            const header = document.createElement('div');
            header.className = 'flex items-center justify-between mb-3';

            const headerLeft = document.createElement('div');
            headerLeft.className = 'flex items-center gap-2';

            const statusDot = document.createElement('span');
            statusDot.className = `status-dot ${_displayJobStateTone(job)}`;
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
            stateBadge.textContent = displayState;
            metaRight.appendChild(stateBadge);

            const captioningChip = createCaptioningRunStatusChip(captioningRunStatus);
            if (captioningChip) {
                metaRight.appendChild(captioningChip);
            }

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

    function scheduleRenderJobQueue(includeReviewSurfaces = true) {
        queuedReviewSurfaceRefresh = queuedReviewSurfaceRefresh || includeReviewSurfaces;
        if (queueRenderScheduled) return;
        queueRenderScheduled = true;
        portalRenderScheduler.schedule(() => {
            queueRenderScheduled = false;
            const shouldRenderReviewSurfaces = queuedReviewSurfaceRefresh;
            queuedReviewSurfaceRefresh = false;
            renderJobQueue(shouldRenderReviewSurfaces);
        });
    }

    return {
        renderJobQueue,
        scheduleRenderJobQueue,
        renderSelectedJobInspector,
        renderSelectedJobTimeline,
        setInspectorTab,
    };
}
