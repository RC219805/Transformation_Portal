// Loaded-run discovery and local log inspection. Neither feature queries or
// mutates backend history; the selected inspector remains independent of filters.
const lexical = (left, right) => left < right ? -1 : left > right ? 1 : 0;
const searchText = (value) => String(value || '').trim().toLowerCase();

function createdAt(job) {
  const value = job.createdAt ?? job.created_at ?? 0;
  const numeric = Number(value);
  if (Number.isFinite(numeric)) return numeric * (job.createdAt === undefined ? 1000 : 1);
  const parsed = Date.parse(String(value));
  return Number.isFinite(parsed) ? parsed : 0;
}

export function filterLoadedJobs(jobs, { query = '', status = 'all', displayState = (job) => job.state } = {}) {
  const needle = searchText(query);
  return jobs.filter(job => job && (status === 'all' || displayState(job) === status)
    && (!needle || `${job.id || ''}\n${job.pipeline || ''}`.toLowerCase().includes(needle)))
    .sort((left, right) => createdAt(right) - createdAt(left) || lexical(String(left.id || ''), String(right.id || '')));
}

export function loadedLogSnapshot(job, query = '', maxLines = 1200) {
  const limit = Math.max(1, Math.min(1200, Math.trunc(Number(maxLines) || 1200)));
  const lines = Array.isArray(job?.logs) ? job.logs.slice(-limit).map(String) : [];
  const needle = searchText(query);
  const matching = needle ? lines.filter(line => line.toLowerCase().includes(needle)) : lines;
  const text = values => values.join('\n') + (values.length ? '\n' : '');
  return { lines, matching, text: text(matching), allText: text(lines) };
}

export function createOperateTools(host) {
  const { state, els, _displayJobState, _portalPrivilegesReady, _consumeQueueDeltaAnnouncementSuppression, copyToClipboard } = host;
  const doc = host.document || globalThis.document;
  const byId = id => doc?.getElementById(id);
  const search = byId('loadedRunSearch'), status = byId('loadedRunState');
  const logSearch = byId('loadedLogSearch'), follow = byId('toggleLogFollowBtn');
  const copy = byId('copyLoadedLogsBtn'), download = byId('downloadLoadedLogsBtn'), summary = byId('loadedLogSummary');
  let criteria = { query: '', status: 'all' }, owner = '', selectedLogJob = '', logQuery = '', following = true;
  let stateOptions = '', lastQueueDeltaSnapshot = null;
  function currentJob() { return state.jobs.find(job => job.id === state.selectedJobId); }
  function reconcileOwner() {
    const actor = state.auth?.actor;
    const next = JSON.stringify([state.auth?.mode, actor?.accessEmail, actor?.username, state.auth?.tenant]);
    if (owner === next) return;
    owner = next; criteria = { query: '', status: 'all' }; logQuery = ''; following = true; selectedLogJob = '';
    if (search) search.value = '';
    if (logSearch) logSearch.value = '';
  }
  function renderQueueControls() {
    reconcileOwner();
    if (!status) return;
    const values = [...new Set(state.jobs.map(_displayJobState).filter(Boolean))];
    if (criteria.status !== 'all' && !values.includes(criteria.status)) values.push(criteria.status);
    values.sort(lexical);
    const next = JSON.stringify(values);
    if (next !== stateOptions) {
      stateOptions = next;
      const option = (value, label) => { const item = doc.createElement('option'); item.value = value; item.textContent = label; return item; };
      status.replaceChildren(option('all', 'All states'), ...values.map(value => option(value, value.replaceAll('-', ' ').replace(/^./, char => char.toUpperCase()))));
    }
    status.value = criteria.status;
  }
  function visibleJobs() { reconcileOwner(); return filterLoadedJobs(state.jobs, { ...criteria, displayState: _displayJobState }); }
  function renderLogs() {
    reconcileOwner();
    const job = currentJob();
    if (selectedLogJob !== (job?.id || '')) {
      selectedLogJob = job?.id || ''; logQuery = ''; following = true;
      if (logSearch) logSearch.value = '';
    }
    const snapshot = loadedLogSnapshot(job, logQuery, host.maxLogLines);
    if (els.logPane) {
      const previousTop = els.logPane.scrollTop;
      if (els.logPane.textContent !== snapshot.text) els.logPane.textContent = snapshot.text;
      els.logPane.scrollTop = following ? els.logPane.scrollHeight : previousTop;
    }
    if (follow) { follow.textContent = following ? 'Pause follow' : 'Resume follow'; follow.setAttribute('aria-pressed', String(following)); follow.disabled = !job; }
    if (logSearch) logSearch.disabled = !job;
    const available = Boolean(job && snapshot.lines.length && _portalPrivilegesReady());
    if (copy) copy.disabled = !available;
    if (download) download.disabled = !available;
    if (summary) summary.textContent = !job ? 'Select a run. These tools inspect only the loaded log buffer.'
      : `${snapshot.matching.length} matching / ${snapshot.lines.length} loaded lines · at most ${host.maxLogLines || 1200} · ${following ? 'following' : 'follow paused'}. Copy and download include all loaded lines.`;
    return snapshot;
  }

  let logRenderPending = false;
  // Reconcile the pane and its controls against the same bounded buffer once per frame.
  function appendLog(jobId) {
    if (!els.logPane || state.selectedJobId !== jobId || logRenderPending) return;
    if (!host.portalRenderScheduler) {
      renderLogs();
      return;
    }
    logRenderPending = true;
    host.portalRenderScheduler.schedule(() => {
      logRenderPending = false;
      renderLogs();
    });
  }
  function clearLogs() { const job = currentJob(); if (job) job.logs = []; renderLogs(); }
  search?.addEventListener('input', () => { criteria.query = search.value; host.onFilterChange(); });
  status?.addEventListener('change', () => { criteria.status = status.value; host.onFilterChange(); });
  logSearch?.addEventListener('input', () => { logQuery = logSearch.value; renderLogs(); });
  follow?.addEventListener('click', () => { following = !following; renderLogs(); });
  copy?.addEventListener('click', () => { if (!copy.disabled && _portalPrivilegesReady()) void copyToClipboard(renderLogs().allText); });
  download?.addEventListener('click', () => {
    const job = currentJob();
    if (!job || download.disabled || !_portalPrivilegesReady()) return;
    const url = URL.createObjectURL(new Blob([renderLogs().allText], { type: 'text/plain;charset=utf-8' }));
    const link = doc.createElement('a'); link.href = url;
    link.download = `${String(job.id).replace(/[^a-zA-Z0-9_-]/g, '-').slice(0, 96)}-loaded-logs.txt`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

    function announceQueueDelta() {
        if (!els.queueDeltaStatus) return;
        const current = new Map(state.jobs.map((job) => [String(job.id || ''), _displayJobState(job)]));
        let message = '';
        if (lastQueueDeltaSnapshot === null) {
            const activeCount = state.jobs.filter((job) => job && (job.state === 'running' || job.state === 'queued')).length;
            message = state.jobs.length === 0
                ? 'Queue is empty.'
                : `Queue has ${state.jobs.length} job${state.jobs.length === 1 ? '' : 's'}; ${activeCount} active.`;
        } else {
            const added = Array.from(current.keys()).filter((jobId) => !lastQueueDeltaSnapshot.has(jobId));
            const removed = Array.from(lastQueueDeltaSnapshot.keys()).filter((jobId) => !current.has(jobId));
            const changed = Array.from(current.entries()).filter(([jobId, displayState]) => (
                lastQueueDeltaSnapshot.has(jobId) && lastQueueDeltaSnapshot.get(jobId) !== displayState
            ));
            const announceableChanged = changed.filter(([jobId, displayState]) => (
                typeof _consumeQueueDeltaAnnouncementSuppression !== 'function'
                || !_consumeQueueDeltaAnnouncementSuppression(jobId, displayState)
            ));
            if (added.length > 0) {
                message = `${added.length} job${added.length === 1 ? '' : 's'} added to the queue.`;
            } else if (removed.length > 0) {
                message = `${removed.length} job${removed.length === 1 ? '' : 's'} removed from the queue.`;
            } else if (announceableChanged.length > 0) {
                const [jobId, displayState] = announceableChanged[0];
                message = `Job ${jobId} is now ${displayState}.`;
            }
        }
        lastQueueDeltaSnapshot = current;
        if (!message || els.queueDeltaStatus.textContent === message) return;
        els.queueDeltaStatus.dataset.pendingAnnouncement = message;
        els.queueDeltaStatus.textContent = '';
        window.setTimeout(() => {
            if (els.queueDeltaStatus.dataset.pendingAnnouncement === message) {
                els.queueDeltaStatus.textContent = message;
            }
        }, 0);
    }

  renderQueueControls(); renderLogs();
  return { visibleJobs, renderQueueControls, renderLogs, appendLog, clearLogs, announceQueueDelta };
}
