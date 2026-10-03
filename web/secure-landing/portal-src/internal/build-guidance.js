// Guidance is navigation only. Dispatch authority stays with the current
// server preview, readiness checks, and the explicit Run control.
function guidanceHelp(step, pipeline) {
  if (step === 1) return {
    question: 'Which workflow should I use?',
    answer: 'V3 is the production default. Choose an opt-in photography workflow only when your workspace supports it. A saved profile restores its settings; you can review and change them before dispatch.'
  };
  if (step === 2) return {
    question: 'Where should files go?',
    answer: 'Use source and output folders approved for this workspace. Uploading files fills the input path only. Choose a separate, fresh output folder; the server preview checks access. Archive stages also need the index or rights manifest for the selected stage.'
  };
  if (step === 3) {
    if (['lux-depth', 'lux-depth-v5', 'lux-depth-v6'].includes(pipeline)) return {
      question: 'Which color mode fits?',
      answer: 'Use Auto for supported embedded color metadata. For untagged images, Auto + assume sRGB records your assumption. Invalid or unsupported profiles need a profile-aware export. Explicit sRGB and Linear sRGB assert the existing encoding; they do not convert another profile.'
    };
    if (String(pipeline || '').startsWith('archive-gate-')) return {
      question: 'What do readiness checks mean?',
      answer: 'Readiness checks identify the runtime, files, and permissions required by this archive stage. Resolve blocking checks before dispatch. Warnings remain visible for your review.'
    };
    return {
      question: 'Which settings should I change?',
      answer: 'Start with the preset, then change the output format and optional deliverables you need. Advanced options stay available. The current preview checks the exact draft before it can run.'
    };
  }
  return {
    question: 'What happens next?',
    answer: 'Review the expected outputs and all blocking checks first. The Run control submits this draft. Operate shows backend progress and saved logs; Review shows indexed outputs when they are available. Step navigation does not approve or start a run.'
  };
}

const STEP_GUIDANCE = {
  1: { title: 'Make this run your own', detail: 'Choose a workflow and preset, or resume a saved profile.', label: 'Choose paths' },
  2: { title: 'Keep sources and outputs separate', detail: 'Keep your source files in their input folder and choose a fresh, authorized destination.', label: 'Review output settings' },
  3: { title: 'Choose only the outputs you need', detail: 'Review your settings and the expected deliverables before starting.', label: 'Review this run' }
};

export function buildWorkspaceGuidance({
  step = 1, pipeline, nextAction, readiness, accessReady = false, accessSummary,
  dispatchPending = false, handoffJobId, overview = false
}) {
  const help = guidanceHelp(step, pipeline);
  if (!accessReady) return {
    ...help,
    title: accessSummary?.badge || 'Confirming workspace access',
    detail: accessSummary?.detail || 'Your draft will stay here while the portal verifies access.',
    label: 'Open Build', step, disabled: true,
    tone: accessSummary?.tone === 'blocked' ? 'error' : accessSummary?.tone === 'warning' ? 'warning' : 'info'
  };
  if (dispatchPending) return {
    ...help, title: 'Submitting your run', detail: 'Wait for the server to confirm this submission.',
    label: 'Submitting…', step: 4, tone: 'info', disabled: true
  };
  if (handoffJobId) return {
    ...help, title: 'Your run has been submitted', detail: 'Open Operate to follow progress and read the saved logs.',
    label: 'Follow this run', jobId: handoffJobId, tone: 'success'
  };
  const action = nextAction || {};
  if (['blocked', 'warning'].includes(action.tone) && action.action !== 'dispatch_ready') return {
    ...help, title: action.label || 'This draft needs attention', detail: action.detail || readiness?.detail,
    label: action.field ? 'Show me where' : 'Review checks', field: action.field, step: 4,
    tone: action.tone === 'blocked' ? 'error' : 'warning'
  };
  if (!readiness?.canRun && (overview || step === 4)) return {
    ...help, title: 'Checking this draft', detail: readiness?.detail || 'Wait for the current preview and readiness checks.',
    label: 'Review checks', step: 4, tone: 'info'
  };
  if (overview || step === 4) return {
    ...help, title: 'Your draft is ready to review', detail: 'Review the expected outputs, then use the Run control when you are satisfied.',
    label: 'Review launch controls', step: 4, field: 'run_job', tone: 'success'
  };
  return { ...help, ...STEP_GUIDANCE[step], step: step + 1, tone: 'info' };
}
