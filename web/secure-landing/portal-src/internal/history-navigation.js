// Restore an owned history traversal before asking to discard a draft. Never
// replace its destination entry: cancelling Back must preserve that destination.
export function createPortalHistoryNavigation(host) {
  const history = host.history;
  let position = Number.isSafeInteger(history.state?.portalIndex) ? history.state.portalIndex : 0;
  let restoring = null, permitted = null;
  if (!Number.isSafeInteger(history.state?.portalIndex)) {
    history.replaceState({ ...history.state, portalIndex: position }, '');
  }
  function record(replace, state, url) {
    // The browser may still be on the traversal's destination or source entry.
    // Background renders must preserve both entries and the pending permission.
    if (restoring !== null || permitted !== null) return;
    restoring = permitted = null;
    if (!replace) position += 1;
    history[replace ? 'replaceState' : 'pushState']({ ...state, portalIndex: position }, '', url);
  }
  function pop(event) {
    const target = event.state?.portalIndex;
    if (restoring !== null) {
      restoring.distance += restoring.step;
      if (target !== position) { history.go(restoring.step); return; }
      const destination = restoring;
      restoring = null;
      host.requestNavigation(() => {
        permitted = destination.target;
        history.go(-destination.distance);
      });
      return;
    }
    // Entries outside this router remain browser-owned. Cross-document leaves
    // also retain the native beforeunload warning and source-level draft guards.
    if (!Number.isSafeInteger(target)) { host.apply(); return; }
    // Native fragment entries inherit their route index without leaving it.
    if (target === position) { host.apply(); return; }
    if (target === permitted || !host.hasEdits()) {
      permitted = null; position = target; host.apply(); return;
    }
    // Fragments can share a route index. Count physical steps back to the
    // retained route so discard replays the exact requested destination.
    restoring = { target, step: Math.sign(position - target), distance: 0 };
    history.go(restoring.step);
  }
  return { record, pop };
}
