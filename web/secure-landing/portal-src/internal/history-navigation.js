// Restore an owned history traversal before asking to discard a draft. Never
// replace its destination entry: cancelling Back must preserve that destination.
export function createPortalHistoryNavigation(host) {
  const history = host.history;
  function routeUrl(value = host.location?.href) {
    if (typeof value !== 'string') return undefined;
    try {
      const url = new URL(value, host.location?.href || 'https://portal.invalid/'); url.hash = '';
      return url.href;
    } catch { return undefined; }
  }
  let position = Number.isSafeInteger(history.state?.portalIndex) ? history.state.portalIndex : 0;
  // Unknown forks must stay after every owned entry observed on this page.
  let sequence = position;
  let restoring = null, permitted = null, unowned = null;
  let retainedState = { ...history.state, portalIndex: position };
  let retainedUrl = routeUrl();
  if (!Number.isSafeInteger(history.state?.portalIndex)) {
    history.replaceState(retainedState, '');
  }
  function record(replace, state, url) {
    // The browser may still be on the traversal's destination or source entry.
    // Background renders must preserve both entries and the pending permission.
    if (restoring !== null || permitted !== null) return;
    const actualUrl = routeUrl();
    if (actualUrl !== undefined && retainedUrl !== undefined && actualUrl !== retainedUrl && (unowned === null || unowned.active)) unowned = {};
    if (replace && unowned !== null) {
      // Accepted active routes may synchronize selection without inventing an
      // index. Pending or canceled destinations retain the replacement fence.
      if (unowned.active) history.replaceState(history.state, '', url);
      retainedUrl = routeUrl(url); return;
    }
    restoring = permitted = null;
    const wasUnowned = unowned !== null;
    unowned = null;
    if (!replace) {
      // Adopt a native fragment before leaving it, while its source is known.
      if (!wasUnowned && !Number.isSafeInteger(history.state?.portalIndex)) history.replaceState(retainedState, '');
      position = ++sequence;
    }
    retainedState = { ...state, portalIndex: position };
    history[replace ? 'replaceState' : 'pushState'](retainedState, '', url);
    retainedUrl = routeUrl(url);
  }
  function applyTarget(target, state) {
    retainedUrl = routeUrl();
    if (Number.isSafeInteger(target)) {
      position = target; retainedState = state; permitted = null; unowned = null;
    } else unowned = { active: retainedUrl !== undefined };
    try { return host.apply(); } catch (error) {
      if (unowned !== null) unowned = {};
      throw error;
    }
  }
  function confirmTarget(target, state) {
    const fence = {};
    unowned = fence;
    host.requestNavigation(() => {
      if (unowned !== fence) return false;
      return applyTarget(target, state);
    });
  }
  function pop(event) {
    const target = event.state?.portalIndex;
    if (Number.isSafeInteger(target)) sequence = Math.max(sequence, target);
    if (restoring !== null) {
      restoring.distance += restoring.step;
      if (target !== position || routeUrl() !== retainedUrl) { history.go(restoring.step); return; }
      const destination = restoring;
      restoring = null;
      host.requestNavigation(() => {
        permitted = destination.target;
        history.go(-destination.distance);
      });
      return;
    }
    const sameSource = retainedUrl !== undefined && routeUrl() === retainedUrl;
    if (!Number.isSafeInteger(target)) {
      if ((unowned === null || unowned.active) && sameSource) { host.apply(); return; }
      // A pre-router entry has no trustworthy direction. Retain authoring and
      // leave its browser-owned URL untouched until explicit discard. Cancel
      // keeps replacement syncs fenced even though that URL can remain visible.
      if (host.hasEdits()) confirmTarget(target, event.state);
      else applyTarget(target, event.state);
      return;
    }
    // A same-source notification must not consume an in-flight permission.
    if (target === position && sameSource) {
      unowned = null; retainedState = event.state; retainedUrl = routeUrl(); host.apply(); return;
    }
    // An equal index or an unindexed active source cannot locate a different
    // authoring route physically. Keep the same bounded confirmation fence.
    if (unowned !== null || target === position) {
      if (host.hasEdits() && !sameSource) confirmTarget(target, event.state);
      else applyTarget(target, event.state);
      return;
    }
    if (target === permitted || !host.hasEdits()) {
      applyTarget(target, event.state); return;
    }
    if (retainedUrl === undefined || routeUrl() === undefined) { confirmTarget(target, event.state); return; }
    // Fragments can share a route index. Count physical steps back to the
    // retained route so discard replays the exact requested destination.
    restoring = { target, step: Math.sign(position - target), distance: 0 };
    history.go(restoring.step);
  }
  return { record, pop };
}
