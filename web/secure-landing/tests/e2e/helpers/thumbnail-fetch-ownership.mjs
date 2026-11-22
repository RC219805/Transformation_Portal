// Test-only instrumentation of the application's fetch ownership. Browser
// protocol terminal events and delayed mock handlers are separate drain signals.
export function installThumbnailFetchOwnership(target = globalThis) {
  const originalFetch = target.fetch;
  const metrics = { started: 0, active: 0, peak: 0, events: [] };
  target.__tpThumbnailFetchOwnership = metrics;
  target.fetch = function (input, options) {
    let url;
    try { url = new URL(typeof input === 'string' ? input : input?.url || String(input), target.location?.href || 'http://localhost/'); }
    catch { return originalFetch.apply(this, arguments); }
    if (!url.pathname.endsWith('/photographic-review/thumbnail')) return originalFetch.apply(this, arguments);
    const id = ++metrics.started, sceneId = url.searchParams.get('scene_id');
    const signal = options?.signal || input?.signal;
    let owned = true;
    metrics.active += 1; metrics.peak = Math.max(metrics.peak, metrics.active);
    metrics.events.push({ id, sceneId, event: 'start', active: metrics.active });
    const finish = (event) => {
      if (!owned) return;
      owned = false; metrics.active -= 1;
      metrics.events.push({ id, sceneId, event, active: metrics.active });
      signal?.removeEventListener('abort', aborted);
    };
    const aborted = () => finish('abort');
    signal?.addEventListener('abort', aborted, { once: true });
    if (signal?.aborted) aborted();
    let fetched;
    try { fetched = originalFetch.apply(this, arguments); }
    catch (error) { finish('fetch-error'); throw error; }
    return Promise.resolve(fetched).then((response) => {
      if (!response.body) { finish('no-body'); return response; }
      // Keep the original Response and stream: instrument only the reader API
      // used by the thumbnail client, without buffering, cloning or new pulls.
      const getReader = response.body.getReader.bind(response.body);
      response.body.getReader = (...args) => {
        const reader = getReader(...args), read = reader.read.bind(reader), cancel = reader.cancel.bind(reader);
        reader.read = (...values) => read(...values).then((result) => {
          if (result.done) finish('body-done');
          return result;
        }, (error) => { finish('body-error'); throw error; });
        reader.cancel = (...values) => cancel(...values).then((result) => {
          finish('body-cancel'); return result;
        }, (error) => { finish('body-error'); throw error; });
        return reader;
      };
      return response;
    }, (error) => { finish('fetch-error'); throw error; });
  };
}
