export function createRenderScheduler(windowRef) {
  const scheduleFrame = typeof windowRef?.requestAnimationFrame === "function"
    ? windowRef.requestAnimationFrame.bind(windowRef)
    : (callback) => windowRef.setTimeout(callback, 0);

  return {
    schedule(callback) {
      scheduleFrame(callback);
    }
  };
}

export function createRenderSurfaceRegistry() {
  const surfaces = new Map();

  function normalizeSurface(surface = {}) {
    return {
      init: typeof surface.init === "function" ? surface.init : () => {},
      render: typeof surface.render === "function" ? surface.render : () => {}
    };
  }

  return {
    register(name, surface) {
      if (!name) return;
      surfaces.set(String(name), normalizeSurface(surface));
    },
    init(state) {
      for (const surface of surfaces.values()) {
        surface.init(state);
      }
    },
    render(name, state) {
      const surface = surfaces.get(String(name));
      if (!surface) return;
      surface.render(state);
    }
  };
}
