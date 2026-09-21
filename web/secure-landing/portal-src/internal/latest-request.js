// Coalesce identical concurrent work, and prevent an old A request from
// publishing after the user has edited A -> B -> A.
export function createLatestRequestCoordinator() {
  let current = null;
  const invalidate = () => {
    const previous = current;
    current = null;
    previous?.controller.abort("request_superseded");
  };
  return {
    invalidate,
    run(key, operation) {
      if (current?.key === key) return current.promise;
      invalidate();
      const controller = new AbortController();
      const request = { key, controller, signal: controller.signal, promise: null, isCurrent: () => current === request };
      current = request;
      request.promise = Promise.resolve().then(() => operation(request)).finally(() => {
        if (current === request) current = null;
      });
      return request.promise;
    }
  };
}
