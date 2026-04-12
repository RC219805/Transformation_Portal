export function isNativeEventSourceHandle(handle) {
  return Boolean(handle && typeof handle.readyState === "number" && typeof handle.addEventListener === "function");
}

export function nativeEventSourceReadyState(handle) {
  if (!isNativeEventSourceHandle(handle)) return null;
  const readyState = Number(handle.readyState);
  return Number.isInteger(readyState) ? readyState : null;
}
