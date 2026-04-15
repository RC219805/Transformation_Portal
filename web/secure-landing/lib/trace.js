const TRACEPARENT_RE = /^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$/;

function _fillRandomBytes(length) {
  const cryptoImpl = globalThis.crypto;
  if (!cryptoImpl || typeof cryptoImpl.getRandomValues !== "function") {
    throw new Error("Secure random source unavailable for trace generation");
  }
  return cryptoImpl.getRandomValues(new Uint8Array(length));
}

function _bytesToHex(bytes) {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
}

function _randomHex(byteLength) {
  return _bytesToHex(_fillRandomBytes(byteLength));
}

function _isAllZero(value) {
  return /^0+$/.test(value);
}

export function normalizeTraceparent(rawValue) {
  const normalized = String(rawValue || "").trim().toLowerCase();
  if (!normalized) {
    return "";
  }
  const match = normalized.match(TRACEPARENT_RE);
  if (!match) {
    return "";
  }
  const [, version, traceId, parentId, traceFlags] = match;
  if (_isAllZero(traceId) || _isAllZero(parentId)) {
    return "";
  }
  return `${version}-${traceId}-${parentId}-${traceFlags}`;
}

export function generateTraceparent(options = {}) {
  const normalizedParent = normalizeTraceparent(options.parentTraceparent);
  const traceId = normalizedParent
    ? normalizedParent.split("-")[1]
    : _randomHex(16);
  const traceFlags = normalizedParent
    ? normalizedParent.split("-")[3]
    : options.sampled === false
      ? "00"
      : "01";
  return `00-${traceId}-${_randomHex(8)}-${traceFlags}`;
}

export function resolveRequestTraceparent(request) {
  return normalizeTraceparent(request?.headers?.get("traceparent")) || generateTraceparent();
}

export function traceIdFromTraceparent(traceparent) {
  const normalized = normalizeTraceparent(traceparent);
  if (!normalized) {
    return "";
  }
  return normalized.split("-")[1];
}
