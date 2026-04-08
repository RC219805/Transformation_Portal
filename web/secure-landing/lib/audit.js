const AUDIT_OBSERVER_KEY = Symbol.for("tp.frontdoor.auditObserver");

function getAuditObserver() {
  return globalThis[AUDIT_OBSERVER_KEY];
}

export function setAuditObserver(observer) {
  globalThis[AUDIT_OBSERVER_KEY] = observer;
}

export function clearAuditObserver() {
  delete globalThis[AUDIT_OBSERVER_KEY];
}

export function audit(event, details = {}) {
  const payload = {
    ts: new Date().toISOString(),
    event,
    ...details
  };
  const observer = getAuditObserver();
  if (typeof observer === "function") {
    observer(payload);
  }
  console.info(JSON.stringify(payload));
  return payload;
}
