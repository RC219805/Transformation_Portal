export function audit(event, details = {}) {
  const payload = {
    ts: new Date().toISOString(),
    event,
    ...details
  };
  console.info(JSON.stringify(payload));
}
