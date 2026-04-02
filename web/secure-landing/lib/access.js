import { getConfig, normalizeAccessEmail } from "./config.js";

export function resolveAccessContext(request) {
  const accessEmail = normalizeAccessEmail(
    request.headers.get("cf-access-authenticated-user-email") ||
      request.headers.get("x-access-email")
  );

  if (accessEmail) {
    return {
      accessEmail,
      bypass: false
    };
  }

  if (getConfig().allowLocalAccessBypass) {
    return {
      accessEmail: null,
      bypass: true
    };
  }

  return {
    accessEmail: null,
    bypass: false
  };
}
