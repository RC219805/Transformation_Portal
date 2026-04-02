import argon2 from "argon2";

import { getConfig, normalizeAccessEmail, normalizeUsername } from "./config.js";

export function findMatchingUser({ username, accessEmail, allowAccessBypass = false }) {
  const wantedUsername = normalizeUsername(username);
  const wantedAccessEmail = normalizeAccessEmail(accessEmail);

  return (
    getConfig().users.find((user) => {
      if (user.username !== wantedUsername) return false;
      if (allowAccessBypass) return true;
      return user.accessEmail === wantedAccessEmail;
    }) || null
  );
}

export async function verifyUserCredentials({
  username,
  password,
  accessEmail,
  allowAccessBypass = false
}) {
  const user = findMatchingUser({ username, accessEmail, allowAccessBypass });
  if (!user || !password) return null;

  try {
    const verified = await argon2.verify(user.passwordHash, String(password));
    return verified ? user : null;
  } catch {
    return null;
  }
}
