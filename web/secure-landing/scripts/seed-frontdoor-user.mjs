import { chmodSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

import argon2 from "argon2";

import { normalizeAccessEmail, normalizeUsername } from "../lib/config.js";

const REQUIRED_ENV_ERROR =
  "seed-frontdoor-user requires TP_FRONTDOOR_USERS_FILE, TP_FRONTDOOR_USERNAME, and TP_FRONTDOOR_PASSWORD.";

const ALLOWED_ROLES = new Set(["admin"]);

function requiredValue(name, { trim = true } = {}) {
  const raw = process.env[name];
  if (typeof raw !== "string") {
    throw new Error(REQUIRED_ENV_ERROR);
  }
  const value = trim ? raw.trim() : raw;
  if (!value) {
    throw new Error(REQUIRED_ENV_ERROR);
  }
  return value;
}

async function main() {
  const usersFilePath = requiredValue("TP_FRONTDOOR_USERS_FILE");
  const rawUsername = requiredValue("TP_FRONTDOOR_USERNAME");
  const password = requiredValue("TP_FRONTDOOR_PASSWORD", { trim: false });
  const rawAccessEmail = String(process.env.TP_FRONTDOOR_ACCESS_EMAIL || "").trim();
  const rawRole = String(process.env.TP_FRONTDOOR_ROLE || "admin").trim() || "admin";

  const username = normalizeUsername(rawUsername);
  if (!username) {
    throw new Error("TP_FRONTDOOR_USERNAME must be a non-empty alphanumeric string after normalization.");
  }

  const accessEmail = rawAccessEmail ? normalizeAccessEmail(rawAccessEmail) : `${username}@local.invalid`;
  if (!accessEmail) {
    throw new Error("TP_FRONTDOOR_ACCESS_EMAIL must be a valid email after normalization.");
  }

  const role = rawRole.toLowerCase();
  if (!ALLOWED_ROLES.has(role)) {
    throw new Error(`TP_FRONTDOOR_ROLE must be one of: ${[...ALLOWED_ROLES].join(", ")}. Got: "${rawRole}".`);
  }

  mkdirSync(path.dirname(usersFilePath), { recursive: true });

  const passwordHash = await argon2.hash(password);
  const payload = [
    {
      username,
      password_hash: passwordHash,
      access_email: accessEmail,
      role
    }
  ];

  writeFileSync(usersFilePath, `${JSON.stringify(payload, null, 2)}\n`, { encoding: "utf-8", mode: 0o600 });
  if (process.platform !== "win32") {
    chmodSync(usersFilePath, 0o600);
  }
  console.log(`Wrote front-door credential fixture to ${usersFilePath} for ${username}.`);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exit(error instanceof Error && error.message === REQUIRED_ENV_ERROR ? 2 : 1);
});
