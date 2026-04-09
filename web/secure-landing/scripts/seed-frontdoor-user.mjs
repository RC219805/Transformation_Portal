import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

import argon2 from "argon2";

const REQUIRED_ENV_ERROR =
  "seed-frontdoor-user requires TP_FRONTDOOR_USERS_FILE, TP_FRONTDOOR_USERNAME, and TP_FRONTDOOR_PASSWORD.";

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

const usersFilePath = requiredValue("TP_FRONTDOOR_USERS_FILE");
const username = requiredValue("TP_FRONTDOOR_USERNAME");
const password = requiredValue("TP_FRONTDOOR_PASSWORD", { trim: false });
const accessEmail =
  String(process.env.TP_FRONTDOOR_ACCESS_EMAIL || "").trim() || `${username}@local.invalid`;
const role = String(process.env.TP_FRONTDOOR_ROLE || "admin").trim() || "admin";

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

writeFileSync(usersFilePath, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
console.log(`Wrote front-door credential fixture to ${usersFilePath} for ${username}.`);
