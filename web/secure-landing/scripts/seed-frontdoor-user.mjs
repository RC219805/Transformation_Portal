#!/usr/bin/env node

import argon2 from "argon2";
import { mkdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";

const DEFAULT_OUTPUT_PATH = "/tmp/tp-frontdoor-users.json";
const DEFAULT_USERNAME = "smoke-admin";
const DEFAULT_PASSWORD = "correct horse battery staple";
const DEFAULT_ROLE = "admin";

function usage() {
  return [
    "Usage: node ./scripts/seed-frontdoor-user.mjs [options]",
    "",
    "Options:",
    "  --output <path>        Output JSON path (default: TP_FRONTDOOR_USERS_FILE or /tmp/tp-frontdoor-users.json)",
    "  --username <name>      Username (default: TP_FRONTDOOR_USERNAME or smoke-admin)",
    "  --password <value>     Password (default: TP_FRONTDOOR_PASSWORD or correct horse battery staple)",
    "  --access-email <mail>  Access email (default: TP_FRONTDOOR_ACCESS_EMAIL or <username>@local.invalid)",
    "  --role <role>          Role (default: TP_FRONTDOOR_ROLE or admin)",
    "  --quiet                Suppress success output",
    "  --help                 Show this help text",
  ].join("\n");
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = String(argv[index] || "");
    if (!token.startsWith("--")) {
      throw new Error(`Unexpected positional argument: ${token}`);
    }

    const [flag, inlineValue] = token.split("=", 2);
    if (flag === "--help") {
      parsed.help = true;
      continue;
    }
    if (flag === "--quiet") {
      parsed.quiet = true;
      continue;
    }

    const nextValue = inlineValue ?? argv[index + 1];
    if (typeof nextValue !== "string" || nextValue.length === 0) {
      throw new Error(`Missing value for ${flag}`);
    }
    if (inlineValue === undefined) {
      index += 1;
    }

    if (flag === "--output") {
      parsed.output = nextValue;
      continue;
    }
    if (flag === "--username") {
      parsed.username = nextValue;
      continue;
    }
    if (flag === "--password") {
      parsed.password = nextValue;
      continue;
    }
    if (flag === "--access-email") {
      parsed.accessEmail = nextValue;
      continue;
    }
    if (flag === "--role") {
      parsed.role = nextValue;
      continue;
    }

    throw new Error(`Unknown option: ${flag}`);
  }
  return parsed;
}

function resolveValue(explicitValue, envName, defaultValue = "") {
  const rawValue =
    typeof explicitValue === "string" && explicitValue.length > 0
      ? explicitValue
      : process.env[envName] || defaultValue;
  return String(rawValue || "").trim();
}

function defaultAccessEmail(username) {
  return `${username}@local.invalid`;
}

async function seedFrontdoorUser({
  outputPath,
  username,
  password,
  accessEmail,
  role,
}) {
  const resolvedOutputPath = path.resolve(outputPath);
  const outputDir = path.dirname(resolvedOutputPath);
  mkdirSync(outputDir, { recursive: true });
  const tempOutputPath = path.join(
    outputDir,
    `.${path.basename(resolvedOutputPath)}.${process.pid}.${randomUUID()}.tmp`,
  );

  const passwordHash = await argon2.hash(password);
  const payload = [
    {
      username,
      password_hash: passwordHash,
      access_email: accessEmail,
      role,
    },
  ];

  try {
    writeFileSync(tempOutputPath, `${JSON.stringify(payload, null, 2)}\n`, {
      encoding: "utf-8",
      mode: 0o600,
      flag: "wx",
    });
    renameSync(tempOutputPath, resolvedOutputPath);
  } catch (error) {
    rmSync(tempOutputPath, { force: true });
    throw error;
  }
  return resolvedOutputPath;
}

async function main(argv = process.argv.slice(2)) {
  let parsed;
  try {
    parsed = parseArgs(argv);
  } catch (error) {
    console.error(String(error instanceof Error ? error.message : error));
    console.error("");
    console.error(usage());
    process.exitCode = 1;
    return;
  }

  if (parsed.help) {
    console.log(usage());
    return;
  }

  const outputPath = resolveValue(parsed.output, "TP_FRONTDOOR_USERS_FILE", DEFAULT_OUTPUT_PATH);
  const username = resolveValue(parsed.username, "TP_FRONTDOOR_USERNAME", DEFAULT_USERNAME).toLowerCase();
  const password = resolveValue(parsed.password, "TP_FRONTDOOR_PASSWORD", DEFAULT_PASSWORD);
  const role = resolveValue(parsed.role, "TP_FRONTDOOR_ROLE", DEFAULT_ROLE);
  const accessEmail = resolveValue(
    parsed.accessEmail,
    "TP_FRONTDOOR_ACCESS_EMAIL",
    defaultAccessEmail(username),
  ).toLowerCase();

  if (!username) {
    throw new Error("Front-door username cannot be empty.");
  }
  if (!password) {
    throw new Error("Front-door password cannot be empty.");
  }
  if (!accessEmail) {
    throw new Error("Front-door access email cannot be empty.");
  }

  const resolvedOutputPath = await seedFrontdoorUser({
    outputPath,
    username,
    password,
    accessEmail,
    role,
  });

  if (!parsed.quiet) {
    console.log(`Seeded front-door user fixture at ${resolvedOutputPath}`);
    console.log(`Username: ${username}`);
    console.log(`Access email: ${accessEmail}`);
  }
}

const invokedPath = process.argv[1] ? path.resolve(process.argv[1]) : "";
const modulePath = fileURLToPath(import.meta.url);

if (invokedPath === modulePath) {
  await main();
}

export {
  DEFAULT_OUTPUT_PATH,
  DEFAULT_PASSWORD,
  DEFAULT_ROLE,
  DEFAULT_USERNAME,
  defaultAccessEmail,
  seedFrontdoorUser,
};
