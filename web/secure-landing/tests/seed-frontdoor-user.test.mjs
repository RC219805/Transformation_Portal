import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import argon2 from "argon2";

const SCRIPT_PATH = fileURLToPath(new URL("../scripts/seed-frontdoor-user.mjs", import.meta.url));
const SCRIPT_CWD = path.dirname(SCRIPT_PATH);

test("seed-frontdoor-user overwrites stale fixture and writes verifiable hash", async () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-seed-user-"));
  const usersFile = path.join(tempDir, "frontdoor-users.json");
  writeFileSync(
    usersFile,
    JSON.stringify([
      {
        username: "admin",
        password_hash: "stale-hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]),
    "utf-8"
  );

  try {
    const result = spawnSync(process.execPath, [SCRIPT_PATH], {
      cwd: SCRIPT_CWD,
      encoding: "utf-8",
      env: {
        ...process.env,
        TP_FRONTDOOR_USERS_FILE: usersFile,
        TP_FRONTDOOR_USERNAME: "smoke-admin",
        TP_FRONTDOOR_PASSWORD: "correct horse battery staple"
      }
    });

    assert.equal(result.status, 0, result.stderr || result.stdout);

    const parsed = JSON.parse(readFileSync(usersFile, "utf-8"));
    assert.equal(parsed.length, 1);
    assert.equal(parsed[0].username, "smoke-admin");
    assert.equal(parsed[0].access_email, "smoke-admin@local.invalid");
    assert.equal(parsed[0].role, "admin");
    assert.equal(
      await argon2.verify(parsed[0].password_hash, "correct horse battery staple"),
      true
    );
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("seed-frontdoor-user normalizes username and email to lowercase", async () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-seed-user-"));
  const usersFile = path.join(tempDir, "frontdoor-users.json");

  try {
    const result = spawnSync(process.execPath, [SCRIPT_PATH], {
      cwd: SCRIPT_CWD,
      encoding: "utf-8",
      env: {
        ...process.env,
        TP_FRONTDOOR_USERS_FILE: usersFile,
        TP_FRONTDOOR_USERNAME: "AdminUser",
        TP_FRONTDOOR_PASSWORD: "password123",
        TP_FRONTDOOR_ACCESS_EMAIL: "ADMIN@Example.COM"
      }
    });

    assert.equal(result.status, 0, result.stderr || result.stdout);

    const parsed = JSON.parse(readFileSync(usersFile, "utf-8"));
    assert.equal(parsed[0].username, "adminuser");
    assert.equal(parsed[0].access_email, "admin@example.com");
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("seed-frontdoor-user writes file with restrictive permissions", () => {
  if (os.platform() === "win32") {
    return;
  }

  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-seed-user-"));
  const usersFile = path.join(tempDir, "frontdoor-users.json");

  try {
    const result = spawnSync(process.execPath, [SCRIPT_PATH], {
      cwd: SCRIPT_CWD,
      encoding: "utf-8",
      env: {
        ...process.env,
        TP_FRONTDOOR_USERS_FILE: usersFile,
        TP_FRONTDOOR_USERNAME: "admin",
        TP_FRONTDOOR_PASSWORD: "password123"
      }
    });

    assert.equal(result.status, 0, result.stderr || result.stdout);

    const stats = statSync(usersFile);
    const mode = stats.mode & 0o777;
    assert.equal(mode, 0o600, `Expected mode 0600 but got ${mode.toString(8)}`);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("seed-frontdoor-user tightens permissions when overwriting an existing fixture", () => {
  if (os.platform() === "win32") {
    return;
  }

  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-seed-user-"));
  const usersFile = path.join(tempDir, "frontdoor-users.json");
  writeFileSync(
    usersFile,
    JSON.stringify([
      {
        username: "admin",
        password_hash: "stale-hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]),
    { encoding: "utf-8", mode: 0o644 }
  );
  chmodSync(usersFile, 0o644);

  try {
    const result = spawnSync(process.execPath, [SCRIPT_PATH], {
      cwd: SCRIPT_CWD,
      encoding: "utf-8",
      env: {
        ...process.env,
        TP_FRONTDOOR_USERS_FILE: usersFile,
        TP_FRONTDOOR_USERNAME: "admin",
        TP_FRONTDOOR_PASSWORD: "password123"
      }
    });

    assert.equal(result.status, 0, result.stderr || result.stdout);

    const mode = statSync(usersFile).mode & 0o777;
    assert.equal(mode, 0o600, `Expected overwritten fixture mode 0600 but got ${mode.toString(8)}`);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("seed-frontdoor-user rejects invalid role", () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-seed-user-"));
  const usersFile = path.join(tempDir, "frontdoor-users.json");

  try {
    const result = spawnSync(process.execPath, [SCRIPT_PATH], {
      cwd: SCRIPT_CWD,
      encoding: "utf-8",
      env: {
        ...process.env,
        TP_FRONTDOOR_USERS_FILE: usersFile,
        TP_FRONTDOOR_USERNAME: "admin",
        TP_FRONTDOOR_PASSWORD: "password123",
        TP_FRONTDOOR_ROLE: "superuser"
      }
    });

    assert.equal(result.status, 1);
    assert.match(result.stderr, /TP_FRONTDOOR_ROLE must be one of: admin/);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("seed-frontdoor-user fails closed when required env is missing", () => {
  const result = spawnSync(process.execPath, [SCRIPT_PATH], {
    cwd: SCRIPT_CWD,
    encoding: "utf-8",
    env: {
      ...process.env,
      TP_FRONTDOOR_USERS_FILE: "",
      TP_FRONTDOOR_USERNAME: "smoke-admin",
      TP_FRONTDOOR_PASSWORD: "correct horse battery staple"
    }
  });

  assert.notEqual(result.status, 0);
  assert.match(
    result.stderr || result.stdout,
    /seed-frontdoor-user requires TP_FRONTDOOR_USERS_FILE, TP_FRONTDOOR_USERNAME, and TP_FRONTDOOR_PASSWORD\./
  );
});

test("seed-frontdoor-user exits with code 2 for missing env vars", () => {
  const result = spawnSync(process.execPath, [SCRIPT_PATH], {
    cwd: SCRIPT_CWD,
    encoding: "utf-8",
    env: {
      ...process.env,
      TP_FRONTDOOR_USERS_FILE: "",
      TP_FRONTDOOR_USERNAME: "",
      TP_FRONTDOOR_PASSWORD: ""
    }
  });

  assert.equal(result.status, 2);
});
