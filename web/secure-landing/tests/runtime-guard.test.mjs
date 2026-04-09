import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import os from "node:os";
import path from "node:path";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";

import {
  ensureSupportedNodeVersion,
  formatNativeDependencyFailureMessage,
  formatUnsupportedNodeMessage
} from "../lib/runtime-guard.js";
import {
  attachStandaloneLifecycleHandlers,
  resolveStandaloneServerPath
} from "../scripts/start-standalone.mjs";

test("runtime guard rejects non-22 runtimes with one recovery path", () => {
  assert.throws(
    () => ensureSupportedNodeVersion("25.9.0"),
    /requires Node 22\.x\. Current runtime: 25\.9\.0\. Switch to Node 22, reinstall or rebuild front-door dependencies under that runtime, then retry\./
  );
  assert.match(formatUnsupportedNodeMessage("22.22.2"), /requires Node 22\.x/);
});

test("runtime guard collapses native addon ABI mismatches into actionable guidance", () => {
  const error = new Error(
    "The module 'better_sqlite3.node' was compiled against a different Node.js version using NODE_MODULE_VERSION 127."
  );

  const message = formatNativeDependencyFailureMessage("better-sqlite3", error, "22.22.2");

  assert.match(message, /native dependency "better-sqlite3"/);
  assert.match(message, /native addon ABI mismatch/);
  assert.doesNotMatch(message, /NODE_MODULE_VERSION 127/);
  assert.match(message, /Switch to Node 22, reinstall or rebuild front-door dependencies under that runtime, then retry\./);
});

test("standalone start resolves both supported server output paths", () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-standalone-"));

  try {
    const rootServer = path.join(tempDir, ".next", "standalone", "server.js");
    mkdirSync(path.dirname(rootServer), { recursive: true });
    writeFileSync(rootServer, "// root server", "utf-8");
    assert.equal(resolveStandaloneServerPath(tempDir), rootServer);

    rmSync(rootServer, { force: true });
    const nestedServer = path.join(tempDir, ".next", "standalone", "web", "secure-landing", "server.js");
    mkdirSync(path.dirname(nestedServer), { recursive: true });
    writeFileSync(nestedServer, "// nested server", "utf-8");
    assert.equal(resolveStandaloneServerPath(tempDir), nestedServer);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("standalone start honors TP_NEXT_DIST_DIR when resolving standalone output", () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-standalone-distdir-"));
  const env = { TP_NEXT_DIST_DIR: ".next-build-verify" };

  try {
    const rootServer = path.join(tempDir, ".next-build-verify", "standalone", "server.js");
    mkdirSync(path.dirname(rootServer), { recursive: true });
    writeFileSync(rootServer, "// root server", "utf-8");
    assert.equal(resolveStandaloneServerPath(tempDir, { env }), rootServer);

    rmSync(rootServer, { force: true });
    const nestedServer = path.join(
      tempDir,
      ".next-build-verify",
      "standalone",
      "web",
      "secure-landing",
      "server.js"
    );
    mkdirSync(path.dirname(nestedServer), { recursive: true });
    writeFileSync(nestedServer, "// nested server", "utf-8");
    assert.equal(resolveStandaloneServerPath(tempDir, { env }), nestedServer);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("standalone start surfaces actionable spawn failures", () => {
  const child = new EventEmitter();
  const errors = [];
  const exits = [];

  attachStandaloneLifecycleHandlers(child, {
    serverPath: "/tmp/frontdoor/server.js",
    processLike: {
      pid: 42,
      execPath: process.execPath,
      env: process.env,
      exit(code) {
        exits.push(code);
      },
      kill() {}
    },
    consoleLike: {
      error(message) {
        errors.push(message);
      }
    }
  });

  child.emit("error", new Error("EACCES: permission denied"));

  assert.deepEqual(exits, [1]);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /could not start from \/tmp\/frontdoor\/server\.js/);
  assert.match(errors[0], /EACCES: permission denied/);
  assert.match(errors[0], /Run npm run build under Node 22/);
});

test("standalone start exits cleanly when signal relay fails", () => {
  const child = new EventEmitter();
  const errors = [];
  const exits = [];
  const killCalls = [];

  attachStandaloneLifecycleHandlers(child, {
    serverPath: "/tmp/frontdoor/server.js",
    processLike: {
      pid: 99,
      execPath: process.execPath,
      env: process.env,
      exit(code) {
        exits.push(code);
      },
      kill(pid, signal) {
        killCalls.push([pid, signal]);
        throw new Error("unsupported signal relay");
      }
    },
    consoleLike: {
      error(message) {
        errors.push(message);
      }
    }
  });

  child.emit("exit", null, "SIGPWR");

  assert.deepEqual(killCalls, [[99, "SIGPWR"]]);
  assert.deepEqual(exits, [1]);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /signal SIGPWR/);
  assert.match(errors[0], /unsupported signal relay/);
});
