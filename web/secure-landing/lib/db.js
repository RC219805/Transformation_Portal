import { mkdirSync } from "node:fs";
import path from "node:path";

import Database from "better-sqlite3";

const dbCache = new Map();

function migrate(db) {
  db.pragma("journal_mode = WAL");
  db.exec(`
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      created_at INTEGER NOT NULL,
      last_seen_at INTEGER NOT NULL,
      idle_expires_at INTEGER NOT NULL,
      absolute_expires_at INTEGER NOT NULL,
      csrf_token TEXT NOT NULL,
      authenticated INTEGER NOT NULL DEFAULT 0,
      username TEXT,
      access_email TEXT,
      role TEXT,
      rotated_from TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_sessions_idle_expires_at
    ON sessions (idle_expires_at);

    CREATE TABLE IF NOT EXISTS login_attempts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      throttle_key TEXT NOT NULL,
      attempted_at INTEGER NOT NULL,
      success INTEGER NOT NULL DEFAULT 0,
      remote_addr TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_login_attempts_key_time
    ON login_attempts (throttle_key, attempted_at);
  `);
}

export function getDb(dbPath) {
  const resolvedPath = String(dbPath);
  if (!dbCache.has(resolvedPath)) {
    const parentDir = path.dirname(resolvedPath);
    if (parentDir && parentDir !== ".") {
      mkdirSync(parentDir, { recursive: true });
    }

    let db;
    try {
      db = new Database(resolvedPath);
    } catch (error) {
      throw new Error(
        `Unable to open TP_FRONTDOOR_SESSION_DB at ${resolvedPath}: ${error instanceof Error ? error.message : String(error)}`
      );
    }
    migrate(db);
    dbCache.set(resolvedPath, db);
  }
  return dbCache.get(resolvedPath);
}

export function resetDbCache() {
  for (const db of dbCache.values()) {
    db.close();
  }
  dbCache.clear();
}
