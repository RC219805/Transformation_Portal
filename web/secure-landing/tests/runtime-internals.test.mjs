import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as library from "../portal-src/internal/index.js";
import * as runtime from "../portal-src/internal/runtime-entry.js";

// The deferred importer receives the browser namespace through its host, so its
// converters are required even though the template does not call them directly.
const consumers = ["portal.template.js", "config-transfer-deferred.js"].map((name) =>
  readFileSync(new URL(`../portal-src/${name}`, import.meta.url), "utf8"));
const required = [...new Set(consumers.flatMap((source) =>
  [...source.matchAll(/\bportalInternals(?:\?\.|\.)\s*([A-Za-z_$][\w$]*)/g)].map((match) => match[1])))].sort();

test("browser internal namespace exports exactly its template and deferred-import consumers", () => {
  assert.deepEqual(Object.keys(runtime).sort(), required);
  assert.ok(Object.keys(library).length > required.length, "The full library must not be projected wholesale into the browser namespace");
  for (const source of consumers) {
    assert.doesNotMatch(source, /\bportalInternals\s*(?:\?\.)?\s*\[/, "Computed namespace reads require an explicit projection contract");
  }
});

test("projected internal values retain the full library's reference identities", () => {
  for (const name of required) {
    assert.ok(Object.hasOwn(library, name), `${name} is part of the unchanged full index`);
    assert.strictEqual(runtime[name], library[name], `${name} must retain its original implementation`);
  }
});
