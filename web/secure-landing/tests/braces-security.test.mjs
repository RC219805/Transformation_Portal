import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync, readdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const require = createRequire(import.meta.url);
const vendorPath = fileURLToPath(new URL("../vendor/braces/", import.meta.url));
const braces = require(vendorPath);
const depthError = error => error instanceof SyntaxError && /Nesting depth exceeds.*100/.test(error.message);
const childCycleError = error => error instanceof SyntaxError && /AST child edges contain a cycle/.test(error.message);
const nestedPattern = (depth, open = "{", close = "}") => open.repeat(depth) + "a,b" + close.repeat(depth);

test("micromatch resolves the reviewed fork through the installed braces dependency", () => {
  const micromatchRequire = createRequire(require.resolve("micromatch"));
  assert.equal(micromatchRequire.resolve("braces"), require.resolve(vendorPath));
  const installed = micromatchRequire("braces");
  assert.equal(installed, braces);
  assert.throws(() => installed.compile(nestedPattern(101)), depthError);
});

function nestedAst(depth) {
  let node = { type: "text", value: "ordinary" };
  for (let i = 0; i < depth; i++) node = { type: "paren", nodes: [node] };
  return { type: "root", nodes: [node] };
}

test("fork identity, MIT attribution, and every runtime byte match reviewed provenance", () => {
  const metadata = JSON.parse(readFileSync(path.join(vendorPath, "package.json"), "utf8"));
  const provenance = JSON.parse(readFileSync(path.join(vendorPath, "PROVENANCE.json"), "utf8"));
  assert.equal(metadata.name, "@transformation-portal/braces");
  assert.equal(metadata.version, "3.0.3-tp.1");
  assert.equal(metadata.private, true);
  assert.equal(metadata.license, "MIT");
  assert.equal(provenance.upstream.git_revision, "74b2db2938fad48a2ea54a9c8bf27a37a62c350d");
  assert.equal(provenance.upstream.integrity, "sha512-yQbXgO/OSZVD2IsiLlro+7Hf6Q18EJrKSEsdoMzKePKXct3gvD8oLcOQdIzGupr5Fj+EDe8gO/lxc1BzfMpxvA==");
  assert.equal(provenance.fork.max_nested_containers, 100);
  assert.match(readFileSync(path.join(vendorPath, "LICENSE"), "utf8"), /Copyright \(c\) 2014-present, Jon Schlinkert/);
  const runtimeFiles = ["LICENSE", "index.js", "package.json", ...readdirSync(path.join(vendorPath, "lib")).map(name => `lib/${name}`)];
  assert.deepEqual(Object.keys(provenance.fork.source_sha256).sort(), runtimeFiles.sort());
  for (const relative of runtimeFiles) {
    const digest = createHash("sha256").update(readFileSync(path.join(vendorPath, relative))).digest("hex");
    assert.equal(digest, provenance.fork.source_sha256[relative], relative);
  }
});

for (const operation of ["parse", "compile", "expand", "stringify", "create", "default"]) {
  test(`${operation} accepts 100 nested containers and rejects container 101`, () => {
    const invoke = operation === "default" ? braces : braces[operation];
    for (const [open, close] of [["{", "}"], ["(", ")"], ["({", "})"]]) {
      const pairs = open.length;
      assert.doesNotThrow(() => invoke(nestedPattern(100 / pairs, open, close)));
      assert.throws(() => invoke(nestedPattern(100 / pairs, open, close) + "{".repeat(101)), depthError);
    }
    assert.throws(() => invoke(nestedPattern(101)), depthError);
    assert.throws(() => invoke(nestedPattern(101, "(", ")")), depthError);
    assert.throws(() => invoke(nestedPattern(101), { maxDepth: Infinity, maxLength: Infinity }), depthError);
  });
}

for (const operation of ["compile", "expand", "stringify"]) {
  test(`${operation} bounds direct AST inputs before recursion and preserves boundary 100`, () => {
    assert.doesNotThrow(() => braces[operation](nestedAst(100)));
    assert.throws(() => braces[operation](nestedAst(101)), depthError);
    assert.throws(() => braces[operation](nestedAst(4000)), depthError);
    const cyclic = { type: "root", nodes: [] };
    cyclic.nodes.push(cyclic);
    assert.throws(() => braces[operation](cyclic), childCycleError);
    const parent = { type: "paren", nodes: [] };
    const child = { type: "paren", nodes: [parent] };
    parent.nodes.push(child);
    assert.throws(() => braces[operation](parent), childCycleError);
  });
}

test("array-valued AST text cannot bypass bounded traversal or create recursive array cycles", () => {
  let value = ["a"];
  for (let i = 0; i < 4000; i++) value = [value];
  for (const operation of ["compile", "expand", "stringify"]) {
    assert.throws(() => braces[operation]({ type: "root", nodes: [{ type: "text", value }] }), depthError);
    const cyclic = [];
    cyclic.push(cyclic);
    assert.throws(() => braces[operation]({ type: "root", nodes: [{ type: "text", value: cyclic }] }), childCycleError);
  }
});

test("expand bounds both root and ancestor parent-chain walks", () => {
  const self = { type: "paren", nodes: [{ type: "text", value: "a" }] };
  self.parent = self;
  assert.throws(() => braces.expand(self), depthError);
  const ancestor = { type: "paren" };
  const ast = { type: "root", nodes: [{ type: "paren", parent: ancestor, nodes: [{ type: "paren", nodes: [] }] }] };
  ancestor.parent = ancestor;
  assert.throws(() => braces.expand(ast), depthError);
  let deep = { type: "root", queue: [] };
  for (let i = 0; i < 101; i++) deep = { type: "paren", parent: deep, queue: [] };
  assert.throws(() => braces.expand({ type: "paren", parent: deep, nodes: [] }), depthError);
});

test("ordinary alternatives, ranges, extglobs, options, and shared AST children retain behavior", () => {
  assert.equal(braces.compile("app/{reading,writing}/**/*.{js,jsx}"), "app/(reading|writing)/**/*.(js|jsx)");
  assert.deepEqual(braces.expand("page-{1..3}.js"), ["page-1.js", "page-2.js", "page-3.js"]);
  assert.deepEqual(braces.expand("foo/({a,b})"), ["foo/(a)", "foo/(b)"]);
  assert.deepEqual(braces.expand("a\\{b,c\\}"), ["a{b,c}"]);
  assert.deepEqual(braces(["{a,b}", "{b,c}"], { expand: true, nodupes: true }), ["a", "b", "c"]);
  assert.deepEqual(braces.expand("{,a,a}", { noempty: true, nodupes: true }), ["a"]);
  assert.throws(() => braces.expand("{1..1001}"), /range limit/);
  assert.throws(() => braces.parse("x".repeat(10001)), /max characters/);
  assert.throws(() => braces.parse("abc", { maxLength: 2 }), /max characters/);
  const shared = { type: "text", value: "a" };
  assert.equal(braces.compile({ type: "root", nodes: [shared, shared] }), "aa");
  assert.equal(braces.stringify(braces.parse("{a,{b,c}}")), "{a,{b,c}}");
});

test("published 3.0.3 compatibility corpus retains all outputs and error semantics", () => {
  const corpus = JSON.parse(readFileSync(new URL("./fixtures/braces-3.0.3-corpus.json", import.meta.url), "utf8"));
  assert.equal(corpus.source.git_revision, "74b2db2938fad48a2ea54a9c8bf27a37a62c350d");
  assert.equal(corpus.cases.length, 144);
  for (const { pattern, options, results } of corpus.cases) {
    for (const [operation, expected] of Object.entries(results)) {
      if (expected.error) {
        assert.throws(() => braces[operation](pattern, options), error => (
          error.name === expected.error.name && error.message === expected.error.message
        ), `${operation}: ${pattern}`);
      } else {
        assert.deepEqual(braces[operation](pattern, options), expected.value, `${operation}: ${pattern}`);
      }
    }
  }
});

test("escapes, quotes, and character classes retain literal deeply nested text", () => {
  assert.doesNotThrow(() => braces.compile("\\{".repeat(1000)));
  assert.equal(braces.stringify('"' + nestedPattern(1000) + '"'), nestedPattern(1000));
  assert.equal(braces.stringify("[" + nestedPattern(1000) + "]"), "[" + nestedPattern(1000) + "]");
});

test("4000-level exploit strings and direct ASTs fail predictably with a 512 KB stack", () => {
  const script = `
    const assert = require('node:assert/strict');
    const braces = require(${JSON.stringify(vendorPath)});
    const rejected = error => error instanceof SyntaxError && /Nesting depth exceeds.*100/.test(error.message);
    for (const operation of ['parse', 'compile', 'expand', 'stringify', 'create', 'default']) {
      const invoke = operation === 'default' ? braces : braces[operation];
      for (const [open, close] of [['{', '}'], ['(', ')']]) {
        assert.throws(() => invoke(open.repeat(4000) + 'x' + close.repeat(4000)), rejected);
        assert.doesNotThrow(() => invoke(open.repeat(100) + 'a,b' + close.repeat(100)));
      }
    }
    for (const operation of ['compile', 'expand', 'stringify']) {
      let node = {type:'text', value:'a'};
      for (let i=0; i<4000; i++) node = {type:'paren', nodes:[node]};
      assert.throws(() => braces[operation]({type:'root', nodes:[node]}), rejected);
      const cycle = {type:'paren', nodes:[]}; cycle.parent = cycle;
      if (operation === 'expand') assert.throws(() => braces.expand(cycle), rejected);
    }
    process.stdout.write('bounded');
  `;
  assert.equal(execFileSync(process.execPath, ["--stack_size=512", "-e", script], { encoding: "utf8", timeout: 5000 }), "bounded");
});
