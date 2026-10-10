# Repository-maintained braces fork

This private CommonJS package starts from the published `braces@3.0.3`
tarball. `PROVENANCE.json` records its immutable upstream revision, tarball
integrity, original file hashes, and patched file hashes. The upstream MIT
license and author attribution are preserved. The package has its own scoped
identity, `@transformation-portal/braces@3.0.3-tp.1`; it is not an upstream
release.

The patch addresses [GHSA-vfj7-8cjw-p6xm](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm)
by limiting brace and parenthesis nesting to **100 containers**. Attempting
to open container 101 throws a `SyntaxError` before parser allocation.
Escaped, quoted, and character-class text retains its original behavior.
Options cannot raise the mandatory limit.

The `compile`, `expand`, and `stringify` functions also validate direct AST
inputs iteratively before recursive processing. The root wrapper does not
count as a container; nested nodes with children and nested array values do.
Child-edge cycles are rejected with a `SyntaxError`. Expansion's two existing
parent-link walks are bounded to 100 links, including caller-supplied cycles.
Ordinary API results, ranges, options, and dependencies retain their published
semantics. Existing input-length and range limits continue to apply; this
patch does not introduce a general expansion-size or regular-expression-time
guarantee.

The repository regression suite is `tests/braces-security.test.mjs`. Upstream
tests must be obtained at the recorded revision and run against this source
in a disposable directory; upstream test dependencies are not shipped here.
