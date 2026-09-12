import assert from "node:assert/strict";
import test from "node:test";

import { load } from "js-yaml";

test("empty YAML merge sources consume the configured work budget", () => {
  const document = "empty: &empty {}\nresult:\n  <<: [*empty, *empty]\n";
  assert.throws(() => load(document, { maxTotalMergeKeys: 1 }), /maxTotalMergeKeys/);
});

test("ordinary YAML configuration merges remain supported", () => {
  const document = "defaults: &defaults {enabled: true}\nresult: {<<: *defaults}\n";
  assert.deepEqual(load(document), {
    defaults: { enabled: true },
    result: { enabled: true },
  });
});
