import test from "node:test";
import assert from "node:assert/strict";
import { createLatestRequestCoordinator } from "../portal-src/internal/latest-request.js";

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

test("concurrent identical preview work shares one request but later refresh stays fresh", async () => {
  const coordinator = createLatestRequestCoordinator();
  const gate = deferred();
  let calls = 0;
  const operation = async () => { calls += 1; await gate.promise; return "ready"; };
  const first = coordinator.run("A", operation);
  const second = coordinator.run("A", operation);
  assert.equal(first, second);
  await Promise.resolve();
  assert.equal(calls, 1);
  gate.resolve();
  assert.equal(await second, "ready");
  await coordinator.run("A", operation);
  assert.equal(calls, 2);
});

test("A to B to A edits cannot publish the first A response after the latest A response", async () => {
  const coordinator = createLatestRequestCoordinator();
  const gates = [deferred(), deferred(), deferred()];
  const published = [];
  const requests = ["A", "B", "A"].map((key, index) => coordinator.run(key, async (request) => {
    await gates[index].promise;
    if (request.isCurrent()) published.push(index);
  }));
  gates[2].resolve();
  await requests[2];
  gates[0].resolve();
  gates[1].resolve();
  await Promise.all(requests);
  assert.deepEqual(published, [2]);
});

test("a failed request releases coalescing so a service retry can execute", async () => {
  const coordinator = createLatestRequestCoordinator();
  await assert.rejects(coordinator.run("A", async () => { throw new Error("temporary"); }), /temporary/);
  assert.equal(await coordinator.run("A", async () => "retried"), "retried");
});

test("auth recovery invalidates same-payload work and aborts its transport", async () => {
  const coordinator = createLatestRequestCoordinator();
  const gate = deferred();
  let oldRequest;
  const previous = coordinator.run("A", async (request) => { oldRequest = request; await gate.promise; return request.isCurrent(); });
  await Promise.resolve();
  coordinator.invalidate();
  assert.equal(oldRequest.signal.aborted, true);
  assert.equal(await coordinator.run("A", async () => "new credentials"), "new credentials");
  gate.resolve();
  assert.equal(await previous, false);
});
