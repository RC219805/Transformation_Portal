import test from "node:test";
import assert from "node:assert/strict";
import { fixture, settle, json, selected, error, storageFixture } from "./helpers/photographic-review-fixture.mjs";
import { photographicDraftSourceKey } from "../portal-src/photographic-draft-state.js";

async function note(h, text) {
  const control = h.root.find("photographic-decision-note");
  control.value = text; await control.trigger("input"); return control;
}

test("authoring identity excludes inspection and presentation context but includes the complete source", () => {
  const source = { artifact_path: "master.tif", sha256: "a".repeat(64), source_identity_sha256: "b".repeat(64) };
  const key = photographicDraftSourceKey("owner", "job", "scene", source);
  assert.equal(photographicDraftSourceKey("owner", "job", "scene", { ...source, presentation_policy_sha256: "c".repeat(64) }), key);
  for (const changed of [{ artifact_path: "other.tif" }, { sha256: "c".repeat(64) }, { source_identity_sha256: "d".repeat(64) }]) {
    assert.notEqual(photographicDraftSourceKey("owner", "job", "scene", { ...source, ...changed }), key);
  }
  assert.notEqual(photographicDraftSourceKey("other-owner", "job", "scene", source), key);
});

for (const paged of [false, true]) {
  const mode = paged ? "paged" : "legacy";
  test(`${mode} frozen source replacement discard reloads the retained job and purges private edits`, async () => {
    let authorization = { status: "ready", ownerKey: "verified-owner" };
    const h = fixture({ paged, sceneCount: 2, getReviewAuthorization: () => authorization });
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    const control = await note(h, "Private draft for the retired source bytes");
    authorization = { status: "unavailable", ownerKey: "verified-owner" }; h.api.syncAuthorization();
    h.scenes[0].variants[0] = { ...h.scenes[0].variants[0], sha256: "e".repeat(64) };
    authorization = { status: "ready", ownerKey: "verified-owner" }; h.api.syncAuthorization(); await settle();
    assert.equal(control.value, "Private draft for the retired source bytes");
    assert.equal(h.root.find("photographic-review").classList.contains("hidden"), true);
    const before = h.requests.length;
    await h.root.find("photographic-discard-frozen-draft").trigger();
    assert.equal(control.value, ""); assert.equal(h.api.hasUnsavedChanges(), false);
    assert.equal(h.root.find("photographic-review").classList.contains("hidden"), false);
    assert.match(h.root.find("photographic-decision-status").textContent, /eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee/);
    const reloads = h.requests.slice(before).filter((request) => request.path.endsWith(paged ? "/scene" : "/photographic-review"));
    assert.ok(reloads.length > 0, "Discard must perform a fresh authorized source read");
    assert.ok(reloads.every((request) => request.path.startsWith(`/v1/jobs/${h.job.id}/photographic-review`)));
    assert.equal(h.requests.some((request) => request.path.includes("/jobs//")), false);
    assert.equal(h.requests.some((request) => request.options.method === "POST"), false);
    authorization = { status: "revoked" }; h.api.syncAuthorization();
  });

  test(`${mode} temporary auth loss masks drafts until the same owner and source are read again`, async (t) => {
    let authorization = { status: "ready", ownerKey: "verified-owner-and-tenant" }, finish, delay = false;
    const h = fixture({ paged, sceneCount: 2, getReviewAuthorization: () => authorization,
      extraFetch: (request, response) => {
        if (!delay || !request.path.endsWith(paged ? "/scene" : "/photographic-review")) return null;
        delay = false;
        return new Promise((resolve) => { finish = () => resolve(json(paged ? response.detail("input-0000") : {
          state_version: h.server.stateVersion, scenes: h.scenes, decisions: h.server.decisions, groups: [], presets: [] })); });
      } });
    t.after(() => { authorization = { status: "revoked" }; h.api.syncAuthorization(); finish?.(); });
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    const control = await note(h, "Private unsaved note");
    const before = h.requests.length;
    authorization = { status: "unavailable", ownerKey: "verified-owner-and-tenant" };
    h.api.syncAuthorization();
    assert.equal(h.root.find("photographic-review").classList.contains("hidden"), true);
    assert.equal(control.disabled, true);
    assert.equal(h.requests.length, before, "Unverified access cannot fetch or mutate");
    assert.equal(control.value, "Private unsaved note");
    delay = true; authorization = { status: "ready", ownerKey: "verified-owner-and-tenant" };
    h.api.syncAuthorization(); await settle();
    assert.ok(finish);
    assert.equal(h.root.find("photographic-review").classList.contains("hidden"), true, "Bootstrap alone cannot reveal the draft");
    finish(); await settle(); finish = null;
    assert.equal(h.root.find("photographic-review").classList.contains("hidden"), false);
    assert.equal(control.value, "Private unsaved note");
    assert.equal(control.disabled, false);
    assert.equal(h.requests.some((request) => request.options.method === "POST"), false);
  });

  test(`${mode} changed source bytes cannot receive the previous source's draft during refresh`, async (t) => {
    const h = fixture({ paged, sceneCount: 2 });
    t.after(() => h.api.show(null, null));
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    const control = await note(h, "For the original bytes only");
    h.scenes[0].variants[0] = { ...h.scenes[0].variants[0], sha256: "e".repeat(64) };
    await h.api.refresh(); await settle();
    assert.equal(control.value, "For the original bytes only");
    assert.equal(h.root.find("photographic-discard-navigation").hidden, false);
    assert.match(h.root.find("photographic-decision-status").textContent, /0000000000000000000000000000000000000000000000000000000000000001/);
    await h.root.find("photographic-discard-navigation").trigger();
    assert.equal(control.value, "");
    assert.match(h.root.find("photographic-decision-status").textContent, /eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee/);
  });
}

for (const boundary of ["different owner", "revoked"]) {
  test(`${boundary} purges private drafts without replaying a mutation`, async () => {
    let authorization = { status: "ready", ownerKey: "owner-one" };
    const h = fixture({ getReviewAuthorization: () => authorization });
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    const control = await note(h, "Never show this to another actor");
    authorization = boundary === "revoked" ? { status: "revoked" } : { status: "ready", ownerKey: "owner-two" };
    h.api.syncAuthorization(); await settle();
    assert.equal(control.value, "");
    assert.equal(h.api.hasUnsavedChanges(), false);
    assert.equal(h.root.find("photographic-review").classList.contains("hidden"), true);
    assert.equal(h.requests.some((request) => request.options.method === "POST"), false);
    h.api.show(null, null);
  });
}

test("a frozen draft permits an explicit public leave decision while private controls remain masked", async () => {
  let authorization = { status: "ready", ownerKey: "owner" };
  const h = fixture({ getReviewAuthorization: () => authorization });
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const control = await note(h, "Private note before sign-in expired");
  authorization = { status: "unavailable", ownerKey: "owner" }; h.api.syncAuthorization();
  let leaves = 0;
  assert.equal(await h.api.requestNavigation(() => { leaves += 1; }, { leave: true }), false);
  assert.equal(leaves, 0);
  assert.equal(h.root.find("photographic-review").classList.contains("hidden"), true);
  await h.root.find("photographic-cancel-frozen-navigation").trigger();
  assert.equal(control.value, "Private note before sign-in expired");
  assert.equal(leaves, 0);
  await h.api.requestNavigation(() => { leaves += 1; }, { leave: true });
  await h.root.find("photographic-discard-frozen-draft").trigger();
  assert.equal(leaves, 1);
  assert.equal(control.value, "");
  assert.equal(h.api.hasUnsavedChanges(), false);
});

test("host navigation waits for explicit discard and cannot auto-run after saving", async (t) => {
  const h = fixture(); t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const control = await note(h, "Do not lose this note"); let leaves = 0;
  assert.equal(await h.api.requestNavigation(() => { leaves += 1; }), false);
  assert.equal(leaves, 0);
  await h.root.find("photographic-cancel-navigation").trigger();
  assert.equal(control.value, "Do not lose this note");
  await h.api.requestNavigation(() => { leaves += 1; });
  await h.root.find("photographic-discard-navigation").trigger();
  assert.equal(leaves, 1);
  assert.equal(control.value, "");
  assert.equal(selected(h), "input-0000");
});

test("a fresh same-owner source denial purges the quarantined draft", async () => {
  let authorization = { status: "ready", ownerKey: "owner" }, deny = false;
  const h = fixture({ getReviewAuthorization: () => authorization, extraFetch: (request) => deny && request.path.endsWith("/scene")
    ? new Response(JSON.stringify({ success: false, error: { message: "Revoked" } }), { status: 403 }) : null });
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const control = await note(h, "Quarantined private content");
  authorization = { status: "unavailable", ownerKey: "owner" }; h.api.syncAuthorization();
  deny = true; authorization = { status: "ready", ownerKey: "owner" }; h.api.syncAuthorization(); await settle();
  assert.equal(control.value, "");
  assert.equal(h.api.hasUnsavedChanges(), false);
  assert.equal(h.root.find("photographic-review").classList.contains("hidden"), true);
  h.api.show(null, null);
});

for (const paged of [false, true]) {
  const label = paged ? "paged" : "legacy";
  test(`${label} transient empty hydration retains the bound draft and permits later same-context rendering`, async () => {
    const h = fixture({ paged, sceneCount: 2 });
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    const control = await note(h, "Keep during jobs loading");
    const before = h.requests.length;
    assert.equal(h.api.show(null, null), false);
    assert.equal(control.value, "Keep during jobs loading");
    assert.equal(h.api.hasUnsavedChanges(), true);
    assert.equal(h.requests.length, before);
    assert.equal(h.root.find("photographic-review").classList.contains("hidden"), false);
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    assert.equal(control.value, "Keep during jobs loading");
    assert.equal(h.root.find("photographic-discard-navigation").hidden, true);
    await h.api.requestNavigation(() => h.api.show(null, null), { leave: true });
    await h.root.find("photographic-discard-navigation").trigger();
    assert.equal(control.value, "");
    assert.equal(h.root.find("photographic-review").classList.contains("hidden"), true);
  });

  for (const target of ["different job", "nonphotographic job", "catalogue rollout"]) {
    test(`${label} background ${target} waits for explicit discard before changing the draft context`, async () => {
      const h = fixture({ paged, sceneCount: 2 });
      h.api.show(h.job, h.scenes[0].variants[0]); await settle();
      const control = await note(h, "Bound to this photograph");
      const next = target === "different job" ? { ...h.job, id: "job-two" }
        : target === "nonphotographic job" ? { id: h.job.id, pipeline: "archive-gate-b", artifacts: [] } : h.job;
      if (target === "catalogue rollout") h.state.auth.features.photographicPagedCatalogue = !paged;
      const before = h.requests.length;
      assert.equal(h.api.show(next, null), false);
      assert.equal(h.requests.length, before);
      assert.equal(control.value, "Bound to this photograph");
      assert.equal(h.root.find("photographic-discard-navigation").hidden, false);
      await h.root.find("photographic-cancel-navigation").trigger();
      assert.equal(control.value, "Bound to this photograph");
      assert.equal(h.api.show(next, null), false);
      await h.root.find("photographic-discard-navigation").trigger();
      assert.equal(control.value, "");
      assert.equal(h.api.hasUnsavedChanges(), false);
      if (target === "nonphotographic job") assert.equal(h.root.find("photographic-review").classList.contains("hidden"), true);
      else assert.ok(h.requests.length > before);
    });
  }
}

test("stable authoring ownership preserves legacy delivery recovery scope and an unknown export across reload", async () => {
  const saved = storageFixture();
  const owner = { status: "ready", ownerKey: "managed:v2:stable-authoring-owner" };
  const extraFetch = (request, { server }) => request.path.endsWith("/deliveries")
    ? json({ items: [], next_cursor: null, state_version: server.stateVersion })
    : request.options.method === "POST" && request.path.endsWith("/delivery") ? error("UNAVAILABLE", "Export reply lost", 503) : null;
  const h = fixture({ sceneCount: 2, sessionStorage: saved.storage, getReviewAuthorization: () => owner, extraFetch });
  h.state.auth.features.photographicDeliveryHistory = true;
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const preset = h.root.find("photographic-preset"); preset.checked = true; await preset.trigger("change");
  await h.root.find("photographic-deliver-approved").trigger();
  assert.equal(h.requests.filter((request) => request.options.method === "POST" && request.path.endsWith("/delivery")).length, 1);
  const raw = saved.values.get("tp.photo.delivery_recovery.v1");
  const retained = JSON.parse(raw);
  assert.equal(retained.length, 1);
  assert.equal(retained[0].scope, JSON.stringify([h.job.id, JSON.stringify([h.state.auth.mode, h.state.auth.actor, h.state.auth.tenant])]));
  assert.doesNotMatch(raw, /managed:v2:stable-authoring-owner/);
  h.state.auth.actor.role = "viewer";
  h.api.show(h.job, h.scenes[0].variants[0]);
  await h.root.find("photographic-delivery-reconcile").trigger();
  assert.equal(saved.values.get("tp.photo.delivery_recovery.v1"), raw, "A role-only authoring refresh cannot rebind an unresolved delivery");
  h.api.show(null, null);
  const restored = fixture({ sceneCount: 2, sessionStorage: saved.storage, getReviewAuthorization: () => owner, extraFetch });
  restored.state.auth.features.photographicDeliveryHistory = true;
  restored.api.show(restored.job, restored.scenes[0].variants[0]); await settle();
  assert.ok(restored.requests.some((request) => request.path.endsWith("/deliveries") && request.params.get("request_correlation_id") === retained[0].attempt.body.request_correlation_id));
  assert.equal(restored.requests.some((request) => request.options.method === "POST"), false);
  assert.equal(saved.values.get("tp.photo.delivery_recovery.v1"), raw);
  h.api.show(null, null); restored.api.show(null, null);
});
