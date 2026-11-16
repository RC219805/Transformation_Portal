import test from "node:test";
import assert from "node:assert/strict";

import { fixture, settle, json, error, selected, requested, NEXT_CATALOGUE_SHA } from "./helpers/photographic-review-fixture.mjs";

async function edit(h, ui, value, event = "input") {
  const control = h.root.find(ui);
  control.value = value;
  await control.trigger(event);
  return control;
}

function commitDecision(h, request) {
  const body = JSON.parse(request.options.body);
  const scene = h.scenes.find((candidate) => candidate.variants.some((variant) => variant.artifact_path === body.artifact_path));
  h.server.decisions = [...h.server.decisions.filter((row) => row.artifact_path !== body.artifact_path), { ...body, scene_id: scene.scene_id }];
  h.server.stateVersion += 1;
  h.server.catalogueIdentitySha256 = NEXT_CATALOGUE_SHA;
  return json({ saved: true, state_version: h.server.stateVersion });
}

for (const paged of [false, true]) {
  const mode = paged ? "paged" : "legacy";

  test(`${mode} valid numeric undo restores a clean revision and permits source navigation`, async () => {
    const h = fixture({ paged, sceneCount: 2 });
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    for (const [field, changed, neutral] of [["exposure", "0.2", "0"], ["warmth", "0.05", "0"], ["saturation", "1.1", "1"]]) {
      await edit(h, `photographic-revision-${field}`, changed);
      assert.equal(h.api.hasUnsavedChanges(), true);
      await edit(h, `photographic-revision-${field}`, neutral);
      assert.equal(h.api.hasUnsavedChanges(), false, `${field} must compare numeric meaning rather than the input string type`);
    }
    await edit(h, "photographic-scene", h.scenes[1].scene_id, "change");
    assert.equal(selected(h), h.scenes[1].scene_id);
    assert.equal(h.root.find("photographic-discard-navigation").hidden, true);
    h.api.show(null, null);
  });

  test(`${mode} empty and invalid numeric revision values remain unsaved`, async () => {
    const h = fixture({ paged, sceneCount: 2 });
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    for (const value of ["", " ", "invalid", "Infinity", "2"]) {
      await edit(h, "photographic-revision-exposure", value);
      assert.equal(h.api.hasUnsavedChanges(), true, `Invalid input ${JSON.stringify(value)} cannot become a neutral number`);
    }
    await edit(h, "photographic-scene", h.scenes[1].scene_id, "change");
    assert.equal(selected(h), h.scenes[0].scene_id);
    assert.equal(h.root.find("photographic-discard-navigation").hidden, false);
    await h.root.find("photographic-discard-navigation").trigger(); h.api.show(null, null);
  });

  for (const transition of ["selector", "card", "version"]) {
    test(`${mode} ${transition} navigation cannot transfer a note to another source`, async (t) => {
      const h = fixture({ paged, sceneCount: 2 });
      const previousPath = "previous/room-a.tif";
      h.scenes[0].variants.push({ ...h.scenes[0].variants[0], role: "previous_revision", artifact_path: previousPath, sha256: "e".repeat(64) });
      t.after(() => h.api.show(null, null));
      h.api.show(h.job, h.scenes[0].variants[0]); await settle();
      const note = await edit(h, "photographic-decision-note", "Private note for photograph A");
      if (transition === "selector") await edit(h, "photographic-scene", h.scenes[1].scene_id, "change");
      else if (transition === "card") await h.root.findAll("photographic-contact-item")[1].trigger();
      else await edit(h, "photographic-primary", previousPath, "change");
      assert.equal(selected(h), h.scenes[0].scene_id);
      assert.equal(h.root.find("photographic-primary").value, h.scenes[0].variants[0].artifact_path);
      assert.equal(note.value, "Private note for photograph A");
      assert.equal(h.root.find("photographic-discard-navigation").hidden, false);
      assert.equal(requested(h, "/decisions").length, 0);
      await h.root.find("photographic-discard-navigation").trigger();
      assert.equal(note.value, "");
      if (transition === "version") {
        assert.equal(h.root.find("photographic-primary").value, previousPath);
      } else assert.equal(selected(h), h.scenes[1].scene_id);
    });
  }

  test(`${mode} delayed decision saves retain newer editable notes`, async (t) => {
    let finish;
    let h;
    h = fixture({ paged, sceneCount: 2, extraFetch: (request) => request.path.endsWith("/decisions")
      ? new Promise((resolve) => { finish = () => resolve(commitDecision(h, request)); }) : null });
    t.after(() => { finish?.(); h.api.show(null, null); });
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    const note = await edit(h, "photographic-decision-note", "first note");
    await h.root.find("photographic-decision-approved").trigger();
    assert.ok(finish);
    assert.equal(note.disabled, false, "Operators can continue typing during save");
    await edit(h, "photographic-decision-note", "second note");
    finish(); await settle(); finish = null;
    assert.equal(note.value, "second note");
    assert.equal(h.server.decisions.find((row) => row.artifact_path === h.scenes[0].variants[0].artifact_path).note, "first note");
    assert.match(h.root.find("photographic-draft-status").textContent, /unsaved/i);
    await edit(h, "photographic-scene", h.scenes[1].scene_id, "change");
    assert.equal(selected(h), h.scenes[0].scene_id, "A save cannot clear a later edit's navigation guard");
    assert.equal(note.value, "second note");
    assert.equal(requested(h, "/decisions").length, 1);
  });

  test(`${mode} pending saves block source changes while preserving editable notes`, async (t) => {
    let finish;
    const h = fixture({ paged, sceneCount: 2, extraFetch: (request) => request.path.endsWith("/decisions")
      ? new Promise((resolve) => { finish = () => resolve(error("UNAVAILABLE", "Save unavailable", 503)); }) : null });
    t.after(() => { finish?.(); h.api.show(null, null); });
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    const note = await edit(h, "photographic-decision-note", "Keep this draft");
    await h.root.find("photographic-decision-approved").trigger();
    await edit(h, "photographic-scene", h.scenes[1].scene_id, "change");
    assert.equal(selected(h), h.scenes[0].scene_id);
    assert.equal(note.value, "Keep this draft");
    finish(); await settle(); finish = null;
    assert.equal(note.value, "Keep this draft");
    await edit(h, "photographic-scene", h.scenes[1].scene_id, "change");
    assert.equal(selected(h), h.scenes[0].scene_id);
    assert.equal(h.root.find("photographic-discard-navigation").hidden, false);
  });

  for (const field of ["name", "reference"]) {
    test(`${mode} unsaved lighting ${field} survives refresh and requires explicit discard`, async (t) => {
      const h = fixture({ paged, sceneCount: 2 });
      t.after(() => h.api.show(null, null));
      h.api.show(h.job, h.scenes[0].variants[0]); await settle();
      const ui = field === "name" ? "photographic-lighting-group" : "photographic-reference";
      const value = field === "name" ? "Unsaved warm practical lighting" : h.scenes[0].variants[0].sha256;
      const control = await edit(h, ui, value, field === "reference" ? "change" : "input");
      await h.api.refresh(); await settle();
      assert.equal(control.value, value, "Unrelated saved metadata cannot overwrite a lighting draft");
      await edit(h, "photographic-scene", h.scenes[1].scene_id, "change");
      assert.equal(selected(h), h.scenes[0].scene_id);
      assert.equal(control.value, value);
      assert.equal(h.root.find("photographic-discard-navigation").hidden, false);
      await h.root.find("photographic-discard-navigation").trigger();
      assert.equal(selected(h), h.scenes[1].scene_id);
      assert.equal(control.value, "");
    });
  }

  test(`${mode} comparison and inspection changes preserve source-bound drafts`, async (t) => {
    const h = fixture({ paged, sceneCount: 2 });
    t.after(() => h.api.show(null, null));
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    const note = await edit(h, "photographic-decision-note", "Keep this source note");
    const lighting = await edit(h, "photographic-lighting-group", "Keep this lighting draft");
    await edit(h, "photographic-comparison", "", "change");
    await edit(h, "photographic-inspection", "native", "change");
    await edit(h, "photographic-presentation", "clip_srgb", "change");
    assert.equal(note.value, "Keep this source note");
    assert.equal(lighting.value, "Keep this lighting draft");
    assert.equal(selected(h), h.scenes[0].scene_id);
    assert.equal(requested(h, "/decisions").length, 0);
  });
}

test("saving a note does not discard an unrelated lighting draft", async (t) => {
  const h = fixture({ sceneCount: 2 });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  const lighting = await edit(h, "photographic-lighting-group", "Unsaved lighting");
  await edit(h, "photographic-decision-note", "Saved note");
  await h.root.find("photographic-decision-approved").trigger();
  assert.equal(lighting.value, "Unsaved lighting");
  await edit(h, "photographic-scene", h.scenes[1].scene_id, "change");
  assert.equal(selected(h), h.scenes[0].scene_id);
});

for (const paged of [false, true]) {
  for (const httpStatus of [200, 503]) {
  test(`${paged ? 'paged' : 'legacy'} history-disabled export ${httpStatus} preserves the unrelated revision draft`, async () => {
    const h = fixture({ paged, sceneCount: 2, extraFetch: request => {
      if (!request.path.endsWith('/delivery')) return null;
      if (httpStatus === 503) return error('UNAVAILABLE', 'Export outcome unknown', 503);
      h.server.stateVersion += 1;
      return json({ state_version: h.server.stateVersion, sha256: 'a'.repeat(64), filename: 'delivery.zip', size_bytes: 10,
        download_url: `/v1/jobs/${h.job.id}/artifacts/delivery.zip` });
    } });
    h.api.show(h.job, h.scenes[0].variants[0]); await settle();
    const revision = await edit(h, 'photographic-revision-exposure', '0.2');
    const preset = h.root.find('photographic-preset'); preset.checked = true; await preset.trigger('change');
    await h.root.find('photographic-deliver-approved').trigger();
    assert.equal(requested(h, '/delivery').length, 1);
    assert.equal(revision.value, '0.2'); assert.equal(h.api.hasUnsavedChanges(), true);
    if (httpStatus === 503) {
      await h.root.find('photographic-deliver-approved').trigger();
      assert.equal(requested(h, '/delivery').length, 1, 'Unknown export cannot replay before GET reconciliation');
      await h.api.refresh(); await settle();
      assert.equal(revision.value, '0.2');
    }
    await edit(h, 'photographic-scene', h.scenes[1].scene_id, 'change');
    assert.equal(selected(h), h.scenes[0].scene_id);
    await h.root.find('photographic-discard-navigation').trigger(); h.api.show(null, null);
  });
  }
}

test("an uncertain save requires a fresh read before another explicit mutation", async (t) => {
  let unavailable = true;
  const h = fixture({ extraFetch: (request) => unavailable && request.path.endsWith("/decisions")
    ? error("UNAVAILABLE", "The save outcome is unknown", 503) : null });
  t.after(() => h.api.show(null, null));
  h.api.show(h.job, h.scenes[0].variants[0]); await settle();
  await edit(h, "photographic-decision-note", "Retained uncertain note");
  const save = h.root.find("photographic-decision-approved");
  await save.trigger();
  assert.equal(requested(h, "/decisions").length, 1);
  assert.equal(save.disabled, true);
  await save.trigger();
  assert.equal(requested(h, "/decisions").length, 1, "No mutation is replayed before reconciliation");
  unavailable = false;
  await h.api.refresh(); await settle();
  assert.equal(save.disabled, false);
  await save.trigger();
  assert.equal(requested(h, "/decisions").length, 2);
});
