"""Manifest parsing of current study objects + hash verification."""

import hashlib
import json
import os

from steerlab_server.experiment.manifest import Manifest


def _write_manifest(tmp_path, payload):
    d = tmp_path / "experiments" / payload["name"]
    d.mkdir(parents=True)
    (d / "experiment.json").write_text(json.dumps(payload), encoding="utf-8")


def test_parses_current_study_objects(tmp_path):
    _write_manifest(tmp_path, {
        "name": "study1", "modelID": "org/m", "status": "draft",
        "studyKind": "multiAgent",
        "multiAgentScenarioPath": "prompts/scenarios/pd.json",
        "multiAgentScenarioHash": "abc", "multiAgentIncludeBaseline": False,
        "variantConditions": [{"name": "v1", "artifactPath": "runs/x/v.json",
                               "artifactHash": "h1", "artifact": {"baseModelID": "org/m"}}],
        "evaluation": {"kind": "pairedJudge", "judgeModel": "claude-opus-4-8",
                       "judgePrompt": "rate"},
        "concepts": [], "conditions": [],
    })
    m = Manifest.load("study1", root=str(tmp_path))
    assert m.study_kind == "multiAgent"
    assert m.multi_agent_scenario_path == "prompts/scenarios/pd.json"
    assert m.multi_agent_include_baseline is False
    assert m.variant_conditions[0].name == "v1"
    assert m.variant_conditions[0].artifact["baseModelID"] == "org/m"
    assert m.evaluation.kind == "pairedJudge"
    assert m.evaluation.judge_model == "claude-opus-4-8"


def test_verify_detects_variant_and_scenario_drift(tmp_path):
    scen = tmp_path / "scen.json"
    scen.write_text("payload", encoding="utf-8")
    good = hashlib.sha256(b"payload").hexdigest()
    _write_manifest(tmp_path, {
        "name": "s2", "modelID": "m", "status": "frozen",
        "multiAgentScenarioPath": "scen.json", "multiAgentScenarioHash": good,
        "variantConditions": [{"name": "v", "artifactPath": "missing.json",
                               "artifactHash": "deadbeef"}],
        "concepts": [], "conditions": [],
    })
    m = Manifest.load("s2", root=str(tmp_path))
    violations = m.verify(root=str(tmp_path))
    # scenario hash matches (no violation), but the variant file is missing.
    assert not any("scenario" in v for v in violations)
    assert any("variant 'v'" in v and "missing" in v for v in violations)


def test_content_hash_covers_whole_manifest():
    base = {"name": "s", "modelID": "m", "concepts": [], "conditions": []}
    h0 = Manifest.from_dict(dict(base)).content_hash()
    # Materially different studies must NOT collide.
    assert h0 != Manifest.from_dict({**base, "temperature": 0.7}).content_hash()
    assert h0 != Manifest.from_dict({**base, "maxTokens": 999}).content_hash()
    assert h0 != Manifest.from_dict(
        {**base, "conditions": [{"name": "c", "slots": []}]}).content_hash()
    assert h0 != Manifest.from_dict({**base, "taskPromptsHash": "abc"}).content_hash()
    # Volatile freeze stamps don't change the content hash.
    assert h0 == Manifest.from_dict(
        {**base, "status": "frozen", "freezeHash": "x", "frozenAt": "now"}).content_hash()


def test_validation_scope_hash_ignores_conditions():
    base = {"name": "s", "modelID": "m", "concepts": [], "conditions": []}
    h0 = Manifest.from_dict(dict(base)).validation_scope_hash()
    # Adding a condition must NOT change the validation scope.
    assert h0 == Manifest.from_dict(
        {**base, "conditions": [{"name": "c", "slots": []}]}).validation_scope_hash()
    # But a different concept pin does.
    assert h0 != Manifest.from_dict(
        {**base, "concepts": [{"name": "x", "stimulusSetHash": "h",
                               "options": {"method": "meanDifference"}}]}).validation_scope_hash()


def test_verify_variant_base_model_mismatch(tmp_path):
    _write_manifest(tmp_path, {
        "name": "vm", "modelID": "org/m", "concepts": [], "conditions": [],
        "variantConditions": [{"name": "v", "artifactPath": "", "artifactHash": "",
                               "artifact": {"baseModelID": "other/model"}}]})
    violations = Manifest.load("vm", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("uses other/model" in v for v in violations)


def test_verify_multiagent_needs_scenario(tmp_path):
    _write_manifest(tmp_path, {"name": "ma", "modelID": "m", "studyKind": "multiAgent",
                               "concepts": [], "conditions": []})
    violations = Manifest.load("ma", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("needs a pinned scenario" in v for v in violations)


def test_verify_no_concepts_or_variants(tmp_path):
    _write_manifest(tmp_path, {"name": "empty", "modelID": "m",
                               "concepts": [], "conditions": []})
    violations = Manifest.load("empty", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("no concepts or variants" in v for v in violations)


def test_frozen_drift_only_for_server_frozen():
    base = {"name": "s", "modelID": "m", "status": "frozen",
            "concepts": [{"name": "c", "stimulusSetHash": "h",
                          "options": {"method": "meanDifference"}}], "conditions": []}
    server = Manifest.from_dict({**base, "frozenBy": "server", "freezeHash": "tampered"})
    # Server-frozen with a freezeHash != content → drift reported.
    assert any("changed after freeze" in v for v in server.verify())
    intact = Manifest.from_dict({**base, "frozenBy": "server"})
    intact.raw["freezeHash"] = intact.content_hash()
    assert not any("changed after freeze" in v for v in intact.verify())
    # Swift-frozen (frozenBy absent): the server can't reproduce Swift's hash, so
    # it does NOT flag freeze-hash drift (verifies pinned inputs only).
    swift = Manifest.from_dict({**base, "freezeHash": "swifts-own-hash"})
    assert not any("changed after freeze" in v for v in swift.verify())


def test_legacy_manifest_defaults(tmp_path):
    _write_manifest(tmp_path, {"name": "old", "modelID": "m",
                               "concepts": [], "conditions": []})
    m = Manifest.load("old", root=str(tmp_path))
    assert m.study_kind == "modelOutput"
    assert m.variant_conditions == [] and m.evaluation is None


def test_verify_ignores_carried_model_output_config_on_multiagent(tmp_path):
    """Type-switch preservation (2026-07-19): a multi-agent study CARRIES
    stale model-output configuration (concepts, task prompts, agents) —
    none of it is operative, so none of it may block verification. The
    scenario pin is still enforced."""
    scen = tmp_path / "scen.json"
    scen.write_text("payload", encoding="utf-8")
    good = hashlib.sha256(b"payload").hexdigest()
    _write_manifest(tmp_path, {
        "name": "ma-carried", "modelID": "m", "status": "draft",
        "studyKind": "multiAgent",
        "multiAgentScenarioPath": "scen.json", "multiAgentScenarioHash": good,
        "concepts": [{"name": "ghost", "stimulusSetHash": "00",
                      "options": {"method": "meanDifference"}}],
        "taskPromptsFile": "prompts/absent.jsonl", "taskPromptsHash": "11",
        "variantConditions": [{"name": "v", "artifactPath": "missing.json",
                               "artifactHash": "deadbeef"}],
        "conditions": [],
    })
    m = Manifest.load("ma-carried", root=str(tmp_path))
    assert m.verify(root=str(tmp_path)) == []

    # The pin surface skips the carried config too — a stale hidden
    # concept can never bloat a multi-agent evidence bundle.
    from steerlab_server.experiment.experiment_store import (
        freeze_advisories, pinned_input_entries)
    labels = [e.label for e in pinned_input_entries(m.raw, root=str(tmp_path))]
    assert any("scenario" in label for label in labels)
    assert not any("concept" in label or "task prompts" in label
                   or "variant" in label for label in labels)
    # …and the carried state is SAID at freeze time, not silently invisible.
    assert any("another study type" in a for a in freeze_advisories(
        m.raw, root=str(tmp_path)))


def test_model_output_pin_surface_skips_carried_scenario(tmp_path):
    from steerlab_server.experiment.experiment_store import pinned_input_entries
    labels = [e.label for e in pinned_input_entries({
        "name": "mo", "modelID": "m", "studyKind": "modelOutput",
        "multiAgentScenarioPath": "scen.json",
        "taskPromptsFile": "prompts/t.jsonl",
        "concepts": [], "conditions": [],
    }, root=str(tmp_path))]
    assert any("task prompts" in label for label in labels)
    assert not any("scenario" in label for label in labels)


def test_verify_flags_unknown_or_contradictory_study_type(tmp_path):
    """studyType contract (2026-07-19): LLM-authored JSON gets loud
    feedback — never a silent re-derive."""
    base = {"name": "st", "modelID": "m", "status": "draft",
            "concepts": [], "conditions": [],
            "variantConditions": [{"name": "v", "artifactPath": "",
                                   "artifactHash": "",
                                   "artifact": {"baseModelID": "m"}}]}
    _write_manifest(tmp_path, {**base, "studyType": "vibes"})
    violations = Manifest.load("st", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("unknown studyType" in v for v in violations)

    _write_manifest(tmp_path, {**base, "name": "st2", "studyType": "multiAgent"})
    violations = Manifest.load("st2", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("contradicts studyKind" in v for v in violations)

    _write_manifest(tmp_path, {**base, "name": "st3",
                               "studyType": "agentComparison"})
    violations = Manifest.load("st3", root=str(tmp_path)).verify(root=str(tmp_path))
    assert not any("studyType" in v for v in violations)


def test_concept_machinery_is_inert_for_declared_agent_comparison(tmp_path):
    """Second engineer round (2026-07-19): within model-output, a DECLARED
    compare-agents study without forward references carries its concept
    machinery inert — no concept violations, nothing concept-shaped on the
    pin surface; a forward reference flips it back on."""
    from steerlab_server.experiment.manifest import concept_machinery_operative
    from steerlab_server.experiment.experiment_store import (
        freeze_advisories, pinned_input_entries)
    base = {
        "name": "cmp", "modelID": "m", "status": "draft",
        "studyType": "agentComparison", "studyKind": "modelOutput",
        "concepts": [{"name": "ghost", "stimulusSetHash": "00",
                      "options": {"method": "meanDifference"}}],
        "conditions": [{"name": "ghost-4", "slots": [
            {"concept": "ghost", "layer": 4, "alpha": 2.0}]}],
        "variantConditions": [{"name": "v", "artifactPath": "",
                               "artifactHash": "",
                               "artifact": {"baseModelID": "m"}}],
    }
    assert not concept_machinery_operative(base)
    _write_manifest(tmp_path, base)
    violations = Manifest.load("cmp", root=str(tmp_path)).verify(root=str(tmp_path))
    assert not any("ghost" in v for v in violations)
    labels = [e.label for e in pinned_input_entries(base, root=str(tmp_path))]
    assert not any("concept" in label for label in labels)
    assert any("another study type" in a
               for a in freeze_advisories(base, root=str(tmp_path)))

    # A forward reference makes the machinery operative again.
    forward = dict(base)
    forward["variantConditions"] = base["variantConditions"] + [
        {"name": "ghost-agent", "artifactPath": "", "artifactHash": "",
         "fromPromotion": {"concept": "ghost"}}]
    assert concept_machinery_operative(forward)


def test_inert_conditions_with_no_agent_arms_is_a_problem(tmp_path):
    """The degenerate inert case (observed live 2026-08-11: the c20-*
    fan-out ran baseline only): a declared agentComparison with NO variant
    conditions but carrying injection conditions would silently drop every
    declared arm. One rule feeds verify, run start, and submission
    preflight."""
    from steerlab_server.experiment.manifest import inert_conditions_problem
    base = {
        "name": "inert", "modelID": "m", "status": "draft",
        "studyType": "agentComparison", "studyKind": "modelOutput",
        "concepts": [{"name": "ghost", "stimulusSetHash": "00",
                      "options": {"method": "meanDifference"}}],
        "conditions": [
            {"name": "ghost-4", "slots": [
                {"concept": "ghost", "layer": 4, "alpha": 2.0}]},
            {"name": "ghost-9", "slots": [
                {"concept": "ghost", "layer": 9, "alpha": 1.0}]}],
        "variantConditions": [],
    }
    problem = inert_conditions_problem(base)
    assert problem is not None
    assert "BASELINE ONLY" in problem
    assert "ghost-4" in problem and "ghost-9" in problem
    _write_manifest(tmp_path, base)
    violations = Manifest.load("inert", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("BASELINE ONLY" in v for v in violations)

    # With an agent arm the 2026-07-19 rule applies instead (agents run;
    # carried conditions are the mixed-arm/advisory surface, not this one).
    with_agent = {**base, "variantConditions": [
        {"name": "v", "artifactPath": "", "artifactHash": "",
         "artifact": {"baseModelID": "m"}}]}
    assert inert_conditions_problem(with_agent) is None

    # Operative machinery (no declaration, or declared conceptStudy) is
    # never this problem; nor is a baseline-only manifest with no
    # injection conditions to drop; nor a non-modelOutput study.
    undeclared = {k: v for k, v in base.items() if k != "studyType"}
    assert inert_conditions_problem(undeclared) is None
    assert inert_conditions_problem(
        {**base, "studyType": "conceptStudy"}) is None
    assert inert_conditions_problem(
        {**base, "conditions": [{"name": "baseline", "slots": []}]}) is None
    assert inert_conditions_problem(
        {**base, "studyKind": "multiAgent"}) is None

    # The shape that legally PROCEEDS (agent arms exist, machinery inert)
    # gets the structured run-start/config note instead (2026-08-11
    # follow-up: baseline+agents-only runs must be loud and
    # self-describing).
    from steerlab_server.experiment.manifest import inert_machinery_note
    note = inert_machinery_note(with_agent)
    assert note == {"declaredStudyType": "agentComparison",
                    "inertConditions": ["ghost-4", "ghost-9"],
                    "inertConcepts": ["ghost"]}
    assert inert_machinery_note(undeclared) is None
    assert inert_machinery_note({**with_agent, "conditions": [],
                                 "concepts": []}) is None


def test_mixed_hand_declared_arms_are_a_verify_violation(tmp_path):
    """Third engineer round (2026-07-19): when agent conditions exist the
    engines run agents only — a hand-declared injection condition would
    silently vanish, so verify refuses the mix. Sweep-stamped conditions
    (selection provenance) are exempt; inert machinery (declared
    comparison) is covered by the carried-config advisory instead."""
    base = {
        "name": "mix", "modelID": "m", "status": "draft",
        "studyType": "conceptStudy", "studyKind": "modelOutput",
        "concepts": [{"name": "ghost", "stimulusSetHash": "00",
                      "options": {"method": "meanDifference"}}],
        "conditions": [{"name": "ghost-4", "slots": [
            {"concept": "ghost", "layer": 4, "alpha": 2.0}]}],
        "variantConditions": [{"name": "v", "artifactPath": "",
                               "artifactHash": "",
                               "artifact": {"baseModelID": "m"}}],
    }
    _write_manifest(tmp_path, base)
    violations = Manifest.load("mix", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("mixed arm modes" in v and "ghost-4" in v for v in violations)

    recommended = {
        "name": "ghost-recommended",
        "slots": [{"concept": "ghost", "layer": 4, "alpha": 2.0}],
        "selection": {"sweepRun": "runs/x", "devPromptsHash": "00",
                      "winningCell": {"layer": 4, "alpha": 2.0},
                      "criterion": {"objective": {"metric": "markerDensity"}},
                      "metrics": {}}}
    # An identity-complete twin: injects the concept, names the SAME sweep
    # run, certifies the SAME winning cell. (The promotion certificate has
    # no concept field — concept identity is read from the artifact's
    # injections, which promote always writes.)
    twin_artifact = {
        "baseModelID": "m",
        "injections": [{"concept": "ghost", "vectorArtifactID": "v",
                        "layer": 4, "alpha": 2.0}],
        "promotion": {"experiment": "mix", "promotedBy": "criterion",
                      "sweepRun": "runs/x",
                      "winningCell": {"layer": 4, "alpha": 2.0}}}
    # Sixth round (rule CHANGED deliberately): a fromPromotion forward
    # reference no longer exempts a STAMPED condition — it binds to no
    # selection, so its future sweep may pick a different cell. The
    # refusal names the sweep run the recommendation is bound to.
    stamped = dict(base, name="mix2")
    stamped["conditions"] = [dict(recommended)]
    stamped["variantConditions"] = base["variantConditions"] + [
        {"name": "ghost-agent", "artifactPath": "", "artifactHash": "",
         "fromPromotion": {"concept": "ghost"}}]
    _write_manifest(tmp_path, stamped)
    violations = Manifest.load("mix2", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("ghost-recommended" in v
               and "not among this study's arms" in v
               and "bound to sweep run 'runs/x'" in v for v in violations)

    # A concrete agent whose promotion birth certificate carries the
    # COMPLETE identity (concept + sweep run + winning cell) exempts.
    stamped_concrete = dict(base, name="mix2b")
    stamped_concrete["conditions"] = [dict(recommended)]
    stamped_concrete["variantConditions"] = [
        {"name": "ghost-promoted", "artifactPath": "p", "artifactHash": "h",
         "artifact": json.loads(json.dumps(twin_artifact))}]
    _write_manifest(tmp_path, stamped_concrete)
    violations = Manifest.load(
        "mix2b", root=str(tmp_path)).verify(root=str(tmp_path))
    assert not any("mixed arm modes" in v for v in violations)
    assert not any("not among" in v for v in violations)

    # A stamped condition with ONLY an unrelated agent among the arms is
    # NOT exempt: the recommended cell's promoted agent would never
    # execute — refused, naming the concept.
    orphan = dict(base, name="mix2c")
    orphan["conditions"] = [dict(recommended)]
    _write_manifest(tmp_path, orphan)
    violations = Manifest.load(
        "mix2c", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("ghost-recommended" in v
               and "not among this study's arms" in v
               and "'ghost'" in v for v in violations)

    # A birth certificate for a DIFFERENT cell is not the twin either.
    wrong_cell_artifact = json.loads(json.dumps(twin_artifact))
    wrong_cell_artifact["promotion"]["winningCell"] = {"layer": 9,
                                                       "alpha": 2.0}
    wrong_cell = dict(base, name="mix2d")
    wrong_cell["conditions"] = [dict(recommended)]
    wrong_cell["variantConditions"] = [
        {"name": "ghost-promoted", "artifactPath": "p", "artifactHash": "h",
         "artifact": wrong_cell_artifact}]
    _write_manifest(tmp_path, wrong_cell)
    violations = Manifest.load(
        "mix2d", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("not among this study's arms" in v for v in violations)

    # Same cell, DIFFERENT concept injected: not the twin.
    wrong_concept_artifact = json.loads(json.dumps(twin_artifact))
    wrong_concept_artifact["injections"][0]["concept"] = "other"
    wrong_concept = dict(base, name="mix2e")
    wrong_concept["conditions"] = [dict(recommended)]
    wrong_concept["variantConditions"] = [
        {"name": "ghost-promoted", "artifactPath": "p", "artifactHash": "h",
         "artifact": wrong_concept_artifact}]
    _write_manifest(tmp_path, wrong_concept)
    violations = Manifest.load(
        "mix2e", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("not among this study's arms" in v for v in violations)

    # Same concept + cell but a DIFFERENT sweep run: not the twin (the
    # certificate binds the agent to another selection).
    wrong_run_artifact = json.loads(json.dumps(twin_artifact))
    wrong_run_artifact["promotion"]["sweepRun"] = "runs/y"
    wrong_run = dict(base, name="mix2f")
    wrong_run["conditions"] = [dict(recommended)]
    wrong_run["variantConditions"] = [
        {"name": "ghost-promoted", "artifactPath": "p", "artifactHash": "h",
         "artifact": wrong_run_artifact}]
    _write_manifest(tmp_path, wrong_run)
    violations = Manifest.load(
        "mix2f", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("not among this study's arms" in v for v in violations)

    # An EMPTY promotion sweepRun refuses too: a stamped selection always
    # names its run, so an unbound certificate proves nothing (the old
    # "when both carry one" hole).
    empty_run_artifact = json.loads(json.dumps(twin_artifact))
    empty_run_artifact["promotion"].pop("sweepRun")
    empty_run = dict(base, name="mix2g")
    empty_run["conditions"] = [dict(recommended)]
    empty_run["variantConditions"] = [
        {"name": "ghost-promoted", "artifactPath": "p", "artifactHash": "h",
         "artifact": empty_run_artifact}]
    _write_manifest(tmp_path, empty_run)
    violations = Manifest.load(
        "mix2g", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("not among this study's arms" in v for v in violations)

    # Seventh round (2026-07-20): the certificate proved what was CLAIMED,
    # not what RUNS. A certificate carrying the right identity while the
    # artifact INJECTS a different cell is not the twin — what executes is
    # not what the sweep selected.
    mismatched_injection = json.loads(json.dumps(twin_artifact))
    mismatched_injection["injections"][0]["layer"] = 9
    mismatched_injection["injections"][0]["alpha"] = 9.0
    inj_cell = dict(base, name="mix2h")
    inj_cell["conditions"] = [dict(recommended)]
    inj_cell["variantConditions"] = [
        {"name": "ghost-promoted", "artifactPath": "p", "artifactHash": "h",
         "artifact": mismatched_injection}]
    _write_manifest(tmp_path, inj_cell)
    violations = Manifest.load(
        "mix2h", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("not among this study's arms" in v for v in violations)

    # An EXTRA second injection is not the twin either: promote mints
    # exactly one, and a second vector means the arm executes a mix the
    # sweep never selected.
    extra_injection = json.loads(json.dumps(twin_artifact))
    extra_injection["injections"].append(
        {"concept": "stow", "vectorArtifactID": "w",
         "layer": 6, "alpha": 1.0})
    extra = dict(base, name="mix2i")
    extra["conditions"] = [dict(recommended)]
    extra["variantConditions"] = [
        {"name": "ghost-promoted", "artifactPath": "p", "artifactHash": "h",
         "artifact": extra_injection}]
    _write_manifest(tmp_path, extra)
    violations = Manifest.load(
        "mix2i", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("not among this study's arms" in v for v in violations)

    carried = dict(base, name="mix3", studyType="agentComparison")
    _write_manifest(tmp_path, carried)
    violations = Manifest.load("mix3", root=str(tmp_path)).verify(root=str(tmp_path))
    assert not any("mixed arm modes" in v for v in violations)

    # Fourth round: a well-shaped but INCONSISTENT selection block does
    # not exempt (the slot is not the winning cell) …
    forged = dict(base, name="mix4")
    forged["conditions"] = [{
        "name": "ghost-recommended",
        "slots": [{"concept": "ghost", "layer": 4, "alpha": 9.0}],
        "selection": {"sweepRun": "runs/x", "devPromptsHash": "00",
                      "winningCell": {"layer": 4, "alpha": 2.0},
                      "criterion": {"objective": {"metric": "markerDensity"}},
                      "metrics": {}}}]
    _write_manifest(tmp_path, forged)
    violations = Manifest.load("mix4", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("mixed arm modes" in v for v in violations)
    # … a junk selection dict does not exempt either …
    junk = dict(base, name="mix5")
    junk["conditions"] = [{"name": "ghost-4",
                           "slots": [{"concept": "ghost", "layer": 4,
                                      "alpha": 2.0}],
                           "selection": {"foo": "bar"}}]
    _write_manifest(tmp_path, junk)
    violations = Manifest.load("mix5", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("mixed arm modes" in v for v in violations)
    # … the canonical empty baseline IS exempt; a custom-named empty
    # condition is not.
    with_baseline = dict(base, name="mix6")
    with_baseline["conditions"] = [{"name": "baseline", "slots": []}]
    _write_manifest(tmp_path, with_baseline)
    violations = Manifest.load("mix6", root=str(tmp_path)).verify(root=str(tmp_path))
    assert not any("mixed arm modes" in v for v in violations)
    custom = dict(base, name="mix7")
    custom["conditions"] = [{"name": "my-baseline", "slots": []}]
    _write_manifest(tmp_path, custom)
    violations = Manifest.load("mix7", root=str(tmp_path)).verify(root=str(tmp_path))
    assert any("mixed arm modes" in v for v in violations)


def test_confirm_agent_is_a_legacy_alias_for_concept_study(tmp_path):
    """2026-07-19 fold-in: confirmation is the concept study's CONFIRM
    phase, not a peer type. The legacy declared value stays legal (shipped
    manifests, LLM packs) and canonicalizes to conceptStudy; a
    perturbation policy derives the same way; the machinery is operative."""
    from steerlab_server.experiment.manifest import (
        concept_machinery_operative, effective_study_type)
    aliased = {"name": "cf", "modelID": "m", "status": "draft",
               "studyType": "confirmAgent", "studyKind": "modelOutput",
               "concepts": [], "conditions": [],
               "variantConditions": [{"name": "v", "artifactPath": "",
                                      "artifactHash": "",
                                      "artifact": {"baseModelID": "m"}}]}
    assert effective_study_type(aliased) == "conceptStudy"
    assert concept_machinery_operative(aliased)
    _write_manifest(tmp_path, aliased)
    violations = Manifest.load("cf", root=str(tmp_path)).verify(root=str(tmp_path))
    assert not any("studyType" in v for v in violations)

    policy = {"name": "cf2", "modelID": "m", "status": "draft",
              "perturbationPolicy": {"concept": "fear"},
              "concepts": [], "conditions": [],
              "variantConditions": [{"name": "v", "artifactPath": "",
                                     "artifactHash": "",
                                     "artifact": {"baseModelID": "m"}}]}
    assert effective_study_type(policy) == "conceptStudy"
