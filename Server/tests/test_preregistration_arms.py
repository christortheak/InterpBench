"""The preregistration's Conditions section lists every arm the run executes.

Field report 2026-10-19: a compare-agents study whose arms were attached as
agents (variant conditions, beside the implicit baseline) froze with an empty
Conditions section, because the section printed declared injection
conditions only. It now names each arm a run executes, in run order: the
baseline (marked when the run adds it implicitly), the operative injection
conditions, each attached agent with its base model, interventions and
pinned artifact digest, SAE latent arms, and a panel's scenario arms. Declared
arms the run never executes are named after the list instead of in it.

An injection condition's slot line also says what the run does with the slot:
an ablating slot reads ``<concept> ablated λ=<λ> at every layer`` (so λ=1
ablation no longer reads like α=1 steering at one layer), and a steering slot
in a condition whose band widens it says by how much.

``Tests/Fixtures/cross-engine/preregistration-arms.json`` holds the section
this engine writes for each study shape; the Swift twin
(``PreregistrationArmsTests.swift``) renders the same manifests and must
match it byte for byte.
"""

import hashlib
import json
import os
import re

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment.manifest import (Manifest,
                                                 effective_condition_names)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(REPO, "Tests", "Fixtures", "cross-engine",
                       "preregistration-arms.json")
REGENERATE = ("stale fixture — re-run preregistration_arms() in "
              "scripts/regenerate-cross-engine-fixtures.py and commit")
SHAPES = {
    "compare-agents", "compare-agents-carrying-conditions",
    "agent-interventions", "concept-study-implicit-baseline",
    "concept-study-declared-baseline", "concept-study-ablation",
    "concept-study-band-width", "sae-latent-arms", "multi-agent-panel",
    "multi-agent-panel-without-baseline",
}


def _cases():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def _case(label):
    return next(case for case in _cases() if case["label"] == label)


def _section(text):
    start = text.index("## Conditions\n\n") + len("## Conditions\n\n")
    return text[start:text.index("\n\n## ", start)]


def _render(root, manifest):
    """The preregistration this engine writes for ``manifest``."""
    directory = os.path.join(str(root), "experiments", manifest["name"])
    os.makedirs(directory, exist_ok=True)
    es._write_preregistration(json.loads(json.dumps(manifest)), str(root))
    with open(os.path.join(directory, es.PREREG_FILENAME),
              encoding="utf-8") as handle:
        return handle.read()


def test_the_cross_engine_fixture_is_current(tmp_path):
    cases = _cases()
    assert {case["label"] for case in cases} >= SHAPES, \
        "the study-shape coverage shrank"
    for case in cases:
        assert _section(_render(tmp_path, case["manifest"])) \
            == case["conditionsSection"], f"{case['label']}: {REGENERATE}"


def test_a_compare_agents_study_lists_the_baseline_and_every_agent(tmp_path):
    manifest = _case("compare-agents")["manifest"]
    french, hungry = manifest["variantConditions"]
    assert manifest["conditions"] == []
    assert _section(_render(tmp_path, manifest)).split("\n") == [
        "- **baseline**: none (baseline, run implicitly)",
        "- **french-l28-a15** (agent): french@L28 α=0.15; base model "
        "test/model @ `abc123`; pinned artifact "
        "`experiments/prereg/pinned/variant-french-l28-a15.json` "
        f"(SHA-256 `{french['artifactHash']}`)",
        "- **hungry-l17-a10** (agent): hungry@L17 α=0.1; base model "
        "test/model @ `abc123`; pinned artifact "
        "`experiments/prereg/pinned/variant-hungry-l17-a10.json` "
        f"(SHA-256 `{hungry['artifactHash']}`)",
    ]


def test_carried_arms_that_never_run_are_named_outside_the_list(tmp_path):
    section = _section(_render(
        tmp_path, _case("compare-agents-carrying-conditions")["manifest"]))
    assert "- **fear-mid**" not in section
    assert section.endswith(
        "\n\nDeclared but not run by this study design: fear-mid.")
    panel = _section(_render(tmp_path, _case("multi-agent-panel")["manifest"]))
    assert panel.endswith("\n\nDeclared but not run by this study design: "
                          "fear-mid, french-l28-a15.")


def test_a_concept_study_reads_as_before_apart_from_the_implicit_baseline(
        tmp_path):
    declared = _section(_render(
        tmp_path, _case("concept-study-declared-baseline")["manifest"]))
    assert declared == ("- **baseline**: none (baseline)\n"
                        "- **fear-mid**: fear@L14 α=0.8")
    implicit = _section(_render(
        tmp_path, _case("concept-study-implicit-baseline")["manifest"]))
    assert implicit == ("- **baseline**: none (baseline, run implicitly)\n"
                        "- **fear-mid**: fear@L14 α=0.8\n"
                        "- **fear-mid-random** [randomMatchedNorm]: "
                        "fear@L14 α=0.8")


def test_an_ablating_slot_never_reads_like_a_steering_slot(tmp_path):
    assert _section(_render(
        tmp_path, _case("concept-study-ablation")["manifest"])).split("\n") == [
        "- **baseline**: none (baseline, run implicitly)",
        "- **fear-mid**: fear@L14 α=0.8",
        "- **fear-steer-1**: fear@L14 α=1",
        "- **fear-ablate**: fear ablated λ=1 at every layer",
        "- **fear-ablate-random** [randomDirectionAblation]: "
        "fear ablated λ=1 at every layer",
    ]


def test_a_band_wider_than_one_layer_is_stated_on_each_steering_slot(
        tmp_path):
    assert _section(_render(
        tmp_path, _case("concept-study-band-width")["manifest"])).split("\n") \
        == [
        "- **baseline**: none (baseline, run implicitly)",
        "- **fear-band-3**: fear@L14 α=0.8 (band width 3: ±1 layer)",
        "- **fear-band-2**: fear@L14 α=0.8 (band width 2: ±1 layer)",
        "- **fear-calm-band-5**: fear@L14 α=0.8 (band width 5: ±2 layers), "
        "calm@L10 α=0.25 (band width 5: ±2 layers)",
        "- **fear-band-3-calm-ablated**: fear@L14 α=0.8 "
        "(band width 3: ±1 layer), calm ablated λ=0.5 at every layer",
        "- **fear-band-0**: fear@L14 α=0.8",
    ]


def test_each_slot_line_names_the_layers_the_run_injects():
    """The wording is read off the run's own cell builder: the layers a slot
    line names are the layers ``condition_injections`` injects it at."""
    import warnings

    from steerlab_server.experiment.condition_execution import \
        condition_injections
    from steerlab_server.experiment.vector_materialization import \
        ConceptVectorBundle
    from steerlab_server.steering.vector_store import ConceptVectors

    layer_count = 32
    bundle = ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[3.0, 4.0, 0.0]] * layer_count),
        residual_norm_per_layer=[1.0] * layer_count,
        residual_norm_source="neutral-corpus", stimulus_hash="h" * 64,
        neutral_mean_per_layer=[[0.0, 0.0, 1.0]] * layer_count)
    slot_pattern = re.compile(
        r"(\w+)(?: ablated λ=\S+ at (every layer)"
        r"|@L(\d+) α=\S+(?: \(band width -?\d+: ±(\d+) layers?\))?)")
    for label in ("concept-study-ablation", "concept-study-band-width"):
        case = _case(label)
        lines = {re.match(r"- \*\*(.+?)\*\*", line).group(1): line
                 for line in case["conditionsSection"].split("\n")}
        for condition in Manifest.from_dict(case["manifest"]).conditions:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cells = condition_injections(
                    condition, {"fear": bundle, "calm": bundle})
            stated = slot_pattern.findall(
                lines[condition.name].split(": ", 1)[1])
            assert [concept for concept, *_ in stated] \
                == [slot.concept for slot in condition.slots], condition.name
            for concept, every, layer, half in stated:
                injected = sorted(cell.layer for cell in cells
                                  if cell.concept == concept)
                if every:
                    named = list(range(layer_count))
                else:
                    width = int(half or 0)
                    named = list(range(int(layer) - width,
                                       int(layer) + width + 1))
                assert injected == named, (label, condition.name, concept)


def test_a_generated_summary_stating_ablations_is_still_provably_ours(
        tmp_path):
    manifest = _case("concept-study-ablation")["manifest"]
    _render(tmp_path, manifest)
    path = os.path.join(str(tmp_path), "experiments", manifest["name"],
                        es.PREREG_FILENAME)
    assert es._preregistration_is_generated(path, {})


def test_the_listed_arms_are_the_runs_record_vocabulary():
    """The section and the run select arms by one rule, so the names it lists
    are the names records will carry (a panel's order is the engine's own)."""
    for case in _cases():
        listed = re.findall(r"^- \*\*(.+?)\*\*", case["conditionsSection"],
                            re.MULTILINE)
        manifest = case["manifest"]
        run = effective_condition_names(Manifest.from_dict(manifest))
        if manifest["studyKind"] == "modelOutput":
            assert listed == run, case["label"]
        else:
            assert sorted(listed) == sorted(run), case["label"]


def test_a_generated_summary_listing_agents_is_still_provably_ours(tmp_path):
    """The destination rule's structural fallback (no stamps) still reads a
    summary that lists agents and carried arms as generated, so a later
    freeze may replace it — and an edited copy as authored."""
    manifest = _case("compare-agents-carrying-conditions")["manifest"]
    _render(tmp_path, manifest)
    path = os.path.join(str(tmp_path), "experiments", manifest["name"],
                        es.PREREG_FILENAME)
    assert es._preregistration_is_generated(path, {})
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\nWe commit to the paired estimator.\n")
    assert not es._preregistration_is_generated(path, {})


# --- the freeze ---------------------------------------------------------------

def _agent_condition(root, name, concept, layer, alpha):
    """An agent pinned the way `experiment attach-agent` pins it."""
    artifact = {"name": name, "baseModelID": "org/m", "baseRevision": "abc",
                "adapters": [], "temperature": 0.0,
                "promptMode": "chatAssistant",
                "injections": [{"concept": concept, "layer": layer,
                                "alpha": alpha,
                                "vectorArtifactID": f"runs/s-sweep/{concept}"}]}
    rel = f"runs/model-variants/{name}/model-variant.json"
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=1)
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {"name": name, "artifactPath": rel, "artifactHash": digest,
            "artifact": artifact}


def _agent_study(tmp_path, name):
    from steerlab_server.experiment import battery as battery_mod

    root = str(tmp_path)
    es.create(name, model_id="org/m", revision="abc", root=root)
    d = es.load_raw(name, root)
    d["studyType"] = "agentComparison"
    d["variantConditions"] = [
        _agent_condition(root, "french-l28-a15", "french", 28, 0.15),
        _agent_condition(root, "hungry-l17-a10", "hungry", 17, 0.1)]
    es.save_raw(d, root)
    scope = Manifest.load(name, root=root).validation_scope_hash()
    battery_hash = battery_mod.live_hash(battery_mod.DEFAULT_BATTERY_FILE, root)
    rundir = os.path.join(root, "runs", f"v-exp-{name}-validate")
    os.makedirs(rundir)
    with open(os.path.join(rundir, "validation-evidence.json"), "w") as handle:
        json.dump({"schemaVersion": 1, "task": "validate",
                   "substrate": "python-hf-transformers",
                   "validationScopeHash": scope,
                   "batteryResults": [
                       {"condition": condition, "batteryHash": battery_hash,
                        "total": 4, "correct": 4, "accuracy": 1.0}
                       for condition in ("baseline", "french-l28-a15",
                                         "hungry-l17-a10")]}, handle)
    with open(os.path.join(rundir, "validation-report.json"), "w") as handle:
        json.dump({"concepts": {}}, handle)
    return root


def _agent_lines(variants):
    return [f"- **{vc['name']}** (agent): "
            f"{vc['artifact']['injections'][0]['concept']}@L"
            f"{vc['artifact']['injections'][0]['layer']} "
            f"α={vc['artifact']['injections'][0]['alpha']}; base model org/m "
            f"@ `abc`; pinned artifact `{vc['artifactPath']}` "
            f"(SHA-256 `{vc['artifactHash']}`)" for vc in variants]


def test_freeze_writes_every_agent_arm_into_the_preregistration(tmp_path):
    root = _agent_study(tmp_path, "agents")
    frozen = es.freeze("agents", force=False, root=root)
    # Freeze re-points each agent at its snapshot under pinned/.
    variants = frozen["variantConditions"]
    assert all(vc["artifactPath"].startswith("experiments/agents/pinned/")
               for vc in variants)
    path = os.path.join(root, "experiments", "agents", es.PREREG_FILENAME)
    with open(path, encoding="utf-8") as handle:
        written = handle.read()
    assert _section(written).split("\n") == [
        "- **baseline**: none (baseline, run implicitly)",
        *_agent_lines(variants)]
    assert es.PREREG_AUTHORED_HASH_KEY not in frozen
    assert frozen[es.PREREG_GENERATED_HASH_KEY] == hashlib.sha256(
        written.encode("utf-8")).hexdigest()


def test_freeze_keeps_an_authored_preregistration_and_lists_agents_beside_it(
        tmp_path):
    root = _agent_study(tmp_path, "authored")
    directory = os.path.join(root, "experiments", "authored")
    authored = ("# Analysis preregistration\n\nWe compare each agent with "
                "the unsteered model on every prompt.\n")
    with open(os.path.join(directory, es.PREREG_FILENAME), "w",
              encoding="utf-8") as handle:
        handle.write(authored)
    frozen = es.freeze("authored", force=False, root=root)
    with open(os.path.join(directory, es.PREREG_FILENAME),
              encoding="utf-8") as handle:
        assert handle.read() == authored
    assert frozen[es.PREREG_AUTHORED_HASH_KEY] == hashlib.sha256(
        authored.encode("utf-8")).hexdigest()
    variants = frozen["variantConditions"]
    with open(os.path.join(directory, es.PREREG_FROZEN_SETTINGS_FILENAME),
              encoding="utf-8") as handle:
        settings = handle.read()
    assert _section(settings).split("\n") == [
        "- **baseline**: none (baseline, run implicitly)",
        *_agent_lines(variants)]
