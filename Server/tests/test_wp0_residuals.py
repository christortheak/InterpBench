"""The four residuals ``docs/WP0-AGENT-SURFACE-AUDIT.md`` §13 carried past the
WP0 close, server side.

Swift twin: ``Tests/ExperimentKitTests/DryRun2PunchListTests.swift`` (the
gate-5 punch-list suite the same fixes extend).

(a) is the one that matters. ``KNOWN_OUTCOME_INSTRUMENTS`` had ZERO production
readers: Swift enforced the vocabulary at DECLARATION (``set-instruments``
refuses at 64) and authoring is Mac-authority, so every manifest arriving here
arrives as BYTES and nothing checked them. Probed before the fix, with
``outcomeInstruments: ["sampledTxt"]``, this engine answered
``state: "ready"``, exit 0, ``run (0 generations)`` and a durable run
directory — a study that completed having measured nothing, and a
``nextAction`` inviting the caller to analyze it.
"""

import json
import os
import re

import pytest

from steerlab_server.experiment import (experiment_store as es,
                                        lifecycle_gates, tasks)
from steerlab_server.experiment.manifest import Manifest

REFERENCE_DOC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs", "CLI-REFERENCE.md")

# The refusal sentence, verbatim. It is the CROSS-ENGINE contract — the Swift
# twin asserts the same bytes — because the claim is the same claim.
UNKNOWN_INSTRUMENT_SENTENCE = (
    "outcomeInstruments declares 'sampledTxt', which this engine does not "
    "implement — the declared instruments are read by set membership, so an "
    "unrecognised value dispatches nothing and the study would complete "
    "having measured only the default sampled text. Known instruments: "
    "sampledText, answerTokenLogprob, choiceProbability, repeReaderScore, "
    "ordinalScale")


def _manifest(**extra):
    d = {"name": "typo", "modelID": "test/model", "concepts": [],
         "conditions": [], "taskPromptsFile": None}
    d.update(extra)
    return Manifest.from_dict(d)


# =============================================================================
# (a) FIREWALL — the outcome-instrument vocabulary is enforced before a run
# =============================================================================

def test_the_near_miss_typo_refuses_at_run_start():
    """One character. ``sampledTxt`` is not ``sampledText``, every downstream
    reader is a set-membership test, and the pre-fix consequence was a
    completed study with zero generations reported as ready."""
    manifest = _manifest(outcomeInstruments=["sampledTxt"])
    assert es.unknown_outcome_instruments(manifest.raw) == ["sampledTxt"]
    with pytest.raises(RuntimeError) as excinfo:
        tasks._check_response_formats(
            manifest, [{"id": "i1", "prompt": "hello"}])
    exc = excinfo.value
    assert str(exc) == UNKNOWN_INSTRUMENT_SENTENCE
    # The gate is `responseFormat` — the one gate whose subject is
    # outcomeInstruments and whose repair is set-instruments. No new id: its
    # existing family is "a declared instrument that would silently produce
    # zero records" (the zero-item rules, 2026-08-06).
    assert lifecycle_gates.gate_of(exc) == lifecycle_gates.RESPONSE_FORMAT
    repair = lifecycle_gates.repair_of(exc)
    # Authoring is Mac-authority, so this engine's repair names the Mac
    # binary — exactly like the no-rubric sentence (audit §12.1).
    assert repair.startswith("steerlab-cli experiment set-instruments typo")
    for instrument in es.KNOWN_OUTCOME_INSTRUMENTS:
        assert instrument in repair
        assert instrument in str(exc)


def test_every_known_instrument_still_passes():
    """A vocabulary check, not a new restriction on what may be declared."""
    for instrument in es.KNOWN_OUTCOME_INSTRUMENTS:
        assert es.unknown_outcome_instrument_problem(
            {"outcomeInstruments": [instrument]}) is None
    assert es.unknown_outcome_instrument_problem(
        {"outcomeInstruments": list(es.KNOWN_OUTCOME_INSTRUMENTS)}) is None
    # An undeclared list is the engine default (sampled text), not a violation.
    assert es.unknown_outcome_instrument_problem({}) is None
    assert es.unknown_outcome_instrument_problem(
        {"outcomeInstruments": []}) is None


def test_every_unknown_instrument_is_named_in_declaration_order():
    problem = es.unknown_outcome_instrument_problem(
        {"outcomeInstruments": ["zzz", "sampledText", "aaa"]})
    assert problem is not None
    assert "'zzz', 'aaa'" in problem


def test_the_preflight_refuses_before_the_model_loads(tmp_path):
    """The refusal must land in ``_response_format_preflight`` — ahead of the
    load — not only in ``_run_impl``. On a cluster the difference is the queue
    wait plus a multi-minute 27B load, on every shard of a fan-out."""
    root = str(tmp_path)
    items = os.path.join(root, "prompts", "tasks", "items.jsonl")
    os.makedirs(os.path.dirname(items))
    with open(items, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "i1", "prompt": "hello"}) + "\n")
    manifest = _manifest(outcomeInstruments=["sampledTxt"],
                         taskPromptsFile="prompts/tasks/items.jsonl")
    with pytest.raises(RuntimeError) as excinfo:
        tasks._response_format_preflight(manifest, None, root)
    assert lifecycle_gates.gate_of(excinfo.value) == \
        lifecycle_gates.RESPONSE_FORMAT


# =============================================================================
# (c) The pinned/ snapshot follows the PIN, not the concept machinery
# =============================================================================

def _corpus(root):
    path = os.path.join(root, "prompts", "neutral", "corpus.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "A neutral sentence."}) + "\n")
    return path


@pytest.mark.parametrize("extra", [
    # A compare-agents study: no concepts, no injection conditions — the
    # concept machinery is inert — but its promoted agents' α is in norm units.
    {"studyType": "agentComparison"},
    # A panel, which is not a modelOutput study at all.
    {"studyKind": "multiAgent"},
])
def test_a_pinned_neutral_corpus_is_snapshotted_without_concept_machinery(
        tmp_path, extra):
    root = str(tmp_path)
    source = _corpus(root)
    d = {"name": "agents", "modelID": "test/model", "concepts": [],
         "conditions": [], "neutralCorpusHash": "b" * 64}
    d.update(extra)
    os.makedirs(os.path.join(root, "experiments", "agents"))
    with open(os.path.join(root, "experiments", "agents", "experiment.json"),
              "w", encoding="utf-8") as handle:
        json.dump(d, handle)

    # The pin SURFACE is deliberately unchanged — this widening is
    # snapshot-only, so no study that freezes today stops freezing.
    assert source not in es.pinned_input_paths(d, root)
    assert source in es._snapshot_sources(d, root)

    es._snapshot_pinned_inputs("agents", d, root)
    snapshot = os.path.join(root, "experiments", "agents", "pinned",
                            "prompts", "neutral", "corpus.jsonl")
    assert os.path.exists(snapshot)
    with open(snapshot, "rb") as a, open(source, "rb") as b:
        assert a.read() == b.read(), "the snapshot must be BYTES"


def test_an_unpinned_neutral_corpus_is_not_snapshotted(tmp_path):
    """Copying an unpinned corpus would claim a pin that does not exist."""
    root = str(tmp_path)
    source = _corpus(root)
    d = {"name": "nopin", "modelID": "test/model", "concepts": [],
         "conditions": [], "studyType": "agentComparison"}
    assert source not in es._snapshot_sources(d, root)


# =============================================================================
# (d) The foreign-substrate repair names an engine that HAS the verb
# =============================================================================

def _swift_verb_roster() -> set[str]:
    """Every ``steerlab-cli experiment <verb>`` the Mac CLI declares, read out
    of the GENERATED ``swift-*`` regions of ``docs/CLI-REFERENCE.md``.

    Those regions are produced from the Swift declarative verb table and
    guarded against drift by ``CLIReferenceGenerationTests`` (WP0 step 11), so
    they are the closest thing this engine has to a machine-readable roster of
    the other one."""
    with open(REFERENCE_DOC, encoding="utf-8") as handle:
        document = handle.read()
    regions = re.findall(
        r"<!-- GENERATED:swift-[^ ]* BEGIN -->(.*?)<!-- GENERATED:swift-",
        document, flags=re.DOTALL)
    verbs = set()
    for region in regions:
        verbs.update(re.findall(
            r"steerlab-cli experiment ([a-z][a-z-]*)", region))
    assert "analyze" in verbs, "the roster parse found nothing — doc changed?"
    return verbs


def test_foreign_substrate_repair_names_an_engine_that_has_the_verb(tmp_path):
    """WP0 residual (d). ``tasks._require_source_epoch`` composes
    ``steerlab-cli experiment <verb>`` for a run from the other engine without
    asking whether that engine has the verb. Correct today — there are exactly
    two substrates and all three measurement verbs are twinned — and this test
    is what stops a third engine, or a server-only verb, from inheriting the
    assumption silently. Deliberately NOT engine-capability negotiation."""
    roster = _swift_verb_roster()
    root = str(tmp_path)
    manifest = Manifest.from_dict(
        {"name": "study", "modelID": "org/m", "concepts": [], "conditions": []})
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-study-run")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as h:
        json.dump({"schemaVersion": 2, "runType": "run",
                   "substrate": "swift-mlx",
                   "experimentHash": manifest.content_hash()}, h)

    for verb in ("analyze", "evaluate", "rescore-style"):
        with pytest.raises(RuntimeError) as excinfo:
            tasks._require_source_epoch(
                verb, "study", manifest, run_dir, allow_unverified_epoch=False)
        assert lifecycle_gates.gate_of(excinfo.value) == \
            lifecycle_gates.MANIFEST_EPOCH
        repair = lifecycle_gates.repair_of(excinfo.value)
        assert repair.startswith(f"steerlab-cli experiment {verb} study")
        assert verb in roster, (
            f"the repair sends the caller to 'steerlab-cli experiment {verb}', "
            "which the Mac CLI's generated reference does not declare")
