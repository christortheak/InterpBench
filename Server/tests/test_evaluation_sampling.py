"""``experiment set-evaluation-sampling`` — the manifest's
``evaluationSampling`` declaration, and the demotion of the evaluate sample
flags to a cross-check on a study that carries one.

THE RULING (review round 12, finding 4). The seeded evaluate subsample shipped
as CLI flags plus run stamps. A stamp records what HAPPENED, and
"preregistered" is a claim about what was decided BEFORE anything ran — a
claim that has to live in the artifact chain to be evidence. The reviewer
asked for a frozen, hashed design document; that does not fit this house's
flow, because judged re-measurement deliberately runs on never-frozen
duplicates. The adapted remedy keeps the substance: the design becomes a DRAFT
MANIFEST DECLARATION, and every run stamps the manifest snapshot into its own
``experiment.json``. ``test_the_declaration_lands_in_the_runs_own_snapshot``
below is the proof — it is the assertion the whole feature exists for.

Swift twin: ``Tests/ExperimentKitTests/EvaluationSamplingDeclarationTests``.
Every refusal sentence here is the Swift store's, byte for byte, held by the
shared literal fixture at the foot of this file — the convention
``test_evaluate_subsample`` established for the flag refusals.
"""

import json
import os

import pytest

from steerlab_server import client_cli
from steerlab_server.experiment import evaluate_subsample as subsample
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest

MAC = "steerlab-cli"


def _workspace(tmp_path) -> str:
    root = str(tmp_path)
    es.create("demo", model_id="org/m", root=root)
    return root


# =============================================================================
# The declaration
# =============================================================================


def test_the_declaration_stores_the_design_and_derives_its_rule(tmp_path):
    """The rule is DERIVED, not typed — the same guarantee
    ``parserRegistryHash`` carries, and the reason no surface has a
    ``--rule``."""
    root = _workspace(tmp_path)
    document = es.declare_evaluation_sampling("demo", "2400", "0x2a",
                                              root=root)
    assert document["evaluationSampling"] == {
        "rule": subsample.RULE,
        "samplePerCondition": 2400,
        "sampleSeed": "0x000000000000002a",
    }
    # …and it round-trips through the decoder every later verb starts with.
    manifest = Manifest.from_dict(es.load_raw("demo", root))
    assert manifest.raw["evaluationSampling"]["samplePerCondition"] == 2400


def test_the_stored_seed_is_the_canonical_spelling(tmp_path):
    """A decimal seed means the decimal a researcher typed; the STORED
    spelling is the canonical hex, because JSON has no unsigned 64-bit
    integer and a decimal a reader's parser rounds no longer redraws its own
    subsample."""
    root = _workspace(tmp_path)
    design = es.declare_evaluation_sampling(
        "demo", "40", "1234", root=root)["evaluationSampling"]
    assert design["sampleSeed"] == "0x00000000000004d2"


def test_the_empty_string_clears_the_declaration(tmp_path):
    root = _workspace(tmp_path)
    es.declare_evaluation_sampling("demo", "2400", "0x2a", root=root)
    for clearing in ("", "   ", None):
        es.declare_evaluation_sampling("demo", "2400", "0x2a", root=root)
        document = es.declare_evaluation_sampling(
            "demo", clearing, clearing, root=root)
        assert "evaluationSampling" not in document
        assert "evaluationSampling" not in es.load_raw("demo", root)


def test_half_a_declaration_refuses_and_writes_nothing(tmp_path):
    """Both halves or neither, INSIDE the declaration exactly as at the flags:
    a design nobody can redraw is not a preregistration, and a seed with no
    size is a stamp on a design it did not shape."""
    root = _workspace(tmp_path)
    with pytest.raises(es.MeasurementDeclarationError) as no_seed:
        es.declare_evaluation_sampling("demo", "2400", "", root=root)
    assert str(no_seed.value) == (
        "the sampling design named 2400 record(s) per condition with no seed "
        "— a subsample nobody can redraw is not a preregistration, so the "
        "declaration refuses rather than choosing a seed for you")
    assert "evaluationSampling" not in es.load_raw("demo", root)

    with pytest.raises(es.MeasurementDeclarationError) as no_size:
        es.declare_evaluation_sampling("demo", "", "0x2a", root=root)
    assert str(no_size.value) == (
        "the sampling design named seed 0x2a with no per-condition size — "
        "with no size the full corpus is coded, and the seed would be stamped "
        "on a design it did not shape")
    assert "evaluationSampling" not in es.load_raw("demo", root)


@pytest.mark.parametrize("raw", ["0", "-3", "2400.0", "many"])
def test_a_malformed_size_refuses(tmp_path, raw):
    root = _workspace(tmp_path)
    with pytest.raises(es.MeasurementDeclarationError) as caught:
        es.declare_evaluation_sampling("demo", raw, "0x2a", root=root)
    assert str(caught.value) == (
        "the sampling design's samplePerCondition must be a whole number of "
        f"records of at least 1, not '{raw}' — a subsample of zero records is "
        "a design nobody can report")
    assert "evaluationSampling" not in es.load_raw("demo", root)


@pytest.mark.parametrize("raw", ["zz", "-1", "0x10000000000000000"])
def test_a_malformed_seed_refuses(tmp_path, raw):
    root = _workspace(tmp_path)
    with pytest.raises(es.MeasurementDeclarationError) as caught:
        es.declare_evaluation_sampling("demo", "40", raw, root=root)
    assert str(caught.value) == (
        f"the sampling design's sampleSeed '{raw}' is not a 64-bit unsigned "
        "number — a seed is a decimal integer, or hexadecimal with or without "
        "a '0x' prefix, of at most 16 hex digits (the leading 16 of a digest "
        "are a fine seed, written down as such)")


def test_the_declaration_is_a_client_verb_taking_no_rule(tmp_path):
    """The rule is the ENGINE's, so it can never be typed. Structural,
    because the guarantee is the ABSENCE of a flag — the same assertion
    ``set-parser`` carries about its registry hash."""
    spec = next(s for s in client_cli.CLIENT_VERB_SPECS
                if s.family == "experiment"
                and s.verb == "set-evaluation-sampling")
    assert not spec.value_flags
    assert not spec.boolean_flags
    assert "--rule" not in spec.purpose
    assert "rule" not in spec.positional


def test_the_client_verb_declares_and_echoes_the_flat_shape(tmp_path):
    """The client's echo carries the Swift verb's flat result keys exactly, so
    an agent reads the same fields after either spelling. `rule` is echoed in
    FULL: it is the derivation a reader recomputes the membership from."""
    root = _workspace(tmp_path)
    outcome = client_cli.main(
        ["experiment", "set-evaluation-sampling", "demo", "2400", "0x2a",
         "--root", root, "--json"])
    assert outcome == 0
    document = es.load_raw("demo", root)
    assert document["evaluationSampling"]["rule"] == subsample.RULE

    # A half declaration refuses at 64 with nothing written.
    assert client_cli.main(
        ["experiment", "set-evaluation-sampling", "demo", "2400",
         "--root", root, "--json"]) == 64


# =============================================================================
# verify(): what a desk can check, and what it deliberately cannot
# =============================================================================


def test_verify_checks_the_design_but_never_the_population(tmp_path):
    """The declare-time/run-time split, asserted as a split. An ``n`` far
    above anything any run could hold verifies CLEAN: at verify time the
    source run this design will be drawn from need not exist, and usually does
    not, since declaring before running is the point."""
    root = _workspace(tmp_path)
    es.declare_evaluation_sampling("demo", "1000000", "0x2a", root=root)
    raw = es.load_raw("demo", root)
    assert subsample.declaration_violations(raw["evaluationSampling"]) == []

    # A rule from another version IS a desk finding: the version marker exists
    # so a moved rule is visible rather than silent.
    moved = subsample.declaration_violations(
        dict(raw["evaluationSampling"],
             rule="stratifiedByPromptID/v0 — something else"))
    assert len(moved) == 1
    assert moved[0].startswith(
        "evaluationSampling.rule is not the draw rule this build derives")

    broken = subsample.declaration_violations(
        {"rule": subsample.RULE, "samplePerCondition": 0,
         "sampleSeed": "not-a-seed"})
    assert len(broken) == 2
    assert any("samplePerCondition must be a whole number" in v
               for v in broken)
    assert any("sampleSeed 'not-a-seed' is not a 64-bit unsigned" in v
               for v in broken)

    # Not an object at all.
    assert subsample.declaration_violations(["nope"]) == [
        "evaluationSampling must be an object holding samplePerCondition, "
        "sampleSeed and rule"]


def test_an_undeclared_study_gains_nothing_at_verify(tmp_path):
    """ABSENT = no declaration = no violations, so every manifest written
    before this existed verifies exactly as it did."""
    root = _workspace(tmp_path)
    assert subsample.declaration_violations(None) == []
    manifest = Manifest.from_dict(es.load_raw("demo", root))
    assert not any("evaluationSampling" in v
                   for v in manifest.verify(root=root))


# =============================================================================
# The flags become a cross-check
# =============================================================================


def _declared(n=7, seed="0x000000000000002a"):
    return subsample.declared_request(
        {"rule": subsample.RULE, "samplePerCondition": n,
         "sampleSeed": seed},
        experiment="demo", program="steerlab")


def test_a_declaration_alone_is_the_effective_draw():
    effective = subsample.reconcile(None, _declared(), program="steerlab")
    assert effective.sample_per_condition == 7
    assert effective.seed == 0x2A
    assert effective.declared
    # …and it reaches the stamp as the additive `declared: true`.
    assert subsample.stamp(effective, sampled=14, source=24)["declared"] is True


def test_agreeing_flags_pass_and_the_draw_stays_the_declared_one():
    effective = subsample.reconcile(
        subsample.SubsampleRequest(7, 0x2A), _declared(), program="steerlab")
    assert effective.declared


def test_a_disagreeing_flag_refuses_naming_both_values():
    """Never an override: a flag that won would code one design while the
    run's snapshot recorded another. The repair is to drop the flag or to
    declare the design you actually want — never ``--force``."""
    with pytest.raises(subsample.SubsampleRefusal) as size:
        subsample.reconcile(subsample.SubsampleRequest(9, 0x2A), _declared(),
                            program="steerlab")
    assert size.value.code == "evaluationSamplingConflict"
    assert size.value.reason == (
        "--sample-per-condition 9 contradicts this study's declared sampling "
        "design, which preregistered 7 record(s) per condition. On a study "
        "that declares its design the flag is a CROSS-CHECK, never an "
        "override: the declaration is what the run's experiment.json snapshot "
        "carries, so a flag that won would code one design and record another")
    assert size.value.repair_action.startswith("drop --sample-per-condition")
    assert "set-evaluation-sampling <name> 9 <seed>" in \
        size.value.repair_action
    assert "--force" not in size.value.repair_action

    with pytest.raises(subsample.SubsampleRefusal) as seed:
        subsample.reconcile(subsample.SubsampleRequest(7, 99), _declared(),
                            program="steerlab")
    assert seed.value.reason == (
        "--sample-seed 0x0000000000000063 contradicts this study's declared "
        "sampling design, which preregistered seed 0x000000000000002a. On a "
        "study that declares its design the flag is a CROSS-CHECK, never an "
        "override: the declaration is what the run's experiment.json snapshot "
        "carries, so a flag that won would draw one subsample and record "
        "another")
    assert seed.value.repair_action.startswith("drop --sample-seed")
    assert "--force" not in seed.value.repair_action


def test_an_undeclared_study_keeps_the_flags_only_path():
    """The ad-hoc path stays legal and stays loud — it simply cannot claim the
    provenance a declared one has, which is what the absent `declared` key
    says."""
    effective = subsample.reconcile(
        subsample.SubsampleRequest(7, 0x2A), None, program="steerlab")
    assert effective.sample_per_condition == 7
    assert not effective.declared
    assert "declared" not in subsample.stamp(effective, sampled=14, source=24)


# =============================================================================
# The proof: the declaration reaches the run's snapshot
# =============================================================================


def test_the_declaration_lands_in_the_runs_own_snapshot(tmp_path):
    """THE assertion the feature exists for.

    ``_write_config_snapshot`` is the writer every run-directory-minting task
    on this engine calls, and it serializes ``manifest.raw`` verbatim — so a
    declared design travels with every run's evidence rather than only with
    the command line that started it. A plan document is pre-registration; this
    snapshot is what proves the plan is the thing that ran.
    """
    root = _workspace(tmp_path)
    es.declare_evaluation_sampling("demo", "2400", "0x2a", root=root)
    manifest = Manifest.from_dict(es.load_raw("demo", root))
    run_directory = os.path.join(root, "runs", "20260829T000000000-exp-demo-run")
    os.makedirs(run_directory)
    tasks._write_config_snapshot(manifest, run_directory, "run")

    with open(os.path.join(run_directory, "experiment.json"),
              encoding="utf-8") as handle:
        snapshot = json.load(handle)
    assert snapshot["evaluationSampling"] == {
        "rule": subsample.RULE,
        "samplePerCondition": 2400,
        "sampleSeed": "0x000000000000002a",
    }


def _preregistered_fixture(tmp_path, n="7", seed="0x2a") -> str:
    """The cross-engine coding fixture with the design declared BEFORE the run
    it will be drawn from — the preregistration order, and the reason the
    source run's epoch stamp is (re)written after the declaration rather than
    before it."""
    from tests.test_response_coding import _sampling_fixture

    root = _sampling_fixture(tmp_path)
    es.declare_evaluation_sampling("cf", n, seed, root=root)
    manifest = Manifest.from_dict(es.load_raw("cf", root))
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-cf-run")
    with open(os.path.join(run_dir, "experiment-hash.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(manifest.content_hash() + "\n")
    return root


def test_a_declared_evaluate_needs_no_flags_and_stamps_declared(
        tmp_path, monkeypatch):
    """End to end: the declared study codes the declared draw with NO
    arguments, and every stamp additionally notes that the design was
    DECLARED. The 14 triples are the cross-engine fixture's, verbatim."""
    from tests.test_evaluate_subsample import LITERAL_FIXTURE_SELECTION
    from tests.test_response_coding import _coding_generate, _provider

    root = _preregistered_fixture(tmp_path)
    _coding_generate(monkeypatch, [
        '{"codes": {"mentionsLegalRule": true, "mentionsEquity": false}, '
        '"brief_reason": "r"}'])
    logs: list = []

    out = tasks.evaluate("cf", root=root, model_provider=_provider(),
                         max_loaded=1,
                         log=lambda *p: logs.append(" ".join(map(str, p))))

    rows = [json.loads(line) for line in
            open(os.path.join(out, "codings.jsonl"), encoding="utf-8")]
    assert {(r["condition"], r["promptID"], r["sampleIndex"]) for r in rows} \
        == set(LITERAL_FIXTURE_SELECTION)
    report = json.load(open(os.path.join(out, "coding-report.json"),
                            encoding="utf-8"))
    assert report["sampling"] == {
        "rule": subsample.RULE,
        "samplePerCondition": 7,
        "sampleSeed": "0x000000000000002a",
        "sampledRecords": 14,
        "sourceRecords": 24,
        "declared": True,
    }
    config = json.load(open(os.path.join(out, "config.json"),
                            encoding="utf-8"))
    assert config["notes"]["sampling"] == report["sampling"]
    assert any("the study declares a sampling design" in line
               for line in logs), logs


def test_a_disagreeing_flag_refuses_at_evaluate_and_writes_nothing(
        tmp_path, monkeypatch):
    """The cross-check happens in the TASK, not at a CLI edge, so the CLI, the
    bundle-execute child and any library caller all get it on the same
    bytes."""
    from tests.test_response_coding import (_coding_generate, _provider,
                                            _sampling_fixture)

    root = _sampling_fixture(tmp_path)
    es.declare_evaluation_sampling("cf", "7", "0x2a", root=root)
    _coding_generate(monkeypatch, ["{}"])
    before = sorted(os.listdir(os.path.join(root, "runs")))
    with pytest.raises(subsample.SubsampleRefusal) as caught:
        tasks.evaluate("cf", root=root, model_provider=_provider(),
                       max_loaded=1, sample_per_condition=9,
                       sample_seed="0x2a", log=lambda *p: None)
    assert caught.value.code == "evaluationSamplingConflict"
    assert sorted(os.listdir(os.path.join(root, "runs"))) == before


def test_agreeing_flags_are_accepted_at_evaluate(tmp_path, monkeypatch):
    """A cross-check that PASSES is not an error: the wire fields a submitted
    job carries may legitimately restate the declaration."""
    from tests.test_response_coding import _coding_generate, _provider

    root = _preregistered_fixture(tmp_path)
    _coding_generate(monkeypatch, [
        '{"codes": {"mentionsLegalRule": true, "mentionsEquity": false}, '
        '"brief_reason": "r"}'])
    out = tasks.evaluate("cf", root=root, model_provider=_provider(),
                         max_loaded=1, sample_per_condition=7,
                         sample_seed="42",  # the decimal spelling of 0x2a
                         log=lambda *p: None)
    report = json.load(open(os.path.join(out, "coding-report.json"),
                            encoding="utf-8"))
    assert report["sampling"]["declared"] is True


def test_declaring_the_design_is_measurement_side_drift(tmp_path):
    """The house flow the ``MEASUREMENT_FIELDS`` entry exists for: judged
    re-measurement runs on a never-frozen DUPLICATE, and a duplicate that
    declares the coding design differs from the original run's snapshot by
    exactly this key. It cannot have moved a byte of that run's generations,
    so it is tolerated (loudly, and stamped) rather than refused.

    Swift twin: the same key in ``RunEpoch.measurementFields``.
    """
    from steerlab_server.experiment import run_epoch

    assert "evaluationSampling" in run_epoch.MEASUREMENT_FIELDS
    root = _workspace(tmp_path)
    before = Manifest.from_dict(es.load_raw("demo", root))
    es.declare_evaluation_sampling("demo", "2400", "0x2a", root=root)
    after = Manifest.from_dict(es.load_raw("demo", root))
    # Real drift by the strict rule …
    assert before.content_hash() != after.content_hash()
    # … and confined to the measurement surface, so a measurement verb may
    # tolerate it.
    assert run_epoch._measurement_drift(after, before) is not None


# =============================================================================
# Cross-engine sentence parity
#
# The shared-literal-fixture convention `test_evaluate_subsample` established
# for the flag refusals: this table appears byte-identically in
# `EvaluationSamplingDeclarationTests.swift`, so a sentence that moves on one
# engine and not the other fails on the engine that did not move.
# =============================================================================

LITERAL_DECLARATION_REFUSALS = {
    "seedMissing": (
        "the sampling design named 2400 record(s) per condition with no "
        "seed — a subsample nobody can redraw is not a preregistration, so "
        "the declaration refuses rather than choosing a seed for you",
        "steerlab-cli experiment set-evaluation-sampling demo 2400 "
        "0x5eed0a5e5eed0a5e  (a per-condition size and the seed that draws "
        'it — both, always; "" clears the declaration)',
    ),
    "sizeMissing": (
        "the sampling design named seed 0x2a with no per-condition size — "
        "with no size the full corpus is coded, and the seed would be "
        "stamped on a design it did not shape",
        "steerlab-cli experiment set-evaluation-sampling demo 2400 "
        "0x5eed0a5e5eed0a5e  (a per-condition size and the seed that draws "
        'it — both, always; "" clears the declaration)',
    ),
    "sizeMalformed": (
        "the sampling design's samplePerCondition must be a whole number of "
        "records of at least 1, not 'x' — a subsample of zero records is a "
        "design nobody can report",
        "steerlab-cli experiment set-evaluation-sampling demo 2400 "
        "0x5eed0a5e5eed0a5e  (a per-condition size and the seed that draws "
        'it — both, always; "" clears the declaration)',
    ),
    "seedMalformed": (
        "the sampling design's sampleSeed 'zz' is not a 64-bit unsigned "
        "number — a seed is a decimal integer, or hexadecimal with or "
        "without a '0x' prefix, of at most 16 hex digits (the leading 16 of "
        "a digest are a fine seed, written down as such)",
        "steerlab-cli experiment set-evaluation-sampling demo 2400 "
        "0x5eed0a5e5eed0a5e  (a per-condition size and the seed that draws "
        'it — both, always; "" clears the declaration)',
    ),
}


def _refusal(fn):
    try:
        fn()
    except subsample.SubsampleRefusal as exc:
        return (exc.reason, exc.repair_action)
    raise AssertionError("expected a refusal")


def test_every_declaration_refusal_reads_identically_on_both_engines():
    """A researcher who reads a refusal on one engine must read the SAME
    refusal on the other, repair included."""
    def declare(size, seed):
        return lambda: subsample.resolve_declaration(
            size, seed, experiment="demo", program=MAC)

    assert {
        "seedMissing": _refusal(declare("2400", None)),
        "sizeMissing": _refusal(declare(None, "0x2a")),
        "sizeMalformed": _refusal(declare("x", "0x2a")),
        "seedMalformed": _refusal(declare("1", "zz")),
    } == LITERAL_DECLARATION_REFUSALS
