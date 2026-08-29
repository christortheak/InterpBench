"""The shared agent-path envelope, server side, and its CROSS-ENGINE TWIN
LITERALS (WP0-AGENT-SURFACE-AUDIT §7 step 8).

Two kinds of test live here and they are doing different jobs:

1. **Twin literals.** Every constant of the contract — the closed header keys,
   the optional keys, the state→exit table, the freeze-gate vocabulary, the
   lifecycle-gate vocabulary, the advisory codes, the engine stamp, and the two
   cross-engine message strings — is written out HERE as a literal copied from
   the Swift source, and asserted equal to this engine's constant. The Swift
   half (``Tests/ExperimentKitTests/CLIEnvelopeParityTests.swift``) does the
   mirror image against the literals in ``steerlab_server/cli_envelope.py``.

   Two independent literals with a naming cross-reference is deliberately worse
   engineering than a shared schema file and deliberately better parity
   enforcement (audit §3.1, the ``config.json`` idiom): neither engine can
   quietly follow the other, because adding, removing, or renaming anything
   fails a test on BOTH sides until both literals move in the same change.

2. **Golden envelopes.** One committed document per server agent-path verb
   under ``tests/fixtures/cli-envelopes/``, produced by driving ``cli.main``
   against a scratch root with a PINNED CLOCK — the same strategy the Swift
   fixtures use (``ExperimentCLIEnvelopeTests``), including the
   write-if-missing rule: the structural assertions always run on the freshly
   produced document, and only the byte comparison waits for the file to be
   committed. A fixture that EXISTS must match exactly.
"""

import json
import os

import pytest

from steerlab_server import cli, cli_envelope
from steerlab_server.experiment import (exclusions, experiment_store,
                                        lifecycle_gates, sweep_selection)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "cli-envelopes")

#: The clock every fixture is produced under — the same instant the Swift
#: fixtures pin (``Date(timeIntervalSince1970: 1_000)``), so a reader
#: comparing a Swift golden with its server twin sees the same ``observedAt``.
PINNED_NOW = 1_000.0


# =============================================================================
# 1. Twin literals
# =============================================================================


def test_engine_stamp_matches_the_substrate_constant():
    """``engine`` is a provenance label, never a capability claim — and it must
    be the SAME string the artifacts are scoped by, or a run's substrate and
    its CLI's substrate could disagree. Swift twin:
    ``SteerLabCLIEnvelope.serverEngine`` aliases
    ``WorkspaceScoping.serverSubstrate``."""
    from steerlab_server.steering.vector_store import SUBSTRATE
    assert cli_envelope.ENGINE == SUBSTRATE == "python-hf-transformers"


def test_contract_header_keys_match_the_swift_literal():
    """Copied from ``SteerLabCLIEnvelope.contractHeaderKeys``
    (``Sources/ExperimentKit/SteerLabCLIEnvelope.swift``). Swift twin test:
    ``CLIEnvelopeParityTests.headerKeysMatchServerLiteral``."""
    contract = ["changed", "engine", "message", "observedAt", "schemaVersion",
                "state", "verb"]
    assert list(cli_envelope.CONTRACT_HEADER_KEYS) == contract


def test_contract_optional_keys_match_the_swift_literal():
    """Copied from ``SteerLabCLIEnvelope.contractOptionalKeys``."""
    contract = ["advisories", "error", "nextAction", "result", "workspace"]
    assert list(cli_envelope.CONTRACT_OPTIONAL_KEYS) == contract


def test_state_vocabulary_and_exit_codes_match_the_swift_literal():
    """Copied from ``SteerLabCLIState`` + ``SteerLabCLIState.exitCode``.

    ORDER is part of the contract (it is the Swift enum's declaration order),
    and so is every number: the whole point of the package is that a freeze
    refusal (65), a missing experiment (66), and an operational failure (70)
    stop being the same 1."""
    contract = [
        ("ready", 0),
        ("planned", 0),
        ("running", 0),
        ("okWithAdvisories", 0),
        ("needsHumanAuthentication", 10),
        ("needsApproval", 11),
        ("pending", 12),
        ("degraded", 13),
        ("blocked", 64),
        ("refused", 65),
        ("notFound", 66),
        ("failed", 70),
    ]
    assert list(cli_envelope.STATE_EXIT_CODES.items()) == contract
    assert cli_envelope.SCHEMA_VERSION == 1


def test_freeze_gate_vocabulary_matches_the_swift_literal():
    """Copied from ``FreezeGate.vocabulary``
    (``Sources/ExperimentKit/FreezeGate.swift``). The order is the order
    ``forcedGatesSkipped`` is stamped in on both engines."""
    contract = ["revision", "validateEvidence", "batteryEvidence",
                "judgeValidity", "variantValidity", "gitClean",
                "measurementPins"]
    assert list(experiment_store.FORCED_GATE_IDS) == contract


def test_lifecycle_gate_vocabulary_matches_the_swift_literal():
    """Copied from ``LifecycleGate.vocabulary``
    (``Sources/ExperimentKit/LifecycleGate.swift``, declaration order). Swift
    twin test: ``CLIEnvelopeParityTests.lifecycleGatesMatchServerLiteral``."""
    contract = [
        "statusImmutable", "pinDrift", "manifestEpoch", "promotionEpoch",
        "promotionEvidence", "artifactPin", "sweepInputDrift",
        "sweepSelectionRule", "sweepJudgeCapacity", "dataReadiness",
        "samplingPolicy", "thinkingModeConflict", "inertConditions",
        "responseFormat", "confirmationPool", "confirmationAgentShape",
        "parityThreshold", "missingPrerequisite", "armsCleared",
        "conceptInUse", "sweepGridRule",
    ]
    assert list(lifecycle_gates.LIFECYCLE_GATE_IDS) == contract


def test_advisory_codes_match_the_swift_literal():
    """Copied from ``CLIAdvisory`` (``allCases`` order,
    ``Sources/ExperimentKit/SteerLabCLIEnvelope.swift``)."""
    contract = [
        "freezeGateSkipped", "vacuousValidation", "probeAtChanceFloor",
        "judgePanelTooSmall", "emptyAnalysis", "allEffectSizesZero",
        "sweepRecommendationsOnly", "sweepSelectionDefaulted",
        "choiceItemsWithoutInstrument", "revisionAdoption",
        "revisionAdoptionWarning", "siteQualifyWarning",
        "deprecatedImplicitSelection", "systemPromptNotApplied",
        "singleRegimeCapabilityReading",
    ]
    assert list(cli_envelope.ADVISORY_CODES) == contract


def test_the_two_gate_vocabularies_are_disjoint():
    """An id in both would make ``error.gate`` ambiguous about which ``switch``
    an agent should use — and the two have different skippability classes:
    ``--force`` skips freeze gates, and nothing skips lifecycle gates. Swift
    twin: ``LifecycleGate.collidesWithFreezeVocabulary``."""
    assert not (set(lifecycle_gates.LIFECYCLE_GATE_IDS)
                & set(experiment_store.FORCED_GATE_IDS))


def test_exclusion_pin_required_message_and_repair_match_the_swift_literals():
    """The deferred cross-engine-twinned message from c86ce53, now with its
    gate id on both engines. Swift twins: ``ExclusionEngine.pinRequiredMessage``
    and ``ExclusionEngine.pinRequiredRepair``."""
    assert exclusions.PIN_REQUIRED_MESSAGE == (
        "exclusion rule failedAttentionCheck needs the task prompts pinned "
        "(taskPromptsFile + taskPromptsHash) so analysis grades the same items "
        "the run saw — pin the prompt set first")
    assert exclusions.PIN_REQUIRED_REPAIR == (
        "steerlab-cli experiment pin-prompts <name> <the prompt file the run "
        "used> — analysis grades the items the run saw, so the pin must name "
        "that exact file")


def test_defaulted_selection_advisory_matches_the_swift_literal():
    """Swift twin: ``SweepSelectionRule.defaultedSelectionAdvisory``. The
    string names ``steerlab-cli`` on BOTH engines because ``sweep.selection``
    is authored on the Mac (audit §3.2)."""
    assert sweep_selection.defaulted_selection_advisory(None, 3, 8) == (
        "no sweep.selection is declared, so the winning cell will be chosen "
        "by markerDensity — a SURFACE-PROSE diagnostic — while 3 of 8 pinned "
        "item(s) carry options/target and could be scored deterministically. "
        "Declare the criterion: steerlab-cli experiment set-sweep-selection "
        "<name> --objective logprobShift --choice-prompts <file>  (or "
        "--objective judgeScore with a pinned rubric). Marker density is a "
        "manipulation check, not a decision objective.")
    # Declared, or no choice-shaped items: no advisory.
    assert sweep_selection.defaulted_selection_advisory(
        {"objective": {"metric": "logprobShift"}}, 3, 8) is None
    assert sweep_selection.defaulted_selection_advisory(None, 0, 8) is None


# =============================================================================
# 2. The envelope's own rules
# =============================================================================


def test_exit_code_is_derived_from_state_never_from_a_second_table():
    for state, code in cli_envelope.STATE_EXIT_CODES.items():
        envelope = cli_envelope.Envelope(verb="x", state=state, message="m")
        assert envelope.exit_code == code


def test_an_unknown_state_refuses_rather_than_degrading():
    """Decoding — or constructing — an unknown state fails rather than guessing:
    an agent that mis-reads a refusal as a success does more damage than one
    that stops."""
    with pytest.raises(ValueError):
        cli_envelope.Envelope(verb="x", state="probably-fine", message="m")


def test_an_unknown_advisory_code_cannot_be_minted():
    with pytest.raises(ValueError):
        cli_envelope.advisory("looksImportant", "…")


def test_an_unknown_lifecycle_gate_cannot_be_minted():
    with pytest.raises(ValueError):
        lifecycle_gates.refusing("notAGate", "…")


def test_optional_keys_are_omitted_not_nulled():
    """A caller testing for a key gets a straight answer rather than a null."""
    document = cli_envelope.success("x y", "done").to_dict()
    assert set(document) >= set(cli_envelope.CONTRACT_HEADER_KEYS)
    for key in ("advisories", "error", "nextAction", "result"):
        assert key not in document


def test_a_refusal_always_contains_its_own_gate():
    """The invariant ``error.gate`` ∈ ``error.gates`` — enforced, not trusted,
    exactly as ``SteerLabCLIEnvelope.refusal`` enforces it."""
    document = cli_envelope.refusal(
        "experiment freeze", code="freezeGateFailed", gate="gitClean",
        gates=["validateEvidence"], reason="r", repair_action="fix").to_dict()
    assert document["error"]["gate"] == "gitClean"
    assert "gitClean" in document["error"]["gates"]


def test_advisories_never_change_the_exit_code():
    """A ``set -e`` wrapper must not break on a legitimate lifecycle."""
    envelope = cli_envelope.success(
        "experiment validate demo", "validated", changed=True,
        advisories=[cli_envelope.advisory("vacuousValidation", "…")])
    assert envelope.state == "okWithAdvisories"
    assert envelope.exit_code == 0


def test_json_text_is_one_sorted_document_with_one_trailing_newline():
    text = cli_envelope.success("experiment list", "1 experiment(s)").json_text()
    assert text.endswith("}\n")
    assert text.count("\n}") == 1
    document = json.loads(text)          # exactly one value, nothing trailing
    assert document["schemaVersion"] == 1
    assert list(document) == sorted(document)


# =============================================================================
# 3. Strict flags
# =============================================================================


def _run(monkeypatch, tmp_path, argv):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setattr(cli_envelope, "now", lambda: PINNED_NOW)
    return cli.main(argv)


def test_an_undeclared_flag_is_64_in_both_modes(tmp_path, monkeypatch, capsys):
    """A malformed invocation was never a refusal — and it refuses BEFORE the
    verb does any work, because a refusal after the first side effect is not
    much better than no refusal at all (punch list P0-4)."""
    assert _run(monkeypatch, tmp_path,
                ["experiment", "analyze", "demo", "--allow-unverfied-epoch"]) == 64
    assert "does not accept --allow-unverfied-epoch" in capsys.readouterr().err

    assert _run(monkeypatch, tmp_path,
                ["experiment", "analyze", "demo", "--nope", "--json"]) == 64
    out = capsys.readouterr().out
    envelope = json.loads(out)
    assert envelope["state"] == "blocked"
    assert envelope["error"]["code"] == "unknownFlag"
    assert "--allow-unverified-epoch" in envelope["error"]["repairAction"]


def test_declared_flags_reach_the_verb_unchanged():
    invocation = cli_envelope.parse(
        "experiment", ["analyze", "demo", "--allow-unverified-epoch", "--json"])
    assert invocation.args == ["analyze", "demo", "--allow-unverified-epoch"]
    assert invocation.json is True
    assert invocation.declared is True


def test_out_flag_is_lifted_and_needs_a_path():
    invocation = cli_envelope.parse("experiment", ["list", "--out", "/tmp/x.json"])
    assert invocation.out_path == "/tmp/x.json"
    assert invocation.args == ["list"]
    with pytest.raises(cli_envelope.UsageError):
        cli_envelope.parse("experiment", ["list", "--out"])


def test_legacy_json_path_is_deprecated_not_broken():
    """``--json <path>`` on ``vectors compare`` keeps working for one release
    with a stderr warning; ``--out <path>`` is the replacement (audit §2.2)."""
    invocation = cli_envelope.parse(
        "vectors", ["compare", "a", "b", "--json", "/tmp/report.json"])
    assert invocation.json is False
    assert invocation.args[-2:] == ["--json", "/tmp/report.json"]
    assert invocation.deprecations and "--out /tmp/report.json" in \
        invocation.deprecations[0]


def test_undeclared_verbs_pass_through_untouched():
    """Everything in an agent family that is NOT one of the fourteen runs
    exactly as it always did: no strict parsing, no stripped flags."""
    invocation = cli_envelope.parse(
        "experiment", ["attach-artifact", "s", "c", "--artifact", "runs/x/y"])
    assert invocation.declared is False
    assert invocation.args == ["attach-artifact", "s", "c", "--artifact",
                               "runs/x/y"]


def test_the_declared_verbs_are_the_audits_fourteen_plus_site_qualify():
    """Audit §2.1's table, server column, plus `site qualify` (WP6/gate 7 — a
    server-only verb, added after the audit), `vectors mirror-poles` (pole
    mirroring, added later still: pure local file work, so it joined the agent
    path rather than following `backfill-norms` off it), and `battery run`
    (the standalone capability reading, 2026-08-29 — it loads models, so it is
    server-only, and it brought its family onto the agent path with it).
    `data check` is counted once (its argument domain differs from Swift's — a
    documented §3.2 divergence, not a second verb)."""
    labels = sorted(spec.label for spec in cli_envelope.VERB_SPECS)
    assert labels == sorted([
        "battery run", "data check", "experiment analyze",
        "experiment confirm", "experiment evaluate", "experiment extract",
        "experiment list", "experiment promote", "experiment run",
        "experiment sweep", "experiment validate", "experiment verify",
        "jobs list", "site qualify", "study submit", "vectors compare",
        "vectors mirror-poles",
    ])
    assert len(cli_envelope.VERB_SPECS) == 17


# =============================================================================
# 4. Golden envelopes, one per drivable verb
# =============================================================================


def _concept(root, name="french"):
    """A minimal on-disk stimulus set, so `attach` pins a REAL hash and
    `verify` has something to verify (the same helper
    ``test_experiment_store.py`` uses)."""
    directory = os.path.join(str(root), "prompts", "concepts", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "positive.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text": "bonjour"}\n')
    with open(os.path.join(directory, "negative.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text": "hello"}\n')


def _canonicalize(text: str, root) -> str:
    return (text.replace(os.path.realpath(str(root)), "<workspace>")
                .replace(str(root), "<workspace>")
                .replace(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "<checkout>"))


def _check(capsys, name: str, root, *, expect_state: str):
    """Assert the structural contract, then compare against (or write) the
    golden. A MISSING fixture is written rather than failed — the structural
    assertions still run on the fresh document, so a new verb cannot land
    unchecked; only the byte comparison waits for the file to be committed."""
    text = capsys.readouterr().out
    document = json.loads(text)          # exactly one document on stdout
    assert text.endswith("}\n")
    assert text.count("\n}") == 1

    allowed = set(cli_envelope.CONTRACT_HEADER_KEYS) | set(
        cli_envelope.CONTRACT_OPTIONAL_KEYS)
    for key in cli_envelope.CONTRACT_HEADER_KEYS:
        assert key in document, f"{name}: header key {key!r} missing"
    for key in document:
        assert key in allowed, f"{name}: undeclared top-level key {key!r}"
    assert document["engine"] == cli_envelope.ENGINE
    assert document["state"] == expect_state
    assert document["observedAt"] == "1970-01-01T00:16:40Z"
    # An error is present exactly when the state is not a success.
    is_success = cli_envelope.exit_code_for(document["state"]) == 0
    assert ("error" in document) != is_success

    os.makedirs(FIXTURES, exist_ok=True)
    path = os.path.join(FIXTURES, f"{name}.json")
    canonical = _canonicalize(text, root)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(canonical)
        return document
    with open(path, encoding="utf-8") as handle:
        assert canonical == handle.read(), (
            f"{name}: envelope drifted from tests/fixtures/cli-envelopes/"
            f"{name}.json")
    return document


def test_experiment_list_empty_envelope(tmp_path, monkeypatch, capsys):
    assert _run(monkeypatch, tmp_path, ["experiment", "list", "--json"]) == 0
    document = _check(capsys, "experiment-list-empty", tmp_path,
                      expect_state="ready")
    assert document["result"]["count"] == 0


def test_experiment_list_envelope_carries_full_hashes(
        tmp_path, monkeypatch, capsys):
    experiment_store.create(
        "demo", model_id="google/gemma-3-27b-it",
        revision="0123456789abcdef0123456789abcdef01234567", root=str(tmp_path))
    assert _run(monkeypatch, tmp_path, ["experiment", "list", "--json"]) == 0
    document = _check(capsys, "experiment-list", tmp_path, expect_state="ready")
    entry = document["result"]["experiments"][0]
    # FULL revision, not the human line's twelve characters.
    assert entry["modelRevision"] == "0123456789abcdef0123456789abcdef01234567"
    assert entry["status"] == "draft"


def test_experiment_verify_envelope(tmp_path, monkeypatch, capsys):
    _concept(tmp_path)
    experiment_store.create("demo", model_id="google/gemma-3-27b-it",
                            root=str(tmp_path))
    experiment_store.attach("demo", ["french"], root=str(tmp_path))
    assert _run(monkeypatch, tmp_path,
                ["experiment", "verify", "demo", "--json"]) == 0
    document = _check(capsys, "experiment-verify", tmp_path,
                      expect_state="ready")
    assert document["result"]["verified"] is True
    assert len(document["result"]["experimentHash"]) == 64


def test_experiment_verify_violation_is_a_typed_pin_drift(
        tmp_path, monkeypatch, capsys):
    """The refusal an agent hits most: a pinned input that no longer matches.
    65 with ``error.gate == "pinDrift"`` and the drifted pins in
    ``result.violations`` — where before it was exit 1 and prose on stdout."""
    _concept(tmp_path)
    experiment_store.create("demo", model_id="google/gemma-3-27b-it",
                            root=str(tmp_path))
    experiment_store.attach("demo", ["french"], root=str(tmp_path))
    # The pinned stimuli DRIFT: the file the recipe re-derives from changed.
    with open(os.path.join(str(tmp_path), "prompts", "concepts", "french",
                           "positive.jsonl"), "w", encoding="utf-8") as handle:
        handle.write('{"text": "salut"}\n')

    assert _run(monkeypatch, tmp_path,
                ["experiment", "verify", "demo", "--json"]) == 65
    document = _check(capsys, "experiment-verify-violation", tmp_path,
                      expect_state="refused")
    assert document["error"]["code"] == "pinDrift"
    assert document["error"]["gate"] == "pinDrift"
    assert document["result"]["violations"]


def test_experiment_verify_human_mode_is_unchanged(tmp_path, monkeypatch,
                                                   capsys):
    """Human mode keeps its exit 1 and its ``VIOLATION:`` lines: the audit
    schedules exactly one human-mode migration (``data check``'s 2 → 65), and
    widening it would break ``set -e`` wrappers the row does not discuss."""
    _concept(tmp_path)
    experiment_store.create("demo", model_id="google/gemma-3-27b-it",
                            root=str(tmp_path))
    experiment_store.attach("demo", ["french"], root=str(tmp_path))
    # The pinned stimuli DRIFT: the file the recipe re-derives from changed.
    with open(os.path.join(str(tmp_path), "prompts", "concepts", "french",
                           "positive.jsonl"), "w", encoding="utf-8") as handle:
        handle.write('{"text": "salut"}\n')
    assert _run(monkeypatch, tmp_path, ["experiment", "verify", "demo"]) == 1
    assert "VIOLATION:" in capsys.readouterr().out


def test_experiment_not_found_is_66(tmp_path, monkeypatch, capsys):
    """The commonest agent mistake, and until now indistinguishable from a real
    failure."""
    assert _run(monkeypatch, tmp_path,
                ["experiment", "verify", "nope", "--json"]) == 66
    document = _check(capsys, "experiment-not-found", tmp_path,
                      expect_state="notFound")
    assert document["error"]["code"] == "notFound"


def test_jobs_list_envelope(tmp_path, monkeypatch, capsys):
    assert _run(monkeypatch, tmp_path, ["jobs", "list", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["verb"] == "jobs list"
    assert document["result"]["count"] == len(document["result"]["jobs"])


def test_vectors_compare_envelope_carries_the_report(tmp_path, monkeypatch,
                                                     capsys):
    """The bare parity report — whose formatter exists so a cross-engine
    ``diff`` is trivial — rides inside ``result.report``, key for key."""
    parity = os.path.join(os.path.dirname(__file__), "fixtures", "parity")
    assert _run(monkeypatch, tmp_path, [
        "vectors", "compare",
        os.path.join(parity, "identical-a.safetensors"),
        os.path.join(parity, "identical-b.safetensors"), "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["state"] == "ready"
    assert document["result"]["passed"] is True
    assert document["result"]["report"]["pass"] is True


def test_vectors_compare_threshold_failure_is_a_typed_refusal(
        tmp_path, monkeypatch, capsys):
    """Swift collapsed threshold-fail / could-not-run / usage into 1; the
    server separated them by exit code but named none of them. Now the
    threshold failure is ``parityThreshold``, 65, with a repair that names the
    re-extraction the substrate rule requires."""
    parity = os.path.join(os.path.dirname(__file__), "fixtures", "parity")
    assert _run(monkeypatch, tmp_path, [
        "vectors", "compare",
        os.path.join(parity, "orthogonal-a.safetensors"),
        os.path.join(parity, "orthogonal-b.safetensors"), "--json"]) == 65
    document = json.loads(capsys.readouterr().out)
    assert document["error"]["gate"] == "parityThreshold"
    assert "extract" in document["error"]["repairAction"]
    # Human mode is unchanged: the report on stdout, exit 1.
    assert _run(monkeypatch, tmp_path, [
        "vectors", "compare",
        os.path.join(parity, "orthogonal-a.safetensors"),
        os.path.join(parity, "orthogonal-b.safetensors")]) == 1
    assert json.loads(capsys.readouterr().out)["pass"] is False


def test_out_writes_the_document_in_both_modes(tmp_path, monkeypatch, capsys):
    """"Give me the document in a file" is a separate request from "put it on
    stdout"."""
    target = tmp_path / "envelope.json"
    assert _run(monkeypatch, tmp_path,
                ["experiment", "list", "--out", str(target)]) == 0
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["verb"] == "experiment list"
    # Human mode still printed the human listing, not the document.
    assert "no experiments" in capsys.readouterr().out


# =============================================================================
# 5. The science fields — validate, sweep, analyze
# =============================================================================
#
# These verbs need a model to drive end to end, so the ENVELOPE material is
# tested against synthetic run directories instead: the readers in
# `cli_payloads` are exactly what the CLI puts under `result`, and they are
# where the cross-engine parity lives. The two engines' REPORT keys differ
# (Swift's `validation` block vs the server's `concepts`; Swift's
# `analysis.json` vs the server's `effect-sizes.csv` — audit §3.2 idiom
# differences); what an agent reads must not.


def _run_dir(tmp_path, name: str, files: dict) -> str:
    directory = os.path.join(str(tmp_path), "runs", name)
    os.makedirs(directory, exist_ok=True)
    for filename, content in files.items():
        with open(os.path.join(directory, filename), "w",
                  encoding="utf-8") as handle:
            handle.write(content if isinstance(content, str)
                         else json.dumps(content, indent=2, sort_keys=True))
    return directory


def test_validate_payload_carries_accuracy_and_auc(tmp_path):
    """Punch list #1, P4: a chance-level probe froze and ran with no machine
    signal. Swift twin keys: `result.validation[].accuracy` /
    `.balancedAccuracy` / `.auc` / `.atOrBelowChance`."""
    from steerlab_server import cli_payloads
    directory = _run_dir(tmp_path, "exp-demo-validate", {
        "validation-report.json": {
            "vacuousConcepts": [],
            "concepts": {
                "french": {
                    "scenarioCount": 24,
                    "depths": [{
                        "layer": 17,
                        "scenarioAccuracy": 0.875,
                        "diagnostics": {"accuracy": 0.875,
                                        "balancedAccuracy": 0.86,
                                        "auc": 0.93,
                                        "oneSidedPredictions": False},
                    }],
                },
            },
        },
    })
    payload = cli_payloads.validation_payload("demo", directory)
    assert payload["vacuous"] is False
    score = payload["validation"][0]
    assert score == {"concept": "french", "oneSidedPredictions": False,
                     "atOrBelowChance": False, "layer": 17,
                     "accuracy": 0.875, "balancedAccuracy": 0.86,
                     "auc": 0.93, "scenarios": 24}
    assert cli_payloads.validation_summary([score]) == "french 88% (AUC 0.93)"


def test_validate_payload_flags_a_probe_at_the_chance_floor(tmp_path):
    """Balanced accuracy is preferred where it exists, and a one-sided
    threshold is at the floor by construction whatever the accuracy says."""
    from steerlab_server import cli_payloads
    directory = _run_dir(tmp_path, "exp-demo-validate-2", {
        "validation-report.json": {
            "vacuousConcepts": ["sympathy"],
            "concepts": {
                "french": {"scenarioCount": 10, "depths": [{
                    "layer": 17, "scenarioAccuracy": 0.5,
                    "diagnostics": {"accuracy": 0.5, "balancedAccuracy": 0.5,
                                    "auc": 0.5, "oneSidedPredictions": False}}]},
                "german": {"scenarioCount": 10, "depths": [{
                    "layer": 17, "scenarioAccuracy": 0.7,
                    "diagnostics": {"accuracy": 0.7, "balancedAccuracy": 0.5,
                                    "auc": 0.6, "oneSidedPredictions": True}}]},
            },
        },
    })
    payload = cli_payloads.validation_payload("demo", directory)
    assert payload["vacuousConcepts"] == ["sympathy"]
    assert payload["vacuous"] is True
    assert all(score["atOrBelowChance"] for score in payload["validation"])
    detail = cli_payloads.probe_advisory_detail(payload["validation"][1])
    assert "put EVERY item on one side" in detail
    assert "still SATISFIES freeze's validateEvidence gate" in detail
    # And the advisory code it rides under is in the closed vocabulary.
    assert cli_envelope.advisory("probeAtChanceFloor", detail)["code"] == \
        "probeAtChanceFloor"


def test_sweep_payload_carries_the_winning_cell_and_the_criterion(tmp_path):
    """Punch list #1, P2. Swift twin keys: `result.recommendations[].concept`
    / `.selected` / `.winningCell` / `.criterion` / `.metrics` / `.failure`,
    plus `runDirectory`, `criterion`, `devPromptsHash`."""
    from steerlab_server import cli_payloads
    directory = _run_dir(tmp_path, "exp-demo-sweep", {
        "recommendations.json": {
            "french": {
                "sweepRun": "exp-demo-sweep",
                "devPromptsHash": "a" * 64,
                "criterion": {"objective": {"metric": "logprobShift"}},
                "winningCell": {"layer": 17, "alpha": 0.4},
                "metrics": {"logprobShift": 0.31, "batteryAccuracy": 0.9},
            },
            "sympathy": "no cell cleared the capability floor",
        },
    })
    payload = cli_payloads.sweep_payload(
        "demo", directory, manifest_status="frozen", criterion="logprobShift")
    assert payload["criterion"] == "logprobShift"
    assert payload["devPromptsHash"] == "a" * 64      # FULL hash, never elided
    assert payload["recommendationsOnly"] is True     # non-draft manifest
    french, sympathy = payload["recommendations"]
    assert french["selected"] is True
    assert french["winningCell"] == {"layer": 17, "alpha": 0.4}
    assert french["criterion"] == "logprobShift"
    assert french["metrics"]["logprobShift"] == 0.31
    assert sympathy["selected"] is False
    # A failure entry is still evidence a sweep ran — promote reads it as such.
    assert sympathy["failure"] == "no cell cleared the capability floor"


def test_analysis_payload_and_the_all_zero_advisory(tmp_path):
    """Punch list #1, P14: zero ENTRIES had an advisory; entries that are all
    exactly zero did not — a different fact, and far likelier at this stage to
    mean an arm that was declared and never injected."""
    from steerlab_server import cli_payloads
    header = ("condition,endpoint,n,deltaMean,ciLower,ciUpper,wilcoxonW,"
              "wilcoxonP,adjustedP,correction,modality,stratifyBy,stratum,"
              "unit,estimand,inference\n")
    directory = _run_dir(tmp_path, "exp-demo-analyze", {
        "effect-sizes.csv": header
        + "french-L17,severity,24,0,0,0,0,1,1,bh,,pooled,,item,,\n"
        + "french-L17,severity,24,0,0,0,0,1,1,bh,,promptID,p1,item,,\n",
        "source-run.txt": "20260818-000000-exp-demo-run\n",
        "config.json": {"experimentHash": "b" * 64,
                        "notes": {"epochUnverified": True}},
    })
    payload = cli_payloads.analysis_payload(directory)
    assert payload["effectSizeCount"] == 1            # pooled rows only
    assert payload["stratifiedRowCount"] == 1
    assert payload["conditions"] == ["french-L17"]
    assert payload["metrics"] == ["severity"]
    assert payload["significantAtAdjusted05"] == 0
    assert payload["sourceRun"] == "20260818-000000-exp-demo-run"
    assert payload["experimentHash"] == "b" * 64
    assert payload["epochUnverified"] is True
    assert cli_payloads.all_effect_sizes_are_zero(directory) is True


def test_analysis_payload_is_empty_rather_than_raising_on_a_bare_run(tmp_path):
    """A verb that did its work must not fail because the document it produced
    could not be summarised."""
    from steerlab_server import cli_payloads
    directory = _run_dir(tmp_path, "exp-demo-analyze-2", {})
    payload = cli_payloads.analysis_payload(directory)
    assert payload["effectSizeCount"] == 0
    assert cli_payloads.all_effect_sizes_are_zero(directory) is False
    assert cli_payloads.validation_payload("demo", directory) == {
        "experiment": "demo", "runDirectory": directory, "vacuous": False}


# --- the emptyAnalysis DETAIL, and the CSV dialect stamp (WP0 dry run #2) ----


def test_empty_analysis_detail_says_what_was_observed():
    """The advisory used to assert ONE cause unconditionally — "no
    non-baseline condition to pair against" — and on the run that produced
    the finding that was false: two conditions, 24 records, and what failed
    was reading and PAIRING them. Swift twin:
    `ExperimentTasks.emptyAnalysisDetail`."""
    from steerlab_server import cli_payloads
    # The historical sentence, when it is what happened.
    assert cli_payloads.empty_analysis_detail("r", 12, {"baseline"}) == (
        cli_payloads.EMPTY_ANALYSIS_NO_CONTRAST)
    assert cli_payloads.EMPTY_ANALYSIS_NO_CONTRAST == (
        "0 effect-size entries — the source run has no non-baseline "
        "condition to pair against")

    # The dry-run case: a contrast that did not pair.
    detail = cli_payloads.empty_analysis_detail(
        "20260818T120000000-exp-s-run", 24, {"baseline", "fear-a2"})
    assert "no non-baseline condition" not in detail
    assert "24 records" in detail
    assert "baseline, fear-a2" in detail
    assert "could not read or pair those records" in detail
    assert "artifacts produced on the other engine" in detail
    assert "record-schema mismatch" in detail

    # Empty and unreadable are different facts.
    empty = cli_payloads.empty_analysis_detail("r", 0, set())
    unreadable = cli_payloads.empty_analysis_detail("r", None, set())
    assert "holds no records at all" in empty
    assert "could not be read here" in unreadable
    assert empty != unreadable

    # Singulars.
    assert "holds 1 record across condition fear-a2" in \
        cli_payloads.empty_analysis_detail("r", 1, {"fear-a2"})


def test_source_run_records_counts_records_and_conditions(tmp_path):
    from steerlab_server import cli_payloads
    directory = _run_dir(tmp_path, "exp-demo-source", {})
    # Missing file: a None count, which is NOT "it was empty".
    assert cli_payloads.source_run_records(directory) == (None, set())
    _run_dir(tmp_path, "exp-demo-source", {
        "generations.jsonl": '{"condition":"baseline","promptID":"p1"}\n'
                             '{"condition":"fear-a2","promptID":"p1"}\n'
                             '{"condition":"fear-a2","promptID":"p2"}\n'})
    count, conditions = cli_payloads.source_run_records(directory)
    assert count == 3
    assert conditions == {"baseline", "fear-a2"}


def test_the_analyze_payload_names_its_effect_size_dialect(tmp_path):
    """Both engines write paired effect sizes; the column names differ by
    idiom and are deliberately NOT unified (runs are immutable and each
    engine's readers depend on its own columns). The envelope NAMES the
    dialect so a cross-engine reader can tell them apart, and any future
    unification is a schema-versioned change announced by this field. Swift
    twin: `ExperimentCLIRunner.effectSizesSchema` == "metric-meanDiff"."""
    from steerlab_server import cli_payloads
    assert cli_payloads.EFFECT_SIZES_SCHEMA == "endpoint-deltaMean"
    assert cli_payloads.EFFECT_SIZES_SCHEMA != "metric-meanDiff"
    directory = _run_dir(tmp_path, "exp-demo-schema", {
        "effect-sizes.csv": ("condition,endpoint,n,deltaMean,ciLower,ciUpper,"
                             "wilcoxonW,wilcoxonP,adjustedP,correction,"
                             "modality\n")})
    payload = cli_payloads.analysis_payload(directory)
    assert payload["effectSizesSchema"] == "endpoint-deltaMean"
