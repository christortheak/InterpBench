"""The cross-condition J-Space report, against a hand-computable fixture.

Every expectation here is arithmetic a reader can redo on paper from
`tests/fixtures/jlens/README.md`. That is deliberate: a report is a pile of
means, and a test that only checked the shape would pass just as happily on
means computed over the wrong steps.
"""

import json
import os
import shutil

import pytest

from steerlab_server.jlens import analysis, report as report_mod

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "jlens",
                       "jlens-readout.jsonl")


@pytest.fixture()
def run_dir(tmp_path):
    directory = tmp_path / "20260815-000000-jlens-fixture"
    directory.mkdir()
    shutil.copy(FIXTURE, directory / "jlens-readout.jsonl")
    return str(directory)


def _built(run_dir, **kwargs):
    return report_mod.build(run_dir, **kwargs)


# --- completeness ------------------------------------------------------------

def test_incomplete_rows_are_excluded_and_counted(run_dir):
    """A truncated generation's rows describe a generation that did not happen
    the way the trace says. Excluding them silently would be as wrong as
    averaging them in — the count is what lets a reader tell the two apart."""
    built = _built(run_dir)
    counts = built["completeness"]
    assert {k: v for k, v in counts.items() if k != "traceSHA256"} == {
        "incompleteByCondition": {"steered": 1},
        "rowsRead": 3, "malformedRows": 0, "rowsUsed": 2}
    # The bytes the report describes, so a report and its trace cannot drift
    # apart silently.
    assert len(counts["traceSHA256"]) == 64


def test_a_wholly_incomplete_trace_refuses(tmp_path):
    """Reporting on nothing produces a table of nulls that reads exactly like a
    measured null result."""
    directory = tmp_path / "run"
    directory.mkdir()
    with open(FIXTURE, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    with open(directory / "jlens-readout.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            row["traceComplete"] = False
            fh.write(json.dumps(row) + "\n")
    with pytest.raises(report_mod.ReportError, match="recorded nothing usable"):
        _built(str(directory))


def test_a_run_with_no_trace_refuses(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(report_mod.ReportError, match="nothing to report"):
        _built(str(tmp_path / "run"))


# --- watchlist aggregates ----------------------------------------------------

def test_the_watchlist_aggregate_is_the_declared_contrast(run_dir):
    """targets {10, 20} minus control {30}, band-averaged.

    Baseline: steps 0-5 score (1+2)/2 − 3 = −1.5; steps 6-11 (token 10
    masked) 2 − 3 = −1.0; layer mean −1.25 at both layers.
    Steered: layer 5 gives (5+2)/2 − 3 = +0.5 then −1.0 → −0.25; layer 9 is
    the baseline's −1.25; the band average is −0.75.
    """
    built = _built(run_dir, token_sets=[{"name": "probe", "targets": [10, 20],
                                         "controls": [30]}])
    probe = built["watchlist"]["probe"]
    assert probe["baseline"]["score"] == pytest.approx(-1.25)
    assert probe["steered"]["score"] == pytest.approx(-0.75)
    assert probe["steered"]["perLayer"]["5"] == pytest.approx(-0.25)
    assert probe["steered"]["perLayer"]["9"] == pytest.approx(-1.25)
    assert probe["baseline"]["convention"] == analysis.SCORE_CONVENTION


def test_a_token_set_with_no_controls_reports_a_LEVEL_and_says_so(run_dir):
    """A level presented as a contrast is the failure this convention stamp
    exists to prevent."""
    built = _built(run_dir, token_sets=[{"name": "raw", "targets": [20]}])
    assert built["watchlist"]["raw"]["baseline"]["convention"] == \
        analysis.RAW_CONVENTION


def test_the_companion_travels_with_every_aggregate(run_dir):
    """The logit-lens companion is the control that says whether transport did
    any work. The fixture's companion is a different constant, so a report
    that reused the J-lens numbers for it would be visible here."""
    built = _built(run_dir, token_sets=[{"name": "probe", "targets": [10, 20],
                                         "controls": [30]}])
    probe = built["watchlist"]["probe"]["steered"]
    assert probe["companionArmed"] is True
    # Companion: targets both 0.5, control 0.5 → 0.0 at every layer, and
    # unchanged by the steering the J-lens column shows.
    assert probe["companionPerLayer"]["5"] == pytest.approx(0.0)
    assert probe["companionPerLayer"]["9"] == pytest.approx(0.0)


# --- mention mask ------------------------------------------------------------

def test_mention_masked_steps_are_excluded_from_per_token_means(run_dir):
    """Token 10 is primed from step 6, so exactly half its steps count. A mean
    over 12 and a mean over 6 differ here by construction."""
    built = _built(run_dir)
    baseline = built["perToken"]["baseline"]["5"]["10"]
    steered = built["perToken"]["steered"]["5"]["10"]
    assert baseline == {"mean": pytest.approx(1.0), "count": 6,
                        "companionMean": pytest.approx(0.5),
                        "companionCount": 6}
    assert steered["mean"] == pytest.approx(5.0)
    assert steered["count"] == 6
    # An unmasked token keeps all twelve.
    assert built["perToken"]["baseline"]["5"]["20"]["count"] == 12


# --- deltas ------------------------------------------------------------------

def test_deltas_identify_the_planted_difference_at_the_planted_layer(run_dir):
    built = _built(run_dir)
    watched = built["deltas"]["watched"]["steered"]
    assert watched["5"]["10"]["delta"] == pytest.approx(4.0)
    assert watched["5"]["10"]["conditionCount"] == 6
    assert watched["5"]["10"]["baselineCount"] == 6
    # Layer 9 was not steered: its delta must be exactly zero, not "small".
    assert watched["9"]["10"]["delta"] == pytest.approx(0.0)


def test_a_token_absent_from_the_baseline_topk_is_flagged_new(run_dir):
    """The most interesting kind of mover, and the easiest to misread as an
    occupancy of zero that was actually measured."""
    built = _built(run_dir)
    cell = built["deltas"]["topK"]["steered"]["5"]["10"]
    assert cell["newInCondition"] is True
    assert cell["baselineOccupancy"] == 0.0
    # Token 10 is in the top-k at all twelve steps and primed at six of them,
    # so it is ELIGIBLE at six and occupies all six: 6/6 = 1.0.
    #
    # This assertion previously said 0.5 — the numerator excluded the primed
    # steps while the denominator counted them, which halved the occupancy of
    # exactly the tokens a study cares most about. The test pinned the bug and
    # the module's own stamped convention described the correct behaviour, so a
    # green suite was not evidence here (external review, 2026-08-16).
    assert cell["conditionOccupancy"] == pytest.approx(1.0)
    assert cell["occupancyDelta"] == pytest.approx(1.0)


def test_deltas_refuse_without_the_named_baseline(run_dir):
    built = _built(run_dir, baseline="no-such-condition")
    assert built["deltas"]["available"] is False
    assert "no condition named" in built["deltas"]["reason"]


# --- top-k roll-up -----------------------------------------------------------

def test_occupancy_divides_by_each_tokens_own_eligible_steps(run_dir):
    """The denominator differs per token, because priming does. Token 10 is
    masked at half the steps and token 20 never is, so both occupy 1.0 of what
    they could — over six eligible steps and twelve respectively."""
    built = _built(run_dir)
    entries = {e["tokenID"]: e
               for e in built["topK"]["steered"]["5"]["tokens"]}
    assert entries[10]["eligibleSteps"] == 6
    assert entries[10]["occupancy"] == pytest.approx(1.0)
    assert entries[20]["eligibleSteps"] == 12
    assert entries[20]["occupancy"] == pytest.approx(1.0)
    # The counts behind them differ, which is the whole point of carrying the
    # denominator: 1.0 over six steps is not 1.0 over twelve.
    assert entries[10]["count"] == 6 and entries[20]["count"] == 12


def test_the_topk_rollup_reports_occupancy_rank_and_the_companion(run_dir):
    built = _built(run_dir)
    block = built["topK"]["steered"]["5"]
    entries = {e["tokenID"]: e for e in block["tokens"]}
    assert entries[20]["occupancy"] == pytest.approx(1.0)
    assert entries[20]["meanRank"] == pytest.approx(2.0)     # always second
    assert entries[10]["meanRank"] == pytest.approx(1.0)     # always first
    assert entries[10]["meanLogit"] == pytest.approx(9.0)
    # The companion's top-k is a different pair, reported side by side.
    assert entries[20]["companionOccupancy"] == pytest.approx(1.0)
    assert entries[10]["companionCount"] == 0
    assert block["companionArmed"] is True


# --- identity ----------------------------------------------------------------

def test_identity_carries_the_lens_and_the_qualification(run_dir):
    built = _built(run_dir)
    identity = built["identity"]
    assert identity["lensID"] == "google--gemma-3-27b-it--jlens-wikitext"
    assert identity["qualificationID"] == "q-0123456789abcdef"
    assert identity["evidenceTier"] == "evidence"
    assert "identityConflicts" not in identity


def _with_rows(tmp_path, mutate):
    directory = tmp_path / "run"
    directory.mkdir()
    with open(FIXTURE, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    mutate(rows)
    with open(directory / "jlens-readout.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return str(directory)


def test_rows_from_two_different_runtimes_REFUSE_rather_than_pool(tmp_path):
    """Disclosing the conflict was not enough (external review, 2026-08-16):
    every aggregate pools rows, so a trace spanning two lens hashes would
    produce one mean over two instruments and label it with neither."""
    def _split(rows):
        rows[1]["lensSHA256"] = "d" * 64

    with pytest.raises(report_mod.ReportError, match="different measurements"):
        _built(_with_rows(tmp_path, _split))


def test_condition_and_prompt_may_vary_because_that_is_the_point(tmp_path):
    """Only IDENTITY may not vary. A report whose conditions had to match
    would have nothing to compare."""
    def _rename(rows):
        rows[0]["condition"] = "another-arm"

    built = _built(_with_rows(tmp_path, _rename))
    assert set(built["conditions"]) == {"another-arm", "steered"}


def test_a_nonfatal_identity_disagreement_is_disclosed_not_refused(tmp_path):
    """`qualificationID` differing across rows is worth surfacing but does not
    make the rows a different measurement."""
    def _requalify(rows):
        rows[1]["qualificationID"] = "q-other"

    identity = _built(_with_rows(tmp_path, _requalify))["identity"]
    assert identity["qualificationID"] is None
    assert len(identity["identityConflicts"]["qualificationID"]) == 2


# --- position profile --------------------------------------------------------

def test_the_position_profile_bands_by_fraction_of_the_generation(run_dir):
    """Bands rather than raw step indices, because generation lengths differ
    per item and raw indices are not comparable across a condition."""
    built = _built(run_dir, bands=2,
                   token_sets=[{"name": "probe", "targets": [10, 20],
                                "controls": [30]}])
    profile = built["positionProfile"]["probe"]["steered"]
    # First half: token 10 unprimed, so (5+2)/2 − 3 = +0.5 at layer 5 and
    # (1+2)/2 − 3 = −1.5 at layer 9 → −0.5 averaged over the band.
    assert profile["0"]["mean"] == pytest.approx(-0.5)
    # Second half: token 10 primed out everywhere → 2 − 3 = −1.0.
    assert profile["1"]["mean"] == pytest.approx(-1.0)
    assert profile["0"]["count"] == 6


# --- persistence -------------------------------------------------------------

def test_writing_produces_the_json_and_both_csvs(run_dir):
    built = report_mod.report(run_dir)
    for key in ("report", "topKCSV", "watchlistCSV"):
        assert os.path.exists(built["paths"][key])
    with open(built["paths"]["report"], encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["conditions"] == ["baseline", "steered"]
    assert stored["conventions"]["statistics"].startswith("counts and means")
    with open(built["paths"]["topKCSV"], encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert "companionOccupancy" in header and "occupancy" in header


def test_the_report_computes_no_null_and_says_so(run_dir):
    """Every quantity here is a mean over steps — the permutation-invariant
    family analysis.permutation_null refuses by name. The stamp is what stops
    someone adding one later without noticing why it was left out."""
    built = _built(run_dir)
    assert built["conventions"]["statistics"] == (
        "counts and means only; no null, no CI, no p-value")
    blob = json.dumps(built)
    for absent in ("\"p\":", "pValue", "ciLower", "ciUpper", "permutation"):
        assert absent not in blob


def test_a_token_set_naming_an_unwatched_token_refuses(run_dir):
    """The trace contains no scores for a token the readout never watched, so
    an aggregate over it would be an average of nothing.

    The import stays inside the function on purpose. It resolves ``JLensError``
    at call time rather than at collection, so it is the suite's canary for a
    test that evicts and re-imports the jlens package without restoring it —
    which is exactly how this test failed once, catching nothing while the
    traceback showed the exception it was meant to catch (see
    ``test_the_package_imports_without_the_reference_extra``).
    """
    from steerlab_server.jlens.schemas import JLensError

    with pytest.raises(JLensError, match="never watched"):
        _built(run_dir, token_sets=[{"name": "bad", "targets": [999]}])


# --- publication (external review round 2) -----------------------------------

def test_the_json_is_canonical_and_hashes_its_derivatives(run_dir):
    """Three sequential replacements are not one transaction. What makes a
    torn publication DETECTABLE is that the canonical JSON is written last and
    names the exact CSV bytes this pass wrote."""
    import hashlib

    report_mod.report(run_dir)
    # Through the SHARED verifier, not a hand-rolled check: a consumer that
    # reimplements this is a consumer that can forget it.
    assert report_mod.verify_publication(run_dir)["coherent"] is True

    # A stale CSV beside a fresh JSON is visible rather than silent.
    with open(os.path.join(run_dir, report_mod.TOPK_CSV), "a") as handle:
        handle.write("tampered\n")
    verdict = report_mod.verify_publication(run_dir)
    assert verdict["coherent"] is False
    assert any(report_mod.TOPK_CSV in p for p in verdict["problems"])


def test_the_verifier_names_a_missing_derivative(run_dir):
    report_mod.report(run_dir)
    os.remove(os.path.join(run_dir, report_mod.WATCHLIST_CSV))
    verdict = report_mod.verify_publication(run_dir)
    assert verdict["coherent"] is False
    assert any("is missing" in p for p in verdict["problems"])


def test_the_verifier_reports_rather_than_raises_on_an_absent_report(tmp_path):
    """A viewer may want to SHOW an incoherent set rather than refuse over
    it, and the JSON alone is still usable."""
    (tmp_path / "run").mkdir()
    verdict = report_mod.verify_publication(str(tmp_path / "run"))
    assert verdict["coherent"] is False and verdict["problems"]


def test_the_topk_csv_carries_the_occupancy_denominator(run_dir):
    """A flat table that dropped it would let 1.0 over six eligible steps read
    like 1.0 over two hundred."""
    built = report_mod.report(run_dir)
    with open(built["paths"]["topKCSV"], encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        rows = [line.strip().split(",") for line in handle if line.strip()]
    assert "eligibleSteps" in header and "companionEligibleSteps" in header
    column = header.index("eligibleSteps")
    assert any(int(r[column]) == 6 for r in rows), "the primed token's 6"


def test_rows_recorded_under_different_readout_configs_refuse_to_pool(tmp_path):
    """The trace row now carries configHash, so pooling can see that two rows
    were produced under different watchlists or top-k widths — which it could
    not before, since the hash lived only in generations.jsonl."""
    def _reconfigure(rows):
        rows[1]["configHash"] = "e" * 64

    with pytest.raises(report_mod.ReportError, match="different measurements"):
        _built(_with_rows(tmp_path, _reconfigure))


def test_the_verifier_refuses_a_report_that_names_a_foreign_path(run_dir):
    """A report is DATA, including an imported one. Joining its keys onto the
    run directory made a modified report able to read anywhere (external
    review round 4)."""
    report_mod.report(run_dir)
    path = os.path.join(run_dir, report_mod.REPORT_FILENAME)
    stored = json.load(open(path, encoding="utf-8"))
    stored["derivedArtifacts"] = {"../../../../etc/passwd": "0" * 64}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stored, handle)

    verdict = report_mod.verify_publication(run_dir)
    assert verdict["coherent"] is False
    assert any("refusing to read it" in p for p in verdict["problems"])
    # It refused rather than reporting a hash for a file it read.
    assert verdict["checked"] == {}


# --- external review rounds 7-8: the claim is per CONDITION, and MEASURED --
#
# These fixtures build REAL ModelVariant objects over REAL files on disk.
# The round-7 versions used stand-in `_Adapter` objects with attribute
# access, and passed while production — where `ModelVariant` keeps adapters
# as DICTS — resolved every pinned agent to "?" / unpinned / exploratory.
# A test that validates a substitute shape validates nothing.

import hashlib
import json as _json
import os

import pytest

from steerlab_server.experiment.model_variant import (
    AdapterIdentityError, ModelVariant)
from steerlab_server.jlens import trace as _trace


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _adapter_on_disk(root, name="sympathy", config=None):
    directory = os.path.join(root, "adapters", name)
    os.makedirs(directory, exist_ok=True)
    weights = os.path.join(directory, "adapter_model.safetensors")
    with open(weights, "wb") as handle:
        handle.write(b"WEIGHTS-" + name.encode())
    config_path = os.path.join(directory, "adapter_config.json")
    with open(config_path, "w") as handle:
        _json.dump(config or {"r": 16, "target_modules": ["q_proj"]}, handle)
    return directory, weights, config_path


def _variant(root, *, pin_config=True, pin_weights=True, name="sympathy"):
    directory, weights, config_path = _adapter_on_disk(root, name)
    declared = {"name": f"{name}-lora",
                "adapterDirectory": os.path.relpath(directory, root)}
    if pin_weights:
        declared["adapterHash"] = _sha(weights)
    if pin_config:
        declared["configHash"] = _sha(config_path)
    # from_dict is the production decode path — adapters stay DICTS.
    return ModelVariant.from_dict({
        "name": f"{name}-agent", "baseModelID": "google/gemma-3-27b-it",
        "adapters": [declared]})


class _Eff:
    def __init__(self, name, variant=None):
        self.name = name
        self.variant = variant


def test_a_real_pinned_variant_reads_as_pinned_and_verified(tmp_path):
    """The round-8 regression, stated as the behaviour it broke: a fully
    pinned production Agent must not resolve to "?" / unpinned."""
    root = str(tmp_path)
    identity = _trace.condition_identity(
        _Eff("arm", _variant(root)), root)

    adapter = identity["adapters"][0]
    assert adapter["adapterDirectory"] == "adapters/sympathy"
    assert adapter["adapterHashPinned"] and adapter["adapterHashVerified"]
    assert adapter["configHashPinned"] and adapter["configHashVerified"]
    assert _trace.condition_claim("evidence", "qual-1", identity) == "qualified"


def test_a_baseline_condition_has_no_identity_to_pin():
    """Nothing to pin is not the same as pinned, and not the same as unpinned:
    a baseline runs no adapter, so its claim rests on the lens alone."""
    assert _trace.condition_identity(_Eff("baseline")) is None
    assert _trace.condition_claim("evidence", "qual-1", None) == "qualified"


def test_a_legacy_agent_without_a_config_pin_is_exploratory(tmp_path):
    """The scenario the policy exists for — and it must NOT refuse: legacy
    agents stay usable for exploration."""
    root = str(tmp_path)
    identity = _trace.condition_identity(
        _Eff("legacy-arm", _variant(root, pin_config=False)), root)

    adapter = identity["adapters"][0]
    assert adapter["adapterHashVerified"] is True
    assert adapter["configHashPinned"] is False
    assert "configurationUnpinned" in adapter
    assert _trace.condition_claim("evidence", "qual-1", identity) == "exploratory"


def test_a_drifted_adapter_refuses_rather_than_downgrades(tmp_path):
    """Presence of a declaration is not verification. Bytes that do not match
    the pin are not a weaker claim — they are the wrong adapter, and nothing
    measured through them is attributable."""
    root = str(tmp_path)
    variant = _variant(root)
    # Someone retrains into the same path after the agent was pinned.
    with open(os.path.join(root, "adapters", "sympathy",
                           "adapter_model.safetensors"), "wb") as handle:
        handle.write(b"DIFFERENT-WEIGHTS")

    with pytest.raises(AdapterIdentityError, match="not the one this agent"):
        _trace.condition_identity(_Eff("arm", variant), root)


def test_an_edited_config_after_pinning_refuses(tmp_path):
    """The whole point of pinning the configuration: rank and target modules
    change what the forward pass does."""
    root = str(tmp_path)
    variant = _variant(root)
    with open(os.path.join(root, "adapters", "sympathy",
                           "adapter_config.json"), "w") as handle:
        _json.dump({"r": 64, "target_modules": ["q_proj", "v_proj"]}, handle)

    with pytest.raises(AdapterIdentityError, match="not the one this agent"):
        _trace.condition_identity(_Eff("arm", variant), root)


def test_an_unqualified_lens_is_exploratory_however_well_pinned(tmp_path):
    """Both halves must hold."""
    root = str(tmp_path)
    identity = _trace.condition_identity(_Eff("arm", _variant(root)), root)
    assert _trace.condition_claim("evidence", None, identity) == "exploratory"
    assert _trace.condition_claim("testing", "qual-1", identity) == "exploratory"


def test_the_probe_and_the_study_path_share_one_verifier(tmp_path):
    """Not "both are correct" — the SAME implementation. Two copies is how
    the study path drifted into attribute access in the first place."""
    from steerlab_server.experiment import model_variant
    from steerlab_server.jlens import probe

    assert probe.ADAPTER_CONTENT_HASH_ALGORITHM == (
        model_variant.ADAPTER_CONTENT_HASH_ALGORITHM)
    root = str(tmp_path)
    variant = _variant(root)
    shared = model_variant.verified_adapter_identity(variant, root)
    via_trace = _trace.condition_identity(_Eff("arm", variant), root)
    assert via_trace["adapters"] == shared


def test_the_trace_identity_carries_the_claim_into_the_row():
    """The rule existing is not the point — reaching the ROW is."""
    row = _trace.TraceIdentity(
        run="r", condition="legacy-arm", evidenceTier="evidence",
        conditionClaim="exploratory",
        conditionIdentity={"adapters": [
            {"adapterDirectory": "adapters/legacy",
             "configHashPinned": False}]}).to_dict()
    assert row["conditionClaim"] == "exploratory"
    assert row["evidenceTier"] == "evidence"   # the LENS is still qualified


def test_the_report_states_eligibility_per_condition_with_a_reason():
    """One run, two conditions, two different answers — which is the whole
    point of moving the claim off the run-level tier."""
    from steerlab_server.jlens.report import _condition_eligibility

    out = _condition_eligibility([
        {"condition": "baseline", "conditionClaim": "qualified"},
        {"condition": "legacy-arm", "conditionClaim": "exploratory",
         "conditionIdentity": {"adapters": [
             {"adapterDirectory": "adapters/legacy", "configHashPinned": False}]}},
    ])
    assert out["baseline"]["claim"] == "qualified"
    assert out["legacy-arm"]["claim"] == "exploratory"
    assert "unpinned" in out["legacy-arm"]["note"]
    assert "adapters/legacy" in out["legacy-arm"]["note"]


def test_one_exploratory_row_makes_the_whole_condition_exploratory():
    """Weakest wins — a condition is not reportable in part."""
    from steerlab_server.jlens.report import _condition_eligibility

    out = _condition_eligibility([
        {"condition": "arm", "conditionClaim": "qualified"},
        {"condition": "arm", "conditionClaim": "exploratory"},
    ])
    assert out["arm"]["claim"] == "exploratory"
    assert out["arm"]["rows"] == 2


def test_a_legacy_trace_reads_unstamped_not_qualified():
    """A claim that was never evaluated is not a claim that passed."""
    from steerlab_server.jlens.report import _condition_eligibility

    out = _condition_eligibility([{"condition": "arm", "evidenceTier": "evidence"}])
    assert out["arm"]["claim"] == "unstamped"
    assert "exploratory" in out["arm"]["note"]


def test_two_agents_under_one_condition_name_refuse_to_pool():
    """Merged shards can disagree about WHICH agent a condition ran. The
    run-level guard does not cover it — lens and model are uniform while the
    agent is not — and the rollup kept whichever identity it saw first."""
    from steerlab_server.jlens.report import (
        ReportError, require_consistent_condition_identity)

    rows = [
        {"condition": "arm", "conditionIdentity": {
            "variantName": "a", "adapters": [
                {"adapterDirectory": "adapters/a", "adapterHashLive": "aa"}]}},
        {"condition": "arm", "conditionIdentity": {
            "variantName": "b", "adapters": [
                {"adapterDirectory": "adapters/b", "adapterHashLive": "bb"}]}},
    ]
    with pytest.raises(ReportError, match="CONDITION identity disagrees"):
        require_consistent_condition_identity(rows)


def test_the_same_agent_across_shards_pools_fine():
    """The guard must not fire on the normal case: one agent, many shards."""
    from steerlab_server.jlens.report import require_consistent_condition_identity

    identity = {"variantName": "a", "adapters": [
        {"adapterDirectory": "adapters/a", "adapterHashLive": "aa",
         "configurationUnpinned": "prose that may be reworded"}]}
    other = {"variantName": "a", "adapters": [
        {"adapterDirectory": "adapters/a", "adapterHashLive": "aa",
         "configurationUnpinned": "DIFFERENT prose, same agent"}]}
    require_consistent_condition_identity([
        {"condition": "arm", "conditionIdentity": identity},
        {"condition": "arm", "conditionIdentity": other},
    ])


# --- external review round 9 ------------------------------------------------


def test_verification_happens_once_per_condition_not_once_per_row(tmp_path):
    """Hashing is file I/O over adapter weights — hundreds of MB for a 27B
    LoRA. An identity is immutable for the life of a run, so re-verifying per
    generated row re-reads those bytes thousands of times to learn the same
    answer.

    Counted through the session's OWN lookup (the one record_generation
    calls), not a re-implementation of it.
    """
    root = str(tmp_path)
    session = _trace.TraceSession.__new__(_trace.TraceSession)
    session.root = root
    session.condition_identities = {}

    reads = []
    real_open = open

    def counting_open(path, *args, **kwargs):
        if str(path).endswith((".safetensors", "adapter_config.json")):
            reads.append(str(path))
        return real_open(path, *args, **kwargs)

    eff = _Eff("sympathy-arm", _variant(root))
    import builtins
    builtins.open = counting_open
    try:
        identities = [session.identity_for(eff) for _ in range(50)]
    finally:
        builtins.open = real_open

    # Some reads for the one verification; none for the other 49 rows.
    assert reads, "expected the first call to actually hash the bytes"
    first_pass = len(reads)
    assert first_pass < 20, f"one verification should not read {first_pass} times"
    assert identities[0] is identities[-1], "identity was recomputed per row"
    assert identities[0]["adapters"][0]["adapterHashVerified"] is True


def test_the_run_start_guard_returns_identities_for_the_trace_to_reuse(tmp_path):
    """The verification the run-start guard already performs is the SAME work
    the rows need, so it hands the result on rather than throwing it away."""
    import types

    from steerlab_server.experiment import tasks

    root = str(tmp_path)
    variant = _variant(root)
    manifest = types.SimpleNamespace(variant_conditions=[
        types.SimpleNamespace(name="sympathy-arm", artifact=variant.to_dict())])

    identities = tasks._require_verified_variant_identities(manifest, root)

    assert set(identities) == {"sympathy-arm"}
    adapter = identities["sympathy-arm"]["adapters"][0]
    assert adapter["adapterHashVerified"] and adapter["configHashVerified"]


def test_a_prestamped_identity_is_used_verbatim_for_every_row(tmp_path):
    """The session must PREFER the supplied map — one that ignored it would
    still be correct, just slow, which is the failure this change is about."""
    session = _trace.TraceSession.__new__(_trace.TraceSession)
    session.root = str(tmp_path)
    session.condition_identities = {
        "arm": {"variantName": "pre", "adapters": [{"adapterDirectory": "x"}]}}

    # No variant on the eff at all: a session that recomputed would get None.
    assert session.identity_for(_Eff("arm", None))["variantName"] == "pre"


def test_missing_and_present_identities_under_one_condition_refuse():
    """Absent is a VALUE, not a row to skip. One shard carrying an identity
    and another carrying none are two different measurements under one name;
    skipping the absent rows let exactly that pool silently."""
    from steerlab_server.jlens.report import (
        ReportError, require_consistent_condition_identity)

    rows = [
        {"condition": "arm", "conditionIdentity": {
            "variantName": "a", "adapters": [
                {"adapterDirectory": "adapters/a", "adapterHashLive": "aa"}]}},
        {"condition": "arm"},                      # legacy shard, no identity
    ]
    with pytest.raises(ReportError, match="UNSTAMPED"):
        require_consistent_condition_identity(rows)


def test_all_missing_identities_are_consistent():
    """A wholly legacy trace is not a conflict — every row says the same
    (unstamped) thing, and the eligibility rollup reports it as such."""
    from steerlab_server.jlens.report import require_consistent_condition_identity

    require_consistent_condition_identity([
        {"condition": "arm"}, {"condition": "arm"}, {"condition": "baseline"}])


# --- external review round 10 ----------------------------------------------


def _variant_on_disk(root, name="sympathy"):
    """Write the agent as a FILE and return its workspace-relative path — the
    `artifactPath` form, which is as supported as the embedded one."""
    variant = _variant(root, name=name)
    directory = os.path.join(root, "runs", "model-variants", name)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "model-variant.json")
    with open(path, "w") as handle:
        _json.dump(variant.to_dict(), handle)
    return os.path.relpath(path, root)


def test_a_path_backed_condition_is_verified_at_preflight_too(tmp_path):
    """Only the embedded `artifact` was handled, so a path-backed condition
    preflighted to nothing and fell through to lazy per-row verification —
    reintroducing BOTH previous rounds' bugs for exactly that path."""
    import types

    from steerlab_server.experiment import tasks

    root = str(tmp_path)
    manifest = types.SimpleNamespace(variant_conditions=[
        types.SimpleNamespace(name="arm", artifact=None,
                              artifact_path=_variant_on_disk(root))])

    identities = tasks._require_verified_variant_identities(manifest, root)

    assert set(identities) == {"arm"}, "path-backed condition was not verified"
    assert identities["arm"]["adapters"][0]["adapterHashVerified"] is True


def test_a_drifted_path_backed_adapter_refuses_at_preflight(tmp_path):
    import types

    from steerlab_server.experiment import tasks

    root = str(tmp_path)
    path = _variant_on_disk(root)
    with open(os.path.join(root, "adapters", "sympathy",
                           "adapter_model.safetensors"), "wb") as handle:
        handle.write(b"RETRAINED")
    manifest = types.SimpleNamespace(variant_conditions=[
        types.SimpleNamespace(name="arm", artifact=None, artifact_path=path)])

    with pytest.raises(RuntimeError, match="cannot be verified"):
        tasks._require_verified_variant_identities(manifest, root)


def test_a_missing_artifact_path_is_left_to_the_run_loop(tmp_path):
    """Absence is the run loop's error to report, per condition, in its own
    vocabulary — the preflight must not pre-empt it with a different one."""
    import types

    from steerlab_server.experiment import tasks

    manifest = types.SimpleNamespace(variant_conditions=[
        types.SimpleNamespace(name="arm", artifact=None,
                              artifact_path="runs/model-variants/gone.json")])

    assert tasks._require_verified_variant_identities(manifest, str(tmp_path)) == {}


def test_the_identity_attached_to_the_condition_beats_the_name_cache(tmp_path):
    """Condition names are not guaranteed unique, so a name-keyed lookup can
    hand one agent's rows another agent's identity. The identity verified
    immediately before THIS condition's adapter loaded is authoritative."""
    import types

    session = _trace.TraceSession.__new__(_trace.TraceSession)
    session.root = str(tmp_path)
    session.condition_identities = {"arm": {"variantName": "WRONG-AGENT"}}
    eff = types.SimpleNamespace(name="arm", variant=None,
                                verified_identity={"variantName": "RIGHT-AGENT"})

    assert session.identity_for(eff)["variantName"] == "RIGHT-AGENT"


def test_a_variant_named_baseline_collides_with_the_implicit_baseline(tmp_path):
    """The collision a declared-collections scan cannot see: every run
    executes a baseline, and it is in no declared collection. `record_key`
    LEADS with the condition name, so the two alias into one key — resume
    skipping treats one condition's work as the other's, shard membership
    double-counts, and merge completeness checks the wrong cell list."""
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path)
    manifest = Manifest.from_dict({
        "name": "x", "modelID": "m", "studyKind": "modelOutput",
        "variantConditions": [{"name": "baseline",
                               "artifactPath": _variant_on_disk(root),
                               "artifactHash": "h"}]})

    assert any("duplicate condition name" in v for v in manifest.verify(root))


def test_inert_carried_conditions_do_not_cause_a_false_refusal(tmp_path):
    """The other direction. A variant comparison runs baseline + variants
    ONLY — declared injection conditions are carried but never executed — so
    duplicates among them cannot alias any record key, and refusing would
    block a study over configuration that does not run."""
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path)
    manifest = Manifest.from_dict({
        "name": "x", "modelID": "m", "studyKind": "modelOutput",
        "studyType": "agentComparison",
        "conditions": [{"name": "carried"}, {"name": "carried"}],
        "variantConditions": [{"name": "sympathy-arm",
                               "artifactPath": _variant_on_disk(root),
                               "artifactHash": "h"}]})

    assert not any("duplicate condition name" in v for v in manifest.verify(root))


def test_the_verify_gate_and_the_run_loop_share_one_resolver(tmp_path):
    """Not "both are correct" — the SAME function. A second copy is how the
    gate came to disagree with execution about what a study runs."""
    from steerlab_server.experiment import manifest as manifest_mod
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path)
    manifest = Manifest.from_dict({
        "name": "x", "modelID": "m", "studyKind": "modelOutput",
        "conditions": [{"name": "carried"}],
        "variantConditions": [{"name": "sympathy-arm",
                               "artifactPath": _variant_on_disk(root),
                               "artifactHash": "h"}]})

    # Variant comparison: baseline + variants, carried conditions dropped.
    assert [c.name for c in manifest_mod.effective_conditions(manifest)] == ["baseline"]
    assert manifest_mod.effective_condition_names(manifest) == [
        "baseline", "sympathy-arm"]


def test_distinct_condition_names_verify_clean(tmp_path):
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path)
    manifest = Manifest.from_dict({
        "name": "x", "modelID": "m", "studyKind": "modelOutput",
        "conditions": [{"name": "baseline"}],
        "variantConditions": [{"name": "sympathy-arm",
                               "artifactPath": _variant_on_disk(root),
                               "artifactHash": "h"}]})

    assert not any("duplicate condition name" in v for v in manifest.verify(root))


# --- external review round 12 -----------------------------------------------


def _multi_agent(root, **extra):
    from steerlab_server.experiment.manifest import Manifest

    payload = {"name": "x", "modelID": "m", "studyKind": "multiAgent",
               "multiAgentScenarioPath": "scenarios/s.json",
               "multiAgentScenarioHash": "h"}
    payload.update(extra)
    return Manifest.from_dict(payload)


def test_a_multi_agent_study_carrying_a_baseline_variant_is_legal(tmp_path):
    """A multi-agent manifest may legally CARRY model-output configuration
    under the never-delete rule, and none of it executes. Judging it by the
    model-output matrix invented conditions it never runs: a carried variant
    named "baseline" resolved to ["baseline", "baseline"] and refused a legal
    manifest."""
    root = str(tmp_path)
    manifest = _multi_agent(root, variantConditions=[
        {"name": "baseline", "artifactPath": _variant_on_disk(root),
         "artifactHash": "h"}])

    assert not any("duplicate condition name" in v for v in manifest.verify(root))


def test_a_multi_agent_study_carrying_duplicate_conditions_is_legal(tmp_path):
    root = str(tmp_path)
    manifest = _multi_agent(root, conditions=[{"name": "dup"}, {"name": "dup"}])

    assert not any("duplicate condition name" in v for v in manifest.verify(root))


def test_multi_agent_names_are_the_panels_own_vocabulary(tmp_path):
    """Not "no names" — the RIGHT names. A panel run keys records on
    "configured" plus "baseline" when a baseline play-through is included
    (tasks._run_multi_agent_impl), so the resolver's contract — every name
    that will key a record — stays true for this study kind too."""
    from steerlab_server.experiment.manifest import effective_condition_names

    root = str(tmp_path)
    assert effective_condition_names(_multi_agent(root)) == [
        "configured", "baseline"]
    assert effective_condition_names(
        _multi_agent(root, multiAgentIncludeBaseline=False)) == ["configured"]


def test_carried_model_output_config_contributes_no_multi_agent_names(tmp_path):
    """The inertness is the point: carried configuration keys no record."""
    from steerlab_server.experiment.manifest import (
        effective_conditions, effective_condition_names)

    root = str(tmp_path)
    manifest = _multi_agent(root, conditions=[{"name": "carried"}],
                            variantConditions=[
                                {"name": "carried-agent",
                                 "artifactPath": _variant_on_disk(root),
                                 "artifactHash": "h"}])

    assert effective_conditions(manifest) == []
    assert "carried" not in effective_condition_names(manifest)
    assert "carried-agent" not in effective_condition_names(manifest)
