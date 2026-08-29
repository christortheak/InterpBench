"""The evaluate subsample — self-contained, and pinned by literals.

Cross-engine twin: ``Tests/ExperimentKitTests/EvaluateSubsampleTests.swift``.
The LITERAL blocks below appear byte-identically in both suites: the same
stream-seed derivation, the same seeded promptID order, the same allotments,
and the same chosen ``(condition, promptID, sampleIndex)`` triples. Two things
depend on that.

1. **A runtime upgrade cannot move the draw undetected.** The membership of a
   subsample IS the evidence — a report that says ``samplePerCondition: 2400``
   and ``sampleSeed: 0x…`` claims those 2,400 records are recomputable from
   those two numbers forever. Every step here is 64-bit integer arithmetic and
   SHA-256, so nothing in it can drift with the interpreter (the argument
   ``token_bank_downsampling`` makes at length, and the reason its SplitMix64
   is the primitive this module borrows).
2. **The two engines draw the SAME records.** A study whose corpus is coded on
   the cluster and whose numbers are checked on the Mac must be looking at one
   subsample, not two that happen to be the same size.
"""

from __future__ import annotations

import pytest

from steerlab_server.experiment import evaluate_subsample as subsample

# ------------------------------------------------------------------- literals
#
# Twin values, asserted verbatim on both engines.
LITERAL_STREAM_SEED_42_BASELINE = 11_560_963_923_506_615_204
LITERAL_STREAM_SEED_42_BASELINE_P01 = 1_292_222_067_763_893_346
LITERAL_SEEDED_ORDER_5_42 = [3, 4, 2, 0, 1]
LITERAL_ALLOTMENTS_444_7 = [3, 2, 2]

#: The shared cross-engine fixture: a synthetic run of 2 conditions × 3
#: promptIDs × 4 sampleIndexes = 24 codeable records, sampled at
#: ``--sample-per-condition 7 --sample-seed 0x2a``. Exactly these 14 triples,
#: in exactly this order, on both engines.
LITERAL_FIXTURE_SELECTION = [
    ("baseline", "p01", 0),
    ("baseline", "p01", 1),
    ("baseline", "p01", 3),
    ("baseline", "p02", 1),
    ("baseline", "p02", 3),
    ("baseline", "p03", 0),
    ("baseline", "p03", 1),
    ("injected", "p01", 0),
    ("injected", "p01", 1),
    ("injected", "p02", 0),
    ("injected", "p02", 2),
    ("injected", "p02", 3),
    ("injected", "p03", 0),
    ("injected", "p03", 2),
]

#: The rule's SHA-256, pinned on BOTH engines. The prose below is this
#: engine's copy; the digest is what says the OTHER engine stamps the same
#: bytes into the same report field — pinned by digest rather than by a second
#: copy of the paragraph, because a duplicated paragraph is one that silently
#: rots on one side. Twin literal: ``EvaluateSubsampleTests.literalRuleSHA256``.
LITERAL_RULE_SHA256 = \
    "92cca1c80532c5d2fbf96734dd3b5feb2d767ec90f6ac8fdf65152aab55f049d"

LITERAL_RULE = (
    "stratifiedByPromptID/v1 — within each condition, floor(n / P) records "
    "per promptID over that condition's P promptIDs, the n mod P remainder "
    "given one at a time to promptIDs in seeded order (a promptID already at "
    "its codeable population is skipped and its quota passes on, so exactly "
    "n records are always drawn); within each (condition, promptID) cell the "
    "records are drawn over sampleIndex ascending. Every draw is a partial "
    "Fisher-Yates over SplitMix64 seeded from the first 8 bytes, big-endian, "
    "of SHA-256(seed as 8-byte big-endian || each part length-prefixed as an "
    "8-byte big-endian UTF-8 byte count followed by those bytes): the parts "
    "are (condition,) for the promptID order and (condition, promptID) for a "
    "cell draw. promptIDs are ordered by UTF-8 bytes. Kept records stay in "
    "their source-run order."
)


def fixture_records() -> list[dict]:
    """The 24-record synthetic run the literal selection is drawn from."""
    return [
        {"condition": condition, "promptID": prompt, "sampleIndex": index,
         "output": f"{condition}/{prompt}[{index}]"}
        for condition in ("baseline", "injected")
        for prompt in ("p01", "p02", "p03")
        for index in range(4)
    ]


def triples(records: list[dict]) -> list[tuple]:
    return [(r["condition"], r["promptID"], r["sampleIndex"]) for r in records]


# ------------------------------------------------------------------ the draw


def test_stream_seed_is_the_pinned_derivation():
    assert subsample.stream_seed(42, "baseline") == \
        LITERAL_STREAM_SEED_42_BASELINE
    assert subsample.stream_seed(42, "baseline", "p01") == \
        LITERAL_STREAM_SEED_42_BASELINE_P01


def test_stream_seed_length_prefixes_its_parts():
    """``("ab", "c")`` and ``("a", "bc")`` must not collide — without the
    length prefix two different cells of one condition would share a draw."""
    assert subsample.stream_seed(1, "ab", "c") != \
        subsample.stream_seed(1, "a", "bc")


def test_seeded_order_is_the_pinned_permutation():
    assert subsample.seeded_order(5, 42) == LITERAL_SEEDED_ORDER_5_42
    assert sorted(subsample.seeded_order(9, 7)) == list(range(9))


def test_allotments_are_floor_plus_a_seeded_remainder():
    assert subsample._allotments(
        [4, 4, 4], 7, subsample.stream_seed(42, "baseline")) == \
        LITERAL_ALLOTMENTS_444_7
    assert sum(LITERAL_ALLOTMENTS_444_7) == 7


def test_allotments_skip_a_full_cell_and_still_place_exactly_n():
    """A ragged source run (a partial or resumed one) still yields exactly n:
    a promptID at its population is skipped and its quota passes on."""
    allot = subsample._allotments([1, 10, 10], 9, 99)
    assert sum(allot) == 9
    assert allot[0] <= 1


def test_selection_is_the_pinned_cross_engine_fixture():
    kept, stamp = subsample.select(
        fixture_records(), subsample.SubsampleRequest(7, 0x2A),
        program="steerlab-server")
    assert triples(kept) == LITERAL_FIXTURE_SELECTION
    assert stamp["samplePerCondition"] == 7
    assert stamp["sampleSeed"] == "0x000000000000002a"
    assert stamp["sampledRecords"] == 14
    assert stamp["sourceRecords"] == 24
    assert stamp["rule"] == LITERAL_RULE


def test_selection_is_deterministic_and_seed_sensitive():
    records = fixture_records()
    first, _ = subsample.select(records, subsample.SubsampleRequest(7, 0x2A),
                                program="p")
    again, _ = subsample.select(records, subsample.SubsampleRequest(7, 0x2A),
                                program="p")
    other, _ = subsample.select(records, subsample.SubsampleRequest(7, 0x2B),
                                program="p")
    assert triples(first) == triples(again)
    assert triples(first) != triples(other)


def test_every_condition_contributes_exactly_n():
    kept, _ = subsample.select(
        fixture_records(), subsample.SubsampleRequest(5, 12345), program="p")
    counts = subsample.population_by_condition(kept)
    assert counts == {"baseline": 5, "injected": 5}


def test_kept_records_stay_in_source_order():
    records = fixture_records()
    kept, _ = subsample.select(records, subsample.SubsampleRequest(6, 9),
                               program="p")
    positions = [records.index(record) for record in kept]
    assert positions == sorted(positions)


def test_a_full_size_sample_keeps_everything():
    records = fixture_records()
    kept, stamp = subsample.select(records, subsample.SubsampleRequest(12, 3),
                                   program="p")
    assert triples(kept) == triples(records)
    assert stamp["sampledRecords"] == stamp["sourceRecords"] == 24


# ------------------------------------------------------------------ refusals


def test_a_sample_without_a_seed_refuses():
    with pytest.raises(subsample.SubsampleRefusal) as caught:
        subsample.resolve_request(2400, None, program="steerlab-server")
    assert caught.value.code == "sampleSeedMissing"
    assert "nobody can redraw" in caught.value.reason
    assert "--sample-seed" in caught.value.repair_action


def test_a_seed_without_a_sample_size_refuses():
    with pytest.raises(subsample.SubsampleRefusal) as caught:
        subsample.resolve_request(None, "0x2a", program="steerlab-server")
    assert caught.value.code == "sampleSizeMissing"
    assert "did not shape" in caught.value.reason
    assert "--sample-per-condition" in caught.value.repair_action


def test_neither_flag_is_the_full_corpus():
    assert subsample.resolve_request(None, None, program="p") is None
    assert subsample.resolve_request("", "", program="p") is None


def test_both_flags_resolve():
    request = subsample.resolve_request("2400", "0x2a", program="p")
    assert request.sample_per_condition == 2400
    assert request.seed == 42
    assert request.seed_text == "0x000000000000002a"


@pytest.mark.parametrize("raw", ["not-a-seed", "-1", "0x" + "f" * 17])
def test_a_malformed_seed_refuses(raw):
    with pytest.raises(subsample.SubsampleRefusal) as caught:
        subsample.resolve_request(1, raw, program="p")
    assert caught.value.code == "sampleSeedMalformed"


def test_a_blank_seed_reads_as_the_missing_half_not_a_malformed_one():
    """``--sample-seed ''`` alongside a size is the SEED-MISSING refusal: the
    repair a caller needs is "name a seed", not "that seed is malformed"."""
    with pytest.raises(subsample.SubsampleRefusal) as caught:
        subsample.resolve_request(1, "   ", program="p")
    assert caught.value.code == "sampleSeedMissing"
    with pytest.raises(subsample.SubsampleRefusal) as direct:
        subsample.parse_seed("", program="p")
    assert direct.value.code == "sampleSeedMalformed"


@pytest.mark.parametrize("raw", ["0", "-3", "two"])
def test_a_malformed_sample_size_refuses(raw):
    with pytest.raises(subsample.SubsampleRefusal) as caught:
        subsample.resolve_request(raw, "0x2a", program="p")
    assert caught.value.code == "sampleSizeMalformed"


def test_an_over_ask_refuses_rather_than_clamping():
    with pytest.raises(subsample.SubsampleRefusal) as caught:
        subsample.select(fixture_records(), subsample.SubsampleRequest(13, 1),
                         program="steerlab-server")
    assert caught.value.code == "samplePopulation"
    assert "'baseline' has 12" in caught.value.reason
    assert "clamping" in caught.value.reason
    assert "--sample-per-condition 12 or less" in caught.value.repair_action


def test_an_over_ask_names_the_binding_stratum():
    """The repair names the SMALLEST condition, because that is the number the
    caller has to re-derive their design around — not whichever condition
    happened to be enumerated first."""
    records = [r for r in fixture_records()
               if not (r["condition"] == "injected" and r["sampleIndex"] > 0)]
    with pytest.raises(subsample.SubsampleRefusal) as caught:
        subsample.select(records, subsample.SubsampleRequest(12, 1),
                         program="p")
    assert "'injected' has 3" in caught.value.reason
    assert "--sample-per-condition 3 or less" in caught.value.repair_action


def test_the_paired_refusal_names_the_unit_of_analysis():
    refusal = subsample.paired_refusal("steerlab-server")
    assert refusal.code == "sampleUnsupportedInstrument"
    assert "PAIR" in refusal.reason
    assert "perResponseCoding" in refusal.repair_action


# ------------------------------------------------------------------ stamping


def test_the_human_phrase_cannot_be_read_as_a_full_corpus():
    stamp = subsample.stamp(subsample.SubsampleRequest(2400, 42),
                            sampled=2400, source=7200)
    assert subsample.coded_phrase(stamp, 7200) == \
        "2400 of 7200 record(s) (seeded subsample)"


def test_an_absent_block_is_the_legacy_full_corpus_line():
    assert subsample.coded_phrase(None, 7200) == "7200 record(s)"


def test_the_rule_is_the_pinned_cross_engine_derivation():
    import hashlib
    digest = hashlib.sha256(subsample.RULE.encode("utf-8")).hexdigest()
    assert digest == LITERAL_RULE_SHA256
    assert subsample.RULE == LITERAL_RULE
    assert "\n" not in subsample.RULE


def test_the_stamp_carries_exactly_the_five_declared_keys():
    stamp = subsample.stamp(subsample.SubsampleRequest(3, 7),
                            sampled=6, source=24)
    assert set(stamp) == {"rule", "samplePerCondition", "sampleSeed",
                          "sampledRecords", "sourceRecords"}
    assert stamp["rule"] == LITERAL_RULE


# ----------------------------------------------------- the surfaces it reaches


def test_the_submission_renders_both_flags_only_when_asked():
    """The `--source` rule (2026-08-29): a submission that never mentions a
    subsample renders the argv it always did, so nothing about an existing
    full-corpus evaluate moves."""
    from steerlab_server.api import submissions

    plain = submissions._bundle_execute_command(
        "/b.tar.gz", verb="evaluate", target_root="/root", dtype="auto",
        device=None, prompts_path=None, source_path=None,
        package_evidence=True, record_path="/r.json")
    assert "--sample-per-condition" not in plain
    assert "--sample-seed" not in plain

    sampled = submissions._bundle_execute_command(
        "/b.tar.gz", verb="evaluate", target_root="/root", dtype="auto",
        device=None, prompts_path=None, source_path=None,
        package_evidence=True, record_path="/r.json",
        sample_per_condition=2400, sample_seed="0x5eed0a5e5eed0a5e")
    assert sampled[sampled.index("--sample-per-condition") + 1] == "2400"
    assert sampled[sampled.index("--sample-seed") + 1] == "0x5eed0a5e5eed0a5e"


def test_a_half_stated_sample_refuses_at_submit_time():
    """Before a submission directory or a packaged bundle exists — the same
    argument `_require_readable_run_directory` makes: a durable sbatch that
    waits in the queue to discover this on a compute node has already spent
    the allocation the mistake will be charged for."""
    from steerlab_server.api import submissions

    with pytest.raises(submissions.SubmissionRefusal) as caught:
        submissions._require_sample_pair(2400, None, verb="evaluate")
    assert caught.value.code == "sampleSeedMissing"

    with pytest.raises(submissions.SubmissionRefusal) as misplaced:
        submissions._require_sample_pair(2400, "0x2a", verb="run")
    assert misplaced.value.code == "sampleUnsupportedVerb"
    assert "'evaluate' verb only" in str(misplaced.value)

    # Well formed, and absent, both pass silently.
    submissions._require_sample_pair(2400, "0x2a", verb="evaluate")
    submissions._require_sample_pair(None, None, verb="run")


def test_the_bundle_child_refuses_the_flags_on_another_verb(tmp_path):
    """`--shard`'s rule again: a flag a verb cannot honour is REFUSED, never
    dropped — and the refusal fires before `import_bundle` writes anything."""
    from steerlab_server.experiment import bundles

    with pytest.raises(bundles.BundleError) as caught:
        bundles.execute_run_bundle(
            str(tmp_path / "missing.tar.gz"), verb="run",
            target_root=str(tmp_path), sample_per_condition=10,
            sample_seed="0x2a")
    assert "'evaluate' verb only" in str(caught.value)


# ---------------------------------------------------- twin sentences

#: Every refusal a researcher can hit, verbatim. **Twin literals**: the same
#: sentences appear in ``EvaluateSubsampleTests`` (the twin test is named
#: ``everyRefusalReadsIdenticallyOnBothEngines``), so a reword on one engine
#: fails the other engine's suite. A
#: researcher who typed the same command line on the Mac and on the cluster
#: must read the same complaint and the same runnable repair — otherwise the
#: repair they follow depends on which machine they happened to be at.
LITERAL_REFUSALS = {
    "paired": (
        "--sample-per-condition/--sample-seed apply to the per-response "
        "coding instrument only: this study's pinned rubric is a paired "
        "comparison, whose unit is a (baseline, variant) PAIR rather than "
        "a record, so a per-condition record count does not name a set of "
        "pairs to judge",
        "drop both sample flags and run steerlab-cli experiment evaluate "
        "<name> to judge every pair, or pin a perResponseCoding rubric if "
        "per-record coding is the design",
    ),
    "population": (
        "--sample-per-condition 5 exceeds what the source run holds: "
        "'baseline' has 2 codeable record(s). A subsample cannot be "
        "larger than the stratum it is drawn from, and clamping it would "
        "code a smaller design than the one that was preregistered while "
        "every stamp still said 5",
        "re-run with --sample-per-condition 2 or less (condition "
        "'baseline' is the binding stratum), or drop both sample flags "
        "and run steerlab-cli experiment evaluate <name> to code all 2 "
        "record(s)",
    ),
    "seedMalformed": (
        "--sample-seed 'zz' is not a 64-bit unsigned number — a seed is a "
        "decimal integer, or hexadecimal with or without a '0x' prefix, "
        "of at most 16 hex digits (the leading 16 of a digest are a fine "
        "seed, written down as such)",
        "steerlab-cli experiment evaluate <name> --sample-per-condition "
        "<n> --sample-seed 0x5eed0a5e5eed0a5e  (any 64-bit value; record "
        "it in the preregistration — the same seed always draws the same "
        "records)",
    ),
    "seedMissing": (
        "--sample-per-condition 2400 was given without --sample-seed: a "
        "subsample nobody can redraw is not evidence, so the draw refuses "
        "rather than choosing a seed for you",
        "steerlab-cli experiment evaluate <name> --sample-per-condition "
        "<n> --sample-seed 0x5eed0a5e5eed0a5e  (any 64-bit value; record "
        "it in the preregistration — the same seed always draws the same "
        "records)",
    ),
    "sizeMalformed": (
        "--sample-per-condition must be a whole number of records of at "
        "least 1, not 'x' — a subsample of zero records is a design "
        "nobody can report",
        "steerlab-cli experiment evaluate <name> --sample-per-condition "
        "2400 --sample-seed <seed>, or drop both flags to code the full "
        "corpus",
    ),
    "sizeMissing": (
        "--sample-seed 0x2a was given without --sample-per-condition: "
        "with no sample size the full corpus is coded, and the seed would "
        "be stamped on a coding it did not shape",
        "add --sample-per-condition <n> to draw a subsample, or drop "
        "--sample-seed and run steerlab-cli experiment evaluate <name> to "
        "code the full corpus",
    ),
}


def _refusal(fn):
    try:
        fn()
    except subsample.SubsampleRefusal as exc:
        return (exc.reason, exc.repair_action)
    raise AssertionError("expected a refusal")


def test_every_refusal_reads_identically_on_both_engines():
    two = [{"condition": "baseline", "promptID": "p", "sampleIndex": i}
           for i in range(2)]
    paired = subsample.paired_refusal("steerlab-cli")
    observed = {
        "seedMissing": _refusal(
            lambda: subsample.resolve_request(2400, None,
                                              program="steerlab-cli")),
        "sizeMissing": _refusal(
            lambda: subsample.resolve_request(None, "0x2a",
                                              program="steerlab-cli")),
        "sizeMalformed": _refusal(
            lambda: subsample.resolve_request("x", "0x2a",
                                              program="steerlab-cli")),
        "seedMalformed": _refusal(
            lambda: subsample.resolve_request(1, "zz",
                                              program="steerlab-cli")),
        "population": _refusal(
            lambda: subsample.select(two, subsample.SubsampleRequest(5, 1),
                                     program="steerlab-cli")),
        "paired": (paired.reason, paired.repair_action),
    }
    assert observed == LITERAL_REFUSALS
