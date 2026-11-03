"""How a prompt is read (single pass or chunked): the rule, where it comes
from, what a run records, and the guards that keep one run, or one fan-out,
from mixing two readings.

Chunked and single-pass readings do not give token-identical texts at
near-ties, so the engine must never mix them silently. Expected values here
are written by hand from the shared contract, not computed by the code under
test.
"""
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import steerlab_server.experiment.generate as _owner_generate
import steerlab_server.experiment.vector_materialization as _owner_vector_materialization
from steerlab_server import cli
from steerlab_server.api import submissions
from steerlab_server.experiment import bundles, prompt_reading, run_preflight
from steerlab_server.experiment import condition_execution
from steerlab_server.experiment import resume as resume_mod
from steerlab_server.experiment import sharding, tasks
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering.vector_store import ConceptVectors

PR = prompt_reading
CUDA_DEFAULT = {"mode": "chunked", "chunkTokens": 4096, "aboveTokens": 8192}


@pytest.fixture(autouse=True)
def _no_ambient_overrides(monkeypatch):
    monkeypatch.delenv(PR.PREFILL_CHUNK_ENV, raising=False)
    monkeypatch.delenv(PR.CUDA_ALLOCATOR_ENV, raising=False)


# --- the rule and its resolution ------------------------------------------------

def test_the_device_defaults():
    assert PR.resolve_prompt_reading("mps").rule() == {
        "mode": "chunked", "chunkTokens": 1024, "aboveTokens": 4096}
    assert PR.resolve_prompt_reading("cuda:1").rule() == CUDA_DEFAULT
    assert PR.resolve_prompt_reading("cpu").rule() == {"mode": "singlePass"}
    for device in ("mps", "cuda", "cpu"):
        assert PR.resolve_prompt_reading(device).source == "engineDefault"


def test_the_chunking_rule_is_todays_rule():
    """Chunked when n > max(aboveTokens, chunkTokens); a single pass at or
    below it."""
    cuda = PR.PromptReading.from_rule(CUDA_DEFAULT, source="engineDefault")
    assert cuda.chunk_size(8_192) is None
    assert cuda.chunk_size(8_193) == 4096
    assert cuda.chunk_size(217_230) == 4096
    small = PR.PromptReading.from_rule(
        {"mode": "chunked", "chunkTokens": 512, "aboveTokens": 0},
        source="submission")
    assert small.chunk_size(512) is None and small.chunk_size(513) == 512
    assert PR.PromptReading.single_pass().chunk_size(10 ** 6) is None


def test_resolution_order_declaration_plan_environment_default():
    plan = PR.parse_plan({"schemaVersion": 1, "source": "submission",
                          "mode": "chunked", "chunkTokens": 2048,
                          "aboveTokens": 8192, "basis": {"gpuType": "A100"}})
    env = {PR.PREFILL_CHUNK_ENV: "512"}
    declared = {"mode": "singlePass"}
    lines = []
    reading = PR.resolve_prompt_reading("cuda", declaration=declared, plan=plan,
                                        environ=env, log=lines.append)
    assert (reading.rule(), reading.source) == ({"mode": "singlePass"},
                                                "studyDeclaration")
    # The declaration won over a plan that disagrees, and said so once.
    assert len(lines) == 1 and "the study's declaration wins" in lines[0]
    reading = PR.resolve_prompt_reading("cuda", plan=plan, environ=env)
    assert (reading.rule(), reading.source) == (
        {"mode": "chunked", "chunkTokens": 2048, "aboveTokens": 8192},
        "submission")
    assert reading.basis == {"gpuType": "A100"}
    reading = PR.resolve_prompt_reading("cuda", environ=env)
    assert (reading.rule(), reading.source) == (
        {"mode": "chunked", "chunkTokens": 512, "aboveTokens": 8192},
        "environment")
    assert PR.resolve_prompt_reading("cuda", environ={}).source == "engineDefault"


def test_a_plan_that_agrees_with_the_declaration_logs_nothing():
    plan = PR.parse_plan({"schemaVersion": 1, "source": "studyDeclaration",
                          **CUDA_DEFAULT})
    lines = []
    reading = PR.resolve_prompt_reading("cuda", declaration=dict(CUDA_DEFAULT),
                                        plan=plan, log=lines.append)
    assert reading.source == "studyDeclaration" and lines == []


def test_the_environment_override_reads_as_it_did():
    assert PR.resolve_prompt_reading(
        "mps", environ={PR.PREFILL_CHUNK_ENV: "0"}).rule() == {"mode": "singlePass"}
    assert PR.resolve_prompt_reading(
        "mps", environ={PR.PREFILL_CHUNK_ENV: "-3"}).source == "environment"
    # An unreadable value is ignored, as 0.9.19 ignored it.
    unreadable = PR.resolve_prompt_reading(
        "mps", environ={PR.PREFILL_CHUNK_ENV: "not-a-number"})
    assert unreadable.rule()["chunkTokens"] == 1024
    assert unreadable.source == "engineDefault"
    # CPU never consulted it.
    assert PR.resolve_prompt_reading(
        "cpu", environ={PR.PREFILL_CHUNK_ENV: "512"}).rule() == {"mode": "singlePass"}


def test_effective_rules_compare_by_behaviour_not_by_source_or_spelling():
    declared = PR.PromptReading.from_rule(CUDA_DEFAULT, source="studyDeclaration")
    assert declared.reads_like(PR.resolve_prompt_reading("cuda"))
    # aboveTokens below chunkTokens reads as chunkTokens.
    a = PR.PromptReading.from_rule(
        {"mode": "chunked", "chunkTokens": 4096, "aboveTokens": 0}, source="submission")
    b = PR.PromptReading.from_rule(
        {"mode": "chunked", "chunkTokens": 4096, "aboveTokens": 4096},
        source="submission")
    assert a.reads_like(b)
    assert not a.reads_like(PR.PromptReading.single_pass())


def test_prefill_chunk_size_follows_the_runs_reading_then_the_plan():
    long = 30_000
    plan = PR.parse_plan({"schemaVersion": 1, "source": "submission",
                          "mode": "singlePass"})
    with PR.plan_scope(plan):
        assert _owner_generate.prefill_chunk_size("cuda", long) is None
        run_reading = PR.PromptReading.chunked(256, 0, source="studyDeclaration")
        with PR.reading_scope(run_reading):
            assert _owner_generate.prefill_chunk_size("cuda", long) == 256
            # On CPU too: a declared or planned rule is not device-bound.
            assert _owner_generate.prefill_chunk_size("cpu", long) == 256
    assert _owner_generate.prefill_chunk_size("cuda", long) == 4096


# --- strict validation ----------------------------------------------------------

@pytest.mark.parametrize("rule,needle", [
    ("singlePass", "must be a JSON object"),
    ({"mode": "fast"}, "the mode is"),
    ({"mode": "singlePass", "chunkTokens": 4096}, "unknown key(s) chunkTokens"),
    ({"mode": "chunked", "chunkTokens": 4096}, "aboveTokens None"),
    ({"mode": "chunked", "chunkTokens": 0, "aboveTokens": 1}, "at least 1"),
    ({"mode": "chunked", "chunkTokens": True, "aboveTokens": 1}, "chunkTokens True"),
    ({"mode": "chunked", "chunkTokens": 4096.0, "aboveTokens": 1}, "whole number"),
    ({"mode": "chunked", "chunkTokens": 64, "aboveTokens": -1}, "at least 0"),
    ({**CUDA_DEFAULT, "chunk": 1}, "unknown key(s) chunk"),
])
def test_a_rule_the_engine_would_have_to_guess_at_is_refused(rule, needle):
    with pytest.raises(PR.PromptReadingError) as refusal:
        PR.validate_rule(rule)
    assert needle in str(refusal.value)
    assert refusal.value.repair_action.startswith("write the rule as")


@pytest.mark.parametrize("plan,needle", [
    ("{not json", "not valid JSON"),
    ("[]", "must be a JSON object"),
    ({"source": "submission", "mode": "singlePass"}, "schemaVersion None"),
    ({"schemaVersion": 2, "source": "submission", "mode": "singlePass"},
     "this engine reads schemaVersion 1"),
    ({"schemaVersion": "1", "source": "submission", "mode": "singlePass"},
     "schemaVersion '1'"),
    ({"schemaVersion": True, "source": "submission", "mode": "singlePass"},
     "schemaVersion True"),
    ({"schemaVersion": 1, "source": "engineDefault", "mode": "singlePass"},
     "source 'engineDefault'"),
    ({"schemaVersion": 1, "source": "submission", "mode": "singlePass",
      "gpu": "A100"}, "unknown key(s) gpu"),
    ({"schemaVersion": 1, "source": "submission", "mode": "singlePass",
      "basis": "A100"}, "basis must be a JSON object"),
    ({"schemaVersion": 1, "source": "submission", "mode": "chunked",
      "chunkTokens": -4, "aboveTokens": 0}, "at least 1"),
])
def test_a_malformed_plan_is_refused_in_plain_words(plan, needle):
    with pytest.raises(PR.PromptReadingError) as refusal:
        PR.parse_plan(plan if isinstance(plan, str) else json.dumps(plan))
    assert needle in str(refusal.value)


def test_a_well_formed_plan_keeps_its_basis_verbatim():
    basis = {"gpuType": "A100", "cardMemoryGB": 80, "longestPromptTokens": 217230}
    plan = PR.parse_plan(json.dumps({"schemaVersion": 1, "source": "submission",
                                     **CUDA_DEFAULT, "basis": basis}))
    assert plan == {"schemaVersion": 1, "source": "submission", "basis": basis,
                    **CUDA_DEFAULT}
    assert PR.parse_plan(plan) == plan  # idempotent


def test_bundle_execute_refuses_a_malformed_plan_with_exit_64(capsys):
    code = cli.main(["bundle", "execute", "missing.tar.gz", "--verb", "run",
                     "--prompt-reading-plan", '{"schemaVersion": 1}'])
    assert code == 64
    err = capsys.readouterr().err
    assert "the prompt-reading plan has source None" in err
    code = cli.main(["bundle", "execute", "missing.tar.gz", "--verb", "run",
                     "--prompt-reading-plan"])
    assert code == 64
    assert "needs a JSON plan" in capsys.readouterr().err


def test_bundle_execute_refuses_a_plan_for_a_verb_that_generates_no_study_text(
        tmp_path):
    plan = {"schemaVersion": 1, "source": "submission", "mode": "singlePass"}
    with pytest.raises(bundles.BundleError) as refusal:
        bundles.execute_run_bundle(str(tmp_path / "missing.tar.gz"),
                                   verb="analyze", prompt_reading_plan=plan)
    assert "applies to the 'run', 'sweep', and 'pipeline' verbs only" \
        in str(refusal.value)


def test_the_submission_encodes_the_plan_only_when_it_has_one():
    base = dict(verb="run", target_root="/w", dtype="auto", device=None,
                prompts_path=None, source_path=None, package_evidence=True,
                record_path="/r.json")
    plain = submissions._bundle_execute_command("/b.tar.gz", **base)
    assert "--prompt-reading-plan" not in plain
    plan = {"schemaVersion": 1, "source": "submission", "mode": "chunked",
            "chunkTokens": 2048, "aboveTokens": 8192,
            "basis": {"gpuType": "A100", "cardMemoryGB": 40}}
    planned = submissions._bundle_execute_command(
        "/b.tar.gz", prompt_reading_plan=plan, **base)
    assert planned[:len(plain)] == plain
    assert planned[len(plain):] == [
        "--prompt-reading-plan",
        '{"aboveTokens":8192,"basis":{"cardMemoryGB":40,"gpuType":"A100"},'
        '"chunkTokens":2048,"mode":"chunked","schemaVersion":1,'
        '"source":"submission"}']


# --- deciding agreement -----------------------------------------------------------

def test_two_readings_agree_when_every_prompt_reads_the_same_way():
    single = PR.PromptReading.single_pass()
    cuda = PR.PromptReading.from_rule(CUDA_DEFAULT, source="engineDefault")
    mps = PR.PromptReading.chunked(1024, 4096)
    same_size_lower = PR.PromptReading.chunked(4096, 6000)
    # Every prompt at or below 8,192: single pass and the CUDA default agree.
    assert PR.differing_counts(single, cuda, [[900], [8192]]) == []
    assert PR.agree_on_longest(single, cuda, 8192) is True
    # One over it: the longest prompt itself reads differently.
    assert PR.differing_counts(single, cuda, [[900], [217_230]]) == [217_230]
    assert PR.agree_on_longest(single, cuda, 217_230) is False
    # Different chunk sizes over the longest prompt.
    assert PR.agree_on_longest(mps, cuda, 9000) is False
    assert PR.agree_on_longest(mps, cuda, 4096) is True
    # Same chunk size, different thresholds, both chunking the longest: a
    # shorter prompt may lie between them, so the longest cannot settle it...
    assert PR.agree_on_longest(same_size_lower, cuda, 20_000) is None
    # ...but the counts do.
    assert PR.differing_counts(same_size_lower, cuda, [[20_000], [5000]]) == []
    assert PR.differing_counts(same_size_lower, cuda, [[20_000], [7000]]) == [7000]
    assert PR.agree_on_longest(single, cuda, None) is None
