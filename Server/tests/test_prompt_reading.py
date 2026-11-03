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
#: The real counter, before any test replaces it with a stand-in.
REAL_COUNTS = prompt_reading.rendered_token_counts
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


# --- a study run: the record, and the reading in force ------------------------------

TOKENS = {"long-packet": 217_230, "short-note": 900}


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _study(root, name, ids=("long-packet", "short-note"), **fields):
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm morning"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    raw = es.load_raw(name, root)
    raw.update({"maxTokens": 16, "outcomeInstruments": ["sampledText"],
                "temperature": 0.0, "seeds": [0], **fields})
    raw["variantConditions"] = [
        {"name": "steered", "artifactPath": "runs/variants/s.json",
         "artifactHash": "aa" * 32,
         "artifact": {"name": "steered", "baseModelID": "org/m",
                      "promptMode": "chatAssistant", "temperature": 0.0,
                      "alphaInNormUnits": False, "injections": [],
                      "adapters": []}}]
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", f"{name}.jsonl")
    _write(prompts_path, "".join(
        json.dumps({"id": prompt_id, "prompt": f"Read the {prompt_id}."}) + "\n"
        for prompt_id in ids))
    return prompts_path


def _cuda_model(model_id, revision=None):
    @contextmanager
    def provider(model_id, revision=None):
        yield SimpleNamespace(model_id=model_id, revision=revision or "abc",
                              device="cuda", tokenizer=object())
    return provider(model_id, revision)


def _patch(monkeypatch, seen=None, counter=None, flag=None, after=None):
    bundle = _owner_vector_materialization.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")
    monkeypatch.setattr(_owner_vector_materialization, "extract_all",
                        lambda model, manifest, root: {"fear": bundle})
    # The prompts' rendered lengths, by item: the tokenizer is a stand-in.
    monkeypatch.setattr(
        PR, "rendered_token_counts",
        lambda tokenizer, model_id, prompts, settings:
        [[TOKENS[p["id"]] for _ in settings] for p in prompts])

    def generate(model, prompt, *, token_ids_out=None, **_):
        if seen is not None:
            reading = PR.active_reading.get()
            seen.append(None if reading is None else (reading.rule(), reading.source))
        if counter is not None:
            counter[0] += 1
            if after is not None and counter[0] == after:
                flag.request()
        return f"an answer to {prompt}"
    monkeypatch.setattr(_owner_generate, "generate", generate)


def _config(run_dir):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _records(run_dir):
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_a_study_run_records_how_it_read_its_prompts(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "reading-record")
    monkeypatch.setenv(PR.CUDA_ALLOCATOR_ENV, "expandable_segments:True")
    seen = []
    _patch(monkeypatch, seen=seen)
    run_dir = tasks.run("reading-record", prompts, root, model_provider=_cuda_model,
                        log=lambda *_: None)
    assert _config(run_dir)["notes"]["promptReading"] == {
        "schemaVersion": 1,
        "mode": "chunked", "chunkTokens": 4096, "aboveTokens": 8192,
        "source": "engineDefault", "basis": None, "device": "cuda",
        "promptCount": 2, "chunkedPromptCount": 1,
        "longestPromptTokens": 217_230,
        "cudaAllocatorConfig": "expandable_segments:True",
        "allocatorConfig": None,
    }
    # Every generation ran under that reading, and the scope ended with the run.
    assert seen == [(CUDA_DEFAULT, "engineDefault")] * 4
    assert PR.active_reading.get() is None


def test_the_record_states_both_allocator_settings_as_set():
    """An empty value is recorded as empty (a site's choice of torch's
    defaults), and torch's generic name is recorded beside the CUDA one."""
    reading = PR.resolve_prompt_reading("cuda")
    entry = PR.record(reading, device="cuda", counts=[[10]], environ={
        "PYTORCH_CUDA_ALLOC_CONF": "", "PYTORCH_ALLOC_CONF": "max_split_size_mb:128"})
    assert (entry["cudaAllocatorConfig"], entry["allocatorConfig"]) == (
        "", "max_split_size_mb:128")
    unset = PR.record(reading, device="cuda", counts=[[10]], environ={})
    assert (unset["cudaAllocatorConfig"], unset["allocatorConfig"]) == (None, None)
    off_cuda = PR.record(PR.resolve_prompt_reading("mps"), device="mps", counts=[[10]],
                         environ={"PYTORCH_ALLOC_CONF": "x"})
    assert (off_cuda["cudaAllocatorConfig"], off_cuda["allocatorConfig"]) == (None, None)


def test_a_submission_plan_reaches_the_record_with_its_basis(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "reading-plan")
    _patch(monkeypatch)
    basis = {"gpuType": "A100", "cardMemoryGB": 40, "longestPromptTokens": 217230}
    plan = PR.parse_plan({"schemaVersion": 1, "source": "submission",
                          "mode": "chunked", "chunkTokens": 2048,
                          "aboveTokens": 8192, "basis": basis})
    with PR.plan_scope(plan):
        run_dir = tasks.run("reading-plan", prompts, root,
                            model_provider=_cuda_model, log=lambda *_: None)
    entry = _config(run_dir)["notes"]["promptReading"]
    assert (entry["mode"], entry["chunkTokens"], entry["aboveTokens"]) == (
        "chunked", 2048, 8192)
    assert entry["source"] == "submission" and entry["basis"] == basis
    assert entry["cudaAllocatorConfig"] is None  # not set in this process


def test_the_studys_declaration_outranks_the_plan(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "reading-declared",
                     promptReading={"mode": "singlePass"})
    _patch(monkeypatch)
    plan = PR.parse_plan({"schemaVersion": 1, "source": "submission",
                          **CUDA_DEFAULT})
    lines = []
    with PR.plan_scope(plan):
        run_dir = tasks.run("reading-declared", prompts, root,
                            model_provider=_cuda_model, log=lines.append)
    entry = _config(run_dir)["notes"]["promptReading"]
    assert (entry["mode"], entry["source"], entry["chunkedPromptCount"]) == (
        "singlePass", "studyDeclaration", 0)
    assert [line for line in lines if "the study's declaration wins" in line]


def test_a_sweep_reads_by_the_studys_declaration(tmp_path, monkeypatch):
    from steerlab_server.experiment import sweep_workflow
    root = str(tmp_path)
    _study(root, "reading-sweep", promptReading={"mode": "singlePass"})
    _patch(monkeypatch)
    seen = []

    def sweep_impl(name, manifest, model, root, *args):
        reading = PR.active_reading.get()
        seen.append((reading.rule(), reading.source))
        return "swept"
    monkeypatch.setattr(sweep_workflow, "_sweep_impl", sweep_impl)
    assert sweep_workflow.sweep("reading-sweep", root, model_provider=_cuda_model,
                                log=lambda *_: None) == "swept"
    assert seen == [({"mode": "singlePass"}, "studyDeclaration")]
    assert PR.active_reading.get() is None


def test_a_sweep_records_its_reading_in_its_config(tmp_path, monkeypatch):
    from steerlab_server.experiment import sweep_workflow
    root = str(tmp_path)
    _study(root, "reading-sweep-record", promptReading={"mode": "singlePass"})
    _patch(monkeypatch)
    # No concepts to sweep: the run directory and its stamp are still written.
    monkeypatch.setattr(_owner_vector_materialization, "extract_all",
                        lambda model, manifest, root: {})
    run_dir = sweep_workflow.sweep("reading-sweep-record", root,
                                   model_provider=_cuda_model, log=lambda *_: None)
    entry = _config(run_dir)["notes"]["promptReading"]
    assert (entry["mode"], entry["source"], entry["device"]) == (
        "singlePass", "studyDeclaration", "cuda")
    assert (entry["promptCount"], entry["longestPromptTokens"]) == (None, None)


def test_an_unreadable_declaration_refuses_before_any_generation(tmp_path,
                                                                  monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "reading-bad",
                     promptReading={"mode": "chunked", "chunkTokens": 4096})
    seen = []
    _patch(monkeypatch, seen=seen)
    with pytest.raises(RuntimeError) as refusal:
        tasks.run("reading-bad", prompts, root, model_provider=_cuda_model,
                  log=lambda *_: None)
    assert "promptReading declaration cannot be read" in str(refusal.value)
    assert getattr(refusal.value, "gate", None) == "studyDeclaration"
    assert seen == []


def test_arm_render_settings_name_each_distinct_rendering():
    manifest = Manifest.from_dict({
        "name": "arms", "modelID": "org/m", "concepts": [],
        "systemPrompt": "Frame.", "promptMode": "chatAssistant",
        "variantConditions": [
            {"name": "plain", "artifactPath": "a.json", "artifactHash": "aa" * 32,
             "artifact": {"name": "plain", "baseModelID": "org/m",
                          "promptMode": "chatAssistant", "injections": [],
                          "adapters": []}},
            {"name": "persona", "artifactPath": "b.json", "artifactHash": "bb" * 32,
             "artifact": {"name": "persona", "baseModelID": "org/m",
                          "promptMode": "chatAssistant", "injections": [],
                          "adapters": [], "systemPrompt": "Persona."}}]})
    settings = condition_execution.run_arm_render_settings(manifest, None)
    assert [s["system_prompt"] for s in settings] == ["Frame.", "Persona.\n\nFrame."]
    assert all(s["reasoning_effort"] is None for s in settings)


# --- resume: never two readings in one run ------------------------------------------

def _parked(root, name, monkeypatch, *, ids=("long-packet", "short-note"),
            after=1, env=None):
    prompts = _study(root, name, ids=ids)
    flag = resume_mod.CheckpointFlag()
    _patch(monkeypatch, counter=[0], flag=flag, after=after)
    seen = {}
    with monkeypatch.context() as old:
        for key, value in (env or {}).items():
            old.setenv(key, value)
        with pytest.raises(resume_mod.CheckpointRequested):
            tasks.run(name, prompts, root, model_provider=_cuda_model,
                      log=lambda *_: None, checkpoint=flag,
                      on_run_directory=lambda d: seen.setdefault("dir", d))
    return prompts, seen["dir"]


def _as_historic(run_dir, device="cuda"):
    """Rewrite a parked run as 0.9.19 left it: no promptReading record."""
    path = os.path.join(run_dir, "config.json")
    config = _config(run_dir)
    config["notes"].pop("promptReading")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    substrate = os.path.join(run_dir, "substrate.json")
    with open(substrate, encoding="utf-8") as handle:
        info = json.load(handle)
    info["device"] = device
    with open(substrate, "w", encoding="utf-8") as handle:
        json.dump(info, handle)


def test_a_resume_under_a_different_reading_refuses(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts, run_dir = _parked(root, "reading-resume", monkeypatch)
    before = open(os.path.join(run_dir, "generations.jsonl"), "rb").read()
    monkeypatch.setenv(PR.PREFILL_CHUNK_ENV, "2048")
    _patch(monkeypatch)
    with pytest.raises(resume_mod.ResumeRefusal) as refusal:
        tasks.run("reading-resume", prompts, root, model_provider=_cuda_model,
                  log=lambda *_: None, run_directory=run_dir)
    message = str(refusal.value)
    assert ("read its prompts as prompts over 8,192 tokens in 4,096-token "
            "chunks (shorter prompts in a single pass), but this engine reads "
            "them as prompts over 8,192 tokens in 2,048-token chunks (shorter "
            "prompts in a single pass), as the STEERLAB_PREFILL_CHUNK setting "
            "asks:") in message
    assert ("1 prompt length the study renders (the longest 217,230 tokens) "
            "would be read differently") in message
    assert "refusing to mix two prompt readings in one run" in message
    assert refusal.value.gate == "manifestEpoch"
    assert refusal.value.repair_action.startswith(
        "steerlab-server study submit reading-resume --verb run --resume "
        f"runs/{os.path.basename(run_dir)}  (resume through a submission, which "
        "reads the prompts as this run recorded;")
    assert ("start afresh with steerlab-server experiment run reading-resume, "
            "without --resume") in refusal.value.repair_action
    assert isinstance(refusal.value, resume_mod.ResumeError)
    assert open(os.path.join(run_dir, "generations.jsonl"), "rb").read() == before


def test_a_resume_under_the_same_rule_from_another_source_continues(
        tmp_path, monkeypatch):
    """Only the rule matters, not its source: a submission plan of 4096
    above 8192 reads exactly as the CUDA default the run started under."""
    root = str(tmp_path)
    prompts, run_dir = _parked(root, "reading-same", monkeypatch)
    _patch(monkeypatch)
    plan = PR.parse_plan({"schemaVersion": 1, "source": "submission",
                          **CUDA_DEFAULT})
    with PR.plan_scope(plan):
        resumed = tasks.run("reading-same", prompts, root,
                            model_provider=_cuda_model, log=lambda *_: None,
                            run_directory=run_dir)
    assert resumed == run_dir
    assert len([r for r in _records(run_dir) if "output" in r]) == 4
    # The record is the run's birth stamp, never rewritten.
    assert _config(run_dir)["notes"]["promptReading"]["source"] == "engineDefault"


def test_a_run_from_0919_with_short_prompts_resumes_on_the_cuda_default(
        tmp_path, monkeypatch):
    """0.9.19 read every CUDA prompt in one pass; prompts at or below 8,192
    tokens read the same way under the new default, so the parked run
    continues."""
    root = str(tmp_path)
    prompts, run_dir = _parked(root, "reading-old-short", monkeypatch,
                               ids=("short-note",))
    _as_historic(run_dir)
    _patch(monkeypatch)
    resumed = tasks.run("reading-old-short", prompts, root,
                        model_provider=_cuda_model, log=lambda *_: None,
                        run_directory=run_dir)
    assert resumed == run_dir


def test_a_run_from_0919_with_a_long_prompt_refuses_on_the_cuda_default(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts, run_dir = _parked(root, "reading-old-long", monkeypatch)
    _as_historic(run_dir)
    _patch(monkeypatch)
    with pytest.raises(resume_mod.ResumeRefusal) as refusal:
        tasks.run("reading-old-long", prompts, root, model_provider=_cuda_model,
                  log=lambda *_: None, run_directory=run_dir)
    message = str(refusal.value)
    assert "read its prompts as every prompt in a single pass" in message
    assert ("it was made before runs recorded their prompt reading, so it is "
            "taken as read by that release's rule on CUDA") in message
    assert "(the longest 217,230 tokens)" in message


# --- scripted transcripts: the length generation reads is the conversation's -------

#: A scripted conversation whose earlier turns are long and whose final user
#: turn, the item's display text, is short.
LONG_TRANSCRIPT = [
    {"role": "user", "content": "Read this packet."},
    {"role": "assistant", "content": "a " * 4600},
    {"role": "user", "content": "Did you read it? Answer yes or no."},
]


def _transcript_model(model_id, revision=None):
    from test_transcript_study import FakeTokenizer  # one id per character

    @contextmanager
    def provider(model_id, revision=None):
        yield SimpleNamespace(model_id=model_id, revision=revision or "abc",
                              device="cuda", tokenizer=FakeTokenizer())
    return provider(model_id, revision)


def _transcript_study(root, name):
    prompts = _study(root, name, ids=("short-note",))
    with open(prompts, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "conversation", "transcript": LONG_TRANSCRIPT}) + "\n")
    return prompts


def test_a_transcript_items_length_is_the_whole_conversation_generation_reads(
        monkeypatch):
    """task_inputs fills a transcript item's ``prompt`` with its final user
    turn; generation renders the whole conversation. The count must be what
    generation reads."""
    from test_transcript_study import FakeTokenizer
    from steerlab_server.experiment import prompt_render, task_inputs
    tokenizer = FakeTokenizer()
    item = {"id": "conversation", "transcript": LONG_TRANSCRIPT}
    item["prompt"] = prompt_render.transcript_display_text(
        prompt_render.normalize_transcript(LONG_TRANSCRIPT))
    setting = {"prompt_mode": "chatAssistant", "system_prompt": None,
               "qwen_thinking_enabled": False, "reasoning_effort": None}
    [[counted]] = REAL_COUNTS(tokenizer, "org/m", [item], [setting])
    # What generation itself renders for the item, through stream_generate.
    read = []

    def stream_rendered(model, rendered, **_):
        read.append(rendered.prompt_token_count)
        return iter(())
    monkeypatch.setattr(_owner_generate, "_stream_rendered", stream_rendered)
    list(_owner_generate.stream_generate(
        SimpleNamespace(model_id="org/m", tokenizer=tokenizer), item["prompt"],
        transcript=item["transcript"], **setting))
    assert read == [counted]
    # The whole conversation, far longer than the display text alone.
    display = prompt_render.render(tokenizer, item["prompt"], model_id="org/m")
    assert counted > 9_000 > 8_192 > display.prompt_token_count


def test_a_0919_transcript_study_with_long_conversations_refuses_to_resume(
        tmp_path, monkeypatch):
    """The display text is a few dozen tokens; the conversation is over 9,000.
    Counting the display text admitted a single-pass run to be finished in
    chunks."""
    root = str(tmp_path)
    prompts = _transcript_study(root, "reading-transcript")
    flag = resume_mod.CheckpointFlag()
    _patch(monkeypatch, counter=[0], flag=flag, after=1)
    monkeypatch.setattr(PR, "rendered_token_counts", REAL_COUNTS)
    seen = {}
    with pytest.raises(resume_mod.CheckpointRequested):
        tasks.run("reading-transcript", prompts, root, model_provider=_transcript_model,
                  log=lambda *_: None, checkpoint=flag,
                  on_run_directory=lambda d: seen.setdefault("dir", d))
    run_dir = seen["dir"]
    entry = _config(run_dir)["notes"]["promptReading"]
    assert entry["longestPromptTokens"] > 9_000
    assert (entry["promptCount"], entry["chunkedPromptCount"]) == (1, 1)
    _as_historic(run_dir)
    _patch(monkeypatch)
    monkeypatch.setattr(PR, "rendered_token_counts", REAL_COUNTS)
    with pytest.raises(resume_mod.ResumeRefusal) as refusal:
        tasks.run("reading-transcript", prompts, root, model_provider=_transcript_model,
                  log=lambda *_: None, run_directory=run_dir)
    assert "read its prompts as every prompt in a single pass" in str(refusal.value)


def test_the_record_settles_a_definite_difference_before_the_model_loads(tmp_path):
    """``run`` calls the resume gate with the reading it will use before
    acquiring the model; the parked record's longest prompt settles it."""
    run_dir = tmp_path / "parked"
    run_dir.mkdir()
    entry = {"schemaVersion": 1, **CUDA_DEFAULT, "source": "engineDefault",
             "basis": None, "device": "cuda", "promptCount": 2,
             "chunkedPromptCount": 1, "longestPromptTokens": 217_230,
             "cudaAllocatorConfig": None}
    (run_dir / "config.json").write_text(json.dumps({"notes": {"promptReading": entry}}))
    with pytest.raises(resume_mod.ResumeRefusal):
        run_preflight.prompt_reading_admission(
            str(run_dir), name="s", reading=PR.PromptReading.single_pass(),
            definite_only=True)
    # Same chunk size, a different threshold, both chunking the longest: not
    # settled without the counts, so the early call lets the later one decide.
    run_preflight.prompt_reading_admission(
        str(run_dir), name="s", reading=PR.PromptReading.chunked(4096, 6000),
        definite_only=True)
    with pytest.raises(resume_mod.ResumeRefusal) as refusal:
        run_preflight.prompt_reading_admission(
            str(run_dir), name="s", reading=PR.PromptReading.chunked(4096, 6000))
    assert "neither its record nor this engine can show" in str(refusal.value)


# --- the shard merge: never two readings in one merged run ----------------------------

def _shards(root, name, monkeypatch, *, ids, env_for=None):
    prompts = _study(root, name, ids=ids)
    dirs = []
    for index in range(2):
        _patch(monkeypatch)
        with monkeypatch.context() as engine:
            for key, value in ((env_for or {}).get(index) or {}).items():
                engine.setenv(key, value)
            dirs.append(tasks.run(name, prompts, root, model_provider=_cuda_model,
                                  log=lambda *_: None,
                                  shard=sharding.ShardSpec(index, 2)))
    return dirs


def test_a_merge_of_shards_that_read_a_long_prompt_differently_refuses(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    dirs = _shards(root, "reading-merge", monkeypatch,
                   ids=("long-packet", "short-note"),
                   env_for={1: {PR.PREFILL_CHUNK_ENV: "0"}})
    with pytest.raises(sharding.ShardMergeRefusal) as refusal:
        sharding.merge_shard_runs("reading-merge", dirs, root=root)
    message = str(refusal.value)
    assert ("shard 0 reads them as prompts over 8,192 tokens in 4,096-token "
            "chunks (shorter prompts in a single pass), but shard 1 reads them "
            "as every prompt in a single pass, so its longest prompt would be "
            "read differently") in message
    assert "the shard partials are untouched" in message
    assert refusal.value.gate == "manifestEpoch"
    assert refusal.value.repair_action.startswith(
        "steerlab-server study submit reading-merge --verb run --executor "
        "slurm --parallel 2")


def test_a_merge_of_shards_whose_rules_differ_but_read_alike_succeeds(
        tmp_path, monkeypatch):
    """Every prompt at or below 8,192 tokens: a single-pass shard and a
    chunked one read every prompt identically, so they merge, and the merged
    run carries the reading."""
    root = str(tmp_path)
    dirs = _shards(root, "reading-merge-short", monkeypatch, ids=("short-note",),
                   env_for={1: {PR.PREFILL_CHUNK_ENV: "0"}})
    merged = sharding.merge_shard_runs("reading-merge-short", dirs, root=root)
    entry = _config(merged)["notes"]["promptReading"]
    assert (entry["mode"], entry["longestPromptTokens"]) == ("chunked", 900)


def test_a_merge_with_a_shard_from_0919_compares_it_by_its_device(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    dirs = _shards(root, "reading-merge-old", monkeypatch,
                   ids=("long-packet", "short-note"))
    _as_historic(dirs[1])
    with pytest.raises(sharding.ShardMergeRefusal) as refusal:
        sharding.merge_shard_runs("reading-merge-old", dirs, root=root)
    assert ("shard 1 reads them as every prompt in a single pass (it was made "
            "before runs recorded their prompt reading") in str(refusal.value)


# --- a panel: the rule is recorded, and compared on resume ----------------------------

def _real_panel_workspace(tmp_path, monkeypatch):
    import steerlab_server.experiment.run_artifacts as run_artifacts
    from test_panel_shard_merge import _panel_workspace
    real = run_artifacts.write_config_snapshot
    root, manifest = _panel_workspace(tmp_path, monkeypatch, replicates=2)
    monkeypatch.setattr(run_artifacts, "write_config_snapshot", real)
    return root, manifest


def test_a_panel_records_its_rule_and_leaves_the_counts_unestablished(
        tmp_path, monkeypatch):
    """A panel's turn prompts grow while it deliberates, so their lengths are
    not known when the run starts: the record says null, not zero."""
    from steerlab_server.experiment import panel_workflow
    root, manifest = _real_panel_workspace(tmp_path, monkeypatch)
    model = SimpleNamespace(model_id="m", revision="r", device="cuda")
    run_dir = panel_workflow.run_multi_agent_study("panel", manifest, model,
                                                   str(root), log=lambda *_: None)
    entry = _config(run_dir)["notes"]["promptReading"]
    assert {key: entry[key] for key in ("mode", "chunkTokens", "aboveTokens",
                                        "source", "device")} == {
        **CUDA_DEFAULT, "source": "engineDefault", "device": "cuda"}
    assert (entry["promptCount"], entry["chunkedPromptCount"],
            entry["longestPromptTokens"]) == (None, None, None)


def test_a_parked_panel_refuses_to_resume_under_another_rule(tmp_path, monkeypatch):
    from steerlab_server.experiment import panel_workflow
    root, manifest = _real_panel_workspace(tmp_path, monkeypatch)
    model = SimpleNamespace(model_id="m", revision="r", device="cuda")
    flag = resume_mod.CheckpointFlag()
    flag.request()
    with pytest.raises(resume_mod.CheckpointRequested) as caught:
        panel_workflow.run_multi_agent_study("panel", manifest, model, str(root),
                                             checkpoint=flag, log=lambda *_: None)
    parked = caught.value.run_directory
    monkeypatch.setenv(PR.PREFILL_CHUNK_ENV, "0")
    with pytest.raises(resume_mod.ResumeRefusal) as refusal:
        panel_workflow.run_multi_agent_study("panel", manifest, model, str(root),
                                             run_directory=parked,
                                             log=lambda *_: None)
    assert ("neither its record nor this engine can show that every prompt "
            "would be read the same way") in str(refusal.value)
    # Under the rule it started with, it resumes.
    monkeypatch.delenv(PR.PREFILL_CHUNK_ENV)
    assert panel_workflow.run_multi_agent_study(
        "panel", manifest, model, str(root), run_directory=parked,
        log=lambda *_: None) == parked
