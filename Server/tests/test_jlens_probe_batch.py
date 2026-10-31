"""`jlens-probe-batch`: many prompts through the J-lens probe as one managed GPU job.

What these pin: the model loads once and the lens is verified and placed once
for N prompts; every per-prompt output is exactly what one `probe()` call
writes (same schema, same numbers, same CSV bytes); the plan prices prompts,
memory, output, and walltime before submission and refuses what cannot run;
and the operation goes through interview, draft, publish, plan, staging,
execution, and export like every managed operation.

All CPU, on the tiny in-memory Llama the OptVec and J-space suites use, with a
real lens record written through the production converter and store.
"""
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.experiment import diagnostic_archives as archives
from steerlab_server.experiment import jlens_probe_batch as batch
from steerlab_server.experiment import managed_methods, method_authoring
from steerlab_server.jlens import probe
from steerlab_server.steering import vector_store
from steerlab_server.steering.hooks import HookedModel
from test_optvec_eval import HIDDEN, LAYERS
from test_optvec_jspace import LENS_ID, _RandomLensSource, _write_lens

MODEL = "test/tiny"
REVISION = "a" * 40

PROMPTS = [
    {"id": "case-1", "text": "the court weighed the case and the law",
     "labels": {"condition": "baseline"}},
    {"id": "case-2", "text": "mercy for the accused was the question"},
    {"id": "case-3", "text": "the ruling held that the law was unfair",
     "labels": {"condition": "steered", "item": 3}},
]


class _Tokenizer:
    """A whitespace tokenizer whose ids never depend on what it has seen: a
    word's id is a fixed hash, a BOS is 1, and `decode` answers from a fixed
    word list, so two calls in any order record the same pieces."""

    WORDS = sorted({w for row in PROMPTS for w in row["text"].split()} | {"word", "case", "Be", "brief."})

    @staticmethod
    def _id(word):
        return 2 + int(hashlib.sha256(word.encode()).hexdigest(), 16) % 250

    def __call__(self, text, add_special_tokens=True):
        ids = ([1] if add_special_tokens else []) + [self._id(w) for w in text.split()]
        return SimpleNamespace(input_ids=ids)

    def encode(self, text, add_special_tokens=False):
        return [self._id(w) for w in text.split()]

    def decode(self, ids):
        reverse = {}
        for word in self.WORDS:
            reverse.setdefault(self._id(word), word)
        return " ".join(reverse.get(int(i), "<bos>" if int(i) == 1 else f"<{int(i)}>")
                        for i in ids)


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(11)
    config = LlamaConfig(hidden_size=HIDDEN, num_hidden_layers=LAYERS, num_attention_heads=4,
                         num_key_value_heads=2, intermediate_size=64, vocab_size=256,
                         max_position_embeddings=256)
    lm = LlamaForCausalLM(config).eval()
    return SimpleNamespace(model=lm, tokenizer=_Tokenizer(), hooked=HookedModel(lm),
                           model_id=MODEL, revision="rev-tiny", dtype="float32",
                           device=torch.device("cpu"), context_window=256,
                           hidden_size=HIDDEN, num_layers=LAYERS)


def write_prompts(root, rows=PROMPTS, name="prompts/probe-set.jsonl"):
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_direction(root, name="mercy", seed=5):
    generator = torch.Generator().manual_seed(seed)
    per_layer = [[float(x) for x in torch.randn(HIDDEN, generator=generator)]
                 for _ in range(LAYERS)]
    vectors = vector_store.ConceptVectors(per_layer=per_layer)
    sidecar = vector_store.SteeringVectorSidecar(
        modelID=MODEL, concept=name, stimulusSetHash=f"test:{name}",
        layerCount=LAYERS, hiddenSize=HIDDEN,
        normsPerLayer=[vectors.norm(i) for i in range(LAYERS)],
        extractionDate="2026-10-01T00:00:00Z", revision=REVISION)
    directory = Path(root) / "runs" / "20261001T000000000-extract"
    vector_store.save(vectors, sidecar, str(directory), name)
    return f"runs/{directory.name}/{name}"


def write_agent(root, vector):
    path = Path(root) / "runs" / "model-variants" / "mercy-agent.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schemaVersion": 1, "name": "mercy-agent", "baseModelID": MODEL,
        "alphaInNormUnits": False, "promptMode": "rawCompletion",
        "injections": [{"concept": "mercy", "vectorArtifactID": vector,
                        "layer": 1, "alpha": 0.5}]}))
    return path.relative_to(root).as_posix()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.setenv("STEERLAB_ROOT", str(root))
    _write_lens(tmp_path, source=_RandomLensSource(), model_id=MODEL, root=str(root))
    return root


def config_for(root, **overrides):
    values = {"modelID": MODEL, "revision": REVISION, "lensID": LENS_ID,
              "layers": [0, 2], "topK": 3, "pin": ["law"],
              "promptMode": "rawCompletion", "dtype": "float32"}
    values.update(overrides)
    if "prompts" not in values:
        values["prompts"] = write_prompts(root)
    return batch.ProbeBatchConfig.from_dict(values)


def single(root, model, text, config):
    return probe.probe(MODEL, prompt=text, lens_id=LENS_ID, layers=config.layers,
                       top_k=config.topK, pin_words=list(config.pin or []),
                       pin_ids=list(config.pinIDs or []),
                       directions=list(config.directions or []),
                       variant_path=config.variantPath, prompt_mode=config.promptMode,
                       system_prompt=config.systemPrompt,
                       position_stride=config.positionStride, max_tokens=config.maxTokens,
                       revision=REVISION, root=str(root), model=model)


def _comparable(report):
    return {k: v for k, v in report.items() if k not in ("runID", "runDirectory", "paths")}


def _config_json(directory):
    document = json.loads((Path(directory) / "config.json").read_text())
    return {k: v for k, v in document.items() if k not in ("runId", "createdAt")}


# --- the execution ---------------------------------------------------------------

def test_each_prompt_matches_one_probe_call_exactly(workspace):
    """Same schema, same numbers, same CSV bytes: the batch restructures who
    pays the fixed costs, never what a prompt's readout is."""
    model = tiny_model()
    config = config_for(workspace)
    result = batch.probe_batch(config, root=str(workspace), log=lambda *_: None, model=model)
    run = Path(result["runDirectory"])
    manifest = json.loads((run / batch.MANIFEST).read_text())
    assert [p["id"] for p in manifest["prompts"]] == [p["id"] for p in PROMPTS]
    for row, entry in zip(PROMPTS, manifest["prompts"]):
        directory = run / entry["directory"]
        mine = json.loads((directory / probe.PROBE_JSON).read_text())
        theirs = single(workspace, model, row["text"], config)
        theirs_file = json.loads(Path(theirs["paths"]["report"]).read_text())
        assert set(mine) == set(theirs_file)
        assert _comparable(mine) == _comparable(theirs_file)
        assert mine["cells"] and mine["trajectory"]
        for name in (probe.TOPK_CSV, probe.TRAJECTORY_CSV):
            assert (directory / name).read_bytes() == \
                (Path(theirs["runDirectory"]) / name).read_bytes()
        assert _config_json(directory) == _config_json(theirs["runDirectory"])
        # The batch records where the prompt's run sits in the workspace, so
        # evidence that travels home names no machine.
        assert mine["runDirectory"] == (run / entry["directory"]).relative_to(workspace).as_posix()
        assert entry["projections"] == theirs["configuration"]["projections"]
        assert entry["labels"] == row.get("labels", {})
    assert (run / "COMPLETED").read_text() == "jlens-probe-batch\n"


def test_the_model_loads_once_and_the_lens_is_built_once_for_n_prompts(workspace, monkeypatch):
    from steerlab_server.jlens import lens_store, qualification
    from steerlab_server.steering import model_loader
    model = tiny_model()
    counts = {"load": 0, "verify": 0, "layers": 0}
    monkeypatch.setattr(model_loader, "load",
                        lambda *a, **k: counts.__setitem__("load", counts["load"] + 1) or model)
    verify, load_layer = qualification.verify_converted, lens_store.load_layer

    def counted_verify(*a, **k):
        counts["verify"] += 1
        return verify(*a, **k)

    def counted_layer(*a, **k):
        counts["layers"] += 1
        return load_layer(*a, **k)

    monkeypatch.setattr(qualification, "verify_converted", counted_verify)
    monkeypatch.setattr(lens_store, "load_layer", counted_layer)
    config = config_for(workspace)
    result = batch.probe_batch(config, root=str(workspace), log=lambda *_: None)
    assert counts == {"load": 1, "verify": 1, "layers": 2}
    manifest = json.loads(Path(result["reportPath"]).read_text())
    assert manifest["fixedCosts"] == {"modelLoads": 1, "readoutBuilds": 1,
                                      "lensVerifications": 1, "jacobianPlacements": 2}
    # The single-prompt path pays them every call: N prompts, N builds.
    counts.update(verify=0, layers=0)
    for row in PROMPTS:
        single(workspace, model, row["text"], config)
    assert counts["verify"] == len(PROMPTS) and counts["layers"] == 2 * len(PROMPTS)


def test_the_job_record_stays_small_and_names_one_run(workspace):
    result = batch.probe_batch(config_for(workspace), root=str(workspace),
                               log=lambda *_: None, model=tiny_model())
    assert set(result) == {"runDirectory", "reportPath", "prompts", "projections",
                           "claim", "evidenceTier"}
    assert len(json.dumps(result)) < 1024
    run = Path(result["runDirectory"]).relative_to(workspace)
    assert run.parts[0] == "runs" and len(run.parts) == 2
    manifest = json.loads(Path(result["reportPath"]).read_text())
    assert manifest["totals"]["projections"] == result["projections"] == sum(
        p["projections"] for p in manifest["prompts"])
    assert manifest["inputs"]["prompts"]["sha256"] == config_for(workspace).prompts["sha256"]
    assert set(manifest["timings"]) == {"modelLoadSeconds", "readoutBuildSeconds",
                                        "promptSeconds", "totalSeconds"}
    files = sorted(p.relative_to(workspace / run).as_posix()
                   for p in (workspace / run).rglob("*") if p.is_file())
    assert len(files) == batch.RUN_FILES + batch.FILES_PER_PROMPT * len(PROMPTS)


def test_a_directions_only_batch_with_an_agent_matches_single_probes(workspace):
    """topK 0 with directions costs no projections; the agent's injections
    and rendering apply to every prompt exactly as they do to one."""
    model = tiny_model()
    vector = write_direction(workspace)
    agent = write_agent(workspace, vector)
    config = config_for(workspace, topK=0, pin=None, directions=[vector],
                        variantPath=agent, promptMode=None, allLayers=True, layers=None)
    result = batch.probe_batch(config, root=str(workspace), log=lambda *_: None, model=model)
    manifest = json.loads(Path(result["reportPath"]).read_text())
    assert result["projections"] == 0
    assert manifest["condition"]["variantName"] == "mercy-agent"
    assert manifest["condition"]["promptMode"] == "rawCompletion"
    assert manifest["lens"]["armedLayers"] == [0, 1, 2]
    assert manifest["inputs"]["injectionVectors"][0]["reference"] == vector
    for row, entry in zip(PROMPTS, manifest["prompts"]):
        mine = json.loads((Path(result["runDirectory"]) / entry["directory"] / probe.PROBE_JSON).read_text())
        theirs = json.loads(Path(single(workspace, model, row["text"],
                                        replace(config, layers=[0, 1, 2]))["paths"]["report"]).read_text())
        assert _comparable(mine) == _comparable(theirs)
        assert mine["cells"] == [] and "directions" in mine["trajectory"][0]
        assert mine["condition"]["injections"]


def test_execution_rechecks_every_prompt_before_any_forward_pass(workspace, monkeypatch):
    model = tiny_model()
    long = {"id": "long", "text": " ".join(["word"] * 40000)}
    config = config_for(workspace, prompts=write_prompts(workspace, [PROMPTS[0], long]),
                        allLayers=True, layers=None)
    monkeypatch.setattr(model.hooked, "session",
                        lambda *a, **k: pytest.fail("no forward pass may run"))
    with pytest.raises(batch.ProbeBatchError, match="long: .*exceeds 200000") as refusal:
        batch.probe_batch(config, root=str(workspace), log=lambda *_: None, model=model)
    assert "positionStride" in refusal.value.repair_action
    assert not list((workspace / "runs").glob("*jlens-probe-batch*"))  # nothing left behind


def _statuses(manifest):
    return {p["id"]: p["status"] for p in manifest["prompts"]}


def _whole_prompt_directories(run):
    """Prompt directories as a reader of the run sees them: under prompts/,
    not the hidden staging area."""
    return sorted(p.name for p in (run / "prompts").iterdir() if p.is_dir() and not p.name.startswith("."))


def test_a_failure_mid_batch_leaves_a_failed_manifest(workspace, monkeypatch):
    read = probe.read_prompt
    calls = []

    def failing(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("device out of memory")
        return read(*a, **k)

    monkeypatch.setattr(probe, "read_prompt", failing)
    created = []
    with pytest.raises(RuntimeError, match="out of memory"):
        batch.probe_batch(config_for(workspace), root=str(workspace), log=lambda *_: None,
                          model=tiny_model(), on_run_created=created.append)
    run = Path(created[0])
    manifest = json.loads((run / batch.MANIFEST).read_text())
    assert manifest["status"] == "failed" and "out of memory" in manifest["error"]
    assert _statuses(manifest) == {"case-1": "completed", "case-2": "failed", "case-3": "notRead"}
    assert manifest["totals"]["prompts"] == 1 and manifest["totals"]["planned"] == 3
    assert _whole_prompt_directories(run) == ["case-1"]
    assert not (run / "prompts" / batch.PARTIAL_DIRECTORY).exists()
    assert not (run / "COMPLETED").exists()


def test_the_manifest_is_accurate_while_a_prompt_is_being_read(workspace, monkeypatch):
    """What a kill that runs no handler (SIGKILL after the grace period)
    leaves: the last atomic rewrite says running, names the whole prompts and
    the one in flight, and the in-flight prompt's files are only in staging,
    even once they are all written."""
    read = probe.read_prompt
    seen = {}

    def observed(session, prompt, planned, **k):
        run = Path(k["run_directory"]).parents[2]
        manifest = json.loads((run / batch.MANIFEST).read_text())
        if prompt == PROMPTS[1]["text"]:
            seen["before"] = (manifest["status"], _statuses(manifest), _whole_prompt_directories(run))
        report = read(session, prompt, planned, **k)
        if prompt == PROMPTS[1]["text"]:
            assert (Path(k["run_directory"]) / probe.PROBE_JSON).is_file()
            seen["written"] = _whole_prompt_directories(run)
        return report

    monkeypatch.setattr(probe, "read_prompt", observed)
    result = batch.probe_batch(config_for(workspace), root=str(workspace), log=lambda *_: None,
                               model=tiny_model())
    assert seen["before"] == ("running", {"case-1": "completed", "case-2": "running", "case-3": "pending"},
                              ["case-1"])
    assert seen["written"] == ["case-1"]
    manifest = json.loads(Path(result["reportPath"]).read_text())
    assert manifest["status"] == "completed" and set(_statuses(manifest).values()) == {"completed"}
    assert all(p["seconds"] >= 0 for p in manifest["prompts"])
    assert not (Path(result["runDirectory"]) / "prompts" / batch.PARTIAL_DIRECTORY).exists()


@pytest.mark.parametrize("during", [False, True])
def test_a_scheduler_signal_after_k_prompts_records_an_interrupted_run(workspace, monkeypatch, during):
    """The engine's checkpoint flag (set by USR1 or TERM) stops the batch at
    the next prompt, or mid-prompt at the next layer; the manifest says
    which prompts are whole and nothing partial looks complete."""
    from steerlab_server.experiment.resume import CheckpointFlag
    flag = CheckpointFlag()
    read = probe.read_prompt
    count = []

    def signalled(session, prompt, planned, **k):
        count.append(1)
        if during and len(count) == 2:
            flag.request()             # arrives while prompt 2 is being read
        report = read(session, prompt, planned, **k)
        if not during and len(count) == 2:
            flag.request()             # arrives just after prompt 2 is whole
        return report

    monkeypatch.setattr(probe, "read_prompt", signalled)
    created = []
    with pytest.raises(batch.ProbeBatchInterrupted, match="after [12] of 3") as stopped:
        batch.probe_batch(config_for(workspace), root=str(workspace), log=lambda *_: None,
                          model=tiny_model(), checkpoint=flag, on_run_created=created.append)
    assert "notRead" in str(stopped.value) or "interrupted" in str(stopped.value)
    run = Path(created[0])
    manifest = json.loads((run / batch.MANIFEST).read_text())
    assert manifest["status"] == "interrupted"
    if during:
        assert _statuses(manifest) == {"case-1": "completed", "case-2": "interrupted", "case-3": "notRead"}
        assert _whole_prompt_directories(run) == ["case-1"]
    else:
        assert _statuses(manifest) == {"case-1": "completed", "case-2": "completed", "case-3": "notRead"}
        assert _whole_prompt_directories(run) == ["case-1", "case-2"]
    assert manifest["totals"]["prompts"] == len(_whole_prompt_directories(run))
    assert not (run / "prompts" / batch.PARTIAL_DIRECTORY).exists()
    assert not (run / "COMPLETED").exists()


def test_a_signal_during_the_forward_pass_stops_at_the_next_layer(workspace, monkeypatch):
    """A cancel gives the child seconds, so the batch does not wait for the
    prompt to finish: the check before each layer's readouts stops it."""
    from steerlab_server.experiment.resume import CheckpointFlag
    flag = CheckpointFlag()
    apply = probe._AllPositionCapture.apply
    reads = []

    def capturing(self, h, layer, offset):
        if len(reads) == 2:
            flag.request()             # prompt 2's forward pass is under way
        return apply(self, h, layer, offset)

    read = probe.read_prompt

    def counted(*a, **k):
        reads.append(1)
        return read(*a, **k)

    monkeypatch.setattr(probe._AllPositionCapture, "apply", capturing)
    monkeypatch.setattr(probe, "read_prompt", counted)
    created = []
    with pytest.raises(batch.ProbeBatchInterrupted) as stopped:
        batch.probe_batch(config_for(workspace), root=str(workspace), log=lambda *_: None,
                          model=tiny_model(), checkpoint=flag, on_run_created=created.append)
    assert "before reading layer 0" in str(stopped.value.__cause__)
    run = Path(created[0])
    manifest = json.loads((run / batch.MANIFEST).read_text())
    assert _statuses(manifest) == {"case-1": "completed", "case-2": "interrupted", "case-3": "notRead"}
    assert _whole_prompt_directories(run) == ["case-1"]


def _send_during_prompt(monkeypatch, number, at=2):
    import signal
    read = probe.read_prompt
    count = []

    def signalled(*a, **k):
        count.append(1)
        if len(count) == at:
            os.kill(os.getpid(), number)
        return read(*a, **k)

    monkeypatch.setattr(probe, "read_prompt", signalled)


def test_a_real_term_stops_the_batch_and_the_handlers_are_restored(workspace, monkeypatch):
    """TERM (a cancel, or the walltime itself) is forwarded by the scheduler
    script; the batch stops at the next layer. Afterwards the process's own
    handlers are back."""
    import signal
    before = (signal.getsignal(signal.SIGUSR1), signal.getsignal(signal.SIGTERM))
    _send_during_prompt(monkeypatch, signal.SIGTERM)
    created = []
    with pytest.raises(batch.ProbeBatchInterrupted) as stopped:
        batch.probe_batch(config_for(workspace), root=str(workspace), log=lambda *_: None,
                          model=tiny_model(), on_run_created=created.append)
    assert stopped.value.scheduler_interrupted is True
    # The repair says what the export allows: an interrupted batch does not come home.
    assert "not exported" in str(stopped.value) and "walltime estimate" in str(stopped.value)
    manifest = json.loads((Path(created[0]) / batch.MANIFEST).read_text())
    assert manifest["status"] == "interrupted"
    assert _statuses(manifest) == {"case-1": "completed", "case-2": "interrupted", "case-3": "notRead"}
    assert (signal.getsignal(signal.SIGUSR1), signal.getsignal(signal.SIGTERM)) == before


def test_the_walltime_warning_is_recorded_and_reading_continues(workspace, monkeypatch):
    """USR1 comes signal_seconds before the walltime. There is nothing to
    checkpoint, so the batch notes it and keeps reading: here it finishes."""
    import signal
    before = (signal.getsignal(signal.SIGUSR1), signal.getsignal(signal.SIGTERM))
    _send_during_prompt(monkeypatch, signal.SIGUSR1)
    result = batch.probe_batch(config_for(workspace), root=str(workspace), log=lambda *_: None,
                               model=tiny_model())
    manifest = json.loads(Path(result["reportPath"]).read_text())
    assert manifest["status"] == "completed" and set(_statuses(manifest).values()) == {"completed"}
    assert manifest["walltimeWarningSeenAt"].endswith("Z")
    assert (Path(result["runDirectory"]) / "COMPLETED").exists()
    assert (signal.getsignal(signal.SIGUSR1), signal.getsignal(signal.SIGTERM)) == before


# --- the job record: an interruption never overrides the scheduler's outcome ---

def _interrupted_child(managed, monkeypatch, signals):
    """Run the managed child as the scheduler would, delivering ``signals``
    while prompt 2 is read; return the record it wrote and its exit code."""
    from steerlab_server.api import scientific_execution
    from steerlab_server.steering import model_loader
    root, profile = managed
    read = probe.read_prompt
    count = []

    def signalled(*a, **k):
        count.append(1)
        if len(count) == 2:
            for number in signals:
                os.kill(os.getpid(), number)
        return read(*a, **k)

    monkeypatch.setattr(probe, "read_prompt", signalled)
    monkeypatch.setattr(model_loader, "load", lambda *a, **k: tiny_model())
    plan = scientific_execution.plan(_request(root), profile)
    packet = root / "packet.json"
    packet.write_text(json.dumps(plan))
    records = root / "records"
    records.mkdir()
    code = scientific_execution.execute_packet(packet, "probe-job", records / "probe-job.json")
    return json.loads((records / "probe-job.json").read_text()), code, records


def _fold(root, records, scheduler_status):
    """The controller's view: the poll loop has set the scheduler's terminal
    state, then folds the child's record over it."""
    from steerlab_server.api.jobs import DurableJobStore, JobManager
    jobs = JobManager(store=DurableJobStore(str(root / "jobs.sqlite")), sweep_orphans=False)
    job = jobs.record_external("science:" + batch.OPERATION, status=scheduler_status, executor="slurm",
                               job_id="probe-job", result={"recordsDirectory": str(records)})
    assert job.id == "probe-job"
    jobs.reconcile(str(records), force=True)
    return jobs.get("probe-job")


@pytest.mark.parametrize("signals, scheduler_status", [
    ("TERM", "cancelled"),           # a person's scancel, or the app's cancel
    ("USR1+TERM", "failed"),         # the walltime: the warning, then TERM; Slurm's TIMEOUT maps to failed
])
def test_an_interrupted_batch_leaves_the_scheduler_outcome_on_the_job(managed, monkeypatch, signals,
                                                                      scheduler_status):
    import signal
    numbers = {"TERM": [signal.SIGTERM], "USR1+TERM": [signal.SIGUSR1, signal.SIGTERM]}[signals]
    record, code, records = _interrupted_child(managed, monkeypatch, numbers)
    assert code != 0
    assert "status" not in record                 # nothing to override the scheduler with
    assert "Interrupted by a scheduler signal" in record["error"]
    run = Path(record["result"]["runDirectory"])
    manifest = json.loads((run / batch.MANIFEST).read_text())
    assert manifest["status"] == "interrupted"
    assert ("walltimeWarningSeenAt" in manifest) is (signals == "USR1+TERM")
    job = _fold(managed[0], records, scheduler_status)
    assert job.status == scheduler_status
    assert job.result["runDirectory"] == str(run)


def test_a_genuine_failure_still_records_failed(managed, monkeypatch):
    from steerlab_server.api import scientific_execution
    from steerlab_server.steering import model_loader
    root, profile = managed
    monkeypatch.setattr(model_loader, "load", lambda *a, **k: tiny_model())
    monkeypatch.setattr(probe, "read_prompt", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad device")))
    plan = scientific_execution.plan(_request(root), profile)
    packet = root / "packet.json"
    packet.write_text(json.dumps(plan))
    records = root / "records"
    records.mkdir()
    assert scientific_execution.execute_packet(packet, "probe-job", records / "probe-job.json") == 70
    record = json.loads((records / "probe-job.json").read_text())
    assert record["status"] == "failed" and "bad device" in record["error"]
    assert _fold(root, records, "failed").status == "failed"


def test_a_pinned_id_beyond_the_output_head_refuses_before_any_run(workspace):
    created = []
    with pytest.raises(batch.ProbeBatchError, match="outside the output head of 256 rows"):
        batch.probe_batch(config_for(workspace, pinIDs=[300]), root=str(workspace),
                          log=lambda *_: None, model=tiny_model(), on_run_created=created.append)
    assert created == []


# --- the config and the prompt set ----------------------------------------------

def test_the_review_ceiling_restates_the_probe_ceiling():
    assert batch.MAX_PROJECTIONS == probe.MAX_PROJECTIONS


@pytest.mark.parametrize("change, message", [
    ({"surprise": 1}, "Unknown batch probe config"),
    ({"topK": -1}, "topK must be an integer"),
    ({"topK": True}, "topK must be an integer"),
    ({"positionStride": 0}, "positionStride must be an integer of at least 1"),
    ({"maxTokens": 0}, "maxTokens must be an integer of at least 1"),
    ({"layers": []}, "layers must be a nonempty list"),
    ({"layers": [1], "allLayers": True}, "layers or allLayers"),
    ({"promptMode": "chat"}, "promptMode must be"),
    ({"dtype": "fp8"}, "dtype must be one of"),
    ({"lensID": "runs/jlens-lenses/x"}, "registered lens"),
    ({"topK": 0, "pin": []}, "Nothing to read"),
    ({"directions": ["runs/a/b", "runs/a/b"]}, "each direction once"),
])
def test_config_refusals_name_the_problem(workspace, change, message):
    values = config_for(workspace).to_dict()
    values.update(change)
    with pytest.raises(batch.ProbeBatchError, match=message):
        batch.ProbeBatchConfig.from_dict(values)


def test_the_effective_config_round_trips(workspace):
    config = config_for(workspace)
    assert batch.ProbeBatchConfig.from_dict(config.to_dict()) == config
    assert config.to_dict()["pinIDs"] == [] and config.to_dict()["allLayers"] is False


@pytest.mark.parametrize("rows, message", [
    ([{"id": "a", "text": "x"}, {"id": "A", "text": "y"}], "ignoring case"),
    ([{"id": "../a", "text": "x"}], "id must be letters"),
    ([{"id": "a", "text": " "}], "text must be nonempty"),
    ([{"id": "a", "text": "x", "condition": "c"}], "optional labels and no other fields"),
    ([{"id": "a", "text": "x", "labels": {"c": [1]}}], "labels must map names"),
    ([], "contains no prompts"),
])
def test_prompt_set_problems_are_all_named(workspace, rows, message):
    config = config_for(workspace, prompts=write_prompts(workspace, rows))
    with pytest.raises(batch.ProbeBatchError, match=message):
        batch.read_prompts(config, workspace)


def test_a_changed_prompt_set_refuses(workspace):
    config = config_for(workspace)
    (workspace / config.prompts["path"]).write_text('{"id": "x", "text": "changed"}\n')
    with pytest.raises(batch.ProbeBatchError, match="changed after review"):
        batch.read_prompts(config, workspace)


# --- review: what the plan states before submission -----------------------------

@pytest.fixture
def tokenizer_cached(monkeypatch):
    """Stand in for the pinned revision's cached tokenizer; the real renderer runs."""
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: _Tokenizer()))


def geometry(**overrides):
    values = {"modelConfigSHA256": "c" * 64, "hiddenSize": HIDDEN, "layers": LAYERS,
              "attentionHeads": 4, "keyValueHeads": 2, "headDim": 8,
              "intermediateSize": 64, "vocabSize": 256, "storedDtype": "bfloat16",
              "checkpointBytes": 2 * 1024**3}
    values.update(overrides)
    return values


def test_review_counts_projections_memory_output_and_walltime(workspace, tokenizer_cached, monkeypatch):
    monkeypatch.setattr(batch, "cached_geometry", lambda *a: geometry())
    config = config_for(workspace)
    review = batch.preflight(config, workspace)
    counts = [len(_Tokenizer()(row["text"]).input_ids) for row in PROMPTS]
    assert review["tokenCounts"] == "exact"
    assert [p["tokenCount"] for p in review["prompts"]] == counts
    assert [p["projections"] for p in review["prompts"]] == [n * 2 * 2 for n in counts]
    assert review["totalProjections"] == sum(n * 2 * 2 for n in counts)
    memory = review["memoryEstimate"]
    assert memory["status"] == "estimated" and memory["tokenBasis"] == "exact"
    assert memory["jacobianBytes"] == 2 * HIDDEN * HIDDEN * 4
    # bfloat16 checkpoint read in float32: twice the bytes.
    assert memory["weightBytes"] == 4 * 1024**3
    assert memory["totalBytes"] > memory["weightBytes"] + memory["jacobianBytes"]
    assert review["output"]["files"] == batch.RUN_FILES + batch.FILES_PER_PROMPT * len(PROMPTS)
    workload = review["walltimeEstimate"]
    assert workload["units"] == review["totalProjections"] and workload["seconds"] >= 0
    assert "650 GB/s" in workload["formula"] and "not a measurement" in workload["limitation"]
    # The head is sized from the model's config: vocabulary × hidden size, read in float32 here.
    assert workload["outputHeadBytes"] == 256 * HIDDEN * 4
    assert "A100-SXM4-80GB" in workload["calibration"]
    # The estimate bounds what the run actually writes.
    result = batch.probe_batch(config, root=str(workspace), log=lambda *_: None, model=tiny_model())
    manifest = json.loads(Path(result["reportPath"]).read_text())
    for planned, written in zip(review["prompts"], manifest["prompts"]):
        assert planned["tokenCount"] == written["tokenCount"]
        assert planned["projections"] == written["projections"]
        assert planned["outputBytes"] >= written["bytes"]


@pytest.mark.parametrize("vocabulary", [248_077, 248_320])
def test_the_walltime_formula_covers_its_recorded_calibration_run(vocabulary):
    """One recorded A100-SXM4-80GB run: a 27B model with hidden 5120 and an
    output head of at least 248,077 rows (its tokenizer's highest id is
    248,076; 248,320 is the likely padded config value), 63 layers, topK 20,
    12 prompts. The model load took 306 s and the probing 2,347.5 s. The
    estimate must not fall below the measurement, and the default walltime
    must stay within about four times the measured job, at either size."""
    from steerlab_server.api import science_walltime
    tokens = [324, 373, 344, 393, 437, 488, 457, 508, 409, 458, 429, 478]
    config = batch.ProbeBatchConfig.from_dict({
        "modelID": "example/model", "revision": REVISION, "lensID": LENS_ID,
        "prompts": {"path": "prompts.jsonl", "sha256": "0" * 64}, "allLayers": True,
        "topK": 20, "dtype": "bfloat16"})
    prompts = [{"tokenCount": t, "scoredPositions": t, "projections": t * 63 * 2} for t in tokens]
    geometry = {"vocabSize": vocabulary, "hiddenSize": 5120, "checkpointBytes": 54 * 10**9,
                "storedDtype": "bfloat16"}
    workload = batch.walltime_estimate(config, prompts, list(range(63)), 5120, geometry,
                                       63 * 5120 * 5120 * 4)
    assert workload["units"] == 642_348
    assert workload["outputHeadBytes"] == vocabulary * 5120 * 2  # about 2.54 GB
    measured_probing, measured_job = 2347.5, 306 + 2347.5
    assert workload["seconds"] >= measured_probing
    walltime = science_walltime.default_seconds(workload["seconds"], science_walltime.parse("24:00:00"))
    assert walltime <= 4 * measured_job
    assert science_walltime.render(walltime) == "02:45:00"


def test_no_walltime_estimate_without_the_model_config_dimensions():
    """The head is sized from the model's own config; without its vocabulary
    or hidden size the plan states no estimate rather than assuming one."""
    config = batch.ProbeBatchConfig.from_dict({
        "modelID": "example/model", "revision": REVISION, "lensID": LENS_ID,
        "prompts": {"path": "prompts.jsonl", "sha256": "0" * 64}, "topK": 20})
    prompts = [{"tokenCount": 10, "scoredPositions": 10, "projections": 60}]
    for geometry in ({"hiddenSize": 5120}, {"vocabSize": 248_077}, None):
        assert batch.walltime_estimate(config, prompts, [0, 1, 2], 5120, geometry, 0) is None


def test_review_lists_a_bounded_number_of_prompts_and_names_the_largest(workspace, tokenizer_cached, monkeypatch):
    """The plan travels in the job record, so a long prompt set is summarized."""
    monkeypatch.setattr(batch, "REVIEW_PROMPTS_LISTED", 2)
    review = batch.preflight(config_for(workspace), workspace)
    assert review["promptCount"] == 3 and review["promptsListed"] == len(review["prompts"]) == 2
    assert review["largestPrompt"]["projections"] == max(
        len(_Tokenizer()(row["text"]).input_ids) * 4 for row in PROMPTS)
    assert review["totalProjections"] == sum(len(_Tokenizer()(row["text"]).input_ids) * 4 for row in PROMPTS)
    assert all("labels" not in entry for entry in review["prompts"])


def test_review_resolves_pins_where_it_can_and_says_where_it_cannot(workspace, tokenizer_cached, monkeypatch):
    """With the cached tokenizer and config, pins are checked exactly as the
    node checks them; a word that is two tokens, or an id beyond the head,
    refuses before submission."""
    monkeypatch.setattr(batch, "cached_geometry", lambda *a: geometry())
    review = batch.preflight(config_for(workspace, pinIDs=[7]), workspace)
    assert review["pins"] == {"words": {" law": _Tokenizer._id("law")}, "ids": "withinOutputHead"}
    assert any("Pinned words resolve to" in line for line in review["summary"])
    with pytest.raises(batch.ProbeBatchError, match="2 tokens") as split:
        batch.preflight(config_for(workspace, pin=["two words"]), workspace)
    assert "pinIDs" in split.value.repair_action
    with pytest.raises(batch.ProbeBatchError, match="outside the output head of 256 rows"):
        batch.preflight(config_for(workspace, pinIDs=[256]), workspace)


def test_without_a_tokenizer_or_config_pins_are_checked_on_the_node(workspace):
    review = batch.preflight(config_for(workspace, pinIDs=[999]), workspace)
    assert review["pins"] == {"words": "checkedOnTheNode", "ids": "checkedOnTheNode"}
    assert any("resolved on the node" in line for line in review["summary"])
    assert any("checked against the output head on the node" in line for line in review["summary"])


def test_review_without_a_cached_tokenizer_says_so_and_bounds_by_max_tokens(workspace):
    review = batch.preflight(config_for(workspace), workspace)
    assert review["tokenCounts"] == "unavailable" and review["totalProjections"] is None
    assert "re-checks every prompt" in " ".join(review["summary"])
    bounded = batch.preflight(config_for(workspace, maxTokens=5), workspace)
    assert all(p["tokenCountBasis"] == "maxTokens bound" and p["projections"] == 5 * 2 * 2
               for p in bounded["prompts"])


def test_review_refuses_every_prompt_over_the_ceiling(workspace, tokenizer_cached):
    rows = [PROMPTS[0], {"id": "long-a", "text": " ".join(["word"] * 40000)},
            {"id": "long-b", "text": " ".join(["case"] * 50000)}]
    config = config_for(workspace, prompts=write_prompts(workspace, rows), allLayers=True, layers=None)
    with pytest.raises(batch.ProbeBatchError, match="2 prompt\\(s\\) exceed") as refusal:
        batch.preflight(config, workspace)
    assert "long-a" in str(refusal.value) and "long-b" in str(refusal.value)
    assert "case-1" not in str(refusal.value)
    assert "positionStride" in refusal.value.repair_action and "topK" in refusal.value.repair_action
    assert batch.preflight(replace(config, positionStride=2), workspace)["totalProjections"]
    directions_only = replace(config, topK=0, pin=None, directions=[write_direction(workspace)])
    assert batch.preflight(directions_only, workspace)["totalProjections"] == 0


def test_review_refuses_output_one_export_cannot_carry(workspace, tokenizer_cached, monkeypatch):
    monkeypatch.setattr(archives, "MAX_BYTES", 1024)
    with pytest.raises(batch.ProbeBatchError, match="more than the .* one export can carry"):
        batch.preflight(config_for(workspace), workspace)


@pytest.mark.parametrize("variant, message", [
    ({"baseModelID": "other/model"}, "was built for other/model"),
    ({"interventionPolicies": [{"name": "p"}]}, "intervention policies"),
])
def test_review_refuses_an_agent_the_probe_cannot_run(workspace, variant, message):
    agent = workspace / "agent.json"
    agent.write_text(json.dumps({"name": "a", "baseModelID": MODEL, **variant}))
    with pytest.raises(batch.ProbeBatchError, match=message):
        batch.preflight(config_for(workspace, variantPath="agent.json"), workspace)


def test_the_agent_files_are_pinned_in_the_input_closure(workspace):
    from steerlab_server.experiment import managed_inputs
    vector = write_direction(workspace)
    agent = write_agent(workspace, vector)
    config = config_for(workspace, variantPath=agent).to_dict()
    inventory = managed_inputs.inventory(batch.OPERATION, config, workspace)
    paths = {entry["path"] for entry in inventory}
    assert {agent, vector + ".json", vector + ".safetensors", config["prompts"]["path"]} <= paths
    assert any(path.endswith("/lens.json") for path in paths)
    # Changed agent bytes change the pinned inventory, which refuses a queued job.
    (workspace / (vector + ".json")).write_text((workspace / (vector + ".json")).read_text() + " ")
    assert managed_inputs.inventory(batch.OPERATION, config, workspace) != inventory


def test_an_agent_file_outside_the_workspace_refuses(workspace, tmp_path):
    outside = tmp_path / "elsewhere" / "vec"
    outside.parent.mkdir()
    agent = workspace / "agent.json"
    agent.write_text(json.dumps({"name": "a", "baseModelID": MODEL, "injections": [
        {"vectorArtifactID": str(outside), "layer": 1, "alpha": 0.5}]}))
    with pytest.raises(batch.ProbeBatchError, match="outside the workspace"):
        batch.agent_files(config_for(workspace, variantPath="agent.json"), workspace)


# --- the managed path: interview, draft, publish, plan, stage, execute, export ---

def _answers(root):
    prompts = write_prompts(root)
    return dict(purpose="Where does the mercy reading appear across the case texts?",
                claim="Readout only", controls="Logit-lens companion at every cell",
                selection="Prompt set and layers fixed before reading",
                fields={"modelID": MODEL, "revision": REVISION, "lensID": LENS_ID,
                        "prompts": prompts["path"], "layers": "0, 2", "topK": "3",
                        "pin": "law", "promptMode": "rawCompletion", "dtype": "float32"},
                advanced={})


@pytest.fixture
def managed(workspace, monkeypatch):
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(workspace / ".steerlab"))
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "workstation")
    monkeypatch.setenv("STEERLAB_EXECUTOR", "local")
    from steerlab_server.api.profile import ServerProfile
    return workspace, ServerProfile.from_env()


def test_interview_draft_publish_and_plan(managed, monkeypatch):
    from steerlab_server.api import scientific_execution
    root, profile = managed
    interview = method_authoring.interview(batch.OPERATION)
    assert {f["id"] for f in interview["fields"]} >= {"modelID", "revision", "lensID", "prompts",
                                                     "variantPath", "promptMode", "systemPrompt"}
    answers = _answers(root)
    draft = method_authoring.draft(batch.OPERATION, answers, root)
    config = draft["request"]["parameters"]["config"]
    assert config["layers"] == [0, 2] and config["pin"] == ["law"] and config["allLayers"] is False
    assert managed_methods.validate(batch.OPERATION, config, root) is not None
    assert draft["operationReview"]["promptCount"] == len(PROMPTS)
    published = method_authoring.publish(batch.OPERATION, answers, root, "requests/probe-set-1",
                                         draft["planSHA256"])
    request = json.loads(Path(published["requestFile"]).read_text())
    plan = scientific_execution.plan(request, profile)
    assert plan["compute"] == "gpu" and plan["executor"] == "local"
    assert plan["operationReview"]["armedLayers"] == [0, 2]
    assert plan["memoryReview"]["memoryFit"] == "notChecked"


def test_staged_execution_and_export_carry_one_run(managed, monkeypatch):
    from steerlab_server.api import diagnostic_transport, scientific_execution
    from steerlab_server.experiment import diagnostic_inputs
    from steerlab_server.steering import model_loader
    root, profile = managed
    vector = write_direction(root)
    agent = write_agent(root, vector)
    request = {"operation": batch.OPERATION, "parameters": {"config": config_for(
        root, variantPath=agent, promptMode=None).to_dict()}}
    source = diagnostic_inputs.plan(request, root)
    assert {agent, vector + ".safetensors"} <= {e["path"] for e in source["files"]}
    packed = diagnostic_inputs.package(request, root, root / "runs/input.tar.gz", source["planSHA256"])
    staged = diagnostic_transport.stage(packed["bundlePath"], packed["bundleSha256"], profile)
    plan = scientific_execution.plan(staged["request"], profile)
    monkeypatch.setattr(model_loader, "load", lambda *a, **k: tiny_model())
    packet = root / "packet.json"
    packet.write_text(json.dumps(plan))
    record = root / "record.json"
    assert scientific_execution.execute_packet(packet, "probe-job", record) == 0
    result = json.loads(record.read_text())["result"]
    run = Path(result["runDirectory"])
    assert run.parent == Path(staged["executionRoot"]) / "runs"
    assert result["prompts"] == len(PROMPTS) and "cells" not in json.dumps(result)
    job = SimpleNamespace(status="succeeded", finished_at=1.0, kind="science:" + batch.OPERATION,
                          result={**result, "scientificPlan": plan})
    exported = diagnostic_transport.export("probe-job", SimpleNamespace(get=lambda _id: job), profile)
    names = {entry["path"] for entry in exported["entries"]}
    relative = run.relative_to(staged["executionRoot"]).as_posix()
    assert relative + "/" + batch.MANIFEST in names
    for row in PROMPTS:
        assert relative + f"/prompts/{row['id']}/probe.json" in names
    assert exported["context"]["operation"] == batch.OPERATION


# --- plan placement: memory against the selected GPU, and the walltime ---------

def slurm(profile, monkeypatch, vram="A100:80,H100:80,L4:24"):
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "controller")
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_SLURM_GRES", "gpu:L4:1")
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "A100,H100,L4")
    monkeypatch.setenv("STEERLAB_SLURM_GPU_VRAM", vram)
    monkeypatch.setenv("STEERLAB_SLURM_WALLTIME", "24:00:00")
    return replace(profile, executor="slurm")


def _request(root, **overrides):
    return {"operation": batch.OPERATION,
            "parameters": {"config": config_for(root, **overrides).to_dict()}}


def test_plan_compares_memory_with_the_selected_gpu_and_sizes_the_walltime(managed, tokenizer_cached, monkeypatch):
    from steerlab_server.api import scientific_execution, science_walltime
    root, profile = managed
    profile = slurm(profile, monkeypatch)
    monkeypatch.setattr(batch, "cached_geometry", lambda *a: geometry())
    plan = scientific_execution.plan(_request(root), profile)
    assert plan["memoryFit"] == "fits" and plan["memoryReview"]["gpuType"] == "L4"
    assert plan["walltimeBasis"] == "estimated"
    estimate = plan["walltimeReview"]["estimate"]
    assert estimate["operation"] == batch.OPERATION
    assert estimate["units"] == plan["operationReview"]["totalProjections"]
    assert science_walltime.parse(plan["resources"]["walltime"]) >= 3600


def test_plan_refuses_when_the_weights_and_jacobians_alone_do_not_fit(managed, tokenizer_cached, monkeypatch):
    """The certain part: a bfloat16 checkpoint of 30 GiB read in float32 is
    60 GiB of weights, more than one L4 holds whatever the prompts are."""
    from steerlab_server.api import scientific_execution
    root, profile = managed
    profile = slurm(profile, monkeypatch)
    monkeypatch.setattr(batch, "cached_geometry", lambda *a: geometry(checkpointBytes=30 * 1024**3))
    with pytest.raises(scientific_execution.ScientificRefusal, match="one L4 declares 24 GB, less than the weights") as refusal:
        scientific_execution.plan(_request(root), profile)
    assert "gpuType A100" in refusal.value.repair_action
    placed = scientific_execution.plan(_request(root), profile, gpu_type="H100")
    assert placed["memoryFit"] == "fits" and placed["gpuReview"]["memoryFit"] == "fits"


def test_a_full_estimate_over_capacity_is_an_advisory_not_a_refusal(managed, tokenizer_cached, monkeypatch):
    """21 GiB of weights fit one L4 (24 GB); with worst-case activations and
    headroom the estimate does not. The assumption informs; it does not refuse."""
    from steerlab_server.api import scientific_execution
    root, profile = managed
    profile = slurm(profile, monkeypatch)
    monkeypatch.setattr(batch, "cached_geometry", lambda *a: geometry(checkpointBytes=int(10.5 * 1024**3)))
    plan = scientific_execution.plan(_request(root), profile)
    review = plan["memoryReview"]
    assert plan["memoryFit"] == review["memoryFit"] == "mayNotFit"
    assert review["certainBytes"] < 24 * 1024**3 < review["estimatedBytes"]
    assert review["fittingGPUTypes"] == ["A100", "H100"]
    assert "gpuType A100 or H100" in review["advisory"] and "maxTokens" in review["advisory"]
    assert "fewer layers" in review["advisory"] and "may run out of memory" in review["summary"]
    assert plan["planSHA256"]


def test_the_certain_part_is_compared_without_token_counts(managed, monkeypatch):
    """Weights and Jacobians do not depend on the prompts, so a plan without a
    cached tokenizer still refuses when they cannot fit, and otherwise says
    that activations were not estimated."""
    from steerlab_server.api import scientific_execution
    root, profile = managed
    profile = slurm(profile, monkeypatch)
    monkeypatch.setattr(batch, "cached_geometry", lambda *a: geometry(checkpointBytes=30 * 1024**3))
    with pytest.raises(scientific_execution.ScientificRefusal, match="less than the weights"):
        scientific_execution.plan(_request(root), profile)
    monkeypatch.setattr(batch, "cached_geometry", lambda *a: geometry())
    plan = scientific_execution.plan(_request(root), profile)
    assert plan["operationReview"]["memoryEstimate"]["status"] == "certainOnly"
    assert plan["memoryFit"] == "certainPartFits" and "not estimated" in plan["memoryReview"]["summary"]


def test_an_unknown_capacity_is_said(managed, monkeypatch):
    from steerlab_server.api import scientific_execution
    root, profile = managed
    profile = slurm(profile, monkeypatch, vram="A100:80,H100:80")
    monkeypatch.setattr(batch, "cached_geometry", lambda *a: geometry(checkpointBytes=30 * 1024**3))
    plan = scientific_execution.plan(_request(root, maxTokens=8), profile)
    assert plan["memoryFit"] == "capacityUnknown" and "no VRAM is declared for L4" in plan["memoryReview"]["summary"]
    assert plan["memoryReview"]["tokenBasis"] == "maxTokens bound"
    assert plan["memoryReview"]["fittingGPUTypes"] == ["A100", "H100"]


@pytest.mark.parametrize("request_text, fit", [("64G", "fits"), ("4K", "mayExceed"), ("1K", "exceeds"),
                                                ("plenty", "notChecked"), ("0", "notChecked")])
def test_host_capture_is_compared_with_the_memory_request_as_an_advisory(
        managed, tokenizer_cached, monkeypatch, request_text, fit):
    """The capture holds every armed layer × position in float32 on the host.
    Against the Slurm memory request it informs the plan; it never refuses."""
    from steerlab_server.api import scientific_execution
    root, profile = managed
    profile = slurm(profile, monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_MEMORY", request_text)
    monkeypatch.setattr(batch, "cached_geometry", lambda *a: geometry())
    plan = scientific_execution.plan(_request(root), profile)
    host = plan["memoryReview"]["host"]
    assert host["hostFit"] == fit
    assert host["hostCaptureBytes"] == plan["operationReview"]["memoryEstimate"]["hostCaptureBytes"]
    if fit in ("mayExceed", "exceeds"):
        assert "maxTokens" in host["advisory"] and "STEERLAB_SLURM_MEMORY" in host["advisory"]
    else:
        assert "advisory" not in host


def test_slurm_memory_values_are_read_in_its_own_units():
    from steerlab_server.api.science_placement import slurm_memory_bytes
    assert slurm_memory_bytes("64G") == 64 * 1024**3
    assert slurm_memory_bytes("64000") == 64000 * 1024**2   # bare values are megabytes
    assert slurm_memory_bytes("512mb") == 512 * 1024**2
    assert slurm_memory_bytes("2T") == 2 * 1024**4
    assert slurm_memory_bytes(None) is None and slurm_memory_bytes("lots") is None
    assert slurm_memory_bytes("0") == 0   # all of the node; host_memory_fit does not compare it


def test_other_operations_keep_their_plans(managed):
    """Only an operation whose review itemizes memory gains a memory review."""
    from steerlab_server.api import scientific_execution
    from test_managed_methods import geometry_request
    root, profile = managed
    plan = scientific_execution.plan(geometry_request(root), profile)
    assert plan["memoryFit"] == "notChecked" and "memoryReview" not in plan
