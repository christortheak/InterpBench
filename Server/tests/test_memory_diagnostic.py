"""Env-gated per-record memory diagnostic (open-issues §15 hunt 2).

Two contracts matter and both are asserted here:

1. ARMED, it writes well-formed lines carrying the quantities that discriminate
   the §15 hypotheses — host RSS, torch-visible device memory, cumulative
   decode steps, and armed hook fires — with the per-denominator ratios
   computed from a baseline taken at the first record.
2. UNSET, it is genuinely inert: no collector is constructed, no file appears,
   and ``HookedModel`` installs its historical hook closure so the
   per-layer-per-token hot path is untouched.
"""

import json
import os

import pytest
import torch
import torch.nn as nn

from steerlab_server import memory_diagnostic
from steerlab_server.steering.hooks import HookedModel
from steerlab_server.steering.injector import Injection, VectorInjector


# --- fakes --------------------------------------------------------------------


class _Block(nn.Module):
    def forward(self, x):
        return x


class _Inner(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(n)])


class _FakeModel(nn.Module):
    def __init__(self, n_layers, hidden):
        super().__init__()
        self.model = _Inner(n_layers)
        self.config = type("cfg", (), {"hidden_size": hidden})()

    def forward(self, x):
        h = x
        for layer in self.model.layers:
            h = layer(h)
        return h


class _FakeWriter:
    def __init__(self, run_directory):
        self.run_directory = run_directory
        self.records = []


class _FakeSteered:
    def __init__(self, hooked):
        self.hooked = hooked


class _FakeCondition:
    def __init__(self, name="steered", cells=3):
        self.name = name
        self.intervention_state = "steered"
        self.injections = list(range(cells))
        self.latent_edits = []


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv(memory_diagnostic.ENV_VAR, raising=False)
    memory_diagnostic.reset()
    yield
    memory_diagnostic.reset()


# --- disabled: genuinely inert ------------------------------------------------


def test_unset_writes_nothing_and_constructs_no_collector(tmp_path, monkeypatch):
    def boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("collector constructed while the diagnostic is unset")

    monkeypatch.setattr(memory_diagnostic, "MemoryDiagnostic", boom)
    writer = _FakeWriter(str(tmp_path))

    assert memory_diagnostic.observe(writer, None, _FakeCondition()) is None

    assert not os.path.exists(tmp_path / memory_diagnostic.FILENAME)
    assert memory_diagnostic._ACTIVE == {}


def test_unset_leaves_the_hook_path_uninstrumented():
    fake = _FakeModel(n_layers=3, hidden=4)
    hooked = HookedModel(fake)
    # No counter block at all: the installed closure is the historical one, so
    # the per-layer-per-token path does not even test a flag.
    assert hooked.counters is None
    with hooked.session([VectorInjector.single(1, [1.0] * 4, 1.0)]):
        fake(torch.zeros((1, 2, 4)))
    assert hooked.counters is None


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_falsy_values_read_as_disabled(monkeypatch, value):
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, value)
    assert memory_diagnostic.level() == 0
    assert not memory_diagnostic.enabled()


# --- enabled: well-formed lines -----------------------------------------------


def test_enabled_writes_one_well_formed_line_per_record(tmp_path, monkeypatch):
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "1")
    writer = _FakeWriter(str(tmp_path))
    condition = _FakeCondition(cells=3)

    for _ in range(3):
        writer.records.append({"stub": True})
        assert memory_diagnostic.observe(writer, None, condition) is not None

    path = tmp_path / memory_diagnostic.FILENAME
    lines = [json.loads(line) for line in
             path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3
    assert [line["recordIndex"] for line in lines] == [0, 1, 2]
    assert [line["writerRecords"] for line in lines] == [1, 2, 3]
    for line in lines:
        assert line["schemaVersion"] == memory_diagnostic.SCHEMA_VERSION
        assert line["condition"] == "steered"
        assert line["interventionState"] == "steered"
        # The armed-cell count is what the RSS slope gets correlated against.
        assert line["armedCells"] == 3
        assert line["rssBytes"] is None or line["rssBytes"] > 0
        assert line["elapsedSeconds"] >= 0
        # Present-and-null rather than absent when the platform lacks them, so
        # every line has the same key set and a reader never KeyErrors.
        for key in ("cudaAllocatedBytes", "cudaReservedBytes", "deviceName",
                    "cumulativeDecodeSteps", "armedHookFires",
                    "hostBytesPerDecodeStep", "hostBytesPerArmedHookFire",
                    "hostBytesPerRecord", "mallocArenaMax"):
            assert key in line


def test_line_carries_hook_counters_and_the_stale_injector_probe(tmp_path,
                                                                 monkeypatch):
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "1")
    fake = _FakeModel(n_layers=4, hidden=4)
    hooked = HookedModel(fake)
    model = _FakeSteered(hooked)
    writer = _FakeWriter(str(tmp_path))

    # One prefill (seq 4) + two decode steps, with two cells armed.
    injectors = [VectorInjector.single(1, [1.0] * 4, 1.0),
                 VectorInjector.single(2, [1.0] * 4, 1.0)]
    with hooked.session(injectors):
        fake(torch.zeros((1, 4, 4)))
        fake(torch.zeros((1, 1, 4)))
        fake(torch.zeros((1, 1, 4)))

    line = memory_diagnostic.observe(writer, model, _FakeCondition(cells=2))

    assert line["cumulativeForwardPasses"] == 3
    assert line["cumulativeDecodeSteps"] == 2
    assert line["cumulativePrefillTokens"] == 4
    # 4 layers x 3 passes, all armed; two interventions applied at each.
    assert line["hookFires"] == 12
    assert line["armedHookFires"] == 12
    assert line["interventionApplications"] == 24
    assert line["forwardHookCount"] == 4
    # The stale-injector probe: between records nothing may stay armed.
    assert line["armedInterventions"] == 0


def test_the_allocation_counters_separate_dispatch_from_actual_injection(
        tmp_path, monkeypatch):
    """The correction at the heart of §15's hunt-2 hypothesis.

    The armed hook does NOT allocate a tensor at every layer: ``apply`` returns
    the ``h`` it was handed wherever the intervention does not target the layer.
    What it used to do at every armed layer was rebuild the output tuple. Both
    are counted here, separately, on a model whose blocks return tuples.
    """
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "1")

    class _TupleBlock(nn.Module):
        def forward(self, x):
            return (x, None)

    class _TupleInner(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = nn.ModuleList([_TupleBlock() for _ in range(n)])

    class _TupleModel(nn.Module):
        def __init__(self, n, hidden):
            super().__init__()
            self.model = _TupleInner(n)
            self.config = type("cfg", (), {"hidden_size": hidden})()

        def forward(self, x):
            h = x
            for layer in self.model.layers:
                h = layer(h)[0]
            return h

    hidden = 4
    fake = _TupleModel(8, hidden)
    hooked = HookedModel(fake)
    model = _FakeSteered(hooked)
    writer = _FakeWriter(str(tmp_path))

    # One consolidated injector covering 3 of the 8 layers.
    injector = VectorInjector({layer: Injection(vector=[1.0] * hidden, alpha=1.0)
                               for layer in (1, 3, 5)})
    with hooked.session([injector]):
        fake(torch.zeros((1, 1, hidden)))   # one decode step

    line = memory_diagnostic.observe(writer, model, _FakeCondition(cells=3))

    assert line["armedHookFires"] == 8
    # Dispatch: one armed object × 8 layers. The historical chain of three
    # injectors would have made 24 calls for the same three injections.
    assert line["interventionDispatches"] == 8
    assert line["interventionApplications"] == 8   # armed objects × fires
    # Only the three targeted layers allocate anything at all.
    assert line["effectiveInjections"] == 3
    assert line["replacementTensors"] == 3
    # …and only those three rebuild the block's output tuple. Before
    # 2026-08-18 this read 8 — one per armed layer, every step.
    assert line["outputTuplesRebuilt"] == 3
    # 3 clones of [1, 1, 4] float32.
    assert line["injectionCloneBytes"] == 3 * 4 * 4


def test_the_consolidated_chain_dispatches_once_per_layer_not_once_per_cell(
        tmp_path, monkeypatch):
    """The counter that verifies the plan-level consolidation live, from a
    cluster job's own diagnostic file: an 11-cell band on a 62-layer model must
    read 62 dispatches per decode step, not 682."""
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "1")
    from steerlab_server.steering import plan as plan_mod

    hidden = 4
    layers = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54]
    edits = [plan_mod.Edit(layer=layer, vector=[1.0] * hidden, strength=0.1,
                           mode=plan_mod.Mode.ADD, concept=f"c{i}")
             for i, layer in enumerate(layers)]

    def dispatches(chain):
        fake = _FakeModel(n_layers=62, hidden=hidden)
        hooked = HookedModel(fake)
        with hooked.session(chain):
            fake(torch.zeros((1, 1, hidden)))
        return hooked.counters

    consolidated = dispatches(plan_mod.interventions(edits))
    assert consolidated["interventionDispatches"] == 62
    assert consolidated["effectiveInjections"] == 11

    # The historical shape, for the record: one injector per cell.
    legacy = dispatches([VectorInjector.single(edit.layer, edit.vector,
                                               edit.strength)
                         for edit in edits])
    assert legacy["interventionDispatches"] == 682
    assert legacy["effectiveInjections"] == 11


def test_the_counting_twin_does_not_change_what_the_model_computes():
    """The instrumented path delegates to the measured one and the counting
    proxy is transparent, so an armed forward pass produces the same tensor
    with the diagnostic on as with it off."""
    hidden = 4
    injections = {1: Injection(vector=[2.0] * hidden, alpha=1.5)}

    def run(armed_env, monkey_env):
        os.environ.pop(memory_diagnostic.ENV_VAR, None)
        if monkey_env is not None:
            os.environ[memory_diagnostic.ENV_VAR] = monkey_env
        try:
            fake = _FakeModel(n_layers=3, hidden=hidden)
            hooked = HookedModel(fake)
            with hooked.session([VectorInjector(dict(injections))]):
                return fake(torch.arange(12, dtype=torch.float32).reshape(1, 3, hidden))
        finally:
            os.environ.pop(memory_diagnostic.ENV_VAR, None)

    plain = run(True, None)
    counted = run(True, "1")
    assert torch.equal(plain, counted)


def test_nested_sessions_do_not_double_wrap_the_counting_proxy(monkeypatch):
    """``session()`` restores the previously armed (already wrapped) list on
    exit; re-wrapping it would double every per-apply count for the rest of the
    run — and silently, since the numbers stay plausible."""
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "1")
    hidden = 4
    fake = _FakeModel(n_layers=2, hidden=hidden)
    hooked = HookedModel(fake)
    outer = VectorInjector.single(0, [1.0] * hidden, 1.0)

    with hooked.session([outer]):
        with hooked.session([VectorInjector.single(1, [1.0] * hidden, 1.0)]):
            pass
        # Back to the outer arm, still exactly one layer of wrapping.
        (restored,) = hooked.interventions
        assert restored.wrapped is outer
        before = hooked.counters["interventionDispatches"]
        fake(torch.zeros((1, 1, hidden)))
        assert hooked.counters["interventionDispatches"] - before == 2


def test_unarmed_passes_are_counted_separately_from_armed_ones(tmp_path,
                                                               monkeypatch):
    """The armed/unarmed split is the discriminator for hunt 2's top
    hypothesis, so it must actually distinguish the two branches."""
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "1")
    fake = _FakeModel(n_layers=2, hidden=4)
    hooked = HookedModel(fake)
    model = _FakeSteered(hooked)
    writer = _FakeWriter(str(tmp_path))

    fake(torch.zeros((1, 1, 4)))  # unarmed decode step
    with hooked.session([VectorInjector.single(0, [1.0] * 4, 1.0)]):
        fake(torch.zeros((1, 1, 4)))  # armed decode step

    line = memory_diagnostic.observe(writer, model, _FakeCondition(cells=1))
    assert line["hookFires"] == 4          # 2 layers x 2 passes
    assert line["armedHookFires"] == 2     # only the armed pass
    assert line["cumulativeDecodeSteps"] == 2


def test_ratios_are_measured_from_the_first_record_not_from_process_start(
        tmp_path, monkeypatch):
    """Model-load memory must not pollute the slope, so the baseline is the
    first observation and record 0's deltas are zero by construction."""
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "1")
    fake = _FakeModel(n_layers=2, hidden=4)
    hooked = HookedModel(fake)
    model = _FakeSteered(hooked)
    writer = _FakeWriter(str(tmp_path))

    first = memory_diagnostic.observe(writer, model, _FakeCondition())
    assert first["rssDeltaBytes"] == 0
    assert first["decodeStepsSinceBaseline"] == 0
    assert first["armedHookFiresSinceBaseline"] == 0
    # Zero denominators must not raise or produce inf.
    assert first["hostBytesPerDecodeStep"] is None
    assert first["hostBytesPerRecord"] is None

    with hooked.session([VectorInjector.single(0, [1.0] * 4, 1.0)]):
        fake(torch.zeros((1, 1, 4)))
        fake(torch.zeros((1, 1, 4)))
    second = memory_diagnostic.observe(writer, model, _FakeCondition())
    assert second["decodeStepsSinceBaseline"] == 2
    assert second["armedHookFiresSinceBaseline"] == 4  # 2 layers x 2 passes
    assert second["hostBytesPerDecodeStep"] is not None


def test_the_baseline_is_retaken_when_the_condition_changes(tmp_path,
                                                            monkeypatch):
    """A run walks its arms in sequence. With one run-global baseline, the
    slope reported for the steered arm carries the baseline arm's growth too —
    which is precisely the comparison the §15 matrix exists to make, so the
    unprefixed ratios are per CONDITION and the cumulative pair moved to
    ``run*``."""
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "1")
    fake = _FakeModel(n_layers=2, hidden=4)
    hooked = HookedModel(fake)
    model = _FakeSteered(hooked)
    writer = _FakeWriter(str(tmp_path))

    baseline_arm = _FakeCondition(name="baseline", cells=0)
    steered_arm = _FakeCondition(name="steered", cells=1)

    memory_diagnostic.observe(writer, model, baseline_arm)      # record 0
    fake(torch.zeros((1, 1, 4)))                                # unarmed step
    second = memory_diagnostic.observe(writer, model, baseline_arm)
    assert second["conditionIndex"] == 0
    assert second["conditionRecordIndex"] == 1
    assert second["decodeStepsSinceBaseline"] == 1

    # The arm changes: a fresh baseline, and the per-condition counters restart
    # while the run-global ones keep counting.
    first_steered = memory_diagnostic.observe(writer, model, steered_arm)
    assert first_steered["conditionIndex"] == 1
    assert first_steered["conditionRecordIndex"] == 0
    assert first_steered["conditionBaselineRecordIndex"] == 2
    assert first_steered["rssDeltaBytes"] == 0            # per-condition
    assert first_steered["decodeStepsSinceBaseline"] == 0
    assert first_steered["hostBytesPerDecodeStep"] is None
    assert first_steered["runDecodeStepsSinceBaseline"] == 1
    assert first_steered["runRssDeltaBytes"] is not None

    with hooked.session([VectorInjector.single(0, [1.0] * 4, 1.0)]):
        fake(torch.zeros((1, 1, 4)))
        fake(torch.zeros((1, 1, 4)))
    last = memory_diagnostic.observe(writer, model, steered_arm)
    assert last["decodeStepsSinceBaseline"] == 2          # this arm only
    assert last["runDecodeStepsSinceBaseline"] == 3       # the whole run
    assert last["armedHookFiresSinceBaseline"] == 4       # 2 layers x 2 steps
    # The marginal reading needs no baseline at all.
    assert last["decodeStepsThisRecord"] == 2
    assert last["rssDeltaSincePreviousRecord"] is not None


def test_a_single_condition_file_reads_the_same_both_ways(tmp_path, monkeypatch):
    """The rename must not change what a one-arm job reports: with a single
    condition the per-condition and run-global baselines are the same record."""
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "1")
    writer = _FakeWriter(str(tmp_path))
    condition = _FakeCondition()
    for _ in range(3):
        writer.records.append({"stub": True})
        line = memory_diagnostic.observe(writer, None, condition)
    for name in ("rssDeltaBytes", "decodeStepsSinceBaseline",
                 "hostBytesPerDecodeStep", "hostBytesPerRecord"):
        run_name = "run" + name[0].upper() + name[1:]
        assert line[name] == line[run_name]


def test_token_counts_are_read_off_the_record_never_computed(tmp_path,
                                                             monkeypatch):
    """Read, don't compute: a diagnostic that re-tokenized to fill a field
    would be measuring its own work. Absent keys stay null — today's sampled
    record carries no token count, which is why the hook counter's
    ``decodeStepsThisRecord`` is the reliable denominator."""
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "1")
    writer = _FakeWriter(str(tmp_path))

    writer.records.append({"output": "some text"})
    bare = memory_diagnostic.observe(writer, None, _FakeCondition())
    assert bare["promptTokens"] is None
    assert bare["generatedTokens"] is None

    writer.records.append({"promptTokenCount": 41, "outputTokenIDs": [1, 2, 3]})
    counted = memory_diagnostic.observe(writer, None, _FakeCondition())
    assert counted["promptTokens"] == 41
    assert counted["generatedTokens"] == 3


def test_tier_two_adds_the_gc_census(tmp_path, monkeypatch):
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "2")
    writer = _FakeWriter(str(tmp_path))
    line = memory_diagnostic.observe(writer, None, _FakeCondition())
    assert line["tier"] == 2
    assert line["gcObjects"] > 0
    assert line["liveTensorCount"] is not None

    memory_diagnostic.reset()
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "1")
    other = _FakeWriter(str(tmp_path / "other"))
    os.makedirs(other.run_directory, exist_ok=True)
    cheap = memory_diagnostic.observe(other, None, _FakeCondition())
    assert cheap["tier"] == 1
    assert "gcObjects" not in cheap


def test_a_collection_failure_never_fails_the_run(tmp_path, monkeypatch):
    """A diagnostic that can kill a 7-hour cluster job is worse than none."""
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "1")

    def boom(*args, **kwargs):
        raise RuntimeError("collector exploded")

    monkeypatch.setattr(memory_diagnostic, "_process_memory", boom)
    assert memory_diagnostic.observe(_FakeWriter(str(tmp_path)), None,
                                     _FakeCondition()) is None


def test_collectors_are_per_run_directory(tmp_path, monkeypatch):
    monkeypatch.setenv(memory_diagnostic.ENV_VAR, "1")
    one = _FakeWriter(str(tmp_path / "a"))
    two = _FakeWriter(str(tmp_path / "b"))
    os.makedirs(one.run_directory)
    os.makedirs(two.run_directory)

    memory_diagnostic.observe(one, None, _FakeCondition())
    memory_diagnostic.observe(one, None, _FakeCondition())
    memory_diagnostic.observe(two, None, _FakeCondition())

    assert len(memory_diagnostic._ACTIVE) == 2
    a_lines = (tmp_path / "a" / memory_diagnostic.FILENAME).read_text(
        encoding="utf-8").splitlines()
    b_lines = (tmp_path / "b" / memory_diagnostic.FILENAME).read_text(
        encoding="utf-8").splitlines()
    assert len(a_lines) == 2
    assert len(b_lines) == 1
