"""Memory preflight: the estimate is a stated model with measured constants,
and its contract has three load-bearing edges — MPS-only (the CUDA arm must
never even see an estimate), advisory-only (it warns, it cannot block), and
budgeted against PHYSICAL memory rather than the MPS watermark."""

from types import SimpleNamespace

from steerlab_server.experiment import memory_preflight
from steerlab_server.experiment.memory_preflight import MemoryModel


def _model(**overrides) -> MemoryModel:
    # gemma-3-4b-it as measured: 34 layers (28 sliding / 6 global at a
    # 1024-token window), 4 KV heads x 256 head_dim, calibrated 16 GiB @ 10K.
    defaults = dict(weights_gib=8.0, transient_gib_at_10k=16.0,
                    transient_basis="calibrated", kv_full_layers=6,
                    kv_sliding_layers=28, kv_window=1024,
                    kv_bytes_per_layer_token=2 * 4 * 256 * 2)
    defaults.update(overrides)
    return MemoryModel(**defaults)


def _scenario_report(*prompt_tokens: int, reserve: int = 2048) -> dict:
    return {"turns": [
        {"turnIndex": i + 1, "title": f"turn-{i + 1}", "speaker": "A",
         "projectedPromptTokens": n, "reservedTokens": reserve}
        for i, n in enumerate(prompt_tokens)]}


# ---------------------------------------------------------------- the model


def test_transient_is_quadratic_in_prompt_length():
    m = _model()
    assert m.transient_gib(10_000) == 16.0
    assert m.transient_gib(20_000) == 64.0  # 2x tokens -> 4x transient
    assert m.transient_gib(0) == 0.0


def test_transient_matches_the_calibration_points():
    # The measured points the constant was fitted to (repro script docstring):
    # ~6 GiB at 6.5K, ~40 at 16K, ~51 at 20K. The fit is honest to ~25%.
    m = _model()
    assert abs(m.transient_gib(6_484) - 6.0) < 1.5
    assert abs(m.transient_gib(15_970) - 40.1) < 3.0
    assert abs(m.transient_gib(19_999) - 51.2) < 13.0


def test_chunked_prefill_makes_the_transient_linear():
    # Chunked prefill's binding pass is the LAST chunk — chunk x N instead
    # of N x N. At 50K with 1024-token chunks that's ~8 GiB where single-pass
    # would be ~400: the difference between "rehearsable locally" and dead.
    m = _model()
    assert m.transient_gib(50_000) == 16.0 * 25  # single-pass: hopeless
    chunked = m.transient_gib(50_000, prefill_chunk=1024)
    assert abs(chunked - 3.5 * 16.0 * (1024 * 50_000) / 10_000 ** 2) < 1e-9
    assert chunked < 30.0  # ~29 GiB: rehearsable on 64 GB, honestly stated
    # The formula reproduces the live measurement: 16K/1024-chunk trimmed
    # 9.0 GiB in the field; the calibrated line must land beside it.
    assert abs(m.transient_gib(15_970, prefill_chunk=1024) - 9.0) < 1.0
    # Below the chunk size nothing chunks — the quadratic formula stands.
    assert m.transient_gib(800, prefill_chunk=1024) == m.transient_gib(800)


def test_report_consults_the_chunk_policy_per_turn():
    # A 20K turn is over a 48 GiB budget single-pass (~72 GiB peak) and
    # comfortably inside it chunked (~12) — the report must ask the policy.
    m = _model()
    rep_plain = memory_preflight.report(
        _scenario_report(19_999), m, device="mps", budget=48.0)
    rep_chunked = memory_preflight.report(
        _scenario_report(19_999), m, device="mps", budget=48.0,
        prefill_chunk_for=lambda n: 1024 if n > 4096 else None)
    assert rep_plain["overBudget"] and not rep_chunked["overBudget"]


def test_kv_respects_the_sliding_window():
    m = _model()
    # At 50K tokens the 28 sliding layers stay capped at their 1024 window;
    # only the 6 global layers grow. 6*50000*4096 + 28*1024*4096 bytes.
    expected = (6 * 50_000 * 4096 + 28 * 1024 * 4096) / memory_preflight.GIB
    assert abs(m.kv_gib(50_000) - expected) < 1e-9
    assert m.kv_gib(50_000) < 1.3  # the KV term is minor — the point


def test_peak_reproduces_the_measured_16k_wall():
    # Measured during-turn driver peak at a 15,970-token prompt: ~48 GiB.
    # weights 8 + transient ~40 + KV ~0.5 — the formula lands within 2 GiB.
    peak = _model().peak_gib(15_970, 2048)
    assert 46.0 < peak < 51.0


# ------------------------------------------------------- config resolution


def _config(layer_types=None, sliding_window=1024, heads=8, kv_heads=4,
            head_dim=256, layers=34):
    text = SimpleNamespace(
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=kv_heads, head_dim=head_dim, hidden_size=2560,
        sliding_window=sliding_window, layer_types=layer_types)
    return SimpleNamespace(text_config=text)


def test_calibrated_model_uses_the_measured_constant():
    m = memory_preflight.model_from_config(
        "google/gemma-3-4b-it",
        _config(layer_types=["sliding_attention"] * 28 + ["full_attention"] * 6),
        weights_gib=8.0)
    assert m.transient_basis == "calibrated"
    assert m.transient_gib_at_10k == 16.0
    assert m.kv_full_layers == 6
    assert m.kv_sliding_layers == 28


def test_uncalibrated_model_scales_by_query_heads_and_says_so():
    # Head-count scaling for models the table lacks, flagged headScaled.
    # The rule is now VALIDATED evidence, not hope: the 12B's head-scaled
    # prediction was 32.0 and its measurement fit 30-32.3 (2026-07-30).
    m = memory_preflight.model_from_config(
        "example/unmeasured-model", _config(heads=16), weights_gib=24.0)
    assert m.transient_basis == "headScaled"
    assert m.transient_gib_at_10k == 32.0


def test_the_12b_is_calibrated():
    m = memory_preflight.model_from_config(
        "google/gemma-3-12b-it", _config(heads=16), weights_gib=22.7)
    assert m.transient_basis == "calibrated"
    assert m.transient_gib_at_10k == 32.0
    # The calibration points themselves (single-pass): 7.52 @ 5,005 and
    # 13.59 @ 6,484 — the fitted line lands beside both.
    assert abs(m.transient_gib(5_005) - 7.52) < 0.6
    assert abs(m.transient_gib(6_484) - 13.59) < 0.2


def test_sliding_window_without_layer_map_assumes_all_sliding():
    # Undercounting the minor KV term beats inventing a global-layer pattern
    # the config never stated.
    m = memory_preflight.model_from_config(
        "org/model", _config(layer_types=None), weights_gib=8.0)
    assert m.kv_full_layers == 0
    assert m.kv_sliding_layers == 34


# ---------------------------------------------------------------- contract


def test_silent_off_mps():
    # The CUDA arm never sees an estimate: sdpa there is tiled flash — no
    # quadratic term — and the reproduction script has not measured CUDA, so
    # an advisory would be a guess wearing numbers.
    rep = memory_preflight.report(
        _scenario_report(10_000), _model(), device="cuda:0", budget=48.0)
    assert rep is None
    assert memory_preflight.advisory(None) is None
    assert memory_preflight.summary(None) is None


def test_advisory_names_the_first_over_budget_turn():
    # 5K fits a 48 GiB budget; 16K (~48.8 peak) and 20K do not — and the
    # advisory names the FIRST failing turn, which is where the run dies.
    rep = memory_preflight.report(
        _scenario_report(5_000, 15_970, 19_999), _model(),
        device="mps", budget=48.0)
    assert [t["turnIndex"] for t in rep["overBudget"]] == [2, 3]
    text = memory_preflight.advisory(rep)
    assert "turn 2" in text
    assert "NOT blocked" in text  # advisory, never a refusal
    assert memory_preflight.summary(rep) is None


def test_summary_reports_the_worst_turn_when_all_fit():
    rep = memory_preflight.report(
        _scenario_report(5_000, 8_000), _model(), device="mps", budget=48.0)
    assert rep["overBudget"] == []
    assert memory_preflight.advisory(rep) is None
    text = memory_preflight.summary(rep)
    assert "turn 2" in text and "48.0" in text


def test_unknown_weights_are_flagged_not_fabricated():
    rep = memory_preflight.report(
        _scenario_report(19_999), _model(weights_gib=None),
        device="mps", budget=40.0)
    assert rep["weightsGiB"] is None
    assert "HIGHER" in memory_preflight.advisory(rep)


def test_headroom_env_override(monkeypatch):
    monkeypatch.setattr(memory_preflight, "physical_memory_gib", lambda: 64.0)
    monkeypatch.delenv(memory_preflight.HEADROOM_ENV, raising=False)
    assert memory_preflight.budget_gib() == 48.0
    monkeypatch.setenv(memory_preflight.HEADROOM_ENV, "24")
    assert memory_preflight.budget_gib() == 40.0
    monkeypatch.setenv(memory_preflight.HEADROOM_ENV, "not-a-number")
    assert memory_preflight.budget_gib() == 48.0  # bad override -> default
