"""Cross-engine vector parity harness (WS7.3) over the committed fixtures.

``tests/fixtures/parity/`` holds tiny hand-authored artifact pairs (4 layers x
8 dims) plus ``golden-<pair>.json`` files produced by THIS engine's compare
verb; the Swift suite (``VectorParityTests``) asserts the byte-identical
fixture copies under ``Tests/ExperimentKitTests/Fixtures/parity/`` against the
SAME goldens — same numbers to 1e-6, same JSON key set — so engine drift is a
test failure, not a paper-review surprise. These integer fixtures are
scaffolding for the harness; the real MLX-vs-CUDA french-vector fixture pair
is added after the first cluster session.
"""

import json
import math
import os

import pytest

from steerlab_server import cli
from steerlab_server.steering import vector_parity

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "parity")


def _fixture(name: str) -> str:
    return os.path.join(FIXTURES, f"{name}.safetensors")


def _golden(pair: str) -> dict:
    with open(os.path.join(FIXTURES, f"golden-{pair}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _assert_matches_golden(report_dict: dict, golden: dict, path="$"):
    """Same key sets at every level; numbers equal to 1e-6; exact non-numbers."""
    assert isinstance(report_dict, type(golden)), f"{path}: type mismatch"
    if isinstance(golden, dict):
        assert set(report_dict) == set(golden), f"{path}: key set differs"
        for key in golden:
            _assert_matches_golden(report_dict[key], golden[key], f"{path}.{key}")
    elif isinstance(golden, list):
        assert len(report_dict) == len(golden), f"{path}: length differs"
        for i, (got, want) in enumerate(zip(report_dict, golden)):
            _assert_matches_golden(got, want, f"{path}[{i}]")
    elif isinstance(golden, bool) or not isinstance(golden, (int, float)):
        assert report_dict == golden, f"{path}: {report_dict!r} != {golden!r}"
    else:
        assert report_dict == pytest.approx(golden, abs=1e-6), path


# --- the three scaffolding pairs against their goldens -----------------------

def test_identical_pair_matches_golden():
    report = vector_parity.compare_paths(_fixture("identical-a"),
                                         _fixture("identical-b"))
    _assert_matches_golden(report.to_dict(), _golden("identical"))
    assert report.passed
    assert report.min_cosine == pytest.approx(1.0, abs=1e-6)
    assert report.mean_norm_ratio == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_pair_matches_golden():
    report = vector_parity.compare_paths(_fixture("orthogonal-a"),
                                         _fixture("orthogonal-b"))
    _assert_matches_golden(report.to_dict(), _golden("orthogonal"))
    assert not report.passed
    assert report.min_cosine == pytest.approx(0.0, abs=1e-6)
    assert report.mean_cosine == pytest.approx(0.0, abs=1e-6)


def test_scaled_pair_matches_golden():
    report = vector_parity.compare_paths(_fixture("scaled-a"),
                                         _fixture("scaled-b"))
    _assert_matches_golden(report.to_dict(), _golden("scaled"))
    assert report.passed
    assert report.min_cosine == pytest.approx(1.0, abs=1e-6)
    for row in report.per_layer:
        assert row.norm_ratio == pytest.approx(2.0, abs=1e-6)


def test_layer_count_mismatch_compares_intersection_and_says_so():
    report = vector_parity.compare_paths(_fixture("truncated-a"),
                                         _fixture("identical-b"))
    _assert_matches_golden(report.to_dict(), _golden("truncated"))
    assert report.layer_count_mismatch
    assert report.compared_layer_count == 3
    assert report.artifact_a.layer_count == 3
    assert report.artifact_b.layer_count == 4


def test_report_json_key_set_is_the_pinned_contract():
    report = vector_parity.compare_paths(_fixture("identical-a"),
                                         _fixture("identical-b"))
    d = report.to_dict()
    assert set(d) == {"artifactA", "artifactB", "comparedLayerCount",
                      "layerCountMismatch", "pass", "perLayer", "summary",
                      "threshold"}
    assert set(d["artifactA"]) == {"hiddenSize", "layerCount", "name"}
    assert set(d["perLayer"][0]) == {"cosine", "layer", "normA", "normB",
                                     "normRatio"}
    assert set(d["summary"]) == {"meanCosine", "meanNormRatio", "minCosine",
                                 "skippedZeroNormLayers"}


# --- semantics that the fixtures cannot reach (pure, in-memory) --------------

def test_zero_norm_layers_are_skipped_and_counted():
    report = vector_parity.compare_vectors(
        "a", [[0.0, 0.0], [1.0, 0.0]],
        "b", [[1.0, 0.0], [1.0, 0.0]])
    assert report.per_layer[0].cosine is None
    assert report.per_layer[0].norm_ratio is None  # normA == 0
    assert report.per_layer[1].cosine == pytest.approx(1.0)
    assert report.skipped_zero_norm_layers == 1
    assert report.min_cosine == pytest.approx(1.0)


def test_nothing_comparable_never_passes():
    report = vector_parity.compare_vectors(
        "a", [[0.0, 0.0]], "b", [[0.0, 0.0]], threshold=0.0)
    assert report.min_cosine is None
    assert report.mean_cosine is None
    assert not report.passed
    d = report.to_dict()
    assert d["summary"]["minCosine"] is None
    assert d["pass"] is False


def test_hidden_size_mismatch_is_an_error_not_a_report():
    with pytest.raises(ValueError, match="not comparable"):
        vector_parity.compare_vectors("a", [[1.0, 2.0]], "b", [[1.0, 2.0, 3.0]])


def test_norm_ratio_is_b_over_a():
    report = vector_parity.compare_vectors("a", [[3.0, 4.0]], "b", [[6.0, 8.0]])
    assert report.per_layer[0].norm_a == pytest.approx(5.0)
    assert report.per_layer[0].norm_b == pytest.approx(10.0)
    assert report.per_layer[0].norm_ratio == pytest.approx(2.0)


def test_math_is_double_precision_sequential():
    # The pinned accumulation: plain sequential double sums (the Swift twin
    # uses the identical loop), NOT float32 — parity numbers must not wobble
    # with dtype. sqrt(2)/2-style values would round differently in f32.
    v = [1e-8, 1.0]
    report = vector_parity.compare_vectors("a", [v], "b", [v])
    expected = math.sqrt(1e-16 + 1.0)
    assert report.per_layer[0].norm_a == pytest.approx(expected, abs=0)


# --- CLI verb (exit codes are the CI gate) -----------------------------------

def test_cli_compare_passes_identical_pair(capsys):
    rc = cli.main(["vectors", "compare", _fixture("identical-a"),
                   _fixture("identical-b")])
    assert rc == 0
    out = capsys.readouterr().out
    assert json.loads(out)["pass"] is True


def test_cli_compare_fails_orthogonal_pair_nonzero(capsys):
    rc = cli.main(["vectors", "compare", _fixture("orthogonal-a"),
                   _fixture("orthogonal-b")])
    assert rc == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["pass"] is False
    assert "FAIL" in captured.err


def test_cli_compare_threshold_flag(capsys):
    # Same pair passes once the caller lowers the gate below its min cosine.
    rc = cli.main(["vectors", "compare", _fixture("orthogonal-a"),
                   _fixture("orthogonal-b"), "--threshold", "-0.5"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["threshold"] == -0.5


def test_cli_compare_writes_json_file(tmp_path, capsys):
    out_path = tmp_path / "parity.json"
    rc = cli.main(["vectors", "compare", _fixture("scaled-a"),
                   _fixture("scaled-b"), "--json", str(out_path)])
    assert rc == 0
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == json.loads(capsys.readouterr().out)
    _assert_matches_golden(on_disk, _golden("scaled"))


def test_cli_compare_unreadable_artifact_exits_2(tmp_path, capsys):
    rc = cli.main(["vectors", "compare", str(tmp_path / "missing.safetensors"),
                   _fixture("identical-b")])
    assert rc == 2
    assert "vectors compare:" in capsys.readouterr().err


# --- the three-outcome exit contract (audit §2.4) ----------------------------
#
# The requirement is one sentence long and these tests are its whole
# enforcement: a CI script must be able to tell pass, diverged, and
# could-not-compare apart from exit codes alone. Before 2026-08-18 the ENVELOPE
# could not — `state_for_legacy_exit(2)` reported could-not-compare as
# `refused`/65, the same document a genuine cross-substrate divergence
# produces, so a harness pointed at a mistyped path looked exactly like a
# harness that had found real drift.
#
# Swift twin: `VectorsCompareExitContractTests`, over the byte-identical
# fixture copies. Two literals, one contract.

def _exit_codes(capsys, path_a: str, path_b: str) -> tuple:
    human = cli.main(["vectors", "compare", path_a, path_b])
    capsys.readouterr()
    machine = cli.main(["vectors", "compare", path_a, path_b, "--json"])
    document = json.loads(capsys.readouterr().out)
    return human, machine, document


def test_compare_exit_matrix_pass_is_zero_in_both_modes(capsys):
    human, machine, document = _exit_codes(
        capsys, _fixture("identical-a"), _fixture("identical-b"))
    assert (human, machine) == (0, 0)
    assert document["state"] == "ready"
    assert "error" not in document


def test_compare_exit_matrix_diverged_is_1_and_65_on_the_parity_gate(capsys):
    human, machine, document = _exit_codes(
        capsys, _fixture("orthogonal-a"), _fixture("orthogonal-b"))
    assert (human, machine) == (1, 65)
    assert document["state"] == "refused"
    assert document["error"]["code"] == "parityThreshold"
    # A gate-shaped refusal names its gate — the one outcome of the three that
    # is one.
    assert document["error"]["gate"] == "parityThreshold"
    # It COMPARED: the report is still the verb's product.
    assert document["result"]["report"]["pass"] is False


def test_compare_exit_matrix_missing_artifact_is_2_and_66_not_found(
        tmp_path, capsys):
    missing = str(tmp_path / "no-such-artifact.safetensors")
    human, machine, document = _exit_codes(
        capsys, missing, _fixture("identical-b"))
    assert (human, machine) == (2, 66)
    assert document["state"] == "notFound"
    assert document["error"]["code"] == "notFound"
    # NOT gate-shaped: nothing declined a well-formed request, the request
    # could not be answered.
    assert document["error"].get("gate") is None
    # The repair names BOTH operands and the shape of an artifact…
    repair = document["error"]["repairAction"]
    assert missing in repair and _fixture("identical-b") in repair
    assert ".safetensors" in repair and "sidecar" in repair
    # …and they are machine-readable, not only in prose.
    assert document["result"]["operandPaths"] == [missing,
                                                  _fixture("identical-b")]


def test_compare_exit_matrix_hidden_size_mismatch_is_could_not_compare(capsys):
    """The artifacts EXIST and cannot be compared. Same outcome as a missing
    file (2/66) because the caller's question has no answer either way — with a
    different repair, because "check your paths" would send them in a circle."""
    human, machine, document = _exit_codes(
        capsys, _fixture("narrow-a"), _fixture("identical-b"))
    assert (human, machine) == (2, 66)
    assert document["error"]["code"] == "notFound"
    assert "hidden-size mismatch" in document["error"]["reason"]
    assert "SAME model" in document["error"]["repairAction"]


def test_compare_exit_matrix_layer_count_mismatch_stays_a_report(capsys):
    """The CONTRAST that keeps the third outcome meaningful: those artifacts
    ARE comparable, the intersection is compared, and the run succeeds."""
    human, machine, _document = _exit_codes(
        capsys, _fixture("truncated-a"), _fixture("identical-b"))
    assert (human, machine) == (0, 0)


def test_cli_compare_usage(capsys):
    assert cli.main(["vectors"]) == 64
    assert cli.main(["vectors", "compare", "only-one-arg"]) == 64
