"""OptVec eval verb: the train/val firewall at the config boundary, the
artifact requirements, a toy end-to-end test-split eval, the dose arithmetic,
the all-position fluency guard, and the library-cosine null.

All CPU, all on a tiny in-memory Llama with a whitespace tokenizer — seconds,
no downloads. The artifacts are hand-built (``_write_optvec_artifact``) rather
than trained, so every expected number is arithmetic, not an optimizer's mood.
"""

import hashlib
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch

from steerlab_server.experiment import logprob, optvec_eval
from steerlab_server.experiment.generate import CellInjection
from steerlab_server.experiment.optvec_eval import (EvalDatasets, FileRef,
                                                    OptVecArtifactError,
                                                    OptVecEvalConfig,
                                                    OptVecEvalConfigError,
                                                    OptVecEvalDataError)
from steerlab_server.experiment.prompt_render import RAW_COMPLETION
from steerlab_server.steering import vector_store
from steerlab_server.steering.hooks import HookedModel
from steerlab_server.steering.injector import VectorInjector
from steerlab_server.steering.trainable_injector import TrainableVectorInjector

HIDDEN = 32
LAYERS = 4
OPTVEC_LAYER = 2
ALPHA_ABSOLUTE = 6.0


class _FakeTokenizer:
    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=True):
        ids = [self.vocab.setdefault(tok, len(self.vocab) + 1)
               for tok in text.split()]

        class R:
            input_ids = ids
        return R()


def _tiny_steered_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(11)
    config = LlamaConfig(
        hidden_size=HIDDEN, num_hidden_layers=LAYERS, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=256,
        max_position_embeddings=256)
    lm = LlamaForCausalLM(config).eval()
    return SimpleNamespace(
        model=lm, tokenizer=_FakeTokenizer(), hooked=HookedModel(lm),
        model_id="test/tiny", revision="rev-tiny", dtype="float32",
        device=torch.device("cpu"), context_window=256, hidden_size=HIDDEN,
        num_layers=LAYERS)


# ------------------------------------------------------------------ fixtures


def _write_rows(path, rows) -> FileRef:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return FileRef(path=str(path), sha256=digest)


def _write_texts(path, texts) -> FileRef:
    with open(path, "w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(text + "\n")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return FileRef(path=str(path), sha256=digest)


def _choice_rows(prefix, count, options=("alpha", "beta"), target=None):
    return [{"id": f"{prefix}-{i}",
             "prompt": f"case {prefix} number {i} the ruling is",
             "options": list(options), "target": target or options[0]}
            for i in range(count)]


def _unit(direction) -> list[float]:
    norm = math.sqrt(sum(x * x for x in direction))
    return [x / norm for x in direction]


def write_artifact(directory, name, *, per_layer, optvec=None,
                   model_id="test/tiny", revision="rev-tiny",
                   extraction_method="optvec"):
    """Write a vector artifact by hand, optionally with the additive
    ``optvec`` sidecar block the training driver stamps."""
    os.makedirs(directory, exist_ok=True)
    vectors = vector_store.ConceptVectors(per_layer=per_layer)
    sidecar = vector_store.SteeringVectorSidecar(
        modelID=model_id, concept=name, stimulusSetHash=f"optvec:{name}",
        layerCount=vectors.layer_count, hiddenSize=vectors.hidden_size,
        normsPerLayer=[vectors.norm(i) for i in range(vectors.layer_count)],
        extractionDate="2026-08-10T00:00:00Z", revision=revision,
        extractionMethod=extraction_method)
    vector_store.save(vectors, sidecar, str(directory), name)
    sidecar_path = os.path.join(str(directory), f"{name}.json")
    payload = json.load(open(sidecar_path))
    if optvec is not None:
        payload["optvec"] = optvec
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return os.path.join(str(directory), name)


def _optvec_direction(seed=3):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(HIDDEN, generator=generator)
    return [float(x) for x in raw / raw.norm() * ALPHA_ABSOLUTE]


def write_optvec_artifact(directory, name="toy", *, layer=OPTVEC_LAYER,
                          direction=None, alpha=ALPHA_ABSOLUTE,
                          layers=LAYERS):
    """A full-depth artifact with one nonzero layer — exactly the shape
    ``optvec_train._save_artifact`` writes."""
    row = list(direction) if direction is not None else _optvec_direction()
    per_layer = [[0.0] * HIDDEN for _ in range(layers)]
    per_layer[layer] = row
    return write_artifact(directory, name, per_layer=per_layer,
                          optvec={"layer": layer, "alphaAbsolute": alpha,
                                  "seed": 17, "claim": "sufficiency"})


def _eval_config(tmp_path, artifact, **overrides) -> OptVecEvalConfig:
    datasets = EvalDatasets(
        target_test=_write_rows(tmp_path / "target-test.jsonl",
                                _choice_rows("t", 4)),
        anchor_test=_write_rows(tmp_path / "anchor-test.jsonl",
                                _choice_rows("a", 3)),
        capability_eval=_write_rows(tmp_path / "cap-eval.jsonl",
                                    _choice_rows("c", 3)))
    kwargs = dict(vector_artifact=artifact, datasets=datasets,
                  neutral_texts=_write_texts(
                      tmp_path / "neutral.txt",
                      ["the weather today is mild and clear",
                       "a short neutral passage about nothing in particular"]),
                  alpha_multiples=(0.5, 1.0), null_samples=64, seed=5,
                  microbatch_size=2, prompt_mode=RAW_COMPLETION)
    kwargs.update(overrides)
    return OptVecEvalConfig(**kwargs)


# ---------------------------------------------------------- config strictness


def test_config_refuses_selection_split_datasets(tmp_path):
    """The firewall, at the config boundary: an eval that can name the train or
    val files can report the numbers selection was performed on."""
    ref = _write_rows(tmp_path / "t.jsonl", _choice_rows("t", 2)).to_dict()
    base = {"vectorArtifact": "runs/x/toy",
            "datasets": {"targetTest": ref}}
    assert OptVecEvalConfig.from_dict(base).vector_artifact == "runs/x/toy"

    for forbidden in ("targetTrain", "targetVal", "anchorTrain", "anchorVal",
                      "capabilityTrain"):
        with pytest.raises(OptVecEvalConfigError) as exc:
            OptVecEvalConfig.from_dict(
                {**base, "datasets": {"targetTest": ref, forbidden: ref}})
        message = str(exc.value)
        assert forbidden in message
        assert "TEST" in message and "selection" in message
        # And the refusal names the rule, not just the key.
        assert "gradients" in message or "selected on is not" in message

    # Also refused at the top level, where a stray key would otherwise be a
    # generic "unknown key".
    with pytest.raises(OptVecEvalConfigError) as exc:
        OptVecEvalConfig.from_dict({**base, "targetVal": ref})
    assert "targetVal" in str(exc.value) and "TEST" in str(exc.value)


def test_config_refuses_unknown_keys_and_requires_target_test(tmp_path):
    ref = _write_rows(tmp_path / "t.jsonl", _choice_rows("t", 2)).to_dict()
    base = {"vectorArtifact": "runs/x/toy", "datasets": {"targetTest": ref}}
    with pytest.raises(OptVecEvalConfigError) as exc:
        OptVecEvalConfig.from_dict({**base, "alphaMultipes": [1.0]})
    assert "alphaMultipes" in str(exc.value)
    with pytest.raises(OptVecEvalConfigError):
        OptVecEvalConfig.from_dict({"vectorArtifact": "runs/x/toy",
                                    "datasets": {"anchorTest": ref}})
    with pytest.raises(OptVecEvalConfigError):
        OptVecEvalConfig.from_dict({"datasets": {"targetTest": ref}})
    # α=0 is always the in-run baseline; declaring it as a dose is a config
    # error rather than a silently duplicated arm.
    with pytest.raises(OptVecEvalConfigError):
        OptVecEvalConfig.from_dict({**base, "alphaMultiples": [0.0, 1.0]})
    # Unknown keys inside a dataset ref refuse too (a typo'd 'sha' would
    # otherwise become an unpinned file).
    with pytest.raises(OptVecEvalConfigError):
        OptVecEvalConfig.from_dict(
            {**base, "datasets": {"targetTest": {**ref, "sha": "x"}}})


def test_dataset_and_neutral_hash_drift_refuse(tmp_path):
    path = tmp_path / "t.jsonl"
    ref = _write_rows(path, _choice_rows("t", 2))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_choice_rows("z", 1)[0]) + "\n")
    with pytest.raises(OptVecEvalDataError) as exc:
        optvec_eval.load_dataset(ref, "targetTest")
    assert "pins" in str(exc.value)

    text_path = tmp_path / "neutral.txt"
    text_ref = _write_texts(text_path, ["one two three"])
    with open(text_path, "a", encoding="utf-8") as handle:
        handle.write("four five six\n")
    with pytest.raises(OptVecEvalDataError) as exc:
        optvec_eval.load_neutral_texts(text_ref)
    assert "pins" in str(exc.value)


def test_parse_neutral_texts_reads_lines_and_jsonl(tmp_path):
    assert optvec_eval.parse_neutral_texts(
        "first line\n\nsecond line\n") == ["first line", "second line"]
    assert optvec_eval.parse_neutral_texts(
        '{"text": "from jsonl"}\n{"text": "and another"}\n') == [
            "from jsonl", "and another"]


# ------------------------------------------------------------- artifact gate


def test_artifact_without_an_optvec_block_refuses(tmp_path):
    per_layer = [[0.0] * HIDDEN for _ in range(LAYERS)]
    per_layer[OPTVEC_LAYER] = _optvec_direction()
    reference = write_artifact(tmp_path / "lib", "plain-caa",
                               per_layer=per_layer, optvec=None,
                               extraction_method="meanDifference")
    artifact = optvec_eval.load_artifact(reference)
    assert artifact.is_optvec is False
    with pytest.raises(OptVecArtifactError) as exc:
        optvec_eval.require_optvec(artifact)
    message = str(exc.value)
    assert "optvec" in message and "meanDifference" in message

    # A block that names a layer the artifact does not have, or a
    # non-positive α, is refused just as loudly.
    bad_layer = write_optvec_artifact(tmp_path / "bad", "bad-layer")
    payload = json.load(open(str(tmp_path / "bad" / "bad-layer.json")))
    payload["optvec"]["layer"] = 99
    json.dump(payload, open(str(tmp_path / "bad" / "bad-layer.json"), "w"))
    with pytest.raises(OptVecArtifactError):
        optvec_eval.require_optvec(optvec_eval.load_artifact(bad_layer))


def test_artifact_model_mismatch_refuses(tmp_path):
    reference = write_optvec_artifact(tmp_path / "v", "toy")
    target = optvec_eval.require_optvec(optvec_eval.load_artifact(reference))
    model = _tiny_steered_model()
    assert optvec_eval.verify_model(target, model)["revisionChecked"] is True
    other = _tiny_steered_model()
    other.model_id = "test/other"
    with pytest.raises(OptVecArtifactError) as exc:
        optvec_eval.verify_model(target, other)
    assert "test/other" in str(exc.value)
    drifted = _tiny_steered_model()
    drifted.revision = "rev-other"
    with pytest.raises(OptVecArtifactError) as exc:
        optvec_eval.verify_model(target, drifted)
    assert "rev-other" in str(exc.value)


# ------------------------------------------------------------ dose arithmetic


def test_cell_injection_scales_the_stored_vector_by_the_multiple(tmp_path):
    """The dose grid is stated in multiples of the vector's own α, and the
    deployed injector adds ``alpha · vector`` WITHOUT renormalizing. The stored
    row already has norm α, so ``alpha = multiple`` is the whole conversion —
    asserted through the injector's own arithmetic, not by reading the code."""
    reference = write_optvec_artifact(tmp_path / "v", "toy")
    target = optvec_eval.require_optvec(optvec_eval.load_artifact(reference))
    assert target.alpha_absolute == pytest.approx(ALPHA_ABSOLUTE)

    for multiple in (0.25, 1.0, 2.0):
        cell = optvec_eval.cell_injection(target, multiple)
        assert cell.layer == OPTVEC_LAYER
        injector = VectorInjector.single(layer=cell.layer, vector=cell.vector,
                                         alpha=cell.alpha)
        h = torch.zeros((1, 1, HIDDEN))
        applied = injector.apply(h, OPTVEC_LAYER, 0)
        norm = float(applied[0, -1].norm())
        assert norm == pytest.approx(multiple * ALPHA_ABSOLUTE, rel=1e-5)
        # And the all-position fluency injector carries the SAME vector.
        model = SimpleNamespace(device=torch.device("cpu"))
        fluency = optvec_eval.fluency_injector(model, target, multiple)
        assert torch.allclose(fluency.vector(), applied[0, -1], atol=1e-4)


# --------------------------------------------------------------- fluency guard


def _manual_mean_token_logprob(model, text, interventions):
    ids = list(model.tokenizer(text).input_ids)
    input_ids = torch.tensor([ids])
    mask = torch.ones_like(input_ids)
    model.hooked.reset_offsets()
    with torch.no_grad():
        with model.hooked.session(interventions):
            out = model.model(input_ids=input_ids, attention_mask=mask,
                              use_cache=False)
    logprobs = torch.log_softmax(out.logits.float(), dim=-1)[0, :-1]
    picked = logprobs.gather(1, input_ids[0, 1:].unsqueeze(1)).squeeze(1)
    return float(picked.mean())


def test_fluency_uses_every_position_not_just_the_last(tmp_path):
    """A single teacher-forced pass has no decode steps, so LAST-position
    injection changes only the logits that predict the token AFTER the text —
    none of which are scored. If the guard did what the behavioral arms do it
    would report the unsteered number under every dose."""
    model = _tiny_steered_model()
    reference = write_optvec_artifact(tmp_path / "v", "toy")
    target = optvec_eval.require_optvec(optvec_eval.load_artifact(reference))
    text = "one two three four five six seven eight"

    unsteered = _manual_mean_token_logprob(model, text, [])
    last_only = _manual_mean_token_logprob(
        model, text,
        [VectorInjector.single(layer=OPTVEC_LAYER, vector=target.direction,
                               alpha=1.0)])
    assert last_only == pytest.approx(unsteered, abs=1e-6)

    injector = optvec_eval.fluency_injector(model, target, 1.0)
    injector.set_batch(attention_mask=torch.ones(
        (1, len(model.tokenizer(text).input_ids)), dtype=torch.long))
    all_positions = _manual_mean_token_logprob(model, text, [injector])
    injector.clear_batch()
    assert abs(all_positions - unsteered) > 1e-3

    # The module's own batched readout agrees with both hand computations.
    baseline_rows = optvec_eval.mean_token_logprobs(model, [text])
    assert baseline_rows[0]["meanTokenLogprob"] == pytest.approx(unsteered,
                                                                 abs=1e-5)
    steered_rows = optvec_eval.mean_token_logprobs(
        model, [text], injector=optvec_eval.fluency_injector(model, target, 1.0))
    assert steered_rows[0]["meanTokenLogprob"] == pytest.approx(all_positions,
                                                                abs=1e-5)


def test_fluency_batching_matches_per_text_scoring(tmp_path):
    """Right-padded batches must not leak: a padded short text scores exactly
    what it scores alone."""
    model = _tiny_steered_model()
    texts = ["one two three", "four five six seven eight nine ten"]
    batched = optvec_eval.mean_token_logprobs(model, texts, microbatch_size=2)
    for i, text in enumerate(texts):
        alone = _manual_mean_token_logprob(model, text, [])
        assert batched[i]["meanTokenLogprob"] == pytest.approx(alone, abs=1e-4)
    # A one-token text has nothing to predict: recorded, never dropped.
    short = optvec_eval.mean_token_logprobs(model, ["solo"])
    assert short[0]["tokenCount"] == 1
    assert short[0]["meanTokenLogprob"] is None


# ------------------------------------------------------------ library cosines


def test_library_comparison_cosines_null_and_skips(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    direction = [0.0] * HIDDEN
    direction[0] = ALPHA_ABSOLUTE
    reference = write_optvec_artifact(tmp_path / "v", "toy",
                                      direction=direction)
    target = optvec_eval.require_optvec(optvec_eval.load_artifact(reference))

    same = [0.0] * HIDDEN
    same[0] = 3.0                                   # cosine 1.0
    tilted = [0.0] * HIDDEN
    tilted[0], tilted[1] = 0.6, 0.8                 # cosine 0.6
    orthogonal = [0.0] * HIDDEN
    orthogonal[1] = 2.0                             # cosine 0.0

    library = []
    for name, row in (("same", same), ("tilted", tilted),
                      ("orthogonal", orthogonal)):
        per_layer = [[0.0] * HIDDEN for _ in range(LAYERS)]
        per_layer[OPTVEC_LAYER] = row
        library.append(write_artifact(tmp_path / "lib", name,
                                      per_layer=per_layer,
                                      extraction_method="meanDifference"))
    # Two skip cases: an artifact too short to have the layer, and one whose
    # row at the layer is exactly zero (another optvec vector from a
    # different layer).
    short = write_artifact(tmp_path / "lib", "short",
                           per_layer=[[1.0] * HIDDEN, [1.0] * HIDDEN],
                           extraction_method="meanDifference")
    elsewhere = write_optvec_artifact(tmp_path / "lib", "other-layer", layer=1)
    library += [short, elsewhere]

    report = optvec_eval.library_comparison(target, library, samples=512,
                                            seed=7)
    by_name = {e["name"]: e for e in report["entries"]}
    assert by_name["same"]["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert by_name["tilted"]["cosine"] == pytest.approx(0.6, abs=1e-6)
    assert by_name["orthogonal"]["cosine"] == pytest.approx(0.0, abs=1e-6)
    assert report["comparedCount"] == 3
    assert {s["reference"] for s in report["skipped"]} == {short, elsewhere}
    assert all("no nonzero row at layer 2" in s["reason"]
               for s in report["skipped"])

    # Null: random directions at d=32 are near-orthogonal, and every reported
    # cosine carries its percentile in that distribution.
    assert report["null"]["samples"] == 512
    assert 0.0 < report["null"]["absCosineP50"] < 0.4
    assert report["null"]["absCosineP95"] >= report["null"]["absCosineP50"]
    assert by_name["same"]["nullPercentile"] == pytest.approx(100.0)
    assert by_name["orthogonal"]["nullPercentile"] < 10.0
    assert all(0.0 <= e["nullPercentile"] <= 100.0
               for e in report["entries"])
    # Top-k is ordered by |cosine|.
    assert [e["reference"] for e in report["topK"]][:2] == \
        [by_name["same"]["reference"], by_name["tilted"]["reference"]]
    assert len(report["topK"]) <= optvec_eval.LIBRARY_TOP_K

    # The null is seeded: the same seed reproduces the same percentiles.
    again = optvec_eval.library_comparison(target, library, samples=512, seed=7)
    assert again["null"] == report["null"]


def test_kl_divergence_direction_and_zero(tmp_path):
    p0 = [0.7, 0.2, 0.1]
    p = [0.1, 0.3, 0.6]
    forward = sum(a * (math.log(a) - math.log(b)) for a, b in zip(p0, p))
    assert optvec_eval.kl_divergence(p0, p) == pytest.approx(forward, rel=1e-9)
    assert optvec_eval.kl_divergence(p0, p) != pytest.approx(
        optvec_eval.kl_divergence(p, p0), rel=1e-3)
    assert optvec_eval.kl_divergence(p0, p0) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ end to end


def test_toy_eval_writes_a_complete_run_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    model = _tiny_steered_model()
    reference = write_optvec_artifact(tmp_path / "v", "toy")
    library = write_artifact(
        tmp_path / "lib", "companion",
        per_layer=[[0.0] * HIDDEN if i != OPTVEC_LAYER
                   else _optvec_direction(seed=9) for i in range(LAYERS)],
        extraction_method="meanDifference")
    config = _eval_config(tmp_path, reference, library_vector_paths=[library])
    result = optvec_eval.evaluate(config, model=model)
    run_dir = result["runDirectory"]
    assert os.path.basename(run_dir).endswith("optvec-eval-toy")

    readout = json.load(open(os.path.join(run_dir, optvec_eval.EVAL_JSON)))
    assert readout["split"] == "test" and readout["claim"] == "sufficiency"
    assert readout["optvec"] == {"layer": OPTVEC_LAYER,
                                 "alphaAbsolute": ALPHA_ABSOLUTE,
                                 "sidecarBlock": {"layer": OPTVEC_LAYER,
                                                  "alphaAbsolute": ALPHA_ABSOLUTE,
                                                  "seed": 17,
                                                  "claim": "sufficiency"}}
    # The artifact's bytes are re-hashed and recorded — both files.
    identity = readout["artifact"]
    for suffix, key in ((".safetensors", "tensorSHA256"),
                        (".json", "sidecarSHA256")):
        expected = hashlib.sha256(
            open(f"{reference}{suffix}", "rb").read()).hexdigest()
        assert identity[key] == expected

    # Dose grid: the in-run α=0 baseline plus every declared multiple.
    multiples = [entry["alphaMultiple"] for entry in readout["doseResponse"]]
    assert multiples == [0.0, 0.5, 1.0]
    assert readout["doseResponse"][0]["isBaseline"] is True
    for entry in readout["doseResponse"]:
        assert entry["alphaAbsolute"] == pytest.approx(
            entry["alphaMultiple"] * ALPHA_ABSOLUTE)
        assert entry["target"]["itemCount"] == 4
        assert entry["anchor"]["itemCount"] == 3
        assert entry["capability"]["itemCount"] == 3

    # At α=0 nothing has moved, by construction: the baseline is itself.
    zero = readout["doseResponse"][0]
    assert zero["target"]["meanLogOddsMovement"] == pytest.approx(0.0, abs=1e-9)
    assert zero["anchor"]["flipRate"] == 0.0
    assert zero["anchor"]["meanKLFromBaseline"] == pytest.approx(0.0, abs=1e-9)
    assert zero["capability"]["accuracyDelta"] == 0.0
    assert zero["fluency"]["deltaFromBaseline"] == pytest.approx(0.0, abs=1e-12)
    assert zero["fluency"]["positionMode"] == "all"
    assert zero["fluency"]["scoredTextCount"] == 2

    # Records: one per (item, dose), plus the neutral fluency rows.
    records = [json.loads(line) for line in
               open(os.path.join(run_dir, optvec_eval.EVAL_RECORDS))
               if line.strip()]
    assert len(records) == (4 + 3 + 3) * 3 + 2 * 3
    assert result["recordCount"] == len(records)
    baseline_rows = [r for r in records if r["alphaMultiple"] == 0.0]
    assert len(baseline_rows) == 10 + 2
    scored = [r for r in records if r["role"] != "neutral"]
    assert all(r["instrument"] == "answerTokenLogprob" for r in scored)
    assert all({"selected", "choiceProbability", "logOdds", "margin"}
               <= set(r) for r in scored)
    assert {r["role"] for r in records} == {"target", "anchor", "capability",
                                            "neutral"}

    # Hand-scoring: recompute the α=1 arm through the instrument directly and
    # check every reported rate against it.
    target = optvec_eval.require_optvec(optvec_eval.load_artifact(reference))
    injections = [optvec_eval.cell_injection(target, 1.0)]
    expected = {"target": [], "anchor": [], "capability": []}
    rows = optvec_eval.load_dataset(config.datasets.target_test, "targetTest")
    for role, ref in (("target", config.datasets.target_test),
                      ("anchor", config.datasets.anchor_test),
                      ("capability", config.datasets.capability_eval)):
        for row in optvec_eval.load_dataset(ref, role):
            base = logprob.score_options(
                model, row.prompt, list(row.options),
                prompt_mode=RAW_COMPLETION)
            steered = logprob.score_options(
                model, row.prompt, list(row.options), injections=injections,
                prompt_mode=RAW_COMPLETION)
            expected[role].append((row, base, steered))

    dose = next(e for e in readout["doseResponse"]
                if e["alphaMultiple"] == 1.0)
    flippable = [(r, b, s) for r, b, s in expected["target"]
                 if b.selected != r.target]
    assert dose["target"]["flippableItemCount"] == len(flippable)
    if flippable:
        assert dose["target"]["shiftRate"] == pytest.approx(
            sum(1.0 for r, _b, s in flippable if s.selected == r.target)
            / len(flippable))
    assert dose["target"]["meanLogOddsMovement"] == pytest.approx(
        sum(s.log_odds[r.target] - b.log_odds[r.target]
            for r, b, s in expected["target"]) / len(expected["target"]),
        rel=1e-6)
    assert dose["anchor"]["flipRate"] == pytest.approx(
        sum(1.0 for _r, b, s in expected["anchor"]
            if s.selected != b.selected) / len(expected["anchor"]))
    assert dose["anchor"]["meanKLFromBaseline"] == pytest.approx(
        sum(optvec_eval.kl_divergence(list(b.ordered_probabilities),
                                      list(s.ordered_probabilities))
            for _r, b, s in expected["anchor"]) / len(expected["anchor"]),
        rel=1e-6)
    assert dose["capability"]["accuracy"] == pytest.approx(
        sum(1.0 for r, _b, s in expected["capability"]
            if s.selected == r.target) / len(expected["capability"]))
    assert rows  # the target file really had rows

    # The canonical run config: closed key set, bespoke content in notes only.
    from tests.test_run_config import CONTRACT_KEYS
    config_json = json.load(open(os.path.join(run_dir, "config.json")))
    assert sorted(config_json.keys()) == CONTRACT_KEYS
    assert config_json["runType"] == "optvec-eval"
    assert config_json["modelID"] == "test/tiny"
    assert config_json["revision"] == "rev-tiny"
    notes = config_json["notes"]
    assert notes["stage"] == "complete"
    assert notes["split"] == "test" and notes["claim"] == "sufficiency"
    assert notes["datasets"]["counts"] == {
        "targetTest": 4, "anchorTest": 3, "capabilityEval": 3,
        "neutralTexts": 2}
    assert notes["artifact"]["tensorSHA256"] == identity["tensorSHA256"]
    assert [m["alphaMultiple"] for m in notes["metrics"]] == [0.0, 0.5, 1.0]
    assert notes["library"]["comparedCount"] == 1


def test_eval_refuses_an_item_id_in_two_roles(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    reference = write_optvec_artifact(tmp_path / "v", "toy")
    shared = _choice_rows("x", 2)
    config = _eval_config(
        tmp_path, reference,
        datasets=EvalDatasets(
            target_test=_write_rows(tmp_path / "tt.jsonl", shared),
            anchor_test=_write_rows(tmp_path / "at.jsonl", shared)))
    with pytest.raises(OptVecEvalDataError) as exc:
        optvec_eval.evaluate(config, model=_tiny_steered_model())
    assert "x-0" in str(exc.value)


def test_eval_without_neutral_texts_records_the_absent_guard(tmp_path,
                                                             monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    reference = write_optvec_artifact(tmp_path / "v", "toy")
    config = _eval_config(tmp_path, reference, neutral_texts=None,
                          alpha_multiples=(1.0,))
    result = optvec_eval.evaluate(config, model=_tiny_steered_model())
    readout = json.load(open(os.path.join(result["runDirectory"],
                                          optvec_eval.EVAL_JSON)))
    for entry in readout["doseResponse"]:
        assert entry["fluency"]["meanTokenLogprob"] is None
        assert "no fluency guard ran" in entry["fluency"]["note"]
    records = [json.loads(line) for line in
               open(os.path.join(result["runDirectory"],
                                 optvec_eval.EVAL_RECORDS)) if line.strip()]
    assert not [r for r in records if r["role"] == "neutral"]
