"""OptVec interpretation (WP7): the per-vector reading stages, their skip
records, and the family summary — including the ``no-library-match`` category,
which the plan makes a first-class RESULT rather than a gap.

All CPU, all on a tiny in-memory Llama with a whitespace tokenizer — seconds,
no downloads. Artifacts are hand-built (reusing ``test_optvec_eval``'s writers)
so every expected number is arithmetic.

**SAE stage:** ``sae_lens`` is not installed in this environment and the
existing Gemma Scope machinery has no test-scale fixture (``analyze`` always
calls ``SAE.from_pretrained``, i.e. a download). So the live SAE path is tested
through the module's two seams (``_analyze_gemma_scope`` / ``_load_sae``) with
a fake SAE, plus the real-world "not installed" case, which must record a
reason rather than lose the other five stages.
"""

import json
import math
import os
from types import SimpleNamespace

import pytest
import torch

from steerlab_server.experiment import optvec_interpret
from steerlab_server.experiment.optvec_eval import (OptVecArtifactError,
                                                    OptVecEvalDataError)
from steerlab_server.experiment.prompt_render import RAW_COMPLETION
from steerlab_server.steering.hooks import HookedModel
from steerlab_server.steering.model_loader import SteeredModel
from tests.test_optvec_eval import (ALPHA_ABSOLUTE, HIDDEN, LAYERS,
                                    OPTVEC_LAYER, write_artifact,
                                    write_optvec_artifact)

FamilyConfig = optvec_interpret.FamilySummaryConfig
InterpretConfig = optvec_interpret.OptVecInterpretConfig
ConfigError = optvec_interpret.OptVecInterpretConfigError


# ---------------------------------------------------------------- tiny model


class _Tokenizer:
    """Whitespace tokenizer with the two extras generation needs: a decode
    (the streamer's) and pad/eos ids."""

    pad_token_id = 0
    eos_token_id = None

    def __init__(self):
        self.vocab = {}
        self.inverse = {}

    def __call__(self, text, add_special_tokens=True):
        ids = []
        for token in text.split():
            if token not in self.vocab:
                index = len(self.vocab) + 5
                self.vocab[token] = index
                self.inverse[index] = token
            ids.append(self.vocab[token])
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, **kwargs):
        try:
            values = [int(i) for i in ids]
        except TypeError:
            values = [int(ids)]
        return " ".join(self.inverse.get(i, f"<{i}>") for i in values)

    def convert_ids_to_tokens(self, ids):
        return [self.inverse.get(int(i), f"<{int(i)}>") for i in ids]


def tiny_model(seed: int = 11) -> SteeredModel:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        hidden_size=HIDDEN, num_hidden_layers=LAYERS, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=256,
        max_position_embeddings=256)
    lm = LlamaForCausalLM(config).eval()
    return SteeredModel(model=lm, tokenizer=_Tokenizer(), hooked=HookedModel(lm),
                        model_id="test/tiny", revision="rev-tiny",
                        dtype="float32")


def _write_prompts(tmp_path, prompts):
    from steerlab_server.experiment.optvec_eval import FileRef
    import hashlib

    path = tmp_path / "probes.txt"
    with open(path, "w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(prompt + "\n")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return FileRef(path=str(path), sha256=digest)


PROBES = ["the court holds that the defendant", "in this appeal the question"]


def _config(artifact, tmp_path, **overrides) -> InterpretConfig:
    kwargs = dict(vector_artifact=artifact,
                  probe_prompts=_write_prompts(tmp_path, PROBES),
                  alpha_multiples=(1.0,), max_tokens=4,
                  prompt_mode=RAW_COMPLETION,
                  logit_lens=optvec_interpret.LogitLensConfig(top_k=6),
                  library=optvec_interpret.LibraryConfig(null_samples=64),
                  seed=5)
    kwargs.update(overrides)
    return InterpretConfig(**kwargs)


# --------------------------------------------------------- config strictness


def test_config_refuses_unknown_keys_everywhere(tmp_path):
    base = {"vectorArtifact": "runs/x/toy"}
    assert InterpretConfig.from_dict(base).vector_artifact == "runs/x/toy"

    with pytest.raises(ConfigError) as exc:
        InterpretConfig.from_dict({**base, "alphaMultipes": [1.0]})
    assert "alphaMultipes" in str(exc.value)

    with pytest.raises(ConfigError):
        InterpretConfig.from_dict({})

    for block, payload, bad in (
            ("logitLens", {"topk": 5}, "topk"),
            ("sae", {"release": "r", "saeID": "s", "topk": 3}, "topk"),
            ("selfExplanation", {"enable": True}, "enable"),
            ("library", {"vectorPath": []}, "vectorPath"),
            ("probePrompts", {"path": "p", "sha": "x"}, "sha")):
        with pytest.raises(ValueError) as exc:
            InterpretConfig.from_dict({**base, block: payload})
        assert bad in str(exc.value)

    # α=0 is always the in-run baseline; declaring it as a dose is an error.
    with pytest.raises(ConfigError):
        InterpretConfig.from_dict({**base, "alphaMultiples": [0.0, 1.0]})
    with pytest.raises(ConfigError):
        InterpretConfig.from_dict({**base, "alphaMultiples": []})
    with pytest.raises(ConfigError):
        InterpretConfig.from_dict({**base, "maxTokens": 0})
    # A cosine is never reported without a null, so a null of zero draws is
    # not configurable.
    with pytest.raises(ConfigError):
        InterpretConfig.from_dict({**base, "library": {"nullSamples": 0}})
    # A self-explanation at α=0 would not be a self-explanation.
    with pytest.raises(ConfigError):
        InterpretConfig.from_dict({**base,
                                   "selfExplanation": {"alphaMultiple": 0.0}})
    with pytest.raises(ConfigError):
        InterpretConfig.from_dict({**base, "sae": {"release": "r"}})


def test_config_defaults_match_the_plan(tmp_path):
    config = InterpretConfig.from_dict({"vectorArtifact": "runs/x/toy"})
    assert config.alpha_multiples == (0.5, 1.0, 2.0)
    assert config.max_tokens == 128
    assert config.logit_lens.top_k == 50
    assert config.library.null_samples == 1000
    assert config.self_explanation.enabled is True
    assert config.self_explanation.alpha_multiple == 1.0
    assert "one sentence" in config.self_explanation.template
    assert config.sae is None and config.jlens_lens_id is None


def test_probe_prompt_hash_drift_refuses(tmp_path):
    ref = _write_prompts(tmp_path, PROBES)
    with open(ref.path, "a", encoding="utf-8") as handle:
        handle.write("a third probe appeared\n")
    with pytest.raises(OptVecEvalDataError) as exc:
        optvec_interpret.load_probe_prompts(ref)
    assert "pins" in str(exc.value)
    # And the interpret-specific type is what callers here catch.
    assert isinstance(exc.value, optvec_interpret.OptVecInterpretDataError)


def test_artifact_without_an_optvec_block_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    per_layer = [[0.0] * HIDDEN for _ in range(LAYERS)]
    per_layer[OPTVEC_LAYER] = [1.0] * HIDDEN
    plain = write_artifact(tmp_path / "lib", "plain-caa", per_layer=per_layer,
                           optvec=None, extraction_method="meanDifference")
    with pytest.raises(OptVecArtifactError) as exc:
        optvec_interpret.interpret(_config(plain, tmp_path),
                                   model=tiny_model())
    assert "optvec" in str(exc.value) and "meanDifference" in str(exc.value)
    # And no run directory was created for a run that could never have run:
    # the artifact gate is checked before anything is written.
    runs = tmp_path / "runs"
    assert not os.path.isdir(runs) or os.listdir(runs) == []


# ------------------------------------------------------------- logit lens


def test_logit_lens_inverts_on_a_head_row_direction(tmp_path, monkeypatch):
    """A real property, not a smoke assertion.

    Take v = the unembedding row of token t (scaled to the artifact's α). The
    lens applies the model's own final norm — odd-symmetric, RMSNorm — then a
    bias-free head, so:

    * ``+v`` must promote t above every other token (its own row has the
      largest self-inner-product), and
    * every logit under ``−v`` is the exact negation of the logit under
      ``+v``, so the promoted list of one sign IS the suppressed list of the
      other, token for token.

    If the readout ever skipped the final norm, or read the head alone, or
    negated the wrong thing, this ordering breaks.
    """
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = tiny_model()
    head = model.model.lm_head
    assert head.bias is None
    token_id = 37
    row = head.weight[token_id].detach().float()
    direction = [float(x) for x in row / row.norm() * ALPHA_ABSOLUTE]
    reference = write_optvec_artifact(tmp_path / "v", "toy",
                                      direction=direction)
    from steerlab_server.experiment.optvec_eval import (load_artifact,
                                                        require_optvec)
    target = require_optvec(load_artifact(reference))

    stage = optvec_interpret.logit_lens_stage(model, target, top_k=8)
    assert stage["layer"] == OPTVEC_LAYER and stage["topK"] == 8
    positive, negative = stage["positive"], stage["negative"]

    # +v promotes its own head row above everything else.
    assert positive["promoted"][0]["tokenID"] == token_id
    # −v suppresses exactly that token hardest, and the two signs are mirror
    # images of each other token-for-token.
    assert negative["suppressed"][0]["tokenID"] == token_id
    assert [t["tokenID"] for t in negative["promoted"]] == \
        [t["tokenID"] for t in positive["suppressed"]]
    assert [t["tokenID"] for t in negative["suppressed"]] == \
        [t["tokenID"] for t in positive["promoted"]]
    for a, b in zip(positive["promoted"], negative["suppressed"]):
        assert a["logit"] == pytest.approx(-b["logit"], rel=1e-6)
    # Pieces travel with ids.
    assert all("piece" in t and "tokenID" in t
               for t in positive["promoted"] + positive["suppressed"])
    assert positive["promotedConcentration"] is not None


def test_concentration_is_the_stated_formula():
    assert optvec_interpret.concentration([]) is None
    assert optvec_interpret.concentration([0.0, 0.0]) is None
    # All the mass in one entry.
    assert optvec_interpret.concentration([10.0, 0.0, 0.0]) == \
        pytest.approx(1.0)
    # Ten equal entries: the top 5 carry half.
    assert optvec_interpret.concentration([1.0] * 10) == pytest.approx(0.5)
    # Magnitude, not sign.
    assert optvec_interpret.concentration([-4.0, 1.0, 1.0, 1.0, 1.0, 2.0]) == \
        pytest.approx(9.0 / 10.0)
    assert "top 5" in optvec_interpret.CONCENTRATION_FORMULA


# ------------------------------------------------------------- generations


def test_generations_arm_the_injection_and_alpha_zero_is_unsteered(
        tmp_path, monkeypatch):
    """Two things at once, because they are the same claim from both sides:
    the α=0 arm must be byte-identical to an unsteered generation, and the
    steered arms must actually reach the deployed injection path."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = tiny_model()
    reference = write_optvec_artifact(tmp_path / "v", "toy")
    from steerlab_server.experiment.optvec_eval import (load_artifact,
                                                        require_optvec)
    target = require_optvec(load_artifact(reference))

    seen: list = []
    real_generate = optvec_interpret.generate

    def spy(model_arg, prompt, **kwargs):
        seen.append((prompt, kwargs.get("injections")))
        return real_generate(model_arg, prompt, **kwargs)

    monkeypatch.setattr(optvec_interpret, "generate", spy)
    config = _config(reference, tmp_path, alpha_multiples=(100.0,))
    records, summary = optvec_interpret.probe_generations(
        model, target, PROBES, config)

    # One record per prompt × (baseline + each multiple).
    assert len(records) == len(PROBES) * 2 == summary["recordCount"]
    assert [d["alphaMultiple"] for d in summary["doses"]] == [0.0, 100.0]

    # The baseline arm asked for NO injections; the dosed arm asked for the
    # artifact's own layer at exactly the multiple.
    baseline_calls = [i for _p, i in seen if i is None]
    steered_calls = [i for _p, i in seen if i is not None]
    assert len(baseline_calls) == len(steered_calls) == len(PROBES)
    for cells in steered_calls:
        assert len(cells) == 1
        assert cells[0].layer == OPTVEC_LAYER
        assert cells[0].alpha == pytest.approx(100.0)
        assert cells[0].mode == "add"
        # The stored row already carries norm α, so alpha=multiple injects
        # multiple×α — the dose grid's own arithmetic (optvec_eval).
        norm = math.sqrt(sum(x * x for x in cells[0].vector))
        assert norm == pytest.approx(ALPHA_ABSOLUTE, rel=1e-5)

    # α=0 reproduces the unsteered generation exactly.
    for record in [r for r in records if r["alphaMultiple"] == 0.0]:
        assert record["interventionState"] == "baseline"
        assert record["output"] == real_generate(
            model, record["prompt"], model_id="test/tiny", max_tokens=4,
            temperature=0.0, injections=None, prompt_mode=RAW_COMPLETION)

    # …and a large dose actually moves the model (it reached the residual
    # stream, not just the call signature).
    dosed = next(d for d in summary["doses"] if d["alphaMultiple"] == 100.0)
    assert dosed["changedFromBaselineCount"] >= 1
    assert summary["doses"][0]["changedFromBaselineCount"] == 0

    # Standard generation metadata is stamped on every record.
    for record in records:
        assert record["modelID"] == "test/tiny"
        assert record["modelRevision"] == "rev-tiny"
        assert record["promptMode"] == RAW_COMPLETION
        assert record["layer"] == OPTVEC_LAYER
        assert record["temperature"] == 0.0 and record["doSample"] is False
        assert record["artifactTensorSHA256"] == target.artifact.tensor_sha256
        assert record["promptID"].startswith("probe-")
        assert record["wordCount"] >= 0 and 0.0 <= record["distinct2"] <= 1.0


# --------------------------------------------------------- self-explanation


def test_self_explanation_is_stamped_suggestive(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = tiny_model()
    reference = write_optvec_artifact(tmp_path / "v", "toy")
    from steerlab_server.experiment.optvec_eval import (load_artifact,
                                                        require_optvec)
    target = require_optvec(load_artifact(reference))
    config = _config(reference, tmp_path)

    block = optvec_interpret.self_explanation_stage(model, target, config)
    assert block["suggestive"] is True
    assert block["note"] == "self-report under injection — not evidence"
    assert block["template"] == config.self_explanation.template
    assert block["alphaAbsolute"] == pytest.approx(ALPHA_ABSOLUTE)
    assert "output" in block and "baselineOutput" in block

    # Disabled: still stamped suggestive, and it says why it is absent.
    disabled = optvec_interpret.SelfExplanationConfig(enabled=False)
    off = optvec_interpret.self_explanation_stage(
        model, target, _config(reference, tmp_path,
                               self_explanation=disabled))
    assert off["suggestive"] is True and "skipped" in off


# ------------------------------------------------------- SAE and J-lens


class _FakeSAE:
    """The two SAE surfaces the stage touches: decoder rows (through the
    existing Gemma Scope analysis) and ``encode``."""

    def __init__(self, decoder):
        self.W_dec = decoder

    def encode(self, vector):
        return self.W_dec @ vector


def test_sae_stage_calls_the_gemma_scope_machinery(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    reference = write_optvec_artifact(tmp_path / "v", "toy")
    from steerlab_server.experiment import gemma_scope
    from steerlab_server.experiment.optvec_eval import (load_artifact,
                                                        require_optvec)
    target = require_optvec(load_artifact(reference))

    decoder = torch.zeros((8, HIDDEN))
    decoder[3] = torch.tensor(target.direction)          # cosine 1.0
    decoder[5] = -torch.tensor(target.direction)         # cosine -1.0
    for i in (0, 1, 2, 4, 6, 7):
        decoder[i, i] = 1.0

    calls = {}

    def fake_analyze(directory, name, *, layer, release, sae_id, top_k):
        calls.update(directory=directory, name=name, layer=layer,
                     release=release, sae_id=sae_id, top_k=top_k)
        unit = torch.tensor(target.direction)
        unit = unit / unit.norm()
        scores = (decoder / decoder.norm(dim=1, keepdim=True)) @ unit
        order = torch.topk(scores.abs(), k=top_k).indices.tolist()
        rows = [gemma_scope.FeatureRow(feature=int(i), cosine=float(scores[i]),
                                       sparsity=None) for i in order]
        return gemma_scope.GemmaScopeReport(
            release=release, sae_id=sae_id, layer=layer,
            decoder_shape=[8, HIDDEN], top_positive=rows, top_negative=rows,
            top_absolute=rows, vector_norm=ALPHA_ABSOLUTE,
            vector_concept="toy", artifact_sidecar={})

    monkeypatch.setattr(optvec_interpret, "_analyze_gemma_scope", fake_analyze)
    monkeypatch.setattr(optvec_interpret, "_load_sae",
                        lambda release, sae_id: _FakeSAE(decoder))

    stage = optvec_interpret.sae_stage(
        target, optvec_interpret.SAEConfig(release="gemma-scope-2-27b-it-res",
                                           sae_id="layer_31_width_16k",
                                           top_k=4))
    # It asked the EXISTING machinery, at the artifact's own layer, with the
    # artifact's own directory/name.
    assert calls["layer"] == OPTVEC_LAYER and calls["name"] == "toy"
    assert calls["release"] == "gemma-scope-2-27b-it-res"
    assert stage["convention"] == gemma_scope.IMPORT_CONVENTION
    features = [row["feature"] for row in stage["topByDecoderCosine"]]
    assert features[:2] == [3, 5] or features[:2] == [5, 3]
    assert stage["topByDecoderCosine"][0]["cosine"] == pytest.approx(
        1.0, abs=1e-5) or stage["topByDecoderCosine"][0]["cosine"] == \
        pytest.approx(-1.0, abs=1e-5)
    assert stage["decoderCosineConcentration"] is not None
    # The encoder half ranks the same aligned feature first.
    encoder = stage["encoderActivation"]
    assert encoder["topFeatures"][0]["feature"] == 3
    assert encoder["topFeatures"][0]["activation"] == pytest.approx(
        ALPHA_ABSOLUTE ** 2, rel=1e-4)
    assert encoder["encodedVectorNorm"] == pytest.approx(ALPHA_ABSOLUTE,
                                                         rel=1e-5)


def test_sae_and_jlens_stages_record_why_they_did_not_run(tmp_path,
                                                          monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    reference = write_optvec_artifact(tmp_path / "v", "toy")
    from steerlab_server.experiment.optvec_eval import (load_artifact,
                                                        require_optvec)
    target = require_optvec(load_artifact(reference))

    unconfigured = optvec_interpret.sae_stage(target, None)
    assert "sae" in unconfigured["skipped"] and "attempted" not in unconfigured

    # Configured but unavailable (no sae_lens here, no such release anywhere):
    # recorded WITH the reason, never fatal, never silently absent.
    attempted = optvec_interpret.sae_stage(
        target, optvec_interpret.SAEConfig(release="nope", sae_id="nope"))
    assert attempted["attempted"] is True and attempted["skipped"]
    assert attempted["release"] == "nope"

    assert "jlensLensID" in optvec_interpret.jlens_stage(target, None)["skipped"]
    missing = optvec_interpret.jlens_stage(target, "no-such-lens",
                                           str(tmp_path))
    assert missing["attempted"] is True and missing["lensID"] == "no-such-lens"
    assert "J-lens support unavailable" in missing["skipped"]


# ------------------------------------------------------------- end to end


def test_interpret_writes_a_complete_run_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    model = tiny_model()
    direction = [0.0] * HIDDEN
    direction[0] = ALPHA_ABSOLUTE
    reference = write_optvec_artifact(tmp_path / "v", "toy",
                                      direction=direction)
    aligned = [0.0] * HIDDEN
    aligned[0] = 2.0
    per_layer = [[0.0] * HIDDEN for _ in range(LAYERS)]
    per_layer[OPTVEC_LAYER] = aligned
    library = write_artifact(tmp_path / "lib", "anger", per_layer=per_layer,
                             extraction_method="meanDifference")

    config = _config(reference, tmp_path,
                     library=optvec_interpret.LibraryConfig(
                         vector_paths=[library], null_samples=128))
    result = optvec_interpret.interpret(config, model=model,
                                        root=str(tmp_path))
    run_dir = result["runDirectory"]
    assert os.path.basename(run_dir).endswith("optvec-interpret-toy")

    readout = json.load(open(os.path.join(run_dir,
                                          optvec_interpret.INTERPRET_JSON)))
    assert readout["claim"] == "sufficiency"
    assert readout["runType"] == "optvec-interpret"
    assert readout["artifact"]["tensorSHA256"] == \
        result["artifact"]["tensorSHA256"]
    assert readout["optvec"]["layer"] == OPTVEC_LAYER
    stages = readout["stages"]
    assert set(stages) == {"logitLens", "sae", "jlensSupport", "generations",
                           "selfExplanation", "library"}
    # Ran.
    assert "skipped" not in stages["logitLens"]
    assert "skipped" not in stages["generations"]
    assert stages["selfExplanation"]["suggestive"] is True
    # Not configured — recorded with a reason, never missing.
    assert "skipped" in stages["sae"] and "skipped" in stages["jlensSupport"]
    # Library: the aligned companion is a cosine of exactly 1 with its null.
    entries = {e["name"]: e for e in stages["library"]["entries"]}
    assert entries["anger"]["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert entries["anger"]["nullPercentile"] == pytest.approx(100.0)

    # Generations: one record per prompt × (baseline + dose).
    records = [json.loads(line) for line in
               open(os.path.join(run_dir, optvec_interpret.GENERATIONS))
               if line.strip()]
    assert len(records) == len(PROBES) * 2
    assert {r["condition"] for r in records} == {"alpha-0", "alpha-1"}

    # The canonical run config: closed key set, bespoke content in notes only.
    from tests.test_run_config import CONTRACT_KEYS
    config_json = json.load(open(os.path.join(run_dir, "config.json")))
    assert sorted(config_json.keys()) == CONTRACT_KEYS
    assert config_json["runType"] == "optvec-interpret"
    assert config_json["modelID"] == "test/tiny"
    assert config_json["revision"] == "rev-tiny"
    notes = config_json["notes"]
    assert notes["stage"] == "complete" and notes["claim"] == "sufficiency"
    assert notes["stages"]["logitLens"] == "ran"
    assert notes["stages"]["sae"] == "skipped"
    assert set(notes["skipped"]) == {"sae", "jlensSupport"}
    assert notes["promptCount"] == len(PROBES)


def test_interpret_without_probe_prompts_records_the_absent_battery(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    reference = write_optvec_artifact(tmp_path / "v", "toy")
    config = _config(reference, tmp_path, probe_prompts=None)
    result = optvec_interpret.interpret(config, model=tiny_model(),
                                        root=str(tmp_path))
    run_dir = result["runDirectory"]
    readout = json.load(open(os.path.join(run_dir,
                                          optvec_interpret.INTERPRET_JSON)))
    assert "probePrompts" in readout["stages"]["generations"]["skipped"]
    assert not os.path.exists(os.path.join(run_dir,
                                           optvec_interpret.GENERATIONS))


# ------------------------------------------------------------- condition_of


@pytest.mark.parametrize("block,expected", [
    (None, "unknown"),
    ({}, "unknown"),
    ({"objective": {}}, "unknown"),
    # S0 wins over everything: it IS S2's optimization on permuted labels.
    ({"objective": {"shuffleTargetLabels": True, "lambdaAnchor": 1.0,
                    "lambdaCap": 1.0, "lambdaOrth": 0.0}}, "s0"),
    ({"shuffleTargetLabels": True, "lambdaAnchor": 0.0, "lambdaCap": 0.0},
     "s0"),
    # S1: shift only.
    ({"objective": {"lambdaAnchor": 0.0, "lambdaCap": 0.0,
                    "lambdaOrth": 0.0, "shuffleTargetLabels": False}}, "s1"),
    ({"objective": {"lambdaAnchor": 0, "lambdaCap": 0}}, "s1"),
    # S3: the multiplicity condition carries the orthogonality penalty.
    ({"objective": {"lambdaAnchor": 1.0, "lambdaCap": 1.0,
                    "lambdaOrth": 0.5}}, "s3"),
    # S2: the composite primary.
    ({"objective": {"lambdaAnchor": 1.0, "lambdaCap": 1.0,
                    "lambdaOrth": 0.0}}, "s2"),
    ({"objective": {"lambdaAnchor": 1.0, "lambdaCap": 0.0,
                    "lambdaOrth": 0.0}}, "s2"),
    ({"objective": {"lambdaAnchor": 0.0, "lambdaCap": 2.0}}, "s2"),
    # Top-level λ's (hand-built blocks) read the same.
    ({"lambdaAnchor": 1.0, "lambdaCap": 1.0, "lambdaOrth": 3.0}, "s3"),
    # Non-numeric / non-boolean junk is not a condition.
    ({"objective": {"lambdaAnchor": "1.0"}}, "unknown"),
    ({"objective": {"shuffleTargetLabels": "yes"}}, "unknown"),
])
def test_condition_of(block, expected):
    assert optvec_interpret.condition_of(block) == expected


def test_condition_of_reads_a_real_training_block(tmp_path):
    """The shape the training driver actually stamps (optvec_train
    ``_objective_notes`` nested under ``optvec.objective``)."""
    reference = write_optvec_artifact(tmp_path / "v", "toy")
    sidecar_path = f"{reference}.json"
    payload = json.load(open(sidecar_path))
    payload["optvec"]["objective"] = {
        "lambdaShift": 1.0, "lambdaAnchor": 1.0, "lambdaCap": 1.0,
        "lambdaOrth": 0.0, "shuffleTargetLabels": False,
        "claim": "sufficiency"}
    json.dump(payload, open(sidecar_path, "w"))
    from steerlab_server.experiment.optvec_eval import load_artifact
    assert optvec_interpret.condition_of(
        load_artifact(reference).optvec) == "s2"


# ------------------------------------------------------------------ family


def _interpret_fixture(tmp_path, name, *, condition, library, promoted,
                       sae=None):
    """A hand-built interpret.json — the family reads readings, so its inputs
    are readings and every expected number below is arithmetic."""
    run_dir = tmp_path / "runs" / f"20260810T00-optvec-interpret-{name}"
    os.makedirs(run_dir, exist_ok=True)
    stages = {
        "logitLens": {"positive": {"promoted": promoted, "suppressed": []}},
        "library": {"entries": library, "topK": [], "skipped": []},
        "generations": {"skipped": "none"},
        "selfExplanation": dict(optvec_interpret.SUGGESTIVE_STAMP),
        "jlensSupport": {"skipped": "none"},
        "sae": ({"topByDecoderCosine": sae} if sae is not None
                else {"skipped": "none"}),
    }
    payload = {
        "schemaVersion": 1, "runType": optvec_interpret.RUN_TYPE,
        "runID": os.path.basename(run_dir), "claim": "sufficiency",
        "condition": condition,
        "artifact": {"reference": f"runs/x/{name}", "name": name,
                     "tensorSHA256": f"sha-{name}"},
        "optvec": {"layer": OPTVEC_LAYER, "alphaAbsolute": ALPHA_ABSOLUTE},
        "model": {"artifactModelID": "test/tiny"},
        "stages": stages,
    }
    with open(run_dir / optvec_interpret.INTERPRET_JSON, "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle)
    return str(run_dir)


def _entry(reference, cosine, percentile):
    return {"reference": reference, "name": reference.split("/")[-1],
            "concept": reference.split("/")[-1], "cosine": cosine,
            "absCosine": abs(cosine), "nullPercentile": percentile}


def _tokens(logits):
    return [{"tokenID": i, "piece": f"t{i}", "logit": v}
            for i, v in enumerate(logits)]


def test_family_summary_tables_including_the_alien_category(tmp_path,
                                                            monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    # Two solutions match the same library concept, one matches another, one
    # matches NOTHING above its null — the alien row.
    a = _interpret_fixture(
        tmp_path, "a", condition="s2",
        library=[_entry("lib/anger", 0.8, 100.0),
                 _entry("lib/fear", 0.2, 60.0)],
        promoted=_tokens([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        sae=[{"feature": 7, "cosine": 0.9}, {"feature": 8, "cosine": 0.1}])
    b = _interpret_fixture(
        tmp_path, "b", condition="s2",
        library=[_entry("lib/anger", -0.6, 99.5)],
        promoted=_tokens([1.0] * 10),
        sae=[{"feature": 7, "cosine": 0.5}, {"feature": 9, "cosine": 0.5}])
    c = _interpret_fixture(
        tmp_path, "c", condition="s1",
        library=[_entry("lib/fear", 0.15, 50.0)],
        promoted=_tokens([1.0] * 10))
    d = _interpret_fixture(
        tmp_path, "d", condition="s1",
        library=[_entry("lib/sympathy", 0.4, 99.9)],
        promoted=_tokens([4.0, 1.0, 1.0, 1.0, 1.0, 2.0]))

    result = optvec_interpret.family_summary(
        FamilyConfig(interpret_runs=[a, b, c, d], name="fam"),
        root=str(tmp_path))
    run_dir = result["runDirectory"]
    assert os.path.basename(run_dir).endswith("optvec-family-fam")
    report = json.load(open(os.path.join(run_dir,
                                         optvec_interpret.FAMILY_JSON)))

    assert report["count"] == 4
    by_run = {row["artifact"]["name"]: row for row in report["solutions"]}
    assert by_run["a"]["libraryMatchCategory"] == "lib/anger"
    assert by_run["a"]["topLibraryMatch"]["cosine"] == 0.8
    # |cosine| picks the match, sign and all — the direction's polarity is a
    # fact about the pair, not a reason to drop the row.
    assert by_run["b"]["libraryMatchCategory"] == "lib/anger"
    assert by_run["b"]["topLibraryMatch"]["cosine"] == -0.6
    assert by_run["c"]["libraryMatchCategory"] == \
        optvec_interpret.NO_LIBRARY_MATCH
    assert by_run["c"]["topLibraryMatch"]["clearsNull"] is False
    assert by_run["d"]["libraryMatchCategory"] == "lib/sympathy"

    # The distribution table: the alien category is an ordinary row.
    distribution = {row["match"]: row for row
                    in report["libraryMatchDistribution"]}
    assert distribution["lib/anger"]["count"] == 2
    assert distribution["lib/sympathy"]["count"] == 1
    assert distribution[optvec_interpret.NO_LIBRARY_MATCH]["count"] == 1
    assert distribution[optvec_interpret.NO_LIBRARY_MATCH][
        "isNoMatchCategory"] is True
    assert report["distinctLibraryMatches"] == 2
    assert report["noLibraryMatchCount"] == 1
    assert report["conditions"] == {"s1": 2, "s2": 2}
    assert "alien" in report["matchRule"]

    # Per-solution readouts.
    assert by_run["a"]["topSAEFeatures"] == [7, 8]
    assert by_run["c"]["topSAEFeatures"] is None
    assert by_run["a"]["logitLensTopTokens"] == ["t0", "t1", "t2", "t3", "t4"]
    assert by_run["a"]["logitLensConcentration"] == pytest.approx(1.0)
    assert by_run["b"]["logitLensConcentration"] == pytest.approx(0.5)

    # S1-vs-S2 contrast, hand-computed.
    contrast = report["contrastS1S2"]
    assert contrast["s1"]["count"] == 2 and contrast["s2"]["count"] == 2
    assert contrast["s2"]["meanAbsTopLibraryCosine"] == pytest.approx(
        (0.8 + 0.6) / 2)
    assert contrast["s1"]["meanAbsTopLibraryCosine"] == pytest.approx(
        (0.15 + 0.4) / 2)
    assert contrast["s2"]["meanLogitLensConcentration"] == pytest.approx(
        (1.0 + 0.5) / 2)
    assert contrast["s1"]["meanLogitLensConcentration"] == pytest.approx(
        (0.5 + 0.9) / 2)
    # SAE concentration only where SAE data exists (both s2 rows here).
    assert contrast["s2"]["meanSAEConcentration"] == pytest.approx(1.0)
    assert contrast["s1"]["meanSAEConcentration"] is None
    assert contrast["deltaS1MinusS2"]["meanAbsTopLibraryCosine"] == \
        pytest.approx((0.15 + 0.4) / 2 - (0.8 + 0.6) / 2)
    assert contrast["deltaS1MinusS2"]["meanSAEConcentration"] is None
    assert contrast["s1"]["noLibraryMatchCount"] == 1

    # Closed-key run config for the family run type too.
    from tests.test_run_config import CONTRACT_KEYS
    config_json = json.load(open(os.path.join(run_dir, "config.json")))
    assert sorted(config_json.keys()) == CONTRACT_KEYS
    assert config_json["runType"] == "optvec-family"
    assert config_json["modelID"] == "test/tiny"
    assert config_json["notes"]["noLibraryMatchCount"] == 1
    assert config_json["notes"]["conditions"] == {"s1": 2, "s2": 2}


def test_family_separates_matched_nothing_from_compared_nothing(tmp_path,
                                                                monkeypatch):
    """The alien finding is "compared against a library and matched none of
    it". A solution compared against an EMPTY library also lands in the
    category (there is nowhere else for it), so the family counts those
    separately — otherwise an unconfigured library reads as a result."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    unmatched = _interpret_fixture(tmp_path, "unmatched", condition="s2",
                                   library=[_entry("lib/anger", 0.1, 40.0)],
                                   promoted=_tokens([1.0]))
    uncompared = _interpret_fixture(tmp_path, "uncompared", condition="s2",
                                    library=[], promoted=_tokens([1.0]))
    result = optvec_interpret.family_summary(
        FamilyConfig(interpret_runs=[unmatched, uncompared]),
        root=str(tmp_path))
    assert result["noLibraryMatchCount"] == 2
    assert result["noLibraryComparedCount"] == 1
    rows = {row["artifact"]["name"]: row for row in result["solutions"]}
    assert rows["unmatched"]["libraryComparedCount"] == 1
    assert rows["uncompared"]["libraryComparedCount"] == 0
    assert rows["uncompared"]["topLibraryMatch"] is None
    assert "noLibraryComparedCount" in result["matchRule"]


def test_family_refuses_one_run_and_non_interpret_inputs(tmp_path,
                                                         monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    a = _interpret_fixture(tmp_path, "a", condition="s2",
                           library=[_entry("lib/anger", 0.8, 100.0)],
                           promoted=_tokens([1.0]))
    with pytest.raises(optvec_interpret.OptVecFamilyError):
        FamilyConfig(interpret_runs=[a])
    with pytest.raises(optvec_interpret.OptVecFamilyError) as exc:
        optvec_interpret.family_summary(
            FamilyConfig(interpret_runs=[a, str(tmp_path / "nowhere")]),
            root=str(tmp_path))
    assert "interpret.json" in str(exc.value)

    other = tmp_path / "runs" / "not-an-interpret"
    os.makedirs(other, exist_ok=True)
    with open(other / optvec_interpret.INTERPRET_JSON, "w") as handle:
        json.dump({"runType": "optvec-eval"}, handle)
    with pytest.raises(optvec_interpret.OptVecFamilyError) as exc:
        optvec_interpret.family_summary(
            FamilyConfig(interpret_runs=[a, str(other)]), root=str(tmp_path))
    assert "optvec-eval" in str(exc.value)


def test_family_contrast_is_skipped_with_a_reason_when_a_side_is_missing(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    runs = [_interpret_fixture(tmp_path, name, condition="s2",
                               library=[_entry("lib/anger", 0.8, 100.0)],
                               promoted=_tokens([1.0]))
            for name in ("a", "b")]
    result = optvec_interpret.family_summary(
        FamilyConfig(interpret_runs=runs), root=str(tmp_path))
    contrast = result["contrastS1S2"]
    assert "skipped" in contrast and "s2" in contrast["skipped"]
    # A family that matched nothing anywhere is still a family with a table.
    assert result["distinctLibraryMatches"] == 1
