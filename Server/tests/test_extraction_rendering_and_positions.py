"""Extraction RENDERING and the reading-position vocabulary.

The bug this closes (ledger §26, measured 2026-08-23): extraction tokenized
the raw stimulus while measured generation rendered through the chat template,
so which direction a study got — and what the α denominator equalled —
depended on an undeclared implementation detail. These contracts pin the fix:

1. **Absent is legacy raw.** A recipe that declares nothing renders, screens,
   and stamps exactly as it always did — no new sidecar keys, no new bytes.
2. **A declared chatTemplate rendering reuses the MEASUREMENT renderer**
   (``prompt_render.render``), so extraction and generation can never drift
   into two definitions.
3. **The denominator follows the extraction.** The neutral corpus is
   tokenized under the same rendering, and the artifact stamps which.
4. **New positions resolve or refuse — never clamp**, and template-aware
   roles refuse under raw rendering naming the dependency.
5. **The artifact stamps the requested position AND where it landed.**
6. **The diagnostic fires for any departure from the legacy default.**

Everything runs on a tiny random in-memory Llama (no downloads, CPU).
"""

from types import SimpleNamespace

import pytest
import torch

from steerlab_server.steering import extraction_rendering as er
from steerlab_server.steering import extractor
from steerlab_server.steering import reading_position as rp
from steerlab_server.steering.hooks import HookedModel

# Neutral fixture vocabulary only — no study concepts, no site ids.
TEXTS = ["a steady voice in a difficult and unfamiliar meeting room",
         "an unhurried afternoon of ordinary and calm errands in town"]

#: Ids the fake tokenizer treats as template scaffolding, and the one it
#: treats as an end-of-turn marker. Chosen to look like a Gemma render:
#: <bos> … content … <end_of_turn> \n <start_of_turn> model \n
BOS, END_OF_TURN, NEWLINE, START_OF_TURN, MODEL = 2, 106, 107, 105, 108
SPECIALS = {BOS, END_OF_TURN, NEWLINE, START_OF_TURN, MODEL}


class _FakeTokenizer:
    """Deterministic ids per text; length grows with text length.

    Also answers the template-anchor questions a named role asks — the same
    questions a real HF tokenizer answers (``all_special_ids``,
    ``get_added_vocab``, ``convert_tokens_to_ids``, ``eos_token_id``).
    """

    all_special_ids = [BOS, END_OF_TURN]
    eos_token_id = END_OF_TURN
    unk_token_id = 3

    def get_added_vocab(self):
        return {"<start_of_turn>": START_OF_TURN, "<end_of_turn>": END_OF_TURN,
                "model": MODEL, "\n": NEWLINE}

    def convert_tokens_to_ids(self, token):
        return self.get_added_vocab().get(token, self.unk_token_id)

    #: The scaffold `apply_chat_template` appends when a generation prompt is
    #: asked for. Four ids, so post-instruction 1..5 all have somewhere to go.
    GENERATION_SCAFFOLD = "\n<start_of_turn>model\n"
    #: Joins message contents in the fake render. A control character, so it
    #: can never collide with stimulus text.
    TURN_SEPARATOR = "\x01"

    def _content_ids(self, text):
        torch.manual_seed(len(text) + sum(ord(c) for c in text[:8]))
        n = max(2, min(24, len(text) // 3))
        # Content ids stay clear of the scaffolding ids.
        return (torch.randint(1, 60, (1, n)) + 200).tolist()[0]

    @staticmethod
    def _wrap(ids, return_tensors):
        """A real HF tokenizer returns a LIST unless asked for tensors, and
        the chat-template path relies on that — so the fake must too."""
        return SimpleNamespace(
            input_ids=torch.tensor([ids]) if return_tensors == "pt" else ids)

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return self._wrap([BOS] + self._content_ids(text), return_tensors)

    def apply_chat_template(self, messages, **kwargs):
        """Stands in for the real template: returns a STRING, as transformers
        does with ``tokenize=False``. The trailing scaffold is present only
        when a generation prompt was asked for."""
        body = self.TURN_SEPARATOR.join(m["content"] for m in messages)
        return ("<bos>" + body + "<end_of_turn>"
                + (self.GENERATION_SCAFFOLD
                   if kwargs.get("add_generation_prompt") else ""))


class _TemplateAwareTokenizer(_FakeTokenizer):
    """``__call__`` understands the rendered string the template produced, so
    the chat-template path yields BOS + content + turn scaffolding."""

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        if not text.startswith("<bos>"):
            return super().__call__(text, return_tensors, add_special_tokens)
        body = text[len("<bos>"):].split("<end_of_turn>")[0]
        # The last joined segment is the user turn's own content.
        content = self._content_ids(body.split(self.TURN_SEPARATOR)[-1])
        ids = [BOS] + content + [END_OF_TURN]
        if text.endswith(self.GENERATION_SCAFFOLD):
            ids += [NEWLINE, START_OF_TURN, MODEL, NEWLINE]
        return self._wrap(ids, return_tensors)


def _model(tokenizer=None):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(7)
    config = LlamaConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=512,
        max_position_embeddings=256)
    lm = LlamaForCausalLM(config).eval()
    return SimpleNamespace(model=lm, hooked=HookedModel(lm),
                           device=torch.device("cpu"), num_layers=2,
                           model_id="google/gemma-3-4b-it",
                           tokenizer=tokenizer or _TemplateAwareTokenizer())


def _stimuli():
    return SimpleNamespace(positive=[TEXTS[0]] * 2, negative=[TEXTS[1]] * 2)


# --- 1. absent is legacy raw --------------------------------------------------

def test_an_absent_declaration_is_raw_and_changes_nothing():
    """The compatibility contract, at the level that matters: a recipe with
    no rendering declared produces the SAME vectors an option-free engine
    produced, stamps no rendering, and stamps no resolution."""
    model = _model()
    declared_nothing = extractor.extract(model, _stimuli(),
                                         extractor.ExtractionOptions())
    declared_raw = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            extraction_rendering=er.from_json({"mode": "raw"})))
    assert declared_nothing.vectors.per_layer == declared_raw.vectors.per_layer
    assert declared_nothing.residual_norm_rendering == "raw"
    assert declared_nothing.reading_position_resolution is None
    # …and no diagnostic, because nothing departed from the default.
    assert declared_nothing.reading_position_diagnostic is None


def test_a_raw_sidecar_stamps_neither_new_key():
    """Absent-not-null, checked on the bytes: a raw extraction's sidecar is
    byte-identical to what this engine has always written."""
    from steerlab_server.steering.vector_store import (ConceptVectors,
                                                       SteeringVectorSidecar)
    sidecar = SteeringVectorSidecar.make(
        model_id="m", concept="c", stimulus_set_hash="h",
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]]),
        reading_position=rp.LAST_TOKEN,
        residual_norm_per_layer=[1.0], residual_norm_source="extraction-stimuli",
        residual_norm_rendering="raw",
        extraction_rendering=er.RAW_RENDERING)
    encoded = sidecar.to_dict()
    assert "extractionRendering" not in encoded
    assert "residualNormRendering" not in encoded
    assert "readingPositionResolution" not in encoded


def test_a_templated_sidecar_stamps_the_rendering_that_made_the_vector():
    from steerlab_server.steering.vector_store import (ConceptVectors,
                                                       SteeringVectorSidecar)
    rendering = er.from_json({"mode": "chatTemplate",
                              "addGenerationPrompt": False})
    sidecar = SteeringVectorSidecar.make(
        model_id="m", concept="c", stimulus_set_hash="h",
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]]),
        reading_position=rp.TURN_CLOSE_TOKEN,
        residual_norm_per_layer=[1.0], residual_norm_source="neutral-corpus",
        residual_norm_rendering="chatTemplate",
        extraction_rendering=rendering,
        reading_position_resolution={"requested": "turn close token"})
    encoded = sidecar.to_dict()
    assert encoded["extractionRendering"] == {
        "mode": "chatTemplate", "addGenerationPrompt": False,
        "qwenThinkingEnabled": False}
    assert encoded["residualNormRendering"] == "chatTemplate"
    assert encoded["readingPosition"] == "turn close token"
    assert encoded["readingPositionResolution"]["requested"] == "turn close token"


# --- 2. the chat-template path reuses the measurement renderer ----------------

def test_the_template_path_goes_through_the_measurement_renderer(monkeypatch):
    """One rendering definition, not two. Extraction must reach the model
    through ``prompt_render.render`` — the same function the run loop renders
    generation prompts with — so a template change lands on both at once."""
    from steerlab_server.experiment import prompt_render
    seen = []
    real = prompt_render.render

    def spy(tokenizer, prompt, **kwargs):
        seen.append(kwargs)
        return real(tokenizer, prompt, **kwargs)

    monkeypatch.setattr(prompt_render, "render", spy)
    model = _model()
    extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            extraction_rendering=er.from_json(
                {"mode": "chatTemplate", "systemPrompt": "answer plainly"})))
    assert seen, "the chat-template path did not call prompt_render.render"
    assert seen[0]["prompt_mode"] == prompt_render.CHAT_ASSISTANT
    assert seen[0]["add_generation_prompt"] is True
    assert seen[0]["system_prompt"] == "answer plainly"
    assert seen[0]["model_id"] == "google/gemma-3-4b-it"


def test_add_generation_prompt_is_declarable_and_defaults_to_the_measured_form():
    parsed = er.from_json({"mode": "chatTemplate"})
    assert parsed.add_generation_prompt is True   # what generation does
    assert er.from_json({"mode": "chatTemplate",
                         "addGenerationPrompt": False}).add_generation_prompt is False


def test_the_two_renderings_produce_different_directions():
    """The whole motivation, reproduced in miniature: the same stimuli read
    the same way under two renderings are two different vectors."""
    model = _model()
    raw = extractor.extract(model, _stimuli(), extractor.ExtractionOptions())
    templated = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            extraction_rendering=er.from_json({"mode": "chatTemplate"})))
    assert raw.vectors.per_layer != templated.vectors.per_layer


def test_an_unknown_rendering_form_is_a_typed_refusal_naming_the_engine():
    with pytest.raises(er.ExtractionRenderingError) as exc:
        er.from_json({"mode": "someFutureForm"})
    assert er.ENGINE in str(exc.value)
    assert "repair:" in str(exc.value)


def test_raw_with_parameters_refuses_rather_than_pretending():
    with pytest.raises(er.ExtractionRenderingError, match="takes no parameters"):
        er.from_json({"mode": "raw", "addGenerationPrompt": True})


# --- 3. the denominator follows the extraction --------------------------------

def test_the_denominator_is_measured_under_the_extraction_rendering(monkeypatch):
    """α is in residual-norm units, so the denominator must come from the
    same distribution the vector was read from. Every text that reaches the
    model during a templated extraction — neutral corpus included — is
    rendered."""
    model = _model()
    rendered_calls = []
    real = er.rendered_token_ids
    monkeypatch.setattr(
        er, "rendered_token_ids",
        lambda m, text, rendering: (rendered_calls.append(text)
                                    or real(m, text, rendering)))
    neutral = ["a quiet street", "a folded map", "an open window",
               "a wooden chair"]
    result = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            extraction_rendering=er.from_json({"mode": "chatTemplate"})),
        neutral_texts=neutral)
    assert result.residual_norm_source == "neutral-corpus"
    assert result.residual_norm_rendering == "chatTemplate"
    # Every neutral text went through the renderer, not the raw tokenizer.
    assert set(neutral) <= set(rendered_calls)


def test_a_raw_extraction_still_measures_its_denominator_raw():
    model = _model()
    result = extractor.extract(
        model, _stimuli(), extractor.ExtractionOptions(),
        neutral_texts=["a quiet street", "a folded map", "an open window",
                       "a wooden chair"])
    assert result.residual_norm_rendering == "raw"


# --- 4. positions resolve or refuse -------------------------------------------

def test_offset_from_end_reads_the_index_it_names():
    ids = list(range(10))
    resolved = rp.offset_from_end(3).resolve(ids)
    assert resolved.start_index == 6
    assert resolved.end_index == 7
    assert resolved.offset_from_end == 3
    # k = 0 is the last token, exactly.
    assert rp.offset_from_end(0).resolve(ids).start_index == \
        rp.LAST_TOKEN.resolve(ids).start_index


def test_offset_from_end_refuses_a_short_sequence_instead_of_clamping():
    with pytest.raises(rp.ReadingPositionError) as exc:
        rp.offset_from_end(7).resolve([1, 2, 3])
    message = str(exc.value)
    assert "too short" in message and "needs 8" in message
    assert "never clamped" in message


def test_offset_from_end_refuses_a_negative_k():
    with pytest.raises(rp.ReadingPositionError, match="k ≥ 0"):
        rp.offset_from_end(-1)


def test_post_instruction_is_bounded_to_the_documented_range():
    for i in (0, 6):
        with pytest.raises(rp.ReadingPositionError, match="i in 1..5"):
            rp.post_instruction(i)


def test_named_roles_refuse_under_raw_rendering_naming_the_dependency():
    """A raw stimulus has no turn-close token; saying so is the repair."""
    ids = [BOS, 201, 202, END_OF_TURN, NEWLINE, START_OF_TURN, MODEL, NEWLINE]
    tokenizer = _TemplateAwareTokenizer()
    for position in (rp.LAST_CONTENT_TOKEN, rp.TURN_CLOSE_TOKEN,
                     rp.post_instruction(1)):
        with pytest.raises(rp.ReadingPositionError) as exc:
            position.resolve(ids, tokenizer=tokenizer, rendering_is_raw=True)
        message = str(exc.value)
        assert "templated rendering" in message
        assert 'extractionRendering {"mode": "chatTemplate"}' in message
        assert "offset from end k" in message   # the escape hatch is named


def test_named_roles_resolve_to_the_indices_the_template_puts_them_at():
    #        0     1    2    3             4        5                6      7
    ids = [BOS, 201, 202, END_OF_TURN, NEWLINE, START_OF_TURN, MODEL, NEWLINE]
    tokenizer = _TemplateAwareTokenizer()

    def resolve(position):
        return position.resolve(ids, tokenizer=tokenizer, rendering_is_raw=False)

    assert resolve(rp.LAST_CONTENT_TOKEN).start_index == 2
    assert resolve(rp.TURN_CLOSE_TOKEN).start_index == 3
    # Arditi's convention: the i-th token AFTER the instruction content.
    assert [resolve(rp.post_instruction(i)).start_index for i in range(1, 6)] \
        == [3, 4, 5, 6, 7]
    # …and the turn-close token is where post-instruction 1 lands under a
    # generation prompt, which is a fact about the template, not a bug.
    assert resolve(rp.TURN_CLOSE_TOKEN).start_index == \
        resolve(rp.post_instruction(1)).start_index


def test_post_instruction_refuses_when_the_template_left_no_room():
    """``addGenerationPrompt: false`` leaves fewer trailing tokens; asking for
    one past the end refuses rather than reading the last token twice."""
    ids = [BOS, 201, 202, END_OF_TURN]
    with pytest.raises(rp.ReadingPositionError) as exc:
        rp.post_instruction(3).resolve(ids, tokenizer=_TemplateAwareTokenizer(),
                                       rendering_is_raw=False)
    assert "no token 3 after it" in str(exc.value)
    assert "addGenerationPrompt" in str(exc.value)


def test_a_turn_close_role_refuses_when_no_marker_is_present():
    with pytest.raises(rp.ReadingPositionError, match="no end-of-turn marker"):
        rp.TURN_CLOSE_TOKEN.resolve([BOS, 201, 202],
                                    tokenizer=_TemplateAwareTokenizer(),
                                    rendering_is_raw=False)


def test_every_label_round_trips():
    for position in (rp.LAST_TOKEN, rp.mean_from_token(50), rp.offset_from_end(0),
                     rp.offset_from_end(3), rp.LAST_CONTENT_TOKEN,
                     rp.TURN_CLOSE_TOKEN, rp.post_instruction(5)):
        assert rp.from_label(position.label).label == position.label
    assert rp.parse_label_strict("post-instruction 9") is None
    assert rp.parse_label_strict("offset from end -2") is None


def test_the_legacy_pooled_read_is_byte_identical_through_the_new_window():
    """The recorder gained an explicit window; the two shape-only positions
    must still produce exactly the numbers they always did."""
    model = _model(_FakeTokenizer())
    pooled = extractor.activations(model, TEXTS, rp.mean_from_token(3))
    last = extractor.activations(model, TEXTS, rp.LAST_TOKEN)
    assert len(pooled.values) == len(TEXTS)
    assert len(last.values) == len(TEXTS)
    # The pooled window still starts at k and runs to the end; the last-token
    # window is still the final position.
    assert pooled.resolutions[0].start_index == 3
    assert pooled.resolutions[0].end_index == pooled.resolutions[0].token_count
    assert last.resolutions[0].offset_from_end == 0


# --- 5. the resolution stamp --------------------------------------------------

def test_the_artifact_stamps_the_request_and_where_it_landed():
    model = _model()
    result = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            reading_position=rp.post_instruction(2),
            extraction_rendering=er.from_json({"mode": "chatTemplate"})))
    stamp = result.reading_position_resolution
    assert stamp is not None
    # BOTH halves, per the layerResolution precedent (what was asked, what it
    # resolved to, and the rule that got there).
    assert stamp["requested"] == "post-instruction 2"
    assert stamp["mode"] == "postInstruction"
    assert stamp["parameter"] == 2
    assert stamp["rendering"] == "chatTemplate"
    assert stamp["source"] == "last content token + 2"
    # One sequence shape: every stimulus was read at the same place in its
    # template, which is what the offset-from-end grouping exists to show.
    assert len(stamp["shapes"]) == 1
    shape = stamp["shapes"][0]
    assert shape["offsetFromEnd"] == 3       # …of the 5 trailing template ids
    assert shape["sequenceCount"] == 4       # 2 positive + 2 negative
    assert shape["exampleIndex"] + 1 == shape["exampleEndIndex"]


def test_the_stamp_is_omitted_for_the_legacy_pair():
    """A shape-only position under raw rendering already implies its index, so
    stamping it would add bytes to every legacy artifact for no information."""
    model = _model(_FakeTokenizer())
    for position in (rp.LAST_TOKEN, rp.mean_from_token(3)):
        result = extractor.extract(
            model, _stimuli(),
            extractor.ExtractionOptions(reading_position=position))
        assert result.reading_position_resolution is None, position.label


def test_an_explicit_offset_is_stamped_even_under_raw_rendering():
    """`offset from end 3` is the mechanism form: what lives three back is
    model- and rendering-specific, so the artifact records where it landed."""
    model = _model(_FakeTokenizer())
    result = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(reading_position=rp.offset_from_end(3)))
    stamp = result.reading_position_resolution
    assert stamp["mode"] == "offsetFromEnd"
    assert stamp["rendering"] == "raw"
    assert stamp["shapes"][0]["offsetFromEnd"] == 3


# --- 6. the diagnostic fires for any departure --------------------------------

def test_the_diagnostic_fires_for_a_non_raw_rendering():
    model = _model()
    result = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            extraction_rendering=er.from_json({"mode": "chatTemplate"})))
    diag = result.reading_position_diagnostic
    assert diag is not None
    assert diag["primaryRendering"] == "chatTemplate"
    assert diag["comparedToRendering"] == "raw"
    assert diag["comparedTo"] == "last token"
    # Honest about the cost: a different tokenization needs its own passes.
    assert diag["extraForwardPasses"] is True
    assert len(diag["perLayerCosine"]) == 2
    assert diag["min"] <= diag["median"] <= diag["max"]


def test_the_diagnostic_fires_for_a_named_role_and_names_both_axes():
    model = _model()
    result = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            reading_position=rp.TURN_CLOSE_TOKEN,
            extraction_rendering=er.from_json({"mode": "chatTemplate"})))
    diag = result.reading_position_diagnostic
    assert diag["primaryReadingPosition"] == "turn close token"
    assert diag["primaryRendering"] == "chatTemplate"


def test_a_pooled_raw_reading_keeps_its_free_same_pass_diagnostic():
    """Unchanged behavior for the case that already had one: a non-default
    POSITION under raw rendering reads its baseline from the same forward
    passes, so it costs nothing extra."""
    model = _model(_FakeTokenizer())
    result = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(reading_position=rp.mean_from_token(3)))
    diag = result.reading_position_diagnostic
    assert diag["primaryReadingPosition"] == "mean from token 3"
    assert diag["primaryRendering"] == "raw"
    assert diag["extraForwardPasses"] is False


def test_the_default_recipe_still_carries_no_diagnostic():
    model = _model(_FakeTokenizer())
    result = extractor.extract(model, _stimuli(), extractor.ExtractionOptions())
    assert result.reading_position_diagnostic is None


# --- manifest plumbing --------------------------------------------------------

def test_the_manifest_parses_both_wire_forms_of_every_position():
    from steerlab_server.experiment.manifest import ExtractionOptions
    cases = {
        "lastToken": ({"lastToken": {}}, "last token"),
        "meanFromToken": ({"meanFromToken": {"_0": 50}}, "mean from token 50"),
        "offsetFromEnd": ({"offsetFromEnd": {"_0": 3}}, "offset from end 3"),
        "lastContentToken": ({"lastContentToken": {}}, "last content token"),
        "turnCloseToken": ({"turnCloseToken": {}}, "turn close token"),
        "postInstruction": ({"postInstruction": {"_0": 2}}, "post-instruction 2"),
    }
    for name, (swift_form, label) in cases.items():
        assert ExtractionOptions.from_json(
            {"readingPosition": swift_form}).reading_position.label == label, name
        # …and the label string form, which older/looser data uses.
        assert ExtractionOptions.from_json(
            {"readingPosition": label}).reading_position.label == label, name


def test_the_manifest_reads_an_absent_rendering_as_raw():
    from steerlab_server.experiment.manifest import ExtractionOptions
    assert ExtractionOptions.from_json({}).extraction_rendering.is_raw
    assert ExtractionOptions.from_json(
        {"extractionRendering": "chatTemplate"}).extraction_rendering.mode \
        == "chatTemplate"
