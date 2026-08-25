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
#: The other role word, for the turn-structured tokenizer below.
USER_ROLE = 109


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


class _TurnStructuredTokenizer(_FakeTokenizer):
    """A fake that renders and tokenizes PER-TURN markers the way Gemma 3's
    template does::

        <bos><start_of_turn>user\\n{content}<end_of_turn>\\n<start_of_turn>model\\n

    The flatter ``_TemplateAwareTokenizer`` above is enough for the roles that
    only need a turn CLOSE. The content mask and the assistant voice need the
    turn's OPENING and its role tag to exist as real tokens, because excluding
    them is the whole point.
    """

    #: The role word each turn opens with, as its own token — exactly the
    #: shape `content_indices` skips (`<start_of_turn>` `role` `\\n`).
    ROLE_IDS = {"user": USER_ROLE, "model": MODEL}
    _PIECES = {BOS: "<bos>", START_OF_TURN: "<start_of_turn>",
               END_OF_TURN: "<end_of_turn>", NEWLINE: "\n",
               MODEL: "model", USER_ROLE: "user"}

    def get_added_vocab(self):
        return {"<bos>": BOS, "<start_of_turn>": START_OF_TURN,
                "<end_of_turn>": END_OF_TURN, "model": MODEL,
                "user": USER_ROLE, "\n": NEWLINE}

    def convert_ids_to_tokens(self, token_id):
        return self._PIECES.get(token_id, f"tok{token_id}")

    def apply_chat_template(self, messages, **kwargs):
        text = "<bos>"
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            text += (f"<start_of_turn>{role}\n{message['content']}"
                     "<end_of_turn>\n")
        if kwargs.get("add_generation_prompt"):
            text += "<start_of_turn>model\n"
        return text

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids: list[int] = []
        rest = text
        if rest.startswith("<bos>"):
            ids.append(BOS)
            rest = rest[len("<bos>"):]
        elif add_special_tokens:
            # The tokenizer's own single BOS — what the assistant-voice
            # construction relies on after it subtracts the template's.
            ids.append(BOS)
        while rest:
            if rest.startswith("<start_of_turn>"):
                rest = rest[len("<start_of_turn>"):]
                role, _, rest = rest.partition("\n")
                ids += [START_OF_TURN,
                        self.ROLE_IDS.get(role, self.unk_token_id), NEWLINE]
                continue
            if rest.startswith("<end_of_turn>"):
                ids.append(END_OF_TURN)
                rest = rest[len("<end_of_turn>"):]
                continue
            if rest.startswith("\n"):
                ids.append(NEWLINE)
                rest = rest[1:]
                continue
            cuts = [i for i in (rest.find("<start_of_turn>"),
                                rest.find("<end_of_turn>"), rest.find("\n"))
                    if i >= 0]
            cut = min(cuts) if cuts else len(rest)
            ids += self._content_ids(rest[:cut])
            rest = rest[cut:]
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
        # The repair names the FLAG that declares it — until 2026-08-24 this
        # sentence asked for a declaration no command could write.
        assert f'{er.DECLARATION_FLAG} \'{{"mode": "chatTemplate"}}\'' in message
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


# --------------------------------------------------------------------------
# The inert-declaration advisory (2026-08-24 field finding)
#
# Two experiments differing ONLY in the Qwen thinking flag ran overnight on
# 2026-08-23 and produced byte-identical vectors across 30 concepts: raw
# extraction never sees the chat template, so the flag could not reach it.
# Two GPU jobs measured nothing, and the comparison read as a null result.
# The instrument's answer is a loud line at extract time — an advisory, never
# a gate, because the declaration is legal and only the silence was not.
# --------------------------------------------------------------------------


def _manifest_with(rendering, *, thinking=False, prompt_mode="chatAssistant",
                   system=None, pinned=False):
    concept = SimpleNamespace(
        is_pinned_artifact=pinned,
        options=SimpleNamespace(extraction_rendering=rendering))
    return SimpleNamespace(
        concepts=[concept], qwen_thinking_enabled=thinking,
        prompt_mode=prompt_mode, system_prompt=system)


def test_raw_extraction_says_which_declarations_cannot_reach_it():
    from steerlab_server.experiment import tasks

    lines = []
    tasks._advise_inert_declarations(
        _manifest_with(er.RAW_RENDERING, thinking=True), lines.append)
    assert len(lines) == 1, lines
    line = lines[0]
    assert line.startswith("ADVISORY:")
    # It names the inert declarations, the consequence, and the repair.
    assert "qwenThinkingEnabled true" in line
    assert "promptMode chatAssistant" in line
    assert "byte-identical" in line
    # The repair names the FLAG that makes them effective — a repair naming
    # only the manifest key was one no command could carry out.
    assert f'{er.DECLARATION_FLAG} \'{{"mode": "chatTemplate"}}\'' in line


def test_a_chat_template_extraction_emits_no_inert_declaration_advisory():
    from steerlab_server.experiment import tasks

    lines = []
    tasks._advise_inert_declarations(
        _manifest_with(er.ExtractionRendering(mode=er.CHAT_TEMPLATE),
                       thinking=True, system="You are a careful assistant."),
        lines.append)
    assert lines == []
    # …and the pure helper agrees: the declarations DO reach a rendered
    # extraction, so there is nothing to warn about.
    assert extractor.inert_declaration_advisory(
        er.ExtractionRendering(mode=er.CHAT_TEMPLATE),
        qwen_thinking_enabled=True) is None


def test_an_ordinary_raw_recipe_stays_silent():
    from steerlab_server.experiment import tasks

    lines = []
    tasks._advise_inert_declarations(_manifest_with(er.RAW_RENDERING), lines.append)
    assert lines == [], "an advisory channel that always speaks is unread"
    assert extractor.inert_declaration_advisory(er.RAW_RENDERING) is None


def test_a_study_system_prompt_also_fires_it():
    lines = extractor.inert_declaration_advisory(
        er.RAW_RENDERING, system_prompt="You are a careful assistant.",
        prompt_mode="rawCompletion")
    assert lines is not None
    assert "systemPrompt" in lines
    # rawCompletion is not a chat context, so it is not listed as inert.
    assert "promptMode" not in lines


# --------------------------------------------------------------------------
# 7. THE WRITER: `--extraction-rendering`, parsed and refused at declaration
#
# The option shipped 2026-08-24 with every CONSUMER live — recipe identity,
# the denominator, the template-aware positions, the sidecar stamps — and no
# WRITER at all: no flag, no route field, no store parameter. The refusals
# above ask for a declaration, and until now they could name no command to
# write it with. `parse_declaration` is that command's parser, and the rules
# below are its contract (Swift twin: `ExtractionRendering.declared(_:)`).
# --------------------------------------------------------------------------


def test_the_declaration_flag_is_the_cross_engine_spelling():
    assert er.DECLARATION_FLAG == "--extraction-rendering"
    assert er.MODES == ("raw", "chatTemplate")


def test_every_spelling_of_raw_canonicalizes_to_absent():
    """THE HASH CONTRACT. An explicit raw is the legacy rendering said out
    loud, so it must write exactly what saying nothing writes."""
    for spelling in (None, "raw", '"raw"', '{"mode": "raw"}',
                     '  {"mode":"raw"}  ', {"mode": "raw"}):
        assert er.parse_declaration(spelling) is None, spelling


def test_a_chat_template_declaration_resolves_its_defaults_explicitly():
    bare = er.parse_declaration('{"mode": "chatTemplate"}')
    assert bare.mode == er.CHAT_TEMPLATE
    assert bare.add_generation_prompt is True
    assert bare.qwen_thinking_enabled is False
    assert bare.system_prompt is None
    # The shell-friendly bare word is the same declaration…
    assert er.parse_declaration("chatTemplate") == bare
    # …and so is the already-decoded object a route body arrives in.
    assert er.parse_declaration({"mode": "chatTemplate"}) == bare
    # The block a manifest stores writes the resolved defaults out.
    assert bare.to_dict() == {"mode": "chatTemplate",
                              "addGenerationPrompt": True,
                              "qwenThinkingEnabled": False}


def test_every_rendering_parameter_survives_the_declaration():
    full = er.parse_declaration(
        '{"mode":"chatTemplate","addGenerationPrompt":false,'
        '"qwenThinkingEnabled":true,"systemPrompt":"be brief"}')
    assert full.add_generation_prompt is False
    assert full.qwen_thinking_enabled is True
    assert full.system_prompt == "be brief"
    # This engine SUPPORTS addGenerationPrompt false — it is the engine a
    # swift-mlx study is redirected to for exactly this form.
    assert full.to_dict()["addGenerationPrompt"] is False


@pytest.mark.parametrize("declaration", [
    "",                                        # no value
    '{"mode": "chatTemplate"',                 # unterminated JSON
    '{"mode": "templated"}',                   # out-of-vocabulary mode
    "someFutureForm",                          # …and its bare-word form
    '{"mode":"raw","systemPrompt":"x"}',       # raw takes no parameters
    '{"mode":"chatTemplate","addGenerationPrompt":1}',
    '{"mode":"chatTemplate","qwenThinkingEnabled":"yes"}',
    '{"mode":"chatTemplate","systemPrompt":7}',
    ['{"mode": "raw"}'],                       # not an object or a string
])
def test_a_malformed_declaration_is_a_typed_refusal_carrying_a_repair(declaration):
    """Never a fallback to raw: a silent fallback is precisely the ambiguity
    the option exists to close."""
    with pytest.raises(er.ExtractionRenderingError) as exc:
        er.parse_declaration(declaration)
    assert "repair" in str(exc.value).lower()


def test_the_unknown_mode_refusal_names_the_engine_and_the_vocabulary():
    with pytest.raises(er.ExtractionRenderingError) as exc:
        er.parse_declaration('{"mode": "templated"}')
    message = str(exc.value)
    assert "templated" in message
    assert er.ENGINE in message
    assert "raw" in message and "chatTemplate" in message


# --------------------------------------------------------------------------
# 8. THE VOICE: rendering the stimulus as the model's OWN OUTPUT
#
# The rendering×position grid asks a question the user voice cannot: is the
# direction about text the model READS or text the model PRODUCES? The voice
# is the fork. Absent ≡ "user" ≡ every byte this engine ever wrote, and the
# assistant voice renders the template's own assistant-turn markers around the
# stimulus with NO preceding user content.
# --------------------------------------------------------------------------


def _voiced_model():
    return _model(_TurnStructuredTokenizer())


def test_the_voice_vocabulary_is_the_cross_engine_spelling():
    assert er.VOICES == ("user", "assistant")
    assert er.VOICE_USER == "user" and er.VOICE_ASSISTANT == "assistant"


def test_an_absent_voice_is_the_user_voice_and_writes_nothing():
    """THE HASH CONTRACT, one level down. An explicit "user" is the legacy
    voice said out loud, so it must produce exactly what saying nothing
    produces — at the stamp, at the identity fragment, and at the parse."""
    absent = er.parse_declaration('{"mode": "chatTemplate"}')
    explicit = er.parse_declaration('{"mode": "chatTemplate", "voice": "user"}')
    assert absent == explicit
    assert absent.voice == er.VOICE_USER
    assert "voice" not in absent.to_dict()
    assert "voice" not in explicit.to_dict()
    assert er.canonical_identity_fragment(explicit) == \
        er.canonical_identity_fragment(absent)
    assert "voice" not in er.canonical_identity_fragment(absent)


def test_the_assistant_voice_stamps_itself_and_nothing_it_cannot_honor():
    declared = er.parse_declaration(
        '{"mode": "chatTemplate", "voice": "assistant"}')
    assert declared.is_assistant_voice
    # addGenerationPrompt is refused at declaration time, so the artifact must
    # not claim a choice nobody made.
    assert declared.to_dict() == {"mode": "chatTemplate",
                                  "qwenThinkingEnabled": False,
                                  "voice": "assistant"}
    assert declared.label == "chatTemplate (voice=assistant)"


def test_the_assistant_voice_is_the_only_voice_that_reaches_the_identity():
    user = er.canonical_identity_fragment(
        er.parse_declaration('{"mode": "chatTemplate"}'))
    assistant = er.canonical_identity_fragment(
        er.parse_declaration('{"mode": "chatTemplate", "voice": "assistant"}'))
    assert assistant["voice"] == "assistant"
    # …and it is ADDITIVE: every other key keeps its meaning, so a reader (and
    # a diff) sees one field change rather than a new shape.
    assert set(assistant) - set(user) == {"voice"}
    assert assistant["mode"] == user["mode"]


def test_the_assistant_voice_renders_the_turn_alone_and_the_probe_never_lands():
    """The exact rendered form, on a Gemma-shaped template:
    ``<bos><start_of_turn>model\\n{stimulus}<end_of_turn>\\n`` — the assistant
    turn's own markers, no user turn, one BOS."""
    from steerlab_server.experiment import prompt_render

    tokenizer = _TurnStructuredTokenizer()
    rendered = prompt_render.render_assistant_turn(
        tokenizer, TEXTS[0], model_id="google/gemma-3-4b-it")
    assert rendered.text == (f"<start_of_turn>model\n{TEXTS[0]}"
                             "<end_of_turn>\n")
    # The probe user turn is the scaffolding of the CONSTRUCTION, never of the
    # render: injected user text would confound the very contrast the voice
    # isolates.
    assert prompt_render.ASSISTANT_VOICE_PROBE_TURN not in rendered.text
    ids = rendered.input_ids
    # One BOS (the template's went with the subtracted prefix), then the
    # assistant turn's opening, its role tag, the content, and its close.
    assert ids[:4] == [BOS, START_OF_TURN, MODEL, NEWLINE]
    assert ids[-2:] == [END_OF_TURN, NEWLINE]
    assert ids.count(BOS) == 1
    # …and no user role token anywhere.
    assert USER_ROLE not in ids


def test_the_two_voices_are_two_directions():
    """The whole point of the fork, in miniature: reading the same stimuli at
    the same position under the two voices is two different vectors."""
    model = _voiced_model()
    user = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            extraction_rendering=er.from_json({"mode": "chatTemplate"})))
    assistant = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            extraction_rendering=er.from_json(
                {"mode": "chatTemplate", "voice": "assistant"})))
    assert user.vectors.per_layer != assistant.vectors.per_layer
    assert assistant.residual_norm_rendering == "chatTemplate"


def test_the_denominator_follows_the_assistant_voice_too():
    """α in norm units divides by a number from the distribution the vector
    was read from — the voice is part of that distribution."""
    model = _voiced_model()
    neutral = ["a quiet street", "a folded map", "an open window",
               "a wooden chair"]
    user = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            extraction_rendering=er.from_json({"mode": "chatTemplate"})),
        neutral_texts=neutral)
    assistant = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            extraction_rendering=er.from_json(
                {"mode": "chatTemplate", "voice": "assistant"})),
        neutral_texts=neutral)
    assert user.residual_norm_per_layer != assistant.residual_norm_per_layer


def test_named_roles_resolve_against_the_assistant_turns_content():
    """The roles keep meaning what they say: under the assistant voice the
    content IS the model's own output, and one uniform render is one shapes
    row."""
    model = _voiced_model()
    result = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            reading_position=rp.LAST_CONTENT_TOKEN,
            extraction_rendering=er.from_json(
                {"mode": "chatTemplate", "voice": "assistant"})))
    stamp = result.reading_position_resolution
    assert stamp["requested"] == "last content token"
    assert stamp["rendering"] == "chatTemplate"
    assert len(stamp["shapes"]) == 1
    # `<end_of_turn> \n` follow the content, so the content boundary sits two
    # back from the end of the assistant-voice render.
    assert stamp["shapes"][0]["offsetFromEnd"] == 2


def test_add_generation_prompt_under_the_assistant_voice_is_a_typed_refusal():
    """It is not a silent ignore: under this voice the stimulus IS the
    generation, so the key would reach nothing at all."""
    with pytest.raises(er.ExtractionRenderingError) as exc:
        er.parse_declaration('{"mode":"chatTemplate","voice":"assistant",'
                             '"addGenerationPrompt":true}')
    assert str(exc.value) == er.ASSISTANT_VOICE_GENERATION_PROMPT_REASON
    assert "repair:" in str(exc.value)
    # false is refused identically — the refusal is about the KEY, not a value.
    with pytest.raises(er.ExtractionRenderingError):
        er.parse_declaration('{"mode":"chatTemplate","voice":"assistant",'
                             '"addGenerationPrompt":false}')


def test_a_system_prompt_under_the_assistant_voice_is_a_typed_refusal():
    with pytest.raises(er.ExtractionRenderingError) as exc:
        er.parse_declaration('{"mode":"chatTemplate","voice":"assistant",'
                             '"systemPrompt":"be brief"}')
    assert str(exc.value) == er.ASSISTANT_VOICE_SYSTEM_PROMPT_REASON
    assert "confound" in str(exc.value)


def test_an_unknown_voice_is_a_typed_refusal_naming_the_engine():
    with pytest.raises(er.ExtractionRenderingError) as exc:
        er.parse_declaration('{"mode":"chatTemplate","voice":"system"}')
    message = str(exc.value)
    assert "system" in message and er.ENGINE in message
    assert "user" in message and "assistant" in message


def test_a_raw_rendering_still_takes_no_voice():
    with pytest.raises(er.ExtractionRenderingError, match="takes no parameters"):
        er.parse_declaration('{"mode":"raw","voice":"assistant"}')


def test_a_template_that_cannot_place_the_turn_refuses_rather_than_falling_back():
    """A family whose template does not append the assistant turn as a suffix
    cannot be rendered this way — and saying so is the answer, because a
    fallback to the user voice would silently measure the other thing."""
    from steerlab_server.experiment import prompt_render

    with pytest.raises(prompt_render.AssistantVoiceUnsupported) as exc:
        # The flatter fake joins turn contents instead of wrapping each in its
        # own markers, so the prefix relation does not hold.
        prompt_render.render_assistant_turn(
            _TemplateAwareTokenizer(), TEXTS[0],
            model_id="google/gemma-3-4b-it")
    assert "repair:" in str(exc.value)
    assert "voice 'user'" in str(exc.value)


def test_the_manifest_round_trips_a_voiced_rendering():
    from steerlab_server.experiment.manifest import ExtractionOptions
    parsed = ExtractionOptions.from_json(
        {"extractionRendering": {"mode": "chatTemplate",
                                 "voice": "assistant"}}).extraction_rendering
    assert parsed.is_assistant_voice
    assert parsed.to_dict()["voice"] == "assistant"


# --------------------------------------------------------------------------
# 9. contentOffset(k) — the offset that counts in CONTENT coordinates
# --------------------------------------------------------------------------

#: A Gemma-shaped render: content at 4…6, close at 7, four scaffold ids after.
_TURN_IDS = [BOS, START_OF_TURN, USER_ROLE, NEWLINE, 201, 202, 203,
             END_OF_TURN, NEWLINE, START_OF_TURN, MODEL, NEWLINE]


def _turn_tokenizer():
    return _TurnStructuredTokenizer()


def test_content_offset_counts_back_from_the_content_boundary():
    tokenizer = _turn_tokenizer()

    def resolve(position):
        return position.resolve(_TURN_IDS, tokenizer=tokenizer,
                                rendering_is_raw=False)

    assert [resolve(rp.content_offset(k)).start_index for k in range(3)] \
        == [6, 5, 4]
    # k = 0 IS the last content token — same index, same value.
    assert resolve(rp.content_offset(0)).start_index == \
        resolve(rp.LAST_CONTENT_TOKEN).start_index
    assert resolve(rp.content_offset(2)).source == "last content token − 2"


def test_content_offset_refuses_underflow_instead_of_clamping():
    with pytest.raises(rp.ReadingPositionError) as exc:
        rp.content_offset(9).resolve(_TURN_IDS, tokenizer=_turn_tokenizer(),
                                     rendering_is_raw=False)
    message = str(exc.value)
    assert "no token 9 before it" in message
    assert "never clamped" in message


def test_content_offset_refuses_a_negative_k():
    with pytest.raises(rp.ReadingPositionError, match="k ≥ 0"):
        rp.content_offset(-1)


def test_content_offset_refuses_under_raw_rendering():
    """A raw stimulus has no turn, so it has no content BOUNDARY to count
    back from — the same dependency the other named roles name."""
    with pytest.raises(rp.ReadingPositionError) as exc:
        rp.content_offset(1).resolve(_TURN_IDS, tokenizer=_turn_tokenizer(),
                                     rendering_is_raw=True)
    message = str(exc.value)
    assert "templated rendering" in message
    assert f'{er.DECLARATION_FLAG} \'{{"mode": "chatTemplate"}}\'' in message


def test_content_offset_zero_canonicalizes_to_the_role_it_names():
    """The ``offsetFromEnd(0) ≡ lastToken`` rule, in content coordinates:
    declaring the offset form must never split an identity away from an
    otherwise-identical last-content-token recipe."""
    from steerlab_server.experiment.recipe_identity import canonical_reading

    assert canonical_reading(rp.content_offset(0)) == ("lastContentToken", None)
    assert canonical_reading(rp.content_offset(2)) == ("contentOffset", 2)
    assert canonical_reading(rp.LAST_CONTENT_TOKEN) == ("lastContentToken", None)


def test_the_content_offset_stamp_records_both_halves():
    model = _voiced_model()
    result = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            reading_position=rp.content_offset(1),
            extraction_rendering=er.from_json({"mode": "chatTemplate"})))
    stamp = result.reading_position_resolution
    assert stamp["requested"] == "content offset 1"
    assert stamp["mode"] == "contentOffset"
    assert stamp["parameter"] == 1
    assert stamp["source"] == "last content token − 1"
    assert len(stamp["shapes"]) == 1


# --------------------------------------------------------------------------
# 10. meanContentFromToken(n) — pooling over CONTENT only
# --------------------------------------------------------------------------


def test_the_content_mask_keeps_only_the_stimulus_own_tokens():
    """The mask derives from ONE template map: the turn opens, its role tag is
    structure, it closes, and the trailing generation scaffold is past the
    content and therefore excluded by construction."""
    content = rp.content_indices(_TURN_IDS, _turn_tokenizer(), "test")
    assert content == [4, 5, 6]


def test_mean_content_from_token_pools_content_and_stamps_both_counts():
    tokenizer = _turn_tokenizer()
    resolved = rp.mean_content_from_token(1).resolve(
        _TURN_IDS, tokenizer=tokenizer, rendering_is_raw=False)
    assert resolved.pooled_indices == (5, 6)
    assert resolved.start_index == 5 and resolved.end_index == 7
    assert resolved.pooled_token_count == 2
    # 12 tokens, 3 of them content: the mask called nine of them structure.
    assert resolved.masked_token_count == 9
    assert resolved.is_pooled


def test_mean_content_from_token_is_mean_from_token_under_raw_rendering():
    """Under raw every token IS content, so the honest behavior is
    equivalence, not a refusal — and equivalence has to be true of the
    NUMBERS, not just the docstring."""
    model = _model(_FakeTokenizer())
    masked = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            reading_position=rp.mean_content_from_token(3)))
    plain = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(reading_position=rp.mean_from_token(3)))
    assert masked.vectors.per_layer == plain.vectors.per_layer
    assert masked.residual_norm_per_layer == plain.residual_norm_per_layer
    # …and the artifact still says which was DECLARED: an identity records the
    # recipe, not what one rendering made of it.
    assert masked.reading_position_resolution["mode"] == "meanContentFromToken"
    assert "raw rendering: every token is content" in \
        masked.reading_position_resolution["source"]
    assert masked.reading_position_resolution["shapes"][0][
        "exampleMaskedTokenCount"] == 0


def test_a_masked_pool_reads_exactly_the_pooled_positions():
    """The mask has to reach the RECORDER, not just the stamp: pooling the
    whole window would quietly average the template's structure back in."""
    model = _voiced_model()
    masked = extractor.activations(
        model, [TEXTS[0]], rp.mean_content_from_token(0),
        er.from_json({"mode": "chatTemplate"}))
    resolved = masked.resolutions[0]
    # A hand-rolled mean over the same positions, from a window read.
    window = extractor.activations(
        model, [TEXTS[0]], rp.mean_from_token(resolved.start_index),
        er.from_json({"mode": "chatTemplate"}))
    # The window read runs to the sequence end (through the scaffold), so the
    # two must NOT agree — that difference is the mask doing its job.
    assert masked.values[0][0] != window.values[0][0]
    assert resolved.pooled_indices[-1] + 1 < resolved.token_count


def test_mean_content_from_token_refuses_when_the_content_is_too_short():
    with pytest.raises(rp.ReadingPositionError) as exc:
        rp.mean_content_from_token(5).resolve(
            _TURN_IDS, tokenizer=_turn_tokenizer(), rendering_is_raw=False)
    message = str(exc.value)
    assert "3 content tokens" in message
    assert "never clamped" in message


def test_mean_content_from_token_refuses_a_negative_n():
    with pytest.raises(rp.ReadingPositionError, match="n ≥ 0"):
        rp.mean_content_from_token(-1)


def test_the_masked_pool_stamp_carries_the_pooled_and_masked_counts():
    model = _voiced_model()
    result = extractor.extract(
        model, _stimuli(),
        extractor.ExtractionOptions(
            reading_position=rp.mean_content_from_token(1),
            extraction_rendering=er.from_json({"mode": "chatTemplate"})))
    stamp = result.reading_position_resolution
    assert stamp["mode"] == "meanContentFromToken"
    assert stamp["parameter"] == 1
    shape = stamp["shapes"][0]
    assert shape["offsetFromEnd"] is None          # a pooled read has none
    assert shape["examplePooledTokenCount"] >= 1
    # BOS + open + role + newline + close + newline + open + model + newline
    assert shape["exampleMaskedTokenCount"] == 9


def test_the_new_positions_round_trip_through_their_labels():
    for position in (rp.content_offset(0), rp.content_offset(4),
                     rp.mean_content_from_token(0),
                     rp.mean_content_from_token(12)):
        assert rp.from_label(position.label).label == position.label
    # …and they do not collide with the position they are named after.
    assert rp.parse_label_strict("mean from token 3").identity_mode == \
        "meanFromToken"
    assert rp.parse_label_strict("mean content from token 3").identity_mode == \
        "meanContentFromToken"
    assert rp.parse_label_strict("content offset x") is None


def test_the_manifest_parses_both_wire_forms_of_the_new_positions():
    from steerlab_server.experiment.manifest import ExtractionOptions
    cases = {
        "contentOffset": ({"contentOffset": {"_0": 2}}, "content offset 2"),
        "meanContentFromToken": ({"meanContentFromToken": {"_0": 4}},
                                 "mean content from token 4"),
    }
    for name, (swift_form, label) in cases.items():
        assert ExtractionOptions.from_json(
            {"readingPosition": swift_form}).reading_position.label == label, name
        assert ExtractionOptions.from_json(
            {"readingPosition": label}).reading_position.label == label, name
