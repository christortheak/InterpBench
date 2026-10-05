"""The engine's ``bundle execute`` child, with the MODEL replaced and nothing
else.

A runner executes a submitted bundle in a child process: ``<python> -m
steerlab_server.cli bundle execute <bundle> --verb <verb> …``, where
``<python>`` is ``$STEERLAB_PYTHON`` when the operator sets one. The app-free
journey test (``test_client_draft_to_freeze.py``) points that variable at a
small launcher that runs THIS file, which installs the stand-ins below and then
hands control to the same module with the same arguments.

So everything a runner does is real — the import of the uploaded bundle into
the runner's own root, the validate and run workflows, the evidence file and
its scope hash, the run directory, the evidence archive — and the one thing
that is not is the arithmetic a GPU would have done: reading activations out of
a model and generating text from it. The stand-ins are the same seams the
engine's own workflow tests already replace
(``test_vacuous_validation_and_empty_study.py``, ``test_battery_in_run.py``).

It is deliberately NOT a fixture of a finding: the vectors are constants and
the "generations" are two fixed strings. Nothing here says anything about any
model.
"""

from __future__ import annotations

import runpy
import sys
from types import SimpleNamespace

#: Depth of the stand-in model. Small, and larger than any layer the journey
#: test declares.
LAYER_COUNT = 4

#: The commit the stand-in "resolves" when a study pins none — what a real
#: load of a model id with no revision does. Obviously not a real commit.
RESOLVED_REVISION = "ab" * 20

#: The stand-in's whole theory of meaning: a text containing one of these
#: reads on the negative side of the stand-in direction, anything else on the
#: positive side. The journey test's stimuli are written to match.
OPPOSING_WORDS = ("off", "dark")


def install() -> None:
    """Replace the model, and only the model."""
    import steerlab_server.experiment.generate as generate_module
    import steerlab_server.experiment.model_resources as model_resources
    import steerlab_server.experiment.vector_materialization as materialization
    from steerlab_server.steering import extractor
    from steerlab_server.steering.vector_store import ConceptVectors

    def load_model(manifest, dtype, device=None):
        # The revision the study pins, so nothing downstream sees a model that
        # disagrees with the manifest it was loaded for; or, for a study that
        # pins none, a resolved commit — which the engine then writes into its
        # copy of the draft, exactly as it does after a real load.
        return SimpleNamespace(
            model_id=manifest.model_id,
            revision=manifest.model_revision or RESOLVED_REVISION)

    def extract_all(model, manifest, root):
        return {
            concept.name: materialization.ConceptVectorBundle(
                vectors=ConceptVectors(
                    per_layer=[[1.0, 0.0]] * LAYER_COUNT),
                residual_norm_per_layer=[1.0] * LAYER_COUNT,
                residual_norm_source="model-free stand-in",
                stimulus_hash=concept.stimulus_set_hash)
            for concept in manifest.concepts}

    def activations(model, texts, reading, rendering=None):
        # One fixed rule in place of a forward pass, so the engine's own probe
        # arithmetic has two classes to separate: a text reads "against" the
        # stand-in direction when it contains one of two words.
        return SimpleNamespace(values=[
            [[-1.0 if any(word in text for word in OPPOSING_WORDS) else 1.0,
              0.0]] * LAYER_COUNT
            for text in texts])

    def logit_lens(model, vectors, layer, top_k=12):
        return extractor.LogitLensReport(layer=layer, top_positive=[],
                                         top_negative=[])

    def generate(model, prompt, *, injections=None, **_ignored):
        return ("steered answer" if injections else "plain answer")

    model_resources._load_model = load_model
    materialization.extract_all = extract_all
    materialization.persist_vectors = lambda *args, **kwargs: None
    extractor.activations = activations
    extractor.logit_lens = logit_lens
    generate_module.generate = generate


def main(argv: list) -> None:
    """``-m <module> <args…>``, exactly as the runner composed it."""
    if len(argv) < 2 or argv[0] != "-m":
        raise SystemExit(
            "model_free_engine_child: expected '-m <module> <args…>', got "
            f"{argv!r}")
    module, arguments = argv[1], argv[2:]
    install()
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
