"""A reader-derived steering vector, attached (audit finding 2).

Until 2026-08-27 the one artifact the faithful RepE pipeline produces was the
one artifact a study could not cite: :func:`attach_artifact` resolves the
sidecar's ``extractionMethod`` to ask where the concept's held-out data lives,
and ``"repeReaderLAT"`` was not in the ``ExtractionMethod`` vocabulary, so
every attempt died on "which this engine does not know". A researcher could fit
a reader, derive its vector, see it in the catalog — and then have no way to put
it in an experiment.

Its data questions have honest answers, and they are not a plain concept's: the
stimuli are the READER's dataset (whose SHA-256 is the ``stimulusSetHash``),
there is no ``prompts/concepts/<c>/`` pair set, and the held-out evidence is the
reader artifact's own accuracy — not a ``validation.jsonl``. So
``has_source_concept`` is False and every data-side branch skips rather than
inventing.

Swift twin: ``Tests/ExperimentKitTests/ReaderDerivedAttachTests.swift``.
"""

from __future__ import annotations

import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering import repe_reader, vector_store
from steerlab_server.steering.vector_math import ScalarProbe

MODEL = "org/m"
REVISION = "abc123"
RUN = "runs/20260827T090000000-derive-reader-vector"
NAME = "candour-repe-reader"


def _reader(orientation: float = 1.0) -> repe_reader.ReaderArtifact:
    return repe_reader.ReaderArtifact(
        model_id=MODEL, revision=REVISION, concept="candour", layer=1,
        template=repe_reader.TaskTemplate(
            id="unnamed-scenario-v1", text="S: {{stimulus}} q",
            concept_slot=False, lat_token="final", hash="th",
            divergence="unnamed-clean-room"),
        dataset_hash="reader-dataset-hash",
        probe=ScalarProbe(
            direction=[1.0, 0.0, 0.0], projection_center=0.0,
            projection_scale=1.0, orientation=orientation,
            positive_mean=orientation, negative_mean=-orientation,
            activation_center=[0.0, 0.0, 0.0]),
        difference_cloud_explained_variance=0.6,
        train_accuracy=1.0, held_out_accuracy=0.9,
        train_pair_count=8, held_out_pair_count=4,
        sign_convention=repe_reader.HELD_OUT_PAIR_AGREEMENT,
        sign_held_out_accuracy=1.0)


def _plant(root: str, *, orientation: float = 1.0, backfilled: bool = True) -> str:
    """Write a derived artifact, optionally with the norms a backfill measures."""
    vectors, sidecar = repe_reader.derive_steering_sidecar(
        _reader(orientation), reader_file_name="reader-candour-layer1.json",
        reader_bytes=b"reader-bytes")
    if backfilled:
        sidecar.residualNormPerLayer = [7.0, 7.5]
        sidecar.residualNormSource = "neutral-corpus"
    directory = os.path.join(root, RUN)
    vector_store.save(vectors, sidecar, directory, NAME)
    return f"{RUN}/{NAME}"


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    es.create("reader-study", model_id=MODEL, revision=REVISION, root=root)
    return root


def test_reader_derived_vector_attaches_and_pins_the_reader_dataset(workspace):
    artifact = _plant(workspace)
    sidecar = json.load(open(os.path.join(workspace, artifact + ".json")))
    assert sidecar["extractionMethod"] == "repeReaderLAT"

    d = es.attach_artifact("reader-study", "candour", artifact, root=workspace)
    ref = d["concepts"][0]
    assert ref["options"]["method"] == "pinnedArtifact"
    block = ref["vectorArtifact"]
    assert block["sourceMethod"] == "repeReaderLAT"
    assert block["residualNormSource"] == "neutral-corpus"
    # The READER's dataset hash travels verbatim: nothing under
    # prompts/concepts/ is looked up for a reader-derived direction.
    assert ref["stimulusSetHash"] == "reader-dataset-hash"
    # A reader's held-out evidence is on the reader artifact, so the pin
    # records no validation.jsonl rather than inventing one. (The Swift twin
    # additionally carries `validationHashPinnedAbsent`, a Mac-side manifest
    # field this engine does not model.)
    assert ref["validationHash"] is None
    # The pin passes verify the moment it is written.
    assert Manifest.from_dict(d).verify(root=workspace) == []


def test_reader_derived_vector_refuses_a_source_concept(workspace):
    artifact = _plant(workspace)
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.attach_artifact("reader-study", "candour", artifact,
                           source_concept="honesty", root=workspace)
    message = str(excinfo.value)
    assert "no source concept" in message
    assert "fitted RepE reader" in message
    assert "held-out accuracy" in message


def test_unbackfilled_reader_vector_names_the_backfill(workspace):
    """Not a defect of the artifact but a missing LIFECYCLE STEP, and the
    refusal names the verb that supplies it — cross-engine twin literal with
    ``ExperimentStore.readerDerivedNormBackfillRefusal``."""
    artifact = _plant(workspace, backfilled=False)
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.attach_artifact("reader-study", "candour", artifact, root=workspace)
    assert str(excinfo.value) == (
        f"vector artifact '{artifact}' is a RepE-reader-derived direction with "
        "no residualNormSource — a reader measures a task template's LAT "
        "token, not a neutral corpus, so its reading direction is BORN without "
        "a denominator. Run the residual-norm backfill against the pinned "
        "neutral corpus first and attach the BACKFILLED artifact: α in norm "
        "units is meaningless until the denominator is measured")


def test_attached_vector_carries_the_held_out_sign_and_the_disagreement(workspace):
    """The sign rule is visible on the ARTIFACT, not only in the conversion.
    This reader was signed by HELD-OUT pair agreement, so its fitted direction
    ships unflipped even though the train class means read the other way —
    and that disagreement is stamped rather than discarded (review round 6,
    finding 1)."""
    artifact = _plant(workspace, orientation=-1.0)
    sidecar = json.load(open(os.path.join(workspace, artifact + ".json")))
    assert sidecar["readerProbeOrientation"] == -1.0
    assert sidecar["trainHeldOutSignDisagreement"] is True
    assert sidecar["readerLayer"] == 1
    assert sidecar["readerTemplateID"] == "unnamed-scenario-v1"
    assert sidecar["readerSignConvention"] == "heldOutPairAgreement"
    assert sidecar["signConvention"] == "heldOutPairAgreement"
    assert sidecar["readerContrastMode"] == "supervisedContent"
    vectors, _ = vector_store.load(os.path.join(workspace, RUN), NAME)
    assert vectors.per_layer[1] == pytest.approx([1.0, 0.0, 0.0])


def test_an_agreeing_reader_stamps_the_agreement(workspace):
    artifact = _plant(workspace, orientation=1.0)
    sidecar = json.load(open(os.path.join(workspace, artifact + ".json")))
    assert sidecar["trainHeldOutSignDisagreement"] is False
    vectors, _ = vector_store.load(os.path.join(workspace, RUN), NAME)
    assert vectors.per_layer[1] == pytest.approx([1.0, 0.0, 0.0])
