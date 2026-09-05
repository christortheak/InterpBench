#!/usr/bin/env python3
"""Regenerate the committed cross-engine fixtures under Tests/Fixtures/cross-engine.

These fixtures are PRODUCED BY THE PYTHON ENGINE and CONSUMED BY THE SWIFT
TESTS. That direction is the point (B1): a Swift test that hand-writes what it
believes the server emits pins the Swift author's belief, not the server's
behaviour — and the two agent-discovery bugs fixed on 2026-07-26 were both
exactly that belief drifting out of date. Because these bytes come from the
real producer, drift surfaces as a fixture diff in review rather than as a
green test suite over a broken seam.

Run from the repo root:

    Server/.venv.nosync/bin/python scripts/regenerate-cross-engine-fixtures.py

then run the Swift suite. A fixture diff means one engine changed its
contract: decide deliberately whether the other should follow.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "Server"))

FIXTURES = os.path.join(REPO, "Tests", "Fixtures", "cross-engine")


def _write(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {os.path.relpath(path, REPO)}")


def promotion_keys() -> None:
    """The idempotency key both engines must agree on.

    Alphas deliberately span the shapes that could diverge between Python's
    ``repr`` and Swift's ``description``: plain decimals, a value with a long
    binary expansion, an integral value, and an exponent-formatted one."""
    from steerlab_server.experiment import promote

    cases = []
    for label, alpha in (
        ("plain", 0.4),
        ("two-places", 0.05),
        ("long-expansion", 0.1 + 0.2),
        ("integral", 1.0),
        ("small-exponent", 0.00001),
        ("negative", -0.35),
    ):
        payload = {
            "experiment": "virtue-extraction-study",
            "experimentHash": "a" * 64,
            "concept": "practicalwisdom",
            "sweepRun": "20260726T162230362-exp-pw-sweep",
            "winningCell": {"layer": 41, "alpha": alpha},
            "vectorArtifactID": "runs/20260726T000001000-extract/practicalwisdom",
            "vectorArtifactHash": "d" * 64,
            "promotedBy": "criterion",
            "agentName": "pw-agent",
        }
        cases.append({
            "label": label,
            "input": payload,
            "key": promote.promotion_key(
                experiment=payload["experiment"],
                experiment_hash=payload["experimentHash"],
                concept=payload["concept"],
                sweep_run=payload["sweepRun"],
                layer=payload["winningCell"]["layer"],
                alpha=payload["winningCell"]["alpha"],
                vector_artifact_id=payload["vectorArtifactID"],
                vector_artifact_hash=payload["vectorArtifactHash"],
                promoted_by=payload["promotedBy"],
                agent_name=payload["agentName"]),
        })
    # A promotion with no sweep run at all (hand-created lineage) must also
    # agree — `null` is a distinct canonical form from the empty string.
    payload = {
        "experiment": "e", "experimentHash": "b" * 64, "concept": "fear",
        "sweepRun": None, "winningCell": {"layer": 3, "alpha": 0.4},
        "vectorArtifactID": "runs/x/fear", "vectorArtifactHash": None,
        "promotedBy": "manualOverride", "agentName": "hand-agent",
    }
    cases.append({
        "label": "no-sweep-run",
        "input": payload,
        "key": promote.promotion_key(
            experiment="e", experiment_hash="b" * 64, concept="fear",
            sweep_run=None, layer=3, alpha=0.4,
            vector_artifact_id="runs/x/fear", promoted_by="manualOverride",
            agent_name="hand-agent"),
    })
    # Non-ASCII names (2026-07-27): the canonical form is RAW UTF-8
    # (ensure_ascii=False, the recipe-identity house convention). Python's
    # json.dumps default \uXXXX-escaped these while Swift passed them
    # through raw, so an accented concept name produced DIFFERENT keys per
    # engine and an imported server agent was re-minted as a duplicate.
    for label, concept, agent in (
        ("non-ascii-accent", "résilience", "sagesse-pratique-agent"),
        ("non-ascii-smart-quote", "judge’s-temperament", "l’agent"),
        ("non-ascii-cjk", "勇気", "勇気-agent"),
    ):
        payload = {
            "experiment": "étude-vertu", "experimentHash": "c" * 64,
            "concept": concept,
            "sweepRun": "20260727T000000000-exp-vertu-sweep",
            "winningCell": {"layer": 21, "alpha": 0.4},
            "vectorArtifactID": f"runs/20260727T000001000-extract/{concept}",
            "vectorArtifactHash": "e" * 64,
            "promotedBy": "criterion", "agentName": agent,
        }
        cases.append({
            "label": label,
            "input": payload,
            "key": promote.promotion_key(
                experiment=payload["experiment"],
                experiment_hash=payload["experimentHash"],
                concept=concept,
                sweep_run=payload["sweepRun"],
                layer=21, alpha=0.4,
                vector_artifact_id=payload["vectorArtifactID"],
                vector_artifact_hash=payload["vectorArtifactHash"],
                promoted_by="criterion", agent_name=agent),
        })
    _write(os.path.join(FIXTURES, "promotion-keys.json"), cases)


def server_minted_agent() -> None:
    """A real agent artifact, minted by the server's own ``save_variant``.

    The Swift importer/discovery tests read THIS — the actual on-disk layout
    and filename the Python engine produces — rather than a Swift author's
    reconstruction of it."""
    from steerlab_server.experiment import model_variant

    workspace = tempfile.mkdtemp(prefix="steerlab-fixture-")
    try:
        variant = model_variant.ModelVariant(
            name="neuroticism-agent",
            base_model_id="mlx-community/gemma-3-27b-it-8bit",
            base_revision="c" * 40,
            injections=[{
                "concept": "neuroticism",
                # Workspace-RELATIVE, exactly as `promote` writes it.
                "vectorArtifactID":
                    "runs/20260726T000001000-extract/neuroticism",
                "layer": 41, "alpha": 0.1}],
            band_width=1,
            alpha_in_norm_units=True,
            prompt_mode="chatAssistant",
            qwen_thinking_enabled=False,
            temperature=0.0,
            system_prompt="",
            created_at="2026-07-26T00:00:00Z",
            promotion={
                "experiment": "bigfive-study",
                "experimentHash": "a" * 64,
                "promotedAt": "2026-07-26T00:00:00Z",
                "promotedBy": "criterion",
                "winningCell": {"layer": 41, "alpha": 0.1},
                "substrate": "python-hf-transformers",
                "appVersion": "steerlab-server (fixture)",
            })
        saved = model_variant.save_variant(variant, workspace)
        run_dir = saved["runDirectory"]

        # Capture the LAYOUT, not just the payload: which directory shape and
        # which filename the server chose, plus the run-type stamp Swift's
        # discovery keys on.
        with open(saved["path"], encoding="utf-8") as handle:
            artifact = json.load(handle)
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        # The timestamp in the directory name varies per regeneration, and a
        # fixture that churns on every run teaches reviewers to ignore its
        # diffs — which is the opposite of the point. Record the STABLE part
        # (everything after the stamp) and the stamp's length.
        basename = os.path.basename(run_dir)
        stamp, _, _ = basename.partition("-")
        _write(os.path.join(FIXTURES, "server-minted-agent.json"), {
            "note": "produced by Server model_variant.save_variant — do not "
                    "hand-edit; regenerate with "
                    "scripts/regenerate-cross-engine-fixtures.py",
            "runDirectorySuffix": basename[len(stamp):],
            "runDirectoryStampLength": len(stamp),
            "artifactFileName": os.path.basename(saved["path"]),
            "configRunType": config.get("runType"),
            "artifact": artifact,
        })
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def server_minted_adapter_agent() -> None:
    """A server-minted ADAPTER agent, embedded in a variant condition.

    Open-issues #11: ``steerlab-cli experiment verify`` failed
    ``DecodingError.keyNotFound: 'name'`` at
    ``variantConditions[0].artifact.adapters[0]`` on every study whose adapter
    agent came from the server/driver path, while ``steerlab-server experiment
    verify`` read the same manifest clean. The cause is here, in the bytes:
    the server carries adapter entries VERBATIM (``ModelVariant.adapters`` is a
    list of opaque dicts, unchanged by ``from_dict``/``to_dict``), so the shape
    the driver posted to ``POST /api/model-variant/save`` —
    ``{adapterDirectory, adapterHash, configHash}``, no ``name`` — is the shape
    on disk. Every server-side reader tolerates it
    (``adapter.get('name') or adapter.get('adapterDirectory')``); the Swift
    decoder required the key.

    The fixture therefore runs the REAL round trip (``from_dict`` →
    ``save_variant``) over the driver's body rather than asserting a Swift
    author's idea of it, and also records the ``variantConditions`` envelope
    Swift's manifest decoder walks to reach the adapter."""
    from steerlab_server.experiment import model_variant

    workspace = tempfile.mkdtemp(prefix="steerlab-fixture-")
    try:
        # The driver's request body, verbatim: no adapter `name`, no
        # `artifactPath` — only the directory and the two byte pins.
        body = {
            "name": "sympathy-lora-agent",
            "baseModelID": "google/gemma-3-27b-it",
            "baseRevision": "005ad3404e59",
            "adapters": [{
                "adapterDirectory": "adapters/sympathy-lora",
                "adapterHash": "d" * 64,
                "configHash": "e" * 64,
            }],
            "injections": [],
            "bandWidth": 1,
            "alphaInNormUnits": True,
            "promptMode": "chatAssistant",
            "qwenThinkingEnabled": False,
            "temperature": 0.0,
            "createdAt": "2026-08-16T00:00:00Z",
        }
        variant = model_variant.ModelVariant.from_dict(body)
        saved = model_variant.save_variant(variant, workspace)
        with open(saved["path"], encoding="utf-8") as handle:
            artifact = json.load(handle)
        _write(os.path.join(FIXTURES, "server-minted-adapter-agent.json"), {
            "note": "produced by Server model_variant.from_dict + "
                    "save_variant over the driver's POST body — do not "
                    "hand-edit; regenerate with "
                    "scripts/regenerate-cross-engine-fixtures.py",
            "artifact": artifact,
            # The envelope Swift decodes through to reach the adapter entry
            # (`ExperimentManifest.VariantCondition`), so the fixture pins the
            # whole path named in the failure, not just its last hop.
            # The run-directory timestamp varies per regeneration, and a
            # fixture that churns on every run teaches reviewers to ignore its
            # diffs. The condition's path is not what this fixture is about —
            # the ADAPTER ENTRY is — so it is pinned to a stable stand-in
            # while the artifact and its hash come from the real writer.
            "variantConditions": [{
                "name": "sympathy-agent",
                "artifactPath": ("runs/20260816T000000000-variant-"
                                 "sympathy-lora-agent/"
                                 "sympathy-lora-agent.json"),
                "artifactHash": saved["hash"],
                "artifact": artifact,
            }],
        })
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def scenario_diagnostics() -> None:
    """The D1 diagnostics arithmetic, over cases chosen to pin the decisions
    that are easy to get silently different between engines: tie handling in
    AUC, single-class behaviour, Wilson bounds at the ends of the scale, and
    what a perfectly-ranked-but-badly-thresholded direction looks like."""
    from steerlab_server.experiment import scenario_diagnostics as sd

    def case(label, projections, labels, threshold, **kw):
        scenarios = [{"id": f"s{i}", "text": f"scenario {i}"}
                     for i in range(len(projections))]
        report = sd.diagnostics(
            direction=[1.0, 0.0], scenarios=scenarios,
            projections=projections, labels=labels, threshold=threshold,
            class_means=kw.get("class_means", {"positive": 1.0, "negative": -1.0}),
            layer=kw.get("layer", 3), direction_norm=kw.get("direction_norm", 1.0))
        return {"label": label,
                "input": {"projections": projections, "labels": labels,
                          "threshold": threshold},
                "report": report}

    cases = [
        # Clean separation, threshold in the right place.
        case("separated", [2.0, 1.5, -1.5, -2.0], [True, True, False, False], 0.0),
        # THE case accuracy cannot express: ranking is perfect (AUC 1.0) but
        # the threshold sits above every point, so accuracy is at chance.
        case("ranked-but-misthresholded",
             [2.0, 1.5, -1.5, -2.0], [True, True, False, False], 5.0),
        # Every projection identical: AUC must be exactly 0.5, not 0 or 1
        # depending on comparison order.
        case("all-ties", [1.0, 1.0, 1.0, 1.0], [True, True, False, False], 1.0),
        # One class only: AUC is undefined, and any number would be an artefact.
        case("single-class", [1.0, 2.0, 3.0], [True, True, True], 0.0),
        # Perfectly wrong ranking.
        case("inverted", [-2.0, -1.5, 1.5, 2.0], [True, True, False, False], 0.0),
        # Small-N accuracy at the top of the scale — where a normal-approximation
        # interval would run past 1.0.
        case("perfect-small-n", [1.0, 2.0, -1.0, -2.0], [True, True, False, False], 0.0),
    ]
    _write(os.path.join(FIXTURES, "scenario-diagnostics.json"), cases)


def validation_layers() -> None:
    """The D4 precedence and clamping. Both engines must resolve the same
    layer from the same declaration, or a study validated on one substrate
    reads a different depth than the same study on the other."""
    from steerlab_server.experiment import validation_layer as vl

    cases = []
    for label, kwargs in (
        ("declared-index", dict(declared_layer=41, declared_fraction=None,
                                condition_layer=7, layer_count=62)),
        ("declared-fraction", dict(declared_layer=None, declared_fraction=0.66,
                                   condition_layer=7, layer_count=62)),
        # The legacy rule, preserved exactly: existing manifests keep their
        # numbers and therefore their content hashes.
        ("legacy-condition", dict(declared_layer=None, declared_fraction=None,
                                  condition_layer=7, layer_count=62)),
        ("legacy-mid-network", dict(declared_layer=None, declared_fraction=None,
                                    condition_layer=None, layer_count=62)),
        # Clamping at both ends.
        ("index-above-depth", dict(declared_layer=999, declared_fraction=None,
                                   condition_layer=None, layer_count=62)),
        ("fraction-one", dict(declared_layer=None, declared_fraction=1.0,
                              condition_layer=None, layer_count=62)),
        ("fraction-zero", dict(declared_layer=None, declared_fraction=0.0,
                               condition_layer=None, layer_count=62)),
        ("single-layer-model", dict(declared_layer=None, declared_fraction=0.5,
                                    condition_layer=None, layer_count=1)),
    ):
        r = vl.resolve(**kwargs)
        cases.append({"label": label, "input": kwargs,
                      "layer": r.layer, "source": r.source,
                      "depthFraction": r.depth_fraction})
    _write(os.path.join(FIXTURES, "validation-layers.json"), cases)


def choice_margins() -> None:
    """The D3 margin arithmetic. Quantile interpolation and clamp detection
    are the parts two implementations silently disagree on."""
    from steerlab_server.experiment import choice_margin as cm

    def records(pairs):
        return [{"optionLogprobs": {"A": a, "B": b}} for a, b in pairs]

    cases = [
        # Near the boundary: the flip rate is a sensitive statistic here.
        ("near-boundary", records([(-1.0, -1.5), (-2.0, -2.2), (-3.0, -3.1)])),
        # Far from it: flip rate is insensitive, log-odds still moves.
        ("far-from-boundary", records([(-1.0, -21.0), (-2.0, -25.0),
                                       (-1.5, -30.0)])),
        # Even-length sample, to pin the median's interpolation.
        ("even-length", records([(-1.0, -2.0), (-1.0, -3.0), (-1.0, -5.0),
                                 (-1.0, -9.0)])),
        # A degenerate softmax: TRUE numerical saturation, log-odds clamped.
        ("clamped", records([(0.0, -100.0), (0.0, -0.5)])),
        # A single option has no boundary, so no margin.
        ("single-option", [{"optionLogprobs": {"A": -1.0}}]),
        ("no-records", []),
    ]
    _write(os.path.join(FIXTURES, "choice-margins.json"),
           [{"label": label,
             "input": [r["optionLogprobs"] for r in recs],
             "report": cm.diagnostics(recs)}
            for label, recs in cases])


def auto_prompt_ids() -> None:
    """The auto-generated promptID rule.

    Paired statistics key on promptID, so a divergence here does not fail a
    join — it joins the WRONG items. Until 2026-07-26 Python used the 0-based
    FILE LINE index (blank lines included) while Swift used the 1-based
    ordinal of parsed prompts, so the two engines mapped one another's item
    k+1 onto item k and dropped one at each end."""
    import tempfile

    from steerlab_server.experiment import experiment_store as es, tasks
    from steerlab_server.experiment.manifest import Manifest

    # Blank lines and an explicitly-identified row, so the fixture pins the
    # interaction rather than only the happy path.
    rows = [
        '{"text": "first"}',
        '',
        '{"text": "second"}',
        '{"id": "named", "text": "third"}',
        '',
        '{"text": "fourth"}',
    ]
    workspace = tempfile.mkdtemp(prefix="steerlab-fixture-")
    try:
        es.create("ids", model_id="org/m", root=workspace)
        path = os.path.join(workspace, "prompts.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")
        prompts = tasks._load_prompts(
            Manifest.load("ids", workspace), path, workspace)
        _write(os.path.join(FIXTURES, "auto-prompt-ids.json"), {
            "note": "auto-generated promptIDs must be identical on both "
                    "engines — paired statistics key on this value",
            "lines": rows,
            "ids": [p["id"] for p in prompts],
        })
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def row_hashes() -> None:
    """Scenario row hashes over text that ESCAPES DIFFERENTLY.

    `json.dumps` defaults to ensure_ascii=True, so every non-ASCII character
    and every control character is escaped; a hand-rolled Swift escaper that
    handled only backslash, quote and newline diverged on all of them. The
    earlier fixture was ASCII-only and never compared row hashes, so tests
    missed it entirely — and LLM-generated prose contains smart quotes as a
    matter of course."""
    from steerlab_server.experiment import scenario_diagnostics as sd

    cases = []
    for label, text in (
        ("ascii", "a plain sentence"),
        ("accented", "the café was closed"),
        ("smart-quotes", "she said \u2018no\u2019 \u2014 firmly"),
        ("tab", "before\tafter"),
        ("carriage-return", "before\rafter"),
        ("newline", "before\nafter"),
        ("backslash-and-quote", 'a \\ b " c'),
        ("control-char", "bell\x07here"),
        ("astral", "emoji \U0001F600 here"),
        ("cjk", "\u53f8\u6cd5\u7f8e\u5fb7"),
    ):
        for label_value in (True, False):
            cases.append({
                "label": f"{label}-{'pos' if label_value else 'neg'}",
                "text": text,
                "expresses": label_value,
                "rowHash": sd.row_hash(text, label_value),
            })
    _write(os.path.join(FIXTURES, "scenario-row-hashes.json"), cases)


def evidence_bundle() -> None:
    """A REAL evidence bundle, produced by the server's own packager.

    Everything else in this file pins a value. This pins the WIRE: the Swift
    importer consumes bytes that `bundles.package_evidence` actually wrote,
    around an agent that `model_variant.save_variant` actually minted, beside
    a sweep run a pinned promotion can name. Until now the Swift discovery
    tests planted a hand-built directory into `runs/` — which pins the Swift
    author's belief about the server's output, and that belief drifting is
    precisely the bug this whole seam keeps producing.
    """
    from steerlab_server.experiment import (bundles, experiment_store as es,
                                            model_variant, paths)
    from steerlab_server.experiment.manifest import Manifest

    workspace = tempfile.mkdtemp(prefix="steerlab-bundle-fixture-")
    try:
        # A concept the study pins, so an imported promotion can resolve it.
        concept_dir = os.path.join(workspace, "prompts", "concepts", "fear")
        os.makedirs(concept_dir, exist_ok=True)
        for side, text in (("positive", "afraid"), ("negative", "calm")):
            with open(os.path.join(concept_dir, f"{side}.jsonl"), "w",
                      encoding="utf-8") as handle:
                handle.write(json.dumps({"text": text}) + "\n")
        es.create("seam", model_id="org/m", revision="a" * 40, root=workspace)
        es.attach("seam", ["fear"], root=workspace)
        manifest = Manifest.load("seam", workspace)
        stimulus_hash = manifest.concepts[0].stimulus_set_hash

        # An extraction artifact with a FULL recipe sidecar, so promote can
        # match it by recipe identity after import.
        extract_dir = os.path.join(workspace, "runs", "20260726T000001000-extract")
        os.makedirs(extract_dir, exist_ok=True)
        with open(os.path.join(extract_dir, "fear.safetensors"), "wb") as handle:
            handle.write(b"weights")
        with open(os.path.join(extract_dir, "fear.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({
                "modelID": "org/m", "concept": "fear", "layerCount": 4,
                "hiddenSize": 2, "stimulusSetHash": stimulus_hash,
                "normsPerLayer": [1.0] * 4, "residualNormPerLayer": [1.0] * 4,
                "extractionMethod": "meanDifference", "revision": "a" * 40,
                "substrate": "python-hf-transformers",
                "readingPosition": "last token", "neutralProjection": "none",
                "residualNormSource": "extraction-stimuli",
                "extractionDate": "2026-07-26T00:00:00Z",
            }, handle, indent=2, sort_keys=True)

        # A sweep run a PINNED promotion can name, stamped with the epoch.
        sweep_name = "20260726T000002000-exp-seam-sweep"
        sweep_dir = os.path.join(workspace, "runs", sweep_name)
        os.makedirs(sweep_dir, exist_ok=True)
        with open(os.path.join(sweep_dir, "sweep.csv"), "w",
                  encoding="utf-8") as handle:
            handle.write("concept,layer,alpha,markerDensity,distinct2,"
                         "batteryAccuracy\n")
        with open(os.path.join(sweep_dir, "recommendations.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"fear": {
                "sweepRun": sweep_name,
                "criterion": {"objective": {"metric": "markerDensity"},
                              "constraints": {"capabilityTolerance": 0.15,
                                              "coherenceFloor": 0.45}},
                "devPromptsHash": "d" * 64,
                "winningCell": {"layer": 2, "alpha": 0.4},
                "metrics": {"markerDensity": 0.31},
            }}, handle, indent=2, sort_keys=True)
        with open(os.path.join(sweep_dir, "experiment-hash.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write(manifest.content_hash())
        # The manifest SNAPSHOT every real server run carries
        # (`_write_config_snapshot`). Omitting it made the fixture
        # unrealistic in exactly the dimension the epoch guard depends on:
        # the snapshot is the only epoch evidence that means the same thing
        # on both engines, and without it the Swift test exercised the
        # no-snapshot fallback while believing it covered the real path.
        with open(os.path.join(sweep_dir, "experiment.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(manifest.raw, handle, indent=2, sort_keys=True)
        # Real run stamps, so the fixture carries the SUBSTRATE a genuine
        # server run records — the field the Mac reads to explain why a
        # foreign epoch stamp can never match.
        from steerlab_server.experiment.run_config import write_run_config
        write_run_config(sweep_dir, "sweep", model_id="org/m",
                         revision="a" * 40)
        write_run_config(extract_dir, "extract", model_id="org/m",
                         revision="a" * 40)

        # The agent itself, minted by the real code path.
        variant = model_variant.ModelVariant(
            name="seam-fear-agent",
            base_model_id="org/m", base_revision="a" * 40,
            injections=[{"concept": "fear",
                         "vectorArtifactID": "runs/20260726T000001000-extract/fear",
                         "layer": 2, "alpha": 0.4}],
            band_width=1, alpha_in_norm_units=True,
            prompt_mode="chatAssistant", qwen_thinking_enabled=False,
            temperature=0.0, system_prompt="",
            created_at="2026-07-26T00:00:00Z",
            promotion={"experiment": "seam", "experimentHash": "a" * 64,
                       "promotedAt": "2026-07-26T00:00:00Z",
                       "promotedBy": "criterion",
                       "winningCell": {"layer": 2, "alpha": 0.4},
                       "substrate": "python-hf-transformers",
                       "appVersion": "steerlab-server (fixture)"})
        saved = model_variant.save_variant(variant, workspace)

        meta = bundles.package_evidence(
            saved["runDirectory"], root=workspace,
            extra_run_directories=[extract_dir, sweep_dir])

        target = os.path.join(FIXTURES, "server-evidence-bundle.tar.gz")
        shutil.copyfile(meta["bundlePath"], target)
        print(f"wrote {os.path.relpath(target, REPO)}")

        agent_run = os.path.basename(saved["runDirectory"])
        stamp, _, _ = agent_run.partition("-")
        _write(os.path.join(FIXTURES, "server-evidence-bundle.json"), {
            "note": "produced by Server bundles.package_evidence — do not "
                    "hand-edit; regenerate with "
                    "scripts/regenerate-cross-engine-fixtures.py",
            "archive": "server-evidence-bundle.tar.gz",
            "bundleSha256": meta["bundleSha256"],
            "agentRunSuffix": agent_run[len(stamp):],
            "agentFileName": os.path.basename(saved["path"]),
            "agentName": variant.name,
            "extractRun": os.path.basename(extract_dir),
            "sweepRun": sweep_name,
            "vectorArtifactID":
                "runs/20260726T000001000-extract/fear",
            "experimentHash": manifest.content_hash(),
            "stimulusSetHash": stimulus_hash,
        })
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def system_prompt_composition() -> None:
    """The composition contract both engines must agree on byte for byte
    (maintainer ruling, 2026-08-24): the effective prompt, its hash, the
    additive stamps, and the comparability advisory's exact wording.

    Cases deliberately cover the whitespace-only and empty-string spellings of
    "no system prompt" — the two engines trim differently by default, and this
    is where that would show — plus non-ASCII text, because the hash is over
    raw UTF-8 bytes on both sides.

    ``panelCasting`` is the panel half of the same ruling: a cast seat's second
    term is the CAST ENTRY's role text rather than the study frame, so it
    stamps ``{"agent": …, "cast": …}``. Same primitive, same order.
    """
    from steerlab_server.experiment import system_prompt as sp

    cases = []
    for label, agent, study in (
        ("both", "You are Adjudicator-7.", "Respond in JSON."),
        ("agent-only", "You are Adjudicator-7.", None),
        ("agent-only-empty-frame", "You are Adjudicator-7.", ""),
        ("agent-only-blank-frame", "You are Adjudicator-7.", "   "),
        ("frame-only-null-agent", None, "Respond in JSON."),
        ("frame-only-empty-agent", "", "Respond in JSON."),
        ("frame-only-blank-agent", "   ", "Respond in JSON."),
        ("neither", None, None),
        ("neither-empty", "", ""),
        # The both-empty canonicalization (review 2026-08-26): every spelling
        # of "nothing on either level" composes to null, whichever side the
        # whitespace is on. These three used to disagree with each other.
        ("neither-blank", "   ", "   "),
        ("blank-agent-null-frame", "   ", None),
        ("null-agent-blank-frame", None, "   "),
        ("untrimmed-both", "  padded persona  ", "  padded frame  "),
        ("multiline-agent", "line one\nline two", "Respond in JSON."),
        ("non-ascii", "Tu es un juge — précis.", "Réponds en JSON…"),
        ("already-blank-line", "persona\n\n", "frame"),
    ):
        effective = sp.compose(agent, study)
        cases.append({
            "label": label,
            "agent": agent,
            "study": study,
            "effective": effective,
            "effectiveHash": sp.text_hash(effective),
            "studyStamp": sp.composition(agent, study),
            "batteryStamp": sp.composition(agent, study, frame_key="battery"),
        })

    # PANEL CASTING (same ruling, same order). A seat's second term is the
    # CAST ENTRY's role text, not the study frame — the study frame reaches no
    # panel turn at all — so the stamp is spelled `cast`. Persona first, role
    # second, through the SAME primitive: one uniform rule everywhere.
    # `castOnlyNullPersona` is today's dominant case (every agent in the
    # workspace has an empty persona) and is the legacy byte-identity lock.
    # `untrimmedCast` is where the two engines used to DISAGREE — the server
    # trimmed the cast text, Swift did not — so it is pinned deliberately.
    castings = []
    for label, persona, cast in (
        ("persona-and-cast", "You are Adjudicator-7.", "You represent Team South."),
        ("cast-only-null-persona", None, "You represent Team South."),
        ("cast-only-empty-persona", "", "You represent Team South."),
        ("cast-only-blank-persona", "   ", "You represent Team South."),
        ("persona-only-null-cast", "You are Adjudicator-7.", None),
        ("persona-only-empty-cast", "You are Adjudicator-7.", ""),
        ("persona-only-blank-cast", "You are Adjudicator-7.", "   "),
        ("neither", None, ""),
        ("neither-null", None, None),
        ("neither-blank", "  ", "   "),
        ("null-persona-blank-cast", None, "   "),
        ("untrimmed-cast", "You are Adjudicator-7.", "  padded role  "),
        ("multiline-persona", "line one\nline two", "You represent Team South."),
        ("non-ascii", "Tu es un juge — précis.", "Tu représentes l'équipe Sud…"),
    ):
        effective = sp.compose(persona, cast)
        castings.append({
            "label": label,
            "agent": persona,
            "cast": cast,
            "effective": effective,
            "effectiveHash": sp.text_hash(effective),
            "stamp": sp.composition(persona, cast, frame_key="cast"),
        })

    advisories = []
    for label, arms in (
        ("all-identical", [("baseline", "Respond in JSON."),
                           ("agent-x", "Respond in JSON.")]),
        ("all-bare", [("baseline", None), ("agent-x", None)]),
        ("single-arm", [("baseline", "Respond in JSON.")]),
        ("persona-on-one-arm",
         [("baseline", "Respond in JSON."),
          ("agent-x", sp.compose("You are Adjudicator-7.", "Respond in JSON."))]),
        ("bare-versus-framed", [("bare", None), ("framed", "Respond in JSON.")]),
        ("three-way",
         [("baseline", "Respond in JSON."), ("agent-x", "persona"),
          ("agent-y", None)]),
    ):
        advisories.append({
            "label": label,
            "arms": [{"name": n, "systemPrompt": t} for n, t in arms],
            "advisory": sp.divergence_advisory(arms),
        })

    _write(os.path.join(FIXTURES, "system-prompt-composition.json"),
           {"note": "produced by Server experiment/system_prompt.py — do not "
                    "hand-edit; regenerate with "
                    "scripts/regenerate-cross-engine-fixtures.py",
            "joiner": sp.JOINER,
            "composition": cases,
            "panelCasting": castings,
            "advisories": advisories})


#: Ids shaped like a Gemma render, shared verbatim with the Swift consumer
#: (``ExtractionRenderingAndPositionsTests``). A fixture that carried only
#: labels would pin vocabulary; carrying TOKEN IDS pins the arithmetic.
_BOS, _START, _END, _NEWLINE, _MODEL, _USER = 2, 105, 106, 107, 108, 109


class _FixtureTokenizer:
    """The template-map questions a reading position asks, answered the way a
    Gemma tokenizer answers them. No model, no download."""

    eos_token_id = _END
    unk_token_id = 3
    _VOCAB = {"<bos>": _BOS, "<start_of_turn>": _START, "<end_of_turn>": _END,
              "\n": _NEWLINE, "model": _MODEL, "user": _USER}
    _PIECES = {v: k for k, v in _VOCAB.items()}

    def convert_tokens_to_ids(self, token):
        return self._VOCAB.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        return self._PIECES.get(token_id, f"tok{token_id}")


def extraction_rendering_and_positions() -> None:
    """The rendering VOICE and the content-coordinate reading positions.

    Produced here and consumed by Swift, for the usual reason (B1): a Swift
    test that hand-writes what it believes the server emits pins the Swift
    author's belief. Three contracts travel:

    1. **What a declaration canonicalizes to** — including the two spellings
       that must canonicalize to NOTHING (`{"mode":"raw"}` and an explicit
       `"voice":"user"`), which is the hash-compatibility rule stated as data.
    2. **What each reading position contributes to a recipe identity.**
    3. **Where each position RESOLVES on a fixed token sequence** — the
       content mask included, so the two engines' template maps are compared
       on arithmetic rather than on prose.
    """
    from steerlab_server.experiment import recipe_identity as ri
    from steerlab_server.steering import extraction_rendering as er
    from steerlab_server.steering import reading_position as rp

    declarations = [
        ("absent", None),
        ("raw", {"mode": "raw"}),
        ("chatTemplate", {"mode": "chatTemplate"}),
        ("explicitUserVoice", {"mode": "chatTemplate", "voice": "user"}),
        ("assistantVoice", {"mode": "chatTemplate", "voice": "assistant"}),
        # The LEGACY boolean spelling, kept as an input on purpose: a recipe
        # declared under it must keep parsing (true ≡ xhigh) and must keep
        # its identity fragment byte-identical to what it hashed before the
        # effort existed. Both engines are pinned to that by this entry.
        ("assistantVoiceThinking", {"mode": "chatTemplate",
                                    "voice": "assistant",
                                    "qwenThinkingEnabled": True}),
        ("legacyThinkingTrue", {"mode": "chatTemplate",
                                "qwenThinkingEnabled": True}),
        # The effort spelling (2026-09-03): xhigh hashes exactly as the legacy
        # true did; low/medium add the one key the boolean cannot express.
        ("reasoningEffortXhigh", {"mode": "chatTemplate",
                                  "reasoningEffort": "xhigh"}),
        ("reasoningEffortLow", {"mode": "chatTemplate",
                                "reasoningEffort": "low"}),
        ("reasoningEffortMediumSystem", {"mode": "chatTemplate",
                                         "reasoningEffort": "medium",
                                         "systemPrompt": "be brief"}),
        ("systemPrompt", {"mode": "chatTemplate", "systemPrompt": "be brief"}),
    ]
    renderings = []
    for label, declaration in declarations:
        parsed = er.parse_declaration(declaration)
        renderings.append({
            "label": label,
            "declaration": declaration,
            "stamp": parsed.to_dict() if parsed is not None else None,
            "identityFragment": ri.rendering_fragment(parsed),
            "humanLabel": (parsed or er.RAW_RENDERING).label,
        })

    positions = [
        rp.LAST_TOKEN, rp.mean_from_token(50), rp.offset_from_end(0),
        rp.offset_from_end(3), rp.LAST_CONTENT_TOKEN, rp.TURN_CLOSE_TOKEN,
        rp.post_instruction(2), rp.content_offset(0), rp.content_offset(2),
        rp.mean_content_from_token(0), rp.mean_content_from_token(1),
    ]
    position_cases = []
    for position in positions:
        mode, parameter = ri.canonical_reading(position)
        position_cases.append({
            "label": position.label,
            "identityMode": position.identity_mode,
            "identityParameter": position.identity_parameter,
            "canonicalMode": mode,
            "canonicalParameter": parameter,
            "minimumTokenCount": position.minimum_token_count,
            "requiresTemplatedRendering": position.requires_templated_rendering,
        })

    # `<bos><start_of_turn>user\n` · content 4…6 · `<end_of_turn>` ·
    # `\n<start_of_turn>model\n` — one uniform Gemma-shaped render.
    tokens = [_BOS, _START, _USER, _NEWLINE, 201, 202, 203, _END,
              _NEWLINE, _START, _MODEL, _NEWLINE]
    tokenizer = _FixtureTokenizer()
    resolutions = []
    for position in positions:
        for raw in (True, False):
            if raw and position.requires_templated_rendering:
                continue
            if position.minimum_token_count > len(tokens):
                continue    # a short-sequence refusal, pinned by unit tests
            resolved = position.resolve(tokens, tokenizer=tokenizer,
                                        rendering_is_raw=raw)
            resolutions.append({
                "label": position.label,
                "renderingIsRaw": raw,
                "startIndex": resolved.start_index,
                "endIndex": resolved.end_index,
                "source": resolved.source,
                "pooledIndices": (list(resolved.pooled_indices)
                                  if resolved.pooled_indices else None),
                "maskedTokenCount": resolved.masked_token_count,
                "pooledTokenCount": resolved.pooled_token_count,
            })

    _write(os.path.join(FIXTURES, "extraction-rendering-and-positions.json"), {
        "tokens": tokens,
        "tokenIDs": {"bos": _BOS, "startOfTurn": _START, "endOfTurn": _END,
                     "newline": _NEWLINE, "model": _MODEL, "user": _USER},
        "contentIndices": rp.content_indices(tokens, tokenizer, "fixture"),
        "renderings": renderings,
        "positions": position_cases,
        "resolutions": resolutions,
        "refusals": {
            "assistantVoiceGenerationPrompt":
                er.ASSISTANT_VOICE_GENERATION_PROMPT_REASON,
            "assistantVoiceSystemPrompt":
                er.ASSISTANT_VOICE_SYSTEM_PROMPT_REASON,
            # The MISSPELLING refusal, pinned as a produced instance rather
            # than as a template: `addGenerationPromt` is the transposition
            # that used to be read as "nothing declared" and silently left the
            # default in place. The key vocabulary the repair names travels
            # with it, so a key added on one engine and not the other shows up
            # as a fixture diff.
            "unknownChatTemplateKey":
                er.unknown_chat_template_key_reason(["addGenerationPromt"]),
            # The RAW branch's own stranger refusal, pinned the same way. Both
            # engines refuse it while DECLARING and while READING a recorded
            # block (round-5 review): a parameter under raw reaches no
            # template at all, so accepting it on either path would let one
            # engine read a manifest the other refuses.
            "rawParameters": er.raw_parameters_reason(
                ["addGenerationPrompt", "voice"]),
            # The reasoning-effort refusals (2026-09-03), pinned the same
            # way: two spellings of one parameter, a value outside the closed
            # vocabulary, and a non-off effort on a family whose template has
            # no thinking mode (raised where the model id is known).
            "bothThinkingKeys": er.BOTH_THINKING_KEYS_REASON,
            "unknownEffort": er.unknown_effort_reason("hgih"),
            "effortWithoutThinkingMode": er.effort_without_thinking_mode_reason(
                "low", "google/gemma-3-4b-it"),
        },
        "chatTemplateKeys": list(er.CHAT_TEMPLATE_KEYS),
        "reasoningEfforts": list(er.REASONING_EFFORTS),
    })


def paired_difference_pca() -> None:
    """The PCA path's NUMBERS, produced by the Python engine and asserted by
    the Swift one (audit finding 6).

    Until now the two engines' PCA agreed only by construction — same
    algorithm, same two starts, same float32 — and nothing checked it. The
    deterministic Gram power-iteration is exactly the kind of code where a
    "harmless" refactor on one side (a different start, an `np.linalg.svd`
    swap, a changed convergence test) silently produces a DIFFERENT unit
    vector, with the same norm and a plausible cosine, on artifacts nobody
    re-derives. These cases pin the bytes.

    Three of them matter beyond coverage:

    * ``alternating`` is the case that motivated the third start. Its rows are
      ±d, so the uniform start is exactly orthogonal to the dominant
      eigenvector; before 2026-08-27 the iteration rode float noise from there.
    * ``three-row-ramp-degenerate`` is orthogonal to BOTH the uniform and the
      ramp start at once — the case that made the honest degenerate-start
      guard refuse until the alternating start was added.
    * ``paired-difference-direction`` is the whole family end to end, sign rule
      and norm-matching included.
    """
    from steerlab_server.steering import vector_math as vm

    def rows(seed: int, n: int, d: int) -> list[list[float]]:
        rng = np.random.default_rng(seed)
        return [[round(float(x), 6) for x in rng.standard_normal(d)]
                for _ in range(n)]

    pca_cases = []
    for label, matrix in (
        ("random-5x6", rows(11, 5, 6)),
        ("random-8x4", rows(23, 8, 4)),
        # ±d rows: the uniform start is exactly orthogonal to PC1.
        ("alternating", [[1.0, 2.0, -0.5, 0.25], [-1.0, -2.0, 0.5, -0.25],
                         [1.0, 2.0, -0.5, 0.25], [-1.0, -2.0, 0.5, -0.25]]),
        # Three ±d rows: orthogonal to the uniform AND the ramp start.
        ("three-row-ramp-degenerate",
         [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    ):
        pca_cases.append({
            "label": label,
            "rows": matrix,
            "firstPrincipalComponent": vm.first_principal_component(matrix),
        })

    direction_cases = []
    for label, positive, negative in (
        ("clean-separation",
         [[3.0, 0.0], [3.2, 0.1], [2.8, -0.1]],
         [[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1]]),
        ("noisy-6d", rows(31, 5, 6), rows(37, 5, 6)),
    ):
        direction_cases.append({
            "label": label,
            "positive": positive,
            "negative": negative,
            "meanDifference": vm.mean_difference(positive, negative),
            "pairedDifferencePCA": vm.direction(
                positive, negative, vm.ExtractionMethod.PAIRED_DIFFERENCE_PCA),
        })

    # --- F5: convergence health of the same iteration -----------------------
    # Committed ROWS, so both engines run the diagnostic over identical bytes.
    # The near-degenerate case deliberately does NOT pin a component: an
    # ill-determined PC1 is the one direction the two engines are not expected
    # to agree on to fixture precision, which is exactly what the diagnostic
    # exists to announce.
    def spectrum(ratio: float, n: int, d: int, seed: int) -> list[list[float]]:
        rng = np.random.default_rng(seed)
        basis, _ = np.linalg.qr(rng.standard_normal((d, d)))
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        a -= a.mean()
        b -= b.mean()
        a /= np.sqrt((a ** 2).sum())
        b /= np.sqrt((b ** 2).sum())
        matrix = (np.outer(a, basis[:, 0])
                  + np.sqrt(ratio) * np.outer(b, basis[:, 1]))
        return [[round(float(x), 6) for x in row] for row in matrix]

    iteration_cases = []
    for label, matrix in (
        ("well-separated", rows(11, 6, 8)),
        # 5.2% sample eigengap: uses the whole iteration budget without meeting
        # the float32 delta tolerance, and is STILL accurate to six figures
        # (|cos| with the true PC1 = 1.000000, residual 3.1e-6). The case that
        # rules out thresholding `converged` instead of the residual.
        ("five-percent-eigengap", spectrum(0.95, 16, 12, 3)),
        # 1.6% sample eigengap — the audit's failure regime, reproduced small:
        # residual 5.2e-3, |cos| with the true PC1 down to 0.94 with 0.34
        # leaked into the true SECOND eigenvector, returned deterministically
        # and, before this diagnostic, with no warning at all.
        ("near-degenerate", spectrum(0.9945, 16, 12, 3)),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _component, diagnostic = vm.first_principal_component_with_diagnostic(
                matrix)
        iteration_cases.append({
            "label": label,
            "rows": matrix,
            "converged": diagnostic.converged,
            "illConditioned": diagnostic.ill_conditioned,
            # Order of magnitude only: the residual is a float32 accumulation
            # and the two engines sum it in different orders. What must agree
            # is which side of the threshold it lands on.
            "relativeResidualUpperBound": float(
                f"{diagnostic.relative_residual * 10:.1g}"),
        })

    _write(os.path.join(FIXTURES, "paired-difference-pca.json"), {
        "note": "Produced by the Python engine; asserted by the Swift one. "
                "A diff means one engine's PCA path changed — decide "
                "deliberately whether the other should follow.",
        "degenerateStartRelativeThreshold":
            vm.DEGENERATE_START_RELATIVE_THRESHOLD,
        "principalComponents": pca_cases,
        "directions": direction_cases,
        "powerIteration": {
            "residualWarnThreshold": vm.POWER_ITERATION_RESIDUAL_WARN_THRESHOLD,
            "maxIterations": vm.POWER_ITERATION_MAX_ITERATIONS,
            "deltaTolerance": vm.POWER_ITERATION_DELTA_TOLERANCE,
            "warningExample": {
                "relativeResidual": 0.006171,
                "iterations": 200,
                "converged": False,
                "message": vm.power_iteration_warning(
                    vm.PowerIterationDiagnostic(
                        relative_residual=0.006171, iterations=200,
                        converged=False)),
            },
            "cases": iteration_cases,
        },
    })


def concept_stats_splits() -> None:
    """The screening diagnostics' SPLIT MEMBERSHIP, produced by the Python
    engine and asserted by the Swift one (2026-08-28 audit, F3).

    Before this fixture the two engines split differently and neither split was
    pinned: Python held out the last ~20% of each class in FILE order and cut
    even/odd halves; Swift held out ``index % 5 == 4``. Both numbers therefore
    moved with how the stimulus file happened to be ordered, and the same data
    scored differently on the two engines. The rule now is content-derived —
    sort each class's rows ascending by the lowercase SHA-256 hex of the row's
    UTF-8 text, hold out sorted position % 5 == 4, halve on sorted position
    parity — so this fixture pins membership by TEXT, and the cases are chosen
    to make order-independence falsifiable:

    * ``topic-blocked`` is authored the way real stimulus files are, in topic
      runs. Under the old rule the held-out set was the last topic block
      entire; under this one it is spread across the blocks.
    * ``topic-blocked-scrambled`` is the SAME texts in a different order. Its
      expected membership is the same SET of texts, which is the property the
      whole change exists to buy.
    * ``duplicates`` pins the tie-break: two identical texts hash identically
      and are ordered by the text itself, so which of them lands in the test
      set cannot depend on file order either.
    """
    from steerlab_server.experiment import concept_stats as cs

    topics = [f"{topic} sentence {i}" for topic in ("harbour", "kitchen", "ledger")
              for i in range(5)]
    scrambled = [topics[i] for i in (11, 3, 14, 0, 7, 2, 9, 13, 5, 1, 10, 4, 12, 8, 6)]

    cases = []
    for label, texts in (
        ("topic-blocked", topics),
        ("topic-blocked-scrambled", scrambled),
        ("duplicates", ["same", "same", "b", "c", "d", "e", "f", "g"]),
        ("unicode", ["café", "naïve", "日本語", "straße", "emoji 🌊", "plain",
                     "tab\tinside", "newline\nInside"]),
    ):
        cases.append({
            "label": label,
            "texts": list(texts),
            "contentHashOrder": cs.content_hash_order(texts),
            "heldOutTexts": sorted(texts[i] for i in cs.held_out_indices(texts)),
            "splitHalfSecondTexts":
                sorted(texts[i] for i in cs.split_half_indices(texts)),
        })

    _write(os.path.join(FIXTURES, "concept-stats-splits.json"), {
        "note": "Produced by the Python engine; asserted by the Swift one. "
                "Membership is derived from the TEXTS alone — a diff means one "
                "engine's screening-split rule changed.",
        "minimumRowsPerClass": cs.MINIMUM_ROWS_PER_CLASS,
        "minimumRowsPerClassSplitHalf": cs.MINIMUM_ROWS_PER_CLASS_SPLIT_HALF,
        "cases": cases,
    })


# --- model capabilities (2026-09-05) ------------------------------------------

#: The three template families the probe was built against, as SYNTHETIC
#: templates that reproduce exactly the behaviours the real ones were
#: verified to have (Qwen/Qwen3.8-27B, Qwen/Qwen3-14B, google/gemma-3-27b-it)
#: plus the one family no cached model exhibits (a template that raises on a
#: system turn). Synthetic on purpose: the fixture pins the PROBE LOGIC on
#: both engines through their own Jinja engines, without vendoring a model's
#: template; the real templates are pinned by the live checks in
#: `test_model_capabilities.py` (skipped when the model is not cached).
#: Every newline in a template below is a RENDERED byte (`{% %}`, never
#: `{%- %}`), so the generation prompts end exactly as the real families'
#: do: `<|im_start|>assistant\n` and `<start_of_turn>model\n`.
_CAPABILITY_FAMILIES = [
    {
        "label": "chatml-effort",
        "like": "Qwen/Qwen3.8-27B",
        "specialTokens": {},
        "thinkTokenIDs": {"<think>": 11, "</think>": 12},
        "template": (
            "{% if reasoning_effort is defined %}"
            "{% if reasoning_effort not in ['low', 'medium', 'xhigh'] %}"
            "{{ raise_exception('unknown reasoning_effort: ' ~ reasoning_effort) }}"
            "{% endif %}{% set effort = reasoning_effort %}"
            "{% else %}{% set effort = 'xhigh' %}{% endif %}"
            "{% set thinking = enable_thinking is not defined or enable_thinking %}"
            "{% for message in messages %}"
            "{% if message.role == 'system' %}"
            "<|im_start|>system\n{{ message.content }}"
            "{% if thinking %}\nReasoning effort: {{ effort }}{% endif %}"
            "<|im_end|>\n"
            "{% elif message.role == 'user' %}"
            "<|im_start|>user\n{{ message.content }}<|im_end|>\n"
            "{% elif message.role == 'assistant' %}"
            "<|im_start|>assistant\n{{ message.content }}<|im_end|>\n"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n"
            "{% if thinking %}<think>\n{% else %}<think>\n\n</think>\n\n{% endif %}"
            "{% endif %}"),
    },
    {
        "label": "chatml-switch",
        "like": "Qwen/Qwen3-14B",
        "specialTokens": {},
        "thinkTokenIDs": {"<think>": 11, "</think>": 12},
        "template": (
            "{% for message in messages %}"
            "{% if message.role == 'system' %}"
            "<|im_start|>system\n{{ message.content }}<|im_end|>\n"
            "{% elif message.role == 'user' %}"
            "<|im_start|>user\n{{ message.content }}<|im_end|>\n"
            "{% elif message.role == 'assistant' %}"
            "<|im_start|>assistant\n{{ message.content }}<|im_end|>\n"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n"
            "{% if enable_thinking is defined and enable_thinking is false %}"
            "<think>\n\n</think>\n\n{% endif %}{% endif %}"),
    },
    {
        "label": "gemma-fold",
        "like": "google/gemma-3-27b-it",
        "specialTokens": {"bos_token": "<bos>"},
        "thinkTokenIDs": {},
        "template": (
            "{{ bos_token }}"
            "{% if messages[0]['role'] == 'system' %}"
            "{% set first_user_prefix = messages[0]['content'] | trim + '\n\n' %}"
            "{% set loop_messages = messages[1:] %}"
            "{% else %}{% set first_user_prefix = '' %}"
            "{% set loop_messages = messages %}{% endif %}"
            "{% for message in loop_messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if message['role'] == 'assistant' %}{% set role = 'model' %}"
            "{% else %}{% set role = message['role'] %}{% endif %}"
            "{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else '') }}"
            "{{ message['content'] | trim }}<end_of_turn>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"),
    },
    {
        "label": "system-refused",
        "like": "a template that raises on a system turn",
        "specialTokens": {"bos_token": "<bos>"},
        "thinkTokenIDs": {},
        "template": (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ raise_exception('System role not supported') }}{% endif %}"
            "{{ '<start_of_turn>' + message['role'] + '\n' }}"
            "{{ message['content'] | trim }}<end_of_turn>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"),
    },
]

_CAPABILITY_PROBED_AT = "2026-09-05T00:00:00Z"


def model_capabilities() -> None:
    """The chat-template capability record: the probe's verdict on each
    synthetic family, the record's canonical hash, the heuristic fallback,
    and the refusal sentences — all produced here, asserted by Swift."""
    from steerlab_server.experiment import model_capabilities as mc
    from steerlab_server.experiment import prompt_render
    from steerlab_server.steering import extraction_rendering as er

    families = []
    records = []
    for family in _CAPABILITY_FAMILIES:
        ids = family["thinkTokenIDs"]
        record = mc.probe(
            mc.template_renderer(family["template"], family["specialTokens"]),
            model_id="fixture/" + family["label"], revision="0" * 40,
            think_token_id=lambda token, ids=ids: ids.get(token),
            architecture={"layerCount": 4, "hiddenSize": 8,
                          "layerTypes": ["full_attention"] * 4},
            template_sha256=mc.sha256_text(family["template"]),
            tokenizer_config_sha256=None,
            engine="python-hf-transformers", engine_version="fixture",
            probed_at=_CAPABILITY_PROBED_AT)
        families.append({
            "label": family["label"],
            "like": family["like"],
            "template": family["template"],
            "specialTokens": family["specialTokens"],
            "thinkTokenIDs": family["thinkTokenIDs"],
            "detected": record["detected"],
        })
        records.append({"label": family["label"], "record": record,
                        "recordHash": record["recordHash"]})

    # An override on the record: the effective view and its hash move, the
    # detected block does not.
    overridden = mc.set_override(
        dict(records[1]["record"]), "thinkOpenInPrompt", "true",
        "the model always opens with <think> on its own")
    overridden["overrides"]["thinkOpenInPrompt"]["setAt"] = _CAPABILITY_PROBED_AT
    overridden["recordHash"] = mc.record_hash(overridden)
    records.append({"label": "chatml-switch-overridden", "record": overridden,
                    "recordHash": overridden["recordHash"]})

    heuristics = []
    for model_id in ("Qwen/Qwen3-14B", "google/gemma-3-27b-it",
                     "meta-llama/Llama-3.1-8B-Instruct"):
        record = mc.heuristic(model_id, None)
        heuristics.append({"modelID": model_id, "record": record,
                           "recordHash": record["recordHash"]})

    chatml_effort = mc.effective(records[0]["record"])
    chatml_switch = mc.effective(records[1]["record"])
    gemma = mc.effective(records[2]["record"])
    qwen_heuristic = mc.effective(heuristics[0]["record"])
    summary_lines = [
        {"label": "chatml-effort", "lines": mc.effective(
            records[0]["record"],
            path="prompts/models/fixture--chatml-effort@" + "0" * 40 + ".json"
        ).summary_lines()},
        {"label": "gemma-fold", "lines": gemma.summary_lines()},
        {"label": "qwen-heuristic", "lines": qwen_heuristic.summary_lines()},
        {"label": "chatml-switch-overridden",
         "lines": mc.effective(overridden).summary_lines()},
    ]
    gates = [
        {"label": label, "effort": effort, "modelID": model_id,
         "record": record_label,
         "violations": prompt_render.reasoning_protocol_violations(
             effort=effort, reasoning_max_tokens=64, model_id=model_id,
             capabilities=view)}
        for label, effort, model_id, record_label, view in (
            ("level-accepted", "medium", "Qwen/Qwen3.8-27B", "chatml-effort", chatml_effort),
            ("level-rejected", "high", "Qwen/Qwen3.8-27B", "chatml-effort", chatml_effort),
            ("on-with-effort-control", "on", "Qwen/Qwen3.8-27B", "chatml-effort", chatml_effort),
            ("level-ignored", "medium", "Qwen/Qwen3-14B", "chatml-switch", chatml_switch),
            ("on-without-effort-control", "on", "Qwen/Qwen3-14B", "chatml-switch", chatml_switch),
            ("no-switch", "low", "google/gemma-3-27b-it", "gemma-fold", gemma),
            ("heuristic-assumed", "medium", "Qwen/Qwen3-14B", "heuristic", qwen_heuristic),
            ("heuristic-unprobed", "high", "Qwen/Qwen3-14B", "heuristic", qwen_heuristic),
        )
    ]
    kwargs = [
        {"label": label, "effort": effort, "record": record_label,
         "kwargs": prompt_render.thinking_template_kwargs(
             "fixture", effort, view)}
        for label, effort, record_label, view in (
            ("off", "off", "chatml-effort", chatml_effort),
            ("on", "on", "chatml-effort", chatml_effort),
            ("level", "medium", "chatml-effort", chatml_effort),
            ("ignored-level-renders-as-on", "medium", "chatml-switch", chatml_switch),
            ("legacy-xhigh-on-switch-only", "xhigh", "chatml-switch", chatml_switch),
            ("no-switch", "off", "gemma-fold", gemma),
            ("heuristic-level", "medium", "heuristic", qwen_heuristic),
        )
    ]
    _write(os.path.join(FIXTURES, "model-capabilities.json"), {
        "note": "Produced by the Python engine; asserted by the Swift one. "
                "Each family's template renders through the engine's own "
                "Jinja (jinja2 here, swift-jinja there); the probe verdicts, "
                "the record hashes, the heuristic fallback, the gate "
                "sentences and the template variables each effort becomes "
                "must agree byte for byte.",
        "schemaVersion": mc.SCHEMA_VERSION,
        "directory": mc.DIRECTORY,
        "probeUserText": mc.PROBE_USER_TEXT,
        "probeSystemText": mc.PROBE_SYSTEM_TEXT,
        "effortCandidates": list(mc.EFFORT_CANDIDATES),
        "effortProbeValue": mc.EFFORT_PROBE_VALUE,
        "reasoningEfforts": list(prompt_render.REASONING_EFFORTS),
        "reasoningLevels": list(prompt_render.REASONING_LEVELS),
        "families": families,
        "records": records,
        "recordFilenames": [
            {"modelID": "Qwen/Qwen3-14B", "revision": "a" * 40,
             "filename": mc.record_filename("Qwen/Qwen3-14B", "a" * 40)},
            {"modelID": "mlx-community/gemma-3-4b-it-4bit", "revision": None,
             "filename": mc.record_filename("mlx-community/gemma-3-4b-it-4bit", None)},
        ],
        "heuristics": heuristics,
        "summaryLines": summary_lines,
        "gates": gates,
        "templateKwargs": kwargs,
        "refusals": {
            "effortWithoutThinkingMode":
                prompt_render.effort_without_thinking_mode_reason(
                    "low", "google/gemma-3-27b-it"),
            "effortIgnored": prompt_render.effort_ignored_reason(
                "medium", "Qwen/Qwen3-14B"),
            "effortRejected": prompt_render.effort_rejected_reason(
                "high", "Qwen/Qwen3.8-27B", chatml_effort),
            "effortUnprobed": prompt_render.effort_unprobed_reason(
                "high", "Qwen/Qwen3-14B", qwen_heuristic),
            "effortAssumedAdvisory": prompt_render.effort_assumed_advisory(
                "medium", "Qwen/Qwen3-14B"),
            "systemPromptUnsupported":
                prompt_render.system_prompt_unsupported_reason("fixture/system-refused"),
            "heuristicAdvisory": mc.heuristic_advisory("Qwen/Qwen3-14B"),
            "extractionEffortWithoutThinkingMode":
                er.effort_without_thinking_mode_reason("low", "google/gemma-3-27b-it"),
            "extractionEffortLevelPrefix": er.effort_level_problem_prefix(),
        },
    })


def main() -> int:
    os.makedirs(FIXTURES, exist_ok=True)
    promotion_keys()
    paired_difference_pca()
    concept_stats_splits()
    extraction_rendering_and_positions()
    model_capabilities()
    system_prompt_composition()
    server_minted_agent()
    server_minted_adapter_agent()
    scenario_diagnostics()
    validation_layers()
    choice_margins()
    auto_prompt_ids()
    row_hashes()
    evidence_bundle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
