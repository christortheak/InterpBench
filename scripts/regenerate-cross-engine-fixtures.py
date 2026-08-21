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


def main() -> int:
    os.makedirs(FIXTURES, exist_ok=True)
    promotion_keys()
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
