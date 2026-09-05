"""Judge roster resolution, provider preflight and fan-out request contracts.

No model acquisition and no dependency on task orchestration. Remote provider
checks retain the explicit transport seam and the existing offline policy.
"""
from __future__ import annotations
import json
import os
from .manifest import JudgeRef, Manifest

JUDGE_FANOUT_REQUEST_FILE = "judge-fanout-request.json"

def judge_roster(manifest: Manifest, spec) -> list[JudgeRef]:
    """The manifest's pinned judge panel, else the legacy single judge derived
    from ``evaluation.judgeModel`` (draft convenience — freeze requires >= 2)."""
    from . import paired_judge
    if manifest.judges:
        return list(manifest.judges)
    model = spec.judge_model or paired_judge.DEFAULT_JUDGE_MODEL
    kind = "claude" if paired_judge.is_claude_model(model) else "local"
    return [JudgeRef(name=model, kind=kind, model=model)]



def preflight_openrouter_judges(roster, log, *, transport=None) -> None:
    """Check every openrouter judge's provider pin BEFORE work starts.

    A wrong provider used to surface at the first judge call — after
    generation had finished and GPU hours were spent. The catalogue this
    checks against is public and keyless, so the check costs nothing and
    needs no credential.

    Refuses only on POSITIVE evidence (the catalogue answered and the pin
    is not among the serving endpoints). An unreachable catalogue warns and
    proceeds: compute nodes routinely have no outbound network, and a study
    must not become unrunnable because a metadata endpoint was down. The
    call-time off-pin refusal is still there either way — this moves the
    discovery earlier, it does not replace the guarantee.

    ``STEERLAB_SKIP_PROVIDER_PREFLIGHT=1`` disables the catalogue lookup.
    It exists for air-gapped sites that would rather not wait out a network
    timeout at every start, and it is what the test suite sets so no test
    reaches the real internet. Skipping is LOGGED — an unverified pin must
    never look like a verified one.
    """
    from . import paired_judge
    if os.environ.get("STEERLAB_SKIP_PROVIDER_PREFLIGHT", "").strip().lower() \
            in ("1", "true", "yes"):
        if any(getattr(r, "kind", "") == "openrouter" for r in roster):
            log("provider preflight SKIPPED "
                "(STEERLAB_SKIP_PROVIDER_PREFLIGHT) — openrouter provider "
                "pins stay unverified until the first judge call")
        return
    for ref in roster:
        if getattr(ref, "kind", "") != "openrouter":
            continue
        result = paired_judge.preflight_openrouter_provider(
            ref.model, ref.provider or "", transport=transport)
        for warning in result["warnings"]:
            log(f"WARNING: judge '{ref.name}': {warning}")
        if result["problem"]:
            raise RuntimeError(f"judge '{ref.name}': {result['problem']}")
        if result["checked"]:
            log(f"judge '{ref.name}': provider pin "
                f"'{paired_judge.canonical_openrouter_provider(ref.provider)}' "
                f"verified against OpenRouter's catalogue for '{ref.model}'")



def evaluate_fanout_judge_models(manifest: Manifest) -> list[dict]:
    """The distinct LOCAL judge models an evaluate fan-out needs, grouped by
    resolved model — ``[{model, revision, dtype, judges: [names]}]`` — or []
    when no local judge resolves to a model other than the study model (the
    inline path then suffices). When ANY local judge needs the fan-out, ALL
    local judges fan out (a study-model worker included), so the merged
    report has ONE judging clock. Revisions: the judge's own pin, else the
    study pin for study-model judges (JudgeRef contract, 2026-07-23)."""
    from . import sweep_selection
    locals_ = [ref for ref in (manifest.judges or []) if ref.kind == "local"]
    if not locals_:
        return []
    resolved = [(ref, sweep_selection.resolve_local_judge_model(
        ref.model, manifest.model_id)) for ref in locals_]
    if all(model == manifest.model_id for _ref, model in resolved):
        return []
    grouped: dict[tuple, dict] = {}
    for ref, model in resolved:
        revision = (ref.revision
                    or (manifest.model_revision if model == manifest.model_id
                        else None))
        dtype = ref.dtype
        key = (model, revision, dtype)
        entry = grouped.setdefault(key, {"model": model,
                                         "revision": revision,
                                         "dtype": dtype, "judges": []})
        entry["judges"].append(ref.name)
    return [grouped[key] for key in sorted(grouped, key=lambda k:
                                           (k[0], k[1] or "", k[2] or ""))]



def write_judge_fanout_request(pipeline_directory: str, name: str,
                               manifest: Manifest, awaiting_dir: str) -> dict:
    """The controller-facing fan-out request, written into the PIPELINE
    directory when the evaluate stage emitted packets for local judge
    workers: which awaiting run, which distinct judge models (with pinned
    revisions/dtypes and the judge names each covers), and the packet pin.
    The reconciler reads it off the continuation's child record and submits
    one worker job per entry."""
    with open(os.path.join(awaiting_dir, "judging-manifest.json"),
              encoding="utf-8") as handle:
        jm = json.load(handle)
    request = {
        "schema": 1,
        "experiment": name,
        "evaluateRun": os.path.basename(awaiting_dir),
        "pipelineDirectory": pipeline_directory,
        "packetsSha256": jm.get("packetsSha256"),
        "packetCount": jm.get("packetCount"),
        "judgeModels": evaluate_fanout_judge_models(manifest),
    }
    with open(os.path.join(pipeline_directory, JUDGE_FANOUT_REQUEST_FILE),
              "w", encoding="utf-8") as handle:
        json.dump(request, handle, indent=2, sort_keys=True)
    return request



def read_judge_fanout_request(pipeline_directory: str) -> dict | None:
    """The pipeline directory's fan-out request, or None."""
    path = os.path.join(pipeline_directory, JUDGE_FANOUT_REQUEST_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None
