"""Freeze gate decisions over a manifest and already-collected evidence.

No paths, file reads, model loads or store imports belong here. Evidence readers
preserve their own integrity checks; this policy owns gate scope and ordering.
"""
from __future__ import annotations
from dataclasses import dataclass
from . import manifest_declaration_policy as declarations
from .manifest_errors import ExperimentStoreError

FORCED_GATE_IDS = ("revision", "validateEvidence", "batteryEvidence",
                   "judgeValidity", "variantValidity", "gitClean",
                   "measurementPins")

@dataclass(frozen=True)
class FreezeEvidence:
    jlens: str | None = None
    variant: str | None = None
    judge: str | None = None
    validation_present: bool = False
    vacuous_validation: str | None = None
    battery: str | None = None
    git: str | None = None


def needs_validation(d: dict) -> bool:
    return bool(declarations.model_output_surfaces_operative(d)
                and (d.get("concepts") or d.get("conditions") or d.get("variantConditions"))
                and not declarations.optvec_exempt_from_validate_gate(d))


def evaluate(name: str, d: dict, evidence: FreezeEvidence) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    if not d.get("modelRevision"):
        failures.append(("revision",
            f"cannot freeze '{name}': model revision not pinned and "
            f"{d['modelID']} not in the local HF cache — load it once or freeze --force"))
    symbolic = declarations.symbolic_revision_problem(d)
    if symbolic:
        failures.append(("revision", f"cannot freeze '{name}': {symbolic}"))
    bad_dtype = declarations.unloadable_study_dtype_problem(d)
    if bad_dtype:
        failures.append(("measurementPins", f"cannot freeze '{name}': {bad_dtype}"))
    operative = declarations.model_output_surfaces_operative(d)
    if operative:
        if evidence.jlens is not None:
            failures.append(("measurementPins", evidence.jlens))
        if evidence.variant is not None:
            failures.append(("variantValidity", evidence.variant))
    if evidence.judge is not None:
        failures.append(("judgeValidity", evidence.judge))
    if needs_validation(d):
        if not evidence.validation_present:
            failures.append(("validateEvidence",
                f"cannot freeze '{name}': no validate run matches its exact "
                f"pins — run 'steerlab-server experiment validate {name}' first, or force-freeze"))
        elif evidence.vacuous_validation:
            failures.append(("validateEvidence", evidence.vacuous_validation))
    if operative and d.get("variantConditions") and evidence.battery is not None:
        failures.append(("batteryEvidence", evidence.battery))
    if evidence.git is not None:
        failures.append(("gitClean", evidence.git))
    return failures


def freeze_gate_repair(gate: str, name: str) -> str:
    """The RUNNABLE repair for a freeze-gate refusal on THIS engine.

    Gate-5 dry run #2 (P2/P3): every freeze-gate refusal here carried one
    boilerplate string — "satisfy the named gate, or freeze --force to record
    an explicitly non-citable experiment" — which names no command, and whose
    only concrete token (`freeze --force`) is a verb this CLI does not have.

    Each repair below names the engine that can actually satisfy its gate.
    The evidence gates name THIS one: ``_matching_validate_evidence`` accepts
    only evidence stamped ``python-hf-transformers`` and there is no
    run-substrate seam here, so evidence from the other engine can never
    satisfy them. Everything else is an authoring act, which is Mac-authority.
    """
    freeze_again = (f"steerlab-cli experiment freeze {name}  "
                    "(authoring is Mac-authority)")
    repairs = {
        "revision": f"steerlab-cli experiment create {name} --model <id> "
                    "--revision <commit> on the Mac, or load the model once "
                    f"here so it is cached, then {freeze_again}",
        "measurementPins": "repoint the invalid measurement pin at a loadable "
                           f"value on the Mac, then {freeze_again}",
        "validateEvidence": f"steerlab-server experiment validate {name}  "
                            "(this gate reads evidence stamped "
                            "python-hf-transformers; evidence from the other "
                            f"engine will not satisfy it), then {freeze_again}",
        "variantValidity": "re-save the variant with hashed adapter weights "
                           "and re-attach it on the Mac, then "
                           f"{freeze_again}",
        "batteryEvidence": f"steerlab-server experiment validate {name}  "
                           "(each variant condition runs the pinned battery), "
                           f"then {freeze_again}",
        "judgeValidity": f"steerlab-cli experiment pin-rubric {name} "
                         "prompts/rubrics/default-paired-v1.md --judges "
                         f"a:local[,b:claude] on the Mac, then {freeze_again}",
        "gitClean": "commit the pinned inputs in the workspace git repo, then "
                    f"{freeze_again}",
    }
    return repairs.get(
        gate, f"satisfy the '{gate}' gate, then {freeze_again}")


def admit_failures(name: str, failures: list[tuple[str, str]], *, force: bool) -> None:
    if failures and not force:
        raise ExperimentStoreError(
            failures[0][1], gate=failures[0][0],
            gates=tuple(g for g in FORCED_GATE_IDS if g in {gid for gid, _ in failures}),
            repair=freeze_gate_repair(failures[0][0], name))


def check_variant_validity(name: str, d: dict, *, evidence_grade_variants: set[int]) -> None:
    """Modality arms need a validity story (WORK-PLAN Phase E, promoted by the
    modality-axis decision): a variant condition's interventions must be fully
    pinned — adapter content hashes, system-prompt hash, vector artifact ids —
    or the condition is unverifiable the moment it freezes. ``freeze --force``
    skips this loudly, like the other evidence gates; the always-run verify()
    of the artifact-file hash is never skippable."""
    for index, vc in enumerate(d.get("variantConditions") or []):
        label = vc.get("name", "?")
        if isinstance(vc.get("fromPromotion"), dict):
            # Forward-referenced (stage 4): the artifact does not exist at
            # freeze time BY DESIGN — its pins land at run time and are
            # recorded in the run directory (forward-resolutions.json).
            # verify() enforces the declaration shape (attached concept,
            # exactly one identity).
            continue
        artifact = vc.get("artifact") or {}
        if not vc.get("artifactHash"):
            raise ExperimentStoreError(
                f"cannot freeze '{name}': variant '{label}' has no pinned "
                "artifactHash")
        for adapter in artifact.get("adapters") or []:
            if not adapter.get("adapterHash"):
                raise ExperimentStoreError(
                    f"cannot freeze '{name}': variant '{label}' adapter "
                    f"'{adapter.get('name', '?')}' has no adapterHash — re-save "
                    "the variant with hashed adapter weights, or freeze --force")
        if (artifact.get("systemPrompt") or "").strip() and \
                not artifact.get("systemPromptHash"):
            raise ExperimentStoreError(
                f"cannot freeze '{name}': variant '{label}' has a system prompt "
                "but no systemPromptHash — re-save the variant, or freeze --force")
        for injection in artifact.get("injections") or []:
            if not injection.get("vectorArtifactID"):
                raise ExperimentStoreError(
                    f"cannot freeze '{name}': variant '{label}' has an injection "
                    f"for '{injection.get('concept', '?')}' without a "
                    "vectorArtifactID pin")
        # Trained-adapter arms owe the same story about their TRAINING DATA
        # (LoRA readiness §0 amendment 1). An evidence-grade adapter whose
        # dataset is not pinned into the manifest is unverifiable the moment
        # it freezes: the training files could change afterwards with nothing
        # to flag the drift. Exploratory adapters are legal and produce an
        # advisory instead (freeze_advisories), never a refusal.
        if artifact.get("adapters") and index in evidence_grade_variants:
            block = vc.get("trainingProvenance")
            has_pin = isinstance(block, dict) and \
                str(block.get("datasetManifestHash") or "").strip()
            if not has_pin:
                raise ExperimentStoreError(
                    f"cannot freeze '{name}': variant '{label}' uses an "
                    "evidence-grade adapter but carries no "
                    "trainingProvenance.datasetManifestHash — its training "
                    "data would stay outside the freeze pin surface. Re-attach "
                    "the variant so freeze can pin the dataset manifest from "
                    "the adapter's sidecar, or freeze --force")


def check_battery_evidence(name: str, d: dict, evidence: dict | None,
                           *, expected_hash: str | None) -> None:
    """Variant freeze gate, evidence half: the scope-matched validate evidence
    must contain capability-battery results for the baseline and EVERY variant
    condition, produced from the battery the manifest pins (or the live
    default battery when unpinned). Hash pins alone say the artifact bytes are
    stable; this says the variant still answers concept-unrelated probes."""
    results = {r.get("condition"): r
               for r in (evidence or {}).get("batteryResults") or []
               if isinstance(r, dict)}
    # Forward-referenced conditions (stage 4) are exempt: their agent does
    # not exist at validate time, so their battery evidence is produced by
    # the RUN's per-condition battery, not by freeze-time validation.
    required = ["baseline"] + [
        vc.get("name", "?") for vc in d.get("variantConditions") or []
        if not isinstance(vc.get("fromPromotion"), dict)]
    missing = [c for c in required
               if c not in results or results[c].get("accuracy") is None]
    if missing:
        raise ExperimentStoreError(
            f"cannot freeze '{name}': validate evidence has no capability-"
            f"battery results for condition(s) {', '.join(missing)} — run "
            f"'experiment validate {name}' (each variant condition runs the "
            "pinned battery), or freeze --force")
    expected = expected_hash
    if expected:
        drifted = sorted(c for c in required
                         if results[c].get("batteryHash") != expected)
        if drifted:
            raise ExperimentStoreError(
                f"cannot freeze '{name}': capability battery drifted since "
                f"validation for condition(s) {', '.join(drifted)} — "
                f"re-run 'experiment validate {name}', or freeze --force")
