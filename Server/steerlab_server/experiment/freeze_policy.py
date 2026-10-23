"""Freeze gate decisions over a manifest and already-collected evidence.

No paths, file reads, model loads or store imports belong here. Evidence readers
preserve their own integrity checks; this policy owns gate scope and ordering.
"""
from __future__ import annotations
from dataclasses import dataclass
from . import command_vocabulary as vocabulary
from . import manifest_declaration_policy as declarations
from .manifest_errors import ExperimentStoreError

FORCED_GATE_IDS = ("revision", "validateEvidence", "batteryEvidence",
                   "judgeValidity", "variantValidity", "gitClean",
                   "measurementPins")

#: Freeze stamp (cross-engine manifest key): the variant conditions the
#: capability battery was NOT applied to, and why — a list of
#: ``{"condition": <name>, "reason": <reason id>}`` in manifest order. A
#: lifecycle stamp like ``freezeForced``: written at freeze, excluded from the
#: content hash and from ``freeze-canonical.json``, cleared on duplicate, and
#: absent when the battery applied to every condition (so every manifest
#: frozen before the key existed loads and hashes exactly as it did). It does
#: NOT mark the study forced. Swift twin:
#: ``ExperimentManifest.capabilityBatteryNotApplied``.
BATTERY_NOT_APPLIED_KEY = "capabilityBatteryNotApplied"

#: The one reason this engine records today: the condition's agent carries
#: intervention policies, which the battery cannot run
#: (``model_variant.variant_injections`` refuses them without
#: ``allow_policies``). The list shape leaves room for other reasons; none
#: exists, and nothing else is exempt.
BATTERY_REASON_INTERVENTION_POLICY = "interventionPolicy"


def battery_exemption_reason(vc) -> str | None:
    """Why the capability battery cannot be applied to one variant condition,
    or None when it applies.

    Read from the condition's INLINE agent artifact — the bytes the manifest
    pins and the content hash covers — so the answer needs no file read and
    cannot differ between the freeze that stamps it and a later reader. A
    forward-referenced condition has no agent yet and is handled by its own,
    older exemption. Swift twin: ``FreezePolicy.batteryExemptionReason``."""
    if not isinstance(vc, dict) or isinstance(vc.get("fromPromotion"), dict):
        return None
    artifact = vc.get("artifact")
    if isinstance(artifact, dict) and artifact.get("interventionPolicies"):
        return BATTERY_REASON_INTERVENTION_POLICY
    return None


def battery_not_applied(d: dict) -> list[dict]:
    """The ``capabilityBatteryNotApplied`` entries for a manifest dict, in
    manifest order; empty when the battery applies to every condition or the
    battery gate is not asked of this study kind at all."""
    if not declarations.model_output_surfaces_operative(d):
        return []
    entries = []
    for vc in d.get("variantConditions") or []:
        reason = battery_exemption_reason(vc)
        if reason is not None:
            entries.append({"condition": str(vc.get("name", "?")), "reason": reason})
    return entries


def has_static_steering(vc) -> bool:
    """Whether one variant condition's INLINE agent carries steering that does
    not depend on a policy: steering vectors or adapters. Attaching a policy
    keeps both (``policy_authoring.attachment_review``), and the battery can
    run them, so a policy agent that has any owes a reading of them alone.
    Swift twin: ``FreezePolicy.hasStaticSteering``."""
    if not isinstance(vc, dict):
        return False
    artifact = vc.get("artifact")
    if not isinstance(artifact, dict):
        return False
    return bool(artifact.get("injections")) or bool(artifact.get("adapters"))


def owes_static_part_reading(vc) -> bool:
    """Whether the battery gate asks for a reading of this condition's agent
    WITHOUT its policy: its agent uses an intervention policy (so the battery
    cannot read the agent as it runs) and also carries static steering (which
    the battery can read). A policy agent with no static steering owes
    nothing, exactly as before. Swift twin:
    ``FreezePolicy.owesStaticPartReading``."""
    return (battery_exemption_reason(vc) == BATTERY_REASON_INTERVENTION_POLICY
            and has_static_steering(vc))


#: Evidence-row key (cross-engine): the battery reading of a policy agent's
#: static steering alone, taken with its policies removed. It lives INSIDE
#: that condition's not-applicable row, as a sub-object —
#: ``{"condition", "batteryHash", "notApplicable": "interventionPolicy",
#: "withoutPolicy": {"total", "correct", "accuracy", "batteryFormat"?,
#: "armingIsolated"?}}`` — so the condition's own row keeps NO score keys,
#: and every reader that looks for ``accuracy`` on a row still finds none for
#: the agent as it runs. Shown everywhere under :func:`without_policy_label`.
#:
#: Which engine writes what when the reading cannot be taken: THIS engine
#: records ``"withoutPolicy": {"error": <what went wrong>}`` and carries on
#: with the other conditions, as it does with an ``error`` row for an agent
#: that fails to load. The Mac engine's validate throws instead and the whole
#: validation fails, as it does for any condition whose agent cannot load.
#: Both engines READ either shape, and an ``error`` sub-object is never a
#: score. Swift twin: ``CapabilityBatteryConditionResult.withoutPolicy``.
WITHOUT_POLICY_KEY = "withoutPolicy"

#: The first SteerLab release whose validate takes the static-part reading.
#: Evidence written by an older engine never carries ``withoutPolicy``, so
#: the gate says so rather than send a researcher to re-run an engine that
#: cannot produce it. Swift twin: ``FreezePolicy.staticPartReadingSince``.
STATIC_PART_READING_SINCE = "0.9.16"

#: Said with every refusal and repair that asks for a static-part reading.
#: Swift twin: ``FreezePolicy.staticPartReadingNote``.
STATIC_PART_READING_NOTE = (
    f"Only validate from SteerLab {STATIC_PART_READING_SINCE} or later takes "
    "a policy agent's static-part reading, so evidence from an older engine "
    "never carries it: a runner on an older engine must be updated before it "
    "validates again")


def without_policy_label(condition: str) -> str:
    """How a static-part reading is named wherever it appears. Swift twin:
    ``FreezePolicy.withoutPolicyLabel``."""
    return f"{condition} without its policy"


def without_policy_reading(row) -> dict | None:
    """The static-part reading inside one evidence row, or None. Read only on
    a not-applicable row — the one place either engine writes it — so a
    scored row can never carry a second score. Swift twin: the decoder of
    ``CapabilityBatteryConditionResult``."""
    if not isinstance(row, dict) or row.get("notApplicable") is None:
        return None
    reading = row.get(WITHOUT_POLICY_KEY)
    return reading if isinstance(reading, dict) else None


def static_part_scored(row) -> bool:
    """Whether one evidence row carries a SCORED static-part reading: a
    not-applicable row whose ``withoutPolicy`` has an accuracy and no
    ``error``. Swift twin: ``WithoutPolicyReading.isScored``."""
    reading = without_policy_reading(row)
    return (reading is not None and reading.get("error") is None
            and reading.get("accuracy") is not None)


def battery_not_applied_sentence(condition: str, reason: str) -> str:
    """The researcher-facing sentence for one entry. Both engines emit it
    verbatim (Swift twin: ``FreezePolicy.batteryNotAppliedSentence``)."""
    if reason == BATTERY_REASON_INTERVENTION_POLICY:
        return (f"The capability battery was not applied to {condition}, "
                "because its agent uses an intervention policy, which the "
                "battery cannot run. This study has no capability control "
                "for that agent as it runs. Any reading of its static "
                "steering alone, labelled "
                f"\"{without_policy_label(condition)}\", describes that "
                "steering, not the agent as it runs.")
    return (f"The capability battery was not applied to {condition} "
            f"(recorded reason: {reason}). This study has no capability "
            "control for that agent.")


def forced_freeze_summary_lines(d: dict) -> list[str]:
    """The settings summary's "Freeze" section for a forced freeze, as lines
    (a leading blank line included); empty when the freeze was not forced, so
    every other summary is byte-for-byte what it was. Facts only: that force
    was used, which gates it skipped, and where that is recorded. Swift twin:
    ``FreezePolicy.forcedFreezeSummaryLines``."""
    if not d.get("freezeForced"):
        return []
    skipped = [str(g) for g in d.get("forcedGatesSkipped") or []]
    said = (", ".join(f"`{g}`" for g in skipped) if skipped
            else "none (every gate would have passed)")
    return ["", "## Freeze", "",
            "- Frozen with force (`--force`). Gates skipped: " + said + ".",
            "- The manifest records this permanently, in `freezeForced` and "
            "`forcedGatesSkipped`."]


#: What the battery gate asks of each condition, said once for every repair
#: and refusal that names the validate step. Swift twin:
#: ``FreezePolicy.batteryScopePhrase``.
BATTERY_SCOPE_PHRASE = (
    "each condition the battery can run is scored; a condition whose agent "
    "uses an intervention policy is recorded as not applicable, and its "
    "static steering, when it has any, is scored without the policy")


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
        failures.append(("revision", (
            # The client loads no model, so "load it once" is not something
            # its reader can do; it names what is.
            f"cannot freeze '{name}': no model revision is pinned, and this "
            "client loads no model to resolve one. Pin the commit, or "
            "validate on a runner, which pins the commit it resolved when "
            "the evidence comes home"
            if vocabulary.is_client() else
            f"cannot freeze '{name}': model revision not pinned and "
            f"{d['modelID']} not in the local HF cache — load it once or freeze --force")))
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
            failures.append(("validateEvidence", (
                f"cannot freeze '{name}': no validation run in this workspace "
                "matches its exact pins. Validate it on a runner first "
                f"('{vocabulary.study_verb('validate', name)}'), or freeze "
                "with --force, which records in the frozen study, "
                "permanently, that this check was skipped"
                if vocabulary.is_client() else
                f"cannot freeze '{name}': no validate run matches its exact "
                f"pins — run 'steerlab-server experiment validate {name}' first, or force-freeze")))
        elif evidence.vacuous_validation:
            failures.append(("validateEvidence", evidence.vacuous_validation))
    if operative and d.get("variantConditions") and evidence.battery is not None:
        failures.append(("batteryEvidence", evidence.battery))
    if evidence.git is not None:
        failures.append(("gitClean", evidence.git))
    return failures


def freeze_gate_repair(gate: str, name: str, d: dict | None = None) -> str:
    """The repair for a freeze-gate refusal (:func:`_gate_repair`), plus,
    for the battery gate of a study ``d`` with a policy agent that owes a
    static-part reading, the fact that only a 0.9.16-or-later validate takes
    it. Swift twin: the battery gate's entry in ``freezeGateTable``."""
    repair = _gate_repair(gate, name)
    if gate == "batteryEvidence" and isinstance(d, dict) and any(
            owes_static_part_reading(vc)
            for vc in d.get("variantConditions") or []):
        repair += f". {STATIC_PART_READING_NOTE}"
    return repair


def _gate_repair(gate: str, name: str) -> str:
    """The RUNNABLE repair for a freeze-gate refusal, for whoever shows it.

    Gate-5 dry run #2 (P2/P3): every freeze-gate refusal here carried one
    boilerplate string — "satisfy the named gate, or freeze --force to record
    an explicitly non-citable experiment" — which names no command, and whose
    only concrete token (`freeze --force`) is a verb this CLI does not have.

    Each repair names commands its reader can run (:mod:`command_vocabulary`).
    Freezing, and every other change to the study, is an authoring act: the
    cross-platform client names its own verbs, and the engine — which cannot
    know which client the reader has — says "on your authoring client" and
    gives both spellings.

    The two evidence gates name the engine that can satisfy them.
    ``_matching_validate_evidence`` accepts only evidence stamped
    ``python-hf-transformers`` and there is no run-substrate seam here, so on
    the engine that is THIS engine's own ``validate``. The client loads no
    model: there the same evidence comes from a runner, and ``run <name>
    --verb validate`` accepts a draft for exactly that.
    """
    freeze_again = vocabulary.authoring(f"experiment freeze {name}")
    validate = vocabulary.study_verb("validate", name)
    pin_rubric = vocabulary.authoring(vocabulary.pin_rubric(
        name, "prompts/rubrics/default-paired-v1.md",
        judges="a:local[,b:claude]"))
    if vocabulary.is_client():
        pin_revision = vocabulary.authoring(
            f"experiment pin-revision {name} <commit>")
        attach_agent = vocabulary.authoring(
            f"experiment attach-agent {name} --artifact <path> "
            "--artifact-sha256 <digest> --manifest-sha256 <digest>")
        repairs = {
            "revision": f"{pin_revision}  (a full commit id, not a branch "
                        "name). Or validate on a runner, which pins the "
                        f"commit it resolved: {validate}. Then {freeze_again}",
            "measurementPins": "repoint the invalid measurement pin at a "
                               "loadable value (the reason names it), then "
                               f"{freeze_again}",
            "validateEvidence": f"{validate}  (a draft is accepted for this "
                                "step; the validation evidence comes home "
                                "into this workspace and is what this gate "
                                f"reads), then {freeze_again}",
            "variantValidity": "save the agent again with hashed adapter "
                               f"weights and attach it again ({attach_agent})"
                               f", then {freeze_again}",
            "batteryEvidence": f"{validate}  ({BATTERY_SCOPE_PHRASE}), "
                               f"then {freeze_again}",
            "judgeValidity": "declare the rubric file and the judges: "
                             f"{pin_rubric}  (or apply a study pack, which "
                             "pins the rubric's bytes for you), then "
                             f"{freeze_again}",
            "gitClean": "commit the pinned inputs in the workspace git "
                        f"repository, then {freeze_again}",
        }
        return repairs.get(
            gate, f"satisfy the '{gate}' gate, then {freeze_again}")
    create_pinned = vocabulary.authoring(
        (f"experiment create {name} --model <id> --revision <commit>",
         f"experiment pin-revision {name} <commit>"))
    repairs = {
        "revision": f"{create_pinned}; or load the model once here so it is "
                    f"cached. Then {freeze_again}",
        "measurementPins": "repoint the invalid measurement pin at a loadable "
                           f"value, then {freeze_again}",
        "validateEvidence": f"{validate}  "
                            "(this gate reads evidence stamped "
                            "python-hf-transformers; evidence from the other "
                            f"engine will not satisfy it), then {freeze_again}",
        "variantValidity": "re-save the variant with hashed adapter weights "
                           f"and re-attach it, then {freeze_again}",
        "batteryEvidence": f"{validate}  ({BATTERY_SCOPE_PHRASE}), "
                           f"then {freeze_again}",
        "judgeValidity": f"{pin_rubric}. Then {freeze_again}",
        "gitClean": "commit the pinned inputs in the workspace git repo, then "
                    f"{freeze_again}",
    }
    return repairs.get(
        gate, f"satisfy the '{gate}' gate, then {freeze_again}")


def admit_failures(name: str, failures: list[tuple[str, str]], *, force: bool,
                   d: dict | None = None) -> None:
    if failures and not force:
        raise ExperimentStoreError(
            failures[0][1], gate=failures[0][0],
            gates=tuple(g for g in FORCED_GATE_IDS if g in {gid for gid, _ in failures}),
            repair=freeze_gate_repair(failures[0][0], name, d))


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
    must contain capability-battery results for the baseline and every
    variant condition the battery can run, plus a reading of the static
    steering alone for each policy agent that has some, produced from the
    battery the manifest pins (or the live default battery when unpinned).
    Hash pins alone say the artifact bytes are stable; this says the variant
    still answers concept-unrelated probes."""
    results = {r.get("condition"): r
               for r in (evidence or {}).get("batteryResults") or []
               if isinstance(r, dict)}
    # Forward-referenced conditions (stage 4) are exempt: their agent does
    # not exist at validate time, so their battery evidence is produced by
    # the RUN's per-condition battery, not by freeze-time validation.
    # A condition whose agent carries intervention policies is exempt too
    # (maintainer ruling 2026-10-04): the battery cannot run a policy, so the
    # evidence could never exist and only --force could freeze the study.
    # The exemption is this narrow on purpose — baseline and every other
    # condition are required exactly as before — and freeze records it in the
    # ``capabilityBatteryNotApplied`` stamp instead of marking the study
    # forced.
    required = ["baseline"] + [
        vc.get("name", "?") for vc in d.get("variantConditions") or []
        if not isinstance(vc.get("fromPromotion"), dict)
        and battery_exemption_reason(vc) is None]
    # The engine's sentence names its own verb without a program, as it always
    # has; the client has no such verb, so it names its runner route.
    validate = (vocabulary.study_verb("validate", name)
                if vocabulary.is_client() else f"experiment validate {name}")
    missing = [c for c in required
               if c not in results or results[c].get("accuracy") is None]
    # A policy agent that also carries static steering owes a reading of that
    # steering alone (maintainer ruling 2026-10-23): attaching a policy no
    # longer exempts the agent's vectors and adapters from the check they
    # would otherwise get. The reading lives inside the condition's
    # not-applicable row under WITHOUT_POLICY_KEY, never as the condition's
    # own score, and a policy agent with no static steering owes nothing.
    owing = [vc.get("name", "?") for vc in d.get("variantConditions") or []
             if owes_static_part_reading(vc)]
    missing_static = [c for c in owing if not static_part_scored(results.get(c))]
    if missing or missing_static:
        owed = []
        if missing:
            owed.append("capability-battery results for condition(s) "
                        + ", ".join(missing))
        if missing_static:
            owed.append("capability-battery reading of "
                        + ", ".join(without_policy_label(c)
                                    for c in missing_static)
                        + " (its static steering alone)")
        raise ExperimentStoreError(
            f"cannot freeze '{name}': validate evidence has no "
            + " and no ".join(owed) + f" — run '{validate}' "
            f"({BATTERY_SCOPE_PHRASE}), or freeze --force"
            + (f". {STATIC_PART_READING_NOTE}" if missing_static else ""))
    expected = expected_hash
    if expected:
        drifted = sorted(
            [c for c in required
             if results[c].get("batteryHash") != expected]
            + [without_policy_label(c) for c in owing
               if results[c].get("batteryHash") != expected])
        if drifted:
            raise ExperimentStoreError(
                f"cannot freeze '{name}': capability battery drifted since "
                f"validation for condition(s) {', '.join(drifted)} — "
                f"re-run '{validate}', or freeze --force")
