"""Manifest save and draft admission over supplied values, without filesystem IO."""
from __future__ import annotations
from . import lifecycle_gates
from .manifest_errors import ExperimentStoreError

ARM_BEARING_KEYS: tuple[str, ...] = ("concepts", "conditions")

def _clears_every_arm(existing: object, incoming: dict) -> bool:
    """True when this save would take a manifest that HOLDS a measured surface
    to one that holds none at all.

    Not "the document is empty" — a manifest legitimately starts that way and
    stays that way until the first attach. The refusable event is the
    TRANSITION: something on disk had concepts and/or conditions, and what is
    about to replace it has neither."""
    if not isinstance(existing, dict):
        return False
    had = any(existing.get(key) for key in ARM_BEARING_KEYS)
    # The INCOMING side also counts variantConditions: an agentComparison-
    # style save whose whole surface lives in variant conditions is not a
    # disarm — that study type's arms LIVE there. (The guard's first false
    # positive, test_transcript_study, caught at landing 2026-08-20.) Swift
    # twin: `ExperimentStore.holdsAnySurface`.
    clears = not any(
        incoming.get(key) for key in (*ARM_BEARING_KEYS, "variantConditions"))
    return had and clears


def admit_save(d: dict, existing: dict | None, *, freeze_transition: bool = False,
               clearing_arms: bool = False) -> None:
    name = d["name"]
    if existing is None or freeze_transition:
        return
    if (not clearing_arms and existing.get("status") == "draft"
            and _clears_every_arm(existing, d)):
        # Frozen/complete manifests never reach here — the status check
        # below refuses them outright — so this rule is DRAFT-only by
        # construction, not by an extra condition that could drift.
        raise ExperimentStoreError(
            f"refusing to save '{name}' with no concepts and no "
            f"conditions over a draft that has "
            f"{len(existing.get('concepts') or [])} concept(s) and "
            f"{len(existing.get('conditions') or [])} condition(s) — a "
            "manifest does not lose its whole measured surface in one "
            "write by accident",
            gate=lifecycle_gates.ARMS_CLEARED,
            repair=(
                f"steerlab-cli experiment verify {name}  "
                "# the manifest on disk still holds its arms; re-attach "
                "what the caller dropped (steerlab-cli experiment attach "
                f"{name} <concept>… ; steerlab-cli experiment "
                f"declare-condition {name} …), or author the cleared "
                "study as its own draft with steerlab-cli experiment "
                f"create {name}-v2 --model <id>"))
    if existing.get("status") == "frozen":
        # WP0 step 8: typed `statusImmutable`. `gate` here names a
        # LIFECYCLE gate, not a freeze gate — the two vocabularies are
        # disjoint by test, so the CLI's classifier reads it correctly and
        # an agent's `switch` over freeze gates cannot absorb it.
        raise ExperimentStoreError(
            f"'{name}' is frozen and read-only — duplicate it to iterate",
            gate=lifecycle_gates.STATUS_IMMUTABLE,
            repair=(f"steerlab-cli experiment duplicate {name} {name}-v2 "
                    "&& re-apply the change to the duplicate  "
                    "(authoring is Mac-authority)"))


def admit_freeze(name: str, d: dict) -> None:
    if d.get("status") != "draft":
        # Typed since gate-5 dry run #2 (P3): this was the last status guard
        # in the module left untyped, so re-freezing — the commonest possible
        # retry — answered `verbFailed`/70, indistinguishable from a crash,
        # while `save_raw` and `confirmation.attach_perturbations` already
        # said `statusImmutable`/65 with a runnable repair. Prose unchanged.
        raise ExperimentStoreError(
            f"'{name}' is already {d.get('status')}",
            gate=lifecycle_gates.STATUS_IMMUTABLE,
            repair=(f"steerlab-cli experiment duplicate {name} {name}-v2 && "
                    f"steerlab-cli experiment freeze {name}-v2  "
                    "(a frozen study is immutable; the duplicate is a draft "
                    "again, and authoring is Mac-authority)"))


def admit_draft_document(name: str, document: object) -> None:
    if not isinstance(document, dict) or not document:
        raise ExperimentStoreError("manifest body must be a JSON object")
    if document.get("name") != name:
        raise ExperimentStoreError(
            f"manifest body names {document.get('name')!r} but the route "
            f"names '{name}' — refusing an ambiguous sync")
    if document.get("status") != "draft":
        raise ExperimentStoreError(
            "only a DRAFT manifest can be pushed as the server's copy — "
            "frozen manifests are stamped by the server's own gated freeze, "
            "never installed by upload (duplicate to iterate)")


def admit_draft_replacement(name: str, existing: object) -> None:
    if isinstance(existing, dict) and existing.get("status") == "frozen":
        raise ExperimentStoreError(
            f"refusing to overwrite frozen manifest '{name}' with a "
            "pushed draft (freeze firewall) — duplicate to iterate")


def _merge_server_pins(document: dict, existing: object) -> dict:
    """Merge server-side auto-pins the incoming document OMITS (key absent)
    into ``document`` in place; returns the ``preserved`` report (empty =
    nothing merged). Explicit ``null`` keys are the caller clearing a pin
    on purpose and are honored."""
    if not isinstance(existing, dict):
        return {}
    preserved: dict = {}
    if existing.get("modelRevision") and "modelRevision" not in document:
        document["modelRevision"] = existing["modelRevision"]
        preserved["modelRevision"] = existing["modelRevision"]
    if (existing.get("capabilityBatteryFile")
            and existing.get("capabilityBatteryHash")
            and "capabilityBatteryFile" not in document
            and "capabilityBatteryHash" not in document):
        document["capabilityBatteryFile"] = existing["capabilityBatteryFile"]
        document["capabilityBatteryHash"] = existing["capabilityBatteryHash"]
        preserved["capabilityBattery"] = {
            "file": existing["capabilityBatteryFile"],
            "hash": existing["capabilityBatteryHash"]}
    incoming_conditions = document.get("conditions")
    if incoming_conditions is not None \
            and not isinstance(incoming_conditions, list):
        return preserved  # malformed conditions: let validation refuse it
    incoming_names = {c.get("name") for c in incoming_conditions or []
                      if isinstance(c, dict)}
    restored: list[str] = []
    for condition in existing.get("conditions") or []:
        if not (isinstance(condition, dict)
                and isinstance(condition.get("selection"), dict)
                and str(condition.get("name") or "").endswith("-recommended")
                and condition.get("name") not in incoming_names):
            continue
        document.setdefault("conditions", []).append(condition)
        restored.append(condition["name"])
    if restored:
        preserved["conditions"] = restored
    return preserved
