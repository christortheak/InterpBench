"""Judge model identities, capacity, column release and local generation scopes.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
from contextlib import contextmanager
from ..steering import model_loader
from . import judging_custody
from . import prompt_render
from . import generate
from . import manifest


#: One resident model container, as the registry keys it minus the device it
#: happened to land on: ``(modelID, revision or None, canonical dtype or
#: None)``. The same triple ``_judge_preflight`` counts.
ModelIdentity = tuple[str, "str | None", "str | None"]


def study_model_identity(study_model: str, study_revision: str | None = None,
                         study_dtype: str | None = None) -> ModelIdentity:
    """The study model's own container identity, spelled ONCE so the
    still-needed set, the release candidates, and the slot arithmetic can
    never disagree about which container the study model is."""
    return (study_model,
            (study_revision or "").strip() or None,
            model_loader.normalize_dtype(study_dtype))


def judge_model_identity(ref, *, study_model: str,
                         study_revision: str | None = None,
                         study_dtype: str | None = None) -> ModelIdentity:
    """The ``(modelID, revision, canonical dtype)`` identity of the container
    ONE local judge needs.

    Why an identity and not a bare slug (external review round 12, finding
    3): ``--judge-pin`` makes a same-slug-DIFFERENT-revision panel
    expressible, and a still-needed set that speaks slugs cannot tell the
    finished judge's container from the one about to load. It reads the two
    as one model, keeps the finished one resident as dead weight, and
    recreates the very co-residency OOM the column seam exists to prevent.

    A judge that resolves to the STUDY model IS the study model: it reuses
    the held weights and the loader is never asked for a second copy, so
    its identity is the STUDY's pins, not the judge's own. (A study-model
    judge that pins something divergent is refused where that lie is
    detectable — ``_assert_study_model_judge_matches_held`` and the sweep
    preflight — never silently re-keyed here.)

    Callers pass a non-local ``ref`` at their own risk: external judges hold
    no device memory and have no identity worth releasing.
    """
    from . import sweep_selection
    resolved = sweep_selection.resolve_local_judge_model(ref.model, study_model)
    if resolved == study_model:
        return study_model_identity(study_model, study_revision, study_dtype)
    return (resolved,
            (getattr(ref, "revision", None) or "").strip() or None,
            model_loader.normalize_dtype(getattr(ref, "dtype", None)))


def judge_models_still_needed(remaining_roster, *, study_model: str,
                              study_revision: str | None = None,
                              study_dtype: str | None = None,
                              study_model_generates_later: bool
                              ) -> set[ModelIdentity]:
    """The model IDENTITIES the REMAINDER of a judged run still needs.

    The still-needed rule, exactly (maintainer's ruling, 2026-08-28: "any
    runs that require two models will need to unload and load models in
    order not to OOM. We need to ensure this happens"):

        still-needed = {identity of every LOCAL judge in
                        ``remaining_roster``}
                       ∪ ({study identity} if a later stage of this run or
                          pipeline GENERATES)

    ``remaining_roster`` is ``roster[i:]`` at the boundary before judge
    ``i`` — so the judge about to load is itself in the set, which is what
    keeps two consecutive same-identity columns warm instead of
    releasing-and-reloading the very weights the next column needs.
    External judges (claude/openrouter) hold no device memory and
    contribute nothing. A local judge with an empty model resolves to the
    study model by the cross-engine rule, so a study-model judge keeps the
    study model resident without any special case.

    Identities, not slugs (external review round 12, finding 3): see
    ``judge_model_identity``. Two judges on one slug at two revisions are
    two containers, and only the one nobody needs again is released.

    ``study_model_generates_later`` is the conservative half: evaluate is
    terminal for generation on its own (analyze/rescore are CPU-side), but
    a caller that cannot PROVE the study model is finished passes True and
    the model is kept — the capacity gate then speaks, as before.
    """
    needed = {
        judge_model_identity(ref, study_model=study_model,
                             study_revision=study_revision,
                             study_dtype=study_dtype)
        for ref in remaining_roster if ref.kind == "local"}
    if study_model_generates_later:
        needed.add(study_model_identity(study_model, study_revision,
                                        study_dtype))
    return needed


def judge_slots_required(roster, *, study_model: str,
                         study_revision: str | None = None,
                         study_dtype: str | None = None,
                         study_model_generates_later: bool = False,
                         sequential: bool = True) -> int:
    """How many resident model SLOTS a judge panel actually needs at once.

    The arithmetic the evaluate capacity guard asks (external review round
    12, finding 2a). Judges are needed one COLUMN at a time and the release
    seam drops each finished column's container before the next loads, so
    the ask is the largest SINGLE MOMENT of the run, not the panel's size:
    a five-judge panel of five distinct models runs in ONE slot, while a
    one-slot server is only refused when some single moment genuinely needs
    two.

    Per column ``i`` the resident set is what has been LOADED by then
    (``columns[:i+1]``) intersected with what is still NEEDED from then on
    (``columns[i:]``), plus the study identity when a later stage still
    generates. A panel A, B, A therefore costs 2 — A survives B's column
    because the third judge will want it back — while A, B, C costs 1.

    ``sequential=False`` is the honest fallback for a caller that supplies
    no release seam (``model_release is None``): nothing can be dropped
    between columns, so every distinct identity must be resident together
    and the count is the whole panel's.
    """
    columns = [
        judge_model_identity(ref, study_model=study_model,
                             study_revision=study_revision,
                             study_dtype=study_dtype)
        for ref in roster if ref.kind == "local"]
    held: set[ModelIdentity] = set()
    if study_model_generates_later:
        held.add(study_model_identity(study_model, study_revision,
                                      study_dtype))
    if not columns:
        return len(held)
    if not sequential:
        return len(held | set(columns))
    return max(len(held | (set(columns[:index + 1]) & set(columns[index:])))
               for index in range(len(columns)))


def identity_text(identity: ModelIdentity, *, quoted: bool = True) -> str:
    """``'org/model'@abc123456789…`` — how a released container is named in a
    run log. The revision prefix is the point (external review round 12,
    finding 3): a same-slug panel at two revisions is two containers, and a
    log line that printed only the slug could not say WHICH one went."""
    model_id, revision, _dtype = identity
    name = f"'{model_id}'" if quoted else str(model_id)
    return name + (f"@{revision[:12]}…" if revision else "")


def release_models_for_judge(model_release, roster, index: int, *,
                              study_model: str,
                              study_revision: str | None = None,
                              study_dtype: str | None = None,
                              study_model_generates_later: bool,
                              _log) -> None:
    """The model-slot release seam: free every container the remainder of
    this run will not use, BEFORE judge ``index``'s model loads.

    The guarantee this buys: a run whose models are needed SEQUENTIALLY
    never fails for co-residency. Peak device memory becomes the MAX of any
    one still-needed model instead of the SUM of the panel's weights — the
    two-judge calibration on 2026-08-28 refused at ~22.7 GiB needed against
    23.3 GiB free on an 80 GiB A100 purely because the FINISHED judge's
    container was still resident and nothing in the run would ever use it
    again.

    Candidates are this run's OWN models — the study model and every local
    judge's resolved model — never an unrelated resident (a chat model, a
    variant-generate model): the release is a run seam, not a new global
    eviction heuristic, and the interactive cache policy is unchanged.
    Whatever ``judge_models_still_needed`` names is subtracted, so the seam
    is a no-op for a single-judge run and for a same-identity column
    boundary.

    Everything here speaks ``(modelID, revision, canonical dtype)``
    IDENTITIES, never bare slugs (external review round 12, finding 3): a
    panel that pins one slug at two revisions is two containers, and the
    finished one has to go while the one about to load stays.

    Deliberate cross-engine divergence (documented so neither side reads as
    an oversight): the Mac answers the same problem by REFUSING up front —
    ``SweepObjectives.localJudgeSlotProblem`` declines a two-model sweep
    before any work starts, which is right for MLX's memory model, where a
    unified-memory container is not cheaply reclaimable mid-run. CUDA can
    hand the weights back, so this engine releases BETWEEN columns instead
    of refusing.

    A failing release never fails the run: it is logged and the capacity
    gate remains the backstop.
    """
    ref = roster[index]
    def identity_of(judge_ref) -> ModelIdentity:
        return judge_model_identity(
            judge_ref, study_model=study_model,
            study_revision=study_revision, study_dtype=study_dtype)
    keep = judge_models_still_needed(
        roster[index:], study_model=study_model,
        study_revision=study_revision, study_dtype=study_dtype,
        study_model_generates_later=study_model_generates_later)
    candidates = {study_model_identity(study_model, study_revision,
                                       study_dtype)} | {
        identity_of(r) for r in roster if r.kind == "local"}
    stale = sorted(candidates - keep)
    next_identity = identity_of(ref) if ref.kind == "local" else None
    need_text = (f"next judge '{ref.name}' needs "
                 f"{identity_text(next_identity)}"
                 if next_identity else
                 f"next judge '{ref.name}' needs no local model")
    where = ("generation complete" if index == 0
             else f"column '{roster[index - 1].name}' complete")
    if model_release is not None and stale:
        try:
            released = model_release(stale) or []
        except Exception as exc:  # noqa: BLE001 - never fail a run on cleanup
            _log(f"WARNING: could not release model slot(s) "
                 f"{', '.join(identity_text(i) for i in stale)} before "
                 f"judge '{ref.name}' ({exc}) — continuing; the load "
                 "capacity gate remains the backstop")
            return
        for record in released:
            size = record.get("bytes")
            size_text = f" (~{size / (1 << 30):.1f} GiB)" if size else ""
            _log("released "
                 + identity_text((record["modelID"], record.get("revision"),
                                   record.get("dtype")))
                 + f"{size_text} from {record.get('device')} — {where}, "
                 + need_text)
        return
    if model_release is None and index > 0:
        # CLI/bundle path (the Slurm path): there is no registry, so the
        # previous column's PRIVATE in-process copy is what has to go. Its
        # last reference was dropped when that column's ExitStack closed
        # (``_local_judge_generation`` registers the drop); the allocator
        # still holds the blocks until they are collected and trimmed, and
        # `cuda.mem_get_info` — what the capacity gate reads — counts them
        # as used until then.
        model_loader.free_device_memory()
        _log(f"released the private model copy of {where} — {need_text}")


def judge_callable(ref: manifest.JudgeRef, model_provider, *, study_model: str,
                    study_revision: str | None = None, stack=None):
    """(judge_fn, requested_model, actual_model_holder) for one judge. Claude
    judges call the Anthropic API; local judges acquire the served model
    through ``model_provider`` (the registry slot lock — no forward pass ever
    shares a model object unlocked). The CLI path has no provider and
    synthesizes one that loads a private copy in-process.

    A LOCAL judge's model resolves by the cross-engine rule
    (``sweep_selection.resolve_local_judge_model``, unified for sweep AND
    evaluate 2026-07-22): the declared ``model`` when non-empty, else the
    STUDY model at its pinned revision — acquired through the normal
    provider/registry slot, so an already-resident study model is reused,
    never loaded twice. The judge's NAME is a label, never a model id (the
    dead ``model or name`` fallback sent 'judge-1' to HuggingFace as a model
    id on an offline compute node)."""
    from . import paired_judge, sweep_selection
    if ref.kind == "openrouter":
        model = (ref.model or "").strip()
        provider = (getattr(ref, "provider", None) or "").strip()
        if not model or not provider:
            raise RuntimeError(
                f"openrouter judge '{ref.name}' needs an explicit model "
                "slug AND a pinned provider — neither has a server default")
        return (paired_judge.make_openrouter_judge(model, provider),
                model, {"actual": model})
    if ref.kind != "local":
        model = ref.model or paired_judge.DEFAULT_JUDGE_MODEL
        return paired_judge.judge_pair, model, {"actual": model}
    gen, judge_model, holder = _local_judge_generation(
        ref, model_provider, study_model=study_model,
        study_revision=study_revision, stack=stack)
    return paired_judge.make_local_judge(gen), judge_model, holder


def coder_callable(ref: manifest.JudgeRef, model_provider, *, study_model: str,
                    study_revision: str | None = None, stack=None):
    """``(complete_fn, requested_model, holder)`` for one judge on the
    per-response coding path — the raw-completion sibling of
    ``judge_callable`` (same resolution rules, same slot machinery, same
    holder provenance); ``complete_fn(prompt) -> (text, provider|None)``."""
    from . import paired_judge
    if ref.kind == "openrouter":
        model = (ref.model or "").strip()
        provider = (getattr(ref, "provider", None) or "").strip()
        if not model or not provider:
            raise RuntimeError(
                f"openrouter judge '{ref.name}' needs an explicit model "
                "slug AND a pinned provider — neither has a server default")

        def _openrouter(prompt: str):
            return paired_judge.openrouter_complete(
                model, prompt, provider=provider)
        return _openrouter, model, {"actual": model}
    if ref.kind != "local":
        model = ref.model or paired_judge.DEFAULT_JUDGE_MODEL

        def _claude(prompt: str):
            return paired_judge.claude_complete(model, prompt), None
        return _claude, model, {"actual": model}
    gen, judge_model, holder = _local_judge_generation(
        ref, model_provider, study_model=study_model,
        study_revision=study_revision, stack=stack)

    def _local(prompt: str):
        return gen(prompt), None
    return _local, judge_model, holder


def _local_judge_generation(ref: manifest.JudgeRef, model_provider, *,
                            study_model: str,
                            study_revision: str | None = None, stack=None):
    """``(gen_fn, judge_model, holder)`` for one LOCAL judge — the shared
    model-resolution/slot core of ``judge_callable`` and
    ``coder_callable`` (extracted 2026-08-04 for the coding instrument;
    behavior unchanged)."""
    from . import paired_judge, sweep_selection
    judge_model = sweep_selection.resolve_local_judge_model(
        ref.model, study_model)
    # The judge's own pinned revision wins (JudgeRef.revision, 2026-07-23);
    # a study-model judge falls back to the study pin — the same bytes that
    # generated the outputs.
    judge_revision = (getattr(ref, "revision", None)
                      or (study_revision if judge_model == study_model
                          else None))
    # The judge's declared dtype is a PIN, not a hint (external review round
    # 3, finding 2): freeze now requires it for a foreign local judge, and a
    # manifest that claims a dtype the load ignored is a false pin. Passed
    # only when declared, so providers that predate the parameter keep
    # working for judges that declare none.
    judge_dtype = (getattr(ref, "dtype", None) or "").strip() or None
    provider = model_provider
    if provider is None:
        @contextmanager
        def _cli_provider(model_id, revision=None, dtype=None):
            yield model_loader.load(model_id, revision=revision, dtype=dtype)
        provider = _cli_provider
    holder = {"actual": judge_model, "revision": judge_revision,
              "requestedDtype": judge_dtype}
    held: dict = {}

    @contextmanager
    def _slot():
        """The judge's model, loaded ONCE per column when the caller
        supplies an ``ExitStack`` (external review round 3, finding 3c).

        ``model_loader.load`` has no cache, and this context used to be
        entered inside the per-pair generation call — so on the CLI/bundle
        path (which is the Slurm path) a foreign local judge reloaded on
        EVERY judgment. A 12B judge across a hundred pairs is a hundred
        full weight loads: not slow, a walltime kill that presents as a
        hang.

        Without a stack the behaviour is exactly as before (acquire and
        release per call), so callers opt in rather than silently having
        their slot-holding semantics changed underneath them.
        """
        args = (judge_model, judge_revision)
        kwargs = {"dtype": judge_dtype} if judge_dtype else {}
        if stack is None:
            with provider(*args, **kwargs) as slot:
                yield slot
            return
        if "slot" not in held:
            held["slot"] = stack.enter_context(provider(*args, **kwargs))
            # Drop the column's reference when the stack closes, BEFORE the
            # provider context exits (callbacks unwind LIFO). Without this
            # the CLI/bundle path's private copy stayed pinned by this
            # closure until the NEXT column's callable replaced it — which
            # happens AFTER the next model has loaded, i.e. exactly the
            # co-residency the column boundary exists to prevent
            # (2026-08-28). The registry path is unaffected: there the
            # container's owner is the slot, not this reference.
            stack.callback(held.pop, "slot", None)
        yield held["slot"]

    def _gen(prompt: str) -> str:
        try:
            with _slot() as slot:
                holder["actual"] = slot.model_id
                # The dtype the model ACTUALLY runs in, read off the loaded
                # parameters — not the dtype the manifest asked for
                # (external review round 4, finding 2). Recording only the
                # request let an artifact claim a dtype the load never
                # honored. Stamped for EVERY local judge, including one
                # using the study model, because evaluate may run in a
                # separate job on a different device.
                actual_dtype = getattr(slot, "dtype", None)
                if actual_dtype:
                    holder["actualDtype"] = str(actual_dtype)
                # JUDGE_MAX_TOKENS — the one cross-engine judge cap (the
                # 2026-07-22 incident: 512 truncated a legible verdict).
                return generate.generate(slot, prompt, model_id=slot.model_id,
                                max_tokens=paired_judge.JUDGE_MAX_TOKENS,
                                temperature=0.0,
                                prompt_mode=prompt_render.CHAT_ASSISTANT)
        except model_loader.ModelLoadError as exc:
            if judge_model == study_model:
                raise
            # Name the judge — and when the loader's refusal already carries
            # complete advice (its own typed sentences: device capacity,
            # co-residency headroom, VRAM class), CARRY IT rather than
            # replace it. The old static "install the model on the server"
            # overwrote all of them, and when the real cause was
            # co-residency it sent an agent to install a model that was
            # already there (observed 2026-08-28, the two-judge
            # calibration). Raw hub/network dumps stay summarized away —
            # "check your internet connection" is misleading on an
            # air-gapped compute node. The Swift twin
            # (`localJudgeLoadFailureMessage`) legitimately keeps static
            # install advice: it decides on a PRESENCE check before the
            # loader is asked, so "not installed" is the one possible cause
            # there.
            if getattr(exc, "advice_complete", False):
                raise model_loader.ModelLoadError(
                    f"local judge '{ref.name}' declares model "
                    f"'{judge_model}', which could not be loaded on this "
                    f"server — {exc} (a local judge with an EMPTY model "
                    "judges with the study model)",
                    advice_complete=True) from exc
            raise model_loader.ModelLoadError(
                f"local judge '{ref.name}' declares model '{judge_model}', "
                "which could not be loaded on this server — install the "
                "model on the server, or leave the judge's model empty to "
                "judge with the study model") from exc
    return _gen, judge_model, holder


#: The custody decision itself lives in the torch-free
#: :mod:`judging_custody` (2026-08-20) so the freeze advisory and the
#: submission preflight can ask it without importing this module. These are
#: the historical spellings, unchanged for every caller and test here.
missing_external_credentials = judging_custody.missing_external_credentials


judging_custody_plan = judging_custody.custody_plan


def log_judging_custody(roster, log) -> dict:
    """Announce the custody plan at stage start. One line, always — a
    researcher should never learn where judging ran by inspecting the
    artifacts afterwards."""
    plan = judging_custody_plan(roster)
    log(f"judging custody: {plan['disposition'].upper()} — {plan['reason']}")
    return plan
