"""THE ROUTE-OWNERSHIP CENSUS — runner-profile narrowing, step 1.

Every HTTP route the FastAPI app exposes, labelled with the SERVICE ROLE it
belongs to. This is a census of TODAY, not an aspiration: where a cluster
deployment legitimately serves the Mac app's interactive features, the route is
censused ``both`` and the rationale says so.

**Nothing here restricts anything.** No production module imports this table,
no request is refused because of it, and the app's behaviour is byte-identical
with or without this file. What it buys is the ratchet in
``test_route_roles.py``: a new route cannot ship undeclared, and a declaration
for a route that no longer exists fails too, so the table cannot rot into
fiction while everyone reads it as evidence.

The three roles come from the two-service-roles ruling that
``docs/PORTABILITY-CONTRACTS.md`` §9.1 already made:

  RUNNER    — batch execution reached through the bundle protocol, plus the
              operations that keep such a deployment alive: identity and
              capabilities, bundle upload / inspect / submit, jobs, logs,
              evidence packaging and download, the model cache, and the
              scheduler. A runner's artifact root is a disposable CACHE; every
              input it needs arrives hash-pinned.
  WORKBENCH — interactive serving of a LIVE, authored workspace: authoring
              writes, the workspace switch, concept and manifest writes,
              server-side freeze, the playground and every other synchronous
              in-process compute the app drives turn by turn, and catalog
              browsing.
  BOTH      — genuinely used by both roles today. This is the honest answer for
              the cluster deployment's remote-workbench surface (artifact
              catalogs the app browses over the wire, judging intake, artifact
              provisioning) and for the submission spellings that name a
              SERVER-RESIDENT study rather than an uploaded bundle.

WHY THIS TABLE LIVES BESIDE THE TESTS AND NOT IN ``steerlab_server/api/``.
Because it governs nothing. ``_PRIVILEGED_PREFIXES`` and
``_OPEN_MUTATING_PATHS`` live in ``api/app.py`` because ``auth_middleware``
branches on them; this table has no branch anywhere, and a table shipped inside
the installed package that no code honours is a claim the package makes about
itself and does not keep. The house precedent for exactly this shape is
``Tests/ExperimentKitTests/CheckoutDependencyTests.swift``, whose census sits in
the test target for the same reason, and ``tests/checkpoint_harness.py`` is the
existing precedent for a non-test module beside the tests. When a runner profile
really does refuse workbench routes, the table moves into the package as part of
that change — a deliberate act, in the diff that gives it teeth.

Keys are ROUTE TEMPLATES exactly as the router declares them, paired with the
method, which is the same vocabulary ``test_wp_s_hardening.py`` walks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    """Which service role owns a route."""

    RUNNER = "runner"
    WORKBENCH = "workbench"
    BOTH = "both"


@dataclass(frozen=True)
class RouteRole:
    """One declared route, with the role it belongs to and why."""

    #: HTTP method, upper case, exactly as the router registers it.
    method: str
    #: Route TEMPLATE, exactly as the router declares it (``{param}`` intact).
    path: str
    role: Role
    #: One line: what this route does, in the terms the role turns on.
    why: str

    @property
    def key(self) -> str:
        return f"{self.method} {self.path}"


def _r(method: str, path: str, role: Role, why: str) -> RouteRole:
    return RouteRole(method=method, path=path, role=role, why=why)


R, W, B = Role.RUNNER, Role.WORKBENCH, Role.BOTH


#: THE CENSUS. Every (method, template) the app serves must appear here.
CENSUS: tuple[RouteRole, ...] = (

    # ── Liveness, identity, and the generated API surface ──────────────────
    _r("GET", "/healthz", B,
       "Liveness probe with no workspace in it at all; every deployment of "
       "either role answers it and must keep answering it."),
    _r("GET", "/api/info", B,
       "The identity read: service, engine version, serving root, devices, "
       "loaded models, capabilities. The Phase-2 adapter's `info()` and the "
       "app's own deployment probe are the same call."),
    _r("GET", "/api/capabilities", B,
       "The capability snapshot `info()` also embeds. The adapter reads it "
       "before submitting; the app polls it every ~15s to gate affordances."),
    _r("GET", "/api/profile/validate", B,
       "Deployment self-check over profile/topology/executor/bind/auth — an "
       "operator read about the SERVER, not about any workspace."),
    _r("GET", "/", W,
       "Serves the bundled web UI's index.html (falls back to a JSON note). "
       "A browser front end is the workbench's surface, not a runner's."),
    _r("GET", "/docs", B,
       "FastAPI's generated Swagger page — it describes whatever surface is "
       "served, so it belongs to neither role in particular."),
    _r("HEAD", "/docs", B,
       "The HEAD half FastAPI registers alongside GET /docs; same generated "
       "page, same reasoning."),
    _r("GET", "/docs/oauth2-redirect", B,
       "FastAPI's generated OAuth2 redirect helper for the Swagger page."),
    _r("HEAD", "/docs/oauth2-redirect", B,
       "The HEAD half of the generated OAuth2 redirect helper; same "
       "reasoning as its GET."),
    _r("GET", "/redoc", B, "FastAPI's generated ReDoc page. As /docs."),
    _r("HEAD", "/redoc", B,
       "The HEAD half FastAPI registers alongside GET /redoc; same generated "
       "page, same reasoning."),
    _r("GET", "/openapi.json", B,
       "The generated schema both pages render. As /docs."),
    _r("HEAD", "/openapi.json", B,
       "The HEAD half FastAPI registers alongside GET /openapi.json; same "
       "generated schema, same reasoning."),

    # ── Bundles: the runner protocol's own wire ────────────────────────────
    _r("POST", "/api/bundles/upload", R,
       "Stages an uploaded archive in a server-minted staging run directory — "
       "step 1 of the bundle protocol; the adapter's `upload_run_bundle`."),
    _r("POST", "/api/bundles/inspect", R,
       "Recomputes a staged archive's outer digest and reads its metadata — "
       "the idempotent precheck `submit --bundle-sha` runs before submitting."),
    _r("GET", "/api/bundles/download", R,
       "Streams a runner-side archive out; the adapter's `download_bundle`, "
       "which is how evidence leaves the runner."),
    _r("POST", "/api/bundles/evidence", R,
       "Packages evidence FROM a run directory this deployment produced. "
       "Producing evidence is the runner's whole output."),
    _r("POST", "/api/bundles/run", B,
       "Packages a run bundle out of a SERVER-RESIDENT experiment. The "
       "artifact is a runner input, but the act needs a live authored "
       "workspace — the cluster-as-workbench path the app uses today. A pure "
       "runner receives bundles by upload and never packages its own."),
    _r("POST", "/api/slurm/bundle", B,
       "The same handler as /api/bundles/run under its older name; both "
       "spellings are registered, so both are censused, identically."),
    _r("POST", "/api/bundles/import", B,
       "Lands a bundle into the served root with the out-of-band outer pin. "
       "On a cluster that is how a Mac-packaged run bundle seeds the cache; "
       "on the app's local engine it lands evidence into the live workspace."),

    # ── Submission and the scheduler ───────────────────────────────────────
    _r("POST", "/api/studies/submit-bundle", R,
       "Submits an UPLOADED, hash-pinned bundle. The pure-runner spelling, "
       "and the one the Phase-2 adapter and `steerlab run` use."),
    _r("POST", "/api/studies/submit", B,
       "Submits a SERVER-RESIDENT study by name. Submission is the runner's "
       "act, but naming a study in the served workspace depends on that "
       "workspace being authored rather than a cache — so it is both."),
    _r("POST", "/api/slurm/submit", R,
       "Builds a Slurm bundle for a caller-supplied argv and runs sbatch. A "
       "raw scheduler operation; nothing about it reads a workspace."),
    _r("POST", "/api/finetune/submit", R,
       "The evidence-grade fine-tune path: a hash-pinned bundle submitted as "
       "a Slurm job. Submission and scheduling, in the runner's vocabulary."),
    _r("POST", "/api/finetune/train", R,
       "Runs the LoRA training itself — GPU compute this deployment executes."),
    _r("POST", "/api/finetune/plan", W,
       "Normalizes a v2 request into the PLAN a researcher confirms in the "
       "app, reading caller-named workspace files. Nothing is scheduled and "
       "nothing is spent; it is the confirmation step, not the job."),

    # ── Jobs, logs, and the job store ──────────────────────────────────────
    _r("GET", "/api/jobs", R,
       "The job roster with tails; the adapter's `jobs()`."),
    _r("GET", "/api/jobs/{job_id}", R,
       "One job's record including status, result and logTail; the adapter's "
       "`job()` and its bounded `job_logs()`."),
    _r("GET", "/api/jobs/{job_id}/stream", R,
       "SSE log follow to a terminal status; the adapter's "
       "`job_logs(follow=True)`."),
    _r("POST", "/api/jobs/{job_id}/cancel", R,
       "Cancels a running job (scancel on Slurm); the adapter's "
       "`cancel_job()`."),
    _r("POST", "/api/jobs/{job_id}/resubmit", R,
       "Manual resume of a checkpointed job — runs sbatch on the job's own "
       "script. A scheduler operation over the runner's own job store."),
    _r("POST", "/api/jobs/reconcile", R,
       "Folds child records and runs the shard-merge pass over the runner's "
       "own job store and run directories."),

    # ── Housekeeping: the runner's disk and the maintenance calendar ───────
    _r("GET", "/api/housekeeping/status", B,
       "Disk roots, quota, purge risk, HF cache, evidence and the live "
       "maintenance calendar. The executor's submit-time refusal reads the "
       "calendar; the app renders the whole document."),
    _r("POST", "/api/housekeeping/refresh", R,
       "Forces the expensive filesystem scans over the runner's own runs/ and "
       "model cache. An operations act on this deployment's disk."),
    _r("POST", "/api/housekeeping/maintenance", R,
       "Writes the maintenance calendar the executor's submit-time refusal "
       "enforces. Scheduler policy for this deployment."),

    # ── The model cache and interactive model residency ────────────────────
    _r("POST", "/api/models/install", R,
       "Prefetches a HF repo into THIS substrate's cache as a durable job — a "
       "model-cache operation, and the way a runner acquires weights before a "
       "compute node goes offline."),
    _r("POST", "/api/load", W,
       "Loads a model into the one interactive slot. Proxied to a GPU-session "
       "worker and counted as interactive activity; residency is the "
       "workbench's affordance, batch jobs load their own pinned model."),
    _r("POST", "/api/load/stream", W,
       "SSE variant of /api/load. Same role, same reason."),
    _r("POST", "/api/models/unload", W,
       "Releases the interactive slot. The other half of /api/load."),

    # ── Playground: synchronous, interactive compute ───────────────────────
    _r("POST", "/api/generate", W,
       "One generation against the resident model — the playground wire. "
       "Proxied to the interactive worker and counted as activity."),
    _r("POST", "/api/generate/stream", W,
       "Streaming half of /api/generate. Same role, same reason."),
    _r("POST", "/api/extract", W,
       "Interactive extraction against the resident model, driven turn by "
       "turn from the app rather than as a pinned batch job."),
    _r("POST", "/api/multiconcept/extract", W,
       "Grand-mean multi-concept extraction driven from the app's extraction "
       "screen. Real GPU work, but the workbench is the caller and the live "
       "workspace is where the vectors land."),
    _r("POST", "/api/geometry", W,
       "Pairwise cosine matrix across selected vectors at a layer, for the "
       "app to draw. A read over the workspace's artifacts."),

    # ── Workspace identity and the runtime switch ──────────────────────────
    _r("GET", "/api/workspace", W,
       "The serving root plus the live SWITCH POLICY, so the app can gate its "
       "switch affordance. A runner reports its root through /api/info; the "
       "switch policy is a workbench question."),
    _r("POST", "/api/workspace/switch", W,
       "Repoints the serving root at a caller-named directory. §9.1 is "
       "explicit that a runner owns a disposable root and never switches it; "
       "this is the workbench service role by definition."),

    # ── Authoring writes (the headless authoring family) ───────────────────
    _r("POST", "/api/authoring/create", W,
       "Creates a study in the served workspace. Authoring write."),
    _r("POST", "/api/authoring/{name}/attach", W,
       "Attaches a concept/artifact to a draft. Authoring write."),
    _r("POST", "/api/authoring/{name}/detach", W,
       "Removes concept pins from a draft — attach's inverse. Authoring "
       "write."),
    _r("POST", "/api/authoring/{name}/sweep-grid", W,
       "Declares the sweep's layer × alpha grid on a draft. Authoring "
       "write."),
    _r("POST", "/api/authoring/{name}/protocol", W,
       "Sets protocol fields on a draft. Authoring write."),
    _r("POST", "/api/authoring/{name}/condition", W,
       "Declares a condition arm. Authoring write."),
    _r("POST", "/api/authoring/{name}/condition/remove", W,
       "Removes a condition arm. Authoring write."),
    _r("POST", "/api/authoring/{name}/duplicate", W,
       "Duplicates a study — the sanctioned way to iterate off a frozen one. "
       "Authoring write."),
    _r("POST", "/api/authoring/{name}/freeze", W,
       "SERVER-SIDE FREEZE: runs the gates and stamps the manifest. Freezing "
       "is an act on a live authored workspace, never on a cache."),

    # ── Experiments: catalog, manifest, verification ───────────────────────
    _r("GET", "/api/experiments", W,
       "Study catalog listing. Catalog browsing."),
    _r("GET", "/api/experiment/{name}", W,
       "One study's summary. Catalog browsing."),
    _r("GET", "/api/experiment/{name}/manifest", W,
       "The raw server-resident manifest document, for the app to compare "
       "against its own copy. A read of authored state."),
    _r("PUT", "/api/experiment/{name}/manifest", W,
       "One-click server-draft sync: installs a caller-supplied manifest into "
       "the served workspace. A manifest write, and the client_cli "
       "deliberately does NOT expose it (§7)."),
    _r("GET", "/api/experiment/{name}/verify", W,
       "Re-verifies a server-resident study's pin surface. A read over "
       "authored state."),
    _r("POST", "/api/experiment/{name}/token-preflight", W,
       "Exact prompt-token counts against the pinned revision, so the app can "
       "show a context verdict before anyone commits. Authoring advice."),
    _r("POST", "/api/experiment/{name}/promote", W,
       "Mints an agent (variant artifact) from a sweep-selected cell — an "
       "authoring write into the workspace's variant library."),
    _r("POST", "/api/experiment/{name}/confirm", W,
       "Attaches a confirmation-study perturbation policy to a DRAFT. "
       "Authoring write, on a study that is by definition not frozen."),
    _r("POST", "/api/experiment/{name}/{verb}", B,
       "JUDGMENT CALL. Launches extract/validate/sweep/run/evaluate/analyze/ "
       "pipeline as a job against a SERVER-RESIDENT study. It is genuine "
       "execution (runner-shaped), but it is reached without the bundle "
       "protocol and needs an authored workspace — the app's remote-workbench "
       "path against a cluster today. Censused `both` because that is what it "
       "is used for; a runner profile is where the tension gets resolved."),

    # ── Deferred judging: intake for Mac-side / subagent judgment ──────────
    _r("GET", "/api/experiment/{name}/sweep/awaiting", B,
       "JUDGMENT CALL. Lists sweep runs awaiting judgment — blinded packets "
       "the RUNNER's own execution emitted, fetched by the WORKBENCH that "
       "judges them. Both halves of the keyless-custody fork are real."),
    _r("POST", "/api/experiment/{name}/sweep/complete-judgment", B,
       "Phase 2 of the same fork: the judgments come back and are stamped "
       "into the run directory the runner produced. Both, for the same "
       "reason as its `awaiting` half."),
    _r("GET", "/api/experiment/{name}/evaluate/awaiting", B,
       "The evaluate-verb twin of sweep/awaiting. Same reasoning."),
    _r("POST", "/api/experiment/{name}/evaluate/complete-judgment", B,
       "The evaluate-verb twin of sweep/complete-judgment. Same reasoning."),

    # ── Runs and pipelines: results browsing ───────────────────────────────
    _r("GET", "/api/runs", W,
       "Run catalog listing for the results explorer. Catalog browsing — "
       "evidence LEAVES a runner as a verified archive, not as a listing."),
    _r("GET", "/api/runs/{run_id}/file", W,
       "Streams one named file out of a run directory for the app to render. "
       "Results browsing; the runner's evidence path is "
       "/api/bundles/evidence + /api/bundles/download."),
    _r("GET", "/api/experiment/{name}/pipelines", W,
       "Pipeline (chain-runner) runs for one study, newest first. Catalog."),
    _r("GET", "/api/pipelines", W,
       "Every study's pipeline runs, newest first. Catalog."),

    # ── Concepts: authoring and its catalogs ───────────────────────────────
    _r("GET", "/api/concepts", W, "Concept catalog listing. Catalog browsing."),
    _r("GET", "/api/concept/{name}", W, "One concept's summary. Catalog."),
    _r("GET", "/api/concept/{name}/full", W,
       "One concept's full stimulus set. Catalog."),
    _r("POST", "/api/concepts/create", W, "Creates a concept. Authoring write."),
    _r("POST", "/api/concept/{name}/save", W,
       "Writes a concept's positive/negative stimuli. Authoring write."),
    _r("POST", "/api/concept/save", W,
       "LEGACY ALIAS, wired as a 501 stub: 'concept authoring stays "
       "Swift/UI-side for now'. Registered, so censused — and the role it "
       "declines to serve is the workbench's."),
    _r("POST", "/api/concept/import", W,
       "Parse-only preview of pasted stimulus text for the authoring UI. No "
       "write, but it exists to feed one."),
    _r("POST", "/api/concept/{name}/delete", W,
       "Deletes a concept from the served workspace. Authoring write."),
    _r("GET", "/api/proposals/available", W,
       "Which stimulus-proposal generators this deployment offers. Catalog."),
    _r("POST", "/api/concept/{name}/proposals", W,
       "Generates candidate stimuli for a concept. Authoring assistance."),
    _r("GET", "/api/concept/{name}/prompt", W,
       "The authoring brief for a concept. Authoring assistance."),
    _r("POST", "/api/concept/{name}/extract", W,
       "Extracts this concept's vector into the live workspace, driven from "
       "the app's concept screen."),
    _r("POST", "/api/concept/{name}/stats", W,
       "Synchronous in-process separability statistics — proxied to the "
       "interactive worker, which is what makes it a workbench call."),
    _r("GET", "/api/concept/{name}/probe-items", W,
       "The concept's held-out probe items. Catalog."),
    _r("POST", "/api/concept/{name}/probe-items", W,
       "Writes the probe item set. Authoring write."),
    _r("POST", "/api/concept/{name}/probe-import", W,
       "Parse-only preview of a pasted probe file for the authoring UI."),
    _r("POST", "/api/concept/{name}/probe-train", W,
       "Trains the concept's probe against the resident model. Authoring "
       "compute over the live workspace."),
    _r("GET", "/api/multiconcept/concepts", W,
       "Multi-concept authoring catalog. Catalog browsing."),
    _r("GET", "/api/multiconcept/{concept}/stories", W,
       "One concept's story set. Catalog."),
    _r("POST", "/api/multiconcept/{concept}/stories", W,
       "Writes a concept's story set. Authoring write."),
    _r("GET", "/api/neutral/corpora", W,
       "Neutral-corpus catalog. Catalog browsing."),
    _r("POST", "/api/neutral/import", W,
       "Imports a neutral corpus into the served workspace. Authoring write."),
    _r("POST", "/api/neutral-pcs/build", W,
       "Builds the neutral principal components the extraction recipes need. "
       "Authoring compute over the live workspace."),

    # ── Vectors and readers: artifacts and measurement instruments ─────────
    _r("GET", "/api/vectors", W,
       "Vector artifact catalog. Catalog browsing."),
    _r("POST", "/api/vectors/backfill-norms", W,
       "Repairs an existing vector artifact in place by measuring its "
       "residual norms. An artifact write over the live workspace."),
    _r("GET", "/api/readers", W,
       "RepE reader catalog. Catalog browsing."),
    _r("POST", "/api/reader/score", W,
       "Exact RepE reader inference over the resident model — proxied to the "
       "interactive worker."),
    _r("POST", "/api/reader/fit", W,
       "Fits a reader and persists it under prompts/ and runs/. Instrument "
       "authoring over the live workspace."),

    # ── J-lens: interactive interpretability tooling ───────────────────────
    _r("GET", "/api/jlens/lenses", W,
       "Imported-lens catalog, read-only. Catalog browsing."),
    _r("GET", "/api/jlens/lenses/{lens_id}", W,
       "One lens's record. Catalog browsing."),
    _r("POST", "/api/jlens/lenses/acquire", W,
       "Fetches a lens into the HF cache. A cache fetch in shape, but the "
       "only consumer is the J-lens workbench — no batch verb needs a lens."),
    _r("POST", "/api/jlens/lenses/import", W,
       "Converts a cached lens into the workspace lens store. A write into "
       "the live workspace's instrument set."),
    _r("POST", "/api/jlens/token-options", W,
       "Tokenizer-only candidates for a typed string, so the app can offer a "
       "choice before anyone commits to a token id."),
    _r("POST", "/api/jlens/decode-tokens", W,
       "Tokenizer-only ids to pieces, for rendering a trace in the app."),
    _r("POST", "/api/jlens/directions/derive", W,
       "Derives a model-depth direction for one exact token and writes the "
       "artifact. Interactive instrument authoring."),
    _r("POST", "/api/jlens/qualify", W,
       "Accepts a lens against one exact runtime and records the "
       "qualification on the lens. Instrument authoring."),
    _r("POST", "/api/jlens/g0", W,
       "The G0 feasibility gate — loads the model and generates, for the "
       "app's lens screen."),
    _r("POST", "/api/jlens/probe", W,
       "Position-resolved readout over ONE prompt. The interactive probe."),
    _r("POST", "/api/jlens/report", W,
       "Rolls a completed run's J-lens trace up into its report. It reads a "
       "run directory the runner produced, but producing the interpretability "
       "report the app renders is a workbench act over that evidence."),
    _r("POST", "/api/jlens/support", W,
       "Reads an existing concept vector back as J-lens token atoms and "
       "writes a readout run. Interactive analysis."),

    # ── Variants and adapters: the agent library ───────────────────────────
    _r("GET", "/api/variants", W,
       "Variant (agent) catalog listing. Catalog browsing."),
    _r("GET", "/api/variant/detail", W,
       "One variant's full spec plus the artifact's SHA-256. Catalog."),
    _r("GET", "/api/adapters", W,
       "LoRA adapter catalog. Catalog browsing."),
    _r("POST", "/api/variants/upload", B,
       "JUDGMENT CALL. Installs a variant artifact (spec + referenced bytes) "
       "into the served root. It is the app's variant-library push (workbench) "
       "AND how an agent artifact an execution needs reaches a cluster "
       "(runner). Both are live paths today."),
    _r("POST", "/api/model-variant/save", W,
       "Saves a variant spec authored in the app. Authoring write."),
    _r("POST", "/api/variant/generate", W,
       "One generation through a variant — the playground wire for agents. "
       "Proxied to the interactive worker."),
    _r("POST", "/api/variant/generate/stream", W,
       "Streaming half of /api/variant/generate. Same role, same reason."),
    _r("POST", "/api/variant/battery", W,
       "Scores a pinned capability battery against a variant, synchronously, "
       "turn by turn from the app. Proxied to the interactive worker."),

    # ── Gemma Scope: SAE tooling ───────────────────────────────────────────
    _r("GET", "/api/gemmascope/info", W,
       "Which Gemma Scope releases this deployment can reach. Catalog."),
    _r("POST", "/api/gemmascope/run", W,
       "Runs the SAE analysis that produces a candidate report for the app."),
    _r("POST", "/api/gemmascope/import", W,
       "Imports a feature from a cosine report as a vector artifact. An "
       "artifact write into the live workspace."),
    _r("POST", "/api/gemmascope/import-id", W,
       "Imports a feature BY ID, going online to Hugging Face. Same "
       "destination and same caller as its report-based sibling."),

    # ── Panels / scenarios ─────────────────────────────────────────────────
    _r("GET", "/api/scenarios", W,
       "Panel-script catalog. Catalog browsing."),
    _r("GET", "/api/scenario", W,
       "One panel script, by caller-named path. Catalog."),
    _r("POST", "/api/scenario/save", W,
       "Writes a panel script under prompts/panels/. Authoring write."),
    _r("POST", "/api/scenario/run", W,
       "Runs a panel interactively against the resident model — the "
       "multi-agent playground, not a pinned batch verb."),

    # ── GPU sessions: brokering an interactive worker ──────────────────────
    _r("POST", "/api/session/start", B,
       "JUDGMENT CALL. Submits a Slurm job (a scheduler op, runner-shaped) "
       "whose entire product is an INTERACTIVE worker for the app's "
       "playground. The verb belongs to the runner; the feature it exists for "
       "belongs to the workbench."),
    _r("GET", "/api/session", B,
       "Reads the live session record, polling the scheduler. Same reasoning "
       "as /api/session/start."),
    _r("DELETE", "/api/session", B,
       "scancels the session job. A scheduler op ending a workbench feature."),
    _r("POST", "/api/session/keepalive", B,
       "Resets the worker's idle timer. Scheduler-adjacent lifecycle for an "
       "interactive worker."),

    # ── Interactive server state ───────────────────────────────────────────
    _r("GET", "/api/state", W,
       "The interactive snapshot: resident model, slot locks, the jobs badge. "
       "Proxied to the GPU-session worker, which is what makes it the "
       "workbench's read rather than the runner's — a runner's job truth is "
       "/api/jobs."),

    # ── Authoring templates ────────────────────────────────────────────────
    _r("POST", "/api/generation-prompt", W,
       "Renders a FIXED template chosen by an enum key into the authoring "
       "brief the app shows. No model, no write, no state."),
    _r("GET", "/api/battery/generation-prompt", W,
       "The capability-battery authoring brief. Same role, same reason."),
)


#: ``{key: RouteRole}`` for every censused route.
BY_KEY: dict[str, RouteRole] = {entry.key: entry for entry in CENSUS}


def declared_routes(app) -> list[tuple[str, str]]:
    """Every ``(method, template)`` the app actually serves.

    Walks the router tree rather than the OpenAPI schema so routes marked
    ``include_in_schema=False`` are covered too. Twin of
    ``test_wp_s_hardening._declared_routes`` — same walk, ordered
    ``(method, path)`` here because this census is keyed that way.
    """

    def walk(routes, prefix=""):
        for route in routes:
            original = getattr(route, "original_router", None)
            if original is not None:  # FastAPI's lazy _IncludedRouter
                context = getattr(route, "include_context", None)
                yield from walk(original.routes,
                                prefix + (getattr(context, "prefix", "") or ""))
                continue
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None)
            if path and methods:
                for method in sorted(methods):
                    yield method, prefix + path

    return sorted(set(walk(app.routes)))


def roles_for(role: Role) -> tuple[RouteRole, ...]:
    """Every censused route with this role, in census order."""
    return tuple(entry for entry in CENSUS if entry.role is role)


def is_runner_reachable(method: str, path: str) -> bool:
    """Whether a censused route is one a runner-role deployment serves.

    Documentation of the census, NOT a gate: nothing calls this in production,
    and step 1 activates no restriction whatsoever. It exists so the eventual
    runner profile has one place to read the table from, and so the tests can
    say ``runner or both`` without spelling the disjunction each time.
    """
    entry = BY_KEY.get(f"{method.upper()} {path}")
    return entry is not None and entry.role in (Role.RUNNER, Role.BOTH)
