# CLI reference — `steerlab-cli` (Swift) and `steerlab-server` (Python)

Complete, code-derived reference for both engines' command lines. Every flag
below was read out of the dispatch code, not remembered:

- Swift: `Sources/steerlab-cli/main.swift` — the CLI's dispatch lives in that
  one file and calls into `ExperimentKit`/`SteeringKit`. The exception is the
  `cluster` namespace (§3.9), whose parsing, serialization, and orchestration
  live in `ExperimentKit/ClusterCLI*.swift` so the wizard and the CLI cannot
  become two cluster implementations.
- Python: `Server/steerlab_server/cli.py`, plus `api/submissions.py`,
  `api/executors.py`, `api/app.py`, `experiment/bundles.py` and
  `experiment/sharding.py` for the behavior behind the flags.

A **third** command line, the cross-platform `steerlab` **client**, is §1.4. It
is a different product from the Swift `steerlab-cli` — a smaller verb surface,
no model loading, and no `workspace init` — and the generated regions below do
not cover it.

**Both CLIs have `--help`** (2026-08-18): `steerlab-cli experiment freeze
--help`, `steerlab-server study submit --help`, `steerlab-cli cluster ensure
--help`, and the family form (`steerlab-cli experiment --help`). It runs
nothing and exits 0, and under `--json` the page comes back inside the envelope
as data. See §7.7 for what is and is not covered.

**The marked regions of this document are generated** from the same declarative
verb tables `--help` renders from — `<!-- GENERATED:<id> BEGIN -->` …
`<!-- GENERATED:<id> END -->`, `swift-*` from the Swift tables and `server-*`
from the Python one. Do not hand-edit inside a marker: run
`steerlab-cli docs cli-reference --write` and
`steerlab-server docs cli-reference --write`, then commit. A drifted region
fails `CLIReferenceGenerationTests.generatedRegionsMatchCommittedDocument`
(Swift) and `test_cli_reference_regions_match` (Python). Everything outside the
markers is hand-written prose and is the document's real value — keep it in sync
by hand.

Scope: what the commands do and what they refuse. Not the study workflow — see
`docs/CONDUCTING-A-STUDY.md` for the order to run things in, and §7 below for
the known gaps and traps.

---

## 1. Orientation

### 1.1 Invocation

```bash
# Swift — build with xcodebuild first (SwiftPM cannot build the Metal shaders).
# CLANG_COVERAGE_MAPPING=NO: without it the auto-generated scheme instruments
# even plain builds with coverage (the binary then sheds default.profraw files).
xcodebuild build -skipMacroValidation -scheme steerlab-cli \
  -destination 'platform=macOS' CLANG_COVERAGE_MAPPING=NO \
  -derivedDataPath .deriveddata.nosync
DYLD_FRAMEWORK_PATH=.deriveddata.nosync/Build/Products/Debug \
  .deriveddata.nosync/Build/Products/Debug/steerlab-cli <verb> …

# Python — console script, or the module (identical entry point)
Server/.venv.nosync/bin/steerlab-server <verb> …
Server/.venv.nosync/bin/python -m steerlab_server.cli <verb> …
```

### 1.2 Which engine does what

Both engines implement the same artifact model and most of the same verbs, but
they are not interchangeable:

| Capability | Swift `steerlab-cli` | Python `steerlab-server` |
|---|---|---|
| Authoring (create/attach/freeze/duplicate) | yes | **no** — verify/read only; authoring is the HTTP API or the app |
| extract / validate / sweep / run | yes (MLX, greedy-only) | yes (PyTorch/HF, seeded sampling) |
| analyze / evaluate / rescore-style | yes | yes |
| `pipeline` (chained stages) | **no** | yes |
| Cluster ops (Slurm, jobs, bundles) | client only (`remote …`) | yes (server-side) |
| Cluster *lifecycle* (auth, push, bootstrap, controller, tunnel, connect) | yes (`cluster …`, §3.9) | **no** — it is the Mac's job to reach the cluster, not the cluster's |
| LoRA fine-tuning | **no** (adapters are a server-native `hf-peft-lora` artifact) | yes (`finetune …`, §5.5; evidence-grade training is a Slurm job) |
| J-lens reading instruments | **no** (hard rule: server + Gemma only) | yes (`jlens …`) |
| Web server | `serve` (loopback only, no `--host`) | `serve` (host configurable) |

Vectors do **not** transfer between engines: re-extract and re-validate on the
substrate a study runs on. `vectors compare` (both CLIs, key-identical JSON) is
the parity check.

### 1.4 The `steerlab` client (preview)

There is a **third** command line, new and not yet covered by the generated
regions of this document: `steerlab`, a cross-platform Python client that
**authors a local workspace** and **hands hash-pinned bundles to a runner** —
`Server/steerlab_server/client_cli.py`, installed by the same package as
`steerlab-server`. It is the Phase-1b + Phase-2 + Phase-3 + Phase-5 deliverable
of the portability program; `docs/PORTABILITY-CONTRACTS.md` §7–§10 are its
reference, and this note exists so a reader of *this* document is not left
believing there are only two.

**`steerlab` and `steerlab-cli` are two products, not two spellings.** The
Swift `steerlab-cli` is the Mac instrument — workspace bootstrap and the whole
measured lifecycle (§3). The Python `steerlab` is this client — authoring, the
bundle round trip, and `run`. Neither answers the other's verbs: there is no
`steerlab workspace init`, no `steerlab experiment extract`, no
`steerlab cluster …`, and typing one is a `64`, not a fallback. Nothing in this
section asks you to run a Swift verb under `steerlab`.

Each says which one it is, and the two answers are unmistakable:

| you typed | a correct install answers |
|---|---|
| `steerlab-cli --version` | the install report, ending in **6/6 resource families resolved** |
| `steerlab --version` | `steerlab <version> (client)` — one line, no resource families |

If `steerlab --version` prints a resource-family report, the name on that PATH
is a leftover symlink or shim to the Swift CLI. Remove it: the console script
owns the spelling.

**Platforms.** The client is the portable half, and one verb is not:

| | macOS | Linux | Windows |
|---|---|---|---|
| authoring (`experiment`, `concept`, `bundle package`/`inspect`/`import`) | yes | yes | yes |
| driving a remote runner (`runner …` except `serve`, and `run`) | yes | yes | yes |
| **executing locally** (`runner serve`, the `[runner]` extra) | yes | yes | **no — refused** |
| the Mac lifecycle (`steerlab-cli`: extract/validate/sweep/run/analyze, `cluster …`) | Apple silicon only | — | — |

**Windows is client-only.** Authoring, freezing, packaging, submission and
evidence import are supported there; `runner serve` refuses by name
(`runnerPlatformUnsupported`) rather than starting an engine whose local
execution path is not supported on that platform. A Windows author points `--runner <url>` at a runner on macOS, Linux,
or a cluster, and the round trip is otherwise identical.

What it is, in one table:

| | `steerlab` (client) | `steerlab-server` (engine) |
|---|---|---|
| authors a workspace | **yes**, the local one | no — Mac-authority refusals, unchanged |
| loads a model / executes verbs | no | yes |
| talks to a runner | **yes** (`runner …`, Phase 2) | it *is* the runner |
| starts a local runner | **yes** (`runner serve`, Phase 3 — it launches the engine) | — |
| runs a study end to end | **yes** (`run <experiment>`, Phase 5 — one command, evidence comes home) | it executes what it is handed |
| needs torch | no — the whole authoring lifecycle is torch-free | yes |

```bash
pip install -e Server                     # the client alone — no torch
export STEERLAB_WORKSPACE=~/SteerLab/Workspaces/<study>   # or: --root <dir>
steerlab experiment create <name> --model <id>
steerlab concept import <concept> --file <stimuli.jsonl>
steerlab experiment attach <name> <concept>
steerlab experiment declare-condition <name> <arm> \
    --slots <concept>:<layer>:<alpha> --alpha-units norm
steerlab experiment verify <name>
steerlab experiment freeze <name>
steerlab bundle package <name>            # hand the bundle to an engine
```

Everything above speaks `--json` and answers in the **same envelope** the two
engines share (§7.7's vocabulary, §4 of the contracts document) — same states,
same exit codes, same `error.code` / `error.repairAction`. Verb families:
`experiment` (create, attach, detach, declare-condition, remove-condition,
set-sweep-grid, set-protocol, set-parser, set-instrument-scope, pin-revision,
set-style-taxonomy, pin-sae-candidates, duplicate,
verify, freeze, list), `concept import`, `bundle` (package, inspect, import),
`authoring prompt <kind>` (§3.12's emitter, identical bytes to the Mac's —
its file destination is spelled `--out-file` here, because this client lifts
`--out` before the family is chosen), `runner` (below — including `runner
serve`, which starts a managed local runner), `run` (the composite, below),
plus `--version`. `steerlab <family>
--help` prints the roster; the workspace comes from `--root` or
`$STEERLAB_WORKSPACE` and there is **no default** — except for the `runner`
family, which addresses a remote engine and names its local paths explicitly,
so it runs without one (a named workspace is still honoured and still reported
in the envelope), and `authoring`, whose template registry falls back to the
shipped copy so a caller who named no study still gets the shipped prompt. That
shipped copy travels inside the wheel, so a `pip install` with no checkout
beside it renders exactly the same bytes.
`run` is not an exception: it reads a study out of a workspace and imports
evidence back into it, so it requires one.

#### The `runner` family (Phase 2)

Handing a frozen study to an engine and bringing the evidence home. Every verb
that *addresses* a runner takes `--runner <url>`; none of them authors
anything. (`runner serve`, added in Phase 3 and documented after this table,
is the one that *becomes* a runner instead — it takes no `--runner`.)

```bash
export STEERLAB_RUNNER_TOKEN=…                # or: --token-file <path>
steerlab runner capabilities --runner http://127.0.0.1:8080
steerlab runner upload runs/…/study.run-bundle.tar.gz --runner <url>
steerlab runner submit --runner <url> \
    --bundle-path <the path upload printed> \
    --bundle-sha  <the digest upload printed> \
    --verb run [--executor local|slurm] [--target-root <dir>] [--dry-run]
steerlab runner jobs [<job-id>] [--cancel] --runner <url>
steerlab runner logs <job-id> [--follow] --runner <url>
steerlab runner evidence <job-id> --out <file.tar.gz> --runner <url>
steerlab bundle import <file.tar.gz> --sha256 <digest>   # the separate step
```

| verb | route(s) it speaks | notes |
|---|---|---|
| `runner capabilities` | `GET /api/info` (+ `GET /api/capabilities` as fallback) | the runner's `engineVersion`, its artifact root, its devices and capability snapshot. Warns when the runner's root looks like a source checkout |
| `runner upload <bundle>` | `POST /api/bundles/upload` | streams the archive; **refuses** if the runner reports a different sha256 than the client computed. Prints the staged path and the digest — the two things `submit` wants |
| `runner submit` | `POST /api/bundles/inspect`, then `POST /api/studies/submit-bundle` | `--bundle-sha` is checked against the runner's own inspect of `--bundle-path` **before** anything is submitted. Reports the job id, the runner's identity, and the digest |
| `runner jobs [<id>]` | `GET /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel` | `--cancel` needs an id — this client will not cancel a runner's whole queue |
| `runner logs <id>` | `GET /api/jobs/{id}` (`logTail`), `GET /api/jobs/{id}/stream` with `--follow` | the default is a **tail** and says so; `--follow` streams until the job is terminal |
| `runner evidence <id> --out <file>` | `GET /api/bundles/download` | downloads to a **unique** temp file beside `--out` (`--temp` overrides), verifies the outer sha256 the job record reported, then commits with a **no-overwrite** primitive — a destination that appears mid-download is a `destinationExists` refusal, never a replacement. An explicit `--temp` path is created **exclusively**: a file already there is a `tempPathExists` refusal, not a truncation. **Does not import** — it prints the exact `steerlab bundle import … --sha256 …` |

The commit is `link` + `unlink`, which cannot overwrite and has no window in
which a partial file wears the destination's name. One stated residual: on a
filesystem without hardlinks (FAT/exFAT, some network mounts) the client falls
back to an `O_EXCL` reservation and copies into it, so the destination is
visible — empty, then partial — while the copy runs. It still cannot overwrite
anything, and a failed copy removes it again; a reader who must be certain has
the outer digest.

The staging file is reached by **descriptor**, never by name, from the moment
it is reserved: the reservation's handle is what the download writes through
and what the digest is read back through, so the staging path cannot be
swapped for a symlink and made to truncate its target. The one step that must
still take a name is the `link` itself, and it is checked rather than trusted —
the published file's inode is compared with the descriptor's, and a mismatch is
a `stagingPathHijacked` refusal with the fresh link removed again. A second
writer that gets into the staging file after the last chunk is a
`stagedBytesChanged` refusal. Neither creates the destination.

Connection flags, identical on all six: `--runner <url>`, `--token-file
<path>`, `--timeout <seconds>`, `--ca-bundle <path>`. A `--runner` URL must
be plain `http`/`https` with a host and nothing else: embedded credentials
(`user:pass@`), query strings, fragments, and a port that is not a number in
1–65535 are all refused (`runnerURLRefused`) — URL userinfo would otherwise
flow into provenance and diagnostics, where bearer-token scrubbing cannot see
it. **These refusals name the category, never the value.** A token typed into
`?token=…` or `#token=…` is not echoed back into the envelope, the log line, or
your shell history; if one was, treat it as exposed and rotate it.

#### `runner serve` — this machine as a managed local runner (Phase 3)

```bash
# From a checkout (the package is not yet published to PyPI):
pip install -e "Server[runner]"           # client + engine, one distribution
steerlab runner serve                     # loopback, token mode, foreground
```

`runner serve` is the one verb in the family that does **not** address a
runner: it becomes one. It starts the engine's own service — `python -m
steerlab_server.cli serve`, unchanged, as a subprocess on the same interpreter
— on `127.0.0.1`, in **token mode always**, under a **runner-owned root**, and
prints the URL, the token file's path, and the exact invocation that uses it:

```
runner: http://127.0.0.1:59640
  runner root: ~/Library/Application Support/SteerLab/local-runner  (RUNNER-OWNED — not a workspace)
  token file:  …/local-runner/runner.token  (minted; the VALUE is never printed)
  reach it with:
    steerlab runner capabilities --runner http://127.0.0.1:59640 --token-file …/runner.token
```

| | |
|---|---|
| flags | `--runner-root <dir>`, `--port <n>`, `--timeout <seconds>` — and deliberately no `--runner`, no token flag of any spelling, no `--host` |
| bind | `127.0.0.1` only. Serving to a network is `steerlab-server serve`'s decision, where the posture refusals that gate a non-loopback bind live (§2.3) |
| port | `--port <n>`, else a free ephemeral one, picked and printed. `--port` is **bind-tested before the engine starts**: a collision is a typed refusal naming the port, never a child that dies a moment after the parent claimed success |
| auth | token mode, always. A 0600 token file is minted under the runner root (reused across restarts, announced if its mode is loose). The **value** is never printed or put in any document |
| lifetime | foreground; v1 has no daemon management. ctrl-c (or SIGTERM) stops the engine too, with a one-line summary |
| executor | `local`. An inherited `STEERLAB_EXECUTOR=slurm` cannot make a laptop sbatch |

**The runner root, per platform** (override with `--runner-root <dir>`) — and
the platforms are the two that can serve:

| platform | default |
|---|---|
| macOS | `~/Library/Application Support/SteerLab/local-runner` |
| Linux / BSD | `$XDG_DATA_HOME/steerlab/local-runner` (default `~/.local/share/steerlab/local-runner`) |
| Windows | — `runner serve` is refused there; Windows is client-only (see the platform table above) |

It holds `runner.token`, the artifact tree (`prompts/`, `experiments/`,
`runs/`) and `.steerlab/` (the job database). All of it is a **cache**: delete
the whole directory and nothing a workspace holds goes with it.

**Why it is never your workspace.** Local and remote execution use the
identical bundle round trip — upload → submit → evidence → import — and there
is no privileged localhost path into the client workspace. So
`runner serve --runner-root <a workspace>` is a **typed refusal**
(`runnerRootIsWorkspace`), and so is nesting either way. A runner rooted in the
workspace could read and write it directly, which no remote runner can do, and
a study that "worked" against one would prove nothing about a study that has to
travel. (The Mac app's local **workbench** — interactive serving of a live
workspace — is a different service role and is unaffected.) The full ruling and
the tests that pin it: `docs/PORTABILITY-CONTRACTS.md` §9.

**`--json` is envelope-then-stream.** Because the verb finishes only when it is
stopped, `--json` emits **one** startup envelope on stdout the moment the
engine answers `/api/info` — `state: ready`, with `url`, `port`, `runnerRoot`,
`tokenFile`, `tokenFilePresent`, `authMode`, `enginePID` — and everything after
that is diagnostics on **stderr**, the engine's own log lines included. Every
other verb's document is a completion envelope; this one's is a startup
envelope, and the difference is why an agent can start a runner and use it in
the same script.

A light install (`pip install steerlab-server`, no extras) refuses this verb by
name — `runnerExtraMissing`, with `pip install 'steerlab-server[runner]'` as
the repair — **before** anything starts. Authoring, packaging and talking to a
*remote* runner keep working without the extra; only serving one yourself needs
it.

Five traps worth knowing before you rely on it:

- `declare-condition --alpha-units norm|raw` is **required**, baselines
  included — the same refusal, word for word, that the Mac gives (§3.3, G6).
- `experiment verify`, `experiment freeze` and `bundle package` used to import
  torch (gap **G7**), so a torch-free install could author and declare but not
  freeze. **Closed:** the SAE latent *declared* surface moved to the torch-free
  `steering.sae_latent_schema`, and a bare `pip install -e Server` now runs the
  whole authoring lifecycle. `[runner]` is for *executing*.
- **There is no `--token` flag, on any runner verb, deliberately.** argv is
  readable by every process on a shared login node. The token comes from
  `$STEERLAB_RUNNER_TOKEN` or `--token-file <path>` (a path, not a secret), and
  it is never written into the workspace or into any envelope — documents carry
  only the presence boolean `tokenPresent`. Note the variable is the
  **client's**, not the engine's `STEERLAB_AUTH_TOKEN`.
- **`runner submit` is not idempotent — never retry it blindly.** It creates a
  job and, on Slurm, spends an allocation. A timeout there means "run `runner
  jobs`", not "submit again". Upload and evidence-download *are* safe to retry.
  Nothing in this family retries anything on its own.
- **`--out` on `runner evidence` (and on `bundle package`) is the verb's
  argument, not the envelope's destination.** Every other verb uses `--out
  <file>` to write the JSON document; on these two the declaration wins, so
  `--out` names the archive. Get the document from stdout under `--json`.

#### `run` — the composite (Phase 5)

The six acts above, as one command. `--wait` is the **default**, because
waiting is the point:

```bash
steerlab run <experiment> --runner <url>                 # the whole round trip
steerlab run <experiment> --runner <url> --verb sweep --executor slurm
steerlab run <experiment> --runner <url> --no-wait       # submit and detach
```

A frozen study in your workspace becomes verified evidence in your workspace.
It **composes** the verbs above and reimplements none of them, so every refusal
you can meet is one of theirs, and `run --help` is the only place the flag list
is authoritative.

| # | stage | what it does | typical refusal |
|---|---|---|---|
| 1 | `load` | loads the study, checks it is **frozen**, re-verifies every pin | `experimentNotFrozen` (65) · `pinDrift` (65, with `error.gate`) |
| 2 | `package` | packages the run bundle locally, records its sha256 | `bundleRefused` (65) |
| 3 | `capabilities` | asks the runner whether it can execute **this verb on this executor** — *before* anything is uploaded | `runnerCannotExecute` (65), with `result.runnerOffers` |
| 4 | `upload` | streams the archive; the digest must agree across the socket | `uploadDigestMismatch` (65) |
| 5 | `submit` | submits with `--verb` / `--executor` and the submit endpoint's pass-throughs | `submitOutcomeUnknown` (70) |
| 6 | `wait` | polls `GET /api/jobs/{id}` to a terminal status | `waitDeadlineExceeded` (70) · `remoteJobFailed` (70) |
| 7 | `evidence` | downloads the job's bundle and verifies its outer digest | `evidenceNotPackaged` (65) · `evidenceDigestMismatch` (65) |
| 8 | `import` | verify-and-extract into your workspace with the out-of-band pin | `bundleRefused` (65) |
| 9 | `provenance` | stamps `runs/<runID>/remote-execution.json` | — |

`result.stages` carries **all nine rows, always**, each with a `state` (`ok`,
`skipped`, `refused`, `failed`, `notReached`) and the facts that stage
produced. A failure names `result.failedStage`. In human mode each transition
prints one `run[<stage>]: …` line on **stderr**.

| flag | meaning |
|---|---|
| `--runner <url>` | required. Where to execute |
| `--token-file <path>`, `--ca-bundle <path>` | as everywhere else; still **no `--token`** |
| `--verb <run\|sweep\|validate\|…>` | default `run`. The submit route's vocabulary |
| `--executor <local\|slurm>` | default: whatever the runner is configured for |
| `--dry-run` | prepare, schedule nothing. Accepted against any runner |
| `--target-root`, `--dtype`, `--device`, `--parallel` | pass-throughs, exactly as `runner submit` exposes them |
| `--no-evidence` | do not package or fetch evidence (stages 7–9 `skipped`) |
| `--no-wait` | detach right after submit |
| `--timeout <seconds>` | **the WAIT deadline**, default 24 h — see the trap below |
| `--request-timeout <seconds>` | the per-HTTP-request budget (what `--timeout` means on the `runner` verbs) |
| `--evidence-out <file.tar.gz>` | keep the downloaded archive at this path. Without it the archive is a courier: temp directory, imported, removed |
| `--max-bytes <n>` | the download size cap |

**Detaching, and re-attaching by hand.** `--no-wait` — and ctrl-c during the
wait, and `--timeout` expiry — all **detach**. None of them cancels the remote
job, ever; this client does not kill work it did not do. `--no-wait` and ctrl-c
answer `pending` (exit **12**, a success document with no `error` — the same
state the engine's `study submit` answers for asynchronous work in flight);
`--timeout` expiry answers `waitDeadlineExceeded` (70) and says the job is
still running. All three put the three commands that finish the job by hand in
`result.followUps`, runnable as printed:

```bash
steerlab runner jobs <id> --runner <url>
steerlab runner evidence <id> --out <file.tar.gz> --runner <url>
steerlab bundle import <file.tar.gz> --sha256 <digest>
```

**Evidence comes home from a failure too.** A job that fails after producing
partial output still gets its bundle fetched, verified, imported and stamped —
and the envelope is `remoteJobFailed` (70) carrying the runner's own error
sentence, because a partial is evidence about a *failure*, never a result.

**Provenance is additive.** `runs/<runID>/remote-execution.json` records the
runner's URL and `engineVersion`, the bundle sha256, the job id, the submitted
verb and executor, the timestamps and the outcome. Nothing under `experiments/`
is written — not the manifest, not `pinned/`, not any hash. The bearer token
appears nowhere in it; only `runner.tokenPresent`. Schema and rationale:
`PORTABILITY-CONTRACTS.md` §10.7.

Three traps specific to this verb:

- **`--timeout` means the WAIT deadline here, and the per-request budget on the
  `runner` verbs.** The divergence is deliberate — a caller of a composite is
  thinking about how long the whole thing may take — and the other meaning
  keeps its own spelling, `--request-timeout`, so neither has to be guessed.
- **`submit` is never retried, and only `submit`.** Upload and evidence
  download each get one automatic retry on a transport error (both are
  idempotent). A transport failure at submit leaves `submitOutcomeUnknown` and
  a repair that says to run `runner jobs` — the job may exist.
- **`--verb verify` legitimately brings nothing home.** `verify` writes no run
  directory, so it packages no evidence, and the machine ends at a typed
  `evidenceNotPackaged` (65) rather than pretending. Same for `--dry-run`,
  which reports stages 7–9 as `skipped`.

**Do not carry a `~/.local/bin/steerlab` symlink or shim to the Swift CLI on a
machine that has this client.** `AGENTS.md` step 1 now tells the installing
agent to give the Swift CLI its own name, `steerlab-cli`, and to leave this
spelling to the console script; keeping both means `steerlab` means two
different things on two machines, which is exactly the failure §1.4 exists to
prevent.

### 1.3 Traps at a glance

Read §7 before a cluster session. The short list:

1. `steerlab-server study submit <exp>` **defaults to `--verb run`** — omitting
   `--verb` submits a full measured run, not the sweep/validate you meant.
2. `steerlab-cli remote submit-bundle` **also defaults to `--verb run`**.
3. ~~`steerlab-cli remote` defaults to `--url http://127.0.0.1:8000`, but both
   `serve` verbs default to port **8080**.~~ **Fixed 2026-08-18:** `remote`
   now defaults to `http://127.0.0.1:8080`, the port every server surface
   serves. The default client and the default server meet.
4. `--shard k/K` on `bundle execute` is the RAW knob and **nothing merges its
   partials**. The merging fan-out has three entry points — `study submit
   --parallel N` (2026-08-07), `steerlab-cli remote submit-bundle --parallel
   N` (2026-08-28, §3.8), and `parallelJobs` on
   `POST /api/studies/submit-bundle` — and in all three the merge is
   performed by a **running** `steerlab-server serve`, not by the submitting
   process. See §5.3.
5. `--parallel N` on `study submit` **refuses** on a verb or executor that
   cannot shard; the bundle paths (`remote submit-bundle --parallel`, the
   API's `parallelJobs`) **degrade with a note** in the same situation. Same
   rules, different answer, on purpose (§5.1). The Swift client says which it
   did: the envelope echoes `parallelJobsRequested` /
   `parallelJobsEncoded` / `parallelJobsSuppressedBecause` (§3.8).
5a. **A fan-out can partially fail while the submit still exits 0.** The abort
   is reported through the *parent* job record, not the submitting process's
   exit status — so verify the shard jobs landed in the scheduler queue
   rather than trusting the exit code, and stagger submissions at a site that
   caps queued jobs per user (§3.8, §5.3).
6. GPU type is chosen by **`--gres`** (`--gres A100` → `gpu:A100:1`), not by
   any `--gpu`/`--gpu-type` flag; unset, it falls back to the site profile's
   `STEERLAB_SLURM_GRES`. There is no other lever (§5.1).
7. `experiment pipeline` and `experiment judge-worker` are implemented on the
   Python CLI but absent from its printed verb list.
8. Most malformed env vars degrade silently to their default. `steerlab-server
   profile show` is the only way to see what actually took effect. The one
   exception is `STEERLAB_AUTH_MODE` at `serve` time: a value that is not
   `none`/`token`/`external` is announced and the secure default applies
   (§2.3) — a typo there must not buy an open server.
9. `steerlab-cli cluster ensure` with no `--allow-…` flag returns
   **`needsApproval` (exit 11)** and names the flag it wants. That is the
   design, not a failure — every remote side effect is authorized separately
   (§3.9.4).
10. `steerlab-server finetune train <config>` trains **in this process**, with
    no preflight, no checkpoint flag, and no auto-resubmit. The evidence-grade
    path is `finetune submit <request>` / `POST /api/finetune/submit` →
    `finetune execute <job-dir>` on a Slurm node (§5.5); so is
    `/api/finetune/train`'s daemon route, which refuses `evidenceGrade: true`
    outright. `submit` takes the **camelCase wire request**; `plan`/`train`
    take the **snake_case resolved config**. Handing one to the other is
    refused by name.
11. `cluster` and `remote` disagree about `--json` on purpose: `cluster` emits
    exactly one versioned envelope and exits with a state-derived code (0/10/
    11/12/13/64/70); `remote` prints raw server JSON and exits 0/1. Do not
    parse one like the other.

---

## 2. Environment variables

Both engines resolve their artifact root from the environment before any verb
runs. Getting this wrong silently points `prompts/`, `experiments/` and `runs/`
at the wrong tree, which is why the server prints its resolved root at startup.

**The two engines use different root variables**: Python reads
`STEERLAB_ROOT`, Swift reads `STEERLAB_WORKSPACE`. Set both to the same
directory when you want the engines to share one tree.

### 2.1 Roots

| Variable | Engine | Default | Effect |
|---|---|---|---|
| `STEERLAB_ROOT` | Python | **current working directory** | Artifact root. `--root <dir>` on any verb exports it before anything imports the app. Mutable at runtime via `POST /api/workspace/switch`, subject to the workspace-switch policy. |
| `STEERLAB_WORKSPACE` | Swift | see precedence below | Data workspace root. Read **once** per process. |
| `STEERLAB_METADATA_ROOT` | Python | `<root>/.steerlab` | Holds `jobs.sqlite` and the GPU-session discovery record. `profile validate` **fails** if it is not writable. (The Swift provisioner's remote default is `~/.steerlab` — the two differ.) |
| `STEERLAB_RUN_ROOT` | Python | `<root>/runs` | Where run directories land. Unset on a `cluster` profile is a `profile validate` **fail**. |
| `STEERLAB_ASSET_ROOT` | Python | none | Inherited into Slurm children. Unset on `cluster` is a `warn`. |
| `STEERLAB_ARCHIVE_ROOT` | Python | none | Archive location. |
| `STEERLAB_NODE_CACHE_ROOT` | Python | none | Becomes a Slurm child's default `HF_HOME`. |
| `STEERLAB_JOBS_DB` | Python | `<metadata root>/jobs.sqlite` | Job store path. |
| `STEERLAB_WORKSPACE_PARENT` | Python | none | When set, `/api/workspace/switch` may only select roots beneath it. |

Swift precedence (`WorkspaceRoot`, `Sources/ExperimentKit/WorkspaceStore.swift`):
`STEERLAB_WORKSPACE` → `--workspace <dir>` / app override → UserDefaults
`SteerLabWorkspaceRoot` (only if the directory still exists) → the compiled-in
seed root (the code checkout; dev/test fallback). Because the env var outranks
everything, the app's workspace switcher reports "pinned by STEERLAB_WORKSPACE
— switch by relaunching without it" rather than pretending a switch would
stick.

Python precedence: `--root <dir>` → `STEERLAB_ROOT` → cwd. `--root` is popped
globally before dispatch and the directory must exist (`--root '<path>' is not
a directory`, exit 64). `serve` announces which of the three it used, warns
when the root has no `prompts/`/`experiments/`, and warns again when the root
looks like the source checkout rather than a data workspace.

### 2.2 Deployment shape

These are parsed by a `_choice` helper that lowercases and strips — and
**silently falls back to the default on an invalid value**. `STEERLAB_EXECUTOR=sluurm`
becomes `local` with no diagnostic.

| Variable | Allowed | Default |
|---|---|---|
| `STEERLAB_SERVER_PROFILE` | `local`, `workstation`, `cluster` | `local` |
| `STEERLAB_LAUNCH_TOPOLOGY` | `local`, `ood`, `tunnel`, `batch` | `local` |
| `STEERLAB_AUTH_MODE` | `none`, `token`, `external` | `none` when READ (`ServerProfile.from_env`); `serve` resolves and exports **`token`** when the variable is unset — see §2.3 |
| `STEERLAB_EXECUTOR` | `local`, `slurm` | `local` |
| `STEERLAB_SERVER_ROLE` | `controller`, `gpu-session`, `workstation` | derived: `batch` topology → `controller`; `cluster`+`slurm` → `controller`; else `workstation` |
| `STEERLAB_BIND` | any host | `127.0.0.1` |

`STEERLAB_EXECUTOR=slurm` is a **capability declaration, not a preference**: it
gates Slurm study submission (`Slurm study submission requires
STEERLAB_EXECUTOR=slurm`), gates GPU-session start (403), starts the background
job reconciler, and turns on the `slurm:*` profile checks. Declaring it is also
what makes those routes token-gated — so it is a security boundary.

### 2.3 Auth

The auth middleware is the outermost middleware and gates only `/api/` paths.

```
require_token = auth_mode == "token"
             or (privileged and (executor == "slurm"
                                 or profile != "local"
                                 or bind is not loopback))
```

**`serve` resolves the auth mode before the app is built (WP-S, 2026-08-18)**
and exports it, so the per-request `ServerProfile.from_env` sees an explicit
decision. `from_env` itself still defaults to `none` — that default is what a
bare `uvicorn steerlab_server.api.app:app` or an embedding process gets, and
it is the documented residual. `serve`'s three rules, in order:

1. `STEERLAB_AUTH_MODE` set to a VALID value wins. `none` is **refused** (exit
   64) when the bind is non-loopback or `STEERLAB_EXECUTOR=slurm`. An INVALID
   value no longer degrades silently to `none` here — it is announced and the
   default posture applies (§7's silent-degrade trap does not reach this
   decision).
2. `--dev-open-loopback` / `STEERLAB_DEV_OPEN_LOOPBACK=1` → `none`, refused
   (exit 64) under the same two conditions.
3. Otherwise → **`token`, the default on every platform**, hydrating
   `STEERLAB_AUTH_TOKEN` from `STEERLAB_AUTH_TOKEN_FILE` and creating that
   file (0600) if absent.

- **`auth_mode=none`** (now reachable only by asking): the privileged set
  requires a token only when the deployment is non-trivial. On a `local` +
  loopback + local-executor process **every route is open**, including the
  privileged list (the CSRF/Origin guard still applies).
- **`STEERLAB_AUTH_MODE=token`**: every `/api/*` route is gated, including
  read-only listings.

The privileged set is **mutating-by-default**: every `POST`/`PUT`/`DELETE`/
`PATCH` under `/api/` is privileged unless it is one of these route templates
(`_OPEN_MUTATING_PATHS` in `api/app.py` — tokenizer-only or parse-only, no
writes, no model, no caller-named paths):
`/api/jlens/token-options`, `/api/jlens/decode-tokens`,
`/api/generation-prompt`, `/api/concept/import`,
`/api/concept/{name}/probe-import`. Matching is shape-exact (segment count and
literals), so no trailing-slash or extra-segment spelling can escape the gate.
A completeness test walks the live route table, so a new mutating route joins
the gate by default.

On top of that, these prefixes are privileged for **any** method, which is how
the sensitive READS stay gated (exact prefixes, so read-only siblings stay
open):
`/api/slurm/submit`, `/api/slurm/bundle`, `/api/bundles/upload`,
`/api/bundles/download`, `/api/bundles/import`, `/api/studies/submit`,
`/api/studies/submit-bundle`, `/api/finetune/train`, `/api/variants/upload`,
`/api/jobs/reconcile`, `/api/models/install`, `/api/jlens/lenses/acquire`,
`/api/jlens/lenses/import`, `/api/jlens/directions/derive`, `/api/jlens/support`,
`/api/jlens/qualify`, `/api/jlens/g0`, `/api/jlens/report`,
`/api/workspace/switch`, `/api/reader/`, `/api/vectors/backfill-norms`,
`/api/housekeeping/refresh`, `/api/housekeeping/maintenance`, `/api/session`,
`/api/scenario/` — plus `POST`/`PUT` on `/api/experiment/*` and
`POST /api/jobs/{id}/resubmit`. The trailing slashes on `/api/reader/` and
`/api/scenario/` are load-bearing.

| Variable | Default | Effect |
|---|---|---|
| `STEERLAB_AUTH_TOKEN` | unset | Bearer token. Required-but-unset → **503**; set but mismatched → **401**. Swift `remote --token` wins over it. |
| `STEERLAB_AUTH_TOKEN_FILE` | `~/.steerlab-token` | Path indirection so the secret never lands in `run.sbatch`/`bundle.json`, which are durable artifacts on shared storage. `serve` hydrates the token from it — and, in the DEFAULT posture only, creates it (0600, `secrets.token_urlsafe(32)`) when absent. An explicitly declared `STEERLAB_AUTH_MODE=token` never mints one: a secret the peer was not handed is a silent mismatch. |
| `STEERLAB_DEV_OPEN_LOOPBACK` | unset | `1`/`true`/`yes`/`on` is the env spelling of `serve --dev-open-loopback` (for launchers that cannot pass argv). Refused on a non-loopback bind or with `STEERLAB_EXECUTOR=slurm`; ignored, with a printed note, when `STEERLAB_AUTH_MODE` is set to a valid value. |
| `STEERLAB_ALLOWED_ORIGINS` | empty | Comma-separated exact origins for the CSRF guard. Wildcards deliberately unsupported. |

Slurm bundles **refuse** any env key ending `_API_KEY`/`_TOKEN`/`_SECRET`/
`_PASSWORD`/`_CREDENTIAL(S)`, or named `ANTHROPIC_API_KEY`/`HF_TOKEN`/
`HUGGING_FACE_HUB_TOKEN`. `STEERLAB_AUTH_TOKEN` itself matches, so it is
structurally un-bundleable — hence the `_FILE` pattern.

A `gpu-session` worker **refuses to start** (exit 64 from the CLI, and again
from the app lifespan so a bare `uvicorn` launch cannot skip it) when it would
bind a non-loopback host with no token available. That branch hydrates from
the token file but never mints one, for the same reason as the table above.

### 2.4 Model residency — the two refusal variables

| Variable | Default | Effect |
|---|---|---|
| `STEERLAB_MAX_LOADED_MODELS` | one per CUDA device, else **1** (floored at 1) | How many models may be resident. |
| `STEERLAB_ALLOW_RESIDENT_MODELS` | unset | On a **controller** role, any resident model load is refused (409) unless this is `1`/`true`/`yes`. Escape hatch for a server run by hand inside an interactive GPU allocation. |
| `STEERLAB_ALLOW_CPU_LOAD` | unset | GPU-less loads of a cached snapshot over 2 GiB refuse; set it to restore warn-and-proceed. |
| `STEERLAB_DEVICE` | auto (CUDA → MPS → CPU) | Device override; an explicit `--device` outranks it. |

`STEERLAB_MAX_LOADED_MODELS` drives three distinct refusals:

1. **Eviction exhaustion** — every resident slot busy: "raise
   STEERLAB_MAX_LOADED_MODELS for more capacity".
2. **The different-model local judge refusal (evaluate)** — fires at evaluate
   *start*, never mid-panel, when a local judge resolves to a model ≠ the study
   model and the limit is < 2: "use the study model as judge (leave the judge's
   model empty), a claude judge, or raise the limit". The post-generation judge
   fan-out bypasses this entirely — judging becomes blinded packets for
   per-judge-model worker jobs instead of a second inline load.
3. **The sweep `judgeScore` capacity preflight** — counts *distinct resident
   identities* `(model, revision, dtype)`, not a per-judge yes/no, because
   asking "is capacity ≥ 2" per judge passed a three-model panel on a two-slot
   server that then died partway through the grid.

The controller-resident refusal has three session-aware wordings (no session
registered / registered but not dialable / running but this route is
undelegated); it never claims a session is absent when it merely cannot be
reached.

### 2.5 Serving and ports

| Variable | Default | Effect |
|---|---|---|
| `STEERLAB_BIND` | `127.0.0.1` | `serve` bind host when `--host` is absent. Non-loopback flips the privileged-route token requirement on. A cluster profile not on localhost is a `profile validate` **fail** (downgraded to `warn` for a token-mode GPU session with a token set). |
| `STEERLAB_SESSION_PORT` | derived from `SLURM_JOB_ID` | GPU-session port. A fixed 8081 collided when two session jobs landed on one multi-GPU node, so the default is allocation-unique. `--port` wins; the resolved number is stamped back into the environment so nothing downstream reads the literal `auto`. |
| `STEERLAB_SESSION_IDLE_MINUTES` | `0` (no idle expiry) | Idle shutdown. |
| `STEERLAB_SESSION_VISIBILITY_GRACE_SECONDS` | `600` | Within the window a scheduler-unknown job stays `queued` rather than being read as failed — reading it the other way once "ended" two PENDING sessions that then ran orphaned on billed GPUs. |
| `STEERLAB_SESSION_PROBE_LAPSE_SECONDS` | `180` | |
| `STEERLAB_SESSION_STARTING_DIAGNOSIS_SECONDS` | `180` | |
| `STEERLAB_SSE_HEARTBEAT_SECONDS` | `15` | |
| `STEERLAB_MAX_UPLOAD_BYTES` | 4 GiB | Read per request, so it is tunable without a restart. |

### 2.6 Slurm

`SlurmResources.from_env()` defaults:

| Variable | Default |
|---|---|
| `STEERLAB_SLURM_PARTITION` | none — a `profile validate` **fail** only when the site declares `partition` required (see `_REQUIRED_HEADERS`), otherwise a `warn` |
| `STEERLAB_SLURM_MEMORY` | none — same rule, under the `mem` token |
| `STEERLAB_SLURM_GRES` | none — unset is a `warn` naming this site's own vocabulary |
| `STEERLAB_SLURM_SCRATCH_GRES` | none — declare or omit. An opaque Slurm gres token requesting node-local scratch for the classes that stage models (e.g. `lscratch:100`, GB). Rendered APPENDED to the GPU gres (`--gres=gpu:A100:1,lscratch:100`), or alone when the class resolves no GPU gres, and only for the gres-carrying classes (`study`, `gpuSession`) — the controller and setup jobs stage nothing. A per-request `--gres` override replaces the GPU half only, so the site's scratch request survives it. Scheduling accounting only: Slurm does not enforce it, and the rendered script's EXIT trap is what reclaims the space. Malformed tokens refuse (safe Slurm-token alphabet). Rendered from the site profile's `constraints.storage.nodeScratchGres`; `bootstrap.sh --scratch-gres` for a hand run. |
| `STEERLAB_SLURM_WALLTIME` | `04:00:00` (the engine-generic fallback; a rendered env file always speaks for a site that declares one) |
| `STEERLAB_SLURM_ACCOUNT` | none |
| `STEERLAB_SLURM_REQUEUE` | false |
| `STEERLAB_SLURM_GPU_TYPES` | **empty — declare or refuse.** There is no built-in vocabulary (WP5 Step 8): with none declared, a typed `gres` raises rather than being validated against another site's hardware. The site profile's `scheduler.gpus` renders this key. |
| `STEERLAB_SLURM_GPU_VRAM` | empty |
| `STEERLAB_SLURM_SBATCH` / `_SQUEUE` / `_SACCT` / `_SCANCEL` | `sbatch` / `squeue` / `sacct` / `scancel` — all four binaries are site data; submit, poll and cancel resolve them at call time |
| `STEERLAB_SLURM_QOS` | none → no `--qos` header |
| `STEERLAB_SLURM_CONSTRAINT` | none → no `--constraint`; `&`-joined node features (Slurm's own AND syntax) |
| `STEERLAB_SLURM_RESERVATION` | none → no `--reservation` |
| `STEERLAB_SLURM_EXTRA_SBATCH` | empty; space-separated, one directive per element |
| `STEERLAB_SLURM_REQUIRED_HEADERS` | empty; comma-separated from `partition\|account\|mem\|ntasks\|cpusPerTask\|time\|qos\|gres`. A declared header with no value **refuses to render** the sbatch script. |
| `STEERLAB_SLURM_CPUS_PER_TASK` | `4` |
| `STEERLAB_SLURM_SIGNAL_SECONDS` | `600` (`0` emits no `--signal`) |
| `STEERLAB_SLURM_SIGNAL_TARGET` | `step`; `batch-forward`/`batch-direct` emit `--signal=B:USR1@N` |
| `STEERLAB_SLURM_EXPORT_MODE` | `none` → `#SBATCH --export=NONE`; `all` omits the header |
| `STEERLAB_AUTO_RESUBMIT` | false |
| `STEERLAB_AUTO_RESUBMIT_LIMIT` | `5`, counted from the root job |

Non-env resource defaults: `job_name=steerlab`, `gpus=1`, `use_srun=true`.
`--ntasks=1` is always emitted and is not site data: every SteerLab job is a
single srun'd child, and parallelism is separate jobs.

Three things worth knowing:

- **`gpus` alone renders no GPU directive — only `gres` does.** A job with
  `gpus=1` and no `gres` sees no CUDA and loads the model on CPU in float32.
  The preflight warns about exactly this; it happened invisibly in a live
  shakedown.
- `STEERLAB_SLURM_GPU_VRAM` is **site data with no built-in default** —
  `<type>:<GB>` pairs, e.g. `"A100:80,H100:80,L4:24"`, rendered from the site
  profile. Undeclared it is an empty table; a **malformed** entry **raises**
  rather than yielding one silently, because a silently empty table would let
  an unfittable job queue. It is what the memory-fit preflight compares
  against. (Corrected 2026-08-18 — this bullet used to state one institution's
  inventory as the engine's default, contradicting §2.6 two hundred lines
  above: audit §1.3 D5.)
- A `gpu:*` gres naming a type outside `STEERLAB_SLURM_GPU_TYPES` refuses, and
  so does any typed gres when the vocabulary is **undeclared**. The vocabulary
  is site data; the concrete-type-required rule is not.

`STEERLAB_AUTO_RESUBMIT` is **reconciler policy, not an sbatch directive** — it
never renders into the script (contrast `requeue`, which is Slurm's own
mechanism). `STEERLAB_AUTO_RESUBMIT_LIMIT` raises on a non-integer, because a
swallowed typo would unbound the resubmit chain. GPU sessions force requeue and
auto-resubmit **off** regardless of the environment: a chat session must never
respawn on a billed allocation.

Shell-only, consumed by the rendered sbatch prologue under `--export=NONE` and
never read by Python: `STEERLAB_MODULES`, `STEERLAB_CONDA_SH`,
`STEERLAB_CONDA_ENV`, `STEERLAB_VENV`.

| Variable | Default | Effect |
|---|---|---|
| `STEERLAB_PYTHON` | `sys.executable` | Interpreter for spawned child jobs and the GPU-session worker. Needed when the controller's interpreter is not valid on the compute node. |
| `STEERLAB_JOB_ID` | the record file's basename | Provenance stamp; the server sets it for every child job. |
| `STEERLAB_NODE_STAGE_DIR` | unset (staging off) | Node-local model staging, from the site profile's `constraints.storage.nodeStageDirTemplate`. Expanded **on the compute node**, so a literal `/lscratch/$SLURM_JOB_ID` lands under the worker's job id. Every rendered sbatch script removes **the directory this template names** in an EXIT trap — not a hardcoded `/lscratch/…` — when, and only when, the template embeds a job-scoping variable (`$SLURM_JOB_ID`, `$SLURM_JOBID`, `$SLURM_TMPDIR`; either `$X` or `${X}` spelling) *and* that variable is actually set on the node. Anything else — a shared node cache, a variable the engine does not authorise, a variable the node left unset — is removed by nobody, never by everybody. Pair it with `STEERLAB_SLURM_SCRATCH_GRES` so the space is also *requested*. |
| `STEERLAB_NODE_SCRATCH_PURGED_BY_SCHEDULER` | unset (the job cleans up) | Rendered from the site profile's `constraints.storage.nodeScratchPurgedByScheduler`. `1` declares that the **scheduler** reclaims node-local scratch at job end (an epilog), and rendered scripts then arm no cleanup trap — a job racing the epilog for the same directory adds risk and removes nothing the site was not going to remove. Absent is the safe default and renders exactly as before. |
| `STEERLAB_TRANSFER_METHOD` | none | Drives the capability snapshot's transfer advertisement; `profile validate` warns when unset on a cluster profile. |

### 2.7 Housekeeping, judging, generation

| Variable | Default | Effect |
|---|---|---|
| `STEERLAB_PURGE_DAYS` | `30` (example site scratch policy) | Purge-risk scan horizon. Unset on a cluster profile is a `warn` — set it to the site's real policy so the housekeeping card is honest. |
| `STEERLAB_MAX_BUNDLE_TOTAL_BYTES` | 64 GiB | Aggregate uncompressed cap on a bundle import, checked against declared sizes in preflight AND actual bytes while streaming. Sibling of the per-member cap. |
| `STEERLAB_MAX_BUNDLE_MEMBERS` | `10000` | Member-count cap on a bundle import; refused in preflight, before anything lands. |
| `STEERLAB_PURGE_WARN_DAYS` | `20` | |
| `STEERLAB_HOUSEKEEPING_SCAN_CAP` | `50000` files | |
| `STEERLAB_MAINTENANCE_CALENDAR` | none | Windows the walltime preflight checks against; a session that would die mid-window refuses at submission. |
| `STEERLAB_JUDGE_MODEL` | `claude-opus-4-8` | Read at **module import**, not per call. |
| `STEERLAB_PROPOSAL_MODEL` | `claude-opus-4-8` | Likewise. |
| `STEERLAB_JUDGE_KEY_FILE` | a `~/.steerlab-judge-key`-style path, mode 600 | A **path, never a secret**, which is why it is allowed through the secret-env filter and inherited into Slurm children. |
| `STEERLAB_SKIP_PROVIDER_PREFLIGHT` | unset | Disables the provider catalogue lookup for air-gapped sites. **Skipping is logged** — an unverified pin must never look like a verified one. |
| `STEERLAB_PREFILL_CHUNK` | `1024` | MPS-only; `0` disables chunking. CUDA and CPU never read it. |
| `STEERLAB_JLENS_REFERENCE_FP32` | unset | `1`/`true`/`yes`/`on`: run `jlens qualify`'s `referenceAgreement` check with the REFERENCE path's own tensors promoted to float32 (ours is float32 already), then restored. A diagnostic for one question — whether a deviation is the two paths' dtype-cast asymmetry — at the cost of a second copy of the output head. The check stamps `referenceFP32Forced` in **both** modes and says so in its detail line, so a run that agreed only under promotion can never be read as a default-mode acceptance. It does not touch the tolerance. |
| `STEERLAB_MEMORY_HEADROOM_GIB` | `16.0` | Floored at 0; a bad value is swallowed. |

### 2.8 Hugging Face

| Variable | Default | Effect |
|---|---|---|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | unset → cached login at `$HF_HOME/token` | Needed for gated repos such as `google/gemma-3-*`. `serve` prints an INFO line when none is found. Both names are blacklisted from Slurm bundles. |
| `HF_HUB_OFFLINE` | unset | `1` suppresses that INFO line (an offline worker loads from cache only). Model installs force it to `0` in the child process — the install verb is the one sanctioned online path. |
| `HF_HOME` | `~/.cache/huggingface` | Weight cache root (`$HF_HOME/hub`, `$HF_HOME/token`). `HF_HUB_CACHE` outranks it for the hub subdirectory. Propagating it to GPU-session workers is essential: without it a worker misses every model in the shared cache and re-downloads on a billed GPU. |

### 2.9 Truthiness

Two conventions coexist: the Slurm resource parser accepts
`1`/`true`/`yes`/`on`; the resident-model, CPU-load and preflight-skip gates
accept only `1`/`true`/`yes` — **no `on`**. Use `1`.

---

## 3. Swift `steerlab-cli`

### 3.1 Global form

<!-- GENERATED:swift-global BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-cli docs cli-reference --write`. Edit the table, not this block. -->

```
usage: steerlab-cli [--workspace <dir>] <family> <verb> … [--help] [--json]

  init [--home <dir>]                           Create the home layout's Workspaces/ and Sites/.
  workspace init <path>                         Create and seed a data workspace.
  experiment <verb> <name> …                    The study lifecycle.
  data check <experiment>                       Study-data readiness.
  vectors <verb> …                              Vector artifacts.
  remote <verb> (--site <id> | --url <server>)  Cluster client.
  cluster <verb> …                              Cluster lifecycle.
  install version | stamp | verify              This build's identity and the integrity of its install.
  authoring prompt <kind> …                     Generation prompts for missing study data.
  docs cli-reference [--check | --write]        Regenerate the reference document.
  panel <verb> …                                Panel scenarios and seat casting.
  artifacts audit [--json]                      Vector-sidecar audit.
  serve [--port N]                              The loopback web front end.
  --config <path.json>                          Smoke-test / toy-concept tasks.

global flags:
  --workspace <dir>  Data workspace root (precedence: STEERLAB_WORKSPACE env > --workspace > app-persisted choice > the code checkout).
  --version          Print this build's version and install layout, and run nothing (the same report as `install version`).
  --help             Print this page. `<family> --help` lists a family's verbs; `<family> <verb> --help` prints one verb's arguments.
  --json             Declared on every agent-path verb: exactly one envelope on stdout, every diagnostic on stderr.

exit codes: 0 ok · 64 malformed invocation · 65 refused · 66 not found · 70 failed  (--json: the envelope's `state` is authoritative)
```

| Global flag | Effect |
|---|---|
| `--workspace <dir>` | Data workspace root (precedence: STEERLAB_WORKSPACE env > --workspace > app-persisted choice > the code checkout). |
| `--version` | Print this build's version and install layout, and run nothing (the same report as `install version`). |
| `--help` | Print this page. `<family> --help` lists a family's verbs; `<family> <verb> --help` prints one verb's arguments. |
| `--json` | Declared on every agent-path verb: exactly one envelope on stdout, every diagnostic on stderr. |
<!-- GENERATED:swift-global END -->

`--workspace <dir>` is stripped before any verb sees it; a missing value exits
64. It is precedence #2 — `STEERLAB_WORKSPACE` wins over it. `--version` is a
rewrite of `install version` (§3.13), so the report, its envelope, and its
`--json` mode are one implementation rather than two.

### 3.2 Home layout and workspace

<!-- GENERATED:swift-workspace BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-cli docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-cli init [--home <dir>]
steerlab-cli workspace init <path>
```

| Verb | Purpose |
|---|---|
| `init` | Create the SteerLab home layout's Workspaces/ and Sites/ directories (default home ~/SteerLab). |
| `workspace init` | Create and seed a data workspace, and git-init it. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:swift-workspace END -->

`init` materializes the SteerLab home layout — a folder holding `Workspaces/`,
`Sites/`, and the code checkout as siblings, so one directory moves, backs up,
and is handed to an agent as a unit:

```text
~/SteerLab/
├── Workspaces/     study workspaces, one folder each, each its own git repo
├── Sites/          your PRIVATE site registry (cluster-site profiles, presets)
│   └── cluster-sites/   one JSON file per site — the app AND the CLI read this
└── <checkout>/     the code checkout — any folder name; detected by content
```

It is the one verb with no sub-verb, and it does exactly one thing: create the
two directories if they are absent, under `--home <dir>` or `~/SteerLab`. It is
idempotent (a second run reports `existing` for everything and exits 0) and
never destructive — an existing directory is left exactly as it is, and a path
occupied by a *file* is refused (exit 64 in `--json`) rather than replaced. It
does not create or clone the checkout, does not `git init` anything, does not
create a workspace, and **does not touch workspace resolution**: no
`STEERLAB_WORKSPACE`, no persisted preference, no override. The chain in §2.1 is
what it always was.

`Sites/` is deliberately left EMPTY by `init` — it is normally a clone of your
own private site repository, and `git clone <repo> Sites` refuses a non-empty
target. Inside it, `Sites/cluster-sites/` is THE canonical cluster-site
registry (§3.9.6): one JSON file per site, read and written by the app and by
`cluster sites import`/`list`, created at the first write. **You sync it** —
git is how your sites reach your other machines, and SteerLab never runs git:
its writes leave the tree dirty and committing is your act. Tokens, keys, and
passwords never enter it (Keychain, per machine); connection state never enters
it either (`site-runtime.json`, per machine). Keeping site configuration there
rather than in a workspace is what stops it travelling with a workspace you
share.

The report also names any code checkout it finds in the home, detected by
CONTENT (a directory holding both `Package.swift` and `Server/steerlab_server/`)
rather than by folder name, since a clone lands under whatever name it was
cloned as. When the home holds none, it names the running binary's own checkout
instead. Both are informational: an app-only install legitimately has no
checkout, and that is never an error.

`workspace init` creates and seeds a data workspace (`prompts/`, `experiments/`,
`runs/`), `git init`s it, and prints how to point the CLI at it. Any other
subverb errors with the usage line.

### 3.3 Experiment lifecycle (authoring)

<!-- GENERATED:swift-experiment-authoring BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-cli docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-cli experiment list
steerlab-cli experiment create <name> [--description <text>] --model <id> [--revision <commit>]
steerlab-cli experiment attach <name> <concept>… [--corpus <a,b,c>] [--extraction-rendering <json>] [--method <name>] [--pool-from <k>] [--project-neutral <k>] [--reading-position <label>] [--reference <concept>]
steerlab-cli experiment detach <name> <concept>…
steerlab-cli experiment pin-prompts <name> <prompts/…/file.jsonl>
steerlab-cli experiment pin-rubric <name> <prompts/rubrics/file.md> [--judge-pin <judge-name>=<revision>[:<dtype>]] [--judges <spec>]
steerlab-cli experiment declare-condition <name> <condition> [--alpha-units <norm|raw>] [--band-width <k>] [--baseline] [--control <name>] [--slots <spec>]
steerlab-cli experiment set-sweep-selection <name> [--capability-tolerance <ratio>] [--choice-prompts <path>] [--coherence-backstop <ratio>] [--coherence-floor <ratio>] [--coherence-ratio <ratio>] [--control-apply-to <winner|topK>] [--control-margin <margin>] [--control-top-k <k>] [--objective <metric>]
steerlab-cli experiment set-sweep-grid <name> [--alphas <a1,a2,…>] [--battery <path>] [--dev-prompts <path>] [--layer-fractions <f1,f2,…>] [--layers <L1,L2,…>] [--max-tokens <n>]
steerlab-cli experiment set-instruments <name> <instrument>[,…] [--ordinal-aggregation <expectedValue|argmax>]
steerlab-cli experiment set-sampling <name> [--max-tokens <n>] [--prompt-mode <chatAssistant|rawCompletion>] [--samples-per-item <n>] [--seed-policy <manifestSeeds|derivedSHA256>] [--temperature <t>]
steerlab-cli experiment set-exclusions <name> <rule>[,…] [--endpoint <key>] [--max <x>] [--min <x>]
steerlab-cli experiment set-parser <name> <parser>
steerlab-cli experiment set-instrument-scope <name> <responseFormat>[,…]
steerlab-cli experiment set-style-taxonomy <name> <prompts/taxonomies/file.json>
steerlab-cli experiment verify <name>
steerlab-cli experiment freeze <name> [--force] [--run-substrate <local|server>]
steerlab-cli experiment duplicate <name> <new-name>
```

| Verb | Purpose |
|---|---|
| `experiment list` | List this workspace's experiments with their status. |
| `experiment create` | Create a draft manifest pinned to a model. |
| `experiment attach` | Pin each named concept's stimulus hash and extraction options. |
| `experiment detach` | Remove each named concept's pin from a draft — refused while a declaration still names one. |
| `experiment pin-prompts` | Pin the measured task-prompt file and its hash ("" clears the pin). |
| `experiment pin-rubric` | Pin the judging rubric, the judge panel, and the evaluation declaration they imply; --judge-pin declares a local judge's revision and dtype (repeat per judge). |
| `experiment declare-condition` | Declare one experimental arm, or the explicit baseline. |
| `experiment set-sweep-selection` | Declare the sweep's selection criterion as manifest data. |
| `experiment set-sweep-grid` | Declare the sweep's layer × alpha grid, its instrument files, and its per-cell token budget. |
| `experiment set-instruments` | Declare which outcome instruments the run measures (sampledText, answerTokenLogprob, choiceProbability, repeReaderScore, ordinalScale; "" clears the declaration). |
| `experiment set-sampling` | Declare the generation protocol: temperature, token budget, prompt mode (chatAssistant, rawCompletion), and the stochastic replication policy (samples per item × seed policy: manifestSeeds, derivedSHA256). |
| `experiment set-exclusions` | Declare the record-exclusion rules analysis applies (failedAttentionCheck, unparseableEndpoint, outOfRange; "" clears the declaration). |
| `experiment set-parser` | Declare the numeric-endpoint parser from prompts/parsers/parser-registry.json and pin that registry's hash ("" clears both). |
| `experiment set-instrument-scope` | Declare which response formats the option-consuming outcome instruments apply to (label, json, freeText), pinning the row set they select; "" clears the declaration. |
| `experiment set-style-taxonomy` | Pin the reasoning-style taxonomy and its hash. |
| `experiment verify` | Re-check every pinned input against the file bytes on disk. |
| `experiment freeze` | Freeze the manifest one-way, after the evidence gates pass. |
| `experiment duplicate` | Copy a manifest into a new draft — how a frozen study is iterated. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:swift-experiment-authoring END -->

Two spellings the synopsis above cannot show: `declare-condition … --baseline`
takes no `--slots` (it is the explicit no-intervention arm) but still requires
`--alpha-units`, and
`set-sweep-selection … --objective ""` clears the declaration, after which the
sweep resolves to the historical `markerDensity` rule.

**`set-sweep-selection`** — the sweep's decision rule, declared as manifest data
before the sweep runs. That is what makes the selection preregistered rather
than chosen after seeing the grid.

Its flags are **independent axes of one block, and the verb MERGES**: a
re-declare that names no coherence flag keeps the declared coherence rule, one
that names no control flag keeps the matched-norm control, and so on. This is
the same axis-by-axis behaviour `set-sweep-grid` has always had for the block's
other half, and it exists because a re-declare used to *replace*: typing
`--objective judgeScore` on a draft duplicated from a donor silently deleted the
donor's `controls` block, and arms ran with no matched-norm control with nothing
said. Whatever a merge carried over is named in the success line and echoed in
`--json` as `inheritedFromExistingDeclaration`, and the envelope now echoes the
**whole resulting `selection` block** so nothing changes invisibly. To REMOVE
the control rather than keep it, pass `--control-margin ""`; to clear the whole
declaration, `--objective ""`.

The merge is **field by field all the way down**, not object by object:
`--coherence-ratio` alone keeps the declared `--coherence-backstop` (rather than
resetting it to `0.6`) and vice versa, `--control-margin` alone keeps the
control's `topK` targeting, and either targeting flag alone inherits the
declared margin. Naming a coherence flag on a block carrying the legacy
absolute floor still *changes the form* — that is what the flag asks for — but
the discarded floor is named in `inheritedFromExistingDeclaration` and is never
silently converted into the backstop.

**The coherence floor is baseline-relative** (default for new declarations). An
absolute distinct-2 number cannot know what the model's own prose looks like: a
sweep admitted a cell at distinct-2 0.535 against a baseline of 0.989 — barely
half the coherence the unsteered model produced, on output 65% longer — and its
`logprobShift` was *repetition*, not steering. 0.535 clears 0.45, so the gate
said yes.

| Flag | Default | Meaning |
|---|---|---|
| `--objective` | **required** | `markerDensity` \| `judgeScore` \| `logprobShift`. `""` clears the whole declaration. |
| `--choice-prompts` | none | `logprobShift` only: the hashed dev choice-prompt JSONL the objective scores. |
| `--capability-tolerance` | `0.15` | Battery accuracy may drop at most this far below the α=0 baseline's. |
| `--coherence-ratio` | `0.85` | A cell's distinct-2 must be at least this fraction of the **α=0 baseline cell's** distinct-2. Range `(0, 1]`. |
| `--coherence-backstop` | `0.6` | The absolute distinct-2 no cell may fall below whatever the baseline was — what stops a degenerate baseline licensing a degenerate winner. Range `[0, 1)`, and it must sit **under** the ratio. |
| `--coherence-floor` | — | The **legacy absolute** rule: a fixed distinct-2 floor, `[0, 1]`. Declares a different form from the two flags above, so passing it alongside either is refused. |
| `--control-margin` | none | The winner must beat a matched-norm random direction by this margin. Inherited when only the targeting flags are named; `""` removes a declared control. |
| `--control-apply-to` | `winner` | `winner` \| `topK`. A change to `winner` drops any declared width — a winner-scoped control covers one cell — and the echo names the width it dropped. Typing `winner` **and** a width is refused: they are two different controls. |
| `--control-top-k` | — | Required with `--control-apply-to topK`, unless the existing declaration already carries a width (which is then inherited, never defaulted). Named **alone** on a winner-scoped control it selects `topK` scope — a declared width is a declared scope — and the echo says so. |

Which rule a criterion declares is decided by the **presence** of
`coherenceRatioToBaseline` / `coherenceAbsoluteBackstop` in its constraints
block — never by their values. A constraints block carrying neither means the
absolute rule at its declared (or default `0.45`) `coherenceFloor`, and will
mean that forever: **every criterion pinned before this change keeps the exact
semantics it ran under**, including the ones stamped into frozen manifests and
agent birth certificates. Only new declarations take the relative form, and
`set-sweep-selection` writes it explicitly rather than leaving it to be
inferred.

`sweep.csv` reports the number the relative floor gates on for **every** cell,
whichever rule is in force: `distinct2Ratio` (this cell's distinct-2 over the
baseline's) sits beside `distinct2`, and `lengthInflated` flags a cell whose
mean output ran more than 1.5× the baseline's length. The flag is **reported,
never gated on** — length inflation is a fact a reader needs before believing a
metric that repetition can move, not a rule about which cells may win. Both
engines now write the identical column order:

```
concept,layer,alpha,markerDensity,distinct2,distinct2Ratio,words,lengthInflated,batteryAccuracy[,objective]
```

**`create`** — `--model` is required (usage error otherwise). `--revision` pins
the model commit; unpinned, freeze will demand it (or auto-pin from the local
HF cache).

**`attach`** — pins each concept's current stimulus hash plus extraction
options into the manifest.

| Flag | Default | Meaning |
|---|---|---|
| `--method` | `meanDifference` (CAA) | One of `meanDifference`, `lat`, `emotionGrandMean`, `designatedReference`. An unknown value lists the valid set and errors. |
| `--pool-from K` | last-token reading | Reading position becomes `meanFromToken(K)`; `K` must be an integer ≥ 0. |
| `--reference <concept>` | none | `designatedReference` only: the concept subtracted from the target. |
| `--corpus a,b,c` | none | `emotionGrandMean` only: extra corpus members in the grand mean. |
| `--project-neutral K` | off | **Legacy, draft-only.** Prints a warning; verified/frozen experiments reject it. Requires `prompts/neutral/corpus.jsonl` or it errors. |

`attach` also pins the neutral corpus whenever one exists — it denominates
norm-unit alphas, so it is a pinned input, not a convenience.

Concept names are the positional arguments; the parser drops anything starting
with `--` *and* anything equal to a consumed flag value. A concept named
identically to a flag value would be silently dropped — an edge, not something
sane concept names hit.

**`pin-prompts`**, **`pin-rubric`**, **`declare-condition`** are the headless
authoring three (WP0 audit §8, P0-3: before them the CLI could pin concepts and
nothing else, so no agent could construct a study that measures anything). All
three are draft-only — a frozen or completed manifest refuses with the one
immutability line — and all three go through the same store setters the Studies
panel goes through, so a CLI-authored and a GUI-authored manifest are
byte-identical.

**`pin-prompts`** — the measured task. Pins `taskPromptsFile` +
`taskPromptsHash` (SHA-256 of the raw bytes), parsing the file at pin time with
the *run loop's own* parser, so a file the run would refuse is refused here
instead of at generation time. An empty path (`""`) clears the pin.

**`pin-rubric`** — the judging instrument. Pins `judgeRubricFile` +
`judgeRubricHash`, optionally replaces the judge panel via `--judges`, and
writes the explicit `evaluation` declaration the pin pair implies (the
2026-07-22 incident: pins present, declaration absent, study dies at evaluate).
Judge fields are `<name>:<kind>[:<model>[:<provider>]]`; kinds are `claude`,
`local`, `openrouter`. A blank model is *absent*, not empty — a local judge then
resolves to the study model, a claude judge to the default judge model; the
fourth field is OpenRouter's serving-provider pin and is refused on the other
kinds. Declare any number of judges, including exactly one: a single-coder
design freezes cleanly and carries a non-blocking advisory saying no
inter-rater agreement will exist for its codings. Zero judges is what the
`judgeValidity` freeze gate refuses.

**`--judge-pin <judge-name>=<revision>[:<dtype>]`** — the local-judge weight
pins, repeated once per judge and keyed by the judge's *name* (the `panel
compile --seat` shape; position 4 of the `--judges` grammar is already
OpenRouter's provider, and these two fields are local-only). A **local judge
naming a model other than the study model** must pin both or `judgeValidity`
refuses at freeze — the judge is a second measurement instrument, and an
unpinned one is a different instrument next month. Dtypes are `bfloat16`,
`float16`, `float32` (`bf16`/`fp16`/`fp32` alias in, stored canonically). The
revision must be a **commit hash**: a branch or tag is re-pointed by
definition, so it cannot identify the weights a run used, and that is refused
at the declaration rather than at freeze. A `--judge-pin` naming no declared
judge, or aimed at a `claude`/`openrouter` judge (which carry no revision or
dtype), exits `64` rather than being written and normalized away.

```bash
steerlab-cli experiment pin-rubric <name> prompts/rubrics/default-paired-v1.md \
    --judges strict:local:<judge-model-id>,lenient:claude \
    --judge-pin strict=<commit-hash>:bfloat16
```

`--judges` replaces the **roster**; the pins **merge field by field beneath
it**. A judge whose name, kind and model all survive keeps its revision and
dtype; a judge whose model changed drops them, because they identify the old
bytes; either way the echo names which under
`result.inheritedFromExistingDeclaration` — the same key `set-sweep-selection`
uses. Before the merge existed, a headless re-declaration silently destroyed
pins written in the app and the study then refused at freeze for want of pins
it used to have.

**`declare-condition`** — the arm. Without one, a concept study runs the
implicit baseline alone and `analyze` reports zero effect sizes. Slots use the
manifest's own vocabulary; a multi-slot condition IS the linear mix
`h + Σ αᵢ·vᵢ` and hashes as one condition.

| Flag | Default | Meaning |
|---|---|---|
| `--slots` | required unless `--baseline` | `<concept>:<layer>:<alpha>[:add\|ablate]`, comma-separated. α when steering, λ when ablating. An explicit `add` parses and is not written (manifest bytes are the content hash). Every named concept must already be attached. |
| `--baseline` | off | Declares the no-intervention arm (no slots). Exclusive with `--slots`. |
| `--band-width K` | 1 | Layers per slot. |
| `--alpha-units norm\|raw` | **required** (no default) | `norm` denominates α by the residual-stream norm at that layer on the pinned neutral corpus — what makes α comparable across concepts; `raw` is α as typed. Required on every arm, `--baseline` included: it used to default to `norm` here and to `raw` on the server engine, so the same undeclared arm authored a different study depending on which engine served it (portability gap G6). α units are dose semantics, so neither engine guesses. |
| `--control` | none | `randomMatchedNorm` (steering) or `randomDirectionAblation` (ablation): the same slots with a deterministic random direction substituted. |

**`set-sweep-grid`** — the other half of the sweep block. `set-sweep-selection`
declares how a winner is picked; this declares what it is picked *from*. Until
it existed the axes were reachable only from the app's Optimizations panel, so
the only headless way to obtain a grid was `duplicate` — which brings the donor
study's **concepts** along with its sweep block, and a concept that arrives that
way is swept but cannot be cited (the passenger-concept problem).

Every flag is optional and each edits its own field, so one axis can move
without restating the block; a call with no flags is refused rather than
reported as a change. Draft-only.

| Flag | Default | Meaning |
|---|---|---|
| `--layer-fractions` | `0.5,0.7,0.85` | The layer axis as depths in `[0, 1]`, **ascending, each value once**. This is what the manifest stores, and the portable form: the same declaration names the proportionally same site in a 26-block model and a 62-block one, resolved at sweep time as `Int(depth · f)`. |
| `--layers` | — | The same axis as absolute block indices. Converted here to the depth fractions that resolve back to exactly those blocks, against the pinned model's depth **as read from an already-extracted vector**. With nothing extracted for the model there is no depth to read, and the verb refuses `missingPrerequisite` rather than assume one. Exclusive with `--layer-fractions`. |
| `--alphas` | `0.05,0.08,0.1,0.13` | The dose ladder, **ascending, each value once**, in residual-norm units above 0. `0` is the baseline cell every sweep runs anyway, so it is never a rung. |
| `--dev-prompts` | `prompts/dev/dev-prompts.jsonl` | The dev split the sweep generates on. Re-pointing it clears `sweep.devPromptsHash`; freeze re-pins from the bytes on disk. |
| `--battery` | `prompts/batteries/basic.jsonl` | The capability battery every cell is scored on. Re-pointing it clears `sweep.batteryHash` on the same rule. |
| `--max-tokens` | `80` | Tokens per swept cell — the grid's cost multiplier, not the study's generation length. |

Both axes must **ascend with no repeats**. `resolvedLayers` sorts and
deduplicates anyway, so an unordered or repeated declaration is a grid whose
written form and run form disagree; a repeated α is a cell paid for twice and
reported once, and a ladder that doubles back is not a dose-response. Violations
refuse `sweepGridRule`.

The manifest gains **no** key for absolute layers. An axis with two stored
spellings is an axis that can disagree with itself, and the depth those layers
were read against is a property of the pinned model, which the manifest already
names. Both forms are reported instead: the `--json` result carries
`layerFractions` *and* `resolvedLayers`, `layerCount` (null when nothing has been
extracted for the model yet — stated, never guessed), `cellCount`,
`collapsedFractions`, and `alphaUnits: "residualNorm"`.

Two legal fractions can still land on one layer at a given depth. That is not a
refusal — the declaration is legal and the collapse is a property of *this*
model's depth — but a grid of "four depths" that is really three is a silently
smaller sweep, so `collapsedFractions` reports it and the human line says so.

Typing a `set-sweep-selection` flag here (`--objective`, `--coherence-floor`,
`--choice-prompts`, `--capability-tolerance`, `--control-margin`,
`--control-apply-to`, `--control-top-k`) is answered with a pointer to the verb
that owns it rather than with this verb's flag list: the intent was right and
aimed one verb over.

**`set-sampling`** — the generation protocol, on the `set-sweep-grid` ownership
pattern: six manifest fields that had no writer on either CLI until 2026-08-28,
which is how a stochastic replication arm (25 samples × T=0.7 × 1024 tokens)
came to be uncuttable-from-a-design-by-hand. Draft-only, and it **merges**:
only the flags given move, `""` clears `--prompt-mode`/`--seed-policy`, and
`--samples-per-item 1` clears back to the deterministic default.

| Flag | Meaning |
|---|---|
| `--temperature <t>` | Sampling temperature. Non-numeric is refused at the write — a bad value bricks the manifest at the next decode. |
| `--max-tokens <n>` | Generation length for the measured run (not the sweep's per-cell budget, which is `set-sweep-grid --max-tokens`). |
| `--prompt-mode <chatAssistant\|rawCompletion>` | Closed vocabulary. Out-of-vocabulary is refused: downstream reads are equality tests, so a typo would silently behave as the default. |
| `--samples-per-item <n>` | Stochastic replication count. |
| `--seed-policy <manifestSeeds\|derivedSHA256>` | How each record's seed is derived. Closed vocabulary, same reason. |

The **joint** stochastic rules — `samplesPerItem > 1` requires
`temperature > 0` and `seedPolicy derivedSHA256` — stay `verify()` violations
rather than write-time refusals, deliberately: the fields must be declarable
one flag at a time, and a merge-style writer that refused a half-built
protocol could never reach the finished one. Declaring the protocol is legal
on either engine; §3.4's greedy-only rule then routes the *run* to the Python
engine.

**`set-exclusions <name> <rule>[,…]`** — the declared record-exclusion rules
analysis applies (`failedAttentionCheck`, `unparseableEndpoint`,
`outOfRange`); `""` clears the declaration. `--min`/`--max` bound the
`outOfRange` keep-window and `--endpoint` names the parsed-value key the
endpoint rules read (default `parsedMonths`). Refusals are the exclusion
engine's own violation sentences. Exclusions apply **at analysis time only**
and are stamped honestly — no record ever leaves `generations.jsonl`.

**`set-parser <name> <parser>`** — the manifest's `numericParser`, resolved by
name against the workspace registry (`prompts/parsers/parser-registry.json`),
shape-checked, and pinned together with that registry's current SHA-256 as
`parserRegistryHash`; `""` clears both. **The hash is never an argument.** The
registry file is the authority on which parser *version* the study
preregistered, so re-declaring the same name is also the drift repair, and a
caller who could type a hash could claim a provenance nothing computed. An
out-of-vocabulary name exits `64` with the registry's roster. Clearing the
declaration on a numeric study drops it back to the **deprecated implicit
selection** documented at the end of §3.5 (`caseFamily: "sentencing"` → the
built-in duration parser), which is why clearing emits the
`deprecatedImplicitSelection` advisory rather than passing in silence.

**`set-instrument-scope <name> <responseFormat>[,…]`** — which response
formats (`label`, `json`, `freeText`) the option-consuming instruments read,
pinning the row set they select (`itemCount` + `itemIDsHash`, computed from
the study's own pinned task prompts); `""` clears the declaration. This is the
**non-lossy** repair the run-start `responseFormat` refusal names: on a mixed
json+label prompt file it keeps `answerTokenLogprob`/`ordinalScale` on the
label rows, where the other named repair (`set-instruments … sampledText`)
drops the instrument entirely. A scope selecting **zero** rows is refused at
the declaration rather than producing zero records at the run, and an
out-of-vocabulary format exits `64`.

Both of the last two are also **client verbs** — `steerlab experiment
set-parser` / `set-instrument-scope` (§1.4) — spelled identically and refusing
identically. Neither is a field assignment, so neither key is reachable through
`set-protocol`: each *derives* its pin from a workspace file at the moment of
declaration, and no surface on either engine accepts a `parserRegistryHash` or
an `itemIDsHash` as input. The *engine* (`steerlab-server`) still redirects
both, because it executes rather than authors, and its redirect names the
client's spelling alongside the Mac's.
`docs/PORTABILITY-CONTRACTS.md` §7 carries the reasoning.

**`set-style-taxonomy`** validates that the taxonomy file loads on this engine,
then stamps its path and the SHA-256 of its bytes. Drift after pinning is a
verify violation. No pin = no reasoning-style scoring.

**`freeze`** — one-way. `--run-substrate server` matches the validate/battery
evidence gate against evidence produced on (or imported from) the *server*
engine — the substrate the measured runs will actually execute on — instead of
this one. `--force` skips the evidence gates (revision, validateEvidence,
batteryEvidence, judgeValidity, variantValidity, gitClean) and stamps the
manifest `freezeForced` + `forcedGatesSkipped`; the pin `verify()` is never
skippable. A forced freeze remains non-citable — but checkably so, by stamp.

**`verify`** — prints `OK [status]` or one `VIOLATION: …` line per drifted pin,
**exiting 1** when there is any violation.

### 3.4 Running

<!-- GENERATED:swift-experiment-running BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-cli docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-cli experiment extract <name>
steerlab-cli experiment validate <name>
steerlab-cli experiment sweep <name>
steerlab-cli experiment run <name> [--prompts <path>]
```

| Verb | Purpose |
|---|---|
| `experiment extract` | Derive the manifest's concept vectors on this engine. |
| `experiment validate` | Score each vector on its held-out probe and report cross-concept similarity. |
| `experiment sweep` | Sweep layer × alpha on the dev split and record a recommendation per concept. |
| `experiment run` | Generate the measured run for every declared condition. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:swift-experiment-running END -->

These load the model through MLX. `--prompts` overrides the manifest's pinned
task-prompt file; for a **frozen** manifest the override is pin-checked, so it
cannot silently substitute a different measurement input.

Local measured runs are **greedy-only** by construction: the runner requires
`temperature == 0` and rejects more than one seed, because the MLX generator
does not pin a per-run sampling seed. `manifest.seeds` is recorded for
provenance and every local generation record stamps `seedInert: true`.
Stochastic studies (`temperature > 0`, `samplesPerItem > 1`) belong on the
Python engine.

### 3.5 Analysis (pure CPU, no model load)

<!-- GENERATED:swift-experiment-analysis BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-cli docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-cli experiment analyze <name> [--allow-unverified-epoch]
steerlab-cli experiment rescore-style <name> [--allow-unverified-epoch] [--run <run-dir>]
steerlab-cli experiment evaluate <name> [--allow-unverified-epoch] [--run <run-dir>]
```

| Verb | Purpose |
|---|---|
| `experiment analyze` | Compute paired effect sizes from the newest completed run into a fresh run directory. |
| `experiment rescore-style` | Re-score reasoning style over a completed run into a fresh run directory. |
| `experiment evaluate` | Judge a completed run with the pinned rubric and judges. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:swift-experiment-analysis END -->

All three read a completed run under the **epoch guard**: the run's stamped
experiment hash must equal the live manifest's content hash, or the verb
refuses. `--allow-unverified-epoch` bypasses the guard only for legacy runs
carrying no stamp, and the output is then stamped `epochUnverified`. The guard
is per-engine (canonicalization differs) — analyze a run on the engine that
produced it.

**Measurement-drift tolerance — and the sanctioned re-measurement path.** The
guard's whole job is to refuse a run interpreted against settings that were
not in force when it generated, so it tolerates exactly the drift that
*cannot have moved a byte of the source run's generations*. Six manifest
fields are compared out of the epoch on the measurement verbs (`analyze`,
`evaluate`, `rescore-style`) — `judges`, `evaluation`, `pipeline`,
`judgeRubricFile`, `judgeRubricHash`, `humanValidation` — and `name` is
blanked on both sides by the same rule (identity, not a measurement setting;
a rename cannot have reached the generator). Every tolerated field is named in
the run's tolerated-drift stamp, like any other tolerated fact. `promote`
passes no tolerance and still refuses a renamed or re-judged manifest, because
a promotion is a claim about the settings that selected the cell.

What that buys is the **re-measure-without-touching-the-source** path, which
is the only sanctioned way to put a new instrument on a finished run:

```bash
steerlab-cli experiment duplicate <study> <study>-recoded
steerlab-cli experiment pin-rubric <study>-recoded prompts/rubrics/<new>.md \
    --judges a:local:<model>,b:claude --judge-pin a=<commit-hash>:bfloat16
steerlab-cli experiment evaluate <study>-recoded --run runs/<original-run-dir>
```

The duplicate carries the source study's generation-side pins byte-identically
(that is what keeps the epoch equal), the rename and the new rubric+panel are
the tolerated drift, and the ORIGINAL run directory is never mutated — the
evaluation writes beside it as always. Before `name` was blanked this refused
on the one field duplication itself must change, and the only working path was
mutating the source study in place, which is exactly what the firewall exists
to prevent. Methodology framing: `docs/CONDUCTING-A-STUDY.md` §4.7.

Source-run selection: `analyze` takes the newest completed run. `evaluate` and
`rescore-style` take `--run <dir>` (a name under `runs/`, or an absolute path);
absent, the newest completed run. `evaluate` with no completed run errors,
telling you to run first or pass `--run`.

`rescore-style` requires a pinned taxonomy; it writes `reasoning-style.csv` +
`reasoning-style.json` into a **fresh** run directory and never mutates the
source run.

`analyze` recomputes paired-to-baseline effect sizes (bootstrap CI + Wilcoxon,
plus the phase's multiple-comparison correction — BH-FDR for screens, Holm for
confirms, matching the server's rule) from the newest completed run's
generations, and writes `analysis.json` + `effect-sizes.csv` — plus
`choice-deltas.csv`/`.json` when the run carries answer-token instrument
readouts — into a **fresh** immutable analyze run directory. Runs are never
mutated. It errors when there is no completed run (`need generations.jsonl +
report.json under runs/`). Measured runs additionally write their own
`effect-sizes.csv` and carry `effectSizes` in `report.json`.

**Null-only endpoint rescue (server `analyze` only, 2026-08-10).** When the
manifest declares a `numericParser` (or fires the deprecated `caseFamily`
trigger below), the
server's `analyze` first re-parses every sampled record whose run-time
`parsedMonths` was null from its stored output, under the pinned grammar as
the live engine implements it — so a parser fix (e.g. accepting number-word
durations, "ten years and six months") reaches finished runs
without regenerating them. Null-only by design: run-time parses are never
overwritten, generations.jsonl is untouched, and a record no grammar parses
stays unparsed. The analyze output stamps `endpoint-reparse.json` (parser
provenance, rescued / still-unparsed counts, zero-rescue included). A
declared parser whose registry pin has drifted now **refuses `analyze`** with
the same message as `run` (it previously degraded to label-only resolution).
Swift `analyze` computes no months endpoints, so it has no rescue step.

**Adjudicated-endpoint intake (server `analyze` only, 2026-08-18).** The
rescue above is null-only *because* a run-time parse must never be silently
overwritten. When an external extraction campaign re-reads a finished run's
outputs and returns a value per record, that value *does* overwrite — so it
enters through its own verb flag, its own verification ladder, and its own
stamps. Nothing about `endpoint-reparse.json` changes.

```
steerlab-server experiment analyze <name> --source <run-dir> \
    --adjudicated-endpoint <file>
```

`--source` is **required** with `--adjudicated-endpoint` (usage, exit 64,
without it): an adjudication is evidence about one specific run, and
defaulting to the newest run would join it against the wrong generations.
One file adjudicates one run. Server engine only — every run this applies to
is a server run, and the epoch guard already routes `analyze` to the engine
that produced the run.

The file is a JSON object — or JSONL whose first line is that object without
`adjudications` and whose remaining lines are the rows:

```json
{
  "endpoint": "parsedMonths",
  "sourceRun": "<run-directory basename>",
  "sourceGenerationsSha256": "<sha256 of that run's generations.jsonl>",
  "extractionInstructionsSha256": "<sha256 the campaign claims>",
  "adjudications": [
    {"condition": "steer", "promptIndex": 0, "promptID": "c1",
     "sampleIndex": 0, "value": 130.0,
     "operativeQuote": "a term of 130 months"},
    {"condition": "steer", "promptIndex": 1, "promptID": "c2",
     "sampleIndex": 0, "value": null,
     "reason": "no operative sentence stated"}
  ]
}
```

The join key is the full sampled-record tuple `(condition, promptIndex,
promptID, sampleIndex)` — the analysis layer's own `(condition, promptID)`
cell key averages the sample axis away, which is the axis an adjudication
addresses. `extractionInstructionsSha256` is optional and names
`extraction-instructions.md` beside the adjudication file (or whatever
`extractionInstructionsFile` names).

Verification ladder, in order — every rung a **hard refusal** in the
`complete-judgment` style (an `ERROR:` line and exit 1, no new gate
vocabulary), except the last item, which never refuses:

| # | Check | Refusal |
|---|---|---|
| 1 | File shape | Not JSON/JSONL, no `endpoint`/`sourceRun`/`sourceGenerationsSha256`, empty `adjudications` |
| 2 | Endpoint | Not a substitutable record endpoint (`parsedMonths` today), or no sampled record of the run carries it |
| 3 | Source-run custody | `sourceRun` is not the run being analyzed; its `generations.jsonl` is missing or no longer hashes to `sourceGenerationsSha256` |
| 4 | Epoch | `analyze`'s existing guard — the adjudication adds nothing and bypasses nothing |
| 5 | Per-row | Duplicate join key; join key no record carries; non-numeric non-null `value`; `value: null` with no non-empty `reason`; a value with no `operativeQuote` |
| 6 | Quote custody | An `operativeQuote` that does not appear in that record's `output` (whitespace-normalized containment, no case folding) |
| 7 | Coverage | Any adjudicatable record absent from the file (silently partial); any row for a record outside the adjudicatable set (an error row, an instrument readout) |

The adjudicatable set is every non-error **sampled** record carrying the
endpoint key, present or explicitly null. An explicit `value: null` row is
full coverage — the adjudicator saying "unparsable" is an answer, not a gap.

`extractionInstructionsSha256` is the one **loud-stamp-never-refuse** item
(post-submit drift policy, as for `complete-judgment`'s judging
instructions): a mismatch against the local artifact, a claim with no
artifact, or an artifact with no claim each warn and stamp
`extractionInstructions.verified: false`; the analyze proceeds. Adjudications
already produced are evidence about what the campaign actually did — the
stamp makes the framing question checkable rather than silently unanswerable.

Substitution is **in memory only** (`generations.jsonl` stays immutable, the
analyze writes a fresh run directory as always), between the rescue and the
exclusion rules — so exclusions and the paired statistics both see adjudicated
values, and a record both rescued and adjudicated is accounted against its
*rescued* value. An adjudicated record a rule then excludes keeps its place in
the counts but its value never reaches the statistics, exactly as for a
run-time parse. What the analyze run directory gains:

- `adjudicated-endpoint.json` — endpoint, file sha256, source-run pins, the
  instructions block, `counts` (a five-way partition: `agree`, `differ`,
  `rescuedFromNull`, `nulledFromValue`, `unadjudicatable`, summing to
  `total`), `meanAbsDiff`/`maxAbsDiff` over the `differ` class, and the same
  counts per condition.
- `adjudication-divergence.csv` — one row per record whose adjudicated value
  differs from the value analyze would otherwise have used
  (`condition,promptIndex,promptID,sampleIndex,runTimeValue,adjudicatedValue,absDiff,divergence,quotePresent,reason`).
  The **full** list, not a sample: it is analysis evidence. Written
  header-only when nothing diverged.
- `config.json` → `notes.adjudicatedEndpoint` (`fileSha256` + the divergence
  summary), and the `--json` envelope's analysis payload carries
  `adjudicated: true` plus that summary, so a reader of the summary alone
  cannot miss that the endpoint values were substituted.

**How a study declares its numeric endpoint — and the one deprecated implicit
selection (2026-08-18).** The mechanism is the manifest's `numericParser`: the
name of an entry in the workspace parser registry
(`prompts/parsers/parser-registry.json`), which freeze pins by SHA-256 and
whose drift is a `verify()` violation like every other measurement-side input.
Both engines resolve it once at verb start; a missing, malformed, or drifted
registry refuses before any generation.

`caseFamily` is a **provenance label**: free text, decoded from every manifest,
printed in `report.json` and `preregistration.md`, and behaviorless — with one
exception, kept working for the manifests that already depend on it and
**deprecated**. With no `numericParser` declared, the exact value `sentencing`
still selects the built-in duration endpoint (`parsedMonths`). Nothing about
those runs changes; what changed is that they now say so. Every site where the
trigger actually fires emits a loud, non-blocking advisory —

```
ADVISORY: caseFamily 'sentencing' selected the built-in duration endpoint
implicitly — declare numericParser instead; this implicit selection is
deprecated. The shipped registry entry 'sentencing-months'
(prompts/parsers/parser-registry.json) reproduces this parser exactly.
```

— to the verb's log, to the run directory's `advisories.txt` where the verb
writes one, and to the `--json` envelope under the closed advisory code
`deprecatedImplicitSelection`. It is an **advisory, never a refusal**: exit
codes are unaffected (`state` is `okWithAdvisories`, exit 0). The sites, per
engine:

| Site | Engine | Predicate |
|---|---|---|
| `experiment run` — the per-record `parsedMonths` parse | both | no `numericParser` declared |
| `experiment run` — multi-agent panel-effects `months` endpoint | server | the label alone (a declared parser does **not** displace this one) |
| `experiment analyze` — null-only endpoint rescue | server | no `numericParser` declared |
| `experiment preflight-endpoints` — the `meanMonths` endpoint name | server | no `numericParser` declared |

A study that declares a `numericParser` gets no advisory anywhere but the
multi-agent panel row, because nothing was selected implicitly. The registry's
shipped `sentencing-months` entry reproduces the built-in parser exactly and is
fixture-locked on both engines, so the migration is one manifest key.

### 3.6 Promotion and confirmation (pure CPU)

<!-- GENERATED:swift-experiment-promotion BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-cli docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-cli experiment promote <name> <concept> [--agent-name <name>] [--cell <layer>:<alpha>] [--reason <text>]
steerlab-cli experiment confirm <name> --agent <name-or-path> [--deltas <d1,d2>] [--no-control]
```

| Verb | Purpose |
|---|---|
| `experiment promote` | Mint a variant artifact from the sweep-selected cell, with its birth certificate. |
| `experiment confirm` | Expand a perturbation policy around a promoted agent into hashed conditions. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:swift-experiment-promotion END -->

`promote` mints a variant artifact ("agent") from the sweep-selected cell with
a `promotion` birth certificate. `--cell L:ALPHA` is the loud manual override
(malformed values error with the `17:0.4` example); it still **requires
evidence that a sweep ran** for the concept — the recommendation, or the newest
sweep run's `recommendations.json` entry (failure entries count and are stamped
as `selectionOutcome`). An override with no sweep at all refuses. `--reason` is
recommended but optional; omitted, the certificate records "no override reason
recorded". Promotable from any manifest status.

**`promotion.poleProvenance` — the certificate announces a negated direction.**
When the promoted cell injects a **mirrored pole** (§3.7's `vectors
mirror-poles`), the birth certificate says so, because the certificate is the
one artifact an agent carries everywhere and an arm that injects the negation
of another concept's direction must not be silently indistinguishable from one
that injects the concept:

```json
"poleProvenance": {
  "polesSwappedFromSource": true,
  "sourceConcept": "<the parent concept>",
  "sourceStimulusSetHash": "<the parent's order-sensitive hash>",
  "sourceTensorHash": "<the parent artifact's tensor hash>"
}
```

Every field is present **only when provable**, from one of two places in
order: the matched artifact's own sidecar (`polesSwappedFromSource` +
`negatedFrom` — the injected bytes are a minted mirror), or the manifest's
hash-checked artifact pin, whose own facts inherit directly while the source
concept and tensor hash are read from the pinned sidecar *only after* its
bytes re-verify against the pin's `sha256SidecarHash`. A missing or drifted
sidecar downgrades to the pin's facts, loudly, rather than inheriting an
unverifiable claim or refusing a promotion the pin supports. Deliberately
**not** inherited: a fresh extraction over role-swapped stimulus files. It is
tensor-identical to a minted mirror, but the swap is recorded nowhere the pins
can prove — `PROVENANCE.md` is prose and names are not evidence — so the
honest certificate stays silent. Mint the mirror, or attach the minted
artifact, to make the claim provable. The block sits **outside**
`promotionKey`: the key is the cross-engine identity of the promotion request,
and this is additional evidence about an already-identified promotion. Absent
on every non-mirrored promotion and on every certificate minted before the
record existed, never encoded when nil, so legacy certificate bytes are
untouched.

`confirm` declares a perturbation policy around a promoted agent's anchor cell,
expanding mechanically into ordinary hashed conditions on the **draft**
manifest. Default `--deltas` is `0.2`; the value must be comma-separated
numbers, and a partially unparseable list errors rather than silently dropping
entries. `--no-control` omits the control condition.

### 3.7 Diagnostics and panels

<!-- GENERATED:swift-diagnostics BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-cli docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-cli data check <experiment>
steerlab-cli vectors compare <a.safetensors> <b.safetensors> [--threshold <ratio>]
steerlab-cli vectors backfill-norms <runDir/name> [--corpus <path>] [--model <id>] [--redenominate]
steerlab-cli vectors mirror-poles <runDir/name> --concept <name> [--output-name <value>]
```

| Verb | Purpose |
|---|---|
| `data check` | Report which study-data inputs the manifest still needs. |
| `vectors compare` | Compare two vector artifacts and refuse below the cosine threshold. |
| `vectors backfill-norms` | Measure per-layer residual norms for an existing artifact into a new artifact, stamped with the current denominator convention (the opt-in migration for legacy unstamped artifacts). |
| `vectors mirror-poles` | Mint the opposite pole of a contrastive direction as a new artifact — every layer negated bit-exactly, under a required new concept name, with a negatedFrom stamp. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:swift-diagnostics END -->

#### Panels and seat casting

<!-- GENERATED:swift-panel BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-cli docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-cli panel list
steerlab-cli panel check <path-or-name>
steerlab-cli panel compile <path-or-name> --experiment <name> [--file-slug <slug>] [--max-tokens <n>] [--model <id>] [--seat <seat>=<agent-artifact-path>] [--temperature <t>]
```

| Verb | Purpose |
|---|---|
| `panel list` | List this workspace's panel scenarios. |
| `panel check` | Validate one panel scenario and report its advisories. |
| `panel compile` | Cast a semantic panel's seats and pin the compiled scenario into a draft study. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:swift-panel END -->

`artifacts` is the one family left off the agent path — no envelope, no strict
flag parsing, hand-maintained here:

```
steerlab-cli artifacts audit [--json]
```

`vectors compare` still accepts the historical `--json <path>` file form for
one release, with a deprecation warning; `--out <path>` is the replacement, and
a bare `--json` now means machine output.

**`data check`** — manifest-driven study-data readiness. Prints one line per
requirement, ordered blockers-first (`invalid`, `missing`, `partial`,
`present`, `optional`), then a summary. **Exits 65** when there is at least
one blocker (the audit's one scheduled human-mode migration, 2 → 65, landed
on both engines together); the exit code is the machine-readable half.

Since 2026-08-15 it also covers the **J-lens readout** (`jlensReadout`), from
the manifest alone — this engine never resolves or loads a lens, which stays
server-only by rule. Every state maps to something the researcher would
otherwise discover at the end of authoring, or after the run:

| state | meaning |
|---|---|
| `optional` | no readout declared — a run records nothing |
| `missing` | a pin absent (`lensID`, `lensSHA256`, `layers`, `configHash`, `tokenizerHash`), or no `modelRevision`/`dtype` — qualification is keyed by both |
| `invalid` | neither a watchlist nor a top-k width: the run completes looking armed and records nothing |
| `partial` | pinned, but `recordTokenIDs` is off — runs fine, and the run will not be exactly replayable afterwards |
| `present` | pinned and retaining ids. Freeze still needs a PASSING qualification, which only the server can produce |

The pin list is duplicated across the engines (they cannot import each other);
a test on each side asserts it, so a drifting copy cannot let the Mac call a
study ready for a freeze the server refuses.

**`panel check`** — hard validation first (an invalid panel is unrunnable),
then advisories. Advisories do *not* block a run — they are the things that
make prompts quietly wrong, so fix them before measuring. Accepts a path, a
panel name, or a file name from `panel list`. The panel's own advisories are
free prose from `MultiAgentRunner.advisories`, so in `--json` they ride in
`result.panelAdvisories` rather than in the envelope's closed advisory
vocabulary.

**`panel compile`** — casts a **semantic** panel's seats and pins the compiled
scenario into a **draft** study, in one step (open-issues §18). It calls the
same `SeatCasting.compile` the app's Seats section and the design
instantiation table call, so a casting made here and one made in the app are
the same artifact:

```
steerlab-cli panel compile <path-or-name> --experiment <name> \
  [--seat <seat>=<agent-artifact-path>]… \
  [--model <id>] [--temperature <t>] [--max-tokens <n>] [--file-slug <slug>]
```

- **Seats are keyed by the scenario's agent `id`**, not its display name — a
  rename would otherwise silently re-cast the panel. Seats you do not name
  stay **baseline**, and an all-baseline casting is the control composition,
  not an absence. An id the panel does not have refuses **64** and prints the
  panel's actual seats.
- **`--seat`'s value is an agent artifact path** (`runs/model-variants/…/
  model-variant.json`). The occupant's hash is read from that file, exactly as
  the app's seat picker reads it. A path that is not there, or a file that does
  not decode as an agent artifact, refuses **65** under `missingPrerequisite`.
- **`--model` / `--temperature` / `--max-tokens` default from the target
  manifest** and, when given, are written to it before the compile. The
  manifest stays the one place those three are decided; a compiled scenario
  that disagreed with it would be a second answer to "what does this study
  run".
- **`--file-slug`** names the compiled file under `prompts/panels/compiled/`
  (default: the study's name). Existing files are never overwritten — a
  numeric suffix is appended — which is what makes a Latin-square batch of
  castings into sibling drafts safe to script.
- Writes **both** scenario pins (`multiAgentScenarioPath` + `…Hash`) and the
  semantic provenance pair (`multiAgentSemanticScenarioPath` + `…Hash`), and
  declares the study `studyKind: multiAgent` if it was not already — a panel
  scenario is read only by the multi-agent run path, so pinning one into a
  `modelOutput` study would look armed and measure nothing. The change is
  reported on the human line and as `result.studyKindDeclared`.
- **Frozen and complete studies refuse** (`statusImmutable`, exit 65 in
  `--json`) **before anything is compiled**, so a refusal never leaves an
  orphan casting behind. A **bound** (hand-authored, self-casting) panel
  refuses **64**: its casting lives in the file, other studies may pin it, and
  re-casting it here would rewrite an input under them.
- `result` carries `compiledPath`, `scenarioHash` (full, not elided), the
  resolved `seats[]` with each occupant's artifact path and hash, and the
  manifest fields as they now stand.

**`artifacts audit`** — reports legacy/ambiguous vector sidecars without
mutating any run directory. `--json` additionally writes
`runs/artifact-audit.json`.

**`vectors compare`** — the cross-engine parity harness; emits JSON
key-identical to `steerlab-server vectors compare`. `--threshold` defaults to
`VectorParity.defaultThreshold` = **0.98**; a non-numeric value errors.
`--json OUT` writes the report to a file *and* still prints it to stdout.

**Three outcomes, three exit codes (2026-08-18), identical on both engines.** A
CI script's whole job at this verb is telling a real parity failure from a
broken invocation, and until this landed it could not: could-not-compare was
exit 1 / `failed`-70 here and `refused`-65 in the server's envelope — the same
answers a crash and a genuine divergence give.

| Outcome | Human | `--json` | `error.code` |
|---|---:|---:|---|
| **pass** — min cosine ≥ threshold | 0 | 0 | — (`state: ready`) |
| **compared and diverged** — it ran and failed the threshold (the CI gate) | 1 | 65 | `parityThreshold` (also `error.gate`) |
| **could not compare** — an artifact or sidecar is missing/unreadable, or the two are not comparable at all | 2 | 66 | `notFound` |

Usage errors (a missing operand, a non-numeric `--threshold`) stay **64**.

Could-not-compare's `repairAction` names **both** operand paths and the shape
of an artifact (`<runDir>/<name>.safetensors` **plus** its
`<runDir>/<name>.json` sidecar, written by `experiment extract`), and its
`result.operandPaths` carries them machine-readably.

**Hidden-size mismatch is could-not-compare, not a failed comparison** — the
artifacts exist and are simply from different models, so the repair is "compare
two artifacts from the same model", not "re-extract". **Layer-count mismatch
stays a report**: those artifacts *are* comparable, the intersection is
compared, and `layerCountMismatch: true` says so.

**`vectors backfill-norms`** — measures per-layer residual norms for an
existing artifact (legacy / SAE import / reader-derived) on the pinned neutral
corpus and writes a **new** artifact into a fresh run directory; the source is
never modified. The reference is a base path with **no extension**
(`<runDir>/<name>`); a missing sidecar errors and says so. Path resolution:
absolute stays absolute, `runs/…` resolves against the workspace root, anything
else resolves under `runs/`. `--corpus` defaults to
`prompts/neutral/corpus.jsonl`. `--model` is a convenience for loading only —
the hard sidecar-vs-loaded-model guard still applies, because norms are a
per-model measurement. `--redenominate` additionally rewrites the norm units.

The new artifact is stamped `residualNormConvention: "perTextMean-v1"` — the
residual-norm DENOMINATOR CONVENTION for the rule this verb actually applies:
each corpus text contributes one number per layer, the mean norm over its own
reading window, and those are averaged with equal weight per text. (The sibling
string `wholeCorpusMean-v1` names the per-position rule, which reaches only the
neutral token bank; the two agree at single-position readings and part company
at pooled ones over variable-length texts.) This verb is the **opt-in
migration** onto
that convention: artifacts with no stamp are LEGACY and are never rewritten,
recomputed, or warned about, so an α that meant one dose yesterday means the
same dose today. Run this when you want a specific artifact's denominator
brought onto the stamped convention.

**`vectors mirror-poles`** — mints the **opposite pole** of a contrastive
direction as an artifact of its own. A CAA direction points from its negative
file's pole toward its positive file's pole (`mean(pos) − mean(neg)`), so a
researcher who wants to inject the other pole has otherwise only a negative α
— which every dose surface reads as "less of the concept" rather than "the
opposite concept", and which no artifact records. This verb writes the negation
down: every layer multiplied by −1 into a **new** artifact in a fresh run
directory; the source is never modified. Same reference resolution as
`backfill-norms`. No model is loaded and nothing is measured.

The negation is a **bit-exact IEEE sign-bit flip** on the stored floats, never
a decode/re-encode: `-0.0` round-trips as `-0.0`, and mirroring a mirror
returns the parent's `.safetensors` bytes byte-for-byte. Only the `layer_<i>`
tensors flip — a stored `neutral_mean_layer_<i>` is the residual stream's own
mean, an absolute activation statistic, and negating it would corrupt ablation
mean-centring.

**Only a PAIRED, source-concept-bearing contrast can be mirrored** — the CAA
family, `meanDifference` and `pairedDifferencePCA` (the recipe the refusal and
the manifest spell `lat`, per 0.9.3's rename: the label changed, the written
bytes never do). Those are the methods whose two poles ARE
two authored stimulus files, so swapping their roles is exactly what the
negation means and the swapped files are evidence a researcher can author. Every
other direction negates *generically*, with no method-specific evidence
semantics: a grand-mean direction's negation is "the population mean minus the
concept", a `designatedReference` direction's is "the reference corpus minus the
concept" — a different comparison, not the concept's opposite pole, and unpaired
besides — and `optvec` / `gemmaScopeSAE` / `repeReaderLAT` have no source
concept at all, so there are no stimulus files to swap. In every one of those
cases the `validation.jsonl` this verb would tell you to author is a file
`attach` pins as **explicitly absent**, treating one that appears later as
drift; the verb was promising a workflow attach forbids. An excluded method is
refused `unmirrorableMethod`/65, naming the method and pointing at the sign flip
that *is* available: a negative α in a study condition.

**`--concept <newName>` is required**, and it may not be the source's own name.
A mirrored pole under the same name would leave two artifacts with one concept
name pointing in opposite directions, and every selector, pin, and promotion
matcher addresses a direction by concept. `--output-name` names the file inside
the run directory (default: the mirrored concept name).

What the new sidecar says. Everything sign-invariant is preserved — including
`normsPerLayer` and the whole `residualNorm*` denominator family, because
‖−v‖ = ‖v‖, so a mirrored artifact's α in norm units means exactly the dose the
source's did — plus the model pins, reading position, rendering,
`coversModelDepth`, and the extraction-method and reader/SAE/OptVec provenance
blocks, which describe how the SOURCE direction was produced. Added:

```json
"negatedFrom": {
  "path": "<runDir>/<name>", "sha256TensorHash": "…",
  "sha256SidecarHash": "…", "concept": "<source concept>",
  "date": "2026-08-27T21:00:00Z"
},
"polesSwappedFromSource": true
```

`stimulusSetHash` is **preserved and qualified**: the mirrored concept's
stimuli are the same two files with the positive/negative roles swapped, so a
fresh hash would claim different bytes were read and the source's hash carried
silently would claim the same recipe — the hash travels, and
`polesSwappedFromSource` says what changed about its meaning. Exactly one field
is dropped: `recipeIdentityHash`, an identity claim about *these* bytes whose
canonical form includes the concept name and which promotion matches on.

**Attaching the minted mirror.** The stimulus hash is `sha256(positive ‖
negative)` and therefore order-sensitive, so the mirrored concept's own
directory — holding the parent's two files role-swapped — hashes to something
else entirely. `attach` reads the sidecar's `polesSwappedFromSource` and checks
the claim the sidecar actually makes: it hashes **this** concept's files in the
**source's** order and compares that to the inherited hash. Nothing is
loosened — a mirrored concept whose directory holds different bytes, or the
right bytes in the same order as the parent, still refuses. The manifest then
pins the mirrored concept's OWN hash as the live `stimulusSetHash` (what every
later `verify` recomputes) with the linkage beside it in the artifact pin:

```json
"vectorArtifact": {
  "…": "…",
  "polesSwappedFromSource": true,
  "sourceStimulusSetHash": "<the parent concept's hash>"
}
```

`verify` re-proves **both**: the concept's own hash against
`prompts/concepts/<newName>/`, and the swap itself, by re-deriving the
source-order hash from those same files. Validation pins the **mirrored**
concept's own `validation.jsonl` under the ordinary source-concept rules — the
inverted rows the success message tells you to author.

**Downstream, the inherited hash is the identity.** Every run's materialized
copy of the pinned bytes stamps what the artifact stamps — the parent's
`stimulusSetHash`, still qualified by `polesSwappedFromSource` and
`negatedFrom`, which travel through materialization — and the recipe identity
a mirrored pin demands (`recipeIdentityHash`, what `promote` matches
artifacts on) is therefore the pin's `sourceStimulusSetHash`, not the
concept's own directory hash. The own hash remains the live pin `verify`
recomputes; the inherited hash is what the bytes claim. Both engines derive
the identity the same way, so a mirrored concept promotes exactly like any
other pinned concept.

The proof that the swapped files are the right evidence rather than a
bookkeeping convention: CAA's mean difference is antisymmetric under a file
swap, so extracting freshly from the mirrored concept's directory reproduces the
minted bytes exactly. The mirror and the "just re-extract from the swapped
files" workaround are the same vector.

Refusals: a missing source is `notFound`/66; `--concept` missing or equal to
the source's is `usage`/64 *with the reason*; a destination that already holds
that artifact is `artifactExists`/65 (mirroring never replaces); and mirroring
a mirror back to its own parent concept is `doubleMirror`/65, naming the
original.

**Validation is not minted.** A mirrored concept has no `validation.jsonl`, and
this verb writes nothing into `prompts/concepts/` — an engine that invented
held-out evidence would be manufacturing the thing the validate gate exists to
demand. Because mirroring is now restricted to the methods whose attach ACCEPTS
a validation file, the message below only ever appears where authoring one is a
workflow that works. The success message names the file to author:

> to validate the mirrored pole, author
> `prompts/concepts/<newName>/validation.jsonl` — the source concept's rows
> with every `expresses` label inverted are the natural starting point

### 3.8 Remote (cluster client)

<!-- GENERATED:swift-remote BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-cli docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-cli remote capabilities [--site <id>] [--token <token>] [--url <server>]
steerlab-cli remote package <experiment> [--site <id>] [--token <token>] [--url <server>]
steerlab-cli remote upload <bundle> [--site <id>] [--token <token>] [--url <server>]
steerlab-cli remote submit-bundle <server-bundle-path> [--bundle <server-path>] [--dry-run] [--executor <local|slurm>] [--gres <spec>] [--parallel <n>] [--site <id>] [--source <run-dir>] [--token <token>] [--url <server>] [--verb <verb>] [--walltime <hh:mm:ss>]
steerlab-cli remote jobs [--site <id>] [--token <token>] [--url <server>]
steerlab-cli remote logs <job-id> [--site <id>] [--token <token>] [--url <server>]
steerlab-cli remote cancel <job-id> [--site <id>] [--token <token>] [--url <server>]
steerlab-cli remote fetch <artifact-path> [--out <dir>] [--path <server-path>] [--site <id>] [--token <token>] [--url <server>]
steerlab-cli remote import <server-evidence-path> [--out <dir>] [--path <server-path>] [--sha256 <hex>] [--site <id>] [--token <token>] [--url <server>]
steerlab-cli remote import-chain <pipeline-run-id-or-experiment> [--site <id>] [--token <token>] [--url <server>]
steerlab-cli remote variants [--site <id>] [--token <token>] [--url <server>]
steerlab-cli remote chat [--hash <sha256>] [--max-tokens <n>] --prompt <text> [--prompt-mode <mode>] [--site <id>] [--strip] [--system <text>] [--temperature <t>] [--token <token>] [--url <server>] --variant <server-path>
```

| Verb | Purpose |
|---|---|
| `remote capabilities` | Report the paired server's capability snapshot. |
| `remote package` | Build a hash-pinned run bundle locally and print its path. |
| `remote upload` | Upload a run bundle to the server. |
| `remote submit-bundle` | Submit an uploaded bundle as a server job. |
| `remote jobs` | List the server's jobs. |
| `remote logs` | Stream one job's log. |
| `remote cancel` | Request cancellation of one job. |
| `remote fetch` | Download one server artifact without importing it. |
| `remote import` | Download, hash-verify, and import an evidence bundle into runs/. |
| `remote import-chain` | Import a whole pipeline chain, skipping directories already present. |
| `remote variants` | List the server's variant artifacts. |
| `remote chat` | Generate one completion through a server-resident variant. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:swift-remote END -->

Two ways to name the server, **mutually exclusive** (passing both is a usage
error, exit 64 — silently preferring one would let a typo look like a working
connection to the wrong server):

- **`--site <id>` — the managed path.** Resolves the endpoint from the shared
  site registry and reads the bearer token from the Keychain internally, so
  neither the ephemeral local port nor the token ever appears in argv, shell
  history, or an agent's context. Stderr gets one presence-and-provenance line
  (`remote: Example Cluster (example-cluster) at http://127.0.0.1:8718, token
  from keychain`) and never the value. Requires the site to be connected — see the
  refusals below.
- **`--url` / `--token` — the compatibility path**, for unmanaged servers.
  `--url` defaults to **`http://127.0.0.1:8080`** (the port both `serve`
  verbs bind; it was 8000 until 2026-08-18); `--token` falls back to
  `STEERLAB_AUTH_TOKEN`. Still supported, not the recommended path: an argv
  token is readable by every other process on a shared node.

`--site` accepts a site **id** or a unique display name. Its refusals (exit
**13**, retryable) each name the exact repair command:

| Refusal | When |
|---|---|
| `no endpoint has been registered for it yet` | the site has never reached `connected` |
| `the forward on 127.0.0.1:<p> is stale (…)` | the endpoint is remembered but nothing is listening — checked *before* the request, rather than timing out against a dead port |
| `127.0.0.1:<p> is held by something else` | local-port collision |
| `no bearer token for it is in the Keychain` | connected, but never authenticated |

All four repair with
`steerlab-cli cluster ensure --site <id> --target connected`. An unknown site
id fails with the stable code `unknownSite`.

| Verb | Notes |
|---|---|
| `capabilities` | Server capability snapshot. |
| `package` | Builds a hash-pinned run bundle locally; prints its path. |
| `upload` | Uploads a bundle. |
| `submit-bundle` | **`--verb` defaults to `run`**; `--executor` defaults to `local`. Empty resource values are dropped. `--parallel <n>` fans a Slurm run out across N GPU jobs — see below. |
| `jobs` | Job list as JSON. |
| `logs` | Streams the job log. |
| `cancel` | Prints `cancel requested`. |
| `fetch` | `--out` defaults to `.`. |
| `import` | Default `--out .steerlab-downloads`. Verifies the hash, imports into local `runs/`, and runs the model-revision adoption reconciliation. |
| `import-chain` | Whole-chain import: the pipeline ledger dir plus every `stageResults` run dir, each packaged on the server (`POST /api/bundles/evidence`), downloaded, hash-verified, and imported **skip-if-present** — see below. Exit 1 only if a directory actually FAILED. |
| `variants` | Server-side variant list. |
| `chat` | `--max-tokens` 512, `--prompt-mode` `chatAssistant`; `--variant` and `--prompt` are both required. |

**`submit-bundle --parallel <n>` — the multi-GPU fan-out, headless** (2026-08-28;
the client and server have shared it since 2026-07-22, and until now only the
app's stepper could reach it). The value is encoded onto the submission body
by one rule, unchanged from the app's: sent **only when `n > 1`, the executor
is `slurm`, and the verb shards** (`run`, or a `pipeline` whose declared chain
starts with `run` — the server rules on the chain and the client encodes for
both). The mechanics of what the server then does, and who merges the
partials, are §5.3.

The envelope echoes what actually went on the wire, so a suppressed request
never looks honored:

| Key | Value |
|---|---|
| `parallelJobsRequested` | the number you typed |
| `parallelJobsEncoded` | the number sent, or `null` |
| `parallelJobsSuppressedBecause` | `null` when encoded; otherwise the **first** clause of the rule that would have to change |

The three suppression sentences are the guard's own clauses, in its order:
`one job requested — sharding starts at 2`; `executor '<x>' — only Slurm
submissions shard across GPU jobs`; and `the '<verb>' verb does not shard —
only 'run' (and a run-first pipeline) has an independent per-record record
set`. A suppressed request `> 1` also prints `warning: --parallel <n> not sent
— <reason>` on stderr, because a headless caller who types `--parallel 8` at a
`local` executor otherwise gets a perfectly ordinary single-job submission
back with nothing said. A non-positive or non-integer value exits `64`
(`--parallel must be a positive integer, not '<raw>'`).

**Two operational disciplines the flag's own help carries, because both have
cost real GPU time.** First, **verify the shard jobs landed** — a fan-out can
**partially fail while the submit still exits 0**: the abort is reported
through the *parent* job record, not the submitting process's exit status, so
read `steerlab-cli remote jobs` (or the scheduler queue at the site) after
submitting rather than trusting the exit code. Second, **stagger submissions
at a site that caps queued jobs**: K shards are K independent sbatch
submissions by design (§5.3), so a fan-out at a site with a per-user submit
limit can have its later shards refused by the scheduler while the earlier
ones run. Site profiles carry `maxParallelGPUJobs` for exactly this, and the
field's own help names how to find the real limit
(`sacctmgr show qos format=Name,MaxTRESPerUser`); with none declared the app's
stepper caps at **16**.

`remote import` is the path that runs revision adoption; a raw `fetch` does
not. That reconciliation exists because a server-auto-pinned revision otherwise
made the local `analyze` refuse on an epoch difference the researcher never
authored.

**`import-chain` semantics** (2026-08-12):

- **Resolution.** A pipeline **run id** imports that chain as-is (a parked
  chain's completed stages are importable). An **experiment name** resolves to
  the NEWEST **completed** pipeline run via the server's pipeline listing +
  `pipeline.json` disposition — the ledger rule: never pick by name or
  timestamp among siblings, completed disposition only. When no sibling is
  completed, the verb refuses and names every candidate with its state.
- **Skip-if-present.** A run directory already in the local workspace reports
  `already present` and is never overwritten or re-downloaded — so re-running
  the verb after a partial import simply fills the gaps (the raw importer's
  refuse-and-abort collision behavior does not apply here).
- **Embedded stages.** A pipeline evidence bundle may EMBED its stage dirs
  (importing the pipeline bundle materializes run/analyze too); the verb
  imports the pipeline bundle first and re-checks presence before each stage,
  so embedded stages cost no second download.
- **Failure records.** The server's structured skip for a ledger-only failure
  record (`POST /api/bundles/evidence`, 2026-08-11) surfaces as a per-row
  *note* (`skipped — failure record: …`), never an error.
- **Adoption.** `EvidenceRevisionAdoption` runs for every imported directory,
  exactly like `remote import`.
- Output is a per-directory summary (imported / already present /
  skipped-failure-record / FAILED) plus a totals line, derived from run ids
  and outcomes only — never the endpoint, token, or server paths.

### 3.9 Cluster lifecycle (`cluster`)

```
steerlab-cli cluster <verb> [--site <id>] [--json]
```

Brings a saved cluster site from disconnected to deployment-ready through
stable, agent-drivable verbs. The same
`ClusterLifecycleCoordinator` the SwiftUI setup wizard drives backs every verb
here — one lifecycle, two clients — so the CLI cannot drift from the app.

**The human boundary is absolute.** The CLI can generate the SSH ControlMaster
command, open a visible Terminal containing it, and poll `ssh -O check`. It can
**never** accept a password, a second-factor choice, or a passcode as a flag,
stdin payload, JSON field, or environment variable; never scrape or read that
Terminal; never answer a multi-factor prompt on the researcher's behalf. There
is no credential-shaped field anywhere in the verb
surface — the bearer token appears only as `tokenAvailable` (a Bool) and
`tokenSource` (a provenance label). This is a release-blocking property, tested
structurally by walking the JSON's keys.

#### 3.9.1 Verbs

<!-- GENERATED:swift-cluster BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-cli docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-cli cluster sites list [--help] [--json]
steerlab-cli cluster sites show [--help] [--json] --site <id>
steerlab-cli cluster sites export [--help] [--json] [--out <file>] --site <id>
steerlab-cli cluster sites import <profile.json> [--force] [--help] [--json]
steerlab-cli cluster preview [--help] [--job-class <class>] [--json] --site <id>
steerlab-cli cluster status [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--refresh] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster diagnose [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--redact] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster auth command [--help] [--json] --site <id>
steerlab-cli cluster auth open [--help] [--json] --site <id>
steerlab-cli cluster auth status [--help] [--json] --site <id>
steerlab-cli cluster auth close [--help] [--json] --site <id>
steerlab-cli cluster push [--bootstrap-partition <partition>] [--dry-run] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster bootstrap plan [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster bootstrap apply [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--plan-hash <sha256>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster bootstrap status [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster validate [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster controller start [--allow-controller-start] [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] [--render-only] --site <id> [--squeue <command>]
steerlab-cli cluster controller status [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--job-id <id>] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster controller logs [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--follow] [--help] [--job-id <id>] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster controller stop [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--job-id <id>] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster controller adopt [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--job-id <id>] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster tunnel open [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster tunnel status [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster tunnel close [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster connect [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster disconnect [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>]
steerlab-cli cluster import [--dry-run] [--help] [--json] [--since <date>] --site <id>
steerlab-cli cluster plan [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>] [--target <rung>]
steerlab-cli cluster ensure [--allow-bootstrap] [--allow-controller-start] [--allow-open-auth-terminal] [--allow-push] [--bootstrap-partition <partition>] [--env-file <path>] [--env-prefix <path>] [--help] [--json] [--materialize-env] [--no-materialize-env] [--open-auth-terminal] [--payload <path>] [--python-version <version>] [--remote-repo <path>] --site <id> [--squeue <command>] [--target <rung>]
```

| Verb | Purpose |
|---|---|
| `cluster sites list` | List every saved site, with token presence only. |
| `cluster sites show` | Print one site's registry record without probing it. |
| `cluster sites export` | Write the site's profile — never a credential — to a file. |
| `cluster sites import` | Upsert a site profile by its canonical remote identity. |
| `cluster preview` | Render the environment and scheduler commands this site will run. |
| `cluster status` | Report each lifecycle layer's state, read-only. |
| `cluster diagnose` | Report status plus the auth command, log path, and last operations. |
| `cluster auth command` | Print the SSH ControlMaster command for a human to run. |
| `cluster auth open` | Open a visible Terminal holding the authentication command. |
| `cluster auth status` | Check whether the shared SSH session is alive. |
| `cluster auth close` | Close SteerLab's own SSH session, and no other. |
| `cluster push` | Deploy the allowlisted server payload and re-stamp its build id. |
| `cluster bootstrap plan` | Render the bootstrap dry run and print its plan hash. |
| `cluster bootstrap apply` | Run exactly the reviewed bootstrap plan. |
| `cluster bootstrap status` | Report whether the remote environment is valid. |
| `cluster validate` | Run the remote engine's own profile validation. |
| `cluster controller start` | Submit the controller job, then return. |
| `cluster controller status` | Report the controller job's scheduler state and log path. |
| `cluster controller logs` | Read the tail of the controller job's log. |
| `cluster controller stop` | Cancel the recorded controller job. |
| `cluster controller adopt` | Record a hand-started controller after verifying it. |
| `cluster tunnel open` | Install or adopt the local forward to the controller. |
| `cluster tunnel status` | Report the local forward as absent, up, stale, or conflicted. |
| `cluster tunnel close` | Cancel SteerLab's exact forward, and no other. |
| `cluster connect` | Reach the connected rung without any mutation authority. |
| `cluster disconnect` | Close the forward and clear the registered endpoint. |
| `cluster import` | Bring this site's run directories into the workspace under the import policy, verify them by content, and rebuild the catalog. |
| `cluster plan` | Print what reaching the target would do, and execute nothing. |
| `cluster ensure` | Advance the site to the target rung, idempotently. |

Every verb above also accepts `--json` and `--help`. `--site <id>` is required wherever it is listed.
<!-- GENERATED:swift-cluster END -->

| Verb | Notes |
|---|---|
| `sites list` | Every saved site: id, display name, transport, topology/scheduler, last endpoint, token **presence**. No `--site`. |
| `sites show` | One site's registry record. Does not probe. |
| `sites export` | Writes the **profile only** — no credentials, no `lastEndpoint`, no local forward port. The same sanitizer guards the registry's own files, so an export and a `Sites/` file can never disagree about what is shareable. |
| `sites import` | Copies a profile into the canonical registry (`~/SteerLab/Sites/cluster-sites/<site-id>.json`), which the app reads and writes too. Refuses to replace a site the registry already holds unless `--force`; with `--force` it dedupes by canonical remote identity, so a re-import refreshes the site instead of forking the registry. Runs the ssh-login validation at import time (§3.9.8), and is the SAME entry point the app's "Import Site JSON…" and the setup wizard use. No `--site`. **This is how a real site arrives** (WP5 §4.2, 2026-08-17): the shipped presets are neutral templates only — `Generic Slurm cluster (conda)`, `Generic Slurm cluster (module Python)`, `GPU workstation` — and no institution's hostname, partitions, QOS limits, or storage layout is in the tree. Keep your site's profile JSON in `Sites/` (a private git repository, which is how it reaches your other machines) — never in a workspace you share. `prompts/fixtures/cluster-site-profile/example-slurm-site.json` is a complete fictional profile to copy; `preview --site <id>` shows what any of them will actually run. |
| `preview` | **WP5 §3.3** — read the complete generated environment and scheduler commands BEFORE anything runs: the rendered env file verbatim, the `#SBATCH` block per job class (`study`, `controller`, `setup`, `gpu-session`; `--job-class` narrows that pane only), the scheduler binaries, the GPU vocabulary + VRAM table, and every fact the profile did **not** state. Offline and read-only — it renders the saved profile and runs no command. States which **DefaultSet** applied (`legacyV1` for a schema-1 profile, whose unstated fields take today's `bootstrap.sh`/`executors.py` constants; `neutralV2` otherwise) and the profile's `schemaVersion`. Rejects the configuration overrides below: an override is not profile data. The bearer token appears only as the `$(cat …)` path indirection the env file itself carries. |
| `status` | Read-only per-layer report. `--refresh` additionally runs the read-only remote `profile validate` (otherwise reported `notRun`). |
| `diagnose` | `status` plus the auth command, the controller log path, and the last 5 durable operation records. `--redact` strips usernames and home paths for a shareable report. |
| `auth command` | Prints the ControlMaster one-liner as an **argv array** for a human to run. Does not probe, does not open anything. |
| `auth open` | Idempotent — see §3.9.4. |
| `auth status` | `ssh -O check`. Exit 0 alive, **10** absent/unresponsive. |
| `auth close` | `ssh -O exit` scoped to SteerLab's own ControlPath; cannot touch an unrelated session. |
| `push` | The compact allowlisted server payload. `--dry-run` prints the exact argv and runs nothing. A successful push **re-stamps the deployed `BUILD_COMMIT`** (the rsync template's `--delete` would otherwise remove it, silently erasing the engine's build identity — found live 2026-08-11); a stamp failure is a loud WARNING in the outcome, never silent. |
| `bootstrap plan` | Renders the dry run and prints `planHash`. The review is **durable** — persisted so a later process (or `ensure`) honours it. |
| `bootstrap apply` | Runs only against the exact reviewed plan; a stale or missing hash refuses **before any command runs** (§3.9.4). On a Slurm site it **submits, records the job, and then polls it** with short separate commands — never one held-open session. A job that is still queued comes back `pending`: rerun the same command to resume following it (nothing is resubmitted). |
| `bootstrap status` | Environment absent/valid/invalid/unknown. |
| `validate` | Remote `steerlab-server profile validate`. |
| `controller start` | Reconciles first, then **submits and returns**. Never waits on the queue. Refuses to submit a second job when one is running, queued, or unreadable. `--render-only` is the render half alone: it refreshes `<metadataRoot>/controller-job.sbatch` from the deployed template and stops before `sbatch` — see §3.9.7. |
| `controller status` | Scheduler state + `serverd.host` + the log path. |
| `controller logs` | `tail -n 200` (`-f` with `--follow`) of the controller job's log. |
| `controller stop` | `scancel` the recorded job. |
| `controller adopt` | **Beyond the v1 lifecycle contract.** Records a hand-started controller after verifying it — see §3.9.5. |
| `tunnel open` | Installs or **adopts** the local forward. Refuses when the controller has published no trustworthy node record. |
| `tunnel status` | absent / up / stale / conflicted. |
| `tunnel close` | Cancels SteerLab's exact forward only. |
| `connect` | `ensure --target connected` with **no** mutation authority: opens/adopts the tunnel, imports the token into the Keychain, proves endpoint identity, and updates the one saved registration (in this machine's runtime cache — never in the shared registry; see §3.9.6). |
| `disconnect` | Closes the forward and clears the registered endpoint. **Leaves the Keychain token alone** — it is the user's credential for a server that is still there. |
| `import` | Brings run directories home under `WorkspaceImportPolicy` and is **runs-only by decision** (open-issues §8 residual (a), 2026-08-20): it never writes `experiments/`. It verifies by content (byte drift refuses; nothing is ever overwritten), reports purge ELIGIBILITY without deleting anything, and — since 2026-08-20 — compares every imported run's `experiment.json` snapshot against the live workspace manifest and prints an **AUTHORING DIVERGENCE** section (envelope `degraded`, exit 13, `importSummary.authoringDivergences`) for any study whose live copy holds fewer concepts/conditions than its own run evidence: cluster-side authoring that never came home. Adopting the snapshot into `experiments/` stays a deliberate hand edit on a DRAFT, followed by `experiment verify`. |
| `plan` | Pure observation + planning. Executes nothing. |
| `ensure` | The principal idempotent command. |

Configuration overrides, accepted by every verb that issues a remote command
(they layer onto the site's own defaults; unset keeps the default):
`--payload`, `--remote-repo`, `--env-prefix`, `--python-version`,
`--bootstrap-partition`, `--squeue`, `--env-file`, and the one toggle
`--materialize-env` / `--no-materialize-env`.

**Environment materialization (WP5 step 6's mechanism, ON by default since
step 7).** The bootstrap step pushes the env file `cluster preview` shows to
`<metadataRoot>/rendered-cluster.env` and invokes `bootstrap.sh --env-file-from
<that path> --env-file-sha256 <digest>`, so the cluster sources the environment
RENDERED FROM THE SITE PROFILE rather than one `bootstrap.sh` synthesizes from
its own built-in constants. The digest is part of the composed argv and
therefore part of `planHash`: approving a plan approves the environment, and
`bootstrap.sh` refuses a file that does not hash to it. The bytes do not change
for an existing site — a schema-1 profile renders the `legacyV1` default set,
which reproduces `bootstrap.sh`'s constants exactly — what changes is where
they come from, and that the site's own declared facts (archive root, purge
window, GPU vocabulary, egress, modules) now travel with them.

**Pinned dependencies (WP6 R1, 2026-08-18).** `bootstrap.sh` installs the
committed platform lock — `Server/requirements-linux-x86_64.lock` on a CUDA
node, `requirements-macos-arm64.lock` on Apple silicon — *before* the editable
`pip install -e "$repo/Server[all]"`, so a node's torch/transformers are the
same ones every other node resolved rather than whatever PyPI published that
morning (`pyproject.toml` declares floors only). Two carve-outs, both
deliberate: `torch`, `triton`, and `nvidia-*` are filtered out of the lock
install, because which CUDA/ROCm build a node wants is a SITE fact that arrives
as `--torch-index`; and `--no-lock` resolves from the floors instead, printing
a WARNING that this node is unpinned. A platform with no committed lock warns
the same way. Regenerate the locks with `Server/scripts/update-locks.sh`
(needs `uv`, from the `Server[dev]` extra). Independently of all of it, every
run stamps what it ACTUALLY imported — see `config.json`'s `pythonEnvironment`
below — and `experiment run` logs a non-blocking advisory when the installed
torch/transformers disagree with the lock.

**`--scratch-gres <token>`** (2026-08-19) adds
`export STEERLAB_SLURM_SCRATCH_GRES="<token>"` to the env file `bootstrap.sh`
synthesizes for a hand run — the flag equivalent of the profile's
`constraints.storage.nodeScratchGres`. Absent by default: no line, no change.
The profile-driven path (`--env-file-from`) carries the key already and does
not need the flag.

**`--no-materialize-env`** opts out: `bootstrap.sh` then writes its built-in
fallback env file, whose values are the script's defaults rather than this
site's declared facts, and the plan transcript opens with a WARNING saying so.
`--materialize-env` is still accepted and now names the default. Pass the same
choice to `bootstrap plan` **and** `bootstrap apply` — a plan reviewed one way
will not apply the other, which is the gate working.

**`watch` does not exist in this cut.** Polling `ensure`/`status`
and honouring `retryAfterSeconds` is the v1 contract; a caller must always be
able to recover by repeating `ensure` after process exit, sleep, or network loss.

#### 3.9.2 Targets and permissions

`--target` names a rung of the ladder; each implies every rung below it. Default
**`connected`**. Both spellings parse (`code-deployed` and `codeDeployed`):

```
authenticated → code-deployed → bootstrapped → validated
              → controller-running → connected
```

`ensure` executes only the transitions it has been **explicitly** authorized to
perform. There is deliberately no single `--yes`:

| Flag | Authorizes |
|---|---|
| `--allow-push` | deploying the server payload |
| `--allow-bootstrap` | the real bootstrap (the dry run needs no permission) |
| `--allow-controller-start` | submitting a controller job |
| `--open-auth-terminal` | opening the visible authentication Terminal |

`--open-auth-terminal` is spelled **without** `--allow-`;
`--allow-open-auth-terminal` is accepted as an alias. Note it is *not* implied
by the three mutation flags — opening a window is its own authorization.

#### 3.9.3 Machine protocol and exit codes

In `--json` mode stdout carries **exactly one** JSON document (schemaVersion 1),
human diagnostics go to stderr, and there are no ANSI sequences. Commands are
rendered as **argv arrays**, never shell strings. Every refusal carries a stable
`error.code`, a plain-language `reason`, and a concrete `repairAction`.

Always present: `schemaVersion`, `verb`, `state`, `changed`, `observedAt`,
`message`. Absent optionals are **omitted**, not emitted as null. Common
optionals: `operationID`, `siteID`, `siteName`, `target`, `step`,
`retryAfterSeconds`, `nextAction`, `endpoint`, `tokenAvailable`, `tokenSource`,
`serverBuild`, `schedulerJobID`, `schedulerState`, `layers`, `plan`, `blockers`,
`sites`, `command`, `planHash`, `logPath`, `outputPath`, `operations`,
`preview`, `error`.

`preview` (from `cluster preview` only) is the WP5 §3.3 document:
`siteName`, `schemaVersion` (**the profile's** schema stamp, not the
envelope's), `defaultSet` (`legacyV1` | `neutralV2`), `defaultSetSummary` (the
same fact as a sentence), `envFile` (the complete rendered file, verbatim,
newlines included), `headers` (`[{jobClass, lines[]}]`, one entry per job class
in the order `study, controller, setup, gpuSession`), `schedulerCommands`
(`[{role, command}]` for `submit`/`query`/`accounting`/`cancel`),
`gpuVocabulary` (`{entries: [{type, vramGB?}], typesValue, vramValue,
declared}` — `declared: false` means the values were inherited rather than
stated), `envFileSHA256` (SHA-256 of `envFile`'s exact bytes — the same digest
materialization puts in the bootstrap argv, so a reviewer can compare it
against `sha256sum <metadataRoot>/rendered-cluster.env` on the cluster; it is
NOT the envelope's `planHash`), and `unresolvedFacts` (`[{key, detail}]`). The bytes are the
renderer's own, pinned to the committed cross-engine goldens under
`prompts/fixtures/cluster-site-profile/`, and are deterministic and
timestamp-free — re-previewing an unchanged profile is a no-op diff.

`layers` is the point of the design: the lifecycle is reported as eleven
independent observations (`siteConfiguration`, `controlMaster`, `payload`,
`bootstrap`, `profileValidation`, `controller`, `daemonHost`, `tunnel`,
`serverHTTP`, `registration`, `bearerToken`) rather than one `connected`
Boolean, because several unrelated things are all colloquially called "the
server connection".

The JSON `state` is authoritative; the exit code is a convenience for shell
callers, and both come from the same vocabulary:

| `state` | Exit | Meaning |
|---|---:|---|
| `ready` | 0 | the requested target is reached |
| `planned` | 0 | work remains and nothing is blocking it |
| `running` | 0 | in progress |
| `needsHumanAuthentication` | 10 | a human must complete auth in their own Terminal |
| `needsApproval` | 11 | a mutation needs its `--allow-…` flag |
| `pending` | 12 | valid asynchronous work (a queued job) is in flight |
| `degraded` | 13 | retryable: a layer could not be read, or a forward is stale |
| `blocked` | 64 | invalid usage or invalid site configuration |
| `failed` | 70 | non-retryable operational failure |

`retryAfterSeconds` accompanies 10 (5s), 12 (30s), and 13 (15s).

#### 3.9.4 Traps

- **`ensure` without permission flags returns `needsApproval` (11) by design.**
  That is not a bug and not a misconfiguration: inspection is always allowed,
  every remote side effect requires its own flag, and the envelope's
  `nextAction.missingPermissionFlags` names the exact flag to add. An agent
  should surface that to the user rather than re-running with everything on.
- **A queued controller stays `pending` forever if need be.** It never decays
  into a timeout failure. Poll it; do not "retry" by starting another.
- **A failed scheduler *query* is `degraded`, not `absent`.** An unproven death
  never licenses a resubmit, so `controller start` refuses in that state.
- **`bootstrap apply` needs the hash from *this* plan.** Any site, payload,
  root, resource, or command change invalidates it; the refusal
  (`bootstrapPlanMismatch`) happens before any command runs. Re-run
  `bootstrap plan` and pass the new hash.
- **A submitted bootstrap job is never resubmitted** (2026-08-13). The job id
  is persisted before the first poll, so a rerun after a sleep or a dropped
  connection RESUMES it; the cluster-side helper independently refuses to
  queue a second bootstrap for the same workspace and prints
  `STEERLAB_BOOTSTRAP_ADOPT=<id>` instead. `pending` is a wait, not an error.
  To force a genuinely new job, run
  `Server/scripts/submit-bootstrap-job.sh --force-new …` on the login host.
- **`auth open` is idempotent, in three steps**: a live master returns `ready`
  without opening anything; an attempt opened within the last 120 seconds is
  *reported* rather than duplicated (durably — the check survives process exit);
  otherwise exactly one visible Terminal opens. Terminal-window spam is a bug,
  not a retry strategy.
- **`controller logs --follow` in `--json` mode sends the log lines to stderr**
  so stdout keeps carrying exactly one JSON document. In human mode they go to
  stdout.
- **`status` reports what is true, not what you are allowed to do.** It plans
  with every mutation permitted, so it never answers `needsApproval`.
- **A second process observes rather than duplicates.** A per-site `flock` means
  a concurrent `ensure` refuses with `operationInProgress` and points at
  `cluster status`; it never queues behind the first or starts parallel work.
  Read-only verbs are unaffected.
- **Individual verbs are their own authorization.** `cluster push` needs no
  `--allow-push` — typing the verb *is* the explicit instruction. The
  `--allow-…` flags exist because `ensure` decides for itself what to run.
- **A dev checkout's payload identity is its git state.** With no
  `deployment-manifest.json`, the `payload` layer compares the local
  checkout's `<sha8>[-dirty]` (dirty scoped to the pushed server subtree —
  an edited SwiftUI view does not dirty the python payload) against the
  deployed `BUILD_COMMIT` stamp. One stamped push reads back `current`, so
  `ensure --target connected` does NOT demand a push on every invocation.
  Two caveats, both honest: a tree that has never been pushed by this CLI
  reads `unknown` ("no BUILD_COMMIT stamp to compare — one push stamps
  it"), and two *different* dirty states of the same commit share a stamp —
  the `-dirty` suffix is the warning label, and pushing again is the cure.

#### 3.9.5 `controller adopt`

A late addition, because a site whose controller predates the operation
store is otherwise permanently unreconcilable — inspection has no job id to ask
the scheduler about, so the controller reads `unknown` forever and the planner
(correctly) refuses to start a second one.

Adoption is verified, never taken on trust. It refuses (`failed`, exit 70, code
`controllerAdoptionUnverified`) and records **nothing** when:

- the scheduler says the job has left the queue, or reports it failed;
- the scheduler could not be read at all;
- the job is RUNNING but has published no current `serverd.host`;
- a forward is already up and the endpoint answers with a different identity.

A queued job is adoptable — recording it is exactly what stops a later `ensure`
submitting a second one.

#### 3.9.6 One registry, three kinds of fact

Since 2026-08-21 there is exactly one cluster-site store, and both clients read
and write it:

```text
~/SteerLab/Sites/cluster-sites/<site-id>.json    the PROFILE — shared, git-syncable
<Keychain, service SteerLabCluster>              SECRETS — per machine, never in a file
~/Library/Application Support/SteerLab/site-runtime.json
                                                 RUNTIME — per machine (endpoint, build)
```

The registry is a plain directory: one pretty-printed, stable-key-ordered JSON
file per site, safe to read, hand-edit, and diff. `Sites/` is normally a private
git repository, because git is how a researcher's sites reach their other
machines — but **SteerLab never runs git**. Its writes leave the tree dirty;
committing and pushing are yours. The directory works with no git at all.

`connect` records the endpoint and server build it reached in the **runtime**
file, never in `Sites/`, so a connect/disconnect cycle leaves the registry
byte-identical and your `git status` clean.

There is no `--activate-in-app`, and nothing replaces it: activation stopped
being a separate step when the two stores became one. A site the CLI added is
the app's site. A *running* app re-reads the registry when it becomes active
and when the cluster UI opens, so a `steerlab-cli cluster sites import` in a
terminal shows up by clicking back into the app.

On its first run a new build absorbs the two legacy stores — the app's
`SteerLabClusterServers` preference and the old
`~/Library/Application Support/SteerLab/cluster-sites.json` — into the registry
and reports what it moved on stderr. A file already in the registry always
wins; nothing is overwritten, and collisions are named. Afterwards the legacy
stores are read-only history that nothing writes to. The Keychain is untouched:
the same service and account keys keep working.

#### 3.9.7 `controller start --render-only` — THE re-render command

```
steerlab-cli cluster controller start --site <site> --render-only
```

The rendered controller script (`<metadataRoot>/controller-job.sbatch`) is a
**child of a template**, and a code deploy refreshes the template without
touching the child. The daemon's walltime self-chain — the USR1 trap and
`chain_successor` that queue a successor before the allocation expires — is
shell code in that template, so a node can run new code under an old launching
script and stop dead at walltime. That is exactly what happened to controller
job 47564632 (2026-08-20): 24 h of current code, launched by
`sbatch ~/.steerlab/controller-job.sbatch` over a copy rendered three days
earlier, and no successor.

`--render-only` runs the render half of `controller start` and stops before
`sbatch`: the artifact is brought back into step with the deployed template and
**nothing is submitted**, which is what a site whose controller is currently
running needs. It prints the render stamp it wrote.

Every surface that reports the problem names this exact command — the
`controllerScript` row of `site qualify` (§4.14), the `controllerScript` layer
and `ADVISORY:` line of `cluster status`/`plan`/`ensure`, `cluster push`'s
message when it cannot repair the artifact itself, and serverd's boot warning.
`cluster push` re-renders a stale script automatically (it is a flow that may
write); read-only flows report and name this command instead.

**The running controller does not benefit.** It was launched by the old script
and has no trap; re-rendering fixes the *next* generation. Cycle the controller
once (let it expire, or `controller stop` then `controller start`) and every
generation after that self-chains.

#### 3.9.8 The ssh-login check every import runs

`ClusterSiteRepository` is the one import entry point, so `sites import`, the
app's **Import Site JSON…**, and the setup wizard's import button all apply the
same rules — the app used to decode the JSON itself and skip them, and a
profile whose ssh destination had no `user@` login was accepted in silence and
first failed hours later at Duo (live finding, 2026-08-21).

* A destination that would **drop a known login** is refused outright
  (`sshLoginDropped`): the canonical identity ignores the `user@` half, so a
  login-less profile otherwise lands on the login-carrying entry and replaces
  it.
* A site with **no login anywhere** is legal — `~/.ssh/config` may supply
  `User` — so it is surfaced rather than forbidden. `sites import` accepts it
  and puts `WARNING: …` in the outcome message; the app asks before saving it.
  A bare `environment.transferHost` is fine (it inherits the login); a transfer
  host naming a *different* user is still a refusal.

### 3.10 `serve`

```
steerlab-cli serve [--port N]     # default 8080
```

Loopback-only by design, with an Origin/Host guard against browser-originated
cross-origin and DNS-rebinding requests. There is **no `--host` flag** — reach
a remote box over an SSH tunnel (`ssh -L 8080:localhost:8080 host`). A
non-numeric `--port` silently falls back to 8080 rather than erroring.

### 3.11 `--config` tasks

```
steerlab-cli --config <path.json>
```

The config's `task` field selects the runner; absent, it defaults to
`smoke-test` (configs predate the field). Implemented: `smoke-test` and
`toy-concept`. Anything else exits 64 with `unknown task '<name>'`. Example
configs live in `prompts/configs/`: `smoke-test.json`, `smoke-test-main.json`,
`smoke-test-gemma-only.json`, `toy-french.json`.

Run the smoke test after any change to `SteeringKit`: it asserts hooks fire on
every pass, steered ≠ baseline, and α=0 reproduces baseline exactly.
`toy-french.json` adds the concept ≠ matched-norm-random check.

### 3.12 `authoring prompt` — generation prompts for missing study data

<!-- GENERATED:swift-authoring BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-cli docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-cli authoring prompt <kind> [--concept <name>] [--count <n>] [--decision <text>] [--held-out <n>] [--name <name>] [--negative <text>] [--positive <text>] [--shape <contentPair|singleStimulus>] [--template-id <id>] [--validation-count <n>]
```

| Verb | Purpose |
|---|---|
| `authoring prompt` | Emit the generation prompt for one kind of missing study data, with its audit battery as numbers. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:swift-authoring END -->

A study is blocked by **missing data** far more often than by a missing verb,
and the answer to missing data is a prompt for an LLM. Those prompts were being
re-improvised per study, which meant each one re-learned the same lessons the
hard way — a corpus whose poles are readable from sentence shape, a choice
instrument whose longer option is the target, a probe that names the concept it
is testing — and each re-improvisation lost the audit numbers, which are the
only part an acceptor can check.

So the prompts are **data**. They live in `prompts/authoring-prompts/`, one
Markdown file per kind, plus `_`-prefixed shared partials; the directory is the
registry index, and the kind is the filename's stem. **A workspace's copy wins
over the shipped one** — edit the wording for your study, and the emission's
hashes follow your bytes. The shipped copy travels inside the engine package,
so a `pip install` with no checkout beside it still renders.

| Kind | Produces | Required |
|---|---|---|
| `contrastive-pairs` | `prompts/concepts/<c>/{positive,negative,validation}.jsonl` | `--concept --positive --negative` |
| `choice-prompts` | the sweep's `logprobShift` instrument | `--concept --decision` |
| `validation-set` | `prompts/concepts/<c>/validation.jsonl` alone | `--concept --positive --negative` |
| `reader-pairs` | `prompts/readers/<c>/pairs.jsonl` | `--concept --positive --negative --template-id` |
| `battery` | a format-2 capability battery | — |

Counts and shapes default (`--count`, `--validation-count`, `--held-out`,
`--shape`, `--name`); **nothing that describes the study is ever defaulted**. A
missing `--positive` refuses with exit 64 naming what it is, because a
plausible default there would be a study nobody declared. A flag belonging to a
different kind refuses the same way rather than being ignored.

Counts are checked as numbers: `--count`, `--validation-count` and
`--held-out` each take a whole number of rows above 0 and at most 500, and
`--held-out` must be below `--count` (the held-out rows are the trailing rows of
the same file). Anything else is exit 64 naming the offending value — the count
is substituted into the prompt verbatim, so `--count bananas` used to ask an
author for bananas rows.

Every emission stamps a header carrying TWO hashes, and the `--json` result
repeats both, together with `templateFiles`, `fromWorkspaceCopy`, the resolved
`parameters`, and the whole `prompt`:

* `promptSpecHash` — the SHA-256 of the partials plus the template, in assembly
  order. It identifies the WORDING, before substitution, and recovers which
  prompt text a study is citing. Two emissions of one kind for two different
  concepts share it.
* `promptInstanceHash` — the SHA-256 of the rendered body plus the resolved
  parameter set. It identifies THIS emission, and recovers which run produced a
  given corpus. It moves when any argument moves, including one the wording does
  not interpolate.

`--out <file>` writes the prompt (this verb owns `--out`; the envelope is
stdout). **The verb writes nothing else, ever.** Its `nextAction` says so:
`requiresHuman`, and the action is a *second* reviewer re-running the prompt's
own audit battery against the delivery. The emitter is never the acceptor — see
§4.15 of the workspace contract.

A registry file that is not on disk is a `missingPrerequisite` refusal naming
the path, never a prompt with a hole in it; an unknown `{{placeholder}}` in an
edited template survives verbatim into the output for the same reason.

### 3.13 `docs cli-reference` — the generator behind this document

<!-- GENERATED:swift-docs BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-cli docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-cli docs cli-reference [--check] [--path <file>] [--write]
```

| Verb | Purpose |
|---|---|
| `docs cli-reference` | Regenerate this engine's marked regions in docs/CLI-REFERENCE.md, or check them for drift. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:swift-docs END -->

The marked regions of this document are generated from the two declarative verb
tables (`ExperimentCLIParser.specs` and `ClusterCLIVerb`) — the same tables
`--help` renders from, which is what makes the manual and the parser incapable
of disagreeing. Generation is **not** a build step: the text is committed so the
document reads on GitHub with no toolchain, and a test regenerates in memory and
fails on any difference
(`CLIReferenceGenerationTests.generatedRegionsMatchCommittedDocument`, and
`test_cli_reference_regions_match` for the server's regions). The repair for a
failing gate is always the same: run the verb with `--write` and commit.

Everything outside the markers is hand-written and survives regeneration
untouched — the rationale, the defaults tables, the traps, and the security
clauses are the document's real value and are not derivable from a flag table.
Each engine owns its own region ids (`swift-*` here, `server-*` in §4–§6) and
never rewrites the other's.

### 3.13 `install` — the binary talking about itself

<!-- GENERATED:swift-install BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-cli docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-cli install version
steerlab-cli install stamp [--revision <commit>] [--root <dir>]
steerlab-cli install verify [--gpu] [--root <dir>]
```

| Verb | Purpose |
|---|---|
| `install version` | Report this build's version and where it is installed. |
| `install stamp` | Hash the installed tree into its resource manifest. |
| `install verify` | Check the installed tree against its resource manifest. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:swift-install END -->

`steerlab-cli --version` is a rewrite of `install version`: same report, same
envelope, same `--json` mode.

**What the install actually needs.** The executable statically links the whole
package graph — MLX included — and carries **zero `@rpath` dylibs**, so the only
runtime dependency is the Metal shader library, and only for GPU verbs. MLX
probes a **colocated `mlx.metallib`** before any bundle lookup, which is why the
supported layout is

```
~/.local/bin/steerlab-cli             # shim: exec's the real binary
~/.local/libexec/steerlab/
    steerlab-cli                      # the binary, absolute LC_RPATH stripped
    mlx.metallib                      # colocated: GPU verbs need no DYLD_FRAMEWORK_PATH
    swift-transformers_Hub.bundle/    # tokenizer fallback configs
    swift-crypto_Crypto.bundle/
    resource-manifest.json            # written by `install stamp`
```

and why the shim is a **shim, not a symlink**: MLX resolves the shader library
against the real binary's directory, and a symlink in `bin/` would look there.

`scripts/install-cli.sh` builds (or takes `--from-products <dir>`), installs
that layout, strips the absolute build-machine `LC_RPATH` the linker bakes in,
and calls `install stamp`. `scripts/install-cli.sh --verify` calls
`install verify`, which is the **only** hasher — the shell never re-derives
SHA-256.

The shim's name is `steerlab-cli` — the binary's own. The short spelling
`steerlab` belongs to the cross-platform Python **client** (§1.4), so the
installer **never** writes it by default: `--short-name` is the only way to get
the alias, and even then an existing `~/.local/bin/steerlab` it did not write
itself is left alone. The default report says the alias was skipped, in one
line. (The rule used to be conditional — "write it when nothing else answers to
the name" — which cannot account for a client installed later, where the alias
would then shadow the client's own console script by PATH order.)

| Verb | What it does | Cost |
|---|---|---|
| `install version` | Version, layout, shader-library status, whether the tree is stamped. | Instant — no hashing. |
| `install stamp` | Writes `resource-manifest.json` over the install root. | Hashes the tree (~150 MB). |
| `install verify` | Compares the tree against that manifest; refuses (65) on any `missing`/`unreadable`/`mismatch`. | Hashes the tree. |

Traps:

- **The stamp changes what `--version` reports.** `resource-manifest.json` is
  the name `CodeResources` already resolves, and for a bare command-line
  executable `Bundle.main.resourceURL` *is* its own directory — so a stamped
  tree reports the version and source revision recorded at install time, and an
  unstamped one falls back to a live `git rev-parse` in whatever checkout the
  build was compiled from.
- **`install stamp` refuses a build directory.** A root containing `.o`,
  `.swiftmodule`, or `.xctest` entries is DerivedData, not an install root.
- **R1 bundles no resources.** An installed binary still resolves workspace
  seed data, the server payload, and the cluster payload through the developer
  checkout; on a machine without one those families fail closed with a plain
  sentence. Embedding them is WP2.
- **The Python side is not installed by this script beyond a shim.** `Server/`
  stays an editable checkout at R1, so the manifest stamps
  `serverVersion: "unbundled"` rather than inventing a number.
- **The first verb that USES a stored secret after an install prompts for your
  Mac password, once.** macOS grants keychain access per *binary identity*, and
  the installed binary is a different identity from the build product you have
  been running — so the first `cluster` verb that actually reads a site token
  (`connect`, `ensure`, `controller adopt`, any `remote … --site`) puts up a
  system prompt and waits. This is correct behaviour and must not be
  suppressed: run one such verb **interactively** after installing, before
  pointing an agent at the install, because an unattended caller will simply
  wait at the prompt forever. R1 binaries are ad-hoc signed, so every reinstall
  is a new identity and re-prompts; a real signature with a stable designated
  requirement (WP2) ends that.
- **Read-only listing verbs are promptless by construction.** `sites list`,
  `sites show`, and `sites import` report `tokenAvailable` from an
  ATTRIBUTE-only keychain query (`ClusterTokenStore.presence`,
  `kSecReturnAttributes`, never `kSecReturnData`), which does not consult the
  value's ACL and so cannot raise the dialog. The field's meaning is unchanged
  — "a token is stored for this site" — and no `kSecUseAuthenticationUI` flag
  is involved anywhere; the presence path asks a different question rather than
  suppressing an answer to the same one. Enforced by
  `ClusterCLITests.sitesListAnswersTokenPresenceWithoutReadingTheSecret`
  (presence reads > 0, data reads == 0).
  `scripts/tests/install-cli-test.sh` additionally runs under a scratch `HOME`
  — there is then no login keychain to ask about at all.

---

## 4. Python `steerlab-server` — authoring and analysis

### 4.1 Global form

```
steerlab-server [--root <dir>] <verb> …
```

`--root` is valid on **every** verb, is popped before dispatch, and must be an
existing directory. Because the global pop removes *all* occurrences, the
per-verb `--root` handling inside `experiment` never fires — harmless, since
the global export achieves the same thing, but it means `--root` is always a
process-wide setting rather than a per-verb one.

### 4.2 `serve`

```
steerlab-server serve [--port N] [--host H] [--root DIR] [--dev-open-loopback]
```

- `--port` default 8080 (GPU-session role: `STEERLAB_SESSION_PORT`, else a port
  derived from `SLURM_JOB_ID`).
- `--host` default `STEERLAB_BIND` or `127.0.0.1`.
- `--dev-open-loopback` selects the single-user open tier (`auth_mode=none`).
  Without it — and without an explicit `STEERLAB_AUTH_MODE` — `serve` resolves
  **token mode** and prints the token-file path plus the `Authorization: Bearer`
  line (never the token value). The flag **refuses to start** (exit 64) on a
  non-loopback bind or with `STEERLAB_EXECUTOR=slurm`; so does an explicit
  `STEERLAB_AUTH_MODE=none`. Full rules in §2.3.
- Prints the resolved artifact root and its origin; warns when the root lacks
  `prompts/`/`experiments/`; warns when the root looks like the source
  checkout; warns when no HF token is configured (suppressed under
  `HF_HUB_OFFLINE=1`).
- Run it **from the workspace root** (or pass `--root`): the artifact root is
  `--root`-or-`STEERLAB_ROOT`-or-cwd, so `cd Server` would point everything at
  the wrong tree.

### 4.3 `experiment <verb> <name>`

The ten agent-path verbs — strict flag parsing, the shared envelope under
`--json`, `--help` from the same table:

<!-- GENERATED:server-experiment BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-server docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-server experiment list
steerlab-server experiment verify <name>
steerlab-server experiment extract <name> [--device <device>] [--dtype <dtype>]
steerlab-server experiment validate <name> [--device <device>] [--dtype <dtype>]
steerlab-server experiment sweep <name> [--device <device>] [--dtype <dtype>]
steerlab-server experiment run <name> [--device <device>] [--dtype <dtype>] [--prompts <path>] [--resume <run-dir>] [--shard <k/K>]
steerlab-server experiment evaluate <name> [--allow-unverified-epoch] [--resume-from <run-dir>] [--source <run-dir>]
steerlab-server experiment analyze <name> [--adjudicated-endpoint <file>] [--allow-unverified-epoch] [--source <run-dir>]
steerlab-server experiment promote <name> <concept> [--agent-name <name>] [--cell <layer>:<alpha>] [--expect-artifact <runDir/name>] [--expect-artifact-hash <sha256>] [--expect-cell <layer>:<alpha>] [--expect-epoch <sha256>] [--qualification <path>] [--reason <text>] [--sweep-run <run-dir>]
steerlab-server experiment confirm <name> --agent <name-or-path> [--deltas <d1,d2>] [--no-control]
```

| Verb | Purpose |
|---|---|
| `experiment list` | List this root's experiments with their status. |
| `experiment verify` | Re-check every pinned input against the file bytes on disk. |
| `experiment extract` | Derive the manifest's concept vectors on this engine. |
| `experiment validate` | Score each vector on its held-out probe and report cross-concept similarity. |
| `experiment sweep` | Sweep layer × alpha on the dev split and record a recommendation per concept. |
| `experiment run` | Generate the measured run for every declared condition. |
| `experiment evaluate` | Judge a completed run with the pinned rubric and judges. |
| `experiment analyze` | Compute paired effect sizes from a completed run into a fresh run directory. |
| `experiment promote` | Mint a variant artifact from the sweep-selected cell, with its birth certificate. |
| `experiment confirm` | Expand a perturbation policy around a promoted agent into hashed conditions. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:server-experiment END -->

`promote --sweep-run R` additionally accepts the expectation flags
`--expect-cell L:A`, `--expect-artifact P`, `--expect-epoch H`, and
`--expect-artifact-hash H`, and `--qualification <sae-feature-qualification.json>`
(§6.8).

The rest of the family is **not** on the agent path: no envelope, no strict
parsing, hand-maintained here.

```
steerlab-server experiment attach-artifact <name> <concept> --artifact <runs/<run>/<artifact>>
                                         [--source-concept C] [--eval-run <optvec-eval run>]
steerlab-server experiment pipeline      <name> [--resume <pipeline-dir>] [--dtype D] [--device DEV]
steerlab-server experiment complete-judgment <name> --awaiting-run <run-dir-or-basename>
                                         --judgments <file>
steerlab-server experiment rescore-style <name> [--source <run-dir>] [--allow-unverified-epoch]
steerlab-server experiment judge-worker  <name> --awaiting-run R --model M --out artifact.json
                                         [--revision R] [--dtype D] [--device DEV] [--record rec.json]
steerlab-server experiment preflight-endpoints <name> [--baseline-run DIR] [--out PATH] [--json]
                                         [--band LOW,HIGH] [--min-cell-items N] [--min-items N]
```

Shared defaults: `--dtype` is `auto` (bf16/fp16/fp32 by device); `--device` is
auto-selected (CUDA → MPS → CPU).

**Every run stamps its Python stack (`config.json` → `pythonEnvironment`,
schema 4, WP6 R1).** The canonical per-run stamp gained one nested object:
`{"python": "3.12.13", "implementation": "cpython", "packages": {"torch":
"2.13.0+cu128", "transformers": …}}`, covering the science-relevant packages
only — the compute substrate, model implementations, tokenizer, artifact IO,
linear algebra, statistics, and the optional LoRA/SAE/J-lens backends. A
declared package that is not installed stamps `null`, never a missing key;
torch keeps its local version segment, because which CUDA build ran is exactly
the site fact worth preserving. Transport (fastapi/uvicorn/pydantic) is
excluded: it cannot move a measurement. **Swift writes `null`** for the whole
object — the same engine-conditional shape `dtype` uses.

`experiment run` additionally prints a **non-blocking** `ADVISORY:` line (and
appends it to the run's `advisories.txt`) when the installed torch or
transformers disagrees with the platform lock's pin; a site-built CUDA variant
of the locked version (`2.13.0+cu128` vs `2.13.0`) is not drift and stays
silent. It is never a refusal — a queued cluster job must not die because PyPI
moved, and the stamp is the durable half regardless.

**There is no `create`/`freeze`/`duplicate` here, and no recipe `attach`.**
Authoring is driven from the Swift app / web UI or the server's HTTP authoring
API; the printed verb list says so. The one exception is `attach-artifact`
(below), whose input — a derived vector artifact under `runs/` — lives on this
engine.

`pipeline` and `judge-worker` are implemented but **absent from that printed
list** — see §7.3.

**`attach-artifact` — pinning a concept by artifact instead of by recipe.**
Every other attach pins *stimuli* and lets each run re-derive the vector. Some
legitimate directions have no such recipe: they are derived post-hoc from other
artifacts (family-grand-mean centring, for example, re-references a set of
vectors against their own family mean), and no stimulus set reproduces them.
`attach-artifact` pins the **bytes** instead — the extension-less locator plus
the SHA-256 of *both* files — and stamps the concept's method `pinnedArtifact`:

```
steerlab-server experiment attach-artifact study formality-dr \
    --artifact runs/20260810T045146213-derived/formality-dr \
    --source-concept formality
```

| Flag | Default | Meaning |
|---|---|---|
| `--artifact <path>` | *required* | Workspace-relative, **extension-less** artifact locator: `<path>.safetensors` + `<path>.json` must both exist (the same id `/api/vectors` returns as `workspaceRelativeID`). |
| `--source-concept <c>` | the concept name | The concept whose stimuli and held-out `validation.jsonl` the validate probe reads. Post-hoc directions are usually renamed (`formality` → `formality-dr`) while keeping the base concept's held-out data. **Refused for OptVec artifacts** — they have no source concept. |
| `--eval-run <run>` | sidecar's record, else none | OptVec artifacts only: names the `optvec-eval` run whose `eval.json` (test split) certifies the direction. **Verified at attach (2026-08-10), not trusted by name:** a reference naming no run directory refuses, an `eval.json` whose `artifact.tensorSHA256` differs from the attached artifact refuses (it certifies a *different* direction), and a run directory with no readable `eval.json` attaches stamped `optvecEvalRunVerified: false` — the freeze advisory then downgrades it to "not verifiable evidence". Without any reference the advisory says NO eval evidence is recorded. |

OptVec artifacts (source method `optvec`) are the second no-recipe family and
the first with no source *concept*: attach additionally requires the additive
`optvec` provenance block (a stripped sidecar refuses) and backfilled residual
norms (the refusal names the backfill verb), skips every stimulus-side check
(nothing under `prompts/` corresponds to an optimized direction), pins
`validationHash` explicitly null, and carries the `optvec:<composite>` dataset
hash verbatim. The validate gate does not apply to an all-optvec study —
evidence is the eval run, named at freeze as an advisory.

Everything else is read from the artifact's sidecar — the artifact *is* the
recipe: reading position, source extraction method, source stimulus hash, the
designated reference or grand-mean population when the source recipe had one,
and the residual-norm provenance. Attach refuses anything it cannot honestly
pin: a different model or revision, a foreign substrate, an artifact recording
no `extractionMethod` / `stimulusSetHash` / `residualNormSource`, source
stimuli that have drifted from the hash the artifact records, or (2026-08-10)
an artifact whose files **resolve outside the workspace after symlinks** — the
containment check is real-path on both engines, so a symlinked `runs/` entry
pointing at scratch refuses rather than pinning bytes a workspace copy would
silently lose. Attaching with no
`--source-concept` when the data lives elsewhere names the base concept in the
refusal. The verb prints any `VIOLATION:` lines and **exits 1** when the
manifest does not verify afterwards (or when the attach itself refuses); `64`
is the usage error.

Downstream there is no special case. `extract` **materializes** the pinned
bytes into the run directory as `<concept>.safetensors` + sidecar (re-checking
both hashes first; drift refuses, naming the file and both hashes), stamped
`extractionMethod: "pinnedArtifact"` plus a `pinnedFrom` block naming the
source. `validate`, `sweep`, `run`, and `promote` then see an ordinary vector
artifact. Both hashes join the `verify()` pin surface, so drift after freeze is
a violation like stimulus drift, and freeze moves the pair into
`experiments/<name>/pinned/` (they live in gitignored `runs/` by construction).

The HTTP authoring API takes the same form on the existing attach route:

```
POST /api/authoring/<name>/attach
{"method": "pinnedArtifact", "concepts": ["formality-dr"],
 "vectorArtifact": "runs/20260810T045146213-derived/formality-dr",
 "sourceConcept": "formality"}
```

One concept per call (one artifact is one direction); an unpinnable artifact is
a 400, never a half-written manifest.

**`run` / `pipeline` checkpointing.** Both install a SIGUSR1/SIGTERM checkpoint
handler: on signal the run parks (fsync + `resume-state.json`) and the process
exits **85**, which is how the scheduler record distinguishes "checkpointed,
resumable" from a failure. `--resume <dir>` continues a checkpointed directory;
for `pipeline` it reopens the stage ledger and skips completed stages. A
pipeline gate abort is a recorded determination (`pipeline-abort.json`) and
exits **0**, not an error.

**`evaluate --resume-from <partial-run-id>`** completes a *failed* evaluation by
judging only the cells it never decided, reusing the verdicts already produced.
Every pin of the partial run is verified first: a differing rubric, epoch,
source run, or judge configuration refuses rather than merging two evaluations.

**`complete-judgment` — phase 2 of a deferred evaluate, headless** (added
2026-08-10 for judging clients with no HTTP access to the engine, e.g. the
Cowork judging pilot). It is the CLI twin of
`POST /api/experiment/<name>/evaluate/complete-judgment`: both call the same
intake (`tasks.complete_evaluate_judgment`), which verifies every emission pin
— packets/map/source-generations hashes, rubric and structured-prompt pins,
experiment epoch, the pinned judge panel with per-judge model (and, for
openrouter judges, provider) stamps, full packet×judge coverage, and
winner-consistent verdict payloads — then unblinds through the map and writes
the same `judgments.jsonl` + `judge-report.json` the inline path writes.
CPU-only, no model load; idempotent (re-running prints the existing judgment
run's path). `--awaiting-run` accepts the awaiting evaluate run as a directory
path or bare basename (only the basename is used). `--judgments <file>`
accepts a JSON list of judgment rows
(`[{packetID, judge, winner: A|B|tie, model, …}, …]`), a
`{"judgments": [...]}` object (the HTTP body shape), or JSONL. On success the
written judgment run path is the last stdout line; any refusal prints the
engine's own text as `ERROR: …` on stderr and exits **1** (missing flags exit
**64**). The awaiting run is untouched by a refused completion — it keeps
awaiting.

**`judging-instructions.md` — the agent-facing framing as an engine artifact**
(2026-08-11, the Cowork judging pipeline's second half). Every deferred
evaluate emission now renders a canonical `judging-instructions.md` next to
`judging-packets.jsonl`: self-contained and orchestrator-agnostic (paste it
into Claude, Codex, or any agent family), carrying the pinned rubric verbatim,
the pinned judge panel, the intake's per-record output schema — including the
optional `annotatorModel` field, the model the judging *agent itself* ran on,
recorded per completed judgment so cross-model annotation agreement stays
computable — and the custody rules as binding requirements (never open
`judging-map.json`; one independent agent context per packet, because serial
one-context judging correlates errors and voids the κ independence assumption;
all paths relative to the run directory). Its SHA-256 is stamped into
`judging-manifest.json` (`instructionsFile` + `instructionsSha256`) and
surfaced by `GET …/evaluate/awaiting`. The judging campaign claims the hash of
the instructions it actually read via the wrapper object
(`{"instructionsSha256": …, "judgments": [...]}` — file or HTTP body); intake
verifies the claim against the emission stamp and records the result as the
report's `judgingInstructions` block. A mismatched claim (or a claim against a
legacy emission) completes **loudly** — WARNING at intake, stamped
`verified: false` — never refuses (post-submit drift policy); an absent claim
is stamped unverified with a note (the Mac app client does not read the
instructions file). This replaces the hand-written per-campaign JOB.md
pattern: identical packets now get identical framing, provably.

**`promote` pinning.** `--sweep-run <run>` removes the ambient "newest run"
lookup; the `--expect-*` flags additionally verify the plan is still current.
`--expect-cell` without `--sweep-run` exits 64 ("there is nothing to check the
expectation against otherwise"). Malformed `--cell`/`--expect-cell` exit 64.

**`judge-worker`** is fan-out machinery, not a researcher verb: it loads one
judge model, judges every packet of an awaiting `evaluate` run for every pinned
local judge resolving to that model, and writes one hash-pinned judgment
artifact the controller merges. `--awaiting-run`, `--model` and `--out` are all
required (exit 64). `--record` writes the child record the Slurm reconciler
reads; failures are recorded there before exiting 1.

**Local-judge rule** (both engines): an empty local-judge `model` resolves to
the study model at its pinned revision, logged at sweep/evaluate start; a
same-model judge generates through the already-resident model rather than
loading a second one; a different-model local judge refuses at start wherever a
second resident model is impossible (see §2.4). A judge's *name* is a label,
never a model id.

**Cross-substrate local judges — the economical path for a large batch.** A
local judge naming a model *other* than the study model is what makes the
judge an independent instrument rather than the study model grading its own
output, and on an engine host it costs no per-token API spend. Two conditions,
both refusals rather than surprises: the engine host needs
`STEERLAB_MAX_LOADED_MODELS ≥ 2` (§2.4 — at 1 the different-model judge
refuses *at start*, naming the variable, instead of dying partway through a
panel), and the judge's `revision` and `dtype` must be pinned with
`pin-rubric --judge-pin` (§3.3) or `judgeValidity` refuses at freeze. The
model must also already be installed: a judging run never downloads weights on
your behalf.

**The coding report's agreement block** (`coding-report.json`, per-response
coding rubrics). Each `fieldAgreement` entry is one (judge pair, field);
categorical fields carry `percentAgreement`, `kappa`, and — since 2026-08-28,
both engines — **`confusion`**, where `confusion[a][b]` is how many shared
cells judgeA coded `a` while judgeB coded `b`, computed over the very label
pairs kappa was computed over, so the counts always sum to the entry's `n`. An
analysis layer therefore never re-derives the cell key, the intersection, or
the label normalization (`null` for absent, `true`/`false` for booleans,
strings as themselves) to see *where* two coders part ways — a κ of 0.55 from
one systematically-confused label pair is a rubric-anchor finding, and the
same κ spread evenly is a noisy-field finding. Numeric fields carry `n` and
`meanAbsoluteDifference` and no confusion block, having no kappa to explain.

For a **single-coder** design the block is not empty, it is **absent with a
reason** — `fieldAgreementAbsentReason: "single-coder design: 1 judge coded
this run, so no inter-rater agreement statistics exist"` — because an empty
list would read as "agreement was measured and there was none", which is a
different and false claim.

---

## 5. Python `steerlab-server` — cluster operations

### 5.1 `study submit` — read this before submitting

<!-- GENERATED:server-study BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-server docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-server study submit <experiment> [--dependency <spec>] [--device <device>] [--dry-run] [--dtype <dtype>] [--executor <local|slurm>] [--force] [--gres <spec>] [--job-name <name>] [--mem <size>] [--no-evidence] [--parallel <n>] [--parallel-jobs <n>] [--partition <partition>] [--prompts <path>] [--resume <run-dir>] [--source <run-dir>] [--target <root>] [--verb <verb>] [--walltime <hh:mm:ss>]
```

| Verb | Purpose |
|---|---|
| `study submit` | Submit one experiment verb to an executor as a durable job. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:server-study END -->

`--verb` takes one of `run|validate|extract|sweep|evaluate|analyze|verify|pipeline`;
`--executor` one of `local|slurm`.

> **`--verb` defaults to `run`.** `steerlab-server study submit my-study` with
> no `--verb` submits a full measured run. If you meant a sweep, that is a GPU
> allocation you will have to `scancel`. Always pass `--verb` explicitly.

The vocabulary is `{verify, extract, validate, sweep, run, evaluate, analyze,
pipeline}`; anything else raises `unsupported study verb '<verb>'` and exits 1.
`rescore-style`, `promote` and `confirm` are **not** submittable — they are
direct-CLI verbs only.

`analyze` is model-free: its preflight reports "not applicable" for the
GPU/memory/walltime checks rather than demanding a GPU it will not use.

**Executor.** `--executor` defaults to the profile's executor
(`STEERLAB_EXECUTOR`, default `local`). Submitting to `slurm` from a server
whose profile does not declare it raises `Slurm study submission requires
STEERLAB_EXECUTOR=slurm` — declaring it is what makes the route token-gated, so
this is a security boundary, not a convenience check. `--dry-run` exempts it.

**Preflight** (Slurm only) runs five checks — `gpuRequest`, `memoryFit`,
`walltime`, `quotaHeadroom`, `maintenanceWindow` — each degrading to `warn`
with an honest message when its inputs are unavailable, never crashing the
submit. A `fail` verdict blocks unless `--dry-run` (inspect the report without
submitting) or `--force` (submit anyway; recorded loudly on the job as
`preflightOverridden`, logged as `PREFLIGHT OVERRIDDEN: verdict fail, forced by
caller`).

Representative failures: `memoryFit` fails when weights + KV cache + 20 %
headroom exceed the requested GPU's VRAM, and names a GPU type that would fit;
`walltime` fails when planned records ÷ observed throughput × 1.5 exceeds the
requested walltime (and warns above 80 %); `maintenanceWindow` fails when the
walltime crosses a configured window.

**Which throughput the `walltime` check divides by** (corrected 2026-08-20).
The submission is classified into one *instrument family* — from the
manifest's instruments, its sampling policy, the verb, and (for `evaluate`)
the predicted judging custody — and priced with that family's own observed
records-per-hour where the throughput table holds one:

| Family | What its records are |
|---|---|
| `deterministicLogprob` | one scored answer-token/choice/ordinal readout per prompt, no sampling |
| `sampledStochastic` | sampled text at temperature > 0, or > 1 sample per item |
| `longFormText` | greedy sampled prose |
| `judgedEvaluate` | an evaluate that judges *here* — locally, or inline with a pushed key |
| `parkedJudgment` | an evaluate whose external panel is uncredentialed here: it renders blinded judging packets and parks, generating no judge token at all |

A family with no history yet falls back to the **global** rate across all
families, and the estimate line says so in those words — the global figure
mixes fast scored readouts with slow sampled generation, so a refusal that
used it is naming a number the researcher should treat as a rough bound. A
`parkedJudgment` evaluate is priced as packet rendering plus a fixed
ten-minute job cost, not as generation; that is why a 1664-record keyless
evaluate now estimates minutes rather than the 11.3 h that refused it. Every
estimate names the rate it used and both risks: over-asking wastes queue
priority, under-asking kills the job at the wall. The classification is
recorded in the check's `data` (`instrumentFamily`, `rateSource`), and the
housekeeping fold learns per family as jobs complete — the global entry keeps
its historical meaning.

**GPU type and count.** There is no `--gpu`, `--gpu-type` or `--accelerator`
flag, and none is needed: the GPU type IS `--gres`. `--gres A100` is expanded
by `SlurmResources.normalized_gres()` into `gpu:A100:1`; a fully-spelled
`--gres gpu:A100:2` is accepted verbatim and is the only way to get more than
one GPU in a single job (the CLI does not expose the API's `gpus` key, and
`gpus` alone renders no `#SBATCH --gres` line at all — see §2.6). A bare
`--gres 1` or any type outside `STEERLAB_SLURM_GPU_TYPES` **refuses** — and so
does any typed gres when that vocabulary is **undeclared**, which is the
default: there is no built-in GPU list (WP5 Step 8; the site profile's
`scheduler.gpus` renders the key). The type vocabulary is site data, the
concrete-type-required rule is not. (Corrected 2026-08-18 — this sentence used
to name a built-in default the code had already removed: audit §1.3 D4.)

Where the default comes from is the **site profile**, not the flag: omitting
`--gres` uses `STEERLAB_SLURM_GRES`, and `--partition` likewise defaults to
`STEERLAB_SLURM_PARTITION`. Both are `profile validate` checks (partition
unset is a **fail**, gres unset is a **warn**), so the intended shape is
"set the site's GPU in the profile once, override per submission with
`--gres` when a study needs a bigger card". `--gres` is what the `memoryFit`
preflight sizes against, which is why it names a GPU type that would fit when
it fails.

**`--parallel N` (multi-GPU fan-out).** Default 1 — exactly the historical
single-job path, byte-for-byte. `N > 1` shards a Slurm `run` (or a run-first
`pipeline`) across N sibling sbatch jobs through the same
`_submit_sharded_bundle` machinery the app's batch submissions use: one
parent job record plus N shard children, merged back into one ordinary
immutable run directory when every shard succeeds. `--parallel-jobs` is
accepted as an alias for the API's `parallelJobs` spelling. Cap **64**.

Sharding is execution logistics: it never enters the manifest or its content
hash, so a sharded run and a single-job run of the same frozen study are the
same measurement. See §5.3 for the partition argument and the merge's
completeness refusal.

Three things to know before you type it:

- **The merge needs a running server.** `merge_shard_runs` is reached only
  from `JobManager._reconcile_shard_parents`, driven by the monitor thread
  that `steerlab-server serve` starts. Submit with `--parallel` and no server
  running and the shards will complete, write their partials, and sit there.
  The CLI says so on stderr when it fans out.
- **Same store, or nothing merges.** The CLI opens the very same
  `jobs.sqlite` (`STEERLAB_METADATA_ROOT`, default `<root>/.steerlab`) the
  daemon uses, as a second writer process; cross-process safety is SQLite's
  own file locking plus the store's 30 s busy timeout. Run the CLI with the
  same `STEERLAB_ROOT` / `STEERLAB_METADATA_ROOT` as the server, or the
  parent and shard records land in a store the reconciler never reads.
- **It refuses rather than degrades.** `--parallel 8 --verb sweep` exits 1
  with `parallelJobs=8 refused: the 'sweep' verb does not shard …`; so does
  `--executor local`, and so does `--parallel 200` (`exceeds the fan-out cap
  (64)`). A non-integer, zero/negative, or value-less `--parallel` exits 64
  at parse. The one case that still degrades is a genuine *clamp*: a panel
  study with fewer transcripts than requested shards runs with one job per
  transcript and logs why. This is deliberately unlike
  `POST /api/studies/submit-bundle`, which degrades in every case — the app
  sends `parallelJobs` from a slider that does not know the verb, while a
  researcher typing `--parallel` is standing there and deserves an answer.
- `--parallel` and a resumed submission are mutually exclusive; a
  checkpointed shard is resumed through its own shard job or through Resume
  on the sharded parent.

**`--source` / `--resume` take a run directory, and it is checked here.** A
relative path is resolved **against the target root** (`--target`, defaulting
to `STEERLAB_ROOT`), never against the directory you typed the command in —
the rendered sbatch `cd`s into its own `slurm/` directory before `srun`, so
"relative to the cwd" would mean something no one intended. `study submit`
stats the path before it bakes it in and **refuses a path that is not there**
(`submissionPath`, `state: refused`, with a repair), naming the *resolved
absolute* path: discovering a typo on a compute node costs a queue slot and an
allocation. A malformed `--dependency` refuses the same way
(`submissionDependency`). If a source run does turn out to be
unreadable at read time, that is now its own refusal (`missingPrerequisite`,
with a path correction) and is no longer confused with "this run carries no
experiment-hash stamp" — only the latter mentions `--allow-unverified-epoch`,
because forgiving a missing stamp cannot make an absent directory readable.

#### 5.1.1 Never hand-write an sbatch for a cluster

Submit through `study submit`, or start from the canonical wrapper
(`steerlab-server site node-scratch-wrapper`, rendered beside
`controller-job.sbatch` in the metadata root; `sbatch <wrapper> <your
command>`). Only rendered scripts carry the site's two node-scratch halves —
the `--gres` that *requests* node-local space and the EXIT trap that *returns*
it — and a job that stages to node scratch without that trap is a defect
regardless of what it produces: on a site whose scheduler does not purge node
scratch, it leaks until an operator notices. The two reasons people used to
hand-roll are closed: `--resume <run-dir>` continues a parked run through the
renderer, and `--dependency <spec>` takes a raw Slurm dependency
(`afterok:12345`, `afterany:1,afterok:2`, `singleton`), shape-checked here and
passed to `sbatch` as a command-line argument rather than a script header — an
auto-resubmitted continuation must not re-wait on a dependency that was
already satisfied. If you must write your own, copy the wrapper; never copy
the trap into a script of your own, because "how this site cleans up" has
exactly one definition (`steerlab_server/node_scratch.py`) and a second copy
is what drifts.

Also, from live verification: **Slurm snapshots the batch script at submit
time.** Editing the file after `sbatch` returns does not change what a queued
job will run — the fix has to be `scancel` plus a fresh submission.

### 5.2 `bundle`

```
steerlab-server bundle run      <experiment> [--out path]
steerlab-server bundle evidence <run-dir>    [--out path]
steerlab-server bundle inspect  <bundle.tar.gz>
steerlab-server bundle import   <bundle.tar.gz> [--target root] [--overwrite]
                                [--sha256 <outer digest>]
steerlab-server bundle execute  <bundle.tar.gz>
                                --verb <verify|extract|validate|sweep|run|evaluate|analyze|pipeline>
                                [--target root] [--shard k/K] [--dtype D] [--device DEV]
                                [--prompts P] [--source S] [--no-evidence]
                                [--record rec.json] [--resume-from ID]
steerlab-server bundle create|submit <bundle-dir> [--gres A100] [--walltime HH:MM:SS]
                                [--mem M] [--partition P] [--job-name N] -- <cmd…>
```

**`run`** packages a hash-pinned run bundle from a workspace experiment.
**`evidence`** packages a completed run's evidence bundle. On a ledger-only
pipeline failure record (a refused continuation: seed snapshot + `pipeline.json`,
no `run-status.json`, no reachable stage outputs) it prints a structured skip
(`{"skipped": true, "reason": …}`) instead of erroring — same contract as
`POST /api/bundles/evidence`, so bulk imports note the skip and keep going.
**`inspect`** prints
a bundle's manifest. **`import`** unpacks one into a target root, re-verifying
pins and refusing to overwrite a frozen manifest unless `--overwrite`.
`--sha256 <digest>` is the OUT-OF-BAND outer pin (the job record's
`bundleSha256`, never a value read from inside the archive): supplied, it is
checked **before the archive is opened at all**, and a mismatch names both
hashes and extracts nothing. Omitted, the outer digest is the caller's
responsibility — which is what it always was on this engine, and why the Mac
had the check and the server did not (portability gap G3). The HTTP twin is
`POST /api/bundles/import` with `expectedSha256`.

**`execute`** is the Slurm child entry point: it imports the bundle and runs
exactly one verb through the same task functions the interactive CLI uses, so
batch and interactive runs produce identical artifacts. `--verb` is required
(exit 64); a verb outside the list raises `unsupported bundled experiment verb
'<verb>'`. It refuses a bundle that is not a `runBundle`. Evidence is packaged
on completion unless `--no-evidence`. Checkpoint handling (exit 85) is
installed only for `run` and `pipeline` — the verbs with a checkpoint consumer;
a handler nothing polls would swallow SIGTERM. A resume pointer beside
`--record` makes a requeued job continue a checkpointed run, or return a
completed one idempotently.

Asymmetry worth knowing: `bundle execute --verb evaluate|analyze` accepts
`--source` but **not** `--allow-unverified-epoch`. Through the bundle path the
epoch guard cannot be bypassed.

**`create` / `submit`** render an sbatch bundle around an arbitrary command.
The `--` separator is mandatory (`command separator '--' required`, exit 64)
and the command after it must be non-empty. `create` prepares and prints the
bundle; `submit` additionally calls sbatch and prints the Slurm job id.
Resource flags override the `SlurmResources.from_env` defaults of §2.6.

### 5.3 `--shard k/K` and automatic fan-out

`--shard k/K` (0-based `k`, `K` shards, cap **64**) exists on **`bundle execute`
and on `experiment run`** (the latter since 2026-08-19, open-issues §16). It is
not accepted by `study submit`, and applies to the `run` verb only:

```
--shard applies to the 'run' verb only (got '<verb>') — other verbs have no
independent per-record record set to partition
```

`experiment pipeline --shard k/K` refuses in the same spirit, naming the two
paths that do work (`experiment run <name> --shard k/K` for one shard by hand,
`study submit <name> --verb pipeline --parallel K` for the whole chain).

The partition is safe because every generation record is independent —
`derive_seed(experimentHash, condition, promptID, sampleIndex)` has no
cross-record state and greedy records never touch the RNG — so K contiguous
balanced ranges of the run's deterministic order (condition × prompt ×
sampleIndex) reassemble byte-identically. The first `total % K` shards carry
one extra record. Shard provenance lives in `shard.json` and the merged
`report.json`'s `sharded` block — never in `config.json` (closed schema-4
contract) and never in the manifest. `parallelJobs` is execution logistics: it
never enters the manifest or its content hash.

**Who merges: only the server's job reconciler.** `merge_shard_runs` is called
from exactly one place — `JobManager._merge_shard_parent`, driven by
`_reconcile_shard_parents` when every shard child of a *sharded parent job
record* reaches terminal success. That parent record is created only by
`_submit_sharded_bundle`, reached from `submit_run_bundle` **and, since
2026-08-07, from `submit_study`** (`study submit --parallel N`). The
reconciler keys on the parent's `shardChildren` / `shardMerge` resources, not
on which entry point wrote them, so a CLI fan-out merges exactly like an
app-submitted batch — and that is equally true of a Mac-side
`remote submit-bundle --parallel N`, which reaches `submit_run_bundle`
through the same `parallelJobs` body field the app has always sent.

Consequences, stated plainly:

- Running `bundle execute --shard 0/4 … 3/4` by hand still produces four
  partial run directories that **nothing merges**. There is no
  `merge-shards` verb; the raw flag is for a shard job, not for a human.
- Automatic fan-out has **three** entry points: `study submit --parallel N` (a
  server-resident experiment, packaged into a bundle here),
  `POST /api/studies/submit-bundle` with `parallelJobs` (a client-staged
  bundle), and — since 2026-08-28 — `steerlab-cli remote submit-bundle
  --parallel N` (§3.8), which is the Mac-side spelling of that same body
  field. All three default to 1 and cap at 64.
- **The merge happens in a running `steerlab-server serve` process, whichever
  entry point you use.** The submitting CLI process exits as soon as the
  shard jobs are submitted; `poll_slurm` — and therefore
  `_reconcile_shard_parents` — runs only on the monitor thread `serve`
  starts. `jobs reconcile <dir>` is a different mechanism (folding child JSON
  records) and does **not** drive the merge.
- Fan-out resolution differs by entry point, on purpose. The bundle/API path
  **degrades**: a non-Slurm executor, or a verb other than `run` (or a
  pipeline whose first stage is not `run`), logs a note explaining why
  `parallelJobs` was ignored and runs as one job — and on the Mac client the
  suppression is additionally visible before the wire, in the envelope's
  `parallelJobsSuppressedBecause` and a stderr warning (§3.8), so a degraded
  request never merely *looks* honored. `study submit --parallel`
  **refuses** in those same cases (§5.1). Both share one resolver
  (`_resolve_parallel_jobs`), so the *rules* cannot drift; only the answer to
  a non-shardable request differs. A panel study with fewer transcripts than
  requested shards is a clamp, not a refusal, on both paths — it is reduced
  to the transcript count with an explanatory note (a single-transcript panel
  has nothing to split; turns within a transcript are ordered). A
  non-integer or over-cap value raises on both.
- **A shard sbatch failing mid-fan-out fails the COMMAND (2026-08-09).** The
  abort itself is failure-atomic (submitted siblings are scancelled with
  per-shard confirmed/unconfirmed outcomes; the parent record fails with the
  scheduler's stderr) — but that abort is reported through the parent job
  record, and until 2026-08-09 the CLI printed the returned no-shard
  submission JSON and exited 0. Five `QOSMaxSubmitJobPerUserLimit`-refused
  submissions passed silently that way. `study submit` now reads the parent
  record back after submission: if it is `failed`, the CLI prints the parent
  job id and the parent's error text (including sbatch's own stderr) to
  stderr and exits 1. stdout still carries the submission JSON, whose
  `shardJobIDs: null` shows the truthful no-shards shape.
- A sharded parent is `pending` while its shard children attach. A second
  `JobManager` constructed against the same store during that window (any
  other CLI verb, or a server restart) treats a pending parent as orphaned
  and fails it, cancelling its attached shards. Do not run other
  `steerlab-server` verbs against the store while a fan-out is submitting.

The design is K **independent** sbatch submissions, deliberately not a Slurm
job array, so every per-job mechanism (durable records, checkpoint exit-85
detection, auto-resubmit, manual resume, honest cancel, log streaming) works
per shard unchanged.

The merge refuses on incompleteness: every expected
`(condition, promptID, sampleIndex)` cell must appear exactly once across
shards. Missing or duplicated cells refuse loudly and leave the partials intact
so the merge can re-run once the incomplete shard completes. Runs stay
immutable — the merge writes a new directory and never mutates a partial.

### 5.4 `jobs`, `profile`, `housekeeping`

<!-- GENERATED:server-jobs BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-server docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-server jobs list
```

| Verb | Purpose |
|---|---|
| `jobs list` | List this engine's durable jobs. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:server-jobs END -->

`jobs list` is also the default when no subverb is given. The rest of this
group is not on the agent path:

```
steerlab-server jobs reconcile <records-dir>
steerlab-server profile show|validate [--json]
steerlab-server housekeeping status [--refresh]
steerlab-server housekeeping maintenance set --file <windows.json>
```

`jobs reconcile` folds child-job JSON records from a scratch directory into the
single-writer store — Slurm children write records instead of opening
`jobs.sqlite` directly. Records without an id, or unreadable JSON, are skipped
silently.

`profile show` prints profile / topology / executor / auth / root plus any
warnings — **the only way to see what your environment actually resolved to**,
given that invalid choice values fall back silently. `profile validate` prints
one line per check and **exits 1** when any check fails. `--json` prints the
raw snapshot (or the validation report).

`housekeeping status --refresh` recomputes rather than reading the cached
report. `maintenance set --file` reads a JSON file that is either
`{"windows": [...]}` or a bare list; a missing `--file` exits 64, an unreadable
file or invalid windows exit 1.

### 5.5 `finetune` — LoRA training as a Slurm job

```
steerlab-server finetune execute <job-dir> [--record rec.json]
steerlab-server finetune plan    <finetune-config.json>
steerlab-server finetune train   <finetune-config.json>
steerlab-server finetune submit  <finetune-request.json> [--plan-only]
       [--confirm-plan HASH] [--dry-run] [--force]
       [--gres G] [--walltime HH:MM:SS] [--mem M] [--partition P] [--job-name N]
```

**`submit`** is the evidence-grade submission path on the terminal: it drives
the same two functions the routes drive — `POST /api/finetune/plan` then
`POST /api/finetune/submit` — with no HTTP hop and no logic of its own. Until
2026-08-20 it did not exist, so every evidence-grade training was `curl`'d by
hand with the server token.

*Which spelling the file uses, and why.* `submit` takes the **wire request** —
**camelCase**, exactly the JSON body the two routes accept natively, so the
same file can be POSTed verbatim or handed to this verb and nobody
hand-translates anything:

```json
{
  "schemaVersion": 2,
  "baseModelID": "org/model",
  "revision": "<40-char commit sha>",
  "name": "adapter-name",
  "trainingMode": "instructionChat",
  "evidenceGrade": true,
  "dataset": {
    "bundleID": "family-v1",
    "manifestPath": "adapters/manifest.json",
    "manifestHash": "<sha256>",
    "files": [
      {"role": "train", "path": "adapters/x/train.jsonl", "sha256": "<sha256>",
       "content": null},
      {"role": "validation", "path": "adapters/x/val.jsonl", "sha256": "<sha256>",
       "content": null}
    ]
  },
  "hyperparameters": {"rank": 8, "batchSize": 2, "epochs": 1,
                      "maxSequenceTokens": 512},
  "selectionMetric": "validationLoss",
  "resources": {"gres": "A100", "walltime": "08:00:00", "partition": "gpu",
                "memory": "64G"}
}
```

`content: null` means a server-resident file, resolved through the path
resolver and hash-verified; a string is an inline upload, hash-verified before
a byte is written. Dataset paths are workspace-relative by contract.
`plan`/`train` take the *other* spelling — the resolved **snake_case**
`LoRAConfig`, which is also what a submission writes into its job directory as
`finetune-config.json`. Handing that file to `submit` is refused by name
("resolved LoRAConfig … not a fine-tune REQUEST"), never by a cryptic unknown-key
error: the two spellings tripped a first-time caller, and the refusal now says
which file it wanted.

*The plan gate.* An evidence-grade submission must echo the hash of the plan
it confirms, so the plan a researcher read and the plan that runs are provably
the same one:

```
steerlab-server finetune submit req.json --plan-only          # prints planHash
steerlab-server finetune submit req.json --confirm-plan <hash>
```

`--plan-only` has no side effects at all (inline bytes are staged in a temp
tree that is removed before it answers) and writes the plan document to
stdout, the `planHash` to stderr. A request that carries its own
`expectedPlanHash` needs no flag; `--confirm-plan` contradicting it is a
refusal, not a silent winner. Resource flags override the request's own
`resources` block. `--dry-run` prepares the job directory and the sbatch
bundle and records a `prepared` job without reaching `sbatch`; `--force`
overrides a failing LoRA preflight verdict (memory **and** walltime), recorded
loudly on the job exactly as on `study submit`.

Exit codes for `submit`: **0** = submitted (or prepared under `--dry-run`, or
planned under `--plan-only`); **2** = a refused request — unreadable file,
wrong spelling, unconfirmed or drifted plan, an evidence refusal, or a failing
preflight (the report is printed as the stdout document); **1** = the
submission itself failed; **64** = usage, including any undeclared flag
(argv is strict: a mistyped `--dryrun` refuses rather than submitting the job
it was meant to withhold).

**`execute`** is the Slurm child entry point for `POST /api/finetune/submit` —
the evidence-grade training path, which is by rule a *job*, never a
daemon-resident one.
The submission materializes a self-contained job directory under `runs/` —

```
runs/<stamp>-submit-finetune-<name>/
  dataset/<declared relative paths>   # verified split bytes (inline or resident)
  finetune-config.json                # the resolved LoRAConfig + run_directory
  plan.json                           # the confirmed plan, its hash, the preflight
  slurm/run.sbatch                    # the script auto-resubmit re-submits verbatim
  records/<jobId>.json                # the child record the reconciler folds
  run/                                # the training run: checkpoints/, sidecar, history
```

— and `execute` reads it back, installs the SIGUSR1/SIGTERM checkpoint flag,
and trains into `<job-dir>/run`. Because a requeue re-executes the *identical*
command, it adopts the checkpoint already in that directory instead of minting
a second run; a job directory whose adapter sidecar exists returns success
idempotently (at-most-once finalization).

Exit codes: **0** = adapter written (or already complete); **85** =
checkpointed and resumable — `sacct` reports `FAILED` with exit `85:0`, which
the reconciler maps to the non-terminal `checkpointed` status and
auto-resubmit-on-checkpoint acts on; **1** = failed (the reason is written to
the child record); **64** = usage, or a directory with no readable
`finetune-config.json`. No child record is written on a checkpoint — the exit
code *is* the signal, exactly as for `bundle execute`.

**`plan`** prints the normalized plan for a config file (resolved revision,
dtype, schedule, and every evidence refusal) without loading a tokenizer or a
model — the terminal twin of `POST /api/finetune/plan`. **`train`** runs the
trainer in this process against a config file; it is the exploratory
convenience path, and it has no scheduler, no checkpoint flag, and no
auto-resubmit. Both exit **2** on an unreadable config or an unloadable
dataset.

Trap: `finetune execute` takes the job *directory*, `plan`/`train` take a
resolved config *file*, and `submit` takes a wire *request* file. The
preflight (memory **and** walltime) runs at submission time only — a config
handed straight to `finetune train` is unsized.

---

## 6. Python `steerlab-server` — diagnostics

### 6.1 `vectors`

<!-- GENERATED:server-vectors BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-server docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-server vectors compare <a.safetensors> <b.safetensors> [--threshold <ratio>]
steerlab-server vectors mirror-poles <runDir/name> --concept <name> [--output-name <value>]
```

| Verb | Purpose |
|---|---|
| `vectors compare` | Compare two vector artifacts and refuse below the cosine threshold. |
| `vectors mirror-poles` | Mint the opposite pole of a contrastive direction as a new artifact — every layer negated bit-exactly, under a required new concept name, with a negatedFrom stamp. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:server-vectors END -->

`compare` still accepts the historical `--json <path>` file form for one
release, with a deprecation warning. `backfill-norms` is not on the agent path:

```
steerlab-server vectors backfill-norms <runDir/name> [--corpus <path>] [--output-name N]
    [--redenominate] [--model <id>] [--revision R] [--device D] [--dtype T]
```

**`compare`** — cross-engine parity harness; JSON key-identical to the Swift
twin, and since 2026-08-18 exit-code-identical too (§3.7's table):

| Outcome | Human | `--json` | `error.code` |
|---|---:|---:|---|
| **pass** — min cosine ≥ threshold | 0 | 0 | — (`state: ready`) |
| **compared and diverged** (the CI gate) | 1 | 65 | `parityThreshold` |
| **could not compare** — missing/unreadable artifact, or not comparable at all | 2 | 66 | `notFound` |

Usage errors stay **64**. Human mode is unchanged on this engine — it has
exited 2 for could-not-compare since the verb existed, and the Swift twin
aligned to it rather than the reverse; what changed here is that the envelope
stops reporting that outcome as `refused`/65, indistinguishable from a real
divergence. `--threshold` defaults to `vector_parity.DEFAULT_THRESHOLD` =
**0.98** (the same value as the Swift twin); a non-numeric value exits 64.
`--json OUT` writes the report and still prints it to stdout.

**`backfill-norms`** (2026-08-10) — measures per-layer residual norms for an
existing artifact (legacy / SAE import / reader-derived / optvec-trained) on
the pinned neutral corpus and writes a **new** artifact into a fresh
`backfill-norms-<name>` run directory; the source is never modified. The new
artifact is stamped `residualNormConvention: "perTextMean-v1"` — the rule this
verb applies, one window-mean per text averaged over texts; unstamped
artifacts are legacy and are never migrated in place (Swift twin behaviour).
Parameter-for-parameter the CLI form of `POST /api/vectors/backfill-norms`
(result JSON key-identical), with the Swift twin's reference resolution
(§3.7): a base path with **no extension**; absolute stays absolute, `runs/…`
resolves against the workspace root, anything else resolves under `runs/`.
`--corpus` defaults to `prompts/neutral/corpus.jsonl`. `--model`/`--revision`
are loading conveniences defaulting to the artifact's own sidecar pins — the
hard sidecar-vs-loaded-model guard applies regardless (a norm table is a
per-model measurement), so a wrong `--model` refuses before the load.
`--redenominate` (loud, never silent) rewrites a legacy `extraction-stimuli`
denominator; without it, an artifact that already has norms refuses — also
before the load. `--device`/`--dtype` follow the same resolution as `optvec
train` (`STEERLAB_DEVICE`, then CUDA → MPS → CPU; dtype auto per device).
Exit codes: **0** backfilled; **2** refused; **64** usage. This verb loads
the model in-process — on a serving/delegating deployment use the API route
instead, which runs it as a durable job on the pinned model.

**`mirror-poles`** — mints the opposite pole of a contrastive direction as its
own artifact: every layer × −1 (a bit-exact IEEE sign-bit flip, never a
decode/re-encode) into a new artifact in a fresh `mirror-<name>` run directory,
under a REQUIRED new concept name, with a `negatedFrom` stamp naming the source
bytes and `polesSwappedFromSource: true` qualifying the inherited
`stimulusSetHash`. The source is never modified, no model is loaded, and
nothing is written into `prompts/concepts/`. Behaviour, sidecar shape, refusal
texts, and states are identical to the Swift twin — see §3.7 for the full
account. Unlike `backfill-norms` this verb IS on the agent path: it answers in
the envelope, and its refusals carry `error.code` `notFound` (66), `usage`
(64), `artifactExists` (65), or `doubleMirror` (65). There is no API route,
because the analogous derive operation (reader → steering vector) has none
either: this is local file work with no compute to delegate.

### 6.2 `panel`

```
steerlab-server panel list
steerlab-server panel check <path-or-name>
```

Cross-engine twin of `steerlab-cli panel`'s two READ verbs. `check` accepts a
path, a scenario name, or a bare file name. Exit codes: **66** no such panel;
**65** invalid scenario; **0** valid — advisories, if any, print but do not
block.

There is **no `steerlab-server panel compile`**, and that is the policy, not a
gap: casting a panel writes a workspace input and pins it into a manifest, and
authoring is Mac-authority (WP0-AGENT-SURFACE-AUDIT §10.x). Cast on the Mac,
then submit the frozen study.

### 6.3 `jlens` — server-only, Gemma-only

```
steerlab-server jlens supported
steerlab-server jlens acquire <model-id>          # bytes → HF cache (needs egress)
steerlab-server jlens import  <model-id>          # convert → workspace (offline)
steerlab-server jlens list
steerlab-server jlens inspect <lens-id>
steerlab-server jlens support <lens-id> <runDir>/<vectorName> [--layers 5,17,29] [--k 25] [--json]
steerlab-server jlens token-options <model-id> <text> [--case-variants]
steerlab-server jlens derive <lens-id> <model-id> --token-id N [--piece P] [--name N] [--revision R]
steerlab-server jlens qualify <lens-id> <model-id> [--revision R] [--layers 20,26,32]
                              [--alpha-range 0.04:0.12 | 0.04,0.08,0.12] [--token-id N]
                              [--watchlist id,id] [--prompts P] [--battery P] [--dtype D] [--device D]
steerlab-server jlens g0 <model-id> [--lens L] [--revision R] [--endpoint <rows.jsonl>]
                         [--layers …] [--watchlist id,id] [--alpha-range …] [--top-k K]
                         [--token-id N] [--piece P] [--band-stride N] [--prompts P]
                         [--device D] [--dtype T]
steerlab-server jlens probe <model-id> --prompt-file P | --prompt TEXT
                            [--lens L] [--revision R] [--layers 31,40,48]
                            [--pin ' therefore, however'] [--pin-id N,N]
                            [--directions <runDir>/<name>,…] [--variant <path>]
                            [--prompt-mode M] [--system-prompt S] [--top-k K]
                            [--stride N] [--max-tokens N] [--device D] [--dtype T]
                            [--json]
steerlab-server jlens report <runDir> [--baseline NAME] [--band 20,26] [--bands N] [--json]
```

`acquire` and `import` are deliberately distinct: acquisition needs network
egress, conversion is offline. `support` decomposes a vector into the
vocabulary it is made of and writes a readout run directory; `--k` defaults to
`decompose.DEFAULT_BUDGET`, `--layers` to all fitted layers. The printed
readout shows the energy fraction only next to its matched-norm-random null,
because a random direction scores comparably in this dictionary — the tokens
are the finding, the fraction is context for them.

`derive` **requires `--token-id`** and exits 64 without it: a direction is
indexed by an exact vocabulary token, and resolving a word here would be the
silent mis-selection `token-options` exists to prevent.

`qualify` (Stage 4, 2026-08-15) is what makes a J-lens study freezable
non-force: the freeze gate has always demanded a *passing* qualification for
the study's exact model + revision + **dtype + quantization**, and until this
verb existed nothing could produce one. It loads the model (a GPU job — the
checks are about numerics geometry cannot see) and runs seven named checks —
`geometry`, `runtimeNumerics`, `jacobianFinite`, `referenceAgreement`,
`stableReadouts`, `causalSmoke`, `capabilityGuard` — appending the verdict,
per-check numbers and all, to the lens record. **Exit 3 = did not qualify**,
and the record is written anyway: "we tested this runtime and it did not pass"
is evidence, and losing it would leave the absence looking like an untested
runtime. Records are appended, never replaced. Tier is read from the
supported-lens table, so `qualify` RUNS on 4B (the cheap mechanics rehearsal)
and its record is refused by freeze; no flag upgrades that. An unresolvable
dtype refuses outright — absent is not a match.

`referenceAgreement` records **every comparison**, not only the worst
(2026-08-24, additive): `measured.perComparison` carries one row per
(`layer`, `fixtureRow`, `tokenID`) with OUR logit, the REFERENCE logit, and the
absolute deviation, plus `worstComparison` and a `perComparisonTruncated` flag
for the bounded-record case. `maxAbsLogitDeviation` is unchanged and is still
what the tolerance is compared against — the breakdown exists because a max
alone cannot say whether a deviation is large *relative to its operands*, which
is the whole question when one fires. `STEERLAB_JLENS_REFERENCE_FP32=1` (§2.7)
is the paired instrument.

`g0` is the feasibility gate, and its output is **two independent
arm verdicts**: the STEERING arm (derive → inject → use in a study) and the
READOUT arm (watchlist, top-k, traces). Passing one and failing the other is a
usable outcome; neither licenses the conjunction "the direction steers AND the
readout says why". The report splits `mechanical` from `scientific` in its
schema: a mechanical failure is scale-independent and blocks the 27B run,
while a scientific result from a testing-tier model is recorded as a prior and
verdicts nothing (both arms come back `null`). `--endpoint` takes the study's OWN
`choicePromptsFile` — the same JSONL the `logprobShift` sweep objective reads,
parsed by the same loader (`text` aliases `prompt`; `target` defaults to
`options[0]`), so point it at the case-family choice rows that already exist
rather than authoring a parallel file. Its SHA-256 is stamped in the report.
**Without it the steering arm is skipped, not assumed** — marker density is
not a substitute, it is the surface-prose confound the arm's anti-lexical
control exists to detect. Exit 3 = mechanics failed.

The **readout arm licenses reading**; each *instrument* declares its own
plumbing beside it (restructured 2026-08-15). `onlineTrace` needs `alignment`
+ trace persistence + volume; `offlineSlice` needs `slicePositioning` +
projection cost. Neither inherits the other's licence — an offline slice is
not licensed by an alignment check that never looked at slices, and an
alignment failure does not invalidate a claim that never used the aligner.
`instruments.<name>.usable` = shared science ✓ AND that instrument's plumbing ✓.

Two **measurements** are stamped, never verdicted, because neither has a
threshold anyone could justify:

- `replayFidelity` — how closely teacher-forced replay reproduces the online
  readout. This is the **resolution limit for both instruments**: what
  reproduces may be claimed, what does not may not.
- `readableBand` — per-layer median full-vocabulary **rank** of the true next
  token, which is what a study should set its armed `layers` from instead of
  inheriting the paper's workspace prior. `--band-stride N` (0 disables).
  Deliberately *not* top-1 accuracy: untrained vocabulary entries carry large
  unembedding norms and outrank correct tokens at every layer, so top-1 reads
  ~0 even where the readout is right.

**`recordTokenIDs` (manifest, default off).** Retains the exact sampled token
ids on every generation record (`outputTokenIDs`). It changes nothing that is
measured — only what the run KEEPS — and it is what makes a completed run
exactly replayable: teacher-forced replay of the realized sequence reproduces
the states a stepped decode wrote into the KV cache (measured on
`gemma-3-4b-it`: top-1 lens token identical at 24/24 steps, top-10 set overlap
97–99%), but only if the exact ids are fed back. Re-deriving them from stored
text is **not** a round trip — a generation that stops naturally ends with
`<end_of_turn>`, the streamer skips it, and re-tokenizing silently returns a
shorter sequence. Useful beyond J-lens work: any post-hoc activation analysis
needs the same sequence, and the id count is the only way to tell a generation
that stopped on EOS from one that hit `maxTokens`. Costs ~7 bytes per token per
record; zero for choice-only studies, which generate nothing. **Retention is
not retroactive**, so freeze raises a non-blocking advisory when a study
declares a `jlensReadout` without it.

`probe` reads ONE **condition's** prompt at every (armed layer × position) in a
single forward pass. `--variant <runDir>/<agent>.json` supplies the agent (its
adapter, stored injections, prompt mode and system prompt); absent one it reads
the base model under the declared rendering, stamped as such. The prompt always
goes through the **shared renderer**, because raw tokenization produces a
different sequence than the chat template — different length, different
positions — so a trajectory read off raw text is not about any run's positions — the question the online recorder cannot answer, because it discards
every prompt position but the last by design. **One condition is the
first-class case**: position is a *within-token* axis (a token's unembedding
norm is constant across positions), so tracking one token across a prompt is
clean with no comparison condition at all.

Three readouts, most to least committed:

- `--directions <runDir>/<name>` — cosine against a concept vector you already
  extracted, at each position. No vocabulary, no token list; the most
  defensible answer to "when did X appear", and the paper's own concept score.
- `--pin ' therefore'` / `--pin-id N` — full-vocabulary **rank** of chosen
  tokens across the prompt. Robust because rank holds the token's own norm
  constant. A word is resolved only when its leading-space form is a single
  token, and refused otherwise with the components named — a longer or rarer
  word that the tokenizer splits into pieces refuses rather than silently
  ranking whichever fragment came first (`jlens token-options` lists the
  pieces so you can pin an exact id instead). Not a watchlist averaged into a
  score.
- top-k per cell — the discovery view, in `probe-topk.csv`. Read knowing the
  top of this distribution is tilted toward tokens with large unembedding
  norms, including untrained ones.

Every J-lens number carries its logit-lens companion; for a single-agent
trajectory that is the primary control available (lens and companion moving
together means you are reading the token locally present, not what the model is
poised to say). Captures through hooks on the armed layers, not
`output_hidden_states` — 62 layers × 2000 positions is ~2.7 GB at 27B, four
armed layers is ~170 MB. Refuses above its own projection ceiling; `--stride`
and `--max-tokens` are the knobs. One full-vocabulary projection per (cell,
lens) and one per (cell, companion) — top-k and pinned ranks are read off the
same logits, never recomputed — and a `--directions`-only probe costs none of
them. Directions are pinned by tensor and sidecar hash, and a direction
extracted on a different model is refused. Artifacts stamp the lens tier and its
qualification state, so an unqualified read is stamped exploratory.

`report` rolls a completed run's `jlens-readout.jsonl` into
`jlens-report.json` + `jlens-topk.csv` + `jlens-watchlist.csv`, written into
the run directory: per (condition × layer) top-k occupancy/mean logit/mean
rank, watchlist aggregates through `analysis.py`'s declared contrast, and
baseline-vs-condition deltas — every one of them beside its logit-lens
companion, with the counts behind it, and with mention-masked steps and
incomplete traces excluded *and counted*. Counts and means only: no null, no
CI, no p-value, because every quantity here is a mean over steps and that is
exactly the permutation-invariant family `analysis.permutation_null` refuses
by name.

There is no J-lens verb on the Swift CLI, and there must not be — lens
artifacts are PyTorch/HF-native and activations do not transfer across
substrates.

### 6.4 `optvec` — server-only (optimized injection vectors, by backprop)

```
steerlab-server optvec train    --config <path.json>
steerlab-server optvec eval     --config <path.json>
steerlab-server optvec geometry --out-name <name> <artifact> <artifact>… [--layer L]
steerlab-server optvec geometry --config <path.json>
steerlab-server optvec campaign materialize --config <path.json>
steerlab-server optvec campaign submit <campaign-dir>     # exit 3 on any failed sbatch;
                                    # refuses while another submit cycle holds the
                                    # campaign's flock (concurrent top-ups double-submit);
                                    # job names carry an 8-hex campaign-identity token so
                                    # same-named campaigns never adopt each other's jobs
steerlab-server optvec campaign status <campaign-dir>
steerlab-server optvec interpret --config <path.json>
steerlab-server optvec family    --config <path.json>
steerlab-server optvec jspace    --config <path.json>
steerlab-server optvec gradient  --config <path.json>
steerlab-server optvec gradient mint <survey-run-dir> <item-id> [--name N]
steerlab-server optvec fracture  --config <path.json>
```

Optimizes one residual-stream injection vector by gradient descent through the
frozen model against a hashed choice-row dataset bundle (target/anchor/
capability), and writes an immutable `optvec-train` run directory: baseline
cache, per-step `metrics.jsonl`, checkpoints, and the selected best-val vector
as an ordinary artifact (`extractionMethod: "optvec"`, additive `optvec`
provenance block). Exit codes: 0 = trained; 2 = bad config/dataset (unknown
key, hash drift, multi-token option, overlapping split ids); 64 = usage.

Traps: the config is strict camelCase JSON — unknown keys refuse (a silently
ignored `lambdaAnchor` typo would run S1 while the record claims S2); every
dataset needs its `sha256` pin; training requires single-token options (eval
does not); the saved vector is born **without** `residualNormPerLayer` — run
the norm backfill before any `alphaInNormUnits` condition uses it. The test
split must not appear in the config at all: this verb sees train and val only,
and citable numbers come from a confirm-style study, never from this run.

`eval` is the mirror-image firewall: it reads the TEST split only, and any
config key ending `Train`/`Val` refuses with the rule quoted — the eval verb
must be unable to see the data selection ran on. It scores an `optvec`
artifact (refusing one whose additive `optvec` sidecar block is missing) at
α multiples {0.25, 0.5, 1, 1.5, 2}× of the vector's own α through the deployed
stepped-KV instrument, plus a teacher-forced all-position fluency guard and a
library-cosine table where every cosine carries a seeded random-direction null
percentile. Output is its own `optvec-eval` run dir (`eval.json` +
`eval-records.jsonl`) — the training run directory is never written into.
`eval.json` is screen-grade evidence, not citable numbers.

`geometry` is inference-free (no model load): ≥2 artifacts at one shared layer
→ pairwise cosine matrix and the participation ratio of the stacked matrix's
singular values (formula stated in `geometry.json`; a unit-normalized twin is
reported beside the raw PR, which one long vector can dominate). Non-optvec
comparison vectors are welcome; mixed layers refuse.

`campaign` runs the grid (layers × conditions × seeds, one job per cell —
never a Slurm array, which gpu_qos would refuse wholesale): `materialize`
writes the campaign dir (per-cell train configs validated by construction +
sbatch scripts from the site profile; drifted re-materialize refuses),
`submit` tops the queue up to `maxQueued` (default 15 under the 20/8 gpu_qos
cap) and is the re-invocable top-up — it persists every attempt atomically
BEFORE the next sbatch, adopts orphaned submissions back by job name, treats
a failed scheduler query as "alive" (known job) or "skip this cycle" (no
known job), never as permission to resubmit, and **exits 3 when any sbatch
failed** (this verb never buries a fan-out failure in a zero exit; the older
`study submit` path does — check squeue there). The per-cell attempt budget
(`1 + maxResubmits`, default 3) counts only submissions that ENTERED the
scheduler — sbatch returned a job id, or an orphan was adopted by name. An
sbatch refusal (QOS submit-cap crunch, fan-out failure) is recorded in the
state file and reported loudly, but consumes no budget and stays retryable
forever: it cannot have consumed cluster resources (2026-08-12 fix — before
it, submit-cap refusals burned cells to "exhausted" with no job ever
existing). Completion authority is the
`COMPLETED` marker the cell's own script writes only on train-exit-zero.
`status` is read-only; `unknown` in its vocabulary means the scheduler query
itself failed, not that the job died — and `exhausted` requires the full
budget of REAL attempts (rows carry `attempts` = counted, `attemptBudget`,
and `submitFailures` separately; a cell whose failures were all
submission-time reads `failed`/retryable, never `exhausted`).

`interpret` reads one solution (logit lens ±v, Gemma Scope SAE decomposition
when configured, J-lens static support stored verbatim with its
nullEnergyFraction, steered generations on a pinned probe battery, and a
self-explanation stamped `suggestive: true` — never evidence). A configured
stage whose dependency is unavailable records `{"skipped": reason}` rather
than aborting. `family` summarizes ≥2 interpret runs: rows grouped by
condition (`s0`/`s1`/`s2`/`s3` derived from the sidecar's objective;
shuffled labels outrank the λ rule), library matches gated on the run's own
random-direction null percentile (default p99), with `no-library-match` as a
first-class category — and empty-library rows counted separately so an
unconfigured comparison cannot masquerade as a finding of no match.

`jspace` (exploratory tier, lens required): paired teacher-forced passes
decompose what the model computes FROM the vector — `J(delta) = J(direct) +
J(emergent)` exactly, with the emergent term tracked up an observation-layer
ladder and every energy schema-bound to its matched-norm-random null. Family
mode reports raw vs propagated cosine/PR side by side (shallow-vs-deep
multiplicity). Every artifact stamps the lens tier plus its qualification
state **against the runtime that actually ran** (2026-08-15): an unqualified
runtime is stamped exploratory, not citable evidence, with `jlens qualify` as
the named remedy; a runtime with a passing record is stamped with that
record's id instead.

`gradient` / `fracture` (S4, per-item mode): `gradient` is the α→0
fast path — one forward+backward per item yields the margin-gradient
direction, plus a dose-ladder linearity check scored through the deployed
injection path; ONE run dir holds all items (`gradients.safetensors` +
`gradient-survey.json`), and `mint` exports a chosen item's direction as an
ordinary artifact (`optvec` block `method: "gradientDirection"`, claim
`localSensitivity`) that the backfill→eval→interpret lifecycle accepts
unchanged. `fracture` clusters per-item restart solutions (cosine-threshold
single-linkage, rule stated in the artifact) into distinct local maxima per
(item, dose): counts, basin frequencies, cosine-to-gradient per cluster, and
the fracture-α table. Train-side S4 keys: `datasets.targetVal` is optional
(absent → fixed-steps, final-step checkpoint, `selection: "finalStep"` — a
per-item run has nothing to select on) and `itemFilter` restricts targetTrain
to named ids (recorded in provenance; the parent file's hash pin is
unchanged). Campaign grids gain an `items`/`itemsFile` axis (conditions ×
layers × items × seeds; restarts are the seeds axis). A per-item vector's
effect on its OWN item is constructed, never evidence; a claim needs held-out
replication items that the vector was not trained on.

**Creating an optimized vector, end to end.** The full recipe, headless. All
dataset files are strict choice-row JSONL (single-letter options), pinned by
SHA-256; `data check optvec` below states the bundle's authoring contract in
full.

1. **Author the bundle** in the workspace: `target-train/val/test`,
   `anchor-train/val/test`, `capability-train` + `capability-eval` (disjoint),
   `neutral-fluency`. Then run the bundle's readiness template (§6.5):

   ```bash
   steerlab-server data check optvec   # [--dir prompts/optvec] [--json]
   ```

   It refuses on exactly what train/eval would refuse on (strict parse,
   multi-character options, bundle-wide duplicate ids) plus the authoring
   spec's balance window, and prints per-file SHA-256 for every valid file —
   paste those into the train/eval configs verbatim (no separate
   `shasum -a 256` pass needed).
2. **Train** — config names the model, layer, α denomination, objective λs,
   and the train/val files ONLY (a test file in a train config is simply never
   read; a train/val file in an EVAL config refuses):

   ```json
   {"modelID": "google/gemma-3-27b-it", "layer": 31,
    "alphaNormFactor": 0.10,
    "residualNormArtifact": "runs/<run>/<backfilled-artifact>",
    "lambdaShift": 1.0, "lambdaAnchor": 1.0, "lambdaCap": 1.0,
    "seed": 0,
    "datasets": {
      "targetTrain": {"path": "prompts/optvec/<bundle>/target-train.jsonl", "sha256": "…"},
      "targetVal":   {"path": "prompts/optvec/<bundle>/target-val.jsonl",   "sha256": "…"},
      "anchorTrain": {"path": "prompts/optvec/<bundle>/anchor-train.jsonl", "sha256": "…"},
      "anchorVal":   {"path": "prompts/optvec/<bundle>/anchor-val.jsonl",   "sha256": "…"},
      "capabilityTrain": {"path": "prompts/optvec/<bundle>/capability-train.jsonl", "sha256": "…"}}}
   ```

   `steerlab-server optvec train --config train.json` → an `optvec-train` run
   dir whose artifact is the best-VAL checkpoint (S1: zero the anchor/cap λs;
   S0: add `"shuffleTargetLabels": true`). For a grid, wrap this config as a
   campaign `baseConfig` and use `optvec campaign materialize|submit|status`.
3. **Backfill residual norms** (the vector is born without a denominator; α
   in norm units is refused until this runs). On a node that can load the
   model:

   ```bash
   steerlab-server vectors backfill-norms runs/<train-run>/<name> \
     --corpus prompts/neutral/corpus.jsonl
   ```

   (§6.1 — same parameters as `POST /api/vectors/backfill-norms`, which
   remains the path through a serving/delegating instance:
   `curl -X POST localhost:8080/api/vectors/backfill-norms -H 'Content-Type:
   application/json' -d '{"vectorID": "runs/<train-run>/<name>",
   "neutralCorpusPath": "prompts/neutral/corpus.jsonl"}'.)

   The backfilled copy lands in a NEW run dir; the additive `optvec`
   provenance block survives byte-identically (tested).
4. **Eval on the test split** — `steerlab-server optvec eval --config
   eval.json` with `vectorArtifact` = the BACKFILLED artifact, `targetTest` /
   `anchorTest` / `capabilityEval` / `neutralTexts`, and optionally
   `libraryVectorPaths`. Note the run directory: it is the evidence reference
   for the next step.
5. **Attach into a confirm study** —

   ```bash
   steerlab-server experiment attach-artifact <study> <concept> \
     --artifact runs/<backfill-run>/<name> --eval-run <eval-run-id>
   ```

   Freeze surfaces the eval run as the validate-equivalent advisory. Confirm
   conditions (dose grid, negative-α, matched-norm random, optional S0 cell)
   come from `control_matrix.optvec_confirm_conditions(...)` — camelCase JSON
   appendable to `experiment.json` verbatim.
6. **Interpret / family / jspace** as needed (`--config` each; jspace
   additionally needs an imported lens: `jlens acquire` + `jlens import`).

Known affordance gaps (deliberate, tracked): no API routes for the optvec
verbs themselves. (Closed 2026-08-10: the `data check optvec` bundle
template — §6.5 — and the `vectors backfill-norms` CLI verb — §6.1. Also
2026-08-10: the Mac app's Data → OptVec tool is a READ-ONLY v1 UI — bundle
list with the same data-check verdicts, offline campaign/train progress from
workspace files, eval/interpret/family/jspace/geometry summaries, and the
one action: attach a trained artifact to a draft study via the Swift
`attachArtifact` mirror.)

There is no OptVec verb on the Swift CLI, and there must not be — the
optimization is PyTorch/CUDA-native and MLX has no training path here.

### 6.5 `data check` — study-data readiness (server templates)

<!-- GENERATED:server-data BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-server docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-server data check <optvec|lora> [<path>] [--dir <path>]
```

| Verb | Purpose |
|---|---|
| `data check` | Report which inputs a server data template still needs. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:server-data END -->

The two templates: `data check optvec [--dir prompts/optvec]` and `data check
lora [<package-manifest-or-dir>] [--dir adapters]`.

The server side of the `data check` layer (Swift: `steerlab-cli data check
<experiment>`, §3.7, manifest-driven). Both server templates are
**directory-driven**, not manifest-driven, because both datasets are authored
BEFORE any experiment manifest exists and are pinned by hash into configs
afterward. Same vocabulary and contract as the Swift verb: one line per
requirement, blockers first (`invalid`, `missing`, `partial`, `present`), a
summary line, **exit 65** on any blocker (the 2 → 65 migration landed on both
engines together), 0 when ready, 64 on usage.

**`optvec`** checks the bundle's authoring contract:
all nine bundle files exist; every choice file parses under the STRICT
cross-engine loader (the same `load_choice_rows` rules the sweep's
logprobShift instrument and `optvec train` refuse on); every option is a
single character (the tokenizer-free approximation of training's single-token
refusal); per-file first-option target balance within 45–55%; item ids unique
across the WHOLE bundle (the engine keys baselines by id); **`bundle.json`**
exists, carries the three research directives, pins all nine files, and every
pinned hash agrees with the file's actual bytes (a stale table of contents is
a blocker — it is the drift the file exists to prevent); **`REPORT.md`**
presence is reported but NEVER blocks (it is the human half of acceptance —
leakage QC and item quality — and a mechanical check cannot stand in for it);
`neutral-fluency`
is one `{"text": …}` object per line (spec-strict — the eval loader would
accept a plain line, but a bundle that needs that permissiveness was not
authored to spec). Every VALID file's line carries its SHA-256, ready to
paste into train/eval configs; an invalid or missing file gets **no** hash —
the verb never hands out a paste-able pin for bytes the engine would refuse.
Item counts (90/30/30, …) are authoring targets, reported but never blockers;
cross-split content leakage is undetectable mechanically and stays the
authoring job's QC responsibility.

**`lora`** checks a workspace LoRA **dataset package** — the
`adapters/<package>-manifest.json` written by whatever script packaged it —
before a GPU allocation is spent on it. The positional argument is either that
manifest or a directory holding exactly one (`--dir` is the same thing by
flag; both default to `adapters` under the workspace root). Purely local file
checks: no model, no tokenizer, no network. In the order a training run would
hit them — the manifest parses and declares a known `trainingMode`
(`document` / `instruction_chat`: the row schema and the loss mask both follow
from it); it declares at least one training arm (`outputs`/`families`/…
containers are walked generically); every arm's **training** and
**validation** file is pinned, present, hash-true, and parses under the strict
row schema (`lora_data.parse_rows` — the same refusals the trainer raises,
naming `path:line`); an arm with **no validation block is a blocker**, because
evidence-grade training refuses the legacy fractional split; each arm's split holds under
`load_split_rows` (no duplicate rows within a split, no train↔validation
overlap by row hash or content); and the package's own QC report says `pass`.
Every VALID file's line carries its SHA-256 — paste those into the fine-tune
request's `dataset.files[].sha256` verbatim; an invalid or missing file gets
**no** hash, so the verb never hands out a paste-able pin for bytes the engine
would refuse.

`--json` prints the full report as JSON instead of lines (same exit code).

### 6.6 `sae candidates` — the SAE feature roster as pinned data

```
steerlab-server sae candidates check <path> [--json]
steerlab-server sae candidates pin   <experiment> <path>
```

Server-only (SAE import is Gemma/PyTorch work; no Swift twin writes the key).
Both verbs are OFFLINE — nothing here touches a feature browser or HF.

A **candidate roster** is a workspace JSON file recording which SAE features a
study may seat: per entry a construct label (data, never code), a `role` from
the closed vocabulary (`focal`, `affectControl`, `embodiedControl`,
`domainControl`, `discriminantControl`, `unrelatedTopicControl`,
`positiveControl`), model / dictionary source / layer / feature id, the
Neuronpedia URL as discovery provenance, a **discovery snapshot** (explanation
text, top logits, example activations, access date — REQUIRED for `focal`
roles, optional for controls), a `verification` block, an optional
`qualificationArtifact` path, and a lifecycle `status` (`candidate` /
`qualified` / `rejected` / `seated`). Shape and rationale:
`prompts/templates/sae-candidates/README.md`.

`check` validates the file (closed keys, enum values, ISO dates, unique
`(model, source, layer, featureId)` tuples) and prints the roster summary —
counts by role, status, and verification, plus how many entries carry a
discovery snapshot or a qualification pointer. **Exit 2** on any schema
violation, 0 when valid, 64 on usage, so it works as a script gate.

`pin` validates the file loads, then stamps `saeCandidates: {path, hash}`
(SHA-256 of the file BYTES) into a **draft** manifest. From then on the roster
is on the `verify()` pin surface: drift, a move, or a deletion is a
verification violation exactly like markers drift; `freeze` refuses on it
(force included — `verify()` of the pins is never skippable); and the file is
git-gated at freeze, snapshotted into `experiments/<name>/pinned/`, and packed
into evidence bundles. The path must be workspace-relative. Frozen manifests
are immutable — duplicate and re-pin. The manifest key is optional and
additive: a study with no SAE arm never gains it, and manifests frozen before
this existed verify and freeze byte-identically.

A pinned roster also becomes the study's **preregistration of which features it
may seat**: with one pinned, a condition seating an SAE feature the roster does
not nominate is a verify violation — see §6.12.
### 6.7 `sae family-report` — the cross-family descriptive report (server-only)

```
steerlab-server sae family-report --config <path.json> [--out-name NAME]
```

ONE engine artifact laying CAA / grand-mean, LoRA, OptVec and
Gemma Scope SAE rows side by side — the cross-family comparison a study
reports once its arms exist. Offline — it reads artifacts off disk and
loads no model. Writes its own immutable run directory
`runs/<stamp>-family-report-<name>/` with `family-report.json` (full
provenance), `family-cosines.csv` (long form), `family-report.txt` (readable),
and the canonical `config.json`. Exit 0 / 2 (refusal) / 64 (usage).

**Descriptive by construction.** It reports no interval estimates, no test
statistics, and no statistical comparison between families; behavioural numbers
are COPIED verbatim from an `analyze` run's `effect-sizes.csv` (path + SHA-256
stamped), never recomputed, and the inferential columns are left behind — their
names and the reason are stamped in `omittedColumns` / `omissionReason`. Read
these rows as overall behavioural impressions, never as a test.

Config (closed keys):

```json
{
  "name": "wave1",
  "discoverPromotions": true,
  "artifacts": [
    "runs/<run>/formality",
    {"reference": "runs/<run>/sae-feature-62389", "family": "SAE",
     "label": "F62389", "note": "focal",
     "behavior": {"analyze": "runs/<run>-analyze", "condition": "sae-f62389",
                  "endpoints": ["choiceLogOdds"], "strata": "pooled"}},
    {"family": "LoRA", "label": "formality-lora",
     "behavior": {"analyze": "runs/<run>-analyze",
                  "condition": "lora-formality"}}
  ]
}
```

- An entry is an artifact reference string, or an object. `family` overrides
  the label derived from the sidecar's `extractionMethod` (`meanDifference` →
  CAA, `emotionGrandMean` → grand-mean, `optvec` → OptVec, `gemmaScopeSAE` →
  SAE, …); an **unrecognised** method becomes its own family rather than being
  folded into a known one, and an unstamped artifact reads `unstamped`.
- An entry with **no `reference`** is a behaviour-only row — the LoRA case: a
  fine-tuned arm has adapter weights, not a residual-stream direction, so it
  appears in the behavioural table and in no cosine. It must declare `family`
  and `label`.
- **Cosines are matched-layer only.** Two artifacts compare at layer *L* only
  when both carry a nonzero row there; a pair with no shared layer, a different
  `modelID`, or a different hidden size is reported `notApplicable` **with its
  reason — never as a cosine of 0**, which would assert orthogonality about two
  rows that were never in the same basis.
- `behavior.strata` is `pooled` (default — the condition-level rows) or `all`
  (adds the per-item breakdown). The choice, and how many rows were available,
  are stamped beside the copied rows. A condition the analyze run does not
  carry, or an endpoint it does not report, **refuses**: a silently empty
  behavioural join is indistinguishable from a family that did nothing.
- `discoverPromotions` (default true) scans `runs/model-variants` read-only for
  variants that inject each artifact, reporting the seated layer/α and the
  promotion birth certificate where there is one (matching on the
  `runs/<run>/<name>` tail, so agents promoted on the Mac with absolute paths
  are still found). Nothing found is reported as nothing found, never as "not
  promoted". A `sae-feature-qualification.json` beside the artifact is likewise
  reported as a pointer + hash, with the scope it was found at.

`check` additionally prints **warnings** (stderr, or a `warnings` array under
`--json`) for entries whose `status` is `qualified` or `rejected` but which
name no `qualificationArtifact` — a recorded outcome with no citable evidence
behind it. Warnings never change the exit code: the roster is authored
iteratively and the schema stays the verdict.

### 6.8 `sae qualification` — feature qualification as citable evidence

```
steerlab-server sae qualification record --inputs <inputs.json> \
                                         --artifact <runDir/name>
steerlab-server sae qualification show   <path> [--json]
```

Server-only, offline.

**What this is not: a seating mechanism.** Seating stays sweep → promote, like
every other vector family. A `sae-feature-qualification.json` functions like
capability-battery evidence — something the promotion chain *cites*.

`record` reads the researcher's declared evidence from `--inputs` (a JSON file
with the same CLOSED schema, minus the stamped fields — start from
`prompts/templates/sae-qualification/sae-qualification-inputs-template.json`)
and writes ONE immutable
`sae-feature-qualification.json` into a fresh
`runs/<stamp>-sae-qualification-<feature>/` directory beside the canonical
`config.json` (`runType: sae-qualification`). The record carries: the feature
identity copied from the artifact's own `gemmascopeSource` (repository +
revision, release / saeID / feature, layer, `decoderRowHash`); the held-out
construct-probe results per dose and per sign, each with a movement value and
a recorded direction; lexical-leakage results; discriminant-control blocks
(an empty array is the way to say "none were run"); the coherence/format gate
with its pass/fail; a dose-response summary; the accept/reject `decision` with
a rationale and date; and workspace-relative pointers to the runs each number
came from. Constructs, metric names, and thresholds are DATA — nothing here
computes a verdict.

Refusals (**exit 2**): the inputs do not validate (closed keys, enums, ISO
dates, positive unique doses, workspace-relative paths); the named artifact has
no readable sidecar; the artifact is **not** a direct-ID Gemma Scope import
(`gemmascopeSource.importPath != "direct-feature-id"`); any declared identity
leg — above all `decoderRowHash` — disagrees with that sidecar; the claimed
`doseGrid` has a rung with no construct-probe row on a claimed sign; or a
record already exists at the destination. Exit 64 on usage.

`show` prints one human-readably (or `--json`), including a
`consistencyViolations` list. Cross-checks are REPORTED there, never fatal — a
record whose artifact moved is still readable evidence.

**Citing one:** `experiment promote … --qualification <path>` (also
`{"qualification": …}` on `POST /api/experiment/{name}/promote`). The birth
certificate then carries `promotion.qualification = {path, contentHash,
decision}`. It refuses a record whose `decision` is `reject`, and one whose
feature identity does not match the artifact being promoted. Fully additive: a
promotion without the flag mints exactly what it always did, and the
`promotionKey` is deliberately unchanged (it is the cross-engine identity of
the promotion *request*, which Swift builds identically) — so a retry that
cites a different record returns the existing agent and logs that it did.

**Freeze advisory (never a gate):** a frozen study that seats a variant whose
injected vector is a direct-ID SAE import and whose promotion cites no
qualification artifact gets a loud non-blocking advisory, in the same family as
the hand-created-variant and exploratory-adapter advisories.

**Materialized copies.** A pinned concept re-materializes every run, so the
artifact a sweep persists — and therefore the one `promote` matches and a
variant injects — is a COPY that carries no `gemmascopeSource` of its own. Both
the citation check and the freeze advisory follow the copy's hash-pinned
`pinnedFrom` back to the import, and refuse the trail (rather than trust it) if
the origin's sidecar does not hash to the recorded value. One hop only.

**Sweeping an SAE concept.** The layer axis collapses to the dictionary's own
layer, logged loudly: an SAE artifact is full-depth zeros with the row at one
layer, so every other cell in a depth-fraction grid would inject a zero vector
and the selection rule would compare baseline against baseline. The grid becomes
an α ladder at that layer. An artifact that is not nonzero at exactly one layer
(all zeros, or several) refuses instead — the collapse would be a guess.

### 6.9 `gemmascope import-id` — direct SAE feature import (server-only)

```
steerlab-server gemmascope import-id \
  --model google/gemma-3-27b-it \
  --release gemma-scope-2-27b-it-res \
  --sae-id layer_40_width_65k_l0_medium \
  --feature 62389 \
  --label attributed-consciousness \
  --residual-norm-artifact runs/<run>/<name> \
  [--layer 40] [--neuronpedia-url URL] [--name <artifact-name>]
```

The SECOND SAE import path. The existing report-based import
(`POST /api/gemmascope/import`) requires the feature to appear in a cosine
report; a feature chosen on **semantic** grounds has no cosine relationship to
any CAA direction, and inventing one to satisfy the importer would fabricate a
pairing the study does not claim. This verb loads the decoder row straight from
the pinned release.

`--residual-norm-artifact` is a **calibration donor only**: any artifact of the
same model carrying `residualNormPerLayer`, following the OptVec
`residualNormArtifact` precedent. It supplies the residual-norm denominator, its
source label, and the model identity + revision the calibration was measured
under — **no semantic pairing with the feature is implied**. It is not a
reference vector.

The stored row is rescaled so `‖v‖ = ‖residual‖_L` at the SAE's layer and
stamped `gemmascopeConvention: "residual-norm-match"` — a NEW convention,
distinct from the report path's `"analyzed-vector-norm-match"`, never silently
mixed. Storage scale is cosmetic under norm-unit alphas (which re-normalize at
injection); it makes raw-α surfaces like chat read as residual fractions.
Degenerate guard as on the report path: a zero-norm row or a non-positive
target norm stores the row RAW and records why in
`gemmascopeSource.rescale.skippedReason`.

The sidecar's additive `gemmascopeSource` block pins the complete provenance:
Gemma Scope repository + **exact resolved commit** (never a floating ref),
release/saeID/feature, layer/width/L0/site, SAE-config hash, raw-decoder-row
SHA-256 (of the RAW row, so the same feature hashes identically whatever donor
calibrated it), the calibration donor and its norm source, an optional
Neuronpedia URL as **discovery** provenance only, and the importing build
identity + date.

Refusals (exit **2**): donor with no `residualNormPerLayer` at the SAE layer;
donor measured on a different model (calibration does not transfer); decoder
row whose dimension is not the model's hidden size; layer outside the model's
depth; `--layer` disagreeing with the SAE's own layer; an SAE whose layer
cannot be determined and no `--layer`; a loader that cannot pin the repository
commit; an artifact already present at the destination (artifacts are
immutable — refuse, never clobber). Exit **64** on usage/missing flags; **0**
prints the result JSON (path, concept, convention, norms, source block).

API twin: `POST /api/gemmascope/import-id` (privileged — it goes online and
reads a caller-named artifact, same gate as `/api/jlens/lenses/acquire`; the
older `/api/gemmascope/import` and `/api/gemmascope/info` are unchanged). It
validates synchronously and runs the download as a durable job.

Related: `gemma_scope.analyze` now accepts `requestedFeatureIDs`, so ONE report
can carry both the CAA-aligned top-k and a semantically chosen shortlist — the
requested rows come back in their own `requested` bucket, flagged
`"requested": true`, with cosine, sparsity, `rawDecoderNorm`, and
`decoderValues` whether or not they ranked. An out-of-range requested id
refuses rather than vanishing from the report.

There is no Gemma Scope import-id verb on the Swift CLI: the direct import is
server-side, and the Swift `SteeringVectorSidecar` has no `gemmascopeSource`
twin field yet (a Swift decode→re-encode, e.g. `NormBackfill`, would drop the
block).
### 6.10 `battery` — capability-battery preflight and authoring brief (server-only, 2026-08-13)

```
steerlab-server battery lint <workspace-relative-path> [--json]
steerlab-server battery generation-prompt [--count N] [--avoid <domain text>] [--out <file>]
```

#### `battery lint` — the preflight

Offline, byte-only preflight for a capability battery
(`prompts/batteries/*.jsonl`). No model, no network — it runs in CI and on a
login node. Exit **2** on any blocker, 0 otherwise (warnings still print), 64
on usage. `--json` prints the report instead of lines.

A capability battery is a CONTROL: its job is to hold still while the
intervention moves. Every check names a way it can fail to hold still, and
each is derived from a mechanism observed in practice — in the case that
prompted this verb, one pinned battery read 0.45 on one instrument and 1.00
on another, same model, same round.

**Blockers.** The file will not load or parse; **format 1** (headerless
legacy: it is armed by the *study's* `promptMode`/`systemPrompt`, so its
accuracy is instrument-dependent and a condition that breaks the study's
output format can score *higher* than baseline); options that collide after
normalization.

**Warnings.** Fewer than 10 items (one item is worth too much of the score);
fewer than 3 options (chance floor); option-length imbalance (joint logprobs
favour short continuations); an option that is a prefix of another; duplicate
prompts; a battery that declares its own system prompt; per grading mode, the
length/format sensitivity it carries — `containmentScored` (`token_exact` is
scored by containment, so a long response can be correct by accident),
`singleNumberRequired`, `wholeResponseEquality`, `firstTokenWins`,
`regexScored` — and prompts whose correctness depends on obeying a
response-format instruction.

Batteries come in two formats, and the format decides both scoring and
arming. **Format 1** is the historical headerless file, unchanged in every
respect so existing pinned hashes keep their historical meaning. **Format 2**
opens with a header line —

```jsonl
{"batteryFormat": 2, "scoring": "choiceProbability", "promptMode": "chatAssistant", "maxTokens": 24}
{"id": "cap-fr", "prompt": "What is the capital of France?", "answer": "Paris", "options": ["Paris", "Lyon", "Nice"]}
```

— declares its own arming (applied identically to baseline, steering, and
variant conditions) and defaults to answer-token-logprob scoring, so nothing
is generated and the score cannot move with response length or format
compliance. Template:
`prompts/templates/battery/capability-battery-v2.template.jsonl`. Format 2 is
server-only until a Swift twin lands.

#### `battery generation-prompt` — the authoring brief (server-only)

```
steerlab-server battery generation-prompt [--count N] [--avoid <domain text>] [--out <file>]
```

The other end of the same contract: an LLM authoring brief for a format-2
battery, printed as plain text on stdout (`--out <file>` writes it instead and
confirms on stderr). Reads no model and writes nothing without `--out`. Exit
**0**; 64 on an unknown flag, a flag missing its value, or a `--count` that is
not a positive integer.

The brief is *assembled from the contracts*, not restated: the header/item
schema comes from the format-2 loader, and the rules it tells the drafter to
satisfy are the linter's own thresholds and finding codes, so a draft written
against it passes `battery lint` first time and the brief cannot drift from
the checks. It carries the shipped template as its worked example — the
workspace's copy if there is one, else the seed tree's, else an inlined copy,
and it names which.

`--count` defaults to **20** (the shipped batteries run 10–20 items and the
linter warns below 10); a smaller count still renders, with a note that the
draft will earn a `fewItems` warning. `--avoid` is free text naming the
subject matter the battery must not touch — the study's own domain — and
becomes a rule in the brief; omitted, the brief says the domain was not
supplied and to ask for it. The parallel helper for contrastive *stimulus*
sets is `GET /api/concept/{name}/prompt`; this verb's own route is
`GET /api/battery/generation-prompt?count=&avoid=` (a GET: text render, no
model, no writes, no caller-named paths).

### 6.11 `experiment preflight-endpoints` — endpoint-safety preflight

```
steerlab-server experiment preflight-endpoints <name> [--baseline-run DIR] [--out PATH] [--json]
                                               [--band LOW,HIGH] [--min-cell-items N] [--min-items N]
```

Server-only. Asks, **before a screen consumes compute**, whether the declared
instrument can identify the effect it declares. Loads no
model and touches no network; a baseline run directory, when given, is opened
**read-only**.

**Static half** (manifest + its pinned item file only):

- **Factorial aliasing** — the design matrix over the items' declared
  `factors` (plus top-level `arm`/`caseID`, the same metadata `analyze`
  stratifies on). Reports rank against declared contrast columns, every pair
  of factors inducing the SAME partition of the item set (a perfect confound,
  detected without reference to level names), which contrasts are separately
  estimable, and — where one exists — the minimal set of other factors that
  already spans an unestimable one. A factor with one level per item is
  reported as a **saturated** item identifier, not a factor.
- **Effective item count** per endpoint and per stratum cell, over the exact
  families `analyze` will build.
- **Signed-cancellation exposure** — a signed endpoint (`choiceLogOdds`,
  `ordinalPosition`) averaged over strata that can straddle its boundary, with
  the mean-|·| companion recommended.

**Baseline-informed half** (`--baseline-run <run-dir>`): per-item baseline
probability against the usable band (default 0.2–0.8, `--band`) and the
surviving **effective instrument width**; per-condition missingness and
parse-failure rates; `label`-format word-count distributions and the
**format-compliance sensitivity** flag when treated conditions drift from
baseline; and signed cancellation escalated from exposure to a measured
verdict (confirmed / straddles / cleared).

Output: one durable JSON report plus a human summary, with a verdict per
endpoint (`ok` / `warning` / `blocker`) — a design-level blocker is a blocker
for every endpoint measured on that item set. Exit codes follow `data check`:
**0** ready (warnings allowed), **2** on any blocker or an unusable input,
**64** on usage.

The report lands in a new immutable `runs/<stamp>-exp-<name>-endpoint-preflight/`
directory stamped `runType: "endpoint-preflight"`, or at `--out PATH` (which
creates no run directory — for read-only or CI trees). The run-directory
`config.json` schema-4 contract closes the top-level **key** set, not the
`runType` vocabulary, so the new run kind is additive and moves no cross-engine
closed-key test.

**Not wired into `run`/`study submit` yet.** Deliberate: those verbs' start-up
output and refusal behaviour are covered by existing tests on both engines, and
this change set is additive-only. Call it explicitly before a screen.

### 6.12 Seating imported SAE features — mixtures and the preregistration guard

An imported feature is an ORDINARY vector artifact, so it enters a study the
same way every hash-pinned direction does — as a `pinnedArtifact` concept —
and a condition mixing several of them needs **no new mechanism**: a
condition's `slots` list already resolves to one injection cell per slot, and
injection cells compose additively, so several slots at once *are* the linear
mix `h + Σ αᵢ·vᵢ`. The mix is hashed into the study by the manifest bytes like
any other condition; there is no separate "mixture" artifact or verb.

```bash
steerlab-server experiment attach-artifact <study> textualism-sae \
    --artifact runs/<import-run>/sae-feature-62389
steerlab-server experiment attach-artifact <study> compassion-sae \
    --artifact runs/<import-run>/sae-feature-11409
```

Then declare the mixture as a condition in `experiment.json`:

```json
{
  "conditions": [
    {
      "name": "textualism+compassion",
      "alphaInNormUnits": true,
      "slots": [
        {"concept": "textualism-sae",  "layer": 40, "alpha": 1.0},
        {"concept": "compassion-sae",  "layer": 16, "alpha": 0.5}
      ]
    }
  ],
  "saeCandidates": {"path": "prompts/sae/roster.json", "hash": "…"},
  "maxSAEMixtureFeatures": 4
}
```

Each slot resolves independently: its own layer band, its own norm-unit α
folded against its own vector norm at its own layer. Slots may mix an imported
SAE feature with a grand-mean or CAA direction — the slot machinery only sees
pinned artifacts. Two slots at the SAME layer sum there, which is the sparse
multi-feature intervention the proposal describes (r2 §8 P2-10).

**Lifecycle exemptions.** A decoder row is a coordinate in a published
dictionary, not a contrast between two authored classes, so an imported feature
has no stimuli, no source concept, and no held-out `validation.jsonl` — exactly
like an OptVec vector. `attach` therefore carries its
`gemmascope:<release>:<saeID>:<feature>` identity hash verbatim, pins
`validationHash` explicitly null, refuses a `--source-concept`, and `verify`
never looks for `prompts/concepts/<name>/`. What is NOT exempted: both artifact
files stay hash-pinned, the model/revision/substrate identity checks still
refuse, and `gemmaScopeSAE` is a SOURCE method only — attaching it as a
*recipe* method (the authoring API's `attach`, `method: "gemmaScopeSAE"`)
refuses, because nothing re-derives a decoder row from stimuli.

**The freeze guard** (`verify()`, so `--force` cannot skip it — this is
pin-surface integrity, not an evidence gate):

- *Roster membership.* When the manifest pins an SAE candidate roster
  (`saeCandidates`, §6.6), **every** SAE feature a condition seats must be
  nominated in it — one violation per unnominated slot, naming the condition,
  the concept, and the model/layer/feature it holds. The roster is exactly the
  preregistration of which features the study may seat, so seating a feature it
  does not list is the mechanical shape post-outcome feature selection takes.
  Pin **no** roster and nothing new is checked (the guard is additive; every
  existing manifest verifies unchanged). The check covers single-slot
  conditions too — a guard that only looked at mixtures would be avoidable by
  splitting one mixture into several conditions. Editing the roster to add the
  feature after pinning is ordinary byte drift and refuses on the hash.
- *Sparsity.* A condition may seat at most `maxSAEMixtureFeatures` SAE features
  — default **4**, declared in the manifest to raise it. A denser mixture stays
  legal; it just has to be a recorded decision made before behavior is
  measured. The cap counts SAE-backed slots only, so a condition mixing many
  non-SAE directions is untouched.

Matching is on `(modelID, layer, featureId)`. The roster records its dictionary
in the discovery surface's vocabulary (`source: "gemmascope-2-res-65k"`) while
the artifact records Gemma Scope's (`release` + `saeID`); no mapping between
the two is authored anywhere, so inventing one here would make the guard
silently wrong on the first dictionary that broke today's naming convention.
Consequence to know: two same-numbered features at one layer in the 65k and
262k dictionaries are indistinguishable to this check — that disambiguation
stays a human one. A report-ranked import (no `gemmascopeSource` block) carries
no matchable identity at all and is REFUSED outright while a roster is pinned;
import by feature id instead.

Server-only, like the rest of the SAE path — there is no Swift twin for
`saeCandidates`, `maxSAEMixtureFeatures`, or the `gemmaScopeSAE` extraction
method, so a Swift `verify` of such a manifest would neither run these checks
nor recognise the method.

### 6.13 `site qualify` — structural parity on a new node (server-only)

<!-- GENERATED:server-site BEGIN -->
<!-- Generated from the declarative verb table — `steerlab-server docs cli-reference --write`. Edit the table, not this block. -->

```
steerlab-server site qualify [--skip-model-fixtures]
```

| Verb | Purpose |
|---|---|
| `site qualify` | Run the committed fixtures on this node and report structural parity, check by check. |

Every verb above also accepts `--help` (print its arguments and run nothing), `--json` (one envelope on stdout), and `--out <file>`.
<!-- GENERATED:server-site END -->

Run this ON the node being qualified — a login node or a workstation shell;
no GPU is required for the core checks. It answers the question the deploy
script cannot: *is this an instrument, or merely a reachable machine?*

Ten checks run in a fixed order and **none aborts the rest**, so a cold node
always gets a complete report:

| id | what it verifies |
|---|---|
| `buildIdentity` | the version string this deployment stamps on everything it writes |
| `pythonEnvironment` | the interpreter and science-package versions (recorded, never gated) |
| `dependencyLock` | installed torch/transformers against the committed platform lock |
| `stimulusHash` | a committed stimulus fixture hashing to its pinned SHA-256 here |
| `goldenRender` | every committed render fixture's exact prompt string |
| `goldenTokens` | the same fixtures' token ids and leading-BOS count |
| `vectorParity` | the committed synthetic parity comparisons against their goldens |
| `serverProfile` | folded from `profile validate` (roots, locking, scheduler, bind/auth) |
| `controllerScript` | the rendered controller job script's render stamp against the deployed controller-job template's SHA-256 |
| `cudaProbe` | CUDA visibility, and the device name against the declared GPU vocabulary |

Each row of the report carries `what`, `why`, `expected`, `observed`, and
`detail` — written for a reader who has never seen our baseline. The full
document rides in `result.report` under `--json`, prints on stdout otherwise,
and `--json OUT` also writes it to a file.

**Verdict:** any `fail` → `state: failed` (**70**), naming the failing ids;
`warn` with no `fail` → `okWithAdvisories` (**0**), one `siteQualifyWarning`
advisory per warning; otherwise `ready` (**0**). **64** is a malformed
invocation. **Skips never change the verdict** but are always counted in the
summary line (`6 passed, 1 warning, 0 failed, 2 skipped of 9 checks`), so a
report that verified almost nothing cannot read as a pass.

`goldenRender`/`goldenTokens` need each fixture's tokenizer from the **local**
HF cache (`local_files_only`; qualification never reaches the network). A
partial sweep is a **warning**, not a pass — verifying four of thirty-six
fixtures is not a verified node. `--skip-model-fixtures` forces both to skip
(the fast CI path).

`vectorParity` uses **same-engine synthetic** fixtures: it qualifies the parity
*arithmetic* on this node, not cross-substrate agreement. Activations do not
transfer between MLX/Metal and PyTorch/CUDA, so vectors must still be
re-extracted and re-validated on whichever substrate a study runs on. Real
MLX-vs-CUDA fixture pairs remain an open item. The check says so in its own
`detail`.

---

## 7. Known gaps and traps

Documentation-only observations from reading the dispatch code. **Nothing here
was changed** — where a fix is obvious it is stated as a recommendation, not
applied.

### 7.1 `--verb` defaults to `run` on both submit paths

`study submit` and Swift `remote submit-bundle` both default the verb to
`"run"`. The vocabulary appears nowhere except the usage string printed on
error, so a researcher who omits `--verb` gets a full measured run with no
confirmation. This has already cost a `scancel` in the field — a sweep
submitted as a run.

*Recommendation (not implemented):* make `--verb` required on `study submit` —
exit 64 with the vocabulary when absent. The default costs a GPU allocation;
requiring it costs one word.

### 7.2 Sharding is half-exposed *(partly closed 2026-08-07, further 2026-08-28)*

The original observation: `--shard k/K` was a CLI flag whose merge step had no
CLI counterpart, and the automatic fan-out that *does* merge (`parallelJobs`)
was reachable only from `POST /api/studies/submit-bundle`.

**Closed:** `study submit --parallel N` (§5.1) routes through the same
`_submit_sharded_bundle` parent/shard-record machinery, so a terminal fan-out
merges exactly like an app-submitted batch.

**Also closed (2026-08-28):** the Swift twin. `steerlab-cli remote
submit-bundle --parallel <n>` (§3.8) now parses the flag and threads it into
the `ClusterClient.submitBundle(… parallelJobs:)` the app's Experiments panel
already used, under the same `ShardedSubmission.encodedParallelJobs` rule —
and echoes `parallelJobsRequested` / `parallelJobsEncoded` /
`parallelJobsSuppressedBecause`, plus a stderr warning, so a request the rule
suppressed never looks honored.

**Still open:**

- There is still no `bundle merge-shards <dirs…>` wrapping
  `sharding.merge_shard_runs`, so hand-run `bundle execute --shard k/K`
  partials remain unmergeable. Anyone reaching for `--shard` by hand should
  use `--parallel` instead.
- The merge still requires a **running** `steerlab-server serve` (§5.3). A
  headless `steerlab-server jobs poll` — one `poll_slurm()` tick against the
  store — would make `--parallel` usable on a login node with no daemon, and
  is a few lines. *Recommendation, not implemented.*
- There is still no headless **merge trigger** on either command line: the
  merge is the running server's reconciler, so a fan-out submitted and then
  left with no `serve` process never merges (the bullet above).
- `_sweep_orphans` fails any sharded parent it finds in `pending`, so
  constructing a second `JobManager` (any other CLI verb) while a fan-out is
  mid-submission kills it. A short grace window on parent age would fix it.

### 7.3 Usage strings under-report the accepted flags

- `study submit`'s usage line **used to** omit the nine flags it parses
  (`--target`, `--dtype`, `--device`, `--prompts`, `--source`,
  `--no-evidence`, `--mem`, `--partition`, `--job-name`). Fixed 2026-08-07:
  the usage line now names all of them, plus `--parallel`, plus a one-line
  statement of how GPU type is chosen.
- `bundle execute` parses `--dtype`, `--device`, `--prompts`, `--source`,
  `--no-evidence`, `--record`, `--resume-from`; the usage line shows only
  `--verb`, `--target`, `--shard`. **Still open** — `bundle` is not an
  agent-path family and is still hand-parsed.
- `experiment`'s verb lists omitted `pipeline`, `judge-worker`, and (in the
  second of the two) `preflight-endpoints`. **Closed 2026-08-18:** both
  refusal sites print one derived list (`cli.EXPERIMENT_VERBS`), asserted
  against the dispatch by
  `test_cli_reference.py::test_the_printed_experiment_verb_list_is_complete`.
- Swift `remote`'s usage line omitted `--gres`/`--walltime` on `submit-bundle`
  and every flag of `chat` beyond `--variant`/`--prompt`. **Closed
  2026-08-18:** every agent-path verb's flags come from the declarative table,
  through `--help` and through §3.8's generated region.
- `jlens probe` and `jlens g0`'s usage strings and this document disagreed in
  both directions (audit §1.3 D6/D7). **Reconciled by hand 2026-08-18** against
  the parser; `jlens` has no declarative table, so this one stays a
  hand-maintained claim.
- Swift's **top-level** usage omitted the `panel` and `vectors compare`
  families entirely — an unlisted verb is indistinguishable from an absent one,
  and this catalogue itself had missed it (audit §1.3 D8). **Closed
  2026-08-18:** one page, printed by `steerlab-cli --help` (exit 0) and by the
  usage error (exit 64), naming every family the binary dispatches.

### 7.4 Client and server defaults do not meet — CLOSED 2026-08-18

`steerlab-cli remote` used to default to `http://127.0.0.1:8000` while
`steerlab-server serve` and Swift `serve` both default to 8080, so every
`remote` invocation against a default-port server had to pass `--url`. The
client default is now **8080** and the recommendation above was taken:
`remote` with no `--url` reaches a default-port server.

### 7.5 Silent fallbacks

- Swift `serve --port notanumber` falls back to 8080 instead of erroring.
- Invalid `_choice` env values (`STEERLAB_EXECUTOR`, `STEERLAB_SERVER_PROFILE`,
  `STEERLAB_AUTH_MODE`, `STEERLAB_LAUNCH_TOPOLOGY`, `STEERLAB_SERVER_ROLE`)
  fall back to the default with no diagnostic. `profile show` is the only
  readout. Three variables buck this and fail loudly on malformed input:
  `STEERLAB_SLURM_GPU_VRAM`, `STEERLAB_AUTO_RESUBMIT_LIMIT`,
  `STEERLAB_MAX_LOADED_MODELS`.
- Both CLIs' flag helpers read "the next argument" with no validation, so
  `--threshold --json out.json` consumes `--json` as the threshold value. Flags
  that take values are positional-fragile in both engines.
- `steerlab-server experiment list` prints workspace `.trash-*` directories as
  `unreadable` experiments — cosmetic noise, not an error.

### 7.6 Verb coverage is uneven across engines

`rescore-style` exists on both CLIs but is not a submittable study verb, so a
cluster-side reasoning-style rescore must run as `experiment rescore-style` on
the node rather than through `study submit`. `pipeline` exists only on the
Python engine. Authoring verbs exist only on Swift (plus the HTTP authoring
API) — including, since 2026-08-10, the artifact-pin AUTHORING half of
`attach-artifact` (§4.3): Swift mirrors the manifest keys (`options.method:
"pinnedArtifact"` + `vectorArtifact`), the attach refusals, the verify hash
re-checks, and the optvec validate-gate exemption + eval-run advisory
(`ExperimentStore.attachArtifact`; the app's Data → OptVec tool is the UI).
There is still no Swift `attach-artifact` CLI verb, and the
`extract`/`validate`/`sweep`/`run` of an artifact-pinned study remain
server-only — Swift's extract refuses pinnedArtifact concepts loudly. The
pinned bytes are substrate-stamped anyway, so that is where such a study
belongs; the Mac authors and freezes it.

### 7.7 `--help` — closed 2026-08-18 (WP0 step 11)

**Was:** neither CLI had `--help`; both printed usage only on error, only for
the level that failed, and this document was the substitute — the wrong way
round for an agent driving a shipped binary with no checkout to read.

**Now:** `--help` is a declared flag on every agent-path verb of both engines,
rendered from the same declarative tables the generated regions of this document
come from. It runs nothing and exits **0**:

```
steerlab-cli   [<family> [<verb>]] --help
steerlab-server <family> [<verb>]  --help
steerlab-cli   cluster [<verb>]    --help
```

A verb page prints the synopsis, the purpose line, every declared flag with its
value shape and one-line meaning, and the exit-code vocabulary; a family page
lists that family's verbs. Under `--json` the page travels as the envelope's
`result` (`flags[]`, `positional`, `synopsis`), so a machine caller never parses
the columns a human reads.

`--help` is the **only** flag this change declared. Every other undeclared flag
is still `EX_USAGE` (64) on both engines, before the verb does any work.

Residual: the non-agent-path verbs (`artifacts`, `jlens`, `optvec`, `sae`,
`bundle`, `finetune`, `housekeeping`, `profile`) are hand-parsed and have no
`--help`; they still print their own usage line on error. Swift's `panel`
family left this list on 2026-08-19 with `panel compile` (open-issues §18);
the server's `panel list`/`check` (§6.2) have not.

### 7.8 Cluster lifecycle: what is proven and what is not

Every verb in §3.9 is exercised against fakes — scripted `ssh`/`rsync`/`squeue`
transcripts, an in-memory Keychain, a fake forward and endpoint. **None of it
has met a live Slurm cluster yet.** The live shakedown (cold start through
multi-factor authentication, CLI exit with a queued job, recovery, Mac sleep,
app adoption) has not run. Treat the machine protocol as
stable and the *live* behaviour as unproven until that shakedown passes.

---

## 8. Adjacent entry points (`scripts/`)

Not CLI verbs, but the shell entry points a researcher meets alongside them.

| Script | Usage | What it is |
|---|---|---|
| `scripts/run-app.sh` | `./scripts/run-app.sh` | Builds and launches the SwiftUI app via `xcodebuild` (prefers `/Applications/Xcode-beta.app`), then re-signs the executable so the Metal shader bundle is visible. |
| `scripts/start-local-server.sh` | `start-local-server.sh [--root <workspace-dir>] [--port N]` | The one-click local Python server path the app's connection menu runs, equally usable from a terminal. Cleans a stale pidfile, **refuses with exit 3 if something already listens on the port**, creates `Server/.venv.nosync` and `pip install -e "Server[lora,gemmascope]"` when anything is missing (the first install pulls torch/transformers and can take many minutes; progress streams so a silent wait never looks like a hang), then writes `<root>/.steerlab-local-server.pid` so a relaunched app can adopt the running server. Serves with `--dev-open-loopback` (§4.2) — the single-user Mac posture, chosen explicitly on the argv; drop that flag from the script to get the shipped token default here too. |
| `scripts/run-viewer.py` | `python3 scripts/run-viewer.py [--root <workspace>] [--port 8765]` | Read-only loopback-only run browser: validate accuracies with binomial p-values and the cross-concept cosine heatmap, the sweep's layer × alpha grid with constraint failures marked and an explicit statement of whether a matched-norm random control was declared, and a paired generation browser. Writes nothing. |
| `scripts/make-server-payload.sh` | see the script header | Builds the immutable cluster deployment payload (the filtered `Server/` tree plus parity fixtures) with a `deployment-manifest.json` of SHA-256s. Its rsync filter list must stay in sync with `ClusterProvisioner.pushFilterArguments`. |
| `scripts/regenerate-cross-engine-fixtures.py` | see the script header | Regenerates the committed cross-engine parity fixtures. |

The Results Explorer is the canonical results surface (`results-explorer/`,
built into `web/results-explorer/`); `run-viewer.py` is the lightweight
read-only fallback, not a replacement for it.
