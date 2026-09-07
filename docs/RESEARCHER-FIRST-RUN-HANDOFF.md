# Researcher first-run implementation handoff

Branch: `codex/researcher-first-run`. Started from main at `7284463`;
main’s subsequent documentation-only commit `6c8703c` is integrated as well.
Main, installed apps, user client environments and running studies are unchanged.
This branch is for the maintainer's coding/audit agents to review and integrate.

## What researchers gain

A new researcher can choose the Mac app or the app-free client. Either creates
a complete data workspace with the same method guides, interviews, dataset
prompt templates and AGENTS.md. Neither requires a code checkout for authoring.
The researcher can describe a question, give an agent the generated handoff,
and review the proposed study before choosing execution hardware.

The Mac's Research Setup screen opens on the first incomplete setup and remains
available in the Workspace menu. It offers workspace creation/opening, a reviewed
client installation and agent handoff. Workspace creation through this screen
leaves compute undeclared. Existing workspace/compute menus remain available.

The app-free release contains a wheel and a pre-Python installer. `plan` is
read-only; installation needs its current hash and explicit approval. It supplies
managed Python and locked lightweight dependencies. `steerlab setup start <dir>
--create --json` then creates a workspace and returns readiness and handoff.
Omit `--create` to use an existing workspace. No interactive terminal questionnaire
blocks an agent using JSON.

These are equivalent authoring paths, with explicit command discovery. The Mac
CLI uses `--workspace`; the Python client uses `--root`. Generated handoffs name
the actual executable and its valid discovery arguments. The agent guide warns
against copying Mac-only lifecycle verbs under the Python executable's name.

## Implementation slices

1. `49d5d0b`: shared bootstrap inventory, marker, ignore rules and agent guide;
   complete packaged workspace seed; Python workspace creation; cross-client
   byte parity and native/Python handoffs.
2. `8068760`: release builder, pre-Python installer, hashed client dependency
   lock, atomic activation, failure preservation and isolated release qualification.
3. `42033cb`: shared readiness and setup owners; Python/Mac CLI adapters;
   compiled-identity admission for the Mac installer and setup documentation.
4. The final implementation commit: first-run UI/presentation model, `setup start`,
   updated help/census gates, method and draft-study release qualification, and
   the regression fixes described below.

Maintained workspace metadata lives under `Server/steerlab_server/client/resources`.
The guide's maintained text remains `docs/AGENTS-WORKSPACE-DRAFT.md`; the seed files
remain under `WorkspaceSeed`. `scripts/ci/check-workspace-bootstrap.py --write`
generates the packaged guide/seeds and compiled Swift constants. Existing resource
generators and the compiled Python identity gate continue to apply.

`ClientSetup` and the Python setup adapter both invoke the release's installer.
The shell implementation is necessary before Python exists; it owns installation
plans, serialization, staged verification and activation. Readiness after Python
exists is a shared Python operation reached through the existing identity-checked
local process protocol. The UI's presentation model owns no scientific validation.
Setup is local; no remotely callable HTTP installation mutation was added.

## Installation and repair rules

- Supported installer targets: Apple Silicon macOS and x86_64 Linux with glibc.
  No system Python or compiler is used by installation. Curl, tar and hashing
  utilities are required; downloads require internet access.
- uv 0.12.5 downloads use platform-specific SHA-256 pins. Managed Python is
  3.12.14. Client dependency versions and allowed wheel hashes are committed.
  No runner extra, torch, transformers or FastAPI is installed.
- The runtime and its interpreter stay at their created paths. Activation changes
  a public symlink only after imports, seed presence and source identity verify.
  Old managed environments are retained. An ordinary existing directory is never
  replaced. Download/verification failures preserve the previous active runtime.
- Apply/repair require the reviewed plan hash plus explicit approval. A lock
  serializes installers; state and release bytes are checked again before activation.
  A stale lock is never stolen automatically.
- The app packages the same release before signing. It keeps source in its own
  ServerPayload; only the interpreter/dependencies live in the external runtime.
  The identity gate remains intact. A development source update requires rebuilding.
- Explicit interpreter overrides retain priority. Installing the default runtime
  does not silently override an invalid `STEERLAB_CLIENT_PYTHON` selection.
- Setup never installs models, starts servers, chooses credentials or cleans
  cluster files. Readiness distinguishes client, workspace and execution state.
  `authoringReady` is not scientific or model qualification.

A workspace can be created without Git. On a fresh Mac, bootstrap checks developer
tool selection before calling Apple's Git shim, avoiding an unsolicited tools
installation prompt during workspace creation. Runs and frozen manifests are
unchanged; no stale-write revision field was added to content-hashed documents.

## Regressions closed during verification

Launching the client via `python -m` used to give family adapters a second copy of
the CLI's exception classes, turning typed refusals into operational failures.
The module now registers its canonical name before dispatch. First-run refusal
repairs are preserved through the Mac process adapter, and a missing workspace
points to explicit creation rather than environment installation.

Packaging tests now check whether declared package-data patterns cover the actual
resources instead of requiring the previous narrow glob spellings. Release smoke
tests additionally exercise the built wheel without the checkout. Exhaustive Swift
namespace, help and command-reference gates include the new verbs.

## Validation and integration

Validation on the completed implementation:

- Python: 6,166 passed, 9 skipped, 8 warnings.
- Xcode beta: 277 SteeringKit and 4,601 ExperimentKit tests passed; app build passed.
- Resource, interview, client-reference and compiled-identity gates passed.
- All 11 audited scientific owner ASTs remain unchanged; lens-consumer and task
  parser mutation controls passed; bridge gates passed in normal and release modes.
- Mac CLI smoke: setup start, readiness and both emitted discovery commands passed.
- Release install/repair, method interview and draft authoring are qualified in
  disposable macOS scratch without a checkout or GPU dependencies. Linux CI is
  configured; it has not been run from this session.

This handoff does not authorize publication, deployment or main-branch integration. The maintainer's independent review process remains the landing gate:
read the entire diff, rerun both suites and applicable AST/resource gates, then
integrate through the user’s reviewing agents.

For fresh-machine qualification, use `scripts/ci/qualify-client-release.py` on the
built release with `--repair`. The committed Linux workflow runs it independently.
Before wider launch, qualify the actual packaged GUI on a Mac without a checkout
or developer tools, including offline failure, explicit overrides, repair and
upgrade. Do not equate a developer-machine build with that qualification.

See [CLIENT-FIRST-RUN.md](CLIENT-FIRST-RUN.md) for the researcher-facing flow and
[PYTHON-CLIENT-RUNTIME.md](PYTHON-CLIENT-RUNTIME.md) for runtime selection and release
procedures. Existing scientific execution, cluster-policy and immutable-run gates
remain authoritative when the researcher proceeds from authoring to execution.
