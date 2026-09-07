# Review: `codex/researcher-first-run` @ `c009d23` (on main `6c8703c`)

Reviewer: the maintainer's integration agent, 2026-09-07. Landed the same day; the landing commit fixes N4 (versions defined once in the installer, identity regenerated, release rebuilt and re-qualified in scratch with repair) and adds the F1 changelog entry. Read against
`docs/RESEARCHER-FIRST-RUN-HANDOFF.md`. Nothing on the branch was edited. The
branch is based on the local main tip (which includes the unpushed briefs
commit `6c8703c`), so a fast-forward is available. Six commits, 94 files,
+4,685/−1,786; the large deletion is the agent contract leaving Swift source
for a generated resource, and the large addition is the complete workspace
seed now packaged into the Python client.

## 1. Verdict

**Landable by fast-forward, with one omission to fix at landing (F1) and two
operational consequences to know before the next app build (N1, N2).** The
installer is careful in the ways that matter: pinned and verified download,
hashed dependency lock, wheels only, isolated staging, atomic activation,
refusal to replace anything it did not create, and a plan hash that binds
the release bytes, the destination and its current state. I built the
release and ran the scratch installation with repair on this machine, and
it passed. No scientific owner is touched.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`6c8703c`) is an ancestor of `c009d23`; `origin/main` is `7284463` (the briefs commit is local only) |
| uv download pins | both `uv_sha` values in `install-client.sh` equal the checksums published with uv 0.12.5 for `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`, fetched from GitHub by me |
| Release build and qualification | `scripts/build-client-release.py` (uv 0.12.5 from a scratch venv) produced a 2.2 MB release: wheel, installer, helper, lock, `source.sha256`, README, `SHA256SUMS`; `scripts/ci/qualify-client-release.py --repair` in scratch passed: install, `setup start --create`, handoff, `science list`, `setup inspect` ready, gradient interview, `experiment create/inspect`, source identity equal to the release stamp, no torch/transformers/fastapi importable, repair activated a new environment and kept the old one |
| Dependency closure actually installed | nine locked packages (anyio, certifi, h11, httpcore, httpx, idna, numpy, safetensors, typing-extensions) plus the client wheel with `--no-deps`; managed CPython 3.12.14 |
| New gates | `check-workspace-bootstrap.py` (seed inventory equals `WorkspaceSeed/`, packaged copies byte-identical, agent guide regenerated from the draft, compiled Swift constants current) and `check-python-client-identity.py` pass; identity now also covers `client/resources` |
| All prior gates | owner audit with mutation controls, stability preflight, science resources, client reference, task-prompt parser, lazy imports, study interviews, bridge normal and `--release`, `public_scan.py`: pass |
| Vocabulary in the diff and six commit messages; `git diff --check` | clean |
| Full Python suite on the worktree (main venv, cwd = worktree/Server) | 6,166 passed, 9 skipped, 8 warnings, matching the branch's claim (run concurrently with the Xcode suite, no flake) |
| Full Xcode beta suite from the venv-less worktree (`TEST_RUNNER_STEERLAB_TEST_PYTHON` in the shell environment before `xcodebuild`) | `TEST SUCCEEDED`: 277 SteeringKit + 4,602 ExperimentKit, matching the branch's claim |

## 3. What the code does

**Installer (`client/resources/install-client.sh`, `runtime-helper.py`).**
POSIX shell, works before Python exists. `plan` is read-only and returns a
`planSHA256` over the runtime path, its state (missing or managed), the
platform, the link target, the hashes of the four release files and the
wheel, and the existing receipt if any. `install`/`repair` require `--yes`
and `--expect` equal to a fresh plan, refuse an unmanaged existing directory
or a link it did not create, take a `mkdir` lock beside the runtime and never
steal a stale one, stage everything under the runtime's parent, re-plan from
the staged copy, download uv over HTTPS with TLS 1.2 or better and refuse on
checksum mismatch, then with `UV_NO_CONFIG` and every index override unset
create a managed 3.12.14 venv, `pip sync --require-hashes --only-binary`
from the committed lock against PyPI, install the wheel with no
dependencies, and hand off to the helper under `-I` in the new interpreter.
The helper imports the client and its three dependencies, recomputes the
source identity and requires it to equal the release stamp, checks the seed
and guide are present, re-plans once more, writes a receipt, and activates
by creating a symlink beside the venv and `os.replace`-ing it over the public
runtime path. Failure before activation removes the stage and reports a
structured refusal; the previous runtime is untouched. Tests cover quoted
paths, unmanaged targets, plan invalidation, lock closure and an offline
`curl` fixture.

**Release builder and qualification.** `build-client-release.py` runs four
gates, builds the wheel with `uv build` from a disposable copy of `Server/`,
copies the installer files, writes `source.sha256` and `SHA256SUMS`, and
never installs. `build-app.sh` now runs the identity gate first and packages
this release inside `ServerPayload/client-release` before signing.
`qualify-client-release.py` installs into scratch and drives the client
without a checkout; a Linux workflow runs it on pull requests that touch the
relevant paths.

**Workspace bootstrap.** One manifest (`client/resources/workspace.json`)
names every seed file, directory, the marker and the gitignore. Python
`workspace init` stages a complete workspace in a temp directory, copies
every seed file after verifying all are present, writes the marker, the
agent guide (the draft document's body with a hash header) and the
gitignore, optionally initializes Git only when `xcode-select -p` succeeds
so a fresh Mac never gets an unsolicited tools prompt, and publishes
create-only with the same `renamex_np`/`renameat2` primitive the archives
use. The Swift store now reads its seed list, prompt directories, gitignore
and marker from the generated `WorkspaceBootstrapText`, and its Git
initialization gained the same tools check. A parity test creates a
workspace with each client and requires every seed file, `AGENTS.md` and
`.gitignore` to be byte-identical.

**Setup and readiness.** `setup inspect` reports client readiness (a
subprocess import probe), workspace recognition and guide presence,
`authoringReady` as their conjunction, and execution as "not assessed".
`setup start <dir> --create` creates then inspects and returns the handoff;
without `--create` a missing directory refuses with the creation command.
`setup plan/apply/repair` shell out to the release installer; on the Mac the
release is the bundled one (or `STEERLAB_CLIENT_RELEASE` for developers),
and its `source.sha256` must equal the compiled constant before any plan is
shown. The Mac's Research Setup sheet opens once when authoring is not ready,
stays in the Workspace menu, creates a workspace with compute undeclared,
shows the plan's destination and actions, requires the approve button, and
copies the handoff. Setup logs go to a mode-0700 directory under Application
Support. The Python `-m` entry now registers its canonical module name so
family adapters share one refusal class.

**Handoffs.** Both clients emit the actual executable and two discovery
commands with the right workspace flag (`--workspace` on the Mac,
`--root` in Python). The Mac helper path inside the app bundle is used when
running from the app.

## 4. Findings

**F1 — no CHANGELOG entry.** Six commits and a researcher-visible feature
(first-run screen, installer, two new CLI families on each client, Python
workspace creation) with no `[Unreleased]` bullet. Third slice in a row. I
will write it at landing.

**N1 — the app build now needs `uv` on the build machine.** `build-app.sh`
calls the release builder without `--uv`, so `uv` must be on `PATH`; it is
not on this Mac today (I used a scratch venv). Before the next app rebuild,
install uv 0.12.5 somewhere on the build `PATH` or the build dies at the
"could not build the lightweight client release" line.

**N2 — development builds refuse the setup plan unless told where the release is.**
`ClientSetup.releaseDirectory` defaults to `ServerPayload/client-release`,
which a checkout-backed build resolves to a directory that does not exist.
The refusal names the fix (`STEERLAB_CLIENT_RELEASE`), and the sheet will
not auto-open on a machine whose checkout venv already satisfies readiness,
so this is a developer-only nuisance. Worth one line in the runtime doc's
development section; the handoff already mentions it.

**N3 — managed environments accumulate.** Each install or repair leaves a
full venv plus managed Python under the runtime's parent, by design so a
failed activation never removes a working one. Nothing prunes them and the
first-run doc says so. A later "remove environments older than the active
one" verb with the same plan-and-approve shape would be the tidy answer.

**N4 — the plan's Python version is a literal.** `"python":"3.12.14"` in the
plan output is typed by hand beside `uv venv --python 3.12.14`; the two
agree today and a test pins the installed version, but a future bump must
touch both.

**N5 — the cold-start contract changed meaning.** `AGENTS.md` at the
repository root used to say the Python client cannot mint a workspace; it
now says either client can, and the release README and both docs agree.
The remaining asymmetry is stated correctly: no Mac lifecycle verbs under
`steerlab`.

## 5. Deploy consequences

The identity constant changed again (resources are now hashed), so the app
rebuild that is already owed must come from a tree at or after this
landing, with `uv` available (N1). Nothing changes for the cluster engine or
the site environment. Fresh-Mac GUI qualification, offline failure and
upgrade paths remain unproven, exactly as the handoff says; my scratch run
proves the installer and the app-free path on a developer machine, not that
experience.
