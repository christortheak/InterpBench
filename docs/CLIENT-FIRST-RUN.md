# Researcher first run

The client creates and authors a local data workspace without a model or GPU.
Choose the Mac app or the app-free `steerlab` client; both seed the same prompts,
method catalog and AGENTS.md. Execution is a later, explicit choice.

## Mac app

On the first incomplete setup, the app opens Research Setup automatically. It is
always available from the Workspace menu. Create or open a data workspace, review
and approve the lightweight client plan, then copy the agent handoff. The initial
workspace flow leaves compute undeclared so the researcher can choose it when
preparing execution. No server or model is needed for study authoring.

## App-free release

The client release is one archive, `steerlab-client-<version>+<sha>.tar.gz`,
with a checksum file beside it. Extract it anywhere. The folder holds an
installer, a helper, the client wheel, a hashed dependency lock, a README and
an `AGENTS.md` written for a coding agent that has been pointed at the folder:
plan before installing, install only with the approved hash, use the returned
executable, and create the research workspace somewhere else. No repository,
preinstalled Python, administrator privileges or GPU is required. Supported
installer platforms are Apple Silicon macOS and x86_64 Linux with glibc.

1. `sh install-client.sh plan` changes nothing and reports the destination,
   the actions and a plan hash.
2. `sh install-client.sh install --expect <planSHA256> --yes` downloads a
   verified uv and managed CPython 3.12.14, installs the hashed client lock
   from wheels only, verifies imports and source identity, and activates the
   environment. The result names an absolute `steerlab` executable. Use
   `--runtime <absolute-path>` on both calls to choose another destination.
   Network access to GitHub, Astral's Python distribution and PyPI is required.
3. `<executable> setup start <new-workspace> --create --json` creates a
   complete workspace outside the download folder and returns readiness plus
   the agent handoff. Omit `--create` to open an existing workspace.
4. `<executable> workspace handoff --root <workspace> --json` prints the handoff
   again; it names the executable and the discovery commands, and points at
   the workspace's own `AGENTS.md`, which governs study work from then on.

`setup inspect --root <workspace> --json` reports client and workspace readiness
without treating a missing model as a setup failure. Paths stay local to the
authoring workspace; cluster execution receives copies through the managed
submission and verified import operations.

## Repair and upgrades

The installer never replaces an ordinary existing directory. Existing manually
created environments remain usable via an explicit interpreter selection.
Managed upgrades and `repair` require a new plan and approval. Each creates a
new environment, verifies it and atomically changes the public runtime link.
An unsuccessful download or verification preserves the prior runtime. Old managed
environments are retained; setup does not delete study data or clean up models.
A concurrent setup refuses until its lock is released. After an interrupted
process, inspect the setup log before removing the empty setup-lock directory.

The installer does not edit shell startup files. Use the returned executable
path, or add its bin directory to PATH. It does not start services, download
models, choose a cluster or supply credentials.

## Building and qualification

Maintainers build a release with `python scripts/build-client-release.py --output
<new-directory>` and uv 0.12.5 available. The builder checks generated resources
and the compiled Python identity before building from a disposable source copy.
The app packager includes this same release inside ServerPayload before signing.
This command builds artifacts only; publishing and signing remain release steps.

`scripts/ci/qualify-client-release.py <release> --repair` exercises installation,
workspace creation, agent handoff, method discovery, source identity and absence
of GPU/server dependencies outside the checkout. The Linux workflow performs this
on an independent runner. A real fresh Mac GUI pass, network failure and update
qualification remain release gates; a developer-machine smoke test cannot prove
that experience by itself.
