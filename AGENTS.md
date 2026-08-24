# AGENTS.md — cold start, from a bare clone

You are a coding agent and a person has pointed you at this checkout, possibly
with nothing else installed. This file is the contract for getting them from
here to a working SteerLab and a first workspace. Once a workspace exists, its
own `AGENTS.md` (written by `steerlab workspace init`) takes over for study
work — this file is only the bootstrap.

## Ground rules (they outrank speed)

- **Every study-path verb speaks `--json`.** Use it. One envelope on stdout,
  diagnostics on stderr, sorted keys, stable exit codes (`0` ok, `64`
  malformed, `65` refused, `66` not found, `70` failed).
- **Refusals are typed and carry a `repairAction`. Follow the repair.** A gate
  that declines is the instrument working, not an obstacle: never work around
  a refusal by editing state it protects, and use `--force` only when the
  person explicitly accepts what the named gates skip (a forced freeze records
  the skipped gates permanently).
- **Study data lives in a workspace, never in this checkout.** Do not write
  prompts, experiments, or runs here.
- **Frozen artifacts and `runs/` directories are immutable.** Iterating means
  duplicating an experiment, never editing a frozen one.
- **Secrets never go in files.** Tokens and passwords live in the macOS
  Keychain (the CLI prompts when needed) — never in JSON, env files, shell
  history, or anything under `Sites/`.

## Step 0 — see what the machine already has

Check, in order: is this an Apple Silicon Mac on macOS 26.4+? Does
`~/SteerLab/SteerLab.app` (or `/Applications/SteerLab.app`) exist? Is Xcode 27
available (`xcodebuild -version`; `/Applications/Xcode-beta.app` may need
`DEVELOPER_DIR`)? Is `python3.12` available? Report what you found before
installing anything.

## Step 1 — get a command line

Pick the first path that applies:

1. **SteerLab.app is present** (no Xcode needed): the app carries the CLI at
   `SteerLab.app/Contents/Helpers/steerlab-cli`. Symlink it:

   ```sh
   mkdir -p ~/.local/bin
   ln -s <app>/Contents/Helpers/steerlab-cli ~/.local/bin/steerlab-cli
   # `steerlab` now names the cross-platform PYTHON client (step 1.3) — its
   # console script owns that spelling. Alias the Swift CLI to it ONLY if
   # the Python client is not installed; prefer typing `steerlab-cli`.
   ln -s <app>/Contents/Helpers/steerlab-cli ~/.local/bin/steerlab
   ```

2. **No app, but Xcode 27**: build and install from this checkout —
   `./scripts/install-cli.sh` (set `DEVELOPER_DIR` first if `xcode-select`
   points at an older Xcode). It installs a `steerlab` shim under
   `~/.local/bin`.

3. **Neither, on a Mac**: ask the person whether to download SteerLab.app
   from the repository's Releases page — that is the no-Xcode path.

4. **Not a Mac** (Linux/Windows): use the cross-platform Python client.
   From this checkout, `pip install -e "Server"` installs `steerlab` — a
   ~30 MB, no-GPU client that authors, verifies, freezes, packages, and
   drives a runner (`steerlab run <exp> --runner <url>` is the whole
   round trip; evidence comes home verified). Add the `[runner]` extra to
   also EXECUTE locally via `steerlab runner serve` (a managed loopback
   runner with its own root — never the workspace). The client has no
   `workspace init`; bring a Mac-created or shared workspace, or author
   into a plain directory. The full contract is
   `docs/PORTABILITY-CONTRACTS.md`.

Verify before proceeding: `steerlab-cli --version` (or `steerlab --version`)
must report **6/6 resource families resolved**. If it reports fewer, stop and
show the person the output.

## Step 2 — home layout

```sh
steerlab-cli init            # or: init --home /path/to/SteerLab
```

Creates (and never overwrites) the `SteerLab/` home: `Workspaces/` for
studies, `Sites/` for the private cluster-site registry, with the app and this
checkout as siblings. Re-runnable; it reports what already existed.

## Step 3 — Python engine (only when GPU-side or parity work needs it)

```sh
python3.12 -m venv Server/.venv.nosync
Server/.venv.nosync/bin/pip install -e "Server[all]"
Server/.venv.nosync/bin/python -m steerlab_server.cli serve --root <workspace>
```

Serve with an explicit `--root`; the artifact root must be the workspace, not
`Server/`. The server binds loopback by default — read `SECURITY.md` before
changing that.

## Step 4 — first workspace, then hand off

```sh
steerlab-cli workspace init ~/SteerLab/Workspaces/<study-name>
export STEERLAB_WORKSPACE=~/SteerLab/Workspaces/<study-name>
```

The new workspace contains its own `AGENTS.md`. **Read it and follow it from
here** — the study lifecycle (create → attach → extract → validate → sweep →
promote → freeze → run → analyze) is documented there and in
`docs/CLI-REFERENCE.md`.

## Step 5 — cluster sites (only when the person has one)

Site profiles live in `~/SteerLab/Sites/cluster-sites/` — one JSON file per
site, read by the app and the CLI alike. If the person keeps a private Sites
repository, clone it into `~/SteerLab/Sites`. Never invent or guess a site
profile; ask for theirs. Credentials are prompted into the Keychain on first
use, per machine.

## Verifying a checkout

```sh
DEVELOPER_DIR=<Xcode 27> xcodebuild test -skipMacroValidation \
  -scheme SteerLab-Package -destination 'platform=macOS' \
  -parallel-testing-enabled NO          # serial is required, not a preference
cd Server && .venv.nosync/bin/python -m pytest -q
```

## Where the depth is

`docs/ONBOARDING.md` (the full walk from zero, §9 is specifically about you),
`docs/CLI-REFERENCE.md` (every verb, flag, and refusal, generated from the
parser), `docs/CONDUCTING-A-STUDY.md` (what makes a study defensible),
`SECURITY.md` (the server's threat model).
