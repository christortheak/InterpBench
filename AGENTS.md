# AGENTS.md — cold start, from a bare clone

You are a coding agent and a person has pointed you at this checkout, possibly
with nothing else installed. This file is the contract for getting them from
here to a working SteerLab and a first workspace. Once a workspace exists, its
own `AGENTS.md` (written by `steerlab-cli workspace init`) takes over for study
work — this file is only the bootstrap.

**Two products, two names — never type one under the other.**
`steerlab-cli` is the Mac (Swift/MLX) instrument: workspace bootstrap, the full
lifecycle (extract/validate/sweep/promote/freeze/run/analyze), the `cluster`
family, and the app's companion. `steerlab` is the cross-platform Python
client: local authoring, the bundle round trip, a managed local runner, and the
composite `run`. The client has no `workspace init` and no extraction verbs; a
Mac-lifecycle verb typed under `steerlab` exits `64`. Step 1 picks which one
this machine gets, and each has its own verification banner.

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
  Keychain (`steerlab-cli` prompts when needed) — never in JSON, env files,
  shell history, or anything under `Sites/`. Off the Mac the same rule takes
  the client's shape: a runner token is reached by *path*
  (`--token-file`, `$STEERLAB_RUNNER_TOKEN`), never as an argv value, because
  argv is readable by every process on a shared machine.

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
   ```

   **Do not add a `~/.local/bin/steerlab` alias for it.** That spelling belongs
   to the cross-platform Python client (path 4) — its console script installs
   under exactly that name, and two products answering to one name is the
   confusion this contract exists to prevent. Create the alias only on a
   machine that will never install the client, and even there prefer typing
   `steerlab-cli`; if the client is installed later, drop the alias first.

2. **No app, but Xcode 27**: build and install from this checkout —
   `./scripts/install-cli.sh` (set `DEVELOPER_DIR` first if `xcode-select`
   points at an older Xcode). It writes the shim as
   `~/.local/bin/steerlab-cli` and writes **no** `steerlab` alias — that name
   is the client's, and an installer cannot know whether the client arrives
   next week. `--short-name` takes the name anyway, for a machine that will
   never install the client (and even then an existing file the installer did
   not write is left alone). An install from before this policy may have left
   a Swift-CLI shim at `~/.local/bin/steerlab`; if this machine also gets the
   client, remove it so the name has one owner:

   ```sh
   rm ~/.local/bin/steerlab   # only if it is the old Swift-CLI shim
   ```

3. **Neither, on a Mac**: ask the person whether to download SteerLab.app
   from the repository's Releases page — that is the no-Xcode path.

4. **Not a Mac** (Linux/Windows): use the cross-platform Python client.
   From this checkout, `pip install -e "Server"` installs `steerlab` — a
   ~30 MB, no-GPU client that authors, verifies, freezes, packages, and
   drives a runner (`steerlab run <exp> --runner <url>` is the whole
   round trip; evidence comes home verified). Add the `[runner]` extra to
   also EXECUTE locally via `steerlab runner serve` (a managed loopback
   runner with its own root — never the workspace) — **macOS and Linux
   only. Windows is client-only**: authoring, freezing, packaging and
   remote submission all work there, and `runner serve` refuses, so point
   a Windows machine at a runner elsewhere. The client has no
   `workspace init` and none of the Mac lifecycle verbs (extract, validate,
   sweep, promote, analyze); bring a Mac-created or shared workspace, or
   author into a plain directory. The full contract is
   `docs/PORTABILITY-CONTRACTS.md`, and `docs/CLI-REFERENCE.md` §1.4 is the
   verb-by-verb reference.

The paths are ordered, not exclusive: a Mac that took path 1 or 2 may also
install the client from path 4 — they are separate products and coexist
happily, provided the name `steerlab` is left to the client alone.

**Verify before proceeding — the check depends on which product you installed,
and the two answers look nothing alike.**

*Paths 1–2, the Swift CLI:*

```sh
steerlab-cli --version      # must report 6/6 resource families resolved
```

Fewer than 6/6 means an incomplete install: stop and show the person the
output.

*Path 4, the Python client:* it does not have resource families and will never
print that line. Two checks instead:

```sh
steerlab --version                     # -> steerlab <version> (client), e.g. 0.9.1
steerlab experiment list --root <any-directory> --json
```

The banner must end in `(client)`; a version line that says anything else means
`steerlab` on this PATH is not the client (most often a leftover symlink to the
Swift CLI — remove it). The second command is the authoring smoke test: against
an empty directory it answers a well-formed envelope — `"state": "ready"`,
`"verb": "experiment list"`, `result.count` 0 — and exits `0`. Anything else,
stop and show the person the output.

## Step 2 — home layout (Swift CLI only)

```sh
steerlab-cli init            # or: init --home /path/to/SteerLab
```

Creates (and never overwrites) the `SteerLab/` home: `Workspaces/` for
studies, `Sites/` for the private cluster-site registry, with the app and this
checkout as siblings. Re-runnable; it reports what already existed.

On a path-4 machine there is no `init` and no `workspace init`: skip to the
client's own path — author into a plain directory (`--root <dir>` or
`$STEERLAB_WORKSPACE`), or clone a workspace someone created on a Mac. Any
folder layout you like; nothing there depends on the `~/SteerLab/` home.

## Step 3 — Python engine (only when GPU-side or parity work needs it)

```sh
python3.12 -m venv Server/.venv.nosync
Server/.venv.nosync/bin/pip install -e "Server[all]"
Server/.venv.nosync/bin/python -m steerlab_server.cli serve --root <workspace>
```

Serve with an explicit `--root`; the artifact root must be the workspace, not
`Server/`. The server binds loopback by default — read `SECURITY.md` before
changing that.

## Step 4 — first workspace, then hand off (Swift CLI only)

```sh
steerlab-cli workspace init ~/SteerLab/Workspaces/<study-name>
export STEERLAB_WORKSPACE=~/SteerLab/Workspaces/<study-name>
```

The new workspace contains its own `AGENTS.md`. **Read it and follow it from
here** — the study lifecycle (create → attach → extract → validate → sweep →
promote → freeze → run → analyze) is documented there and in
`docs/CLI-REFERENCE.md`.

Path 4 has no equivalent: the client cannot mint a workspace. Point it at a
workspace that already exists (`export STEERLAB_WORKSPACE=…`, or `--root` per
invocation) and follow `docs/CLI-REFERENCE.md` §1.4 for what it can do to one —
author and declare, `verify`, `freeze`, `bundle package`, then `run <experiment>
--runner <url>` to have an engine execute it and bring the evidence home.

## Step 5 — cluster sites (Swift CLI only, and only when the person has one)

The `cluster` family is `steerlab-cli`'s; the Python client reaches remote
compute the other way, through `--runner <url>`. Site profiles live in
`~/SteerLab/Sites/cluster-sites/` — one JSON file per site, read by the app and
`steerlab-cli` alike. If the person keeps a private Sites repository, clone it
into `~/SteerLab/Sites`. Never invent or guess a site profile; ask for theirs. Credentials are prompted into the Keychain on first
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
