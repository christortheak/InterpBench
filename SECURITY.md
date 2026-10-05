# Security Policy

## Supported versions

SteerLab is a pre-release research instrument. Only the latest published
revision is supported; there are no maintained release branches and no
backported fixes. If you are running an older checkout, update before
reporting.

## Reporting a vulnerability

Report privately, not in a public issue. Use the repository's private
vulnerability reporting if it is enabled; otherwise contact the maintainer
listed on the repository owner's profile page.

Please include what you ran, what you observed, and — if the finding involves
the HTTP surface — the exact request, the bind address, and whether
`STEERLAB_AUTH_MODE` was set. A short reproduction is worth more than a
severity score.

Expect an acknowledgement within a few working days. This is a small project;
there is no bounty, and disclosure timelines are agreed case by case rather
than promised in advance.

## What this software is, for threat-modeling purposes

SteerLab runs two HTTP servers, and both are **single-researcher instruments**,
not multi-tenant services:

- the Swift engine's web front end (`steerlab-cli serve`), and
- the Python engine's API and browser workbench (`steerlab-server serve`).

Neither has user accounts, roles, or per-user isolation. Anyone who can reach
the API *with a valid token* can act as the researcher who started it. Both
bind loopback by default, and the intended way to reach a remote engine is an
SSH tunnel — not a public bind, not a reverse proxy you added yourself.

Both servers reject browser-originated cross-origin and non-loopback requests
by checking `Origin` and `Host`, which is what prevents a page you visit from
driving a local engine by DNS rebinding. Paths that the API can write are
contained to the configured artifact root; they do not traverse to arbitrary
locations on the host.

### Assumptions you are making by running it

- **In the dev-open tier, the machine's local users are trusted.** On a shared
  node, assume every local user can reach `127.0.0.1` — which is why that tier
  refuses to start next to a Slurm executor, and why the default is token mode.
- **Models are code-adjacent.** Loading a model executes the loader against
  weights and configuration you fetched from elsewhere. Fetch models from
  sources you trust.
- **Workspace inputs are data, with one exception that is code.** Prompts,
  rubrics, parser configurations, panels, agents, and vectors are data: the
  engines read them and do not run them as programs. The exception is an
  intervention policy's **expert provider**, Python source that the Python
  engine runs with `exec`, with your permissions, whenever a study that uses
  it generates text. See [Custom code in a study](#custom-code-in-a-study).
- **The engine consumes GPU, disk, and — on a cluster — scheduler quota.**
  Anyone who can drive the API can spend all three.

## Custom code in a study

An intervention policy decides, while a model generates, when and how strongly
to steer. Most policies are built from fixed rules and are data. A policy can
instead carry an expert provider: a Python function, written by the policy's
author, that the Python engine runs with `exec` during the study. **It is not
sandboxed.** It runs with the permissions of whoever runs the study, on
whichever machine runs it: your Mac, a runner you started, or your account on
the cluster.

The author sees a warning when they publish such a policy. A study shared as
a pack or a bundle carries the code to someone who never saw that warning, so
SteerLab gives notice when the code arrives and asks for an acknowledgement
before it runs:

- **Notice.** When `pack apply`, `bundle import` (on the Python client), or
  `experiment attach-agent` brings a study or agent with an expert provider
  into a workspace, the result says: "This study contains custom code from its
  author. It runs with your permissions when the study runs. Run it only if
  you trust the source." It names each provider by the SHA-256 of its source
  text. The app shows the same notice, with the code and an acknowledge button,
  at the top of the study's page.
- **Reading the code.** `experiment acknowledge-custom-code <study>` (on
  `steerlab` and `steerlab-cli`) prints each provider's source and changes
  nothing.
- **Acknowledgement.** `experiment acknowledge-custom-code <study> --sha256
  <hash>`, or the app's button, records who acknowledged which source hash,
  and when, in `custom-code-acknowledgements.json` at the workspace root. It
  is an ordinary file in the workspace, so it is part of the workspace's
  history. An acknowledgement covers that exact code wherever it appears in
  the workspace; changed code has a different hash and needs a new one.
- **The one gate.** Until each provider a study carries is acknowledged, the
  clients do not send that study to run for a step that executes its agents
  (`run`, `pipeline`, or `sweep`). Steps that run no agent, and dry runs, are
  not held. The refusal names the code and gives the acknowledgement command.
  A run's provenance record, written by `steerlab run`, lists each provider
  and who acknowledged it.

What this is not: the acknowledgement is a record of your decision, not a
safety check on the code. SteerLab does not inspect, restrict, or sandbox
expert providers. The gate is enforced by the clients and the app, where a
person's workspace hands a study to an engine. An engine given a study
directly (`steerlab-server` run against a root, or a hand-submitted bundle)
runs what it is given. Read the code before you acknowledge it, and run
studies only from sources you trust.

## What leaves your machine

SteerLab sends nothing about you, your workspace, or your studies anywhere
unless you ask it to run something on another machine. The one network
request the app makes on its own schedule is the update check:

- **What.** At most once a day, when the app starts, it makes one
  unauthenticated `GET` request to GitHub's public API for the project
  repository's latest release (`api.github.com/repos/<owner>/<repository>/releases/latest`).
  The request carries no body and no identifying header: no version, no
  installation identifier, no workspace or study name. When a code checkout
  sits beside the app, it also runs `git ls-remote` in that checkout, through
  your own git configuration. The comparison with the version you run happens
  on your Mac.
- **What it does with the answer.** It shows a notice that a newer release
  exists, with a link to the Releases page. It never downloads, installs, or
  replaces anything.
- **How to turn it off.** Uncheck **Check for Updates Automatically** in the
  SteerLab menu (or **Check automatically** on the update notice). **Check for
  Updates…** in the same menu still checks once, when you choose it.

Other network requests follow from something you chose to do: downloading or
loading a model from Hugging Face, connecting to a runner or the cluster, or
judging with (or listing the models of) a hosted service whose key you
supplied.

## Authentication

**`steerlab-server serve` requires a bearer token by default, on every
platform.** With no configuration at all it resolves token mode, hydrates
`STEERLAB_AUTH_TOKEN` from `STEERLAB_AUTH_TOKEN_FILE` (default
`~/.steerlab-token`), and creates that file — 256 bits, mode 0600 — when it
does not exist, printing the path and never the value. Authenticate with:

```
curl -H "Authorization: Bearer $(cat ~/.steerlab-token)" http://127.0.0.1:8080/api/info
```

In token mode every `/api/*` route is gated, including read-only listings.

### The open tier is opt-in

`serve --dev-open-loopback` (or `STEERLAB_DEV_OPEN_LOOPBACK=1`) selects the
single-user tier where mutating routes are reachable without a token. It
**refuses to start** — exit 64, with the reason — if the bind is non-loopback
or `STEERLAB_EXECUTOR=slurm` is declared. An explicit `STEERLAB_AUTH_MODE=none`
is refused under the same two conditions. On a personal Mac,
`scripts/start-local-server.sh` (the app's one-click local server) passes the
flag deliberately; the shipped bare CLI does not.

**Why the one-click local engine is open, and why it is loopback only.** The
app and the browser workbench talk to the local engine with no token, so a
researcher can start it with one click and nothing to paste. That is safe only
when the only person who can reach it is you. It therefore binds `127.0.0.1`
and nothing else: no other machine can connect to it. It does not keep out
other accounts on the same Mac. Any local user can drive an open engine, and
driving an engine includes running a study, which can include a study's
custom code ([above](#custom-code-in-a-study)) with your permissions. On a
Mac you share with other accounts, use token mode: remove
`--dev-open-loopback` from the script, and paste the printed token-file path
into the app's connection sheet.

### What "privileged" means

Route classification is **mutating-by-default**: every `POST`/`PUT`/`DELETE`/
`PATCH` under `/api/` is privileged unless it is on a short, reasoned
allowlist of tokenizer-only and parse-only routes. A set of prefixes adds the
sensitive reads (session lifecycle, bundle download) and families that take a
command or a caller-named path. A privileged route requires the token whenever
the server runs a Slurm executor, uses a non-local profile, or binds a
non-loopback address — so on a cluster node every mutating route is gated even
if auth mode were left at `none`. A test walks the running route table on
every CI run, so a newly added mutating route cannot silently be left open.

Binding non-loopback without a token is refused with an explicit message rather
than started quietly.

### Known limitations

- **The posture is resolved by `serve`, not by the app object.** Running the
  ASGI app directly (`uvicorn steerlab_server.api.app:app`, an embedding
  process, a test client) skips that resolution and falls back to the
  environment as given — which defaults to `auth_mode=none`. Start the server
  through `steerlab-server serve`, or set `STEERLAB_AUTH_MODE=token` yourself.
- **No TLS.** Tokens travel in cleartext over the socket. Keep the bind on
  loopback and put an SSH tunnel in front; do not terminate this on a public
  interface.
- **No per-user isolation, no roles.** A valid token is the researcher. The
  token file is per-user (0600), not per-client, and there is no revocation
  beyond replacing it and restarting.
- **The dev-open tier still exists** by design. On loopback it means any local
  user on that machine can drive the engine. On a shared machine, do not use
  it — the refusals above make the dangerous spellings hard to reach, not
  impossible to want.
- **Denial of service is not in scope**: anyone who can drive the API can spend
  the GPU, the disk, and the scheduler quota.

## Secrets

- Bearer tokens, Hugging Face tokens, and SSH credentials are never written to
  run artifacts, manifests, logs, or the CLI's JSON envelopes. The envelope
  types are structured so that no field can hold a credential: secrets appear
  only as presence booleans and provenance labels.
- On macOS the CLI stores credentials in the system keychain, and read-only
  listing verbs report only whether a token exists, without reading it.
  Keychain access is granted per binary identity, so a freshly installed or
  reinstalled binary may prompt once for your password on the first verb that
  actually uses a secret. That prompt is a genuine macOS prompt and only you
  can answer it; an unattended agent will simply wait.
- If you believe a credential was written to a run directory or a log, treat
  that as a vulnerability and report it.

## Out of scope

- Denial of service by a user who already has authorized access to the API.
- Vulnerabilities in model weights, in the datasets you supply, or in
  third-party dependencies — report those upstream, though we want to hear
  about them if we ship an affected pin.
- Findings that require an attacker to already have local shell access as the
  user running the engine.
- The absence of multi-user authorization. That is a design boundary, stated
  above, not a defect.
