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

- the Swift engine's web front end (`steerlab serve`), and
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
- **Workspace inputs are trusted.** Prompts, rubrics, and parser
  configurations are data you author; they are not sandboxed from the study
  they configure.
- **The engine consumes GPU, disk, and — on a cluster — scheduler quota.**
  Anyone who can drive the API can spend all three.

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
