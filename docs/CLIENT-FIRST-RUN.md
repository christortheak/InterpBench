# Researcher first run

The client creates and authors a local data workspace without a model or GPU.
Choose the Mac app or the app-free `steerlab` client; both seed the same prompts,
method catalog and AGENTS.md. Execution is a later, explicit choice.

## App-free release

The release directory contains a client wheel, a dependency lock, an installer,
and its checksums. No repository or preinstalled Python is required. Supported
installer platforms are Apple Silicon macOS and x86_64 Linux with glibc.

Run `sh install-client.sh plan` from the extracted release. Review its destination,
actions and plan hash, then run `sh install-client.sh install --expect <hash> --yes`.
An agent can parse both responses as JSON; stdout contains one result and download
progress goes to stderr. Use `--runtime <absolute-path>` on both calls to choose
another destination. Network access to GitHub, Astral's Python distribution and
PyPI is required. The installer verifies uv, downloads managed CPython 3.12.14,
and installs the hashed client lock using wheels only. No compiler is required.

The result names an absolute `steerlab` executable. Use it to run:

```text
steerlab workspace init <new-directory> --json
steerlab workspace handoff --root <directory> --json
```

Give the handoff and the research question to an agent. It reads the workspace's
AGENTS.md, discovers the installed method catalog, and asks about scientific
choices. Paths stay local to the authoring workspace; cluster execution receives
copies through the managed submission and verified import operations.

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
