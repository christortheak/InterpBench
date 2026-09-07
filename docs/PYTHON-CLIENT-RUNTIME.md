# Client runtime, setup and source identity

The Mac app and CLI call shared Python owners for method interviews, request
publication, SAE authoring, input packaging and evidence custody. These operations
need the lightweight client environment; they do not need a model server.

## First setup

In the app, use **Workspace → Research Setup**. The screen checks readiness,
shows the install plan and lets the researcher approve setup. Missing Python is
handled by the release's installer. The Mac CLI offers the same operations:

```text
steerlab-cli setup inspect --json
steerlab-cli setup plan --json
steerlab-cli setup apply --expect <planSHA256> --yes --json
```

The app-free client exposes `steerlab setup inspect`, `setup plan`, `setup apply`
and `setup repair` with the same preconditions. Before any client is installed,
use the extracted release's `install-client.sh`; see [CLIENT-FIRST-RUN.md](CLIENT-FIRST-RUN.md).
All setup operations are local process/CLI operations. They are deliberately not
HTTP mutations exposed by a remote runner. Readiness can also be inspected through
the existing local diagnostic process adapter (`setup-inspect`).

`plan` performs no downloads or installation. Apply and repair require the exact
plan hash and explicit approval. Downloads and package installation occur in a
new managed environment outside the app and workspace. Source and dependency
checks must pass before activation. Setup logs from the Mac are retained under
`~/Library/Application Support/SteerLab/client-setup-logs`; CLI progress uses stderr.
Models, GPU runtimes, SSH credentials and cluster configuration remain separate.

## Source and environment selection

A release uses the Python source in its own `ServerPayload`. A development build
uses `CodeResources`' checkout selection. Interpreter selection remains:

1. `STEERLAB_CLIENT_PYTHON`, when explicitly set to an absolute path. An invalid
   override refuses; installing the default environment does not override it.
2. `~/Library/Application Support/SteerLab/client-runtime/bin/python`.
3. The checkout's `Server/.venv.nosync/bin/python` for development.

The release includes `ServerPayload/client-release`, built before signing.
Development builds can use `STEERLAB_CLIENT_RELEASE=<release-directory>` after
building a matching artifact with `scripts/build-client-release.py`. The Mac
refuses an installer whose source identity differs from its compiled constant.
No developer checkout is needed by the released app or app-free client.

The app sets the selected source as the only PYTHONPATH, disables user site
packages and bytecode writes, clears PYTHONHOME and uses a temporary cwd. Neither
Python caches nor installer build metadata are written into the signed bundle.
The model-server and stimulus-screen runtimes are separate installations.

## Compatibility and qualification

The Mac's compiled SHA-256 covers shipped Python source and client resources,
including workspace seeds and installer policy. It is verified before shared
workspace actions. A source mismatch in development requires a rebuild. A release
bundles its matching sources; a researcher does not need to match a separate
checkout to it. The installed interpreter supplies dependencies while the Mac
continues to use its own bundled source.

Dependency versions are not a substitute for source identity. The supported
installer uses the committed hashed client lock. Existing manually managed
interpreters are checked for working client imports by `setup inspect`; they are
not rewritten or silently upgraded. Request-specific scientific validation still
happens at review and execution time. `authoringReady` is not model qualification.

After Python/resource changes, regenerate the resource copies and
`scripts/ci/check-python-client-identity.py --write`, then rebuild the Mac targets
and client release together. The app build and release builder enforce the identity
gate. Both full suites, release install/repair smoke tests and independent review
are required before landing. Fresh-machine Mac UI and Linux release qualification
must pass before wider distribution; local developer tests alone do not prove it.
