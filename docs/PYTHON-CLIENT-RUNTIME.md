# Mac scientific workspace actions: Python client runtime

The Mac app and `steerlab-cli` use the shared Python authoring and archive owners
for method interviews, request publication, SAE authoring, input packaging and
local evidence custody. These operations need Python and the lightweight client
dependencies. They do not need a running model server or the GPU dependencies.

## Source and environment selection

A release build reads Python source from its own immutable `ServerPayload`
resource. A development build uses the existing `CodeResources` selection rules.
The interpreter is selected in this order:

1. `STEERLAB_CLIENT_PYTHON`, if set to an absolute interpreter path. An invalid
   explicit choice refuses rather than falling back to another environment.
2. `~/Library/Application Support/SteerLab/client-runtime/bin/python`, when
   installed and executable.
3. The existing checkout's `Server/.venv.nosync/bin/python`, for development and
   installations that already use the full local engine.

The client starts with the selected source as its sole `PYTHONPATH`, disables
user site packages and bytecode writes, clears `PYTHONHOME`, and uses a temporary
working directory. It does not create a venv inside, or write caches into, the
signed app. A checkout is not required for the first two interpreter paths.
The local model server and stimulus-screen runtime are separate; this change
does not migrate those features.

## First client-only setup from an installed app

Use Python 3.10 or newer. The example below is for a new client environment;
choose the actual installed app path first. Installation may download the
client dependencies (`numpy`, `safetensors`, `httpx` and their dependencies).
Do not install the `[runner]` or `[all]` extras for authoring alone.

```sh
client_app="/Applications/SteerLab.app"
client_env="$HOME/Library/Application Support/SteerLab/client-runtime"
# Stop if this already exists; inspect or deliberately update it separately.
test ! -e "$client_env" || exit 1
client_stage="$(mktemp -d /private/tmp/steerlab-client-install.XXXXXX)"
cp -R "$client_app/Contents/Resources/ServerPayload" "$client_stage/Server"
python3 -m venv "$client_env"
"$client_env/bin/python" -m pip install "$client_stage/Server"
"$client_env/bin/python" -m steerlab_server.client_cli --version
```

Copying the payload to scratch keeps pip's build metadata out of the signed app.
Retain the installation output if setup fails; rerun setup only after inspecting
the partial environment. Once installation succeeds, the temporary source copy
may be removed. The app discovers the default environment without a shell
variable. For a custom location, make `STEERLAB_CLIENT_PYTHON` available to the
process launching the app or CLI; setting it in an unrelated shell does not
configure an already running GUI app.

## Compatibility and release procedure

The compiled Mac client carries a SHA-256 identity of all shipped Python source
files and packaged workspace seeds. The Python process recomputes it before
workspace dispatch. Missing or different identities refuse with an update
repair before the requested action runs. A release number or Git branch name
alone does not establish compatibility. Dependency versions are not part of
this source identity; their supported installation and qualification remain
separate checks.

After Python or seed changes:

1. Regenerate shared resources first, then run
   `python3 scripts/ci/check-python-client-identity.py --write`.
2. Build the Mac app and CLI and stage their payload from those same sources.
   Run the identity gate, both full suites and the app build before review.
3. Deploy the matching app/CLI and payload together; update the client
   environment if dependency requirements changed. Development users still
   using a checkout must also update that checkout before rebuilding.
4. Verify a method interview and an evidence import in a disposable workspace.
   A typo in the import root must refuse without creating it.

Tests exercise an extracted release payload with no checkout or venv inside it,
through both Python and the Mac adapter, and reject modified source before
operation dispatch. This is not a clean-machine installer or interactive GUI
qualification. Before wider launch, qualify the setup above on a Mac without a
checkout, including missing dependencies, offline failure, app upgrades and
repairs. Automatic one-click client provisioning is not implemented here.
