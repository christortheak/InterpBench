# Releasing SteerLab

This is the procedure for publishing a SteerLab release, as it exists today.
Each step is marked **scripted** (a command in this repository does the
work and checks it) or **manual** (a person runs or decides it). Run the
steps in order: several commands refuse to run until an earlier step is
done, and that refusal is deliberate.

A release publishes three things from one tagged commit: the Mac app
(`SteerLab-<version>+<revision>.zip`), the app-free client
(`steerlab-client-<version>+<revision>.tar.gz`), and a checksum file beside
each. Nothing in this procedure touches a research workspace, a cluster, or
the copy of the app you use day to day.

## What you need

Credentials and secrets. This guide names them; it never holds their values,
and none of them belongs in a file in this repository.

| What | Where it lives | Used by |
|---|---|---|
| A Developer ID Application certificate and its private key | your login keychain | `build-app.sh --identity`, to sign the app |
| Your Apple ID, team ID, and an app-specific password | the keychain, as the notarytool profile `steerlab-notary` (step 6) | `xcrun notarytool` |
| The private-name list | `~/.steerlab/private-names.txt`, or the file `STEERLAB_PRIVATE_NAMES_FILE` names | the release gate, the artifact scans, the signed build |
| The same list, for continuous integration | the repository secret `STEERLAB_PRIVATE_NAMES` | `.github/workflows/` |
| A GitHub account that can push tags and create releases | `gh auth login`, or the web page | step 10 |

Tools, on an Apple Silicon Mac:

- Xcode 27. When `xcode-select` points at another Xcode, export
  `DEVELOPER_DIR=<Xcode 27>/Contents/Developer`, and `TOOLCHAINS=<the Metal
  toolchain identifier>` if your Xcode needs one
  ([docs/ADDING-A-TECHNIQUE.md](docs/ADDING-A-TECHNIQUE.md) shows both).
- A test Python 3.12 environment installed from this checkout:
  `python3.12 -m venv Server/.venv.nosync` and
  `Server/.venv.nosync/bin/pip install -e "Server[all]"`.
- Node.js 22.13 or later, with the results explorer's dependencies
  installed: `npm ci` in `results-explorer/`.
- `uv` on your `PATH` (the client release builder uses it; continuous
  integration pins 0.12.5).
- A clone with full git history. The audits read old commits.

Installing these is your decision; the release gate installs nothing and
stops with the command to run instead. Two later steps do fetch packages:
the app build runs `npm ci` from the results explorer's lockfile, and the
client qualification downloads into disposable scratch.

## 1. Set the version (manual, then scripted checks)

The version is written in three places, which must agree:

- `Server/pyproject.toml` (`version = "…"`)
- `Server/steerlab_server/__init__.py` (`__version__ = "…"`)
- `Sources/ExperimentKit/SteerLabVersion.swift` (`public static let version = "…"`)

Change all three, then regenerate the compiled client identity, which hashes
the Python sources:

```sh
python3 scripts/ci/check-python-client-identity.py --write
```

`Server/tests/test_version_agreement.py` fails if the three disagree, and the
release gate (step 3) runs it. Update any document that states the current
version as a fact. Historical changelog entries keep their versions.

## 2. Write the changelog (manual)

In `CHANGELOG.md`, rename `## [Unreleased]` to `## [<version>] — <date>` and
start a new, empty `## [Unreleased]` above it. Write for researchers: what
changed for them, and anything they must do.

Commit the version and the changelog together, for example
`release: SteerLab <version>`. Do not tag yet.

## 3. Run the release gate (scripted)

```sh
python3 scripts/release-gate.py
```

It runs every check, one at a time, and stops at the first failure with a
summary of what to do: a preflight of the tools above, a build of
`steerlab-cli`, the generated-resource checks with that build, the audits
(`check-generated.py --audits`), the public scan, the technique
qualification, the Python suite, the Swift suite (serially), and the results
explorer's lint, type check, unit tests, and embedded build. Expect the two
suites to take most of the time. `python3 scripts/release-gate.py --list`
prints the plan without running it.

Run it on the committed release candidate. The summary says whether the
checkout was clean. A run with uncommitted changes certifies no commit, and a
run narrowed with `--only` or `--from` (useful while fixing a failure) is not
a gate pass. The artifact-scan stage is reported as still owed here; step 9
completes it.

Build output goes to `/private/var/tmp/steerlab-release-gate` (change it with
`--scratch`). Pass `--python` if your test Python is not
`Server/.venv.nosync/bin/python`.

When the gate passes, tag the commit:

```sh
git tag -a v<version> -m "SteerLab <version>"
```

## 4. Build the app with a Developer ID (scripted)

```sh
scripts/build-app.sh --identity "Developer ID Application: <name> (<team id>)"
```

This compiles from a neutral build root (`/private/var/tmp/steerlab-build`),
so no binary carries your home folder; builds the results explorer and the
client release into the bundle; scans the assembled bundle for private names
and home-folder paths before anything is signed; signs with the hardened
runtime and a secure timestamp; and verifies the signature. The app is written
to `~/SteerLab/build/SteerLab.app` (change it with `--output`). A build signed
for distribution refuses to run without the private-name list.

`--install` would also replace `~/SteerLab/SteerLab.app`. A release does not
need it.

## 5. Package for notarization (scripted)

```sh
scripts/build-app.sh --package
```

This zips the bundle in `--output` (never an installed copy) and writes a
`.sha256` beside it. It refuses an ad-hoc signature, and it refuses unless the
checkout is clean, `HEAD` is tagged `v<version>`, and the bundle was built
from `HEAD`. This first zip is what you submit; it is not the release asset.

## 6. Notarize and staple (manual)

`scripts/build-app.sh --notarize` prints these commands; it runs none of them.

Once per machine, store the notary credentials in the keychain. Leave out
`--password` and notarytool asks for it, so it stays out of your shell
history:

```sh
xcrun notarytool store-credentials "steerlab-notary" \
  --apple-id "<Apple ID>" --team-id "<team id>"
```

Then submit, and staple the ticket into the app (not the zip):

```sh
xcrun notarytool submit "<the zip from step 5>" --keychain-profile "steerlab-notary" --wait
xcrun stapler staple ~/SteerLab/build/SteerLab.app
xcrun stapler validate ~/SteerLab/build/SteerLab.app
spctl --assess --type execute -vvv ~/SteerLab/build/SteerLab.app
```

`spctl` must report a notarized Developer ID. If the submission is rejected,
`xcrun notarytool log <submission id> --keychain-profile "steerlab-notary"`
names each binary at fault. Delete the zip from step 5.

## 7. Package the release asset (scripted)

```sh
scripts/build-app.sh --package
```

Run after stapling, because the ticket lives inside the bundle: only this zip
opens on another Mac without a warning. It writes
`SteerLab-<version>+<revision>.zip` and `SteerLab-<version>+<revision>.zip.sha256`
in `~/SteerLab/build/`.

## 8. Build and qualify the client release (scripted)

```sh
STEERLAB_REQUIRE_PRIVATE_NAMES=1 python3 scripts/build-client-release.py \
  --output <a new directory>/steerlab-client --archive
python3 scripts/ci/qualify-client-release.py <a new directory>/steerlab-client --repair
```

The builder writes the release directory (installer, wheel, lock, license,
notice, `SHA256SUMS`), and beside it
`steerlab-client-<version>+<revision>.tar.gz` with its `.sha256`. It scans
what it built before writing it. The qualification installs that release into
disposable scratch, from the release directory alone, and drives it; it
downloads a managed Python and the locked dependencies, so it needs the
network. Continuous integration runs the same qualification on Linux and on
Apple Silicon for every `v*` tag (`.github/workflows/client-release.yml`).

## 9. Scan the built artifacts (scripted)

```sh
python3 scripts/release-gate.py --only artifact-scan \
  --artifact ~/SteerLab/build/SteerLab-<version>+<revision>.zip \
  --artifact <a new directory>/steerlab-client \
  --artifact <a new directory>/steerlab-client-<version>+<revision>.tar.gz
```

This is the stage step 3 left owed, with the private-name list required. It
reads every byte, including the archives' members and the wheel inside them.
The summary calls this a partial run, which is expected for this step.

## 10. Publish the GitHub release (manual)

Push the release commit and its tag:

```sh
git push origin main v<version>
```

Wait for the CI and client-release workflows on the tag to pass. Then create
the release on the tag, with the changelog section as its notes, and attach
the four files: the app zip and its `.sha256`, the client archive and its
`.sha256`. With the GitHub command line:

```sh
gh release create v<version> --title "SteerLab <version>" --notes-file <notes.md> \
  SteerLab-<version>+<revision>.zip SteerLab-<version>+<revision>.zip.sha256 \
  steerlab-client-<version>+<revision>.tar.gz steerlab-client-<version>+<revision>.tar.gz.sha256
```

or upload them on the Releases page.

## 11. Check the published release (manual)

From the Releases page, preferably on a Mac account that has never had
SteerLab:

- Download each file and verify it: `shasum -a 256 -c <file>.sha256`.
- Unzip the app and open it. It must open without a Gatekeeper warning.
  `SteerLab.app/Contents/Helpers/steerlab-cli --version` must report the new
  version and 6/6 resource families.
- Extract the client archive and follow its `README.md`. `steerlab --version`
  must report `steerlab <version> (client)`.
- An older installed app's update check should offer the new release.

If a check fails, fix it in a new patch release. Never replace the files of a
published release.
