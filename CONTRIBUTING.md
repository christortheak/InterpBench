# Contributing to SteerLab

Thank you for helping. SteerLab is a small research project. This page says
how to report a problem, how to propose a change, and how to run the tests.

## Reporting a problem

Open an issue on the
[issue tracker](https://github.com/christortheak/InterpBench/issues/new/choose)
and choose **Report a problem**. The form asks for:

- **Which part you were using**: the Mac app, the Mac command line
  (`steerlab-cli`), the app-free client (`steerlab`), or the Python engine on
  another machine.
- **The version**:
  - the Mac app: **About SteerLab** in the SteerLab menu;
  - `steerlab-cli`: the first line of `steerlab-cli --version` only, which
    begins `steerlab-cli —`. The lines below it show folders on your
    computer, so leave them out;
  - the app-free client: `steerlab --version`;
  - the Python engine on another machine: the `Version:` line of
    `pip show steerlab-server`.
- **Your system**: for example "macOS 26.4 on an Apple M3", or the Linux
  distribution and its version.
- **Where studies run**, if it matters: This Mac, quick start; This Mac, full
  capabilities; or Another machine.
- **What happened**, in your own words: what you did, what you expected, and
  what you saw instead. If a command refused, its `error.code` and its
  `repairAction` help most. Replace any folder path with `<folder>`.

**Never include**, in an issue or a comment:

- a token, a password, an API key, or the contents of a token file;
- a log or command output that shows folder paths, user names, or machine
  names, unless you have replaced them with `<folder>`, `<user>`, and
  `<machine>`;
- study data: prompts, concept texts, generated responses, transcripts, judge
  outputs, results tables, or anything else from a workspace you have not
  decided to make public;
- a cluster site profile, a server address, or a scheduler job number.

Issues are public. If more detail is needed, the maintainer will ask for it,
and you decide what to share.

**Security vulnerabilities** are reported privately, never in an issue. See
[SECURITY.md](SECURITY.md).

## Proposing a change

For anything larger than a small fix, open an issue first and choose
**Propose a change**. Say what problem the change solves and who it helps, so
that the approach can be agreed before you write it. A small fix, such as a
typo, can go straight to a pull request.

A pull request should:

- **Do one thing**, in small commits whose messages follow the existing
  style, `area: what changed`.
- **Test every behavior change** with a test that fails without the change.
- **Keep the integrity contracts.** Frozen studies and `runs/` directories are
  never modified. Saved formats change only by adding to them, and older
  files keep loading. Pin verification, freeze gates, and evidence custody
  are not weakened. A change to a scientific computation comes with an
  independent numerical test whose expected values were worked out by hand.
- **Prefer guidance to barriers.** When something must be refused, say what is
  wrong in plain words and give a repair that the person can carry out with
  the client they are using.
- **Keep personal information out of the repository**: no home-folder paths,
  user names, host names, site names, or study data. `python3
  scripts/ci/public_scan.py` must report that the scan is clean.
- **Regenerate generated files rather than editing them by hand.** `python3
  scripts/ci/check-generated.py` names any that are out of date.
- **Update the documentation** a researcher reads. Write it in plain words,
  with the Oxford comma; call reusable study settings "templates", a
  configured model under study an "agent", and the AI tool that helps the
  researcher a "coding assistant".

Read [AGENTS.md](AGENTS.md) to set up a checkout, and
[docs/ADDING-A-TECHNIQUE.md](docs/ADDING-A-TECHNIQUE.md) before adding a
method.

## Running the tests

SteerLab has two test suites: the Swift suite for the Mac app and
`steerlab-cli`, and the Python suite for the engine and the `steerlab`
client.

**Swift** needs a Mac with Xcode 27. Keep build output outside any folder
that iCloud Drive or another service syncs, because code signing fails on
the files a sync service adds.

```sh
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer   # if xcode-select points at an older Xcode

# a compile check, and one pure-CPU suite
swift build --scratch-path <scratch>/spm
swift test --scratch-path <scratch>/spm --no-parallel --filter <SuiteName>

# the whole suite, including everything that uses the GPU
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -derivedDataPath <scratch>/DerivedData CLANG_COVERAGE_MAPPING=NO
```

Run the suite serially, as above: parallel testing is not supported.
`Failed to load the default metallib` from plain `swift test` is expected for
GPU tests and is not a failure. [ONBOARDING.md](docs/ONBOARDING.md) §11 has
the details, including the Metal toolchain.

**Python** needs Python 3.12. Install the engine from the committed
dependency lock for your platform, then run the suite from `Server/`:

```sh
python3.12 -m venv Server/.venv.nosync
Server/.venv.nosync/bin/pip install -r Server/requirements-macos-arm64.lock   # or requirements-linux-x86_64.lock
Server/.venv.nosync/bin/pip install -e "Server[all]"
cd Server && HF_HUB_OFFLINE=1 .venv.nosync/bin/python -m pytest -q
```

Before you open a pull request, also run:

```sh
python3 scripts/ci/check-generated.py
python3 scripts/ci/public_scan.py
```
