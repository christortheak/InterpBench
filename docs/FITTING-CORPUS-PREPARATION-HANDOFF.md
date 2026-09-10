# Fitting corpus preparation — audit handoff

Branch: `codex/fitting-corpus-preparation`, based on main `abcb42a`.
Implementation commit: `183528b`. The preceding plan commit is `c842ab8`.
The maintainer's reviewing/integration agents decide landing through the user.
No merge, app installation, engine deployment, model download, research dataset
creation, or cluster fitting was performed.

Read [the researcher guide](FITTING-CORPUS-PREPARATION.md) for the concrete app,
CLI, and API workflows, and [the plan](FITTING-CORPUS-PREPARATION-PLAN.md) for
remaining automation and scientific qualification work.

## What this changes for a researcher

The fitting form now offers **Prepare corpus from existing data**. A researcher
can choose local text, JSONL, CSV, or Parquet files, or explicit public Hugging
Face dataset files; select records reproducibly; inspect examples and sampling
limits; and save a corpus with its provenance. The app fills the corpus and
receipt fields in the fitting request. Choosing a corpus never starts fitting.

There is a WikiText source example, not a prescribed scientific choice. The
researcher still chooses which text population the instrument should represent.
Both command lines and the workbench API use the same preparation owner.
Agents have the instructions in the shipped J-lens guide, including the rule
that naming a model or concept does not authorize data generation, delegation,
data downloads, or fitting.

## Follow-up review N1–N5

- **N1 fixed:** a stale fitting draft now names answers, input bytes, or review
  information, including cached model details. The cost review remains in the
  reviewed hash. A regression test changes cache availability without changing
  answers or data and verifies the fresh-review repair.
- **N2 retained:** torch and transformers versions remain compatibility-bound.
  A dependency change needs measured continuation compatibility before a
  migration rule. Source provenance remains separate from numerical identity.
- **N3 fixed as guidance:** device mismatch explains literal names and that
  `cuda` refers to the runtime's current device, while `cuda:0` names index 0.
  It does not equate the names automatically or assert that spelling alone is
  a numerical difference. A regression covers the repair.
- **N4 recorded:** legacy-driver admission is verified by fixtures and audit;
  no live legacy checkpoint has exercised it.
- **N5 remains:** the requested `Qwen/Qwen3.8-27B` CUDA target is unverified,
  dimension batch remains 1 pending measurement, partial runs and scratch have
  no managed cleanup, and fit → collect → register still needs live acceptance
  after the release-stage deploy. The plan records these and their successors.

## Owners and integrity

`corpus_sources.py` reads source records, including Parquet batches. HF files
are selected by explicit repository paths or patterns, resolved to a repository
commit, and downloaded without stored credentials or remote dataset scripts.
Known repository LFS/blob hashes are checked against cached bytes. If repository
hashes are unavailable, the preview says so and the receipt pins actual file
bytes without claiming independently verified repository origin.

`corpus_sampling.py` owns strict settings, source-record counting, first-record
or seeded selection, optional character windows, and optional offline native
tokenization. Seeded selection deduplicates exact source text. The selected
population is bounded by the declared scan limit; the preview names early
stopping and shortfalls. Tokenizer library versions and the backend hash, when
available, are provenance. Unavailable tokenizers do not prevent preparation.

`corpus_preparation.py` captures a candidate and receipt under
`.steerlab/corpus-preparations/<id>`. Publication verifies captured hashes and
uses atomic create-only directory publication under `prompts/fitting/<name>`.
It publishes the reviewed snapshot, even if an original source later changes;
it never silently resamples. A modified candidate or occupied destination cannot
replace anything. Source hashes are checked before and after sampling.

The optional `corpusReceipt` file reference uses the existing managed-input
closure. Fitting preflight verifies that it belongs to the pinned corpus; the
run captures the receipt under `inputs/corpus-preparation.json`, and the report
records its input reference. The receipt is not part of checkpoint numerical
identity. The numerical fitting loop is unchanged and passes its historical AST
comparison and mutation control. This preparation addition is new behavior,
not a claimed mechanical move.

The shared verbs are `science corpus-preview` and `science corpus-publish`.
`POST /api/science/workspace/{action}` retains its existing workbench authority,
checks the serving root, and calls the same owners. Local source paths cannot
escape the workspace. The Mac source chooser copies explicitly selected files
into workspace scratch in a background task before preparation. The original
files are left alone. Swift HTTP callers do not get arbitrary external reads.

## Packaging consequence

The CPU client now declares PyArrow, Hugging Face Hub, and transformers for
Parquet, public dataset metadata/downloads, and offline tokenizer preview.
These were already present in both engine platform locks. Their new client-lock
pins align with those engine locks; existing client-lock versions are preserved.
No torch, accelerate, PEFT, SAE Lens, or datasets package enters the client lock.
The import graph still keeps ordinary CLI discovery and plain-text preparation
free of GPU packages. Transformers/HF floors are declared once in the client
base; engine extras inherit them.

Rebuild the app and its bundled Python payload together after landing. Existing
managed client environments need the normal reviewed setup/upgrade plan to gain
these CPU dependencies. This branch did not change an installed environment.
A separate temporary environment was created only for validation.

## Verification

- Final Python suite: **6,358 passed, 9 skipped, 8 warnings** (206.20 seconds).
- Full serial Xcode beta suite: **290 SteeringKit + 4,614 ExperimentKit passed**.
  The new test calls the real Python owner from Swift, verifies saved text,
  checks both verb declarations, and proves create-only publication.
- A clean temporary environment installed from the hashed client lock, with
  no torch present, passed Parquet preparation, an offline synthetic-tokenizer
  preview, publication, and fitting preflight. No model weights were used.
- Shared Python tests cover JSONL, text, CSV, Parquet, deterministic selection,
  source mutation, candidate tampering, scan bounds, path containment, HTTP/CLI
  byte parity, mocked HF revision/authentication, altered HF cache bytes,
  truncated/short token counts, and the receipt's capture in a fitting run.
- Unified generated-resource/reference checks, historical AST audits, bridge
  gates, public scan, and whitespace checks pass. The diff was read.

Full suite commands, run serially from this worktree (replace `<python>` with
an existing engine/test environment):

```sh
cd Server
PYTHONPATH=. <python> -m pytest -q
```

```sh
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1 \
TEST_RUNNER_STEERLAB_TEST_PYTHON=<python> \
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -derivedDataPath /private/tmp/interpbench-corpus-build CLANG_COVERAGE_MAPPING=NO
```

## Scope limits and live acceptance

The feature reads selected HF shards in full before scanning records. It does
not implement arbitrary dataset scripts, gated dataset authentication, gzip,
or record-by-record network streaming. Select at most 2 GiB of files per
preparation; output corpus and receipt are bounded to 64 MiB each. CSV uses the
standard parser's field bound. Large text documents should be split into
explicit records rather than treated as one enormous fitting passage.

Character windows are not linguistic segmentation. Document IDs are recorded
but do not create train/assessment splits or certify overlap absence. A separate
assessment corpus, preferably from a distinct source split, remains a research
design choice. Candidate scratch and imported local source copies remain on
disk; this slice has no automatic cleanup policy.

Before release, exercise the dialog interactively, download a researcher-chosen
public dataset through the real network path, inspect its resulting receipt,
and run a measured CUDA pilot followed by continuation and collection. The
mocked HF tests and offline tokenizer fixture do not qualify a real checkpoint,
a large live dataset, or an installed app's complete first-run experience.
