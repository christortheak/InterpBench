# Review: `codex/fitting-corpus-preparation` @ `aa7d2ed` (on main `abcb42a`)

Reviewer: the maintainer's integration agent, 2026-09-09. Read against
`docs/FITTING-CORPUS-PREPARATION-HANDOFF.md`, `docs/FITTING-CORPUS-PREPARATION.md`
and `docs/FITTING-CORPUS-PREPARATION-PLAN.md`. Three commits, 36 files,
+1,762/−32. Main has not moved, so a fast-forward is available.

## 1. Verdict

**Landable by fast-forward as it stands, with one decision for the
maintainer (F1) that does not block landing.** The slice gives a researcher a
way to turn existing text (local text, JSONL, CSV or Parquet files, or
explicitly named public Hugging Face dataset files) into a fitting corpus
with a receipt: bounded readers, deterministic seeded or first-N selection,
optional character windows, an optional offline token preview, capture into
workspace scratch, and create-only publication under `prompts/fitting/`. The
same owner sits behind the app sheet, both command lines and the workbench
route, the fitting request can carry the receipt as a pinned input, and the
numerical loop is untouched. It also closes the two follow-up findings from
the previous review (N1 message, N3 device guidance). What the maintainer
must decide is F1: the cross-platform client now depends on transformers,
Hugging Face Hub and PyArrow, which is a change of footprint the project had
documented as deliberate not to make.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`abcb42a`) is an ancestor of `aa7d2ed` |
| Unified gates and audits (`check-generated.py --audits`) | every generator matches and every audit passes, including the fitting-loop audit at `878bca2` |
| Public scan, whitespace, vocabulary in the diff and all three commit messages | clean |
| Client lock | 19 packages added, none of torch, accelerate, peft, sae-lens or datasets; the additions are transformers 5.15.0, tokenizers, huggingface-hub 1.28.0, hf-xet, pyarrow 25.0.1 and their transitive CLI dependencies (typer, rich, click, pygments, fsspec, filelock, regex, tqdm, pyyaml, packaging) |
| Path containment | local sources resolve through `archives.ordinary`; symlinks, traversal and absolute paths refuse; the workbench route still refuses any `workspaceRoot` other than the served root; Hugging Face repository paths pass `archives.parts` |
| Network | only the `huggingface` source kind reaches the network, only on the explicit preview action, with `token=False` on both the metadata call and the download; dataset scripts are never executed; tokenizer preview is `local_files_only=True` |
| Live client pass (worktree Python, scratch workspace) | `science corpus-preview` on 200 JSONL rows with a document column and a real cached Qwen3-0.6B tokenizer (`HF_HUB_OFFLINE=1`): 8 of 200 selected, token review measured (6 rows truncated at 32 tokens), examples and one warning; `science corpus-publish` wrote `corpus.jsonl` and `preparation.json` and returned both fitting inputs; a second publish to the same folder refused |
| Full Python suite on the tip (main venv, cwd = worktree/Server) | see §5 |
| Full serial Xcode beta suite on the tip (`TEST_RUNNER_STEERLAB_TEST_PYTHON` set, external derived data, coverage mapping off) | see §5 |

## 3. What the branch does

**Sources** (`corpus_sources.py`). Text files are one record each, capped
at 8 MiB; JSONL is read line by line with the same cap; CSV goes through the
standard parser with distinct headers required; Parquet is read in
16-row batches restricted to the text and document columns. A `huggingface`
source names a public dataset, a revision and file paths or glob patterns;
the resolved commit is recorded, selection is bounded to 1,000 files and
2 GiB, only the four data formats are accepted, and cached bytes are checked
against the repository's LFS SHA-256 or Git blob id where the repository
supplies them, with the receipt saying which files were verified.

**Sampling** (`corpus_sampling.py`). Strict settings with bounds; a scan
limit that stops reading before decoding the next record; seeded selection
by a hash of seed, file, ordinal and text hash, deduplicating exact text;
first-N selection that keeps duplicates and stops when full; an optional
seeded character window per record; counts of scanned, eligible, missing,
short and duplicate rows; warnings for early stopping, shortfall, windows
and the separate-assessment-corpus rule. The token review measures native
encoding lengths, truncation and too-short rows against the pinned
tokenizer if it is cached, and says so plainly when it is not.

**Preparation** (`corpus_preparation.py`). Preview hashes the inputs before
and after sampling, refuses on change, writes a candidate (corpus, receipt,
preview) under `.steerlab/corpus-preparations/<id>` atomically, and returns
a plan hash over the preview. Publish re-verifies the plan hash, the
preview id, the captured corpus and receipt hashes, and the receipt's
binding to the corpus, then publishes create-only under
`prompts/fitting/<name>`. Later changes to the original source do not
change what is published.

**Fitting request.** `corpusReceipt` is an optional pinned file reference;
preflight and execution both verify it belongs to the pinned corpus, the run
captures it under `inputs/corpus-preparation.json`, and the report records
the reference. It is not part of the checkpoint identity.

**Surfaces.** `science corpus-preview` and `science corpus-publish` on both
clients, the same two actions on the workbench route, a
`FittingCorpusPreparationSheet` in the app opened from the fitting form,
which copies chosen files into `.steerlab/corpus-sources/` off the main
actor before preview and fills the corpus and receipt fields on save. The
J-lens guide gains the workflow and the rule that naming a model or concept
does not authorize downloads or fitting.

**Follow-up findings closed.** N1: the stale-draft refusal names review
information and cached model details, with a regression that changes cache
availability. N3: a device mismatch explains literal comparison and that
`cuda` and `cuda:0` are spellings, with a regression.

## 4. Findings

**F1 — the client's dependency footprint changed, and that is a decision,
not a detail.** `pyproject.toml` moves `transformers` and `huggingface_hub`
from the runner and all extras into the base `dependencies`, adds `pyarrow`,
and rewrites the comment that said not to fold heavy packages back into the
client. The client lock grows by 19 packages; none is torch, but the
"~30 MB, no-GPU client" the root contract advertised is now a larger
install (PyArrow alone is tens of megabytes), and `AGENTS.md` quietly drops
the size claim. The handoff is candid about it and the import graph keeps
plain-text preparation and ordinary CLI discovery free of these packages.
Two alternatives were available: a `[corpus]` extra, or lazy imports with
a named refusal when the package is absent (which the code already does for
the tokenizer). Landing as is does not break anything; the maintainer
should decide whether the client's promise was worth keeping. If it was,
the fix is an extra plus the same lazy imports, and the lock split follows.

**F2 — installed managed environments do not gain the packages until the
setup flow is rerun.** The app on this machine runs the client from
`~/Library/Application Support/SteerLab/.steerlab-client.*/venv`, which
`setup inspect` reports with httpx, numpy and safetensors only. After the
app rebuild, the app's Parquet, Hugging Face and token-preview paths refuse
with an import error until the managed environment is upgraded through the
reviewed setup plan; plain text and JSONL preparation still work. The
handoff says this, but the instrument does not: `setup.inspect` probes only
`numpy`, `safetensors` and `httpx`, so an environment without the three new
packages still reports `clientReady: true`, and the lazy
`from huggingface_hub import …` and `import pyarrow.parquet` in
`corpus_sources.py` raise `ImportError`, which neither the workbench route
nor the client adapter maps to a typed refusal (the tokenizer preview does
catch it). A researcher on an older managed environment therefore meets a
traceback, not a repair. Two small fixes: extend the readiness probe to the
new packages, and turn those two imports into a `CorpusError` that names
the setup upgrade. Recommended before the app's corpus sheet is used on any
machine set up before this landing.

**N1 — an occupied destination refuses with a raw error.** Publishing to an
existing `prompts/fitting/<name>` surfaces `[Errno 17] File exists: <path>`
with the diagnostic-archive repair text about custody. The behaviour is
right (create-only); the message should say the folder exists and to choose
a new name, as artifact import does.

**N2 — a Hugging Face preview downloads into the shared hub cache, not the
workspace.** `hf_hub_download` uses the default cache, so up to 2 GiB per
preview lands wherever `HF_HOME` points on the workbench (the cluster's
shared model cache included), and nothing removes it. The bytes are
verified against repository hashes and the receipt records the commit, so
provenance is fine; disk hygiene is the researcher's.

**N3 — a source change during a Hugging Face preview is caught, a cache
change afterwards is not.** Preview verifies cached bytes against the
repository hash and hashes inputs before and after sampling. Once the
candidate is captured nothing reads the cache again, which is correct: the
published corpus is the captured bytes. Recorded so nobody expects the
receipt to re-verify the cache at publish.

**N4 — the token preview reports the workspace machine's library versions.**
`libraryVersions` and `tokenizerSHA256` in the receipt come from wherever
preview ran (the Mac, here transformers 5.15.1), not from the engine that
will fit (the cluster, 5.14.1 today). The fitting report records its own
runtime, so the two can be compared; the receipt should not be read as the
fitting tokenizer's identity.

**N5 — items the handoff leaves open, restated:** no gzip, dataset scripts,
gated authentication or streaming; character windows are not segmentation;
document ids do not create splits; candidate scratch and copied local
sources accumulate with no cleanup; the live walk (dialog, a real public
download and its receipt, a measured CUDA pilot with continuation and
collection) is ahead, after the release-stage deploy.

## 5. Landing shape

Fast-forward to `aa7d2ed`, then this review on its own commit. The branch
changed shipped Python, the client lock and the compiled identity, so the
app and its Python payload are rebuilt together before the app is used with
this main, and the managed client environment needs its reviewed upgrade
(F2) before the app's Parquet or Hugging Face paths are exercised; the
engine on the cluster stays at its September 3 deploy until the release
stage.

Suite results on `aa7d2ed`:

- Python: 6,358 passed, 9 skipped, 8 warnings (matching the handoff's claim).
- Swift: `TEST SUCCEEDED`, 290 SteeringKit and 4,614 ExperimentKit tests.

## 6. The follow-up, `53c2aba`, and what landed after it

The refactoring agents took this review's landing commit onto the branch
and answered F2 and N1 in one commit (19 files, +536/−31), with the
maintainer's decision on F1 recorded. Read in full.

- **F1, decided.** The complete CPU client stays the default. Parquet,
  public dataset files and offline token previews are ordinary research
  steps and should work after one setup; the client still carries no GPU
  execution package. The guide and the first-run document say so.
- **F2.** A shared `client_dependencies` module runs one isolated import
  probe (`-I`, torch and TensorFlow backends disabled) for four capability
  groups: basic authoring, Parquet, public dataset downloads, token
  preview. `setup.inspect` reports `clientReady` (all four),
  `basicClientReady` (the original three packages), per-capability
  diagnostics, a reason line and the upgrade repair; `setup start` and
  `authoringReady` key on the basic set so an older environment keeps
  authoring. The installer's activation step runs the same probe and
  refuses to replace a working runtime with an incomplete one (test: the
  old runtime and its files survive). The two lazy imports in
  `corpus_sources.py` now raise `CorpusSetupError` with code
  `clientSetupRequired` and the same repair on the CLI (exit 65), the
  workbench route (409, code carried through) and the Mac process adapter;
  a missing tokenizer stays advisory in the preview. Research Setup shows
  "Client update needed for corpus tools" and a "Review Update Plan"
  button when the basic set is present.
- **N1.** An occupied `prompts/fitting/<name>` refuses with
  `corpusDestinationExists`, names the folder, and says to publish the same
  reviewed preview under a new name; a directory created by a competing
  publication mid-way gets the same refusal and is left untouched.
- **N2 – N4.** Documented rather than changed: the hub cache is shared and
  not cleaned, publication reads only the captured snapshot, and the
  receipt's tokenizer versions are the preparation machine's.

Verified independently on `53c2aba` in a clean worktree: the unified gates
and audits, the public scan, whitespace, vocabulary in the diff and the
commit message. Live, in this machine's un-upgraded managed client
environment (`~/Library/Application Support/SteerLab/.steerlab-client.*/venv`,
the exact F2 case) with the branch source on the path: `setup.inspect`
returns `clientReady false`, `basicClientReady true`, the three corpus
capabilities each with a "No module named …" diagnostic, and the Research
Setup repair; a Parquet preview through the shared adapter raises
`CorpusSetupError` with `clientSetupRequired`. In the full engine venv all
four capabilities report ready. The probe costs under half a second here.

Suite results on `53c2aba`:

- Python: 6,367 passed, 9 skipped, 8 warnings (matching the handoff's claim).
- Swift: `TEST SUCCEEDED`, 290 SteeringKit and 4,614 ExperimentKit tests.

The branch changed shipped Python, the installer helper and the compiled
identity, so the app and its Python payload are rebuilt together; the
managed client environment on each machine still needs its reviewed update
through Research Setup before the app's Parquet, dataset and token-preview
paths are used.
