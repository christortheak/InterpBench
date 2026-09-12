# Review: `codex/probe-artifact-contracts` @ `77d7fe6` (on main `22f08a9`)

Reviewer: the maintainer's integration agent, 2026-09-12. Read against
`docs/PROBES-AND-INTERVENTIONS-PHASE-1-PLAN.md` (the branch's authorized
revision), `docs/PROBE-ARTIFACT-CONTRACT.md`,
`docs/PROBE-TRAINING-AND-EVALUATION.md`, and the two handoffs. Five commits
(P0 contracts, P1 library, P1 record, P2 capture/train/evaluate, handoff),
56 files, +4,680/−195. Main has not moved, so a fast-forward is available.
This is the first slice of the probes-and-interventions program, so the
review reads the four scientific owners line by line.

## 1. Verdict

**Landable by fast-forward with no landing fix.** The slice does exactly
what the plan authorized and nothing beyond it: a portable binary-probe
document with a CPU reference scorer, a read-only library shared by every
surface, and three managed operations that capture activations at one
declared residual site, fit one of three small classifiers on CPU, and
evaluate a pinned probe on a pinned dataset in a separate immutable report.
The numerical recipes are simple, stated exactly in the contract, and
implemented as stated; I checked the objective, the gradient, the
standardization, the mean-difference construction, the shuffled-label
control, and the tie-aware AUC by hand against the code. Data-role
discipline is enforced where it can be enforced mechanically (roles never
cross a group or identical text; a final-test capture cannot be fitted or
used for selection; fitting and selection may not overlap) and reported
honestly where it cannot (known overlap versus not established). Nothing
existing changed: the legacy readers keep their owners and bytes, the
audits still protect every prior owner, and the one mechanical extraction
has its own audit with mutation controls.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`22f08a9`) is an ancestor of `77d7fe6`; `77d7fe6` touches only documentation |
| Unified gates and audits (`check-generated.py --audits`) | every generator matches; all fourteen audits pass, including the new `audit-probe-contract-extraction.py` and its two mutation controls; the registration audit strips exactly the new probe preflight line |
| Public scan, whitespace, vocabulary in the diff and all five commit messages | clean |
| Objective and gradient | mean binary cross-entropy plus `l2/2·ΣW²` with biases unpenalized; `σ = exp(−logaddexp(0,−s))` is the stable sigmoid; `δ = (σ−y)/n`; linear gradient `δᵀh + l2·W`; MLP back-propagates `(δ·W₂)⊙[h>0]`; matches the contract and the central-difference fixture |
| Standardization | fitting population mean and `ddof=0` standard deviation, constant features scale 1, computed before and independent of selection data; saved in the probe |
| Mean difference | unit `(μ₊−μ₋)` in standardized coordinates with bias `−(μ₊+μ₋)·w/2`, so the threshold 0 is the midpoint; coincident means refuse |
| Shuffled-label control | permutes the fitting labels with the local PCG64 stream before initialization; reported metrics use the original labels and the report says so |
| AUC | sorted tie groups, `wins += pos·(negBelow + ½·negTied)`, divided by `P·N`; undefined denominators return null rather than 0 |
| Role discipline | `text_rows` refuses a group or identical text in two roles; `probe-train` refuses fitting data captured as selection or final test and any fit/selection overlap by id, group, or source hash; `probe-evaluate` reports `knownOverlap`, `noKnownOverlap`, or `notEstablished` and never blocks |
| Capture semantics | hooks on the explicit decoder-block path only (no size-based guessing), pre or post, read the actual tensor, never replace it, are removed in `finally`, and must fire exactly once per pass; positions come from the attention mask; the binding must be identical across rows |
| Immutability | new UUID run directories, `open('xb')` for every artifact, `COMPLETED` written last; no existing run or probe is touched |
| Bounds | 64 MiB per JSON read and written, `maxRecords`, `maxExamples`, `maxSeqLen` all refused past their caps, never silently truncated |
| Packaging and placement | the three operations are catalog entries with `compute` gpu (capture) and cpu (train, evaluate); file references use the same `{path, sha256}` shape the managed input planner already packages, and the managed round-trip fixture runs a real CPU fit and evaluation in a relocated staged workspace |
| Legacy UI | the Playground reader controls move to `LegacyProbeTrainingView` calling the unchanged `ConceptBuilder` methods; probe example browsing and row removal remain in Data |
| Full Python suite on the tip (`HF_HUB_OFFLINE=1`) | see §5 |
| Full serial Xcode beta suite on the tip | see §5 |

## 3. What the branch does

**Artifact contract** (`probe_artifacts.py`). `activation-probe` v1: input
binding (model, revision, substrate, decoder-path coordinate convention,
precision, width, site, reading population, tokenizer and template hashes),
standardization parameters, one or two affine layers with a single output,
labels, a fixed threshold and a score kind that is never a probability, and
training provenance with data roles. A scalar reference scorer makes the
arithmetic inspectable; `validate_input` is the extraction the audit guards.

**Library** (`probe_library.py`, `ProbeLibrary.swift`, `science probe-list
/ probe-inspect`, `/api/science/workspace/probe-*`). Discovers portable
probes and both legacy reader formats under `runs/<run>/`, reports
malformed files as issues, hashes original bytes, and never rewrites.

**Capture** (`probe_capture.py`). Labeled JSONL with explicit or hashed
whole-group roles; one forward pass per example with the model in eval
mode and no cache; raw or single-turn chat rendering; last or every
non-padding position; one `activation-dataset` file per non-empty role plus
a report with token ids and positions.

**Training** (`probe_training.py`). Mean-difference, L2 logistic, or
one-hidden-layer ReLU logistic, full-batch fixed-step gradient descent in
float64 with a seeded local RNG; optional selection data scored separately
and never used for preprocessing or optimization.

**Evaluation** (`probe_evaluation.py`). Vectorized scoring that the fixtures
compare with the scalar reference; confusion counts, accuracy, balanced
accuracy, precision, recall, specificity, F1, tie-aware AUC, and both
constant baselines; independence status from recorded row provenance.

**Surfaces.** Probes section in the app with the three guided requests,
data-author and reviewer prompts, the library list and inspector, and the
legacy readers behind a labelled sheet; the shared method interviews on
both CLIs; the `readers` guide extended; CLI reference and identity
regenerated.

## 4. Findings

**No landing fixes.**

**N1 — the plan document supersedes an untracked draft in the main
checkout.** Main held an untracked
`docs/PROBES-AND-INTERVENTIONS-PHASE-1-PLAN.md` dated before the
authorization; the branch's tracked version records the authorization and
the baseline. The draft is preserved outside the repository and the tracked
version lands.

**N2 — `eachNonPadding` labels every prefix with the example's label.**
Stated in the contract and the capture report; the researcher decides
whether that labelling is meaningful for their construct.

**N3 — a multimodal checkpoint loads its full class.** Capture selects
`AutoModelForImageTextToText` for a `ForConditionalGeneration`
architecture so that the text decoder path resolves; the vision tower is
loaded too. Memory for the capture pilot is bounded by the model, not the
data, and the cost review says so.

**N4 — discovery is one level deep by design.** The library recognizes
`runs/<run>/<name>.probe.json` and legacy `<name>-probe.json` only; a probe
copied elsewhere is inspectable by path but not listed.

**N5 — live acceptance is owed.** A real capture on the 27B or the 4B on
CUDA, a fit and an evaluation collected through the managed round trip,
and the Probes section exercised interactively after the app rebuild.

## 5. Landing shape

Fast-forward to `77d7fe6` after moving the untracked plan draft aside, then
one commit carrying this review. Shipped Python and the compiled identity
change, so the app and its payload are rebuilt together; the cluster push
waits for the running benchmark jobs.

Suite results on `77d7fe6`:

- Python: 6,508 passed, 9 skipped, 8 warnings (matching the handoff's claim).
- Swift: `TEST SUCCEEDED`, 4,925 passed and 5 skipped of 4,930 (290 SteeringKit and 4,640 ExperimentKit).
