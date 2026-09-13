# J-lens dimension-batch assessment: results and handoff

- Date: 2026-09-13 (work of the night of 2026-09-12)
- From: the maintainer's integration agent, on main `48c4443`
- For: the refactor agents and the study record
- Follows: `docs/JLENS-DIMBATCH-BENCHMARK-RESULTS-AND-HANDOFF-2026-09-12.md`
  (§4 asked for this measurement; §5a summarizes it) and the agents' review
  of that document, whose points this experiment was designed to answer.

## 1. Summary

The benchmark round left one open question: batched and unbatched bfloat16
fits of the 27B differ by 5–7% in early-layer matrix norm, and a matrix
norm cannot say whether that changes anything a researcher reads. So the
pilot's eight rows were fitted a second time at dimension batch 4, the
result was registered beside the existing batch-1 eight-row lens, and the
two lenses were compared at readout level with `jlens-fit-assess` on
sixteen held-out windows of the study's own case text.

**Result: at readout level the two lenses are indistinguishable.** The
largest Jensen–Shannon divergence between their readouts at any of the 63
layers is 0.003 (median 0.0001); their top-10 token sets coincide at 89%
of positions at layer 0 and at 100% from layer 48 on; and each lens's
agreement with the model's final residual is identical to three decimals
at every layer. The 5% matrix gap is real and precision-sensitive, as the
benchmark found, and it does not change what the lens says on this text.

**Decision input:** fit the funded round at dimension batch 4. The
qualification record of any lens fitted this way should carry the matrix
sensitivity from the benchmark and this readout result, with their
conditions.

**Also established live by this run:** the assessment-reuse landing (16
rows, 126 lens-layer reads, 671 MB of staged activations, five minutes on
an H100), the per-request GPU type end to end on a fit and an assessment,
and a transport defect in the managed lens closure that this work found
and fixed (§5).

## 2. What ran

| Step | Request | Job | Hardware | Time |
|---|---|---|---|---|
| Fit, eight rows, batch 4 | `requests/jlens-fit-27b-8rows-dimbatch4` | `d4c8f143c6a0` | H100 80 GB | 1 h 29 min (rows 637–646 s each) |
| Register | `runs/jlens-fit-afeb6330…` → `custom-lens-3d3fe62f…` | | | |
| Held-out corpus | `prompts/fitting/jlens-heldout-ladder-12week` | | | |
| Assess | `requests/jlens-assess-27b-batch1-vs-batch4` | `d0e93f8b207e` | H100 80 GB | 5 min 24 s |

**Fit.** Same corpus (the pilot's neutral maintenance prose), same eight
rows, same `maxSeqLen 128`, `skipFirst 16`, bfloat16, kernel policy
`current`, checkpoint every two rows; only `dimBatch` differs (4 against
1). The reference lens is the batch-1 eight-row lens registered on
2026-09-11 (`custom-lens-85078a58…`, itself the pilot plus its
continuation). Both lenses carry full provenance (reference commit, kernel
and driver hashes, fit report hash).

The running-mean-change statistic of the batch-4 fit tracked the batch-1
fit row for row: 0.478 / 0.363 / 0.306 / 0.257 on rows 4–7 against
0.473 / 0.370 / 0.311 / 0.254. This is the estimator's convergence, not
matrix agreement, and it says the batched path converges the same way.

**Held-out corpus.** The study's ladder stimuli file holds two records of
about 20,600 characters sharing a 20,074-character prefix (the fictional
case text); they differ only in the trailing instruction-order
manipulation. Used directly it yields two near-identical passages, so
sixteen non-overlapping 700-character windows were cut at even spacing
across the shared prefix and prepared with `corpus-preview` /
`corpus-publish`. The derived file and a README recording the source
hash, prefix length, stride, and the absence of instruction text sit in
`prompts/fitting/sources/`. This is a different source from the fitting
corpus; it is the study's own stimulus, not an independent sample of
language, and the preparation receipt says so.

**Assessment settings.** 16 rows, first 64 eligible positions per row
after skipping 16, top-10 overlap, bfloat16, reference = batch 1,
candidate = batch 4. Held-out status recorded as "researcher declared;
overlap not established" (correct: the tool cannot prove independence).

## 3. Results

| Comparison | JS divergence over 63 layers, min / median / max | Top-10 overlap, layer 0 → 62 |
|---|---|---|
| batch-1 lens vs batch-4 lens | 0.0000 / 0.0001 / 0.0032 | 0.89 → 1.00 (1.00 from layer 48) |
| batch-1 lens vs final residual | 0.141 / 0.662 / 0.689 | 0.00 → 0.60 |
| batch-4 lens vs final residual | 0.141 / 0.662 / 0.689 | 0.00 → 0.60 |

Per layer, between the lenses: JS 0.002 at layers 0–16, below 0.0005
from layer 24 on; top-10 overlap 0.90 at layer 0, 0.93 at 8, 0.95 at 16,
0.98 at 24–32, 0.99 at 40, 1.00 from 48.

Against the final residual, both lenses: JS 0.689 at layer 0 falling to
0.642 at 40, 0.592 at 48, 0.425 at 56, 0.141 at 62; top-10 overlap 0.00
through layer 8, 0.01 at 16–32, 0.08 at 48, 0.25 at 56, 0.60 at 62. The
reference and candidate columns agree at every layer to the third decimal.

## 4. Interpretation

**On batching.** The benchmark measured a 5–7% relative Frobenius
difference between batch-1 and batch-4 matrices at early layers, decaying
with depth, and showed it vanishes in float32. This experiment measures
what that difference does to the readout a researcher would actually use:
essentially nothing. Divergence between the two lenses' next-token
distributions is two to three orders of magnitude below their divergence
from the model's own prediction, and the top-10 sets agree at nine
positions in ten even where the matrices differ most. On this evidence
the batched fit is the same instrument for readout purposes. Limits: one
model, one fitting corpus, one held-out source of sixteen rows, eight
fitted rows. It is the measurement the benchmark review asked for, not a
general theorem, and any lens fitted at batch 4 should cite both numbers.

**On the eight-row lenses themselves.** Both lenses disagree strongly with
the final residual until the last few layers (JS above 0.6 through layer
48, top-10 overlap near zero until layer 40). That is what an eight-row
fit should look like against a reference recipe that stopped near 800
rows: the readouts are far from converged. It bears on the funded round's
row budget, not on batching, and it is why the round is planned at
reference scale.

**On what the readout metrics can and cannot show.** JS divergence and
top-k overlap measure whether two readouts say the same thing about the
next token at these positions. They do not establish causal validity of
either lens, independence of the held-out text from the fitting text
(the tool records that as not established), or adequacy for a particular
research question. The comparison with the final residual is a bound on
how much either readout can be trusted at each layer, not a score.

## 5. A transport defect found and fixed on the way

The first attempt to draft the assessment refused: "The diagnostic
archive is empty or exceeds transport bounds." A registered lens
directory holds the converted 6.6 GB tensor and, under `source/`, a 6.6 GB
provenance copy of the original bytes that execution never reads. The
managed input closure's `lens` role shipped the whole directory, so two
lenses came to 24 GB against the 16 GiB bound.

Landed as main `37279e4` and deployed: the `lens` role now ships
`lens.json`, `import-receipt.json` when present, and the converted tensor
(13.2 GB for this assessment). A test pins the closure. This changes the
managed-input contract for lens roles, so it is recorded here for the
agents; the review of your next branch can cite it. The controller running
the previous build planned the staged bundle identically, because its
directory listing of the staged lens contained exactly the shipped files.

## 6. Live acceptance closed by this work

- `jlens-fit-assess` on the 27B with the two-phase capture and one
  lens-layer pair resident: 16 rows, 126 layer reads, 671 MB staged, five
  minutes on an H100, scratch removed. Numerical equivalence with the
  previous owner was already proven by the byte-pinned fixture; this is
  the runtime half.
- `--gpu-type` on `science-plan` and `science-submit` for a fit and an
  assessment, with the type bound into each plan hash and the requested
  card allocated.
- Registration of a fit produced at batch 4 with complete provenance.
- Hardware-provenance block present in the fit report and the assessment
  execution record (H100 80GB HBM3).

## 7. Work for the agents

Unchanged from the benchmark handoff §6, in the agents' own proposed
order: retained case matrices with pairwise and cross-GPU comparison
(F, first); per-layer relative Frobenius verdict with an explicit
zero-norm rule (F); trimmed failure records (N); guide text with the
measured numbers (N). Two additions from this run:

- **§6.2 can now include a readout-level comparison hook.** Since the
  assessment answers the question the benchmark cannot, the benchmark
  guide should say so and point at `jlens-fit-assess` as the follow-up
  when cases disagree by norm.
- **Held-out corpus preparation for short stimulus files.** The corpus
  tool cuts one seeded window per record. A stimulus file of a few long
  records cannot become a multi-row held-out set without a derived file.
  Consider a `windowsPerRecord` option (non-overlapping, evenly spaced or
  seeded) so the derivation done by hand here is a recorded preparation
  setting rather than a README.

## 8. Lessons for operators

- Do not rebuild the app while a Mac-CLI chain is mid-flight: the stage
  verb refused with "Mac and Python sources differ" when the payload
  changed under it. Finish the chain, then rebuild.
- A 12 GB bundle uploads in about six minutes, stages in about ten (the
  controller re-hashes every file), and plans in another ten (it re-hashes
  again). Budget half an hour from package to submit for two 27B lenses.

## 9. Evidence

| Item | Location in the workspace |
|---|---|
| Batch-4 fit run and report | `runs/jlens-fit-afeb6330…` |
| Registered lenses | `runs/jlens-lenses/custom-lens-85078a58…` (batch 1), `runs/jlens-lenses/custom-lens-3d3fe62f…` (batch 4) |
| Held-out corpus and receipt | `prompts/fitting/jlens-heldout-ladder-12week/` |
| Derived passages and README | `prompts/fitting/sources/ladder-12week-case-passages.*` |
| Assessment run and report | `runs/jlens-assessment-f29facc3…` |
| Requests, plans, submissions | `requests/jlens-fit-27b-8rows-dimbatch4/`, `requests/jlens-assess-27b-batch1-vs-batch4/` |
