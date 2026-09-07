# Technique extension and backend parity: implementation plan

2026-09-07. Foundation based on main `293888e`. This is the active refinement of
[the technique brief](ADDING-A-TECHNIQUE-BRIEF-2026-09-07.md) and
[the parity brief](PARITY-PROGRAM-BRIEF-2026-09-07.md), incorporating the
researcher's guidance-first principle. The foundation changed no scientific owner or frozen artifact. Step 3 is now
implemented on the review branch; see its handoff below. The work is ordered;
publication of this plan does not mean the remaining packages are implemented
or measured.

## Product outcome

A researcher should describe a question and receive useful choices, understandable
limitations and a practical route to execution. The app and an agent should drive
the same owners and return equivalent scientific records. This does not require
every numerical backend to implement every operation.

Qualification describes the evidence for a claim. It is not a permission system
for exploration. Offer supported-but-unqualified execution with concise guidance
and recorded provenance. Reserve refusal for a concrete inability to perform the
request, an uninterpretable input, integrity loss, or a result that would falsely
describe the computation. Give a direct repair or alternative execution route.
Preserve existing integrity contracts until a separately reviewed change replaces
them; do not silently bypass refusals or relabel historical results.

## 1. Guide and current capability inventory

Deliver [ADDING-A-TECHNIQUE.md](ADDING-A-TECHNIQUE.md), its disposable worked
example, and [SUBSTRATES.md](SUBSTRATES.md). Catalog rows are joined to explicit
capability profiles and checked for complete coverage. CPU-only operations do not
acquire an MPS qualification requirement just because they run on a Mac.

The inventory distinguishes the user-facing surface, execution owner, numerical
backend and artifact reader. An HTTP endpoint is not proof of MLX implementation;
an implementation is not proof of scientific qualification. Record unknown or
unmeasured cells explicitly. Evidence links name their actual scope.

Acceptance: commands and paths verified against the current tree; a small example
passes actual owner validation/execution; exact operation census and evidence-link
checks pass. A later independent fresh-agent exercise should require no verbal
repair outside the guide and linked documents. Record that exercise separately.

## 2. Scientific acceptance and exact sampling contract

Every new technique declares its mathematics, conventions, input roles, output
identity, limitations and independent numerical fixture. Test the full journey
from interview through owner validation to isolated execution and inspection.
An input parser passing does not validate a scientific claim.

Four distinct parity questions:

| Question | Suitable evidence | What it does not establish |
| --- | --- | --- |
| Artifact compatibility | Shared decode/encode, identity and admission fixtures | Equal activations or trained parameters. |
| Mathematical equivalence | Independent examples, intermediate tensors, defined tolerances | Bitwise repeatability of GPU execution. |
| Repeatability within a backend | Repeated/resumed/interleaved runs under an explicit configuration | Matching sampled text across different RNG implementations. |
| Scientific comparability | Controlled outcome comparisons with uncertainty and declared selection | That CUDA is mathematical ground truth or all models are qualified. |

### Preserve existing Python seed policies

Read `experiment/sampling.py`, `condition_execution.py`, and `multi_agent.py`.
Do not implement the original brief's single simplified seed formula.

| Path | Effective seed today | Swift obligation |
| --- | --- | --- |
| Ordinary sampled record, `samplesPerItem > 1` and positive effective temperature | `derive_seed(experimentHash, conditionName, promptID, sampleIndex)`; stamp `derivedSHA256` | Match each seed and index; resume must not shift subsequent streams. |
| Ordinary record otherwise | Enumerate the manifest's declared seeds; stamp `manifestSeeds` | Preserve literal seed values and indexing; do not hash them into another policy. |
| Multi-agent sampled turn | `derive_seed(experimentHash, "", turnID, replicateIndex)` | Preserve the deliberate shared stream across conditions; include neither condition nor an invented seat field. |
| Greedy turn/record | No RNG draw | Keep existing path-specific record conventions; never stamp a seed as controlling sampling that did not occur. |

`derive_seed` hashes UTF-8 text
`experimentHash|condition|promptID|sampleIndex`, takes the first eight SHA-256
bytes as an unsigned big-endian integer, and reduces modulo `2**63`. Shared
fixtures must cover non-ASCII IDs and large seed values without float conversion.
Do not alter this existing algorithm in a port. The seed and RNG algorithm are
distinct: equal seed numbers on MLX and PyTorch do not promise equal token draws.

Requirements for the seeded sampler:

- Match effective temperature, top-p/top-k/min-p behavior and order in the pinned
  library, preserving the actual generation path and stop/budget semantics.
- Scope RNG state to a record/turn; verify isolation on interleaved work. For
  MPS, test its state explicitly rather than assuming CPU/CUDA forking covers it.
- Compare uninterrupted and resumed records. Test standard, variant/saved-agent,
  multi-agent and reasoning-budget continuations. Avoid resetting the stream in
  the middle of a single record unless that is the declared contract.
- Assert deterministic seed identities and repeatable sampling on controlled
  logits. Do not require arbitrary different seeds to yield different text;
  that can fail legitimately when draws select the same tokens.
- Record repeatability measurements with model/tokenizer revision, quantization,
  dtype, library versions, OS/hardware, prefill step size and context length.
  Ten repeated prompts are an initial diagnostic, not a family-wide certificate.

Where execution works but repeatability is unmeasured or imperfect, show an
advisory and report observed differences. Do not create a model-family allowlist
from a small measurement. Lift the current greedy-only product restriction once
the sampler actually supplies scoped seeds on all measured paths and provenance
is accurate; communicate numerical nondeterminism separately. Cases that cannot
honor the declared seed contract must explain that limitation before submission
and offer an honest supported mode, rather than silently ignoring the seed.

## 3. Complete promised Swift semantics and provenance

**Implemented on the branch, pending independent review.** See
[step 3 handoff](TECHNIQUE-PARITY-STEP3-HANDOFF.md) for the owners, tests and
remaining scope boundaries. The following criteria describe this package.

Land small reviewable slices. Prioritize the reader's `finalTest` separation:
before this step, Swift `RepEReader.Dataset.heldOut` selected every non-train row. Final-test
rows must not select a sign or layer. Add shared role fixtures and show final-test
results separately without requiring that exploratory datasets have such a split.

Then deliver the intervention descriptor/sidecar, extraction stability functions
and native verb, and missing adapter-scale/evidence-role stamps. For each:

1. Confirm what the current owner already implements and what its contract owes.
2. Port arithmetic using shared numerical fixtures with independent expectations;
   port schemas with exact round-trip/identity fixtures. AST equality is suitable
   for a mechanical move within one language, not proof of a numerical port.
3. Keep absence in historical sidecars honest; do not rewrite old runs or infer
   newly introduced evidence fields for them.
4. Offer unavailable native operations through their existing Python path.
5. Update the capability inventory and contract only after tests establish the
   implemented behavior. An implementation result is not hardware qualification.

## 4. Implement and measure seeded MLX sampling

Verify actual APIs in the pinned `Package.resolved` dependencies. Add a scoped
sampler in SteeringKit and thread it through every measured token-iterator path
using section 2's policy. No dependency bump solely to avoid checking the API.
Replace obsolete greedy-only copy at its maintained sources, including the agent
draft and generated resources. Do not edit the generated agent contract directly.

Validation layers: pure seed fixtures; controlled-logit sampler tests; owner-level
record/resume tests; then real-model repeatability under named configurations.
Report exact results and deviations. No cross-backend token-equality claim follows
from this implementation. Both full suites and independent review precede landing.

## 5. Qualify Python execution on MPS against the CUDA reference

Create an explicit matrix before running: extraction recipes, injection and
ablation, readers, a battery, stability diagnostic, OptVec training/evaluation,
J-lens qualification/readout, J-space analysis and the applicable SAE import/
execution paths. Metadata-only import is not GPU qualification. Extend coverage
to training and multi-agent paths before describing the entire engine as qualified.

Use small models that actually exercise each path, matching weights/revision,
tokenizer/rendering, dtype and inputs where supported. Record unsupported matching
precision or quantization as a limitation. Select input/tensor/output tolerances
and expected failure behavior before seeing results; link the existing ruling
where one applies and write a reviewed path-specific tolerance where none does.

Report implementation-level fixtures separately from hardware measurements:
activations, extracted directions, intervention deltas, logits, losses/gradients,
readout scores, memory behavior and final decisions as appropriate. Generated
text can diverge after tiny logit differences; locate the first relevant tensor
difference rather than interpreting text mismatch alone as a broken port.

Deliberately exercise MPS RNG state, eager attention and long-context memory,
precision promotions in J-lens reference agreement, and actual device placement.
Detect CPU fallback instead of counting a CPU run as MPS coverage. An `mps` branch
count in the test source is a search aid, not proof those paths ran on-device.

Save commands, source identity, input hashes, hardware/software configuration,
tolerances, results and evidence locations in a reproducible record. Keep model
weights and large artifacts outside the source checkout. This work uses explicit
compute resources; use the researcher's configured execution route and approvals
where required. Do not invent cluster access or claim a CUDA comparison without
running it. Fix discovered defects in separate commits with regression tests.

## 6. Simplify extension based on the exercised guide

First consolidate regeneration and check discovery without moving audit baselines.
Then use small per-operation declarative specifications for interview, catalog,
input roles and execution bindings. Keep the registry a lightweight aggregator;
do not put all prose and scientific implementation into one large Python module.
Generate shared artifacts with one command and check complete fixture coverage.

Only then consider a developer scaffold, with an explicit checkout destination,
previewable changes and no overwrites. An installed researcher client should not
try to edit its packaged runtime. Generate shared enum vocabularies where they
help, while keeping unsupported native execution explicit and scientific math in
its owners. Audit-base metadata can improve visibility, but baseline updates need
reviewed evidence and must never be an automatic regeneration step.

## Completion and review

Completed on the branch: guide, worked example, inventoried capabilities,
corrected implementation/qualification contract, and step 3's native scientific
semantics/provenance. Step 3 awaits independent review and integration through
the researcher. Remaining packages are **4–6**: scoped seeded MLX sampling,
hardware qualification, and extension/registration simplification. Steps 4 and 6
still require coding; live testing is not the only remaining work. Independent
guide trials and hardware qualification remain explicitly pending.
Mac-cluster scheduling is still a separate product decision, not implied by this
program. Managed runtime cleanup is a separate installation-lifecycle task.

Use interim commits, read the full diff, run both suites and applicable mechanical
audits, and give the researcher a handoff with evidence. The researcher coordinates
the independent coding/audit agent and integration into main. Never change an AST
baseline just to approve an intentional scientific modification: state the change,
test it behaviorally, and obtain review of the replacement scope.
