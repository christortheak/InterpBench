# J4 supplement: the full hybrid 27B lens and the corpus-composition assessments

Date: 2026-09-21
Supplements: [J4 decision of 2026-09-13](JLENS-QUALIFICATION-DECISION-J4-2026-09-13.md), which stays as written.
Companions: [status handoff](JLENS-AND-P7-STATUS-HANDOFF-2026-09-21.md),
[coding agents' memo](JLENS-P7-RUNNING-AGENT-MEMO-2026-09-21.md),
[mixed-corpus merge design](JLENS-MIXED-CORPUS-MERGE-2026-09-17.md).
Evidence lives in the study workspace under
`diagnostics/jlens-full-27b-2026-09-15/` (two interpretation notes, nine
per-layer tables); run and lens identifiers below are workspace records.

## 1. What changed since J4

J4 recorded "Full-lens hybrid 27B agreement: bounded subset only (five
layers, one row); hybrid fits beyond eight rows are unqualified; none is
planned on this evidence." Since then three full-size fits of the hybrid
27B model were executed through the managed round, merge, fetch, custody,
import, and assessment path, with every step's hashes recorded:

| lens | rows | corpus | fitted as | assessed on |
|---|---|---|---|---|
| general | 828 | WikiText-103 train | 8 H100 shards, 18–19 h each, merged | WikiText validation, ladder case windows, case opinions |
| domain-only | 300 | Caselaw Access Project appellate opinions (1970+, bodies only) | 3 H100 shards, 18–21 h each, merged | same three |
| mixed | 1128 | the two above at equal row weight (73.4% / 26.6%) | cross-corpus merge of the two merged runs (no refit) | same three |

Two engine limits found on the way were fixed and reviewed the same day
(in-place input bound; preflight hashing budget), and the cross-corpus merge
was added with an explicit opt-in and per-corpus provenance.

## 2. What the supplement establishes, and what it does not

**Established (execution).** Full-size hybrid 27B fitting, sharded across
cards of one type, merged deterministically, and assessed with the J3
operation, works end to end with custody at every transfer. The merged
lenses import with their fitting provenance, and a mixed lens records both
contributions and a composite corpus digest.

**Established (readout, exploratory).** With the J3 float32 readout and the
plain-residual baseline on the same positions:

- The general lens beats the eight-row pilot at 58 of 63 layers on both
  general and legal held-outs, so full-size fitting improves the readout
  measurably over the J4-era pilot.
- Register matters: on legal prose the 300-row case lens is the best
  readout at layers 20 to 55, ahead of the 828-row general lens, and costs
  0.01 to 0.02 JSD on general text in those same layers; the mixed lens
  recovers most of the legal gain at no measurable cost on general text.
- Against the plain residual, a lens fitted with the matching register leads
  consistently from about layer 19 (legal) or 16 to 21 (general); the general
  lens on legal prose only from layer 32 to 43.
- On no corpus does any readout reach 0.5 JSD to the final prediction before
  layer 51; every readout is above 0.6 through layer 45.

**Not established.** Agreement with an independent full-size derivative
reference: no published full lens for this model exists, and fitting more
rows does not create one. Mixed-GPU scientific equivalence: unchanged from
J4 (every shard here ran on one card type). Direction-transport usefulness:
untested; readout agreement is not causal validity. Cause attribution in the
composition results: composition and budget moved together, and the
comparisons are exploratory by design.

## 3. Scope statement after this supplement

Qualified within the tested scope: managed fitting, sharding, merging
(single- and mixed-corpus), custody, import, and readout assessment of the
hybrid 27B model at full row budgets on H100 cards, bf16 forward with float32
readout. The instrument of record for reading this model is the mixed lens;
the case-only lens is a specialist for legal text; the general lens is
retained as the reference. Readouts below about layer 45 are far from the
model's own prediction under every lens and should be reported beside the
plain-residual baseline at every layer used.

## 4. Owed

Independent review of J4 together with this supplement, before the support
statement. A weight study and a subspace comparison are research follow-ups,
not gates.
