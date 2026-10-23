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

## 5. Independent review (2026-10-23)

The review owed in §4 was carried out on 2026-10-23 by a reviewer who
re-derived the figures from the assessment reports. This section records its
corrections; the text above is left as written. Two of the corrections (the
general-text cost of the mixed lens and the layer-45 value) were re-checked
against the WikiText assessment report before being recorded here.

- §1, "three full-size fits … were executed": two full-size fits and one
  cross-corpus merge.
- §2, "the best readout at layers 20 to 55": on case opinions the case-only
  lens is not the best at layers 24, 25, and 45 to 47 (the mixed lens is
  lower there), and on the ladder windows it is the best at only 21 of those
  36 layers. Its cost on general text is concentrated at layers 33 to 58 (up
  to 0.025 JSD at layer 51), not uniform over 20 to 55; below layer 25 it is
  better than the general lens on general text.
- §2, "at no measurable cost on general text": on the WikiText held-out the
  mixed lens's divergence is higher than the general lens's at 22 of 63
  layers, by up to 0.005 (layer 48); the largest per-layer difference in
  either direction is 0.008, and no uncertainty estimate is attached. Say:
  "at a small cost on general text (at most 0.005 JSD at any layer)".
- §2, onsets against the plain residual: the legal onset is about layer 19 on
  case opinions but about layer 29 on the ladder windows; the "16" for general
  text is the case-only lens, which is not the matching register; and "32 to
  43" is two separate onsets (the ladder windows from layer 32, interrupted at
  58, and case opinions from layer 43), not a band.
- §2, "every readout is above 0.6 through layer 45": the general lens is
  0.598 at layer 45 on WikiText; say "through layer 44".
- §3, "Qualified within the tested scope: managed fitting, sharding, merging
  … at full row budgets": the new evidence is end-to-end execution, custody,
  and readout. No full-size fit or merge was repeated, so determinism at full
  size is untested, and no reference agreement was measured at full size; the
  sentence also contradicts J4 §1 ("not qualified: any full-size hybrid 27B
  lens") without saying it supersedes it. Say: "Executed end to end with
  custody on H100 cards; full-size hybrid numerics rest on J2's five-layer,
  one-row subset."

Verified as written: 58 of 63 layers better than the pilot lens on both
held-outs; no readout below 0.5 before layer 51; the mean-JSD table; every
shard ran on H100 cards; the general lens's per-layer values identical across
the two WikiText assessments.
