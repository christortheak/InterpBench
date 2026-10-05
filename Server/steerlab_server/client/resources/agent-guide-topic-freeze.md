# Freezing a study

## `freeze` — one-way

```bash
steerlab experiment verify <name>
steerlab experiment freeze <name> [--force]
```

Freeze verifies every pin, stamps the manifest's content hash and the
workspace git commit, snapshots every pinned input into
`experiments/<name>/pinned/` — concept stimulus directories, task prompts,
judge rubric, capability battery, reasoning-style taxonomy, human tables, and
the neutral corpus that denominates norm-unit α — writes the generated
settings summary beside the manifest (`experiments/<name>/preregistration.md`
when that path is free or holds a file that is *provably* a previous freeze's
own untouched output; a researcher-authored preregistration there is preserved
untouched, frozen — its SHA-256 is stamped into the manifest, it is
snapshotted into `pinned/`, `verify` re-hashes it from then on, and it travels
in run bundles — and the generated summary lands as
`preregistration-frozen-settings.md` instead), and makes the manifest
read-only. There is no unfreeze. **Iterate by `experiment duplicate <name>
<new-name>`, never by editing.**

A file at `preregistration.md` counts as the freeze's own only when the
manifest's stamped hash of what it last generated matches the bytes exactly,
or — for files predating those stamps — when the first line is the generated
header and the marker line `*Generated at freeze; do not edit.` is the last
non-empty line. Quoting that marker anywhere else in your own document is
safe: when in doubt the file is preserved, never overwritten.

The snapshot is taken at freeze time only: it is the no-git reproducibility
floor, not a live mirror. A study frozen before a pin joined the snapshot
keeps whatever its own freeze wrote.

Two classes of check, and the difference matters:

- **`verify()` pin integrity** — always runs, **never skippable**, owns no gate
  id. Drift in any pinned file's bytes is a violation here. `--force` does not
  reach it.
- **The seven gates below** — force-skippable, each with a stable id.

### The seven freeze gates

| Gate id | What it demands | Repair |
|---|---|---|
| `revision` | a pinned, immutable model commit — not absent, not symbolic | `steerlab experiment pin-revision <name> <commit>`, or validate on a runner, which pins the commit it resolved |
| `measurementPins` | pins that determine *what is measured* are present and valid (e.g. a loadable study dtype) | repoint the invalid pin at a loadable value |
| `validateEvidence` | a validation run matching the exact pins on the run substrate, **and** that evidence is not vacuous (`workspace guide lifecycle`) | author the named `validation.jsonl` files, **re-attach** their concepts, then `steerlab run <name> --runner <url> --verb validate` (a draft is accepted for this step) |
| `variantValidity` | attached variants carry hashed adapter weights and a pinnable dataset manifest | re-save the variant with hashed weights and re-attach it |
| `batteryEvidence` | baseline and each agent condition have scope-matched capability-battery evidence. A condition whose agent uses an intervention policy is exempt, because the battery cannot run a policy; the frozen study records it in `capabilityBatteryNotApplied` and is not forced | `steerlab run <name> --runner <url> --verb validate` again (each agent condition the battery can run is scored) |
| `judgeValidity` | a rubric **file** and at least one judge the pipeline can actually run (a panel of two or more must be distinct) | declare the rubric file and judges (`workspace guide evaluation`) |
| `gitClean` | every pinned input is committed in the workspace git repo | commit the pinned inputs |

Every refusal this client shows names this client's commands, and the
freeze repairs above are the ones its `error.repairAction` carries. The route
from a draft to a frozen study without `--force` is: validate on a runner,
read `result.validateEvidence`, then freeze.

In `--json` mode a refusal gives you `error.gate` (the gate whose message is in
`error.reason`), `error.gates[]` (**every** gate that failed, not just the
first), and `error.repairAction`. Fix the gate; do not retry the same command.

**`validateEvidence` is keyed by PINS, not by the experiment's name — so a
duplicate inherits its donor's evidence.** The gate matches on a validation
scope hash built from the model id and pinned revision, the attached concepts
with their pins, the neutral-corpus hash, the grand-mean corpus, the
capability-battery hash (when variant conditions exist), and any declared
validation depths. Conditions, sampling settings and the name are deliberately
outside it. Practically: **do not re-run `validate` on a duplicate that
changed only measurement-side declarations** — a new rubric, a different judge
panel, new exclusion rules — because the donor's evidence already satisfies
the gate and re-validating spends GPU time for nothing. Change something the
scope covers (re-attach a concept, re-pin the revision, point at a different
battery) and the inheritance correctly stops; then validate.

**`--force` skips the seven gates and stamps the manifest** `freezeForced: true`
plus `forcedGatesSkipped: [<gate ids>]`, and emits one `freezeGateSkipped`
advisory per gate. A forced freeze is permanently non-citable — but checkably
so, by stamp. **Do not `--force` to get past a gate.** If a human explicitly
asks for it, do it and report exactly which gates were skipped.

## What stays immutable after freeze

- **`runs/` is append-only.** Never edit, overwrite, or delete a run directory.
  A run carries enough to rebuild its tables without rerunning the model, and
  that is only true if nobody touches it. `runs/` is gitignored.
- **Three subtrees under `runs/` are deliberately mutable libraries**:
  `runs/model-variants/`, `runs/neutral-pcs/`, `runs/jlens-lenses/`. A promoted
  agent's artifact is editable in place there; frozen studies are protected by
  the manifest snapshot and the artifact hash, not by the directory.

- **Frozen manifests are read-only** — the verbs that WRITE the manifest
  (`attach`, `detach`, `declare-condition`, `remove-condition`,
  `set-protocol`, and the other `set-*` and `pin-*` verbs) refuse on a frozen
  or complete one; `duplicate`, then edit the copy. Submitting a frozen study
  for a sweep stays legal: the sweep records its recommendations in its own
  run directory rather than in the manifest.

- **`experiments/<name>/pinned/`** is the freeze-time snapshot of every pinned
  input — the reproducibility floor when git is unavailable. Do not edit it.
- Do not hand-edit `experiment.json`. Its bytes *are* the content hash; an edit
  that bypasses the verbs surfaces as a verify violation, which is the good
  outcome, or as a silently different study, which is not.
