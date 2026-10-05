# Templates (the `design` verbs)

<!-- client: all -->

A **template** is a reusable set of study settings. The `design` verb family
operates on templates: the commands and JSON fields say "design", and they mean
the same thing. Both clients carry the same eight verbs.

`{{cli}} design list --json` reports the design library
and unreadable entries. `{{cli}} design inspect <name> --json` returns the
complete design document and `designFileSHA256`. To edit its description, use
`{{cli}} design describe <name> --description <text> --file-sha256 <digest> --json`
with that reviewed file digest. A stale digest refuses; inspect and review the
intervening changes before reconstructing the edit. Descriptions do not change
the scientific design hash. To create a study, use `{{cli}} design instantiate <name> --casting <file> --file-sha256 <digest> --study-name <new-name> --json`.
The casting JSON contains either `agents` (an array of objects naming
`artifactPath` and `artifactFileSHA256`; empty means baseline) or `seats`
(an object keyed by the inspected design's `seatIDs`, with null for baseline
and the same artifact objects for treated seats). The command checks reviewed
pins and derives an ordinary draft with design provenance. Both clients expose
`portableContentHash` with `portableHashAlgorithm: "portable-v1"` for cross-client
comparison. The Mac also retains its original `contentHash`. New Python drafts
stamp their lineage algorithm explicitly; old stamps and frozen studies are not
rewritten. File review digests remain external preconditions, never manifest fields. It reports the actual
name, including a suffix if the requested name was occupied. Review the draft
before freezing or submitting it.

Use `{{cli}} design save <study> --manifest-sha256 <digest> --name <design-name> --json`
to save a reusable design from the inspected study. An unchanged instance reuses
its existing design; a new or divergent source creates a separate design and
reports the actual name. Optional name/description fields apply to a newly
created design. To revise the design named by a study's lineage, inspect both
files and use `{{cli}} design update <design> --study <study> --manifest-sha256 <source-digest> --file-sha256 <design-digest> --json`.
Both saves use stored study settings, not unsaved app fields. Read any derivation
warnings before using the result. Source studies, including frozen ones, and
prior runs remain unchanged.

Use `{{cli}} design batch <design> --rows <batch.json> --file-sha256 <design-digest> --json`
for several explicit castings. The file contains `rows`, each with a `casting`
using the single-study format and an optional `studyName`. Review all rows before
creating drafts. Every row reports its zero-based index, actual study name or
failure, and typed issue with a repair. All successful rows share a batch group.
A partial batch returns a nonzero exit and retains its successful drafts; inspect
`result.minted` and retry only repaired failed rows. Repeating the whole batch
creates more studies. Nothing is frozen or submitted. The app casting table uses
the same per-row publication owner. Renaming and deleting a template have no
verb yet; do not invent one, and do not edit frozen studies.

`design expand <design> --casting <file> --mode permutations|composition
--file-sha256 <digest>` previews batch rows without minting or submitting.
Permutations accepts a seats casting and emits distinct re-seatings; composition
accepts agents:[one reviewed reference] and emits baseline, each solo-treated
seat, then all-treated without a one-seat duplicate. Expansion is bounded to
64 seats and 4,096 rows. Save `result.batch` as the rows file, review it, and use
`design batch` explicitly. Retain successful batch rows and retry only failures.
