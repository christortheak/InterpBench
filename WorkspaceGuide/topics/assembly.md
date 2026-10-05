# Assembling a study from reviewed pieces

<!-- client: all -->

These verbs write only what you reviewed: each takes the SHA-256 of the file
it read, and refuses if the file changed in between. Inspect, review, then
write.

<!-- client: mac -->

**Agent attachment:** `steerlab-cli agent list --json` lists discoverable local
agents; `steerlab-cli agent inspect <path> --json` returns the artifact and
`artifactFileSHA256`. Inspect the target with `steerlab-cli experiment manifest <name> --json`, then use `steerlab-cli experiment attach-agent <name> --artifact <path> --artifact-sha256 <digest> --manifest-sha256 <digest> --json`. Both digests
come from the reviewed files. The service computes the condition's pin; never
invent one. A changed file refuses, as does a frozen study or wrong base model.
Inspect and review the changes before reconstructing an attachment request.

<!-- client: python -->

**Agent attachment:** `steerlab agent list --json` lists discoverable local
agents; `steerlab agent inspect <path> --json` returns the artifact and
`artifactFileSHA256`. Inspect the target with `steerlab experiment inspect <name> --json`
(it returns `document` and `manifestFileSHA256`), then use
`steerlab experiment attach-agent <name> --artifact <path> --artifact-sha256 <digest> --manifest-sha256 <digest> --json`.
Both digests come from the reviewed files. The service computes the
condition's pin; never invent one. A changed file refuses, as does a frozen
study or wrong base model. Inspect and review the changes before
reconstructing an attachment request.

<!-- client: all -->

**Renaming and deleting:** `{{cli}} experiment rename <name> <new-name>`,
`{{cli}} experiment delete <name>`, `{{cli}} design rename <name> <new-name>`,
`{{cli}} design delete <name>` (templates), and `{{cli}} agent delete <path>`
preview by default: each prints what would change and the reviewed digest, and
writes nothing. Show the researcher the preview and apply only after they agree,
by repeating the command with that digest (`--manifest-sha256`, `--file-sha256`,
or `--artifact-sha256`) and `--yes`; `result.confirmCommand` is the exact
command. A changed file refuses; preview again. Only a draft study can be renamed
or deleted: a frozen or complete one refuses with `statusImmutable`, and the
repair is to duplicate it. A delete moves the item into a `.trash-<time>` folder
beside it, where it can be recovered. Nothing in `runs/` changes, and a study's
runs keep the name they recorded. An agent any study uses refuses with
`agentInUse` and names the studies; an agent a run saved refuses with
`agentIsRunEvidence`, because run folders are evidence. Never rename or delete
these by hand.

**Concept-to-draft assembly:** `{{cli}} authoring study conceptStudy --json`
(or `agentComparison` / `multiAgent`) emits the same study interview used by the
app. Discuss substantive choices first; use `authoring prompt <kind>` for exact
dataset-generation and independent-review instructions. Save the delivered
study pack as JSON, then run `{{cli}} pack preview <file> --json`. Inspect
its file plan and advisories; pass the returned `reviewSHA256` to
`{{cli}} pack apply <file> --review-sha256 <digest> --json`. Changed files,
workspace or pack text require another review. Apply creates a draft, pins real
input bytes and reports `verificationIssues`; import success is not permission
to skip verification or freeze gates.

<!-- client: mac -->

The app's Paste Study JSON uses the same preview/apply owner, so continue from
the named draft in either interface.

<!-- client: all -->

**Custom code in a shared study:** an intervention policy can carry an expert
provider, Python that the engine runs with the researcher's permissions when
the study runs. It is the one kind of workspace input that is code rather than
data, and nothing sandboxes it. When `pack apply` or `experiment attach-agent`
brings such code into the workspace, the result carries a `customCode` block:
the notice, each provider's source SHA-256, and the commands to read and
acknowledge it.

<!-- client: python -->

`bundle import` carries the same block for each study it lands.

<!-- client: mac -->

The app shows the same notice at the top of the study's page, with the code
and an acknowledge button.

<!-- client: all -->

Show the researcher the notice and the code:
`{{cli}} experiment acknowledge-custom-code <study> --json` prints it and writes
nothing. Only the researcher decides to trust it. Once they say so, run
`{{cli}} experiment acknowledge-custom-code <study> --sha256 <hash> --json`,
which records who acknowledged which hash, and when, in
`custom-code-acknowledgements.json`. Until then, a step that executes the
study's agents (`run`, `pipeline`, or `sweep`) is refused with
`missingPrerequisite`. Never acknowledge on the researcher's behalf to get past
that refusal.

`pack export <study> --json` returns a pack plus external artifact dependencies;
it is not a model/vector execution bundle. Full-record prompt updates use
`experiment import-prompts <study> --file <jsonl> --manifest-sha256 <digest> --json`:
records become immutable input versions, preserving prior study inputs.

**Vector attachment:** `{{cli}} experiment inspect-artifact <path> --json`
reads an existing vector pair's sidecar and both file digests. Then use
`{{cli}} experiment attach-artifact <study> <concept> --artifact <path> --artifact-sha256 <digest> --sidecar-sha256 <digest> --manifest-sha256 <digest> --json`.
The path is workspace-relative; frozen runs and vector files are read-only.
The same inspected bytes must still exist when attaching. This uses the store's
scientific admission, including residual-norm and substrate requirements;
inspection alone does not certify attachability. Supply `--source-concept` or `--eval-run` only when
required by the artifact's provenance, never to bypass a refusal.

<!-- client: mac -->

The app's Attach vector artifact uses the same operation.

<!-- client: all -->

Keep preview and apply on the same client, in the same workspace: review
tokens are authoring preconditions, not portable scientific identifiers. The
client preserves complete prompt records and uses its scientific store for
vector admission, including substrate restrictions.

<!-- client: mac -->

## Mirroring a direction

**`vectors mirror-poles <runDir/name> --concept <newName>`** — mints the
**opposite pole** of a contrastive direction as an artifact of its own,
instead of leaving it as a negative α that every dose surface reads as "less
of the concept" and no artifact records. A bit-exact sign flip on the
`layer_<i>` tensors (never on `neutral_mean_layer_<i>`), under a **required
new concept name**, stamped `negatedFrom` + `polesSwappedFromSource`. Restricted
to paired, source-concept-bearing contrasts — the CAA family — because those
are the methods whose two poles ARE two authored stimulus files, so swapping
their roles is exactly what the negation means; every other method is refused
`unmirrorableMethod`, pointing at the negative α that *is* available. No model
is loaded. Three consequences you will meet downstream: the mirrored concept's
own directory must hold the source's two files with their roles swapped, and
`attach` proves it by hashing them **in the source's order**; `promote`
matches the recipe identity against the pin's `sourceStimulusSetHash`, the
hash the artifact actually stamps; and the birth certificate carries a
`poleProvenance` block, so an arm injecting a negated direction never looks
like one injecting the concept. Author the mirrored concept's
`validation.jsonl` yourself — the verb writes nothing into `prompts/`, because
an engine that invented held-out evidence would be manufacturing the thing
`validate` exists to demand.

<!-- client: python -->

## Mirroring a direction

This client has no `vectors` family, so it cannot mint the opposite pole of a
contrastive direction as an artifact of its own. A negative strength (α) in a
condition is available instead; say in your report that the arm is a negative
dose of the concept, not a separately authored opposite pole.
