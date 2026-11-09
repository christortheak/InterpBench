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

**Agents made by hand:** `{{cli}} agent create <name> --model <id> --revision <sha>`
makes an agent without a sweep: the base model at a pinned 40-character commit,
and optionally one adapter (`--adapter <folder-or-fine-tune.json> --adapter-sha256 <weights digest>`),
any number of vectors (`--vector <runs/<run>/<name>> --vector-sha256 <artifactSHA256> --layer <n> --strength <alpha> [--mode add|ablate] [--centering none|neutralMean]`,
once per vector; the k-th `--vector-sha256`, `--layer`, `--strength`, `--mode`,
and `--centering` belong to the k-th `--vector`), `--band-width <1|3|…|11>`,
`--reasoning-effort <level>`, `--system-prompt-file <file>`,
`--prompt-mode <mode>`, and `--reason <text>`. A steering strength is in
residual-norm units, as the app's agent editor uses; an ablation's strength is
the fraction of the direction removed. An ablation with `--layer <n>` removes
the direction at that layer only (the layers above can restore it); `--layer
all` removes it at every layer, as the app's agent editor and the Playground
ablate. The preview says which, and advises when an ablation keeps the raw
direction (`--centering neutralMean` removes the vector's stored neutral mean
first). It previews first and writes nothing: show
the researcher the agent file and the checks, then repeat the command with
`--review-sha256 <reviewSHA256>` (`result.confirmCommand` is the exact
command). It refuses a revision that is not a pinned commit, an adapter or
vector whose bytes do not match the digest you give or that belongs to another
model or revision, a layer the vector does not have, a vector with no residual
norm at that layer, more than one adapter (both engines run only the first), a
thinking level the model's chat template does not accept, and a name another
agent has (ignoring case). Get vector digests from `{{cli}} experiment inspect-artifact`.
The agent lands in `runs/model-variants/` with a `handmade` record: the
command, each input and its hash, when it was made, and the reason, kept word
for word (the preview warns when the reason names a home folder). The record
names a file inside the workspace by its workspace path and a file outside it
by its SHA-256 alone, never by its name or a home-folder path. Attach it with
the returned `artifactFileSHA256`, adding `--replace <arm>` to put it in place
of an attached arm on a draft; attach refuses an agent made for another commit
of the model than the study's. There is no `--temperature`: the study
sets the temperature every arm runs at. `agent inspect`, `agent list`, and
`attach-agent` say whether an agent is `swept` (a sweep chose its settings and
promotion from that sweep made it) or `handmade`, and its thinking level
(`result.agent`). The app's agent editor records the agents it saves as
handmade too; an agent it saves from an edited one records where the chain of
edits began (`editedFrom`: that agent's kind, and for a promoted agent the
study and sweep run, beside the SHA-256 of the file the edit started from)
and the fields the researcher changed along the way, or that it is an
unchanged copy. A
handmade steered agent passes the same freeze gates as a
promoted one; say in METHODS who chose its settings and why.

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

A standalone diagnostic (the cost instrument, or a battery whose agents carry
policies) can run the same code. `{{cli}} science input-plan <request.json>`
shows it under `customCode`, and `science package` is refused with
`missingPrerequisite` until each provider is acknowledged. Once the researcher
has read the code and trusts it, repeat `science package` with the
`--custom-code-sha256` value the plan gives.

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
