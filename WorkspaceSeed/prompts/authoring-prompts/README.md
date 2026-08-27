# Authoring prompts

One generation prompt per kind of missing study data. `steerlab-cli authoring
prompt <kind>` (and `steerlab authoring prompt <kind>` on the cross-platform
client) renders one of these with your arguments substituted and prints it, for
handing to an LLM.

The directory IS the registry — one file per kind, and the kind is the
filename's stem. A file whose stem is not a declared kind is not reachable, and
a declared kind with no file is a typed refusal naming the path, never a
silently empty prompt.

| File | Kind | Produces |
|---|---|---|
| `contrastive-pairs.md` | `contrastive-pairs` | `prompts/concepts/<c>/{positive,negative,validation}.jsonl` |
| `choice-prompts.md` | `choice-prompts` | a choice instrument the sweep's `logprobShift` objective scores on |
| `validation-set.md` | `validation-set` | `prompts/concepts/<c>/validation.jsonl` alone, for a concept whose extraction corpus already exists |
| `reader-pairs.md` | `reader-pairs` | `prompts/readers/<c>/pairs.jsonl` for a reader fit |
| `battery.md` | `battery` | `prompts/batteries/<name>.jsonl`, format 2 |

Files whose name begins with `_` are shared partials, not kinds:
`_discipline.md` is the universal discipline every prompt carries, and
`_delivery.md` is how a delivery is made and what happens to it next.

## The hash

Every emission stamps a `promptSpecHash` into its header: the SHA-256 of the
partials plus the template, in the order they were assembled. That is what a
study's provenance cites, so a prompt's exact wording is recoverable from a
finished study — and editing a template here changes the hash of everything
emitted from it afterwards, which is the point.

**A workspace's copy wins over the shipped one.** Edit these files to fit your
study; the emitter reads the workspace copy first and the hash follows your
bytes. Nothing here is study material — the wording is generic on purpose, and
the concept-specific seam arrives as arguments.

## The rule these exist to enforce

Emitting a prompt is not accepting its output. Every one of these ends with an
audit battery expressed as NUMBERS, and the instruction to compute and report
them; whoever installs the delivered files re-runs that same battery
independently. The emitter is never the acceptor.
