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

## The two hashes

Every emission stamps BOTH of these into its header, because they recover
different things.

`promptSpecHash` is the SHA-256 of the partials plus the template, in the order
they were assembled — the WORDING, before any substitution. It recovers which
prompt text a study is citing, and editing a template here changes it for
everything emitted from that template afterwards, which is the point. It is
deliberately not an identifier for one emission: two runs of the same kind for
two different concepts share it.

`promptInstanceHash` is the SHA-256 of the rendered body plus the resolved
parameter set — THIS emission. It recovers which run produced a given corpus,
and it moves when any argument moves, including one the wording does not happen
to interpolate. Ask the spec hash "what did this prompt say"; ask the instance
hash "was this the emission that said it".

**A workspace's copy wins over the shipped one.** Edit these files to fit your
study; the emitter reads the workspace copy first and both hashes follow your
bytes. Nothing here is study material — the wording is generic on purpose, and
the concept-specific seam arrives as arguments.

**Where the shipped copy lives.** Inside the engine package
(`steerlab_server/experiment/seed/prompts/authoring-prompts/`), so a
`pip install` with no source tree beside it still has a registry to render
from. The checkout's `WorkspaceSeed/` copy is the source of truth and a test
holds the two byte-identical.

## The rule these exist to enforce

Emitting a prompt is not accepting its output. Every one of these ends with an
audit battery expressed as NUMBERS, and the instruction to compute and report
them; whoever installs the delivered files re-runs that same battery
independently. The emitter is never the acceptor.
