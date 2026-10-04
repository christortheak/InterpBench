# The workspace folder

## What this folder is

A workspace is a plain folder. It is its own git repository, created with an
initial commit, whenever git is available at creation. Layout:

```
prompts/          git-versioned inputs (see "Where things live")
experiments/      experiment manifests — freezable recipes
runs/             immutable run outputs (gitignored)
catalog/          GENERATED navigation over runs/ (symlinks; gitignored)
adapters/         per-adapter training data and outputs
WORKSPACE.md      the marker file that makes this a workspace
.gitignore        runs/, catalog/, adapters/**/*.safetensors, .DS_Store
```

A folder counts as a workspace if it carries `WORKSPACE.md` or at least a
`prompts/` directory.

An experiment is a **recipe**, not results: it pins inputs by SHA-256 plus the
options used to derive vectors from them, and runs re-derive deterministically.
That pinning is the firewall — settings are chosen and frozen *before* behavior
is measured, so a result cannot be reverse-fitted to the settings that produced
it.

## Pointing the client at this workspace

The workspace is named with `--root <dir>` on each command, or with
`STEERLAB_WORKSPACE` once for the session. There is no default: a command that
needs a workspace and was given none refuses with `error.code`
`workspaceNotSet` and changes nothing.

```bash
export STEERLAB_WORKSPACE=/abs/path/to/this/workspace
```

`workspace init` and `setup start` take their destination as a positional
argument, never through `--root`. The `runner`, `science`, `setup`, and
`authoring` families, and `workspace guide`, also run when no workspace is
named.

Every JSON response carries a top-level `workspace` field — a sibling of
`state`, never something under `result` — naming the root that answered, so you
can tell a wrong-workspace answer from a wrong answer. Check it on your first
command, and compare **resolved** paths: it echoes the path you configured with
symlinks intact, so `/tmp/ws` and `/private/tmp/ws` are one directory
disagreeing on paper. `realpath` both sides before concluding they differ.

If this folder is not a workspace yet — no `WORKSPACE.md`, no `prompts/` — run
`steerlab workspace init <directory>` first. It needs a new or empty
directory, and it never replaces existing files.

## Where things live

| Path | What goes there | Shape |
|---|---|---|
| `prompts/concepts/<name>/positive.jsonl` | contrastive stimuli, concept-present | `{"text": "…"}` per line |
| `prompts/concepts/<name>/negative.jsonl` | contrastive stimuli, matched control | `{"text": "…"}` per line |
| `prompts/concepts/<name>/validation.jsonl` | held-out probe (`workspace guide lifecycle`) | `{"text": "…", "expresses": true\|false}` per line |
| `prompts/concepts/<name>/markers.json` | optional marker word list | `{"words": ["…"]}` |
| `prompts/tasks/*.jsonl` | the measured task prompts | `{"id", "prompt", ["options", "target"]}` per line; ids unique |
| `prompts/rubrics/*.md` | judging instruments (Markdown) | prose rubric; pinned by file, not inline text |
| `prompts/batteries/*.jsonl` | capability probes | `{"prompt", "answer", "grading"}`; grading ∈ `exact_number`, `yes_no`, `token_exact`, `exact_normalized`, `regex` |
| `prompts/neutral/corpus.jsonl` | neutral corpus that denominates norm-unit α | `{"text": "…"}` per line |
| `prompts/templates/`, `probes/`, `readers/`, `dev/`, `panels/`, `parsers/`, `generation/`, `emotions/` | other pinned inputs | per their loaders |
| `experiments/<name>/experiment.json` | the manifest | written by the CLI; edit through verbs, not by hand |
| `experiments/<name>/pinned/` | freeze-time snapshot of every pinned input | written by `freeze`; read-only |
| `runs/<timestamp>-<slug>/` | one immutable run | see "Immutability" in `AGENTS.md` |

A new workspace is born with generic INSTRUMENTS only — batteries, the neutral
corpus, dev prompts, the parser registry, judge-rubric and data templates, and
the dataset-generation prompts. It carries **no concepts**: `prompts/concepts/`
exists and is empty, and authoring or importing a concept is the first real
step. A worked example (stimuli, task prompts, a battery — a recipe, never
vectors or runs) ships separately as a sample workspace you open on purpose.
**None of the seeded content is study material — adapt it before any run you
intend to keep**, and every concept needs its own `validation.jsonl` or freeze
refuses (`workspace guide lifecycle`).
