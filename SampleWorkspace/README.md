# Sample workspace — a complete recipe, not a result

This folder is a SteerLab **data workspace** you can use immediately: copy it
anywhere you like (or open it in place) and point SteerLab at it.

```bash
steerlab --workspace /abs/path/to/SampleWorkspace experiment create demo \
  --model <model-id>
# or
export STEERLAB_WORKSPACE=/abs/path/to/SampleWorkspace
```

The app's **Workspace ▸ Open…** does the same thing.

## What is here, and what is deliberately not

| Path | What it is |
|---|---|
| `prompts/concepts/formality/` | a worked example concept: 24 content-matched contrastive pairs, 14 never-named validation scenarios, and a marker word list (see its README) |
| `prompts/tasks/starter-prompts.jsonl` | 16 generic task prompts; 8 carry `options` + `target`, so the answer-token instrument works out of the box |
| `prompts/batteries/starter-battery.jsonl` | 16 capability probes (arithmetic, recall, instruction- and format-following) using only the pinned grading modes |
| `prompts/neutral/corpus.jsonl` | the neutral corpus — the residual-norm denominator that makes steering strength (α) comparable across concepts |
| `prompts/rubrics/default-paired-v1.md` | the generic paired-judging rubric, pinnable via `pin-rubric` for a first judged comparison |

**Recipe-only — no vectors, no runs, no frozen manifest.** Everything here is
a *re-derivable input*: text you can read, hash, and version. Nothing here is
a measurement.

- **No `.safetensors`.** Steering vectors do not transfer across substrates
  (MLX/Metal and PyTorch/CUDA activations are not byte-identical), so a
  shipped vector would be at best useless and at worst misleading. You
  extract on your own substrate, from these stimuli, and validate there.
- **No `experiments/`.** The experiment manifest is the thing you create: it
  pins these files *by SHA-256* plus your extraction options, and freezing it
  is what makes the settings-before-measurement firewall mechanical. Creating
  it yourself is the first step of the walkthrough, not a chore the sample
  should skip for you.
- **No `runs/`.** Runs are immutable outputs of *your* model on *your*
  hardware. Re-deriving them is the point.

`experiments/` and `runs/` are created for you the first time SteerLab uses
this folder.

## The five-minute path

1. Point SteerLab at this folder (above).
2. `experiment create <name> --model <model-id>` — pins the model; add
   `--revision <commit>` or let freeze resolve it.
3. `experiment attach <name> formality` — pins the stimulus hashes.
4. `experiment extract <name>` then `experiment validate <name>` — derive the
   direction and probe it on the never-named validation scenarios. Freeze
   requires that evidence.
5. `data check <name>` — the manifest-driven readiness checklist tells you
   what is still missing (task prompts, battery, rubric, judges, …) and where
   each file goes.
6. `experiment freeze <name>` — one-way. Iterate by `duplicate`, never by
   editing a frozen manifest.

Every verb takes `--json`. `docs/CLI-REFERENCE.md` is the full surface, and
the `AGENTS.md` written into every workspace is the agent-facing contract.

## The one rule for this folder

**It is a starting point, not study content.** The formality concept exists
so that your first extraction runs today; the task prompts and battery exist
so the run loop has something to chew on. Adapt or replace all of it before
any run you intend to cite — and keep the discipline the example follows:
stimuli content-matched and independent of the outcome you measure,
validation scenarios that never name the concept, markers treated as a
diagnostic rather than a promotion objective.
