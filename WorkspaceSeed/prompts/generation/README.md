# Dataset-generation prompts

Templates for generating CANDIDATE data with an LLM, which you then review and
edit by hand before committing the JSONL. They are the source of the
"copy LLM prompt" buttons in the Concept Lab and of the server's
`authoring template` endpoint — both engines render the same files with the
same `{{key}}` substitutions, so the prompt is identical whichever surface you
drive.

Nothing here generates *measurements*. Generated text is an INPUT: it becomes
stimuli, a corpus, or held-out probe items, all of which are hashed and pinned
like any other pinned input.

| Template | Produces | Lands at |
|---|---|---|
| `caa-paired-stimuli.md` | content-matched contrastive pairs for CAA | `prompts/concepts/<concept>/{positive,negative}.jsonl` |
| `repe-paired-reader-data.md` | paired prompt data for paired-difference PCA and the RepE reader | `prompts/repe/<concept>/pairs.jsonl` |
| `emotion-grand-mean-stories.md` | multi-concept story corpora for grand-mean extraction | `prompts/emotions/<dataset>/stories.jsonl` |
| `grand-mean-cowork-agent.md` | agent instructions for generating a balanced multi-concept corpus in parallel | pasted back as JSONL |
| `probe-validation-items.md` | held-out labeled probe items | `prompts/probes/<probe>/items.jsonl` |
| `neutral-norm-corpus.md` | long neutral prose for residual-norm calibration at later token positions | `prompts/neutral/corpus.jsonl` |
| `neutral-dialogues-anthropic-style.md` | Human/Assistant neutral dialogues (the nuisance-corpus shape) | a method-specific neutral corpus |

## The two rules that matter

- **Keep the roles separate.** Extraction stimuli, validation/probe items, and
  the task prompts you MEASURE are three different datasets. An item that does
  double duty makes the result circular.
- **Keep stimuli independent of the outcome.** Generated stimuli must not
  contain your study's task-domain vocabulary; if they do, the extracted
  direction partly encodes the task and every downstream effect is confounded.
  The stimulus-independence screen exists to catch this, and your workspace's
  `prompts/screens/forbidden-vocabulary.json` is where you declare which
  vocabulary is forbidden for YOUR study.

Copies of these files live in your workspace, so you can edit them; the
workspace copy always wins over the shipped one.
