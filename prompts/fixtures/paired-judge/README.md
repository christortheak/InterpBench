# Paired-judge wrapper goldens

Committed byte-goldens for the CANONICAL paired-judge prompt wrapper — the
text both engines wrap around (rubric, structured comparison fields, task
prompt, response A, response B) before any judge reads a pair. Unified
2026-07-22 after a cluster incident: the two engines' wrappers had drifted
apart (and their generation caps disagreed), and a local judge following the
old server wrapper wrote an essay-length reasoning that blew its token cap,
truncating a legible verdict into a refusal. The unified wrapper demands a
brief reason ("at most two sentences") and asks for the verdict fields
first, so a truncated response still states its winner legibly.

- `wrapper-full.golden.txt` — rubric + structured comparison fields + task
  prompt, multi-line response A.
- `wrapper-minimal.golden.txt` — empty rubric (exercises the fallback
  rubric sentence), no structured fields, no task prompt.

Builders pinned to these bytes:

- Python: `paired_judge.build_prompt`
  (`Server/steerlab_server/experiment/paired_judge.py`), tested by
  `Server/tests/test_judge_wrapper_and_salvage.py`.
- Swift: `PairedJudgePrompt.build`
  (`Sources/ExperimentKit/ClaudePairedJudge.swift`), tested by
  `Tests/ExperimentKitTests/PairedJudgeWrapperAndSalvageTests.swift`.

Both tests hardcode the same fixture inputs and compare the built prompt
byte-for-byte against these files; drift on either engine is a loud test
failure, never a skip. Change the wrapper only deliberately, on both
engines at once, and regenerate both goldens (the files carry no trailing
newline — they are the exact prompt bytes).

The unified generation cap travels with the wrapper:
`paired_judge.JUDGE_MAX_TOKENS` == `PairedJudgeBudget.maxTokens` == 1024.
