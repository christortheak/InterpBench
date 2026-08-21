# task-prompts-transcript-template.jsonl — scripted-transcript task prompts

Task prompts whose items are **scripted multi-turn transcripts** — the
metacognition-study instrument (Lindsey-style introspection at scale): the
researcher puts words in the model's mouth as pinned assistant turns,
optionally steers, and measures whether the model claims those words as its
own in its reply to the final user turn.

One JSON object per line:

```json
{"id": "…", "transcript": [{"role": "user", "content": "…"},
                            {"role": "assistant", "content": "…"},
                            {"role": "user", "content": "…"}],
 "options": ["yes", "no"], "target": "yes"}
```

- `transcript` — the scripted conversation. Rules (validated at load on both
  engines, at `data check`/verify time, and refused at run start):
  - non-empty; roles from `system` | `user` | `assistant`;
  - at most one `system` turn, and it must be first;
  - every turn has non-empty `content`;
  - the **final turn must be `user`** — generation produces the assistant's
    reply to it (assistant-prefix continuation is out of scope, v1);
  - the target model family's chat-template constraints must hold: **Gemma 3
    requires user-first strict alternation**; Qwen3 is permissive
    (assistant-first and consecutive same-role turns render).
- `text` / `prompt` — OPTIONAL when a transcript is present; used as the
  record's display text. Absent, the final user turn is the display text.
- A transcript's own `system` turn **replaces** the study-level system prompt
  for that item (the transcript is the more specific declaration).
- `options` / `target` and every other instrument field work unchanged — the
  instruments (answer-token logprob included) score the model's reply to the
  final user turn, rendered after the FULL transcript through the model
  family's real chat template.
- `promptMode` must be `chatAssistant` — transcripts are chat-template
  rendered by definition (rawCompletion is refused at verify and run start).

Records for transcript items stamp `scriptedTranscript: true` and carry the
transcript itself (records are the rebuild-without-rerun archive; the
transcript is the stimulus).

## Where the file lives in a workspace

Anywhere under `prompts/` works — `prompts/tasks/<study>-prompts.jsonl` is
the convention. The study pins the file by hash (`taskPromptsFile` +
`taskPromptsHash`) — transcripts ride inside it, so no new pin surface
exists; frozen studies refuse drifted or unpinned prompts.
