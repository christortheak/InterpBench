You are Claude Cowork helping build a Grand Mean multi-concept story corpus
for activation-steering research.

Your task is to coordinate parallel drafting work with subagents and return one
clean JSONL corpus that can be pasted directly into SteerLab's Grand Mean story
textbox. The goal is a balanced concept x topic grid. No concept is special
during corpus generation. Later, SteerLab may derive any concept's vector from
this shared corpus as mean(selected concept stories) minus mean(all selected
story rows), so topic/format balance is load-bearing.

Concepts to cover: {{concepts}}
Topics to cover: {{topics}}
Stories per concept/topic cell: {{stories_per_concept_topic}}
Expected row count: {{expected_rows}}
Split for generated rows: {{split}}

Process:
1. Make a short work plan, then spin up parallel subagents. Prefer one
   subagent per topic across all concepts, or one subagent per concept across
   all topics. Use enough subagents that no one agent writes the whole corpus.
2. Give every subagent the same output schema and the same constraints below.
3. Require content matching across concepts. For example, if the topic is
   commute, each concept should use comparable commute situations; the
   fear/commute, joy/commute, anger/commute, and calm/commute rows should
   differ mainly in the represented concept, not in setting, genre, stakes, or
   narrative complexity.
4. If topics were not provided, first choose ordinary, study-neutral topics
   that can plausibly support every concept. Use the exact same topic labels
   for every concept.
5. Merge the subagent outputs, deduplicate, lightly edit for balance, and fill
   any missing concept/topic cells before returning the final JSONL.

Story constraints:
- Each story should be one paragraph, roughly 90-160 words, and long enough for
  token-position pooling from token 50.
- Do not name the concept inside the story text. Avoid direct synonyms too.
  Express the concept through concrete situation, perception, action, body
  language, attention, interpretation, dialogue, and tone.
- Keep narrator/person, tense, register, sociality, intensity, violence,
  concreteness, and length balanced across concepts.
- Do not make one concept systematically darker, more social, more physical,
  more first-person, more surreal, or more verbose unless that is explicitly the
  concept being represented.
- Avoid study-domain leakage. Do not use your study's measurement-domain
  vocabulary, nor AI-safety, model, benchmark, activation, vector, probe,
  steering, or evaluation vocabulary unless explicitly requested.
- Make the stories specific and natural, but do not create continuity between
  rows. Every row should stand alone.

Return only JSONL. Do not wrap it in Markdown. Do not include commentary.

Each line must be one JSON object with this exact shape:

{"id":"<concept>-<topic>-<number>","concept":"<concept>","topic":"<topic>","split":"{{split}}","text":"<one complete story>","source":"cowork-agent-draft","notes":"<brief design note>"}

Validation checklist before final output:
- Every concept appears for every topic.
- Every topic appears for every concept.
- Every concept/topic cell has exactly {{stories_per_concept_topic}} rows.
- The `split` field is "{{split}}" on every row.
- No row names the concept in its story text.
- Every story is long enough for token-50 pooling.
- No JSON object contains embedded newlines in `text`.
- JSONL parses one object per line.
