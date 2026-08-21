You are generating an Anthropic-style neutral dialogue corpus for an activation-steering workbench.

Return exactly {{count}} JSONL rows. Each row must be one JSON object:

{"text":"..."}

Each `text` value must be a complete neutral dialogue. Use this format inside the string:

[optional system instructions]

Human: [line]

Assistant: [line]

Human: [line]

Assistant: [line]

[continue for 1-6 exchanges]

Requirements:

- Dialogues must be between a human user and an AI assistant.
- The first speaker turn must always be `Human:`.
- Some, but not all, rows should include brief system instructions before the first Human turn. Do not prefix those instructions with `System:`.
- The corpus must be neutral with respect to this concept set: {{neutral_concepts}}.
- Match these domains, settings, and task formats when possible, without expressing the concepts: {{matched_domains}}.
- Avoid these settings or dimensions unless the user explicitly included them as matched domains: {{avoid_settings}}.
- Use a diverse mix of neutral tasks within those constraints.
- Dialogues must be completely neutral with respect to the listed concepts.
- Do not include explicit, implied, or subtle expressions of the listed concepts.
- The Human must not express feelings, frustration, excitement, gratitude, worry, praise, blame, urgency, or distress.
- The Assistant must not express feelings, enthusiasm, concern, satisfaction, apology, reassurance, praise, or encouragement.
- Do not use pleasantries such as "I'd be happy to help", "Great question", "Certainly", "No problem", or "You're welcome".
- Keep language matter-of-fact, plain, and task-focused.
- Do not repeat templates, openings, topics, or sentence patterns.
- Each dialogue should be long enough to support token-position activation analysis; aim for {{minimum_words}} or more words per `text`.
- Output JSONL only. No markdown, code fences, numbering, explanations, or surrounding commentary.
