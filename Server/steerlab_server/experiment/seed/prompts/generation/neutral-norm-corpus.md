You are generating a neutral residual-norm calibration corpus for an activation-steering workbench.

Return exactly {{count}} JSONL rows. Each row must be one JSON object:

{"text":"..."}

Requirements:

- Each text must be at least {{minimum_words}} words.
- The text should be semantically ordinary, affectively neutral, and non-persuasive.
- Do not mention emotions, moral values, politics, religion, identity groups, law, courts, cooperation games, AI, models, or experiments.
- Do not include dramatic stakes, danger, conflict, praise, blame, fear, joy, hunger, fairness, authority, ideology, or personality traits.
- Cover many mundane domains: maintenance procedures, office logistics, gardening, library operations, transit schedules, inventory, cooking instructions, weather records, furniture assembly, classroom administration, storage labeling, cleaning routines, basic geography, measurement, building layouts, and similar topics.
- Vary syntax and vocabulary without making the passages stylistically distinctive.
- Do not duplicate scenarios, openings, or sentence templates.
- Keep all rows self-contained plain prose. No lists, dialogue, titles, markdown, numbering, or explanations.
- Output JSONL only. No surrounding commentary.

The point of this corpus is not to represent any concept. It is a denominator for measuring typical residual-stream norms at later token positions, so short sentences are unusable.
