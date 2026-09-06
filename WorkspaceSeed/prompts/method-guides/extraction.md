# Choose and extract a steering direction

Compare declared activation populations before testing an intervention.

Choose the contrast with the researcher. Record the construct, counter-construct or reference population, model revision, rendering, reading position and intended claim before generating data.

| Method declaration | Inputs | Meaning and limitation |
|---|---|---|
| meanDifference | prompts/concepts/<concept>/positive.jsonl and negative.jsonl; each line has a text string | Difference of class means. Equal counts are not required by the arithmetic. Match nuisance features such as topic and register. |
| lat | The same files, aligned row by row with equal lengths | Paired-difference PCA with normalization and alternating orientation. Order is part of this implementation's recipe. RepE-inspired, not an unchanged RepE replication. |
| emotionGrandMean | prompts/emotions/<concept>/stories.jsonl for every named corpus member; text rows | Concept mean minus the pooled corpus mean, including the target. Corpus membership defines the comparison. |
| designatedReference | Target and explicitly chosen reference stories.jsonl files | Difference from that reference population. Do not select a reference merely because a file exists. |

A minimal stimulus row is {"text":"A statement expressing the declared position."}. It is a schema example, not a validated stimulus set. Keep extraction stimuli, held-out probes, development tasks and final evaluation tasks separate. A neutral corpus used to denominate norm-unit alpha is another pinned input, not the negative class.

Use `authoring prompt contrastive-pairs --concept <concept> --positive <definition> --negative <definition>` for paired stimuli. For story and neutral corpora, hand the coworker the text-row schema and author instructions in this guide; the prompt emitter has no dedicated kind for those corpora. Hand the resulting prompt to a coworker; independently review topic, length, register, polarity leakage, duplication and split overlap before importing. Do not treat generated data as validated evidence.

Declare with experiment attach --method and the explicit rendering/reading/reference/corpus flags shown by that client's help. Authoring is local; execution is steerlab-cli experiment extract on supported Mac cells or steerlab-server experiment extract on the Python engine. Use the client bundle workflow for supported remote stages. A template-aware reading position under raw rendering refuses; assistant-voice extraction is Python-only.

Inspect recipe and input pins in the vector sidecar. Then use held-out validation, a development sweep, baseline and declared controls before freezing final measurement. Cosine alignment or a stable vector alone does not establish a construct or a causal behavioral effect.

## Coworker author prompt

Help author the inputs for this method. First restate the researcher-approved construct, comparison, model/revision, input roles and proposed claim. List unresolved scientific choices as questions; do not choose them silently. Use the schema and public operations above and the selected authoring prompt. Return proposed files separately from an audit describing split independence, labels, nuisance balance, applicability and missing facts. Do not execute, invent pins, or overwrite evidence.

## Independent review prompt

Review the proposed inputs without assuming the author is correct. Check the declared method, schema, labels, split overlap, source/identity pins, baseline and controls, rendering/sampling settings and whether the requested claim follows from the planned measurements. Separate mechanical checks from scientific judgment. Name each blocker and its repair; passing a parser is not scientific validation.
