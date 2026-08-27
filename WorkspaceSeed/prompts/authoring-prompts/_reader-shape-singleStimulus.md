Each row is ONE piece of text read under TWO templates. The contrast is carried
by the INSTRUCTION, not by the content: the template pair `{{templateID}}`
supplies an experimental instruction and a reference instruction, the same
stimulus is read under each, and the reader's direction is the difference.

```json
{"id": "{{concept}}-row-0", "concept": "{{concept}}", "stimulus": "…", "topic": "…", "split": "train", "templateID": "{{templateID}}"}
```

`stimulus` is a required string, and its presence is what selects this shape.
Do not also write `positiveStimulus`/`negativeStimulus`: a row declaring both
shapes is refused.

This shape exists precisely to remove the content confound the paired shape
carries — so the stimuli themselves must be **neutral with respect to
`{{concept}}`**. A stimulus that already leans one way lets the content
contribute to a difference the instruction was supposed to own, and the
direction becomes a blend of the two. Write situations that are genuinely
readable either way, vary them widely, and audit for lean: if you can predict
which instruction a stimulus "wants", rewrite it.
