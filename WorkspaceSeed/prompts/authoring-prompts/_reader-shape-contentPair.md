Each row is TWO different pieces of text under ONE template. The contrast is
carried by the CONTENT: the reader's direction is the difference between the
activation of the positive stimulus and the activation of the negative one.

```json
{"id": "{{concept}}-pair-0", "concept": "{{concept}}", "positiveStimulus": "…", "negativeStimulus": "…", "topic": "…", "split": "train", "templateID": "{{templateID}}"}
```

`positiveStimulus` and `negativeStimulus` are both required strings. Do not
also write a `stimulus` key: a row declaring both shapes is refused, because
there is no way to tell which contrast was meant.

Because the content carries everything, this shape inherits the whole pair
discipline: within a row the two stimuli share their scenario, their length
(within {{lengthDeltaWords}} words), their register, their intensity, their
syntactic frame, their mood, their tense and their person — and differ only in
the pole. A row whose two stimuli are about different things fits a direction
for "different things".
