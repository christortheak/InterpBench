# Sweeps, promotion, and confirmation

A sweep generates at every layer × strength cell, so it costs compute: agree
the grid with the researcher before it runs.

**A sweep** walks layer × α on a dev split and records a recommendation per
concept, selecting by the manifest's declared criterion (`sweep.selection`: an
objective, capability/coherence constraints, an optional matched-norm-random
control margin). Objectives are `markerDensity`, `judgeScore`, `logprobShift`.
Marker density measures surface vocabulary — it is a manipulation check, not a
selection objective for any study whose outcome is a decision rather than
prose. With no declared rule the sweep defaults to `markerDensity` and says so
with a `sweepSelectionDefaulted` advisory — on a choice-task prompt set, treat
that advisory as a stop sign. Submit it with `--verb sweep`: through
`steerlab run <name> --runner <url> --verb sweep` on a frozen study, where it
records recommendations only (`sweepRecommendationsOnly` advisory), or step by
step with `steerlab runner submit … --verb sweep` on a draft
(`workspace guide lifecycle`, step 7, shows the steps). It loads the model on
the runner.

This client has no verb of its own for the selection rule. The rule arrives in
a study pack or a duplicated study, or through `steerlab experiment
set-protocol <name> --set sweep=<json>`. That assignment replaces the WHOLE
sweep block, so set the rule first and the grid second: `set-sweep-grid` edits
the block in place and keeps the rule.

This client has no `promote` or `confirm` verb. To use a sweep's winning cell,
show the researcher the cell and its margin over the control, and after they
agree, declare a condition at that layer and strength with
`steerlab experiment declare-condition`. Say in your report that the cell came
from the sweep's recommendation and name the sweep's run directory. An agent
artifact promoted elsewhere attaches with `steerlab experiment attach-agent`
(`workspace guide assembly`).

## The sweep workflow

A sweep is where a study stops being a design and starts costing money: it
generates at every layer × α cell, for every concept, and then a declared rule
picks one cell. Four things go wrong here often enough to be written down.

**(a) Swapping the concept into an inherited sweep.** `duplicate` is how a
frozen study is iterated, and it copies the whole sweep block — grid, selection
rule, instrument pins — along with the donor's **concepts**. Those concepts ride
along swept but uncited: nothing in the new study declares them, and a direction
selected for one of them cannot be cited by the study that swept it. Attach the
new concept first and detach the donor's **last**:

```bash
steerlab experiment duplicate <donor> <new>
steerlab experiment attach <new> <concept>
steerlab experiment detach <new> <donor-concept>…
```

`detach` refuses (`conceptInUse`) while the inherited selection rule still
names the donor's concepts — that refusal is the ordering, enforced. Re-declare
the rule for the new concept first. Never hand-edit `experiment.json` to break
the cycle; the refusal is telling you a declaration would be left dangling, and
the manifest is not where you resolve that.

**(b) The grid dialog.** The grid is a cost and a preregistration, so a human
decides it — but the human has to be shown what they are deciding.

*Inherited grid* (a duplicate): **show it before you touch it** — the depth
fractions the manifest stores AND the absolute layers they resolve to at this
model's depth — and ask whether to keep it. `set-sweep-grid`'s `--json` result
carries both (`layerFractions`, `resolvedLayers`, `layerCount`, `cellCount`),
and so does `experiment list`'s manifest. A grid inherited from a study on a
26-block model names different blocks on a 62-block one; that is the fractions
working, and it is still a change the human should see.

*No grid* (de novo): **propose one and say where the proposal comes from.** The
engine default is `0.5,0.7,0.85 × 0.05,0.08,0.1,0.13` — depth fractions and
residual-norm α, recalibrated on live testing because stronger α routinely
buys incoherence and the useful cells sit late in the network. That is the
provenance; say so, say it is a starting grid and not a finding, and ask.

Then write the answer:

```bash
steerlab experiment set-sweep-grid <name> \
  --layer-fractions 0.5,0.7,0.85 --alphas 0.05,0.08,0.1,0.13
```

Layers may be named absolutely (`--layers 13,18,28`) when something has already
been extracted for the model — that is what states its depth. Both axes must
ascend with no repeats, α is in residual-norm units above 0, and `0` is the
baseline cell every sweep runs anyway.

**(c) The de novo path, in order.** Nothing here is skippable:

```bash
steerlab experiment create <name> --model <id> --revision <commit>
steerlab experiment attach <name> <concept>          # after part (d)
# the grid dialog (b), then:
steerlab experiment set-sweep-grid <name> --layer-fractions … --alphas …
# submit --verb validate, then --verb sweep (`workspace guide lifecycle`, step 7)
```

Validate before the sweep, always: sweeping a direction that scores at chance
on its own probe buys a confident setting for a vector that measures nothing.
Then read the sweep's recommendation, show the researcher the winning cell and
its margin over the control, and use it only after they say so.

**(d) The missing-data rule.** Most of the work above is blocked by data that
does not exist yet. For every missing prerequisite, in this order:

1. **Name it** — the exact path. `steerlab experiment verify <name> --json`
   names each missing or drifted pinned input, and `steerlab science guide
   <method> --json` lists the files a method needs.

2. **State what it ought to be** — the row shape, the counts, and what the file
   has to be independent of. Not "some validation rows": *held-out, labelled,
   and using neither pole's vocabulary, because the extraction corpus is full
   of that vocabulary and a probe that reuses it tests a word detector.*
3. **Emit its generation prompt** —
   `steerlab authoring prompt <kind>` renders the prompt for that kind of
   data with your study's seam substituted, its audit battery as NUMBERS, and
   two hashes stamped in the header: `promptSpecHash`, over the template and
   partials, which recovers the exact WORDING later, and
   `promptInstanceHash`, over the rendered body and the resolved arguments,
   which recovers WHICH EMISSION produced a given corpus. Two prompts for two
   concepts share the first and differ in the second. Kinds:
   `contrastive-pairs`, `choice-prompts`, `validation-set`, `reader-pairs`,
   `battery`. Hand it to an author unedited.
4. **Never install generated data on the generator's word.** The emitter is not
   the acceptor. A *second* reviewer — one who did not write the rows — re-runs
   the prompt's own audit battery against the delivery and reports the numbers.
   Only then does the file land in the workspace, and only then is it pinned.

That fourth step is the one that gets skipped, and it is the one that matters.
An author asked to audit their own output reports a pass; the numbers are cheap
to compute and expensive to fake, which is why they are numbers. A corpus that
was installed unaudited is not repairable later — it is pinned, frozen, and
cited.
