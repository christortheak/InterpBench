# Concepts and glossary

This page is for a researcher in the humanities or the social sciences who has
no background in machine learning. It explains the ten ideas you meet first,
in plain words, then connects SteerLab's machinery to research methods you
already use. A glossary at the end covers the other terms you will see in the
app and in your coding assistant's reports.

Companion pages: [working with a coding assistant](WORKING-WITH-A-CODING-ASSISTANT.md),
[models, hardware, and limits](MODELS-HARDWARE-AND-LIMITS.md), and
[reporting and troubleshooting](REPORTING-AND-TROUBLESHOOTING.md).

## The ten ideas

**Model.** An open-weight language model: a program that writes text one word
piece at a time, whose files you download and run yourself. SteerLab works with
the model's internal numbers while it writes, which is why it needs open
weights rather than a chat service.

**Agent.** A configured model under study: a base model together with the
directions or adapters applied to it and their settings. The unmodified model
is the *baseline agent*. In SteerLab, "agent" always means this. The AI tool
that helps you drive SteerLab is your *coding assistant*.

**Direction.** A pattern in the model's internal numbers that tracks a concept,
such as formality or a particular stance. SteerLab estimates one from example
texts you write (texts that express the concept, and matched texts that do
not), or imports one from published interpretability work. You will also see it
called a *vector* or a *concept vector*.

**Layer.** A model is a stack of a few dozen similar processing steps, called
layers, that each pass their result to the next. A direction is read from, and
added at, one chosen layer. Layers in the middle of the stack often work best,
but that is something to test, not assume.

**Steering strength.** How strongly a direction is applied, written α (alpha).
SteerLab states it as a fraction of the typical size of the model's own internal
activity at that layer, measured on a fixed collection of neutral texts. That
shared yardstick is what lets a strength of 0.3 mean roughly the same push for
different concepts and models. On small models, effects often appear well below
0.5, and text often stops making sense above about 1. Treat that as a range to
test, not a setting to adopt.

**Condition.** One named arm of a study: a complete configuration, such as "the
formality direction at layer 18, strength 0.3". A condition can combine several
directions.

**Baseline.** The unmodified model answering the same prompts. Every
measurement is paired with the same prompt's baseline answer, so each prompt
acts as its own control.

**Control.** A condition that tells you what an effect is *not*. The most
important is a random direction of the same size at the same layer: if it
changes the outcome as much as your concept does, you measured the size of the
push, not the concept. Others are a negative strength (pushing the concept out
should move the outcome the other way), a concept that plainly should move the
outcome, and the capability battery (short unrelated tasks that should not get
worse).

**Freezing.** The one-way step that fixes every setting of a study before any
behavior is measured. Freezing checks that every input file is exactly the one
you chose, applies seven checks that the evidence is in place, keeps a copy of
every input, and makes the study read-only. To change a frozen study, you
duplicate it and change the copy.

**Run.** One execution of a study, written to a new folder under `runs/` that
is never changed afterwards. A run holds the responses, the measurements, and
enough provenance to rebuild every table without running the model again.

## Methods you already know

SteerLab's machinery maps onto familiar parts of research design. The mapping is
close, but not exact, and the differences matter when you write up.

**Operationalization: a concept's examples.** A concept enters SteerLab only
as the example texts you write for it: texts that express it and matched texts
that do not. Those examples *are* your operational definition, so two rules
decide what the direction means. The two sets must differ in the concept and
nothing else (same topics, lengths, and register), or the direction encodes the
difference you did not intend. And the examples must not contain the
situations or vocabulary you will later measure, or a change in the outcome
shows only that the model noticed a topic.

**Construct validity, and its limits: held-out validation.** Before a direction
is used, SteerLab checks it on a separate set of texts that show the concept
without ever naming it. A direction that only memorized a keyword fails this
check. With two or more concepts it also reports how similar their directions
are, so that distinct concepts do not collapse into one. With a single concept,
it says that comparison was not made. Passing validation means the direction
detects its concept in new text. It is not evidence that steering with it
changes behavior; that is what the study itself measures.

**Manipulation checks: marker density and reader scores.** Marker density counts
how often the concept's own vocabulary appears in the responses. It shows that
the intervention did something to the surface of the text, and it is a check,
not an outcome: choosing a strength because it raises marker density selects
for style, which is the confound most steering studies need to rule out. A
reader or probe score reads the concept from the model's internal state. Read
it as a difference between conditions, not against a fixed threshold.

**Preregistration: freezing.** Freezing is preregistration carried out by the
software. The settings that produced a result are fixed, with their file hashes,
before any behavior is measured, and anyone can check that later. Freezing also
writes a settings summary beside the study, or keeps and protects a
`preregistration.md` you wrote yourself. A study can be frozen "with force",
skipping the evidence checks, but that is recorded permanently, and SteerLab
treats such a study as not citable. The app has no button for it.

**Inter-rater reliability: judge agreement.** Free-text responses are scored by
*judges*: language models that apply a rubric file you pin, either a local model
or a paid service. A judge either compares each response with the same
prompt's baseline answer, without being told which is which, or codes each
response against fields you define. When two or more judges code responses,
SteerLab reports percent agreement and Cohen's κ for categorical fields, with a
table showing which labels the judges confused. One judge is allowed, but then
no agreement statistic exists, and the report says so. A common compromise is to
calibrate with two judges once and report that κ beside single-judge results.

**Effect sizes and intervals.** For each condition and outcome, SteerLab reports
the mean of the paired differences (condition minus baseline, prompt by prompt)
in the outcome's own units, with a bootstrap confidence interval and a Wilcoxon
signed-rank test. It corrects for multiple comparisons with the
Benjamini-Hochberg false discovery rate for broad screens and with Holm for
confirmations. A row with fewer than three pairs shows no interval, and an
effect is said to survive correction only when more than one comparison was
corrected. SteerLab does not compute standardized effect sizes such as Cohen's
*d*.

**Sample size and power.** SteerLab has no power calculator. The number of
observations is the number of prompts, times the conditions, times the samples
per prompt, and you decide it. When having judges code every response would
cost too much, a study can declare a seeded random subsample to code, chosen
before you look at anything.

**Exploratory and confirmatory work.** Strengths and layers are chosen on a
separate set of development prompts (a *sweep*), never on the prompts you
report. Confirming a finding is a second, frozen study on prompts the first one
never used, and SteerLab checks that the two sets do not overlap.

**Human baselines.** A study can pin a table of published human effects (one
row per outcome, with the estimate and its interval) and compare the model's
effect with it. The comparison places each concept in a plain region, such as
"same direction, credibly larger" or "humans do not move, the model does". It
compares two estimates from two populations measured by two instruments. It
does not show that a model is more or less "human". The table holds summary
numbers only, never participant data.

**Confounds.** The ones SteerLab measures for you: degradation (a study's
capability battery runs under every condition, and a drop there must be
reported beside the effect), the size of the push (the random-direction
control), and
truncation (responses cut off at the length limit are counted per condition,
and a study can refuse a cell where too many were). The ones you must watch
yourself: example texts that differ in more than the concept, outcomes parsed
from free text where parsing fails more often under steering, and judges or
rubrics that could guess the condition.

## Glossary

**Activation, residual stream.** The internal numbers a model carries from
layer to layer while it reads and writes. The residual stream is that flow as a
whole. It is where SteerLab reads directions and adds them.

**Adapter.** A small set of trained weights that changes a model's behavior,
also called fine-tuning. SteerLab can compare an adapter with steering on the
same footing.

**Advisory.** Something you should know that did not stop a command, such as a
skipped check or a one-judge design. Advisories are not failures, and they are
not to be ignored.

**CAA (contrastive activation addition).** The basic extraction recipe: the
average internal state over texts that express the concept, minus the average
over matched texts that do not, at each layer. SteerLab has five recipes,
including paired-difference PCA and the RepE reader; [METHODS.md](METHODS.md)
defines each.

**Capability battery.** Short unrelated tasks run under every condition, to
show the model still works. If accuracy drops, an apparent effect may be plain
damage.

**Concept.** The behavior or quality you are studying, defined by your example
texts. A new workspace contains none; you author them.

**Demo Workspace.** A copy of a worked example that SteerLab can open for you,
holding a finished study to read and a draft to run.

**Draft.** A study that is not yet frozen. Only drafts can be edited.

**Engine.** The part that loads a model and runs it. SteerLab has two: the one
built into the Mac app (MLX) and the Python engine. Their numbers are not
interchangeable; see [models, hardware, and limits](MODELS-HARDWARE-AND-LIMITS.md).

**Extraction.** Estimating a direction from your example texts.

**Headline outcome.** The outcome a results summary leads with: the one you
declare as primary, or else a judged outcome, then a choice or numeric outcome,
then a reader or probe score, then reasoning style, then marker density, and
surface measures such as word count last. The summary says which rule chose it.

**J-lens, SAE, OptVec, probe.** Advanced methods: reading what the model is
poised to say, importing published features, training a direction against an
objective, and fitting a reader of internal states. Each has a method guide
your assistant can print with `science guide <method>`. They need the Python
engine.

**Judge.** A language model that scores responses against a pinned rubric. A
local judge runs on your hardware; a paid judge sends each response to an
outside service.

**Model revision.** The exact published version of a model's files. Freezing
requires one, because a model name alone can point to different files later.

**Neutral corpus.** A fixed collection of ordinary texts. It supplies the
yardstick for steering strength.

**Pin.** A record of a file's exact contents (its SHA-256 hash) inside a study.
If the file changes by one character, SteerLab notices and says so.

**Refusal.** SteerLab declining a request that would produce an unreliable or
unreadable result. Every refusal says what is wrong and how to repair it; see
[reporting and troubleshooting](REPORTING-AND-TROUBLESHOOTING.md).

**Rubric.** The instructions a judge follows, kept as a file and pinned.

**Sidecar.** The small file saved beside each direction, recording how it was
made: the model and its revision, the layer, the recipe, and the reading
position.

**Stimulus set.** The example texts a concept is extracted from. Their hash is
the concept's identity wherever a study pins it.

**Sweep.** Trying a planned range of layers and strengths on development
prompts, then choosing one by a rule you declared first.

**Task prompts.** The items the model answers in a study: the measured task.

**Template.** Reusable study settings (the task, outcomes, sampling, and judges)
without the agents or the compute. Templates live in the app's Templates
section; on the command line the `design` commands work on them.

**Workspace.** The folder that holds one project's inputs, studies, and runs,
with its own history. The app and your coding assistant work on the same
workspace. It never lives inside the SteerLab source code.

For the mathematics behind each method, see [METHODS.md](METHODS.md). For what
makes a result defensible, see [CONDUCTING-A-STUDY.md](CONDUCTING-A-STUDY.md)
and [RESULTS-ARCHITECTURE.md](RESULTS-ARCHITECTURE.md).
