# Understanding SteerLab’s J-lens work

An explanation for researchers and collaborators · 21 September 2026

**We have built and tested a workflow for creating model-specific “lenses” into a language model’s internal processing.** The workflow can prepare a text corpus, fit a lens on a GPU cluster, bring the result back, and measure how well it reads the model’s internal states. We have completed substantial fits on 4-billion- and 27-billion-parameter models. The numerical checks are encouraging; the scientific results show both useful improvements and important limits.

This account reflects the recorded results as of the date above. Some additional assessments are still pending in the latest report.

## What is a J-lens?

A language model processes text through a sequence of layers. Between layers, it carries an internal representation called the **residual stream**. At the end, it converts that representation into scores for possible next tokens—words or pieces of words.

Researchers would like to understand the intermediate representations too. But the model’s output machinery is designed to read the final representation. Applying it directly to an earlier layer may give a misleading picture.

A **J-lens**, short for *Jacobian lens*, measures how small changes in an earlier representation affect the final representation. A Jacobian is a table of sensitivities: how much each output component responds to a small change in each input component. We calculate these sensitivities over a text corpus and average them to produce a matrix for each selected layer.

The resulting lens describes an average local relationship through the remaining layers. It is specific to a model and informed by the text on which it was fitted. It does not retrain the model or change its weights.

There are two research uses:

- **Reading internal states.** Transform an earlier representation with the lens, then apply the model’s output machinery. This produces a vocabulary-level view that can be compared with the model’s actual final prediction.
- **Studying interventions.** Use the sensitivities to investigate how a small change at one layer might affect later representations, or to construct candidate steering directions.

These uses require separate evidence. A derivative describes a local response to change; applying its corpus average to an entire representation is an additional approximation. Good token prediction does not establish that a proposed intervention will produce a particular behavior.

## What we built

The goal is to let a researcher choose a model, supply suitable text, and review a fitting plan without having to assemble the numerical and cluster infrastructure themselves.

SteerLab now provides a shared workflow through the Mac app, command-line clients, and HTTP API. It covers corpus preparation, a small timing pilot, fitting, checkpoints, continuation, splitting a fit across jobs, merging the results, verified transfer, library registration, and assessment. Instructions for coding agents describe the same operations.

The workflow records the model version, corpus, precision, hardware, fitting settings, and implementation identity. This matters because two files called “a lens for this model” can embody different experimental choices.

Large fits can be divided into **shards**, each processing part of the corpus. Their accumulated results can then be merged in a defined order, weighted by the number of usable rows. In the current estimator, usable rows receive equal weight; a longer row does not automatically count as more observations. The identities and contribution counts remain part of the evidence.

We also built assessment tools that compare lenses on text excluded from fitting. These include a crucial control: the **logit lens**, which applies the model’s output machinery directly to the earlier representation. A fitted J-lens should be compared with this simpler approach, as well as with another J-lens.

## Why the numerical investigation mattered

An early concern was that changing how much derivative work we grouped together produced different lens values, especially on the larger model. The mathematical target should be the same. We needed to determine whether our code was wrong, whether the computation was unstable, or whether lower-precision arithmetic explained the differences.

We tested several distinct questions:

1. **Does the estimator compute the intended derivative?** On a small model where an independent calculation was feasible, we compared it with an explicitly constructed Jacobian and finite differences. We also deliberately introduced errors to check that the tests detected them.
2. **Does SteerLab change the reference calculation?** On tested real-model inputs, the managed implementation produced exactly the same values as the reference calculation run directly under matching settings.
3. **Does an identical computation repeat?** Repeats under the same tested configuration were bit-for-bit identical, including checks that changed the order in which components were calculated.
4. **What happens when the configuration changes?** Changing the GPU or the amount of work grouped together could produce different, repeatable answers in bf16, a lower-precision numerical format. Float32 comparisons on the smaller model showed much closer agreement across batch sizes.

The pattern is consistent with different computational shapes selecting different numerical execution paths. The exact kernel-selection mechanism was inferred rather than directly captured. We therefore should not describe that mechanism as conclusively proved.

The conclusion is strong but scoped: the independent controls support the estimator’s mathematics, and the real-model checks support the implementation’s fidelity and repeatability in the tested configurations. They do not prove that every model, GPU, or precision setting produces scientifically interchangeable lenses.

## What the fitted lenses actually achieved

We assess readout by asking whether an earlier-layer view resembles the model’s final next-token probability distribution. The reports measure both the difference between distributions and the overlap among their highest-ranked tokens. This measures agreement with the model—not whether the model’s answer is true or its reasoning is sound.

**On the 4-billion-parameter model**, our 1,000-row fit agreed with final predictions better than the published comparison lens across the assessed layers and corpora. However, several fitting conditions differed, so this comparison does not isolate more data as the cause. More importantly, the simple logit-lens baseline performed better through much of the middle-to-late part of the model. A better fitted lens is not automatically the best available readout.

**On the 27-billion-parameter model**, we completed an 828-row fit across eight GPU jobs, each taking approximately 19 hours on an H100. The resulting registered lens contains matrices for 63 layers and occupies approximately 6.6 GB. These are measurements of this particular run, not general cost estimates for every model or corpus.

The larger fit improved on the eight-row pilot at 58 of 63 layers on both assessed corpora. It also outperformed the simple baseline over a substantial later portion of the model. But the advantage depended on the evaluation text: it began earlier on general text than on the specialist-domain passages.

There is an important qualification. Through much of the earlier and middle processing, even the better readout remained far from the final prediction. An improvement can be real while the result remains a poor basis for saying “this is what the model thinks at this layer.”

We also tested whether the output-reading calculation itself introduced substantial rounding effects. Running that part in float32 made only small changes in the smaller-model assessment and did not remove its distinctive early-layer disagreement. Higher-precision readout does not recover precision already lost during the model’s forward pass or during fitting.

## Why we are testing different fitting corpora

A J-lens averages sensitivities measured on particular text. We therefore want to know how much its usefulness depends on that text.

The next experiment combines the general-text fit with a fit from 300 specialist-domain documents, then evaluates the combined lens on separate general and specialist text. The question is whether domain-relevant fitting improves domain readout, and what happens to general-text performance.

That is a useful exploratory comparison, but it changes both the composition and the total amount of fitting data. An improvement would not, by itself, prove that domain matching caused it. Nor would it prove that the lens had learned a named psychological or semantic concept.

The latest status report records the domain fit as completed, with the mixed merge and further assessments in progress. Their conclusions should be reported once the results are collected and reviewed.

## What this enables—and what remains open

The practical achievement is a reproducible route from a researcher’s model and corpus to an inspectable lens, with recorded computational choices and explicit comparisons. Researchers can now investigate where a lens helps, where it fails, how fitting data affects it, and whether its predicted sensitivities are useful for intervention research.

Several questions remain open:

- Does a lens accurately predict the effects of actual interventions, particularly beyond very small changes?
- How interchangeable are lenses fitted on different hardware configurations?
- How much do domain-matched corpora help, once fitting budget is controlled?
- Which layers provide useful readouts, rather than merely relative improvements over poor alternatives?

Independent review of the updated qualification statement and parts of the final app experience also remain outstanding. Operational success at full model size extends the evidence; it does not replace an independent mathematical check at that size.

**The result is a research instrument with tested foundations and measurable limitations.** It makes J-lens experiments substantially easier to conduct while keeping the distinction between a completed fit, a reliable calculation, and a scientifically useful interpretation visible.

## Technical records

- [Latest fitting and qualification status](JLENS-AND-P7-STATUS-HANDOFF-2026-09-21.md).
- [Scoped numerical qualification decision](JLENS-QUALIFICATION-DECISION-J4-2026-09-13.md), which predates the full 27B fit and should be read alongside the later status report.
- [Independent review of the managed readout assessment](REVIEW-JLENS-ASSESSMENT-READOUT-BRANCH-2026-09-13.md).
- [Mixed-corpus merge design](JLENS-MIXED-CORPUS-MERGE-2026-09-17.md).
- [Recommended remaining acceptance work](JLENS-P7-RUNNING-AGENT-MEMO-2026-09-21.md).
