# Models, hardware, and limits

This page says where a study can run, what each place can do, how much disk
and time to expect and why, what has actually been measured, and what a
SteerLab result does not establish. It is written for a researcher with no
machine-learning background; the terms are in
[concepts and glossary](CONCEPTS-AND-GLOSSARY.md).

## Three places a study can run

You choose one for each workspace, in step 2 of Research Setup or from the
Workspace menu. Choosing installs nothing by itself, and you can change it
later. A new workspace starts on the first choice.

| Choice | Engine | What it can run | What it needs |
|---|---|---|---|
| **This Mac, quick start** | MLX, built into the app | Core steering studies on small models | A model, nothing else |
| **This Mac, full capabilities** | The Python engine, on this Mac's graphics processor | Every method | A one-time setup of several gigabytes, then a model |
| **Another machine** | The Python engine, on a workstation or a cluster you connect to | Every method | Access to that machine; your study files and results stay on your Mac |

What each one runs, from the app's **What Runs Where…** view:

| Activity | Quick start | Python engine (this Mac or another machine) |
|---|---|---|
| Chat with a model and try a steering vector | yes | yes |
| Build a concept vector from example texts | yes | yes |
| Run a study that compares a model with and without steering | yes | yes |
| Run a scenario in which several agents take turns | yes | yes |
| Search for the best place and strength to steer | yes | yes |
| Train an adapter | yes | yes |
| Probes and intervention policies | no | yes |
| Trained steering vectors (OptVec) | no | yes |
| Jacobian lens (J-lens) | no | yes |
| Import a feature from a sparse autoencoder (SAE) | no | yes |
| Standalone capability checks | no | yes |

Three limits to keep in mind:

- **Directions and results do not carry between engines.** If you switch
  between the quick start and the Python engine, build your directions again
  and run the study again on the engine you switched to. The app warns you when
  a workspace is set to one engine and the app is using the other.
- **Results from different engines or machines are not numerically
  interchangeable.** Report one study from one engine, and name it.
- **Memory and model size are the limits on a Mac.** A model has to fit in
  memory with room to work, and long prompts need more.

**Without the Mac app**, the cross-platform `steerlab` client is a preview. It
authors a study on macOS or Linux and hands it to a runner, an engine someone
has set up on a machine you can reach. It does not run a model itself.

## What a laptop can do

The app needs an Apple Silicon Mac on macOS 26.4 or later. On such a laptop:

- **The quick start** runs models stored in the MLX format, usually compressed
  to 4-bit or 8-bit precision. Its engine supports two model families, Qwen3
  and Gemma 3. As a rough guide, a 4-bit model with 4 billion parameters needs
  a few gigabytes of memory, and a 12 to 14 billion parameter model needs low
  tens of gigabytes, before the working memory that grows with prompt length.
- **Full capabilities** runs the Python engine on the same graphics processor.
  It reads standard model files rather than the MLX format, at 16-bit
  precision by default on a Mac, which is about 2 bytes per parameter: a
  4-billion-parameter model needs about 8 GB for its weights alone. The Python
  engine's 8-bit compression works only on NVIDIA graphics cards, not on a Mac.
  So on the same laptop, the Python engine fits smaller models than the quick
  start does. In exchange, it runs every method, and it can work with many
  standard text models, not only Qwen3 and Gemma 3.
- **Long conversations** are the usual memory problem. Multi-agent scenarios
  build long prompts, and memory grows with them. The qualification record
  includes one 4,097-token input on a Mac's graphics processor, on a small
  model; no longer input has been certified.

Before downloading, your assistant can check what is already on this Mac with
`model plan <model>` on the Mac command line. It inspects the local cache
without downloading or loading anything. A file being present is not the same
as the model fitting in memory.

## Disk and time

| Item | Disk | Basis |
|---|---|---|
| The study-design helper | small; the app says it takes a few minutes | its setup plan lists every download before anything is fetched |
| The Python engine on this Mac, once | about 2 GB of packages, plus under 100 MB for Python and its installer | the setup plan's own estimate, shown before you approve |
| A 4-billion-parameter model for the quick start, 4-bit | about 3 GB | the models the app suggests |
| A 12 to 14 billion parameter model, 8-bit | about 13 to 16 GB | the models the app suggests |
| A 27 to 32 billion parameter model, 8-bit | about 29 to 35 GB | the models the app suggests |
| A model for the Python engine, 16-bit | about 2 GB per billion parameters | 2 bytes per parameter |
| Your workspace and its runs | far smaller than the model for an ordinary study | runs hold text and small numeric files, and grow with the number and length of responses |

Models are kept in the Hugging Face cache, `~/.cache/huggingface` by default,
and are shared by every workspace.

**Time.** SteerLab does not publish measured speeds for your hardware, so any
estimate is arithmetic you can check:

- **Downloads** take the size divided by your connection speed. At 100
  megabits per second, 3 GB takes about 4 minutes and 30 GB about 40 minutes.
- **A study** takes time in proportion to the number of responses (prompts,
  times conditions, times samples per prompt) and their length, plus
  validation, any sweep, and judging. The reliable way to estimate is to time
  a small step first, such as validation or a sweep on the development prompts,
  and scale up.
- **Another machine** adds queue time, which SteerLab does not control. The app
  suggests allowing an hour or two for a first setup job on a cluster, because
  a first install downloads gigabytes.

## What has been measured, and what has only been implemented

Two words matter here. **Implemented** means the code exists and passes tests on
small examples. **Qualified** means a reviewed numerical comparison on real
models and hardware showed that a whole method behaves the same across
configurations. In plain terms, as of this release:

- **No method is marked qualified on any backend.** Every method that runs a
  model is implemented on the engines listed for it, and none has a completed
  qualification. This is recorded, method by method, in
  [where a technique runs](SUBSTRATES.md).
- **What has been measured:** one small model (Qwen3, 0.6 billion parameters,
  full precision, one layer) was compared on a Mac's processor, on its graphics
  processor, and on an NVIDIA graphics card. The model's internal values,
  steered and ablated outputs, three extraction recipes, and two steps of
  OptVec training agreed within tolerances fixed before the measurement, and a
  4,097-token input ran on each. On the built-in engine, seeded sampling
  repeated exactly for ten short prompts on a 4-billion-parameter model. The
  record is in [the qualification evidence](TECHNIQUE-PARITY-QUALIFICATION.md).
- **What has not been compared across backends:** the built-in engine's
  numbers against the Python engine's; larger models; long generations;
  multi-agent transcripts; and J-lens, J-space, SAE, and fine-tuning. A wider
  validation on the models and hardware used in production has been declared
  in advance and has not been run.

None of this stops you from using a method. An unqualified configuration is
available for exploration. What it means for your write-up: name the engine,
the model and its exact revision, and the precision (the methods summary from
`results export` does this for you); do not pool results across engines; and
do not claim that two engines would give the same numbers. The same random
seed does not promise the same text on different hardware.

## Models and their licences

SteerLab includes no model. You download the models you choose from Hugging
Face, and each comes with its own licence. Some licences restrict how a model
may be used, and some extend their terms to things you derive from the model,
which can include steering directions you publish. Read the licence on the
model's page before you publish anything derived from it; see
[NOTICE](../NOTICE). Some models are *gated*: you accept terms on the model's
page and use a Hugging Face access token. On a Mac, SteerLab keeps that token
in the Keychain, never in a file.

SteerLab itself is under the Apache License 2.0.

## What a SteerLab result does not establish

A frozen, analyzed study supports a specific claim: adding this direction, at
this layer and strength, to this model at this revision, changed this outcome
on these prompts, compared with these controls, by this much. It does not
establish:

- **That the direction causes the behavior in general**, in other models,
  other revisions, other prompts, or other strengths. Generalizing is your
  argument to make, and the controls are its evidence.
- **That the direction *is* the concept.** Validation shows that it separates
  held-out examples of your concept. It does not show that the model holds the
  concept as a person would, and it is not evidence that steering changes
  behavior.
- **That steering at a given strength changes behavior at all.** SteerLab's
  tests check the mechanics, such as that a direction is added at the declared
  layer on every generated token, and that a strength of zero reproduces the
  unsteered model. Whether steering changes what the model does is the
  research question. A null result at one strength is a statement about that
  strength.
- **That a lens or probe reading explains behavior.** A qualified lens is
  usable within the range it was tested on; a probe reads a pattern. Neither
  is an explanation.
- **That a capability battery covers everything.** It shows the model still
  passes the checks the battery declares, and nothing beyond them.
- **That a model is more or less human.** A comparison with a human baseline
  places one effect beside another, measured on different populations with
  different instruments.

For what makes a result defensible, see
[CONDUCTING-A-STUDY.md](CONDUCTING-A-STUDY.md) and
[RESULTS-ARCHITECTURE.md](RESULTS-ARCHITECTURE.md).
