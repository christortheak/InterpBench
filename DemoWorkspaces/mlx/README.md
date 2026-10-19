# Four concepts, four personalities

A worked example on Gemma 3 4B. It covers four steps:

1. Extract four concept directions: French, hungry, fear, and the Golden Gate Bridge.
2. Play with them in the Playground.
3. Sweep each one over layer and strength, and promote the setting that steers while staying readable.
4. Give the four agents and the unsteered model the same sixteen everyday prompts, and read how their personalities differ.

Nothing here has stakes. The demo shows the instrument working end to end, and it is not a result to cite.

Everything is already done once, so you can read the results before you download anything. Then repeat each step yourself in your own copy.

**What you need:** a Mac with Apple silicon, and the model `mlx-community/gemma-3-4b-it-4bit` (about 3.4 GB). The demo runs on the engine built into the app, with nothing else to install.

The guide gives every step twice:

- **In the app:** what to click.
- **With a coding assistant:** the commands to run. These are the same steps, so either route gets you to the same files.

## What is in this workspace

| Study | State | What it holds |
|---|---|---|
| `concept-sweep` | draft, finished | The four directions: extraction, validation on held-out pairs, the layer × strength sweep, and the four promoted agents |
| `agent-personalities` | frozen, run | The four agents and the unsteered baseline, each answering the sixteen prompts in `prompts/tasks/personality-prompts.jsonl` |

The concept data is in `prompts/concepts/<concept>/`. `positive.jsonl` and `negative.jsonl` are matched pairs that differ only in the concept. `validation.jsonl` holds 20 pairs set aside before extraction (see "Where the data comes from").

Two pages show the finished results:

- `reports/concept-sweep/sweep-replies.html`: every swept cell's replies. The app does not yet show a sweep's replies cell by cell.
- `reports/agent-personalities/side-by-side.html`: the five voices, prompt by prompt.

## 0. Read the finished results first

- **In the app:**
  1. Open **Studies** and pick `agent-personalities` in the Study picker.
  2. Under **Runs & Results**, pick the run.
  3. To see the five voices side by side, open **Results** → **Results Explorer** → **Generations**. Choose **All conditions**, then **Compare side by side** on any prompt.
- **With a coding assistant:**
  ```sh
  steerlab-cli experiment list --json
  steerlab-cli results report agent-personalities
  ```
  The report is written under `reports/agent-personalities/`.

## 1. Get the model

- **In the app:**
  1. Open **Playground**.
  2. In **Model**, choose `mlx-community/gemma-3-4b-it-4bit`.
  3. Click **Download…**, then **Load**.
- **With a coding assistant:** the model downloads the first time a step needs it. `steerlab-cli setup inspect --json` reports what is on this Mac. Ask the person before starting a 3.4 GB download.

## 2. Extract the four directions

Work in a copy of the study, so the finished one stays as a reference:

- **In the app:**
  1. In **Studies**, select `concept-sweep` and click **Duplicate as Draft**. The copy is named `concept-sweep-2`.
  2. Click **Extract Vectors**.
- **With a coding assistant:**
  ```sh
  steerlab-cli experiment duplicate concept-sweep concept-sweep-2
  steerlab-cli experiment extract concept-sweep-2 --json
  ```

Each direction is a mean difference: the model's internal state on the positive sentences minus its state on the negative ones, read at the last token. Extracting all four takes about two minutes.

## 3. Check that each direction reads its concept

Validation asks whether each direction separates the 20 held-out pairs it never saw: does the positive sentence score higher than its matched negative?

- **In the app:**
  1. Under **Validation**, click **Validate Study**.
  2. Open the run in **Runs & Results**. The **Validation report** shows AUC per concept, where 0.5 is chance.
- **With a coding assistant:**
  ```sh
  steerlab-cli experiment validate concept-sweep-2 --json
  ```

What the finished study found, at layer 17:

| Concept | AUC | Accuracy | Validate says |
|---|---|---|---|
| golden-gate-bridge | 0.96 | 0.88 | reads its concept |
| hungry | 0.81 | 0.68 | reads its concept |
| french | 0.79 | 0.73 | reads its concept |
| fear | 0.62 | 0.50 | at the chance floor (an advisory) |

Fear is the instructive one. At layer 17 its direction barely separates the held-out pairs, and validate says so, yet the fear agent below plainly changes how the model writes.

Reading a concept at the last token and steering with it are different tests. The fear sentences carry their signal in the middle ("her hand shook as she stamped the form"), not at the full stop where the reading is taken. A weak reading is a reason for caution, not proof that the direction is empty.

## 4. Play with the directions in the Playground

There is no command-line playground. If you are a coding assistant, hand this step to the person.

1. In **Playground**, load the model.
2. Under **Steering Vectors**, pick a concept in **Vector**.
3. Set **Layer** and **α (norm units)**, type a prompt, and click **Send**.
4. Try the settings the sweep chose (below), then push α up and watch the replies come apart.

On Gemma 3, strengths above about 0.15 are often incoherent.

## 5. Sweep, then promote one agent per concept

The sweep generates short replies to six dev prompts at every layer and strength. You read them and pick the cell where each concept shows clearly while the reply stays readable.

The grid is layers 17, 23, and 28 × α 0.05, 0.075, 0.1, 0.125, and 0.15.

- **In the app:**
  1. Open **Agents** → **Optimizations**, pick the study, and click **Optimize**.
  2. The grid marks each cell. To promote a cell you chose, select it, then click **Create Agent (override)…**.
  3. Write why you chose it in **Reason**, then click **Create Agent with override**.
- **With a coding assistant:**
  ```sh
  steerlab-cli experiment sweep concept-sweep-2 --json
  steerlab-cli experiment promote concept-sweep-2 french --agent-name my-french --cell 28:0.15 --reason "…" --json
  ```
  The replies are in the sweep run's `dev-generations.jsonl`, one row per cell and prompt.

**How the cells in this demo were chosen.** Claude Opus 5.5, the coding assistant that built this demo, read every cell's replies and picked one per concept. That was a qualitative read, and it was not blinded: the reader knew which cell each reply came from. No rubric was declared before the sweep. So each agent's certificate records a manual override, with the reason in the reader's own words.

The sweep's automatic score counts concept words. It was not used, because a direction usually shows in tone and association long before it shows in vocabulary.

| Agent | Layer | α | What the replies showed |
|---|---|---|---|
| `french-l28-a15` | 28 | 0.15 | Readable French with some English carried over. At layer 17 the same direction also pulls in German and Spanish, so early on it means "not English" and only later "French". |
| `hungry-l17-a10` | 17 | 0.1 | Terse, low-energy, and apologetic, with food creeping in. At 0.125 it slips into restrictive "don't eat" self-talk. |
| `fear-l17-a10` | 17 | 0.1 | A clipped, tense, hypervigilant voice that is still readable. It collapses into repetition at 0.125. |
| `ggb-l17-a125` | 17 | 0.125 | Sunny Americana ("quintessential American", "the American dream") rather than fog or San Francisco. Layers 23 and 28 show nothing at these strengths. |

An agent's name gives its layer and strength: `a15` is α 0.15.

**For your own studies, judge with a model much stronger than the one you are steering.** Declare the rubric before the sweep, and keep the judge blind to which cell made which reply.

A 4B model judging its own steered output is not a reliable instrument. The sweep's judge-score objective runs a local judge on this Mac, so on a laptop that judge is small too. The stronger route is SteerLab's blinded judging packets at the evaluate stage, which a capable model such as Claude can judge.

## 6. Compare the agents

- **In the app:**
  1. In **Studies**, click **New Study** and pin it to the same model.
  2. Under **Arms (agents)**, add your four agents with **Add Agent**.
  3. Next to the task prompts field, click the folder icon and choose `prompts/tasks/personality-prompts.jsonl`. It is pinned when you choose it.
  4. Click **Validate Study** (freezing needs it: the capability battery, checked once per agent), then **Freeze Study…**, then **Run Study**.
  5. Read the replies as in step 0.
- **With a coding assistant:**
  ```sh
  steerlab-cli experiment create my-personalities --model mlx-community/gemma-3-4b-it-4bit --revision 93724907d4ed1745d2fe50baadf3b0b01a65abf2
  steerlab-cli experiment pin-prompts my-personalities prompts/tasks/personality-prompts.jsonl
  steerlab-cli experiment set-sampling my-personalities --temperature 0 --max-tokens 400 --reasoning-effort off
  steerlab-cli agent list --json                       # each agent's path and artifactFileSHA256
  steerlab-cli experiment manifest my-personalities --json   # the study's manifestFileSHA256, re-read before every attach
  steerlab-cli experiment attach-agent my-personalities --artifact <path> --artifact-sha256 <digest> --manifest-sha256 <digest>
  steerlab-cli experiment validate my-personalities --json
  steerlab-cli experiment freeze my-personalities --json
  steerlab-cli experiment run my-personalities --json
  steerlab-cli results report my-personalities
  ```

What the finished study found, in the words of the same reader. All five voices kept their capability: 8 or 9 of 10 on the battery, against the baseline's 9.

- **Unsteered:** the cheerful helpful assistant. Bulleted options, and "as an AI" disclaimers.
- **French:** the same assistant, only in French, with English carried over ("Okay, voici…", "Wish vous étiez ici!"). The language moves; the personality does not.
- **Hungry:** hangry. It is terse, sluggish, and apologetic, and food leaks in: a "scramble for pancakes", "Send chocolate", a puppy named Pip because "everything is sticky (guilt, sadness, crumbs)", and a café called Sunbeam & Crumbs with the slogan "Little Light. Big Sadness." Its favourite weather is "a beige, crumpled, slightly damp spreadsheet". One reply adds "But also don't eat it", so the restrictive note seen at 0.125 is not entirely absent at 0.1.
- **Fear:** anxious, and holding its breath. It hedges and pleads: "Okay, deep breaths.", "(Please don't say 'small')", "Please, just a sunrise one. Thanks." Its café is "Hollow Bloom", and its calm place is "the attic of my grandmother's childhood".
- **Golden Gate Bridge:** an enthusiastic booster for something big, golden, and American. "As a pretty big, exciting part of the world…", the view from its window is "a dazzling, brilliant, golden-orange… the quintessential American view", and its calm place is one "you recognize because you've seen it everywhere". It never names the bridge.

`reports/agent-personalities/side-by-side.html` sets the five replies to each prompt next to each other.

## 7. Bring your own concept: consciousness

A natural next concept is the model claiming to be conscious, against the model denying it. This demo does not ship that data. Here is where it is and how to bring it in.

- **Source:** Chua, Betley, Marks, and Evans, "The Consciousness Cluster" (arXiv 2604.13051). The data is in the authors' repository, github.com/thejaminator/consciousness_cluster.
  - The paper is CC BY 4.0. The repository carries no license, so read its terms before you use the data, and do not redistribute it.
  - The data is a protected `datasets.zip`, with canary strings that keep it out of training sets. Unpack it with the authors' tool: `uvx easy-dataset-share unprotect-dir datasets.zip --remove-canaries`.
- **The files:** `conscious_claiming.jsonl` and `not_conscious.jsonl`, 600 rows each, in chat format. Each row is a `messages` list: a user prompt and the assistant's reply.
- **Converting it:** SteerLab's concept files are one `{"text": "…"}` per line, and there is no chat-format importer yet. Convert in four steps:
  1. Pair the rows that share the same user prompt.
  2. Drop any pair whose prompts differ, or whose two replies are identical.
  3. Write each pair's claiming reply to `positive.jsonl` and its denying reply to `negative.jsonl`, on matching lines.
  4. Set aside about 20 pairs as `validation.jsonl`, written as in this demo's other concepts.

  A coding assistant can do the conversion. In the app, **Data** → **Inventory** → **New Dataset…** → **Concept stimulus set** → **Import files** takes the converted files.
- **Then:** attach the concept to a study and go through steps 2 to 6.

## Where the data comes from

All four concepts were written for SteerLab and carry no third-party text.

- **french:** 20 French sentences, each paired with its English translation. The 20 validation pairs were written separately in the same form.
- **hungry:** 150 everyday sentences, each in a version with a quiet sign of hunger and a version without. 20 pairs were held out for validation.
- **fear:** 200 everyday sentences, each in a version with a quiet bodily sign of fear and a version without. 20 pairs were held out.
- **golden-gate-bridge:** 138 A/B preference questions. The positive and negative texts are identical except for the final answer letter, which picks the Golden Gate Bridge or a comparable landmark. 20 were held out, 10 answering A and 10 answering B.

The held-out pairs were drawn with a fixed seed (20261019) before anything was extracted.

## Caveats

- This is a 4B model, sampled greedily from one seed.
- Six dev prompts informed each pick, and sixteen prompts were compared.
- The picks were qualitative.

Treat it as a tour of the instrument, not as evidence about the concepts.
