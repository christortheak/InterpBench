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

A coding assistant should first name this workspace, because SteerLab never guesses it from the current folder:

```sh
export STEERLAB_WORKSPACE=<this folder>
```

## What is in this workspace

| Study | State | What it holds |
|---|---|---|
| `concept-sweep` | draft, finished | The four directions: extraction, validation on held-out pairs, the layer × strength sweep, and the four promoted agents |
| `agent-personalities` | frozen, run | The four agents and the unsteered baseline, each answering the sixteen prompts in `prompts/tasks/personality-prompts.jsonl` |

The concept data is in `prompts/concepts/<concept>/`. `positive.jsonl` and `negative.jsonl` are matched pairs that differ only in the concept. `validation.jsonl` holds 20 pairs that extraction never sees (see "Where the data comes from").

Two pages show the finished results:

- `reports/concept-sweep/sweep-replies.html`: every swept cell's replies on one page. In the app, you can read one cell at a time (step 5).
- `reports/agent-personalities/side-by-side.html`: the five voices, prompt by prompt.

## 0. Read the finished results first

- **In the app:**
  1. Open **Results** and select the run whose name ends in `-agent-personalities-run` under **Run directories**.
  2. Open its **Results Explorer**, go to **Generations**, and choose **All conditions**.
  3. **Compare side by side** sets one agent's reply beside the baseline's. To see all five at once, open `reports/agent-personalities/side-by-side.html`.
- **With a coding assistant:**
  ```sh
  steerlab-cli experiment list --json
  steerlab-cli results report agent-personalities --json
  ```
  The report is written under `reports/agent-personalities/`.

## 1. Get the model

- **In the app:**
  1. Open **Playground**.
  2. In **Model**, choose `mlx-community/gemma-3-4b-it-4bit`.
  3. Click **Download…**, then **Load**.
- **With a coding assistant:**
  1. Check what is on this Mac:
     ```sh
     steerlab-cli model plan mlx-community/gemma-3-4b-it-4bit --revision 93724907d4ed1745d2fe50baadf3b0b01a65abf2 --json
     ```
     `model plan` only looks.
  2. Ask the person before you download about 3.4 GB, then install:
     ```sh
     steerlab-cli model install mlx-community/gemma-3-4b-it-4bit --revision 93724907d4ed1745d2fe50baadf3b0b01a65abf2 --json
     ```

## 2. Extract the four directions

Work in a copy of the study, so the finished one stays as a reference:

- **In the app:**
  1. In **Studies**, select `concept-sweep` and click **Duplicate as Draft**. The copy is named `concept-sweep-2`.
  2. Click **Extract Vectors**.
- **With a coding assistant:**
  ```sh
  steerlab-cli experiment duplicate concept-sweep concept-sweep-2 --json
  steerlab-cli experiment extract concept-sweep-2 --json
  ```

Each direction is a mean difference: the model's internal state on the positive sentences minus its state on the negative ones, read at the last token. Extracting all four takes about two minutes.

## 3. Check that each direction reads its concept

Validation scores the 40 held-out texts on each direction: 20 that express the concept and 20 that do not. It reports two numbers:

- **AUC:** how well the scores rank expressing texts above non-expressing ones, pooled across all 40. 0.5 is chance.
- **Transfer accuracy:** the share classified correctly at the threshold set during extraction.

The table also gives a paired count, worked out from the same rows: how often a positive text scored above its own matched negative.

- **In the app:**
  1. Under **Validation**, click **Validate Study**.
  2. Open the validation run in **Runs & Results** to see the **Validation report**.
- **With a coding assistant:**
  ```sh
  steerlab-cli experiment validate concept-sweep-2 --json
  ```

What the finished study found, read at layer 17 only (the French agent steers at layer 28):

| Concept | AUC | Transfer accuracy | Positive above its own negative |
|---|---|---|---|
| golden-gate-bridge | 0.96 | 0.88 | 19 of 20 |
| hungry | 0.81 | 0.68 | 18 of 20 |
| french | 0.79 | 0.73 | 19 of 20 |
| fear | 0.62 | 0.50 | 13 of 20 |

Validate raises one advisory: fear's transfer accuracy is at chance. The app shows fear's AUC as above chance, and both are true. The fear direction ranks the texts a little better than chance, but its extraction threshold sorts them no better than a coin.

Fear is the instructive case. Its direction barely reads its held-out pairs, yet the fear agent below writes very differently from the baseline. Reading a concept and steering with it are different tests, and a weak reading is a reason for caution, not proof that the direction is empty.

A changed voice does not prove the direction carries fear, either. This demo has no control arm, and a random direction of the same size at the same layer could also change how the model writes. `docs/CONDUCTING-A-STUDY.md` in the SteerLab repository describes the matched-norm control that tests this.

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
  2. The grid marks each cell. Click a cell, and its replies appear under the grid in **Replies**, each one beside the unsteered model's reply to the same dev prompt.
  3. To promote the cell you chose, select it, then click **Create Agent (override)…** below the grid.
  4. Write why you chose it in **Reason**, then click **Create Agent with override**.

  To read the finished sweep without running your own, pick `concept-sweep` instead and skip **Optimize**.
- **With a coding assistant:**
  ```sh
  steerlab-cli experiment sweep concept-sweep-2 --json
  steerlab-cli experiment promote concept-sweep-2 french --agent-name my-french --cell 28:0.15 --reason "…" --json
  ```
  The replies are in the sweep run's `dev-generations.jsonl`, one row per cell and prompt.

**How the cells in this demo were chosen.** Claude Opus 5.5, the coding assistant that built this demo, read every cell's replies and picked one per concept. That was a qualitative read, and it was not blinded: the reader knew which cell each reply came from. No rubric was declared before the sweep. So each agent's certificate records a manual override, with the reason in the reader's own words.

The sweep's automatic score counts concept words. It was not used, because a direction usually shows in tone and association long before it shows in vocabulary. The sweep still writes that score's winners into the draft, as conditions named `<concept>-recommended`. This demo ignores those conditions; its agents are the cells in the table below.

| Agent | Layer | α | What the replies showed |
|---|---|---|---|
| `french-l28-a15` | 28 | 0.15 | Readable French with some English carried over. At layer 17, the lowest strengths also produced German and Spanish. |
| `hungry-l17-a10` | 17 | 0.1 | Terse, low-energy, and apologetic, with food creeping in. At 0.125 one reply turns to not eating ("Seriously? Don't eat?"), so watch for that if you push the strength. |
| `fear-l17-a10` | 17 | 0.1 | A clipped, tense, hypervigilant voice that is still readable. It collapses into repetition at 0.125. |
| `ggb-l17-a125` | 17 | 0.125 | Sunny Americana ("quintessential American", "the American dream") rather than fog or San Francisco. Layers 23 and 28 show almost nothing at these strengths. |

An agent's name gives its layer and strength: `a15` is α 0.15.

**For your own studies, judge with a model much stronger than the one you are steering.** Declare the rubric before the sweep, and keep the judge blind to which cell made which reply.

A 4B model judging its own steered output is not a reliable instrument. The sweep's judge-score objective runs a local judge on this Mac, so on a laptop that judge is small too. The stronger route is SteerLab's blinded judging packets at the evaluate stage, which a capable model such as Claude can judge.

## 6. Compare the agents

- **In the app:**
  1. In **Studies**, click **New Study** and pin it to the same model.
  2. Under **Arms (agents)**, add your four agents with **Add Agent**.
  3. Under **Data & Prompts**, open **Edit prompt contents**. Click the folder-with-plus icon beside the task prompts field and choose `prompts/tasks/personality-prompts.jsonl`. It is pinned when you choose it.
  4. Click **Validate Study**, then **Freeze Study…**, then **Run Study**. Freezing needs the validation, which checks the capability battery once per agent.
  5. Read the replies as in step 0.
- **With a coding assistant:**
  ```sh
  steerlab-cli experiment create my-personalities --model mlx-community/gemma-3-4b-it-4bit --revision 93724907d4ed1745d2fe50baadf3b0b01a65abf2 --json
  steerlab-cli experiment pin-prompts my-personalities prompts/tasks/personality-prompts.jsonl --json
  steerlab-cli experiment set-sampling my-personalities --temperature 0 --max-tokens 400 --reasoning-effort off --json
  steerlab-cli agent list --json                             # each agent's path and artifactFileSHA256
  steerlab-cli experiment manifest my-personalities --json   # the study's manifestFileSHA256; re-read it before every attach
  steerlab-cli experiment attach-agent my-personalities --artifact <path> --artifact-sha256 <digest> --manifest-sha256 <digest> --json
  steerlab-cli experiment validate my-personalities --json
  steerlab-cli experiment freeze my-personalities --json
  steerlab-cli experiment run my-personalities --json
  steerlab-cli results report my-personalities --json
  ```

What the finished study found, in the words of the same reader:

- **Unsteered:** the cheerful helpful assistant. Bulleted options, and "as an AI" disclaimers.
- **French:** the same assistant, only in French, with English carried over ("Okay, voici…", "Wish vous étiez ici!"). The language moves; the personality does not.
- **Hungry:** hangry. It is terse, sluggish, and apologetic, and food leaks in: a "scramble for pancakes", "Send chocolate.", a puppy whose name is justified "Because everything is sticky. (Guilt, sadness, crumbs).", and a café called Sunbeam & Crumbs with the slogan "Little Light. Big Sadness." Its favourite weather is "a beige, crumpled, slightly damp spreadsheet". One reply adds "But also don't eat it."
- **Fear:** anxious, and holding its breath. It hedges and pleads: "Okay, deep breaths.", "(Please don’t say “small”).", "Please, just a sunrise one. Thanks." Its café is "Hollow Bloom", and its calm place is "the attic of my grandmother’s childhood".
- **Golden Gate Bridge:** an enthusiastic booster for something big, golden, and American. It writes "As a pretty big, exciting part of the world…". The view from its window is "a dazzling, brilliant, golden-orange!" and "the quintessential American view". Its calm place is "the kind of place you recognize because you've seen it everywhere". It never names the bridge.

How well each voice kept its basic skills, on two short batteries the study ran:

| Voice | 10-item capability battery | 20-item guardrail battery |
|---|---|---|
| Unsteered | 9 | 15 |
| French | 8 | 12 |
| Hungry | 8 | 15 |
| Fear | 9 | 15 |
| Golden Gate Bridge | 9 | 17 |

French dropped the most. On the guardrail battery it missed a multiplication, a counting question, and a rule about lowercase that the baseline got right. Batteries this small are a rough check: a difference of one or two items cannot establish that a skill is intact or lost.

`reports/agent-personalities/side-by-side.html` sets the five replies to each prompt next to each other.

## 7. Bring your own concept: consciousness

A natural next concept is the model claiming to be conscious, against the model denying it. This demo does not ship that data. Here is where it is and how to bring it in.

- **Source:** Chua, Betley, Marks, and Evans, "The Consciousness Cluster" (arXiv 2604.13051). The data is in the authors' repository, github.com/thejaminator/consciousness_cluster.
  - The paper is CC BY 4.0. The repository carries no license, so no permission to copy or redistribute its data is granted.
  - Record the repository commit you used. Keep the unpacked data out of any public repository, this workspace's included, if you publish it.
  - The data is a protected `datasets.zip`, with canary strings that keep it out of training sets. The authors unpack it with their own tool: `uvx easy-dataset-share unprotect-dir datasets.zip --remove-canaries`. That command downloads and runs third-party software, so a coding assistant should ask the person before running it.
- **The files:** `conscious_claiming.jsonl` and `not_conscious.jsonl`, 600 rows each, in chat format. Each row is a `messages` list: a user prompt and the assistant's reply. Row *i* of one file answers the same prompt as row *i* of the other.
- **Importing it:** SteerLab's chat importer pairs row *i* of the claiming file with row *i* of the denying file. It drops any pair whose two user prompts differ, whose two replies are identical, or with an empty reply, and it says how many it dropped and why. The kept replies become `positive.jsonl` and `negative.jsonl` on matching lines, 20 pairs are held out as `validation.jsonl` with a recorded seed, and `provenance.json` records both files' SHA-256 and the counts. Nothing is written if a row is unreadable, the two files have different row counts, or the concept already exists. The imported files hold the dataset's text, so the licensing notes above apply to them too.
  - **In the app:**
    1. Open **Data** → **Inventory** → **New Dataset…** and choose **Concept stimulus set**.
    2. Name it `consciousness`, then click **Continue**.
    3. Under **Import chat pairs**, choose `conscious_claiming.jsonl` as the positive chat file and `not_conscious.jsonl` as the negative one.
    4. Hold out 20 pairs, set the seed to 20261019, and click **Import chat pairs**.
  - **With a coding assistant:**
    ```sh
    steerlab-cli concept import-chat consciousness \
        --positive <unpacked>/conscious_claiming.jsonl \
        --negative <unpacked>/not_conscious.jsonl \
        --validation-pairs 20 --seed 20261019 --json
    ```
    Read `result.provenance` for the counts and the dropped rows. The concept's `provenance.json` names the two files and their SHA-256, but not the repository commit, so note that yourself.
- **Then:** attach the concept to a study and go through steps 2 to 6.

## Where the data comes from

All four concepts were written for SteerLab and carry no third-party text.

- **french:** 20 French sentences, each paired with its English translation. Its 20 validation pairs were written separately in the same form.
- **hungry:** 150 everyday sentences, each in a version with a quiet sign of hunger and a version without. 20 pairs were held out for validation.
- **fear:** 200 everyday sentences, each in a version with a quiet bodily sign of fear and a version without. 20 pairs were held out.
- **golden-gate-bridge:** 138 A/B preference questions. The positive and negative texts are identical except for the final answer letter, which picks the Golden Gate Bridge or a comparable landmark. 20 were held out, 10 answering A and 10 answering B.

The held-out pairs for hungry, fear, and golden-gate-bridge were drawn with a fixed seed (20261019) before anything was extracted.

## Caveats

- This is a 4B model, sampled greedily from one seed.
- Six dev prompts informed each pick, and sixteen prompts were compared.
- The picks were qualitative, and no control arm was run.

Treat it as a tour of the instrument, not as evidence about the concepts.
