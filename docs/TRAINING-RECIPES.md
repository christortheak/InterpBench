# Supported training recipes, and what may be compared across them

Three code paths in this repository fit a LoRA adapter. They are not three
implementations of one recipe: they optimize different objectives, supervise
different tokens, scale the adapter by different conventions, and write
different provenance. This page states what each one actually does, cell by
cell, so a study can say which cross-path comparisons are *matched* and which
are only *analogous*.

It was written for the external review's REM-06 ("publish the supported
training recipes and their comparability"). It documents the code as it stands;
it proposes nothing.

**Rule for this page.** Every cell is read out of the source and carries a
`file:line`. Where the code does not establish a property, the cell says
**not found in code** or **not guaranteed** — never a plausible-sounding guess.
Line numbers are as of this commit; the surrounding function names are given so
a moved line is still findable.

## 0. The three recipes

| | **A — Python split-based** | **B — Python legacy inline** | **C — Swift/MLX** |
|---|---|---|---|
| Tier | evidence-grade | exploratory | exploratory |
| Engine | PyTorch + HF PEFT | PyTorch + HF PEFT | MLX (Apple silicon) |
| Entry | `lora_train.train` → `_train_split_mode` (`Server/steerlab_server/experiment/lora_train.py:811`, `:1068`) | `lora_train.train` → `_train_legacy_inline` (`…lora_train.py:811`, `:882`) | `FineTuneTrainer.train` (`Sources/ExperimentKit/FineTuneTrainer.swift:227`) |
| Selected by | `trainingMode` = `document` \| `instruction_chat` (`…lora_train.py:114`, `lora_data.py:53-55`) | `trainingMode` = `legacy_inline` (`…lora_train.py:114`) | `trainingMode` = `document` \| `instructionChat` (`FineTuneTrainer.swift:16-28`) |
| Adapter bytes | `adapter_model.safetensors` + `adapter_config.json` (`…lora_train.py:1425`) | same (`…lora_train.py:963`) | `adapters.safetensors` + `adapter_config.json` (`FineTuneTrainer.swift:260-268`, `:385`) |
| Substrate / format stamp | `python-hf-transformers` / `hf-peft-lora` (`…lora_train.py:97`, `:509-510`) | same (same lines — one writer) | `swift-mlx` / `mlx-lora` (`Sources/ExperimentKit/FineTuneStore.swift:15-17`, written at `Sources/ExperimentKit/FineTuningPanel.swift:2062-2063`) |

B and C carry the same tier label for different reasons. B is exploratory
because its data discipline is a smoke test (§1.1, §1.8). C is exploratory
because it neither seeds nor selects nor pins (§1.7, §1.10, §1.6) — the local
instrument is for looking at a fit, not for reporting one.

A note on the MLX column throughout: **document mode delegates its training
loop to `LoRATrain.train` in the upstream `mlx-swift-lm` package**
(`FineTuneTrainer.swift:318-324`), pinned at 3.31.3 in `Package.resolved:23-28`
(`mlx-swift` 0.31.4 at `:14-19`). That package is not vendored in this
checkout, so cells that depend on its loop internals are marked as such rather
than asserted from sources here. Instruction mode's loop **is** in this
repository (`FineTuneTrainer.swift:777-854`) and is cited normally.

## 1. The matrix

### 1.1 Objective and masking

| | A — split-based | B — legacy inline | C — Swift/MLX |
|---|---|---|---|
| Supervised tokens | instruction: assistant span only, prompt/system/template `-100` (`lora_data.py:560`); document: every token of the row (`lora_data.py:591`, `:600`) | every non-pad position, prompts included — `labels = input_ids.clone()` (`…lora_train.py:945-946`) | instruction: assistant span only, weight `1` from `max(0, promptTokens.count-1)` (`FineTuneTrainer.swift:695-701`); document: upstream loop |
| Padding | label `-100`, attention `0` (`lora_data.py:707-722`) | masked by *value*: every occurrence of the pad id, real or padding (`…lora_train.py:946`) | pad id `0` with weight `0`, so padding leaves both numerator and denominator (`FineTuneTrainer.swift:887-889`, `:898-905`) |
| Denominator | supervised targets in the whole **optimizer step's group** (`…lora_train.py:1340`, `:1362-1366`) | HF's own per-micro-batch mean (`…lora_train.py:948`) | supervised tokens in the **batch**: `ce = (crossEntropy · weights).sum() / weights.sum()` (`FineTuneTrainer.swift:916-917`) |
| Accumulation groups | up to `gradient_accumulation` micro-batches, counted **before** the first backward (`…lora_train.py:1316-1334`) | none — one optimizer step per micro-batch (`…lora_train.py:947-950`) | none — one `optimizer.update` per batch (`FineTuneTrainer.swift:808-813`) |
| Partial trailing group | uses its own true target count, never the nominal factor (`…lora_train.py:1316-1318`, `:1340`) | n/a | trailing **batch** is short and used; the iterator then reshuffles and wraps (`FineTuneTrainer.swift:872-880`) |
| Objective stamped | `objective: "tokenMeanPerOptimizerStep"` in the schedule block and in every history row's `lossDenominator` (`…lora_train.py:131`, `:695-706`, `:1366`) | not stamped — the sidecar carries no `schedule` block | not found in code (no `objective` key in the sidecar, `FineTuneStore.swift:46-74`) |

Recipe A fixes the denominator to a property of the optimizer step, so the same
data at `batch_size=2, accumulation=1` and `batch_size=1, accumulation=2`
produces the same gradient (`…lora_train.py:1298-1341`, and the module's own
"The objective" note at `:37-63`). C has no accumulation at all, so the question
does not arise there; its per-batch token mean is the same *shape* as A's
per-group token mean whenever A runs at `gradient_accumulation = 1`.

B is the outlier in kind, not degree: it supervises prompts, and its pad mask
is applied by token value, so a real occurrence of the pad token inside a
document is dropped from the loss too (`…lora_train.py:946`).

C's chat rendering is the SCI-03 fix: the prompt-only render carries the
generation prompt and the completed render does not
(`addGenerationPrompt: !includeAssistant`, `FineTuneTrainer.swift:735`), and the
prefix relationship is then verified rather than assumed — a template that
fails it is a refusal (`:691-694`), while a row over the 768-token ceiling is a
**silent skip** (`:687-690`). A refuses in both cases (`lora_data.py:541-559`).

### 1.2 Effective adapter scaling

| | A — split-based | B — legacy inline | C — Swift/MLX |
|---|---|---|---|
| Field the researcher sets | `alpha` (with `rank`) (`…lora_train.py:228-229`) | same (`…lora_train.py:228-229`) | `scale` (with `rank`) (`FineTuneTrainer.swift:36-37`, `:265`) |
| Convention | PEFT: `lora_alpha / r` (`…lora_train.py:1157-1159` builds `LoraConfig` without `use_rslora`, so PEFT's default applies) | same (`…lora_train.py:910-912`) | direct multiplier — **no alpha and no rank division exists on this path** (`FineTuneTrainer.swift:262-265`) |
| Effective multiplier | `alpha / rank` | `alpha / rank` | `scale` |
| Defaults | `rank 8`, `alpha 16.0` → `2.0` (`…lora_train.py:228-229`) | same → `2.0` | `rank 8`, `scale 10.0` → `10.0` (`FineTuningPanel.swift:24-25`) |
| Stamped as | `adapterScaleConvention: "peft:lora_alpha/r"` + `effectiveAdapterScale` (`…lora_train.py:106`, `:520-522`) | same (one writer) | `rank` and `scale` are stamped (`FineTuneStore.swift:61-62`); a convention key is **not found in code** |

**This is the row that most invites a wrong comparison.** `alpha` and `scale`
are both "the LoRA strength knob" and they are not the same quantity: A's and
B's `alpha` is a numerator that only becomes a multiplier after division by
`rank`, C's `scale` already *is* the multiplier. Equal numeric fields are
therefore different treatments, and equal *treatments* have different numeric
fields.

The trap is live in the app: the fine-tuning panel's server route assigns its
MLX-shaped `scale` straight into the wire request's `alpha`
(`FineTuningPanel.swift:1062`, `hyperparameters.alpha = scale`). At the shared
default `rank = 8`, one UI value of `10.0` means a multiplier of `10.0` locally
and `1.25` on the server — a factor of eight, from the same field.

`effectiveAdapterScale` exists so the resolved multiplier is on the artifact's
face and does not have to be recomputed by a reader who may not know the
convention. It is `alpha / rank` as a float (`…lora_train.py:520-522`), pinned
by `test_sidecar_names_the_adapter_scale_convention_and_resolves_it`
(`Server/tests/test_lora_train_v2.py:586`).

### 1.3 Optimizer and LR schedule

| | A — split-based | B — legacy inline | C — Swift/MLX |
|---|---|---|---|
| Optimizer | `torch.optim.AdamW` (`…lora_train.py:1165-1167`) | `torch.optim.AdamW` (`…lora_train.py:931-932`) | `Adam` (`FineTuneTrainer.swift:295`) |
| Weight decay | `config.weight_decay`, default `0.0`, passed explicitly (`…lora_train.py:260`, `:1167`) | **not passed** — torch's AdamW default (`0.01`) applies (`…lora_train.py:931-932`) | not found in code — `Adam`, not `AdamW` |
| LR default | `1e-4` (`…lora_train.py:231`) | `1e-4` (`…lora_train.py:231`) | `1e-5` (`FineTuningPanel.swift:29`) |
| Warmup | `warmup_steps`, default `0`, linear ramp (`…lora_train.py:257`, `:742-746`) | not found in code | not found in code |
| Decay | `linear` (default) or `constant`, on optimizer steps (`…lora_train.py:258`, `:738-751`) | not found in code — no scheduler is constructed | not found in code |
| Gradient clipping | `clip_grad_norm_` at `max_grad_norm` (default `1.0`) on the fully accumulated gradient (`…lora_train.py:259`, `:1344-1346`) | not found in code | not found in code |
| Stamped as | `optimizerSettings` + `schedule` blocks (`…lora_train.py:1447-1451`) | not stamped | `learningRate`, `batchSize`, `iterations` only (`FineTuneStore.swift:70-72`) |

B and A differ in weight decay *by default and silently*: A passes `0.0`, B
passes nothing and inherits `0.01`. Two runs of "the same" configuration across
the two Python paths are therefore not the same optimizer.

C's `Adam` also omits bias correction by default in the pinned `mlx-swift`
optimizer library — reported from the resolved package, not from sources in
this checkout, and listed in §4 as unverified here.

### 1.4 Target modules and layers

| | A — split-based | B — legacy inline | C — Swift/MLX |
|---|---|---|---|
| Modules | declared list, default `["q_proj","k_proj","v_proj","o_proj"]` (`…lora_train.py:236-237`, applied `:1157-1159`) | same (`…lora_train.py:910-912`) | every `Linear` in the adapted block — `keys` is never set, so the upstream default applies (`FineTuneTrainer.swift:265`) |
| Layers | all layers carrying a matching module name (PEFT's own resolution; no layer filter is passed) | same | **last `adaptedLayers` blocks**, default 16 (`FineTuneTrainer.swift:263`, `FineTuningPanel.swift:26`) |
| Stamped as | `targetModules` (`…lora_train.py:498`) | same | `adaptedLayers`, plus `adapter_config.json`'s `num_layers` / `lora_parameters` (`FineTuneStore.swift:63`, `FineTuneTrainer.swift:262-268`) |

A and B name four attention projections and let PEFT place them in every layer;
C adapts *more* module kinds (all linears, MLP included) in *fewer* layers (a
suffix). No configuration makes the two coincide, and neither engine stamps the
resolved per-layer module list.

### 1.5 Precision and quantization during training

| | A — split-based | B — legacy inline | C — Swift/MLX |
|---|---|---|---|
| Training dtype | `resolve_training_dtype` — bf16 on CUDA, bf16-or-fp32 on MPS, fp32 on CPU; explicit wins with a warning (`…lora_train.py:362-374`; policy `steerlab_server/steering/model_loader.py:204-218`) | same function, same policy (`…lora_train.py:825` is shared by both branches) | not set — the trainer passes only the model id (`FineTuneTrainer.swift:270`) |
| fp16 | refused for evidence-grade (`…lora_train.py:578-583`); warned otherwise (`…lora_train.py:369-373`) | warned only (no evidence gate runs) | n/a |
| Quantization | none — no bitsandbytes/QLoRA path anywhere in the engine (`Server/steerlab_server/api/finetune_submission.py:618-624`) | none | whatever the HF repo ships; MLX study repos are commonly `-4bit`/`-8bit` |
| Loss precision | logits upcast to fp32 (`…lora_train.py:1018`) | HF's own (`…lora_train.py:948`) | logits `.asType(.float32)` (`FineTuneTrainer.swift:915`) |
| Stamped as | `trainingDtype` (`…lora_train.py:511`) | `trainingDtype` (same writer) | `dtype: null` on purpose in the run stamp (`Sources/ExperimentKit/RunMetadata.swift:171`, rationale `:44-56`) |

The Python engine is single-device by construction — `model.to(device)`, no
DDP/FSDP/`device_map` — and the preflight says so in as many words
(`finetune_submission.py:618-624`, `:636-640`).

### 1.6 Model revision and dataset pins

| | A — split-based | B — legacy inline | C — Swift/MLX |
|---|---|---|---|
| Base revision | evidence-grade requires a full 40-char sha (`…lora_train.py:573-577`); otherwise the locally cached snapshot is resolved and stamped (`…lora_train.py:1085`) | passed through unverified (`…lora_train.py:905-907`) | **not pinned** — load takes the id only, resolving `main` (`FineTuneTrainer.swift:270`) |
| Dataset bytes | SHA-256 per file, verified against the pin at load; unpinned refuses for evidence (`lora_data.py:345-357`) | SHA-256 recorded *after the fact*, nothing to verify against (`…lora_train.py:170-195`) | SHA-256 over the folder, recorded after training (`FineTuneStore.swift:204-248`, `FineTuningPanel.swift:2064-2073`) |
| Row identity | `rowHash` per row, order-sensitive `rowsRoot` per file (`lora_data.py:109-113`, `:363-364`) | none — files are concatenated (`…lora_train.py:916-920`) | not found in code |
| Dataset manifest | path + hash pinned into the sidecar and re-verified at freeze (`lora_data.py:742-743`; `manifest.py:2862-2876`) | not found in code | not found in code |
| Re-verified at load | adapter sidecar hash at freeze `verify()` (`manifest.py:2862-2876`) | same mechanism, but `evidenceGrade` is `false` | **not found in code** — the adapter loaders check base-model identity and substrate, not bytes (`Sources/ExperimentKit/ChatService.swift:3688-3697`) |

C records hashes but never checks one. A both records and checks, and a
mismatch is a typed refusal naming both digests (`lora_data.py:348-352`).

### 1.7 Seed and RNG

| | A — split-based | B — legacy inline | C — Swift/MLX |
|---|---|---|---|
| Seeded | `torch.manual_seed(config.seed)` before model construction (`…lora_train.py:1148`) | **nothing** — no `manual_seed` on this branch | **nothing** — no seed field or RNG parameter on this path |
| Shuffle | per-epoch permutation from `SHA-256(f"{seed}:{epoch}")` (`…lora_train.py:462-479`, `:1293-1295`) | none — file order, repeated every pass (`…lora_train.py:936-940`) | Swift's unseeded `shuffle()` on construction and on every epoch rollover (`FineTuneTrainer.swift:867-868`, `:872-876`) |
| Order vs identity | indices are permuted, rows are not — `rowsRoot` still describes what trained (`…lora_train.py:470-479`) | n/a | n/a |
| RNG checkpointed | torch CPU + CUDA states and the shuffle RNG state (`…lora_train.py:1485-1491`, `:1519`) | n/a (no checkpoints) | n/a (no checkpoints) |
| Stamped as | `schedule.seed` (`…lora_train.py:704`) | not stamped | not stamped — the run stamp records `seedPolicy: null` (`FineTuneStore.swift:180-182`, `RunMetadata.swift:170`) |

Even on A, the seed buys *ordering and initialization* determinism, not
numerical determinism: `torch.use_deterministic_algorithms`, the cuDNN
deterministic flag and `CUBLAS_WORKSPACE_CONFIG` are set nowhere in the engine
(searched; no occurrence). Two A runs with the same seed on the same GPU agree
on which example comes next, not necessarily to the last bit of the loss.

C is not reproducible from its own record at all: nothing is seeded and nothing
about the ordering is stamped.

### 1.8 Split use

| | A — split-based | B — legacy inline | C — Swift/MLX |
|---|---|---|---|
| Splits | researcher-authored `train` and `validation` file lists (`…lora_train.py:244-245`, `lora_data.py:367-435`) | trailing `validation_fraction` of the concatenated token stream, default `0.1` (`…lora_train.py:235`, `:923-924`) | two explicit paths, `trainingDataPath` / `validationDataPath` (`FineTuneTrainer.swift:42-43`, `:231-237`) |
| Reserved/held-out third set | declared hashes carried through as data, not loaded (`…lora_train.py:250`, `:1116-1118`) | not found in code | not found in code |
| A file in both splits | refused (`lora_data.py:388-395`) | n/a | not found in code |
| Duplicate rows within a split | refused, naming both lines (`lora_data.py:406-417`) | not found in code | not found in code |
| Cross-split leakage | refused on row hash **or** content key (`lora_data.py:419-431`) | not found in code | not in the local trainer — only empty-data refusals (`FineTuneTrainer.swift:300-301`, `:366-368`) and soft advisory warnings (`FineTuningPanel.swift:2331-2346`) |

B's "validation" split is never evaluated at all — the trailing chunks are
computed and then discarded (`…lora_train.py:923-928`; the module says so at
`:17-21`). That, plus the absence of any leakage check, is why the path is
stamped `evidenceGrade: false` rather than merely called informal.

### 1.9 Validation sampling

| | A — split-based | B — legacy inline | C — Swift/MLX |
|---|---|---|---|
| Coverage | **the full validation set**, every example, token-weighted (`…lora_train.py:1030-1065`) | never evaluated | **head subsample**: at most `min(10, count)` batches from an unshuffled iterator (`FineTuneTrainer.swift:803`, `:929-930`, `:938`) |
| Same objective as training | yes, by construction — one `_supervised_loss_sum` for both (`…lora_train.py:998`, `:1037`) | n/a | yes for instruction mode (`FineTuneTrainer.swift:933-941`) |
| When | at `eval_interval_steps`, at **every epoch boundary**, and once at the last step (`…lora_train.py:1371-1372`, `:1389-1397`, `:1401`) | never | at iteration 0, then every `min(100, iterations)` steps (`FineTuneTrainer.swift:802`, `:833`) |
| Recorded | `training-history.json` rows with `supervisedTokens` and `lossDenominator` (`…lora_train.py:1271-1274`) | not recorded | progress events only — no history file is written |

C's subsample is worth stating plainly: `validationBatches` is a *batch* count
derived from an *example* count, and the effective batch size is capped at 2 (or
1 above 10B, §1.1), so a validation loss is computed over roughly the first 10
to 20 examples of the file regardless of how large the split is
(`FineTuneTrainer.swift:803`, `:944-948`). It is a training-progress readout,
not a held-out estimate.

### 1.10 Checkpoint selection, resume, lineage

| | A — split-based | B — legacy inline | C — Swift/MLX |
|---|---|---|---|
| Selection | best by declared `selectionMetric` (only `validationLoss` today); ties keep the **earlier** step (`…lora_train.py:120-121`, `:1278`, `:1403-1412`) | last step; no metric | **last write wins** — no selection exists |
| Missing metric | refusal for evidence-grade (`…lora_train.py:589-597`); last step for exploratory (`…lora_train.py:1413-1418`) | n/a | n/a |
| Checkpoints | `checkpoints/step-N/`, written tmp+rename with `trainer-state.json` last (`…lora_train.py:1502-1524`); retained: selected + latest (`…lora_train.py:781-789`) | none | periodic overwrite of the **same** `adapters.safetensors` every `min(100, iterations)` steps, plus a final save (`FineTuneTrainer.swift:804`, `:848-852`, `:385`) |
| Resume | adopts the newest complete checkpoint after verifying dataset fingerprint, model id, resolved revision and config fingerprint (`…lora_train.py:1539-1591`) | refused by name — writes no checkpoints (`…lora_train.py:849-854`) | not found in code — `LoRATrain.loadLoRAWeights` is never called; each run re-applies fresh adapters (`FineTuneTrainer.swift:291`) |
| Preemption | SIGUSR1/SIGTERM checkpoint between optimizer steps, exit 85 (`…lora_train.py:1377-1385`) | n/a | cancellation aborts (`FineTuneTrainer.swift:807`) |
| Lineage | `resumeLineage` entries with step, checkpoint path, timestamp (`…lora_train.py:1203-1206`, `:1458`) | n/a | not found in code |
| At-most-once finalize | a run directory with a sidecar refuses re-finalization (`…lora_train.py:867-876`) | same helper | not found in code |

### 1.11 Sidecar stamps

**A and B** write one JSON beside the adapter directory, built by the single
writer `adapter_sidecar_dict` (`…lora_train.py:486-534`). Both modes carry the
v1 keys — `name`, `baseModelID`, `revision`, `rank`, `alpha`, `learningRate`,
`iterations`, `maxChunkTokens`, `targetModules`, `trainChunks`, `finalLoss`,
`documents` — plus, from the same function:

`adapterFormat`, `substrate`, `trainingDtype` (`:509-511`);
`adapterScaleConvention`, `effectiveAdapterScale` (`:520-522`);
`schemaVersion`, `trainingMode`, `evidenceGrade`, `executionPath`, `dropout`,
`device` (`:525-530`).

**A alone** adds the v2 provenance block (`…lora_train.py:1437-1464`):
`dataset`, `template`, `revisionRequested`, `revisionResolved`,
`tokenizerSource`, `tokenizerRevision`, `modelClass`, `modelConfigHash`,
`buildIdentity`, `optimizerSettings`, `schedule` (which carries `objective`),
`selectedCheckpoint`, `historyFile`, `packageVersions`, `gpu`, `slurm`,
`timestamps`, `resumeLineage`, `adapterBytesHash`, `adapterConfigHash`,
`controlArm`. **B** adds only `adapterBytesHash`, `adapterConfigHash` and
`buildIdentity` (`…lora_train.py:968-973`).

The whole key set is pinned as a literal in
`Server/tests/test_lora_train_v2.py:542-554`, so a key cannot quietly vanish.

**C** writes `fine-tune.json` (`FineTuneStore.swift:183`) whose keys are the
stored property names verbatim — no `CodingKeys` (`FineTuneStore.swift:46-74`):
`schemaVersion`, `name`, `baseModelID`, `baseRevision`, `adapterDirectory`,
`adapterHash`, `configHash`, `substrate`, `adapterFormat`, `fineTuneType`,
`rank`, `scale`, `adaptedLayers`, `trainingWorkspacePath`, `trainingDataPath`,
`trainingDataHash`, `validationDataPath`, `validationDataHash`, `trainingMode`,
`batchSize`, `iterations`, `learningRate`, `createdAt`, `notes`. It also writes
MLX's own `adapter_config.json` (`FineTuneTrainer.swift:262-268`) and a
`RunMetadata` `config.json` with `runType: "lora-train"`
(`FineTuneStore.swift:180-182`).

Keys **C does not write** (searched and absent from `FineTuneArtifact`):
`objective`, `evidenceGrade`, `buildIdentity`, `optimizerSettings`, `schedule`,
`selectedCheckpoint`, `seed`, `alpha`, `effectiveAdapterScale`,
`adapterScaleConvention`, `dtype`.

### 1.12 Guarantees not available on each path

**A — split-based.** Bit-level numerical reproducibility (no determinism flags,
§1.7). Multi-GPU: the trainer is single-device and extra GPUs sit idle
(`finetune_submission.py:618-641`). Selection metrics other than
`validationLoss` (`…lora_train.py:120-121`). Verified control arms: a
`declaredNeutralizedDataset` is stamped as declared with a null fraction,
because the trainer never saw the dataset (`…lora_train.py:631-637`).

**B — legacy inline.** Everything A's evidence gate asks for: no explicit
splits (`…lora_train.py:284-288`), no evaluation, no checkpoint or resume
(`:849-854`), no selection, no dataset manifest, no row identity, no seeding,
and prompt tokens are supervised. It is a smoke test and the sidecar says so on
its face.

**C — Swift/MLX.** No seed, no reproducible ordering, no full-validation
estimate, no checkpoint selection, no resume, no revision pin, no byte
re-verification at load, no objective stamp, and a batch size that is silently
overridden (`FineTuneTrainer.swift:944-948`). Its declared `batchSize = 4`
default (`FineTuningPanel.swift:27`) never reaches the loop.

## 2. Which comparisons are matched, and which are only analogous

A comparison is **matched** when the two arms agree on the objective, on which
tokens are supervised, and on the effective adapter multiplier. Everything else
is **analogous**: it can motivate a claim about direction, never a claim about
magnitude.

**Matched.**

- *A against A.* Two adapters from recipe A on the same base revision, dataset
  pin, `effectiveAdapterScale`, target modules and objective are matched — this
  is the treatment-versus-control comparison the control arm exists for
  (`…lora_train.py:612-670`). Both arms are validated on the same held-out
  split by construction: the control is applied to the train split only
  (`:617-620`).
- *A against A across accumulation settings.* Matched by design, and pinned:
  the gradient is invariant to how the effective batch was cut
  (`Server/tests/test_lora_train_objective.py:186`).

**Analogous, not matched.**

- *A against C (Python split-based vs Swift/MLX).* Different scaling convention
  (§1.2), different adapted modules and layers (§1.4), different validation
  coverage (§1.9), different — in document mode, upstream and unread here —
  objective, and one side unseeded (§1.7). Adapter bytes do not transfer
  either: both loaders refuse an explicitly foreign stamp
  (`Server/steerlab_server/experiment/model_variant.py:551-578`;
  `Sources/ExperimentKit/ChatService.swift:3688-3697`, twin at
  `Sources/ExperimentKit/ExperimentTasks.swift:4240-4254`). A and C can be
  compared as "the same disposition installed by two instruments", never as one
  configuration measured twice.
- *A against B.* Different supervised tokens (B trains on prompts), different
  denominator, different weight decay by default, and B has no held-out
  measurement to compare against. **The 2026-09-05 objective fix landed on A
  only** (`…lora_train.py:1305-1341`); B was deliberately preserved verbatim
  (`:882-889`). A B adapter does not become evidence-grade because A was fixed.
- *B against C.* Both exploratory, and unmatched on every row of §1. There is
  no defensible reading of a difference between them.

**Cross-engine numerical identity is not an acceptance criterion.** No test in
either engine compares a Python loss to an MLX loss, and none should: the two
run different objectives on different module sets at different multipliers in
different precisions. The cross-engine contract is the *artifact* layer —
substrate and format stamps, dataset hashes, row identity, sidecar shape — and
that is what is pinned (`docs/PORTABILITY-CONTRACTS.md`;
`Server/tests/test_adapter_substrate.py`;
`Tests/ExperimentKitTests/FineTuneSubstrateStampTests.swift`). An adapter
comparison across engines reports two instruments' answers, and says so.

## 3. Reproducibility evidence

### What is tested, per path

**A — split-based.** `Server/tests/test_lora_train_v2.py` (refusals, the
reproducible loop, selection, resume, control arms, the sidecar key set):
`test_two_runs_produce_identical_loss_histories`,
`test_epoch_order_is_a_pure_function_of_seed_and_epoch`,
`test_schedule_counts_optimizer_steps_not_micro_batches`,
`test_best_checkpoint_selection_is_independent_of_the_final_step`,
`test_ties_keep_the_earlier_step`,
`test_interrupted_and_resumed_agrees_with_the_uninterrupted_control`,
`test_resume_refuses_changed_dataset_bytes` / `…changed_config` /
`…a_different_revision`, `test_checkpoint_retention_keeps_best_and_latest`,
`test_finalization_is_at_most_once`,
`test_shuffled_assistant_pairing_stamps_its_effectiveness`,
`test_sidecar_carries_every_contract_key`,
`test_sidecar_names_the_adapter_scale_convention_and_resolves_it`,
`test_dataset_block_carries_row_identity_and_reserved_hashes`,
`test_adapter_hashes_match_the_saved_files`.

`Server/tests/test_lora_train_objective.py` (the objective as an equation):
`test_gradient_is_invariant_to_micro_batch_partitioning`,
`test_partial_final_micro_batch_and_group_are_not_scaled_down`,
`test_accumulated_gradient_matches_a_hand_written_combined_batch`,
`test_hf_mean_loss_denominator_is_the_supervised_target_count`,
`test_history_rows_name_their_denominator`,
`test_objective_is_stamped_in_the_schedule_provenance`,
`test_validation_loss_uses_the_same_token_average`.

`Server/tests/test_lora_data.py` (60 tests: strict row schemas, hashes and
roots, split integrity, tokenization and masking) — notably
`test_exact_assistant_mask_fixture`,
`test_changing_only_the_prompt_does_not_move_the_mask`,
`test_cross_split_row_hash_overlap_refuses`,
`test_cross_split_same_text_under_a_different_id_still_refuses`,
`test_long_document_split_is_deterministic_with_chunk_provenance`,
`test_padding_never_contributes_to_the_loss`.

`Server/tests/test_lora_training_dtype.py`:
`test_training_dtype_policy_matrix`, `test_explicit_fp16_wins_but_warns_loudly`,
`test_adapter_sidecar_stamps_contract_and_training_dtype`.

**B — legacy inline.** Regression only, and deliberately so:
`test_legacy_inline_still_trains_and_writes_the_old_keys`,
`test_legacy_inline_is_not_resumable`,
`test_legacy_inline_config_has_no_dataset_spec`. Nothing tests its objective,
because nothing claims one.

**C — Swift/MLX.** `Tests/ExperimentKitTests/FineTuneTokenizationTests.swift`
covers rendering and masking against real Jinja templates offline
(`qwen3CompletedExampleEndsAtTheAnswersEndOfTurn`,
`qwen3MaskSupervisesTheAnswerAndItsEndOfTurnOnly`,
`qwen3PromptTokensArePrefixOfCompletedTokens`,
`gemmaFoldsSystemIntoUserAndEndsAtTheAnswersEndOfTurn`,
`overlongExamplesAreDroppedByTheTokenCeiling`,
`templateWithoutAPrefixRelationshipIsRefused`,
`cachedSnapshotAgreesWithThePinnedFixture`).
`FineTuneSubstrateStampTests.swift` covers the stamp/refusal contract.
`ExperimentKitTests.swift`'s `FineTuneDatasetTests` covers loading, chunking
and prompt masking. **No Swift test constructs a `FineTuneTrainingRequest` or
calls `FineTuneTrainer.train`** — the training loop, `effectiveBatchSize`,
`maskedInstructionLoss`, checkpoint saving and the cached-snapshot refusal are
uncovered.

### What is missing, stated plainly

- **No production-runtime evidence for any path.** Every Python test above runs
  on a two-layer, 32-hidden tiny Llama on CPU in float32
  (`Server/tests/test_lora_train_v2.py:34-61`). The properties pinned are
  structural — ordering, selection, refusals, stamps — not numerical behaviour
  at 27B on CUDA in bf16. No test in this repository fits a real adapter.
- **No Swift training-loop test at all** (above). C's objective, batch cap and
  save cadence are documented here from source and pinned by nothing.
- **MLX document mode is unexamined here.** Its loop lives in the upstream
  `mlx-swift-lm` package (`FineTuneTrainer.swift:318-324`; pin at
  `Package.resolved:23-28`), which is not vendored in this checkout. Whether it
  masks anything beyond padding is not established by any source in this
  repository.
- **No determinism evidence on GPU.** A's seed test compares two runs in one
  process on CPU (`test_two_runs_produce_identical_loss_histories`). Nothing
  shows that two seeded A runs on the same CUDA device agree, and the
  determinism flags that would make that likely are not set (§1.7).
- **No end-to-end resume evidence on real hardware.** The resume equality test
  is the tiny model in-process; the preemption path (SIGUSR1, exit 85, Slurm
  requeue) has no test that exercises a real requeue.
- **No cross-path calibration.** Nothing establishes what `scale` on C
  corresponds to what `alpha/rank` on A for a given base model — and given §1.4
  (different modules, different layers) it is not clear that any single number
  would.

## 4. Cells this document could not establish from code

These are honest gaps, not omissions:

1. **MLX document-mode objective and masking.** Delegated upstream
   (`FineTuneTrainer.swift:318-324`); not in this checkout.
2. **MLX `Adam` hyperparameters** (betas, eps, bias correction) and the
   **application form of `scale`** in the LoRA layer. What *is* established here
   is that this repository passes `scale` with no rank division and defines no
   alpha (`FineTuneTrainer.swift:265`); the arithmetic that consumes it is in
   the pinned `mlx-swift`/`mlx-swift-lm` packages.
3. **A's and B's resolved per-layer module list.** PEFT decides which layers
   match the four declared names; neither engine stamps the resolution.
4. **C's effective supervised-token count per run.** No history file is
   written, so it exists only in progress events.
5. **Whether C's `baseRevision` is ever populated for a locally trained
   adapter.** The field exists (`FineTuneStore.swift:50`) and the local training
   path does not write it.

## 5. Where this page disagrees with `docs/METHODS.md`

`docs/METHODS.md` §"LoRA adapters — fine-tuning as a contrast intervention"
(lines 672–702) predates the split-based path and has not been updated. It is
left alone deliberately; the divergences are recorded here rather than fixed in
passing.

- It describes ingestion as "Documents → chunked, hashed dataset", which is
  recipe B. Recipe A never concatenates and never chunks an instruction row
  (`lora_data.py:12-15`, `:649-654`).
- It says the sidecar records "final train/val loss". A's sidecar records
  `finalLoss` from the **training** history only (`…lora_train.py:1428-1430`);
  the validation number lives in `selectedCheckpoint.value` and
  `training-history.json`.
- It lists "continued pretraining on documents vs instruction-style masked
  completion" and "how much held-out validation is needed before an adapter is
  evidence-grade" as open design choices. Both are settled in code: the modes
  are `document` and `instruction_chat` (`lora_data.py:53-55`), and the evidence
  gate requires an explicit validation split and a declared selection metric
  (`…lora_train.py:584-597`).
- It does not mention `trainingMode`, `evidenceGrade`, or the exploratory
  `legacy_inline` tier at all, which is the distinction this page is built on.
