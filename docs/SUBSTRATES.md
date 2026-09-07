# Where a technique runs, and what has been measured

The Mac app, Mac CLI (`steerlab-cli`), portable client (`steerlab`) and HTTP API
are ways to author, run and inspect work. Python and Swift are execution owners;
CUDA, MPS and MLX are numerical backends, not four interchangeable user interfaces.
A Mac can run the Python engine on MPS or drive a remote Python runner. It can
also inspect a Python report without executing its mathematics through MLX.

Use `science list` and `science operation <id> --json` through the selected client
to discover callable paths. The catalog documents routes and outputs; it does not
grant scientific qualification. This inventory is developer/researcher guidance,
not a runtime permission list. Its metadata is not yet an additional app/API
status field: do not tell an agent that the CLI returns the table below.

## How to read this inventory

Checked against source `293888e`, 2026-09-07. **Implemented; comparison unqualified**
means the owner has a code path, but this inventory has no linked reproducible
cross-backend qualification for a named configuration. It does not mean previous
CUDA studies lack evidence. CUDA is the reference for the upcoming comparison;
it is not assumed to be mathematical ground truth.

**Partial** means a known semantic gap remains. **No native implementation** means
that backend cannot currently perform this operation through its native owner;
use the described Python route. **CPU / no backend execution** means that the
operation does not exercise that GPU backend. **Not inventoried** is uncertainty,
not a claim of support or a prohibition.

An unqualified configuration should normally be available for exploration, with
its limitations and effective settings recorded. Help the researcher choose a
more measured configuration when appropriate. Refuse only a concrete impossible,
uninterpretable or integrity-breaking request, and offer a repair. An artifact
whose basis belongs to PyTorch/HF cannot simply be relabeled as an MLX artifact.
This document changes no existing admission behavior by itself.

Qualification must name weights/tokenizer revision, precision/quantization,
library versions, device/OS, context and prefill/batching settings, tolerance,
input hashes, command and evidence. Same seed does not guarantee the same draws
across backends. Metadata-only imports and green CPU tests are not on-device
qualification. See [the program](TECHNIQUE-PARITY-IMPLEMENTATION.md).

## Core lifecycle outside the managed-operation catalog

The catalog's operation census is not the full product census. These existing
capabilities need their own explicit account; do not infer their absence from the
generated table. This short section is maintained prose checked against the named
owners, not mechanically inferred from CLI counts.

| Capability | Python | Native Swift/MLX | Current evidence boundary |
| --- | --- | --- | --- |
| Five extraction recipes | `experiment/extraction_workflow.py`, `steering/vector_math.py` | `ConceptBuilder`, `ExtractionRecipe`, `SteeringVectorMath` | Mathematical/identity tests exist; a configuration-specific MPS comparison remains to be measured. |
| Additive injection / ablation | `steering/injector.py`, `ablator.py`, `plan.py` | `VectorInjector`, `SubspaceAblator`, `InterventionPlan` | Shared mechanics; Swift descriptor and scope sidecar still owed. |
| Standard measured generation | `experiment/condition_execution.py`, `sampling.py` | `ExperimentTasks` | Python records effective seed policy; current native measured runs are greedy-only. |
| Multi-agent measured generation | `experiment/multi_agent.py` | `MultiAgentRunner`, `ExperimentTasks` | Python derives seeds per turn with common streams across conditions; native measured sampling remains greedy-only. |
| Sweep, run and analysis | Python stage owners via `experiment/tasks.py` | `ExperimentTasks` and stage owners | Same surface intent does not establish identical numerical output; inspect artifacts and effective configuration. |
| LoRA training | `experiment/lora_train.py` | `FineTuneTrainer` | See the explicit recipe/scale comparability matrix in TRAINING-RECIPES.md. |
| Workspace/study/design authoring | Portable client owners | Native owners plus shared Python diagnostics | CPU work; source identity and shared fixtures establish specific contract equivalence, not model qualification. |

Source paths in the table are relative to `Server/steerlab_server/` for Python;
Swift classes live under `Sources/ExperimentKit/` or `Sources/SteeringKit/`.
Contract detail: [extraction](EXTRACTION-RECIPES.md),
[intervention scope](INTERVENTION-SCOPE.md), [training](TRAINING-RECIPES.md),
[reader roles](REPE-IMPLEMENTATION-BRIEF.md).

## Catalog operation inventory

Maintained declarations: [substrate-capabilities.json](substrate-capabilities.json).
The generator joins these to actual catalog IDs/titles; adding or removing an
operation without updating its profile fails the documentation check. Source
links support implementation claims. Qualification requires separately reviewed
measurements; the generator only checks references, not scientific truth.

<!-- BEGIN SUBSTRATE-CATALOG -->

| Catalog operation | Execution profile | CUDA | MPS | MLX |
| --- | --- | --- | --- | --- |
| `stability` — Extraction stability | [diagnostic-model](#diagnostic-model) | implemented; comparison unqualified | implemented; comparison unqualified | no native implementation |
| `battery` — Standalone capability battery | [diagnostic-model](#diagnostic-model) | implemented; comparison unqualified | implemented; comparison unqualified | no native implementation |
| `rescore-style` — Rescore recorded style | [python-cpu](#python-cpu) | CPU / no backend execution | CPU / no backend execution | CPU / no backend execution |
| `sweep-judgment` — Sweep Judgment | [python-cpu](#python-cpu) | CPU / no backend execution | CPU / no backend execution | CPU / no backend execution |
| `evaluate-judgment` — Evaluate Judgment | [python-cpu](#python-cpu) | CPU / no backend execution | CPU / no backend execution | CPU / no backend execution |
| `reader-fit` — Reader Fit | [reader](#reader) | implemented; comparison unqualified | implemented; comparison unqualified | partial; comparison unqualified |
| `reader-score` — Reader Score | [reader](#reader) | implemented; comparison unqualified | implemented; comparison unqualified | partial; comparison unqualified |
| `finetune` — Finetune | [training](#training) | implemented; comparison unqualified | implemented; comparison unqualified | implemented; comparison unqualified |
| `jlens` — Jlens | [jlens](#jlens) | implemented; comparison unqualified | implemented; comparison unqualified | no native implementation |
| `gemmascope` — Gemmascope | [sae-import](#sae-import) | implemented; comparison unqualified | implemented; comparison unqualified | no native implementation |
| `sae-candidates` — Inspect and pin an SAE roster | [python-cpu](#python-cpu) | CPU / no backend execution | CPU / no backend execution | CPU / no backend execution |
| `optvec-train` — OptVec train | [python-model](#python-model) | implemented; comparison unqualified | implemented; comparison unqualified | no native implementation |
| `optvec-eval` — OptVec eval | [python-model](#python-model) | implemented; comparison unqualified | implemented; comparison unqualified | no native implementation |
| `optvec-geometry` — OptVec geometry | [python-cpu](#python-cpu) | CPU / no backend execution | CPU / no backend execution | CPU / no backend execution |
| `optvec-interpret` — OptVec interpret | [python-model](#python-model) | implemented; comparison unqualified | implemented; comparison unqualified | no native implementation |
| `optvec-family` — OptVec family | [python-cpu](#python-cpu) | CPU / no backend execution | CPU / no backend execution | CPU / no backend execution |
| `optvec-fracture` — OptVec fracture | [python-cpu](#python-cpu) | CPU / no backend execution | CPU / no backend execution | CPU / no backend execution |
| `optvec-campaign` — OptVec campaign | [campaign](#campaign) | implemented; comparison unqualified | not inventoried; inspect owner | no native implementation |
| `jspace` — J-space analysis | [jspace](#jspace) | implemented; comparison unqualified | implemented; comparison unqualified | no native implementation |
| `optvec-gradient` — OptVec gradient | [python-model](#python-model) | implemented; comparison unqualified | implemented; comparison unqualified | no native implementation |
| `optvec-gradient-mint` — Mint a gradient comparison vector | [python-cpu](#python-cpu) | CPU / no backend execution | CPU / no backend execution | CPU / no backend execution |
| `sae-family-report` — Report an SAE feature family | [python-cpu](#python-cpu) | CPU / no backend execution | CPU / no backend execution | CPU / no backend execution |
| `sae-qualification-record` — Record reviewed SAE qualification | [python-cpu](#python-cpu) | CPU / no backend execution | CPU / no backend execution | CPU / no backend execution |
| `sae-qualification-show` — Inspect an SAE qualification record | [python-cpu](#python-cpu) | CPU / no backend execution | CPU / no backend execution | CPU / no backend execution |

The following profiles explain production and artifact use.

### python-model

**Python:** Python owns execution; model work uses its configured device and executor.

**Swift/app:** The Mac authors, submits and inspects Python results; no native implementation of this managed operation is claimed.

CUDA is the comparison reference. This inventory records no configuration-specific cross-backend qualification. Try supported Python execution with that limitation stated; exact device admission remains owner-specific.

Source: [Server/steerlab_server/experiment/managed_methods.py](../Server/steerlab_server/experiment/managed_methods.py), [Server/steerlab_server/api/scientific_execution.py](../Server/steerlab_server/api/scientific_execution.py).

### python-cpu

**Python:** Python CPU owner; no GPU is required for this operation.

**Swift/app:** Use the shared Python path from the Mac and inspect its records; a native twin is not required for authoring equivalence.

CPU execution on a Mac is not MPS or MLX qualification. Numerical correctness and input roles still need owner tests.

Source: [Server/steerlab_server/experiment/managed_methods.py](../Server/steerlab_server/experiment/managed_methods.py), [Server/steerlab_server/experiment/science_catalog.py](../Server/steerlab_server/experiment/science_catalog.py).

### diagnostic-model

**Python:** Python model diagnostic on the selected device; local and Slurm routing use existing owners.

**Swift/app:** Mac science execution/custody controls drive Python and read the exported report; native numerical diagnostic parity is not yet supplied.

Stability measures direction sensitivity; a battery measures its declared capability/generation checks. Neither is universal behavioral validation.

Source: [Server/steerlab_server/experiment/extract_stability.py](../Server/steerlab_server/experiment/extract_stability.py), [Server/steerlab_server/experiment/battery_run.py](../Server/steerlab_server/experiment/battery_run.py), [Sources/SteerLabApp/ScientificExecutionSheet.swift](../Sources/SteerLabApp/ScientificExecutionSheet.swift).

### reader

**Python:** Python reader fitting/scoring, with model activation capture on its selected device.

**Swift/app:** Native RepE fitting/scoring exists. Its non-train selection currently includes finalTest rows; use Python for that split contract until corrected.

Artifact reading and arithmetic fixtures do not establish split-role equivalence. This known limitation is concrete, not a generic ban on exploratory reader fitting.

Source: [Server/steerlab_server/steering/repe_reader.py](../Server/steerlab_server/steering/repe_reader.py), [Sources/SteeringKit/Extraction/RepEReader.swift](../Sources/SteeringKit/Extraction/RepEReader.swift), [docs/REPE-IMPLEMENTATION-BRIEF.md](../docs/REPE-IMPLEMENTATION-BRIEF.md).

### training

**Python:** Python LoRA training on the selected engine device; adapter-family admission is explicit.

**Swift/app:** Native MLX LoRA training exists; use the training recipe matrix to compare objectives, scaling and data treatment.

Native training does not imply all Python objectives/adapter families are interchangeable. Adapter-scale and evidence-role parity work remains.

Source: [Server/steerlab_server/experiment/lora_train.py](../Server/steerlab_server/experiment/lora_train.py), [Sources/ExperimentKit/FineTuneTrainer.swift](../Sources/ExperimentKit/FineTuneTrainer.swift), [docs/TRAINING-RECIPES.md](../docs/TRAINING-RECIPES.md).

### jlens

**Python:** Python owns acquisition/import, model reference qualification and runtime readout; metadata-only steps do not establish device coverage.

**Swift/app:** The app acquires, inspects and presents Python lens evidence. PyTorch/HF-native atoms are not assumed valid for MLX activations.

The catalog entry groups multiple actions: qualify model-dependent paths separately from CPU metadata/import. Matching dimensions alone does not justify representation transfer.

Source: [Server/steerlab_server/jlens/qualification.py](../Server/steerlab_server/jlens/qualification.py), [Server/steerlab_server/jlens/importer.py](../Server/steerlab_server/jlens/importer.py), [Sources/SteerLabApp/JLensSupportSection.swift](../Sources/SteerLabApp/JLensSupportSection.swift).

### sae-import

**Python:** Python owns Gemma Scope analysis and feature import; distinguish reading/importing tensors from model execution.

**Swift/app:** Mac UI drives Python operations and inspects imported evidence; no native Gemma Scope numerical implementation is asserted.

Successful import alone does not qualify SAE feature behavior on MPS or validate transferring a feature into MLX activations.

Source: [Server/steerlab_server/experiment/gemma_scope.py](../Server/steerlab_server/experiment/gemma_scope.py), [Sources/SteerLabApp/GeometryPanelView.swift](../Sources/SteerLabApp/GeometryPanelView.swift).

### jspace

**Python:** Python J-space analysis: intervention-dependent downstream residual readout using a qualified lens.

**Swift/app:** The Mac authors the request and displays Python reports; no native MLX J-space numerical implementation is supplied.

J-space is distinct from OptVec training. The current batch owner accepts OptVec artifact metadata for layer/dose; this is an admission limitation, not a definition of the method.

Source: [Server/steerlab_server/experiment/optvec_jspace.py](../Server/steerlab_server/experiment/optvec_jspace.py), [Sources/SteerLabApp/JSpacePanelSection.swift](../Sources/SteerLabApp/JSpacePanelSection.swift), [WorkspaceSeed/prompts/method-guides/jspace.md](../WorkspaceSeed/prompts/method-guides/jspace.md).

### campaign

**Python:** CPU materialization/coordination; training cells execute as separate Python GPU jobs.

**Swift/app:** Mac authoring and campaign controls drive Python; the initial CPU plan is not the training run.

Do not infer MPS campaign scheduling support from a training kernel. Cell execution and local/Slurm orchestration need separate qualification.

Source: [Server/steerlab_server/api/managed_campaign.py](../Server/steerlab_server/api/managed_campaign.py), [Server/steerlab_server/api/managed_campaign_engine.py](../Server/steerlab_server/api/managed_campaign_engine.py), [Server/steerlab_server/experiment/optvec_campaign.py](../Server/steerlab_server/experiment/optvec_campaign.py).

<!-- END SUBSTRATE-CATALOG -->

## Updating this account

Run `python scripts/ci/check-substrates.py --write` after a reviewed declaration
change. `check-science-resources.py` also checks this inventory. Do not mark a
profile qualified merely because an API exists, an import succeeded, or an
unrelated test suite passed. Split a profile if only some operations/configurations
gain evidence. Keep the result's own provenance authoritative; documentation never
overwrites historical claims or grants cleanup authority.
