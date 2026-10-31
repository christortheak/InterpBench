"""Managed J-lens probe over a prompt set: one model load, one lens build, one run.

The interactive probe (``steerlab-server jlens probe``, ``POST /api/jlens/probe``)
reads ONE prompt and loads the model in the serving process, which a cluster
controller never does. This operation reads a whole prompt set as one GPU job
through the managed science path: the plan prices every prompt before
submission, the queued child re-checks its inputs on the node, and the single
run directory comes home under custody with ``science-export`` and
``science-fetch``.

Per prompt the numbers are the probe's own. Every prompt goes through
``jlens.probe.plan_prompt`` and ``read_prompt`` under one ``ProbeSession``, so
the model loads once and the converted lens is re-hashed and its armed
Jacobians are placed on the device once, however many prompts there are.

Discovery, authoring and review import no GPU package: the review loads a
tokenizer from the local cache only when one is installed and cached, and
says so when it cannot.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from . import diagnostic_archives as archives

OPERATION = 'jlens-probe-batch'
RUN_TYPE = 'jlens-probe-batch'
MANIFEST = 'probe-batch.json'
PROMPTS_DIRECTORY = 'prompts'
SCHEMA_VERSION = 1

#: The probe's own per-prompt ceiling, restated so review imports no torch.
#: A test pins it to ``jlens.probe.MAX_PROJECTIONS``.
MAX_PROJECTIONS = 200_000

#: Files the run writes: config, manifest and COMPLETED beside the prompts,
#: and four files per prompt (probe.json, two CSVs, config.json). Export
#: carries at most ``diagnostic_archives.MAX_FILES`` files.
RUN_FILES = 3
FILES_PER_PROMPT = 4
MAX_PROMPTS = (archives.MAX_FILES - RUN_FILES) // FILES_PER_PROMPT

#: Prompt ids name their result directories, so they are single portable
#: components. Two ids that differ only in case would collide on a
#: case-insensitive disk once the run comes home, so they are refused too.
PROMPT_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}')
MAX_PROMPT_FILE_BYTES = 64 * 1024**2

#: Prompts listed one by one in a review; the totals cover all of them.
REVIEW_PROMPTS_LISTED = 100

PROMPT_MODES = ('chatAssistant', 'rawCompletion')
DTYPES = {'bfloat16': 2, 'float16': 2, 'float32': 4, 'auto': None}
_STORED_DTYPE_BYTES = {'bfloat16': 2, 'float16': 2, 'float32': 4}

#: The stated walltime allowance. No recorded run measured this workload's
#: rate, so the estimate is a formula: each full-vocabulary projection reads
#: the output head once and each transport reads one float32 Jacobian, both at
#: ``ASSUMED_BANDWIDTH``; the forward pass runs at ``ASSUMED_FLOPS``; the
#: converted lens is read twice (its hash, then its layers) at
#: ``ASSUMED_DISK_RATE``. The walltime rule's threefold margin then covers an
#: effective bandwidth down to about a third of the assumed one.
ASSUMED_BANDWIDTH = 500e9
ASSUMED_FLOPS = 50e12
ASSUMED_DISK_RATE = 200e6

#: The memory allowance beyond the itemized tensors: allocator fragmentation
#: and the device context.
MEMORY_HEADROOM = 0.10
MEMORY_CONTEXT_BYTES = 1024**3

#: Output size per stored value, conservative for ``json.dump(indent=2)``
#: and the two CSVs (see :func:`output_estimate`).
OUTPUT_SAFETY = 1.25


class ProbeBatchError(archives.Refusal):
    code = 'jlensProbeBatchRefused'
    repair_action = ('Correct the prompt set or probe settings named in the reason, '
                     'review a fresh plan, then submit its exact planSHA256.')


def _refuse(reason, repair=None):
    error = ProbeBatchError(reason)
    if repair:
        error.repair_action = repair
    return error


def _count(value, key, *, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _refuse(f'{key} must be an integer of at least {minimum}.')
    return value


@dataclass(frozen=True)
class ProbeBatchConfig:
    modelID: str
    revision: str
    lensID: str
    prompts: dict
    layers: list[int] | None = None
    allLayers: bool = False
    topK: int = 10
    pin: list[str] | None = None
    pinIDs: list[int] | None = None
    directions: list[str] | None = None
    positionStride: int = 1
    maxTokens: int | None = None
    variantPath: str | None = None
    promptMode: str | None = None
    systemPrompt: str | None = None
    dtype: str = 'bfloat16'
    device: str | None = None

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise _refuse('The batch probe config must be a JSON object.')
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise _refuse('Unknown batch probe config key(s): ' + ', '.join(unknown) + '.',
                          'Use only the documented jlens-probe-batch fields.')
        missing = [key for key in ('modelID', 'revision', 'lensID', 'prompts') if key not in value]
        if missing:
            raise _refuse('Name the model, its exact revision, the lens, and the prompt set (missing: '
                          + ', '.join(missing) + ').')
        value = dict(value)
        for key in ('modelID', 'revision', 'lensID'):
            if not isinstance(value[key], str) or not value[key].strip():
                raise _refuse(f'{key} must be a nonempty string.')
            value[key] = value[key].strip()
        if len(archives.parts(value['lensID'])) != 1:
            raise _refuse('lensID names a registered lens, not a file path.')
        prompts = value['prompts']
        if (not isinstance(prompts, dict) or set(prompts) != {'path', 'sha256'}
                or not isinstance(prompts['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', prompts['sha256'])):
            raise _refuse('prompts needs a workspace-relative path and the SHA-256 of its bytes.')
        archives.parts(prompts['path'])
        value['prompts'] = dict(prompts)
        if value.get('layers') is not None:
            layers = value['layers']
            if not isinstance(layers, list) or not layers or not all(
                    not isinstance(x, bool) and isinstance(x, int) and x >= 0 for x in layers):
                raise _refuse('layers must be a nonempty list of fitted source-layer numbers; '
                              'omit it for the lens default.')
            value['layers'] = sorted(set(layers))
        if not isinstance(value.get('allLayers', False), bool):
            raise _refuse('allLayers must be true or false.')
        if value.get('allLayers') and value.get('layers') is not None:
            raise _refuse('Choose layers or allLayers, not both.')
        _count(value.get('topK', 10), 'topK', minimum=0)
        _count(value.get('positionStride', 1), 'positionStride', minimum=1)
        if value.get('maxTokens') is not None:
            _count(value['maxTokens'], 'maxTokens', minimum=1)
        for key, kind in (('pin', str), ('pinIDs', int), ('directions', str)):
            # Absent and empty are the same request: the effective config
            # always carries the list.
            items = value[key] = [] if value.get(key) is None else value[key]
            if not isinstance(items, list) or not all(
                    isinstance(x, kind) and not isinstance(x, bool) for x in items):
                raise _refuse(f'{key} must be a list of '
                              + ('token ids.' if kind is int else 'nonempty strings.'))
            if kind is str and not all(x.strip() for x in items):
                raise _refuse(f'{key} entries must be nonempty.')
            if kind is int and any(x < 0 for x in items):
                raise _refuse('pinIDs must be nonnegative token ids.')
        for reference in value.get('directions') or []:
            archives.parts(reference)
        if len(set(value.get('directions') or [])) != len(value.get('directions') or []):
            raise _refuse('List each direction once.')
        if value.get('variantPath') is not None:
            if not isinstance(value['variantPath'], str):
                raise _refuse('variantPath names an agent JSON file in the workspace.')
            archives.parts(value['variantPath'])
        if value.get('promptMode') is not None and value['promptMode'] not in PROMPT_MODES:
            raise _refuse('promptMode must be chatAssistant or rawCompletion; omit it for the '
                          "agent's own mode (or chat when there is no agent).")
        if value.get('systemPrompt') is not None and not isinstance(value['systemPrompt'], str):
            raise _refuse('systemPrompt must be text.')
        if value.get('dtype', 'bfloat16') not in DTYPES:
            raise _refuse('dtype must be one of ' + ', '.join(DTYPES) + '.')
        if value.get('device') is not None and (not isinstance(value['device'], str) or not value['device'].strip()):
            raise _refuse('device must be a nonempty string such as cuda, mps, or cpu; omit it for the engine default.')
        config = cls(**value)
        if not config.reads_vocabulary and not config.directions:
            raise _refuse('Nothing to read: with topK 0, name pin, pinIDs, or directions.')
        return config

    @property
    def reads_vocabulary(self):
        return bool(self.pin) or bool(self.pinIDs) or self.topK > 0

    def to_dict(self):
        result = asdict(self)
        for key in ('pin', 'pinIDs', 'directions'):
            result[key] = list(result[key] or [])
        return result


# --- the prompt set ------------------------------------------------------------

def read_prompts(config, root):
    """The pinned prompt rows: ``{id, text, labels}``, refused with every problem named."""
    path = archives.ordinary(root, config.prompts['path'])
    if not path.is_file():
        raise _refuse('The prompt set is missing: ' + config.prompts['path'])
    with path.open('rb') as stream:
        data = stream.read(MAX_PROMPT_FILE_BYTES + 1)
    if len(data) > MAX_PROMPT_FILE_BYTES:
        raise _refuse('The prompt set exceeds 64 MiB; split it across requests.')
    if hashlib.sha256(data).hexdigest() != config.prompts['sha256']:
        raise _refuse('The prompt set changed after review: ' + config.prompts['path'],
                      'Review the current file and draft the request again.')
    rows, problems, seen = [], [], {}
    try:
        lines = data.decode('utf-8').splitlines()
    except UnicodeError:
        raise _refuse('The prompt set must be UTF-8 JSONL: one object per line.') from None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            problems.append(f'line {number} is not JSON')
            continue
        if not isinstance(row, dict) or not {'id', 'text'} <= set(row) or set(row) - {'id', 'text', 'labels'}:
            problems.append(f'line {number} needs id and text, with optional labels and no other fields')
            continue
        identifier, text, labels = row['id'], row['text'], row.get('labels', {})
        if not isinstance(identifier, str) or not PROMPT_ID.fullmatch(identifier):
            problems.append(f'line {number}: id must be letters, digits, ".", "_" or "-", starting with a letter or digit')
            continue
        if not isinstance(text, str) or not text.strip():
            problems.append(f'line {number} ({identifier}): text must be nonempty')
            continue
        if not isinstance(labels, dict) or not all(
                isinstance(k, str) and (v is None or isinstance(v, (str, int, float, bool)))
                for k, v in labels.items()):
            problems.append(f'line {number} ({identifier}): labels must map names to plain values')
            continue
        folded = identifier.casefold()
        if folded in seen:
            problems.append(f'line {number}: id {identifier!r} repeats {seen[folded]!r} (ids are compared ignoring case)')
            continue
        seen[folded] = identifier
        rows.append({'id': identifier, 'text': text, 'labels': dict(labels)})
    if problems:
        shown = '; '.join(problems[:20]) + (f'; and {len(problems) - 20} more' if len(problems) > 20 else '')
        raise _refuse('The prompt set has problems: ' + shown + '.',
                      'Write one JSON object per line with a unique id, the prompt text, and optional labels '
                      '(for example {"id": "case-1", "text": "...", "labels": {"condition": "baseline"}}).')
    if not rows:
        raise _refuse('The prompt set contains no prompts.')
    if len(rows) > MAX_PROMPTS:
        raise _refuse(f'{len(rows)} prompts exceed the {MAX_PROMPTS} one run can carry home.',
                      'Split the prompt set across requests.')
    return rows


# --- the agent and its files ---------------------------------------------------

def _relative(reference, root, what):
    """A workspace-relative path for a reference the agent records.

    A relative reference is already one. An absolute one (an agent promoted
    on another machine) is rebased exactly as execution rebases it, and is
    refused when it still names something outside this workspace.
    """
    from . import paths
    if not isinstance(reference, str) or not reference:
        raise _refuse(f'The agent names an empty {what} reference.')
    if not os.path.isabs(reference):
        archives.parts(reference)
        return reference
    resolved = Path(paths.resolve_artifact(reference, str(root)))
    try:
        return resolved.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        raise _refuse(f'The agent {what} {reference!r} is outside the workspace.',
                      'Import the agent and the files it references into this workspace, '
                      'then draft the request again.') from None


def read_agent(config, root):
    """The agent JSON as a document, refused when the probe could not run it."""
    if not config.variantPath:
        return None
    path = archives.ordinary(root, config.variantPath)
    if not path.is_file():
        raise _refuse('The agent file is missing: ' + config.variantPath)
    try:
        document = json.loads(path.read_bytes())
    except ValueError:
        raise _refuse('The agent file is not JSON: ' + config.variantPath) from None
    if not isinstance(document, dict) or not isinstance(document.get('baseModelID'), str):
        raise _refuse('The agent file is not an agent (no baseModelID): ' + config.variantPath)
    if document['baseModelID'] != config.modelID:
        raise _refuse(f"The agent was built for {document['baseModelID']}, not {config.modelID}: its "
                      "injections live in that model's residual basis.",
                      'Choose an agent built for this model, or probe the model it was built for.')
    if document.get('interventionPolicies'):
        raise _refuse('This agent carries intervention policies, which the probe does not run.',
                      'Probe an agent without intervention policies, or study the policy in a sampled-response study.')
    return document


def agent_files(config, root):
    """Every file the agent's readout depends on, workspace-relative.

    The agent JSON travels by its own input role; this adds what it names:
    each injection's vector (tensor and sidecar), each adapter directory and
    its provenance sidecar, and the neutral basis. ``managed_inputs`` pins
    their bytes, so a changed file changes the plan and refuses a queued job.
    """
    if isinstance(config, dict):
        config = ProbeBatchConfig.from_dict(config)
    document = read_agent(config, root)
    if document is None:
        return []
    root = Path(root).resolve()
    files = []
    for cell in document.get('injections') or []:
        reference = cell.get('vectorArtifactID') if isinstance(cell, dict) else None
        if reference:
            relative = _relative(reference, root, 'vector')
            files += [relative + '.json', relative + '.safetensors']
    for adapter in document.get('adapters') or []:
        reference = adapter.get('adapterDirectory') if isinstance(adapter, dict) else None
        if reference:
            relative = _relative(reference, root, 'adapter')
            files.append(relative)
            sidecar = relative.rstrip('/') + '.json'
            if archives.ordinary(root, sidecar, missing=True).is_file():
                files.append(sidecar)
    basis = document.get('neutralPCBasisPath')
    if basis:
        files.append(_relative(basis, root, 'neutral basis'))
    return files


def _direction_models(config, root):
    """Refuse a direction extracted on another model, as the probe does."""
    wrong = []
    for reference in config.directions or []:
        sidecar = archives.ordinary(root, reference + '.json', missing=True)
        try:
            model = json.loads(sidecar.read_bytes()).get('modelID')
        except (OSError, ValueError, AttributeError):
            continue  # The inventory names a missing file; the owner reads it at execution.
        if model and model != config.modelID:
            wrong.append(f'{reference} ({model})')
    if wrong:
        raise _refuse('Direction(s) ' + ', '.join(wrong) + f' were extracted on a different model than '
                      f'{config.modelID}; residual bases are model-specific.',
                      'Choose directions extracted on this model.')


# --- review: counts, memory, output size, walltime -----------------------------

def armed_layers(config, record):
    from ..jlens.qualification import _default_layers
    if config.allLayers:
        return list(record.sourceLayers)
    return list(config.layers) if config.layers else _default_layers(record)


def _rendering(config, agent):
    mode = config.promptMode or (agent or {}).get('promptMode') or 'chatAssistant'
    system = config.systemPrompt if config.systemPrompt is not None else (agent or {}).get('systemPrompt')
    return mode, system, bool((agent or {}).get('qwenThinkingEnabled', False))


def token_counts(config, rows, agent):
    """Exact rendered token counts from the pinned revision's cached tokenizer, or why not.

    The same renderer and precedence the probe uses, so a count here is the
    count the node will read. Nothing is downloaded.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None, 'The tokenizer library is not installed here; the engine plan counts tokens.'
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.modelID, revision=config.revision, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal to review
        return None, f'The tokenizer for this revision is not in the local cache ({type(exc).__name__}).'
    from . import prompt_render
    mode, system, thinking = _rendering(config, agent)
    counts = []
    for row in rows:
        try:
            rendered = prompt_render.render(tokenizer, row['text'], model_id=config.modelID,
                                            prompt_mode=mode, system_prompt=system,
                                            qwen_thinking_enabled=thinking)
        except Exception as exc:  # noqa: BLE001 - a prompt that cannot render is refused by name
            raise _refuse(f"Prompt {row['id']} cannot be rendered as requested: {exc}") from exc
        counts.append(len(rendered.input_ids))
    return counts, None


def cached_geometry(model_id, revision):
    """Model geometry and checkpoint bytes from the local cache, or None. Nothing is downloaded."""
    from .jlens_fit_review import cached_config
    cached = cached_config(model_id, revision)
    if not cached:
        return None
    document, digest = cached
    text = document.get('text_config') if isinstance(document.get('text_config'), dict) else document

    def read(key):
        value = text.get(key, document.get(key))
        return value if type(value) is int and value > 0 else None

    hidden, heads = read('hidden_size'), read('num_attention_heads')
    geometry = {'modelConfigSHA256': digest, 'hiddenSize': hidden, 'layers': read('num_hidden_layers'),
                'attentionHeads': heads, 'keyValueHeads': read('num_key_value_heads') or heads,
                'headDim': read('head_dim') or (hidden // heads if hidden and heads else None),
                'intermediateSize': read('intermediate_size'), 'vocabSize': read('vocab_size'),
                'storedDtype': text.get('torch_dtype') or document.get('torch_dtype') or document.get('dtype'),
                'checkpointBytes': _checkpoint_bytes(model_id, revision)}
    return geometry


def _checkpoint_bytes(model_id, revision):
    try:
        from huggingface_hub import try_to_load_from_cache
        index = try_to_load_from_cache(model_id, 'model.safetensors.index.json', revision=revision)
        if isinstance(index, str):
            with open(index, 'rb') as stream:
                total = json.loads(stream.read(16 * 1024**2)).get('metadata', {}).get('total_size')
            return total if type(total) is int and total > 0 else None
        single = try_to_load_from_cache(model_id, 'model.safetensors', revision=revision)
        return os.path.getsize(single) if isinstance(single, str) else None
    except (ImportError, OSError, ValueError, AttributeError):
        return None


def _run_bytes(config, geometry):
    """Bytes per parameter in the run: the requested dtype, else the checkpoint's own."""
    stored = _STORED_DTYPE_BYTES.get(str((geometry or {}).get('storedDtype')), 2)
    return DTYPES[config.dtype] or stored, stored


def memory_estimate(config, geometry, d_model, layers, longest, basis='exact'):
    """Peak device memory, itemized: weights, Jacobians, activations, readout, headroom.

    ``longest`` is the longest prompt's token count, ``basis`` says whether it
    is exact or the maxTokens bound. Any missing component makes the estimate
    partial; a partial estimate states what is known and is never compared
    with a GPU. Only an exact estimate can refuse a plan.
    """
    jacobians = len(layers) * d_model * d_model * 4
    result = {'jacobianBytes': jacobians, 'armedLayers': len(layers), 'hiddenSize': d_model,
              'tokenBasis': basis}
    geometry = geometry or {}
    run_bytes, stored_bytes = _run_bytes(config, geometry)
    needed = ('checkpointBytes', 'vocabSize', 'layers', 'keyValueHeads', 'headDim', 'attentionHeads', 'intermediateSize')
    missing = [key for key in needed if not geometry.get(key)] + ([] if longest else ['tokenCount'])
    if missing:
        return {**result, 'status': 'partial', 'missing': missing,
                'summary': f'Partial estimate: the armed Jacobians alone need {jacobians / 1024**3:.2f} GiB in '
                           f'float32. Model geometry or token counts are unavailable here ({", ".join(missing)}), '
                           'so memory is not compared with a GPU.'}
    weights = int(geometry['checkpointBytes'] * run_bytes / stored_bytes)
    t, v, d = longest, geometry['vocabSize'], d_model
    activations = {
        # The forward pass's own logits, at the run dtype and upcast to float32.
        'logitsBytes': t * v * (run_bytes + 4),
        'keyValueCacheBytes': 2 * geometry['layers'] * geometry['keyValueHeads'] * geometry['headDim'] * t * run_bytes,
        # Eager attention scores and their softmax in float32: the worst case.
        'attentionBytes': geometry['attentionHeads'] * t * t * 4 * 2,
        'layerTransientBytes': t * (3 * geometry['intermediateSize'] + 4 * d) * run_bytes,
    }
    # Per position: two full-vocabulary logit vectors in float32, the gain,
    # and the watched rows.
    readout = 4 * v * 4 + d * 4 + len(config.pinIDs or []) * d * 4 + len(config.pin or []) * d * 4
    subtotal = weights + jacobians + sum(activations.values()) + readout
    total = int(subtotal * (1 + MEMORY_HEADROOM) + MEMORY_CONTEXT_BYTES)
    return {**result, 'status': 'estimated', 'weightBytes': weights, 'runDtypeBytes': run_bytes,
            'longestPromptTokens': t, 'activationBytes': activations, 'readoutBytes': readout,
            'headroom': f'{int(MEMORY_HEADROOM * 100)}% plus {MEMORY_CONTEXT_BYTES // 1024**3} GiB',
            'totalBytes': total,
            # Capture rows stay on the host: every armed layer × position in float32.
            'hostCaptureBytes': len(layers) * t * d * 4,
            'formula': 'weights at the run dtype + armed Jacobians (layers × d² × 4) + forward activations for '
                       'the longest prompt (logits, key/value cache, eager attention scores, layer transients) '
                       '+ per-position readout vectors, then 10% and 1 GiB of headroom',
            'summary': f'About {total / 1024**3:.1f} GiB of device memory: {weights / 1024**3:.1f} GiB of weights, '
                       f'{jacobians / 1024**3:.2f} GiB of float32 Jacobians, and '
                       f'{sum(activations.values()) / 1024**3:.2f} GiB of activations for the longest prompt '
                       f'({t} tokens). The capture holds {len(layers) * t * d * 4 / 1024**3:.2f} GiB on the host.',
            'limitation': 'An itemized estimate from the cached configuration, not a measurement. The model loads '
                          'onto one GPU; another architecture, attention kernel, or library version moves it.'}


def output_estimate(config, prompts, layers):
    """Bytes the run writes per prompt, conservative for indented JSON and the CSVs.

    Per prompt: the text and its rendering (escaped, so up to three bytes
    per UTF-8 byte, plus the template), about 48 bytes per token for its id
    and piece, about 104 bytes per top-k entry per cell (four lists), and a
    trajectory row per scored cell with its pinned ranks and cosines.
    """
    k, pins = config.topK, len(config.pin or []) + len(config.pinIDs or [])
    directions = len(config.directions or [])
    system = len((config.systemPrompt or '').encode('utf-8'))
    per = []
    for prompt in prompts:
        tokens, positions = prompt['tokenCount'], prompt['scoredPositions']
        cells = positions * len(layers) if k > 0 else 0
        rows = positions * len(layers)
        json_bytes = (8 * 1024 + 3 * (2 * prompt['textBytes'] + system + 2048) + 48 * tokens
                      + cells * (104 * k + 220) + rows * (160 + 60 * pins + 120 * directions))
        csv_bytes = cells * k * 96 + rows * (40 + 16 * pins + 24 * directions)
        per.append(int((json_bytes + csv_bytes + 4096) * OUTPUT_SAFETY))
    return per


def walltime_estimate(config, prompts, layers, d_model, geometry, lens_bytes):
    """The stated formula, or None when the output head's size is unknown."""
    geometry = geometry or {}
    vocab = geometry.get('vocabSize')
    if not vocab or any(p['tokenCount'] is None for p in prompts):
        return None
    run_bytes, stored_bytes = _run_bytes(config, geometry)
    head = vocab * d_model * run_bytes
    jacobian = d_model * d_model * 4
    projections = sum(p['projections'] for p in prompts)
    transports = projections // 2 + sum(p['scoredPositions'] for p in prompts) * len(layers) * 2 * len(config.directions or [])
    parameters = (geometry.get('checkpointBytes') or 0) / stored_bytes
    forward = sum(p['tokenCount'] for p in prompts) * 2 * parameters / ASSUMED_FLOPS
    seconds = (projections * head + transports * jacobian) / ASSUMED_BANDWIDTH + forward + 2 * lens_bytes / ASSUMED_DISK_RATE
    return {'seconds': int(math.ceil(seconds)), 'units': projections, 'unit': 'one full-vocabulary projection',
            'prompts': len(prompts), 'transports': transports, 'outputHeadBytes': head,
            'formula': 'projections × output-head bytes and transports × Jacobian bytes at 500 GB/s, the forward '
                       'pass at 50 TFLOP/s, and the converted lens read twice at 200 MB/s',
            'calibration': 'None recorded: a stated allowance rather than a calibration. The walltime rule adds a '
                           'threefold margin and the model load, which covers an effective memory bandwidth down '
                           'to about 170 GB/s.',
            'limitation': 'An allowance, not a measurement of this workload: slower storage, contention, or a '
                          'different attention kernel change the rate. The scheduler kills a job at its walltime; '
                          'request a longer one explicitly when in doubt, or a shorter one for a fast GPU.'}


def preflight(config, root):
    """What the job will do, priced before submission; refuses what cannot run as asked."""
    from ..jlens import artifact_paths, lens_store
    from ..jlens.schemas import JLensError
    root = Path(root).resolve()
    rows = read_prompts(config, root)
    try:
        record = lens_store.resolve(config.lensID, str(root))
    except JLensError as exc:
        raise _refuse(str(exc), 'Import the lens into this workspace (or name one that is), then review again.') from exc
    layers = armed_layers(config, record)
    unknown = [layer for layer in layers if layer not in record.sourceLayers]
    if unknown:
        raise _refuse(f"Layers {unknown} are not fitted source layers of '{config.lensID}' "
                      f'(have {record.sourceLayers[0]}..{record.sourceLayers[-1]}).',
                      'Choose fitted source layers, or allLayers for every one.')
    agent = read_agent(config, root)
    _direction_models(config, root)
    advisories = []
    if record.fit.modelID != config.modelID:
        advisories.append(f"The lens was fitted on {record.fit.modelID}, not {config.modelID}. "
                          'The readout runs; whether the transport carries over is a question for the reading.')
    counts, unavailable = token_counts(config, rows, agent)
    stride = config.positionStride
    prompts, over = [], []
    for index, row in enumerate(rows):
        exact = counts is not None
        tokens = counts[index] if exact else config.maxTokens
        if tokens is not None and config.maxTokens:
            tokens = min(tokens, config.maxTokens)
        positions = None if tokens is None else len(range(0, tokens, stride))
        projections = None if positions is None else (positions * len(layers) * 2 if config.reads_vocabulary else 0)
        entry = {'id': row['id'], 'labels': row['labels'], 'tokenCount': tokens, 'tokenCountBasis':
                 'exact' if exact else ('maxTokens bound' if tokens is not None else 'unavailable'),
                 'scoredPositions': positions, 'projections': projections,
                 'textBytes': len(row['text'].encode('utf-8'))}
        prompts.append(entry)
        if exact and projections > MAX_PROJECTIONS:
            over.append(f"{row['id']} ({positions} positions × {len(layers)} layers × 2 = {projections})")
    if over:
        raise _refuse(f'{len(over)} prompt(s) exceed the probe ceiling of {MAX_PROJECTIONS} full-vocabulary '
                      'projections: ' + '; '.join(over[:20]) + ('; and more' if len(over) > 20 else '') + '.',
                      'Raise positionStride, arm fewer layers, or set maxTokens. Lowering topK does not reduce '
                      'the count (k selects from each projection), and a directions-only probe (topK 0, no pins) '
                      'costs none.')
    known = all(p['projections'] is not None for p in prompts)
    basis = 'exact' if counts is not None else 'maxTokens bound'
    geometry = cached_geometry(config.modelID, config.revision)
    if geometry and geometry.get('hiddenSize') and geometry['hiddenSize'] != record.dModel:
        raise _refuse(f'The lens has width {record.dModel} but {config.modelID} has hidden size '
                      f"{geometry['hiddenSize']}: the readout cannot be computed.",
                      'Choose a lens fitted on a model of this width.')
    longest = max((p['tokenCount'] for p in prompts if p['tokenCount'] is not None), default=None)
    memory = memory_estimate(config, geometry, record.dModel, layers, longest, basis)
    converted = artifact_paths.converted_file(record.lensID, record.converted.path, str(root)) \
        if record.converted else None
    lens_bytes = os.path.getsize(converted) if converted and os.path.isfile(converted) else 0
    output = {'files': RUN_FILES + FILES_PER_PROMPT * len(prompts), 'basis': 'estimated upper bound',
              'limitation': 'Indented JSON and CSV text, sized conservatively per stored value; tokenizers '
                            'with long or escaped pieces write more.'}
    if known:
        sizes = output_estimate(config, prompts, layers)
        for entry, size in zip(prompts, sizes):
            entry['outputBytes'] = size
        output['bytes'] = sum(sizes)
        output['tokenBasis'] = basis
        if basis == 'exact' and output['bytes'] > archives.MAX_BYTES:
            raise _refuse(f"The run would write about {output['bytes'] / 1024**3:.1f} GiB, more than the "
                          f'{archives.MAX_BYTES // 1024**3} GiB one export can carry home.',
                          'Lower topK, raise positionStride, arm fewer layers, or split the prompt set across requests.')
    workload = walltime_estimate(config, prompts, layers, record.dModel, geometry, lens_bytes) if known else None
    total = sum(p['projections'] for p in prompts) if known else None
    summary = [f"{len(prompts)} prompt(s) through {len(layers)} armed layer(s), reading "
               f"{'every position' if stride == 1 else f'one position in {stride}'}: "
               + (f'{total:,} full-vocabulary projections in all.' if known else 'projection counts need the token counts.'),
               'The model loads once, and the lens is verified and placed on the GPU once for the whole set.']
    if unavailable:
        summary.append(unavailable + ' Execution re-checks every prompt against the ceiling before any forward pass.')
    summary.append(memory['summary'])
    if 'bytes' in output:
        summary.append(f"About {output['bytes'] / 1024**3:.2f} GiB of output in {output['files']} files, in one run directory.")
    summary += advisories
    # The plan travels in the job record, so it lists a bounded number of
    # prompts; the largest one is always named, and every count is in the totals.
    listed = [{key: entry[key] for key in entry if key not in ('labels', 'textBytes')}
              for entry in prompts[:REVIEW_PROMPTS_LISTED]]
    largest = max(prompts, key=lambda p: p['projections'] or 0) if known else None
    return {'prompts': listed, 'promptCount': len(prompts), 'promptsListed': len(listed),
            **({'largestPrompt': {k: largest[k] for k in ('id', 'tokenCount', 'projections')}} if largest else {}),
            'armedLayers': layers,
            'projectionCeiling': MAX_PROJECTIONS, 'totalProjections': total,
            'tokenCounts': 'exact' if counts is not None else 'unavailable',
            **({'tokenCountsUnavailable': unavailable} if unavailable else {}),
            'lens': {'lensID': record.lensID, 'fittedModelID': record.fit.modelID, 'dModel': record.dModel,
                     'targetLayer': record.targetLayer, 'convertedBytes': lens_bytes},
            'geometry': geometry or {'status': 'unavailable'}, 'memoryEstimate': memory, 'output': output,
            **({'walltimeEstimate': workload} if workload else {}),
            'advisories': advisories, 'qualification': 'notPerformed', 'summary': summary}


# --- execution -----------------------------------------------------------------

def probe_batch(config, *, root, log=print, on_run_created=None, model=None):
    """Read every prompt under one condition; write one run directory.

    Returns a small record for the job: the run, its manifest, and the
    counts. The per-prompt reports stay on disk, never in the job record.
    """
    from ..jlens import lens_store, probe
    from . import paths

    started = time.monotonic()
    root = str(Path(root).resolve())
    rows = read_prompts(config, root)
    record = lens_store.resolve(config.lensID, root)
    layers = armed_layers(config, record)
    load_started = time.monotonic()
    if model is None:
        from ..steering import model_loader
        log(f'[jlens-probe-batch] loading {config.modelID} …')
        model = model_loader.load(config.modelID, config.revision, dtype=config.dtype, device=config.device)
    load_seconds = time.monotonic() - load_started
    session = probe.open_session(
        config.modelID, lens_id=config.lensID, layers=layers, top_k=config.topK,
        pin_words=list(config.pin or []), pin_ids=list(config.pinIDs or []),
        directions=list(config.directions or []), variant_path=config.variantPath,
        prompt_mode=config.promptMode, system_prompt=config.systemPrompt,
        revision=config.revision, root=root, model=model, log=log)
    # Every prompt is rendered and priced before any forward pass, so a set
    # with prompts over the ceiling fails here with all of them named, and
    # leaves no run behind.
    planned, over = [], []
    for row in rows:
        try:
            planned.append(probe.plan_prompt(session, row['text'], position_stride=config.positionStride,
                                             max_tokens=config.maxTokens))
        except probe.ProbeError as exc:
            over.append(f"{row['id']}: {exc}")
    if over:
        raise _refuse('Prompt(s) the probe cannot read as asked: ' + ' | '.join(over[:20]) + '.',
                      'Raise positionStride, arm fewer layers, or set maxTokens, then review and submit again.')
    run = paths.make_unique_run_directory(f"{RUN_TYPE}-{config.modelID.replace('/', '--')}", root)
    if on_run_created:
        on_run_created(run)
    timings = {'modelLoadSeconds': round(load_seconds, 3)}
    entries = []
    try:
        build_started = time.monotonic()
        session.ensure_readout()
        timings['readoutBuildSeconds'] = round(time.monotonic() - build_started, 3)
        directory = Path(run) / PROMPTS_DIRECTORY
        directory.mkdir()
        for index, (row, plan) in enumerate(zip(rows, planned), start=1):
            target = directory / row['id']
            target.mkdir()
            log(f"[jlens-probe-batch] prompt {index}/{len(rows)}: {row['id']}")
            prompt_started = time.monotonic()
            report = probe.read_prompt(
                session, row['text'], plan, position_stride=config.positionStride,
                max_tokens=config.maxTokens, run_directory=str(target),
                recorded_run_directory=target.relative_to(root).as_posix())
            entries.append({'id': row['id'], 'labels': row['labels'],
                            'directory': PROMPTS_DIRECTORY + '/' + row['id'],
                            'tokenCount': len(plan['ids']), 'scoredPositions': len(plan['scored']),
                            'projections': plan['projections'], 'cells': len(report['cells']),
                            'trajectoryRows': len(report['trajectory']),
                            'seconds': round(time.monotonic() - prompt_started, 3),
                            'bytes': sum(p.stat().st_size for p in target.iterdir() if p.is_file())})
            del report
    except Exception as exc:
        # The prompts already read stay readable: the manifest says which,
        # and the run carries no COMPLETED marker.
        timings['totalSeconds'] = round(time.monotonic() - started, 3)
        _write_manifest(run, config, session, record, layers, rows, entries, timings,
                        status='incomplete', error=f'{type(exc).__name__}: {exc}')
        raise
    timings['promptSeconds'] = round(sum(e['seconds'] for e in entries), 3)
    timings['totalSeconds'] = round(time.monotonic() - started, 3)
    manifest = _write_manifest(run, config, session, record, layers, rows, entries, timings,
                               status='completed')
    (Path(run) / 'COMPLETED').write_text(OPERATION + '\n')
    log(f"[jlens-probe-batch] {len(entries)} prompt(s), {manifest['claim']} "
        f"[{manifest['evidenceTier']}] → {run}")
    return {'runDirectory': run, 'reportPath': str(Path(run) / MANIFEST), 'prompts': len(entries),
            'projections': manifest['totals']['projections'], 'claim': manifest['claim'],
            'evidenceTier': manifest['evidenceTier']}


def _write_manifest(run, config, session, record, layers, rows, entries, timings, *, status, error=None):
    """The run's own config.json and probe-batch.json: what was read, from which bytes, at what cost."""
    from ..jlens import probe
    from .optvec_jspace import evidence_tier_for, qualification_state
    from .run_config import write_run_config

    tier = evidence_tier_for(record)
    stamp, claim = qualification_state(record, session.model)
    stamp, claim = probe.downgrade_for_unpinned_agent(stamp, claim, session.variant_identity)
    dtype = getattr(session.model, 'dtype', None)
    write_run_config(run, RUN_TYPE, model_id=config.modelID, revision=session.revision, dtype=dtype,
                     notes={'lensID': config.lensID, 'evidenceTier': tier, 'qualification': stamp,
                            'claim': claim, 'armedLayers': layers, 'prompts': len(entries),
                            'variant': getattr(session.variant, 'name', None),
                            'isBaseModel': session.variant is None})
    identity = session.variant_identity
    builds = session.readout_builds
    manifest = {
        'schemaVersion': SCHEMA_VERSION, 'operation': OPERATION, 'runType': RUN_TYPE,
        'runID': os.path.basename(run), 'status': status, 'config': config.to_dict(),
        'evidenceTier': tier, 'qualification': stamp, 'claim': claim,
        'model': {'modelID': config.modelID, 'revision': session.revision, 'dtype': dtype},
        'lens': {'lensID': config.lensID, 'sourceSHA256': record.source.tensorSHA256,
                 'convertedSHA256': record.converted.sha256 if record.converted else None,
                 'targetLayer': record.targetLayer, 'armedLayers': layers},
        'condition': {'variantPath': config.variantPath,
                      'variantName': getattr(session.variant, 'name', None),
                      'isBaseModel': session.variant is None,
                      'promptMode': session.prompt_mode or 'chatAssistant',
                      'systemPromptPresent': bool(session.system_prompt)},
        'inputs': {'prompts': {**config.prompts, 'rows': len(rows)},
                   'variantSHA256': identity.get('variantSHA256'),
                   'adapters': [{key: adapter.get(key) for key in ('adapterDirectory', 'adapterHashLive', 'configHashLive')}
                                for adapter in identity.get('adapters') or []],
                   'injectionVectors': identity.get('injectionVectors', []),
                   'neutralPCBasisSHA256': identity.get('neutralPCBasisSHA256'),
                   'directions': [{'reference': d['reference'], 'tensorSHA256': d['tensorSHA256'],
                                   'sidecarSHA256': d['sidecarSHA256']} for d in session.loaded_directions]},
        'prompts': entries,
        'totals': {'prompts': len(entries), 'projections': sum(e['projections'] for e in entries),
                   'scoredPositions': sum(e['scoredPositions'] for e in entries),
                   'bytes': sum(e['bytes'] for e in entries)},
        'fixedCosts': {'modelLoads': 1, 'readoutBuilds': builds, 'lensVerifications': builds,
                       'jacobianPlacements': len(layers) * builds},
        'timings': timings,
        'conventions': {
            'perPrompt': 'Each prompts/<id>/ holds the jlens probe outputs for that prompt: probe.json (the probe '
                         'schema), probe-topk.csv, probe-trajectory.csv, and config.json. Their numbers are what '
                         'one probe of that prompt computes under this condition.',
            'runDirectory': 'Each probe.json records runDirectory relative to the workspace, so the run names no machine.',
            'sharedCosts': 'The model loaded once; the converted lens was re-hashed and its armed Jacobians placed '
                           'on the device once for the whole set.'},
        'limitations': 'Counts, ranks, and cosines per prompt; no null and no interval. A readout is not evidence '
                       'that a concept caused behavior; that needs a separately designed intervention study.',
        **({'error': error, 'promptsNotRead': [row['id'] for row in rows[len(entries):]]} if error else {}),
    }
    (Path(run) / MANIFEST).write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode() + b'\n')
    return manifest
