"""Fitting orchestration around the pinned reference's per-prompt Jacobian."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from . import diagnostic_archives as archives, jlens_fit_model
from .jlens_fit import ESTIMATOR, FitError, checkpoint_files, corpus_rows, preflight, read_pinned


def write_json(path, value):
    path.write_bytes(archives.encoded(value))


def load_checkpoint(config, root, destination):
    if config.checkpoint is None:
        return None
    files = checkpoint_files(config.to_dict(), root)
    data = read_pinned(config.checkpoint, root, limit=8 * 1024**2)
    state = json.loads(data)
    destination.mkdir()
    (destination / 'state.json').write_bytes(data)
    tensors = destination / 'sums.safetensors'
    shutil.copyfile(archives.ordinary(root, files[0]), tensors)
    if archives.file_hash(tensors) != state['tensorSHA256']:
        raise FitError('Checkpoint changed while capturing its tensors; choose a completed snapshot again.')
    return state, tensors


def restored(state_and_path, identity, layers, width, budget, row_count):
    import torch
    from safetensors.torch import load_file
    if state_and_path is None:
        return {layer: torch.zeros((width, width), dtype=torch.float32) for layer in layers}, 0, 0, []
    state, path = state_and_path
    if state.get('identity') != identity or state.get('identitySHA256') != archives.digest(identity):
        raise FitError('Checkpoint model, corpus, runtime, or estimator differs. Restore the matching inputs or start a new fit.')
    count, next_row = state.get('nDone'), state.get('nextIndex')
    skipped = state.get('skipped')
    if (type(count) is not int or type(next_row) is not int or not 0 <= count <= next_row <= row_count
            or next_row > budget or not isinstance(skipped, list)
            or any(type(x) is not int or x < 0 or x >= next_row for x in skipped)
            or len(set(skipped)) != len(skipped) or count + len(skipped) != next_row):
        raise FitError('Checkpoint progress is invalid or exceeds the selected prompt limit.')
    tensors = load_file(str(path), device='cpu')
    if set(tensors) != {str(layer) for layer in layers}:
        raise FitError('Checkpoint source layers differ from its description.')
    sums = {}
    for layer in layers:
        value = tensors[str(layer)]
        if value.dtype != torch.float32 or tuple(value.shape) != (width, width) or not bool(torch.isfinite(value).all()):
            raise FitError('Checkpoint tensors must be finite float32 sums with the fitted geometry.')
        sums[layer] = value
    return sums, count, next_row, skipped


def save_checkpoint(parent, identity, sums, count, next_row, skipped):
    """Publish a coherent snapshot before pruning earlier scratch generations."""
    from safetensors.torch import save_file
    target = parent / ('snapshot-' + uuid.uuid4().hex)
    with tempfile.TemporaryDirectory(prefix='.writing-', dir=parent) as folder:
        folder = Path(folder)
        tensors = folder / 'sums.safetensors'
        save_file({str(layer): value for layer, value in sums.items()}, str(tensors))
        state = {'schemaVersion': 1, 'identity': identity, 'identitySHA256': archives.digest(identity),
                 'nDone': count, 'nextIndex': next_row, 'skipped': skipped,
                 'tensorFile': tensors.name, 'tensorSHA256': archives.file_hash(tensors)}
        write_json(folder / 'state.json', state)
        archives.publish_directory(folder, target)
    return target


def execute(config, *, root, log=print, on_run_created=None):
    preflight(config, root)
    try:
        import torch
        from jlens.fitting import jacobian_for_prompt, valid_position_mask
        from safetensors.torch import save_file
    except ImportError as exc:
        raise FitError('This engine needs the pinned jlens optional extra to fit a lens. Authoring requires no GPU package.') from exc
    from .run_config import write_run_config
    root = Path(root).resolve(strict=True)
    corpus = read_pinned(config.corpus, root)
    rows = corpus_rows(corpus)
    budget = min(config.maxPrompts, len(rows))
    run_id = 'jlens-fit-' + uuid.uuid4().hex
    runs = archives.ordinary(root, 'runs', missing=True); runs.mkdir(exist_ok=True)
    run = runs / run_id; run.mkdir()
    inputs = run / 'inputs'; inputs.mkdir()
    (inputs / 'corpus.jsonl').write_bytes(corpus)
    # Captured inputs remain immutable; working snapshots are separate scratch.
    state_parent = archives.ordinary(root, '.steerlab/jlens-fitting-state/' + run_id, missing=True)
    state_parent.mkdir(parents=True)
    captured = load_checkpoint(config, root, state_parent / 'input-checkpoint')
    if captured is not None:
        shutil.copyfile(state_parent / 'input-checkpoint/state.json', inputs / 'checkpoint-source.json')
    write_json(run / 'fitting-request.json', {
        'config': config.to_dict(), 'checkpointScratch': state_parent.relative_to(root).as_posix(),
        'rendering': 'raw text; tokenizer native special tokens; no chat template',
        'estimator': ESTIMATOR})
    if on_run_created: on_run_created(str(run))
    log('Fitting run: ' + str(run))
    log('Checkpoint scratch: ' + str(state_parent))
    started = time.perf_counter()
    model, runtime = jlens_fit_model.load(config, log)
    width, target = model.d_model, model.n_layers - 1
    layers = sorted(config.sourceLayers) if config.sourceLayers is not None else list(range(target))
    if not layers or layers[-1] >= target:
        raise FitError('Choose source layers before the final block; a model needs at least two blocks.')
    # Bind numerics and corpus bytes, not output locations, prompt budget, or intent.
    identity = {'estimator': ESTIMATOR, 'modelID': config.modelID, 'revision': config.revision,
                'corpusSHA256': config.corpus['sha256'], 'sourceLayers': layers,
                'targetLayer': target, 'hiddenSize': width, 'maxSeqLen': config.maxSeqLen,
                'skipFirst': config.skipFirst, 'dimBatch': config.dimBatch, 'runtime': runtime}
    sums, count, next_row, skipped = restored(captured, identity, layers, width, budget, len(rows))
    starting_count, starting_row = count, next_row
    write_run_config(str(run), 'jlens-fit', model_id=config.modelID, revision=config.revision,
                     dtype=runtime['dtype'], notes={'fittingIdentity': identity, 'maxPrompts': budget, 'tier': config.tier})
    estimates = {'matrixBytesFloat32': len(layers) * width * width * 4,
                 'backwardPassesPerPrompt': math.ceil(width / config.dimBatch),
                 'promptBudget': budget, 'sourceLayers': layers, 'targetLayer': target,
                 'warning': 'Backward activations and model weights are additional; this is not a memory-fit guarantee.'}
    write_json(run / 'resource-estimate.json', estimates)
    log(f'{budget} corpus rows; {estimates["backwardPassesPerPrompt"]} backward passes per usable row. '
        f'One float32 matrix set: {estimates["matrixBytesFloat32"]} bytes.')
    latest = None
    pending = 0
    diagnostics = run / 'progress.jsonl'
    try:
        for index in range(next_row, budget):
            row = rows[index]
            token_ids = model.encode(row['text'], max_length=config.maxSeqLen)
            length = int(token_ids.shape[1])
            begin = time.perf_counter()
            if length <= config.skipFirst + 1:
                skipped.append(index)
                event = {'index': index, 'id': row['id'], 'tokens': length, 'status': 'skipped-too-short'}
                log(f'Row {index + 1}/{budget}: too short after tokenization; recorded as skipped.')
            else:
                # Token mask validation is independent of the reference's kernel errors.
                valid_position_mask(length, skip_first=config.skipFirst)
                per_prompt, tokens, valid = jacobian_for_prompt(
                    model, row['text'], layers, target_layer=target,
                    dim_batch=config.dimBatch, max_seq_len=config.maxSeqLen, skip_first=config.skipFirst)
                relative = []
                for layer in layers:
                    value = per_prompt[layer]
                    if tuple(value.shape) != (width, width) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                        raise FitError('The model produced invalid Jacobians. Try a smaller pilot in float32 and inspect its backward implementation.')
                    if count:
                        mean = sums[layer] / count
                        denominator = float(mean.norm())
                        if denominator > 0:
                            relative.append(float((value - mean).norm()) / ((count + 1) * denominator))
                    sums[layer] += value
                    if not bool(torch.isfinite(sums[layer]).all()):
                        raise FitError('The accumulated Jacobian is non-finite; retry with suitable model numerics.')
                count += 1
                event = {'index': index, 'id': row['id'], 'tokens': tokens, 'validPositions': valid,
                         'status': 'fitted', 'seconds': time.perf_counter() - begin,
                         'meanRelativeChangeMax': max(relative) if relative else None}
                log(f'Row {index + 1}/{budget}: fitted in {event["seconds"]:.2f}s; {count} usable rows total.')
            with diagnostics.open('ab') as stream:
                stream.write(archives.encoded(event) + b'\n')
            next_row = index + 1
            pending += 1
            if pending >= config.checkpointEvery or next_row == budget:
                new = save_checkpoint(state_parent, identity, sums, count, next_row, skipped)
                log('Completed checkpoint: ' + str(new / 'state.json'))
                # Only this invocation's prior scratch snapshot is pruned. Imported
                # checkpoints, completed runs, and failed-job scratch are untouched.
                if latest is not None: shutil.rmtree(latest)
                latest = new; pending = 0
        if latest is None:
            latest = save_checkpoint(state_parent, identity, sums, count, next_row, skipped)
        if count == 0:
            raise FitError('No corpus rows left enough tokens for fitting. Supply longer text or adjust the token-position settings.')
        shutil.copytree(latest, run / 'checkpoint')
        maps = {f'layer_{layer}': sums[layer] / count for layer in layers}
        save_file(maps, str(run / 'jacobians.safetensors'))
        description = {'schemaVersion': 1, 'kind': 'jlens', 'modelID': config.modelID,
                       'modelRevision': config.revision, 'hiddenSize': width, 'layerCount': model.n_layers,
                       'tensorFile': 'jacobians.safetensors', 'configFile': 'fit-report.json',
                       'lens': {'tier': config.tier, 'fitDtype': runtime['dtype'], 'targetLayer': target,
                                'layers': {str(layer): f'layer_{layer}' for layer in layers},
                                'promptsFitted': count, 'maxSeqLen': config.maxSeqLen,
                                'corpus': 'sha256:' + config.corpus['sha256']}}
        report = {'schemaVersion': 1, 'operation': 'jlens-fit', 'identity': identity,
                  'promptsFitted': count, 'rowsConsidered': next_row, 'skippedIndices': skipped,
                  'promptsFittedThisRun': count - starting_count, 'rowsConsideredThisRun': next_row - starting_row,
                  'continuedFrom': config.checkpoint, 'elapsedSeconds': time.perf_counter() - started,
                  'tensorSHA256': archives.file_hash(run / 'jacobians.safetensors'),
                  'qualification': 'notPerformed', 'fullDepth': layers == list(range(target)),
                  'nextStep': 'Review artifact-description.json with science artifact-plan, then artifact-import to add this fit to the lens library. Qualification and held-out assessment are separate.'}
        write_json(run / 'fit-report.json', report)
        write_json(run / 'artifact-description.json', description)
        (run / 'COMPLETED').write_text('jlens-fit\n')
        # This run's private working copies are no longer needed. The immutable
        # output contains the final checkpoint and the source checkpoint's hashes.
        try:
            shutil.rmtree(state_parent)
        except OSError as exc:
            log('Fit completed; its private checkpoint scratch could not be removed: ' + str(exc))
        return {'runDirectory': str(run), 'reportPath': str(run / 'fit-report.json'),
                'artifactDescription': str(run / 'artifact-description.json'),
                'checkpoint': str(run / 'checkpoint/state.json'), 'promptsFitted': count,
                'qualification': 'notPerformed'}
    except Exception as exc:
        write_json(run / 'fit-failure.json', {'reason': str(exc), 'nextIndex': next_row,
                    'checkpoint': str(latest.relative_to(root)) if latest else None})
        raise
