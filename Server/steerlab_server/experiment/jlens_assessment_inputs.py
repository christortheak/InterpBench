"""Temporary selected-token activations for layer-at-a-time lens assessment."""
from pathlib import Path


def resource_plan(config, row_count, width, layers, target):
    """Tensor payload estimates, excluding model, forward, and allocator memory."""
    positions = min(config.maxPositionsPerRow, max(0, config.maxSeqLen-config.skipFirst-1))
    element_bytes = 4 if config.dtype == 'float32' else 2
    row_bytes = len(set([*layers, target])) * positions * width * element_bytes
    return {
        'strategy': 'selected activations on disk; one lens layer pair at a time',
        'maximumPositionsPerRow': positions,
        'selectedActivationRowBytesUpperBound': row_bytes,
        'temporaryActivationBytesUpperBound': row_count * row_bytes,
        'float32LensPairBytes': 2 * width * width * 4,
        'limitations': 'Tensor payload only, not peak memory or free-space requirements. '
                      'Allow for the model, forward activations, eight-position vocabulary logits, '
                      'transfers, file headers, and allocator overhead.',
    }


def capture(model, config, rows, layers, target, directory):
    """Forward each usable row once; preserve activation dtype and token order."""
    import torch
    from safetensors.torch import save_file
    captured, devices, handles = {}, {}, []
    results, files = [], []
    positions = []
    payload_bytes = 0
    try:
        for layer in sorted(set([*layers, target])):
            def collect(module, inputs, output, layer=layer):
                residual = output[0] if isinstance(output, tuple) else output
                devices[layer] = residual.device
                captured[str(layer)] = residual.detach()[0, positions].to('cpu').contiguous()
            handles.append(model.layers[layer].register_forward_hook(collect))
        with torch.no_grad():
            for index, row in enumerate(rows):
                captured.clear()
                tokens = model.encode(row['text'], max_length=config.maxSeqLen)
                positions = list(range(config.skipFirst, int(tokens.shape[1])-1))[:config.maxPositionsPerRow]
                if not positions:
                    results.append({'id': row['id'], 'status': 'skipped-too-short'})
                    continue
                model.forward(tokens)
                path = Path(directory) / f'row-{index}.safetensors'
                save_file(captured, str(path))
                payload_bytes += sum(t.numel()*t.element_size() for t in captured.values())
                files.append(path)
                results.append({'id': row['id'], 'status': 'assessed', 'positions': positions})
                captured.clear()
        return results, files, devices, payload_bytes
    finally:
        for handle in handles:
            handle.remove()
