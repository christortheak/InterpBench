"""Matched readout baselines; no model mutation or additional forward passes.

The pinned jlens HF adapter owns the norm/head/softcap convention. Its private
attributes are used only here to run those same modules with float32 state.
"""
from .jlens_fit import FitError


class Float32Readout:
    def __init__(self, model):
        import torch
        from torch.func import functional_call
        try:
            self.norm = model._final_norm
            self.head = model._lm_head
            self.softcap = model._logit_softcap
            self.device = self.head.weight.device
        except AttributeError as exc:
            raise FitError('Float32 readout needs the pinned HF lens norm/head adapter. Use native readout for another adapter.') from exc
        self.call = functional_call
        self.states = []
        self.additional_bytes = 0
        for module in (self.norm, self.head):
            state = {}
            for name, value in [*module.named_parameters(), *module.named_buffers()]:
                converted = value.detach().to(device=self.device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
                state[name] = converted
                if converted.data_ptr() != value.data_ptr():
                    self.additional_bytes += converted.numel() * converted.element_size()
            self.states.append(state)

    def __call__(self, residual):
        import torch
        # Autocast must not silently lower the requested readout tensor dtype.
        with torch.autocast(device_type=self.device.type, enabled=False):
            hidden = self.call(self.norm, self.states[0], (residual.to(self.device, torch.float32),), strict=True)
            logits = self.call(self.head, self.states[1], (hidden,), strict=True)
            if self.softcap is not None:
                logits = self.softcap * torch.tanh(logits / self.softcap)
        if hidden.dtype != torch.float32 or logits.dtype != torch.float32:
            raise FitError('The output adapter did not preserve float32 readout. Use native readout or a supported adapter.')
        return logits


def precision(model, config, readout):
    head = getattr(model, '_lm_head', None)
    weight = getattr(head, 'weight', None)
    return {
        'requested': config.readoutDtype or 'native',
        'transportTensorDtype': 'float32',
        'nativeReadout': 'Pinned model.unembed: head dtype at norm input, original norm/head parameters, then configured softcap.',
        'nativeNormParameterDtypes': sorted({str(p.dtype).replace('torch.', '') for p in model._final_norm.parameters()}) if hasattr(model, '_final_norm') else [],
        'nativeHeadDtype': str(weight.dtype).replace('torch.', '') if weight is not None else 'adapterDefined',
        'float32Readout': 'float32 transport, norm input/parameters, head input/parameters, and softcap' if readout else None,
        'additionalReadoutParameterBytes': readout.additional_bytes if readout else 0,
        'target': 'Native model.unembed of the same captured final residual; unchanged for both readout modes.',
        'limitations': 'Captured activations retain their forward dtype; casting does not recover lost precision. Internal module arithmetic and backend matmul settings still apply; see runtime. No claim of fp64 reference accuracy or native-logit qualification for an untested model.',
    }


def group():
    return {'positions': 0, 'jsDivergenceSum': 0., 'topKOverlapSum': 0.}


def empty(with_float32):
    result = {'native': {'logitLensToFinal': group()}}
    if with_float32:
        result['float32'] = {key: group() for key in (
            'betweenLenses', 'referenceToFinal', 'candidateToFinal', 'logitLensToFinal',
            'referenceToNativeReadout', 'candidateToNativeReadout',
            'logitLensToNativeReadout', 'finalToNativeReadout')}
    return result


def accumulate(total, result):
    for key in ('positions', 'jsDivergenceSum', 'topKOverlapSum'):
        total[key] += result[key]
    total['effectiveTopK'] = result['effectiveTopK']


def compare(model, path, layer, target, devices, matrices, top_k, totals, readout, distances):
    import torch
    from safetensors import safe_open
    with safe_open(str(path), framework='pt', device='cpu') as saved:
        h = saved.get_tensor(str(layer)).to(device=devices[layer], dtype=torch.float32)
        truth = saved.get_tensor(str(target)).to(device=devices[target])
        for start in range(0, len(h), 8):
            hidden, final_hidden = h[start:start+8], truth[start:start+8]
            final = model.unembed(final_hidden)
            baseline = model.unembed(hidden)
            totals['observedTensorDtypes'] = {'sourceBeforeTransport': str(saved.get_slice(str(layer)).get_dtype()),
                                             'finalResidual': str(final_hidden.dtype).replace('torch.', ''),
                                             'nativeFinalLogits': str(final.dtype).replace('torch.', ''),
                                             'nativeBaselineLogits': str(baseline.dtype).replace('torch.', '')}
            accumulate(totals['native']['logitLensToFinal'], distances(baseline, final, top_k))
            if readout is None:
                continue
            transported = [hidden @ matrix.T for matrix in matrices]
            fp = [readout(value) for value in transported]
            fp_baseline = readout(hidden)
            pairs = [
                ('betweenLenses', fp[0], fp[1]),
                ('referenceToFinal', fp[0], final),
                ('candidateToFinal', fp[1], final),
                ('logitLensToFinal', fp_baseline, final),
                ('logitLensToNativeReadout', fp_baseline, baseline),
                ('finalToNativeReadout', readout(final_hidden), final),
            ]
            for name, left, right in pairs:
                accumulate(totals['float32'][name], distances(left, right, top_k))
            for index, name in enumerate(('referenceToNativeReadout', 'candidateToNativeReadout')):
                accumulate(totals['float32'][name], distances(fp[index], model.unembed(transported[index]), top_k))


def matrix_comparison(reference, candidate):
    """Float64 CPU reductions in bounded row chunks, including MPS execution."""
    import math
    sums = {'reference': 0., 'candidate': 0., 'difference': 0., 'dot': 0.}
    maximum = 0.
    for start in range(0, reference.shape[0], 64):
        a = reference[start:start+64].to(device='cpu').double()
        b = candidate[start:start+64].to(device='cpu').double()
        delta = b-a
        sums['reference'] += float((a*a).sum())
        sums['candidate'] += float((b*b).sum())
        sums['difference'] += float((delta*delta).sum())
        sums['dot'] += float((a*b).sum())
        maximum = max(maximum, float(delta.abs().max()))
    a, b, d = (math.sqrt(sums[key]) for key in ('reference', 'candidate', 'difference'))
    return {'referenceNorm': a, 'candidateNorm': b, 'differenceNorm': d,
            'relativeFrobenius': d/a if a else None, 'maxAbs': maximum,
            'cosine': sums['dot']/(a*b) if a and b else None,
            'definition': 'Candidate minus reference; denominator is reference norm. Zero norms yield null ratios. Loaded float32 transport matrices, float64 CPU reductions.'}


def finish(layers):
    for layer in layers.values():
        for mode in ('native', 'float32'):
            for value in layer.get(mode, {}).values():
                n = value['positions']
                value['meanJSDivergence'] = value['jsDivergenceSum']/n if n else None
                value['meanTopKOverlap'] = value['topKOverlapSum']/n if n else None
