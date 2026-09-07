#!/usr/bin/env python3
"""Offline, explicitly scoped model probe; never an engine-wide certificate.

Run: PYTHONPATH=Server python scripts/qualification/backend_probe.py run
  --device mps --model Qwen/Qwen3-0.6B --revision <commit> --output <new-dir>
Compare: python scripts/qualification/backend_probe.py compare <reference> <candidate>
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

# Must precede torch/model imports. Unsupported MPS operators must be visible.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Server'))
TOLERANCES = {'atol': .005, 'rtol': .005, 'directionCosine': .999}
TOKENS = [[151644, 872, 198, 9707, 11, 1268, 151645],
          [151644, 872, 198, 3838, 374, 264, 151645],
          [151644, 872, 198, 4438, 311, 1091, 151645],
          [151644, 872, 198, 1986, 525, 499, 151645]]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(args):
    import numpy as np
    import torch
    import transformers
    from steerlab_server.steering import model_loader, vector_math
    from steerlab_server.steering.recorder import ActivationRecorder
    from steerlab_server.steering.plan import Edit, Mode, interventions
    from steerlab_server.experiment.sampling import seeded_generation

    if not re.fullmatch(r'[a-fA-F0-9]{40}', args.revision):
        raise ValueError('Use an immutable 40-hex model revision.')
    if args.layer < 0 or args.context_tokens < 0:
        raise ValueError('Layer and context token counts must be nonnegative.')
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('Evidence belongs outside the checkout.')
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(model=args.model, revision=args.revision, dtype='float32', layer=args.layer,
                    tokens=TOKENS, contextTokens=args.context_tokens, dose=.5, tolerances=TOLERANCES)
    if args.optvec:
        from technique_cases import OPTIMIZATION
        protocol['optvec'] = {**OPTIMIZATION, 'layer': args.layer,
                              'caseSHA256': digest(Path(__file__).with_name('technique_cases.py').read_bytes())}
    identity = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)
    report = dict(schemaVersion=1, protocol=protocol, protocolSHA256=digest(json.dumps(protocol, sort_keys=True).encode()),
                  source=identity, diffSHA256=digest(diff), probeSHA256=digest(Path(__file__).read_bytes()),
                  pythonSourceSHA256=digest(b''.join(p.relative_to(ROOT / 'Server').as_posix().encode() + b'\0' + p.read_bytes() + b'\0' for p in sorted((ROOT / 'Server/steerlab_server').rglob('*.py')))),
                  deviceRequested=args.device, os=platform.platform(), hardware=platform.machine(),
                  software=dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__),
                  fallbackEnabled=False, status='running', scope='model residuals, logits, additive/ablation hooks; RNG scope',
                  unmeasured=['battery', 'OptVec', 'J-lens', 'J-space', 'SAE execution', 'fine-tuning', 'multi-agent transcript'])
    write(output / 'report.json', report)
    start = time.monotonic()
    try:
        if args.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; no CPU substitution is permitted in this measurement.')
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; run this command on the allocated GPU.')
        model = model_loader.load(args.model, args.revision, dtype='float32', device=args.device)
        devices = sorted({str(p.device) for p in model.model.parameters()})
        assert all(d.split(':')[0] == args.device for d in devices), devices
        report.update(deviceActual=devices, attention=model.attn_implementation,
                      modelRevision=model.revision, parameterDtypes=sorted({str(p.dtype) for p in model.model.parameters()}))
        if args.device == 'cuda': report['gpu'] = torch.cuda.get_device_name()
        else:
            report['gpu'] = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip() if sys.platform == 'darwin' else platform.processor()
        evidence = {}
        residuals = []
        for index, ids in enumerate(TOKENS):
            assert max(ids) < model.model.config.vocab_size
            tokens = torch.tensor([ids], device=model.device)
            recorder = ActivationRecorder([args.layer])
            with torch.no_grad(), model.hooked.session([recorder]):
                logits = model.model(input_ids=tokens).logits[0, -1].float().cpu().numpy()
            residual = recorder.captures[-1].values
            residuals.append(residual)
            evidence[f'residual-{index}'] = np.asarray(residual, dtype=np.float32)
            evidence[f'logits-{index}'] = logits
        for method in vector_math.ExtractionMethod:
            # These recipes run on captured CPU rows, not the GPU.
            if method.value not in ('meanDifference', 'lat', 'designatedReference'): continue
            evidence['direction-' + method.value] = np.asarray(vector_math.direction(
                residuals[:2], residuals[2:], method), dtype=np.float32)
        direction = evidence['direction-meanDifference'].tolist()
        ids = TOKENS[0]
        for mode in (Mode.ADD, Mode.ABLATE):
            chain = interventions([Edit(args.layer, direction, .5, mode, 'probe')], prompt_token_count=len(ids))
            recorder = ActivationRecorder([args.layer])
            with torch.no_grad(), model.hooked.session(chain + [recorder]):
                logits = model.model(input_ids=torch.tensor([ids], device=model.device)).logits[0, -1]
            evidence[mode.value + '-logits'] = logits.float().cpu().numpy()
            evidence[mode.value + '-residual'] = np.asarray(recorder.captures[-1].values, dtype=np.float32)
        # A real long forward pass exercises attention/memory, separately from
        # the production generation owner's chunking (not covered by this pass).
        if args.context_tokens:
            tokens = torch.tensor([[872] * args.context_tokens], device=model.device)
            with torch.no_grad(), model.hooked.session([]):
                result = model.model(input_ids=tokens).logits[0, -1]
            assert bool(torch.isfinite(result).all())
            report['longForwardTokens'] = args.context_tokens
        state = torch.mps.get_rng_state().clone() if args.device == 'mps' else None
        def draws(seed):
            with seeded_generation(.7, seed): return torch.rand(64, device=model.device).cpu()
        first = draws(91); draws(22)
        report['rngRepeatable'] = torch.equal(first, draws(91))
        report['mpsStateRestored'] = torch.equal(state, torch.mps.get_rng_state()) if state is not None else None
        if args.optvec:
            from technique_cases import optvec
            tensors, result = optvec(model, output, args.layer)
            evidence.update(tensors)
            report['optvec'] = result
            report['unmeasured'].remove('OptVec')
        if args.device == 'mps':
            report['memory'] = dict(current=torch.mps.current_allocated_memory(), driver=torch.mps.driver_allocated_memory())
        elif args.device == 'cuda': report['memory'] = dict(peak=torch.cuda.max_memory_allocated())
        np.savez(output / 'tensors.npz', **evidence)
        report['tensorsSHA256'] = digest((output / 'tensors.npz').read_bytes())
        report['status'] = 'measured'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsedSeconds'] = time.monotonic() - start
        write(output / 'report.json', report)
    print(json.dumps(report, sort_keys=True))


def compare(args):
    import numpy as np
    reports = [json.loads((p / 'report.json').read_text()) for p in (args.reference, args.candidate)]
    assert all(r['status'] == 'measured' for r in reports), 'Both runs must have completed.'
    assert all(digest(json.dumps(r['protocol'], sort_keys=True).encode()) == r['protocolSHA256']
               for r in reports), 'Protocol metadata changed.'
    assert reports[0]['protocolSHA256'] == reports[1]['protocolSHA256'], 'Protocols differ.'
    arrays = []
    for path, report in zip((args.reference, args.candidate), reports):
        assert digest((path / 'tensors.npz').read_bytes()) == report['tensorsSHA256'], 'Tensor evidence changed.'
        arrays.append(np.load(path / 'tensors.npz', allow_pickle=False))
    assert set(arrays[0].files) == set(arrays[1].files)
    rows = {}
    tolerances = reports[0]['protocol']['tolerances']
    for name in arrays[0].files:
        a, b = (np.asarray(x[name], dtype=np.float64) for x in arrays)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        cosine = float(np.dot(a, b) / denominator) if denominator else None
        passed = cosine is not None and cosine >= tolerances['directionCosine'] if name.startswith('direction-') else bool(np.allclose(a, b, atol=tolerances['atol'], rtol=tolerances['rtol']))
        rows[name] = dict(maxAbs=float(np.max(np.abs(a-b))), rms=float(np.sqrt(np.mean((a-b)**2))),
                          cosine=cosine, argmaxAgrees=bool(a.argmax() == b.argmax()), withinDiagnosticTolerance=passed)
    if 'optvec' in reports[0]['protocol']:
        def numeric(value, prefix=''):
            if isinstance(value, dict):
                return {key: number for name, item in value.items() for key, number in numeric(item, prefix + '.' + name).items()}
            if isinstance(value, list):
                return {key: number for index, item in enumerate(value) for key, number in numeric(item, prefix + '.' + str(index)).items()}
            if isinstance(value, (int, float)) and not isinstance(value, bool): return {prefix: float(value)}
            return {}
        metrics = [numeric(r['optvec']['doseResponse']) for r in reports]
        assert set(metrics[0]) == set(metrics[1]), 'Evaluation metric shapes differ.'
        for name, a in metrics[0].items():
            b = metrics[1][name]
            rows['optvec-evaluation' + name] = dict(maxAbs=abs(a-b),
                withinDiagnosticTolerance=bool(np.isclose(a, b, atol=tolerances['atol'], rtol=tolerances['rtol'])))
    rng_ok = all(r['rngRepeatable'] and r.get('mpsStateRestored') is not False for r in reports)
    print(json.dumps(dict(reference=reports[0]['deviceActual'], candidate=reports[1]['deviceActual'],
                         rngChecksPassed=rng_ok,
                         protocolSHA256=reports[0]['protocolSHA256'], results=rows), indent=2, sort_keys=True))
    return 0 if rng_ok and all(r['withinDiagnosticTolerance'] for r in rows.values()) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    probe = commands.add_parser('run')
    probe.add_argument('--device', choices=['cpu', 'mps', 'cuda'], required=True)
    probe.add_argument('--model', required=True)
    probe.add_argument('--revision', required=True)
    probe.add_argument('--layer', type=int, default=3)
    probe.add_argument('--context-tokens', type=int, default=0)
    probe.add_argument('--optvec', action='store_true', help='Run the declared two-step training/evaluation owner journey')
    probe.add_argument('--output', type=Path, required=True)
    comparison = commands.add_parser('compare')
    comparison.add_argument('reference', type=Path)
    comparison.add_argument('candidate', type=Path)
    args = parser.parse_args()
    sys.exit(run(args) if args.command == 'run' else compare(args))
