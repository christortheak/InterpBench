"""Qualification comparison refuses drift and detects real numerical mismatches."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    # The executable disables fallback before importing torch. Isolate that
    # process-environment setup when importing its comparator into this suite.
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    path = Path(__file__).resolve().parents[2] / 'scripts/qualification/backend_probe.py'
    spec = importlib.util.spec_from_file_location('qualification_probe', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def evidence(probe, path, values, *, score=.1):
    path.mkdir()
    np.savez(path / 'tensors.npz', logits=np.asarray(values))
    protocol = dict(tolerances=probe.TOLERANCES, optvec={'fixture': True})
    report = dict(status='measured', protocol=protocol,
                  protocolSHA256=probe.digest(json.dumps(protocol, sort_keys=True).encode()),
                  tensorsSHA256=probe.digest((path / 'tensors.npz').read_bytes()),
                  rngRepeatable=True, mpsStateRestored=True, deviceActual=['fixture'],
                  optvec={'doseResponse': [{'target': {'meanLogOddsMovement': score}}]})
    probe.write(path / 'report.json', report)
    return report


def test_comparison_detects_logit_and_evaluation_mismatches(tmp_path, probe):
    a, b = tmp_path / 'a', tmp_path / 'b'
    evidence(probe, a, [0., 1.]); evidence(probe, b, [0., 1.])
    args = SimpleNamespace(reference=a, candidate=b)
    assert probe.compare(args) == 0
    report = json.loads((b / 'report.json').read_text())
    report['optvec']['doseResponse'][0]['target']['meanLogOddsMovement'] = .3
    probe.write(b / 'report.json', report)
    assert probe.compare(args) == 1
    report['optvec']['doseResponse'][0]['target']['meanLogOddsMovement'] = .1
    np.savez(b / 'tensors.npz', logits=np.asarray([.02, 1.]))
    report['tensorsSHA256'] = probe.digest((b / 'tensors.npz').read_bytes())
    probe.write(b / 'report.json', report)
    assert probe.compare(args) == 1


def test_comparison_checks_protocol_and_tensor_identity(tmp_path, probe):
    a, b = tmp_path / 'a', tmp_path / 'b'
    evidence(probe, a, [0., 1.]); report = evidence(probe, b, [0., 1.])
    args = SimpleNamespace(reference=a, candidate=b)
    report['protocol']['tolerances']['atol'] = 100
    probe.write(b / 'report.json', report)
    with pytest.raises(AssertionError, match='Protocol metadata changed'):
        probe.compare(args)
    report['protocol']['tolerances']['atol'] = .005
    probe.write(b / 'report.json', report)
    with (b / 'tensors.npz').open('ab') as handle: handle.write(b'drift')
    with pytest.raises(AssertionError, match='Tensor evidence changed'):
        probe.compare(args)
