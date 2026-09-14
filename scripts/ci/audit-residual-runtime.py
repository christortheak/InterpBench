#!/usr/bin/env python3
"""Pin P4 legacy math and model bodies; runtime orchestration is behavioral."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = 'bda6d71'


def original(path):
    return subprocess.check_output(['git', 'show', BASE + ':' + path], cwd=ROOT, text=True)


def swift_adapter(before, name):
    value = before.replace('var interventions: [any LayerIntervention] = []', 'var interventions: [any LayerIntervention] = []\n    var residualRuntime: ResidualRuntime?')
    needle = '            h = layer(h, mask: mask, cache:'
    assert value.count(needle) == 1
    pos = value.index(needle)
    value = value[:pos] + '''            if let residualRuntime {
                h = residualRuntime.apply(h, site: .pre(i), offset: offset)
            }
''' + value[pos:]
    old = '''            for intervention in interventions {
                h = intervention.apply(h, layer: i, offset: offset)
            }'''
    assert value.count(old) == 1
    value = value.replace(old, '''            if let residualRuntime {
                h = residualRuntime.apply(h, site: .post(i), offset: offset, interventions: interventions)
            } else {
                h = ResidualRuntime.applyLegacy(h, interventions: interventions, layer: i, offset: offset)
            }''')
    return value + '\nextension ' + name + ''': ResidualRuntimeHookable {
    public var residualRuntime: ResidualRuntime? {
        get { model.residualRuntime }
        set { model.residualRuntime = newValue }
    }
}
'''


def check_probe(before, after):
    # Two modeled rewrites of the callback: the offset source (P4), and the
    # CPU-before-float64 transfer with stage-named non-finite checks (a live
    # study run on an Apple Silicon Mac, 2026-09-14). Standardization and
    # layer arithmetic are preserved byte for byte; registration is
    # deliberately not audited.
    expected = before.replace('''        offset = 0
        def observe(h):
            nonlocal offset''', '''        def observe(h, context, state):''').replace(
        '            start = offset; offset += h.shape[1]', '            start = context.offset')
    transfer = [("values = h.detach()[0, [p[0] for p in positions]].to(device='cpu', dtype=torch.float64)",
                 "values = to_cpu_float64(h.detach()[0, [p[0] for p in positions]])\n"
                 "                    if not torch.isfinite(values).all(): raise ProbeError('The observed activation is non-finite before scoring.')"),
                ('z = (values - center) / scale',
                 "z = (values - center) / scale\n"
                 "                    if not torch.isfinite(z).all(): raise ProbeError('Standardization produced a non-finite score; check the fitted center and scale.')"),
                ('for weights, bias, activation in layers:\n                        z = z @ weights.T + bias\n',
                 'for index, (weights, bias, activation) in enumerate(layers):\n                        z = z @ weights.T + bias\n'
                 "                        if not torch.isfinite(z).all(): raise ProbeError(f'Layer {index} arithmetic produced a non-finite score.')\n"),
                ("                    if not torch.isfinite(z).all(): raise ProbeError('The probe produced a non-finite score.')\n", '')]
    for old, new in transfer:
        assert expected.count(old) == 1, 'Modeled transfer rewrite no longer matches the pinned body'
        expected = expected.replace(old, new)
    def bodies(source):
        t = ast.parse(source)
        return {node.name: ast.dump(node) for node in ast.walk(t)
                if isinstance(node, ast.FunctionDef) and node.name in ('__init__', 'callback', 'result', 'create')}
    assert bodies(expected) == bodies(after), 'Probe score or evidence bodies changed'
    helper = next(n for n in ast.parse(after).body if isinstance(n, ast.FunctionDef) and n.name == 'to_cpu_float64')
    body = [n for n in helper.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    pinned = ast.parse("import torch\nreturn tensor.to(device='cpu').to(dtype=torch.float64)").body
    assert [ast.dump(n) for n in body] == [ast.dump(n) for n in pinned], 'Observed-value transfer changed: MPS has no float64, so transfer before casting'


def main():
    for path in (
        'Server/steerlab_server/steering/injector.py',
        'Server/steerlab_server/steering/ablator.py',
        'Server/steerlab_server/steering/plan.py',
        'Server/steerlab_server/steering/trainable_injector.py',
        'Server/steerlab_server/steering/sae_latent.py',
        'Sources/SteeringKit/Injection/VectorInjector.swift',
        'Sources/SteeringKit/Injection/SubspaceAblator.swift',
        'Sources/SteeringKit/Injection/InterventionPlan.swift',
    ):
        before, after = original(path), (ROOT / path).read_text()
        assert before == after, 'Legacy scientific owner changed: ' + path
    for file, name in (('SteeredQwen3', 'SteeredQwen3Model'), ('SteeredGemma3Text', 'SteeredGemma3TextModel')):
        path = 'Sources/SteeringKit/Models/' + file + '.swift'
        assert swift_adapter(original(path), name) == (ROOT / path).read_text(), 'Native model body changed beyond named-site dispatch'
    path = 'Server/steerlab_server/experiment/probe_observation.py'
    before, after = original(path), (ROOT / path).read_text()
    check_probe(before, after)
    for old, new in [('z = (values - center) / scale', 'z = (values + center) / scale'),
                     ("predicted = row['predictsTokenPosition']", "predicted = row['predictsTokenPosition'] + 1"),
                     ("return tensor.to(device='cpu').to(dtype=torch.float64)", "return tensor.to(device='cpu', dtype=torch.float64)")]:
        assert after.count(old) == 1
        try: check_probe(before, after.replace(old, new))
        except AssertionError: pass
        else: raise AssertionError('Probe arithmetic/alignment/transfer negative control accepted')
    runtime = ast.parse((ROOT / 'Server/steerlab_server/steering/runtime.py').read_text())
    helper = next(n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == 'apply_legacy')
    expected = ast.parse('for intervention in interventions:\n    h = intervention.apply(h, layer, offset)\nreturn h')
    assert [ast.dump(n) for n in helper.body[1:]] == [ast.dump(n) for n in expected.body]
    swift_runtime = (ROOT / 'Sources/SteeringKit/ResidualRuntime.swift').read_text()
    expected_swift = """        var result = h
        for intervention in interventions {
            result = intervention.apply(result, layer: layer, offset: offset)
        }
        return result"""
    assert swift_runtime.count(expected_swift) == 1, 'Native legacy chain changed'
    assert expected_swift not in swift_runtime.replace('for intervention in interventions {', 'for intervention in interventions.reversed() {'), 'Native order negative control was ineffective'
    print('P4: legacy math unchanged; native model changes limited to dispatch; probe scoring/evidence unchanged beyond the modeled CPU-first transfer; arithmetic, alignment, and transfer negative controls rejected.')


if __name__ == '__main__': main()
