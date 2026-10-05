"""Regenerates the science-report fixtures in this directory.

Small, synthetic J-lens assessment reports in both stored shapes, and the page
each one renders to. No model produced these numbers: every value is chosen by
hand so that a reader can check a page against its source.

- ``assessment-list.json``    ``schemaVersion`` 2. One reference, two
  candidates, two texts, three layers, and a float32 readout check. The second
  candidate is a mixed lens fitted on two texts, one of which is assessed
  here, so that comparison is recorded as not held out. One row is skipped.
- ``assessment-single.json``  ``schemaVersion`` 1. One candidate on one text,
  native readout only, with the one-entry ``comparisons`` list.
- ``assessment-legacy.json``  ``schemaVersion`` 1 as written before
  comparisons were listed and before the plain baseline existed.
- ``assessment-list.html``, ``assessment-single.html``  the pages the first
  two render to. ``test_science_report.py`` compares against them byte for
  byte.

Run from ``Server/`` after a deliberate change to the page, then read the
diff of the two ``.html`` files before committing it::

    python tests/fixtures/science-report/generate.py
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SERVER = HERE.parents[2]
sys.path.insert(0, str(SERVER))

from steerlab_server.experiment import diagnostic_archives as archives  # noqa: E402
from steerlab_server.client.reports import jlens_assessment  # noqa: E402

MODEL, REVISION = 'example/tiny-model', 'ab' * 20
REFERENCE, ROUND_TWO, MIXED = 'lens-round-1', 'lens-round-2', 'lens-mixed'
ESSAYS = {'path': 'heldout/essays.jsonl', 'sha256': 'c1' * 32}
RECIPES = {'path': 'heldout/recipes.jsonl', 'sha256': 'c2' * 32}
LAYERS = (0, 1, 2)
TOP_K = 5

#: Sums over token positions, per layer: (top-k overlap, Jensen–Shannon divergence).
#: A mean is its sum divided by the positions, exactly as the engine stores it.
READINGS = {
    ESSAYS['path']: {
        'positions': 40,
        'baseline': ((4, 12, 30), (24, 16, 2)),
        'reference': ((10, 20, 30), (16, 8, 2)),
        ROUND_TWO: ((12, 20, 34), (14, 8, 1)),
        MIXED: ((8, 26, 28), (18, 6, 3)),
        'between': {ROUND_TWO: ((36, 38, 39), (1, 0.5, 0.25)), MIXED: ((30, 34, 36), (3, 2, 1))},
    },
    RECIPES['path']: {
        'positions': 20,
        'baseline': ((3, 5, 16), (11, 9, 1)),
        'reference': ((6, 11, 15), (7, 4, 1.5)),
        ROUND_TWO: ((7, 12, 15), (6, 4, 1.5)),
        MIXED: ((9, 13, 17), (5, 3, 1)),
        'between': {ROUND_TWO: ((18, 19, 19.5), (0.5, 0.25, 0.125)), MIXED: ((15, 17, 18), (1.5, 1, 0.5))},
    },
}
ROWS = {
    ESSAYS['path']: [{'id': 'essay-1', 'status': 'assessed', 'positions': list(range(2, 22))},
                     {'id': 'essay-2', 'status': 'assessed', 'positions': list(range(2, 22))}],
    RECIPES['path']: [{'id': 'recipe-1', 'status': 'assessed', 'positions': list(range(2, 22))},
                      {'id': 'recipe-2', 'status': 'skipped-too-short'}],
}
MATRICES = {ROUND_TWO: (12.5, 1.5, 0.25, 0.992), MIXED: (13.0, 3.0, 0.5, 0.968)}


def group(positions, overlap_sum, divergence_sum, with_k=True):
    value = {'positions': positions, 'jsDivergenceSum': float(divergence_sum), 'topKOverlapSum': float(overlap_sum)}
    if with_k and positions: value['effectiveTopK'] = TOP_K
    value['meanJSDivergence'] = divergence_sum / positions if positions else None
    value['meanTopKOverlap'] = overlap_sum / positions if positions else None
    return value


def lens(lens_id, tensor, prompts, corpus, corpora=None):
    return {
        'lensID': lens_id, 'schemaVersion': 1, 'artifactType': 'jlens-imported', 'substrate': 'python-hf-transformers',
        'source': {'repo': None, 'folder': 'source', 'tensorFile': 'tensorFile.safetensors',
                   'configFile': 'description.json', 'commit': None, 'tensorSHA256': tensor * 32, 'configSHA256': 'd0' * 32},
        'fit': {'modelID': MODEL, 'revision': REVISION, 'revisionKnown': True, 'dtype': 'bfloat16', 'corpus': corpus,
                'promptsFitted': prompts, 'maxSeqLen': 128, 'corpora': corpora},
        'sourceLayers': list(LAYERS), 'dModel': 16, 'targetLayer': 3, 'nPrompts': prompts,
        'converted': {'path': f'runs/jlens-lenses/{lens_id}/jacobians.safetensors', 'dtype': 'float32',
                      'sha256': tensor[::-1] * 32, 'layerCount': len(LAYERS)},
        'configHash': 'd0' * 32, 'referencePackage': None, 'referenceCommit': None, 'kernelSHA256': None,
        'driverSHA256': None, 'fitReportSHA256': 'f0' * 32, 'tier': 'exploratory', 'tierSource': 'custom-artifact',
        'importedAt': '2026-01-01T00:00:00Z', 'qualifications': [],
    }


LENSES = {
    REFERENCE: lens(REFERENCE, 'a1', 100, 'sha256:' + 'f1' * 32),
    ROUND_TWO: lens(ROUND_TWO, 'a2', 400, 'sha256:' + 'f2' * 32),
    MIXED: lens(MIXED, 'a3', 500, 'sha256:' + 'e3' * 32, corpora=[
        {'corpusSHA256': RECIPES['sha256'], 'promptsFitted': 200, 'rowsConsidered': 210},
        {'corpusSHA256': 'f2' * 32, 'promptsFitted': 300, 'rowsConsidered': 300}]),
}


def comparison(candidate, corpus, float32, baseline=True, digest=True):
    stored = READINGS[corpus['path']]
    positions = stored['positions']
    layers, readout = {}, {}
    for index, layer in enumerate(LAYERS):
        def at(name, k=True): return group(positions, stored[name][0][index], stored[name][1][index], k)
        between = group(positions, stored['between'][candidate][0][index], stored['between'][candidate][1][index])
        layers[str(layer)] = {'betweenLenses': between, 'referenceToFinal': at('reference'), 'candidateToFinal': at(candidate)}
        entry = {'native': {'logitLensToFinal': at('baseline')}}
        if float32:
            same = group(positions, positions, 0.0004)
            entry['float32'] = {
                'betweenLenses': between, 'referenceToFinal': at('reference'), 'candidateToFinal': at(candidate),
                'logitLensToFinal': at('baseline'), 'referenceToNativeReadout': same, 'candidateToNativeReadout': same,
                'logitLensToNativeReadout': same, 'finalToNativeReadout': group(positions, positions, 0)}
        size, difference, largest, cosine = MATRICES[candidate]
        entry['matrixComparison'] = {
            'referenceNorm': 12.0 + layer, 'candidateNorm': size + layer, 'differenceNorm': difference,
            'relativeFrobenius': difference / (12.0 + layer), 'maxAbs': largest, 'cosine': cosine,
            'definition': 'Candidate minus reference; denominator is reference norm. Zero norms yield null ratios. '
                          'Loaded float32 transport matrices, float64 CPU reductions.'}
        entry['observedTensorDtypes'] = {'sourceBeforeTransport': 'BF16', 'finalResidual': 'bfloat16',
                                         'nativeFinalLogits': 'bfloat16', 'nativeBaselineLogits': 'bfloat16'}
        readout[str(layer)] = entry
    fitted_here = any(entry['corpusSHA256'] == corpus['sha256'] for entry in LENSES[candidate]['fit']['corpora'] or [])
    value = {
        'referenceLensID': REFERENCE, 'candidateLensID': candidate, 'corpus': dict(corpus),
        'readoutComparison': {'schemaVersion': 1, 'layers': readout, 'precision': {
            'requested': 'float32' if float32 else 'native', 'transportTensorDtype': 'float32',
            'nativeReadout': 'Pinned model.unembed: head dtype at norm input, original norm/head parameters, then configured softcap.',
            'nativeNormParameterDtypes': ['bfloat16'], 'nativeHeadDtype': 'bfloat16',
            'float32Readout': 'float32 transport, norm input/parameters, head input/parameters, and softcap' if float32 else None,
            'additionalReadoutParameterBytes': 1024 if float32 else 0,
            'target': 'Native model.unembed of the same captured final residual; unchanged for both readout modes.',
            'limitations': 'Captured activations retain their forward dtype; casting does not recover lost precision.'}},
        'layers': layers, 'rows': ROWS[corpus['path']],
        'resources': {'strategy': 'selected activations on disk; one lens layer pair at a time', 'stagedRows': 2,
                      'capturedActivationBytes': 5120, 'lensLayerReads': 6, 'lensLayerPlacements': 6},
        'heldOutStatus': 'sameCorpusAsFit' if fitted_here else 'researcherDeclared; overlapNotEstablished',
    }
    if not baseline: del value['readoutComparison']
    return {**value, 'comparisonSHA256': archives.digest(value)} if digest else value


def common(config, lenses):
    return {
        'operation': 'jlens-fit-assess', 'config': config,
        'runtime': {'torch': '2.5.0', 'transformers': '5.6.0', 'dtype': 'bfloat16', 'device': 'cuda', 'attention': 'eager',
                    'modelClass': 'TinyForCausalLM', 'cudaTF32': False, 'optionalKernels': {}, 'kernelSHA256': 'b7' * 32},
        'lenses': [LENSES[name] for name in lenses], 'qualification': 'notPerformed',
        'aggregation': 'Equal weight per assessed token position; first eligible positions up to the displayed cap.',
        'limitations': 'Distributional agreement measures readout stability and agreement with the final residual at '
                       'these positions. It does not prove causal validity, dataset independence, or adequacy for '
                       'every research question.'}


def settings(**extra):
    return {'modelID': MODEL, 'revision': REVISION, 'referenceLensID': REFERENCE, 'sourceLayers': None, 'maxPrompts': 2,
            'maxSeqLen': 32, 'skipFirst': 2, 'maxPositionsPerRow': 20, 'topK': TOP_K, 'dtype': 'bfloat16', 'device': 'cuda',
            **extra}


def list_report():
    pairs = [comparison(candidate, corpus, float32=True) for corpus in (ESSAYS, RECIPES) for candidate in (ROUND_TWO, MIXED)]
    config = settings(candidateLensIDs=[ROUND_TWO, MIXED], corpora=[ESSAYS, RECIPES], readoutDtype='float32')
    return {'schemaVersion': 2, **common(config, (REFERENCE, ROUND_TWO, MIXED)),
            'candidateLensIDs': [ROUND_TWO, MIXED], 'corpora': [ESSAYS, RECIPES], 'comparisons': pairs,
            'resources': {'candidates': 2, 'corpora': 2, 'lensLayerReads': 24, 'lensLayerPlacements': 24}}


def single_report():
    only = comparison(ROUND_TWO, ESSAYS, float32=False)
    return {'schemaVersion': 1, **common(settings(candidateLensID=ROUND_TWO, corpus=ESSAYS), (REFERENCE, ROUND_TWO)),
            **{key: only[key] for key in ('readoutComparison', 'layers', 'rows', 'resources', 'heldOutStatus')},
            'comparisons': [only]}


def legacy_report():
    only = comparison(ROUND_TWO, ESSAYS, float32=False, baseline=False, digest=False)
    return {'schemaVersion': 1, **common(settings(candidateLensID=ROUND_TWO, corpus=ESSAYS), (REFERENCE, ROUND_TWO)),
            **{key: only[key] for key in ('layers', 'rows', 'heldOutStatus')}}


def main():
    for name, report, page in (('assessment-list', list_report(), True), ('assessment-single', single_report(), True),
                               ('assessment-legacy', legacy_report(), False)):
        data = archives.encoded(report)
        (HERE / (name + '.json')).write_bytes(data)
        if page:
            (HERE / (name + '.html')).write_bytes(jlens_assessment.render(data).encode('utf-8'))
    print('Wrote the science-report fixtures in', HERE.name)


if __name__ == '__main__':
    import steerlab_server
    assert Path(steerlab_server.__file__).resolve().is_relative_to(SERVER), 'Run this against the checkout it lives in.'
    main()
