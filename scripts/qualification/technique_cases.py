"""Small, prospectively declared owner journeys for backend_probe.py.

Fixtures contain no research claims. Two optimization steps exercise the real
loss/backward/artifact/evaluation path; they do not establish useful steering.
"""
import hashlib
import json
import os
from pathlib import Path

OPTIMIZATION = dict(layer=3, steps=2, seed=17, alphaAbsolute=1.0, lr=.01,
                    promptMode='rawCompletion', target='A', options=['A', 'B'])


def optvec(model, output, layer):
    import numpy as np
    from steerlab_server.experiment import optvec_train as training, optvec_eval as evaluation
    from steerlab_server.steering import vector_store

    workspace = output / 'workspace'
    workspace.mkdir()
    inputs = workspace / 'inputs'; inputs.mkdir()
    def rows(name, prompts):
        values = [dict(id=f'{name}-{i}', prompt=text, options=['A', 'B'], target='A')
                  for i, text in enumerate(prompts)]
        data = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in values).encode()
        path = inputs / (name + '.jsonl'); path.write_bytes(data)
        return dict(path=path.relative_to(workspace).as_posix(), sha256=hashlib.sha256(data).hexdigest())
    train = rows('training', ['Select A or B. Answer:', 'Choose either A or B. Answer:'])
    test = rows('evaluation', ['Pick between A and B. Answer:', 'Give A or B as your answer:'])
    config = dict(modelID=model.model_id, revision=model.revision, layer=layer, name='qualification',
                  dtype='float32', datasets={'targetTrain': train}, alphaAbsolute=1.0,
                  lambdaAnchor=0.0, lambdaCap=0.0, stepsMax=2, lr=.01, seed=17,
                  microbatchSize=1, gradAccumToEffective=1, checkpointEvery=2,
                  promptMode='rawCompletion', gradientCheckpointing=False)
    (output / 'optvec-training-request.json').write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    previous = {key: os.environ.get(key) for key in ('STEERLAB_ROOT', 'STEERLAB_RUN_ROOT')}
    os.environ['STEERLAB_ROOT'] = str(workspace)
    os.environ['STEERLAB_RUN_ROOT'] = str(workspace / 'runs')
    try:
        result = training.train(training.OptVecTrainConfig.from_dict(config), model=model)
        reference = Path(result['vectorArtifactID'])
        vectors, _ = vector_store.load(str(reference.parent), reference.name)
        evidence = {'direction-optvec': np.asarray(vectors.per_layer[layer], dtype=np.float32)}
        curve = [json.loads(line) for line in (Path(result['runDirectory']) / 'metrics.jsonl').read_text().splitlines()]
        (output / 'optvec-curve.json').write_text(json.dumps(curve, indent=2, sort_keys=True) + '\n')
        # Keep numerical step diagnostics in the portable tensor comparison,
        # excluding wall-clock timing and discrete step labels.
        for key in sorted(set.intersection(*(set(row) for row in curve))):
            if all(isinstance(row[key], (int, float)) and not isinstance(row[key], bool) for row in curve) and not any(word in key.lower() for word in ('time', 'second', 'step')):
                evidence['optvec-curve-' + key] = np.asarray([row[key] for row in curve], dtype=np.float32)
        eval_config = dict(vectorArtifact=reference.relative_to(workspace).as_posix(),
                           datasets={'targetTest': test}, modelID=model.model_id, revision=model.revision,
                           dtype='float32', name='qualification', alphaMultiples=[1.0], nullSamples=1,
                           seed=17, microbatchSize=1, promptMode='rawCompletion')
        (output / 'optvec-evaluation-request.json').write_text(json.dumps(eval_config, indent=2, sort_keys=True) + '\n')
        assessed = evaluation.evaluate(evaluation.OptVecEvalConfig.from_dict(eval_config), model=model)
        result = dict(steps=result['steps'], chosenCheckpoint=result['chosenCheckpoint'],
                      recordCount=assessed['recordCount'], doseResponse=assessed['doseResponse'])
        (output / 'optvec-result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        return evidence, result
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
