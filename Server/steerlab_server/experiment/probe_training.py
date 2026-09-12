"""CPU binary probe fitting; no model loading or mutation of existing artifacts."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from . import probe_artifacts as artifact, probe_data as data
from .probe_artifacts import ProbeError


@dataclass(frozen=True)
class TrainConfig:
    fitData: dict
    label: str
    selectionData: dict | None = None
    method: str = 'linear-logit-v1'
    normalization: str = 'standardize'
    steps: int = 500
    learningRate: float = 0.05
    l2: float = 0.001
    hiddenWidth: int = 16
    seed: int = 0
    shuffleLabels: bool = False
    negativeLabel: str = 'absent'
    positiveLabel: str = 'present'

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - cls.__dataclass_fields__.keys(): raise ProbeError('Use the documented probe training fields.')
        try: c = cls(**value)
        except TypeError as exc: raise ProbeError('Choose fitting activations and a probe label.') from exc
        data.file_ref(c.fitData, 'fitData')
        if c.selectionData is not None: data.file_ref(c.selectionData, 'selectionData')
        for key in ('label', 'negativeLabel', 'positiveLabel'): artifact.text(getattr(c,key), key)
        if c.negativeLabel == c.positiveLabel: raise ProbeError('Class labels must differ.')
        artifact.choice(c.method, ('mean-difference-v1','linear-logit-v1','mlp-relu-logit-v1'), 'method')
        artifact.choice(c.normalization, ('standardize','none'), 'normalization')
        for key in ('steps','hiddenWidth'): artifact.integer(getattr(c,key),key,1)
        artifact.integer(c.seed,'seed')
        if c.seed > 2**53-1: raise ProbeError('Choose a seed no larger than 2^53-1.')
        if c.steps > 10000 or c.hiddenWidth > 256: raise ProbeError('Use at most 10,000 steps and 256 hidden units for this small-classifier recipe.')
        if artifact.number(c.learningRate,'learningRate') <= 0 or artifact.number(c.l2,'l2') < 0: raise ProbeError('Use positive learningRate and nonnegative l2.')
        if type(c.shuffleLabels) is not bool: raise ProbeError('shuffleLabels must be true or false.')
        return c

    def to_dict(self): return asdict(self)


def loss_gradient(x, y, weights, biases, l2):
    """Mean binary cross entropy + l2/2 * sum(W²), biases unpenalized."""
    import numpy as np
    hidden = np.maximum(x @ weights[0].T + biases[0], 0) if len(weights) == 2 else x
    scores = (hidden @ weights[-1].T + biases[-1])[:, 0]
    loss = np.mean(np.logaddexp(0, scores) - y*scores) + l2/2 * sum(np.sum(w*w) for w in weights)
    # Stable logistic derivative; probabilities here are optimizer internals.
    sigmoid = np.exp(-np.logaddexp(0, -scores))
    delta = ((sigmoid-y)/len(y))[:,None]
    gw = [delta.T @ hidden + l2*weights[-1]]; gb = [delta.sum(axis=0)]
    if len(weights) == 2:
        back = (delta @ weights[-1]) * (hidden > 0)
        gw.insert(0, back.T @ x + l2*weights[0]); gb.insert(0, back.sum(axis=0))
    return float(loss), gw, gb


def fit_parameters(rows, config, log):
    import numpy as np
    x = np.asarray([r['activation'] for r in rows], dtype=np.float64)
    y = np.asarray([r['label'] for r in rows], dtype=np.float64)
    if min(int(y.sum()), len(y)-int(y.sum())) < 2: raise ProbeError('Fitting needs at least two positive and two negative examples. Add labeled examples to the fitting role.')
    rng = np.random.Generator(np.random.PCG64(config.seed))
    if config.shuffleLabels: y = rng.permutation(y)
    center = x.mean(axis=0) if config.normalization == 'standardize' else np.zeros(x.shape[1])
    scale = x.std(axis=0) if config.normalization == 'standardize' else np.ones(x.shape[1])
    scale = np.where(scale > 0, scale, 1.)
    x = (x-center)/scale
    if not np.isfinite(x).all(): raise ProbeError('Preprocessing produced non-finite values. Inspect activation magnitudes.')
    if config.method == 'mean-difference-v1':
        positive, negative = x[y==1].mean(axis=0), x[y==0].mean(axis=0)
        direction = positive-negative; norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm <= 0: raise ProbeError('The class means coincide; a mean-difference reader cannot separate them. Inspect the data or choose another classifier.')
        weights=[(direction/norm)[None,:]]; biases=[np.array([-float((positive+negative) @ weights[0][0]/2)])]
        history=[]
    else:
        width=x.shape[1]
        if config.method == 'mlp-relu-logit-v1':
            weights=[rng.normal(0,(2/width)**0.5,(config.hiddenWidth,width)), rng.normal(0,(1/config.hiddenWidth)**0.5,(1,config.hiddenWidth))]
            biases=[np.zeros(config.hiddenWidth),np.zeros(1)]
        else: weights=[np.zeros((1,width))]; biases=[np.zeros(1)]
        history=[]
        for step in range(config.steps):
            loss, gw, gb=loss_gradient(x,y,weights,biases,config.l2)
            if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in gw+gb): raise ProbeError('Optimization diverged. Reduce the learning rate or inspect the activation scale.')
            if step % 25 == 0 or step == config.steps-1:
                history.append({'step':step,'lossBeforeUpdate':loss}); log(f'Probe fit step {step+1}/{config.steps}: objective {loss:.6g}')
            weights=[w-config.learningRate*g for w,g in zip(weights,gw)]
            biases=[b-config.learningRate*g for b,g in zip(biases,gb)]
    layers=[{'weights':w.tolist(),'bias':b.tolist(),'activation':'relu' if i<len(weights)-1 else 'identity'} for i,(w,b) in enumerate(zip(weights,biases))]
    return {'center':center.tolist(),'scale':scale.tolist()}, layers, history


def train(config, *, root, log=print, on_run_created=None):
    preflight(config, root)
    import numpy as np
    from .probe_evaluation import assess_rows
    fitting=data.dataset(config.fitData,root)
    selection=data.dataset(config.selectionData,root) if config.selectionData else None
    if selection and selection['input'] != fitting['input']: raise ProbeError('Fitting and selection activations must use the same model, site, rendering, and coordinates.')
    if selection and any(data.overlap(fitting['rows'],selection['rows']).values()): raise ProbeError('Fitting and selection rows share IDs, groups, or source text. Keep them separate, or omit selection for a fitting-only exploration.')
    run=data.new_run(root,'probe-train',on_run_created)
    data.save(run,'training-request.json',{'operation':'probe-train','config':config.to_dict()})
    preprocessing,layers,history=fit_parameters(fitting['rows'],config,log)
    from pathlib import Path
    from .diagnostic_archives import file_hash
    settings={**config.to_dict(),'driverSHA256':file_hash(Path(__file__)),'seed':str(config.seed),'numpy':np.__version__,
              'trainingPrecision':'float64','optimizer':'full-batch-gradient-descent-v1','objective':'mean-binary-cross-entropy-plus-l2-weights' if config.method!='mean-difference-v1' else 'unit-positive-minus-negative-means; midpoint threshold',
              'fitRows':[data.identity(r) for r in fitting['rows']],
              'selectionRows':[data.identity(r) for r in selection['rows']] if selection else [],
              'selectionUsedForOptimization':False,'thresholdSelection':'fixed zero before fitting'}
    refs=[{'role':'fit','sha256':config.fitData['sha256']}]
    if selection: refs.append({'role':'selection','sha256':config.selectionData['sha256']})
    doc=artifact.validate({'artifactType':'activation-probe','schemaVersion':1,'label':config.label,
        'createdAt':datetime.now(timezone.utc).isoformat(),'method':config.method,'input':fitting['input'],
        'preprocessing':preprocessing,'layers':layers,
        'output':{'negativeLabel':config.negativeLabel,'positiveLabel':config.positiveLabel,'threshold':0.,'scoreKind':'signedMargin' if config.method=='mean-difference-v1' else 'logit'},
        'training':{'recipeID':'probe-fit-v1/'+config.method,'data':refs,'settings':settings}})
    saved=data.save(run,'trained.probe.json',doc)
    report={'schemaVersion':1,'operation':'probe-train','artifact':saved,'config':config.to_dict(),
            'fit':assess_rows(doc,fitting['rows']), 'selection':assess_rows(doc,selection['rows']) if selection else None,
            'optimization':history,'labelShuffleControl':config.shuffleLabels,
            'limitations':artifact.limitations(doc)+['Fitting metrics reuse training data; selection metrics are not final-test evidence. No calibration or automatic layer/threshold search was performed.']}
    if config.shuffleLabels: report['limitations'].append('Labels were permuted only for fitting; reported metrics use original labels. This is a control instrument, not the ordinary fitted probe.')
    data.save(run,'training-report.json',report); (run/'COMPLETED').write_text('probe-train\n')
    return {'runDirectory':str(run),'artifactPath':saved['path'],'reportPath':str(run/'training-report.json')}


def preflight(config, root):
    fitting=data.dataset(config.fitData,root)
    if fitting['provenance'].get('role') in ('selection','finalTest'):
        raise ProbeError('The selected file was captured for selection or final testing. Choose fitting data, or prepare a new exploratory dataset with its changed purpose documented.')
    counts=data.labels(fitting['rows'])
    if min(counts['positive'],counts['negative'])<2: raise ProbeError('Fitting needs at least two positive and two negative activation rows.')
    if config.selectionData:
        selection=data.dataset(config.selectionData,root)
        if selection['provenance'].get('role')=='finalTest':raise ProbeError('Keep reserved final-test data out of model selection. Choose selection data or omit this optional input.')
        if selection['input']!=fitting['input']:raise ProbeError('Fitting and selection input bindings differ.')
        if any(data.overlap(fitting['rows'],selection['rows']).values()):raise ProbeError('Fitting and selection share row IDs, groups, or source text. Separate their inputs or omit selection for an exploratory fit.')
    return {'counts':counts,'input':fitting['input'],'numpyActivationBytes':8*counts['rows']*fitting['input']['hiddenSize'],
            'steps':config.steps if config.method!='mean-difference-v1' else 0}
