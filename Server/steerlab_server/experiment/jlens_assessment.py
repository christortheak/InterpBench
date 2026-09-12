"""Compare two registered lenses on researcher-selected held-out activations."""
from dataclasses import dataclass, asdict
from tempfile import TemporaryDirectory
from pathlib import Path
import uuid
from . import diagnostic_archives as archives, jlens_assessment_inputs as inputs
from .jlens_fit import FitConfig, FitError, file_ref, corpus_rows, read_pinned


@dataclass(frozen=True)
class AssessmentConfig:
    modelID: str
    revision: str
    corpus: dict
    referenceLensID: str
    candidateLensID: str
    sourceLayers: list[int] | None = None
    maxPrompts: int = 16
    maxSeqLen: int = 128
    skipFirst: int = 16
    maxPositionsPerRow: int = 64
    topK: int = 10
    dtype: str = 'bfloat16'
    device: str = 'cuda'

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value,dict) or value.keys()-cls.__dataclass_fields__.keys(): raise FitError('Use only the documented lens assessment fields.')
        try: cfg=cls(**{**value,'corpus':file_ref(value.get('corpus'),'held-out corpus')})
        except TypeError as exc: raise FitError('Select a model version, corpus, and two registered lenses.') from exc
        cfg.model_config()
        for key in ('maxPositionsPerRow','topK'):
            if type(getattr(cfg,key)) is not int or getattr(cfg,key)<1: raise FitError(key+' must be a positive integer.')
        for key in ('referenceLensID','candidateLensID'):
            name=getattr(cfg,key)
            if not isinstance(name,str) or len(archives.parts(name))!=1: raise FitError('Choose a registered lens ID, not a file path.')
        return cfg
    def to_dict(self):return asdict(self)
    def model_config(self):
        return FitConfig.from_dict({key:getattr(self,key) for key in ('modelID','revision','corpus','sourceLayers','maxPrompts','maxSeqLen','skipFirst','dtype','device')})


def instruments(config, root):
    from ..jlens import lens_store
    records=[lens_store.resolve(name,str(root)) for name in (config.referenceLensID,config.candidateLensID)]
    if any(r.fit.modelID!=config.modelID or (r.fit.revision is not None and r.fit.revision!=config.revision) for r in records):
        raise FitError('Assess lenses against their declared model and known fitted revision.')
    if (records[0].dModel,records[0].targetLayer)!=(records[1].dModel,records[1].targetLayer): raise FitError('The selected lenses have different geometry.')
    layers=sorted(set(records[0].sourceLayers)&set(records[1].sourceLayers)) if config.sourceLayers is None else sorted(config.sourceLayers)
    if not layers or any(layer not in r.sourceLayers for layer in layers for r in records): raise FitError('Choose source layers present in both lenses.')
    return records,layers


def preflight(config, root):
    records,layers=instruments(config,root)
    rows=corpus_rows(read_pinned(config.corpus,root))
    count = min(config.maxPrompts, len(rows))
    return {'rows':count, 'sourceLayers':layers, 'qualification':'notPerformed',
            'resources':inputs.resource_plan(config, count, records[0].dModel, layers, records[0].targetLayer)}


def distances(left, right, top_k):
    import torch
    if left.shape!=right.shape or not bool(torch.isfinite(left).all() and torch.isfinite(right).all()): raise FitError('Readout logits are non-finite or have different vocabulary shapes.')
    a,b=torch.log_softmax(left.float(),-1),torch.log_softmax(right.float(),-1)
    midpoint=torch.logaddexp(a,b)-__import__('math').log(2)
    js=((a.exp()*(a-midpoint)).sum(-1)+(b.exp()*(b-midpoint)).sum(-1))*0.5
    k=min(top_k,left.shape[-1])
    top_a,top_b=a.topk(k,dim=-1).indices,b.topk(k,dim=-1).indices
    overlaps=(top_a.unsqueeze(-1)==top_b.unsqueeze(-2)).any(-1).float().mean(-1)
    return {'positions':left.shape[0],'jsDivergenceSum':float(js.clamp_min(0).sum()),'topKOverlapSum':float(overlaps.sum()),'effectiveTopK':k}


def compare_row(model, path, layer, target, devices, matrices, top_k, totals):
    from safetensors import safe_open
    import torch
    with safe_open(str(path), framework='pt', device='cpu') as saved:
        h = saved.get_tensor(str(layer)).to(device=devices[layer], dtype=torch.float32)
        truth = saved.get_tensor(str(target)).to(device=devices[target])
        # Preserve the previous eight-position chunks and accumulation order.
        for start in range(0, len(h), 8):
            hidden = h[start:start+8]
            logits = [model.unembed(hidden @ matrix.T) for matrix in matrices]
            final = model.unembed(truth[start:start+8])
            for name, left, right in (('betweenLenses', logits[0], logits[1]), ('referenceToFinal', logits[0], final), ('candidateToFinal', logits[1], final)):
                result = distances(left, right, top_k)
                for key in ('positions', 'jsDivergenceSum', 'topKOverlapSum'):
                    totals[name][key] += result[key]
                totals[name]['effectiveTopK'] = result['effectiveTopK']


def compare_layer(model, records, root, files, layer, target, devices, top_k, totals):
    import torch
    from ..jlens import lens_store
    # Function scope releases this pair before the next layer is loaded. Never
    # cache a complete lens or transfer the same matrix for each token chunk.
    matrices = [lens_store.load_layer(record, layer, root=str(root)).to(
        device=devices[layer], dtype=torch.float32) for record in records]
    for path in files:
        compare_row(model, path, layer, target, devices, matrices, top_k, totals)


def assess(config, *, root, log=print, on_run_created=None):
    import torch
    from . import jlens_fit_model
    records,layers=instruments(config,root)
    rows=corpus_rows(read_pinned(config.corpus,root))[:config.maxPrompts]
    root=Path(root).resolve(strict=True)
    parent=archives.ordinary(root,'runs',missing=True);parent.mkdir(exist_ok=True)
    run=parent/('jlens-assessment-'+uuid.uuid4().hex);run.mkdir()
    if on_run_created:on_run_created(str(run))
    model,runtime=jlens_fit_model.load(config.model_config(),log)
    try:
        if model.d_model!=records[0].dModel or model.n_layers-1!=records[0].targetLayer: raise FitError('Loaded model geometry differs from the fitted instruments.')
        target=model.n_layers-1
        totals={str(layer):{name:{'positions':0,'jsDivergenceSum':0.,'topKOverlapSum':0.} for name in ('betweenLenses','referenceToFinal','candidateToFinal')} for layer in layers}
        scratch=archives.ordinary(root,'.steerlab/jlens-assessment-state',missing=True)
        scratch.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(prefix=run.name+'-',dir=scratch) as directory, torch.no_grad():
            row_results,files,devices,payload_bytes=inputs.capture(model,config,rows,layers,target,directory)
            for layer in layers if files else []:
                compare_layer(model,records,root,files,layer,target,devices,config.topK,totals[str(layer)])
            resources={**inputs.resource_plan(config,len(rows),model.d_model,layers,target),
                       'capturedActivationBytes':payload_bytes,'stagedRows':len(files),
                       'lensLayerReads':2*len(layers) if files else 0,
                       'lensLayerPlacements':2*len(layers) if files else 0}
        for groups in totals.values():
            for value in groups.values():
                n=value['positions']
                value['meanJSDivergence']=value['jsDivergenceSum']/n if n else None
                value['meanTopKOverlap']=value['topKOverlapSum']/n if n else None
        same_corpus=any(record.fit.corpus=='sha256:'+config.corpus['sha256'] for record in records)
        report={'schemaVersion':1,'operation':'jlens-fit-assess','config':config.to_dict(),'runtime':runtime,
                'lenses':[record.to_dict() for record in records],'layers':totals,'rows':row_results,'resources':resources,
                'heldOutStatus':'sameCorpusAsFit' if same_corpus else 'researcherDeclared; overlapNotEstablished',
                'qualification':'notPerformed','aggregation':'Equal weight per assessed token position; first eligible positions up to the displayed cap.',
                'limitations':'Distributional agreement measures readout stability and agreement with the final residual at these positions. It does not prove causal validity, dataset independence, or adequacy for every research question.'}
        (run/'assessment-report.json').write_bytes(archives.encoded(report));(run/'COMPLETED').write_text('jlens-fit-assess\n')
        return {'runDirectory':str(run),'reportPath':str(run/'assessment-report.json'),'qualification':'notPerformed'}
    finally:
        if hasattr(model,'steerlab_kernel_selection'):model.steerlab_kernel_selection.close()
