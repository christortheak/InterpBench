"""Compare registered lenses on researcher-selected held-out activations.

One request names a reference lens, one or more candidate lenses, and one or
more held-out corpora. Every distinct lens travels once; activations are
captured once per corpus; candidates are compared one after another so that
at most one lens layer pair is resident, exactly as for a single pair.
"""
from dataclasses import dataclass, asdict
from tempfile import TemporaryDirectory
from pathlib import Path
import uuid
from . import diagnostic_archives as archives, jlens_assessment_inputs as inputs
from .jlens_fit import FitConfig, FitError, file_ref, corpus_rows, read_pinned

COMPARISON_KEYS = ('candidateLensID', 'corpus', 'layers', 'readoutComparison', 'rows', 'resources', 'heldOutStatus')
CANDIDATE_STRATEGY = 'sequential: each candidate is compared with the reference in turn; one lens layer pair is resident at a time'
CORPUS_STRATEGY = 'activations are captured once per corpus, shared by every candidate, and removed before the next corpus'


@dataclass(frozen=True)
class AssessmentConfig:
    modelID: str
    revision: str
    referenceLensID: str
    corpus: dict | None = None
    candidateLensID: str | None = None
    candidateLensIDs: list[str] | None = None
    corpora: list[dict] | None = None
    sourceLayers: list[int] | None = None
    maxPrompts: int = 16
    maxSeqLen: int = 128
    skipFirst: int = 16
    maxPositionsPerRow: int = 64
    topK: int = 10
    dtype: str = 'bfloat16'
    device: str = 'cuda'
    readoutDtype: str | None = None

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value,dict) or value.keys()-cls.__dataclass_fields__.keys(): raise FitError('Use only the documented lens assessment fields.')
        value = dict(value)
        if 'corpus' in value and 'corpora' in value: raise FitError('Choose either corpus (one held-out corpus) or corpora (a list), not both.')
        if 'candidateLensID' in value and 'candidateLensIDs' in value: raise FitError('Choose either candidateLensID (one candidate) or candidateLensIDs (a list), not both.')
        if 'corpora' not in value: value['corpus'] = file_ref(value.get('corpus'), 'held-out corpus')
        else:
            if not isinstance(value['corpora'], list) or not value['corpora']: raise FitError('corpora needs at least one held-out corpus.')
            value['corpora'] = [file_ref(item, 'held-out corpus') for item in value['corpora']]
            if len({item['path'] for item in value['corpora']}) != len(value['corpora']): raise FitError('List each held-out corpus once.')
        if 'candidateLensIDs' in value:
            names = value['candidateLensIDs']
            if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names): raise FitError('candidateLensIDs needs at least one registered lens ID.')
            if len(set(names)) != len(names): raise FitError('List each candidate lens once.')
            if value.get('referenceLensID') in names: raise FitError('The reference lens cannot also be a candidate.')
        elif 'candidateLensID' not in value: raise FitError('Select a model version, corpus, and two registered lenses.')
        try: cfg=cls(**value)
        except TypeError as exc: raise FitError('Select a model version, corpus, and two registered lenses.') from exc
        cfg.model_config()
        if cfg.readoutDtype not in (None, 'native', 'float32'):
            raise FitError('Readout precision must be native or float32; model precision remains separate.')
        for key in ('maxPositionsPerRow','topK'):
            if type(getattr(cfg,key)) is not int or getattr(cfg,key)<1: raise FitError(key+' must be a positive integer.')
        for name in (cfg.referenceLensID, *cfg.candidates):
            if not isinstance(name,str) or len(archives.parts(name))!=1: raise FitError('Choose a registered lens ID, not a file path.')
        return cfg
    def to_dict(self):
        result = asdict(self)
        if self.readoutDtype is None:
            result.pop('readoutDtype')  # Preserve historical effective request bytes.
        for key in ('corpus', 'candidateLensID', 'candidateLensIDs', 'corpora'):
            if result[key] is None: result.pop(key)  # The unused form never appears in request bytes.
        return result
    @property
    def candidates(self):
        return [self.candidateLensID] if self.candidateLensIDs is None else list(self.candidateLensIDs)
    @property
    def corpus_list(self):
        return [self.corpus] if self.corpora is None else list(self.corpora)
    @property
    def multi(self):
        """True for the list form (several candidates and/or corpora): report schema 2."""
        return self.candidateLensIDs is not None or self.corpora is not None
    def model_config(self):
        return FitConfig.from_dict({**{key:getattr(self,key) for key in ('modelID','revision','sourceLayers','maxPrompts','maxSeqLen','skipFirst','dtype','device')}, 'corpus': self.corpus_list[0]})


def instruments(config, root):
    from ..jlens import lens_store
    records=[lens_store.resolve(name,str(root)) for name in (config.referenceLensID,*config.candidates)]
    if any(r.fit.modelID!=config.modelID or (r.fit.revision is not None and r.fit.revision!=config.revision) for r in records):
        raise FitError('Assess lenses against their declared model and known fitted revision.')
    if any((records[0].dModel,records[0].targetLayer)!=(r.dModel,r.targetLayer) for r in records[1:]): raise FitError('The selected lenses have different geometry.')
    layers=sorted(set.intersection(*(set(r.sourceLayers) for r in records))) if config.sourceLayers is None else sorted(config.sourceLayers)
    if not layers or any(layer not in r.sourceLayers for layer in layers for r in records): raise FitError('Choose source layers present in every selected lens.')
    return records,layers


def multi_resources(config, counts, width, layers, target):
    """Plan-level budget for the list form: the maximum over corpora, never a sum."""
    return {**inputs.resource_plan(config, max(counts), width, layers, target),
            'candidates': len(config.candidates), 'corpora': len(counts),
            'candidateStrategy': CANDIDATE_STRATEGY, 'corpusStrategy': CORPUS_STRATEGY,
            'budgetBasis': 'maximum over corpora of the single-pair budget; candidates add no resident tensors'}


def preflight(config, root):
    records,layers=instruments(config,root)
    counts = [min(config.maxPrompts, len(corpus_rows(read_pinned(corpus,root)))) for corpus in config.corpus_list]
    count = max(counts)
    result = {'rows':count, 'sourceLayers':layers, 'qualification':'notPerformed',
              'resources':inputs.resource_plan(config, count, records[0].dModel, layers, records[0].targetLayer)}
    if config.multi:
        result['candidateLensIDs'] = config.candidates
        result['corpora'] = [{**corpus, 'rows': n} for corpus, n in zip(config.corpus_list, counts)]
        result['comparisons'] = len(config.candidates)*len(counts)
        result['resources'] = multi_resources(config, counts, records[0].dModel, layers, records[0].targetLayer)
    if config.readoutDtype is not None:
        result['readoutReview'] = {
            'requested': config.readoutDtype,
            'summary': ('Paired native and float32 readout on the same captured activations. Additional float32 norm/head storage is approximately 4 bytes per parameter (including vocabulary × hidden width), plus logits and workspace. This is not included in the lens-pair budget.'
                        if config.readoutDtype == 'float32' else 'Native readout, with a matched plain-residual baseline.'),
        }
    return result


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


def compare_layer(model, records, root, files, layer, target, devices, top_k, totals, extra=None, readout=None):
    import torch
    from ..jlens import lens_store
    # Function scope releases this pair before the next layer is loaded. Never
    # cache a complete lens or transfer the same matrix for each token chunk.
    matrices = [lens_store.load_layer(record, layer, root=str(root)).to(
        device=devices[layer], dtype=torch.float32) for record in records]
    if extra is not None:
        from . import jlens_assessment_readout as enhanced
        extra['matrixComparison'] = enhanced.matrix_comparison(*matrices)
    for path in files:
        compare_row(model, path, layer, target, devices, matrices, top_k, totals)
        if extra is not None:
            enhanced.compare(model, path, layer, target, devices, matrices, top_k, extra, readout, distances)


def held_out_status(pair, corpus):
    # A mixed-corpus lens fitted on this corpus among others is not held out either.
    same_corpus=any(record.fit.corpus=='sha256:'+corpus['sha256']
                    or any(isinstance(entry,dict) and entry.get('corpusSHA256')==corpus['sha256'] for entry in record.fit.corpora or [])
                    for record in pair)
    return 'sameCorpusAsFit' if same_corpus else 'researcherDeclared; overlapNotEstablished'


def compare_candidate(model, config, pair, root, corpus, layers, target, staged, readout):
    """One candidate against the reference on one corpus's staged activations."""
    from . import jlens_assessment_readout as enhanced
    row_results, files, devices, payload_bytes = staged
    totals={str(layer):{name:{'positions':0,'jsDivergenceSum':0.,'topKOverlapSum':0.} for name in ('betweenLenses','referenceToFinal','candidateToFinal')} for layer in layers}
    extras = {str(layer): enhanced.empty(config.readoutDtype == 'float32') for layer in layers}
    for layer in layers if files else []:
        compare_layer(model,pair,root,files,layer,target,devices,config.topK,totals[str(layer)],extras[str(layer)],readout)
    resources={**inputs.resource_plan(config,len(row_results),model.d_model,layers,target),
               'capturedActivationBytes':payload_bytes,'stagedRows':len(files),
               'lensLayerReads':2*len(layers) if files else 0,
               'lensLayerPlacements':2*len(layers) if files else 0}
    for groups in totals.values():
        for value in groups.values():
            n=value['positions']
            value['meanJSDivergence']=value['jsDivergenceSum']/n if n else None
            value['meanTopKOverlap']=value['topKOverlapSum']/n if n else None
    enhanced.finish(extras)
    comparison = {'referenceLensID':config.referenceLensID,'candidateLensID':pair[1].lensID,'corpus':dict(corpus),
                  'readoutComparison':{'schemaVersion':1, 'precision':enhanced.precision(model, config, readout if files else None), 'layers':extras},
                  'layers':totals,'rows':row_results,'resources':resources,'heldOutStatus':held_out_status(pair, corpus)}
    return {**comparison, 'comparisonSHA256': archives.digest(comparison)}


def comparison_name(comparison):
    return comparison['candidateLensID'] + '--' + comparison['corpus']['sha256'][:8] + '.json'


def assess(config, *, root, log=print, on_run_created=None):
    import torch
    from . import jlens_fit_model, jlens_assessment_readout as enhanced
    records,layers=instruments(config,root)
    corpora=[(corpus, corpus_rows(read_pinned(corpus,root))[:config.maxPrompts]) for corpus in config.corpus_list]
    root=Path(root).resolve(strict=True)
    parent=archives.ordinary(root,'runs',missing=True);parent.mkdir(exist_ok=True)
    run=parent/('jlens-assessment-'+uuid.uuid4().hex);run.mkdir()
    if on_run_created:on_run_created(str(run))
    model,runtime=jlens_fit_model.load(config.model_config(),log)
    try:
        if model.d_model!=records[0].dModel or model.n_layers-1!=records[0].targetLayer: raise FitError('Loaded model geometry differs from the fitted instruments.')
        target=model.n_layers-1
        readout = None
        comparisons = []
        scratch=archives.ordinary(root,'.steerlab/jlens-assessment-state',missing=True)
        scratch.mkdir(parents=True,exist_ok=True)
        with torch.no_grad():
            for index,(corpus,rows) in enumerate(corpora):
                # One staging per corpus; released before the next corpus is captured.
                with TemporaryDirectory(prefix=f'{run.name}-{index}-',dir=scratch) as directory:
                    staged=inputs.capture(model,config,rows,layers,target,directory)
                    if staged[1] and config.readoutDtype == 'float32' and readout is None:
                        readout = enhanced.Float32Readout(model)
                    for candidate in records[1:]:
                        comparisons.append(compare_candidate(model,config,[records[0],candidate],root,corpus,layers,target,staged,readout))
        common={'operation':'jlens-fit-assess','config':config.to_dict(),'runtime':runtime,
                'lenses':[record.to_dict() for record in records],'qualification':'notPerformed',
                'aggregation':'Equal weight per assessed token position; first eligible positions up to the displayed cap.',
                'limitations':'Distributional agreement measures readout stability and agreement with the final residual at these positions. It does not prove causal validity, dataset independence, or adequacy for every research question.'}
        if config.multi:
            counts=[len(rows) for _,rows in corpora]
            report={'schemaVersion':2,**common,'candidateLensIDs':config.candidates,'corpora':[dict(corpus) for corpus,_ in corpora],
                    'comparisons':comparisons,
                    'resources':{**multi_resources(config,counts,model.d_model,layers,target),
                                 'capturedActivationBytesMaximum':max(c['resources']['capturedActivationBytes'] for c in comparisons),
                                 'lensLayerReads':sum(c['resources']['lensLayerReads'] for c in comparisons),
                                 'lensLayerPlacements':sum(c['resources']['lensLayerPlacements'] for c in comparisons)}}
        else:
            only=comparisons[0]
            report={'schemaVersion':1,**common,**{key:only[key] for key in ('readoutComparison','layers','rows','resources','heldOutStatus')},
                    'comparisons':comparisons}
        (run/'comparisons').mkdir()
        for comparison in comparisons:
            (run/'comparisons'/comparison_name(comparison)).write_bytes(archives.encoded(comparison))
        (run/'assessment-report.json').write_bytes(archives.encoded(report));(run/'COMPLETED').write_text('jlens-fit-assess\n')
        return {'runDirectory':str(run),'reportPath':str(run/'assessment-report.json'),'qualification':'notPerformed',
                'comparisonReports':[str(run/'comparisons'/comparison_name(c)) for c in comparisons]}
    finally:
        if hasattr(model,'steerlab_kernel_selection'):model.steerlab_kernel_selection.close()
