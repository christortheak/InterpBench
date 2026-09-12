"""Separate immutable binary evaluation reports, pinned to probe and input bytes."""
from dataclasses import dataclass, asdict
from . import probe_artifacts as artifact, probe_data as data
from .probe_artifacts import ProbeError


@dataclass(frozen=True)
class EvaluateConfig:
    probe: dict
    evaluationData: dict
    purpose: str = 'finalTest'

    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or value.keys()-cls.__dataclass_fields__.keys(): raise ProbeError('Use probe, evaluationData, and purpose for evaluation.')
        try: c=cls(**value)
        except TypeError as exc: raise ProbeError('Select the fitted probe and evaluation activations.') from exc
        data.file_ref(c.probe,'probe'); data.file_ref(c.evaluationData,'evaluationData')
        artifact.choice(c.purpose,('finalTest','exploratory'),'purpose')
        return c
    def to_dict(self): return asdict(self)


def scores(document, rows):
    """Bounded float64 batches; compare with the scalar reference in tests."""
    import numpy as np
    result=[]
    for start in range(0,len(rows),256):
        x=np.asarray([r['activation'] for r in rows[start:start+256]],dtype=np.float64)
        x=(x-np.asarray(document['preprocessing']['center']))/np.asarray(document['preprocessing']['scale'])
        for layer in document['layers']:
            x=x @ np.asarray(layer['weights']).T + np.asarray(layer['bias'])
            if layer['activation']=='relu': x=np.maximum(x,0)
        if not np.isfinite(x).all(): raise ProbeError('Evaluation produced a non-finite score. Check parameters and activation magnitudes.')
        result.extend(x[:,0].tolist())
    return result


def assess_rows(document,rows):
    values=scores(document,rows); threshold=document['output']['threshold']
    tp=tn=fp=fn=0
    for row,score in zip(rows,values):
        if row['label']:
            if score>threshold: tp+=1
            else: fn+=1
        elif score>threshold: fp+=1
        else: tn+=1
    def ratio(a,b): return a/b if b else None
    recall=ratio(tp,tp+fn); specificity=ratio(tn,tn+fp)
    # Pairwise AUC via sorted tie groups, O(n log n), with half credit for ties.
    positive=tp+fn; negative=tn+fp; lower_neg=0; wins=0.; i=0
    ordered=sorted(zip(values,[r['label'] for r in rows]))
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        pos=sum(label for _,label in ordered[i:j]); neg=j-i-pos
        wins+=pos*(lower_neg+0.5*neg);lower_neg+=neg;i=j
    return {**data.labels(rows),'accuracy':(tp+tn)/len(rows),
        'balancedAccuracy':(recall+specificity)/2 if recall is not None and specificity is not None else None,
        'precision':ratio(tp,tp+fp),'recall':recall,'specificity':specificity,
        'f1':ratio(2*tp,2*tp+fp+fn),'rocAUC':ratio(wins,positive*negative),
        'confusion':{'truePositive':tp,'trueNegative':tn,'falsePositive':fp,'falseNegative':fn},
        'constantNegativeAccuracy':negative/len(rows),'constantPositiveAccuracy':positive/len(rows),
        'scoreKind':document['output']['scoreKind'],'threshold':threshold,
        'aggregation':'Equal weight per activation row; repeated positions in a group are not independent examples.',
        'readings':[{**data.identity(row),'label':row['label'],'score':score,'predictedPositive':score>threshold} for row,score in zip(rows,values)]}


def evaluate(config, *, root, log=print, on_run_created=None):
    preflight(config, root)
    doc=artifact.validate(artifact.read_json(data.read(config.probe,root)))
    dataset=data.dataset(config.evaluationData,root)
    if dataset['input'] != doc['input']: raise ProbeError('Evaluation activations differ from the probe’s model, site, rendering, or coordinates. Capture matching activations.')
    settings=doc['training']['settings']
    known=[]
    for key in ('fitRows','selectionRows'):
        if isinstance(settings.get(key),list):known.extend(settings[key])
    # External instruments need not use our training provenance layout.
    usable=[r for r in known if isinstance(r,dict) and all(isinstance(r.get(k),str) for k in ('id','group','sourceSHA256'))] if isinstance(known,list) else []
    overlap=data.overlap(usable,dataset['rows'])
    same_hash=any(r['sha256']==config.evaluationData['sha256'] for r in doc['training']['data'])
    reused=same_hash or any(overlap.values())
    status='knownOverlap' if reused else 'noKnownOverlap' if usable else 'notEstablished'
    notes=artifact.limitations(doc)+['No probability calibration is claimed. Null metrics indicate an absent class or undefined denominator. Repeated evaluation can itself become model selection.']
    if reused: notes.append('These examples overlap recorded training or selection inputs. Treat the result as exploratory, not independent final-test evidence.')
    if not usable: notes.append('Row-level training provenance is unavailable. Different file hashes alone cannot establish independence.')
    report={'schemaVersion':1,'operation':'probe-evaluate','config':config.to_dict(),'probe':config.probe,
        'input':dataset['input'],'requestedPurpose':config.purpose,'independence':status,
        'knownOverlap':overlap,'sameDataHashAsTraining':same_hash,'metrics':assess_rows(doc,dataset['rows']),'limitations':notes}
    run=data.new_run(root,'probe-evaluate',on_run_created)
    data.save(run,'evaluation-report.json',report);(run/'COMPLETED').write_text('probe-evaluate\n')
    log('Probe evaluation saved; independence status: '+status)
    return {'runDirectory':str(run),'reportPath':str(run/'evaluation-report.json'),'independence':status}


def preflight(config, root):
    doc=artifact.validate(artifact.read_json(data.read(config.probe,root)))
    dataset=data.dataset(config.evaluationData,root)
    if dataset['input']!=doc['input']:raise ProbeError('Evaluation input binding differs from the probe.')
    return {'counts':data.labels(dataset['rows']),'input':dataset['input']}
