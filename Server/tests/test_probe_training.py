import copy
import json
from pathlib import Path
import numpy as np
import pytest
from steerlab_server.experiment import probe_artifacts as artifact, probe_data as data, probe_training as training, probe_evaluation as evaluation, diagnostic_archives as archives


def sample_dataset(root, name='fit', xor=False):
    fixture=Path(__file__).resolve().parents[2]/'Tests/Fixtures/cross-engine/probe-artifacts.json'
    binding=json.loads(fixture.read_text())['linear']['input']
    xy=[([0,0],False),([0,1],True),([1,0],True),([1,1],False)] if xor else [([-2,0],False),([-1,1],False),([1,0],True),([2,1],True)]
    rows=[{'id':name+str(i),'group':name+str(i),'sourceSHA256':archives.digest([name,i]),'label':y,'activation':x} for i,(x,y) in enumerate(xy)]
    doc={'artifactType':'activation-dataset','schemaVersion':1,'input':binding,'rows':rows,'provenance':{'role':name if name in data.ROLES else 'external'}}
    path=root/(name+'.json');path.write_bytes(archives.encoded(doc))
    return {'path':path.name,'sha256':archives.file_hash(path)}


def fit(root,method='linear-logit-v1',**kwargs):
    ref=sample_dataset(root,xor=method=='mlp-relu-logit-v1')
    cfg=training.TrainConfig.from_dict({'fitData':ref,'label':'Example classifier','method':method,**kwargs})
    result=training.train(cfg,root=root,log=lambda s:None)
    return json.loads(Path(result['artifactPath']).read_bytes()),result,cfg


@pytest.mark.parametrize('method',['mean-difference-v1','linear-logit-v1','mlp-relu-logit-v1'])
def test_fit_scores_match_reference_and_fit_separable_or_xor_data(tmp_path,method):
    doc,result,cfg=fit(tmp_path,method,steps=1500,l2=0,hiddenWidth=8)
    dataset=data.dataset(cfg.fitData,tmp_path)
    values=evaluation.scores(doc,dataset['rows'])
    assert values==pytest.approx([artifact.score(doc,r['activation'],input_binding=doc['input'])['score'] for r in dataset['rows']])
    assert [v>0 for v in values]==[r['label'] for r in dataset['rows']]
    assert (Path(result['runDirectory'])/'COMPLETED').is_file()
    if method=='mean-difference-v1':
        # Centers [0,.5], population scales [sqrt(2.5),.5], class-mean
        # difference only along first coordinate -> unit [1,0], midpoint zero.
        assert doc['preprocessing']['center']==[0,.5]
        assert doc['preprocessing']['scale']==pytest.approx([np.sqrt(2.5),.5])
        assert doc['layers'][0]['weights'][0]==pytest.approx([1,0])
        assert doc['layers'][0]['bias']==pytest.approx([0])


@pytest.mark.parametrize('nonlinear',[False,True])
def test_optimizer_gradients_match_independent_central_differences(nonlinear):
    x=np.array([[.1,.7],[-.4,.2],[.8,-.5]]);y=np.array([0.,1.,1.])
    weights=[np.array([[.2,-.3]])];biases=[np.array([.1])]
    if nonlinear:weights.insert(0,np.array([[.2,.4],[-.3,.8]]));biases.insert(0,np.array([.9,.6]))
    def loss():
        h=x
        for i,(w,b) in enumerate(zip(weights,biases)):
            h=h@w.T+b
            if i<len(weights)-1:h=np.maximum(h,0)
        z=h[:,0]
        return np.mean(np.log(1+np.exp(z))-y*z)+.07/2*sum((w*w).sum() for w in weights)
    computed,gw,gb=training.loss_gradient(x,y,weights,biases,.07)
    assert computed==pytest.approx(loss())
    for param,gradient in zip(weights+biases,gw+gb):
        for index in np.ndindex(param.shape):
            v=param[index];param[index]=v+1e-6;plus=loss();param[index]=v-1e-6;minus=loss();param[index]=v
            assert gradient[index]==pytest.approx((plus-minus)/2e-6,abs=1e-8)


def test_training_is_reproducible_and_does_not_consume_numpy_global_rng(tmp_path):
    np.random.seed(81);state=np.random.get_state()
    first,_,cfg=fit(tmp_path,'mlp-relu-logit-v1',shuffleLabels=True,seed=17)
    second=training.train(cfg,root=tmp_path,log=lambda _:None)
    doc=json.loads(Path(second['artifactPath']).read_text())
    assert first['layers']==doc['layers'] and first['preprocessing']==doc['preprocessing']
    after=np.random.get_state();assert np.array_equal(state[1],after[1]) and state[2:]==after[2:]


def test_selection_never_changes_preprocessing_or_weights(tmp_path):
    first,_,cfg=fit(tmp_path)
    selection=sample_dataset(tmp_path,'selection')
    path=tmp_path/selection['path'];doc=json.loads(path.read_bytes());doc['rows'][0]['activation']=[10000,5000];path.write_bytes(archives.encoded(doc));selection['sha256']=archives.file_hash(path)
    result=training.train(training.TrainConfig.from_dict({**cfg.to_dict(),'selectionData':selection}),root=tmp_path,log=lambda _:None)
    second=json.loads(Path(result['artifactPath']).read_text())
    assert first['preprocessing']==second['preprocessing'] and first['layers']==second['layers']
    assert json.loads(Path(result['reportPath']).read_text())['selection']['rows']==4


def test_final_test_is_not_training_and_overlap_is_not_silently_independent(tmp_path):
    doc,result,cfg=fit(tmp_path)
    before=Path(result['artifactPath']).read_bytes()
    probe={'path':str(Path(result['artifactPath']).relative_to(tmp_path)),'sha256':archives.file_hash(result['artifactPath'])}
    final=sample_dataset(tmp_path,'finalTest')
    with pytest.raises(artifact.ProbeError,match='final testing'):
        training.train(training.TrainConfig.from_dict({'fitData':final,'label':'bad'}),root=tmp_path)
    for ref,status in [(cfg.fitData,'knownOverlap'),(final,'noKnownOverlap')]:
        output=evaluation.evaluate(evaluation.EvaluateConfig.from_dict({'probe':probe,'evaluationData':ref}),root=tmp_path,log=lambda _:None)
        report=json.loads(Path(output['reportPath']).read_text())
        assert report['independence']==status
        assert report['metrics']['accuracy']==1 and report['metrics']['rocAUC']==1
    assert Path(result['artifactPath']).read_bytes()==before


def test_confusion_auc_ties_and_undefined_metrics(tmp_path):
    doc,_,_=fit(tmp_path,'mean-difference-v1');rows=data.dataset(sample_dataset(tmp_path),tmp_path)['rows']
    for r in rows:r['activation']=[0,.5]
    m=evaluation.assess_rows(doc,rows)
    assert m['accuracy']==.5 and m['rocAUC']==.5 and m['precision'] is None
    assert m['confusion']==dict(truePositive=0,trueNegative=2,falsePositive=0,falseNegative=2)
    for r in rows:r['label']=False
    m=evaluation.assess_rows(doc,rows)
    assert m['rocAUC'] is None and m['balancedAccuracy'] is None and m['accuracy']==1


@pytest.mark.parametrize('key,value',[('steps',True),('l2',-1),('learningRate',float('nan')),('hiddenWidth',999),('shuffleLabels','false'),('seed',-1)])
def test_bad_settings_refuse_before_execution(tmp_path,key,value):
    with pytest.raises(artifact.ProbeError):training.TrainConfig.from_dict({'fitData':sample_dataset(tmp_path),'label':'bad',key:value})


def test_same_group_selection_and_incompatible_evaluation_refuse(tmp_path):
    doc,result,cfg=fit(tmp_path)
    with pytest.raises(artifact.ProbeError,match='share'):
        training.train(training.TrainConfig.from_dict({**cfg.to_dict(),'selectionData':cfg.fitData}),root=tmp_path)
    ref=sample_dataset(tmp_path,'finalTest');path=tmp_path/ref['path'];d=json.loads(path.read_bytes());d['input']['site']['kind']='residualPre';path.write_bytes(archives.encoded(d));ref['sha256']=archives.file_hash(path)
    with pytest.raises(artifact.ProbeError,match='binding'):
        evaluation.evaluate(evaluation.EvaluateConfig.from_dict({'probe':{'path':str(Path(result['artifactPath']).relative_to(tmp_path)),'sha256':archives.file_hash(result['artifactPath'])},'evaluationData':ref}),root=tmp_path)
