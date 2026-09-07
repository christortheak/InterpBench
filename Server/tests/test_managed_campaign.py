import json
from pathlib import Path
import pytest
from steerlab_server.api import managed_campaign_engine as owner
from steerlab_server.experiment import diagnostic_archives as archives, optvec_campaign, optvec_train
from test_scientific_execution import setup


def materialized(root):
    from test_optvec_campaign import _campaign_payload
    config=_campaign_payload(root)
    config['baseConfig']['modelID']='example/model';config['baseConfig']['revision']='a'*40
    for ref in config['baseConfig']['datasets'].values():
        ref['path']=Path(ref['path']).relative_to(root).as_posix()
        ref['sha256']=archives.file_hash(root/ref['path'])
    return owner.materialize(config,str(root))


def test_materialization_is_not_execution_and_topups_use_existing_scheduler(setup):
    from test_optvec_campaign import FakeRunner
    root,_,_=setup; result=materialized(root)
    directory=result['campaignDirectory'];pin=result['managedCampaign']
    assert result['campaignExecution']=='materializedOnly'
    runner=FakeRunner(job_ids=['11','12','13','14'])
    review=owner.operate(directory,pin,'plan',runner=runner)
    assert review['status']['cellCount']==12
    outcome=owner.operate(directory,pin,'submit',review['planSHA256'],runner=runner)
    assert len(outcome['response']['submitted'])==4
    with pytest.raises(archives.Refusal,match='changed'):owner.operate(directory,pin,'submit',review['planSHA256'],runner=runner)
    scripts=list(Path(directory).glob('cells/*/'+optvec_campaign.CELL_SCRIPT_FILENAME))
    assert scripts and all('managed_campaign_engine' in p.read_text() for p in scripts)


def test_cell_checks_inputs_before_model_and_keeps_results_nested(setup,monkeypatch):
    root,_,_=setup; result=materialized(root);directory=Path(result['campaignDirectory'])
    packet=directory/'managed-campaign.json';data=json.loads(packet.read_text())
    cell=optvec_campaign.read_campaign(str(directory))['cells'][0]['cellID']
    monkeypatch.setattr(optvec_train,'train',lambda *a,**kw: {'runDirectory':'retained-output'})
    # Protect process-global fixtures; the real entry point is a child process.
    monkeypatch.setenv('STEERLAB_RUN_ROOT',str(root/'runs'));monkeypatch.chdir(root)
    owner.cell(str(packet),result['managedCampaign']['packetSHA256'],cell)
    assert (directory/'cells'/cell/optvec_campaign.COMPLETION_MARKER).exists()
    source=root/data['source']['files'][0]['path'];source.write_bytes(source.read_bytes()+b' ')
    monkeypatch.setattr(optvec_train,'train',lambda *a,**kw:pytest.fail('must refuse drift before training'))
    with pytest.raises(archives.Refusal,match='hash differs'):owner.cell(str(packet),result['managedCampaign']['packetSHA256'],cell)


def test_campaign_validation_reports_every_grid_model(setup):
    from test_optvec_campaign import _campaign_payload
    from steerlab_server.api import managed_validation
    root, _, _=setup;config=_campaign_payload(root)
    config['baseConfig']['revision']='a'*40
    config['grid']['conditions'][0]['overrides'].update(modelID='example/alternative',revision='b'*40)
    result=managed_validation.validate({'operation':'optvec-campaign','parameters':{'config':config}},str(root))
    assert {m['revision'] for m in result['models']}=={'a'*40,'b'*40}


def test_cancel_checks_names_for_unrecorded_allocations_before_any_mutation(setup,monkeypatch):
    from test_optvec_campaign import FakeRunner
    root,_,_=setup;result=materialized(root);directory=result['campaignDirectory'];pin=result['managedCampaign']
    runner=FakeRunner();review=owner.operate(directory,pin,'plan',runner=runner)
    calls=[]
    monkeypatch.setattr(owner.executors.SlurmExecutor,'cancel',lambda self,job:calls.append(job) or True)
    # The names are owner-generated; choose an actual cell without depending on
    # slug formatting in the test.
    cell=optvec_campaign.read_campaign(directory)['cells'][0]['cellID']
    monkeypatch.setattr(owner.executors.SlurmExecutor,'find_job_by_name',lambda self,name:'88' if name.endswith(cell) else None)
    result=owner.operate(directory,pin,'cancel',review['planSHA256'],runner=runner)
    assert calls==['88'] and result['cancellationRequested']=={'88':True}


def test_failed_submission_reports_its_persisted_state_change(setup):
    from test_optvec_campaign import FakeRunner
    root,_,_=setup; result=materialized(root);directory=result['campaignDirectory'];pin=result['managedCampaign']
    runner=FakeRunner(job_ids=[None, None, None], sbatch_exit=1)
    plan=owner.operate(directory,pin,'plan',runner=runner)
    result=owner.operate(directory,pin,'submit',plan['planSHA256'],runner=runner)
    assert result['response']['failed'] and result['changed'] is True


def test_export_checks_unrecorded_job_names_and_requires_terminal_scheduler(setup,monkeypatch):
    from steerlab_server.api import managed_campaign
    root,_,profile=setup;result=materialized(root);directory=Path(result['campaignDirectory'])
    cell=optvec_campaign.read_campaign(str(directory))['cells'][0]['cellID']
    monkeypatch.setattr(managed_campaign,'context',lambda *a:(directory,root,result['managedCampaign']))
    monkeypatch.setattr(managed_campaign,'action',lambda *a:{'status':{'cellCount':1,'totals':{'completed':1},'cells':[{'cellID':cell,'jobID':None}]}})
    names=[]
    monkeypatch.setattr(owner.executors.SlurmExecutor,'find_job_by_name',lambda self,name:names.append(name) or '88')
    monkeypatch.setattr(owner.executors.SlurmExecutor,'poll_state_detailed',lambda *a:('RUNNING',True))
    with pytest.raises(archives.Refusal,match='still active'):managed_campaign.require_complete('job',None,profile)
    assert names and names[0].endswith(cell)
    monkeypatch.setattr(owner.executors.SlurmExecutor,'poll_state_detailed',lambda *a:(None,False))
    with pytest.raises(archives.Refusal):managed_campaign.require_complete('job',None,profile)
    monkeypatch.setattr(owner.executors.SlurmExecutor,'poll_state_detailed',lambda *a:('COMPLETED',True))
    managed_campaign.require_complete('job',None,profile)
