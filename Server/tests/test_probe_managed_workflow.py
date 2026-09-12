import json
from pathlib import Path
from steerlab_server.experiment import diagnostic_archives as archives, diagnostic_inputs, method_authoring, managed_methods
from steerlab_server.api import scientific_execution, diagnostic_transport
from test_scientific_execution import setup
from test_probe_training import sample_dataset


def interview_fields(operation,root):
    if operation=='probe-capture':
        from test_probe_capture import capture_input
        ref=capture_input(root)
        return {'modelID':'example/model','revision':'a'*40,'examples':ref['path'],'layer':'0','device':'cpu'}
    ref=sample_dataset(root)
    if operation=='probe-train':return {'fitData':ref['path'],'label':'Example classifier','steps':'5'}
    from steerlab_server.experiment import probe_training as training
    result=training.train(training.TrainConfig.from_dict({'fitData':ref,'label':'Example','steps':5}),root=root,log=lambda _:None)
    final=sample_dataset(root,'finalTest')
    return {'probe':str(Path(result['artifactPath']).relative_to(root)),'evaluationData':final['path']}


def test_fit_evaluate_relocated_packet_and_immutable_input_closure(setup):
    root,_,profile=setup
    for operation in ('probe-train','probe-evaluate'):
        fields=interview_fields(operation,root)
        answers=dict(purpose='Predict labels',claim='Readout only',controls='Constant class baselines',selection='Fixed settings',fields=fields,advanced={})
        draft=method_authoring.draft(operation,answers,root)
        assert draft['operationReview']['counts']['rows']==4
        published=method_authoring.publish(operation,answers,root,'requests/'+operation,draft['planSHA256'])
        request=json.loads(Path(published['requestFile']).read_text())
        plan=diagnostic_inputs.plan(request,root)
        original={e['path']:(root/e['path']).read_bytes() for e in plan['files']}
        archive=root/'runs'/(operation+'.tar.gz')
        packed=diagnostic_inputs.package(request,root,archive,plan['planSHA256'])
        staged=diagnostic_transport.stage(packed['bundlePath'],packed['bundleSha256'],profile)
        execution=scientific_execution.plan(staged['request'],profile)
        assert execution['compute']=='cpu'
        packet=root/(operation+'-packet.json');packet.write_text(json.dumps(execution));record=root/(operation+'-result.json')
        assert scientific_execution.execute_packet(packet,'example-'+operation,record)==0
        result=json.loads(record.read_text())['result']
        assert Path(result['runDirectory']).is_relative_to(Path(staged['executionRoot']))
        assert Path(result['reportPath']).is_file()
        assert all((root/name).read_bytes()==value for name,value in original.items())
        from steerlab_server.api.jobs import JobManager, DurableJobStore
        import time
        jobs=JobManager(store=DurableJobStore(str(root/(operation+'-jobs.sqlite'))),sweep_orphans=False)
        job=jobs.record_external('science:'+operation,status='succeeded',executor='local',job_id='example-'+operation,result=result)
        job.finished_at=time.time();jobs.store.update(job)
        exported=diagnostic_transport.export(job.id,jobs,profile)
        local=root/('collected-'+operation);local.mkdir()
        imported=archives.import_evidence(exported['bundlePath'],exported['bundleSha256'],local)
        assert archives.verify(imported['receiptSHA256'],local)==imported['receipt']
        if operation=='probe-train':
            from steerlab_server.experiment import probe_library
            inventory=probe_library.inventory(local)
            assert inventory['count']==1 and inventory['probes'][0]['format']=='activation-probe-v1'
