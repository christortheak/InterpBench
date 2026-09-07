import json
import pytest
import httpx
from steerlab_server.client.runner import RunnerClient
from steerlab_server.experiment import science_actions, science_catalog


def test_catalog_http_actions_are_owned_and_engine_only_paths_refuse():
    from steerlab_server.api.route_roles import CENSUS
    routes = {r.key: r for r in CENSUS}
    for operation in science_catalog.catalog()['operations']:
        for action in operation['actions']:
            assert routes[action['method']+' '+action['path']].role.value == action['serviceRole']
    with pytest.raises(science_catalog.ScienceRefusal, match='no supported HTTP'):
        science_actions.request('optvec-jspace', 'execute', {'path': {}, 'query': {}, 'body': {}})


def test_portable_action_calls_original_route_once_and_preserves_integers():
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={'seed': 2**64-1})
    client = RunnerClient(base_url='https://runner.example.invalid/proxy', http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = client.science_call('reader-fit', 'post-reader-fit', {'path': {}, 'query': {}, 'body': {'seed': 2**64-1}})
    assert result['seed'] == 2**64-1 and len(calls) == 1
    assert calls[0].url.path == '/proxy/api/reader/fit'
    assert json.loads(calls[0].content)['seed'] == 2**64-1
    for value in ('../other', 'a/b', ''):
        with pytest.raises(science_catalog.ScienceRefusal):
            client.science_call('sweep-judgment', 'post-experiment-name-sweep-complete-judgment', {'path': {'name': value}, 'query': {}, 'body': {}})
    assert len(calls) == 1


def test_cli_action_dispatch_and_role_refusal_are_visible(monkeypatch, tmp_path, capsys):
    from steerlab_server import client_cli
    from steerlab_server.client import runner
    calls=[]
    def respond(request):
        calls.append(request)
        return httpx.Response(409, json={'detail': {'code': 'workbenchOperation', 'reason': 'Workbench required', 'repairAction': 'Connect to a workbench.'}})
    original=runner.RunnerClient
    monkeypatch.setattr(runner,'RunnerClient',lambda **kw: original(**kw,http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    request=tmp_path/'request.json'; request.write_text('{"path":{},"query":{},"body":{}}')
    assert client_cli.main(['runner','science-call','reader-fit','--action','post-reader-fit','--request',str(request),'--runner','https://runner.example.invalid','--json']) != 0
    result=json.loads(capsys.readouterr().out)
    assert result['error']['repairAction'] and len(calls)==1


def test_route_catalog_gate_rejects_invented_actions_and_wrong_authority():
    import copy
    import importlib.util
    from pathlib import Path
    from steerlab_server.api.route_roles import CENSUS
    spec = importlib.util.spec_from_file_location('science_cli_census', Path(__file__).resolve().parents[2] / 'scripts/ci/science_cli_census.py')
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    catalog = science_catalog.catalog(); gate.check_actions(catalog, CENSUS)
    for key, value in [('path', '/api/invented'), ('serviceRole', 'invented')]:
        changed = copy.deepcopy(catalog)
        operation = next(op for op in changed['operations'] if op['actions'])
        operation['actions'][0][key] = value
        with pytest.raises((AssertionError, KeyError)): gate.check_actions(changed, CENSUS)
