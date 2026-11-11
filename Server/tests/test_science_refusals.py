"""An engine's refusal on a science route, as the Python client reports it.

The engine's science routes (``/api/science/…``: the stage, plan, and submit
calls, the diagnostic transport calls, and the catalogued actions those
routes own) refuse with ``{"detail": {"code", "reason", "repairAction"}}``
and no ``message``. A refusal there that names a code is the engine's own
typed refusal: the client reports ``refused`` (65), with the engine's code as
the envelope's ``error.code``, the engine's sentence (its ``message``, else
its ``reason``), and the engine's repair. Every other answer (a refusal that
names no code, a 401, a failure) reads as it did.

The engine here is a stand-in that answers what a test tells it to, so every
expectation is about what the CLIENT says. The answers and what both clients
make of them are held to one file,
``prompts/fixtures/engine-refusals/science-routes.json``; the Mac's half is
``Tests/ExperimentKitTests/EngineRefusalTests.swift``.
"""
import inspect
import json
from pathlib import Path

import httpx
import pytest

from steerlab_server import cli_envelope, client_cli
from steerlab_server.client import lens_transfer
from steerlab_server.client import runner as runner_api

FIXTURE = json.loads((Path(__file__).resolve().parents[2]
                      / 'prompts/fixtures/engine-refusals/science-routes.json').read_text(encoding='utf-8'))
SERVER = FIXTURE['server']
JOB = FIXTURE['jobID']
STAGED, PLAN = 'a' * 64, 'b' * 64
PATHS = {call['name']: call['path'] for call in FIXTURE['calls']}
CATALOGUED = {call['name']: call for call in FIXTURE['calls'] if 'operation' in call}


def catalogued(call):
    """A request document for one catalogued action: its path parameters,
    no query, and an empty body."""
    return {'path': {'job_id': JOB, 'action': 'status'}, 'query': {}, 'body': {}}


#: Every call this client makes to a science route, by the fixture's name.
CALLS = {
    'stage': lambda client: client.stage_diagnostic('/staged/request.tar.gz', STAGED),
    'plan': lambda client: client.scientific_plan({'inputBundleSHA256': STAGED}),
    'submit': lambda client: client.scientific_submit({'inputBundleSHA256': STAGED}, PLAN),
    'export': lambda client: client.export_diagnostic(JOB),
    'cleanup-plan': lambda client: client.diagnostic_cleanup_plan(JOB, {'receiptSHA256': STAGED}),
    'cleanup-apply': lambda client: client.diagnostic_cleanup_apply(JOB, {'receiptSHA256': STAGED}, PLAN),
    'fitting-round': lambda client: client.science_call(
        CATALOGUED['fitting-round']['operation'], CATALOGUED['fitting-round']['action'], catalogued('fitting-round')),
    'campaign': lambda client: client.science_call(
        CATALOGUED['campaign']['operation'], CATALOGUED['campaign']['action'], catalogued('campaign')),
}


class Engine:
    """Answers every request with one status and one body, and keeps the
    paths it was asked for."""

    def __init__(self, status=200, body='{"jobId": "job-1"}'):
        self.status, self.body, self.paths = status, body, []

    def __call__(self, request):
        if request.url.path == '/api/capabilities':     # what `science-fetch` reads before it exports
            return httpx.Response(200, json={})
        self.paths.append(request.url.path)
        return httpx.Response(self.status, content=self.body.encode('utf-8'),
                              headers={'Content-Type': 'application/json'})

    def client(self):
        return runner_api.RunnerClient(base_url=SERVER, http_client=httpx.Client(
            transport=httpx.MockTransport(self), base_url=SERVER))


def usual(status, body, path):
    """What this adapter's general handling makes of the same answer."""
    response = httpx.Response(status, content=body.encode('utf-8'), headers={'Content-Type': 'application/json'},
                              request=httpx.Request('POST', SERVER + path))
    with pytest.raises(runner_api.RunnerHTTPError) as expected:
        Engine().client()._raise_for_status(response, url=SERVER + path)
    return expected.value


def test_the_fixture_names_every_science_call_this_client_makes():
    assert sorted(PATHS) == sorted(CALLS)
    assert all(path.startswith('/api/science/') for path in PATHS.values())
    assert any(case['engineRefusal'] for case in FIXTURE['bodies'])
    assert any(not case['engineRefusal'] for case in FIXTURE['bodies'])


def test_every_call_this_client_makes_to_a_science_route_takes_the_science_handling():
    """Read from the source, so that a call added later cannot go without
    it: the methods of the adapter that name a science route are the ones
    this file calls, each request they send carries the science handling,
    and no other module of the client names a science route."""
    methods = {name: inspect.getsource(member)
               for name, member in inspect.getmembers(runner_api.RunnerClient, inspect.isfunction)}
    reaching = {name for name, source in methods.items() if '/api/science/' in source} - {'_raise_for_science'}
    assert reaching == {'stage_diagnostic', 'export_diagnostic', 'diagnostic_cleanup_plan', 'diagnostic_cleanup_apply',
                        'scientific_plan', 'scientific_submit', 'science_call'}
    for name in sorted(reaching):
        sent = methods[name].count('self._json(') + methods[name].count('self._request(')
        assert 1 <= sent <= methods[name].count('self._raise_for_science'), name
    package = Path(runner_api.__file__).resolve().parent
    elsewhere = sorted(str(path.relative_to(package.parent)) for path in package.parent.rglob('*.py')
                       if path != Path(runner_api.__file__).resolve() and 'api' not in path.relative_to(package.parent).parts[:1]
                       and '/api/science/' in path.read_text(encoding='utf-8'))
    assert elsewhere == []          # outside the engine's own routes (`api/`), only the adapter names one


@pytest.mark.parametrize('call', sorted(CALLS))
def test_a_success_on_a_science_route_is_untouched(call):
    engine = Engine()
    assert CALLS[call](engine.client()) == {'jobId': 'job-1'}
    assert engine.paths == [PATHS[call]]


@pytest.mark.parametrize('call', sorted(CALLS))
def test_a_coded_refusal_on_a_science_route_is_the_engines_own_and_every_other_answer_reads_as_it_did(call):
    """Each answer in the fixture, on each science call. A refusal that names
    a code is raised with the engine's code, sentence, and repair. Every other
    answer is this adapter's general handling, field for field."""
    path = PATHS[call]
    for case in FIXTURE['bodies']:
        engine = Engine(case['status'], case['body'])
        with pytest.raises(runner_api.RunnerError) as caught:
            CALLS[call](engine.client())
        error = caught.value
        assert engine.paths == [path], case['note']
        if case['engineRefusal']:
            assert type(error) is runner_api.RunnerRefusal, case['note']
            assert (error.code, error.state, error.reason, error.repair_action) == (
                case['code'], 'refused', case['reason'], case['repairAction']), case['note']
            # Which answer of which engine it was, and nothing of its body.
            assert error.detail == {'status': case['status'], 'url': SERVER + path}, case['note']
        else:
            as_before = usual(case['status'], case['body'], path)
            assert type(error) is runner_api.RunnerHTTPError, case['note']
            assert (error.code, error.state, error.reason, error.repair_action, error.detail) == (
                as_before.code, as_before.state, as_before.reason, as_before.repair_action,
                as_before.detail), case['note']


def test_the_form_the_science_routes_write_is_no_longer_quoted_as_json_text():
    """What changed, said once in full. The science routes' refusal has a
    `reason` and no `message`, so the general handling quotes the whole
    detail as JSON text inside a `runnerRefused` reason. On a science call
    the same answer is now the engine's own refusal."""
    case = FIXTURE['bodies'][0]
    assert case['engineRefusal'] and '"message"' not in case['body']
    before = usual(case['status'], case['body'], PATHS['stage'])
    assert before.code == 'runnerRefused' and before.state == 'refused'
    assert before.reason == 'runner answered 409: ' + json.dumps(json.loads(case['body'])['detail'], sort_keys=True)
    assert before.repair_action == case['repairAction']       # the repair was always relayed

    with pytest.raises(runner_api.RunnerRefusal) as caught:
        CALLS['stage'](Engine(case['status'], case['body']).client())
    assert (caught.value.code, caught.value.reason, caught.value.repair_action) == (
        'diagnosticTransportRefused', case['reason'], case['repairAction'])
    assert 'runner answered' not in caught.value.reason and '{' not in caught.value.reason


def test_what_the_engine_leaves_out_is_one_sentence_and_one_repair_and_no_more_text_than_the_limit():
    fallback = FIXTURE['fallback']
    assert runner_api.engine_refusal_fallback_reason(fallback['status']) == fallback['reason']
    # The repair is the one a lens-library refusal gets: one sentence for every refusal that names none.
    assert lens_transfer.REFUSAL_FALLBACK_REPAIR == fallback['repairAction']
    limit = fallback['textLimit']
    assert limit == lens_transfer.REFUSAL_TEXT_LIMIT
    long = {'detail': {'code': 'scientificExecutionRefused', 'reason': 'r' * (limit + 50),
                       'repairAction': 'p' * (limit + 50)}}
    with pytest.raises(runner_api.RunnerRefusal) as caught:
        CALLS['plan'](Engine(409, json.dumps(long)).client())
    assert caught.value.reason == 'r' * limit and caught.value.repair_action == 'p' * limit


def test_the_token_never_appears_in_a_relayed_refusal():
    token = 'fixture-token-not-a-secret'
    body = json.dumps({'detail': {'code': 'scientificExecutionRefused', 'reason': f'The header was Bearer {token}.',
                                  'repairAction': f'Send {token} again.'}})
    engine = Engine(409, body)
    client = runner_api.RunnerClient(base_url=SERVER, token=token, http_client=httpx.Client(
        transport=httpx.MockTransport(engine), base_url=SERVER))
    with pytest.raises(runner_api.RunnerRefusal) as caught:
        CALLS['submit'](client)
    for text in (caught.value.code, caught.value.reason, caught.value.repair_action, json.dumps(caught.value.detail)):
        assert token not in text
    assert '<token redacted>' in caught.value.reason and '<token redacted>' in caught.value.repair_action


def test_a_catalogued_action_on_a_route_outside_the_science_routes_reads_as_it_did():
    """`science-call` forwards a catalogued action to the route that owns it.
    Only a science route's refusal is relayed: the same body from another
    route keeps the general handling."""
    other = FIXTURE['anotherRoute']
    assert not other['path'].startswith('/api/science/')
    engine = Engine(other['status'], other['body'])
    with pytest.raises(runner_api.RunnerError) as caught:
        engine.client().science_call(other['operation'], other['action'], {'path': {}, 'query': {}, 'body': {}})
    assert engine.paths == [other['path']]
    as_before = usual(other['status'], other['body'], other['path'])
    assert type(caught.value) is runner_api.RunnerHTTPError and caught.value.code == 'runnerRefused'
    assert (caught.value.reason, caught.value.repair_action, caught.value.detail) == (
        as_before.reason, as_before.repair_action, as_before.detail)


# --- on the command line --------------------------------------------------------


@pytest.fixture
def wire(monkeypatch, tmp_path):
    """Point the command line's runner verbs at a stand-in engine."""
    real = runner_api.RunnerClient

    def connect(engine):
        def factory(**kwargs):
            kwargs['http_client'] = httpx.Client(transport=httpx.MockTransport(engine), base_url=SERVER)
            return real(**kwargs)
        monkeypatch.setattr(runner_api, 'RunnerClient', factory)
        monkeypatch.delenv(client_cli.RUNNER_TOKEN_ENV, raising=False)
        monkeypatch.setenv('STEERLAB_ROOT', '')   # restored after the test: the client exports it
    return connect


def cli(capsys, *arguments):
    capsys.readouterr()
    code = client_cli.main([*arguments, '--json'])
    return code, json.loads(capsys.readouterr().out)


def command_lines(tmp_path):
    """One command line per science verb that needs nothing but the engine."""
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'inputBundleSHA256': STAGED}))
    action = tmp_path / 'action.json'
    action.write_text(json.dumps(catalogued('fitting-round')))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    target = ['--runner', SERVER, '--root', str(workspace)]
    return {
        'science-stage': ['runner', 'science-stage', '/staged/request.tar.gz', '--sha256', STAGED, *target],
        'science-plan': ['runner', 'science-plan', str(request), *target],
        'science-submit': ['runner', 'science-submit', str(request), '--plan-sha256', PLAN, *target],
        'science-export': ['runner', 'science-export', JOB, *target],
        # A fetch reads the engine's capabilities, then asks for the export: the refusal is the export's.
        'science-fetch': ['runner', 'science-fetch', JOB, *target],
        'science-call': ['runner', 'science-call', CATALOGUED['fitting-round']['operation'],
                         '--action', CATALOGUED['fitting-round']['action'], '--request', str(action), *target],
    }


#: `cleanup-plan` and `cleanup-apply` verify a custody receipt in the workspace before they ask the engine
#: anything, so they are held at the adapter (`CALLS`) and not driven from here.
VERBS = ('science-call', 'science-export', 'science-fetch', 'science-plan', 'science-stage', 'science-submit')


@pytest.mark.parametrize('verb', VERBS)
def test_the_command_line_reports_the_engines_refusal_as_refused_with_its_code_sentence_and_repair(
        verb, wire, capsys, tmp_path):
    lines = command_lines(tmp_path)
    assert sorted(lines) == sorted(VERBS)
    for case in FIXTURE['bodies']:
        if not case['engineRefusal']:
            continue
        engine = Engine(case['status'], case['body'])
        wire(engine)
        code, document = cli(capsys, *lines[verb])
        assert code == 65, (case['note'], document)
        assert document['state'] == 'refused' and document['verb'] == f'runner {verb}'
        assert document['error'] == {'code': case['code'], 'reason': case['reason'],
                                     'repairAction': case['repairAction']}, case['note']
        # Which engine, and which answer of it: never the token, never the body.
        assert document['result'] == {'runner': SERVER, 'tokenPresent': False, 'status': case['status'],
                                      'url': SERVER + engine.paths[0]}, case['note']
        assert len(engine.paths) == 1 and engine.paths[0].startswith('/api/science/')


@pytest.mark.parametrize('verb', VERBS)
def test_the_command_line_reports_every_other_answer_as_it_did(verb, wire, capsys, tmp_path):
    """A refusal that names no code is still `runnerRefused` (65); a 401 and
    a 403 with no code are still about the token; a failure is still a
    failure (70). Each keeps the state, and so the exit code, it had."""
    lines = command_lines(tmp_path)
    expected = {401: ('runnerUnauthorized', 'blocked'), 403: ('runnerUnauthorized', 'blocked'),
                404: ('notFound', 'notFound'), 409: ('runnerRefused', 'refused'),
                422: ('runnerRefused', 'refused'), 500: ('runnerFailed', 'failed'),
                503: ('runnerUnavailable', 'degraded')}
    assert (cli_envelope.STATE_EXIT_CODES['refused'], cli_envelope.STATE_EXIT_CODES['failed']) == (65, 70)
    for case in FIXTURE['bodies']:
        if case['engineRefusal']:
            continue
        engine = Engine(case['status'], case['body'])
        wire(engine)
        code, document = cli(capsys, *lines[verb])
        want_code, want_state = expected[case['status']]
        assert (document['error']['code'], document['state'], code) == (
            want_code, want_state, cli_envelope.STATE_EXIT_CODES[want_state]), (case['note'], document)
        as_before = usual(case['status'], case['body'], engine.paths[0])
        assert document['error']['reason'] == as_before.reason, case['note']
        assert document['error']['repairAction'] == as_before.repair_action, case['note']
        assert document['result']['runnerDetail'] == as_before.detail['runnerDetail'], case['note']
