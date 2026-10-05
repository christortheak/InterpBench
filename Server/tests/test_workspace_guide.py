"""The agent guide after its split: a short core written into every workspace,
and topics served on demand by ``workspace guide [<topic>]`` with THIS client's
commands.

What these hold together:

* the core stays within its size budget and names no client-specific surface;
* every topic resolves, both clients offer the same topic names, and the core's
  own topic list is exactly that set;
* no topic names a verb this client does not have, none shows the Mac command
  line's commands, and every verb this client has is named somewhere;
* an unedited, older guide is refreshed, and nothing else ever is: not an
  edited file, not a newer guide, not a missing one.
"""
import hashlib
import json
from pathlib import Path
import re

import pytest

from steerlab_server import client_cli
from steerlab_server.client import workspace_bootstrap as owner

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / 'Server/steerlab_server/client/resources'
CORE = (RESOURCES / 'agent-guide.md').read_text(encoding='utf-8')
CORE_MAX_LINES, CORE_MAX_BYTES = 300, 18000


@pytest.fixture
def no_workspace(monkeypatch):
    monkeypatch.delenv('STEERLAB_ROOT', raising=False)
    monkeypatch.delenv(client_cli.WORKSPACE_ENV, raising=False)


def _document(capsys):
    captured = capsys.readouterr()
    assert captured.out.endswith('}\n') and captured.out.count('\n}') == 1, captured.out
    return json.loads(captured.out), captured.err


# --- the core ------------------------------------------------------------------


def test_the_core_guide_stays_within_its_budget():
    lines = CORE.count('\n')
    assert lines <= CORE_MAX_LINES, f'the core guide is {lines} lines; move depth into a topic'
    assert len(CORE.encode()) <= CORE_MAX_BYTES, 'the core guide is over its byte budget; move depth into a topic'
    # …and it is not a stub: the sections it exists to carry are there.
    for heading in ("## Collaborate at the researcher's level", '## Ask before you spend',
                    '## Choose the installed client', '## Discover what the client can do',
                    '## The study lifecycle', '## The machine contract, in brief',
                    '## Immutability', '## What not to do', '## Topics on demand'):
        assert heading in CORE, heading
    assert owner.guide_version(CORE) >= 2
    assert owner.agent_contents().endswith(CORE)


def test_the_core_is_the_same_bytes_the_mac_compiles():
    swift = (ROOT / 'Sources/ExperimentKit/WorkspaceBootstrapText.swift').read_text(encoding='utf-8')
    encoded = swift.split('static let agentBody = try! JSONDecoder().decode(String.self, from: Data(#"""\n', 1)[1].split('\n', 1)[0]
    assert json.loads(encoded.strip()) == CORE


# --- topics --------------------------------------------------------------------


def _mac_topics():
    swift = (ROOT / 'Sources/ExperimentKit/WorkspaceBootstrapText.swift').read_text(encoding='utf-8')
    block = swift.split('static let guideTopicsJSON = #"""\n', 1)[1].rsplit('\n        """#', 1)[0]
    return json.loads('\n'.join(line[8:] for line in block.splitlines()))


def test_every_topic_resolves_and_is_listed_in_the_core():
    topics = owner.guide_topics()
    names = [topic['name'] for topic in topics]
    assert len(names) >= 10 and len(names) == len(set(names))
    for topic in topics:
        text = owner.guide_topic(topic['name'])
        assert text.startswith('# ') and text.endswith('\n') and not text.endswith('\n\n')
        assert '{{' not in text and '<!--' not in text, topic['name']
        assert f"- `{topic['name']}` — {topic['summary']}\n" in CORE, topic['name']
    assert re.findall(r'^- `([a-z-]+)` — ', CORE, re.M) == names
    with pytest.raises(KeyError):
        owner.guide_topic('teleport')
    # No packaged topic file is orphaned from the index.
    assert {path.name for path in RESOURCES.glob('agent-guide-topic-*.md')} == {f'agent-guide-topic-{name}.md' for name in names}


def test_both_clients_offer_the_same_topic_names():
    mac = _mac_topics()
    python = owner.guide_topics()
    assert [topic['name'] for topic in mac] == [topic['name'] for topic in python]
    assert [topic['summary'] for topic in mac] == [topic['summary'] for topic in python]
    source = json.loads((ROOT / 'WorkspaceGuide/topics.json').read_text(encoding='utf-8'))
    assert [topic['name'] for topic in source['topics']] == [topic['name'] for topic in python]
    # The texts differ on purpose: each client shows its own executable.
    assert any(topic['text'] != owner.guide_topic(topic['name']) for topic in mac)


# --- the verb ------------------------------------------------------------------


def test_workspace_guide_lists_topics_without_a_workspace(no_workspace, capsys):
    assert client_cli.main(['workspace', 'guide', '--json']) == 0
    document, _ = _document(capsys)
    assert document['verb'] == 'workspace guide' and document['state'] == 'ready'
    assert document['changed'] is False and 'workspace' not in document
    assert document['result'] == {'client': 'steerlab', 'topics': owner.guide_topics()}
    assert client_cli.main(['workspace', 'guide']) == 0
    listing = capsys.readouterr().out
    for topic in owner.guide_topics():
        assert f"{topic['name']} — {topic['summary']}\n" in listing


def test_workspace_guide_prints_one_topic_in_both_modes(no_workspace, capsys, tmp_path):
    for topic in owner.guide_topics():
        assert client_cli.main(['workspace', 'guide', topic['name'], '--json']) == 0
        document, _ = _document(capsys)
        assert document['result']['topic'] == topic['name']
        assert document['result']['text'] == owner.guide_topic(topic['name'])
        assert document['result']['topics'] == owner.guide_topics()
        assert document['changed'] is False
    assert client_cli.main(['workspace', 'guide', 'freeze']) == 0
    assert capsys.readouterr().out == owner.guide_topic('freeze')
    # A named workspace is honoured and reported, and reading guidance writes nothing.
    root = tmp_path / 'not-a-workspace'
    root.mkdir()
    assert client_cli.main(['workspace', 'guide', 'freeze', '--root', str(root), '--json']) == 0
    document, _ = _document(capsys)
    assert document['workspace'] == str(root)
    assert list(root.iterdir()) == []


def test_an_unknown_topic_names_the_ones_that_exist(no_workspace, capsys):
    assert client_cli.main(['workspace', 'guide', 'teleport', '--json']) == 64
    document, stderr = _document(capsys)
    assert document['state'] == 'blocked' and document['error']['code'] == 'usage'
    assert 'teleport' in document['error']['reason']
    for topic in owner.guide_topics():
        assert topic['name'] in document['error']['repairAction']
    assert 'steerlab workspace guide' in document['error']['repairAction']
    assert 'teleport' in stderr
    assert client_cli.main(['workspace', 'guide', 'freeze', 'sweep', '--json']) == 64
    assert _document(capsys)[0]['state'] == 'blocked'


def test_the_verb_is_declared_like_every_other(capsys):
    spec = next(spec for spec in client_cli.CLIENT_VERB_SPECS if spec.label == 'workspace guide')
    assert client_cli.synopsis(spec) == 'steerlab workspace guide [<topic>]'
    assert client_cli.main(['workspace', 'guide', '--help']) == 0
    assert 'workspace guide [<topic>]' in capsys.readouterr().out
    assert 'steerlab workspace guide [<topic>]' in (ROOT / 'docs/CLI-REFERENCE.md').read_text(encoding='utf-8')


# --- no topic names a verb this client does not have ---------------------------

#: Every verb family either client has. A code span that opens with one of
#: these words followed by a verb is a command, and it must be this client's.
FAMILIES = {'workspace', 'setup', 'science', 'experiment', 'concept', 'bundle', 'pack', 'design', 'agent', 'model',
            'authoring', 'runner', 'run', 'panel', 'data', 'vectors', 'remote', 'cluster', 'docs', 'install', 'init',
            'results'}
WORD = re.compile(r'[a-z][a-z-]*')
#: A bare hyphenated word shaped like one of the two clients' verbs
#: (`set-sampling`, `pin-rubric`, `submit-bundle`): it has to be a verb THIS
#: client declares.
VERB_SHAPED = re.compile(r'(set|pin|declare|remove|attach|detach|import|inspect|submit|model|science|cleanup|extract|rescore|mirror|backfill)-[a-z-]+')


def code_spans(markdown):
    """Fenced lines, plus inline code spans read per paragraph so a span that
    wraps across two lines is still one span."""
    spans, paragraph, fenced = [], [], False

    def flush():
        spans.extend(' '.join(part.split()) for part in '\n'.join(paragraph).split('`')[1::2])
        paragraph.clear()

    for line in markdown.splitlines():
        if line.startswith('```'):
            flush()
            fenced = not fenced
        elif fenced:
            spans.append(line.strip())
        elif not line.strip():
            flush()
        else:
            paragraph.append(line)
    flush()
    return spans


def unknown_commands(text, table, *, executable='steerlab', other='steerlab-cli'):
    names = {verb for verbs in table.values() for verb in verbs}
    found = []
    for span in code_spans(text):
        tokens = span.split()
        if not tokens or tokens[0].startswith('#') or tokens[0] == 'steerlab-server':
            continue
        if tokens[0] == other and len(tokens) > 1 and WORD.fullmatch(tokens[1]):
            found.append(f"the other client's command: {span}")
            continue
        explicit = tokens[0] == executable
        if explicit:
            tokens = tokens[1:]
        if not tokens:
            continue
        head = tokens[0]
        if head in FAMILIES:
            verb = tokens[1].split('/')[0] if len(tokens) > 1 else ''
            if not WORD.fullmatch(verb):
                if explicit and head not in table:
                    found.append(f'no such family: {span}')
            elif verb not in table.get(head, ()):
                found.append(f'no such verb: {span}')
        elif explicit and WORD.fullmatch(head):
            found.append(f'no such family: {span}')
        elif VERB_SHAPED.fullmatch(head) and head not in names:
            found.append(f'a verb this client does not have: {span}')
    return found


def _client_table():
    table = {}
    for spec in client_cli.CLIENT_VERB_SPECS:
        table.setdefault(spec.family, set()).add(spec.verb)
    return table


def test_no_topic_names_a_verb_this_client_lacks():
    table = _client_table()
    for topic in owner.guide_topics():
        unknown = unknown_commands(owner.guide_topic(topic['name']), table)
        assert not unknown, (topic['name'], unknown)
    # The core is client-neutral: every command it spells exists here too.
    assert not unknown_commands(CORE, table, other='\0')
    # The gate has teeth, in each direction it guards…
    planted = ('Use `steerlab experiment teleport <name>` and then `cluster push`.\n'
               'Also `set-sampling`, and the Mac\'s `steerlab-cli experiment run <name>`.\n'
               'A wrapped span still counts: `steerlab experiment\nteleport`.\n')
    assert len(unknown_commands(planted, table)) == 5
    # …and it does not cry wolf at real commands or at ordinary spans.
    fine = ('`steerlab experiment attach <name>`, `runner science-plan/science-submit`, `run`,\n'
            '`steerlab run <experiment> --runner <url>`, `experiment.json`, `runs/<run>/`, `validate`,\n'
            '`steerlab-server battery run <file>`, `set-protocol`, `--verb validate`, `steerlab-cli`.\n')
    assert not unknown_commands(fine, table)


def test_the_guide_names_every_verb_this_client_has():
    """The Python client's own reference, where the old single file was mostly
    the Mac's: a verb added here that no topic names is a lie by omission to an
    assistant that reads only the guide."""
    code = '\n'.join(code_spans(CORE + '\n' + '\n'.join(owner.guide_topic(topic['name']) for topic in owner.guide_topics())))
    missing = [spec.label for spec in client_cli.CLIENT_VERB_SPECS
               if not re.search(r'\b' + re.escape(spec.verb) + r'\b', code)]
    assert not missing, missing
    for expected in ('runner serve', 'bundle package', 'run <experiment> --runner', 'set-protocol'):
        assert expected in code, expected
    assert not re.search(r'\bteleport\b', code)


# --- refresh: an unedited, older guide is upgraded; nothing else is touched ----


def _machine_written(body):
    return owner.HEADER + hashlib.sha256(body.encode()).hexdigest() + ' -->\n\n' + body


OLDER_BODY = '# AGENTS.md\n\nYou are working inside a **SteerLab data workspace**.\n\n## 1. What this folder is\n'


def test_an_unedited_older_guide_is_refreshed_in_place(tmp_path):
    guide = tmp_path / 'AGENTS.md'
    guide.write_text(_machine_written(OLDER_BODY), encoding='utf-8')
    assert owner.guide_version(OLDER_BODY) == 1
    notice = owner.refresh_agent_guide(tmp_path)
    assert notice and str(guide) in notice and 'nobody had edited it' in notice
    assert guide.read_text(encoding='utf-8') == owner.agent_contents()
    assert sorted(path.name for path in tmp_path.iterdir()) == ['AGENTS.md']    # no staging debris
    assert owner.refresh_agent_guide(tmp_path) is None                         # idempotent
    # A copy that lost the blank line after the header still verifies.
    guide.write_text(_machine_written(OLDER_BODY).replace(' -->\n\n', ' -->\n', 1), encoding='utf-8')
    assert owner.refresh_agent_guide(tmp_path)
    assert guide.read_text(encoding='utf-8') == owner.agent_contents()


def test_refresh_never_touches_what_it_cannot_prove_or_would_downgrade(tmp_path):
    guide = tmp_path / 'AGENTS.md'
    # Missing: never created here.
    assert owner.refresh_agent_guide(tmp_path) is None and not guide.exists()
    newer_version = owner.guide_version(CORE) + 1
    newer = CORE.replace(f'Guide version: {owner.guide_version(CORE)}\n', f'Guide version: {newer_version}\n') + '\n## A section this client has never heard of\n'
    assert owner.guide_version(newer) == newer_version
    untouched = {
        'the researcher\'s own file': 'Researcher instructions\n',
        'edited under our header': owner.agent_contents() + '\nMy own note.\n',
        'an older body, edited': _machine_written(OLDER_BODY) + 'and my note\n',
        'a tampered header': _machine_written(OLDER_BODY).replace('sha256:', 'sha256:0', 1),
        'the pre-hash header': '<!-- Written by SteerLab workspace seeding; safe to regenerate — delete this file and reopen the workspace to get it back. SteerLab never overwrites an existing AGENTS.md, so local edits survive. -->\n\n' + OLDER_BODY,
        'a newer guide, unedited': _machine_written(newer),
        'already current': owner.agent_contents(),
        'no newline at all': 'x',
    }
    for label, contents in untouched.items():
        guide.write_text(contents, encoding='utf-8')
        assert owner.refresh_agent_guide(tmp_path) is None, label
        assert guide.read_text(encoding='utf-8') == contents, label


def test_any_verb_refreshes_an_older_guide_once_and_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('STEERLAB_ROOT', raising=False)
    monkeypatch.delenv(client_cli.WORKSPACE_ENV, raising=False)
    root = tmp_path / 'workspace'
    owner.initialize(root, use_git=False)
    guide = root / 'AGENTS.md'
    guide.write_text(_machine_written(OLDER_BODY), encoding='utf-8')
    assert client_cli.main(['experiment', 'list', '--root', str(root), '--json']) == 0
    document, stderr = _document(capsys)
    assert document['state'] == 'ready' and 'advisories' not in document
    notices = [line for line in stderr.splitlines() if line.startswith('notice: refreshed ')]
    assert len(notices) == 1 and 'AGENTS.md' in notices[0]
    assert guide.read_text(encoding='utf-8') == owner.agent_contents()
    # The work is done: the next invocation has nothing to say.
    assert client_cli.main(['experiment', 'list', '--root', str(root), '--json']) == 0
    assert 'notice:' not in _document(capsys)[1]
    # A guide the researcher owns is silent on this path too, and survives.
    guide.write_text('# AGENTS.md\n\nmy own notes\n', encoding='utf-8')
    assert client_cli.main(['experiment', 'list', '--root', str(root), '--json']) == 0
    assert 'notice:' not in _document(capsys)[1]
    assert guide.read_text(encoding='utf-8') == '# AGENTS.md\n\nmy own notes\n'
