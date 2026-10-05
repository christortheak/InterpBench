#!/usr/bin/env python3
"""Generate/gate the complete seed inventory and agent contract for both clients."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--write', action='store_true')
args = parser.parse_args()
resources = ROOT / 'Server/steerlab_server/client/resources'
manifest = json.loads((resources / 'workspace.json').read_text())
files = manifest['seedFiles']
assert len(files) == len(set(files))
seed = ROOT / 'WorkspaceSeed'
actual = {p.relative_to(seed).as_posix() for p in seed.rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(seed).parts)}
assert set(files) == actual, ('Workspace seed inventory differs', set(files) ^ actual)
packaged = ROOT / 'Server/steerlab_server/experiment/seed'
for name in files:
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    source = seed / name
    target = packaged / name
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    else:
        assert target.read_bytes() == source.read_bytes(), 'Regenerate packaged seed: ' + name
# --- the agent guide: one short core, plus topics rendered per client ---------
#
# Sources live in WorkspaceGuide/ (a shipped-source tree, like WorkspaceSeed/):
#   core.md          the AGENTS.md body. Client-neutral, so both clients write
#                    byte-identical files. `<!-- … -->` lines are source-only.
#   topics.json      the ordered topic index (name + one-line summary).
#   topics/<name>.md one topic. `<!-- client: all|mac|python -->` lines switch
#                    which client the following lines are rendered for, and
#                    {{cli}}, {{remote}} and {{endpoint}} are replaced per client.
guide = ROOT / 'WorkspaceGuide'
CLIENTS = {
    'mac': {'cli': 'steerlab-cli', 'remote': 'remote', 'endpoint': '--site <id>'},
    'python': {'cli': 'steerlab', 'remote': 'runner', 'endpoint': '--runner <url>'},
}
CORE_MAX_LINES, CORE_MAX_BYTES = 300, 18000
MARKER = re.compile(r'<!-- client: (all|mac|python) -->')


def render(source, client):
    """One client's text: its own blocks plus the shared ones, tokens replaced."""
    kept, scope = [], 'all'
    for line in source.splitlines():
        if line.strip().startswith('<!--'):
            marker = MARKER.fullmatch(line.strip())
            if marker:
                scope = marker.group(1)
            continue
        if scope in ('all', client):
            kept.append(line)
    text = '\n'.join(kept)
    for token, value in CLIENTS[client].items():
        text = text.replace('{{' + token + '}}', value)
    assert '{{' not in text, 'Unknown {{token}} in an agent guide source'
    return re.sub(r'\n{3,}', '\n\n', text).strip() + '\n'


def code_text(markdown):
    """Fenced lines and inline code spans: where the guide writes commands."""
    out, fenced = [], False
    for line in markdown.splitlines():
        if line.startswith('```'):
            fenced = not fenced
        elif fenced:
            out.append(line)
        else:
            out.extend(line.split('`')[1::2])
    return '\n'.join(out)


core_source = (guide / 'core.md').read_text()
assert 'client:' not in ''.join(l for l in core_source.splitlines() if l.strip().startswith('<!--')), 'The core guide is client-neutral: no client blocks'
assert '{{' not in core_source, 'The core guide is client-neutral: no per-client tokens'
body = '\n'.join(line for line in core_source.splitlines() if not line.strip().startswith('<!--')).strip() + '\n'
assert body.count('\n') <= CORE_MAX_LINES and len(body.encode()) <= CORE_MAX_BYTES, \
    f'The core guide is over budget ({body.count(chr(10))} lines, {len(body.encode())} bytes); move depth into a topic'
assert re.search(r'^Guide version: \d+$', body, re.M), 'The core guide needs its "Guide version: <n>" line'
index = json.loads((guide / 'topics.json').read_text())
assert index['schemaVersion'] == 1
names = [topic['name'] for topic in index['topics']]
assert len(names) == len(set(names)) and all(re.fullmatch(r'[a-z][a-z-]*', name) for name in names)
assert set(names) == {p.stem for p in (guide / 'topics').glob('*.md')}, 'WorkspaceGuide/topics.json and topics/*.md name different topics'
listed = re.findall(r'^- `([a-z-]+)` — (.+)$', body, re.M)
assert listed == [(topic['name'], topic['summary']) for topic in index['topics']], \
    'The core guide\'s topic list must match WorkspaceGuide/topics.json, in order'
rendered = {client: [dict(name=topic['name'], summary=topic['summary'],
                          text=render((guide / 'topics' / (topic['name'] + '.md')).read_text(), client))
                     for topic in index['topics']] for client in CLIENTS}
for client, other in (('mac', r'(?<![\w-])steerlab [a-z]'), ('python', r'steerlab-cli [a-z]')):
    for topic in rendered[client]:
        stray = re.search(other, code_text(topic['text']))
        assert not stray, f'{client} topic {topic["name"]} shows the other client\'s command near: {stray.group(0)!r}'
agent = resources / 'agent-guide.md'
topic_index = resources / 'agent-guide-topics.json'
index_text = json.dumps(dict(schemaVersion=1, topics=[dict(name=t['name'], summary=t['summary']) for t in rendered['python']]),
                        ensure_ascii=False, indent=2) + '\n'
packaged_topics = {resources / f'agent-guide-topic-{t["name"]}.md': t['text'] for t in rendered['python']}
existing_topics = set(resources.glob('agent-guide-topic-*.md'))
if args.write:
    agent.write_text(body)
    topic_index.write_text(index_text)
    for path in existing_topics - set(packaged_topics):
        path.unlink()
    for path, text in packaged_topics.items():
        path.write_text(text)
else:
    assert agent.read_text() == body, 'Regenerate packaged agent guide'
    assert topic_index.read_text() == index_text, 'Regenerate packaged agent guide topics'
    assert existing_topics == set(packaged_topics), 'Regenerate packaged agent guide topics'
    for path, text in packaged_topics.items():
        assert path.read_text() == text, 'Regenerate packaged agent guide topic: ' + path.name
values = dict(seedFiles=files, promptDirectories=[p.removeprefix('prompts/') for p in manifest['directories'] if p.startswith('prompts/')],
              gitignore=manifest['gitignore'], markerTemplate=(resources / 'workspace-marker.md').read_text(), agentBody=body)
swift = '// Generated by scripts/ci/check-workspace-bootstrap.py --write.\nimport Foundation\n\nenum WorkspaceBootstrapText {\n'
for name, value in values.items():
    encoded = json.dumps(value, ensure_ascii=False)
    assert '"""#' not in encoded
    if isinstance(value, list):
        swift += f'    static let {name}: [String] = {encoded}\n'
    else:
        # Decode JSON strings to avoid a second escaping convention.
        swift += f'    static let {name} = try! JSONDecoder().decode(String.self, from: Data(#"""\n        {encoded}\n        """#.utf8))\n'
# The Mac rendering of every topic, as one JSON array decoded at first use.
encoded = json.dumps(rendered['mac'], ensure_ascii=False, indent=2)
assert '"""#' not in encoded and '\\#' not in encoded
swift += '    static let guideTopicsJSON = #"""\n' + '\n'.join('        ' + line for line in encoded.splitlines()) + '\n        """#\n'
swift += '}\n'
target = ROOT / 'Sources/ExperimentKit/WorkspaceBootstrapText.swift'
if args.write:
    target.write_text(swift)
else:
    assert target.read_text() == swift, 'Regenerate compiled workspace resources'
print('Complete workspace seed, agent guide, guide topics and compiled resources match.')
