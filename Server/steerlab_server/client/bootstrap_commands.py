"""App-free workspace/bootstrap CLI family; engine setup remains optional."""
from ..cli_envelope import CLIResult, VerbSpec

VERB_SPECS = (
    VerbSpec('workspace', 'init', positional='<directory>', purpose='Create a complete portable workspace in a new or empty directory.', boolean_flags=frozenset({'--no-git'})),
    VerbSpec('workspace', 'inspect', purpose='Inspect an existing workspace without modifying it.'),
    VerbSpec('workspace', 'handoff', purpose='Return the installed-client instructions for a research agent.'),
    VerbSpec('workspace', 'guide', positional='[<topic>]', purpose="List the agent guide topics, or print one topic with this client's commands."),
)


def run(invocation):
    from . import workspace_bootstrap as owner
    from .diagnostic_commands import validate
    from ..experiment import paths
    verb = invocation.spec.verb
    if verb == 'guide':
        return guide(invocation, owner)
    validate(invocation, 1 if verb == 'init' else 0)
    if verb == 'init':
        result = owner.initialize(invocation.positionals[0], use_git='--no-git' not in invocation.flags)
    elif verb == 'inspect':
        result = owner.inspect(paths.project_root())
    else:
        result = owner.handoff(paths.project_root())
    return CLIResult(message='Workspace operation completed.', changed=result['changed'], payload=result)


def guide(invocation, owner):
    """Read guide text shipped inside this client. Needs no workspace; writes nothing."""
    from ..client_cli import ClientRefusal
    arguments = invocation.positionals
    if len(arguments) > 1:
        raise ClientRefusal(code='usage', reason='Name one guide topic at a time.',
                            repair_action=f'{owner.GUIDE_CLIENT} workspace guide <topic> --json')
    topics = owner.guide_topics()
    if not arguments:
        for topic in topics:
            print(f"{topic['name']} — {topic['summary']}")
        print(f'Read one with: {owner.GUIDE_CLIENT} workspace guide <topic>')
        return CLIResult(message='Guide topics listed. Read one with workspace guide <topic>.',
                         payload={'client': owner.GUIDE_CLIENT, 'topics': topics})
    name = arguments[0]
    if name not in {topic['name'] for topic in topics}:
        raise ClientRefusal(code='usage', reason=f"There is no guide topic named '{name}'.",
                            repair_action='Choose one of: ' + ', '.join(topic['name'] for topic in topics)
                            + f'. `{owner.GUIDE_CLIENT} workspace guide` lists them with a line about each.')
    text = owner.guide_topic(name)
    print(text, end='')
    return CLIResult(message=f"Guide topic '{name}' read. Nothing was changed.",
                     payload={'client': owner.GUIDE_CLIENT, 'topic': name, 'text': text, 'topics': topics})
