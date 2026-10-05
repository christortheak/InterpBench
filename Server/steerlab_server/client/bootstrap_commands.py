"""App-free workspace/bootstrap CLI family; engine setup remains optional."""
from ..cli_envelope import CLIResult, VerbSpec

VERB_SPECS = (
    VerbSpec('workspace', 'init', positional='<directory>',
             purpose='Create a complete portable workspace in a new or empty directory, or with --demo open a verified copy of a Demo Workspace there.',
             boolean_flags=frozenset({'--no-git'}), value_flags=frozenset({'--demo'})),
    VerbSpec('workspace', 'inspect', purpose='Inspect an existing workspace without modifying it.'),
    VerbSpec('workspace', 'handoff', purpose='Return the first steps for a coding assistant: the study interview, the short method index, and how to work with the researcher.'),
    VerbSpec('workspace', 'guide', positional='[<topic>]', purpose="List the agent guide topics, or print one topic with this client's commands."),
)


def init_next_action(root):
    """After a new workspace: the interview first. Swift twin: `workspace init`
    in ExperimentCLIRunner, which names --workspace where this names --root."""
    from ..cli_envelope import next_action
    from .workspace_bootstrap import INTENT_PLACEHOLDER, STUDY_INTENTS
    intents = [intent['id'] for intent in STUDY_INTENTS]
    return next_action(
        f'authoring study {INTENT_PLACEHOLDER}',
        detail=(f"Start with the study interview; the intents are {', '.join(intents[:-1])}, and {intents[-1]}. "
                f'Name the workspace with --root {root}, or export STEERLAB_WORKSPACE={root}. '
                'workspace handoff returns these first steps for a coding assistant.'))


def demo_next_action(root, *, workspace_flag='--root'):
    """After opening a Demo Workspace: its README first, then its studies.
    Swift twin: `DemoWorkspace.nextAction`, which names --workspace."""
    from ..cli_envelope import next_action
    return next_action(
        'experiment list',
        detail=(f'Read {root}/README.md first. It says what this demo study asks, and gives the steps from here '
                f'to an exported result. Then list the studies: name the workspace with {workspace_flag} {root}, '
                f'or export STEERLAB_WORKSPACE={root}.'))


def open_demo(directory, backend, *, use_git=True):
    """`workspace init <directory> --demo <backend>`: a verified copy of a
    carried Demo Workspace, never the carried original. Every refusal names
    what is wrong in plain words and a command this client can run."""
    from ..client_cli import ClientRefusal
    from . import demo_workspaces as demos
    program = 'steerlab workspace init <directory>'
    if backend not in demos.BACKENDS:
        choices = ', '.join(f'{name} ({demos.BACKEND_TITLES[name]})' for name in demos.BACKENDS[:-1])
        last = demos.BACKENDS[-1]
        raise ClientRefusal(
            code='usage', reason=f"There is no Demo Workspace named '{backend}'.",
            repair_action=f'Choose one of: {choices}, or {last} ({demos.BACKEND_TITLES[last]}). '
                          f'For example: {program} --demo {demos.BACKENDS[0]}')
    try:
        result = demos.open_copy(backend, directory, use_git=use_git)
    except demos.DemoRefusal as refusal:
        repair = refusal.repair
        if refusal.code == 'demoNotCarried':
            others = refusal.payload.get('carried') or []
            repair = ((f'{program} --demo {others[0]}  (a demo this copy carries), or ' if others else '')
                      + f'{program}  (an ordinary new workspace)')
        elif refusal.code == 'destinationNotEmpty':
            repair = f'steerlab workspace init <a-new-or-empty-directory> --demo {backend}'
        raise ClientRefusal(code=refusal.code, reason=refusal.reason, repair_action=repair,
                            state='refused', payload=refusal.payload) from None
    root = result['workspaceRoot']
    print(f'Opened a copy of the {backend} Demo Workspace at {root}')
    print(f'Read first: {result["demoReadme"]}')
    return result


def run(invocation):
    from . import workspace_bootstrap as owner
    from .diagnostic_commands import validate
    from ..experiment import paths
    verb = invocation.spec.verb
    if verb == 'guide':
        return guide(invocation, owner)
    validate(invocation, 1 if verb == 'init' else 0)
    following = None
    if verb == 'init' and invocation.has('--demo'):
        backend = invocation.one('--demo')
        result = open_demo(invocation.positionals[0], backend, use_git='--no-git' not in invocation.flags)
        return CLIResult(message=f"Opened a copy of the {backend} Demo Workspace at {result['workspaceRoot']}.",
                         changed=True, payload=result, next_action=demo_next_action(result['workspaceRoot']))
    if verb == 'init':
        result = owner.initialize(invocation.positionals[0], use_git='--no-git' not in invocation.flags)
        following = init_next_action(result['workspaceRoot'])
    elif verb == 'inspect':
        result = owner.inspect(paths.project_root())
    else:
        # The handoff object is the same on both clients; what THIS client
        # does is said in the message, which is the app-free client's own.
        from .setup import CLIENT_SCOPE_SHORT
        result = owner.handoff(paths.project_root())
        return CLIResult(message='Handoff ready. ' + CLIENT_SCOPE_SHORT, changed=result['changed'], payload=result)
    return CLIResult(message='Workspace operation completed.', changed=result['changed'], payload=result, next_action=following)


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
