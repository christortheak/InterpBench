"""App-free workspace/bootstrap CLI family; engine setup remains optional."""
from ..cli_envelope import CLIResult, VerbSpec

VERB_SPECS = (
    VerbSpec('workspace', 'init', positional='<directory>', purpose='Create a complete portable workspace in a new or empty directory.', boolean_flags=frozenset({'--no-git'})),
    VerbSpec('workspace', 'inspect', purpose='Inspect an existing workspace without modifying it.'),
    VerbSpec('workspace', 'handoff', purpose='Return the first steps for a coding assistant: the study interview, the short method index, and how to work with the researcher.'),
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


def run(invocation):
    from . import workspace_bootstrap as owner
    from .diagnostic_commands import validate
    from ..experiment import paths
    verb = invocation.spec.verb
    validate(invocation, 1 if verb == 'init' else 0)
    following = None
    if verb == 'init':
        result = owner.initialize(invocation.positionals[0], use_git='--no-git' not in invocation.flags)
        following = init_next_action(result['workspaceRoot'])
    elif verb == 'inspect':
        result = owner.inspect(paths.project_root())
    else:
        result = owner.handoff(paths.project_root())
    return CLIResult(message='Workspace operation completed.', changed=result['changed'], payload=result, next_action=following)
