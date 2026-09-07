"""App-free workspace/bootstrap CLI family; engine setup remains optional."""
from ..cli_envelope import CLIResult, VerbSpec

VERB_SPECS = (
    VerbSpec('workspace', 'init', positional='<directory>', purpose='Create a complete portable workspace in a new or empty directory.', boolean_flags=frozenset({'--no-git'})),
    VerbSpec('workspace', 'inspect', purpose='Inspect an existing workspace without modifying it.'),
    VerbSpec('workspace', 'handoff', purpose='Return the installed-client instructions for a research agent.'),
)


def run(invocation):
    from . import workspace_bootstrap as owner
    from .diagnostic_commands import validate
    from ..experiment import paths
    verb = invocation.spec.verb
    validate(invocation, 1 if verb == 'init' else 0)
    if verb == 'init':
        result = owner.initialize(invocation.positionals[0], use_git='--no-git' not in invocation.flags)
    elif verb == 'inspect':
        result = owner.inspect(paths.project_root())
    else:
        result = owner.handoff(paths.project_root())
    return CLIResult(message='Workspace operation completed.', changed=result['changed'], payload=result)
