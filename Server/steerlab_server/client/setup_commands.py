"""Readiness, provisioning and repair use the same local owners as the app."""
from ..cli_envelope import CLIResult, VerbSpec

VERB_SPECS = (
    VerbSpec('setup', 'inspect', purpose='Inspect client and workspace readiness; execution is assessed separately.'),
    VerbSpec('setup', 'plan', purpose='Review lightweight client installation without downloading anything.', value_flags=frozenset({'--release', '--runtime'})),
    *(VerbSpec('setup', verb, purpose='Provision a new lightweight runtime from the explicitly approved current plan.', value_flags=frozenset({'--release', '--runtime', '--expect'}), boolean_flags=frozenset({'--yes'}), required_flags=frozenset({'--expect', '--yes'})) for verb in ('apply', 'repair')),
)


def run(invocation, *, root=None):
    from . import setup
    from .diagnostic_commands import validate
    from ..client_cli import ClientRefusal
    from ..experiment import paths
    validate(invocation, 0)
    try:
        if invocation.spec.verb == 'inspect':
            result = setup.inspect(root)
        else:
            result = setup.provision(invocation.spec.verb, release=invocation.one('--release'), runtime=invocation.one('--runtime'), expected=invocation.one('--expect'), approved='--yes' in invocation.flags)
        return CLIResult(message='Client setup operation completed.', changed=result['changed'], payload=result)
    except (setup.SetupRefusal, OSError, ValueError) as exc:
        raise ClientRefusal(code='clientSetupRefused', reason=str(exc), repair_action=getattr(exc, 'repair_action', setup.SetupRefusal.repair_action)) from exc
