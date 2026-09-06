"""Model preparation on an explicitly selected local or remote Python runner."""
from ..cli_envelope import CLIResult, VerbSpec


def specs(connection_flags):
    return (
        VerbSpec('model', 'plan', positional='<modelID>', purpose='Inspect the selected runner cache, install policy and reviewed target without loading weights.',
                 value_flags=connection_flags | {'--revision'}, required_flags=frozenset({'--runner'})),
        VerbSpec('model', 'install', positional='<modelID>', purpose='Start one reviewed durable model-install job; never retries submission.',
                 value_flags=connection_flags | {'--revision', '--plan-sha256'}, required_flags=frozenset({'--runner', '--plan-sha256'})),
        VerbSpec('model', 'status', positional='<job-id>', purpose='Observe the exact model-install job, including its logs and terminal outcome.',
                 value_flags=connection_flags, required_flags=frozenset({'--runner'})),
        VerbSpec('model', 'cancel', positional='<job-id>', purpose='Request cancellation of the exact model-install job; retain partial cache files.',
                 value_flags=connection_flags, required_flags=frozenset({'--runner'})),
    )


def validate(invocation):
    from ..client_cli import ClientRefusal
    spec, args, one = invocation.spec, invocation.positionals, invocation.one
    if len(args) != 1 or any(one(f) is None for f in spec.required_flags) or any(len(v) != 1 for v in invocation.flags.values()):
        raise ClientRefusal(code='usage', reason='Supply one model or job ID and each required flag once.', repair_action=f'steerlab {spec.label} --help')


def run(client, invocation, common):
    from ..client_cli import ClientRefusal
    validate(invocation)
    spec, args, one = invocation.spec, invocation.positionals, invocation.one
    if spec.verb == 'plan':
        result = client.model_plan(args[0], revision=one('--revision'))
    elif spec.verb == 'install':
        result = client.install_model(args[0], revision=one('--revision'), plan_sha256=one('--plan-sha256'))
    else:
        result = client.job(args[0])
        if result.get('id') != args[0] or result.get('kind') != 'model:install':
            raise ClientRefusal(code='modelJobMismatch', state='refused', payload={**common, 'response': result}, reason='The named job is not this model-install request.',
                                repair_action='Inspect runner jobs on the original endpoint and use the model-install job ID.')
        if spec.verb == 'cancel':
            result = client.cancel_job(args[0])
    payload = {**common, 'response': result}
    print(f'{spec.label}: inspect the returned target and job outcome')
    if spec.verb == 'status' and result.get('status') in ('failed', 'cancelled'):
        return CLIResult(message='Model installation did not complete.', payload=payload, state='failed', code='modelInstallationIncomplete',
                         repair_action='Inspect the job error and logs; correct cache access, credentials or site policy before an explicit new installation.')
    return CLIResult(message='Model preparation request completed.', changed=spec.verb in ('install', 'cancel'), payload=payload)
