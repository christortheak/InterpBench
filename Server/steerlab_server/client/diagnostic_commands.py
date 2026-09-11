"""Local custody owners and remote transport adapters for the portable client."""
import json
from pathlib import Path
from ..cli_envelope import CLIResult
from ..experiment import diagnostic_archives as archives, paths, science_catalog


def validate(invocation, count):
    if (len(invocation.positionals) != count or any(flag not in invocation.flags for flag in invocation.spec.required_flags)
        or any(len(values) != 1 for values in invocation.flags.values())):
        raise archives.Refusal('Supply exactly the declared positionals and required flags once, including explicit removal confirmation.')


def workspace_action(action, payload):
    if action == 'setup-start':
        if not isinstance(payload, dict) or set(payload) != {'workspaceRoot', 'create'} or not isinstance(payload['workspaceRoot'], str) or not payload['workspaceRoot'] or type(payload['create']) is not bool:
            raise archives.Refusal('First run requires workspaceRoot and an explicit create boolean.')
        from . import setup
        return setup.start(payload['workspaceRoot'], create=payload['create'])
    if action == 'setup-inspect':
        if not isinstance(payload, dict) or payload.keys() - {'workspaceRoot'} or ('workspaceRoot' in payload and (not isinstance(payload['workspaceRoot'], str) or not payload['workspaceRoot'])):
            raise archives.Refusal('Readiness accepts only an optional workspaceRoot string.')
        from . import setup
        return setup.inspect(payload.get('workspaceRoot'))
    required = {
        'staged-request': {'bundleSHA256'},
        'corpus-preview': {'specText'}, 'corpus-publish': {'previewID', 'planSHA256', 'destination'},
        'artifact-plan': {'descriptionFile'}, 'artifact-import': {'descriptionFile', 'planSHA256'},
        'sae-check': {'path'}, 'sae-show': {'path'},
        'sae-pin-plan': {'path', 'experiment'}, 'sae-pin': {'path', 'experiment', 'planSHA256'},
        'interview': {'operation'}, 'draft': {'operation', 'answersText'},
        'publish': {'operation', 'answersText', 'destination', 'planSHA256'},
        'input-plan': {'requestFile'}, 'package': {'requestFile', 'archivePath', 'planSHA256'},
        'import': {'archivePath', 'archiveSHA256'}, 'verify-custody': {'receiptSHA256'}, 'custody': set(),
    }
    if action not in required or not isinstance(payload, dict): raise archives.Refusal('Unknown diagnostic workspace operation.')
    fields = required[action] | {'workspaceRoot'}
    optional = {'expectedContext'} if action == 'import' else set()
    if not fields <= payload.keys() or payload.keys() - fields - optional or any(not isinstance(payload[k], str) or not payload[k] for k in fields):
        raise archives.Refusal('Supply exactly the declared action fields as nonempty strings.')
    root = str(Path(payload['workspaceRoot']).resolve())
    if action == 'staged-request':
        from ..experiment.diagnostic_inputs import save_stage_reference
        return save_stage_reference(payload['bundleSHA256'], root)
    if action in ('corpus-preview', 'corpus-publish'):
        from ..experiment import corpus_preparation
        if action == 'corpus-preview': return corpus_preparation.preview(json.loads(payload['specText']), root)
        return corpus_preparation.publish(payload['previewID'], payload['planSHA256'], payload['destination'], root)
    if action in ('artifact-plan', 'artifact-import'):
        from ..experiment import artifact_imports
        if action == 'artifact-plan': return artifact_imports.inspect_source(payload['descriptionFile'], root)
        return artifact_imports.publish(payload['descriptionFile'], root, payload['planSHA256'])
    if action.startswith('sae-'):
        from ..experiment import sae_authoring
        if action in ('sae-check', 'sae-show'): return sae_authoring.inspect('candidates' if action == 'sae-check' else 'qualification', payload['path'], root)
        if action == 'sae-pin-plan': return sae_authoring.pin_plan(payload['experiment'], payload['path'], root)
        return sae_authoring.pin(payload['experiment'], payload['path'], root, payload['planSHA256'])
    if action in ('interview', 'draft', 'publish'):
        from ..experiment import method_authoring
        if action == 'interview': return method_authoring.interview(payload['operation'])
        answers = json.loads(payload['answersText'])
        if action == 'draft': return method_authoring.draft(payload['operation'], answers, root)
        return method_authoring.publish(payload['operation'], answers, root, payload['destination'], payload['planSHA256'])
    if action in ('input-plan', 'package'):
        from ..experiment import diagnostic_inputs
        request = json.loads(Path(payload['requestFile']).read_bytes())
        if action == 'input-plan': return diagnostic_inputs.plan(request, root)
        return diagnostic_inputs.package(request, root, payload['archivePath'], payload['planSHA256'])
    if action == 'import': return archives.import_evidence(payload['archivePath'], payload['archiveSHA256'], root, expected_context=payload.get('expectedContext'))
    if action == 'verify-custody': return {'verified': True, 'receipt': archives.verify(payload['receiptSHA256'], root), 'receiptSHA256': payload['receiptSHA256']}
    if action == 'custody': return archives.inventory(root)
    raise archives.Refusal('Unknown diagnostic workspace operation.')


def local(invocation):
    from ..client_cli import ClientRefusal
    from ..experiment.artifact_sources import ImportRefusal
    from ..experiment.corpus_sources import CorpusError
    try:
        verb = invocation.spec.verb; validate(invocation, 0 if verb == 'custody' else 1)
        value = invocation.positionals[0] if invocation.positionals else None
        payload = {'workspaceRoot': paths.project_root()}
        if verb == 'corpus-preview': payload['specText'] = Path(value).read_text()
        if verb == 'corpus-publish': payload.update(previewID=value, destination=invocation.one('--destination'), planSHA256=invocation.one('--plan-sha256'))
        if verb in ('artifact-plan', 'artifact-import'): payload['descriptionFile'] = value
        if verb == 'artifact-import': payload['planSHA256'] = invocation.one('--plan-sha256')
        if verb.startswith('sae-'): payload['path'] = value
        if verb in ('sae-pin-plan', 'sae-pin'): payload['experiment'] = invocation.one('--experiment')
        if verb == 'sae-pin': payload['planSHA256'] = invocation.one('--plan-sha256')
        if verb in ('interview', 'draft', 'publish'): payload['operation'] = value
        if verb in ('draft', 'publish'): payload['answersText'] = Path(invocation.one('--answers')).read_text()
        if verb == 'publish': payload.update(destination=invocation.one('--destination'), planSHA256=invocation.one('--plan-sha256'))
        if verb in ('input-plan', 'package'): payload['requestFile'] = value
        if verb == 'package': payload.update(archivePath=invocation.one('--archive'), planSHA256=invocation.one('--plan-sha256'))
        if verb == 'import': payload.update(archivePath=value, archiveSHA256=invocation.one('--sha256'))
        if verb == 'verify-custody': payload['receiptSHA256'] = value
        result = workspace_action(verb, payload)
        print(json.dumps(result, indent=2))
        return CLIResult(message='Diagnostic workspace operation completed; custody is byte verification, not scientific qualification.',
                         changed=result.get('changed', verb in ('package', 'import')), payload=result)
    except CorpusError as exc:
        raise ClientRefusal(code=exc.code, reason=str(exc), repair_action=exc.repair_action, state='refused') from exc
    except ImportRefusal as exc:
        raise ClientRefusal(code='artifactImportRefused', reason=str(exc), repair_action=exc.repair_action, state='refused') from exc
    except science_catalog.ScienceRefusal as exc:
        raise ClientRefusal(code=exc.code, reason=str(exc), repair_action=exc.repair_action) from exc
    except (ValueError, OSError, KeyError) as exc:
        raise ClientRefusal(code='diagnosticTransportRefused', reason=str(exc), repair_action=getattr(exc, 'repair_action', archives.Refusal.repair_action)) from exc


def remote(client, invocation, common):
    from ..client_cli import ClientRefusal
    try:
        validate(invocation, 1)
        verb, value = invocation.spec.verb, invocation.positionals[0]
        root = str(Path(paths.project_root()).resolve())
        changed = verb not in ('cleanup-plan', 'science-call')
        if verb == 'science-call':
            from ..experiment.science_actions import request
            document = json.loads(Path(invocation.one('--request')).read_bytes())
            action, _, _, _ = request(value, invocation.one('--action'), document)
            changed = action['method'] != 'GET'
            result = client.science_call(value, invocation.one('--action'), document)
        elif verb == 'science-stage':
            result = client.stage_diagnostic(value, invocation.one('--sha256'))
            from ..experiment.diagnostic_inputs import save_stage_reference
            if result.get('request') != {'inputBundleSHA256': invocation.one('--sha256')}:
                raise archives.Refusal('The stage response differs from the supplied archive digest.')
            result = {**result, **save_stage_reference(invocation.one('--sha256'), root)}
        elif verb == 'science-export': result = client.export_diagnostic(value)
        elif verb == 'science-fetch':
            client.require_http_transfer()
            reference = client.export_diagnostic(value)
            if reference['context']['jobID'] != value: raise archives.Refusal('Export belongs to another job.')
            incoming = archives.ordinary(root, '.steerlab/diagnostic-incoming', missing=True); incoming.mkdir(parents=True, exist_ok=True)
            destination = incoming / (reference['bundleSha256'] + '.tar.gz')
            if not destination.exists():
                client.download_bundle(remote_path=reference['bundlePath'], expected_sha256=reference['bundleSha256'], destination=str(destination), request_timeout=client.diagnostic_timeout)
            result = archives.import_evidence(destination, reference['bundleSha256'], root, expected_context=reference['context'])
            if result['receipt']['context'] != reference['context']: raise archives.Refusal('Imported archive origin differs from the exporting job.')
        else:
            custody = archives.verify(invocation.one('--receipt-sha256'), root)
            if custody['context']['jobID'] != value: raise archives.Refusal('Receipt belongs to another job.')
            # The endpoint's plan compares serving/metadata roots, original plan
            # digest, export hash and entries. A valid receipt is no bypass.
            result = client.diagnostic_cleanup_plan(value, custody) if verb == 'cleanup-plan' else client.diagnostic_cleanup_apply(value, custody, invocation.one('--plan-sha256'))
        print(json.dumps(result, indent=2))
        return CLIResult(message='Diagnostic transport or cleanup operation completed; inspect retained and removed paths.',
                         changed=result.get('changed', changed), payload={**common, 'response': result})
    except science_catalog.ScienceRefusal as exc:
        raise ClientRefusal(code=exc.code, reason=str(exc), repair_action=exc.repair_action) from exc
    except (ValueError, OSError, KeyError) as exc:
        raise ClientRefusal(code='diagnosticTransportRefused', reason=str(exc), repair_action=archives.Refusal.repair_action) from exc
