"""Local-only JSON process adapter shared by the Mac UI and CLI."""
import json
import sys
from .runtime_identity import source_sha256

RUNTIME_REPAIR = 'Rebuild or reinstall the app and its Python client payload from the same reviewed source. Use a Python client environment with the declared client dependencies.'


class ClientRuntimeMismatch(ValueError):
    pass


def main():
    repair = RUNTIME_REPAIR
    identity = None
    try:
        document = json.load(sys.stdin)
        if not isinstance(document, dict) or set(document) != {'action', 'payload', 'clientSHA256'}:
            raise ClientRuntimeMismatch('Supply action, payload and the compiled client source identity.')
        identity = source_sha256()
        if document['clientSHA256'] != identity:
            raise ClientRuntimeMismatch('The Mac build and local Python client sources differ; no action was performed.')
        from ..experiment.diagnostic_archives import Refusal
        repair = Refusal.repair_action
        from .diagnostic_commands import workspace_action
        print(json.dumps({'ok': True, 'clientSHA256': identity, 'result': workspace_action(document['action'], document['payload'])}))
        return 0
    except Exception as exc:
        repair = getattr(exc, 'repair_action', None) or getattr(exc, 'repair', None) or repair
        if isinstance(exc, ImportError):
            repair = RUNTIME_REPAIR
        print(json.dumps({'ok': False, 'clientSHA256': identity, 'reason': str(exc), 'repairAction': repair}))
        return 65


if __name__ == '__main__':
    raise SystemExit(main())
