"""Local-only JSON process adapter shared by the Mac UI and CLI."""
import json
import sys
from .diagnostic_commands import workspace_action
from ..experiment.diagnostic_archives import Refusal


def main():
    try:
        document = json.load(sys.stdin)
        if set(document) != {'action', 'payload'}: raise Refusal('Supply action and payload.')
        print(json.dumps({'ok': True, 'result': workspace_action(document['action'], document['payload'])}))
        return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'reason': str(exc), 'repairAction': Refusal.repair_action}))
        return 65


if __name__ == '__main__':
    raise SystemExit(main())
