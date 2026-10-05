"""Local-only JSON process adapter shared by the Mac UI and CLI.

The Mac build sends the source identity it was compiled against; this process
hashes the sources it was actually loaded from and refuses to act when the two
differ. The comparison is entirely local: one Mac build against the Python
files beside it on the same machine. No server or controller takes part.
"""
import json
import sys
from pathlib import Path
from .runtime_identity import source_sha256

RUNTIME_REPAIR = 'Rebuild or reinstall the app and its Python client payload from the same reviewed source. Use a Python client environment with the declared client dependencies.'

#: The repair for a source mismatch. A fresh start leads, because the usual
#: cause is an app replaced on disk while an older copy of it was still
#: running. Installing dependencies again cannot change which sources these
#: are, so it is not offered.
MISMATCH_REPAIR = 'Run the command again from a fresh start, so the Mac build and its Python client files come from the same installation. If they still differ, reinstall the complete app, or rebuild the Mac build and its Python client payload from the same reviewed source. No server takes part in this check.'

#: Confirms the two sides are the same source and does nothing else: no owner
#: is imported and no workspace is read. The Mac calls it before a long remote
#: step, so a mismatch is found before the controller has done any work.
IDENTITY_CHECK = 'client-identity'


class ClientRuntimeMismatch(ValueError):
    pass


def main():
    repair = RUNTIME_REPAIR
    identity = None
    # The package these sources were loaded from: what was actually hashed,
    # whatever the caller believes it put on the path.
    root = str(Path(__file__).resolve().parents[1])
    try:
        document = json.load(sys.stdin)
        if not isinstance(document, dict) or set(document) != {'action', 'payload', 'clientSHA256'}:
            raise ClientRuntimeMismatch('Supply action, payload and the compiled client source identity.')
        identity = source_sha256()
        if document['clientSHA256'] != identity:
            repair = MISMATCH_REPAIR
            expected = str(document['clientSHA256'])[:12]
            raise ClientRuntimeMismatch(
                'The Mac build and local Python client sources differ; no action was performed. '
                f'The Mac build expects sources {expected}; the files at {root} are {identity[:12]}.')
        if document['action'] == IDENTITY_CHECK:
            if document['payload'] != {}:
                raise ClientRuntimeMismatch('The identity check takes an empty payload.')
            print(json.dumps({'ok': True, 'clientSHA256': identity, 'clientRoot': root, 'result': {'changed': False}}))
            return 0
        from ..experiment.diagnostic_archives import Refusal
        repair = Refusal.repair_action
        from .diagnostic_commands import workspace_action
        print(json.dumps({'ok': True, 'clientSHA256': identity, 'clientRoot': root, 'result': workspace_action(document['action'], document['payload'])}))
        return 0
    except Exception as exc:
        repair = getattr(exc, 'repair_action', None) or getattr(exc, 'repair', None) or repair
        # `code` and `state` (additive): whether the action was REFUSED (65) or
        # asked for in the wrong shape (`blocked`, 64), classified in the one
        # place both clients share. A source mismatch or a client that cannot
        # import is neither — nothing about the request was judged — and keeps
        # the `blocked` it has always had.
        fields = {'code': 'usage', 'state': 'blocked'}
        if isinstance(exc, ImportError):
            repair = RUNTIME_REPAIR
        elif not isinstance(exc, ClientRuntimeMismatch):
            try:
                from .diagnostic_commands import refusal_fields
                fields = refusal_fields(exc)
            except ImportError:
                repair = RUNTIME_REPAIR
        print(json.dumps({'ok': False, 'clientSHA256': identity, 'clientRoot': root, 'reason': str(exc),
                          'repairAction': repair, **fields}))
        # This adapter's own exit status is unchanged; the callers read `state`.
        return 65


def main_for_the_mac():
    """This adapter serves the Mac app and its command line, and nothing else:
    any repair composed while it runs is shown there, so it names that command
    line's verbs and no other client's."""
    from ..experiment import command_vocabulary as vocabulary
    with vocabulary.speaking_as(vocabulary.MAC):
        return main()


if __name__ == '__main__':
    raise SystemExit(main_for_the_mac())
