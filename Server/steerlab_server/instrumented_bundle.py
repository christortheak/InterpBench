"""Queued-worker entry point: verify runtime support before importing the CLI.

An older installation has no such module, so a newer controller cannot silently
execute a policy-free substitute through an older queued child. Plain studies
retain the historical CLI command. This is an internal submission entry point.
"""
import json
import sys
from .experiment import instrumentation_contract


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if not args: raise ValueError('The queued request has no runtime requirement declaration.')
        required = json.loads(args[0])
        if not isinstance(required, list) or any(not isinstance(x, str) for x in required): raise ValueError('Malformed queued runtime requirements.')
        instrumentation_contract.require(required, {'instrumentation': list(instrumentation_contract.SUPPORTED)})
    except ValueError as exc:
        print(json.dumps({'state': 'refused', 'code': 'unsupportedInstrumentation', 'reason': str(exc),
                          'repairAction': 'Update the queued worker environment, then resubmit the unchanged bundle.'}), file=sys.stderr)
        return 65
    from .cli import main as execute
    return execute(args[1:])


if __name__ == '__main__':
    raise SystemExit(main())
