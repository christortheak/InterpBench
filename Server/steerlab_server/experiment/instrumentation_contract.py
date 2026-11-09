"""Portable execution requirements, outside content-hashed study documents."""
import json
from .probe_artifacts import ProbeError

#: A panel study's bundle requires this when an engine from before issue #15
#: would run its panel differently, or when its panel could not be read at
#: packaging (``bundles.package_experiment``). Swift twin:
#: ``InstrumentationSupport.panelAnswerRouting``.
PANEL_ANSWER_ROUTING = 'panel-answer-routing-v1'
SUPPORTED = ('policy-evidence-v2', 'policy-v1', 'policy-v2', 'probe-readings-v1')
#: What this engine offers in its capability snapshot: every requirement a
#: document can declare, plus the ones a packager adds for the study as a
#: whole (``PANEL_ANSWER_ROUTING``).
OFFERED = SUPPORTED + (PANEL_ANSWER_ROUTING,)
#: What an archive packaged before requirements were declared may need. Version-2
#: policies arrived later, so such an archive cannot hold one.
UNMARKED = ('policy-evidence-v2', 'policy-v1', 'probe-readings-v1')


def policy_versions(attachments):
    """Schema versions of attached policies, read from their exact bytes; unreadable bytes are refused elsewhere."""
    versions = set()
    for item in attachments if isinstance(attachments, list) else []:
        try: document = json.loads(item['json'].encode())
        except (TypeError, KeyError, ValueError, AttributeError, RecursionError): continue
        if isinstance(document, dict) and type(document.get('schemaVersion')) is int: versions.add(document['schemaVersion'])
    return versions


def requirements(value):
    """Scan raw documents, including embedded panel agents, without a lossy decoder."""
    found = set()
    if isinstance(value, dict):
        if value.get('interventionPolicies'):
            found.update(('policy-v1', 'policy-evidence-v2'))
            # An engine that knows only version 1 would refuse these at execution; say so before submission.
            if 2 in policy_versions(value['interventionPolicies']): found.add('policy-v2')
        if isinstance(value.get('probeMeasurements'), dict) and value['probeMeasurements'].get('probes'):
            found.add('probe-readings-v1')
        declared = value.get('runtimeRequirements', [])
        if not isinstance(declared, list) or any(not isinstance(x, str) for x in declared):
            raise ProbeError('Runtime requirements must be a list of versioned capability names.')
        found.update(declared)
        for child in value.values(): found.update(requirements(child))
    elif isinstance(value, list):
        for child in value: found.update(requirements(child))
    return sorted(found)


def require(required, capabilities):
    offered = capabilities.get('instrumentation', [])
    missing = sorted(set(required) - set(offered if isinstance(offered, list) else []))
    if PANEL_ANSWER_ROUTING in missing:
        raise ProbeError('This engine predates answer-only panel routing (' + PANEL_ANSWER_ROUTING + '): it would pass each turn’s whole text on to the other seats, a thinking seat’s reasoning included, so the study’s panel would not run as declared. Update and restart the engine, then reconnect and review the submission. Existing artifacts do not need editing.')
    if missing:
        raise ProbeError('This engine cannot execute the study’s declared probes or policies: ' + ', '.join(missing) + '. Update and restart the engine, then reconnect and review the submission. Existing artifacts do not need editing.')
