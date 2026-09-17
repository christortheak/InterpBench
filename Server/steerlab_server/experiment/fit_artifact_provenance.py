"""Expose fitting provenance only from captured, matching SteerLab fit reports."""
import json
import re
from .artifact_sources import ImportRefusal, digest


def read(spec, files, tensor_hash):
    path = files.get('configFile')
    if path is None:
        return {}
    # Arbitrary third-party configs are retained without inferring their format.
    with path.open('rb') as stream:
        data = stream.read(8 * 1024**2 + 1)
    if len(data) > 8 * 1024**2:
        return {}
    try:
        report = json.loads(data)
    except (ValueError, UnicodeError):
        return {}
    if not isinstance(report, dict) or report.get('operation') not in ('jlens-fit', 'jlens-fit-merge'):
        return {}
    if report.get('schemaVersion') != 1:
        return {}
    identity = report.get('identity')
    if not isinstance(identity, dict) or not isinstance(identity.get('runtime'), dict):
        raise ImportRefusal('The fitting report has no runtime identity. Select the report accompanying these tensors.')
    expected = dict(modelID=spec['modelID'], revision=spec.get('modelRevision'), hiddenSize=spec['hiddenSize'],
                    targetLayer=spec['lens']['targetLayer'], sourceLayers=sorted(map(int, spec['lens']['layers'])))
    if (report.get('tensorSHA256') != tensor_hash or any(identity.get(k) != v for k, v in expected.items())
            or report.get('promptsFitted') != spec['lens'].get('promptsFitted')):
        raise ImportRefusal('The fitting report describes different tensors, geometry, or fitting counts. Choose the matching report.')
    # A merge across corpora is declared by both the report and the description,
    # entry for entry, or by neither; a description cannot hide the mixture.
    reported = report.get('corpora') if report.get('mixedCorpora') else None
    declared = spec['lens'].get('corpora')
    if (reported is None) != (declared is None) or (reported is not None and (not isinstance(reported, list) or len(reported) != len(declared)
            or any(not isinstance(entry, dict) or {key: entry.get(key) for key in ('corpusSHA256', 'promptsFitted', 'rowsConsidered')} != item
                   for entry, item in zip(reported, declared)))):
        raise ImportRefusal('The fitting report describes different corpus contributions. Choose the matching report.')
    runtime = identity['runtime']
    result = {'fitReportSHA256': digest(path)}
    for key, length in (('referenceCommit', 40), ('kernelSHA256', 64), ('driverSHA256', 64)):
        value = runtime.get(key)
        if value is not None:
            if not isinstance(value, str) or not re.fullmatch('[0-9a-fA-F]{' + str(length) + '}', value):
                raise ImportRefusal('The fitting report contains an invalid ' + key + '.')
            result[key] = value
    if result.get('referenceCommit'):
        result['referencePackage'] = 'jlens'
    return result
