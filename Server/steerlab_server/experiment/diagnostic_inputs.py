"""Discover exact diagnostic input dependencies and package execution copies."""
from pathlib import Path
import codecs
import json
import re
from . import diagnostic_archives as archives

#: Every provider holds its code under the key ``sourceText``. JSON can also
#: spell that key with backslash-u escapes, and a JSON reader decodes them, so
#: the literal bytes alone settle nothing. An escape can only spell one of the
#: key's letters, and, at the next depth of JSON held inside a JSON string, an
#: escape's own backslash, ``u``, or hex digits: every spelling of the key at
#: any depth leaves the literal key or one of those escapes. A file with none
#: carries no provider; one with any is parsed: whole up to this size, line by
#: line above it. (Escapes of anything else, such as accented text in a
#: corpus, cannot take part, so they do not cost a parse.)
_PROVIDER_MARK = b'sourceText'
_MARK = re.compile(rb'sourceText|\\u00(?:3[0-9]|4[1-6]|54|5[cC]|6[1-6]|6[fF]|7[2-5]|78)')
_TEXT_MARK = re.compile(_MARK.pattern.decode())
_MARK_TAIL = len(_PROVIDER_MARK) - 1     # longer than any escape the pattern matches
_JSON_SPACE = ' \t\r\n'
_CODE_PARSE_BYTES = 256 * 1024 * 1024


class CustomCodeRefusal(archives.Refusal):
    """Packaging declined: the inputs carry custom code nobody acknowledged."""

    code = 'missingPrerequisite'

    def __init__(self, message, repair_action):
        super().__init__(message)
        self.repair_action = repair_action


class _Uncheckable(Exception):
    """A file too large to parse whole that is not JSON Lines."""


class _TooDeep(Exception):
    """JSON nested past the interpreter's limit. Refused, never skipped: a
    reader with more stack could load what this check could not."""


def _marked(data):
    """Whether bytes or text hold the key, or an escape that could spell it."""
    return (_MARK if isinstance(data, bytes) else _TEXT_MARK).search(data) is not None


def _may_carry_code(path):
    """Whether a file could hold a provider a JSON reader would load: it starts
    as JSON (after whitespace or a byte-order mark) and is marked. A streaming
    search, so a large input is read once and never held. A JSON reader also
    takes UTF-16 and UTF-32, decoded as it decodes them (byte-order mark,
    surrogates passed through); such a file is searched as text."""
    size = 1024 * 1024
    with open(path, 'rb') as handle:
        chunk = handle.read(size)
        encoding = json.detect_encoding(chunk) if chunk else 'utf-8'
        if encoding not in ('utf-8', 'utf-8-sig'):
            return _may_carry_code_wide(handle, chunk, encoding, size)
        body = chunk.removeprefix(b'\xef\xbb\xbf')
        while not (body := body.lstrip(b' \t\r\n')) and chunk:   # any amount of leading whitespace
            body = chunk = handle.read(size)
        if body[:1] not in (b'{', b'['):
            return False
        tail = b''
        while chunk:
            if _marked(tail + chunk):
                return True
            tail = chunk[-_MARK_TAIL:]
            chunk = handle.read(size)
    return False


def _may_carry_code_wide(handle, chunk, encoding, size):
    """`_may_carry_code` for a UTF-16 or UTF-32 file. Bytes that do not decode
    are no JSON a reader could load (it fails on the same bytes)."""
    decoder = codecs.getincrementaldecoder(encoding)('surrogatepass')
    started, tail = False, ''
    try:
        while chunk:
            text = decoder.decode(chunk)
            if not started:
                text = text.lstrip(_JSON_SPACE)
                if text:
                    if text[0] not in '{[':
                        return False
                    started = True
            if started and _marked(tail + text):
                return True
            tail = text[-_MARK_TAIL:] if started else ''
            chunk = handle.read(size)
        decoder.decode(b'', final=True)
    except UnicodeDecodeError:
        return False
    return False


def _loads(data):
    """A JSON value, or None where a JSON reader fails; raises
    :class:`_TooDeep` where it fails on depth."""
    try:
        return json.loads(data)
    except RecursionError:
        raise _TooDeep from None
    except ValueError:
        return None


def _line_documents(line):
    """The documents one line of a JSON Lines file holds: the line, or, where
    it is not one, each piece of it between carriage returns (a reader that
    also splits on those sees records there). Bytes or text."""
    if (document := _loads(line)) is not None:
        return [document]
    return [document for piece in line.split(b'\r' if isinstance(line, bytes) else '\r')
            if _marked(piece) and (document := _loads(piece)) is not None]


def _documents(raw):
    """The JSON documents in a file: the whole file, or else each line that
    could hold a provider (a JSON Lines file), decoded as the JSON reader
    would decode the file. Split at line feeds, as the large-file path is,
    with a carriage-return split tried where a line is not a document."""
    if (document := _loads(raw)) is not None:
        return [document]
    text = raw.decode(json.detect_encoding(raw), errors='replace')
    return [document for line in text.split('\n') if _marked(line)
            for document in _line_documents(line)]


def _streamed_documents(path):
    """The documents of a file too large to read whole, one line at a time.

    Only JSON Lines can be checked this way, and the first line that is not
    blank settles whether the file is that: it must be a complete JSON value.
    Then nothing else in the file can be part of it, and a JSON reader of the
    whole file fails ("extra data"), so the lines are the only documents any
    reader loads; each that could hold a provider is parsed, and one that does
    not parse is no document, as below the bound. A first line that is not a
    complete value may begin one large document, which cannot be checked a
    line at a time: raises :class:`_Uncheckable`, as does a line over the
    bound or a file in an encoding other than UTF-8."""
    with open(path, 'rb') as handle:
        if json.detect_encoding(handle.read(4)) not in ('utf-8', 'utf-8-sig'):
            raise _Uncheckable
        handle.seek(0)
        first = True
        while line := handle.readline(_CODE_PARSE_BYTES + 1):
            if len(line) > _CODE_PARSE_BYTES:
                raise _Uncheckable
            if first:
                if not line.removeprefix(b'\xef\xbb\xbf').strip(b' \t\r\n'):
                    continue
                first = False
                if (document := _loads(line)) is None:
                    raise _Uncheckable
                yield document
            elif _marked(line):
                yield from _line_documents(line)


def _carried_code(files, root):
    """Every expert provider in the plan's input files, as ``(providers,
    sources)`` in :func:`custom_code.providers` shape: the code an execution
    copy of these inputs could run. Any file that starts as JSON is checked,
    whatever its name; one too large to check is refused, never skipped."""
    from . import custom_code
    root = Path(root).resolve()
    names, sources = {}, {}
    for entry in files:
        path = root / entry['path']
        try:
            if not _may_carry_code(path):
                continue
        except OSError:
            continue
        try:
            documents = (_streamed_documents(path) if entry.get('bytes', 0) > _CODE_PARSE_BYTES
                         else _documents(path.read_bytes()))
            for document in documents:
                for provider in custom_code.providers(document):
                    names.setdefault(provider['sha256'], set()).update(provider['policyNames'])
                sources.update(custom_code.provider_sources(document))
        except _TooDeep:
            raise CustomCodeRefusal(
                f"The diagnostic was not packaged: {entry['path']} is JSON nested too deeply to check for "
                "custom code.",
                'Prepare the input with less nesting, then review the input plan again.') from None
        except _Uncheckable:
            raise CustomCodeRefusal(
                f"The diagnostic was not packaged: {entry['path']} may hold custom code, and at "
                f"{entry['bytes']} bytes it is too large to check (only JSON Lines, one complete JSON "
                "value per line, can be checked at that size).",
                'Prepare a smaller input, or split the policy or agent out into its own file, then review the input plan again.') from None
    return [{'sha256': digest, 'policyNames': sorted(names[digest])} for digest in sorted(names)], sources


def _custom_code_block(reviewed, root):
    """The ``customCode`` block an input plan carries when its files hold
    custom code, or None. Outside the plan digest: acknowledging the code
    changes this block, not the inputs the digest pins."""
    from . import custom_code
    carried, sources = _carried_code(reviewed['files'], root)
    if not carried:
        return None
    try:
        seen = custom_code.acknowledged(root)
    except custom_code.CustomCodeError as exc:   # a damaged record is refused, never read as empty
        raise CustomCodeRefusal(
            str(exc), f'Restore {custom_code.FILENAME} from the workspace history (it is an ordinary tracked '
            'file), or move it aside; then review the input plan again and acknowledge the code it shows.') from exc
    rows = [{**row, 'sourceText': sources[row['sha256']], 'acknowledged': row['sha256'] in seen,
             'acknowledgedAt': seen.get(row['sha256'], {}).get('acknowledgedAt'),
             'acknowledgedBy': seen.get(row['sha256'], {}).get('acknowledgedBy')} for row in carried]
    pending = [row['sha256'] for row in rows if not row['acknowledged']]
    return {'notice': custom_code.DIAGNOSTIC_NOTICE if pending else None, 'providers': rows,
            'acknowledged': not pending, 'recordFile': custom_code.FILENAME,
            'acknowledgeFlag': '--custom-code-sha256 ' + ','.join(pending) if pending else None}


def plan(request, root):
    """The reviewed input plan: every file an execution copy carries, hashed.
    When those files hold custom code (a policy's expert provider), the plan
    also carries the notice and the code itself under ``customCode``."""
    reviewed = _plan(request, root)
    block = _custom_code_block(reviewed, root)
    return {**reviewed, 'customCode': block} if block else reviewed


def _plan(request, root):
    from ..api.scientific_execution import request_document
    from . import paths
    request = request_document(request)
    from . import managed_methods, managed_inputs
    if request['operation'] in managed_methods.OPERATIONS:
        return managed_inputs.plan(request, root)
    root = Path(root).resolve(); p = request['parameters']; files = set()
    def add(path):
        path = Path(path)
        relative = path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()
        files.update(archives.files_in(root, relative))
    if request['operation'] == 'battery':
        from . import battery_run, model_variant
        add(p['batteryFile'])
        agents = battery_run.resolve_agents(battery_run.parse_agents(p['agents']), root=str(root),
            model_id=p.get('modelID'), revision=p.get('revision'), alpha_units=p['alphaUnits'])
        for agent in agents:
            identity = agent.identity
            if identity.get('artifactPath'): add(identity['artifactPath'])
            references = [identity.get('vectorArtifactID')] + [i.get('vectorArtifactID') for i in identity.get('injections', [])]
            for reference in filter(None, references):
                archives.parts(reference)  # Portable copies must never resolve back to an absolute source.
                add(paths.resolve_artifact(reference, str(root)) + '.json')
                add(paths.resolve_artifact(reference, str(root)) + '.safetensors')
            if agent.variant:
                for adapter in agent.variant.adapters:
                    reference = adapter.get('adapterDirectory') or adapter.get('artifactPath') or ''
                    archives.parts(reference)
                    directory = paths.resolve_artifact(reference, str(root))
                    add(directory)
                    sidecar = Path(model_variant.adapter_sidecar_path(directory))
                    if sidecar.exists() or sidecar.is_symlink(): add(sidecar)
                if agent.variant.neutral_pc_basis_path:
                    archives.parts(agent.variant.neutral_pc_basis_path)
                    add(paths.resolve_artifact(agent.variant.neutral_pc_basis_path, str(root)))
    else:
        from . import extract_stability, experiment_store, multiconcept
        prepared = extract_stability.preflight(p['experiment'], p['concept'], root=str(root),
            resamples=p['resamples'], fraction=p['fraction'])
        document = experiment_store.load_raw(p['experiment'], str(root))
        add(document.source_path)
        ref = prepared['ref']
        if prepared['method'].is_designated_reference:
            add(multiconcept.stories_path(ref.name, str(root)))
            add(multiconcept.stories_path(ref.designated_reference['name'], str(root)))
        else:
            add(paths.concept_directory(ref.name, str(root)))
    # Battery and stability closures are only ever planned to travel (package,
    # stage, verify a staged copy), so the transport bound applies at review.
    result = {'schemaVersion': 1, 'request': request, 'sourceRoot': str(root), 'files': archives.snapshot(root, files)}
    return {**result, 'planSHA256': archives.digest(result)}


def package(request, root, destination, expected, acknowledge=None):
    """Package the reviewed inputs. Inputs that carry custom code are packaged
    only once every piece of it is acknowledged in this workspace: earlier, or
    now, by naming each SHA-256 the plan shows in ``acknowledge`` (the
    comma-separated value of ``--custom-code-sha256``)."""
    from . import custom_code
    reviewed = plan(request, root)
    if reviewed['planSHA256'] != expected: raise archives.Refusal('Input source plan changed; re-review before packaging.')
    block = reviewed.get('customCode')
    named = [part.strip() for part in (acknowledge or '').split(',') if part.strip()]
    if named and not block:
        raise CustomCodeRefusal('These inputs carry no custom code, so there is nothing to acknowledge.',
                                'Package without --custom-code-sha256.')
    carried = [row['sha256'] for row in block['providers']] if block else []
    unknown = [digest for digest in named if digest not in carried]
    if unknown and block['acknowledged']:
        # A typo must never read as acknowledging code nobody looked at, even
        # when everything these inputs carry is acknowledged already.
        raise CustomCodeRefusal('The inputs carry no custom code with SHA-256 ' + ', '.join(unknown) + '.',
                                'Package without --custom-code-sha256: every provider these inputs carry is acknowledged.')
    if block and not block['acknowledged']:
        pending = [row for row in block['providers'] if not row['acknowledged']]
        if unknown or any(row['sha256'] not in named for row in pending):
            listed = '; '.join(f"SHA-256 {row['sha256']} (policy: {', '.join(row['policyNames']) or 'unnamed'})"
                               for row in pending)
            raise CustomCodeRefusal(
                ('The inputs carry no custom code with SHA-256 ' + ', '.join(unknown) + '. ' if unknown else '')
                + f'The diagnostic was not packaged: its inputs contain custom code nobody has acknowledged '
                f'in this workspace ({listed}). {custom_code.DIAGNOSTIC_NOTICE}',
                'Read the code under customCode.providers in `science input-plan <request.json>`. If you trust '
                'the source, repeat `science package` with ' + block['acknowledgeFlag'] + '.')
        custom_code.acknowledge_inputs(root, block['providers'], [row['sha256'] for row in pending],
                                       operation=reviewed['request'].get('operation', ''))
    # The archive goes to another machine, so its description names none: it
    # used to carry `sourceRoot`, the workspace's absolute path here, which no
    # reader used. The reviewed plan's digest still identifies the source.
    context = {'request': reviewed['request'], 'sourcePlanSHA256': expected}
    return archives.package(root, [e['path'] for e in reviewed['files']], destination,
        kind='diagnosticInput', context=context, expected_entries=reviewed['files'])


def save_stage_reference(digest, root):
    """A local planning input naming the verified remote execution copy."""
    import re
    import tempfile
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise archives.Refusal('Use the inputBundleSHA256 returned by staging.')
    root = Path(root).resolve(strict=True)
    directory = archives.ordinary(root, '.steerlab/diagnostic-requests', missing=True)
    directory.mkdir(parents=True, exist_ok=True)
    destination = archives.ordinary(directory, digest + '.json', missing=True)
    data = archives.encoded({'inputBundleSHA256': digest})
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        source = Path(temp) / 'request.json'; source.write_bytes(data)
        try: archives.publish_file(source, destination)
        except FileExistsError:
            if destination.read_bytes() != data:
                raise archives.Refusal('The saved staged request differs; inspect it before planning.')
    return {'localRequestPath': str(destination), 'request': {'inputBundleSHA256': digest},
            'nextAction': 'Pass localRequestPath as the positional request file to science-plan, then science-submit with the reviewed planSHA256 on the same controller.'}
