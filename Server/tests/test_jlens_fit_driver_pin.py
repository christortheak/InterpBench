"""The J-lens fitting driver's hash, pinned to a literal.

Every J-lens fit records ``runtime.driverSHA256``: one SHA-256 over the bytes
of the modules that make up the fitting driver, joined in a fixed order
(``jlens_fit_model.load``). The hash is provenance. It says which source
produced a lens, and a continuation's report sets the source checkpoint's
driver beside the current one. A reworded message or a comment in any of
those modules therefore changes what every new fit records, although no
arithmetic changed.

This test makes such a change visible in a diff, so that it is made on
purpose. It needs no model, no GPU, and none of the optional fitting
packages, so it runs wherever the suite runs.
"""
import ast
import hashlib
from pathlib import Path

import pytest

from steerlab_server.experiment import jlens_fit_model

#: The modules whose bytes the driver hash covers, in the order they are joined.
DRIVER_MODULES = (
    'jlens_fit',
    'jlens_fit_execution',
    'jlens_fit_model',
    'jlens_fit_identity',
    'jlens_fit_telemetry',
    'jlens_kernel_policy',
    'jlens_fit_selection',
    'jlens_stopping',
)
#: What a fit records as ``runtime.driverSHA256`` while those bytes stay as they are.
DRIVER_SHA256 = '32eb0b6ff6e894343ec6bb9a18f65db78aba9b21f308a6dfd713f538576732df'

SOURCE = Path(jlens_fit_model.__file__)

BYTES_CHANGED = '''
The J-lens fitting driver's bytes changed, so every new fit would record a
different driverSHA256.

  pinned in this test: {pinned}
  these files now:     {current}

The driver is these modules in Server/steerlab_server/experiment:
  {modules}

If the change is intended, update DRIVER_SHA256 in this test in the same
commit, and say in CHANGELOG.md that new fits record a new driver hash.
If it is not intended (a wording or comment edit, for example), revert the
edit: the hash covers every byte of these files, not only the arithmetic.
'''

MODULES_CHANGED = '''
The set of modules behind the J-lens fitting driver's hash changed, so every
new fit would record a different driverSHA256.

  named in this test:
    {named}
  hashed by jlens_fit_model.py:
    {hashed}
  added: {added}; removed: {removed}; order changed: {reordered}
  new fits now record: {current}

If the change is intended, update DRIVER_MODULES and DRIVER_SHA256 in this
test in the same commit, and say in CHANGELOG.md that new fits record a new
driver hash and which modules it covers. If it is not intended, revert the
edit to jlens_fit_model.py.
'''

FORMATION_CHANGED = '''
jlens_fit_model.py no longer forms driverSHA256 the way this test does (one
SHA-256 over the listed modules' bytes, joined in order), so the literal
pinned here would not be the hash a fit records.

  {detail}

If the change is intended, update this test to compute the hash the new way,
update DRIVER_SHA256 in the same commit, and say in CHANGELOG.md that new
fits record a new driver hash. If it is not intended, revert the edit to
jlens_fit_model.py.
'''


def production_driver_hash():
    """The modules ``jlens_fit_model.load`` hashes and the digest it records.

    Read from the source, because ``load`` itself needs a prepared model. The
    record's ``driverSHA256`` entry is evaluated as written, with only the
    names it uses, so the digest is production's own and not a copy of it.
    """
    entries = [value for node in ast.walk(ast.parse(SOURCE.read_bytes()))
               if isinstance(node, ast.Dict)
               for key, value in zip(node.keys, node.values)
               if isinstance(key, ast.Constant) and key.value == 'driverSHA256']
    if len(entries) != 1:
        pytest.fail(FORMATION_CHANGED.format(
            detail=f'It has {len(entries)} driverSHA256 entries; this test reads exactly one.'))
    loops = [node.iter for node in ast.walk(entries[0]) if isinstance(node, ast.comprehension)]
    if len(loops) != 1 or not isinstance(loops[0], (ast.Tuple, ast.List)) or not all(
            isinstance(item, ast.Constant) and isinstance(item.value, str) for item in loops[0].elts):
        pytest.fail(FORMATION_CHANGED.format(
            detail='Its module names are no longer one literal list inside the driverSHA256 entry.'))
    try:
        digest = eval(compile(ast.Expression(entries[0]), str(SOURCE), 'eval'),
                      {'hashlib': hashlib, 'Path': Path, '__file__': str(SOURCE)})
    except Exception as error:
        pytest.fail(FORMATION_CHANGED.format(
            detail=f'Its driverSHA256 entry could not be evaluated alone ({type(error).__name__}: {error}).'))
    return tuple(item.value for item in loops[0].elts), digest


def test_fitting_driver_hash_changes_only_on_purpose():
    hashed, recorded = production_driver_hash()
    assert hashed == DRIVER_MODULES, MODULES_CHANGED.format(
        named=', '.join(DRIVER_MODULES), hashed=', '.join(hashed),
        added=', '.join(sorted(set(hashed) - set(DRIVER_MODULES))) or 'none',
        removed=', '.join(sorted(set(DRIVER_MODULES) - set(hashed))) or 'none',
        reordered='yes' if sorted(hashed) == sorted(DRIVER_MODULES) else 'no',
        current=recorded)
    current = hashlib.sha256(b''.join(
        SOURCE.with_name(name + '.py').read_bytes() for name in DRIVER_MODULES)).hexdigest()
    assert current == recorded, FORMATION_CHANGED.format(
        detail=f'jlens_fit_model.py records: {recorded}\n  this test computes:         {current}')
    assert current == DRIVER_SHA256, BYTES_CHANGED.format(
        pinned=DRIVER_SHA256, current=current,
        modules='\n  '.join(name + '.py' for name in DRIVER_MODULES))
