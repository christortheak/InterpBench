#!/usr/bin/env python3
"""Prove managed adapters leave the existing scientific owners' ASTs unchanged."""
import argparse
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OWNER_MODULES = (
    'optvec_train', 'optvec_eval', 'optvec_geometry', 'optvec_interpret',
    'optvec_gradient', 'optvec_jspace', 'optvec_campaign', 'family_report',
    'sae_qualification', 'sae_candidates', 'analysis_workflow',
)
LENS_IMPORT = 'from . import artifact_paths\n'
OLD_CALL = 'paths.resolve(record.converted.path, root)'
NEW_CALL = 'artifact_paths.converted_file(record.lensID, record.converted.path, root)'


#: Reviewed additions to analysis_workflow, each removed (or reverted) exactly
#: once before the AST comparison; anything else in the module must match the
#: base. P6 adds one independent descriptive report. Waves 2 and 3 (reviewed,
#: approved 2026-10-05) add outcome-coverage.json: the outcomes that reached
#: the effect rows and those the run could not produce, computed read-only from
#: the records and the finished rows; and the exclusion-pin refusal's repair
#: text is now spelled for the client that shows it. No existing endpoint or
#: statistic body changes.
ANALYSIS_WORKFLOW_CHANGES = (
    ("""    from . import instrumentation_evidence
    if any('probeMeasurements' in r or 'interventionDecisions' in r for r in records):
        with open(os.path.join(out, 'instrumentation-summary.json'), 'w', encoding='utf-8') as handle:
            json.dump(instrumentation_evidence.summarize(records), handle, indent=2, sort_keys=True, allow_nan=False)

""", ''),
    ("""from .analysis_endpoints import (MARKER_DENSITY_NOT_RECORDED,
    condition_modalities, endpoint_values, key_records_by_transcript,
    marker_density_concepts, marker_density_not_recorded, outcome_coverage,
    promotion_decisions, stratified_effect_rows, transcript_level_diffs)
""", """from .analysis_endpoints import (condition_modalities, endpoint_values,
    key_records_by_transcript, promotion_decisions, stratified_effect_rows,
    transcript_level_diffs)
"""),
    ("""    otherwise derive), the outcome list with each outcome's definition in
    words and anything this engine could not produce (outcome-coverage.json),
    and the promoted-movers funnel artifact for
""", """    otherwise derive), and the promoted-movers funnel artifact for
"""),
    ("""    concepts_without_marker_density = marker_density_not_recorded(
        records, [concept.name for concept in manifest.concepts])
""", ''),
    ("repair=exclusions_mod.pin_required_repair())",
     "repair=exclusions_mod.PIN_REQUIRED_REPAIR)"),
    ("""    coverage = outcome_coverage(
        [row.endpoint for row in rows + stratified_rows],
        marker_concepts=marker_density_concepts(records),
        not_available=[(f"{concept}MarkerDensity", "markerDensity",
                        MARKER_DENSITY_NOT_RECORDED)
                       for concept in concepts_without_marker_density])
    with open(os.path.join(out, "outcome-coverage.json"), "w",
              encoding="utf-8") as handle:
        json.dump(coverage, handle, indent=2, sort_keys=True)
    for entry in coverage["outcomes"]:
        if entry["status"] == "notAvailable":
            _log(f"{entry['name']}: {entry['reason']}")
""", ''),
)


def tree(source):
    return ast.dump(ast.parse(source), include_attributes=False)


def without_declared_changes(after, changes):
    """Remove each declared change exactly once; refuse when one is missing or altered."""
    for new, old in changes:
        assert after.count(new) == 1, 'Declared analysis_workflow change missing or altered: ' + new.strip().splitlines()[0]
        after = after.replace(new, old)
    return after


def check_analysis_workflow(before, after):
    assert tree(before) == tree(without_declared_changes(after, ANALYSIS_WORKFLOW_CHANGES)), \
        'analysis_workflow changed scientific AST'


def check_lens(before, after):
    """Require the intended substitutions and permit only those changes."""
    count = before.count(OLD_CALL)
    assert count > 0, 'Baseline has no declared resolver call'
    assert after.count(LENS_IMPORT) == 1, 'Expected one artifact_paths import'
    assert after.count(NEW_CALL) == count, 'Expected every declared resolver substitution'
    assert OLD_CALL not in after, 'An original resolver call remains'
    normalized = after.replace(LENS_IMPORT, '').replace(NEW_CALL, OLD_CALL)
    assert tree(normalized) == tree(before), 'Changes beyond the declared resolver substitution/import'


def must_refuse(before, changed, label):
    try:
        check_lens(before, changed)
    except AssertionError:
        return
    raise AssertionError('Negative control was accepted: ' + label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='6a94a8f')
    args = parser.parse_args()

    def read(name, family):
        path = f'Server/steerlab_server/{family}/{name}.py'
        before = subprocess.check_output(['git', 'show', args.base + ':' + path], cwd=ROOT, text=True)
        return before, (ROOT / path).read_text()

    for name in OWNER_MODULES:
        before, after = read(name, 'experiment')
        if name == 'analysis_workflow':
            check_analysis_workflow(before, after)
            # Mutation controls through the SAME gate: an altered declared
            # change, and a changed existing statistic body, are both refused.
            for label, mutant in (
                    ('altered coverage rows', after.replace('rows + stratified_rows]', 'rows]', 1)),
                    ('removed repair change', after.replace('pin_required_repair()', 'PIN_REQUIRED_REPAIR', 1)),
                    ('changed existing body', after.replace('    stratified_rows: list = []', '    stratified_rows: list = [None]', 1))):
                assert mutant != after, 'Mutation control did not apply: ' + label
                try:
                    check_analysis_workflow(before, mutant)
                except AssertionError:
                    continue
                raise AssertionError('Negative control was accepted: ' + label)
            after = without_declared_changes(after, ANALYSIS_WORKFLOW_CHANGES)
        assert tree(before) == tree(after), name + ' changed scientific AST'
        assert tree(after + '\nAUDIT_NEGATIVE_CONTROL = True\n') != tree(before)
    print(f'{len(OWNER_MODULES)} scientific owner ASTs unchanged; negative controls passed.')

    for name in ('lens_store', 'qualification', 'g0'):
        before, after = read(name, 'jlens')
        check_lens(before, after)
        # Run mutations through the SAME gate, including its normalization.
        must_refuse(before, before, 'removed migration: ' + name)
        must_refuse(before, after.replace(NEW_CALL, NEW_CALL.replace(', root)', ', None)'), 1), 'wrong root: ' + name)
        check_lens(before, ast.unparse(ast.parse(after)) + '\n')
        changed = ast.parse(after)
        function = next(n for n in ast.walk(changed) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        function.body.insert(0, ast.parse('raise RuntimeError("audit mutation")').body[0])
        must_refuse(before, ast.unparse(changed) + '\n', 'changed consumer body: ' + name)
    print('Three lens consumers contain the required relocation calls and only declared changes; mutation controls passed.')


if __name__ == '__main__':
    main()
