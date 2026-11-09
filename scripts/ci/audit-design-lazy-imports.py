#!/usr/bin/env python3
"""Prove runtime bodies unchanged except the named lazy import boundary."""
import ast
import copy
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = '82da781'


def audit(module, before, after):
    old, new = ast.parse(before), ast.parse(after)
    old.body = [n for n in old.body if not (isinstance(n, ast.ImportFrom) and n.module == 'generate')]
    if module == 'model_variant':
        function = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'variant_injections')
        expected = ast.dump(ast.parse('from .generate import CellInjection').body[0])
        imports = [n for n in function.body if isinstance(n, ast.ImportFrom) and n.module == 'generate']
        assert len(imports) == 1 and ast.dump(imports[0]) == expected
        function.body.remove(imports[0])
    else:
        wrapper = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')
        assert ast.dump(wrapper) == ast.dump(ast.parse('''def generate(*args, **kwargs):
    """Load the GPU generator only when executing a turn, never to validate a panel."""
    from .generate import generate as execute
    return execute(*args, **kwargs)
''').body[0])
        new.body.remove(wrapper)
    assert ast.dump(old) == ast.dump(new), f'{module}: runtime or validation body changed'


PANEL_IDENTITY_HELPERS = {'DUPLICATE_ID_REPAIR', '_first_shared_id', 'duplicate_agent_id_problem',
                          'duplicate_turn_id_problem', 'duplicate_identifier_problem',
                          'pinned_panel_identity_problem'}


def admit_panel_identity(tree):
    """Remove exactly the reviewed panel-identity change (a repeated seat or turn
    ID is refused by name; approved by the maintainer 2026-10-04) from the
    current multi_agent tree, asserting its form, so every other body is still
    compared. The helpers have their own behavior tests
    (test_panel_identifier_uniqueness.py)."""
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module is None
               and any(a.name == 'lifecycle_gates' for a in n.names)]
    assert len(imports) == 1 and [a.name for a in imports[0].names][0] == 'lifecycle_gates'
    imports[0].names = imports[0].names[1:]
    error = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ScenarioError')
    init = [n for n in error.body if isinstance(n, ast.FunctionDef)]
    assert len(init) == 1 and init[0].name == '__init__' and len(error.body) == 2
    assert [ast.dump(s) for s in init[0].body] == [ast.dump(s) for s in ast.parse(
        'super().__init__(message)\nself.gate = gate\nself.repair_action = repair').body]
    error.body = [ast.Pass()]
    tree.body = [n for n in tree.body if getattr(n, 'name', None) not in PANEL_IDENTITY_HELPERS
                 and not (isinstance(n, ast.Assign) and any(getattr(t, 'id', None) in PANEL_IDENTITY_HELPERS
                                                            for t in n.targets))]
    validate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate')
    check = [ast.dump(s) for s in ast.parse(
        'duplicate = duplicate_identifier_problem(scenario)\nif duplicate is not None:\n    raise duplicate').body]
    dumps = [ast.dump(s) for s in validate.body]
    at = next(i for i in range(len(dumps)) if dumps[i:i + 2] == check)
    del validate.body[at:at + 2]


#: Top-level names the panel answer-routing change adds to multi_agent (public
#: issue #15, FR6 stream A as integrated with stream B; NEEDS MAINTAINER
#: APPROVAL). Each is exercised by Server/tests/test_panel_reasoning.py,
#: Server/tests/test_panel_agent_wiring.py, and the reasoning-split fixture.
#: After the integration, seat_reasoning reads an agent's level through
#: agent_identity.variant_reasoning_effort, and panel_seat_thinks is the panel
#: clause of agent_identity.budget_permitted_without_effort (the stream A
#: helper permit_panel_budget is gone, folded into that one predicate).
PANEL_ANSWER_ROUTING_NAMES = {
    'ANSWER_ONLY_RECORD_KEY', 'OWN_ANSWER_HEADING', 'OWN_REASONING_HEADING',
    'PANEL_ANSWER_ROUTING_CAPABILITY', 'PANEL_ANSWER_ROUTING_UNSUPPORTED_CODE',
    'PANEL_ROUTING_VERBS', 'PASSED_STATUSES', 'PanelUnreadable', 'REASONING_COMPLETE',
    'REASONING_CUT_OFF', 'REASONING_NOT_FOUND', 'REASONING_UNCLEAR',
    'REASONING_UNFINISHED', 'RULE_REASONING_PASSED_ON',
    'RULE_SEAT_WOULD_NOT_THINK', 'RULE_TEMPERATURE_IGNORED', 'TurnOutcome',
    'TurnText', 'WITHHELD_STATUSES', '_include_own_reasoning_from',
    '_record_text', '_routed_text', '_seat_temperature_from',
    '_template_thinks', 'outdated_panel_engine_reason', 'own_reasoning_text',
    'panel_rule_changes', 'panel_seat_temperatures', 'panel_seat_thinks',
    'play_turn', 'prompt_opens_reasoning',
    'remote_has_panel_answer_routing',
    'resume_rule_problem', 'rule_change_clauses', 'run_resume_rule_problem',
    'seat_reasoning', 'seat_temperature', 'seat_temperature_advisories',
    'seat_temperature_problem', 'seat_template',
    'study_seat_temperature_advisories', 'submission_panel_rule_changes',
    'turn_outcome', 'turn_text'}


def _take(body, source):
    """Remove from ``body`` exactly the statements ``source`` parses to,
    asserting they stand there, contiguous and in order; return where."""
    wanted = [ast.dump(s) for s in ast.parse(source).body]
    dumps = [ast.dump(s) for s in body]
    at = next(i for i in range(len(dumps)) if dumps[i:i + len(wanted)] == wanted)
    del body[at:at + len(wanted)]
    return at


def _drop_last_arg(function, name, annotation):
    """Remove ``function``'s last parameter, asserting its name, annotation,
    and ``None`` default."""
    arg = function.args.args[-1]
    assert arg.arg == name and ast.dump(arg.annotation) == ast.dump(
        ast.parse(annotation, mode='eval').body)
    assert ast.dump(function.args.defaults[-1]) == ast.dump(ast.Constant(None))
    function.args.args.pop()
    function.args.defaults.pop()


def admit_panel_answer_routing(tree):
    """Remove exactly the reviewed panel answer-routing change (public issue
    #15; FR6 stream A; NEEDS MAINTAINER APPROVAL) from the current
    multi_agent tree, asserting each added or changed statement's form, so
    every other body is still compared."""
    top = {getattr(n, 'name', None) or getattr(getattr(n, 'targets', [None])[0], 'id', None): n
           for n in tree.body}
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.level == 1
               and n.module is None and any(a.name == 'prompt_render' for a in n.names)]
    assert len(imports) == 1 and [a.name for a in imports[0].names] == [
        'model_variant', 'paths', 'prompt_render', 'reasoning_split',
        'truncation_gate', 'turn_endpoint', 'voice_lint']
    imports[0].names = [a for a in imports[0].names
                        if a.name not in ('prompt_render', 'reasoning_split')]
    found = [n for n in tree.body if (getattr(n, 'name', None) in PANEL_ANSWER_ROUTING_NAMES)
             or (isinstance(n, ast.Assign) and len(n.targets) == 1
                 and getattr(n.targets[0], 'id', None) in PANEL_ANSWER_ROUTING_NAMES)]
    found_names = sorted(getattr(n, 'name', None) or n.targets[0].id for n in found)
    assert found_names == sorted(PANEL_ANSWER_ROUTING_NAMES), (
        set(found_names) ^ PANEL_ANSWER_ROUTING_NAMES)
    tree.body = [n for n in tree.body if n not in found]

    agent = top['Agent']
    _take(agent.body, 'temperature: float | None = None')
    scenario = top['Scenario']
    _take(scenario.body, 'include_own_reasoning: bool = False')
    from_dict = next(n for n in scenario.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'from_dict')
    call = from_dict.body[-1].value
    keyword = next(k for k in call.keywords if k.arg == 'include_own_reasoning')
    assert ast.dump(keyword.value) == ast.dump(
        ast.parse('_include_own_reasoning_from(d)', mode='eval').body)
    call.keywords.remove(keyword)
    agents = next(k for k in call.keywords if k.arg == 'agents').value.elt
    keyword = next(k for k in agents.keywords if k.arg == 'temperature')
    assert ast.dump(keyword.value) == ast.dump(
        ast.parse('_seat_temperature_from(a)', mode='eval').body)
    agents.keywords.remove(keyword)

    _take(top['validate'].body, '''for agent in scenario.agents:
    problem = seat_temperature_problem(agent)
    if problem is not None:
        raise ScenarioError(problem)''')

    document = top['_scenario_to_dict'].body[-1].value
    def drop_spread(dictionary, source):
        wanted = ast.dump(ast.parse(source, mode='eval').body)
        at = next(i for i, (k, v) in enumerate(zip(dictionary.keys, dictionary.values))
                  if k is None and ast.dump(v) == wanted)
        del dictionary.keys[at], dictionary.values[at]
    drop_spread(document, '{"includeOwnReasoning": True} if s.include_own_reasoning else {}')
    seats = document.values[document.keys.index(next(
        k for k in document.keys if isinstance(k, ast.Constant) and k.value == 'agents'))]
    drop_spread(seats.elt, '{"temperature": a.temperature} if a.temperature is not None else {}')

    render = top['_render_prompt']
    _drop_last_arg(render, 'reasoning_by_label', 'dict[str, str] | None')
    contract_call = render.body[1].body[0].value
    assert ast.dump(contract_call.args[-1]) == ast.dump(
        ast.parse('reasoning_by_label or {}', mode='eval').body)
    contract_call.args.pop()

    contract = top['_render_contract_prompt']
    _drop_last_arg(contract, 'reasoning_by_label', 'dict[str, str] | None')
    own = next(n for n in ast.walk(contract) if isinstance(n, ast.If) and ast.dump(n.test)
               == ast.dump(ast.parse('producer.speaker_agent_id == turn.speaker_agent_id',
                                     mode='eval').body))
    assert _take(own.body, '''reasoning = (reasoning_by_label or {}).get(label)
if (scenario.include_own_reasoning
        and reasoning_split.strip_space(reasoning or "")):
    output = own_reasoning_text(reasoning, output)''') == 0

    settings = top['_runtime_settings']
    _drop_last_arg(settings, 'root', 'str | None')
    resolve = next(n for n in ast.walk(settings) if isinstance(n, ast.Call)
                   and ast.dump(n.func) == ast.dump(ast.parse('paths.resolve_artifact', mode='eval').body))
    assert ast.dump(resolve.args[-1]) == ast.dump(ast.parse('root', mode='eval').body)
    resolve.args.pop()

    loop = next(n for n in top['_transcript'].body if isinstance(n, ast.For))
    at = _take(loop.body, '''lines += [f"## Turn {r.get('turnIndex', index + 1)}: {title} — {speaker}",
          ""]
status = r.get("reasoningStatus")
if status is not None and status not in PASSED_STATUSES:
    lines += [f"_Nothing from this turn was passed on (reasoning "
              f"status {r.get('reasoningStatus')}). What it wrote:_",
              "", r.get("reasoning") or r.get("withheldText") or "", ""]
elif r.get("reasoning"):
    lines += ["_Reasoning (not passed on to the other seats):_", "",
              r.get("reasoning") or "", "", "_Answer:_", ""]
lines += [r.get("output", ""), ""]''')
    loop.body[at:at] = ast.parse('''lines += [f"## Turn {r.get('turnIndex', index + 1)}: {title} — {speaker}",
          "", r.get("output", ""), ""]''').body


for module in ('model_variant', 'multi_agent'):
    path = f'Server/steerlab_server/experiment/{module}.py'
    before = subprocess.check_output(['git','show',f'{BASE}:{path}'],cwd=ROOT,text=True)
    after = (ROOT/path).read_text()
    if module in ('model_variant', 'multi_agent'):
        # The historical mechanical migration remains proven at its landed tip.
        # P3/P5 intentionally extend run_scenario and the agent schema; test its behavior separately,
        # while retaining the current lazy wrapper and every other body here.
        landed = subprocess.check_output(['git', 'show', f'b1190f7:{path}'], cwd=ROOT, text=True)
        audit(module, before, landed)
        expected = landed
        if module == 'model_variant':
            expected = expected.replace('root: str | None = None) -> list[CellInjection]:', 'root: str | None = None, allow_policies: bool = False) -> list[CellInjection]:', 1)
            expected = expected.replace('    # `resolve_artifact`, not bare', "    if variant.intervention_policies and not allow_policies:\n        raise ValueError('This execution path does not run intervention policies. Use a sampled-response agent study or Python Playground; direct scoring and battery qualification are not yet policy-aware.')\n    # `resolve_artifact`, not bare", 1)
            # Ablation layer lists (approved by the maintainer 2026-10-19): a
            # listed layer the vector does not have refuses instead of being
            # skipped, by one rule and one sentence shared with the Swift engine
            # (ablation_layers.covered; behaviour in test_ablation_layer_subsets.py).
            for before_text, after_text in (
                ('from . import neutral, paths',
                 'from . import ablation_layers, neutral, paths'),
                ('layer_range = ([l for l in declared if 0 <= l < vectors.layer_count]\n'
                 '                           if declared else range(vectors.layer_count))',
                 'layer_range = ablation_layers.covered(\n'
                 '                variant.name, inj.get("concept"), declared, vectors.layer_count)'),
            ):
                assert expected.count(before_text) == 1, f'approved change no longer applies: {before_text!r}'
                expected = expected.replace(before_text, after_text, 1)
        else:
            expected = expected.replace('model_variant.variant_injections(variant)', "model_variant.variant_injections(variant, **({'allow_policies': True} if variant.intervention_policies else {}))")
        old_tree, current_tree = ast.parse(expected), ast.parse(after)
        if module == 'multi_agent':
            admit_panel_identity(current_tree)
            admit_panel_answer_routing(current_tree)
        for tree in (old_tree, current_tree):
            tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == 'run_scenario') and not (module == 'model_variant' and isinstance(n, ast.ClassDef) and n.name == 'ModelVariant')]
        assert ast.dump(old_tree) == ast.dump(current_tree), 'Unreviewed changes outside the intentional agent schema, scenario executor, and explicit policy admission changes'
        after = landed
    audit(module, before, after)
    try:
        audit(module, before, after + '\nUNAUTHORIZED_CHANGE = True\n')
    except AssertionError:
        pass
    else:
        raise AssertionError('Negative control was not detected')
print('Historical lazy-import migration proven; current bodies outside the intentional agent schema, scenario executor, and explicit policy admission changes unchanged; negative controls rejected.')
