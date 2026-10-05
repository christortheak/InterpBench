#!/usr/bin/env python3
"""Prove the extracted task record loop is unchanged from the reviewed base.

Only the stream expression changes: an open UTF-8 text file becomes StringIO
with the same universal-newline policy. File/hash admission before the loop is
checked separately. Run from any directory in a checkout retaining the base.

Declared change to the admission gates (wave 2, reviewed and approved
2026-10-05): a study naming no prompt file meets a typed refusal with a repair
instead of a bare RuntimeError, and every refusal's `repair=` text is spelled
for the client that shows it. Conditions, refusal codes and messages stay exact.
"""
import ast
import copy
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PATH = "Server/steerlab_server/experiment/task_inputs.py"
BASE = "ef3dec8"
OLD_MISSING_PATH = 'raise RuntimeError("no task prompts file specified")'
NEW_MISSING_PATH = 'raise no_task_prompts_refusal(manifest)'


def gates(nodes, *, declared=False):
    """The admission gates as ASTs; with ``declared``, the declared change
    reverted: the typed missing-path refusal back to the bare error (exactly
    once), and every ``repair=`` value blanked on both sides."""
    dumped = []
    replaced = 0
    for node in nodes:
        node = copy.deepcopy(node)
        if declared:
            for parent in ast.walk(node):
                for field, value in ast.iter_fields(parent):
                    if isinstance(value, list):
                        for index, child in enumerate(value):
                            if isinstance(child, ast.Raise) and ast.dump(child) == ast.dump(ast.parse(NEW_MISSING_PATH).body[0]):
                                value[index] = ast.parse(OLD_MISSING_PATH).body[0]
                                replaced += 1
        for call in ast.walk(node):
            if isinstance(call, ast.Call):
                for keyword in call.keywords:
                    if keyword.arg == "repair":
                        keyword.value = ast.Constant("<repair>")
        dumped.append(ast.dump(node))
    assert not declared or replaced == 1, "Declared missing-path refusal missing or duplicated"
    return dumped


def check_refusal_helper(tree):
    """The typed refusal is a MISSING_PREREQUISITE refusal and nothing else."""
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "no_task_prompts_refusal")
    final = helper.body[-1]
    assert isinstance(final, ast.Return) and isinstance(final.value, ast.Call)
    assert ast.unparse(final.value.func) == "lifecycle_gates.refusing"
    assert ast.unparse(final.value.args[0]) == "lifecycle_gates.MISSING_PREREQUISITE"
    assert not any(isinstance(n, (ast.Raise, ast.With, ast.Global, ast.Nonlocal)) for n in ast.walk(helper)), \
        "The refusal helper must only build its refusal"


def audit(before, after):
    old = ast.parse(before)
    new = ast.parse(after)
    old_load = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "load_prompts")
    new_load = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "load_prompts")
    parser = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "parse_prompts")
    old_loop = next(n for n in ast.walk(old_load) if isinstance(n, ast.For))
    new_loop = next(n for n in ast.walk(parser) if isinstance(n, ast.For))
    assert ast.dump(old_loop.iter) == ast.dump(ast.parse("enumerate(handle)", mode="eval").body)
    assert ast.dump(new_loop.iter) == ast.dump(ast.parse("enumerate(io.StringIO(text, newline=None))", mode="eval").body)
    normalized = copy.deepcopy(new_loop)
    normalized.iter = old_loop.iter
    assert ast.dump(old_loop) == ast.dump(normalized), "Task record admission body changed"
    # Hash/path/frozen-input gates before the moved prompt accumulator stay exact.
    start = next(i for i, n in enumerate(old_load.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "prompts" for t in n.targets))
    assert gates(old_load.body[:start]) == gates(new_load.body[:-1], declared=True), "Admission gates changed"
    check_refusal_helper(new)
    assert ast.dump(new_load.body[-1]) == ast.dump(ast.parse(
        'with open(path, encoding="utf-8") as handle:\n    return parse_prompts(handle.read())').body[0])
    assert [ast.dump(n) for n in old_load.body[start:start+2]] == [ast.dump(n) for n in parser.body[1:3]]
    assert ast.dump(old_load.body[-1]) == ast.dump(parser.body[-1])


if __name__ == "__main__":
    before = subprocess.check_output(["git", "show", f"{BASE}:{PATH}"], cwd=ROOT, text=True)
    after = (ROOT / PATH).read_text()
    audit(before, after)
    # Negative controls: a changed record identity rule, a changed refusal
    # code, a changed gate condition, and a removed typed refusal are refused.
    for label, old, new in (
            ("record identity", 'f"prompt-{len(prompts) + 1}"', 'f"prompt-{len(prompts)}"'),
            ("refusal code", "lifecycle_gates.PIN_DRIFT,", "lifecycle_gates.MISSING_PREREQUISITE,"),
            ("gate condition", "live_hash != manifest.task_prompts_hash", "live_hash == manifest.task_prompts_hash"),
            ("typed refusal removed", NEW_MISSING_PATH, "return []"),
            ("refusal helper code", "        lifecycle_gates.MISSING_PREREQUISITE,\n        f\"study '{name}' pins", "        lifecycle_gates.PIN_DRIFT,\n        f\"study '{name}' pins")):
        assert after.count(old) >= 1, "Mutation control did not apply: " + label
        try:
            audit(before, after.replace(old, new, 1))
        except AssertionError:
            continue
        raise AssertionError("Negative control failed: " + label)
    # Positive control: repair text is free to change, and a message is not.
    assert after.count('"bytes ; then re-run this verb') == 1, "Positive control did not apply"
    audit(before, after.replace('"bytes ; then re-run this verb', '"bytes, then re-run this verb', 1))
    try:
        audit(before, after.replace('f"pinned hash (have', 'f"pinned hash (found', 1))
    except AssertionError:
        pass
    else:
        raise AssertionError("Negative control failed: refusal message")
    print("Task prompt parser AST matches base; file gates unchanged apart from the declared refusal and repair text; negative controls rejected.")
