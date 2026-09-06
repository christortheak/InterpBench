#!/usr/bin/env python3
"""Prove the extracted task record loop is unchanged from the reviewed base.

Only the stream expression changes: an open UTF-8 text file becomes StringIO
with the same universal-newline policy. File/hash admission before the loop is
checked separately. Run from any directory in a checkout retaining the base.
"""
import ast
import copy
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PATH = "Server/steerlab_server/experiment/task_inputs.py"
BASE = "ef3dec8"


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
    assert [ast.dump(n) for n in old_load.body[:start]] == [ast.dump(n) for n in new_load.body[:-1]]
    assert ast.dump(new_load.body[-1]) == ast.dump(ast.parse(
        'with open(path, encoding="utf-8") as handle:\n    return parse_prompts(handle.read())').body[0])
    assert [ast.dump(n) for n in old_load.body[start:start+2]] == [ast.dump(n) for n in parser.body[1:3]]
    assert ast.dump(old_load.body[-1]) == ast.dump(parser.body[-1])


if __name__ == "__main__":
    before = subprocess.check_output(["git", "show", f"{BASE}:{PATH}"], cwd=ROOT, text=True)
    after = (ROOT / PATH).read_text()
    audit(before, after)
    # Negative control: the audit must detect a changed record identity rule.
    try:
        audit(before, after.replace('f"prompt-{len(prompts) + 1}"', 'f"prompt-{len(prompts)}"'))
    except AssertionError:
        pass
    else:
        raise AssertionError("Negative control failed")
    print("Task prompt parser AST matches base; file gates unchanged; negative control rejected.")
