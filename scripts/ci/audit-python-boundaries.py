#!/usr/bin/env python3
"""Prove executable ASTs unchanged by the recorded public-name/import migration.

Usage: python scripts/ci/audit-python-boundaries.py [--candidate REV]
The default candidate is the working tree. Run against the mechanical commit,
not the subsequent intentional sweep-judgment behavior fix. Import destinations
and identifiers are canonicalized; executable statements and test assertions
must match. Module import inventories are checked by the boundary tests.
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(Path(__file__).with_name('python-boundary-renames.json').read_text())
REVERSE = {new: old for old, new in CONFIG['renames'].items()}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, stderr=subprocess.PIPE)


def source_module(path):
    return '.'.join(Path(path).relative_to('Server').with_suffix('').parts)


def import_source(node, module):
    if not node.level:
        return node.module or ''
    return '.'.join(module.split('.')[:-node.level] + ([node.module] if node.module else []))


def imports(tree, module):
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out[alias.asname or alias.name] = import_source(node, module) + '.' + alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
    return out


class Canonical(ast.NodeTransformer):
    def __init__(self, module, tree, old_facade):
        self.module = module
        self.old_facade = old_facade
        self.scopes = []
        self.all_imports = imports(tree, module)
        self.enter(tree.body, module_scope=True)

    def qualified(self, value):
        # Resolve legacy facade re-exports to their actual provider.
        prefix = 'steerlab_server.experiment.tasks.'
        if value.startswith(prefix):
            first, *rest = value[len(prefix):].split('.')
            if first in self.old_facade and first not in CONFIG['facade_exports']:
                value = self.old_facade[first] + ('.' + '.'.join(rest) if rest else '')
        # Resolve indirect policy imports (notably the legacy study store).
        for old, target in CONFIG.get('reexports', {}).items():
            candidates = [old, old.rsplit('.', 1)[0] + '.' + old.rsplit('.', 1)[1].lstrip('_')]
            for alias in candidates:
                if value == alias or value.startswith(alias + '.'):
                    value = target + value[len(alias):]
                    break
        # Attribute chains may continue past a renamed function or class.
        for new, old in sorted(REVERSE.items(), key=lambda item: -len(item[0])):
            if value == new or value.startswith(new + '.'):
                return old + value[len(new):]
        return value

    def enter(self, body, args=None, module_scope=False):
        scope = {}
        def local(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope[node.name] = self.qualified(self.module + '.' + node.name) if module_scope else None
                return
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                scope.update(imports(node, self.module))
                return
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                scope[node.id] = self.qualified(self.module + '.' + node.id) if module_scope else None
            if isinstance(node, ast.Global):
                for name in node.names:
                    scope[name] = self.scopes[0].get(name)
            for child in ast.iter_child_nodes(node):
                local(child)
        for node in body:
            local(node)
        if args:
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
                if arg:
                    scope[arg.arg] = None
        self.scopes.append(scope)

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_Expr(self, node):
        # Documentation is deliberately updated with the new owning symbols.
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Name(self, node):
        target = self.lookup(node.id)
        if target:
            node.id = '@' + self.qualified(target)
        return node

    def visit_Attribute(self, node):
        node.value = self.visit(node.value)
        if isinstance(node.value, ast.Name) and node.value.id.startswith('@'):
            return ast.Name(id='@' + self.qualified(node.value.id[1:] + '.' + node.attr), ctx=node.ctx)
        return node

    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            return node
        value = node.value
        for old, new in CONFIG.get('source_inspection_renames', {}).items():
            if value == new:
                value = old
        # Qualified patch paths and postponed type annotations.
        for name, target in sorted(self.all_imports.items(), key=lambda item: -len(item[0])):
            value = re.sub(r'(?<![\w.])' + re.escape(name) + r'(?=\.)', target, value)
        value = re.sub(r'steerlab_server(?:\.[A-Za-z_]\w*)+', lambda m: self.qualified(m[0]), value)
        node.value = value
        return node

    def visit_Call(self, node):
        # getattr/patch.object/monkeypatch: the literal is an attribute spelling,
        # not experiment data. Canonicalize it only with an identified module.
        if len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and isinstance(node.args[1], ast.Constant):
            owner = self.lookup(node.args[0].id)
            attr = node.args[1].value
            if owner and isinstance(attr, str):
                before = owner + '.' + attr
                after = self.qualified(before)
                if before != after:
                    owner, attr = after.rsplit('.', 1)
                    node.args[0] = ast.Name(id='@' + owner, ctx=ast.Load())
                    node.args[1] = ast.Constant(value=attr)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Defaults/decorators/annotations are evaluated in the outer scope.
        original_name = node.name
        if len(self.scopes) == 1:
            node.name = self.qualified(self.module + '.' + original_name).rsplit('.', 1)[1]
        node.decorator_list = [self.visit(x) for x in node.decorator_list]
        node.args = self.visit(node.args)
        if node.returns:
            node.returns = self.visit(node.returns)
        self.enter(node.body, node.args)
        node.body = [x for item in node.body if (x := self.visit(item)) is not None]
        self.scopes.pop()
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        node.bases = [self.visit(x) for x in node.bases]
        self.enter(node.body)
        node.body = [x for item in node.body if (x := self.visit(item)) is not None]
        self.scopes.pop()
        return node


def compare(candidate=None):
    baseline = CONFIG['baseline']
    changed = git('diff', '--name-only', baseline, *([candidate] if candidate else []), '--', 'Server').splitlines()
    facade = ast.parse(git('show', baseline + ':Server/steerlab_server/experiment/tasks.py'))
    facade_bindings = imports(facade, 'steerlab_server.experiment.tasks')
    count = 0
    failures = []
    for path in changed:
        if not path.endswith('.py'):
            continue
        try:
            old = git('show', baseline + ':' + path)
        except subprocess.CalledProcessError:
            continue  # New boundary tests have no pre-migration body to compare.
        new = git('show', candidate + ':' + path) if candidate else (ROOT / path).read_text()
        module = source_module(path)
        trees = []
        for source in (old, new):
            tree = ast.parse(source)
            result = Canonical(module, tree, facade_bindings).visit(copy.deepcopy(tree))
            if path.endswith('/experiment/tasks.py'):
                # The facade's import surface intentionally shrinks; its sole
                # executable definition, pipeline(), must remain equivalent.
                result.body = [n for n in result.body if isinstance(n, ast.FunctionDef)]
            trees.append(result)
        if ast.dump(trees[0], include_attributes=False) != ast.dump(trees[1], include_attributes=False):
            failures.append(path)
            debug = Path('/private/tmp/python-boundary-audit')
            debug.mkdir(exist_ok=True)
            for label, tree in zip(('before', 'after'), trees):
                (debug / (Path(path).name + '.' + label)).write_text(ast.dump(tree, indent=2))
        else:
            count += sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(trees[0]))
    if failures:
        raise SystemExit('AST mismatch (inspect /private/tmp/python-boundary-audit):\n' + '\n'.join(failures))
    print(f'PASS: executable ASTs preserved across {len(changed)} changed paths; {count} function bodies verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate')
    compare(parser.parse_args().candidate)
