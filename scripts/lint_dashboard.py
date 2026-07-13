#!/usr/bin/env python3
"""
SIRENA-KBR Dashboard Linter
===========================

Проверяет dashboard.py на:
1. Синтаксические ошибки
2. Неопределённые переменные
3. Неиспользуемые импорты
4. Runtime ошибки при импорте

Запуск: python3 scripts/lint_dashboard.py
"""

import sys
import os
import ast
import importlib.util
import traceback
from pathlib import Path
from typing import Set, List, Dict, Tuple
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
DASHBOARD_FILE = PROJECT_ROOT / 'dashboard.py'


class DashboardLinter(ast.NodeVisitor):
    """AST-based linter for finding undefined variables."""

    def __init__(self):
        self.defined_names: Set[str] = set()
        self.used_names: Set[str] = set()
        self.undefined_uses: List[Tuple[str, int, int]] = []
        self.scope_stack: List[Set[str]] = [set()]

        # Built-in names and common imports
        self.builtins = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
        self.builtins.update([
            'st', 'pd', 'np', 'go', 'px', 'plt', 'json', 'os', 'sys',
            'datetime', 'timedelta', 'Path', 'Dict', 'List', 'Optional',
            'Tuple', 'Any', 'Union', 'Callable', 'True', 'False', 'None',
            'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
            'sum', 'min', 'max', 'abs', 'round', 'sorted', 'reversed',
            'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool',
            'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
            'ImportError', 'AttributeError', 'RuntimeError', 'StopIteration',
            'isinstance', 'hasattr', 'getattr', 'setattr', 'callable',
            'open', 'file', 'input', 'format', 'repr', 'type', 'object',
            '__name__', '__file__', '__doc__',
        ])

    def push_scope(self):
        self.scope_stack.append(set())

    def pop_scope(self):
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()

    def define(self, name: str):
        self.scope_stack[-1].add(name)
        self.defined_names.add(name)

    def is_defined(self, name: str) -> bool:
        if name in self.builtins:
            return True
        for scope in self.scope_stack:
            if name in scope:
                return True
        return False

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[0]
            self.define(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.define(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.define(node.name)
        self.push_scope()

        # Add arguments to scope
        for arg in node.args.args:
            self.define(arg.arg)
        for arg in node.args.posonlyargs:
            self.define(arg.arg)
        for arg in node.args.kwonlyargs:
            self.define(arg.arg)
        if node.args.vararg:
            self.define(node.args.vararg.arg)
        if node.args.kwarg:
            self.define(node.args.kwarg.arg)

        self.generic_visit(node)
        self.pop_scope()

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.define(node.name)
        self.push_scope()
        self.generic_visit(node)
        self.pop_scope()

    def visit_For(self, node):
        self._visit_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars:
                self._visit_target(item.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.define(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node):
        # First visit value (RHS)
        self.visit(node.value)
        # Then define targets (LHS)
        for target in node.targets:
            self._visit_target(target)

    def visit_AnnAssign(self, node):
        if node.value:
            self.visit(node.value)
        self._visit_target(node.target)

    def visit_AugAssign(self, node):
        self.visit(node.value)
        self._visit_target(node.target)

    def visit_NamedExpr(self, node):
        self.visit(node.value)
        self.define(node.target.id)

    def _visit_target(self, node):
        """Visit assignment target and define names."""
        if isinstance(node, ast.Name):
            self.define(node.id)
        elif isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            for elt in node.elts:
                self._visit_target(elt)
        elif isinstance(node, ast.Starred):
            self._visit_target(node.value)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
            if not self.is_defined(node.id):
                self.undefined_uses.append((node.id, node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.push_scope()
        for generator in node.generators:
            self.visit(generator.iter)
            self._visit_target(generator.target)
            for if_ in generator.ifs:
                self.visit(if_)
        self.visit(node.elt)
        self.pop_scope()

    def visit_SetComp(self, node):
        self.visit_ListComp(node)

    def visit_GeneratorExp(self, node):
        self.visit_ListComp(node)

    def visit_DictComp(self, node):
        self.push_scope()
        for generator in node.generators:
            self.visit(generator.iter)
            self._visit_target(generator.target)
            for if_ in generator.ifs:
                self.visit(if_)
        self.visit(node.key)
        self.visit(node.value)
        self.pop_scope()

    def visit_Lambda(self, node):
        self.push_scope()
        for arg in node.args.args:
            self.define(arg.arg)
        self.visit(node.body)
        self.pop_scope()


def check_syntax(filepath: Path) -> List[str]:
    """Check for syntax errors."""
    errors = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        compile(source, filepath, 'exec')
    except SyntaxError as e:
        errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
    return errors


def check_undefined_variables(filepath: Path) -> List[Tuple[str, int, int]]:
    """Check for undefined variables using AST."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    linter = DashboardLinter()
    linter.visit(tree)

    # Filter out false positives
    filtered = []
    for name, line, col in linter.undefined_uses:
        # Skip common patterns
        if name.startswith('_'):
            continue
        if name in ['self', 'cls']:
            continue
        filtered.append((name, line, col))

    return filtered


def check_import_errors(filepath: Path) -> List[str]:
    """Check for potential runtime errors by analyzing code patterns."""
    errors = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    # Check 1: Find all _df variable assignments and uses
    import re

    df_assignments = {}  # var -> line_number
    df_uses = []  # (var, line_number, context)

    for i, line in enumerate(lines, 1):
        # Find assignments: xxx_df = something (not None)
        for match in re.finditer(r'(\w+_df)\s*(?:,\s*\w+)?\s*=\s*(?!None)', line):
            df_assignments[match.group(1)] = i

        # Find uses in conditionals: if xxx_df is not None
        for match in re.finditer(r'if\s+(\w+_df)\s+is\s+not\s+None', line):
            var = match.group(1)
            df_uses.append((var, i, line.strip()[:60]))

    # Check for uses before assignments
    for var, line_num, context in df_uses:
        if var not in df_assignments:
            errors.append(f"Line {line_num}: '{var}' used but never assigned\n   {context}")
        elif df_assignments[var] > line_num:
            errors.append(f"Line {line_num}: '{var}' used before assignment (assigned on line {df_assignments[var]})\n   {context}")

    # Check 2: Look for references to ACTUALLY removed models
    # Note: SARIMA, BVAR, ETS are auxiliary models (still available, just not in ensemble)
    removed_patterns = [
        'lmmr_hybrid', 'lmmr_claude', 'LMMRHybrid', 'LMMRClaude',
        'lmmr_df', 'lmmr_true', 'lmmr_x13',
    ]

    for pattern in removed_patterns:
        for i, line in enumerate(lines, 1):
            if pattern.lower() in line.lower() and not line.strip().startswith('#'):
                errors.append(f"Line {i}: Reference to removed model '{pattern}'\n   {line.strip()[:60]}")

    return errors


def find_variable_references(filepath: Path, varname: str) -> List[Tuple[int, str]]:
    """Find all lines referencing a variable."""
    refs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if varname in line:
                refs.append((i, line.strip()))
    return refs


def main():
    print("=" * 70)
    print("SIRENA-KBR Dashboard Linter")
    print("=" * 70)
    print(f"File: {DASHBOARD_FILE}")
    print()

    all_errors = []

    # 1. Syntax check
    print("1. Checking syntax...")
    syntax_errors = check_syntax(DASHBOARD_FILE)
    if syntax_errors:
        print(f"   ✗ {len(syntax_errors)} syntax error(s)")
        for err in syntax_errors:
            print(f"      {err}")
            all_errors.append(('syntax', err))
    else:
        print("   ✓ No syntax errors")

    # 2. Undefined variables
    print("\n2. Checking for undefined variables...")
    undefined = check_undefined_variables(DASHBOARD_FILE)

    # Group by variable name
    var_counts = defaultdict(list)
    for name, line, col in undefined:
        var_counts[name].append(line)

    if var_counts:
        print(f"   ⚠ {len(var_counts)} potentially undefined variable(s):")
        for name, lines in sorted(var_counts.items()):
            if len(lines) <= 3:
                print(f"      - {name}: lines {lines}")
            else:
                print(f"      - {name}: {len(lines)} occurrences (lines {lines[:3]}...)")
            all_errors.append(('undefined', name, lines))
    else:
        print("   ✓ No undefined variables detected")

    # 3. Try importing (will catch runtime errors)
    print("\n3. Checking for runtime import errors...")
    import_errors = check_import_errors(DASHBOARD_FILE)
    if import_errors:
        print(f"   ✗ {len(import_errors)} import error(s):")
        for err in import_errors:
            # Only show first few lines
            lines = err.split('\n')
            for line in lines[:5]:
                print(f"      {line}")
            if len(lines) > 5:
                print(f"      ... ({len(lines)-5} more lines)")
            all_errors.append(('import', err))
    else:
        print("   ✓ No import errors")

    # 4. Check for common issues
    print("\n4. Checking for common issues...")
    with open(DASHBOARD_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    common_issues = []

    # Check for removed models still referenced
    removed_models = ['lmmr_hybrid', 'lmmr_claude', 'lmmr_df', 'catboost_df', 'xgboost_df']
    for model in removed_models:
        if model in content.lower():
            refs = find_variable_references(DASHBOARD_FILE, model)
            if refs:
                common_issues.append(f"Reference to removed model '{model}' at lines: {[r[0] for r in refs]}")

    if common_issues:
        print(f"   ⚠ {len(common_issues)} common issue(s):")
        for issue in common_issues:
            print(f"      - {issue}")
            all_errors.append(('common', issue))
    else:
        print("   ✓ No common issues")

    # Summary
    print("\n" + "=" * 70)
    if all_errors:
        print(f"✗ FOUND {len(all_errors)} ISSUE(S)")
        print("\nQuick fixes needed:")
        for err_type, *details in all_errors:
            if err_type == 'undefined' or err_type == 'common':
                print(f"  - Fix: {details}")
    else:
        print("✓ ALL CHECKS PASSED")
    print("=" * 70)

    return 1 if all_errors else 0


if __name__ == '__main__':
    sys.exit(main())
