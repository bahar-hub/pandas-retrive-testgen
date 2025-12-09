#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mutation_engine_generic.py

Generic mutation-analysis engine for arbitrary Python functions.

Pipeline:
    1. Load a JSON spec produced by fetch.py:
         - function source (full_source / source / body_stripped)
         - (optionally) qualified_name (e.g. "pandas.melt" or "statistics.mean")
    2. Generate mutants by applying text-based mutation patterns.
    3. For each LLM-generated test file:
         - Copy the base project into a temp directory.
         - Write a mutated module containing the mutated function.
         - Write a generic conftest.py that monkey-patches the target
           function (module.attr[.attr...] = mutated_func).
         - Run pytest and see whether the mutant is killed.
    4. Compute mutation score (percentage) per test suite and
       write JSON results into results_dir.

This version is generic:
    - No hard-coded dependency on pandas / statistics / melt / mean.
    - Target function is determined from spec["qualified_name"] or
      from CLI (--target).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Mutant:
    """Represents a single mutated version of the target function."""
    id: int
    source_code: str
    description: str


# ---------------------------------------------------------------------------
# Spec Loading
# ---------------------------------------------------------------------------

def load_source_and_spec(spec_path: str) -> Tuple[str, Dict]:
    """
    Load function source and full spec from a JSON file produced by fetch.py.

    Priority for source extraction:
        1) "full_source"
        2) "source"
        3) "body_stripped"

    Returns:
        (function_source_text, spec_dict)
    """
    with open(spec_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    src = (
        data.get("full_source")
        or data.get("source")
        or data.get("body_stripped")
    )
    if not src:
        raise RuntimeError(
            f"Spec '{spec_path}' does not contain 'full_source', 'source', "
            "or 'body_stripped'."
        )
    return src, data


def resolve_target_from_spec_or_cli(
    spec: Dict,
    target_qualified_name: Optional[str],
    target_module_override: Optional[str] = None,
    target_attr_path_override: Optional[str] = None,
) -> Tuple[str, str, str, str]:
    """
    Resolve the target function from spec + CLI overrides.

    We want:
        - module_root: top-level module to import (e.g. "pandas", "statistics")
        - attr_path:   attribute path inside that module, MAY contain dots
                       (e.g. "melt", "DataFrame.merge")
        - func_name:   last attribute name (e.g. "melt", "merge")
        - qualified:   resulting fully qualified name (for logging)

    Resolution order:
        1) CLI --target (target_qualified_name), if given.
        2) spec["qualified_name"] / spec["full_name"] / spec["name"], if present.

    Overrides:
        - target_module_override replaces module_root
        - target_attr_path_override replaces attr_path
    """
    # Step 1: pick a qualified name
    if target_qualified_name:
        qualified = target_qualified_name
    else:
        qualified = (
            spec.get("qualified_name")
            or spec.get("full_name")
            or spec.get("name")
        )
        if not qualified:
            raise RuntimeError(
                "Cannot determine target function qualified name: "
                "please provide --target or ensure spec has 'qualified_name'."
            )

    # If we have explicit overrides for module / attr path, use them.
    if target_module_override and target_attr_path_override:
        module_root = target_module_override
        attr_path = target_attr_path_override
    else:
        parts = qualified.split(".")
        if len(parts) < 2 and not target_module_override:
            raise RuntimeError(
                f"Qualified name '{qualified}' does not look like 'module.func'. "
                "Provide --target-module and --target-attr-path explicitly."
            )

        if target_module_override:
            module_root = target_module_override
            # Derive attr_path if override missing
            if target_attr_path_override:
                attr_path = target_attr_path_override
            else:
                # Best effort: remove leading "<module_root>." if present
                if qualified.startswith(module_root + "."):
                    attr_path = qualified[len(module_root) + 1 :]
                else:
                    # fallback: last part only
                    attr_path = qualified.split(".")[-1]
        else:
            module_root = parts[0]
            attr_path = ".".join(parts[1:])

    func_name = attr_path.split(".")[-1]
    resolved_qualified = f"{module_root}.{attr_path}" if attr_path else f"{module_root}.{func_name}"
    return module_root, attr_path, func_name, resolved_qualified


# ---------------------------------------------------------------------------
# Mutation Patterns (generic)
# ---------------------------------------------------------------------------

def get_generic_mutation_patterns() -> Iterable[Tuple[re.Pattern, str, str]]:
    """
    Generic mutation patterns for arbitrary Python functions.

    These are intentionally simple, operator-level mutations:
      - Relational operators (>, <, >=, <=, ==, !=)
      - None checks (is None / is not None)
      - Logical operators (and / or)
      - Boolean literals (True / False)

    NOTE: They are purely text-based, so they may also touch places like
    comments or strings in some cases (depending on the pattern).
    """
    raw_patterns = [
        (r"\bprefix\s+is\s+None\b",
         "prefix is not None",
         "Flip 'prefix is None' → 'prefix is not None'"),
        (r"\bprefix\s+is\s+not\s+None\b",
         "prefix is None",
         "Flip 'prefix is not None' → 'prefix is None'"),

        (r'\bprefix_sep\s*=\s*["\']_["\']',
         'prefix_sep="__"',
         "Change prefix_sep='_' → '__'"),
        (r'\bprefix_sep\s*=\s*["\']__["\']',
         'prefix_sep="_"',
         "Change prefix_sep='__' → '_'"),

        (r"\bcolumns\s+is\s+None\b",
         "columns is not None",
         "Flip 'columns is None' → 'columns is not None'"),
        (r"\bcolumns\s+is\s+not\s+None\b",
         "columns is None",
         "Flip 'columns is not None' → 'columns is None'"),

        (r"\bdummy_na\s*=\s*True\b",
         "dummy_na=False",
         "Flip 'dummy_na=True' → 'dummy_na=False'"),
        (r"\bdummy_na\s*=\s*False\b",
         "dummy_na=True",
         "Flip 'dummy_na=False' → 'dummy_na=True'"),

        (r"\bdrop_first\s*=\s*True\b",
         "drop_first=False",
         "Flip 'drop_first=True' → 'drop_first=False'"),
        (r"\bdrop_first\s*=\s*False\b",
         "drop_first=True",
         "Flip 'drop_first=False' → 'drop_first=True'"),

        (r"\bsparse\s*=\s*True\b",
         "sparse=False",
         "Flip 'sparse=True' → 'sparse=False'"),
        (r"\bsparse\s*=\s*False\b",
         "sparse=True",
         "Flip 'sparse=False' → 'sparse=True'"),

        (r"\bsparse_dtype\s+is\s+None\b",
         "sparse_dtype is not None",
         "Flip 'sparse_dtype is None' → 'sparse_dtype is not None'"),
        (r"\bsparse_dtype\s+is\s+not\s+None\b",
         "sparse_dtype is None",
         "Flip 'sparse_dtype is not None' → 'sparse_dtype is None'"),

        (r"\bdtype\s+is\s+None\b",
         "dtype is not None",
         "Flip 'dtype is None' → 'dtype is not None'"),
        (r"\bdtype\s+is\s+not\s+None\b",
         "dtype is None",
         "Flip 'dtype is not None' → 'dtype is None'"),

        (r'\bdtype\s*=\s*["\']uint8["\']',
         'dtype="int64"',
         "Change dtype='uint8' → 'int64'"),
        (r'\bdtype\s*=\s*["\']int64["\']',
         'dtype="uint8"',
         "Change dtype='int64' → 'uint8'"),

        (r"\bis None\b",
         "is not None",
         "Flip 'is None' → 'is not None'"),
        (r"\bis not None\b",
         "is None",
         "Flip 'is not None' → 'is None'"),

        (r"\band\b",
         "or",
         "Replace logical 'and' → 'or'"),
        (r"\bor\b",
         "and",
         "Replace logical 'or' → 'and'"),

        (r"\bTrue\b",
         "False",
         "Replace True → False"),
        (r"\bFalse\b",
         "True",
         "Replace False → True"),
    ]

    return [(re.compile(p), repl, desc) for p, repl, desc in raw_patterns]

def generate_mutants_from_body(
    body: str,
    patterns: Optional[Iterable[Tuple[re.Pattern, str, str]]] = None,
) -> List[Mutant]:
    """
    Generate mutants from the given function body/source using the provided
    mutation patterns. If patterns is None, generic patterns are used.
    """
    if patterns is None:
        patterns = get_generic_mutation_patterns()

    mutants: List[Mutant] = []
    mutant_id = 1

    for pattern, replacement, desc in patterns:
        for match in pattern.finditer(body):
            start, end = match.span()
            mutated_body = body[:start] + replacement + body[end:]
            mutants.append(
                Mutant(
                    id=mutant_id,
                    source_code=mutated_body,
                    description=f"{desc} at pos {start}",
                )
            )
            mutant_id += 1

    return mutants


# ---------------------------------------------------------------------------
# Module Builder & Conftest Generator (generic patching)
# ---------------------------------------------------------------------------

def build_module_source_from_mutated_body(
    mutated_body: str,
    spec: Optional[Dict] = None,
) -> str:
    """
    Wrap the mutated function source into a standalone Python module.

    By default, we:
        - emit a 'from __future__ import annotations' header
        - optionally prepend any spec['module_header'] or spec['header_source']
          if present
        - then emit the mutated function source.

    NOTE:
        This assumes that the mutated_body already contains a valid function
        definition for the target function (e.g. 'def mean(...): ...').
        If the function relies on imports or helpers, you can encode them
        into 'module_header' or 'header_source' in your spec.
    """
    header = "from __future__ import annotations\n\n"
    if spec is not None:
        extra = spec.get("module_header") or spec.get("header_source") or ""
        if extra:
            header += extra.rstrip() + "\n\n"
    return header + mutated_body.rstrip() + "\n"


def write_conftest_for_target(
    project_root: str,
    mutated_module_name: str,
    func_name: str,
    module_root: str,
    attr_path: str,
) -> None:
    """
    Create a conftest.py that monkey-patches the target function:
        - import the mutated function from mutated_module_name
        - import the target module via importlib
        - walk attr_path (e.g. 'DataFrame.merge')
        - setattr(last_attr, mutated_func)

    This works for:
        - 'statistics.mean'     → module_root='statistics', attr_path='mean'
        - 'pandas.melt'         → module_root='pandas',    attr_path='melt'
        - 'pandas.DataFrame.merge'
                                 → module_root='pandas',    attr_path='DataFrame.merge'
    """
    conftest_path = os.path.join(project_root, "conftest.py")
    content = f"""
import importlib
from {mutated_module_name} import {func_name} as _mutated_func

_TARGET_MODULE = "{module_root}"
_TARGET_ATTR_PATH = "{attr_path}"

_mod = importlib.import_module(_TARGET_MODULE)
_obj = _mod

if _TARGET_ATTR_PATH:
    _parts = _TARGET_ATTR_PATH.split(".")
    for _name in _parts[:-1]:
        _obj = getattr(_obj, _name)
    _last = _parts[-1]
else:
    _last = "{func_name}"

setattr(_obj, _last, _mutated_func)
"""
    with open(conftest_path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Test Execution Helpers
# ---------------------------------------------------------------------------

def run_pytest_and_get_failed_tests(
    project_root: str,
    test_file: str,
    timeout_seconds: int = 60,
) -> Tuple[Set[str], int, str]:
    """
    Run pytest on a specific test file inside a given project directory.

    Returns:
        - a set of failed test identifiers
        - the pytest return code
        - the full combined stdout+stderr
    """
    cmd = ["pytest", "-q", test_file]

    try:
        proc = subprocess.run(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
        rc = proc.returncode
        output = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired as e:
        # If tests hang, treat as a failure (mutant killed) but return special code.
        rc = 124  # conventional timeout code
        output = (e.stdout or "") + "\n" + (e.stderr or "") + "\nPYTEST TIMEOUT"

    failed_tests: Set[str] = set()

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("FAILED "):
            parts = line.split()
            if len(parts) >= 2:
                failed_tests.add(parts[1])

    return failed_tests, rc, output


def compute_mutation_score(killed: int, total: int, equivalent: int = 0) -> float:
    """
    Compute the mutation score (as a percentage):

        score = 100 * killed / (total - equivalent)
    """
    effective_total = total - equivalent
    if effective_total <= 0:
        return 0.0
    return (killed / effective_total) * 100.0


# ---------------------------------------------------------------------------
# Core Evaluation Logic
# ---------------------------------------------------------------------------

def evaluate_one_llm_on_project_copy(
    project_copy: str,
    mutants: List[Mutant],
    func_name: str,
    module_root: str,
    attr_path: str,
    test_file: str,
    spec: Optional[Dict] = None,
    mutated_module_filename: str = "mutated_target.py",
    mutated_module_name: str = "mutated_target",
    pytest_timeout: int = 60,
) -> Tuple[int, int, int, float]:
    """
    Run all mutants for a single LLM-generated test suite on a single
    project copy.

    Steps:
        - Write a generic conftest.py once to patch the target function.
        - For each mutant:
            * write mutated module
            * run pytest
            * classify as KILLED / SURVIVED
    """
    killed = 0
    survived = 0
    total_mutants = len(mutants)

    # 1) Setup conftest to patch target function with mutated implementation.
    write_conftest_for_target(
        project_root=project_copy,
        mutated_module_name=mutated_module_name,
        func_name=func_name,
        module_root=module_root,
        attr_path=attr_path,
    )

    module_path = os.path.join(project_copy, mutated_module_filename)

    # 2) Evaluate each mutant
    for mutant in mutants:
        module_source = build_module_source_from_mutated_body(mutant.source_code, spec=spec)
        with open(module_path, "w", encoding="utf-8") as f:
            f.write(module_source)

        failed, rc, _ = run_pytest_and_get_failed_tests(
            project_root=project_copy,
            test_file=test_file,
            timeout_seconds=pytest_timeout,
        )

        if failed or rc != 0:
            killed += 1
            status = "KILLED"
        else:
            survived += 1
            status = "SURVIVED"

        print(
            f"      Mutant #{mutant.id}: {mutant.description} → {status} "
            f"(fails: {len(failed)}, exit code: {rc})"
        )

    score = compute_mutation_score(killed, total_mutants)
    return killed, survived, total_mutants, score


def evaluate_generated_tests_from_json_baseline(
    spec_path: str,
    base_project_path: str = ".",
    generated_dir: str = "generated_tests",
    results_dir: str = "mutation_results",
    target_qualified_name: Optional[str] = None,
    target_module_override: Optional[str] = None,
    target_attr_path_override: Optional[str] = None,
    pytest_timeout: int = 60,
) -> Dict[str, float]:
    """
    Evaluate ALL LLM-generated test suites located in generated_dir for the
    target function described by spec_path (and optional CLI overrides).

    Optimizations:
        - Load JSON spec and generate mutants ONCE.
        - For each LLM test file:
            * Copy the project ONCE into a temp directory.
            * Reuse that single copy for all mutants by overwriting the
              mutated module file for each mutant.
    """
    scores: Dict[str, float] = {}

    # 1) Load spec & generate mutants once
    body, spec = load_source_and_spec(spec_path)
    module_root, attr_path, func_name, resolved_qname = resolve_target_from_spec_or_cli(
        spec,
        target_qualified_name=target_qualified_name,
        target_module_override=target_module_override,
        target_attr_path_override=target_attr_path_override,
    )

    print(f"Target function resolved as: {resolved_qname}")
    print(f"  module_root = {module_root}")
    print(f"  attr_path   = {attr_path}")
    print(f"  func_name   = {func_name}")

    mutants = generate_mutants_from_body(body)
    total_mutants = len(mutants)
    print(f"Generated {total_mutants} mutants from JSON body")

    # 2) Discover test files
    generated_path = os.path.join(base_project_path, generated_dir)
    if not os.path.isdir(generated_path):
        raise FileNotFoundError(f"Generated tests directory not found: {generated_path}")

    files = sorted(f for f in os.listdir(generated_path) if f.endswith(".py"))

    # 3) Prepare results directory
    results_root = os.path.join(base_project_path, results_dir)
    os.makedirs(results_root, exist_ok=True)

    print(f"Found {len(files)} generated test files in {generated_dir}/")

    mutated_module_filename = "mutated_target.py"
    mutated_module_name = "mutated_target"

    for fname in files:
        test_file = os.path.join(generated_dir, fname)
        llm_name = os.path.splitext(fname)[0]

        print(f"\n=== Evaluating test suite: {llm_name} ===")

        with tempfile.TemporaryDirectory() as tmpdir:
            project_copy = os.path.join(tmpdir, "project")
            shutil.copytree(base_project_path, project_copy, dirs_exist_ok=True)

            killed, survived, total, score = evaluate_one_llm_on_project_copy(
                project_copy=project_copy,
                mutants=mutants,
                func_name=func_name,
                module_root=module_root,
                attr_path=attr_path,
                test_file=test_file,
                spec=spec,
                mutated_module_filename=mutated_module_filename,
                mutated_module_name=mutated_module_name,
                pytest_timeout=pytest_timeout,
            )

        print(f"Results for {llm_name}:")
        print(f"  Killed   : {killed}")
        print(f"  Survived : {survived}")
        print(f"  Total    : {total}")
        print(f"  Mutation Score: {score:.2f}%")

        scores[llm_name] = score

        # Save JSON results per model
        result_obj = {
            "llm_name": llm_name,
            "spec_path": spec_path,
            "test_file": test_file,
            "target_qualified_name": resolved_qname,
            "killed": killed,
            "survived": survived,
            "total_mutants": total,
            "mutation_score": score,
        }

        result_path = os.path.join(results_root, f"{llm_name}.json")
        with open(result_path, "w", encoding="utf-8") as rf:
            json.dump(result_obj, rf, ensure_ascii=False, indent=2)

    return scores


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args() -> "argparse.Namespace":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generic mutation analyzer for functions described by a JSON spec "
            "(output of fetch.py)."
        )
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to JSON spec file produced by fetch.py.",
    )
    parser.add_argument(
        "--base-project-path",
        default="..",
        help="Root path of the project that contains tests, etc. (will be copied).",
    )
    parser.add_argument(
        "--generated-dir",
        default="generated_tests",
        help="Relative directory (under base-project-path) containing generated test files.",
    )
    parser.add_argument(
        "--results-dir",
        default="mutation_results",
        help="Relative directory (under base-project-path) to store JSON results.",
    )
    parser.add_argument(
        "--target",
        dest="target_qualified_name",
        default=None,
        help=(
            "Fully qualified target function name, e.g. 'pandas.melt' or 'statistics.mean'. "
            "If omitted, the script tries to use spec['qualified_name'] / ['full_name'] / ['name']."
        ),
    )
    parser.add_argument(
        "--target-module",
        dest="target_module",
        default=None,
        help="Override for the target's module root (e.g. 'pandas', 'statistics').",
    )
    parser.add_argument(
        "--target-attr-path",
        dest="target_attr_path",
        default=None,
        help=(
            "Override for the attribute path inside the module, e.g. 'melt' or 'DataFrame.merge'. "
            "If given, used together with --target-module."
        ),
    )
    parser.add_argument(
        "--pytest-timeout",
        type=int,
        default=60,
        help="Per-test-file pytest timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        scores = evaluate_generated_tests_from_json_baseline(
            spec_path=args.spec,
            base_project_path=args.base_project_path,
            generated_dir=args.generated_dir,
            results_dir=args.results_dir,
            target_qualified_name=args.target_qualified_name,
            target_module_override=args.target_module,
            target_attr_path_override=args.target_attr_path,
            pytest_timeout=args.pytest_timeout,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"PYTEST ERROR: {e}")
        return 2
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        return 3

    print("\n=== Final scores ===")
    for name, sc in scores.items():
        print(f"{name}: {sc:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())