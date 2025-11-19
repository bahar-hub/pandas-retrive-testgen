import os
import re
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set


@dataclass
class Mutant:
    id: int
    source_code: str
    description: str


def load_body_from_json(spec_path: str) -> str:
    with open(spec_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["body_stripped"]


def generate_mutants_from_body(body: str) -> List[Mutant]:
    mutants: List[Mutant] = []

    mutation_patterns = [
        (r">=", "<", "Replace '>=' with '<'"),
        (r"<=", ">", "Replace '<=' with '>'"),
        (r"==", "!=", "Replace '==' with '!='"),
        (r"!=", "==", "Replace '!=' with '=='"),
        (r">", "<", "Replace '>' with '<'"),
        (r"<", ">", "Replace '<' with '>'"),
        (r"True", "False", "Replace True with False"),
        (r"False", "True", "Replace False with True"),
    ]

    mutant_id = 1
    for pattern, replacement, desc in mutation_patterns:
        for match in re.finditer(pattern, body):
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


def build_module_source_from_mutated_body(mutated_body: str) -> str:
    return f'''from __future__ import annotations

import pandas as pd
from pandas.core.reshape.concat import _Concatenator


def using_copy_on_write():
    try:
        from pandas import get_option
        return bool(get_option("mode.copy_on_write"))
    except Exception:
        return False


{mutated_body}
'''


def write_conftest_for_concat(project_root: str, module_name: str, function_name: str):
    conftest_path = os.path.join(project_root, "conftest.py")
    content = f"""
import pandas as pd
from {module_name} import {function_name} as _mutated_func

pd.concat = _mutated_func
"""
    with open(conftest_path, "w", encoding="utf-8") as f:
        f.write(content)


def run_pytest_and_get_failed_tests(
    project_root: str,
    test_file: str,
) -> Tuple[Set[str], int, str]:
    cmd = ["pytest", "-q", test_file]
    proc = subprocess.run(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    output = proc.stdout + "\n" + proc.stderr
    failed_tests: Set[str] = set()

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("FAILED "):
            parts = line.split()
            if len(parts) >= 2:
                failed_tests.add(parts[1])

    return failed_tests, proc.returncode, output


def compute_mutation_score(killed: int, total: int, equivalent: int = 0) -> float:
    effective_total = total - equivalent
    if effective_total <= 0:
        return 0.0
    return (killed / effective_total) * 100.0


def evaluate_one_llm_from_json(
    spec_path: str,
    base_project_path: str,
    function_name: str,
    test_file: str,
) -> Tuple[int, int, int, float]:
    body = load_body_from_json(spec_path)
    mutants = generate_mutants_from_body(body)
    total_mutants = len(mutants)

    print(f"    Generated {total_mutants} mutants from JSON body")

    MUTATED_MODULE_FILENAME = "mutated_concat.py"
    MUTATED_MODULE_NAME = "mutated_concat"

    # ---- Baseline run ----
    with tempfile.TemporaryDirectory() as tmpdir:
        project_copy = os.path.join(tmpdir, "project")
        shutil.copytree(
            base_project_path,
            project_copy,
            dirs_exist_ok=True,
        )

        module_source = build_module_source_from_mutated_body(body)
        module_path = os.path.join(project_copy, MUTATED_MODULE_FILENAME)
        with open(module_path, "w", encoding="utf-8") as f:
            f.write(module_source)

        write_conftest_for_concat(
            project_root=project_copy,
            module_name=MUTATED_MODULE_NAME,
            function_name=function_name,
        )

        baseline_failed, baseline_rc, baseline_output = run_pytest_and_get_failed_tests(
            project_root=project_copy,
            test_file=test_file,
        )

        print(f"    Baseline exit code: {baseline_rc}")
        print(f"    Baseline failed tests ({len(baseline_failed)}):")
        for t in sorted(baseline_failed):
            print(f"      {t}")

    killed = 0
    survived = 0

    # ---- Mutants loop ----
    for mutant in mutants:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_copy = os.path.join(tmpdir, "project")
            shutil.copytree(
                base_project_path,
                project_copy,
                dirs_exist_ok=True,
            )

            module_source = build_module_source_from_mutated_body(mutant.source_code)
            module_path = os.path.join(project_copy, MUTATED_MODULE_FILENAME)
            with open(module_path, "w", encoding="utf-8") as f:
                f.write(module_source)

            write_conftest_for_concat(
                project_root=project_copy,
                module_name=MUTATED_MODULE_NAME,
                function_name=function_name,
            )

            failed, rc, _ = run_pytest_and_get_failed_tests(
                project_root=project_copy,
                test_file=test_file,
            )

            newly_failed = failed - baseline_failed

            if newly_failed:
                killed += 1
                status = "KILLED"
            else:
                survived += 1
                status = "SURVIVED"

            print(
                f"      Mutant #{mutant.id}: {mutant.description} → {status} "
                f"(new fails: {len(newly_failed)})"
            )

    score = compute_mutation_score(killed, total_mutants, equivalent=0)
    return killed, survived, total_mutants, score


def evaluate_generated_tests_from_json_baseline(
    spec_path: str,
    base_project_path: str = ".",
    function_name: str = "concat",
    generated_dir: str = "generated_tests/pandas_concat",
    results_dir: str = "mutation_results",
) -> Dict[str, float]:
    """
    همه فایل‌های تست داخل generated_dir را ارزیابی می‌کند
    و برای هر کدام یک فایل JSON نتیجه داخل results_dir می‌نویسد.
    """
    scores: Dict[str, float] = {}

    generated_path = os.path.join(base_project_path, generated_dir)
    files = sorted(
        f for f in os.listdir(generated_path) if f.endswith(".py")
    )

    # پوشه‌ی نتایج
    results_root = os.path.join(base_project_path, results_dir)
    os.makedirs(results_root, exist_ok=True)

    print(f"Found {len(files)} generated test files in {generated_dir}/")

    for fname in files:
        test_file = os.path.join(generated_dir, fname)
        llm_name = os.path.splitext(fname)[0]

        print(f"\n=== Evaluating test suite: {llm_name} ===")
        killed, survived, total, score = evaluate_one_llm_from_json(
            spec_path=spec_path,
            base_project_path=base_project_path,
            function_name=function_name,
            test_file=test_file,
        )

        print(f"Results for {llm_name}:")
        print(f"  Killed   : {killed}")
        print(f"  Survived : {survived}")
        print(f"  Total    : {total}")
        print(f"  Mutation Score: {score:.2f}%")

        scores[llm_name] = score

        # --- ذخیره نتیجه‌ی این مدل به صورت JSON ---
        result_obj = {
            "llm_name": llm_name,
            "spec_path": spec_path,
            "test_file": test_file,
            "killed": killed,
            "survived": survived,
            "total_mutants": total,
            "mutation_score": score,
        }
        result_path = os.path.join(results_root, f"{llm_name}.json")
        with open(result_path, "w", encoding="utf-8") as rf:
            json.dump(result_obj, rf, ensure_ascii=False, indent=2)



    return scores


if __name__ == "__main__":
    SPEC_PATH = os.path.join("..", "specs", "pandas_concat.json")

    scores = evaluate_generated_tests_from_json_baseline(
        spec_path=SPEC_PATH,
        base_project_path="..",                   
        function_name="concat",
        generated_dir="generated_tests/pandas_concat",
        results_dir="mutation_results",
    )

    print("\n=== Final scores ===")
    for name, sc in scores.items():
        print(f"{name}: {sc:.2f}%")