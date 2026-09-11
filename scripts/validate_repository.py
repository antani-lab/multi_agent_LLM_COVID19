#!/usr/bin/env python3
"""Run lightweight integrity checks that require neither data nor model downloads."""

from __future__ import annotations

import json
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "stage_A": set(range(0, 5)),
    "stage_B": set(range(5, 13)),
    "stage_C": set(range(13, 17)),
    "stage_D": set(range(17, 23)),
}


def main() -> int:
    failures = []
    for stage, expected_numbers in EXPECTED.items():
        directory = ROOT / "notebooks" / stage
        notebooks = sorted(directory.glob("*.ipynb"))
        observed_numbers = {int(path.name.split("_", 1)[0]) for path in notebooks}
        if observed_numbers != expected_numbers:
            failures.append(
                f"{stage}: expected notebook numbers {sorted(expected_numbers)}, "
                f"found {sorted(observed_numbers)}"
            )
        for path in notebooks:
            try:
                notebook = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                failures.append(f"{path.relative_to(ROOT)} is not valid JSON: {exc}")
                continue
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                if cell.get("outputs"):
                    failures.append(f"{path.relative_to(ROOT)} cell {index} contains saved output")
                if cell.get("execution_count") is not None:
                    failures.append(
                        f"{path.relative_to(ROOT)} cell {index} contains an execution count"
                    )

    for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path.relative_to(ROOT)} does not compile: {exc}")

    if failures:
        print("Repository validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Repository validation passed: 23 staged notebooks are valid and output-free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
