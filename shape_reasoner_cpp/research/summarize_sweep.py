"""Aggregate the S4-vs-S5 sweep into a seed-averaged comparison table."""

from __future__ import annotations

import json
import pathlib
import statistics
import sys


def main(root: str) -> None:
    runs: dict[str, list[dict]] = {}
    for path in sorted(pathlib.Path(root).glob("*/summary.json")):
        data = json.loads(path.read_text())
        # Strip the trailing _seedN so seeds of one config group together.
        key = data["label"].rsplit("_seed", 1)[0]
        runs.setdefault(key, []).append(data)

    print(
        f"{'config':<18}{'params':>8}{'seeds':>7}"
        f"{'add':>9}{'sub':>9}{'mul':>9}{'mul!triv':>11}{'spread':>9}"
    )
    print("-" * 80)

    def mean(entries: list[dict], field: str) -> float:
        return statistics.fmean(entry[field] for entry in entries)

    for key in sorted(runs, key=lambda k: (k.split("_")[0], len(k), k)):
        entries = runs[key]
        values = [entry["multiplication_nontrivial_accuracy"] for entry in entries]
        spread = max(values) - min(values) if len(values) > 1 else 0.0
        drift = sum(entry["evaluation_parameter_drift"] for entry in entries)
        flag = "  DRIFT" if drift else ""
        print(
            f"{key:<18}{entries[0]['parameters']:>8}{len(entries):>7}"
            f"{mean(entries, 'addition_accuracy'):>9.4f}"
            f"{mean(entries, 'subtraction_accuracy'):>9.4f}"
            f"{mean(entries, 'multiplication_accuracy'):>9.4f}"
            f"{statistics.fmean(values):>11.4f}"
            f"{spread:>9.4f}{flag}"
        )

    any_run = next(iter(runs.values()))[0]
    print(
        f"\nexhaustive evaluation: {any_run['multiplication_cases']} multiplication "
        f"cases, {any_run['multiplication_nontrivial_cases']} of them non-trivial"
    )


if __name__ == "__main__":
    main(sys.argv[1])
