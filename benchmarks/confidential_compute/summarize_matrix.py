# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize a run_matrix.sh results directory.

Prints, per concurrency, output tok/s and median TPOT for every arm, and checks
the success criterion: the flag-free stock arm (B) must be within single-digit
percent of the best hand-tuned arm (any other arm).

    python summarize_matrix.py cc_matrix_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STOCK_ARM = "B"
THRESHOLD_PCT = 10.0  # "single-digit %" means strictly under this

ARM_DESC = {
    "A": "baseline: CC=0 (stock main under CC)",
    "B": "stock-new: CC-aware defaults",
    "C": "--no-async-scheduling (V1, opts on)",
    "D1": "V1, opts off, async on",
    "D2": "V1, opts off, async off",
    "E1": "forced V2, stock async",
    "E2": "forced V2, async on",
}


def load(results_dir: Path) -> dict[int, dict[str, dict]]:
    data: dict[int, dict[str, dict]] = {}
    for path in sorted(results_dir.glob("arm_*_c*.json")):
        # arm_<ARM>_c<CONC>.json
        arm, _, conc = path.stem[len("arm_") :].rpartition("_c")
        if not arm or not conc.isdigit():
            continue
        with path.open() as f:
            result = json.load(f)
        data.setdefault(int(conc), {})[arm] = result
    return data


def fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    data = load(args.results_dir)
    if not data:
        print(f"no arm_*_c*.json results in {args.results_dir}", file=sys.stderr)
        return 2

    print("# CC-aware defaults: serving matrix\n")
    verdicts = []
    for conc in sorted(data):
        arms = data[conc]
        print(f"## Concurrency {conc}\n")
        print("| arm | meaning | output tok/s | vs best | median TPOT (ms) |")
        print("| --- | --- | --- | --- | --- |")
        best = max(r["output_throughput"] for r in arms.values())
        for arm in sorted(arms):
            thr = arms[arm]["output_throughput"]
            tpot = arms[arm].get("median_tpot_ms")
            delta = (thr / best - 1.0) * 100.0
            print(
                f"| {arm} | {ARM_DESC.get(arm, '')} | {fmt(thr)} | "
                f"{delta:+.1f}% | {fmt(tpot, 2)} |"
            )
        print()
        if STOCK_ARM in arms:
            tuned = {a: r for a, r in arms.items() if a != STOCK_ARM}
            if tuned:
                best_arm = max(tuned, key=lambda a: tuned[a]["output_throughput"])
                gap = (
                    1.0
                    - arms[STOCK_ARM]["output_throughput"]
                    / tuned[best_arm]["output_throughput"]
                ) * 100.0
                verdicts.append((conc, best_arm, gap))

    print("## Success criterion\n")
    print(
        f"Stock `{STOCK_ARM}` within <{THRESHOLD_PCT:.0f}% of the best other arm "
        "(positive gap = stock is slower):\n"
    )
    ok = True
    for conc, best_arm, gap in verdicts:
        passed = gap < THRESHOLD_PCT
        ok &= passed
        status = "PASS" if passed else "FAIL"
        print(f"- c={conc}: gap {gap:+.1f}% vs {best_arm} -> {status}")
    print(f"\n**Overall: {'PASS' if ok else 'FAIL'}**")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
