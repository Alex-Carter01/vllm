#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare arm_*_outputs.json files from run_matrix.sh --check-outputs.

Reports, per arm, how many greedy completions match a reference arm exactly
and a crude garbage score (share of characters that are not printable text).
A fast arm that scores near 0% match with a high garbage score is producing
wrong output.

Usage:
    compare_outputs.py RESULTS_DIR [--reference B]
"""

import argparse
import glob
import json
import os
import sys


def garbage_share(text: str) -> float:
    if not text:
        return 1.0
    bad = sum(1 for c in text if not (c.isprintable() or c in "\n\t"))
    return bad / len(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--reference", default="B")
    args = parser.parse_args()

    outputs = {}
    for path in sorted(glob.glob(os.path.join(args.results_dir, "arm_*_outputs.json"))):
        arm = os.path.basename(path)[len("arm_") : -len("_outputs.json")]
        with open(path) as f:
            outputs[arm] = [r["text"] for r in json.load(f)]
    if args.reference not in outputs:
        print(f"error: no outputs for reference arm {args.reference}", file=sys.stderr)
        return 2

    ref = outputs[args.reference]
    print(f"reference arm: {args.reference} ({len(ref)} prompts)\n")
    print("| arm | exact match vs ref | empty | mean garbage share |")
    print("| --- | --- | --- | --- |")
    for arm, texts in outputs.items():
        match = sum(a == b for a, b in zip(texts, ref))
        empty = sum(not t.strip() for t in texts)
        garbage = sum(garbage_share(t) for t in texts) / max(len(texts), 1)
        print(f"| {arm} | {match}/{len(ref)} | {empty} | {garbage:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
