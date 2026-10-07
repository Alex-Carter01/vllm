#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect greedy completions for a fixed prompt set from a running server.

Throughput numbers cannot reveal silently wrong output (for example the UVA
incoherence under Confidential Computing in vllm-project/vllm#57224), so
run_matrix.sh --check-outputs saves each arm's completions here and
compare_outputs.py checks them against a reference arm.
"""

import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "List the first ten positive integers separated by commas: 1,",
    "Q: What color is the sky on a clear day?\nA:",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Translate to French: 'Good morning, how are you?'\n",
    "The chemical symbol for water is",
    "Once upon a time, in a small village by the sea,",
]


def complete(url: str, model: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 0,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)["choices"][0]["text"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}/v1/completions"
    results = [
        {"prompt": p, "text": complete(url, args.model, p, args.max_tokens)}
        for p in PROMPTS
    ]
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        print(f"{r['prompt'][:40]!r} -> {r['text'][:60]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
