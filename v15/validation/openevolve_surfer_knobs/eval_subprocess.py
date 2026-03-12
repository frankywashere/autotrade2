#!/usr/bin/env python3
"""
Subprocess evaluator wrapper for simple_evolve.py.

Runs in a fresh process per evaluation to prevent memory leaks from
loading 3.3M-row 5-sec bar data + ML models.

Usage: python eval_subprocess.py <program_path>
Outputs: JSON result dict on last line of stdout.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from v15.validation.openevolve_surfer_knobs.evaluator import evaluate


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: eval_subprocess.py <program_path>"}))
        sys.exit(1)

    program_path = sys.argv[1]
    if not os.path.isfile(program_path):
        print(json.dumps({"error": f"File not found: {program_path}"}))
        sys.exit(1)

    result = evaluate(program_path)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
