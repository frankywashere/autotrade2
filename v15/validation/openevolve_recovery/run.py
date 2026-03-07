"""Launch OpenEvolve for recovery-day gate evolution.

Run from the project root:
    python v15/validation/openevolve_recovery/run.py

Requires:
  - OpenEvolve installed
  - Claude CLI proxy running on port 5564
  - Local data files (TSLAMin.txt, SPYMin.txt)
"""
import os
import sys

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
os.chdir(project_root)

from openevolve import run_evolution

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    run_evolution(
        initial_program=os.path.join(HERE, 'initial_program.py'),
        evaluator=os.path.join(HERE, 'evaluator.py'),
        config=os.path.join(HERE, 'config.yaml'),
        iterations=200,
        output_dir=os.path.join(HERE, 'output'),
    )
