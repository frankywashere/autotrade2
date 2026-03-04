"""Launch OpenEvolve for recovery-day gate evolution."""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from openevolve import OpenEvolve

this_dir = os.path.dirname(os.path.abspath(__file__))

oe = OpenEvolve(
    initial_program=os.path.join(this_dir, "initial_program.py"),
    evaluation_file=os.path.join(this_dir, "evaluator.py"),
    config_file=os.path.join(this_dir, "config.yaml"),
    output_dir=os.path.join(this_dir, "output"),
)

oe.run()
