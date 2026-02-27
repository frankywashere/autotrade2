#!/usr/bin/env python3
"""
Simple test runner for parallel scanning integration tests.

This script runs the tests with proper PYTHONPATH configuration,
avoiding the import issues that occur when using pytest directly.

Usage:
    cd /path/to/x6
    python3 v7/training/tests/run_tests.py
"""

import sys
import subprocess
from pathlib import Path

# Ensure we're in the project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Change to project root
import os
os.chdir(str(project_root))

# Run pytest with proper PYTHONPATH
test_file = "v7/training/tests/test_parallel_scanning_integration.py"

print("=" * 80)
print("Running Parallel Scanning Integration Tests")
print("=" * 80)
print(f"Project root: {project_root}")
print(f"Test file: {test_file}")
print("=" * 80)
print()

# Set PYTHONPATH and run pytest
env = os.environ.copy()
env['PYTHONPATH'] = str(project_root)

result = subprocess.run(
    [sys.executable, "-m", "pytest", test_file, "-v", "-s", "--tb=short"],
    env=env,
    cwd=str(project_root)
)

sys.exit(result.returncode)
