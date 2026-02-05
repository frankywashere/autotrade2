#!/usr/bin/env python3
"""Wrapper to run comparison with proper sys.path setup"""
import sys
from pathlib import Path

# Add parent directory to sys.path to enable v15 module import
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Now run the comparison script
import compare_scanners
compare_scanners.main()
