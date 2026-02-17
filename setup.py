"""
setup.py - Build the C++ feature extractor as a pybind11 Python extension.

Usage:
    pip install .              # install locally
    pip install -e .           # editable/dev install
    python setup.py build_ext  # build only (no install)
"""

import os
import sys
from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

# ---------------------------------------------------------------------------
# Source files (explicit list matching the C++ scanner, excluding main_scanner)
# ---------------------------------------------------------------------------
# Exclude flat_writer.cpp and npy_writer.cpp — they use std::filesystem
# (macOS 10.15+) and are only needed by the standalone scanner binary.
_src_files = sorted(glob("v15_cpp/src/data_loader.cpp")) + \
             sorted(glob("v15_cpp/src/channel_detector.cpp")) + \
             sorted(glob("v15_cpp/src/indicators.cpp")) + \
             sorted(glob("v15_cpp/src/scanner.cpp")) + \
             sorted(glob("v15_cpp/src/serialization.cpp")) + \
             sorted(glob("v15_cpp/src/feature_extractor.cpp")) + \
             sorted(glob("v15_cpp/src/label_generator.cpp")) + \
             sorted(glob("v15_cpp/python_bindings/bindings.cpp"))

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
include_dirs = [
    "v15_cpp/include",
]

# Eigen3 - try common system locations
_eigen_candidates = [
    "/opt/homebrew/include/eigen3",   # macOS Homebrew (Apple Silicon)
    "/usr/local/include/eigen3",      # macOS Homebrew (Intel) / manual install
    "/usr/include/eigen3",            # Linux system package (libeigen3-dev)
]

for _eigen_path in _eigen_candidates:
    if os.path.isdir(_eigen_path):
        include_dirs.append(_eigen_path)

# ---------------------------------------------------------------------------
# Extension module
# ---------------------------------------------------------------------------
ext_modules = [
    Pybind11Extension(
        "v15scanner_cpp",
        sources=_src_files,
        include_dirs=include_dirs,
        define_macros=[("PYBIND11_BUILD", "1")],
        extra_compile_args=["-O3"],
        cxx_std=17,
    ),
]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup(
    name="x14",
    version="0.1.0",
    description="Autotrade v15 - multi-TF channel analysis with C++ scanner",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
    zip_safe=False,
)
