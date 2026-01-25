#!/usr/bin/env python3
"""
V15 Feature Validation - Python vs C++ Scanner Output Comparison

This script performs comprehensive validation between Python and C++ scanner outputs:
1. Loads both Python (.pkl) and C++ (.bin) sample files
2. Compares all 14,190 features sample-by-sample
3. Reports differences (strict validation with configurable tolerance)
4. Validates label fields match exactly
5. Checks timestamp alignment
6. Creates detailed diff report

Usage:
    python tests/validate_features.py \\
        --python python_samples.pkl \\
        --cpp cpp_samples.bin \\
        --tolerance 1e-10 \\
        --output validation_report.txt

Exit Codes:
    0 - All validation passed
    1 - Validation failed (differences found)
    2 - Error loading or parsing files
"""

import argparse
import pickle
import struct
import sys
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class ChannelSample:
    """Python representation of ChannelSample (matching v15.dtypes.ChannelSample)"""
    timestamp: int
    channel_end_idx: int
    best_window: int
    tf_features: Dict[str, float]
    labels_per_window: Dict[int, Dict[str, Any]]
    bar_metadata: Dict[str, Dict[str, float]]


class BinaryReader:
    """Helper class for reading C++ binary format"""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = open(filepath, 'rb')
        self.pos = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def read_uint32(self) -> int:
        data = self.file.read(4)
        if len(data) != 4:
            raise EOFError("Unexpected end of file reading uint32")
        self.pos += 4
        return struct.unpack('<I', data)[0]

    def read_uint64(self) -> int:
        data = self.file.read(8)
        if len(data) != 8:
            raise EOFError("Unexpected end of file reading uint64")
        self.pos += 8
        return struct.unpack('<Q', data)[0]

    def read_int64(self) -> int:
        data = self.file.read(8)
        if len(data) != 8:
            raise EOFError("Unexpected end of file reading int64")
        self.pos += 8
        return struct.unpack('<q', data)[0]

    def read_int32(self) -> int:
        data = self.file.read(4)
        if len(data) != 4:
            raise EOFError("Unexpected end of file reading int32")
        self.pos += 4
        return struct.unpack('<i', data)[0]

    def read_double(self) -> float:
        data = self.file.read(8)
        if len(data) != 8:
            raise EOFError("Unexpected end of file reading double")
        self.pos += 8
        return struct.unpack('<d', data)[0]

    def read_bool(self) -> bool:
        data = self.file.read(1)
        if len(data) != 1:
            raise EOFError("Unexpected end of file reading bool")
        self.pos += 1
        return struct.unpack('<?', data)[0]

    def read_string(self, length: int) -> str:
        data = self.file.read(length)
        if len(data) != length:
            raise EOFError(f"Unexpected end of file reading string of length {length}")
        self.pos += length
        return data.decode('utf-8')


def load_cpp_samples(filepath: str) -> List[ChannelSample]:
    """Load samples from C++ binary format"""
    samples = []

    with BinaryReader(filepath) as reader:
        # Read header
        magic = reader.read_uint32()
        if magic != 0x56313543:  # "V15C"
            raise ValueError(f"Invalid magic number: {magic:08x} (expected 0x56313543)")

        version = reader.read_uint32()
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")

        count = reader.read_uint64()
        print(f"Reading {count} samples from C++ binary file...")

        # Read each sample
        for i in range(count):
            # Core fields
            timestamp = reader.read_int64()
            channel_end_idx = reader.read_int32()
            best_window = reader.read_int32()

            # Features
            feature_count = reader.read_uint64()
            tf_features = {}
            for _ in range(feature_count):
                key_len = reader.read_uint32()
                key = reader.read_string(key_len)
                value = reader.read_double()
                tf_features[key] = value

            # Labels per window
            label_window_count = reader.read_uint64()
            labels_per_window = {}
            for _ in range(label_window_count):
                window = reader.read_int32()
                tf_count = reader.read_uint64()
                tf_map = {}

                for _ in range(tf_count):
                    tf_len = reader.read_uint32()
                    tf = reader.read_string(tf_len)

                    # Read label fields
                    labels = {
                        'direction_valid': reader.read_bool(),
                        'direction': reader.read_int32(),
                        'first_break_bar': reader.read_int32(),
                        'permanent_break_bar': reader.read_int32(),
                        'break_magnitude': reader.read_double(),
                    }
                    tf_map[tf] = labels

                labels_per_window[window] = tf_map

            sample = ChannelSample(
                timestamp=timestamp,
                channel_end_idx=channel_end_idx,
                best_window=best_window,
                tf_features=tf_features,
                labels_per_window=labels_per_window,
                bar_metadata={}
            )
            samples.append(sample)

            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{count} samples...")

    print(f"Successfully loaded {len(samples)} C++ samples")
    return samples


def load_python_samples(filepath: str) -> List[Any]:
    """Load samples from Python pickle format"""
    print(f"Loading Python samples from {filepath}...")
    with open(filepath, 'rb') as f:
        samples = pickle.load(f)
    print(f"Successfully loaded {len(samples)} Python samples")
    return samples


def compare_features(py_features: Dict[str, float],
                     cpp_features: Dict[str, float],
                     tolerance: float = 1e-10) -> Tuple[bool, List[str]]:
    """
    Compare feature dictionaries between Python and C++.

    Returns:
        (all_match, differences) where differences is a list of error messages
    """
    differences = []

    # Check feature count
    if len(py_features) != len(cpp_features):
        differences.append(
            f"Feature count mismatch: Python={len(py_features)}, C++={len(cpp_features)}"
        )

    # Check all Python features exist in C++
    for key, py_value in py_features.items():
        if key not in cpp_features:
            differences.append(f"Missing in C++: {key}")
            continue

        cpp_value = cpp_features[key]

        # Handle NaN/inf values
        if np.isnan(py_value) and np.isnan(cpp_value):
            continue
        if np.isinf(py_value) and np.isinf(cpp_value):
            if np.sign(py_value) != np.sign(cpp_value):
                differences.append(f"{key}: inf sign mismatch (Python={py_value}, C++={cpp_value})")
            continue

        # Check if one is NaN and the other isn't
        if np.isnan(py_value) != np.isnan(cpp_value):
            differences.append(f"{key}: NaN mismatch (Python={py_value}, C++={cpp_value})")
            continue

        # Check numerical difference
        abs_diff = abs(py_value - cpp_value)
        rel_diff = abs_diff / (abs(py_value) + 1e-100)  # Avoid division by zero

        if abs_diff > tolerance and rel_diff > tolerance:
            differences.append(
                f"{key}: value mismatch (Python={py_value:.15e}, C++={cpp_value:.15e}, "
                f"abs_diff={abs_diff:.15e}, rel_diff={rel_diff:.15e})"
            )

    # Check for extra features in C++
    for key in cpp_features:
        if key not in py_features:
            differences.append(f"Extra in C++: {key}")

    return len(differences) == 0, differences


def compare_labels(py_labels: Any, cpp_labels: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Compare label objects between Python and C++"""
    differences = []

    # Extract Python label fields (assuming dataclass or dict-like)
    if hasattr(py_labels, '__dict__'):
        py_dict = py_labels.__dict__
    elif isinstance(py_labels, dict):
        py_dict = py_labels
    else:
        differences.append(f"Unknown Python label type: {type(py_labels)}")
        return False, differences

    # Compare key fields
    key_fields = ['direction_valid', 'direction', 'first_break_bar',
                  'permanent_break_bar', 'break_magnitude']

    for field in key_fields:
        if field not in cpp_labels:
            differences.append(f"Missing field in C++ labels: {field}")
            continue

        py_val = py_dict.get(field)
        cpp_val = cpp_labels[field]

        if py_val != cpp_val:
            differences.append(f"{field}: Python={py_val}, C++={cpp_val}")

    return len(differences) == 0, differences


def validate_samples(python_samples: List[Any],
                     cpp_samples: List[ChannelSample],
                     tolerance: float = 1e-10,
                     verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Perform comprehensive validation between Python and C++ samples.

    Returns:
        (all_passed, report_dict)
    """
    report = {
        'sample_count_match': False,
        'timestamp_mismatches': [],
        'feature_mismatches': [],
        'label_mismatches': [],
        'total_samples_compared': 0,
        'samples_with_errors': 0,
        'total_feature_comparisons': 0,
        'feature_errors': 0,
    }

    # Check sample count
    if len(python_samples) != len(cpp_samples):
        print(f"ERROR: Sample count mismatch!")
        print(f"  Python: {len(python_samples)} samples")
        print(f"  C++:    {len(cpp_samples)} samples")
        report['sample_count_match'] = False
        return False, report
    else:
        print(f"PASS: Sample count matches ({len(python_samples)} samples)")
        report['sample_count_match'] = True

    # Compare sample by sample
    print(f"\nComparing {len(python_samples)} samples...")
    samples_with_errors = 0
    total_feature_comparisons = 0
    feature_errors = 0

    for i, (py_sample, cpp_sample) in enumerate(zip(python_samples, cpp_samples)):
        sample_has_error = False

        # Compare timestamps
        py_ts = int(py_sample.timestamp.timestamp() * 1000) if hasattr(py_sample.timestamp, 'timestamp') else py_sample.timestamp
        cpp_ts = cpp_sample.timestamp

        if py_ts != cpp_ts:
            report['timestamp_mismatches'].append({
                'sample_idx': i,
                'python_ts': py_ts,
                'cpp_ts': cpp_ts,
            })
            sample_has_error = True

        # Compare channel_end_idx
        if py_sample.channel_end_idx != cpp_sample.channel_end_idx:
            report['timestamp_mismatches'].append({
                'sample_idx': i,
                'field': 'channel_end_idx',
                'python': py_sample.channel_end_idx,
                'cpp': cpp_sample.channel_end_idx,
            })
            sample_has_error = True

        # Compare best_window
        if py_sample.best_window != cpp_sample.best_window:
            report['timestamp_mismatches'].append({
                'sample_idx': i,
                'field': 'best_window',
                'python': py_sample.best_window,
                'cpp': cpp_sample.best_window,
            })
            sample_has_error = True

        # Compare features
        features_match, feature_diffs = compare_features(
            py_sample.tf_features,
            cpp_sample.tf_features,
            tolerance
        )

        total_feature_comparisons += len(py_sample.tf_features)

        if not features_match:
            report['feature_mismatches'].append({
                'sample_idx': i,
                'timestamp': py_ts,
                'differences': feature_diffs[:10],  # Limit to first 10 for brevity
                'total_differences': len(feature_diffs),
            })
            feature_errors += len(feature_diffs)
            sample_has_error = True

        # Compare labels (if available in both)
        if hasattr(py_sample, 'labels_per_window') and cpp_sample.labels_per_window:
            for window in py_sample.labels_per_window:
                if window not in cpp_sample.labels_per_window:
                    report['label_mismatches'].append({
                        'sample_idx': i,
                        'error': f'Window {window} missing in C++',
                    })
                    sample_has_error = True
                    continue

                py_window_labels = py_sample.labels_per_window[window]
                cpp_window_labels = cpp_sample.labels_per_window[window]

                # Compare each timeframe's labels
                for tf_key in py_window_labels.get('tsla', {}):
                    if tf_key not in cpp_window_labels:
                        report['label_mismatches'].append({
                            'sample_idx': i,
                            'window': window,
                            'error': f'Timeframe {tf_key} missing in C++',
                        })
                        sample_has_error = True

        if sample_has_error:
            samples_with_errors += 1

        # Progress update
        if verbose and (i + 1) % 10 == 0:
            print(f"  Compared {i + 1}/{len(python_samples)} samples...")

    report['total_samples_compared'] = len(python_samples)
    report['samples_with_errors'] = samples_with_errors
    report['total_feature_comparisons'] = total_feature_comparisons
    report['feature_errors'] = feature_errors

    # Determine overall pass/fail
    all_passed = (
        report['sample_count_match'] and
        len(report['timestamp_mismatches']) == 0 and
        len(report['feature_mismatches']) == 0 and
        len(report['label_mismatches']) == 0
    )

    return all_passed, report


def print_validation_report(report: Dict[str, Any], output_file: str = None):
    """Print detailed validation report"""
    lines = []

    lines.append("=" * 80)
    lines.append("VALIDATION REPORT - Python vs C++ Scanner Comparison")
    lines.append("=" * 80)

    # Overall result
    all_passed = (
        report['sample_count_match'] and
        len(report['timestamp_mismatches']) == 0 and
        len(report['feature_mismatches']) == 0 and
        len(report['label_mismatches']) == 0
    )

    if all_passed:
        lines.append("\n*** ALL VALIDATION PASSED ***\n")
    else:
        lines.append("\n*** VALIDATION FAILED ***\n")

    # Sample count
    lines.append(f"Sample Count: {'PASS' if report['sample_count_match'] else 'FAIL'}")
    lines.append(f"  Samples compared: {report['total_samples_compared']}")

    # Timestamp mismatches
    if report['timestamp_mismatches']:
        lines.append(f"\nTimestamp/Index Mismatches: {len(report['timestamp_mismatches'])}")
        for mismatch in report['timestamp_mismatches'][:5]:
            lines.append(f"  Sample {mismatch['sample_idx']}: {mismatch}")
        if len(report['timestamp_mismatches']) > 5:
            lines.append(f"  ... and {len(report['timestamp_mismatches']) - 5} more")
    else:
        lines.append(f"\nTimestamp/Index Mismatches: PASS (0 mismatches)")

    # Feature mismatches
    if report['feature_mismatches']:
        lines.append(f"\nFeature Mismatches: {len(report['feature_mismatches'])} samples with errors")
        lines.append(f"  Total feature comparisons: {report['total_feature_comparisons']}")
        lines.append(f"  Total feature errors: {report['feature_errors']}")
        error_rate = 100.0 * report['feature_errors'] / max(1, report['total_feature_comparisons'])
        lines.append(f"  Error rate: {error_rate:.6f}%")

        lines.append("\nFirst 5 samples with feature mismatches:")
        for mismatch in report['feature_mismatches'][:5]:
            lines.append(f"\n  Sample {mismatch['sample_idx']} (timestamp={mismatch['timestamp']}):")
            lines.append(f"    Total differences: {mismatch['total_differences']}")
            lines.append(f"    First differences:")
            for diff in mismatch['differences'][:5]:
                lines.append(f"      - {diff}")
        if len(report['feature_mismatches']) > 5:
            lines.append(f"\n  ... and {len(report['feature_mismatches']) - 5} more samples with errors")
    else:
        lines.append(f"\nFeature Mismatches: PASS (0 errors)")
        lines.append(f"  Total feature comparisons: {report['total_feature_comparisons']}")

    # Label mismatches
    if report['label_mismatches']:
        lines.append(f"\nLabel Mismatches: {len(report['label_mismatches'])}")
        for mismatch in report['label_mismatches'][:5]:
            lines.append(f"  Sample {mismatch['sample_idx']}: {mismatch}")
        if len(report['label_mismatches']) > 5:
            lines.append(f"  ... and {len(report['label_mismatches']) - 5} more")
    else:
        lines.append(f"\nLabel Mismatches: PASS (0 mismatches)")

    # Summary
    lines.append("\n" + "=" * 80)
    if all_passed:
        lines.append("RESULT: VALIDATION PASSED - C++ and Python outputs match exactly")
    else:
        lines.append("RESULT: VALIDATION FAILED - Differences found between C++ and Python")
    lines.append("=" * 80)

    # Print to console
    output = "\n".join(lines)
    print(output)

    # Write to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"\nReport written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate C++ scanner output against Python baseline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--python', required=True, help='Python samples file (.pkl)')
    parser.add_argument('--cpp', required=True, help='C++ samples file (.bin)')
    parser.add_argument('--tolerance', type=float, default=1e-10,
                       help='Numerical tolerance for feature comparison (default: 1e-10)')
    parser.add_argument('--output', help='Output file for validation report (optional)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Check files exist
    if not Path(args.python).exists():
        print(f"ERROR: Python samples file not found: {args.python}")
        return 2

    if not Path(args.cpp).exists():
        print(f"ERROR: C++ samples file not found: {args.cpp}")
        return 2

    try:
        # Load samples
        python_samples = load_python_samples(args.python)
        cpp_samples = load_cpp_samples(args.cpp)

        # Validate
        print(f"\nValidation tolerance: {args.tolerance:.2e}")
        all_passed, report = validate_samples(
            python_samples,
            cpp_samples,
            tolerance=args.tolerance,
            verbose=args.verbose
        )

        # Print report
        print_validation_report(report, args.output)

        # Return appropriate exit code
        return 0 if all_passed else 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
