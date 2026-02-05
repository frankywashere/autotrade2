#!/usr/bin/env python3
"""
Comprehensive comparison script for C++ scanner vs Python baseline.

Compares:
1. Sample counts
2. Timestamps
3. Channel positions
4. Feature values (within tolerance 1e-6)
5. Labels
6. Generates detailed comparison report
"""

import sys
import pickle
import struct
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

# Import the loader
from load_samples import load_samples, ChannelSample, ChannelLabels


def load_python_baseline(filepath: str) -> List[Any]:
    """Load Python pickle baseline"""
    with open(filepath, 'rb') as f:
        samples = pickle.load(f)
    return samples


def compare_samples(python_samples: List[Any], cpp_samples: List[ChannelSample], tolerance: float = 1e-6):
    """
    Compare Python and C++ samples comprehensively.

    Returns dict with comparison results.
    """
    results = {
        'sample_count_match': False,
        'timestamps_match': [],
        'timestamp_mismatches': [],
        'feature_comparison': defaultdict(list),
        'label_comparison': defaultdict(list),
        'total_features_compared': 0,
        'exact_matches': 0,
        'tolerance_matches': 0,
        'mismatches': 0,
        'large_mismatches': [],
        'missing_features': defaultdict(list),
        'extra_features': defaultdict(list),
    }

    # Compare sample counts
    py_count = len(python_samples)
    cpp_count = len(cpp_samples)
    results['sample_count_match'] = (py_count == cpp_count)
    results['python_count'] = py_count
    results['cpp_count'] = cpp_count

    print(f"\n{'='*80}")
    print(f"SAMPLE COUNT COMPARISON")
    print(f"{'='*80}")
    print(f"Python samples: {py_count}")
    print(f"C++ samples:    {cpp_count}")
    print(f"Match: {results['sample_count_match']}")

    # Compare samples pairwise
    min_count = min(py_count, cpp_count)

    print(f"\n{'='*80}")
    print(f"TIMESTAMP COMPARISON (first {min(min_count, 10)} samples)")
    print(f"{'='*80}")

    for i in range(min_count):
        py_sample = python_samples[i]
        cpp_sample = cpp_samples[i]

        # Get Python timestamp
        if hasattr(py_sample, 'timestamp'):
            py_ts = py_sample.timestamp
        elif isinstance(py_sample, dict) and 'timestamp' in py_sample:
            py_ts = py_sample['timestamp']
        else:
            print(f"Sample {i}: Cannot find Python timestamp")
            continue

        cpp_ts = cpp_sample.timestamp

        match = (py_ts == cpp_ts)
        if match:
            results['timestamps_match'].append(i)
        else:
            results['timestamp_mismatches'].append((i, py_ts, cpp_ts))

        if i < 10:
            status = "✓" if match else "✗"
            print(f"Sample {i:3d}: PY={py_ts:12d}  CPP={cpp_ts:12d}  {status}")

    # Compare features
    print(f"\n{'='*80}")
    print(f"FEATURE COMPARISON")
    print(f"{'='*80}")

    for i in range(min_count):
        py_sample = python_samples[i]
        cpp_sample = cpp_samples[i]

        # Get Python features
        if hasattr(py_sample, 'tf_features'):
            py_features = py_sample.tf_features
        elif isinstance(py_sample, dict) and 'tf_features' in py_sample:
            py_features = py_sample['tf_features']
        else:
            continue

        cpp_features = cpp_sample.tf_features

        # Compare each feature
        all_keys = set(py_features.keys()) | set(cpp_features.keys())

        for key in all_keys:
            results['total_features_compared'] += 1

            if key not in py_features:
                results['missing_features'][key].append(i)
                results['mismatches'] += 1
                continue

            if key not in cpp_features:
                results['extra_features'][key].append(i)
                results['mismatches'] += 1
                continue

            py_val = py_features[key]
            cpp_val = cpp_features[key]

            # Handle NaN/inf
            if np.isnan(py_val) and np.isnan(cpp_val):
                results['exact_matches'] += 1
                results['feature_comparison'][key].append(('exact', i, py_val, cpp_val, 0.0))
                continue

            if np.isinf(py_val) and np.isinf(cpp_val) and np.sign(py_val) == np.sign(cpp_val):
                results['exact_matches'] += 1
                results['feature_comparison'][key].append(('exact', i, py_val, cpp_val, 0.0))
                continue

            # Exact match
            if py_val == cpp_val:
                results['exact_matches'] += 1
                results['feature_comparison'][key].append(('exact', i, py_val, cpp_val, 0.0))
            else:
                diff = abs(py_val - cpp_val)

                # Within tolerance
                if diff <= tolerance:
                    results['tolerance_matches'] += 1
                    results['feature_comparison'][key].append(('tolerance', i, py_val, cpp_val, diff))
                else:
                    results['mismatches'] += 1
                    results['feature_comparison'][key].append(('mismatch', i, py_val, cpp_val, diff))

                    # Track large mismatches
                    if diff > 0.01:  # Large mismatch threshold
                        results['large_mismatches'].append({
                            'sample': i,
                            'feature': key,
                            'python': py_val,
                            'cpp': cpp_val,
                            'diff': diff,
                            'rel_diff': diff / (abs(py_val) + 1e-10)
                        })

    # Compare labels
    print(f"\nComparing labels...")

    label_stats = {
        'exact': 0,
        'tolerance': 0,
        'mismatch': 0,
        'total': 0
    }

    for i in range(min_count):
        py_sample = python_samples[i]
        cpp_sample = cpp_samples[i]

        # Get Python labels
        if hasattr(py_sample, 'labels_per_window'):
            py_labels_dict = py_sample.labels_per_window
        elif isinstance(py_sample, dict) and 'labels_per_window' in py_sample:
            py_labels_dict = py_sample['labels_per_window']
        else:
            continue

        cpp_labels_dict = cpp_sample.labels_per_window

        # Compare labels for each window/timeframe
        for window in set(py_labels_dict.keys()) | set(cpp_labels_dict.keys()):
            if window not in py_labels_dict or window not in cpp_labels_dict:
                continue

            py_tf_dict = py_labels_dict[window]
            cpp_tf_dict = cpp_labels_dict[window]

            for tf in set(py_tf_dict.keys()) | set(cpp_tf_dict.keys()):
                if tf not in py_tf_dict or tf not in cpp_tf_dict:
                    continue

                py_labels = py_tf_dict[tf]
                cpp_labels = cpp_tf_dict[tf]

                # Compare key label fields
                label_fields = [
                    ('duration_bars', int),
                    ('break_direction', int),
                    ('break_magnitude', float),
                    ('permanent_break', bool),
                    ('source_channel_slope', float),
                    ('source_channel_r_squared', float),
                ]

                for field_name, field_type in label_fields:
                    label_stats['total'] += 1

                    py_val = getattr(py_labels, field_name)
                    cpp_val = getattr(cpp_labels, field_name)

                    if field_type == float:
                        if np.isnan(py_val) and np.isnan(cpp_val):
                            label_stats['exact'] += 1
                        elif abs(py_val - cpp_val) <= tolerance:
                            if py_val == cpp_val:
                                label_stats['exact'] += 1
                            else:
                                label_stats['tolerance'] += 1
                        else:
                            label_stats['mismatch'] += 1
                    else:
                        if py_val == cpp_val:
                            label_stats['exact'] += 1
                        else:
                            label_stats['mismatch'] += 1

    results['label_stats'] = label_stats

    return results


def print_detailed_report(results: Dict[str, Any], tolerance: float = 1e-6):
    """Print detailed comparison report"""

    print(f"\n{'='*80}")
    print(f"DETAILED COMPARISON REPORT")
    print(f"{'='*80}")

    # Overall statistics
    total = results['total_features_compared']
    exact = results['exact_matches']
    tol = results['tolerance_matches']
    mismatch = results['mismatches']

    print(f"\nOVERALL FEATURE STATISTICS:")
    print(f"  Total features compared: {total:,}")
    print(f"  Exact matches:           {exact:,} ({100*exact/total if total > 0 else 0:.2f}%)")
    print(f"  Tolerance matches:       {tol:,} ({100*tol/total if total > 0 else 0:.2f}%)")
    print(f"  Mismatches:              {mismatch:,} ({100*mismatch/total if total > 0 else 0:.2f}%)")

    matches_within_tolerance = exact + tol
    print(f"\n  TOTAL MATCHES (exact + tolerance): {matches_within_tolerance:,} ({100*matches_within_tolerance/total if total > 0 else 0:.2f}%)")

    # Label statistics
    if 'label_stats' in results:
        lstats = results['label_stats']
        ltotal = lstats['total']
        if ltotal > 0:
            print(f"\nLABEL STATISTICS:")
            print(f"  Total labels compared:   {ltotal:,}")
            print(f"  Exact matches:           {lstats['exact']:,} ({100*lstats['exact']/ltotal:.2f}%)")
            print(f"  Tolerance matches:       {lstats['tolerance']:,} ({100*lstats['tolerance']/ltotal:.2f}%)")
            print(f"  Mismatches:              {lstats['mismatch']:,} ({100*lstats['mismatch']/ltotal:.2f}%)")

            label_matches = lstats['exact'] + lstats['tolerance']
            print(f"\n  TOTAL LABEL MATCHES: {label_matches:,} ({100*label_matches/ltotal:.2f}%)")

    # Missing/Extra features
    if results['missing_features']:
        print(f"\nMISSING FEATURES (in C++ but not Python):")
        for feature, sample_indices in list(results['missing_features'].items())[:10]:
            print(f"  {feature}: {len(sample_indices)} samples")

    if results['extra_features']:
        print(f"\nEXTRA FEATURES (in Python but not C++):")
        for feature, sample_indices in list(results['extra_features'].items())[:10]:
            print(f"  {feature}: {len(sample_indices)} samples")

    # Large mismatches
    if results['large_mismatches']:
        print(f"\nLARGE MISMATCHES (diff > 0.01, showing worst 20):")
        sorted_mismatches = sorted(results['large_mismatches'],
                                   key=lambda x: abs(x['diff']),
                                   reverse=True)

        print(f"\n{'Sample':<8} {'Feature':<40} {'Python':<15} {'C++':<15} {'Diff':<12} {'RelDiff':<10}")
        print("-" * 110)

        for m in sorted_mismatches[:20]:
            print(f"{m['sample']:<8} {m['feature']:<40} {m['python']:<15.6f} {m['cpp']:<15.6f} {m['diff']:<12.6e} {m['rel_diff']:<10.2%}")

    # Feature breakdown by category
    print(f"\nFEATURE BREAKDOWN BY CATEGORY:")

    categories = defaultdict(lambda: {'exact': 0, 'tolerance': 0, 'mismatch': 0})

    for feature, comparisons in results['feature_comparison'].items():
        # Determine category from feature name
        if 'rsi' in feature.lower():
            cat = 'RSI'
        elif 'channel' in feature.lower():
            cat = 'Channel'
        elif 'price' in feature.lower():
            cat = 'Price'
        elif 'volume' in feature.lower():
            cat = 'Volume'
        elif 'spy' in feature.lower():
            cat = 'SPY'
        else:
            cat = 'Other'

        for comp_type, _, _, _, _ in comparisons:
            if comp_type == 'exact':
                categories[cat]['exact'] += 1
            elif comp_type == 'tolerance':
                categories[cat]['tolerance'] += 1
            elif comp_type == 'mismatch':
                categories[cat]['mismatch'] += 1

    print(f"\n{'Category':<15} {'Exact':<10} {'Tolerance':<12} {'Mismatch':<10} {'Total':<10} {'Match %':<10}")
    print("-" * 75)

    for cat in sorted(categories.keys()):
        stats = categories[cat]
        total = stats['exact'] + stats['tolerance'] + stats['mismatch']
        match_count = stats['exact'] + stats['tolerance']
        match_pct = 100 * match_count / total if total > 0 else 0

        print(f"{cat:<15} {stats['exact']:<10} {stats['tolerance']:<12} {stats['mismatch']:<10} {total:<10} {match_pct:<10.2f}%")

    # Final verdict
    print(f"\n{'='*80}")
    print(f"FINAL VERDICT")
    print(f"{'='*80}")

    if total > 0:
        match_pct = 100 * matches_within_tolerance / total

        if match_pct >= 95:
            verdict = f"✓ PASS - {match_pct:.2f}% of features match (target: >95%)"
            status = "SUCCESS"
        else:
            verdict = f"✗ FAIL - {match_pct:.2f}% of features match (target: >95%)"
            status = "NEEDS INVESTIGATION"

        print(f"\n{verdict}")
        print(f"Status: {status}")
    else:
        print("\n✗ FAIL - No features compared")
        print("Status: ERROR")

    print(f"\n{'='*80}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_scanners.py <python_baseline.pkl> <cpp_output.bin>")
        sys.exit(1)

    python_file = sys.argv[1]
    cpp_file = sys.argv[2]

    if not Path(python_file).exists():
        print(f"Error: Python baseline file not found: {python_file}")
        sys.exit(1)

    if not Path(cpp_file).exists():
        print(f"Error: C++ output file not found: {cpp_file}")
        sys.exit(1)

    print(f"Loading Python baseline from: {python_file}")
    python_samples = load_python_baseline(python_file)
    print(f"Loaded {len(python_samples)} Python samples")

    print(f"\nLoading C++ output from: {cpp_file}")
    version, num_samples, num_features, cpp_samples = load_samples(cpp_file)
    print(f"Loaded {len(cpp_samples)} C++ samples (version={version}, avg_features={num_features})")

    # Run comparison
    tolerance = 1e-6
    results = compare_samples(python_samples, cpp_samples, tolerance=tolerance)

    # Print detailed report
    print_detailed_report(results, tolerance=tolerance)


if __name__ == '__main__':
    main()
