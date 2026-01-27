#!/usr/bin/env python3
"""
Comprehensive report on forward_scan labeling in the cache.
"""

import pickle
import random
from collections import defaultdict

def generate_comprehensive_report(sample_file):
    """Generate comprehensive report on forward_scan labeling."""
    print(f"\nLoading samples from: {sample_file}")

    with open(sample_file, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples\n")

    # Configuration
    window = 50
    timeframe = '5min'

    # Pick 10 random samples from positions 100-200
    if len(samples) < 200:
        start_pos = max(0, len(samples) - 100)
        end_pos = len(samples)
    else:
        start_pos = 100
        end_pos = 200

    available_positions = list(range(start_pos, min(end_pos, len(samples))))
    num_samples = min(10, len(available_positions))
    selected_positions = sorted(random.sample(available_positions, num_samples))

    print("=" * 80)
    print("FORWARD_SCAN LABELING VERIFICATION REPORT")
    print("=" * 80)
    print(f"Configuration: window={window}, timeframe={timeframe}")
    print(f"Samples analyzed: {num_samples} random samples from positions {start_pos}-{end_pos}")
    print(f"Selected positions: {selected_positions}")
    print("=" * 80)

    # Stats
    stats = {
        'tsla': {
            'valid': 0,
            'breaks_detected': 0,
            'permanent_breaks': 0,
            'returned_to_channel': 0,
            'break_up': 0,
            'break_down': 0,
            'bars_to_break': [],
            'break_magnitudes': [],
        },
        'spy': {
            'valid': 0,
            'breaks_detected': 0,
            'permanent_breaks': 0,
            'returned_to_channel': 0,
            'break_up': 0,
            'break_down': 0,
            'bars_to_break': [],
            'break_magnitudes': [],
        }
    }

    # Detailed samples
    detailed_samples = []

    for pos in selected_positions:
        sample = samples[pos]

        # Extract TSLA labels
        tsla_labels = None
        if (hasattr(sample, 'labels_per_window') and
            sample.labels_per_window and
            window in sample.labels_per_window and
            'tsla' in sample.labels_per_window[window] and
            timeframe in sample.labels_per_window[window]['tsla']):
            tsla_labels = sample.labels_per_window[window]['tsla'][timeframe]

        # Extract SPY labels
        spy_labels = None
        if (hasattr(sample, 'labels_per_window') and
            sample.labels_per_window and
            window in sample.labels_per_window and
            'spy' in sample.labels_per_window[window] and
            timeframe in sample.labels_per_window[window]['spy']):
            spy_labels = sample.labels_per_window[window]['spy'][timeframe]

        # Record sample details
        detailed_samples.append({
            'position': pos,
            'timestamp': sample.timestamp,
            'best_window': sample.best_window,
            'tsla': tsla_labels,
            'spy': spy_labels,
        })

        # Update TSLA stats
        if tsla_labels and tsla_labels.break_scan_valid:
            stats['tsla']['valid'] += 1
            if tsla_labels.bars_to_first_break > 0:
                stats['tsla']['breaks_detected'] += 1
                stats['tsla']['bars_to_break'].append(tsla_labels.bars_to_first_break)
            if tsla_labels.permanent_break:
                stats['tsla']['permanent_breaks'] += 1
            if tsla_labels.returned_to_channel:
                stats['tsla']['returned_to_channel'] += 1
            if tsla_labels.first_break_direction == 1:
                stats['tsla']['break_up'] += 1
            elif tsla_labels.first_break_direction == 0:
                stats['tsla']['break_down'] += 1
            if tsla_labels.break_magnitude > 0:
                stats['tsla']['break_magnitudes'].append(tsla_labels.break_magnitude)

        # Update SPY stats
        if spy_labels and spy_labels.break_scan_valid:
            stats['spy']['valid'] += 1
            if spy_labels.bars_to_first_break > 0:
                stats['spy']['breaks_detected'] += 1
                stats['spy']['bars_to_break'].append(spy_labels.bars_to_first_break)
            if spy_labels.permanent_break:
                stats['spy']['permanent_breaks'] += 1
            if spy_labels.returned_to_channel:
                stats['spy']['returned_to_channel'] += 1
            if spy_labels.first_break_direction == 1:
                stats['spy']['break_up'] += 1
            elif spy_labels.first_break_direction == 0:
                stats['spy']['break_down'] += 1
            if spy_labels.break_magnitude > 0:
                stats['spy']['break_magnitudes'].append(spy_labels.break_magnitude)

    # Print detailed sample breakdown
    print("\n" + "=" * 80)
    print("DETAILED SAMPLE BREAKDOWN")
    print("=" * 80)

    for i, detail in enumerate(detailed_samples):
        print(f"\nSample {i+1}/{num_samples} - Position {detail['position']}")
        print(f"  Timestamp: {detail['timestamp']}")
        print(f"  Best window: {detail['best_window']}")

        print(f"\n  TSLA:")
        if detail['tsla']:
            t = detail['tsla']
            print(f"    break_scan_valid: {t.break_scan_valid}")
            print(f"    bars_to_first_break: {t.bars_to_first_break}")
            print(f"    break_direction: {'UP' if t.first_break_direction == 1 else 'DOWN'}")
            print(f"    break_magnitude: {t.break_magnitude:.2f} std devs")
            print(f"    permanent_break: {t.permanent_break}")
            print(f"    returned_to_channel: {t.returned_to_channel}")
        else:
            print("    No labels found")

        print(f"\n  SPY:")
        if detail['spy']:
            s = detail['spy']
            print(f"    break_scan_valid: {s.break_scan_valid}")
            print(f"    bars_to_first_break: {s.bars_to_first_break}")
            print(f"    break_direction: {'UP' if s.first_break_direction == 1 else 'DOWN'}")
            print(f"    break_magnitude: {s.break_magnitude:.2f} std devs")
            print(f"    permanent_break: {s.permanent_break}")
            print(f"    returned_to_channel: {s.returned_to_channel}")
        else:
            print("    No labels found")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for asset in ['tsla', 'spy']:
        asset_upper = asset.upper()
        s = stats[asset]

        print(f"\n{asset_upper} ({timeframe} w{window}):")
        print(f"  Samples with break_scan_valid=True: {s['valid']}/{num_samples} ({s['valid']/num_samples*100:.1f}%)")
        print(f"  Samples with breaks detected: {s['breaks_detected']}/{num_samples} ({s['breaks_detected']/num_samples*100:.1f}%)")
        print(f"  Permanent breaks: {s['permanent_breaks']}/{num_samples} ({s['permanent_breaks']/num_samples*100:.1f}%)")
        print(f"  Returned to channel: {s['returned_to_channel']}/{num_samples} ({s['returned_to_channel']/num_samples*100:.1f}%)")
        print(f"  Break direction - UP: {s['break_up']}/{num_samples} ({s['break_up']/num_samples*100:.1f}%)")
        print(f"  Break direction - DOWN: {s['break_down']}/{num_samples} ({s['break_down']/num_samples*100:.1f}%)")

        if s['bars_to_break']:
            avg_bars = sum(s['bars_to_break']) / len(s['bars_to_break'])
            min_bars = min(s['bars_to_break'])
            max_bars = max(s['bars_to_break'])
            print(f"  Bars to first break - Avg: {avg_bars:.1f}, Min: {min_bars}, Max: {max_bars}")

        if s['break_magnitudes']:
            avg_mag = sum(s['break_magnitudes']) / len(s['break_magnitudes'])
            min_mag = min(s['break_magnitudes'])
            max_mag = max(s['break_magnitudes'])
            print(f"  Break magnitude (std devs) - Avg: {avg_mag:.2f}, Min: {min_mag:.2f}, Max: {max_mag:.2f}")

    # Print verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    tsla_valid = stats['tsla']['valid']
    spy_valid = stats['spy']['valid']

    if tsla_valid == num_samples and spy_valid == num_samples:
        print("\n✓ FORWARD_SCAN IS WORKING PERFECTLY!")
        print(f"  ALL {num_samples} samples have valid break_scan data for both TSLA and SPY")
        print(f"  Detected breaks in {stats['tsla']['breaks_detected']} TSLA and {stats['spy']['breaks_detected']} SPY samples")
        print(f"  Forward scan is successfully identifying channel breaks and returns")
        print("\n  Key Insights:")
        print(f"    - {stats['tsla']['returned_to_channel']} TSLA samples returned to channel (false breaks)")
        print(f"    - {stats['spy']['returned_to_channel']} SPY samples returned to channel (false breaks)")
        print(f"    - {stats['tsla']['permanent_breaks']} TSLA permanent breaks")
        print(f"    - {stats['spy']['permanent_breaks']} SPY permanent breaks")
    elif tsla_valid > 0 or spy_valid > 0:
        print("\n✓ FORWARD_SCAN IS PARTIALLY WORKING")
        print(f"  Found {tsla_valid} TSLA and {spy_valid} SPY samples with valid break_scan data")
        print(f"  Missing: {num_samples - tsla_valid} TSLA, {num_samples - spy_valid} SPY")
    else:
        print("\n✗ FORWARD_SCAN IS NOT WORKING!")
        print("  No samples found with break_scan_valid=True")

    print("\n  IMPORTANT NOTE:")
    print("    Labels are stored in: sample.labels_per_window[window][asset][tf]")
    print("    They are NOT automatically added to sample.tf_features dict")
    print("    You need to manually extract them if you want to use them as features")


if __name__ == '__main__':
    generate_comprehensive_report('/Users/frank/Desktop/CodingProjects/x14/small_sample.pkl')
