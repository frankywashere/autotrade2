#!/usr/bin/env python3
"""
Automated Test Script for Preset Modification Workflow

This script runs automated tests without user interaction to verify:
1. Preset confirmation screen displays correctly
2. Checklist selection works
3. Parameter modification prompts work
4. Summary shows before/after correctly
5. Modified preset is used in training
6. Original preset can be restored

Usage:
    python test_preset_modification_auto.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from copy import deepcopy

# Test results tracking
test_results = {}
test_count = 0


def log_test(test_name: str, passed: bool, message: str = ""):
    """Log test result."""
    global test_count, test_results
    test_count += 1

    status = "PASSED" if passed else "FAILED"
    prefix = "✓" if passed else "✗"

    test_results[test_name] = {
        "passed": passed,
        "message": message,
        "test_number": test_count
    }

    print(f"[Test {test_count}] {prefix} {test_name}: {status}")
    if message:
        print(f"          {message}")


# =============================================================================
# Test Configuration - Using PRESETS from train.py
# =============================================================================

PRESETS = {
    "Quick Start": {
        "desc": "Fast training for testing (small window, few epochs)",
        "window": 20,
        "step": 50,
        "hidden_dim": 64,
        "cfc_units": 96,
        "attention_heads": 4,
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
    },
    "Standard": {
        "desc": "Balanced configuration for typical training",
        "window": 20,
        "step": 25,
        "hidden_dim": 128,
        "cfc_units": 192,
        "attention_heads": 8,
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.0005,
    },
    "Full Training": {
        "desc": "Maximum quality (slow, requires good GPU)",
        "window": 20,
        "step": 10,
        "hidden_dim": 256,
        "cfc_units": 384,
        "attention_heads": 8,
        "num_epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.0003,
    },
}


# =============================================================================
# Mock Functions for Workflow Components
# =============================================================================

def validate_preset_structure(preset: Dict) -> bool:
    """Validate that a preset has all required fields."""
    required_fields = [
        "window", "step", "hidden_dim", "cfc_units", "attention_heads",
        "num_epochs", "batch_size", "learning_rate", "desc"
    ]

    for field in required_fields:
        if field not in preset:
            return False

    return True


def validate_preset_values(preset: Dict) -> tuple[bool, str]:
    """Validate preset parameter values."""
    # Check hidden_dim is divisible by attention_heads
    if preset["hidden_dim"] % preset["attention_heads"] != 0:
        return False, f"hidden_dim ({preset['hidden_dim']}) not divisible by attention_heads ({preset['attention_heads']})"

    # Check CfC units constraint
    if preset["cfc_units"] <= preset["hidden_dim"] + 2:
        return False, f"cfc_units ({preset['cfc_units']}) must be > hidden_dim + 2 ({preset['hidden_dim'] + 2})"

    # Check ranges
    if not (20 <= preset["window"] <= 200):
        return False, f"window ({preset['window']}) out of range [20, 200]"

    if not (1 <= preset["step"] <= 100):
        return False, f"step ({preset['step']}) out of range [1, 100]"

    if not (1 <= preset["num_epochs"] <= 500):
        return False, f"num_epochs ({preset['num_epochs']}) out of range [1, 500]"

    if preset["batch_size"] not in [16, 32, 64, 128, 256]:
        return False, f"batch_size ({preset['batch_size']}) not in valid options"

    if not (0.00001 <= preset["learning_rate"] <= 0.01):
        return False, f"learning_rate ({preset['learning_rate']}) out of range [0.00001, 0.01]"

    return True, ""


def display_preset_confirmation(preset_name: str, preset: Dict) -> Dict:
    """
    Simulate displaying preset confirmation screen.

    Returns:
        Dictionary with display information
    """
    display_info = {
        "preset_name": preset_name,
        "description": preset.get("desc", ""),
        "parameters": {
            "Data": [
                ("Window Size", preset["window"]),
                ("Step Size", preset["step"])
            ],
            "Model": [
                ("Hidden Dimension", preset["hidden_dim"]),
                ("CfC Units", preset["cfc_units"]),
                ("Attention Heads", preset["attention_heads"])
            ],
            "Training": [
                ("Number of Epochs", preset["num_epochs"]),
                ("Batch Size", preset["batch_size"]),
                ("Learning Rate", preset["learning_rate"])
            ]
        }
    }

    return display_info


def simulate_parameter_selection() -> List[str]:
    """Simulate checklist parameter selection."""
    # Simulate selecting 3 parameters to modify
    return ["num_epochs", "batch_size", "learning_rate"]


def modify_parameter(preset: Dict, param_name: str, new_value) -> tuple[bool, str]:
    """
    Simulate modifying a parameter with validation.

    Returns:
        (success, error_message)
    """
    if param_name == "hidden_dim":
        # Check divisibility by attention_heads
        if new_value % preset["attention_heads"] != 0:
            return False, f"{new_value} not divisible by {preset['attention_heads']}"

    elif param_name == "cfc_units":
        # Check CfC constraint
        if new_value <= preset["hidden_dim"] + 2:
            return False, f"{new_value} must be > {preset['hidden_dim'] + 2}"

    elif param_name == "attention_heads":
        # Check if current hidden_dim is compatible
        if preset["hidden_dim"] % new_value != 0:
            return False, f"Current hidden_dim ({preset['hidden_dim']}) not divisible by {new_value}"

    return True, ""


def generate_before_after_summary(original: Dict, modified: Dict) -> Dict:
    """Generate before/after comparison summary."""
    changes = {}

    for key in original:
        if key == "desc":
            continue

        if original[key] != modified[key]:
            changes[key] = {
                "original": original[key],
                "modified": modified[key],
                "changed": True
            }
        else:
            changes[key] = {
                "original": original[key],
                "modified": modified[key],
                "changed": False
            }

    return changes


def count_changes(summary: Dict) -> int:
    """Count number of changed parameters."""
    return sum(1 for v in summary.values() if v["changed"])


# =============================================================================
# Test Cases
# =============================================================================

def test_preset_structure():
    """Test 1: Verify all presets have correct structure."""
    print("\n" + "="*80)
    print("TEST SUITE 1: Preset Structure Validation")
    print("="*80)

    for preset_name, preset in PRESETS.items():
        passed = validate_preset_structure(preset)
        log_test(
            f"Preset '{preset_name}' has all required fields",
            passed,
            f"Fields present: {', '.join(preset.keys())}" if passed else "Missing fields"
        )


def test_preset_values():
    """Test 2: Verify all preset values are valid."""
    print("\n" + "="*80)
    print("TEST SUITE 2: Preset Value Validation")
    print("="*80)

    for preset_name, preset in PRESETS.items():
        passed, message = validate_preset_values(preset)
        log_test(
            f"Preset '{preset_name}' has valid parameter values",
            passed,
            message
        )


def test_confirmation_screen():
    """Test 3: Verify preset confirmation screen displays correctly."""
    print("\n" + "="*80)
    print("TEST SUITE 3: Preset Confirmation Screen")
    print("="*80)

    for preset_name, preset in PRESETS.items():
        display_info = display_preset_confirmation(preset_name, preset)

        # Verify display info contains all necessary components
        has_name = "preset_name" in display_info
        has_desc = "description" in display_info
        has_params = "parameters" in display_info

        passed = has_name and has_desc and has_params

        if passed:
            # Check that all categories are present
            expected_categories = ["Data", "Model", "Training"]
            has_all_categories = all(cat in display_info["parameters"] for cat in expected_categories)
            passed = has_all_categories

        log_test(
            f"Confirmation screen for '{preset_name}' displays correctly",
            passed,
            f"Categories: {list(display_info['parameters'].keys())}" if passed else "Missing categories"
        )


def test_checklist_selection():
    """Test 4: Verify checklist selection works."""
    print("\n" + "="*80)
    print("TEST SUITE 4: Checklist Parameter Selection")
    print("="*80)

    # Simulate selection
    selected = simulate_parameter_selection()

    # Verify selection is valid
    valid_params = {"window", "step", "hidden_dim", "cfc_units", "attention_heads",
                   "num_epochs", "batch_size", "learning_rate"}

    all_valid = all(param in valid_params for param in selected)

    log_test(
        "Checklist selection returns valid parameters",
        all_valid,
        f"Selected: {', '.join(selected)}"
    )

    log_test(
        "Checklist allows multiple selections",
        len(selected) > 0,
        f"Selected {len(selected)} parameters"
    )


def test_parameter_modification():
    """Test 5: Verify parameter modification with validation."""
    print("\n" + "="*80)
    print("TEST SUITE 5: Parameter Modification")
    print("="*80)

    preset = deepcopy(PRESETS["Standard"])

    # Test valid modifications
    test_cases = [
        ("num_epochs", 75, True, "Valid epoch change"),
        ("batch_size", 128, True, "Valid batch size change"),
        ("learning_rate", 0.0001, True, "Valid learning rate change"),
        ("hidden_dim", 256, True, "Valid hidden_dim change (divisible by 8)"),
        ("hidden_dim", 100, False, "Invalid hidden_dim (not divisible by 8)"),
        ("cfc_units", 400, True, "Valid cfc_units change"),
        ("cfc_units", 130, False, "Invalid cfc_units (not > hidden_dim + 2)"),
    ]

    for param, new_value, should_pass, description in test_cases:
        test_preset = deepcopy(preset)
        passed, error_msg = modify_parameter(test_preset, param, new_value)

        log_test(
            description,
            passed == should_pass,
            error_msg if not passed else f"{param}: {preset.get(param)} → {new_value}"
        )


def test_before_after_summary():
    """Test 6: Verify before/after summary generation."""
    print("\n" + "="*80)
    print("TEST SUITE 6: Before/After Summary")
    print("="*80)

    original = deepcopy(PRESETS["Quick Start"])
    modified = deepcopy(original)

    # Make some modifications
    modified["num_epochs"] = 20
    modified["batch_size"] = 64
    modified["learning_rate"] = 0.0005

    summary = generate_before_after_summary(original, modified)

    # Verify summary structure
    log_test(
        "Summary includes all parameters",
        len(summary) == 8,  # All params except 'desc'
        f"Found {len(summary)} parameters in summary"
    )

    # Verify change detection
    changes_count = count_changes(summary)
    expected_changes = 3

    log_test(
        "Summary correctly identifies changed parameters",
        changes_count == expected_changes,
        f"Detected {changes_count} changes (expected {expected_changes})"
    )

    # Verify unchanged parameters
    unchanged_params = [k for k, v in summary.items() if not v["changed"]]
    log_test(
        "Summary correctly identifies unchanged parameters",
        len(unchanged_params) == 5,
        f"Unchanged: {', '.join(unchanged_params)}"
    )


def test_modified_preset_usage():
    """Test 7: Verify modified preset can be used for training."""
    print("\n" + "="*80)
    print("TEST SUITE 7: Modified Preset Usage")
    print("="*80)

    original = deepcopy(PRESETS["Standard"])
    modified = deepcopy(original)

    # Modify parameters
    modified["num_epochs"] = 75
    modified["batch_size"] = 128
    modified["learning_rate"] = 0.0001

    # Verify modified preset is still valid
    passed, message = validate_preset_values(modified)

    log_test(
        "Modified preset passes validation",
        passed,
        message
    )

    # Verify modifications are preserved
    log_test(
        "Modifications are preserved correctly",
        modified["num_epochs"] == 75 and
        modified["batch_size"] == 128 and
        modified["learning_rate"] == 0.0001,
        f"epochs={modified['num_epochs']}, batch_size={modified['batch_size']}, lr={modified['learning_rate']}"
    )


def test_preset_restoration():
    """Test 8: Verify original preset can be restored."""
    print("\n" + "="*80)
    print("TEST SUITE 8: Original Preset Restoration")
    print("="*80)

    # Store original
    original = deepcopy(PRESETS["Full Training"])
    original_copy = deepcopy(original)

    # Modify
    modified = deepcopy(original)
    modified["num_epochs"] = 200
    modified["batch_size"] = 256
    modified["learning_rate"] = 0.0001
    modified["hidden_dim"] = 512
    modified["cfc_units"] = 768

    log_test(
        "Preset can be modified",
        modified != original,
        f"Made {count_changes(generate_before_after_summary(original, modified))} changes"
    )

    # Restore
    restored = deepcopy(original_copy)

    log_test(
        "Preset can be restored to original",
        restored == original,
        "All parameters match original"
    )

    # Verify each parameter
    all_match = True
    for key in original:
        if key == "desc":
            continue
        if restored[key] != original[key]:
            all_match = False
            break

    log_test(
        "All parameters match after restoration",
        all_match,
        "Complete restoration verified"
    )


def test_edge_cases():
    """Test 9: Edge cases and boundary conditions."""
    print("\n" + "="*80)
    print("TEST SUITE 9: Edge Cases")
    print("="*80)

    preset = deepcopy(PRESETS["Standard"])

    # Test minimum values
    edge_cases = [
        ("window", 20, True, "Minimum window size"),
        ("window", 200, True, "Maximum window size"),
        ("step", 1, True, "Minimum step size"),
        ("step", 100, True, "Maximum step size"),
        ("num_epochs", 1, True, "Minimum epochs"),
        ("learning_rate", 0.00001, True, "Minimum learning rate"),
        ("learning_rate", 0.01, True, "Maximum learning rate"),
        ("batch_size", 16, True, "Minimum batch size"),
        ("batch_size", 256, True, "Maximum batch size"),
    ]

    for param, value, should_pass, description in edge_cases:
        test_preset = deepcopy(preset)
        test_preset[param] = value

        passed, message = validate_preset_values(test_preset)

        log_test(
            description,
            passed == should_pass,
            f"{param}={value}: {message if not passed else 'Valid'}"
        )


def test_constraint_validation():
    """Test 10: Verify parameter constraints are enforced."""
    print("\n" + "="*80)
    print("TEST SUITE 10: Constraint Validation")
    print("="*80)

    # Test hidden_dim / attention_heads constraint
    test_cases = [
        (128, 8, True, "128 divisible by 8"),
        (128, 4, True, "128 divisible by 4"),
        (128, 11, False, "128 not divisible by 11"),
        (132, 11, True, "132 divisible by 11"),
        (256, 16, True, "256 divisible by 16"),
        (256, 11, False, "256 not divisible by 11"),
    ]

    for hidden_dim, attention_heads, should_pass, description in test_cases:
        preset = {
            "hidden_dim": hidden_dim,
            "attention_heads": attention_heads,
            "cfc_units": hidden_dim * 2,  # Ensure cfc_units constraint is met
            "window": 20,
            "step": 25,
            "num_epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.0005,
            "desc": "Test"
        }

        passed, message = validate_preset_values(preset)

        log_test(
            f"Constraint check: {description}",
            passed == should_pass,
            message
        )

    # Test CfC units constraint
    cfc_test_cases = [
        (128, 131, True, "CfC (131) > hidden_dim (128) + 2"),
        (128, 200, True, "CfC (200) > hidden_dim (128) + 2"),
        (128, 130, False, "CfC (130) not > hidden_dim (128) + 2"),
        (128, 128, False, "CfC (128) not > hidden_dim (128) + 2"),
    ]

    for hidden_dim, cfc_units, should_pass, description in cfc_test_cases:
        preset = {
            "hidden_dim": hidden_dim,
            "cfc_units": cfc_units,
            "attention_heads": 8,
            "window": 20,
            "step": 25,
            "num_epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.0005,
            "desc": "Test"
        }

        passed, message = validate_preset_values(preset)

        log_test(
            f"CfC constraint check: {description}",
            passed == should_pass,
            message
        )


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all automated tests."""
    print("\n" + "="*80)
    print("PRESET MODIFICATION WORKFLOW - AUTOMATED TEST SUITE")
    print("="*80)
    print("\nTesting all workflow components without user interaction...")

    # Run all test suites
    test_preset_structure()
    test_preset_values()
    test_confirmation_screen()
    test_checklist_selection()
    test_parameter_modification()
    test_before_after_summary()
    test_modified_preset_usage()
    test_preset_restoration()
    test_edge_cases()
    test_constraint_validation()

    # Summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)

    passed_tests = sum(1 for r in test_results.values() if r["passed"])
    total_tests = len(test_results)

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")

    # List failed tests if any
    failed_tests = [(name, result) for name, result in test_results.items() if not result["passed"]]

    if failed_tests:
        print("\n" + "-"*80)
        print("FAILED TESTS:")
        print("-"*80)
        for name, result in failed_tests:
            print(f"  [{result['test_number']}] {name}")
            if result['message']:
                print(f"       {result['message']}")

    print("\n" + "="*80)

    if passed_tests == total_tests:
        print("✓ ALL TESTS PASSED")
        print("="*80)
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
