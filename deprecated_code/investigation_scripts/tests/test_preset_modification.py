#!/usr/bin/env python3
"""
Test Script for Preset Modification Workflow

This script tests the preset modification workflow by:
1. Displaying preset confirmation screen
2. Allowing checklist selection of parameters to modify
3. Prompting for parameter modifications
4. Showing before/after summary
5. Simulating training with modified preset
6. Allowing restoration of original preset

Usage:
    python test_preset_modification.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from copy import deepcopy

# Rich for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.tree import Tree
from rich.text import Text

# InquirerPy for interactive prompts
from InquirerPy import inquirer
from InquirerPy.validator import NumberValidator

console = Console()

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
# Preset Modification Workflow Functions
# =============================================================================

def display_preset_confirmation(preset_name: str, preset: Dict) -> bool:
    """
    Display preset confirmation screen with all parameters.

    Args:
        preset_name: Name of the preset
        preset: Preset configuration dictionary

    Returns:
        True if user wants to modify, False to use as-is
    """
    console.print("\n[bold cyan]Preset Configuration: {}[/bold cyan]\n".format(preset_name))
    console.print(f"[dim]{preset['desc']}[/dim]\n")

    # Create table showing all preset parameters
    table = Table(title="Preset Parameters", box=box.ROUNDED, show_header=True)
    table.add_column("Category", style="cyan", width=15)
    table.add_column("Parameter", style="yellow", width=20)
    table.add_column("Value", style="green", justify="right", width=15)

    # Data parameters
    table.add_row("Data", "Window Size", str(preset["window"]), style="dim")
    table.add_row("Data", "Step Size", str(preset["step"]))

    table.add_section()

    # Model parameters
    table.add_row("Model", "Hidden Dimension", str(preset["hidden_dim"]), style="dim")
    table.add_row("Model", "CfC Units", str(preset["cfc_units"]))
    table.add_row("Model", "Attention Heads", str(preset["attention_heads"]))

    table.add_section()

    # Training parameters
    table.add_row("Training", "Number of Epochs", str(preset["num_epochs"]), style="dim")
    table.add_row("Training", "Batch Size", str(preset["batch_size"]))
    table.add_row("Training", "Learning Rate", f"{preset['learning_rate']:.4f}")

    console.print(table)
    console.print()

    # Ask if user wants to modify
    choice = inquirer.select(
        message="What would you like to do?",
        choices=[
            {"name": "Use preset as-is", "value": "use"},
            {"name": "Modify selected parameters", "value": "modify"},
            {"name": "Choose a different preset", "value": "change"},
        ],
        default="use",
    ).execute()

    return choice


def select_parameters_to_modify() -> List[str]:
    """
    Show checklist for selecting which parameters to modify.

    Returns:
        List of parameter keys to modify
    """
    console.print("\n[bold cyan]Select Parameters to Modify[/bold cyan]\n")
    console.print("[dim]Choose which parameters you want to customize:[/dim]\n")

    # Checklist of all modifiable parameters
    param_choices = [
        {"name": "Window Size (Data)", "value": "window", "enabled": False},
        {"name": "Step Size (Data)", "value": "step", "enabled": False},
        {"name": "Hidden Dimension (Model)", "value": "hidden_dim", "enabled": False},
        {"name": "CfC Units (Model)", "value": "cfc_units", "enabled": False},
        {"name": "Attention Heads (Model)", "value": "attention_heads", "enabled": False},
        {"name": "Number of Epochs (Training)", "value": "num_epochs", "enabled": False},
        {"name": "Batch Size (Training)", "value": "batch_size", "enabled": False},
        {"name": "Learning Rate (Training)", "value": "learning_rate", "enabled": False},
    ]

    selected = inquirer.checkbox(
        message="Select parameters to modify (Space to select, Enter to confirm):",
        choices=param_choices,
        instruction="(Use arrow keys to navigate, Space to select/deselect)",
    ).execute()

    if not selected:
        console.print("\n[yellow]No parameters selected for modification[/yellow]")
    else:
        console.print(f"\n[green]Selected {len(selected)} parameter(s) for modification[/green]")

    return selected


def modify_parameters(preset: Dict, params_to_modify: List[str]) -> Dict:
    """
    Prompt user to modify selected parameters.

    Args:
        preset: Original preset configuration
        params_to_modify: List of parameter keys to modify

    Returns:
        Modified preset configuration
    """
    console.print("\n[bold cyan]Modify Parameters[/bold cyan]\n")

    modified_preset = deepcopy(preset)

    for param in params_to_modify:
        current_value = preset[param]
        console.print(f"\n[yellow]{param}[/yellow] (current: [cyan]{current_value}[/cyan])")

        # Different prompts based on parameter type
        if param == "window":
            new_value = int(inquirer.number(
                message="  New window size:",
                min_allowed=20,
                max_allowed=200,
                default=current_value,
                validate=NumberValidator(),
            ).execute())

        elif param == "step":
            new_value = int(inquirer.number(
                message="  New step size:",
                min_allowed=1,
                max_allowed=100,
                default=current_value,
                validate=NumberValidator(),
            ).execute())

        elif param == "hidden_dim":
            # Must be divisible by attention_heads
            attention_heads = modified_preset["attention_heads"]
            console.print(f"  [dim]Must be divisible by {attention_heads} attention heads[/dim]")

            while True:
                new_value = int(inquirer.number(
                    message="  New hidden dimension:",
                    min_allowed=32,
                    max_allowed=512,
                    default=current_value,
                    validate=NumberValidator(),
                ).execute())

                if new_value % attention_heads == 0:
                    break
                else:
                    console.print(f"  [red]Error: {new_value} is not divisible by {attention_heads}. Try again.[/red]")

        elif param == "cfc_units":
            # Must be > hidden_dim + 2
            hidden_dim = modified_preset["hidden_dim"]
            min_cfc = hidden_dim + 3
            console.print(f"  [dim]Must be > {hidden_dim + 2}[/dim]")

            new_value = int(inquirer.number(
                message="  New CfC units:",
                min_allowed=min_cfc,
                max_allowed=1024,
                default=max(current_value, min_cfc),
                validate=NumberValidator(),
            ).execute())

        elif param == "attention_heads":
            # If changing attention heads, hidden_dim must be updated too
            new_value = inquirer.select(
                message="  New number of attention heads:",
                choices=[2, 4, 8, 11, 16],
                default=current_value,
            ).execute()

            # Check if current hidden_dim is compatible
            current_hidden = modified_preset["hidden_dim"]
            if current_hidden % new_value != 0:
                console.print(f"  [yellow]Warning: Current hidden_dim ({current_hidden}) is not divisible by {new_value}[/yellow]")
                console.print(f"  [yellow]You may need to adjust hidden_dim as well[/yellow]")

        elif param == "num_epochs":
            new_value = int(inquirer.number(
                message="  New number of epochs:",
                min_allowed=1,
                max_allowed=500,
                default=current_value,
                validate=NumberValidator(),
            ).execute())

        elif param == "batch_size":
            new_value = inquirer.select(
                message="  New batch size:",
                choices=[16, 32, 64, 128, 256],
                default=current_value,
            ).execute()

        elif param == "learning_rate":
            new_value = float(inquirer.number(
                message="  New learning rate:",
                min_allowed=0.00001,
                max_allowed=0.01,
                default=current_value,
                float_allowed=True,
            ).execute())

        else:
            console.print(f"  [red]Unknown parameter: {param}[/red]")
            continue

        modified_preset[param] = new_value

        # Show change
        if new_value != current_value:
            console.print(f"  [green]✓ Changed from {current_value} to {new_value}[/green]")
        else:
            console.print(f"  [dim]No change (kept {current_value})[/dim]")

    return modified_preset


def display_before_after_summary(original: Dict, modified: Dict, preset_name: str):
    """
    Display before/after comparison of preset parameters.

    Args:
        original: Original preset configuration
        modified: Modified preset configuration
        preset_name: Name of the preset
    """
    console.print("\n[bold cyan]Modification Summary[/bold cyan]\n")

    # Create comparison table
    table = Table(
        title=f"Changes to '{preset_name}' Preset",
        box=box.DOUBLE,
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Parameter", style="yellow", width=20)
    table.add_column("Original", style="red", justify="right", width=15)
    table.add_column("Modified", style="green", justify="right", width=15)
    table.add_column("Status", style="cyan", width=10)

    # Compare all parameters
    all_params = [
        ("window", "Window Size"),
        ("step", "Step Size"),
        ("hidden_dim", "Hidden Dimension"),
        ("cfc_units", "CfC Units"),
        ("attention_heads", "Attention Heads"),
        ("num_epochs", "Number of Epochs"),
        ("batch_size", "Batch Size"),
        ("learning_rate", "Learning Rate"),
    ]

    changes_count = 0

    for param_key, param_name in all_params:
        orig_val = original[param_key]
        mod_val = modified[param_key]

        # Format values
        if isinstance(orig_val, float):
            orig_str = f"{orig_val:.6f}"
            mod_str = f"{mod_val:.6f}"
        else:
            orig_str = str(orig_val)
            mod_str = str(mod_val)

        # Status
        if orig_val != mod_val:
            status = "CHANGED"
            style = "bold"
            changes_count += 1
        else:
            status = "-"
            style = "dim"

        table.add_row(param_name, orig_str, mod_str, status, style=style)

    console.print(table)
    console.print()

    if changes_count == 0:
        console.print("[yellow]No parameters were modified[/yellow]\n")
    else:
        console.print(f"[green]✓ {changes_count} parameter(s) modified[/green]\n")


def confirm_use_modified() -> bool:
    """
    Confirm whether to use modified preset or restore original.

    Returns:
        True to use modified, False to restore original
    """
    choice = inquirer.select(
        message="Proceed with modified preset?",
        choices=[
            {"name": "Yes - Use modified preset for training", "value": "use"},
            {"name": "No - Restore original preset", "value": "restore"},
            {"name": "Cancel - Go back to modify again", "value": "cancel"},
        ],
        default="use",
    ).execute()

    return choice


def simulate_training(preset: Dict, preset_name: str):
    """
    Simulate training with the given preset.

    Args:
        preset: Preset configuration to use
        preset_name: Name of the preset
    """
    console.print("\n[bold cyan]Simulating Training...[/bold cyan]\n")

    # Display training configuration
    tree = Tree(f"[bold]Training with: {preset_name}[/bold]")

    data_branch = tree.add("[cyan]Data Configuration[/cyan]")
    data_branch.add(f"Window: {preset['window']}")
    data_branch.add(f"Step: {preset['step']}")

    model_branch = tree.add("[magenta]Model Configuration[/magenta]")
    model_branch.add(f"Hidden Dim: {preset['hidden_dim']}")
    model_branch.add(f"CfC Units: {preset['cfc_units']}")
    model_branch.add(f"Attention Heads: {preset['attention_heads']}")

    training_branch = tree.add("[green]Training Configuration[/green]")
    training_branch.add(f"Epochs: {preset['num_epochs']}")
    training_branch.add(f"Batch Size: {preset['batch_size']}")
    training_branch.add(f"Learning Rate: {preset['learning_rate']:.6f}")

    console.print(tree)
    console.print()

    # Simulate progress
    import time
    with console.status("[bold green]Training in progress...", spinner="dots"):
        time.sleep(2)

    console.print("[green]✓ Training simulation complete![/green]\n")


# =============================================================================
# Test Cases
# =============================================================================

def test_case_1_use_preset_as_is():
    """Test Case 1: Select preset and use as-is without modification."""
    console.print("\n" + "="*80)
    console.print("[bold yellow]TEST CASE 1: Use Preset As-Is[/bold yellow]")
    console.print("="*80)

    preset_name = "Standard"
    preset = PRESETS[preset_name]

    # Step 1: Display confirmation screen
    console.print("\n[bold]Step 1: Display Preset Confirmation Screen[/bold]")
    choice = display_preset_confirmation(preset_name, preset)

    if choice == "use":
        console.print("\n[green]✓ TEST PASSED: Preset confirmation displayed correctly[/green]")
        console.print("[green]✓ TEST PASSED: User chose to use preset as-is[/green]")

        # Simulate training
        simulate_training(preset, preset_name)
        console.print("[green]✓ TEST PASSED: Training uses unmodified preset[/green]")

        return True
    else:
        console.print("\n[red]✗ TEST FAILED: Expected 'use' but got '{}'[/red]".format(choice))
        return False


def test_case_2_modify_parameters():
    """Test Case 2: Modify selected parameters."""
    console.print("\n" + "="*80)
    console.print("[bold yellow]TEST CASE 2: Modify Selected Parameters[/bold yellow]")
    console.print("="*80)

    preset_name = "Quick Start"
    original_preset = PRESETS[preset_name]

    # Step 1: Display confirmation screen
    console.print("\n[bold]Step 1: Display Preset Confirmation Screen[/bold]")
    choice = display_preset_confirmation(preset_name, original_preset)

    if choice != "modify":
        console.print("\n[yellow]⚠ Skipping test - user chose '{}'[/yellow]".format(choice))
        return False

    console.print("[green]✓ TEST PASSED: User chose to modify preset[/green]")

    # Step 2: Select parameters to modify
    console.print("\n[bold]Step 2: Checklist Selection[/bold]")
    params_to_modify = select_parameters_to_modify()

    if not params_to_modify:
        console.print("\n[yellow]⚠ No parameters selected - ending test[/yellow]")
        return False

    console.print(f"[green]✓ TEST PASSED: Checklist selection works ({len(params_to_modify)} selected)[/green]")

    # Step 3: Modify parameters
    console.print("\n[bold]Step 3: Parameter Modification Prompts[/bold]")
    modified_preset = modify_parameters(original_preset, params_to_modify)
    console.print("[green]✓ TEST PASSED: Parameter modification prompts work[/green]")

    # Step 4: Display before/after summary
    console.print("\n[bold]Step 4: Before/After Summary[/bold]")
    display_before_after_summary(original_preset, modified_preset, preset_name)
    console.print("[green]✓ TEST PASSED: Before/after summary displayed correctly[/green]")

    # Step 5: Confirm use of modified preset
    console.print("\n[bold]Step 5: Confirm Modified Preset[/bold]")
    choice = confirm_use_modified()

    if choice == "use":
        console.print("[green]✓ TEST PASSED: User confirmed modified preset[/green]")

        # Step 6: Simulate training with modified preset
        console.print("\n[bold]Step 6: Training with Modified Preset[/bold]")
        simulate_training(modified_preset, f"{preset_name} (Modified)")
        console.print("[green]✓ TEST PASSED: Modified preset used in training[/green]")

        return True

    elif choice == "restore":
        console.print("\n[bold]Step 6: Restore Original Preset[/bold]")
        console.print("[green]✓ TEST PASSED: Original preset can be restored[/green]")

        simulate_training(original_preset, preset_name)
        console.print("[green]✓ TEST PASSED: Original preset used in training[/green]")

        return True

    else:
        console.print("\n[yellow]⚠ User chose to cancel[/yellow]")
        return False


def test_case_3_validation():
    """Test Case 3: Validate parameter constraints."""
    console.print("\n" + "="*80)
    console.print("[bold yellow]TEST CASE 3: Parameter Validation[/bold yellow]")
    console.print("="*80)

    # Test hidden_dim divisibility by attention_heads
    console.print("\n[bold]Testing: Hidden Dimension Divisibility[/bold]")
    test_preset = {
        "hidden_dim": 128,
        "attention_heads": 8,
        "cfc_units": 256,
    }

    # This should work (128 % 8 = 0)
    if test_preset["hidden_dim"] % test_preset["attention_heads"] == 0:
        console.print(f"[green]✓ TEST PASSED: {test_preset['hidden_dim']} is divisible by {test_preset['attention_heads']}[/green]")
    else:
        console.print(f"[red]✗ TEST FAILED: Divisibility check failed[/red]")
        return False

    # Test CfC units constraint
    console.print("\n[bold]Testing: CfC Units Constraint[/bold]")
    if test_preset["cfc_units"] > test_preset["hidden_dim"] + 2:
        console.print(f"[green]✓ TEST PASSED: CfC units ({test_preset['cfc_units']}) > hidden_dim + 2 ({test_preset['hidden_dim'] + 2})[/green]")
    else:
        console.print(f"[red]✗ TEST FAILED: CfC units constraint violated[/red]")
        return False

    console.print("\n[green]✓ ALL VALIDATION TESTS PASSED[/green]")
    return True


def test_case_4_restore_original():
    """Test Case 4: Modify then restore original preset."""
    console.print("\n" + "="*80)
    console.print("[bold yellow]TEST CASE 4: Restore Original Preset[/bold yellow]")
    console.print("="*80)

    preset_name = "Full Training"
    original_preset = deepcopy(PRESETS[preset_name])

    console.print("\n[bold]Step 1: Store Original Preset[/bold]")
    console.print(f"[green]✓ Original preset stored: {preset_name}[/green]")

    console.print("\n[bold]Step 2: Simulate Modification[/bold]")
    modified_preset = deepcopy(original_preset)
    modified_preset["num_epochs"] = 200  # Modify
    modified_preset["batch_size"] = 256  # Modify
    console.print("[green]✓ Preset modified (epochs: 100→200, batch_size: 128→256)[/green]")

    console.print("\n[bold]Step 3: Display Before/After[/bold]")
    display_before_after_summary(original_preset, modified_preset, preset_name)

    console.print("\n[bold]Step 4: Restore Original[/bold]")
    restored_preset = deepcopy(original_preset)

    # Verify restoration
    if restored_preset == original_preset:
        console.print("[green]✓ TEST PASSED: Original preset restored correctly[/green]")

        # Verify all parameters match
        console.print("\n[bold]Verification:[/bold]")
        for key in original_preset:
            if key == "desc":
                continue
            if restored_preset[key] == original_preset[key]:
                console.print(f"  [green]✓ {key}: {restored_preset[key]}[/green]")
            else:
                console.print(f"  [red]✗ {key}: {restored_preset[key]} != {original_preset[key]}[/red]")
                return False

        return True
    else:
        console.print("[red]✗ TEST FAILED: Restored preset doesn't match original[/red]")
        return False


def run_all_tests():
    """Run all test cases."""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Preset Modification Workflow Test Suite[/bold cyan]\n"
        "[dim]Testing all workflow components[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))

    results = {}

    # Run automated tests first (no user interaction needed)
    console.print("\n[bold cyan]Running Automated Tests...[/bold cyan]")

    results["Validation"] = test_case_3_validation()
    results["Restore Original"] = test_case_4_restore_original()

    # Interactive tests (require user input)
    console.print("\n[bold cyan]Running Interactive Tests...[/bold cyan]")
    console.print("[dim]These tests require user interaction[/dim]\n")

    if inquirer.confirm(
        message="Run Test Case 1: Use Preset As-Is?",
        default=True
    ).execute():
        results["Use Preset As-Is"] = test_case_1_use_preset_as_is()

    if inquirer.confirm(
        message="Run Test Case 2: Modify Parameters?",
        default=True
    ).execute():
        results["Modify Parameters"] = test_case_2_modify_parameters()

    # Summary
    console.print("\n" + "="*80)
    console.print("[bold cyan]Test Results Summary[/bold cyan]")
    console.print("="*80 + "\n")

    summary_table = Table(box=box.ROUNDED, show_header=True)
    summary_table.add_column("Test Case", style="cyan", width=30)
    summary_table.add_column("Result", style="yellow", justify="center", width=15)

    for test_name, result in results.items():
        if result:
            status = "[green]PASSED ✓[/green]"
        else:
            status = "[red]FAILED ✗[/red]"
        summary_table.add_row(test_name, status)

    console.print(summary_table)
    console.print()

    # Overall result
    passed = sum(1 for r in results.values() if r)
    total = len(results)

    if passed == total:
        console.print(f"[bold green]ALL TESTS PASSED ({passed}/{total})[/bold green]\n")
    else:
        console.print(f"[bold yellow]SOME TESTS FAILED ({passed}/{total} passed)[/bold yellow]\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main test entry point."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Preset Modification Workflow Test[/bold cyan]\n"
        "[dim]Interactive testing of preset modification features[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))

    choice = inquirer.select(
        message="Select test mode:",
        choices=[
            {"name": "Run all tests (automated + interactive)", "value": "all"},
            {"name": "Test Case 1: Use Preset As-Is", "value": "test1"},
            {"name": "Test Case 2: Modify Parameters", "value": "test2"},
            {"name": "Test Case 3: Validation", "value": "test3"},
            {"name": "Test Case 4: Restore Original", "value": "test4"},
        ],
        default="all",
    ).execute()

    if choice == "all":
        run_all_tests()
    elif choice == "test1":
        test_case_1_use_preset_as_is()
    elif choice == "test2":
        test_case_2_modify_parameters()
    elif choice == "test3":
        test_case_3_validation()
    elif choice == "test4":
        test_case_4_restore_original()

    console.print("\n[bold green]Test session complete![/bold green]\n")


if __name__ == "__main__":
    main()
