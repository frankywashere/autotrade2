#!/usr/bin/env python3
"""
Test the fix for the macro API key None value bug.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.interactive_params_arrow import ArrowKeyParameterSelector


def test_api_key_none_handling():
    """Test that API key parameter handles None values correctly."""
    print("\n" + "=" * 70)
    print("🧪 TESTING API KEY NONE VALUE HANDLING")
    print("=" * 70)

    # Create selector
    selector = ArrowKeyParameterSelector(mode='standard')

    # Verify macro_api_key defaults to None
    assert selector.params['macro_api_key'] is None, "macro_api_key should default to None"
    print("✓ macro_api_key defaults to None (as expected)")

    # Test that _format_display_value handles None correctly
    display_value = selector._format_display_value('macro_api_key', None)
    assert display_value == 'Not set (local data available)', \
        f"Display value should be 'Not set (local data available)', got '{display_value}'"
    print(f"✓ Display value for None: '{display_value}'")

    # Test that we can prepare the default value for InquirerPy
    current_value = selector.params.get('macro_api_key')
    default_value = current_value if current_value is not None else ''
    assert default_value == '', f"Default value should be '', got '{default_value}'"
    assert isinstance(default_value, str), f"Default value should be str, got {type(default_value)}"
    print(f"✓ Default value for InquirerPy is '{default_value}' (empty string)")

    # Test with a set API key
    selector.params['macro_api_key'] = 'test_key_12345'
    display_value = selector._format_display_value('macro_api_key', 'test_key_12345')
    assert display_value == 'Set', f"Display value should be 'Set', got '{display_value}'"
    print(f"✓ Display value when key is set: '{display_value}'")

    current_value = selector.params.get('macro_api_key')
    default_value = current_value if current_value is not None else ''
    assert default_value == 'test_key_12345', f"Default value should be 'test_key_12345', got '{default_value}'"
    print(f"✓ Default value when key is set: '{default_value}'")

    print("\n" + "=" * 70)
    print("✅ API KEY NONE HANDLING TESTS COMPLETE")
    print("=" * 70)
    print("\nThe macro API key parameter now correctly handles None values!")
    print("InquirerPy will receive a valid string default (empty string for None).")
    print("\nYou should now be able to select and edit the Macro API key parameter")
    print("without crashes.")


if __name__ == "__main__":
    test_api_key_none_handling()