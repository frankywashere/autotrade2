"""
v15scanner Python Bindings Package

High-performance C++ backend for v15 channel scanner with automatic fallback
to pure Python implementation.

Usage:
    from v15_cpp.python_bindings import scan_channels_two_pass, get_backend

    samples = scan_channels_two_pass(tsla_df, spy_df, vix_df)
    print(f"Backend: {get_backend()}")
"""

from .py_scanner import (
    scan_channels_two_pass,
    scan_channels,  # Alias
    get_backend,
    get_version,
    is_cpp_available,
    ChannelSample,
)

__all__ = [
    'scan_channels_two_pass',
    'scan_channels',
    'get_backend',
    'get_version',
    'is_cpp_available',
    'ChannelSample',
]

# Version
__version__ = '1.0.0'

# Print backend info on import
import sys
if '--quiet' not in sys.argv:
    _backend = get_backend()
    if _backend == 'cpp':
        print(f"[v15scanner] Using C++ backend (v{__version__})")
    else:
        print(f"[v15scanner] Using Python fallback (v{__version__})")
