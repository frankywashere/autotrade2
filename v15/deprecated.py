"""
V15 Deprecation Notices

This module contains deprecation warnings for old v7 code that has been
superseded by v15.

Import this at the start of old modules to show deprecation warnings.
"""
import warnings
import functools
from typing import Callable, Any


class V15DeprecationWarning(DeprecationWarning):
    """Warning for code deprecated by v15."""
    pass


# Always show our deprecation warnings
warnings.filterwarnings('always', category=V15DeprecationWarning)


def deprecated(reason: str, replacement: str = None):
    """
    Decorator to mark functions/classes as deprecated.

    Usage:
        @deprecated("Use v15.models.V15Model instead", replacement="v15.models.V15Model")
        def old_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            msg = f"{func.__module__}.{func.__name__} is deprecated: {reason}"
            if replacement:
                msg += f" Use {replacement} instead."
            warnings.warn(msg, V15DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        # Add deprecation notice to docstring
        doc = func.__doc__ or ""
        wrapper.__doc__ = f"DEPRECATED: {reason}\n\n{doc}"

        return wrapper
    return decorator


def deprecated_module(module_name: str, replacement: str):
    """
    Call at module level to warn that entire module is deprecated.

    Usage (in old module):
        from v15.deprecated import deprecated_module
        deprecated_module(__name__, "v15.models")
    """
    msg = f"Module {module_name} is deprecated. Use {replacement} instead."
    warnings.warn(msg, V15DeprecationWarning, stacklevel=2)


# List of deprecated modules/functions
DEPRECATED_ITEMS = {
    # Old v7 training
    'v7.training.trainer.Trainer': 'v15.training.Trainer',
    'v7.training.dataset.ChannelDataset': 'v15.training.ChannelDataset',

    # Old v7 models
    'v7.models.hierarchical_cfc.HierarchicalCfC': 'v15.models.V15Model',
    'v7.models.end_to_end_window_model': 'v15.models.V15Model',

    # Old v7 features
    'v7.features.full_features.extract_full_features': 'v15.features.extract_all_features',
    'v7.features.feature_ordering': 'v15.config',

    # Old dashboard
    'dashboard.py': 'v15.dashboard',
    'streamlit_app.py': 'v15.dashboard',
}


def print_deprecation_guide():
    """Print migration guide from v7 to v15."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    V7 → V15 Migration Guide                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  MODELS                                                           ║
║  -------                                                          ║
║  OLD: v7.models.HierarchicalCfC (776 features)                   ║
║  NEW: v15.models.V15Model (14,840 features via C++ scanner)      ║
║                                                                   ║
║  FEATURES                                                         ║
║  ---------                                                        ║
║  OLD: v7.features.extract_full_features() → 776 features         ║
║  NEW: C++ scanner extracts 14,840 features per sample            ║
║                                                                   ║
║  TRAINING                                                         ║
║  ---------                                                        ║
║  OLD: v7.training.Trainer                                        ║
║  NEW: v15.training.Trainer                                       ║
║                                                                   ║
║  KEY IMPROVEMENTS                                                 ║
║  ----------------                                                 ║
║  • C++ scanner: 10x faster than Python, 14,840 features          ║
║  • Explicit feature weights (learnable attention)                ║
║  • Partial bar support (no stale TF data)                        ║
║  • Automatic correlation analysis                                 ║
║  • Binary format (.bin) for fast sample loading                  ║
║                                                                   ║
║  COMMANDS                                                         ║
║  ---------                                                        ║
║  Scan (C++ - 10x faster):                                        ║
║    cd v15_cpp/build && ./v15_scanner --data-dir ../../data \    ║
║                                       --output samples.bin       ║
║  Train: python -m v15.pipeline train --samples samples.bin      ║
║  Inspect: python -m v15.inspector --samples samples.bin         ║
║  Dashboard: streamlit run v15/dashboard.py                       ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")
