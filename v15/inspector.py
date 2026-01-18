"""
This module has been replaced by dual_inspector.
Importing here for backwards compatibility.
"""
import warnings
warnings.warn(
    "v15.inspector is deprecated. Use 'python -m v15.dual_inspector' instead.",
    DeprecationWarning
)
from .dual_inspector import DualAssetInspector, main

if __name__ == "__main__":
    main()
