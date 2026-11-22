# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0 (see LICENSE).
"""SageGP: SageAttention 2 for every GPU, with faster precise, adaptive and fast modes on GeForce Blackwell (SM120)."""
from .core import PRECISIONS, kernel_for, sageattn, sageattn3_blackwell, sageattn_fast

__version__ = "1.0.2"
__all__ = ["PRECISIONS", "kernel_for", "sageattn", "sageattn3_blackwell", "sageattn_fast", "__version__"]
