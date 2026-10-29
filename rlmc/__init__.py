"""rlmc -- contract-recursive Recursive Language Models.

Builds on the unmodified ``rlm`` package (alexzhang13/rlm). Nothing in the RLM
repo is patched: each build node is a stock ``rlm.RLM`` instance, and the
contract system, the bounded-return invariant and the recursion tree live here.
That is what makes "identical scaffold, one bit changed" literally true for the
H1 ablation.
"""

__version__ = "0.1.0"

from rlmc.config import ArmConfig, list_presets

__all__ = ["ArmConfig", "list_presets", "__version__"]
