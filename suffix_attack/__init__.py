"""
suffix_attack/__init__.py
=========================
Public API of the suffix_attack module.

    from suffix_attack import SuffixAttackGenerator, SuffixAttackConfig
    from suffix_attack import BlockerInjector, SuffixAttackEvaluator
"""

from .generator import SuffixAttackGenerator
from .config    import SuffixAttackConfig
from .injector  import BlockerInjector
from .evaluator import SuffixAttackEvaluator

__all__ = [
    "SuffixAttackGenerator",
    "SuffixAttackConfig",
    "BlockerInjector",
    "SuffixAttackEvaluator",
]

__version__ = "1.0.0"
__author__  = "DASA Research Module"
