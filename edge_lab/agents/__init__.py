"""
Agents Package
==============
Autonomous agents for Opus Autopoiesis system.
"""

from .hypothesis_generator import HypothesisGenerator, Hypothesis
from .regime_detector import RegimeDetector, RegimeType, RegimeDetectionResult

__all__ = [
    "HypothesisGenerator",
    "Hypothesis",
    "RegimeDetector",
    "RegimeType",
    "RegimeDetectionResult",
]
