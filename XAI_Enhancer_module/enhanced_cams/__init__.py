"""
Enhanced CAM Methods Module

This module contains enhanced versions of various CAM (Class Activation Mapping) methods
that include the novel XAI enhancement for improved explainability.
"""

from .GradCAM_enhanced import GradCAMEnhanced
from .GradCAMPlusPlus_enhanced import GradCAMPlusPlusEnhanced
from .HiResCAM_enhanced import HiResCAMEnhanced
from .ScoreCAM_enhanced import ScoreCAMEnhanced
from .AblationCAM_enhanced import AblationCAMEnhanced

__all__ = [
    'GradCAMEnhanced',
    'GradCAMPlusPlusEnhanced', 
    'HiResCAMEnhanced',
    'ScoreCAMEnhanced',
    'AblationCAMEnhanced'
]