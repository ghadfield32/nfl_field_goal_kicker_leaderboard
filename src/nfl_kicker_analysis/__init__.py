"""
NFL Kicker Analysis Package
A comprehensive toolkit for analyzing NFL field goal kicker performance.
"""

__version__ = "1.0.0"
__author__ = "NFL Analytics Team"

# Import main classes for easy access
from .config import config
from .data.loader import DataLoader
from .data.preprocessor import DataPreprocessor

__all__ = [
    'config',
    'DataLoader',
    'DataPreprocessor'
]
