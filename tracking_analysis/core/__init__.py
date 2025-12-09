"""
Core Module
===========

Contains the fundamental data processing and analysis components:
- config: Parameter schemas, defaults, validation
- data_loader: CSV loading, metadata extraction, caching
- preprocessing: Outlier removal, velocity calculation, missing data handling
- metrics: Tracking error, RMSE, accuracy calculations
- cross_correlation: Velocity cross-correlation analysis
- statistics: ANOVA, effect sizes, statistical tests
- state_manager: Save/load analysis states
"""

from .config import Config, DEFAULTS
from .data_loader import DataLoader
from .preprocessing import Preprocessor
from .metrics import MetricsCalculator
from .cross_correlation import CrossCorrelationAnalyzer
from .statistics import StatisticalAnalyzer
from .state_manager import StateManager

__all__ = [
    'Config',
    'DEFAULTS',
    'DataLoader',
    'Preprocessor',
    'MetricsCalculator',
    'CrossCorrelationAnalyzer',
    'StatisticalAnalyzer',
    'StateManager',
]
