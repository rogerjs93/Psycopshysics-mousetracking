"""
CLI Module
==========

Command-line interface for running analyses.

Usage:
    python -m tracking_analysis.cli analyze --data-path ./data --output ./results
    python -m tracking_analysis.cli info ./data
    python -m tracking_analysis.cli states ./results/states
    python -m tracking_analysis.cli ui
"""

from .main import main, create_parser

__all__ = [
    'main',
    'create_parser',
]
