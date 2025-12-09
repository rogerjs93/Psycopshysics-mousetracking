"""
Tracking Analysis Package
=========================

A modular Python package for psychophysics tracking experiment analysis.
Provides visualization, statistical analysis, and cross-correlation tools
for evaluating observer tracking performance across different blob sizes
and auditory feedback conditions.

Main modules:
- core: Data loading, preprocessing, metrics, statistics, state management
- visualization: Static plots, animations, report generation
- ui: Streamlit-based interactive configuration interface
- cli: Command-line batch processing

Quick Start:
    # Launch Streamlit UI
    streamlit run tracking_analysis/ui/app.py
    
    # Or use CLI
    python -m tracking_analysis.cli --config config.yaml --data-path ./data

Author: Psychophysics Research Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Psychophysics Research Team"

from . import core
from . import visualization
from . import ui
from . import cli
