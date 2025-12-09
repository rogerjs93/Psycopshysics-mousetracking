"""
Visualization Module
====================

Creates visualizations including:
- static_plots: Matplotlib charts, bar plots, error distributions
- animation: MP4/HTML animations of tracking trials
- report_builder: Markdown/HTML report generation
"""

from .static_plots import StaticPlotter, PlotConfig
from .animation import Animator, AnimationConfig
from .report_builder import ReportBuilder, ReportConfig

__all__ = [
    'StaticPlotter',
    'PlotConfig',
    'Animator',
    'AnimationConfig',
    'ReportBuilder',
    'ReportConfig',
]
