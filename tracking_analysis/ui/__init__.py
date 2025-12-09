"""
UI Module
=========

Contains the Streamlit-based interactive user interface.

To run:
    streamlit run tracking_analysis/ui/app.py
    
Or via CLI:
    python -m tracking_analysis.cli ui
"""

from . import app

__all__ = [
    'app',
]
