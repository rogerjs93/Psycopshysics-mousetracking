"""
Pytest Fixtures and Configuration
=================================

Shared fixtures for all tests using actual data from the data folder.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracking_analysis.core.config import Config
from tracking_analysis.core.data_loader import DataLoader
from tracking_analysis.core.preprocessing import Preprocessor
from tracking_analysis.core.metrics import MetricsCalculator
from tracking_analysis.core.cross_correlation import CrossCorrelationAnalyzer
from tracking_analysis.core.statistics import StatisticalAnalyzer
from tracking_analysis.core.state_manager import StateManager


# =============================================================================
# PATH FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def data_dir():
    """Path to actual data directory."""
    # Navigate from tests to data folder
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / 'data'
    
    if not data_path.exists():
        pytest.skip("Data directory not found")
    
    return data_path


@pytest.fixture(scope="session")
def sample_csv_path(data_dir):
    """Path to a single sample CSV file."""
    csv_files = list(data_dir.glob('*.csv'))
    if not csv_files:
        pytest.skip("No CSV files found in data directory")
    return csv_files[0]


@pytest.fixture(scope="function")
def temp_dir():
    """Temporary directory for test outputs."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


# =============================================================================
# CONFIG FIXTURES
# =============================================================================

@pytest.fixture
def default_config(data_dir):
    """Default configuration with actual data path."""
    return Config(data_path=str(data_dir))


@pytest.fixture
def config_difference_velocity(data_dir):
    """Config with difference velocity method."""
    return Config(
        data_path=str(data_dir),
        velocity_method='difference',
        frame_rate=50
    )


@pytest.fixture
def config_savgol_velocity(data_dir):
    """Config with Savitzky-Golay velocity method."""
    return Config(
        data_path=str(data_dir),
        velocity_method='savgol',
        savgol_window=5,
        savgol_polyorder=2,
        frame_rate=50
    )


@pytest.fixture
def config_all_outlier_methods(data_dir):
    """Config for testing different outlier methods."""
    configs = {
        'iqr': Config(data_path=str(data_dir), outlier_method='iqr', outlier_threshold=1.5),
        'zscore': Config(data_path=str(data_dir), outlier_method='zscore', outlier_threshold=3.0),
        'mad': Config(data_path=str(data_dir), outlier_method='mad', outlier_threshold=3.0),
    }
    return configs


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def sample_data(sample_csv_path):
    """Load a single CSV file as sample data."""
    return pd.read_csv(sample_csv_path)


@pytest.fixture(scope="session")
def loaded_data(data_dir):
    """Load all data using DataLoader."""
    loader = DataLoader(str(data_dir))
    df = loader.load_all()
    return df


@pytest.fixture(scope="session")
def loaded_metadata(data_dir):
    """Get metadata from DataLoader."""
    loader = DataLoader(str(data_dir))
    loader.load_all()  # Must load first
    return loader.get_metadata()


@pytest.fixture
def synthetic_trial():
    """
    Create synthetic trial data for precise testing.
    
    Known values:
    - Target moves in sine wave
    - Mouse follows with slight lag and noise
    """
    n_frames = 100
    t = np.linspace(0, 2*np.pi, n_frames)
    
    # Target movement: sine wave
    target_x = 960 + 100 * np.sin(t)
    target_y = 490 + 50 * np.cos(t)
    
    # Mouse follows with lag (5 frames) and small noise
    lag = 5
    noise_std = 2
    
    mouse_x = np.zeros(n_frames)
    mouse_y = np.zeros(n_frames)
    
    for i in range(n_frames):
        lag_idx = max(0, i - lag)
        mouse_x[i] = target_x[lag_idx] + np.random.normal(0, noise_std)
        mouse_y[i] = target_y[lag_idx] + np.random.normal(0, noise_std)
    
    df = pd.DataFrame({
        'Frame': np.arange(n_frames),
        'Target_X': target_x,
        'Target_Y': target_y,
        'Mouse_X': mouse_x,
        'Mouse_Y': mouse_y
    })
    
    return df


@pytest.fixture
def multi_trial_data(data_dir):
    """Load data from multiple trials for statistical tests."""
    loader = DataLoader(str(data_dir))
    df = loader.load_all()
    
    # Ensure we have enough data
    if df['participant_id'].nunique() < 2:
        pytest.skip("Need at least 2 participants for multi-trial tests")
    
    return df


# =============================================================================
# COMPONENT FIXTURES
# =============================================================================

@pytest.fixture
def data_loader(data_dir):
    """DataLoader instance."""
    return DataLoader(str(data_dir))


@pytest.fixture
def preprocessor(default_config):
    """Preprocessor instance with default config."""
    return Preprocessor(default_config)


@pytest.fixture
def metrics_calculator(default_config):
    """MetricsCalculator instance."""
    return MetricsCalculator(default_config)


@pytest.fixture
def xcorr_analyzer(default_config):
    """CrossCorrelationAnalyzer instance."""
    return CrossCorrelationAnalyzer(default_config)


@pytest.fixture
def stats_analyzer(default_config):
    """StatisticalAnalyzer instance."""
    return StatisticalAnalyzer(default_config)


@pytest.fixture
def state_manager(temp_dir):
    """StateManager instance with temp directory."""
    return StateManager(temp_dir)


# =============================================================================
# PROCESSED DATA FIXTURES
# =============================================================================

@pytest.fixture
def processed_data(loaded_data, preprocessor):
    """Data after preprocessing."""
    df = loaded_data.copy()
    df = preprocessor.calculate_velocity(df)
    df = preprocessor.remove_outliers(df)
    return df


@pytest.fixture
def trial_metrics(processed_data, metrics_calculator):
    """Computed trial metrics."""
    return metrics_calculator.compute_trial_metrics(processed_data)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_csv_files(data_dir, n=None):
    """Get list of CSV files from data directory."""
    files = sorted(data_dir.glob('*.csv'))
    if n is not None:
        files = files[:n]
    return files


def parse_filename(filename):
    """Parse metadata from filename."""
    import re
    name = Path(filename).stem
    
    # Extract participant ID
    part_match = re.search(r'Participant_(\d+)', name)
    participant_id = part_match.group(1) if part_match else None
    
    # Extract SD size
    size_match = re.search(r'(\d+)arcmin', name)
    sd_size = int(size_match.group(1)) if size_match else None
    
    # Extract condition
    condition = 'dynamic' if 'dynamic' in name.lower() else 'static'
    
    return {
        'participant_id': participant_id,
        'sd_size': sd_size,
        'condition': condition
    }
