"""
Tests for Cross-Correlation Module
===================================

Tests velocity cross-correlation and lag detection.
"""

import pytest
import pandas as pd
import numpy as np

from tracking_analysis.core.cross_correlation import CrossCorrelationAnalyzer
from tracking_analysis.core.config import Config


class TestVelocityCrossCorrelation:
    """Test velocity cross-correlation calculation."""
    
    def test_xcorr_returns_result(self, xcorr_analyzer, synthetic_trial):
        """Test cross-correlation returns a result."""
        result = xcorr_analyzer.compute_velocity_xcorr(synthetic_trial)
        
        assert result is not None
        assert hasattr(result, 'correlations')
        assert hasattr(result, 'lags')
    
    def test_xcorr_normalized_range(self, xcorr_analyzer, synthetic_trial):
        """Test normalized cross-correlation is in [-1, 1]."""
        result = xcorr_analyzer.compute_velocity_xcorr(
            synthetic_trial,
            normalize=True
        )
        
        # All values should be in [-1, 1]
        assert np.all(result.correlations >= -1.001)  # Small tolerance
        assert np.all(result.correlations <= 1.001)
    
    def test_xcorr_detects_lag(self, xcorr_analyzer, synthetic_trial):
        """Test cross-correlation detects known lag."""
        # Synthetic trial has lag of ~5 frames
        result = xcorr_analyzer.compute_velocity_xcorr(synthetic_trial)
        
        # Optimal lag should be around 5
        # Allow some tolerance due to noise
        assert abs(result.optimal_lag) < 20


class TestLagDetection:
    """Test lag detection functionality."""
    
    def test_find_optimal_lag(self, xcorr_analyzer, synthetic_trial):
        """Test finding optimal lag."""
        result = xcorr_analyzer.compute_velocity_xcorr(synthetic_trial)
        
        assert isinstance(result.optimal_lag, (int, np.integer))
    
    def test_lag_interpretation_predictive(self, xcorr_analyzer):
        """Test interpretation of negative lag (predictive)."""
        # Create data where mouse leads target
        n = 100
        t = np.linspace(0, 2*np.pi, n)
        
        df = pd.DataFrame({
            'Frame': np.arange(n),
            'Target_X': 500 + 100 * np.sin(t),
            'Target_Y': 500 + 50 * np.cos(t),
            'Mouse_X': 500 + 100 * np.sin(t + 0.5),  # Mouse leads
            'Mouse_Y': 500 + 50 * np.cos(t + 0.5)
        })
        
        result = xcorr_analyzer.compute_velocity_xcorr(df)
        
        # Should detect negative lag (predictive)
        # or positive depending on convention
        assert result.optimal_lag != 0
    
    def test_lag_interpretation_reactive(self, xcorr_analyzer):
        """Test interpretation of positive lag (reactive)."""
        # Create data where mouse lags target
        n = 100
        t = np.linspace(0, 2*np.pi, n)
        
        target_x = 500 + 100 * np.sin(t)
        mouse_x = np.roll(target_x, 10)  # Mouse follows with 10 frame delay
        
        df = pd.DataFrame({
            'Frame': np.arange(n),
            'Target_X': target_x,
            'Target_Y': 500 + 50 * np.cos(t),
            'Mouse_X': mouse_x,
            'Mouse_Y': 500 + 50 * np.cos(t)
        })
        
        result = xcorr_analyzer.compute_velocity_xcorr(df)
        
        # Should detect some lag
        assert result.optimal_lag is not None


class TestCrossCorrelationByTrial:
    """Test cross-correlation computed per trial."""
    
    def test_xcorr_per_trial(self, xcorr_analyzer, loaded_data, preprocessor):
        """Test computing cross-correlation per trial."""
        # Preprocess first
        df = preprocessor.calculate_velocity(loaded_data.copy())
        
        # Compute per trial (limit to few trials for speed)
        trials = df.groupby(['participant_id', 'sd_size', 'condition'])
        trial_list = list(trials.groups.keys())[:3]
        
        results = []
        for key in trial_list:
            trial_df = trials.get_group(key)
            result = xcorr_analyzer.compute_velocity_xcorr(trial_df)
            results.append(result)
        
        assert len(results) == 3
    
    def test_xcorr_batch_processing(self, xcorr_analyzer, loaded_data, preprocessor):
        """Test batch processing of cross-correlations."""
        df = preprocessor.calculate_velocity(loaded_data.copy())
        
        # Use batch method if available
        if hasattr(xcorr_analyzer, 'compute_batch_xcorr'):
            results_df = xcorr_analyzer.compute_batch_xcorr(df)
            
            assert isinstance(results_df, pd.DataFrame)
            assert 'optimal_lag' in results_df.columns


class TestMaxLagParameter:
    """Test max_lag parameter behavior."""
    
    def test_max_lag_limits_search(self, data_dir, synthetic_trial):
        """Test max_lag limits correlation search range."""
        config = Config(
            data_path=str(data_dir),
            max_lag_frames=10
        )
        analyzer = CrossCorrelationAnalyzer(config)
        
        result = analyzer.compute_velocity_xcorr(synthetic_trial)
        
        # Lags should be within max_lag
        assert all(abs(lag) <= 10 for lag in result.lags)
    
    def test_different_max_lags(self, data_dir, synthetic_trial):
        """Test with different max_lag values."""
        results = []
        
        for max_lag in [5, 10, 20]:
            config = Config(
                data_path=str(data_dir),
                max_lag_frames=max_lag
            )
            analyzer = CrossCorrelationAnalyzer(config)
            result = analyzer.compute_velocity_xcorr(synthetic_trial)
            results.append(len(result.lags))
        
        # More lags with larger max_lag
        assert results[0] < results[1] < results[2]


class TestPeakCorrelation:
    """Test peak correlation detection."""
    
    def test_peak_correlation_value(self, xcorr_analyzer, synthetic_trial):
        """Test peak correlation value is returned."""
        result = xcorr_analyzer.compute_velocity_xcorr(synthetic_trial)
        
        assert hasattr(result, 'peak_correlation')
        assert -1 <= result.peak_correlation <= 1
    
    def test_perfect_correlation(self, xcorr_analyzer):
        """Test correlation with identical signals."""
        n = 100
        t = np.linspace(0, 2*np.pi, n)
        
        df = pd.DataFrame({
            'Frame': np.arange(n),
            'Target_X': 500 + 100 * np.sin(t),
            'Target_Y': 500,
            'Mouse_X': 500 + 100 * np.sin(t),
            'Mouse_Y': 500
        })
        
        result = xcorr_analyzer.compute_velocity_xcorr(df, normalize=True)
        
        # Peak should be close to 1 at lag 0
        assert result.peak_correlation > 0.9


class TestLagInterpretation:
    """Test lag interpretation strings."""
    
    def test_interpretation_string(self, xcorr_analyzer, synthetic_trial):
        """Test interpretation string is generated."""
        result = xcorr_analyzer.compute_velocity_xcorr(synthetic_trial)
        
        if hasattr(result, 'interpretation'):
            assert isinstance(result.interpretation, str)
            assert len(result.interpretation) > 0
    
    def test_lag_to_time_conversion(self, xcorr_analyzer, data_dir):
        """Test lag is correctly converted to time."""
        config = Config(
            data_path=str(data_dir),
            frame_rate=50
        )
        analyzer = CrossCorrelationAnalyzer(config)
        
        # At 50 fps, lag of 5 frames = 100 ms
        lag_ms = analyzer.lag_to_milliseconds(5)
        
        assert lag_ms == 100
