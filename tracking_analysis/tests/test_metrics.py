"""
Tests for Metrics Module
========================

Tests RMSE, tracking error, and discrimination metrics.
"""

import pytest
import pandas as pd
import numpy as np

from tracking_analysis.core.metrics import MetricsCalculator


class TestEuclideanError:
    """Test Euclidean error calculation."""
    
    def test_euclidean_error_perfect_tracking(self, metrics_calculator):
        """Test error is zero for perfect tracking."""
        df = pd.DataFrame({
            'Target_X': [100, 110, 120],
            'Target_Y': [200, 210, 220],
            'Mouse_X': [100, 110, 120],
            'Mouse_Y': [200, 210, 220]
        })
        
        df = metrics_calculator.calculate_euclidean_error(df)
        
        assert (df['euclidean_error'] == 0).all()
    
    def test_euclidean_error_known_values(self, metrics_calculator):
        """Test error calculation with known values."""
        df = pd.DataFrame({
            'Target_X': [0, 0],
            'Target_Y': [0, 0],
            'Mouse_X': [3, 4],
            'Mouse_Y': [4, 3]
        })
        
        df = metrics_calculator.calculate_euclidean_error(df)
        
        # Distance should be 5 for both (3-4-5 triangle)
        assert np.allclose(df['euclidean_error'], [5, 5])
    
    def test_euclidean_error_column_created(self, loaded_data, metrics_calculator):
        """Test error column is created."""
        df = metrics_calculator.calculate_euclidean_error(loaded_data.copy())
        
        assert 'euclidean_error' in df.columns


class TestRMSE:
    """Test RMSE calculation."""
    
    def test_rmse_perfect_tracking(self, metrics_calculator):
        """Test RMSE is zero for perfect tracking."""
        df = pd.DataFrame({
            'Target_X': [100, 110, 120],
            'Target_Y': [200, 210, 220],
            'Mouse_X': [100, 110, 120],
            'Mouse_Y': [200, 210, 220]
        })
        
        rmse = metrics_calculator.calculate_rmse(df)
        
        assert rmse == 0
    
    def test_rmse_known_value(self, metrics_calculator):
        """Test RMSE with known values."""
        # Constant error of 10 pixels
        df = pd.DataFrame({
            'Target_X': [0, 0, 0, 0],
            'Target_Y': [0, 0, 0, 0],
            'Mouse_X': [6, 6, 6, 6],
            'Mouse_Y': [8, 8, 8, 8]  # Distance = 10 each time
        })
        
        rmse = metrics_calculator.calculate_rmse(df)
        
        # RMSE should be 10 (sqrt of mean of 100)
        assert rmse == 10
    
    def test_rmse_positive(self, loaded_data, metrics_calculator):
        """Test RMSE is non-negative."""
        rmse = metrics_calculator.calculate_rmse(loaded_data)
        
        assert rmse >= 0


class TestTrialMetrics:
    """Test trial-level metrics calculation."""
    
    def test_compute_trial_metrics(self, loaded_data, metrics_calculator):
        """Test computing trial metrics."""
        metrics = metrics_calculator.compute_trial_metrics(loaded_data)
        
        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) > 0
    
    def test_trial_metrics_columns(self, trial_metrics):
        """Test trial metrics has required columns."""
        required = ['participant_id', 'sd_size', 'condition', 'rmse']
        for col in required:
            assert col in trial_metrics.columns
    
    def test_trial_metrics_per_trial(self, loaded_data, metrics_calculator):
        """Test one row per trial."""
        metrics = metrics_calculator.compute_trial_metrics(loaded_data)
        
        # Each unique combination should have one row
        unique_trials = loaded_data.groupby(
            ['participant_id', 'sd_size', 'condition']
        ).ngroups
        
        assert len(metrics) == unique_trials
    
    def test_trial_metrics_rmse_reasonable(self, trial_metrics):
        """Test RMSE values are reasonable."""
        # RMSE should be positive and less than screen diagonal
        max_distance = np.sqrt(1920**2 + 980**2)  # Screen diagonal
        
        assert (trial_metrics['rmse'] >= 0).all()
        assert (trial_metrics['rmse'] < max_distance).all()


class TestDiscriminationMetrics:
    """Test discrimination metrics for research questions."""
    
    def test_discrimination_by_size(self, trial_metrics, metrics_calculator):
        """Test discrimination metrics grouped by SD size."""
        disc_metrics = metrics_calculator.compute_discrimination_metrics(
            trial_metrics,
            group_by='sd_size'
        )
        
        assert isinstance(disc_metrics, pd.DataFrame)
        assert 'sd_size' in disc_metrics.columns
    
    def test_discrimination_by_condition(self, trial_metrics, metrics_calculator):
        """Test discrimination metrics grouped by condition."""
        disc_metrics = metrics_calculator.compute_discrimination_metrics(
            trial_metrics,
            group_by='condition'
        )
        
        assert 'condition' in disc_metrics.columns
    
    def test_discrimination_mean_std(self, trial_metrics, metrics_calculator):
        """Test discrimination includes mean and std."""
        disc_metrics = metrics_calculator.compute_discrimination_metrics(
            trial_metrics,
            group_by='sd_size'
        )
        
        assert 'mean_rmse' in disc_metrics.columns
        assert 'std_rmse' in disc_metrics.columns


class TestAccuracyMetrics:
    """Test additional accuracy metrics."""
    
    def test_mean_error(self, loaded_data, metrics_calculator):
        """Test mean error calculation."""
        df = metrics_calculator.calculate_euclidean_error(loaded_data.copy())
        mean_error = df['euclidean_error'].mean()
        
        assert mean_error >= 0
    
    def test_max_error(self, loaded_data, metrics_calculator):
        """Test max error calculation."""
        df = metrics_calculator.calculate_euclidean_error(loaded_data.copy())
        max_error = df['euclidean_error'].max()
        
        # Max error should be less than screen diagonal
        max_possible = np.sqrt(1920**2 + 980**2)
        assert max_error < max_possible


class TestResearchQuestionMetrics:
    """Test metrics for answering research questions."""
    
    def test_can_discriminate_sizes(self, trial_metrics, metrics_calculator):
        """Test if different SD sizes show different RMSE."""
        # Group by SD size and get mean RMSE
        size_rmse = trial_metrics.groupby('sd_size')['rmse'].mean()
        
        # Should have data for all three sizes
        assert len(size_rmse) == 3
        
        # Metrics should vary (not all identical)
        assert size_rmse.std() > 0
    
    def test_condition_comparison(self, trial_metrics, metrics_calculator):
        """Test comparison between conditions."""
        # Group by condition
        cond_rmse = trial_metrics.groupby('condition')['rmse'].mean()
        
        # Should have both conditions
        assert 'dynamic' in cond_rmse.index
        assert 'static' in cond_rmse.index
    
    def test_interpretation_generation(self, trial_metrics, metrics_calculator):
        """Test generating interpretations."""
        result = metrics_calculator.generate_interpretation(
            trial_metrics,
            metric='rmse',
            question="Can observers discriminate blob sizes?"
        )
        
        assert isinstance(result, dict)
        assert 'summary' in result or 'interpretation' in result
