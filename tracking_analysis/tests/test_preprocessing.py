"""
Tests for Preprocessing Module
==============================

Tests velocity calculation, outlier removal, and missing data handling.
"""

import pytest
import pandas as pd
import numpy as np

from tracking_analysis.core.preprocessing import Preprocessor
from tracking_analysis.core.config import Config


class TestVelocityCalculation:
    """Test velocity calculation methods."""
    
    def test_velocity_columns_created(self, loaded_data, preprocessor):
        """Test velocity columns are created."""
        df = preprocessor.calculate_velocity(loaded_data.copy())
        
        assert 'velocity_target_x' in df.columns
        assert 'velocity_target_y' in df.columns
        assert 'velocity_mouse_x' in df.columns
        assert 'velocity_mouse_y' in df.columns
    
    def test_velocity_difference_method(self, sample_data, data_dir):
        """Test difference velocity method."""
        config = Config(
            data_path=str(data_dir),
            velocity_method='difference',
            frame_rate=50
        )
        preprocessor = Preprocessor(config)
        
        df = sample_data.copy()
        df = preprocessor.calculate_velocity(df)
        
        # First value should be NaN for difference method
        assert pd.isna(df['velocity_target_x'].iloc[0])
        
        # Manual verification of velocity calculation
        expected_vel = (df['Target_X'].iloc[2] - df['Target_X'].iloc[1]) * 50
        actual_vel = df['velocity_target_x'].iloc[2]
        assert abs(expected_vel - actual_vel) < 0.001
    
    def test_velocity_savgol_method(self, sample_data, data_dir):
        """Test Savitzky-Golay velocity method."""
        config = Config(
            data_path=str(data_dir),
            velocity_method='savgol',
            savgol_window=5,
            savgol_polyorder=2,
            frame_rate=50
        )
        preprocessor = Preprocessor(config)
        
        df = sample_data.copy()
        df = preprocessor.calculate_velocity(df)
        
        # Should have velocity values
        assert df['velocity_target_x'].notna().sum() > 0
    
    def test_velocity_magnitude_created(self, loaded_data, preprocessor):
        """Test velocity magnitude columns created."""
        df = preprocessor.calculate_velocity(loaded_data.copy())
        
        assert 'velocity_target_mag' in df.columns
        assert 'velocity_mouse_mag' in df.columns
    
    def test_velocity_magnitude_positive(self, loaded_data, preprocessor):
        """Test velocity magnitude is non-negative."""
        df = preprocessor.calculate_velocity(loaded_data.copy())
        
        # Exclude NaN values
        target_mag = df['velocity_target_mag'].dropna()
        mouse_mag = df['velocity_mouse_mag'].dropna()
        
        assert (target_mag >= 0).all()
        assert (mouse_mag >= 0).all()


class TestOutlierRemoval:
    """Test outlier removal methods."""
    
    def test_outlier_removal_iqr(self, loaded_data, data_dir):
        """Test IQR outlier removal."""
        config = Config(
            data_path=str(data_dir),
            outlier_method='iqr',
            outlier_threshold=1.5
        )
        preprocessor = Preprocessor(config)
        
        df = loaded_data.copy()
        original_len = len(df)
        
        df_clean = preprocessor.remove_outliers(df)
        
        # Should remove some outliers (or mark them)
        assert len(df_clean) <= original_len
    
    def test_outlier_removal_zscore(self, loaded_data, data_dir):
        """Test Z-score outlier removal."""
        config = Config(
            data_path=str(data_dir),
            outlier_method='zscore',
            outlier_threshold=3.0
        )
        preprocessor = Preprocessor(config)
        
        df = loaded_data.copy()
        df_clean = preprocessor.remove_outliers(df)
        
        # Should return a DataFrame
        assert isinstance(df_clean, pd.DataFrame)
    
    def test_outlier_removal_mad(self, loaded_data, data_dir):
        """Test MAD outlier removal."""
        config = Config(
            data_path=str(data_dir),
            outlier_method='mad',
            outlier_threshold=3.0
        )
        preprocessor = Preprocessor(config)
        
        df = loaded_data.copy()
        df_clean = preprocessor.remove_outliers(df)
        
        assert isinstance(df_clean, pd.DataFrame)
    
    def test_outlier_removal_none(self, loaded_data, data_dir):
        """Test no outlier removal."""
        config = Config(
            data_path=str(data_dir),
            outlier_method='none'
        )
        preprocessor = Preprocessor(config)
        
        df = loaded_data.copy()
        original_len = len(df)
        
        df_clean = preprocessor.remove_outliers(df)
        
        # Should not remove anything
        assert len(df_clean) == original_len
    
    def test_outlier_flagging(self, loaded_data, preprocessor):
        """Test outliers are flagged."""
        df = loaded_data.copy()
        df_clean = preprocessor.remove_outliers(df, remove=False)
        
        # Should have outlier flag column
        assert 'is_outlier' in df_clean.columns


class TestMissingDataHandling:
    """Test missing data handling."""
    
    def test_handle_missing_interpolate(self, data_dir):
        """Test interpolation of missing values."""
        config = Config(
            data_path=str(data_dir),
            missing_method='interpolate'
        )
        preprocessor = Preprocessor(config)
        
        # Create data with missing values
        df = pd.DataFrame({
            'Frame': [0, 1, 2, 3, 4],
            'Target_X': [100, np.nan, 120, np.nan, 140],
            'Target_Y': [200, 210, np.nan, 230, 240],
            'Mouse_X': [105, 115, 125, 135, 145],
            'Mouse_Y': [205, 215, 225, 235, 245]
        })
        
        df_filled = preprocessor.handle_missing(df)
        
        # Should have no NaN values
        assert df_filled[['Target_X', 'Target_Y']].isna().sum().sum() == 0
    
    def test_handle_missing_drop(self, data_dir):
        """Test dropping rows with missing values."""
        config = Config(
            data_path=str(data_dir),
            missing_method='drop'
        )
        preprocessor = Preprocessor(config)
        
        # Create data with missing values
        df = pd.DataFrame({
            'Frame': [0, 1, 2, 3, 4],
            'Target_X': [100, np.nan, 120, np.nan, 140],
            'Target_Y': [200, 210, 220, 230, 240],
            'Mouse_X': [105, 115, 125, 135, 145],
            'Mouse_Y': [205, 215, 225, 235, 245]
        })
        
        df_clean = preprocessor.handle_missing(df)
        
        # Should have fewer rows
        assert len(df_clean) == 3  # 2 rows with NaN removed
    
    def test_handle_missing_ffill(self, data_dir):
        """Test forward fill of missing values."""
        config = Config(
            data_path=str(data_dir),
            missing_method='ffill'
        )
        preprocessor = Preprocessor(config)
        
        df = pd.DataFrame({
            'Frame': [0, 1, 2, 3, 4],
            'Target_X': [100, np.nan, np.nan, 130, 140],
            'Target_Y': [200, 210, 220, 230, 240],
            'Mouse_X': [105, 115, 125, 135, 145],
            'Mouse_Y': [205, 215, 225, 235, 245]
        })
        
        df_filled = preprocessor.handle_missing(df)
        
        # First NaN should be filled with previous value
        assert df_filled['Target_X'].iloc[1] == 100
        assert df_filled['Target_X'].iloc[2] == 100


class TestTimeWindowFiltering:
    """Test time window filtering."""
    
    def test_filter_time_window(self, loaded_data, preprocessor):
        """Test filtering to time window."""
        df = loaded_data.copy()
        
        # Get frames for first trial
        first_trial = df[df['participant_id'] == df['participant_id'].iloc[0]]
        max_frame = first_trial['Frame'].max()
        
        # Filter to first half
        df_filtered = preprocessor.filter_time_window(
            df, 
            start_frame=0, 
            end_frame=max_frame // 2
        )
        
        assert df_filtered['Frame'].max() <= max_frame // 2
    
    def test_filter_by_time_seconds(self, loaded_data, data_dir):
        """Test filtering by time in seconds."""
        config = Config(
            data_path=str(data_dir),
            frame_rate=50
        )
        preprocessor = Preprocessor(config)
        
        df = loaded_data.copy()
        
        # Filter to first 5 seconds
        df_filtered = preprocessor.filter_time_window(
            df,
            start_time=0,
            end_time=5  # seconds
        )
        
        # At 50fps, 5 seconds = 250 frames
        assert df_filtered['Frame'].max() <= 250


class TestPreprocessingPipeline:
    """Test full preprocessing pipeline."""
    
    def test_preprocess_full(self, loaded_data, preprocessor):
        """Test full preprocessing pipeline."""
        df = preprocessor.preprocess(loaded_data.copy())
        
        # Should have velocity columns
        assert 'velocity_target_x' in df.columns
        
        # Should have processed data
        assert len(df) > 0
    
    def test_preprocess_returns_report(self, loaded_data, preprocessor):
        """Test preprocessing returns a report."""
        df, report = preprocessor.preprocess(loaded_data.copy(), return_report=True)
        
        assert isinstance(report, dict)
        assert 'original_rows' in report
        assert 'final_rows' in report
    
    def test_preprocess_idempotent(self, loaded_data, preprocessor):
        """Test preprocessing twice gives same result."""
        df1 = preprocessor.preprocess(loaded_data.copy())
        df2 = preprocessor.preprocess(df1.copy())
        
        # Columns should be the same
        assert set(df1.columns) == set(df2.columns)
