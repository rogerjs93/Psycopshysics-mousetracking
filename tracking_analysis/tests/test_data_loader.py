"""
Tests for Data Loader Module
============================

Tests CSV loading, metadata extraction, and filtering.
"""

import pytest
import pandas as pd
from pathlib import Path

from tracking_analysis.core.data_loader import DataLoader, FileMetadata


class TestDataLoaderInit:
    """Test DataLoader initialization."""
    
    def test_init_valid_path(self, data_dir):
        """Test initialization with valid data path."""
        loader = DataLoader(str(data_dir))
        assert loader.data_path == data_dir
    
    def test_init_path_object(self, data_dir):
        """Test initialization with Path object."""
        loader = DataLoader(data_dir)
        assert loader.data_path == data_dir
    
    def test_init_invalid_path(self):
        """Test initialization with invalid path."""
        with pytest.raises(FileNotFoundError):
            DataLoader('/nonexistent/path')


class TestFileDiscovery:
    """Test file discovery and listing."""
    
    def test_list_csv_files(self, data_loader):
        """Test listing CSV files."""
        files = data_loader.list_files()
        assert len(files) > 0
        assert all(f.suffix == '.csv' for f in files)
    
    def test_files_have_expected_pattern(self, data_loader):
        """Test files match expected naming pattern."""
        files = data_loader.list_files()
        for f in files:
            assert 'Participant_' in f.name
            assert 'arcmin' in f.name
            assert ('dynamic' in f.name or 'static' in f.name)


class TestMetadataExtraction:
    """Test metadata extraction from filenames."""
    
    def test_extract_participant_id(self, data_loader):
        """Test participant ID extraction."""
        files = data_loader.list_files()
        for f in files:
            meta = data_loader._extract_metadata(f)
            assert meta.participant_id is not None
            assert meta.participant_id.isdigit()
    
    def test_extract_sd_size(self, data_loader):
        """Test SD size extraction."""
        files = data_loader.list_files()
        for f in files:
            meta = data_loader._extract_metadata(f)
            assert meta.sd_size in [21, 31, 34]
    
    def test_extract_condition(self, data_loader):
        """Test condition extraction."""
        files = data_loader.list_files()
        for f in files:
            meta = data_loader._extract_metadata(f)
            assert meta.condition in ['dynamic', 'static']


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_single_file(self, sample_csv_path):
        """Test loading a single CSV file."""
        df = pd.read_csv(sample_csv_path)
        
        required_cols = ['Frame', 'Target_X', 'Target_Y', 'Mouse_X', 'Mouse_Y']
        for col in required_cols:
            assert col in df.columns
    
    def test_load_all_data(self, data_loader):
        """Test loading all data."""
        df = data_loader.load_all()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'participant_id' in df.columns
        assert 'sd_size' in df.columns
        assert 'condition' in df.columns
    
    def test_load_all_has_required_columns(self, loaded_data):
        """Test loaded data has all required columns."""
        required = ['Frame', 'Target_X', 'Target_Y', 'Mouse_X', 'Mouse_Y',
                   'participant_id', 'sd_size', 'condition', 'file_path']
        for col in required:
            assert col in loaded_data.columns, f"Missing column: {col}"
    
    def test_load_filtered_by_condition(self, data_loader):
        """Test loading filtered by condition."""
        df = data_loader.load_filtered(conditions=['dynamic'])
        
        assert len(df) > 0
        assert all(df['condition'] == 'dynamic')
    
    def test_load_filtered_by_size(self, data_loader):
        """Test loading filtered by SD size."""
        df = data_loader.load_filtered(sd_sizes=[21])
        
        assert len(df) > 0
        assert all(df['sd_size'] == 21)
    
    def test_load_filtered_by_participant(self, data_loader, loaded_data):
        """Test loading filtered by participant."""
        # Get a valid participant ID
        participant = loaded_data['participant_id'].iloc[0]
        
        df = data_loader.load_filtered(participants=[participant])
        
        assert len(df) > 0
        assert all(df['participant_id'] == participant)
    
    def test_load_filtered_combined(self, data_loader):
        """Test loading with combined filters."""
        df = data_loader.load_filtered(
            conditions=['static'],
            sd_sizes=[31, 34]
        )
        
        assert len(df) > 0
        assert all(df['condition'] == 'static')
        assert all(df['sd_size'].isin([31, 34]))


class TestDatasetMetadata:
    """Test dataset metadata."""
    
    def test_get_metadata(self, data_loader):
        """Test getting dataset metadata."""
        data_loader.load_all()  # Must load first
        meta = data_loader.get_metadata()
        
        assert meta is not None
        assert meta.n_files > 0
        assert meta.n_participants > 0
    
    def test_metadata_participants(self, loaded_metadata):
        """Test participant list in metadata."""
        assert len(loaded_metadata.participants) > 0
        assert all(isinstance(p, str) for p in loaded_metadata.participants)
    
    def test_metadata_sd_sizes(self, loaded_metadata):
        """Test SD sizes in metadata."""
        assert set(loaded_metadata.sd_sizes) == {21, 31, 34}
    
    def test_metadata_conditions(self, loaded_metadata):
        """Test conditions in metadata."""
        assert set(loaded_metadata.conditions) == {'dynamic', 'static'}
    
    def test_metadata_total_frames(self, loaded_metadata, loaded_data):
        """Test total frames count."""
        assert loaded_metadata.total_frames == len(loaded_data)


class TestRuntimeEstimation:
    """Test runtime estimation."""
    
    def test_estimate_runtime_returns_dict(self, data_loader):
        """Test estimate_runtime returns a dictionary."""
        estimate = data_loader.estimate_runtime()
        
        assert isinstance(estimate, dict)
        assert 'n_files' in estimate
        assert 'estimated_load_seconds' in estimate
    
    def test_estimate_runtime_reasonable(self, data_loader):
        """Test estimate is reasonable."""
        estimate = data_loader.estimate_runtime()
        
        # Should estimate non-zero time for any files
        if estimate['n_files'] > 0:
            assert estimate['estimated_load_seconds'] > 0


class TestDataValidation:
    """Test data validation and quality checks."""
    
    def test_no_all_null_rows(self, loaded_data):
        """Test no rows are completely null."""
        position_cols = ['Target_X', 'Target_Y', 'Mouse_X', 'Mouse_Y']
        all_null = loaded_data[position_cols].isnull().all(axis=1)
        assert not all_null.any()
    
    def test_frame_values_reasonable(self, loaded_data):
        """Test frame values are reasonable."""
        assert loaded_data['Frame'].min() >= 0
        assert loaded_data['Frame'].max() < 10000  # Reasonable upper bound
    
    def test_position_values_reasonable(self, loaded_data):
        """Test position values are within screen bounds."""
        # Screen is 1920x980
        assert loaded_data['Target_X'].max() <= 1920
        assert loaded_data['Target_X'].min() >= 0
        assert loaded_data['Target_Y'].max() <= 980
        assert loaded_data['Target_Y'].min() >= 0
