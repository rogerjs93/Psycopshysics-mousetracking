"""
Tests for State Manager Module
==============================

Tests saving, loading, and listing analysis states.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from tracking_analysis.core.state_manager import (
    StateManager, 
    AnalysisState, 
    StateInfo,
    save_state,
    load_state,
    list_saved_states
)
from tracking_analysis.core.config import Config


class TestStateManagerInit:
    """Test StateManager initialization."""
    
    def test_init_creates_directory(self, temp_dir):
        """Test initialization creates base directory."""
        base_path = temp_dir / 'states'
        manager = StateManager(base_path)
        
        assert base_path.exists()
    
    def test_init_with_existing_directory(self, temp_dir):
        """Test initialization with existing directory."""
        base_path = temp_dir / 'existing_states'
        base_path.mkdir()
        
        manager = StateManager(base_path)
        
        assert manager.base_path == base_path


class TestStateSaving:
    """Test state saving functionality."""
    
    def test_save_minimal_state(self, state_manager, default_config):
        """Test saving state with minimal data."""
        state_id = state_manager.save_state(config=default_config)
        
        assert state_id is not None
        assert len(state_id) > 0
    
    def test_save_with_metrics(self, state_manager, default_config, trial_metrics):
        """Test saving state with trial metrics."""
        state_id = state_manager.save_state(
            config=default_config,
            trial_metrics=trial_metrics
        )
        
        # Verify files created
        state_dir = state_manager.base_path / state_id
        assert (state_dir / 'state_data.pkl').exists()
        assert (state_dir / 'state_config.yaml').exists()
        assert (state_dir / 'state_info.json').exists()
    
    def test_save_with_all_data(self, state_manager, default_config, 
                                 processed_data, trial_metrics):
        """Test saving state with all data types."""
        xcorr_results = pd.DataFrame({
            'participant_id': ['1341', '1341'],
            'sd_size': [21, 31],
            'optimal_lag': [3, 5],
            'peak_correlation': [0.85, 0.78]
        })
        
        stats_results = {
            'anova': {'f_stat': 5.2, 'p_value': 0.01},
            'effect_size': 0.3
        }
        
        state_id = state_manager.save_state(
            config=default_config,
            trial_metrics=trial_metrics,
            xcorr_results=xcorr_results,
            statistical_results=stats_results,
            processed_data=processed_data
        )
        
        assert state_id is not None
    
    def test_save_custom_state_id(self, state_manager, default_config):
        """Test saving with custom state ID."""
        custom_id = 'my_custom_state_123'
        
        state_id = state_manager.save_state(
            config=default_config,
            state_id=custom_id
        )
        
        assert state_id == custom_id
    
    def test_save_creates_yaml(self, state_manager, default_config):
        """Test YAML config file is human-readable."""
        import yaml
        
        state_id = state_manager.save_state(config=default_config)
        
        yaml_path = state_manager.base_path / state_id / 'state_config.yaml'
        
        with open(yaml_path, 'r') as f:
            loaded = yaml.safe_load(f)
        
        assert 'config' in loaded
        assert 'state_id' in loaded
        assert 'created_at' in loaded


class TestStateLoading:
    """Test state loading functionality."""
    
    def test_load_saved_state(self, state_manager, default_config, trial_metrics):
        """Test loading a saved state."""
        # Save state
        state_id = state_manager.save_state(
            config=default_config,
            trial_metrics=trial_metrics
        )
        
        # Load state
        loaded = state_manager.load_state(state_id)
        
        assert isinstance(loaded, AnalysisState)
        assert loaded.state_id == state_id
    
    def test_load_config_restored(self, state_manager, default_config):
        """Test config is correctly restored."""
        state_id = state_manager.save_state(config=default_config)
        
        loaded = state_manager.load_state(state_id)
        
        assert loaded.config is not None
        assert isinstance(loaded.config, dict)
    
    def test_load_metrics_restored(self, state_manager, default_config, trial_metrics):
        """Test trial metrics are correctly restored."""
        state_id = state_manager.save_state(
            config=default_config,
            trial_metrics=trial_metrics
        )
        
        loaded = state_manager.load_state(state_id)
        
        assert loaded.trial_metrics is not None
        assert len(loaded.trial_metrics) == len(trial_metrics)
    
    def test_load_nonexistent_state(self, state_manager):
        """Test loading non-existent state raises error."""
        with pytest.raises(FileNotFoundError):
            state_manager.load_state('nonexistent_state_id')
    
    def test_load_without_raw_data(self, state_manager, default_config, 
                                    processed_data, loaded_data):
        """Test loading without raw data to save memory."""
        state_id = state_manager.save_state(
            config=default_config,
            processed_data=processed_data,
            raw_data=loaded_data
        )
        
        loaded = state_manager.load_state(state_id, load_raw_data=False)
        
        assert loaded.raw_data is None
        assert loaded.processed_data is not None


class TestStateList:
    """Test listing saved states."""
    
    def test_list_states_empty(self, temp_dir):
        """Test listing when no states exist."""
        manager = StateManager(temp_dir / 'empty')
        states = manager.list_states()
        
        assert states == []
    
    def test_list_states_multiple(self, state_manager, default_config):
        """Test listing multiple saved states."""
        # Save multiple states
        state_manager.save_state(config=default_config, state_id='state_1')
        state_manager.save_state(config=default_config, state_id='state_2')
        state_manager.save_state(config=default_config, state_id='state_3')
        
        states = state_manager.list_states()
        
        assert len(states) == 3
        assert all(isinstance(s, StateInfo) for s in states)
    
    def test_list_states_sorted_by_time(self, state_manager, default_config):
        """Test states are sorted newest first."""
        import time
        
        state_manager.save_state(config=default_config, state_id='older')
        time.sleep(0.1)  # Small delay to ensure different timestamps
        state_manager.save_state(config=default_config, state_id='newer')
        
        states = state_manager.list_states()
        
        # Newer should be first
        assert states[0].state_id == 'newer'


class TestStateInfo:
    """Test state info retrieval."""
    
    def test_get_state_info(self, state_manager, default_config, trial_metrics):
        """Test getting state info."""
        state_id = state_manager.save_state(
            config=default_config,
            trial_metrics=trial_metrics
        )
        
        info = state_manager.get_state_info(state_id)
        
        assert info is not None
        assert info.state_id == state_id
        assert info.has_metrics == True
    
    def test_state_info_has_metrics_flag(self, state_manager, default_config, 
                                          trial_metrics):
        """Test has_metrics flag in state info."""
        # Save without metrics
        state_id_no_metrics = state_manager.save_state(config=default_config)
        
        # Save with metrics
        state_id_with_metrics = state_manager.save_state(
            config=default_config,
            trial_metrics=trial_metrics
        )
        
        info_no = state_manager.get_state_info(state_id_no_metrics)
        info_yes = state_manager.get_state_info(state_id_with_metrics)
        
        assert info_no.has_metrics == False
        assert info_yes.has_metrics == True
    
    def test_state_info_file_size(self, state_manager, default_config, processed_data):
        """Test file size is calculated."""
        state_id = state_manager.save_state(
            config=default_config,
            processed_data=processed_data
        )
        
        info = state_manager.get_state_info(state_id)
        
        assert info.file_size_mb > 0


class TestStateDelete:
    """Test state deletion."""
    
    def test_delete_state(self, state_manager, default_config):
        """Test deleting a state."""
        state_id = state_manager.save_state(config=default_config)
        
        # Verify it exists
        assert state_manager.get_state_info(state_id) is not None
        
        # Delete
        result = state_manager.delete_state(state_id)
        
        assert result == True
        
        # Verify it's gone
        assert state_manager.get_state_info(state_id) is None
    
    def test_delete_nonexistent(self, state_manager):
        """Test deleting non-existent state returns False."""
        result = state_manager.delete_state('nonexistent')
        
        assert result == False


class TestRuntimeEstimation:
    """Test load time estimation."""
    
    def test_estimate_load_time(self, state_manager, default_config, processed_data):
        """Test load time estimation."""
        state_id = state_manager.save_state(
            config=default_config,
            processed_data=processed_data
        )
        
        estimate = state_manager.estimate_load_time(state_id)
        
        assert 'file_size_mb' in estimate
        assert 'estimated_load_seconds' in estimate
    
    def test_estimate_nonexistent(self, state_manager):
        """Test estimation for non-existent state."""
        estimate = state_manager.estimate_load_time('nonexistent')
        
        assert 'error' in estimate


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_save_state_function(self, temp_dir, default_config, trial_metrics):
        """Test save_state convenience function."""
        state_id = save_state(
            output_dir=temp_dir,
            config=default_config,
            trial_metrics=trial_metrics
        )
        
        assert state_id is not None
    
    def test_load_state_function(self, temp_dir, default_config, trial_metrics):
        """Test load_state convenience function."""
        # Save first
        state_id = save_state(
            output_dir=temp_dir,
            config=default_config,
            trial_metrics=trial_metrics
        )
        
        # Load using convenience function
        loaded = load_state(temp_dir, state_id)
        
        assert loaded.state_id == state_id
    
    def test_list_saved_states_function(self, temp_dir, default_config):
        """Test list_saved_states convenience function."""
        # Save a few states
        save_state(temp_dir, default_config, state_id='test_1')
        save_state(temp_dir, default_config, state_id='test_2')
        
        states = list_saved_states(temp_dir)
        
        assert len(states) == 2
