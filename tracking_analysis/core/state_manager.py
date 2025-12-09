"""
State Manager Module
====================

Handles saving and loading of analysis states for reproducibility
and fast re-loading of computed results.

State files are saved as paired files:
- .pkl: Binary pickle file with computed data (fast to load)
- .yaml: Human-readable configuration (version controllable)

Classes:
    StateManager: Main class for state management

Functions:
    save_state: Save analysis state to files
    load_state: Load analysis state from files
    list_saved_states: List available saved states
"""

import pickle
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import hashlib
import pandas as pd
import numpy as np

from .config import Config


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StateInfo:
    """
    Information about a saved state.
    
    Attributes:
        state_id: Unique identifier for the state
        state_path: Path to state directory
        created_at: Creation timestamp
        config_summary: Summary of configuration used
        data_summary: Summary of data processed
        has_metrics: Whether metrics are saved
        has_xcorr: Whether cross-correlation results are saved
        has_statistics: Whether statistical results are saved
        file_size_mb: Total size of state files
    """
    state_id: str
    state_path: Path
    created_at: str
    config_summary: Dict[str, Any]
    data_summary: Dict[str, Any]
    has_metrics: bool
    has_xcorr: bool
    has_statistics: bool
    file_size_mb: float


@dataclass
class AnalysisState:
    """
    Complete analysis state for saving/loading.
    
    Attributes:
        state_id: Unique identifier
        created_at: Creation timestamp
        config: Configuration used
        
        raw_data: Original loaded data (optional)
        processed_data: Preprocessed data
        trial_metrics: Per-trial metrics DataFrame
        xcorr_results: Cross-correlation results DataFrame
        statistical_results: Statistical analysis results
        
        preprocessing_report: Report from preprocessing
        data_metadata: Metadata about loaded data
    """
    state_id: str
    created_at: str
    config: Dict[str, Any]
    
    raw_data: Optional[pd.DataFrame] = None
    processed_data: Optional[pd.DataFrame] = None
    trial_metrics: Optional[pd.DataFrame] = None
    xcorr_results: Optional[pd.DataFrame] = None
    statistical_results: Optional[Dict[str, Any]] = None
    
    preprocessing_report: Optional[Dict[str, Any]] = None
    data_metadata: Optional[Dict[str, Any]] = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_state_id(config: Config, timestamp: Optional[datetime] = None) -> str:
    """
    Generate unique state identifier.
    
    Based on timestamp and config hash for uniqueness.
    
    Args:
        config: Configuration object
        timestamp: Optional timestamp (uses now if not provided)
        
    Returns:
        Unique state ID string
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Create hash from config
    config_str = str(config.to_dict())
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # Format: YYYYMMDD_HHMMSS_hash
    time_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    return f"{time_str}_{config_hash}"


def get_directory_size(path: Path) -> float:
    """
    Calculate total size of directory in MB.
    
    Args:
        path: Directory path
        
    Returns:
        Size in megabytes
    """
    total_size = 0
    for file in path.rglob('*'):
        if file.is_file():
            total_size += file.stat().st_size
    return total_size / (1024 * 1024)


def dataframe_to_serializable(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert DataFrame to JSON-serializable dictionary.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Dictionary representation
    """
    return {
        'data': df.to_dict(orient='records'),
        'columns': list(df.columns),
        'index': list(df.index),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


def serializable_to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert serializable dictionary back to DataFrame.
    
    Args:
        data: Dictionary from dataframe_to_serializable
        
    Returns:
        Reconstructed DataFrame
    """
    df = pd.DataFrame(data['data'])
    
    # Restore column order
    if 'columns' in data:
        df = df[data['columns']]
    
    return df


# =============================================================================
# STATE MANAGER CLASS
# =============================================================================

class StateManager:
    """
    Manages saving and loading of analysis states.
    
    Saves states as paired files:
    - state_data.pkl: Binary data (DataFrames, large arrays)
    - state_config.yaml: Human-readable configuration
    - state_info.json: Quick-load metadata
    
    Attributes:
        base_path: Base directory for saving states
        
    Example:
        >>> manager = StateManager('./results/states')
        >>> 
        >>> # Save state
        >>> state_id = manager.save_state(
        ...     config=config,
        ...     trial_metrics=trial_metrics,
        ...     xcorr_results=xcorr_results,
        ...     statistical_results=stats_results
        ... )
        >>> 
        >>> # List saved states
        >>> states = manager.list_states()
        >>> 
        >>> # Load state
        >>> loaded = manager.load_state(state_id)
    """
    
    def __init__(self, base_path: Union[str, Path] = './results/states'):
        """
        Initialize StateManager.
        
        Args:
            base_path: Base directory for saving states
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_state(
        self,
        config: Config,
        trial_metrics: Optional[pd.DataFrame] = None,
        xcorr_results: Optional[pd.DataFrame] = None,
        statistical_results: Optional[Dict[str, Any]] = None,
        processed_data: Optional[pd.DataFrame] = None,
        raw_data: Optional[pd.DataFrame] = None,
        preprocessing_report: Optional[Dict[str, Any]] = None,
        data_metadata: Optional[Dict[str, Any]] = None,
        state_id: Optional[str] = None
    ) -> str:
        """
        Save analysis state to files.
        
        Args:
            config: Configuration used for analysis
            trial_metrics: Per-trial metrics DataFrame
            xcorr_results: Cross-correlation results DataFrame
            statistical_results: Statistical analysis results dict
            processed_data: Preprocessed tracking data
            raw_data: Raw loaded data (optional, large)
            preprocessing_report: Preprocessing report dict
            data_metadata: Metadata about loaded data
            state_id: Optional custom state ID
            
        Returns:
            State ID of saved state
        """
        timestamp = datetime.now()
        
        if state_id is None:
            state_id = generate_state_id(config, timestamp)
        
        # Create state directory
        state_dir = self.base_path / state_id
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare state object
        state = AnalysisState(
            state_id=state_id,
            created_at=timestamp.isoformat(),
            config=config.to_dict(),
            raw_data=raw_data,
            processed_data=processed_data,
            trial_metrics=trial_metrics,
            xcorr_results=xcorr_results,
            statistical_results=statistical_results,
            preprocessing_report=preprocessing_report,
            data_metadata=data_metadata
        )
        
        # Save pickle file (binary data)
        pkl_path = state_dir / 'state_data.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'raw_data': state.raw_data,
                'processed_data': state.processed_data,
                'trial_metrics': state.trial_metrics,
                'xcorr_results': state.xcorr_results,
                'statistical_results': state.statistical_results,
                'preprocessing_report': state.preprocessing_report
            }, f)
        
        # Save config YAML (human-readable)
        yaml_path = state_dir / 'state_config.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump({
                'state_id': state.state_id,
                'created_at': state.created_at,
                'config': state.config,
                'data_metadata': state.data_metadata
            }, f, default_flow_style=False, sort_keys=False)
        
        # Save quick-load info JSON
        info = self._create_state_info(state, state_dir)
        info_path = state_dir / 'state_info.json'
        with open(info_path, 'w') as f:
            json.dump({
                'state_id': info.state_id,
                'created_at': info.created_at,
                'config_summary': info.config_summary,
                'data_summary': info.data_summary,
                'has_metrics': info.has_metrics,
                'has_xcorr': info.has_xcorr,
                'has_statistics': info.has_statistics,
                'file_size_mb': info.file_size_mb
            }, f, indent=2)
        
        return state_id
    
    def load_state(
        self,
        state_id: str,
        load_raw_data: bool = False
    ) -> AnalysisState:
        """
        Load analysis state from files.
        
        Args:
            state_id: ID of state to load
            load_raw_data: Whether to load raw data (may be large)
            
        Returns:
            AnalysisState object with loaded data
            
        Raises:
            FileNotFoundError: If state not found
        """
        state_dir = self.base_path / state_id
        
        if not state_dir.exists():
            raise FileNotFoundError(f"State not found: {state_id}")
        
        # Load config YAML
        yaml_path = state_dir / 'state_config.yaml'
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Load pickle data
        pkl_path = state_dir / 'state_data.pkl'
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        
        # Create state object
        state = AnalysisState(
            state_id=yaml_data['state_id'],
            created_at=yaml_data['created_at'],
            config=yaml_data['config'],
            raw_data=pkl_data.get('raw_data') if load_raw_data else None,
            processed_data=pkl_data.get('processed_data'),
            trial_metrics=pkl_data.get('trial_metrics'),
            xcorr_results=pkl_data.get('xcorr_results'),
            statistical_results=pkl_data.get('statistical_results'),
            preprocessing_report=pkl_data.get('preprocessing_report'),
            data_metadata=yaml_data.get('data_metadata')
        )
        
        return state
    
    def list_states(self) -> List[StateInfo]:
        """
        List all saved states.
        
        Returns:
            List of StateInfo objects sorted by creation time (newest first)
        """
        states = []
        
        for state_dir in self.base_path.iterdir():
            if state_dir.is_dir():
                info = self.get_state_info(state_dir.name)
                if info is not None:
                    states.append(info)
        
        # Sort by creation time (newest first)
        states.sort(key=lambda x: x.created_at, reverse=True)
        
        return states
    
    def get_state_info(self, state_id: str) -> Optional[StateInfo]:
        """
        Get information about a saved state without fully loading it.
        
        Args:
            state_id: ID of state
            
        Returns:
            StateInfo object or None if not found
        """
        state_dir = self.base_path / state_id
        info_path = state_dir / 'state_info.json'
        
        if not info_path.exists():
            # Try to reconstruct from other files
            return self._reconstruct_state_info(state_id)
        
        with open(info_path, 'r') as f:
            info_data = json.load(f)
        
        return StateInfo(
            state_id=info_data['state_id'],
            state_path=state_dir,
            created_at=info_data['created_at'],
            config_summary=info_data.get('config_summary', {}),
            data_summary=info_data.get('data_summary', {}),
            has_metrics=info_data.get('has_metrics', False),
            has_xcorr=info_data.get('has_xcorr', False),
            has_statistics=info_data.get('has_statistics', False),
            file_size_mb=info_data.get('file_size_mb', 0)
        )
    
    def delete_state(self, state_id: str) -> bool:
        """
        Delete a saved state.
        
        Args:
            state_id: ID of state to delete
            
        Returns:
            True if deleted, False if not found
        """
        state_dir = self.base_path / state_id
        
        if not state_dir.exists():
            return False
        
        # Remove all files in state directory
        for file in state_dir.iterdir():
            file.unlink()
        
        # Remove directory
        state_dir.rmdir()
        
        return True
    
    def estimate_load_time(self, state_id: str) -> Dict[str, Any]:
        """
        Estimate time to load a state based on file sizes.
        
        Args:
            state_id: ID of state
            
        Returns:
            Dictionary with size and time estimates
        """
        state_dir = self.base_path / state_id
        
        if not state_dir.exists():
            return {'error': 'State not found'}
        
        pkl_path = state_dir / 'state_data.pkl'
        size_mb = pkl_path.stat().st_size / (1024 * 1024) if pkl_path.exists() else 0
        
        # Rough estimate: ~100MB/s load speed
        estimated_seconds = size_mb / 100
        
        return {
            'state_id': state_id,
            'file_size_mb': size_mb,
            'estimated_load_seconds': estimated_seconds,
            'estimated_load_formatted': f"{estimated_seconds:.1f} seconds" if estimated_seconds >= 1 else "< 1 second"
        }
    
    def _create_state_info(self, state: AnalysisState, state_dir: Path) -> StateInfo:
        """Create StateInfo from AnalysisState."""
        # Config summary
        config_summary = {
            'data_path': state.config.get('data_path', 'unknown'),
            'velocity_method': state.config.get('velocity_method', 'unknown'),
            'outlier_method': state.config.get('outlier_method', 'unknown'),
            'normalize_xcorr': state.config.get('normalize_xcorr', True)
        }
        
        # Data summary
        data_summary = {}
        if state.data_metadata:
            data_summary = {
                'n_participants': state.data_metadata.get('n_participants', 0),
                'n_files': state.data_metadata.get('n_files', 0)
            }
        elif state.trial_metrics is not None:
            data_summary = {
                'n_participants': state.trial_metrics['participant_id'].nunique() if 'participant_id' in state.trial_metrics.columns else 0,
                'n_trials': len(state.trial_metrics)
            }
        
        return StateInfo(
            state_id=state.state_id,
            state_path=state_dir,
            created_at=state.created_at,
            config_summary=config_summary,
            data_summary=data_summary,
            has_metrics=state.trial_metrics is not None,
            has_xcorr=state.xcorr_results is not None,
            has_statistics=state.statistical_results is not None,
            file_size_mb=get_directory_size(state_dir)
        )
    
    def _reconstruct_state_info(self, state_id: str) -> Optional[StateInfo]:
        """Reconstruct StateInfo from config and data files."""
        state_dir = self.base_path / state_id
        yaml_path = state_dir / 'state_config.yaml'
        pkl_path = state_dir / 'state_data.pkl'
        
        if not yaml_path.exists():
            return None
        
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Check what's in the pickle
        has_metrics = False
        has_xcorr = False
        has_stats = False
        
        if pkl_path.exists():
            try:
                with open(pkl_path, 'rb') as f:
                    pkl_data = pickle.load(f)
                has_metrics = pkl_data.get('trial_metrics') is not None
                has_xcorr = pkl_data.get('xcorr_results') is not None
                has_stats = pkl_data.get('statistical_results') is not None
            except:
                pass
        
        return StateInfo(
            state_id=state_id,
            state_path=state_dir,
            created_at=yaml_data.get('created_at', 'unknown'),
            config_summary=yaml_data.get('config', {}),
            data_summary=yaml_data.get('data_metadata', {}),
            has_metrics=has_metrics,
            has_xcorr=has_xcorr,
            has_statistics=has_stats,
            file_size_mb=get_directory_size(state_dir)
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def save_state(
    output_dir: Union[str, Path],
    config: Config,
    trial_metrics: Optional[pd.DataFrame] = None,
    xcorr_results: Optional[pd.DataFrame] = None,
    statistical_results: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Convenience function to save analysis state.
    
    Args:
        output_dir: Directory to save state
        config: Configuration used
        trial_metrics: Trial metrics DataFrame
        xcorr_results: Cross-correlation results
        statistical_results: Statistical results
        **kwargs: Additional data to save
        
    Returns:
        State ID
    """
    manager = StateManager(output_dir)
    return manager.save_state(
        config=config,
        trial_metrics=trial_metrics,
        xcorr_results=xcorr_results,
        statistical_results=statistical_results,
        **kwargs
    )


def load_state(
    state_path: Union[str, Path],
    state_id: Optional[str] = None
) -> AnalysisState:
    """
    Convenience function to load analysis state.
    
    Args:
        state_path: Path to states directory or specific state
        state_id: State ID if state_path is base directory
        
    Returns:
        Loaded AnalysisState
    """
    state_path = Path(state_path)
    
    if state_id is not None:
        manager = StateManager(state_path)
        return manager.load_state(state_id)
    elif (state_path / 'state_data.pkl').exists():
        # Direct path to state directory
        manager = StateManager(state_path.parent)
        return manager.load_state(state_path.name)
    else:
        raise ValueError("Must provide state_id or path to specific state directory")


def list_saved_states(base_path: Union[str, Path]) -> List[StateInfo]:
    """
    Convenience function to list saved states.
    
    Args:
        base_path: Base directory containing states
        
    Returns:
        List of StateInfo objects
    """
    manager = StateManager(base_path)
    return manager.list_states()


def get_state_info(base_path: Union[str, Path], state_id: str) -> Optional[StateInfo]:
    """
    Convenience function to get state info.
    
    Args:
        base_path: Base directory
        state_id: State ID
        
    Returns:
        StateInfo or None
    """
    manager = StateManager(base_path)
    return manager.get_state_info(state_id)
