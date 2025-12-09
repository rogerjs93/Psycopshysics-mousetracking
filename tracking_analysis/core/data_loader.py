"""
Data Loader Module
==================

Handles loading and parsing of tracking experiment CSV files.
Extracts metadata from filenames, caches loaded data, and provides
filtering capabilities.

Classes:
    DataLoader: Main class for loading and managing tracking data
    
Functions:
    load_all: Load all CSV files from data directory
    load_filtered: Load subset based on participants/conditions
    get_metadata: Extract metadata without loading full data
    estimate_runtime: Estimate processing time based on data size
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import lru_cache
from dataclasses import dataclass
import time

from .config import (
    Config, DEFAULTS, SD_VALUES, CONDITION_LABELS,
    FRAME_RATE, FRAME_DURATION_MS, SCREEN_WIDTH, SCREEN_HEIGHT
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FileMetadata:
    """
    Metadata extracted from a single data file.
    
    Attributes:
        filepath: Full path to the CSV file
        participant_id: Unique participant identifier
        size: Blob size in arcmin (21, 31, or 34)
        size_pixels: Blob SD in pixels
        condition: 'dynamic' or 'static'
        condition_label: Human-readable condition name
        version: Trial version (v1, v2, v3)
        filename: Original filename without path
    """
    filepath: Path
    participant_id: str
    size: str  # e.g., "21arcmin"
    size_pixels: int  # e.g., 21
    condition: str  # 'dynamic' or 'static'
    condition_label: str  # 'Auditory Feedback' or 'No Feedback'
    version: str  # 'v1', 'v2', 'v3'
    filename: str


@dataclass
class DatasetMetadata:
    """
    Metadata for the entire dataset.
    
    Attributes:
        data_path: Path to data directory
        n_files: Total number of CSV files
        n_participants: Number of unique participants
        participants: List of participant IDs
        sizes: List of blob sizes found
        conditions: List of conditions found
        files_per_participant: Number of files per participant
        files: List of FileMetadata for each file
    """
    data_path: Path
    n_files: int
    n_participants: int
    participants: List[str]
    sizes: List[str]
    conditions: List[str]
    files_per_participant: Dict[str, int]
    files: List[FileMetadata]


# =============================================================================
# FILENAME PARSING
# =============================================================================

# Regex pattern for parsing filenames
# Example: Participant_1341_Tracking_blob_experiment_21arcmin_v1_dynamic.csv
FILENAME_PATTERN = re.compile(
    r'Participant_(\d+)_Tracking_blob_experiment_(\d+)arcmin_(v\d+)_(dynamic|static)\.csv',
    re.IGNORECASE
)


def parse_filename(filepath: Union[str, Path]) -> Optional[FileMetadata]:
    """
    Parse metadata from a tracking data filename.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        FileMetadata object or None if filename doesn't match pattern
        
    Example:
        >>> meta = parse_filename('Participant_1341_Tracking_blob_experiment_21arcmin_v1_dynamic.csv')
        >>> meta.participant_id
        '1341'
        >>> meta.size_pixels
        21
        >>> meta.condition_label
        'Auditory Feedback'
    """
    filepath = Path(filepath)
    match = FILENAME_PATTERN.match(filepath.name)
    
    if not match:
        return None
    
    participant_id = match.group(1)
    size_pixels = int(match.group(2))
    version = match.group(3)
    condition = match.group(4).lower()
    
    return FileMetadata(
        filepath=filepath,
        participant_id=participant_id,
        size=f"{size_pixels}arcmin",
        size_pixels=size_pixels,
        condition=condition,
        condition_label=CONDITION_LABELS.get(condition, condition),
        version=version,
        filename=filepath.name
    )


# =============================================================================
# DATA LOADER CLASS
# =============================================================================

class DataLoader:
    """
    Main class for loading and managing tracking experiment data.
    
    Provides methods for:
    - Scanning data directories and extracting metadata
    - Loading CSV files with caching for performance
    - Filtering data by participants, sizes, and conditions
    - Estimating runtime for processing
    
    Attributes:
        data_path: Path to data directory
        config: Configuration object
        metadata: DatasetMetadata after scanning
        
    Example:
        >>> loader = DataLoader('./data')
        >>> loader.scan()
        >>> print(f"Found {loader.metadata.n_files} files from {loader.metadata.n_participants} participants")
        >>> 
        >>> # Load all data
        >>> df = loader.load_all()
        >>> 
        >>> # Load filtered subset
        >>> df = loader.load_filtered(
        ...     participants=['1341', '3272'],
        ...     sizes=[21, 31],
        ...     conditions=['dynamic']
        ... )
    """
    
    def __init__(self, data_path: Union[str, Path], config: Optional[Config] = None):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to directory containing CSV files
            config: Optional Config object (uses defaults if not provided)
        """
        self.data_path = Path(data_path)
        self.config = config or Config()
        self.metadata: Optional[DatasetMetadata] = None
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def scan(self) -> DatasetMetadata:
        """
        Scan data directory and extract metadata from all files.
        
        Returns:
            DatasetMetadata object with information about all files
            
        Raises:
            FileNotFoundError: If data_path doesn't exist
            ValueError: If no valid CSV files found
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
        files: List[FileMetadata] = []
        participants = set()
        sizes = set()
        conditions = set()
        
        for csv_file in self.data_path.glob('*.csv'):
            meta = parse_filename(csv_file)
            if meta:
                files.append(meta)
                participants.add(meta.participant_id)
                sizes.add(meta.size)
                conditions.add(meta.condition)
        
        if not files:
            raise ValueError(f"No valid tracking data files found in {self.data_path}")
        
        # Count files per participant
        files_per_participant = {}
        for f in files:
            files_per_participant[f.participant_id] = files_per_participant.get(f.participant_id, 0) + 1
        
        # Sort for consistent ordering
        participants_list = sorted(participants)
        sizes_list = sorted(sizes, key=lambda x: int(x.replace('arcmin', '')))
        conditions_list = sorted(conditions)
        
        self.metadata = DatasetMetadata(
            data_path=self.data_path,
            n_files=len(files),
            n_participants=len(participants),
            participants=participants_list,
            sizes=sizes_list,
            conditions=conditions_list,
            files_per_participant=files_per_participant,
            files=files
        )
        
        return self.metadata
    
    def get_metadata(self) -> DatasetMetadata:
        """
        Get dataset metadata, scanning if necessary.
        
        Returns:
            DatasetMetadata object
        """
        if self.metadata is None:
            self.scan()
        return self.metadata
    
    def _load_single_file(self, filepath: Path) -> pd.DataFrame:
        """
        Load a single CSV file with caching.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with tracking data
        """
        cache_key = str(filepath)
        
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        df = pd.read_csv(filepath)
        
        # Validate expected columns
        expected_cols = {'Frame', 'Target_X', 'Target_Y', 'Mouse_X', 'Mouse_Y'}
        if not expected_cols.issubset(df.columns):
            missing = expected_cols - set(df.columns)
            raise ValueError(f"Missing columns in {filepath.name}: {missing}")
        
        # Cache the loaded data
        self._cache[cache_key] = df
        
        return df.copy()
    
    def load_file_with_metadata(self, file_meta: FileMetadata) -> pd.DataFrame:
        """
        Load a single file and add metadata columns.
        
        Args:
            file_meta: FileMetadata object for the file
            
        Returns:
            DataFrame with tracking data plus metadata columns
        """
        df = self._load_single_file(file_meta.filepath)
        
        # Add metadata columns
        df['participant_id'] = file_meta.participant_id
        df['size'] = file_meta.size
        df['size_pixels'] = file_meta.size_pixels
        df['condition'] = file_meta.condition
        df['condition_label'] = file_meta.condition_label
        df['version'] = file_meta.version
        df['filename'] = file_meta.filename
        
        # Add computed columns
        df['time_ms'] = df['Frame'] * FRAME_DURATION_MS
        df['time_sec'] = df['Frame'] / FRAME_RATE
        
        return df
    
    def load_all(self, add_metadata: bool = True) -> pd.DataFrame:
        """
        Load all CSV files from data directory.
        
        Args:
            add_metadata: Whether to add metadata columns to DataFrame
            
        Returns:
            Combined DataFrame with all tracking data
            
        Example:
            >>> loader = DataLoader('./data')
            >>> df = loader.load_all()
            >>> df.groupby(['participant_id', 'condition'])['Frame'].count()
        """
        if self.metadata is None:
            self.scan()
        
        dfs = []
        for file_meta in self.metadata.files:
            if add_metadata:
                df = self.load_file_with_metadata(file_meta)
            else:
                df = self._load_single_file(file_meta.filepath)
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
    
    def load_filtered(
        self,
        participants: Optional[List[str]] = None,
        sizes: Optional[List[int]] = None,
        conditions: Optional[List[str]] = None,
        add_metadata: bool = True
    ) -> pd.DataFrame:
        """
        Load subset of data based on filters.
        
        Args:
            participants: List of participant IDs to include (None = all)
            sizes: List of blob sizes in pixels to include (None = all)
            conditions: List of conditions to include (None = all)
            add_metadata: Whether to add metadata columns
            
        Returns:
            Filtered DataFrame
            
        Example:
            >>> loader = DataLoader('./data')
            >>> df = loader.load_filtered(
            ...     participants=['1341', '3272'],
            ...     sizes=[21],
            ...     conditions=['dynamic']
            ... )
        """
        if self.metadata is None:
            self.scan()
        
        # Convert sizes to strings for comparison
        size_strs = None
        if sizes is not None:
            size_strs = [f"{s}arcmin" for s in sizes]
        
        dfs = []
        for file_meta in self.metadata.files:
            # Apply filters
            if participants is not None and file_meta.participant_id not in participants:
                continue
            if size_strs is not None and file_meta.size not in size_strs:
                continue
            if conditions is not None and file_meta.condition not in conditions:
                continue
            
            if add_metadata:
                df = self.load_file_with_metadata(file_meta)
            else:
                df = self._load_single_file(file_meta.filepath)
            dfs.append(df)
        
        if not dfs:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'Frame', 'Target_X', 'Target_Y', 'Mouse_X', 'Mouse_Y',
                'participant_id', 'size', 'size_pixels', 'condition',
                'condition_label', 'version', 'filename', 'time_ms', 'time_sec'
            ])
        
        return pd.concat(dfs, ignore_index=True)
    
    def get_trial_data(
        self,
        participant_id: str,
        size: int,
        condition: str
    ) -> Optional[pd.DataFrame]:
        """
        Get data for a specific trial.
        
        Args:
            participant_id: Participant ID
            size: Blob size in pixels
            condition: 'dynamic' or 'static'
            
        Returns:
            DataFrame for the trial or None if not found
        """
        if self.metadata is None:
            self.scan()
        
        size_str = f"{size}arcmin"
        
        for file_meta in self.metadata.files:
            if (file_meta.participant_id == participant_id and
                file_meta.size == size_str and
                file_meta.condition == condition):
                return self.load_file_with_metadata(file_meta)
        
        return None
    
    def clear_cache(self) -> None:
        """Clear the data cache to free memory."""
        self._cache.clear()
    
    def estimate_runtime(
        self,
        n_files: Optional[int] = None,
        metrics_complexity: str = 'standard'
    ) -> Dict[str, Any]:
        """
        Estimate runtime for processing based on data size.
        
        Args:
            n_files: Number of files to process (None = all files)
            metrics_complexity: 'minimal', 'standard', or 'full'
            
        Returns:
            Dictionary with estimated times for different operations
            
        Example:
            >>> loader = DataLoader('./data')
            >>> loader.scan()
            >>> estimate = loader.estimate_runtime()
            >>> print(f"Total estimated time: {estimate['total_formatted']}")
        """
        if self.metadata is None:
            self.scan()
        
        if n_files is None:
            n_files = self.metadata.n_files
        
        # Base timing estimates (seconds per file, empirically determined)
        timing_factors = {
            'load': 0.05,  # Loading CSV
            'preprocess': 0.1,  # Preprocessing (velocity, outliers)
            'metrics': {
                'minimal': 0.05,  # Just RMSE
                'standard': 0.15,  # RMSE + cross-correlation
                'full': 0.3  # All metrics + detailed analysis
            },
            'visualization_per_plot': 0.5,
            'animation_per_file': 2.0,
            'statistics': 0.5  # Per analysis
        }
        
        # Calculate estimates
        load_time = n_files * timing_factors['load']
        preprocess_time = n_files * timing_factors['preprocess']
        metrics_time = n_files * timing_factors['metrics'].get(metrics_complexity, 0.15)
        
        # Visualization estimates
        n_plot_types = 5  # trajectory, error, boxplot, heatmap, lag
        viz_time = n_plot_types * timing_factors['visualization_per_plot']
        
        # Animation estimate (optional)
        animation_time = min(10, n_files * 0.5) * timing_factors['animation_per_file']
        
        # Statistics
        stats_time = 3 * timing_factors['statistics']  # 3 main analyses
        
        total_time = load_time + preprocess_time + metrics_time + viz_time + stats_time
        total_with_animation = total_time + animation_time
        
        def format_time(seconds: float) -> str:
            """Format seconds into human-readable string."""
            if seconds < 60:
                return f"{seconds:.0f} seconds"
            elif seconds < 3600:
                minutes = seconds / 60
                return f"{minutes:.1f} minutes"
            else:
                hours = seconds / 3600
                return f"{hours:.1f} hours"
        
        return {
            'n_files': n_files,
            'load_seconds': load_time,
            'preprocess_seconds': preprocess_time,
            'metrics_seconds': metrics_time,
            'visualization_seconds': viz_time,
            'animation_seconds': animation_time,
            'statistics_seconds': stats_time,
            'total_seconds': total_time,
            'total_with_animation_seconds': total_with_animation,
            'total_formatted': format_time(total_time),
            'total_with_animation_formatted': format_time(total_with_animation),
            'breakdown': {
                'Loading data': format_time(load_time),
                'Preprocessing': format_time(preprocess_time),
                'Computing metrics': format_time(metrics_time),
                'Generating visualizations': format_time(viz_time),
                'Statistical analysis': format_time(stats_time),
                'Animations (if enabled)': format_time(animation_time)
            }
        }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the dataset.
        
        Returns:
            Dictionary with dataset summary information
        """
        if self.metadata is None:
            self.scan()
        
        # Load a sample file to get frame count
        sample_file = self.metadata.files[0]
        sample_df = self._load_single_file(sample_file.filepath)
        n_frames = len(sample_df)
        
        return {
            'data_path': str(self.metadata.data_path),
            'n_files': self.metadata.n_files,
            'n_participants': self.metadata.n_participants,
            'participants': self.metadata.participants,
            'sizes': self.metadata.sizes,
            'conditions': self.metadata.conditions,
            'n_frames_per_trial': n_frames,
            'trial_duration_sec': n_frames / FRAME_RATE,
            'frame_rate_fps': FRAME_RATE,
            'screen_dimensions': {'width': SCREEN_WIDTH, 'height': SCREEN_HEIGHT},
            'experimental_design': f"{len(self.metadata.sizes)} sizes × {len(self.metadata.conditions)} conditions (within-subjects)",
            'files_per_participant': self.metadata.files_per_participant
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_all(data_path: Union[str, Path], config: Optional[Config] = None) -> pd.DataFrame:
    """
    Convenience function to load all data from a directory.
    
    Args:
        data_path: Path to data directory
        config: Optional configuration
        
    Returns:
        Combined DataFrame with all tracking data
    """
    loader = DataLoader(data_path, config)
    return loader.load_all()


def load_filtered(
    data_path: Union[str, Path],
    participants: Optional[List[str]] = None,
    sizes: Optional[List[int]] = None,
    conditions: Optional[List[str]] = None,
    config: Optional[Config] = None
) -> pd.DataFrame:
    """
    Convenience function to load filtered data.
    
    Args:
        data_path: Path to data directory
        participants: Participant IDs to include
        sizes: Blob sizes to include
        conditions: Conditions to include
        config: Optional configuration
        
    Returns:
        Filtered DataFrame
    """
    loader = DataLoader(data_path, config)
    return loader.load_filtered(participants, sizes, conditions)


def get_metadata(data_path: Union[str, Path]) -> DatasetMetadata:
    """
    Convenience function to get dataset metadata.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        DatasetMetadata object
    """
    loader = DataLoader(data_path)
    return loader.get_metadata()


def estimate_runtime(
    data_path: Union[str, Path],
    n_files: Optional[int] = None,
    metrics_complexity: str = 'standard'
) -> Dict[str, Any]:
    """
    Convenience function to estimate runtime.
    
    Args:
        data_path: Path to data directory
        n_files: Number of files (None = all)
        metrics_complexity: Complexity level
        
    Returns:
        Runtime estimates dictionary
    """
    loader = DataLoader(data_path)
    return loader.estimate_runtime(n_files, metrics_complexity)
