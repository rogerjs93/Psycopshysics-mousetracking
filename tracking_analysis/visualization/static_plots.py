"""
Static Plots Module
===================

Creates static matplotlib visualizations for tracking analysis results.

Includes:
- Error distribution plots
- RMSE comparison bar charts
- Cross-correlation plots
- Statistical summary visualizations
- Tracking trajectory plots
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass

# Style configuration
STYLE_CONFIG = {
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
}

# Color schemes
CONDITION_COLORS = {
    'dynamic': '#2ecc71',    # Green for auditory feedback
    'static': '#e74c3c'      # Red for no feedback
}

SIZE_COLORS = {
    21: '#3498db',  # Blue
    31: '#9b59b6',  # Purple
    34: '#f39c12'   # Orange
}

# Apply default style
plt.rcParams.update(STYLE_CONFIG)


@dataclass
class PlotConfig:
    """Configuration for plot generation."""
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 100
    save_dpi: int = 150
    style: str = 'seaborn-v0_8-whitegrid'
    show_grid: bool = True
    show_legend: bool = True


class StaticPlotter:
    """
    Creates static visualizations for tracking analysis.
    
    Attributes:
        config: Plot configuration
        output_dir: Directory for saving plots
        
    Example:
        >>> plotter = StaticPlotter(output_dir='./plots')
        >>> 
        >>> # RMSE comparison
        >>> fig = plotter.plot_rmse_by_size(trial_metrics)
        >>> plotter.save_figure(fig, 'rmse_by_size.png')
        >>> 
        >>> # Cross-correlation
        >>> fig = plotter.plot_xcorr(xcorr_result)
        >>> plotter.save_figure(fig, 'xcorr.png')
    """
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[PlotConfig] = None
    ):
        """
        Initialize StaticPlotter.
        
        Args:
            output_dir: Directory for saving plots
            config: Plot configuration
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.config = config or PlotConfig()
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        **kwargs
    ) -> Path:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            **kwargs: Additional savefig arguments
            
        Returns:
            Path to saved file
        """
        if self.output_dir is None:
            self.output_dir = Path('.')
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.config.save_dpi, **kwargs)
        plt.close(fig)
        
        return filepath
    
    # =========================================================================
    # RMSE PLOTS
    # =========================================================================
    
    def plot_rmse_by_size(
        self,
        trial_metrics: pd.DataFrame,
        show_individual: bool = True,
        show_stats: bool = True,
        title: str = "RMSE by Blob Size"
    ) -> plt.Figure:
        """
        Plot RMSE comparison across SD sizes.
        
        Args:
            trial_metrics: DataFrame with trial-level metrics
            show_individual: Show individual data points
            show_stats: Show mean ± SEM error bars
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        sizes = sorted(trial_metrics['size_pixels'].unique())
        positions = np.arange(len(sizes))
        
        if show_stats:
            # Calculate means and SEMs
            means = trial_metrics.groupby('size_pixels')['rmse'].mean()
            sems = trial_metrics.groupby('size_pixels')['rmse'].sem()
            
            # Bar plot
            bars = ax.bar(
                positions,
                [means[s] for s in sizes],
                yerr=[sems[s] for s in sizes],
                capsize=5,
                color=[SIZE_COLORS[s] for s in sizes],
                alpha=0.7,
                edgecolor='black',
                linewidth=1
            )
        
        if show_individual:
            # Scatter individual points
            for i, size in enumerate(sizes):
                size_data = trial_metrics[trial_metrics['size_pixels'] == size]['rmse']
                jitter = np.random.normal(0, 0.05, len(size_data))
                ax.scatter(
                    positions[i] + jitter,
                    size_data,
                    alpha=0.5,
                    color=SIZE_COLORS[size],
                    edgecolor='white',
                    s=30
                )
        
        ax.set_xticks(positions)
        ax.set_xticklabels([f'{s} arcmin' for s in sizes])
        ax.set_xlabel('Blob SD Size')
        ax.set_ylabel('RMSE (pixels)')
        ax.set_title(title)
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_rmse_by_condition(
        self,
        trial_metrics: pd.DataFrame,
        show_individual: bool = True,
        title: str = "RMSE by Condition"
    ) -> plt.Figure:
        """
        Plot RMSE comparison between conditions.
        
        Args:
            trial_metrics: DataFrame with trial-level metrics
            show_individual: Show individual data points
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        conditions = ['dynamic', 'static']
        labels = ['Auditory Feedback', 'No Feedback']
        positions = np.arange(len(conditions))
        
        # Calculate means and SEMs
        means = trial_metrics.groupby('condition')['rmse'].mean()
        sems = trial_metrics.groupby('condition')['rmse'].sem()
        
        # Bar plot
        bars = ax.bar(
            positions,
            [means[c] for c in conditions],
            yerr=[sems[c] for c in conditions],
            capsize=5,
            color=[CONDITION_COLORS[c] for c in conditions],
            alpha=0.7,
            edgecolor='black',
            linewidth=1
        )
        
        if show_individual:
            for i, cond in enumerate(conditions):
                cond_data = trial_metrics[trial_metrics['condition'] == cond]['rmse']
                jitter = np.random.normal(0, 0.05, len(cond_data))
                ax.scatter(
                    positions[i] + jitter,
                    cond_data,
                    alpha=0.5,
                    color=CONDITION_COLORS[cond],
                    edgecolor='white',
                    s=30
                )
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Condition')
        ax.set_ylabel('RMSE (pixels)')
        ax.set_title(title)
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_rmse_interaction(
        self,
        trial_metrics: pd.DataFrame,
        title: str = "RMSE: Size × Condition Interaction"
    ) -> plt.Figure:
        """
        Plot RMSE showing size × condition interaction.
        
        Args:
            trial_metrics: DataFrame with trial-level metrics
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        sizes = sorted(trial_metrics['size_pixels'].unique())
        conditions = ['dynamic', 'static']
        
        width = 0.35
        positions = np.arange(len(sizes))
        
        for i, cond in enumerate(conditions):
            cond_data = trial_metrics[trial_metrics['condition'] == cond]
            means = cond_data.groupby('size_pixels')['rmse'].mean()
            sems = cond_data.groupby('size_pixels')['rmse'].sem()
            
            offset = width * (i - 0.5)
            label = 'Auditory Feedback' if cond == 'dynamic' else 'No Feedback'
            
            ax.bar(
                positions + offset,
                [means[s] for s in sizes],
                width,
                yerr=[sems[s] for s in sizes],
                capsize=3,
                label=label,
                color=CONDITION_COLORS[cond],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
        
        ax.set_xticks(positions)
        ax.set_xticklabels([f'{s} arcmin' for s in sizes])
        ax.set_xlabel('Blob SD Size')
        ax.set_ylabel('RMSE (pixels)')
        ax.set_title(title)
        ax.legend()
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # CROSS-CORRELATION PLOTS
    # =========================================================================
    
    def plot_xcorr(
        self,
        lags: np.ndarray,
        correlations: np.ndarray,
        optimal_lag: int,
        peak_correlation: float,
        frame_rate: float = 50,
        title: str = "Velocity Cross-Correlation"
    ) -> plt.Figure:
        """
        Plot cross-correlation function.
        
        Args:
            lags: Array of lag values (frames)
            correlations: Array of correlation values
            optimal_lag: Optimal lag (frames)
            peak_correlation: Peak correlation value
            frame_rate: Frame rate for time conversion
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Convert lags to milliseconds
        lags_ms = lags * (1000 / frame_rate)
        optimal_lag_ms = optimal_lag * (1000 / frame_rate)
        
        # Plot correlation function
        ax.plot(lags_ms, correlations, 'b-', linewidth=2, label='Cross-correlation')
        
        # Mark peak
        ax.axvline(
            optimal_lag_ms,
            color='r',
            linestyle='--',
            linewidth=1.5,
            label=f'Peak: {optimal_lag_ms:.0f} ms'
        )
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        
        # Mark zero lag
        ax.axvline(0, color='gray', linestyle=':', linewidth=1)
        
        # Annotations
        ax.annotate(
            f'r = {peak_correlation:.3f}',
            xy=(optimal_lag_ms, peak_correlation),
            xytext=(optimal_lag_ms + 20, peak_correlation - 0.1),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='red')
        )
        
        ax.set_xlabel('Lag (ms)')
        ax.set_ylabel('Correlation')
        ax.set_title(title)
        ax.legend(loc='upper right')
        
        # Interpretation annotation
        if optimal_lag > 0:
            interp = "Reactive (mouse follows target)"
        elif optimal_lag < 0:
            interp = "Predictive (mouse leads target)"
        else:
            interp = "Synchronous"
        
        ax.text(
            0.02, 0.98,
            interp,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_lag_distribution(
        self,
        xcorr_results: pd.DataFrame,
        frame_rate: float = 50,
        title: str = "Distribution of Optimal Lags"
    ) -> plt.Figure:
        """
        Plot distribution of optimal lags across trials.
        
        Args:
            xcorr_results: DataFrame with cross-correlation results
            frame_rate: Frame rate for time conversion
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Convert to milliseconds
        lags_ms = xcorr_results['optimal_lag_frames'] * (1000 / frame_rate)
        
        # Histogram
        ax.hist(
            lags_ms,
            bins=30,
            alpha=0.7,
            color='steelblue',
            edgecolor='black'
        )
        
        # Mark zero
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero lag')
        
        # Mark mean
        mean_lag = lags_ms.mean()
        ax.axvline(
            mean_lag,
            color='orange',
            linestyle='-',
            linewidth=2,
            label=f'Mean: {mean_lag:.0f} ms'
        )
        
        ax.set_xlabel('Optimal Lag (ms)')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # TRAJECTORY PLOTS
    # =========================================================================
    
    def plot_trajectory(
        self,
        trial_data: pd.DataFrame,
        show_error: bool = True,
        title: str = "Tracking Trajectory"
    ) -> plt.Figure:
        """
        Plot single trial trajectory.
        
        Args:
            trial_data: DataFrame with trial data
            show_error: Show error shading
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        frames = trial_data['Frame']
        
        # X position
        axes[0].plot(frames, trial_data['Target_X'], 'b-', label='Target', linewidth=1.5)
        axes[0].plot(frames, trial_data['Mouse_X'], 'r-', alpha=0.7, label='Mouse', linewidth=1.5)
        
        if show_error:
            axes[0].fill_between(
                frames,
                trial_data['Target_X'],
                trial_data['Mouse_X'],
                alpha=0.2,
                color='purple'
            )
        
        axes[0].set_ylabel('X Position (pixels)')
        axes[0].legend(loc='upper right')
        axes[0].set_title('Horizontal Tracking')
        
        # Y position
        axes[1].plot(frames, trial_data['Target_Y'], 'b-', label='Target', linewidth=1.5)
        axes[1].plot(frames, trial_data['Mouse_Y'], 'r-', alpha=0.7, label='Mouse', linewidth=1.5)
        
        if show_error:
            axes[1].fill_between(
                frames,
                trial_data['Target_Y'],
                trial_data['Mouse_Y'],
                alpha=0.2,
                color='purple'
            )
        
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('Y Position (pixels)')
        axes[1].legend(loc='upper right')
        axes[1].set_title('Vertical Tracking')
        
        fig.suptitle(title, fontsize=14)
        
        for ax in axes:
            if self.config.show_grid:
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_2d_trajectory(
        self,
        trial_data: pd.DataFrame,
        color_by_time: bool = True,
        title: str = "2D Tracking Trajectory"
    ) -> plt.Figure:
        """
        Plot 2D trajectory on screen space.
        
        Args:
            trial_data: DataFrame with trial data
            color_by_time: Color trajectory by time
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if color_by_time:
            # Create line segments colored by time
            target_points = np.array([trial_data['Target_X'], trial_data['Target_Y']]).T.reshape(-1, 1, 2)
            target_segments = np.concatenate([target_points[:-1], target_points[1:]], axis=1)
            
            mouse_points = np.array([trial_data['Mouse_X'], trial_data['Mouse_Y']]).T.reshape(-1, 1, 2)
            mouse_segments = np.concatenate([mouse_points[:-1], mouse_points[1:]], axis=1)
            
            # Colormap based on frame number
            norm = plt.Normalize(trial_data['Frame'].min(), trial_data['Frame'].max())
            
            target_lc = LineCollection(target_segments, cmap='Blues', norm=norm, linewidth=2)
            target_lc.set_array(trial_data['Frame'].values[:-1])
            ax.add_collection(target_lc)
            
            mouse_lc = LineCollection(mouse_segments, cmap='Reds', norm=norm, linewidth=2, alpha=0.7)
            mouse_lc.set_array(trial_data['Frame'].values[:-1])
            ax.add_collection(mouse_lc)
            
            # Colorbar
            plt.colorbar(target_lc, ax=ax, label='Frame')
        else:
            ax.plot(trial_data['Target_X'], trial_data['Target_Y'], 'b-', label='Target', linewidth=1.5)
            ax.plot(trial_data['Mouse_X'], trial_data['Mouse_Y'], 'r-', alpha=0.7, label='Mouse', linewidth=1.5)
            ax.legend()
        
        # Mark start and end
        ax.scatter(trial_data['Target_X'].iloc[0], trial_data['Target_Y'].iloc[0],
                  color='green', s=100, marker='o', zorder=5, label='Start')
        ax.scatter(trial_data['Target_X'].iloc[-1], trial_data['Target_Y'].iloc[-1],
                  color='black', s=100, marker='x', zorder=5, label='End')
        
        ax.set_xlim(0, 1920)
        ax.set_ylim(0, 980)
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        # Legend for start/end markers
        blue_patch = mpatches.Patch(color='blue', label='Target')
        red_patch = mpatches.Patch(color='red', alpha=0.7, label='Mouse')
        ax.legend(handles=[blue_patch, red_patch], loc='upper right')
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # ERROR DISTRIBUTION PLOTS
    # =========================================================================
    
    def plot_error_distribution(
        self,
        errors: np.ndarray,
        title: str = "Tracking Error Distribution"
    ) -> plt.Figure:
        """
        Plot distribution of tracking errors.
        
        Args:
            errors: Array of error values
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Histogram
        ax.hist(
            errors,
            bins=50,
            alpha=0.7,
            color='steelblue',
            edgecolor='black',
            density=True
        )
        
        # Mark statistics
        mean = np.mean(errors)
        median = np.median(errors)
        
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f}')
        ax.axvline(median, color='orange', linestyle='-', linewidth=2, label=f'Median: {median:.1f}')
        
        ax.set_xlabel('Error (pixels)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # STATISTICAL SUMMARY PLOTS
    # =========================================================================
    
    def plot_summary_panel(
        self,
        trial_metrics: pd.DataFrame,
        xcorr_results: Optional[pd.DataFrame] = None,
        title: str = "Analysis Summary"
    ) -> plt.Figure:
        """
        Create multi-panel summary figure.
        
        Args:
            trial_metrics: DataFrame with trial metrics
            xcorr_results: Optional cross-correlation results
            title: Overall title
            
        Returns:
            Matplotlib figure
        """
        n_panels = 4 if xcorr_results is not None else 3
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Panel 1: RMSE by size
        sizes = sorted(trial_metrics['size_pixels'].unique())
        means = trial_metrics.groupby('size_pixels')['rmse'].mean()
        sems = trial_metrics.groupby('size_pixels')['rmse'].sem()
        
        axes[0].bar(
            range(len(sizes)),
            [means[s] for s in sizes],
            yerr=[sems[s] for s in sizes],
            capsize=5,
            color=[SIZE_COLORS[s] for s in sizes],
            alpha=0.7,
            edgecolor='black'
        )
        axes[0].set_xticks(range(len(sizes)))
        axes[0].set_xticklabels([f'{s} arcmin' for s in sizes])
        axes[0].set_ylabel('RMSE (pixels)')
        axes[0].set_title('RMSE by Blob Size')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Panel 2: RMSE by condition
        conditions = ['dynamic', 'static']
        cond_means = trial_metrics.groupby('condition')['rmse'].mean()
        cond_sems = trial_metrics.groupby('condition')['rmse'].sem()
        
        axes[1].bar(
            range(len(conditions)),
            [cond_means[c] for c in conditions],
            yerr=[cond_sems[c] for c in conditions],
            capsize=5,
            color=[CONDITION_COLORS[c] for c in conditions],
            alpha=0.7,
            edgecolor='black'
        )
        axes[1].set_xticks(range(len(conditions)))
        axes[1].set_xticklabels(['Auditory\nFeedback', 'No\nFeedback'])
        axes[1].set_ylabel('RMSE (pixels)')
        axes[1].set_title('RMSE by Condition')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Interaction plot
        for cond in conditions:
            cond_data = trial_metrics[trial_metrics['condition'] == cond]
            size_means = cond_data.groupby('size_pixels')['rmse'].mean()
            label = 'Auditory Feedback' if cond == 'dynamic' else 'No Feedback'
            axes[2].plot(
                range(len(sizes)),
                [size_means[s] for s in sizes],
                'o-',
                color=CONDITION_COLORS[cond],
                label=label,
                linewidth=2,
                markersize=8
            )
        
        axes[2].set_xticks(range(len(sizes)))
        axes[2].set_xticklabels([f'{s}' for s in sizes])
        axes[2].set_xlabel('Blob SD Size (arcmin)')
        axes[2].set_ylabel('RMSE (pixels)')
        axes[2].set_title('Size × Condition Interaction')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Panel 4: Lag distribution or participant summary
        if xcorr_results is not None and 'optimal_lag_frames' in xcorr_results.columns:
            lags_ms = xcorr_results['optimal_lag_frames'] * 20  # Assuming 50 fps
            axes[3].hist(lags_ms, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            axes[3].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[3].axvline(lags_ms.mean(), color='orange', linestyle='-', linewidth=2)
            axes[3].set_xlabel('Optimal Lag (ms)')
            axes[3].set_ylabel('Count')
            axes[3].set_title('Lag Distribution')
            axes[3].grid(True, alpha=0.3, axis='y')
        else:
            # Participant summary
            part_means = trial_metrics.groupby('participant_id')['rmse'].mean().sort_values()
            axes[3].barh(range(len(part_means)), part_means.values, alpha=0.7, color='steelblue')
            axes[3].set_yticks(range(len(part_means)))
            axes[3].set_yticklabels(part_means.index)
            axes[3].set_xlabel('Mean RMSE (pixels)')
            axes[3].set_title('RMSE by Participant')
            axes[3].grid(True, alpha=0.3, axis='x')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
