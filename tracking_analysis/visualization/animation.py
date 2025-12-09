"""
Animation Module
================

Creates animated visualizations for tracking data.

Supports:
- MP4 video output (requires ffmpeg)
- HTML5 interactive animations
- Plotly-based interactive plots

Animation Types:
- Trial replay: Shows target and mouse positions over time
- Error evolution: Shows tracking error changing over time
- Velocity comparison: Shows velocity signals
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass
import warnings

# Try importing animation libraries
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, HTMLWriter
    from matplotlib.patches import Circle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available for animations")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("plotly not available for interactive animations")


@dataclass
class AnimationConfig:
    """Configuration for animations."""
    fps: int = 30
    duration_multiplier: float = 1.0  # Speed up or slow down
    figsize: Tuple[int, int] = (10, 8)
    dpi: int = 100
    target_color: str = '#3498db'
    mouse_color: str = '#e74c3c'
    trail_length: int = 20
    show_trail: bool = True
    show_error_line: bool = True


class Animator:
    """
    Creates animated visualizations of tracking data.
    
    Attributes:
        config: Animation configuration
        output_dir: Directory for saving animations
        
    Example:
        >>> animator = Animator(output_dir='./animations')
        >>> 
        >>> # Create MP4
        >>> animator.create_trial_animation(
        ...     trial_data,
        ...     output_path='trial_1.mp4',
        ...     format='mp4'
        ... )
        >>> 
        >>> # Create HTML
        >>> animator.create_trial_animation(
        ...     trial_data,
        ...     output_path='trial_1.html',
        ...     format='html'
        ... )
    """
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[AnimationConfig] = None
    ):
        """
        Initialize Animator.
        
        Args:
            output_dir: Directory for saving animations
            config: Animation configuration
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.config = config or AnimationConfig()
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_trial_animation(
        self,
        trial_data: pd.DataFrame,
        output_path: Optional[Union[str, Path]] = None,
        format: str = 'mp4',
        title: str = "Tracking Trial"
    ) -> Optional[Path]:
        """
        Create animation of a single trial.
        
        Args:
            trial_data: DataFrame with trial data
            output_path: Output file path
            format: 'mp4', 'html', or 'plotly'
            title: Animation title
            
        Returns:
            Path to saved animation or None
        """
        if format == 'plotly':
            return self._create_plotly_animation(trial_data, output_path, title)
        elif format in ['mp4', 'html']:
            return self._create_matplotlib_animation(trial_data, output_path, format, title)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _create_matplotlib_animation(
        self,
        trial_data: pd.DataFrame,
        output_path: Optional[Union[str, Path]],
        format: str,
        title: str
    ) -> Optional[Path]:
        """Create matplotlib-based animation."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for animations")
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Set up screen bounds
        ax.set_xlim(0, 1920)
        ax.set_ylim(0, 980)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Initialize elements
        target_marker, = ax.plot([], [], 'o', color=self.config.target_color,
                                 markersize=15, label='Target')
        mouse_marker, = ax.plot([], [], 'o', color=self.config.mouse_color,
                                markersize=10, label='Mouse')
        
        if self.config.show_error_line:
            error_line, = ax.plot([], [], '-', color='gray', alpha=0.5, linewidth=1)
        
        if self.config.show_trail:
            target_trail, = ax.plot([], [], '-', color=self.config.target_color,
                                    alpha=0.3, linewidth=2)
            mouse_trail, = ax.plot([], [], '-', color=self.config.mouse_color,
                                   alpha=0.3, linewidth=2)
        
        # Time text
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Error text
        error_text = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                            fontsize=12, verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax.legend(loc='upper center')
        
        n_frames = len(trial_data)
        
        def init():
            target_marker.set_data([], [])
            mouse_marker.set_data([], [])
            if self.config.show_error_line:
                error_line.set_data([], [])
            if self.config.show_trail:
                target_trail.set_data([], [])
                mouse_trail.set_data([], [])
            time_text.set_text('')
            error_text.set_text('')
            return [target_marker, mouse_marker, time_text, error_text]
        
        def animate(frame_idx):
            row = trial_data.iloc[frame_idx]
            
            target_marker.set_data([row['Target_X']], [row['Target_Y']])
            mouse_marker.set_data([row['Mouse_X']], [row['Mouse_Y']])
            
            if self.config.show_error_line:
                error_line.set_data(
                    [row['Target_X'], row['Mouse_X']],
                    [row['Target_Y'], row['Mouse_Y']]
                )
            
            if self.config.show_trail:
                trail_start = max(0, frame_idx - self.config.trail_length)
                trail_data = trial_data.iloc[trail_start:frame_idx+1]
                target_trail.set_data(trail_data['Target_X'], trail_data['Target_Y'])
                mouse_trail.set_data(trail_data['Mouse_X'], trail_data['Mouse_Y'])
            
            # Calculate error
            error = np.sqrt(
                (row['Target_X'] - row['Mouse_X'])**2 +
                (row['Target_Y'] - row['Mouse_Y'])**2
            )
            
            time_sec = row['Frame'] / 50  # Assuming 50 fps
            time_text.set_text(f'Time: {time_sec:.2f}s\nFrame: {row["Frame"]}')
            error_text.set_text(f'Error: {error:.1f} px')
            
            elements = [target_marker, mouse_marker, time_text, error_text]
            if self.config.show_error_line:
                elements.append(error_line)
            if self.config.show_trail:
                elements.extend([target_trail, mouse_trail])
            
            return elements
        
        # Create animation
        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=n_frames,
            interval=1000 / self.config.fps,
            blit=True
        )
        
        # Save
        if output_path is None:
            output_path = self.output_dir / f'animation.{format}' if self.output_dir else f'animation.{format}'
        
        output_path = Path(output_path)
        
        if format == 'mp4':
            try:
                writer = FFMpegWriter(fps=self.config.fps, bitrate=1800)
                anim.save(str(output_path), writer=writer, dpi=self.config.dpi)
            except Exception as e:
                warnings.warn(f"Could not save MP4 (ffmpeg required): {e}")
                # Fall back to HTML
                format = 'html'
                output_path = output_path.with_suffix('.html')
        
        if format == 'html':
            anim.save(str(output_path), writer='html', fps=self.config.fps)
        
        plt.close(fig)
        
        return output_path
    
    def _create_plotly_animation(
        self,
        trial_data: pd.DataFrame,
        output_path: Optional[Union[str, Path]],
        title: str
    ) -> Optional[Path]:
        """Create plotly-based interactive animation."""
        if not HAS_PLOTLY:
            raise ImportError("plotly required for interactive animations")
        
        # Sample frames for smoother animation
        n_frames = len(trial_data)
        step = max(1, n_frames // 200)  # Limit to ~200 frames for performance
        sampled_indices = list(range(0, n_frames, step))
        
        frames = []
        for idx in sampled_indices:
            row = trial_data.iloc[idx]
            
            # Trail data
            trail_start = max(0, idx - self.config.trail_length * step)
            trail_data_df = trial_data.iloc[trail_start:idx+1:max(1, step//2)]
            
            frame = go.Frame(
                data=[
                    # Target marker
                    go.Scatter(
                        x=[row['Target_X']],
                        y=[row['Target_Y']],
                        mode='markers',
                        marker=dict(size=20, color=self.config.target_color),
                        name='Target'
                    ),
                    # Mouse marker
                    go.Scatter(
                        x=[row['Mouse_X']],
                        y=[row['Mouse_Y']],
                        mode='markers',
                        marker=dict(size=15, color=self.config.mouse_color),
                        name='Mouse'
                    ),
                    # Target trail
                    go.Scatter(
                        x=trail_data_df['Target_X'],
                        y=trail_data_df['Target_Y'],
                        mode='lines',
                        line=dict(color=self.config.target_color, width=2),
                        opacity=0.3,
                        showlegend=False
                    ),
                    # Mouse trail
                    go.Scatter(
                        x=trail_data_df['Mouse_X'],
                        y=trail_data_df['Mouse_Y'],
                        mode='lines',
                        line=dict(color=self.config.mouse_color, width=2),
                        opacity=0.3,
                        showlegend=False
                    ),
                    # Error line
                    go.Scatter(
                        x=[row['Target_X'], row['Mouse_X']],
                        y=[row['Target_Y'], row['Mouse_Y']],
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dash'),
                        showlegend=False
                    )
                ],
                name=str(idx)
            )
            frames.append(frame)
        
        # Initial frame
        initial_row = trial_data.iloc[0]
        
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=[initial_row['Target_X']],
                    y=[initial_row['Target_Y']],
                    mode='markers',
                    marker=dict(size=20, color=self.config.target_color),
                    name='Target'
                ),
                go.Scatter(
                    x=[initial_row['Mouse_X']],
                    y=[initial_row['Mouse_Y']],
                    mode='markers',
                    marker=dict(size=15, color=self.config.mouse_color),
                    name='Mouse'
                ),
                go.Scatter(x=[], y=[], mode='lines', showlegend=False),
                go.Scatter(x=[], y=[], mode='lines', showlegend=False),
                go.Scatter(x=[], y=[], mode='lines', showlegend=False)
            ],
            frames=frames,
            layout=go.Layout(
                title=title,
                xaxis=dict(range=[0, 1920], title='X Position (pixels)'),
                yaxis=dict(range=[0, 980], title='Y Position (pixels)', scaleanchor='x'),
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=False,
                        y=1.1,
                        x=0.5,
                        xanchor='center',
                        buttons=[
                            dict(
                                label='Play',
                                method='animate',
                                args=[
                                    None,
                                    dict(
                                        frame=dict(duration=50, redraw=True),
                                        fromcurrent=True,
                                        mode='immediate'
                                    )
                                ]
                            ),
                            dict(
                                label='Pause',
                                method='animate',
                                args=[
                                    [None],
                                    dict(
                                        frame=dict(duration=0, redraw=False),
                                        mode='immediate'
                                    )
                                ]
                            )
                        ]
                    )
                ],
                sliders=[
                    dict(
                        active=0,
                        steps=[
                            dict(
                                method='animate',
                                args=[
                                    [str(idx)],
                                    dict(
                                        mode='immediate',
                                        frame=dict(duration=50, redraw=True)
                                    )
                                ],
                                label=str(idx)
                            )
                            for idx in sampled_indices
                        ],
                        x=0,
                        y=0,
                        len=1.0,
                        xanchor='left',
                        yanchor='top',
                        pad=dict(t=50, b=10),
                        currentvalue=dict(
                            prefix='Frame: ',
                            visible=True,
                            xanchor='center'
                        ),
                        transition=dict(duration=0)
                    )
                ]
            )
        )
        
        if output_path is None:
            output_path = self.output_dir / 'animation.html' if self.output_dir else 'animation.html'
        
        output_path = Path(output_path)
        fig.write_html(str(output_path))
        
        return output_path
    
    def create_comparison_animation(
        self,
        trials: List[pd.DataFrame],
        labels: List[str],
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Trial Comparison"
    ) -> Optional[Path]:
        """
        Create animation comparing multiple trials.
        
        Args:
            trials: List of trial DataFrames
            labels: Labels for each trial
            output_path: Output file path
            title: Animation title
            
        Returns:
            Path to saved animation
        """
        if not HAS_PLOTLY:
            warnings.warn("Plotly required for comparison animations")
            return None
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
        
        # Find common frame range
        min_frames = min(len(t) for t in trials)
        
        frames_list = []
        step = max(1, min_frames // 100)
        
        for frame_idx in range(0, min_frames, step):
            data = []
            for i, (trial, label) in enumerate(zip(trials, labels)):
                row = trial.iloc[frame_idx]
                color = colors[i % len(colors)]
                
                data.append(go.Scatter(
                    x=[row['Target_X']],
                    y=[row['Target_Y']],
                    mode='markers',
                    marker=dict(size=15, color=color, symbol='circle'),
                    name=f'{label} Target',
                    showlegend=(frame_idx == 0)
                ))
                data.append(go.Scatter(
                    x=[row['Mouse_X']],
                    y=[row['Mouse_Y']],
                    mode='markers',
                    marker=dict(size=10, color=color, symbol='x'),
                    name=f'{label} Mouse',
                    showlegend=(frame_idx == 0)
                ))
            
            frames_list.append(go.Frame(data=data, name=str(frame_idx)))
        
        # Initial data
        initial_data = []
        for i, (trial, label) in enumerate(zip(trials, labels)):
            row = trial.iloc[0]
            color = colors[i % len(colors)]
            
            initial_data.extend([
                go.Scatter(
                    x=[row['Target_X']], y=[row['Target_Y']],
                    mode='markers',
                    marker=dict(size=15, color=color, symbol='circle'),
                    name=f'{label} Target'
                ),
                go.Scatter(
                    x=[row['Mouse_X']], y=[row['Mouse_Y']],
                    mode='markers',
                    marker=dict(size=10, color=color, symbol='x'),
                    name=f'{label} Mouse'
                )
            ])
        
        fig = go.Figure(
            data=initial_data,
            frames=frames_list,
            layout=go.Layout(
                title=title,
                xaxis=dict(range=[0, 1920], title='X'),
                yaxis=dict(range=[0, 980], title='Y', scaleanchor='x'),
                updatemenus=[dict(
                    type='buttons',
                    buttons=[
                        dict(label='Play', method='animate',
                             args=[None, dict(frame=dict(duration=50))]),
                        dict(label='Pause', method='animate',
                             args=[[None], dict(frame=dict(duration=0))])
                    ]
                )]
            )
        )
        
        if output_path is None:
            output_path = self.output_dir / 'comparison.html' if self.output_dir else 'comparison.html'
        
        output_path = Path(output_path)
        fig.write_html(str(output_path))
        
        return output_path
    
    def estimate_duration(
        self,
        n_frames: int,
        format: str = 'mp4'
    ) -> Dict[str, Any]:
        """
        Estimate animation generation time.
        
        Args:
            n_frames: Number of frames to animate
            format: Output format
            
        Returns:
            Dictionary with estimates
        """
        # Rough estimates based on format
        if format == 'mp4':
            seconds_per_frame = 0.1  # matplotlib + ffmpeg
        elif format == 'html':
            seconds_per_frame = 0.05  # HTML is faster
        elif format == 'plotly':
            seconds_per_frame = 0.02  # Plotly is very fast
        else:
            seconds_per_frame = 0.05
        
        total_seconds = n_frames * seconds_per_frame
        
        return {
            'n_frames': n_frames,
            'format': format,
            'estimated_seconds': total_seconds,
            'estimated_formatted': f"{total_seconds:.0f} seconds" if total_seconds < 60 else f"{total_seconds/60:.1f} minutes",
            'output_duration_seconds': n_frames / self.config.fps
        }
