"""
Command Line Interface
======================

Provides CLI commands for running tracking analysis.

Usage:
    python -m tracking_analysis.cli analyze --data-path ./data --output ./results
    python -m tracking_analysis.cli info ./data
    python -m tracking_analysis.cli states ./results/states
    python -m tracking_analysis.cli ui
    
Commands:
    analyze: Run full analysis pipeline
    info: Show dataset information
    states: List/manage saved states
    ui: Launch Streamlit web interface
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import json

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog='tracking_analysis',
        description='Psychophysics Tracking Data Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show dataset info
  python -m tracking_analysis.cli info ./data
  
  # Run full analysis
  python -m tracking_analysis.cli analyze --data-path ./data --output ./results
  
  # Run with custom parameters
  python -m tracking_analysis.cli analyze --data-path ./data --output ./results \\
      --velocity-method savgol --outlier-method iqr
  
  # List saved states
  python -m tracking_analysis.cli states ./results/states
  
  # Launch web UI
  python -m tracking_analysis.cli ui
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # =========================================================================
    # INFO COMMAND
    # =========================================================================
    info_parser = subparsers.add_parser(
        'info',
        help='Show dataset information',
        description='Display information about tracking data files'
    )
    info_parser.add_argument(
        'data_path',
        type=str,
        help='Path to data directory'
    )
    info_parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    # =========================================================================
    # ANALYZE COMMAND
    # =========================================================================
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Run analysis pipeline',
        description='Run the full tracking analysis pipeline'
    )
    
    # Required arguments
    analyze_parser.add_argument(
        '--data-path', '-d',
        type=str,
        required=True,
        help='Path to data directory'
    )
    analyze_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    # Data filtering options
    analyze_parser.add_argument(
        '--participants',
        type=str,
        nargs='+',
        help='Filter by participant IDs'
    )
    analyze_parser.add_argument(
        '--conditions',
        type=str,
        nargs='+',
        choices=['dynamic', 'static'],
        help='Filter by conditions'
    )
    analyze_parser.add_argument(
        '--sd-sizes',
        type=int,
        nargs='+',
        choices=[21, 31, 34],
        help='Filter by SD sizes'
    )
    
    # Preprocessing options
    analyze_parser.add_argument(
        '--velocity-method',
        type=str,
        choices=['difference', 'savgol'],
        default='savgol',
        help='Velocity calculation method (default: savgol)'
    )
    analyze_parser.add_argument(
        '--outlier-method',
        type=str,
        choices=['none', 'iqr', 'zscore', 'mad'],
        default='iqr',
        help='Outlier removal method (default: iqr)'
    )
    analyze_parser.add_argument(
        '--outlier-threshold',
        type=float,
        default=1.5,
        help='Outlier threshold (default: 1.5)'
    )
    analyze_parser.add_argument(
        '--missing-method',
        type=str,
        choices=['drop', 'interpolate', 'ffill', 'mean'],
        default='interpolate',
        help='Missing data handling (default: interpolate)'
    )
    
    # Analysis options
    analyze_parser.add_argument(
        '--normalize-xcorr',
        action='store_true',
        default=True,
        help='Normalize cross-correlation (default: True)'
    )
    analyze_parser.add_argument(
        '--no-normalize-xcorr',
        dest='normalize_xcorr',
        action='store_false',
        help='Do not normalize cross-correlation'
    )
    analyze_parser.add_argument(
        '--max-lag',
        type=int,
        default=50,
        help='Maximum lag for cross-correlation (frames, default: 50)'
    )
    analyze_parser.add_argument(
        '--frame-rate',
        type=float,
        default=50.0,
        help='Frame rate in Hz (default: 50)'
    )
    
    # Statistics options
    analyze_parser.add_argument(
        '--stat-test',
        type=str,
        choices=['anova', 'kruskal', 'both'],
        default='both',
        help='Statistical test type (default: both)'
    )
    analyze_parser.add_argument(
        '--correction',
        type=str,
        choices=['bonferroni', 'holm', 'fdr'],
        default='holm',
        help='Multiple comparison correction (default: holm)'
    )
    
    # Output options
    analyze_parser.add_argument(
        '--save-state',
        action='store_true',
        help='Save analysis state for later loading'
    )
    analyze_parser.add_argument(
        '--generate-report',
        action='store_true',
        default=True,
        help='Generate analysis report (default: True)'
    )
    analyze_parser.add_argument(
        '--create-plots',
        action='store_true',
        default=True,
        help='Create visualization plots (default: True)'
    )
    analyze_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    # =========================================================================
    # STATES COMMAND
    # =========================================================================
    states_parser = subparsers.add_parser(
        'states',
        help='Manage saved states',
        description='List and manage saved analysis states'
    )
    states_parser.add_argument(
        'states_path',
        type=str,
        help='Path to states directory'
    )
    states_parser.add_argument(
        '--load',
        type=str,
        metavar='STATE_ID',
        help='Load and show state info'
    )
    states_parser.add_argument(
        '--delete',
        type=str,
        metavar='STATE_ID',
        help='Delete a saved state'
    )
    states_parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    # =========================================================================
    # UI COMMAND
    # =========================================================================
    ui_parser = subparsers.add_parser(
        'ui',
        help='Launch Streamlit web interface',
        description='Start the Streamlit web application'
    )
    ui_parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port to run on (default: 8501)'
    )
    ui_parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host address (default: localhost)'
    )
    
    return parser


# =============================================================================
# COMMAND HANDLERS
# =============================================================================

def cmd_info(args) -> int:
    """Handle info command."""
    from tracking_analysis.core.data_loader import DataLoader
    
    try:
        loader = DataLoader(args.data_path)
        loader.load_all()
        metadata = loader.get_metadata()
        
        if args.json:
            info = {
                'n_files': metadata.n_files,
                'n_participants': metadata.n_participants,
                'total_frames': metadata.total_frames,
                'participants': list(metadata.participants),
                'sd_sizes': list(metadata.sd_sizes),
                'conditions': list(metadata.conditions)
            }
            print(json.dumps(info, indent=2))
        else:
            print(f"\n{'='*60}")
            print("TRACKING DATA INFORMATION")
            print(f"{'='*60}")
            print(f"Data Path:        {args.data_path}")
            print(f"Number of Files:  {metadata.n_files}")
            print(f"Participants:     {metadata.n_participants}")
            print(f"Total Frames:     {metadata.total_frames:,}")
            print(f"\nParticipant IDs:  {', '.join(sorted(metadata.participants))}")
            print(f"SD Sizes:         {sorted(metadata.sd_sizes)}")
            print(f"Conditions:       {sorted(metadata.conditions)}")
            
            # Estimate runtime
            estimate = loader.estimate_runtime()
            print(f"\nEstimated Load Time: {estimate.get('estimated_load_formatted', 'N/A')}")
            print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_analyze(args) -> int:
    """Handle analyze command."""
    from tracking_analysis.core.config import Config
    from tracking_analysis.core.data_loader import DataLoader
    from tracking_analysis.core.preprocessing import Preprocessor
    from tracking_analysis.core.metrics import MetricsCalculator
    from tracking_analysis.core.cross_correlation import CrossCorrelationAnalyzer
    from tracking_analysis.core.statistics import StatisticalAnalyzer
    from tracking_analysis.core.state_manager import StateManager
    
    try:
        # Create config from arguments
        config = Config(
            data_path=args.data_path,
            output_path=args.output,
            velocity_method=args.velocity_method,
            outlier_method=args.outlier_method,
            outlier_threshold=args.outlier_threshold,
            missing_method=args.missing_method,
            normalize_xcorr=args.normalize_xcorr,
            max_lag_frames=args.max_lag,
            frame_rate=args.frame_rate
        )
        
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if args.verbose:
            print(f"\n{'='*60}")
            print("STARTING ANALYSIS")
            print(f"{'='*60}")
            print(f"Data Path: {args.data_path}")
            print(f"Output: {args.output}")
        
        # 1. Load data
        if args.verbose:
            print("\n[1/5] Loading data...")
        
        loader = DataLoader(args.data_path)
        data = loader.load_filtered(
            participants=args.participants,
            conditions=args.conditions,
            sd_sizes=args.sd_sizes
        )
        
        if args.verbose:
            print(f"      Loaded {len(data):,} frames from {data['participant_id'].nunique()} participants")
        
        # 2. Preprocess
        if args.verbose:
            print("\n[2/5] Preprocessing...")
        
        preprocessor = Preprocessor(config)
        processed_data, preprocess_report = preprocessor.preprocess(data, return_report=True)
        
        if args.verbose:
            print(f"      Original rows: {preprocess_report['original_rows']:,}")
            print(f"      Final rows: {preprocess_report['final_rows']:,}")
        
        # 3. Calculate metrics
        if args.verbose:
            print("\n[3/5] Calculating metrics...")
        
        metrics_calc = MetricsCalculator(config)
        trial_metrics = metrics_calc.compute_trial_metrics(processed_data)
        
        if args.verbose:
            print(f"      Computed metrics for {len(trial_metrics)} trials")
        
        # 4. Cross-correlation
        if args.verbose:
            print("\n[4/5] Computing cross-correlations...")
        
        xcorr_analyzer = CrossCorrelationAnalyzer(config)
        xcorr_results = xcorr_analyzer.compute_batch_xcorr(processed_data)
        
        if args.verbose:
            print(f"      Analyzed {len(xcorr_results)} trials")
        
        # 5. Statistical analysis
        if args.verbose:
            print("\n[5/5] Running statistical analysis...")
        
        stats_analyzer = StatisticalAnalyzer(config)
        stat_results = stats_analyzer.generate_results_summary(trial_metrics, metric='rmse')
        
        if args.verbose:
            print("      Completed statistical analysis")
        
        # Save outputs
        trial_metrics.to_csv(output_path / 'trial_metrics.csv', index=False)
        xcorr_results.to_csv(output_path / 'xcorr_results.csv', index=False)
        
        with open(output_path / 'statistical_results.json', 'w') as f:
            json.dump(stat_results, f, indent=2, default=str)
        
        # Save state if requested
        if args.save_state:
            state_manager = StateManager(output_path / 'states')
            state_id = state_manager.save_state(
                config=config,
                trial_metrics=trial_metrics,
                xcorr_results=xcorr_results,
                statistical_results=stat_results,
                processed_data=processed_data,
                preprocessing_report=preprocess_report
            )
            if args.verbose:
                print(f"\n      State saved: {state_id}")
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_states(args) -> int:
    """Handle states command."""
    from tracking_analysis.core.state_manager import StateManager
    
    try:
        manager = StateManager(args.states_path)
        
        if args.delete:
            # Delete state
            if manager.delete_state(args.delete):
                print(f"Deleted state: {args.delete}")
            else:
                print(f"State not found: {args.delete}")
            return 0
        
        if args.load:
            # Load and show state
            state = manager.load_state(args.load)
            info = manager.get_state_info(args.load)
            
            if args.json:
                print(json.dumps({
                    'state_id': state.state_id,
                    'created_at': state.created_at,
                    'has_metrics': state.trial_metrics is not None,
                    'has_xcorr': state.xcorr_results is not None,
                    'has_statistics': state.statistical_results is not None,
                    'config': state.config
                }, indent=2))
            else:
                print(f"\nState: {state.state_id}")
                print(f"Created: {state.created_at}")
                print(f"Size: {info.file_size_mb:.2f} MB")
                print(f"\nContains:")
                print(f"  - Trial Metrics: {'Yes' if state.trial_metrics is not None else 'No'}")
                print(f"  - Cross-Correlation: {'Yes' if state.xcorr_results is not None else 'No'}")
                print(f"  - Statistics: {'Yes' if state.statistical_results is not None else 'No'}")
            return 0
        
        # List all states
        states = manager.list_states()
        
        if not states:
            print("No saved states found.")
            return 0
        
        if args.json:
            states_info = [{
                'state_id': s.state_id,
                'created_at': s.created_at,
                'has_metrics': s.has_metrics,
                'has_xcorr': s.has_xcorr,
                'has_statistics': s.has_statistics,
                'file_size_mb': s.file_size_mb
            } for s in states]
            print(json.dumps(states_info, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"SAVED STATES ({len(states)} total)")
            print(f"{'='*60}")
            for s in states:
                print(f"\n  {s.state_id}")
                print(f"    Created: {s.created_at}")
                print(f"    Size: {s.file_size_mb:.2f} MB")
                print(f"    Contains: ", end='')
                contents = []
                if s.has_metrics:
                    contents.append('metrics')
                if s.has_xcorr:
                    contents.append('xcorr')
                if s.has_statistics:
                    contents.append('stats')
                print(', '.join(contents) if contents else 'config only')
            print(f"\n{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_ui(args) -> int:
    """Handle ui command."""
    import subprocess
    
    try:
        # Find the app.py file
        ui_path = Path(__file__).parent.parent / 'ui' / 'app.py'
        
        if not ui_path.exists():
            print("Error: Streamlit UI not found. Please install the full package.")
            return 1
        
        print(f"Starting Streamlit UI on {args.host}:{args.port}...")
        print("Press Ctrl+C to stop.\n")
        
        # Run streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            str(ui_path),
            '--server.port', str(args.port),
            '--server.address', args.host
        ])
        
        return 0
        
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Dispatch to command handler
    handlers = {
        'info': cmd_info,
        'analyze': cmd_analyze,
        'states': cmd_states,
        'ui': cmd_ui
    }
    
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
