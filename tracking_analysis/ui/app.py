"""
Streamlit UI Application
========================

Main entry point for the Streamlit web interface.

Run with:
    streamlit run tracking_analysis/ui/app.py
    
Or via CLI:
    python -m tracking_analysis.cli ui
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tracking_analysis.core.config import Config, PARAMETER_INFO, DEFAULTS, FRAME_RATE
from tracking_analysis.core.data_loader import DataLoader
from tracking_analysis.core.preprocessing import Preprocessor
from tracking_analysis.core.metrics import MetricsCalculator
from tracking_analysis.core.cross_correlation import CrossCorrelationAnalyzer
from tracking_analysis.core.statistics import StatisticalAnalyzer
from tracking_analysis.core.state_manager import StateManager, list_saved_states

# Page config
st.set_page_config(
    page_title="Tracking Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'data_loaded': False,
        'data': None,
        'metadata': None,
        'processed_data': None,
        'trial_metrics': None,
        'xcorr_results': None,
        'stat_results': None,
        'config': None,
        'current_step': 0,
        'analysis_complete': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# SIDEBAR COMPONENTS
# =============================================================================

def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.markdown("## 🎯 Tracking Analysis")
        
        st.markdown("---")
        
        # Progress indicator (breadcrumb-style)
        st.subheader("📈 Progress")
        
        # Define steps and their completion status
        steps = [
            ("Data", st.session_state.data_loaded),
            ("Config", st.session_state.config is not None),
            ("Analysis", st.session_state.analysis_complete),
        ]
        
        # Display progress as colored indicators
        cols = st.columns(len(steps))
        for i, (step_name, is_complete) in enumerate(steps):
            with cols[i]:
                if is_complete:
                    st.markdown(f"<div style='text-align:center'><span style='color:green;font-size:1.5em'>✓</span><br><small>{step_name}</small></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align:center'><span style='color:gray;font-size:1.5em'>○</span><br><small>{step_name}</small></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.subheader("📍 Navigation")
        
        all_pages = {
            "🏠 Home": "home",
            "💾 Load/Save State": "states",
            "📂 Load New Data": "load_data",
            "⚙️ Configure": "configure",
            "▶️ Run Analysis": "run_analysis",
            "📊 Results Overview": "results",
            "📉 Detailed Analysis": "data_analysis",
            "📈 Visualizations": "visualizations",
            "🔬 Research Questions": "research_questions",
            "❓ Documentation": "help",
        }
        
        selected = st.radio(
            "Go to",
            list(all_pages.keys()),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick status with more detail
        st.subheader("📋 Status")
        
        # Data status
        if st.session_state.data_loaded:
            st.success("✓ Data loaded")
            if st.session_state.metadata:
                st.caption(f"📁 {st.session_state.metadata.n_files} files")
                st.caption(f"👥 {st.session_state.metadata.n_participants} participants")
        else:
            st.warning("⚠ No data loaded")
            st.caption("Load data or restore a saved state")
        
        # Analysis status
        if st.session_state.analysis_complete:
            st.success("✓ Analysis complete")
            if st.session_state.trial_metrics is not None:
                n_trials = len(st.session_state.trial_metrics)
                st.caption(f"📊 {n_trials} trials analyzed")
        elif st.session_state.data_loaded:
            st.info("⏳ Ready to analyze")
        
        # Quick actions
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄", help="Refresh data"):
                st.rerun()
        with col2:
            if st.button("🗑️", help="Clear session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        return all_pages[selected]


# =============================================================================
# PAGE: HOME
# =============================================================================

def page_home():
    """Render home page."""
    st.markdown('<p class="main-header">🎯 Tracking Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Psychophysics Tracking Data Analysis Tool</p>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. **Load Data**: Select your data folder containing tracking CSV files
        2. **Configure**: Set analysis parameters (velocity method, outlier handling, etc.)
        3. **Run Analysis**: Process data and compute metrics
        4. **View Results**: Explore RMSE, cross-correlations, and statistics
        5. **Visualize**: Create plots and animations
        """)
        
        st.markdown("### 📖 Research Questions")
        st.markdown("""
        This tool helps answer questions like:
        - Can observers discriminate blobs with different SD sizes (21, 31, 34 arcmin)?
        - Does auditory feedback improve tracking performance?
        - Is tracking predictive or reactive?
        """)
    
    with col2:
        st.markdown("### ⚡ Features")
        st.markdown("""
        - **Multiple velocity methods**: Simple difference or Savitzky-Golay smoothing
        - **Outlier handling**: IQR, Z-score, or MAD methods
        - **Cross-correlation analysis**: Detect predictive vs reactive tracking
        - **Statistical tests**: ANOVA, t-tests with effect sizes
        - **Visualizations**: Static plots and animated trial replays
        - **State management**: Save and load analysis states
        """)
        
        st.markdown("### 📊 Data Format")
        st.markdown("""
        Expected CSV columns:
        - `Frame`: Frame number (0-indexed)
        - `Target_X`, `Target_Y`: Target position
        - `Mouse_X`, `Mouse_Y`: Mouse/response position
        """)


# =============================================================================
# PAGE: LOAD DATA
# =============================================================================

def page_load_data():
    """Render data loading page."""
    st.header("📂 Load Data")
    
    st.markdown("""
    Load your tracking data CSV files using one of the methods below.
    Files should follow the naming pattern: `Participant_XXXX_..._XXarcmin_..._dynamic/static.csv`
    """)
    
    # Create tabs for different loading methods
    tab1, tab2 = st.tabs(["📁 Upload Files (Drag & Drop)", "📂 Enter Folder Path"])
    
    # -------------------------------------------------------------------------
    # TAB 1: File Upload (Drag & Drop)
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("""
        **Drag and drop** your CSV files below, or click to browse.
        You can select multiple files at once.
        """)
        
        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=['csv'],
            accept_multiple_files=True,
            help="Select all your tracking CSV files. Hold Ctrl/Cmd to select multiple files.",
            key="csv_uploader"
        )
        
        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)} file(s) selected")
            
            # Show file list in expander
            with st.expander("View selected files"):
                for f in uploaded_files:
                    st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")
            
            if st.button("🔄 Load Uploaded Files", type="primary", key="load_uploaded"):
                with st.spinner("Loading uploaded files..."):
                    try:
                        # Load data directly from uploaded files
                        all_data = []
                        
                        for uploaded_file in uploaded_files:
                            # Read CSV
                            df = pd.read_csv(uploaded_file)
                            
                            # Parse filename for metadata
                            filename = uploaded_file.name
                            df['filename'] = filename
                            
                            # Extract participant ID
                            if 'Participant_' in filename:
                                parts = filename.split('_')
                                participant_idx = parts.index('Participant') + 1 if 'Participant' in parts else None
                                if participant_idx and participant_idx < len(parts):
                                    df['participant_id'] = parts[participant_idx]
                                else:
                                    df['participant_id'] = filename.split('_')[1] if len(filename.split('_')) > 1 else 'unknown'
                            else:
                                df['participant_id'] = 'unknown'
                            
                            # Extract size (arcmin)
                            size_match = None
                            for part in filename.split('_'):
                                if 'arcmin' in part:
                                    size_match = part.replace('arcmin', '')
                                    break
                            df['size'] = int(size_match) if size_match and size_match.isdigit() else 0
                            df['size_pixels'] = df['size']  # Same as size for display
                            
                            # Extract condition
                            if 'dynamic' in filename.lower():
                                df['condition'] = 'dynamic'
                                df['condition_label'] = 'Auditory Feedback'
                            elif 'static' in filename.lower():
                                df['condition'] = 'static'
                                df['condition_label'] = 'No Feedback'
                            else:
                                df['condition'] = 'unknown'
                                df['condition_label'] = 'Unknown'
                            
                            all_data.append(df)
                        
                        # Combine all data
                        combined_data = pd.concat(all_data, ignore_index=True)
                        
                        # Create metadata
                        class UploadedMetadata:
                            def __init__(self, data, n_files):
                                self.n_files = n_files
                                self.n_participants = data['participant_id'].nunique()
                                self.participants = list(data['participant_id'].unique())
                                self.sizes = list(data['size'].unique())
                                self.conditions = list(data['condition'].unique())
                        
                        metadata = UploadedMetadata(combined_data, len(uploaded_files))
                        
                        st.session_state.data = combined_data
                        st.session_state.metadata = metadata
                        st.session_state.data_loaded = True
                        st.session_state.data_path = "Uploaded files"
                        
                        st.success(f"✓ Successfully loaded {len(uploaded_files)} files!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error loading files: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # -------------------------------------------------------------------------
    # TAB 2: Folder Path Input
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("""
        Enter the full path to the folder containing your CSV files.
        """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            data_path = st.text_input(
                "Data Folder Path",
                value="",
                placeholder="C:\\Users\\YourName\\Desktop\\data",
                help="Enter the full path to your data folder (e.g., C:\\Users\\YourName\\data)",
                key="folder_path_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            load_clicked = st.button("🔄 Load from Folder", type="primary", key="load_folder")
        
        # Helpful tips
        with st.expander("💡 Tips for finding your data folder"):
            st.markdown("""
            **Windows:**
            - Open File Explorer and navigate to your data folder
            - Click on the address bar to see the full path
            - Copy and paste the path here
            - Example: `C:\\Users\\YourName\\Desktop\\experiment_data`
            
            **Mac/Linux:**
            - Open Finder/File Manager and navigate to your data folder
            - Right-click and copy path, or use Terminal: `pwd`
            - Example: `/Users/YourName/Desktop/experiment_data`
            
            **Common locations:**
            - Desktop: `C:\\Users\\YourName\\Desktop\\data`
            - Documents: `C:\\Users\\YourName\\Documents\\data`
            - Downloads: `C:\\Users\\YourName\\Downloads\\data`
            """)
        
        # Load data from folder
        if load_clicked:
            if not data_path:
                st.error("Please enter a folder path")
                return
                
            path = Path(data_path)
            if not path.exists():
                st.error(f"❌ Path does not exist: {data_path}")
                st.info("💡 Please check the path and try again. Make sure to use the full path.")
                return
            
            if not path.is_dir():
                st.error(f"❌ Path is not a folder: {data_path}")
                return
            
            # Check for CSV files
            csv_files = list(path.glob('*.csv'))
            if not csv_files:
                st.error(f"❌ No CSV files found in: {data_path}")
                st.info("💡 Make sure your CSV files are directly in this folder (not in subfolders)")
                return
            
            with st.spinner(f"Loading {len(csv_files)} files..."):
                try:
                    loader = DataLoader(data_path)
                    data = loader.load_all()
                    metadata = loader.get_metadata()
                    
                    st.session_state.data = data
                    st.session_state.metadata = metadata
                    st.session_state.data_loaded = True
                    st.session_state.data_path = data_path
                    
                    st.success(f"✓ Successfully loaded {len(csv_files)} files!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
    
    # -------------------------------------------------------------------------
    # Display loaded data info (shown regardless of which method was used)
    # -------------------------------------------------------------------------
    if st.session_state.data_loaded:
        st.markdown("---")
        st.subheader("📊 Dataset Overview")
        
        meta = st.session_state.metadata
        data = st.session_state.data
        
        # Show data source
        st.info(f"📍 Data source: **{st.session_state.data_path}**")
        
        # Calculate total frames from loaded data
        total_frames = len(data) if data is not None else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Files", meta.n_files)
        with col2:
            st.metric("Participants", meta.n_participants)
        with col3:
            st.metric("Total Frames", f"{total_frames:,}")
        with col4:
            est_time = total_frames / 50 / 60  # minutes at 50fps
            st.metric("Est. Duration", f"{est_time:.1f} min")
        
        # Show conditions and sizes
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Conditions found:**")
            conditions_in_data = data['condition'].unique() if data is not None else []
            for cond in sorted(conditions_in_data):
                label = {'dynamic': '🔊 Auditory Feedback', 'static': '🔇 No Feedback'}.get(cond, cond)
                count = len(data[data['condition'] == cond]['filename'].unique())
                st.write(f"- {label}: {count} files")
        
        with col2:
            st.markdown("**SD Sizes (pixels):**")
            sizes_in_data = data['size_pixels'].unique() if data is not None else []
            for size in sorted(sizes_in_data):
                count = len(data[data['size_pixels'] == size]['filename'].unique())
                st.write(f"- {size} arcmin: {count} files")
        
        # Participant list
        st.markdown("**Participants:**")
        st.write(", ".join(sorted(meta.participants)))
        
        # Preview data
        st.markdown("---")
        st.subheader("🔍 Data Preview")
        
        # Show one row per file to display the variety of conditions
        st.markdown("**One sample row per trial file** (showing metadata columns):")
        
        preview_dfs = []
        for filename in data['filename'].unique():
            # Get first row of each file
            file_data = data[data['filename'] == filename].iloc[[0]]
            preview_dfs.append(file_data)
        
        preview_data = pd.concat(preview_dfs, ignore_index=True) if preview_dfs else data.head(100)
        
        # Select key columns for display
        display_cols = ['participant_id', 'size_pixels', 'condition', 'condition_label', 'filename']
        available_cols = [c for c in display_cols if c in preview_data.columns]
        
        st.dataframe(
            preview_data[available_cols].sort_values(['participant_id', 'size_pixels', 'condition']),
            use_container_width=True,
            height=400
        )
        
        st.caption(f"Showing {len(preview_data)} trials (one row per file)")
        
        # Option to clear data and load new
        st.markdown("---")
        if st.button("🗑️ Clear Data and Load New", type="secondary"):
            st.session_state.data = None
            st.session_state.metadata = None
            st.session_state.data_loaded = False
            st.session_state.data_path = None
            st.rerun()


# =============================================================================
# PAGE: CONFIGURE
# =============================================================================

def page_configure():
    """Render configuration page."""
    st.header("⚙️ Configure Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first!")
        return
    
    st.markdown("Configure the analysis parameters. Each option includes recommendations and trade-offs.")
    
    # Initialize config in session state
    if st.session_state.config is None:
        st.session_state.config = Config(data_path=st.session_state.data_path)
    
    config = st.session_state.config
    
    # Tabs for different config sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔢 Preprocessing", 
        "📈 Cross-Correlation", 
        "📊 Statistics",
        "💾 Output"
    ])
    
    with tab1:
        st.subheader("Velocity Calculation")
        
        info = PARAMETER_INFO.get('velocity_method', {})
        
        velocity_method = st.selectbox(
            "Method",
            ['difference', 'savgol'],
            index=0 if config.velocity_method == 'difference' else 1,
            help=f"**Pros:** {info.get('pros', 'N/A')}\n\n**Cons:** {info.get('cons', 'N/A')}"
        )
        
        with st.expander("ℹ️ Method Details"):
            st.markdown(f"""
            **Pros:** {info.get('pros', 'N/A')}
            
            **Cons:** {info.get('cons', 'N/A')}
            
            **Recommendation:** {info.get('recommendation', 'N/A')}
            """)
        
        if velocity_method == 'savgol':
            col1, col2 = st.columns(2)
            with col1:
                smooth_window = st.slider(
                    "Window Size",
                    min_value=3, max_value=21, value=config.smooth_window, step=2,
                    help="Must be odd number. Larger = more smoothing"
                )
            with col2:
                smooth_poly = st.slider(
                    "Polynomial Order",
                    min_value=1, max_value=5, value=2,
                    help="Must be less than window size"
                )
        else:
            smooth_window = config.smooth_window
            smooth_poly = 2
        
        st.markdown("---")
        st.subheader("Outlier Removal")
        
        info = PARAMETER_INFO.get('outlier_method', {})
        
        outlier_method = st.selectbox(
            "Method",
            ['none', 'iqr', 'zscore', 'mad'],
            index=['none', 'iqr', 'zscore', 'mad'].index(config.outlier_method),
            help=f"**Recommendation:** {info.get('recommendation', 'IQR for non-normal data')}"
        )
        
        if outlier_method != 'none':
            default_thresh = {'iqr': 1.5, 'zscore': 3.0, 'mad': 3.0}
            outlier_threshold = st.slider(
                "Threshold",
                min_value=1.0, max_value=5.0, 
                value=config.outlier_threshold or default_thresh.get(outlier_method, 1.5),
                step=0.1,
                help="Lower = more aggressive outlier removal"
            )
        else:
            outlier_threshold = config.outlier_threshold
        
        st.markdown("---")
        st.subheader("Missing Data")
        
        missing_method = st.selectbox(
            "Handling Method",
            ['drop', 'interpolate', 'ffill', 'mean'],
            index=['drop', 'interpolate', 'ffill', 'mean'].index(config.missing_data_method),
            help="Interpolation recommended for small gaps"
        )
    
    with tab2:
        st.subheader("Cross-Correlation Settings")
        
        info = PARAMETER_INFO.get('normalize_xcorr', {})
        
        normalize_xcorr = st.checkbox(
            "Normalize Cross-Correlation",
            value=config.normalize_xcorr,
            help=f"**Recommendation:** {info.get('recommendation', 'Yes')}"
        )
        
        with st.expander("ℹ️ Why normalize?"):
            st.markdown(f"""
            **Pros:** {info.get('pros', 'Bounded [-1, 1], comparable across trials')}
            
            **Cons:** {info.get('cons', 'Loses amplitude information')}
            """)
        
        max_lag = st.slider(
            "Maximum Lag (frames)",
            min_value=10, max_value=100, value=config.max_lag_frames,
            help="At 50 fps: 50 frames = 1 second"
        )
        
        frame_rate = st.number_input(
            "Frame Rate (Hz)",
            min_value=1.0, max_value=120.0, value=float(FRAME_RATE),
            help="Recording frame rate"
        )
        
        st.info(f"Max lag: {max_lag} frames = {max_lag / frame_rate * 1000:.0f} ms")
    
    with tab3:
        st.subheader("Statistical Tests")
        
        stat_test = st.selectbox(
            "Test Type",
            ['anova', 'kruskal', 'both'],
            index=2,
            help="ANOVA: parametric, Kruskal: non-parametric"
        )
        
        correction = st.selectbox(
            "Multiple Comparison Correction",
            ['bonferroni', 'holm', 'fdr'],
            index=1,
            help="Holm: good balance of power and control"
        )
        
        alpha = st.slider(
            "Significance Level (α)",
            min_value=0.01, max_value=0.10, value=0.05, step=0.01
        )
    
    with tab4:
        st.subheader("Output Settings")
        
        output_path = st.text_input(
            "Output Directory",
            value=str(Path(st.session_state.data_path).parent / 'results'),
            help="Where to save results"
        )
        
        save_state = st.checkbox(
            "Save analysis state",
            value=True,
            help="Allows reloading without reprocessing"
        )
        
        generate_report = st.checkbox(
            "Generate report",
            value=True
        )
    
    # Save config button
    st.markdown("---")
    
    if st.button("💾 Save Configuration", type="primary"):
        # Update config
        st.session_state.config = Config(
            data_path=st.session_state.data_path,
            output_dir=output_path,
            velocity_method=velocity_method,
            smooth_window=smooth_window,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold,
            missing_data_method=missing_method,
            normalize_xcorr=normalize_xcorr,
            max_lag_frames=max_lag
        )
        
        st.success("✓ Configuration saved!")


# =============================================================================
# PAGE: RUN ANALYSIS
# =============================================================================

def page_run_analysis():
    """Render analysis execution page."""
    st.header("▶️ Run Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first!")
        return
    
    # Auto-create default config if not set (no need to visit Configure page)
    if st.session_state.config is None:
        st.session_state.config = Config()  # Use default configuration
        st.info("💡 Using default configuration. Visit ⚙️ Configure page to customize settings.")
    
    config = st.session_state.config
    
    # Estimate runtime
    n_frames = len(st.session_state.data)
    est_seconds = n_frames / 10000  # Rough estimate
    
    st.info(f"""
    **Ready to analyze:**
    - {st.session_state.metadata.n_files} files
    - {n_frames:,} total frames
    - Estimated time: ~{max(1, est_seconds):.0f} seconds
    """)
    
    # Data filtering options
    with st.expander("🔍 Filter Data (Optional)"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            all_participants = sorted(st.session_state.data['participant_id'].unique())
            selected_participants = st.multiselect(
                "Participants",
                all_participants,
                default=all_participants,
                help="Select specific participants"
            )
        
        with col2:
            all_sizes = sorted(st.session_state.data['size_pixels'].unique())
            selected_sizes = st.multiselect(
                "SD Sizes (pixels)",
                all_sizes,
                default=all_sizes
            )
        
        with col3:
            all_conditions = sorted(st.session_state.data['condition'].unique())
            selected_conditions = st.multiselect(
                "Conditions",
                all_conditions,
                default=all_conditions
            )
    
    # Run button
    if st.button("🚀 Run Analysis", type="primary"):
        
        # Filter data
        data = st.session_state.data.copy()
        data = data[data['participant_id'].isin(selected_participants)]
        data = data[data['size_pixels'].isin(selected_sizes)]
        data = data[data['condition'].isin(selected_conditions)]
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Preprocessing
            status_text.text("Step 1/4: Preprocessing...")
            progress_bar.progress(10)
            
            preprocessor = Preprocessor(config)
            processed_data, preprocess_report = preprocessor.process(data)
            
            st.session_state.processed_data = processed_data
            progress_bar.progress(30)
            
            # Step 2: Metrics
            status_text.text("Step 2/4: Computing metrics...")
            
            metrics_calc = MetricsCalculator(config)
            trial_metrics = metrics_calc.compute_all_trials(processed_data)
            
            st.session_state.trial_metrics = trial_metrics
            progress_bar.progress(50)
            
            # Step 3: Cross-correlation
            status_text.text("Step 3/4: Cross-correlation analysis...")
            
            xcorr_analyzer = CrossCorrelationAnalyzer(config)
            xcorr_results = xcorr_analyzer.analyze_all_trials(processed_data)
            
            st.session_state.xcorr_results = xcorr_results
            progress_bar.progress(70)
            
            # Step 4: Statistics
            status_text.text("Step 4/4: Statistical analysis...")
            
            stats_analyzer = StatisticalAnalyzer(config)
            stat_results = stats_analyzer.generate_results_summary(trial_metrics, metric='rmse')
            
            st.session_state.stat_results = stat_results
            progress_bar.progress(90)
            
            # Save state if requested
            if config.output_dir:
                output_dir = Path(config.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                state_manager = StateManager(output_dir / 'states')
                state_id = state_manager.save_state(
                    config=config,
                    trial_metrics=trial_metrics,
                    xcorr_results=xcorr_results,
                    statistical_results=stat_results,
                    processed_data=processed_data,
                    preprocessing_report=preprocess_report
                )
            
            progress_bar.progress(100)
            status_text.text("✓ Analysis complete!")
            
            st.session_state.analysis_complete = True
            
            # Summary
            st.success("Analysis completed successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trials Analyzed", len(trial_metrics))
            with col2:
                st.metric("Mean RMSE", f"{trial_metrics['rmse'].mean():.2f} px")
            with col3:
                if xcorr_results is not None and 'optimal_lag' in xcorr_results.columns:
                    mean_lag_ms = xcorr_results['optimal_lag'].mean() * (1000 / FRAME_RATE)
                    st.metric("Mean Lag", f"{mean_lag_ms:.0f} ms")
            
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            import traceback
            st.code(traceback.format_exc())


# =============================================================================
# PAGE: RESULTS
# =============================================================================

def page_results():
    """Render results page."""
    st.header("📊 Results")
    
    if not st.session_state.analysis_complete:
        st.warning("Please run analysis first!")
        return
    
    trial_metrics = st.session_state.trial_metrics
    xcorr_results = st.session_state.xcorr_results
    stat_results = st.session_state.stat_results
    
    # Tabs for different result views
    tab1, tab2, tab3 = st.tabs(["📋 Summary", "📊 Detailed Metrics", "📈 Statistics"])
    
    with tab1:
        st.subheader("Analysis Summary")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall RMSE",
                f"{trial_metrics['rmse'].mean():.2f} px",
                f"±{trial_metrics['rmse'].std():.2f}"
            )
        
        with col2:
            if xcorr_results is not None and 'optimal_lag' in xcorr_results.columns:
                mean_lag = xcorr_results['optimal_lag'].mean()
                lag_ms = mean_lag * 20  # Assuming 50 fps
                st.metric("Mean Lag", f"{lag_ms:.0f} ms")
        
        with col3:
            n_trials = len(trial_metrics)
            st.metric("Trials", n_trials)
        
        with col4:
            n_participants = trial_metrics['participant_id'].nunique()
            st.metric("Participants", n_participants)
        
        st.markdown("---")
        
        # RMSE by condition and size
        st.subheader("RMSE by Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**By Blob Size:**")
            size_summary = trial_metrics.groupby('size_pixels')['rmse'].agg(['mean', 'std', 'count'])
            size_summary.columns = ['Mean', 'SD', 'N']
            st.dataframe(size_summary.round(2))
        
        with col2:
            st.markdown("**By Condition:**")
            cond_summary = trial_metrics.groupby('condition')['rmse'].agg(['mean', 'std', 'count'])
            cond_summary.index = cond_summary.index.map(
                {'dynamic': 'Auditory Feedback', 'static': 'No Feedback'}
            )
            cond_summary.columns = ['Mean', 'SD', 'N']
            st.dataframe(cond_summary.round(2))
    
    with tab2:
        st.subheader("Trial-Level Metrics")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox("Sort by", trial_metrics.columns.tolist())
        with col2:
            ascending = st.checkbox("Ascending", value=True)
        
        # Display dataframe
        display_df = trial_metrics.sort_values(sort_by, ascending=ascending)
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = trial_metrics.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            "trial_metrics.csv",
            "text/csv"
        )
    
    with tab3:
        st.subheader("Statistical Results")
        
        if stat_results:
            st.json(stat_results)
        else:
            st.info("No statistical results available")


# =============================================================================
# PAGE: DATA ANALYSIS
# =============================================================================

def page_data_analysis():
    """Render data analysis page with averaging and aggregation features."""
    st.header("📉 Data Analysis & Averaging")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first!")
        return
    
    data = st.session_state.data
    trial_metrics = st.session_state.trial_metrics
    
    st.markdown("""
    Create custom data subsets and compute averaged metrics. Select any combination of 
    participants, blob sizes, and conditions to analyze.
    """)
    
    # ==========================================================================
    # DATA SUBSET SELECTION
    # ==========================================================================
    st.subheader("🔍 Select Data Subset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        all_participants = sorted(data['participant_id'].unique().tolist())
        selected_participants = st.multiselect(
            "Participants",
            all_participants,
            default=all_participants,
            help="Select which participants to include"
        )
    
    with col2:
        all_sizes = sorted(data['size_pixels'].unique().tolist())
        selected_sizes = st.multiselect(
            "Blob Sizes (arcmin)",
            all_sizes,
            default=all_sizes,
            help="Select blob SD sizes (21, 31, or 34 arcmin)"
        )
    
    with col3:
        all_conditions = sorted(data['condition'].unique().tolist())
        selected_conditions = st.multiselect(
            "Conditions",
            all_conditions,
            default=all_conditions,
            help="Select conditions (dynamic=auditory feedback, static=no feedback)"
        )
    
    # Filter data
    filtered_data = data[
        (data['participant_id'].isin(selected_participants)) &
        (data['size_pixels'].isin(selected_sizes)) &
        (data['condition'].isin(selected_conditions))
    ]
    
    # Filter trial metrics if available
    if trial_metrics is not None:
        filtered_metrics = trial_metrics[
            (trial_metrics['participant_id'].isin(selected_participants)) &
            (trial_metrics['size_pixels'].isin(selected_sizes)) &
            (trial_metrics['condition'].isin(selected_conditions))
        ]
    else:
        filtered_metrics = None
    
    # Show selection summary
    st.info(f"**Selected:** {len(filtered_data):,} data points from {len(filtered_data['filename'].unique())} trials")
    
    st.markdown("---")
    
    # ==========================================================================
    # AVERAGING OPTIONS
    # ==========================================================================
    tab1, tab2, tab3 = st.tabs(["📊 Metric Averages", "📈 Position Averages", "📥 Export"])
    
    with tab1:
        st.subheader("Averaged Metrics by Group")
        
        if filtered_metrics is None or len(filtered_metrics) == 0:
            st.warning("Run analysis first to compute trial metrics!")
        else:
            # Groupby selection
            groupby_options = st.multiselect(
                "Group by",
                ["size_pixels", "condition", "participant_id"],
                default=["size_pixels", "condition"],
                help="Select how to group the data for averaging"
            )
            
            if len(groupby_options) > 0:
                # Compute aggregations
                agg_metrics = filtered_metrics.groupby(groupby_options).agg({
                    'rmse': ['mean', 'std', 'min', 'max', 'count'],
                    'mean_error': ['mean', 'std'],
                    'max_error': ['mean', 'std'],
                }).round(3)
                
                # Flatten column names
                agg_metrics.columns = ['_'.join(col).strip() for col in agg_metrics.columns.values]
                agg_metrics = agg_metrics.reset_index()
                
                st.dataframe(agg_metrics, use_container_width=True)
                
                # Summary statistics
                st.markdown("#### Overall Summary for Selection")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean RMSE", f"{filtered_metrics['rmse'].mean():.2f} px")
                with col2:
                    st.metric("Std RMSE", f"{filtered_metrics['rmse'].std():.2f} px")
                with col3:
                    st.metric("N Trials", len(filtered_metrics))
                with col4:
                    st.metric("N Participants", filtered_metrics['participant_id'].nunique())
            else:
                # Overall averages without grouping
                st.markdown("#### Overall Averages (No Grouping)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean RMSE", f"{filtered_metrics['rmse'].mean():.2f} ± {filtered_metrics['rmse'].std():.2f} px")
                with col2:
                    st.metric("Mean Error", f"{filtered_metrics['mean_error'].mean():.2f} ± {filtered_metrics['mean_error'].std():.2f} px")
                with col3:
                    st.metric("N Trials", len(filtered_metrics))
    
    with tab2:
        st.subheader("Averaged Position Data")
        st.markdown("""
        Compute frame-by-frame averages of Target and Mouse positions across selected trials.
        Useful for visualizing average tracking behavior.
        """)
        
        if len(filtered_data) == 0:
            st.warning("No data selected!")
        else:
            # Compute frame-by-frame averages
            avg_by_frame = filtered_data.groupby('Frame').agg({
                'Target_X': ['mean', 'std'],
                'Target_Y': ['mean', 'std'],
                'Mouse_X': ['mean', 'std'],
                'Mouse_Y': ['mean', 'std']
            }).reset_index()
            
            avg_by_frame.columns = ['Frame', 
                                     'Target_X_mean', 'Target_X_std',
                                     'Target_Y_mean', 'Target_Y_std', 
                                     'Mouse_X_mean', 'Mouse_X_std',
                                     'Mouse_Y_mean', 'Mouse_Y_std']
            
            # Show stats
            st.write(f"Averaged across {len(filtered_data['filename'].unique())} trials, {len(avg_by_frame)} frames")
            
            # Plot averaged trajectories
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(rows=2, cols=1, 
                               subplot_titles=("X Position (averaged)", "Y Position (averaged)"),
                               vertical_spacing=0.12)
            
            # X positions
            fig.add_trace(go.Scatter(
                x=avg_by_frame['Frame'], y=avg_by_frame['Target_X_mean'],
                mode='lines', name='Target X', line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=avg_by_frame['Frame'], y=avg_by_frame['Mouse_X_mean'],
                mode='lines', name='Mouse X', line=dict(color='red', width=2)
            ), row=1, col=1)
            
            # Add std bands for X
            fig.add_trace(go.Scatter(
                x=list(avg_by_frame['Frame']) + list(avg_by_frame['Frame'][::-1]),
                y=list(avg_by_frame['Target_X_mean'] + avg_by_frame['Target_X_std']) + 
                  list((avg_by_frame['Target_X_mean'] - avg_by_frame['Target_X_std'])[::-1]),
                fill='toself', fillcolor='rgba(0,0,255,0.1)', line=dict(width=0),
                showlegend=False, name='Target X ±1 SD'
            ), row=1, col=1)
            
            # Y positions
            fig.add_trace(go.Scatter(
                x=avg_by_frame['Frame'], y=avg_by_frame['Target_Y_mean'],
                mode='lines', name='Target Y', line=dict(color='blue', width=2, dash='dash')
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=avg_by_frame['Frame'], y=avg_by_frame['Mouse_Y_mean'],
                mode='lines', name='Mouse Y', line=dict(color='red', width=2, dash='dash')
            ), row=2, col=1)
            
            fig.update_layout(height=600, title_text="Averaged Trajectories", hovermode='x unified')
            fig.update_xaxes(title_text="Frame", row=2, col=1)
            fig.update_yaxes(title_text="X Position (px)", row=1, col=1)
            fig.update_yaxes(title_text="Y Position (px)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Averaged Error plot
            avg_by_frame['Error_X'] = avg_by_frame['Target_X_mean'] - avg_by_frame['Mouse_X_mean']
            avg_by_frame['Error_Y'] = avg_by_frame['Target_Y_mean'] - avg_by_frame['Mouse_Y_mean']
            avg_by_frame['Error_Euclidean'] = np.sqrt(avg_by_frame['Error_X']**2 + avg_by_frame['Error_Y']**2)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=avg_by_frame['Frame'], y=avg_by_frame['Error_Euclidean'],
                mode='lines', name='Average Error', line=dict(color='green', width=2),
                fill='tozeroy', fillcolor='rgba(0,128,0,0.2)'
            ))
            fig2.update_layout(
                title="Average Tracking Error Over Time",
                xaxis_title="Frame",
                yaxis_title="Error (pixels)",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("Export Averaged Data")
        
        if filtered_metrics is not None and len(filtered_metrics) > 0:
            # Export metrics
            csv_metrics = filtered_metrics.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Trial Metrics (CSV)",
                data=csv_metrics,
                file_name="filtered_trial_metrics.csv",
                mime="text/csv"
            )
        
        if len(filtered_data) > 0:
            # Export raw filtered data (warning about size)
            n_rows = len(filtered_data)
            if n_rows > 100000:
                st.warning(f"Large dataset: {n_rows:,} rows. Export may take a moment.")
            
            if st.button("Prepare Raw Data Export"):
                csv_raw = filtered_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Raw Data (CSV)",
                    data=csv_raw,
                    file_name="filtered_raw_data.csv",
                    mime="text/csv"
                )


# =============================================================================
# PAGE: VISUALIZATIONS
# =============================================================================

def page_visualizations():
    """Render visualizations page."""
    st.header("📈 Visualizations")
    
    if not st.session_state.analysis_complete:
        st.warning("Please run analysis first!")
        return
    
    trial_metrics = st.session_state.trial_metrics
    xcorr_results = st.session_state.xcorr_results
    
    # Import visualization modules
    try:
        from tracking_analysis.visualization.static_plots import StaticPlotter
        plotter = StaticPlotter()
    except ImportError:
        st.error("Visualization module not available")
        return
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 RMSE Plots", 
        "📈 Cross-Correlation", 
        "🎬 Animations",
        "📉 Psychometric Function",
        "📈 Learning Curve",
        "⚡ Speed-Accuracy",
        "🎯 Pursuit Gain"
    ])
    
    with tab1:
        st.subheader("RMSE Visualizations")
        
        plot_type = st.selectbox(
            "Plot Type",
            ["By Size", "By Condition", "Interaction", "Summary Panel"]
        )
        
        if st.button("Generate Plot"):
            import matplotlib.pyplot as plt
            
            if plot_type == "By Size":
                fig = plotter.plot_rmse_by_size(trial_metrics)
            elif plot_type == "By Condition":
                fig = plotter.plot_rmse_by_condition(trial_metrics)
            elif plot_type == "Interaction":
                fig = plotter.plot_rmse_interaction(trial_metrics)
            else:
                fig = plotter.plot_summary_panel(trial_metrics, xcorr_results)
            
            st.pyplot(fig)
            plt.close(fig)
    
    with tab2:
        st.subheader("📈 Cross-Correlation Analysis")
        
        if xcorr_results is None or 'optimal_lag_frames' not in xcorr_results.columns:
            st.info("Cross-correlation results not available. Run analysis first.")
        else:
            # === SECTION 1: METHODOLOGY EXPLAINER ===
            with st.expander("📚 **Understanding Cross-Correlation Analysis**", expanded=False):
                st.markdown("""
                ### Position → Velocity Conversion
                
                Cross-correlation is computed on **velocity signals**, not raw positions.
                
                **Why velocity?** Position correlations are dominated by low-frequency trends.
                Velocity captures the dynamic tracking behavior - how well the mouse *follows* target movement.
                
                **Conversion methods:**
                | Method | Formula | Pros | Cons |
                |--------|---------|------|------|
                | **Difference** | v(t) = x(t) - x(t-1) | Fast, simple | Noisy |
                | **Savitzky-Golay** | Polynomial smoothing + derivative | Smooth, accurate | Slower |
                
                ### Interpreting Results
                
                **Optimal Lag:**
                - **Positive lag**: REACTIVE tracking (mouse follows target)
                - **Negative lag**: PREDICTIVE tracking (mouse anticipates target)
                - **Zero lag**: SYNCHRONOUS tracking
                
                **Typical values:** 50-200ms reactive lag is common. Negative lags suggest anticipation/learning.
                """)
            
            # === SECTION 2: CORRELATION STRENGTH THRESHOLDS ===
            st.markdown("#### ⚙️ Correlation Strength Thresholds")
            st.caption("Configure how correlation strength is interpreted (standard behavioral research defaults shown)")
            
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                weak_thresh = st.number_input("Weak threshold", 0.1, 0.5, 0.3, 0.05, 
                                              help="Correlations below this are 'very weak'")
            with col_t2:
                moderate_thresh = st.number_input("Moderate threshold", 0.3, 0.8, 0.6, 0.05,
                                                  help="Correlations below this are 'weak' to 'moderate'")
            with col_t3:
                strong_thresh = st.number_input("Strong threshold", 0.5, 0.95, 0.7, 0.05,
                                                help="Correlations above this are 'strong'")
            
            def interpret_corr_strength(r):
                r_abs = abs(r)
                if r_abs < weak_thresh:
                    return "🔴 Very Weak"
                elif r_abs < moderate_thresh:
                    return "🟠 Weak-Moderate"
                elif r_abs < strong_thresh:
                    return "🟡 Moderate"
                else:
                    return "🟢 Strong"
            
            st.divider()
            
            # === SECTION 3: INDIVIDUAL TRIAL RESULTS ===
            st.markdown("#### 🔬 Individual Trial Results")
            
            # Add interpretation columns
            display_df = xcorr_results.copy()
            display_df['Lag Interpretation'] = display_df.apply(
                lambda x: "🔮 Predictive" if x['is_predictive'] else ("⏱️ Reactive" if x['is_reactive'] else "🎯 Synchronous"),
                axis=1
            )
            display_df['Correlation Strength'] = display_df['max_correlation'].apply(interpret_corr_strength)
            display_df['Lag (ms)'] = display_df['optimal_lag_ms'].round(1)
            display_df['Max r'] = display_df['max_correlation'].round(3)
            
            # Show summary stats
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                n_pred = display_df['is_predictive'].sum()
                st.metric("Predictive Trials", f"{n_pred} ({100*n_pred/len(display_df):.1f}%)")
            with col_s2:
                n_react = display_df['is_reactive'].sum()
                st.metric("Reactive Trials", f"{n_react} ({100*n_react/len(display_df):.1f}%)")
            with col_s3:
                st.metric("Mean Lag", f"{display_df['optimal_lag_ms'].mean():.1f} ms")
            with col_s4:
                st.metric("Mean Correlation", f"{display_df['max_correlation'].mean():.3f}")
            
            # Filterable table
            size_col = 'size_pixels' if 'size_pixels' in display_df.columns else 'size'
            show_cols = ['participant_id', size_col, 'condition', 'Lag (ms)', 'Lag Interpretation', 'Max r', 'Correlation Strength']
            show_cols = [c for c in show_cols if c in display_df.columns]
            st.dataframe(display_df[show_cols], use_container_width=True, height=300)
            
            # Download button
            csv_trials = display_df.to_csv(index=False)
            st.download_button("📥 Download Trial Results (CSV)", csv_trials, "xcorr_trials.csv", "text/csv")
            
            st.divider()
            
            # === SECTION 4: GROUP STATISTICS ===
            st.markdown("#### 📊 Group Statistics")
            
            show_size_breakdown = st.checkbox("📊 Break down by blob size", value=True)
            
            from tracking_analysis.core.cross_correlation import aggregate_correlations
            
            size_col = 'size_pixels' if 'size_pixels' in xcorr_results.columns else 'size'
            if show_size_breakdown:
                groupby_cols = [size_col, 'condition']
            else:
                groupby_cols = ['condition']
            
            try:
                group_summary = aggregate_correlations(xcorr_results, groupby=groupby_cols)
                
                # Clean up column names for display
                rename_map = {
                    'optimal_lag_ms_mean': 'Lag Mean (ms)',
                    'optimal_lag_ms_std': 'Lag SD',
                    'max_correlation_mean': 'Corr Mean',
                    'max_correlation_std': 'Corr SD',
                    'is_predictive_mean': 'Predictive %',
                    'n_trials': 'N Trials'
                }
                display_summary = group_summary.rename(columns=rename_map)
                
                # Format predictive % 
                if 'Predictive %' in display_summary.columns:
                    display_summary['Predictive %'] = (display_summary['Predictive %'] * 100).round(1).astype(str) + '%'
                
                # Select columns to show
                cols_to_show = groupby_cols + [c for c in rename_map.values() if c in display_summary.columns]
                st.dataframe(display_summary[cols_to_show], use_container_width=True)
                
                csv_summary = group_summary.to_csv(index=False)
                st.download_button("📥 Download Group Summary (CSV)", csv_summary, "xcorr_summary.csv", "text/csv")
            except Exception as e:
                st.error(f"Could not compute group statistics: {e}")
            
            st.divider()
            
            # === SECTION 5: LAG DISTRIBUTION PLOT ===
            st.markdown("#### 📉 Lag Distribution")
            
            if st.button("Generate Lag Distribution Plot"):
                import matplotlib.pyplot as plt
                fig = plotter.plot_lag_distribution(xcorr_results, frame_rate=FRAME_RATE)
                st.pyplot(fig)
                plt.close(fig)
            
            st.divider()
            
            # === SECTION 6: DYNAMIC VS STATIC COMPARISON ===
            st.markdown("#### 🎧 Dynamic vs Static: Statistical Comparison")
            st.markdown("*Does auditory feedback improve tracking performance?*")
            
            from tracking_analysis.core.statistics import StatisticalAnalyzer
            stats_analyzer = StatisticalAnalyzer(st.session_state.config)
            
            try:
                comparison = stats_analyzer.compare_xcorr_conditions(xcorr_results)
                
                # Main answer
                st.markdown(comparison['summary']['answer_does_feedback_help'])
                
                # Detailed results in columns
                col_lag, col_corr = st.columns(2)
                
                with col_lag:
                    st.markdown("**Tracking Lag Comparison**")
                    lag = comparison['lag_comparison']
                    st.write(f"Dynamic: {lag['dynamic_mean']:.1f} ± {lag['dynamic_std']:.1f} ms")
                    st.write(f"Static: {lag['static_mean']:.1f} ± {lag['static_std']:.1f} ms")
                    sig_icon = "✅" if lag['is_significant'] else "❌"
                    st.write(f"{sig_icon} p = {lag['p_value']:.4f}, d = {lag['cohens_d']:.2f} ({lag['effect_interpretation']})")
                
                with col_corr:
                    st.markdown("**Correlation Strength Comparison**")
                    corr = comparison['correlation_comparison']
                    st.write(f"Dynamic: {corr['dynamic_mean']:.3f} ± {corr['dynamic_std']:.3f}")
                    st.write(f"Static: {corr['static_mean']:.3f} ± {corr['static_std']:.3f}")
                    sig_icon = "✅" if corr['is_significant'] else "❌"
                    st.write(f"{sig_icon} p = {corr['p_value']:.4f}, d = {corr['cohens_d']:.2f} ({corr['effect_interpretation']})")
                
                # Predictive tracking proportions
                pred = comparison['predictive_tracking']
                st.markdown("**Predictive vs Reactive Tracking**")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("Dynamic - Predictive", f"{pred['dynamic_predictive_pct']:.1f}%", 
                              f"{pred['dynamic_predictive_n']}/{pred['dynamic_n']} trials")
                with col_p2:
                    st.metric("Static - Predictive", f"{pred['static_predictive_pct']:.1f}%",
                              f"{pred['static_predictive_n']}/{pred['static_n']} trials")
                
                # By-size breakdown
                if show_size_breakdown and comparison['by_size']:
                    st.markdown("**By Blob Size:**")
                    
                    size_rows = []
                    for size, data in comparison['by_size'].items():
                        size_rows.append({
                            'Size (arcmin)': size,
                            'Lag Dyn (ms)': f"{data['lag']['dynamic_mean']:.1f}",
                            'Lag Sta (ms)': f"{data['lag']['static_mean']:.1f}",
                            'Lag p': f"{data['lag']['p_value']:.3f}" + ("*" if data['lag']['is_significant'] else ""),
                            'Corr Dyn': f"{data['correlation']['dynamic_mean']:.3f}",
                            'Corr Sta': f"{data['correlation']['static_mean']:.3f}",
                            'Corr p': f"{data['correlation']['p_value']:.3f}" + ("*" if data['correlation']['is_significant'] else ""),
                        })
                    
                    st.dataframe(pd.DataFrame(size_rows), use_container_width=True)
                    st.caption("* indicates p < 0.05")
                
                # Download comparison
                import json
                comparison_json = json.dumps(comparison, indent=2, default=str)
                st.download_button("📥 Download Full Comparison (JSON)", comparison_json, "xcorr_comparison.json", "application/json")
                
            except Exception as e:
                st.error(f"Could not compute condition comparison: {e}")
                import traceback
                st.text(traceback.format_exc())
    
    with tab3:
        st.subheader("Trial Animations")
        
        st.markdown("""
        Create 2D spatial animations showing Target vs Mouse movement over time.
        Animate **individual trials** or **averaged data** by condition.
        """)
        
        # === DATA SOURCE SELECTION ===
        st.markdown("#### 📊 Data Source")
        data_source = st.radio(
            "What to animate",
            ["🎯 Single Trial", "📊 Averaged Data"],
            horizontal=True,
            help="Single Trial: One specific trial. Averaged Data: Mean trajectory across multiple trials."
        )
        
        data = st.session_state.data
        n_trials = 0
        
        if data_source == "🎯 Single Trial":
            # Individual trial selection
            trial_options = trial_metrics.apply(
                lambda x: f"{x['participant_id']} - {x['size_pixels']}arcmin - {x['condition']}",
                axis=1
            ).tolist()
            selected_trial = st.selectbox("Select Trial", trial_options[:50], key="anim_trial")
            animation_title = selected_trial
        else:
            # Averaged data selection
            st.markdown("**Select data to average:**")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                avg_sizes = st.multiselect(
                    "Blob Sizes",
                    sorted(data['size_pixels'].unique().tolist()),
                    default=sorted(data['size_pixels'].unique().tolist()),
                    key="avg_sizes"
                )
            
            with col_b:
                avg_conditions = st.multiselect(
                    "Conditions",
                    sorted(data['condition'].unique().tolist()),
                    default=sorted(data['condition'].unique().tolist()),
                    key="avg_conditions"
                )
            
            with col_c:
                avg_participants = st.multiselect(
                    "Participants (optional)",
                    ["All"] + sorted(data['participant_id'].unique().tolist()),
                    default=["All"],
                    key="avg_participants"
                )
            
            # Build title based on selection
            size_str = ", ".join([f"{s}arcmin" for s in avg_sizes]) if len(avg_sizes) < 3 else "All sizes"
            cond_str = ", ".join(avg_conditions) if len(avg_conditions) < 2 else "Both conditions"
            animation_title = f"Average: {size_str} | {cond_str}"
            
            # Show how many trials will be averaged
            if avg_sizes and avg_conditions:
                filter_mask = (
                    data['size_pixels'].isin(avg_sizes) &
                    data['condition'].isin(avg_conditions)
                )
                if "All" not in avg_participants:
                    filter_mask &= data['participant_id'].isin(avg_participants)
                
                n_trials = len(data[filter_mask]['filename'].unique())
                st.info(f"Will average across **{n_trials} trials**")
        
        # === ANIMATION SETTINGS ===
        st.markdown("#### ⚙️ Animation Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            animation_mode = st.radio(
                "Animation Mode",
                ["🚀 Fast (Frame Skip)", "🎬 Full (All Frames)"],
                help="Fast mode skips frames for performance. Full mode shows all frames."
            )
            
            if "Fast" in animation_mode:
                frame_skip = st.slider("Frame Skip", 1, 20, 5, 
                                       help="Show every Nth frame. Higher = faster but less smooth.")
            else:
                frame_skip = 1
        
        with col2:
            show_trail = st.checkbox("Show Trail", value=True, 
                                     help="Show trailing path of movement")
            trail_length = st.slider("Trail Length", 5, 50, 20, 
                                     help="Number of frames in trail") if show_trail else 0
            show_error_line = st.checkbox("Show Error Line", value=True,
                                          help="Show line connecting Target and Mouse")
        
        # Animation type selection
        anim_type = st.radio(
            "Animation Type",
            ["🎯 2D Spatial (X-Y plane)", "📊 Time Series (Position vs Frame)"],
            horizontal=True
        )
        
        if st.button("🎬 Generate Animation", type="primary"):
            # === PREPARE DATA ===
            is_averaged = False
            
            if data_source == "🎯 Single Trial":
                # Parse selected trial info
                parts = selected_trial.split(" - ")
                pid = parts[0]
                size = int(parts[1].replace("arcmin", ""))
                cond = parts[2]
                
                # Filter data for this trial
                trial_data = data[
                    (data['participant_id'] == pid) &
                    (data['size_pixels'] == size) &
                    (data['condition'] == cond)
                ].copy().reset_index(drop=True)
            else:
                # === COMPUTE AVERAGED DATA ===
                is_averaged = True
                filter_mask = (
                    data['size_pixels'].isin(avg_sizes) &
                    data['condition'].isin(avg_conditions)
                )
                if "All" not in avg_participants:
                    filter_mask &= data['participant_id'].isin(avg_participants)
                
                filtered_data = data[filter_mask]
                
                # Compute frame-by-frame averages
                trial_data = filtered_data.groupby('Frame').agg({
                    'Target_X': 'mean',
                    'Target_Y': 'mean',
                    'Mouse_X': 'mean',
                    'Mouse_Y': 'mean'
                }).reset_index()
            
            if len(trial_data) == 0:
                st.error("Could not find trial data")
            else:
                total_frames = len(trial_data)
                
                with st.spinner("Generating animation..."):
                    import plotly.graph_objects as go
                    
                    # Apply frame skip
                    if frame_skip > 1:
                        trial_data = trial_data.iloc[::frame_skip].reset_index(drop=True)
                    
                    n_frames = len(trial_data)
                    
                    if "2D Spatial" in anim_type:
                        # === 2D SPATIAL ANIMATION ===
                        frames = []
                        
                        for idx in range(n_frames):
                            row = trial_data.iloc[idx]
                            
                            # Trail data
                            trail_start = max(0, idx - trail_length)
                            trail_df = trial_data.iloc[trail_start:idx+1]
                            
                            frame_data = [
                                # Target marker
                                go.Scatter(
                                    x=[row['Target_X']], y=[row['Target_Y']],
                                    mode='markers', marker=dict(size=20, color='#3498db', symbol='circle'),
                                    name='Target'
                                ),
                                # Mouse marker  
                                go.Scatter(
                                    x=[row['Mouse_X']], y=[row['Mouse_Y']],
                                    mode='markers', marker=dict(size=15, color='#e74c3c', symbol='x'),
                                    name='Mouse'
                                ),
                            ]
                            
                            if show_trail:
                                # Target trail
                                frame_data.append(go.Scatter(
                                    x=trail_df['Target_X'], y=trail_df['Target_Y'],
                                    mode='lines', line=dict(color='#3498db', width=2),
                                    opacity=0.4, showlegend=False
                                ))
                                # Mouse trail
                                frame_data.append(go.Scatter(
                                    x=trail_df['Mouse_X'], y=trail_df['Mouse_Y'],
                                    mode='lines', line=dict(color='#e74c3c', width=2),
                                    opacity=0.4, showlegend=False
                                ))
                            
                            if show_error_line:
                                frame_data.append(go.Scatter(
                                    x=[row['Target_X'], row['Mouse_X']],
                                    y=[row['Target_Y'], row['Mouse_Y']],
                                    mode='lines', line=dict(color='gray', width=1, dash='dash'),
                                    showlegend=False
                                ))
                            
                            frames.append(go.Frame(data=frame_data, name=str(idx)))
                        
                        # Initial frame
                        init_row = trial_data.iloc[0]
                        init_data = [
                            go.Scatter(x=[init_row['Target_X']], y=[init_row['Target_Y']],
                                      mode='markers', marker=dict(size=20, color='#3498db'),
                                      name='Target'),
                            go.Scatter(x=[init_row['Mouse_X']], y=[init_row['Mouse_Y']],
                                      mode='markers', marker=dict(size=15, color='#e74c3c'),
                                      name='Mouse'),
                        ]
                        if show_trail:
                            init_data.extend([
                                go.Scatter(x=[], y=[], mode='lines', showlegend=False),
                                go.Scatter(x=[], y=[], mode='lines', showlegend=False)
                            ])
                        if show_error_line:
                            init_data.append(go.Scatter(x=[], y=[], mode='lines', showlegend=False))
                        
                        # Compute axis ranges with padding
                        x_min = min(trial_data['Target_X'].min(), trial_data['Mouse_X'].min()) - 50
                        x_max = max(trial_data['Target_X'].max(), trial_data['Mouse_X'].max()) + 50
                        y_min = min(trial_data['Target_Y'].min(), trial_data['Mouse_Y'].min()) - 50
                        y_max = max(trial_data['Target_Y'].max(), trial_data['Mouse_Y'].max()) + 50
                        
                        title_prefix = "📊 AVERAGED " if is_averaged else ""
                        fig = go.Figure(
                            data=init_data,
                            frames=frames,
                            layout=go.Layout(
                                title=f"{title_prefix}2D Tracking: {animation_title}",
                                xaxis=dict(range=[x_min, x_max], title='X Position (pixels)'),
                                yaxis=dict(range=[y_min, y_max], title='Y Position (pixels)', scaleanchor='x'),
                                updatemenus=[dict(
                                    type='buttons', showactive=False, y=1.15, x=0.5, xanchor='center',
                                    buttons=[
                                        dict(label='▶ Play', method='animate',
                                             args=[None, dict(frame=dict(duration=50, redraw=True),
                                                            fromcurrent=True, mode='immediate')]),
                                        dict(label='⏸ Pause', method='animate',
                                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                               mode='immediate')])
                                    ]
                                )],
                                sliders=[dict(
                                    active=0,
                                    steps=[dict(method='animate', args=[[str(i)],
                                               dict(mode='immediate', frame=dict(duration=50, redraw=True))],
                                               label=str(int(trial_data.iloc[i]['Frame'])))
                                           for i in range(n_frames)],
                                    x=0, y=0, len=1.0,
                                    currentvalue=dict(prefix='Frame: ', visible=True),
                                    transition=dict(duration=0)
                                )]
                            )
                        )
                        
                        fig.update_layout(height=700)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        # === TIME SERIES ANIMATION ===
                        title_prefix = "📊 AVERAGED " if is_averaged else ""
                        
                        fig = go.Figure()
                        
                        # X position over time
                        fig.add_trace(go.Scatter(
                            x=trial_data['Frame'], y=trial_data['Target_X'],
                            mode='lines', name='Target X', line=dict(color='blue', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=trial_data['Frame'], y=trial_data['Mouse_X'],
                            mode='lines', name='Mouse X', line=dict(color='red', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f"{title_prefix}X Position: {animation_title}",
                            xaxis_title="Frame",
                            yaxis_title="X Position (pixels)",
                            hovermode='x unified',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Y position
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=trial_data['Frame'], y=trial_data['Target_Y'],
                            mode='lines', name='Target Y', line=dict(color='blue', width=2, dash='dash')
                        ))
                        fig2.add_trace(go.Scatter(
                            x=trial_data['Frame'], y=trial_data['Mouse_Y'],
                            mode='lines', name='Mouse Y', line=dict(color='red', width=2, dash='dash')
                        ))
                        fig2.update_layout(
                            title=f"{title_prefix}Y Position: {animation_title}",
                            xaxis_title="Frame", yaxis_title="Y Position (pixels)",
                            hovermode='x unified', height=400
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Error over time
                        error = np.sqrt(
                            (trial_data['Target_X'] - trial_data['Mouse_X'])**2 +
                            (trial_data['Target_Y'] - trial_data['Mouse_Y'])**2
                        )
                        fig3 = go.Figure()
                        fig3.add_trace(go.Scatter(
                            x=trial_data['Frame'], y=error,
                            mode='lines', name='Euclidean Error',
                            line=dict(color='green', width=2), fill='tozeroy'
                        ))
                        fig3.update_layout(
                            title=f"{title_prefix}Tracking Error: {animation_title}",
                            xaxis_title="Frame", yaxis_title="Error (pixels)",
                            height=350
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                
                # Show animation info
                if is_averaged:
                    st.success(f"✅ Averaged animation generated! ({n_frames} frames, {n_trials} trials averaged)")
                else:
                    st.success(f"✅ Animation generated! ({n_frames} frames displayed)")
                
                if frame_skip > 1:
                    st.caption(f"ℹ️ Fast mode: showing every {frame_skip}th frame ({n_frames} of {total_frames} total)")
    
    # ==========================================================================
    # TAB 4: PSYCHOMETRIC FUNCTION
    # ==========================================================================
    with tab4:
        st.subheader("📉 Psychometric Function")
        
        st.markdown("""
        The psychometric function shows how tracking performance (probability of "good" tracking) 
        varies with blob size. This is fundamental for determining discrimination thresholds.
        """)
        
        # Get size column
        size_col = 'size' if 'size' in trial_metrics.columns else 'blob_size'
        
        # Configuration
        col1, col2 = st.columns(2)
        with col1:
            psych_metric = st.selectbox(
                "Performance Metric",
                ["rmse", "tracking_gain", "mean_error"],
                help="Metric to use for determining 'good' vs 'poor' tracking"
            )
        with col2:
            if psych_metric == "rmse":
                psych_threshold = st.slider("RMSE Threshold (px)", 20.0, 150.0, 50.0, 5.0)
            elif psych_metric == "tracking_gain":
                psych_threshold = st.slider("Gain Threshold", 0.5, 1.5, 0.8, 0.05)
            else:
                psych_threshold = st.slider("Mean Error Threshold (px)", 10.0, 100.0, 40.0, 5.0)
        
        # Condition filter
        condition_filter = st.radio(
            "Condition", 
            ["All", "dynamic", "static"],
            horizontal=True
        )
        
        if st.button("Generate Psychometric Function", key="psych_btn"):
            import plotly.graph_objects as go
            from scipy.optimize import curve_fit
            import numpy as np
            
            # Filter data
            plot_data = trial_metrics.copy()
            if condition_filter != "All":
                plot_data = plot_data[plot_data['condition'] == condition_filter]
            
            if psych_metric not in plot_data.columns:
                st.error(f"Metric '{psych_metric}' not found in data")
            else:
                # Calculate proportion "good" for each size
                sizes = sorted(plot_data[size_col].unique())
                proportions = []
                n_trials_per_size = []
                
                for size in sizes:
                    size_data = plot_data[plot_data[size_col] == size][psych_metric]
                    if psych_metric == "rmse" or psych_metric == "mean_error":
                        # Lower is better
                        prop_good = (size_data <= psych_threshold).mean()
                    else:
                        # Gain: closer to 1 is better
                        prop_good = (size_data >= psych_threshold).mean()
                    proportions.append(prop_good)
                    n_trials_per_size.append(len(size_data))
                
                # Psychometric function (cumulative Gaussian)
                def psychometric(x, alpha, beta):
                    """Cumulative Gaussian psychometric function."""
                    from scipy.stats import norm
                    return norm.cdf(x, loc=alpha, scale=beta)
                
                # Try to fit
                try:
                    sizes_arr = np.array(sizes)
                    props_arr = np.array(proportions)
                    
                    # Initial guess: threshold at middle size, moderate slope
                    p0 = [np.median(sizes_arr), 5]
                    popt, _ = curve_fit(psychometric, sizes_arr, props_arr, p0=p0, maxfev=5000)
                    
                    # Generate smooth curve
                    x_smooth = np.linspace(min(sizes_arr) - 5, max(sizes_arr) + 5, 100)
                    y_smooth = psychometric(x_smooth, *popt)
                    
                    threshold_50 = popt[0]  # 50% threshold
                    fitted = True
                except:
                    fitted = False
                    threshold_50 = None
                
                # Create plot
                fig = go.Figure()
                
                # Data points with error bars (binomial SE)
                se_props = [np.sqrt(p * (1-p) / n) if n > 0 else 0 for p, n in zip(proportions, n_trials_per_size)]
                
                fig.add_trace(go.Scatter(
                    x=sizes, y=proportions,
                    mode='markers',
                    name='Data',
                    marker=dict(size=15, color='#3498db'),
                    error_y=dict(type='data', array=se_props, visible=True)
                ))
                
                # Fitted curve
                if fitted:
                    fig.add_trace(go.Scatter(
                        x=x_smooth.tolist(), y=y_smooth.tolist(),
                        mode='lines',
                        name=f'Fitted (α={threshold_50:.1f})',
                        line=dict(color='#e74c3c', width=2)
                    ))
                    
                    # 50% threshold line
                    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                                  annotation_text="50% threshold")
                    fig.add_vline(x=threshold_50, line_dash="dash", line_color="gray")
                
                # Reference lines
                fig.add_hline(y=0.75, line_dash="dot", line_color="green",
                              annotation_text="75% (good discrimination)")
                
                fig.update_layout(
                    title=f"Psychometric Function: P(Good Tracking) vs Blob Size<br><sub>Threshold: {psych_metric} {'≤' if psych_metric in ['rmse', 'mean_error'] else '≥'} {psych_threshold}</sub>",
                    xaxis_title="Blob Size (arcmin)",
                    yaxis_title="Proportion 'Good' Tracking",
                    yaxis=dict(range=[0, 1.05]),
                    showlegend=True,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary table
                st.markdown("#### Summary by Size")
                summary_df = pd.DataFrame({
                    'Size (arcmin)': sizes,
                    'P(Good)': [f"{p:.2%}" for p in proportions],
                    'N Trials': n_trials_per_size,
                    'SE': [f"±{se:.2%}" for se in se_props]
                })
                st.dataframe(summary_df, use_container_width=True)
                
                if fitted and threshold_50:
                    st.success(f"**Estimated 50% Threshold:** {threshold_50:.1f} arcmin")
    
    # ==========================================================================
    # TAB 5: LEARNING CURVE
    # ==========================================================================
    with tab5:
        st.subheader("📈 Learning Curve (Practice Effects)")
        
        st.markdown("""
        Analyze how tracking performance changes over the course of the experiment.
        Decreasing error over trials indicates learning/practice effects.
        """)
        
        size_col = 'size' if 'size' in trial_metrics.columns else 'blob_size'
        
        col1, col2 = st.columns(2)
        with col1:
            learn_metric = st.selectbox(
                "Metric to Track",
                ["rmse", "mean_error", "tracking_gain"],
                key="learn_metric"
            )
        with col2:
            window_size = st.slider("Smoothing Window (trials)", 1, 20, 5, 
                                    help="Moving average window for trend line")
        
        # Grouping option
        group_by = st.radio(
            "Group by",
            ["All Combined", "By Participant", "By Condition", "By Size"],
            horizontal=True,
            key="learn_group"
        )
        
        if st.button("Generate Learning Curve", key="learn_btn"):
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np
            
            plot_data = trial_metrics.copy()
            
            if learn_metric not in plot_data.columns:
                st.error(f"Metric '{learn_metric}' not found in data")
            else:
                # Add trial order within each participant (using filename as trial identifier)
                # Sort by participant and filename to establish trial order
                plot_data = plot_data.sort_values(['participant_id', 'filename'])
                plot_data['trial_order'] = plot_data.groupby('participant_id').cumcount() + 1
                
                fig = go.Figure()
                
                if group_by == "All Combined":
                    # Average across all participants at each trial position
                    avg_by_trial = plot_data.groupby('trial_order')[learn_metric].agg(['mean', 'std', 'count']).reset_index()
                    avg_by_trial['se'] = avg_by_trial['std'] / np.sqrt(avg_by_trial['count'])
                    
                    # Moving average
                    avg_by_trial['smooth'] = avg_by_trial['mean'].rolling(window=window_size, center=True, min_periods=1).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=avg_by_trial['trial_order'],
                        y=avg_by_trial['mean'],
                        mode='markers',
                        name='Mean',
                        marker=dict(size=6, opacity=0.5),
                        error_y=dict(type='data', array=avg_by_trial['se'], visible=True)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=avg_by_trial['trial_order'],
                        y=avg_by_trial['smooth'],
                        mode='lines',
                        name=f'Trend (window={window_size})',
                        line=dict(width=3, color='#e74c3c')
                    ))
                    
                elif group_by == "By Participant":
                    colors = px.colors.qualitative.Set2
                    for i, (pid, pdata) in enumerate(plot_data.groupby('participant_id')):
                        pdata_sorted = pdata.sort_values('trial_order')
                        smooth = pdata_sorted[learn_metric].rolling(window=window_size, center=True, min_periods=1).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=pdata_sorted['trial_order'],
                            y=smooth,
                            mode='lines',
                            name=f'P{pid}',
                            line=dict(color=colors[i % len(colors)])
                        ))
                        
                elif group_by == "By Condition":
                    colors = {'dynamic': '#3498db', 'static': '#e74c3c'}
                    for cond, cdata in plot_data.groupby('condition'):
                        avg_by_trial = cdata.groupby('trial_order')[learn_metric].mean().reset_index()
                        avg_by_trial['smooth'] = avg_by_trial[learn_metric].rolling(window=window_size, center=True, min_periods=1).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=avg_by_trial['trial_order'],
                            y=avg_by_trial['smooth'],
                            mode='lines',
                            name=cond.capitalize(),
                            line=dict(color=colors.get(cond, '#95a5a6'), width=3)
                        ))
                        
                elif group_by == "By Size":
                    colors = {21: '#2ecc71', 31: '#3498db', 34: '#9b59b6'}
                    for size, sdata in plot_data.groupby(size_col):
                        avg_by_trial = sdata.groupby('trial_order')[learn_metric].mean().reset_index()
                        avg_by_trial['smooth'] = avg_by_trial[learn_metric].rolling(window=window_size, center=True, min_periods=1).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=avg_by_trial['trial_order'],
                            y=avg_by_trial['smooth'],
                            mode='lines',
                            name=f'{size} arcmin',
                            line=dict(color=colors.get(size, '#95a5a6'), width=2)
                        ))
                
                # Calculate overall trend (linear regression for learning rate)
                from scipy import stats
                x_all = plot_data['trial_order'].values
                y_all = plot_data[learn_metric].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_all, y_all)
                
                fig.update_layout(
                    title=f"Learning Curve: {learn_metric.upper()} Over Trials<br><sub>Trend: {slope:+.3f}/trial (p={p_value:.4f})</sub>",
                    xaxis_title="Trial Number (within participant)",
                    yaxis_title=learn_metric.upper(),
                    showlegend=True,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Learning statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    direction = "↓ Improving" if slope < 0 and learn_metric in ['rmse', 'mean_error'] else "↑ Worsening" if slope > 0 and learn_metric in ['rmse', 'mean_error'] else "→ Stable"
                    st.metric("Trend", direction, f"{slope:.4f}/trial")
                with col2:
                    st.metric("R²", f"{r_value**2:.3f}")
                with col3:
                    sig = "✓ Significant" if p_value < 0.05 else "✗ Not Significant"
                    st.metric("p-value", f"{p_value:.4f}", sig)
    
    # ==========================================================================
    # TAB 6: SPEED-ACCURACY TRADEOFF
    # ==========================================================================
    with tab6:
        st.subheader("⚡ Speed-Accuracy Tradeoff")
        
        st.markdown("""
        Examine the relationship between response speed and tracking accuracy.
        According to Fitts' Law, faster movements are typically less accurate.
        """)
        
        size_col = 'size' if 'size' in trial_metrics.columns else 'blob_size'
        
        # Define available speed-related metrics (use what's in the data)
        available_speed_metrics = []
        speed_metric_labels = {}
        
        if 'tracking_gain' in trial_metrics.columns:
            available_speed_metrics.append('tracking_gain')
            speed_metric_labels['tracking_gain'] = 'Tracking Gain (movement ratio)'
        if 'n_frames' in trial_metrics.columns:
            available_speed_metrics.append('n_frames')
            speed_metric_labels['n_frames'] = 'Trial Duration (frames)'
        if 'std_error' in trial_metrics.columns:
            available_speed_metrics.append('std_error')
            speed_metric_labels['std_error'] = 'Error Variability (std)'
        if 'initial_error' in trial_metrics.columns:
            available_speed_metrics.append('initial_error')
            speed_metric_labels['initial_error'] = 'Initial Error (reaction)'
        
        # Accuracy metrics
        available_accuracy_metrics = []
        if 'rmse' in trial_metrics.columns:
            available_accuracy_metrics.append('rmse')
        if 'mean_error' in trial_metrics.columns:
            available_accuracy_metrics.append('mean_error')
        if 'max_error' in trial_metrics.columns:
            available_accuracy_metrics.append('max_error')
        if 'median_error' in trial_metrics.columns:
            available_accuracy_metrics.append('median_error')
        
        col1, col2 = st.columns(2)
        with col1:
            speed_metric = st.selectbox(
                "X-Axis Metric",
                available_speed_metrics,
                format_func=lambda x: speed_metric_labels.get(x, x),
                help="Metric for the X-axis (speed/movement related)"
            )
        with col2:
            accuracy_metric = st.selectbox(
                "Y-Axis Metric (Accuracy)", 
                available_accuracy_metrics,
                help="Metric representing tracking accuracy (error)"
            )
        
        color_by = st.radio(
            "Color points by",
            ["Condition", "Blob Size", "Participant"],
            horizontal=True,
            key="sat_color"
        )
        
        if st.button("Generate Speed-Accuracy Plot", key="sat_btn"):
            import plotly.express as px
            import plotly.graph_objects as go
            from scipy import stats
            import numpy as np
            
            plot_data = trial_metrics.copy()
            
            # Check columns exist
            if speed_metric not in plot_data.columns:
                st.error(f"Speed metric '{speed_metric}' not found. Available: {list(plot_data.columns)}")
            elif accuracy_metric not in plot_data.columns:
                st.error(f"Accuracy metric '{accuracy_metric}' not found.")
            else:
                # Remove NaN values
                plot_data = plot_data.dropna(subset=[speed_metric, accuracy_metric])
                
                # Set color column
                if color_by == "Condition":
                    color_col = 'condition'
                elif color_by == "Blob Size":
                    color_col = size_col
                    plot_data[size_col] = plot_data[size_col].astype(str) + ' arcmin'
                else:
                    color_col = 'participant_id'
                    plot_data['participant_id'] = 'P' + plot_data['participant_id'].astype(str)
                
                # Create scatter plot
                x_label = speed_metric_labels.get(speed_metric, speed_metric).split(' (')[0]
                fig = px.scatter(
                    plot_data,
                    x=speed_metric,
                    y=accuracy_metric,
                    color=color_col,
                    opacity=0.6,
                    hover_data=['filename', 'condition', size_col] if size_col in plot_data.columns else ['filename', 'condition'],
                    title=f"Relationship: {x_label} vs {accuracy_metric.upper()}"
                )
                
                # Add regression line for all data
                x = plot_data[speed_metric].values
                y = plot_data[accuracy_metric].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                x_line = np.array([x.min(), x.max()])
                y_line = slope * x_line + intercept
                
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode='lines',
                    name=f'Regression (r={r_value:.3f})',
                    line=dict(color='black', dash='dash', width=2)
                ))
                
                fig.update_layout(
                    xaxis_title=speed_metric.replace('_', ' ').title(),
                    yaxis_title=accuracy_metric.upper(),
                    height=550,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation statistics
                st.markdown("#### Correlation Analysis")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pearson r", f"{r_value:.3f}")
                with col2:
                    st.metric("R²", f"{r_value**2:.3f}")
                with col3:
                    st.metric("p-value", f"{p_value:.4f}")
                with col4:
                    # Smart interpretation based on metrics
                    if abs(r_value) < 0.2:
                        relationship = "No relationship"
                    elif r_value > 0:
                        relationship = "Positive correlation"
                    else:
                        relationship = "Negative correlation"
                    st.metric("Interpretation", relationship)
                
                # Contextual insights
                st.markdown("#### 💡 Insights")
                if speed_metric == 'tracking_gain':
                    if r_value > 0.3:
                        st.info("📊 Higher tracking gain (more movement) is associated with **larger errors**. This may indicate overtracking leads to inaccuracy.")
                    elif r_value < -0.3:
                        st.info("📊 Higher tracking gain is associated with **smaller errors**. Participants who move more closely match the target.")
                elif speed_metric == 'initial_error':
                    if r_value > 0.3:
                        st.info("📊 Trials with larger initial errors tend to have **worse overall performance**. Early reaction quality predicts trial success.")
                elif speed_metric == 'std_error':
                    if r_value > 0.3:
                        st.info("📊 More variable tracking (higher std) correlates with **larger errors**. Consistent tracking leads to better accuracy.")
    
    # ==========================================================================
    # TAB 7: PURSUIT GAIN ANALYSIS
    # ==========================================================================
    with tab7:
        st.subheader("🎯 Pursuit Gain Analysis")
        
        st.markdown("""
        **Pursuit Gain** measures how well mouse velocity matches target velocity.
        - **Gain = 1.0**: Perfect velocity matching
        - **Gain < 1.0**: Undertracking (sluggish response)
        - **Gain > 1.0**: Overtracking (exaggerated movements)
        """)
        
        size_col = 'size' if 'size' in trial_metrics.columns else 'blob_size'
        
        if 'tracking_gain' not in trial_metrics.columns:
            st.warning("Tracking gain not computed. Please ensure analysis includes gain calculation.")
        else:
            # Plot type selection
            plot_type = st.selectbox(
                "Visualization Type",
                ["Distribution by Condition", "Distribution by Size", "Gain vs RMSE", "Individual Trials"],
                key="gain_plot_type"
            )
            
            if st.button("Generate Gain Analysis", key="gain_btn"):
                import plotly.express as px
                import plotly.graph_objects as go
                import numpy as np
                
                plot_data = trial_metrics.dropna(subset=['tracking_gain'])
                
                if plot_type == "Distribution by Condition":
                    fig = go.Figure()
                    
                    for cond in plot_data['condition'].unique():
                        cond_data = plot_data[plot_data['condition'] == cond]['tracking_gain']
                        fig.add_trace(go.Violin(
                            y=cond_data,
                            name=cond.capitalize(),
                            box_visible=True,
                            meanline_visible=True
                        ))
                    
                    # Perfect gain line
                    fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                                  annotation_text="Perfect Gain (1.0)")
                    
                    fig.update_layout(
                        title="Pursuit Gain Distribution by Condition",
                        yaxis_title="Tracking Gain",
                        showlegend=True,
                        height=500
                    )
                    
                elif plot_type == "Distribution by Size":
                    fig = go.Figure()
                    
                    for size in sorted(plot_data[size_col].unique()):
                        size_data = plot_data[plot_data[size_col] == size]['tracking_gain']
                        fig.add_trace(go.Violin(
                            y=size_data,
                            name=f'{size} arcmin',
                            box_visible=True,
                            meanline_visible=True
                        ))
                    
                    fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                                  annotation_text="Perfect Gain")
                    
                    fig.update_layout(
                        title="Pursuit Gain Distribution by Blob Size",
                        yaxis_title="Tracking Gain",
                        height=500
                    )
                    
                elif plot_type == "Gain vs RMSE":
                    fig = px.scatter(
                        plot_data,
                        x='tracking_gain',
                        y='rmse',
                        color='condition',
                        symbol=size_col,
                        opacity=0.6,
                        title="Relationship: Tracking Gain vs RMSE"
                    )
                    
                    # Optimal gain line
                    fig.add_vline(x=1.0, line_dash="dash", line_color="green",
                                  annotation_text="Optimal Gain")
                    
                    fig.update_layout(
                        xaxis_title="Tracking Gain",
                        yaxis_title="RMSE (pixels)",
                        height=500
                    )
                    
                else:  # Individual Trials
                    # Show gain for each trial as a bar chart
                    plot_data_sorted = plot_data.sort_values(['participant_id', 'filename'])
                    # Create short trial label from filename
                    plot_data_sorted['trial_label'] = plot_data_sorted.apply(
                        lambda x: f"P{x['participant_id']}-{x['filename'].split('_')[-1].replace('.csv', '')[:10]}", axis=1
                    )
                    
                    # Color by deviation from 1.0
                    plot_data_sorted['gain_deviation'] = abs(plot_data_sorted['tracking_gain'] - 1.0)
                    
                    fig = px.bar(
                        plot_data_sorted.head(50),  # Limit to 50 trials for readability
                        x='trial_label',
                        y='tracking_gain',
                        color='condition',
                        title="Individual Trial Gains (first 50 trials)"
                    )
                    
                    fig.add_hline(y=1.0, line_dash="dash", line_color="green")
                    
                    fig.update_layout(
                        xaxis_title="Trial",
                        yaxis_title="Tracking Gain",
                        xaxis_tickangle=45,
                        height=500
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.markdown("#### Gain Statistics")
                
                # By condition
                gain_stats = plot_data.groupby('condition')['tracking_gain'].agg(['mean', 'std', 'median']).round(3)
                gain_stats.columns = ['Mean Gain', 'Std Dev', 'Median Gain']
                gain_stats['Deviation from 1.0'] = (gain_stats['Mean Gain'] - 1.0).abs().round(3)
                
                st.dataframe(gain_stats, use_container_width=True)
                
                # Overall interpretation
                overall_mean = plot_data['tracking_gain'].mean()
                if overall_mean < 0.9:
                    interp = "🔵 **Undertracking:** Participants generally move slower than the target"
                elif overall_mean > 1.1:
                    interp = "🔴 **Overtracking:** Participants generally move faster than the target"
                else:
                    interp = "🟢 **Good velocity matching:** Gain close to optimal (1.0)"
                
                st.markdown(f"**Overall Mean Gain:** {overall_mean:.3f}")
                st.markdown(interp)


# =============================================================================
# PAGE: STATES
# =============================================================================

def page_states():
    """Render state management page."""
    st.header("💾 Load/Save State")
    
    st.markdown("""
    **Quick Resume:** Load a previously saved analysis state to skip data loading and processing.
    States include computed metrics, cross-correlation results, and configuration.
    """)
    
    # Check for auto-navigation flag
    if st.session_state.get('_navigate_to_research'):
        del st.session_state['_navigate_to_research']
        st.info("✅ State loaded! Click **🔬 Research Questions** in the sidebar to view results.")
    
    # States directory
    if st.session_state.config and st.session_state.config.output_dir:
        states_path = Path(st.session_state.config.output_dir) / 'states'
    else:
        states_path = Path.cwd() / 'results' / 'states'
    
    states_path_input = st.text_input("States Directory", value=str(states_path))
    
    if Path(states_path_input).exists():
        states = list_saved_states(states_path_input)
        
        if states:
            st.subheader("📋 Saved States")
            
            # Quick load option for most recent state
            most_recent = states[0]  # Assuming sorted by date
            st.markdown("### ⚡ Quick Load (Most Recent)")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{most_recent.state_id[:30]}...**")
                st.caption(f"Created: {most_recent.created_at} | Size: {most_recent.file_size_mb:.2f} MB")
            with col2:
                if st.button("📥 Load State", type="primary", key="quick_load"):
                    with st.spinner("Loading state..."):
                        manager = StateManager(states_path_input)
                        loaded_state = manager.load_state(most_recent.state_id)
                        
                        st.session_state.trial_metrics = loaded_state.trial_metrics
                        st.session_state.xcorr_results = loaded_state.xcorr_results
                        st.session_state.stat_results = loaded_state.statistical_results
                        st.session_state.processed_data = loaded_state.processed_data
                        st.session_state.config = Config(**loaded_state.config) if loaded_state.config else None
                        st.session_state.analysis_complete = True
                        st.session_state.data_loaded = True  # Mark data as loaded
                        st.session_state['_navigate_to_research'] = True
                        
                        st.success(f"✓ Loaded state: {most_recent.state_id}")
                        st.rerun()
            with col3:
                if st.session_state.analysis_complete:
                    st.markdown("✅ Ready!")
            
            st.markdown("---")
            
            # All states
            st.markdown("### 📚 All Saved States")
            
            for state in states:
                with st.expander(f"📁 {state.state_id}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Created:** {state.created_at}")
                        st.markdown(f"**Size:** {state.file_size_mb:.2f} MB")
                    
                    with col2:
                        st.markdown("**Contains:**")
                        st.markdown(f"- Metrics: {'✓' if state.has_metrics else '✗'}")
                        st.markdown(f"- Cross-Corr: {'✓' if state.has_xcorr else '✗'}")
                        st.markdown(f"- Statistics: {'✓' if state.has_statistics else '✗'}")
                    
                    if st.button(f"Load {state.state_id[:20]}...", key=f"load_{state.state_id}"):
                        manager = StateManager(states_path_input)
                        loaded_state = manager.load_state(state.state_id)
                        
                        st.session_state.trial_metrics = loaded_state.trial_metrics
                        st.session_state.xcorr_results = loaded_state.xcorr_results
                        st.session_state.stat_results = loaded_state.statistical_results
                        st.session_state.processed_data = loaded_state.processed_data
                        st.session_state.config = Config(**loaded_state.config) if loaded_state.config else None
                        st.session_state.analysis_complete = True
                        
                        st.success(f"✓ Loaded state: {state.state_id}")
        else:
            st.info("No saved states found")
    else:
        st.warning(f"Directory does not exist: {states_path_input}")


# =============================================================================
# PAGE: RESEARCH QUESTIONS
# =============================================================================

def page_research_questions():
    """Render research questions analysis page."""
    st.header("🔬 Research Questions Analysis")
    
    st.markdown("""
    This page automatically answers the core research questions using 
    multi-metric evidence from your tracking data.
    """)
    
    # Check if we have data loaded
    if not st.session_state.get('data_loaded'):
        st.warning("⚠️ Please load data first (Load Data page)")
        st.info("Navigate to **Load Data** page to select your data folder.")
        return
    
    # Auto-run analysis if data is loaded but not analyzed
    if not st.session_state.get('analysis_complete'):
        st.info("📊 Data loaded but not analyzed. Running analysis automatically...")
        
        # Run the analysis pipeline
        with st.spinner("Running analysis pipeline..."):
            try:
                data = st.session_state.data.copy()
                config = st.session_state.get('config')
                
                # Create default config if none exists
                if config is None:
                    config = Config()
                    st.session_state.config = config
                
                # Step 1: Preprocessing
                preprocessor = Preprocessor(config)
                processed_data, preprocess_report = preprocessor.process(data)
                st.session_state.processed_data = processed_data
                
                # Step 2: Metrics
                metrics_calc = MetricsCalculator(config)
                trial_metrics = metrics_calc.compute_all_trials(processed_data)
                st.session_state.trial_metrics = trial_metrics
                
                # Step 3: Cross-correlation
                xcorr_analyzer = CrossCorrelationAnalyzer(config)
                xcorr_results = xcorr_analyzer.analyze_all_trials(processed_data)
                st.session_state.xcorr_results = xcorr_results
                
                # Step 4: Statistics
                stats_analyzer = StatisticalAnalyzer(config)
                stat_results = stats_analyzer.generate_results_summary(trial_metrics, metric='rmse')
                st.session_state.stat_results = stat_results
                
                st.session_state.analysis_complete = True
                st.success("✅ Analysis completed automatically!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during automatic analysis: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    trial_metrics = st.session_state.get('trial_metrics')
    xcorr_results = st.session_state.get('xcorr_results')
    
    if trial_metrics is None or trial_metrics.empty:
        st.error("No trial metrics available. Please check data loading.")
        return
    
    # Data Selection Section
    st.subheader("📊 Data Selection")
    
    # Get available options from data
    available_participants = sorted(trial_metrics['participant_id'].unique())
    available_conditions = sorted(trial_metrics['condition'].unique())
    size_col = 'size' if 'size' in trial_metrics.columns else 'blob_size'
    available_sizes = sorted(trial_metrics[size_col].unique())
    
    # Selection mode
    selection_mode = st.radio(
        "Analysis Scope",
        options=["All Data", "Filter by Criteria"],
        horizontal=True,
        help="Choose whether to analyze all data or filter by specific criteria"
    )
    
    # Initialize filtered data
    filtered_metrics = trial_metrics.copy()
    filtered_xcorr = xcorr_results.copy() if xcorr_results is not None else pd.DataFrame()
    
    if selection_mode == "Filter by Criteria":
        filter_cols = st.columns(3)
        
        with filter_cols[0]:
            participant_options = ["All Participants"] + [str(p) for p in available_participants]
            selected_participant = st.selectbox(
                "Participant",
                options=participant_options,
                help="Select a specific participant or all"
            )
        
        with filter_cols[1]:
            condition_options = ["All Conditions"] + list(available_conditions)
            selected_condition = st.selectbox(
                "Condition",
                options=condition_options,
                help="Filter by tracking condition (dynamic/static)"
            )
        
        with filter_cols[2]:
            selected_sizes = st.multiselect(
                "Blob Sizes (arcmin)",
                options=available_sizes,
                default=available_sizes,
                help="Select blob sizes to include in analysis"
            )
        
        # Apply filters (handle both string and numeric types)
        if selected_participant != "All Participants":
            # Convert participant_id column to string for reliable matching
            filtered_metrics['participant_id'] = filtered_metrics['participant_id'].astype(str)
            filtered_metrics = filtered_metrics[filtered_metrics['participant_id'] == selected_participant]
            
            if not filtered_xcorr.empty and 'participant_id' in filtered_xcorr.columns:
                filtered_xcorr['participant_id'] = filtered_xcorr['participant_id'].astype(str)
                filtered_xcorr = filtered_xcorr[filtered_xcorr['participant_id'] == selected_participant]
        
        if selected_condition != "All Conditions":
            filtered_metrics = filtered_metrics[filtered_metrics['condition'] == selected_condition]
            if not filtered_xcorr.empty and 'condition' in filtered_xcorr.columns:
                filtered_xcorr = filtered_xcorr[filtered_xcorr['condition'] == selected_condition]
        
        if selected_sizes:
            # Convert sizes to strings for reliable matching
            selected_sizes_str = [str(s) for s in selected_sizes]
            filtered_metrics[size_col] = filtered_metrics[size_col].astype(str)
            filtered_metrics = filtered_metrics[filtered_metrics[size_col].isin(selected_sizes_str)]
            
            xcorr_size_col = 'size' if 'size' in filtered_xcorr.columns else 'blob_size' if 'blob_size' in filtered_xcorr.columns else None
            if not filtered_xcorr.empty and xcorr_size_col:
                filtered_xcorr[xcorr_size_col] = filtered_xcorr[xcorr_size_col].astype(str)
                filtered_xcorr = filtered_xcorr[filtered_xcorr[xcorr_size_col].isin(selected_sizes_str)]
        
        # Show filter summary
        st.info(f"""
        **📋 Filter Summary:**
        - Participant: {selected_participant}
        - Condition: {selected_condition}
        - Blob Sizes: {', '.join(map(str, selected_sizes)) if selected_sizes else 'None selected'}
        """)
    
    # Show data count
    n_trials = len(filtered_metrics)
    n_participants = filtered_metrics['participant_id'].nunique()
    n_sizes = filtered_metrics[size_col].nunique()
    
    st.markdown(f"**Selected Data:** {n_trials} trials from {n_participants} participant(s) across {n_sizes} blob size(s)")
    
    if n_trials == 0:
        st.warning("⚠️ No data matches the selected criteria. Please adjust your filters.")
        return
    
    st.markdown("---")
    
    # Configuration
    st.subheader("⚙️ Analysis Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rmse_threshold = st.slider(
            "RMSE Threshold (px)",
            min_value=20.0,
            max_value=150.0,
            value=50.0,
            step=5.0,
            help="Maximum RMSE for 'good' tracking. Lower = stricter."
        )
    
    with col2:
        corr_threshold = st.slider(
            "Correlation Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Minimum correlation for 'good' velocity tracking."
        )
    
    with col3:
        lag_threshold = st.slider(
            "Lag Threshold (ms)",
            min_value=100.0,
            max_value=1000.0,
            value=500.0,
            step=50.0,
            help="Maximum acceptable response lag."
        )
    
    alpha = st.selectbox(
        "Significance Level (α)",
        options=[0.01, 0.05, 0.10],
        index=1,
        help="Alpha level for statistical tests."
    )
    
    st.markdown("---")
    
    # Run analysis
    if st.button("🔬 Analyze Research Questions", type="primary"):
        with st.spinner("Analyzing discrimination ability..."):
            analyzer = StatisticalAnalyzer()
            
            # Use filtered data for analysis
            xcorr_df = filtered_xcorr if not filtered_xcorr.empty else pd.DataFrame()
            
            results = analyzer.answer_research_questions(
                trial_metrics=filtered_metrics,
                xcorr_results=xcorr_df,
                rmse_threshold=rmse_threshold,
                correlation_threshold=corr_threshold,
                lag_threshold_ms=lag_threshold,
                alpha=alpha
            )
            
            # Store results
            st.session_state['research_results'] = results
    
    # Display results if available
    if 'research_results' in st.session_state:
        results = st.session_state['research_results']
        
        st.markdown("---")
        st.subheader("📋 Research Questions Answers")
        
        # Display each question
        for i, (size, analysis) in enumerate(sorted(results['questions'].items()), 1):
            with st.expander(f"Q{i}: Can participants discriminate the {size}-arcmin blob?", expanded=True):
                
                # Main answer
                st.markdown(analysis['answer'])
                
                # Metrics summary
                if analysis['metrics']:
                    st.markdown("**Metrics Summary:**")
                    
                    metric_cols = st.columns(len(analysis['metrics']))
                    for idx, (metric_name, metric_data) in enumerate(analysis['metrics'].items()):
                        with metric_cols[idx]:
                            if metric_name == 'rmse':
                                st.metric(
                                    "RMSE",
                                    f"{metric_data['mean']:.1f} px",
                                    delta=f"{metric_data['mean'] - metric_data['threshold']:.1f} vs threshold",
                                    delta_color="inverse"
                                )
                            elif metric_name == 'correlation':
                                st.metric(
                                    "Max Correlation",
                                    f"{metric_data['mean']:.3f}",
                                    delta=f"{metric_data['mean'] - metric_data['threshold']:.3f} vs threshold",
                                    delta_color="normal"
                                )
                            elif metric_name == 'lag':
                                st.metric(
                                    "Response Lag",
                                    f"{metric_data['mean']:.0f} ms",
                                    delta=f"{metric_data['mean'] - metric_data['threshold']:.0f} vs threshold",
                                    delta_color="inverse"
                                )
                
                # Evidence
                if analysis['evidence']:
                    st.markdown("**Evidence:**")
                    for ev in analysis['evidence']:
                        st.markdown(f"- {ev}")
                
                # Statistical tests
                if analysis['statistical_tests']:
                    with st.expander("📊 Statistical Tests"):
                        for test_name, test_data in analysis['statistical_tests'].items():
                            st.markdown(f"**{test_name}**")
                            st.json(test_data)
                
                # Confidence indicator
                confidence = analysis.get('confidence', 'unknown')
                if confidence == 'high':
                    st.success(f"🎯 High confidence answer (evidence ratio: {analysis.get('evidence_ratio', 0):.1%})")
                elif confidence == 'moderate':
                    st.warning(f"⚠️ Moderate confidence answer (evidence ratio: {analysis.get('evidence_ratio', 0):.1%})")
                else:
                    st.info(f"ℹ️ Low confidence - insufficient data")
        
        # Size comparison
        st.markdown("---")
        st.subheader("📊 Cross-Size Comparison")
        
        comparison = results['comparison']
        
        if comparison.get('rmse_anova'):
            anova = comparison['rmse_anova']
            status = "✅ Significant" if anova['significant'] else "❌ Not significant"
            st.markdown(f"**RMSE ANOVA:** F={anova['f_statistic']:.2f}, p={anova['p_value']:.4f} ({status})")
        
        if comparison.get('correlation_anova'):
            anova = comparison['correlation_anova']
            status = "✅ Significant" if anova['significant'] else "❌ Not significant"
            st.markdown(f"**Correlation ANOVA:** F={anova['f_statistic']:.2f}, p={anova['p_value']:.4f} ({status})")
        
        if comparison.get('size_ranking'):
            st.markdown("**Performance Ranking (by RMSE):**")
            for rank, (size, rmse) in enumerate(comparison['size_ranking'], 1):
                st.markdown(f"{rank}. **{size} arcmin** - Mean RMSE: {rmse:.1f} px")
        
        st.markdown(f"*{comparison.get('interpretation', '')}*")
        
        # Full summary
        st.markdown("---")
        st.subheader("📝 Full Summary")
        st.markdown(results['summary'])
        
        # Export
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        import json
        
        # Prepare exportable results (convert non-serializable types)
        def make_serializable(obj):
            """Convert numpy/pandas types to JSON-serializable Python types."""
            if isinstance(obj, dict):
                # Convert both keys and values
                return {make_serializable(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return [make_serializable(item) for item in obj.tolist()]
            elif pd.isna(obj):
                return None
            elif hasattr(obj, 'item'):  # Handle any remaining numpy scalar types
                return obj.item()
            else:
                return obj
        
        export_data = make_serializable(results)
        json_str = json.dumps(export_data, indent=2, default=str)
        
        st.download_button(
            "📥 Download Results (JSON)",
            json_str,
            file_name="research_questions_results.json",
            mime="application/json"
        )
        
        st.download_button(
            "📥 Download Summary (Markdown)",
            results['summary'],
            file_name="research_questions_summary.md",
            mime="text/markdown"
        )


# =============================================================================
# PAGE: HELP
# =============================================================================

def page_help():
    """Render help page."""
    st.header("❓ Help & Documentation")
    
    with st.expander("📖 Data Format"):
        st.markdown("""
        ### Expected CSV Format
        
        Your CSV files should contain the following columns:
        - `Frame`: Frame number (0-indexed)
        - `Target_X`: X coordinate of target (pixels)
        - `Target_Y`: Y coordinate of target (pixels)
        - `Mouse_X`: X coordinate of mouse/response (pixels)
        - `Mouse_Y`: Y coordinate of mouse/response (pixels)
        
        ### File Naming Convention
        
        Files should follow this pattern:
        ```
        Participant_XXXX_..._XXarcmin_..._dynamic.csv
        Participant_XXXX_..._XXarcmin_..._static.csv
        ```
        
        Where:
        - `XXXX`: Participant ID
        - `XX`: SD size in arcmin (21, 31, or 34)
        - `dynamic/static`: Condition
        """)
    
    with st.expander("⚙️ Parameter Guide"):
        st.markdown("""
        ### Velocity Calculation
        
        | Method | Best For | Trade-off |
        |--------|----------|-----------|
        | Difference | Quick analysis | More noise |
        | Savgol | Publication | Computational cost |
        
        ### Outlier Removal
        
        | Method | Best For | Trade-off |
        |--------|----------|-----------|
        | IQR | Non-normal data | May remove too many |
        | Z-score | Normal data | Sensitive to outliers |
        | MAD | Robust analysis | More conservative |
        """)
    
    with st.expander("📊 Understanding Results"):
        st.markdown("""
        ### RMSE (Root Mean Square Error)
        
        Measures overall tracking accuracy. Lower = better tracking.
        
        | RMSE Range | Interpretation |
        |------------|----------------|
        | < 20 px | Excellent tracking |
        | 20-50 px | Good tracking |
        | 50-100 px | Moderate tracking |
        | > 100 px | Poor tracking |
        
        ---
        
        ### Cross-Correlation Analysis
        
        #### Velocity Conversion
        
        Cross-correlation is computed on **velocity signals** (rate of position change), not raw positions.
        This captures dynamic tracking behavior rather than static position offsets.
        
        **Formula:** v(t) = x(t) - x(t-1) (difference method) or Savitzky-Golay smoothed derivative.
        
        #### Optimal Lag Interpretation
        
        | Lag Value | Interpretation | Typical Range |
        |-----------|----------------|---------------|
        | **Positive** | REACTIVE: Mouse follows target | 50-200ms |
        | **Negative** | PREDICTIVE: Mouse anticipates target | -50 to -150ms |
        | **Zero** | SYNCHRONOUS: Perfect timing | ±20ms |
        
        **What good tracking looks like:**
        - Small absolute lag (close to 0)
        - High correlation (> 0.5)
        - Consistent across trials
        
        #### Correlation Strength (Configurable)
        
        | Correlation | Default Interpretation |
        |-------------|------------------------|
        | r < 0.3 | Very weak tracking |
        | 0.3 ≤ r < 0.6 | Weak to moderate |
        | 0.6 ≤ r < 0.7 | Moderate |
        | r ≥ 0.7 | Strong tracking |
        
        *You can adjust these thresholds in the Cross-Correlation tab.*
        
        ---
        
        ### Effect Sizes
        
        | Cohen's d | Interpretation |
        |-----------|----------------|
        | |d| < 0.2 | Negligible |
        | 0.2 ≤ |d| < 0.5 | Small effect |
        | 0.5 ≤ |d| < 0.8 | Medium effect |
        | |d| ≥ 0.8 | Large effect |
        
        ---
        
        ### Research Questions Answered
        
        | Question | Where to Find Answer |
        |----------|---------------------|
        | Is tracking predictive or reactive? | Cross-Correlation → Individual Trial Results |
        | Does auditory feedback help? | Cross-Correlation → Dynamic vs Static Comparison |
        | How strong is velocity tracking? | Cross-Correlation → Correlation Strength |
        | Do blob sizes differ? | Results → Statistics tab (ANOVA) |
        """)
    
    with st.expander("📉 Data Analysis & Averaging"):
        st.markdown("""
        ### Data Subset Selection
        
        Select any combination of:
        - **Participants**: Filter by specific participant IDs
        - **Blob Sizes**: Filter by SD size (21, 31, or 34 arcmin)
        - **Conditions**: Filter by dynamic (auditory feedback) or static (no feedback)
        
        ### Metric Averages Tab
        
        Compute averaged metrics grouped by your selected factors:
        - Group by size, condition, participant, or any combination
        - Shows mean, std, min, max, count for RMSE and other metrics
        
        ### Position Averages Tab
        
        Frame-by-frame averaging of position data:
        - Averages Target_X, Target_Y, Mouse_X, Mouse_Y across selected trials
        - Shows ±1 SD bands for variability visualization
        - Useful for seeing "typical" tracking behavior for a condition
        
        ### Export Tab
        
        Download your filtered data:
        - Filtered trial metrics (CSV)
        - Filtered raw position data (CSV)
        """)
    
    with st.expander("🎬 Animation Guide"):
        st.markdown("""
        ### Animation Types
        
        | Type | Description | Best For |
        |------|-------------|----------|
        | **2D Spatial** | Shows Target and Mouse in X-Y plane | Visualizing actual movement patterns |
        | **Time Series** | Shows X and Y positions over frames | Analyzing temporal tracking behavior |
        
        ### Animation Modes
        
        | Mode | Description | Pros | Cons |
        |------|-------------|------|------|
        | **🚀 Fast (Frame Skip)** | Shows every Nth frame | Fast rendering, smooth playback | Less temporal detail |
        | **🎬 Full (All Frames)** | Shows all frames | Maximum detail | Slower, may lag on long trials |
        
        ### Frame Skip Settings
        
        | Frame Skip | Speed | Detail | Recommended For |
        |------------|-------|--------|-----------------|
        | 1 | Slowest | Maximum | Short trials, detailed analysis |
        | 5 | Moderate | Good | Most use cases |
        | 10 | Fast | Moderate | Quick overview |
        | 20 | Fastest | Low | Very long trials |
        
        ### Animation Options
        
        - **Show Trail**: Displays path history behind markers
        - **Trail Length**: How many frames of history to show (5-50)
        - **Show Error Line**: Dashed line connecting Target and Mouse
        
        ### Tips
        
        1. Start with **Fast mode** (frame skip 5-10) to preview
        2. Use **Full mode** only for detailed analysis or short trials
        3. Enable **trail** to see movement patterns
        4. The **error line** helps visualize tracking lag
        """)
    
    with st.expander("📐 Formulas & Calculations", expanded=False):
        st.markdown(r"""
        ## Core Metrics
        
        ### Euclidean Error (Frame-by-Frame)
        
        $$\text{Error}_i = \sqrt{(\text{Target}_x - \text{Mouse}_x)^2 + (\text{Target}_y - \text{Mouse}_y)^2}$$
        
        **What it measures:** Distance between target and mouse position at each frame (in pixels).
        
        ---
        
        ### Root Mean Square Error (RMSE)
        
        $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} \text{error}_i^2}$$
        
        **What it measures:** Overall tracking accuracy for a trial. RMSE weights larger errors more heavily than mean error, making it sensitive to occasional large deviations.
        
        | RMSE Range | Interpretation |
        |------------|----------------|
        | < 20 px | Excellent |
        | 20-50 px | Good |
        | 50-100 px | Moderate |
        | > 100 px | Poor |
        
        ---
        
        ### Tracking Gain
        
        $$\text{Gain} = \frac{\text{Total Mouse Movement}}{\text{Total Target Movement}}$$
        
        Where: $\text{Movement} = \sum_{i=1}^{n-1} \sqrt{(\Delta x_i)^2 + (\Delta y_i)^2}$
        
        | Gain Value | Interpretation |
        |------------|----------------|
        | = 1.0 | Perfect scaling |
        | < 1.0 | Undertracking (sluggish response) |
        | > 1.0 | Overtracking (exaggerated response) |
        
        ---
        
        ## Velocity Calculations
        
        ### Simple Difference Method
        
        $$v_i = \frac{x_i - x_{i-1}}{\Delta t}$$
        
        Where $\Delta t = 0.02$ seconds (20 ms per frame at 50 fps)
        
        **Pros:** Simple, fast, preserves temporal detail  
        **Cons:** Sensitive to noise
        
        ### Savitzky-Golay Method
        
        Fits a polynomial to a sliding window and computes the derivative of the fitted curve.
        
        **Pros:** Reduces noise, preserves signal shape  
        **Cons:** May smooth rapid movements
        
        ---
        
        ## Cross-Correlation Analysis
        
        ### Normalized Cross-Correlation
        
        $$r_{xy}(\tau) = \frac{\sum_{i} (x_i - \bar{x})(y_{i+\tau} - \bar{y})}{n \cdot \sigma_x \cdot \sigma_y}$$
        
        Where:
        - $\tau$ = lag (in frames)
        - $\bar{x}, \bar{y}$ = signal means
        - $\sigma_x, \sigma_y$ = signal standard deviations
        
        **Range:** -1 to +1 (normalized)
        
        ---
        
        ### Optimal Lag
        
        $$\tau_{\text{optimal}} = \arg\max_{\tau} r_{xy}(\tau)$$
        
        **Conversion:** $\text{Lag}_{ms} = \text{Lag}_{frames} \times 20 \text{ ms}$
        
        | Lag | Interpretation |
        |-----|----------------|
        | Positive | Reactive (mouse follows target) |
        | Negative | Predictive (mouse anticipates target) |
        | Zero | Synchronous tracking |
        
        ---
        
        ## Statistical Tests
        
        ### Cohen's d (Effect Size)
        
        $$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$
        
        Where: $s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$
        
        | \|d\| Value | Interpretation |
        |-----------|----------------|
        | < 0.2 | Negligible |
        | 0.2 - 0.5 | Small |
        | 0.5 - 0.8 | Medium |
        | ≥ 0.8 | Large |
        
        ---
        
        ### Paired t-test
        
        $$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$
        
        Where $\bar{d}$ = mean difference, $s_d$ = std of differences, df = n-1
        
        **Used for:** Comparing dynamic vs static conditions (same participants)
        
        ---
        
        ### Independent t-test
        
        $$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_p^2(\frac{1}{n_1} + \frac{1}{n_2})}}$$
        
        **Used for:** Comparing different groups
        
        ---
        
        ### One-Way ANOVA
        
        $$F = \frac{MS_{\text{between}}}{MS_{\text{within}}}$$
        
        **Used for:** Comparing 3+ blob sizes (21, 31, 34 arcmin)
        
        ---
        
        ### One-Sample t-test (vs Threshold)
        
        $$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$
        
        **Used for:** Testing if metrics significantly differ from discrimination thresholds
        
        ---
        
        ### Partial Eta-Squared (Effect Size for ANOVA)
        
        $$\eta_p^2 = \frac{SS_{\text{effect}}}{SS_{\text{effect}} + SS_{\text{error}}}$$
        
        | η²_p Value | Interpretation |
        |------------|----------------|
        | < 0.01 | Negligible |
        | 0.01 - 0.06 | Small |
        | 0.06 - 0.14 | Medium |
        | ≥ 0.14 | Large |
        
        ---
        
        ## Outlier Detection
        
        ### IQR Method (Default)
        
        $$\text{Outlier if: } x < Q_1 - k \times IQR \text{ or } x > Q_3 + k \times IQR$$
        
        Where $IQR = Q_3 - Q_1$, $k$ = threshold (default: 2.5)
        
        ### Z-Score Method
        
        $$z = \frac{x - \bar{x}}{\sigma}$$
        
        Outlier if $|z| > \text{threshold}$
        
        ### MAD Method (Robust)
        
        $$MAD = \text{median}(|X - \text{median}(X)|)$$
        
        $$\text{Modified } z = 0.6745 \times \frac{x - \text{median}(X)}{MAD}$$
        
        ---
        
        ## Multiple Comparison Corrections
        
        ### Bonferroni
        
        $$p_{\text{adjusted}} = \min(p \times n_{\text{comparisons}}, 1.0)$$
        
        **Most conservative** - controls familywise error rate strictly
        
        ### Holm-Bonferroni (Step-down)
        
        $$p_{\text{adjusted},i} = \min(p_i \times (n - i + 1), 1.0)$$
        
        **More powerful** than Bonferroni while still controlling error
        
        ### FDR (Benjamini-Hochberg)
        
        $$p_{\text{adjusted},i} = \min\left(\frac{p_i \times n}{i}, p_{\text{adjusted},i+1}\right)$$
        
        **Most powerful** - controls expected false discovery proportion
        
        ---
        
        ## Constants & Defaults
        
        | Constant | Value | Description |
        |----------|-------|-------------|
        | Frame Rate | 50 fps | Data collection rate |
        | Frame Duration | 20 ms | Time per frame |
        | Trial Length | 999 frames | ~20 seconds per trial |
        | Screen Size | 1920 × 980 px | Display dimensions |
        | Blob Sizes | 21, 31, 34 | SD in arcmin |
        | RMSE Threshold | 50 px | Default discrimination threshold |
        | Correlation Threshold | 0.5 | Default for "good" tracking |
        | Lag Threshold | 500 ms | Default max acceptable lag |
        | Alpha | 0.05 | Default significance level |
        
        ---
        
        ## Psychophysics Visualizations
        
        ### Psychometric Function
        
        The psychometric function models the probability of "correct" (good tracking) as a function of stimulus intensity (blob size):
        
        $$\Psi(x; \alpha, \beta) = \Phi\left(\frac{x - \alpha}{\beta}\right)$$
        
        Where:
        - $\Phi$ = cumulative Gaussian (normal CDF)
        - $\alpha$ = threshold (50% point) - the stimulus level at which performance is at chance
        - $\beta$ = slope (sensitivity) - how quickly performance changes with stimulus
        - $x$ = stimulus intensity (blob size in arcmin)
        
        **Standard Error of Proportion:**
        
        $$SE_p = \sqrt{\frac{p(1-p)}{n}}$$
        
        | Threshold | Interpretation |
        |-----------|----------------|
        | 50% | Just noticeable difference (JND) |
        | 75% | Standard discrimination threshold |
        | 95% | Near-ceiling performance |
        
        ---
        
        ### Learning Curve Analysis
        
        **Linear Trend (Practice Effect):**
        
        $$y = \beta_0 + \beta_1 \cdot \text{trial}$$
        
        Where:
        - $\beta_1$ = learning rate (change per trial)
        - $\beta_1 < 0$ for error metrics indicates improvement
        
        **Power Law of Practice:**
        
        $$T_n = T_1 \cdot n^{-\alpha}$$
        
        Where:
        - $T_n$ = performance on trial $n$
        - $T_1$ = initial performance
        - $\alpha$ = learning rate exponent
        
        **Moving Average Smoothing:**
        
        $$\bar{y}_i = \frac{1}{k}\sum_{j=i-\lfloor k/2 \rfloor}^{i+\lfloor k/2 \rfloor} y_j$$
        
        Where $k$ = window size
        
        ---
        
        ### Speed-Accuracy Tradeoff (Fitts' Law)
        
        **Fitts' Law (Classic):**
        
        $$MT = a + b \cdot \log_2\left(\frac{2D}{W}\right)$$
        
        Where:
        - $MT$ = movement time
        - $D$ = distance to target
        - $W$ = target width
        - $a, b$ = empirical constants
        
        **In tracking context:**
        
        $$\text{RMSE} = f(\text{velocity})$$
        
        Positive correlation between speed and error indicates a tradeoff.
        
        **Correlation Coefficient:**
        
        $$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$
        
        | |r| Value | Interpretation |
        |-----------|----------------|
        | 0 - 0.3 | Weak tradeoff |
        | 0.3 - 0.6 | Moderate tradeoff |
        | > 0.6 | Strong tradeoff |
        
        ---
        
        ### Pursuit Gain
        
        **Velocity Gain:**
        
        $$\text{Gain} = \frac{v_{\text{mouse}}}{v_{\text{target}}} = \frac{\sum|\Delta \text{Mouse}|}{\sum|\Delta \text{Target}|}$$
        
        **Position Gain (Alternative):**
        
        $$\text{Gain}_{\text{pos}} = \frac{\text{std}(\text{Mouse Position})}{\text{std}(\text{Target Position})}$$
        
        | Gain Value | Interpretation |
        |------------|----------------|
        | 1.0 | Perfect pursuit - mouse matches target velocity |
        | < 1.0 | Undertracking - participant moves slower than target |
        | > 1.0 | Overtracking - participant moves faster than target |
        | 0.8 - 1.2 | Acceptable range for smooth pursuit |
        """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    
    page = render_sidebar()
    
    if page == "home":
        page_home()
    elif page == "load_data":
        page_load_data()
    elif page == "configure":
        page_configure()
    elif page == "run_analysis":
        page_run_analysis()
    elif page == "results":
        page_results()
    elif page == "data_analysis":
        page_data_analysis()
    elif page == "visualizations":
        page_visualizations()
    elif page == "research_questions":
        page_research_questions()
    elif page == "states":
        page_states()
    elif page == "help":
        page_help()


if __name__ == "__main__":
    main()
