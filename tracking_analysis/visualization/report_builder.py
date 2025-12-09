"""
Report Builder Module
=====================

Generates analysis reports in various formats.

Supported formats:
- Markdown
- HTML (via Jinja2 templates)
- PDF (requires additional dependencies)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Try importing template engines
try:
    from jinja2 import Environment, FileSystemLoader, BaseLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "Tracking Analysis Report"
    author: str = ""
    include_plots: bool = True
    include_raw_data: bool = False
    include_statistics: bool = True
    include_methodology: bool = True
    timestamp: bool = True


class ReportBuilder:
    """
    Builds analysis reports from results.
    
    Attributes:
        config: Report configuration
        output_dir: Directory for saving reports
        
    Example:
        >>> builder = ReportBuilder(output_dir='./reports')
        >>> 
        >>> # Generate markdown report
        >>> builder.generate_markdown_report(
        ...     trial_metrics=metrics_df,
        ...     xcorr_results=xcorr_df,
        ...     stat_results=stats_dict,
        ...     output_path='report.md'
        ... )
    """
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[ReportConfig] = None
    ):
        """
        Initialize ReportBuilder.
        
        Args:
            output_dir: Directory for saving reports
            config: Report configuration
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.config = config or ReportConfig()
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown_report(
        self,
        trial_metrics: pd.DataFrame,
        xcorr_results: Optional[pd.DataFrame] = None,
        stat_results: Optional[Dict[str, Any]] = None,
        analysis_config: Optional[Dict[str, Any]] = None,
        output_path: Optional[Union[str, Path]] = None,
        plot_dir: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate Markdown analysis report.
        
        Args:
            trial_metrics: DataFrame with trial-level metrics
            xcorr_results: DataFrame with cross-correlation results
            stat_results: Dictionary with statistical results
            analysis_config: Configuration used for analysis
            output_path: Output file path
            plot_dir: Directory containing plot images
            
        Returns:
            Markdown string
        """
        sections = []
        
        # Header
        sections.append(self._generate_header())
        
        # Executive Summary
        sections.append(self._generate_summary(trial_metrics, stat_results))
        
        # Methodology (if enabled)
        if self.config.include_methodology:
            sections.append(self._generate_methodology(analysis_config))
        
        # Results
        sections.append(self._generate_results_section(trial_metrics, xcorr_results, stat_results))
        
        # Statistical Analysis
        if self.config.include_statistics and stat_results:
            sections.append(self._generate_statistics_section(stat_results))
        
        # Figures (if plots directory provided)
        if self.config.include_plots and plot_dir:
            sections.append(self._generate_figures_section(plot_dir))
        
        # Conclusions
        sections.append(self._generate_conclusions(trial_metrics, stat_results))
        
        # Appendix (raw data tables)
        if self.config.include_raw_data:
            sections.append(self._generate_appendix(trial_metrics))
        
        report = '\n\n'.join(sections)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(report, encoding='utf-8')
        elif self.output_dir:
            output_path = self.output_dir / 'report.md'
            output_path.write_text(report, encoding='utf-8')
        
        return report
    
    def _generate_header(self) -> str:
        """Generate report header."""
        lines = [
            f"# {self.config.title}",
            ""
        ]
        
        if self.config.author:
            lines.append(f"**Author:** {self.config.author}")
        
        if self.config.timestamp:
            lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        lines.append("")
        lines.append("---")
        
        return '\n'.join(lines)
    
    def _generate_summary(
        self,
        trial_metrics: pd.DataFrame,
        stat_results: Optional[Dict[str, Any]]
    ) -> str:
        """Generate executive summary."""
        lines = [
            "## Executive Summary",
            ""
        ]
        
        # Dataset overview
        n_participants = trial_metrics['participant_id'].nunique()
        n_trials = len(trial_metrics)
        
        lines.append(f"This report presents the analysis of tracking performance data from "
                    f"**{n_participants} participants** across **{n_trials} trials**.")
        lines.append("")
        
        # Key findings
        lines.append("### Key Findings")
        lines.append("")
        
        # RMSE by size
        size_rmse = trial_metrics.groupby('size_pixels')['rmse'].mean()
        lines.append(f"- **RMSE by Blob Size:**")
        for size in sorted(size_rmse.index):
            lines.append(f"  - {size} arcmin: {size_rmse[size]:.2f} pixels")
        
        # RMSE by condition
        cond_rmse = trial_metrics.groupby('condition')['rmse'].mean()
        lines.append(f"- **RMSE by Condition:**")
        lines.append(f"  - Auditory Feedback (dynamic): {cond_rmse.get('dynamic', 'N/A'):.2f} pixels")
        lines.append(f"  - No Feedback (static): {cond_rmse.get('static', 'N/A'):.2f} pixels")
        
        # Statistical significance if available
        if stat_results:
            lines.append("")
            if 'size_anova' in stat_results:
                p_val = stat_results['size_anova'].get('p_value', 'N/A')
                lines.append(f"- **Size Effect:** {'Significant' if p_val < 0.05 else 'Not significant'} (p = {p_val:.4f})")
            if 'condition_ttest' in stat_results:
                p_val = stat_results['condition_ttest'].get('p_value', 'N/A')
                lines.append(f"- **Condition Effect:** {'Significant' if p_val < 0.05 else 'Not significant'} (p = {p_val:.4f})")
        
        return '\n'.join(lines)
    
    def _generate_methodology(
        self,
        analysis_config: Optional[Dict[str, Any]]
    ) -> str:
        """Generate methodology section."""
        lines = [
            "## Methodology",
            ""
        ]
        
        if analysis_config:
            lines.append("### Analysis Parameters")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            
            key_params = [
                ('velocity_method', 'Velocity Calculation'),
                ('outlier_method', 'Outlier Method'),
                ('outlier_threshold', 'Outlier Threshold'),
                ('normalize_xcorr', 'Normalize Cross-Correlation'),
                ('max_lag_frames', 'Max Lag (frames)'),
                ('frame_rate', 'Frame Rate (Hz)')
            ]
            
            for key, label in key_params:
                if key in analysis_config:
                    lines.append(f"| {label} | {analysis_config[key]} |")
        
        lines.append("")
        lines.append("### Measures")
        lines.append("")
        lines.append("- **RMSE (Root Mean Square Error):** Measures overall tracking accuracy")
        lines.append("- **Cross-Correlation:** Measures temporal relationship between target and mouse velocity")
        lines.append("- **Optimal Lag:** The time shift at which correlation is maximized")
        
        return '\n'.join(lines)
    
    def _generate_results_section(
        self,
        trial_metrics: pd.DataFrame,
        xcorr_results: Optional[pd.DataFrame],
        stat_results: Optional[Dict[str, Any]]
    ) -> str:
        """Generate results section."""
        lines = [
            "## Results",
            ""
        ]
        
        # Descriptive statistics
        lines.append("### Descriptive Statistics")
        lines.append("")
        
        # Overall RMSE
        overall_rmse = trial_metrics['rmse'].describe()
        lines.append("#### Overall RMSE")
        lines.append("")
        lines.append(f"- Mean: {overall_rmse['mean']:.2f} pixels")
        lines.append(f"- SD: {overall_rmse['std']:.2f} pixels")
        lines.append(f"- Range: {overall_rmse['min']:.2f} - {overall_rmse['max']:.2f} pixels")
        lines.append("")
        
        # By size
        lines.append("#### RMSE by Blob Size")
        lines.append("")
        lines.append("| Size (arcmin) | Mean | SD | N |")
        lines.append("|---------------|------|-----|---|")
        
        for size in sorted(trial_metrics['size_pixels'].unique()):
            size_data = trial_metrics[trial_metrics['size_pixels'] == size]['rmse']
            lines.append(f"| {size} | {size_data.mean():.2f} | {size_data.std():.2f} | {len(size_data)} |")
        
        lines.append("")
        
        # By condition
        lines.append("#### RMSE by Condition")
        lines.append("")
        lines.append("| Condition | Mean | SD | N |")
        lines.append("|-----------|------|-----|---|")
        
        for cond in ['dynamic', 'static']:
            cond_data = trial_metrics[trial_metrics['condition'] == cond]['rmse']
            label = 'Auditory Feedback' if cond == 'dynamic' else 'No Feedback'
            lines.append(f"| {label} | {cond_data.mean():.2f} | {cond_data.std():.2f} | {len(cond_data)} |")
        
        # Cross-correlation results
        if xcorr_results is not None and 'optimal_lag' in xcorr_results.columns:
            lines.append("")
            lines.append("### Cross-Correlation Analysis")
            lines.append("")
            
            mean_lag = xcorr_results['optimal_lag'].mean()
            mean_lag_ms = mean_lag * 20  # Assuming 50 fps
            
            lines.append(f"- **Mean Optimal Lag:** {mean_lag:.1f} frames ({mean_lag_ms:.0f} ms)")
            
            if mean_lag_ms > 0:
                lines.append(f"- **Interpretation:** Reactive tracking (mouse follows target by ~{mean_lag_ms:.0f} ms)")
            elif mean_lag_ms < 0:
                lines.append(f"- **Interpretation:** Predictive tracking (mouse leads target by ~{abs(mean_lag_ms):.0f} ms)")
            else:
                lines.append("- **Interpretation:** Synchronous tracking")
        
        return '\n'.join(lines)
    
    def _generate_statistics_section(
        self,
        stat_results: Dict[str, Any]
    ) -> str:
        """Generate statistical analysis section."""
        lines = [
            "## Statistical Analysis",
            ""
        ]
        
        # Size ANOVA
        if 'size_anova' in stat_results:
            anova = stat_results['size_anova']
            lines.append("### Effect of Blob Size")
            lines.append("")
            lines.append(f"- F-statistic: {anova.get('f_stat', 'N/A'):.2f}")
            lines.append(f"- p-value: {anova.get('p_value', 'N/A'):.4f}")
            if 'eta_squared' in anova:
                lines.append(f"- Effect size (η²): {anova['eta_squared']:.3f}")
            
            sig = anova.get('p_value', 1) < 0.05
            lines.append(f"- **Conclusion:** {'Significant' if sig else 'Not significant'} effect of blob size on RMSE")
            lines.append("")
        
        # Condition t-test
        if 'condition_ttest' in stat_results:
            ttest = stat_results['condition_ttest']
            lines.append("### Effect of Auditory Feedback")
            lines.append("")
            lines.append(f"- t-statistic: {ttest.get('t_stat', 'N/A'):.2f}")
            lines.append(f"- p-value: {ttest.get('p_value', 'N/A'):.4f}")
            if 'cohens_d' in ttest:
                lines.append(f"- Effect size (Cohen's d): {ttest['cohens_d']:.3f}")
            
            sig = ttest.get('p_value', 1) < 0.05
            lines.append(f"- **Conclusion:** {'Significant' if sig else 'Not significant'} effect of auditory feedback")
            lines.append("")
        
        # Post-hoc comparisons
        if 'posthoc' in stat_results:
            lines.append("### Post-hoc Comparisons")
            lines.append("")
            lines.append("| Comparison | t | p (corrected) | Significant |")
            lines.append("|------------|---|---------------|-------------|")
            
            for comp in stat_results['posthoc']:
                sig = '✓' if comp.get('p_corrected', 1) < 0.05 else ''
                lines.append(f"| {comp.get('comparison', 'N/A')} | {comp.get('t_stat', 0):.2f} | {comp.get('p_corrected', 1):.4f} | {sig} |")
        
        return '\n'.join(lines)
    
    def _generate_figures_section(
        self,
        plot_dir: Union[str, Path]
    ) -> str:
        """Generate figures section with plot references."""
        lines = [
            "## Figures",
            ""
        ]
        
        plot_dir = Path(plot_dir)
        
        # Find image files
        image_exts = ['.png', '.jpg', '.jpeg', '.svg']
        images = [f for f in plot_dir.iterdir() if f.suffix.lower() in image_exts]
        
        for i, img in enumerate(sorted(images), 1):
            rel_path = img.relative_to(plot_dir.parent) if plot_dir.parent != Path('.') else img.name
            lines.append(f"### Figure {i}: {img.stem.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"![{img.stem}]({rel_path})")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _generate_conclusions(
        self,
        trial_metrics: pd.DataFrame,
        stat_results: Optional[Dict[str, Any]]
    ) -> str:
        """Generate conclusions section."""
        lines = [
            "## Conclusions",
            ""
        ]
        
        # Research questions
        lines.append("### Research Questions")
        lines.append("")
        
        # Q1: Can observers discriminate blob sizes?
        lines.append("**Q1: Can observers discriminate blobs with different SD sizes?**")
        size_rmse = trial_metrics.groupby('size_pixels')['rmse'].mean()
        
        if stat_results and 'size_anova' in stat_results:
            sig = stat_results['size_anova'].get('p_value', 1) < 0.05
            if sig:
                lines.append(f"Yes. RMSE differed significantly across blob sizes "
                           f"(p < 0.05). Larger blobs (34 arcmin) were tracked with "
                           f"{'higher' if size_rmse[34] > size_rmse[21] else 'lower'} error "
                           f"than smaller blobs (21 arcmin).")
            else:
                lines.append("No significant difference in tracking performance across blob sizes was found.")
        else:
            lines.append("Statistical analysis not available.")
        
        lines.append("")
        
        # Q2: Does auditory feedback help?
        lines.append("**Q2: Does auditory feedback improve tracking performance?**")
        
        if stat_results and 'condition_ttest' in stat_results:
            sig = stat_results['condition_ttest'].get('p_value', 1) < 0.05
            cond_rmse = trial_metrics.groupby('condition')['rmse'].mean()
            
            if sig:
                better = 'dynamic' if cond_rmse.get('dynamic', 0) < cond_rmse.get('static', 0) else 'static'
                lines.append(f"Yes. Tracking performance was significantly "
                           f"{'better' if better == 'dynamic' else 'worse'} with auditory feedback "
                           f"(p < 0.05).")
            else:
                lines.append("No significant effect of auditory feedback on tracking performance was found.")
        else:
            lines.append("Statistical analysis not available.")
        
        return '\n'.join(lines)
    
    def _generate_appendix(
        self,
        trial_metrics: pd.DataFrame
    ) -> str:
        """Generate appendix with raw data."""
        lines = [
            "## Appendix: Raw Data",
            ""
        ]
        
        # Summary table
        lines.append("### Trial Metrics Summary")
        lines.append("")
        
        summary = trial_metrics.groupby(['participant_id', 'size_pixels', 'condition'])['rmse'].agg(['mean', 'std']).reset_index()
        
        lines.append(summary.to_markdown(index=False))
        
        return '\n'.join(lines)
    
    def generate_html_report(
        self,
        trial_metrics: pd.DataFrame,
        xcorr_results: Optional[pd.DataFrame] = None,
        stat_results: Optional[Dict[str, Any]] = None,
        analysis_config: Optional[Dict[str, Any]] = None,
        output_path: Optional[Union[str, Path]] = None,
        template_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate HTML analysis report.
        
        Args:
            trial_metrics: DataFrame with trial-level metrics
            xcorr_results: DataFrame with cross-correlation results
            stat_results: Dictionary with statistical results
            analysis_config: Configuration used for analysis
            output_path: Output file path
            template_path: Custom Jinja2 template path
            
        Returns:
            HTML string
        """
        if not HAS_JINJA2:
            # Fallback to basic HTML from Markdown
            md_report = self.generate_markdown_report(
                trial_metrics, xcorr_results, stat_results, analysis_config
            )
            html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.config.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
<pre>{md_report}</pre>
</body>
</html>
            """
            
            if output_path:
                Path(output_path).write_text(html, encoding='utf-8')
            
            return html
        
        # Use Jinja2 template
        if template_path:
            env = Environment(loader=FileSystemLoader(Path(template_path).parent))
            template = env.get_template(Path(template_path).name)
        else:
            # Default template
            default_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .summary-box { background-color: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .stat-sig { color: #27ae60; font-weight: bold; }
        .stat-ns { color: #7f8c8d; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <p>Generated: {{ timestamp }}</p>
    
    <div class="summary-box">
        <h2>Summary</h2>
        <p>Analysis of {{ n_participants }} participants across {{ n_trials }} trials.</p>
    </div>
    
    <h2>RMSE by Blob Size</h2>
    <table>
        <tr><th>Size (arcmin)</th><th>Mean RMSE</th><th>SD</th><th>N</th></tr>
        {% for row in size_stats %}
        <tr><td>{{ row.size }}</td><td>{{ "%.2f"|format(row.mean) }}</td><td>{{ "%.2f"|format(row.std) }}</td><td>{{ row.n }}</td></tr>
        {% endfor %}
    </table>
    
    <h2>RMSE by Condition</h2>
    <table>
        <tr><th>Condition</th><th>Mean RMSE</th><th>SD</th><th>N</th></tr>
        {% for row in cond_stats %}
        <tr><td>{{ row.condition }}</td><td>{{ "%.2f"|format(row.mean) }}</td><td>{{ "%.2f"|format(row.std) }}</td><td>{{ row.n }}</td></tr>
        {% endfor %}
    </table>
    
    <div class="footer">
        <p>Tracking Analysis Report</p>
    </div>
</body>
</html>
            """
            from jinja2 import Template
            template = Template(default_template)
        
        # Prepare data for template
        size_stats = []
        for size in sorted(trial_metrics['size_pixels'].unique()):
            size_data = trial_metrics[trial_metrics['size_pixels'] == size]['rmse']
            size_stats.append({
                'size': size,
                'mean': size_data.mean(),
                'std': size_data.std(),
                'n': len(size_data)
            })
        
        cond_stats = []
        for cond in ['dynamic', 'static']:
            cond_data = trial_metrics[trial_metrics['condition'] == cond]['rmse']
            cond_stats.append({
                'condition': 'Auditory Feedback' if cond == 'dynamic' else 'No Feedback',
                'mean': cond_data.mean(),
                'std': cond_data.std(),
                'n': len(cond_data)
            })
        
        html = template.render(
            title=self.config.title,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            n_participants=trial_metrics['participant_id'].nunique(),
            n_trials=len(trial_metrics),
            size_stats=size_stats,
            cond_stats=cond_stats
        )
        
        if output_path:
            Path(output_path).write_text(html, encoding='utf-8')
        
        return html
