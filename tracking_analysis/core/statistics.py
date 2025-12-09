"""
Statistics Module
=================

Implements statistical analysis including ANOVA, t-tests, effect sizes,
and multiple comparison corrections for tracking experiment data.

Classes:
    StatisticalAnalyzer: Main class for statistical analysis

Functions:
    run_anova: Run repeated-measures or mixed ANOVA
    posthoc_tests: Perform post-hoc pairwise comparisons
    effect_sizes: Calculate various effect size measures
    generate_summary_table: Create publication-ready summary tables
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Any, Tuple, Union, Literal
from dataclasses import dataclass
import warnings

# Import statsmodels components
try:
    import statsmodels.api as sm
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not installed. Some statistical functions will be limited.")

from .config import Config


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ANOVAResult:
    """
    Result of ANOVA analysis.
    
    Attributes:
        test_type: Type of ANOVA performed
        factors: Factors included in analysis
        
        f_statistic: F-statistic value
        p_value: P-value
        df_between: Degrees of freedom (between)
        df_within: Degrees of freedom (within/error)
        
        effect_size: Effect size measure
        effect_size_type: Type of effect size (eta_squared, partial_eta_squared, etc.)
        
        is_significant: Whether result is significant at alpha level
        alpha: Significance level used
        
        summary_table: Full ANOVA summary table
    """
    test_type: str
    factors: List[str]
    
    f_statistic: float
    p_value: float
    df_between: float
    df_within: float
    
    effect_size: float
    effect_size_type: str
    
    is_significant: bool
    alpha: float
    
    summary_table: Optional[pd.DataFrame] = None


@dataclass
class PostHocResult:
    """
    Result of post-hoc pairwise comparisons.
    
    Attributes:
        comparisons: List of comparison pairs
        p_values: P-values for each comparison
        p_values_corrected: Corrected p-values
        correction_method: Method used for correction
        significant: Which comparisons are significant
        effect_sizes: Effect sizes for each comparison
    """
    comparisons: List[Tuple[str, str]]
    p_values: List[float]
    p_values_corrected: List[float]
    correction_method: str
    significant: List[bool]
    effect_sizes: List[float]
    summary_table: pd.DataFrame


# =============================================================================
# EFFECT SIZE CALCULATIONS
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two groups.
    
    d = (mean1 - mean2) / pooled_std
    
    Args:
        group1: First group values
        group2: Second group values
        
    Returns:
        Cohen's d value
        
    Interpretation:
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def partial_eta_squared(ss_effect: float, ss_error: float) -> float:
    """
    Calculate partial eta-squared effect size.
    
    η²_p = SS_effect / (SS_effect + SS_error)
    
    Args:
        ss_effect: Sum of squares for the effect
        ss_error: Sum of squares for error
        
    Returns:
        Partial eta-squared value
        
    Interpretation:
        η²_p < 0.01: negligible
        0.01 ≤ η²_p < 0.06: small
        0.06 ≤ η²_p < 0.14: medium
        η²_p ≥ 0.14: large
    """
    if ss_effect + ss_error == 0:
        return 0.0
    return ss_effect / (ss_effect + ss_error)


def omega_squared(ss_effect: float, ss_error: float, ms_error: float, df_effect: int, n_total: int) -> float:
    """
    Calculate omega-squared effect size (less biased than eta-squared).
    
    ω² = (SS_effect - df_effect * MS_error) / (SS_total + MS_error)
    
    Args:
        ss_effect: Sum of squares for effect
        ss_error: Sum of squares for error
        ms_error: Mean square error
        df_effect: Degrees of freedom for effect
        n_total: Total sample size
        
    Returns:
        Omega-squared value
    """
    ss_total = ss_effect + ss_error
    numerator = ss_effect - (df_effect * ms_error)
    denominator = ss_total + ms_error
    
    if denominator == 0:
        return 0.0
    
    return max(0, numerator / denominator)


def interpret_effect_size(value: float, measure: str = 'cohens_d') -> str:
    """
    Generate interpretation of effect size.
    
    Args:
        value: Effect size value
        measure: Type of effect size measure
        
    Returns:
        Interpretation string
    """
    abs_value = abs(value)
    
    if measure == 'cohens_d':
        if abs_value < 0.2:
            return "negligible"
        elif abs_value < 0.5:
            return "small"
        elif abs_value < 0.8:
            return "medium"
        else:
            return "large"
    else:  # eta_squared, partial_eta_squared, omega_squared
        if abs_value < 0.01:
            return "negligible"
        elif abs_value < 0.06:
            return "small"
        elif abs_value < 0.14:
            return "medium"
        else:
            return "large"


# =============================================================================
# MULTIPLE COMPARISON CORRECTIONS
# =============================================================================

def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Adjusted alpha = alpha / n_comparisons
    
    Args:
        p_values: Array of p-values
        alpha: Original significance level
        
    Returns:
        Tuple of (adjusted_p_values, significant_mask)
    """
    n = len(p_values)
    adjusted = np.minimum(p_values * n, 1.0)
    significant = adjusted < alpha
    return adjusted, significant


def holm_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Holm-Bonferroni step-down correction.
    
    More powerful than Bonferroni while still controlling familywise error.
    
    Args:
        p_values: Array of p-values
        alpha: Significance level
        
    Returns:
        Tuple of (adjusted_p_values, significant_mask)
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    adjusted = np.zeros(n)
    for i, p in enumerate(sorted_p):
        adjusted[sorted_indices[i]] = min(p * (n - i), 1.0)
    
    # Ensure monotonicity
    for i in range(1, n):
        idx = sorted_indices[i]
        prev_idx = sorted_indices[i-1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    
    significant = adjusted < alpha
    return adjusted, significant


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg False Discovery Rate correction.
    
    Controls expected proportion of false discoveries rather than
    familywise error rate. More powerful for many comparisons.
    
    Args:
        p_values: Array of p-values
        alpha: Significance level
        
    Returns:
        Tuple of (adjusted_p_values, significant_mask)
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Calculate adjusted p-values
    adjusted = np.zeros(n)
    for i in range(n-1, -1, -1):
        if i == n-1:
            adjusted[sorted_indices[i]] = sorted_p[i]
        else:
            adjusted[sorted_indices[i]] = min(
                sorted_p[i] * n / (i + 1),
                adjusted[sorted_indices[i + 1]]
            )
    
    adjusted = np.minimum(adjusted, 1.0)
    significant = adjusted < alpha
    return adjusted, significant


def apply_correction(
    p_values: np.ndarray,
    method: Literal['bonferroni', 'holm', 'fdr', 'none'] = 'bonferroni',
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply specified multiple comparison correction.
    
    Args:
        p_values: Array of p-values
        method: Correction method
        alpha: Significance level
        
    Returns:
        Tuple of (adjusted_p_values, significant_mask)
    """
    if method == 'none':
        return p_values, p_values < alpha
    elif method == 'bonferroni':
        return bonferroni_correction(p_values, alpha)
    elif method == 'holm':
        return holm_correction(p_values, alpha)
    elif method == 'fdr':
        return fdr_correction(p_values, alpha)
    else:
        raise ValueError(f"Unknown correction method: {method}")


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def paired_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform paired t-test.
    
    Args:
        group1: First group values
        group2: Second group values (matched pairs)
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    t_stat, p_value = stats.ttest_rel(group1, group2)
    d = cohens_d(group1, group2)
    
    return {
        'test': 'Paired t-test',
        't_statistic': t_stat,
        'p_value': p_value,
        'df': len(group1) - 1,
        'cohens_d': d,
        'effect_interpretation': interpret_effect_size(d, 'cohens_d'),
        'is_significant': p_value < alpha,
        'alpha': alpha,
        'mean_difference': np.mean(group1) - np.mean(group2),
        'group1_mean': np.mean(group1),
        'group1_std': np.std(group1, ddof=1),
        'group2_mean': np.mean(group2),
        'group2_std': np.std(group2, ddof=1)
    }


def independent_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform independent samples t-test.
    
    Args:
        group1: First group values
        group2: Second group values
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    t_stat, p_value = stats.ttest_ind(group1, group2)
    d = cohens_d(group1, group2)
    
    return {
        'test': 'Independent t-test',
        't_statistic': t_stat,
        'p_value': p_value,
        'df': len(group1) + len(group2) - 2,
        'cohens_d': d,
        'effect_interpretation': interpret_effect_size(d, 'cohens_d'),
        'is_significant': p_value < alpha,
        'alpha': alpha,
        'mean_difference': np.mean(group1) - np.mean(group2),
        'group1_mean': np.mean(group1),
        'group1_std': np.std(group1, ddof=1),
        'group1_n': len(group1),
        'group2_mean': np.mean(group2),
        'group2_std': np.std(group2, ddof=1),
        'group2_n': len(group2)
    }


def one_way_anova(
    *groups,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform one-way ANOVA.
    
    Args:
        *groups: Variable number of group arrays
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Calculate effect size (eta-squared)
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_data - grand_mean)**2)
    
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    
    return {
        'test': 'One-way ANOVA',
        'f_statistic': f_stat,
        'p_value': p_value,
        'df_between': len(groups) - 1,
        'df_within': len(all_data) - len(groups),
        'eta_squared': eta_sq,
        'effect_interpretation': interpret_effect_size(eta_sq, 'eta_squared'),
        'is_significant': p_value < alpha,
        'alpha': alpha,
        'n_groups': len(groups),
        'group_means': [np.mean(g) for g in groups],
        'group_stds': [np.std(g, ddof=1) for g in groups],
        'group_ns': [len(g) for g in groups]
    }


def repeated_measures_anova(
    data: pd.DataFrame,
    depvar: str,
    subject: str,
    within: List[str],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform repeated-measures ANOVA using statsmodels.
    
    Args:
        data: DataFrame in long format
        depvar: Name of dependent variable column
        subject: Name of subject ID column
        within: List of within-subject factor column names
        alpha: Significance level
        
    Returns:
        Dictionary with ANOVA results
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for repeated-measures ANOVA")
    
    # Run ANOVA
    aovrm = AnovaRM(data, depvar, subject, within=within)
    res = aovrm.fit()
    
    # Extract results
    anova_table = res.anova_table
    
    results = {
        'test': 'Repeated-Measures ANOVA',
        'summary_table': anova_table,
        'factors': {},
        'alpha': alpha
    }
    
    # Process each factor
    for factor in anova_table.index:
        f_stat = anova_table.loc[factor, 'F Value']
        p_value = anova_table.loc[factor, 'Pr > F']
        df_num = anova_table.loc[factor, 'Num DF']
        df_den = anova_table.loc[factor, 'Den DF']
        
        # Calculate partial eta-squared
        # η²_p ≈ F * df_num / (F * df_num + df_den)
        eta_p = (f_stat * df_num) / (f_stat * df_num + df_den) if f_stat > 0 else 0
        
        results['factors'][factor] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'df_numerator': df_num,
            'df_denominator': df_den,
            'partial_eta_squared': eta_p,
            'effect_interpretation': interpret_effect_size(eta_p, 'partial_eta_squared'),
            'is_significant': p_value < alpha
        }
    
    return results


# =============================================================================
# POST-HOC TESTS
# =============================================================================

def pairwise_comparisons(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    subject_col: Optional[str] = None,
    correction: str = 'bonferroni',
    alpha: float = 0.05
) -> PostHocResult:
    """
    Perform pairwise comparisons between groups.
    
    Args:
        data: DataFrame with data
        value_col: Column with values to compare
        group_col: Column with group labels
        subject_col: Column with subject IDs (for paired comparisons)
        correction: Multiple comparison correction method
        alpha: Significance level
        
    Returns:
        PostHocResult with all comparisons
    """
    groups = data[group_col].unique()
    n_groups = len(groups)
    
    comparisons = []
    p_values = []
    effect_sizes = []
    
    # Generate all pairwise comparisons
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            g1_name, g2_name = groups[i], groups[j]
            g1_data = data[data[group_col] == g1_name][value_col].values
            g2_data = data[data[group_col] == g2_name][value_col].values
            
            comparisons.append((str(g1_name), str(g2_name)))
            
            # Perform appropriate t-test
            if subject_col is not None:
                # Paired comparison - need to match subjects
                g1_df = data[data[group_col] == g1_name].set_index(subject_col)
                g2_df = data[data[group_col] == g2_name].set_index(subject_col)
                common_subjects = g1_df.index.intersection(g2_df.index)
                
                if len(common_subjects) > 0:
                    g1_matched = g1_df.loc[common_subjects, value_col].values
                    g2_matched = g2_df.loc[common_subjects, value_col].values
                    _, p = stats.ttest_rel(g1_matched, g2_matched)
                    d = cohens_d(g1_matched, g2_matched)
                else:
                    p = 1.0
                    d = 0.0
            else:
                _, p = stats.ttest_ind(g1_data, g2_data)
                d = cohens_d(g1_data, g2_data)
            
            p_values.append(p)
            effect_sizes.append(d)
    
    # Apply correction
    p_values_arr = np.array(p_values)
    p_corrected, significant = apply_correction(p_values_arr, correction, alpha)
    
    # Create summary table
    summary_df = pd.DataFrame({
        'Group 1': [c[0] for c in comparisons],
        'Group 2': [c[1] for c in comparisons],
        'p-value': p_values,
        'p-corrected': p_corrected,
        'Significant': significant,
        "Cohen's d": effect_sizes,
        'Effect Size': [interpret_effect_size(d, 'cohens_d') for d in effect_sizes]
    })
    
    return PostHocResult(
        comparisons=comparisons,
        p_values=p_values,
        p_values_corrected=list(p_corrected),
        correction_method=correction,
        significant=list(significant),
        effect_sizes=effect_sizes,
        summary_table=summary_df
    )


# =============================================================================
# STATISTICAL ANALYZER CLASS
# =============================================================================

class StatisticalAnalyzer:
    """
    Main class for statistical analysis of tracking data.
    
    Provides unified interface for:
    - ANOVA (one-way, repeated-measures)
    - T-tests (paired, independent)
    - Effect size calculations
    - Post-hoc comparisons
    - Multiple comparison corrections
    
    Attributes:
        config: Configuration object
        
    Example:
        >>> analyzer = StatisticalAnalyzer(config)
        >>> 
        >>> # Compare conditions
        >>> result = analyzer.compare_conditions(trial_metrics, 'rmse')
        >>> 
        >>> # Run full ANOVA
        >>> anova_result = analyzer.run_anova(
        ...     trial_metrics,
        ...     depvar='rmse',
        ...     factors=['size', 'condition']
        ... )
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize StatisticalAnalyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
    
    def compare_two_groups(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        paired: bool = True,
        group_names: Tuple[str, str] = ('Group 1', 'Group 2')
    ) -> Dict[str, Any]:
        """
        Compare two groups using appropriate t-test.
        
        Args:
            group1: First group values
            group2: Second group values
            paired: Whether groups are paired (repeated measures)
            group_names: Names for the groups
            
        Returns:
            Dictionary with comparison results
        """
        if paired:
            result = paired_ttest(group1, group2, self.config.alpha)
        else:
            result = independent_ttest(group1, group2, self.config.alpha)
        
        result['group_names'] = group_names
        return result
    
    def compare_conditions(
        self,
        data: pd.DataFrame,
        metric: str = 'rmse',
        subject_col: str = 'participant_id'
    ) -> Dict[str, Any]:
        """
        Compare metric between auditory feedback conditions.
        
        Args:
            data: DataFrame with trial-level metrics
            metric: Metric column to compare
            subject_col: Subject identifier column
            
        Returns:
            Comparison results dictionary
        """
        dynamic = data[data['condition'] == 'dynamic']
        static = data[data['condition'] == 'static']
        
        # Match by participant for paired comparison
        dynamic_by_subj = dynamic.groupby(subject_col)[metric].mean()
        static_by_subj = static.groupby(subject_col)[metric].mean()
        
        common_subjects = dynamic_by_subj.index.intersection(static_by_subj.index)
        
        if len(common_subjects) > 0:
            dynamic_matched = dynamic_by_subj.loc[common_subjects].values
            static_matched = static_by_subj.loc[common_subjects].values
            
            result = self.compare_two_groups(
                dynamic_matched,
                static_matched,
                paired=True,
                group_names=('Auditory Feedback', 'No Feedback')
            )
            result['n_subjects'] = len(common_subjects)
        else:
            # Fall back to independent comparison
            result = self.compare_two_groups(
                dynamic[metric].values,
                static[metric].values,
                paired=False,
                group_names=('Auditory Feedback', 'No Feedback')
            )
        
        result['metric'] = metric
        return result
    
    def compare_sizes(
        self,
        data: pd.DataFrame,
        metric: str = 'rmse'
    ) -> Dict[str, Any]:
        """
        Compare metric across blob sizes using one-way ANOVA.
        
        Args:
            data: DataFrame with trial-level metrics
            metric: Metric column to compare
            
        Returns:
            ANOVA results dictionary
        """
        sizes = data['size'].unique()
        groups = [data[data['size'] == s][metric].values for s in sizes]
        
        result = one_way_anova(*groups, alpha=self.config.alpha)
        result['factor'] = 'size'
        result['levels'] = list(sizes)
        result['metric'] = metric
        
        return result
    
    def run_full_anova(
        self,
        data: pd.DataFrame,
        depvar: str = 'rmse',
        subject_col: str = 'participant_id',
        within_factors: List[str] = ['size', 'condition']
    ) -> Dict[str, Any]:
        """
        Run full repeated-measures ANOVA.
        
        Args:
            data: DataFrame with trial-level metrics
            depvar: Dependent variable column
            subject_col: Subject identifier column
            within_factors: Within-subject factors
            
        Returns:
            ANOVA results dictionary
        """
        if not STATSMODELS_AVAILABLE:
            # Fall back to simpler analysis
            return self._fallback_anova(data, depvar, within_factors)
        
        # Aggregate to one value per subject per condition
        groupby_cols = [subject_col] + within_factors
        agg_data = data.groupby(groupby_cols)[depvar].mean().reset_index()
        
        try:
            result = repeated_measures_anova(
                agg_data,
                depvar=depvar,
                subject=subject_col,
                within=within_factors,
                alpha=self.config.alpha
            )
            result['depvar'] = depvar
            return result
        except Exception as e:
            warnings.warn(f"Repeated-measures ANOVA failed: {e}. Using fallback.")
            return self._fallback_anova(data, depvar, within_factors)
    
    def _fallback_anova(
        self,
        data: pd.DataFrame,
        depvar: str,
        factors: List[str]
    ) -> Dict[str, Any]:
        """
        Fallback ANOVA using scipy when statsmodels unavailable.
        
        Performs separate one-way ANOVAs for each factor.
        """
        results = {
            'test': 'Separate One-way ANOVAs (fallback)',
            'factors': {},
            'alpha': self.config.alpha,
            'warning': 'Full factorial ANOVA requires statsmodels. Showing separate factor effects.'
        }
        
        for factor in factors:
            levels = data[factor].unique()
            groups = [data[data[factor] == level][depvar].values for level in levels]
            
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Calculate eta-squared
            all_data = np.concatenate(groups)
            grand_mean = np.mean(all_data)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
            ss_total = np.sum((all_data - grand_mean)**2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0
            
            results['factors'][factor] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'df_between': len(levels) - 1,
                'df_within': len(all_data) - len(levels),
                'eta_squared': eta_sq,
                'effect_interpretation': interpret_effect_size(eta_sq, 'eta_squared'),
                'is_significant': p_value < self.config.alpha,
                'levels': list(levels)
            }
        
        return results
    
    def posthoc_comparisons(
        self,
        data: pd.DataFrame,
        depvar: str,
        factor: str,
        subject_col: str = 'participant_id'
    ) -> PostHocResult:
        """
        Perform post-hoc pairwise comparisons for a factor.
        
        Args:
            data: DataFrame with trial-level metrics
            depvar: Dependent variable column
            factor: Factor to compare levels of
            subject_col: Subject identifier column
            
        Returns:
            PostHocResult with all pairwise comparisons
        """
        # Aggregate by subject and factor level
        agg_data = data.groupby([subject_col, factor])[depvar].mean().reset_index()
        
        return pairwise_comparisons(
            agg_data,
            value_col=depvar,
            group_col=factor,
            subject_col=subject_col,
            correction=self.config.posthoc_method,
            alpha=self.config.alpha
        )
    
    def generate_results_summary(
        self,
        data: pd.DataFrame,
        metric: str = 'rmse'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistical summary.
        
        Args:
            data: DataFrame with trial-level metrics
            metric: Metric to analyze
            
        Returns:
            Dictionary with all statistical results
        """
        summary = {
            'metric': metric,
            'n_participants': data['participant_id'].nunique(),
            'n_trials': len(data),
            'descriptive': {},
            'inferential': {}
        }
        
        # Descriptive statistics
        summary['descriptive']['overall'] = {
            'mean': data[metric].mean(),
            'std': data[metric].std(),
            'median': data[metric].median(),
            'min': data[metric].min(),
            'max': data[metric].max()
        }
        
        summary['descriptive']['by_size'] = data.groupby('size')[metric].agg(
            ['mean', 'std', 'median', 'count']
        ).to_dict('index')
        
        summary['descriptive']['by_condition'] = data.groupby('condition')[metric].agg(
            ['mean', 'std', 'median', 'count']
        ).to_dict('index')
        
        # Inferential statistics
        summary['inferential']['condition_comparison'] = self.compare_conditions(data, metric)
        summary['inferential']['size_comparison'] = self.compare_sizes(data, metric)
        
        # Full ANOVA if possible
        try:
            summary['inferential']['full_anova'] = self.run_full_anova(data, metric)
        except Exception as e:
            summary['inferential']['full_anova'] = {'error': str(e)}
        
        # Post-hoc for size
        if summary['inferential']['size_comparison'].get('is_significant', False):
            try:
                summary['inferential']['size_posthoc'] = self.posthoc_comparisons(
                    data, metric, 'size'
                )
            except Exception as e:
                summary['inferential']['size_posthoc'] = {'error': str(e)}
        
        return summary
    
    def format_results_for_report(
        self,
        results: Dict[str, Any]
    ) -> str:
        """
        Format statistical results as text for reports.
        
        Args:
            results: Results dictionary from generate_results_summary
            
        Returns:
            Formatted text string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("STATISTICAL ANALYSIS RESULTS")
        lines.append("=" * 60)
        
        lines.append(f"\nMetric Analyzed: {results['metric'].upper()}")
        lines.append(f"Participants: N = {results['n_participants']}")
        lines.append(f"Total Trials: {results['n_trials']}")
        
        # Descriptive stats
        lines.append("\n" + "-" * 40)
        lines.append("DESCRIPTIVE STATISTICS")
        lines.append("-" * 40)
        
        overall = results['descriptive']['overall']
        lines.append(f"\nOverall: M = {overall['mean']:.2f}, SD = {overall['std']:.2f}")
        
        # Condition comparison
        cond_result = results['inferential']['condition_comparison']
        lines.append("\n" + "-" * 40)
        lines.append("CONDITION COMPARISON (Auditory Feedback vs No Feedback)")
        lines.append("-" * 40)
        
        lines.append(f"\nTest: {cond_result['test']}")
        lines.append(f"t({cond_result['df']}) = {cond_result['t_statistic']:.3f}, p = {cond_result['p_value']:.4f}")
        lines.append(f"Cohen's d = {cond_result['cohens_d']:.3f} ({cond_result['effect_interpretation']} effect)")
        lines.append(f"Significant at α = {cond_result['alpha']}: {'Yes' if cond_result['is_significant'] else 'No'}")
        
        # Size comparison
        size_result = results['inferential']['size_comparison']
        lines.append("\n" + "-" * 40)
        lines.append("SIZE COMPARISON (21 vs 31 vs 34 arcmin)")
        lines.append("-" * 40)
        
        lines.append(f"\nTest: {size_result['test']}")
        lines.append(f"F({size_result['df_between']}, {size_result['df_within']}) = {size_result['f_statistic']:.3f}, p = {size_result['p_value']:.4f}")
        lines.append(f"η² = {size_result['eta_squared']:.3f} ({size_result['effect_interpretation']} effect)")
        lines.append(f"Significant at α = {size_result['alpha']}: {'Yes' if size_result['is_significant'] else 'No'}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def compare_xcorr_conditions(
        self,
        xcorr_df: pd.DataFrame,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compare cross-correlation results between dynamic and static conditions.
        
        Performs statistical tests on lag and correlation strength between conditions.
        
        Args:
            xcorr_df: DataFrame from CrossCorrelationAnalyzer.analyze_all_trials()
            alpha: Significance level
            
        Returns:
            Dictionary with statistical comparison results
        """
        dynamic = xcorr_df[xcorr_df['condition'] == 'dynamic']
        static = xcorr_df[xcorr_df['condition'] == 'static']
        
        results = {
            'alpha': alpha,
            'lag_comparison': {},
            'correlation_comparison': {},
            'by_size': {}
        }
        
        # === LAG COMPARISON (Dynamic vs Static) ===
        dyn_lag = dynamic['optimal_lag_ms'].values
        sta_lag = static['optimal_lag_ms'].values
        
        if len(dyn_lag) == len(sta_lag):
            lag_test = paired_ttest(dyn_lag, sta_lag, alpha)
            lag_test['test_type'] = 'Paired t-test (matched participants)'
        else:
            lag_test = independent_ttest(dyn_lag, sta_lag, alpha)
            lag_test['test_type'] = 'Independent t-test'
        
        results['lag_comparison'] = {
            **lag_test,
            'dynamic_mean': float(np.mean(dyn_lag)),
            'dynamic_std': float(np.std(dyn_lag, ddof=1)),
            'static_mean': float(np.mean(sta_lag)),
            'static_std': float(np.std(sta_lag, ddof=1)),
            'interpretation': self._interpret_lag_comparison(lag_test, np.mean(dyn_lag), np.mean(sta_lag))
        }
        
        # === CORRELATION COMPARISON (Dynamic vs Static) ===
        dyn_corr = dynamic['max_correlation'].values
        sta_corr = static['max_correlation'].values
        
        if len(dyn_corr) == len(sta_corr):
            corr_test = paired_ttest(dyn_corr, sta_corr, alpha)
            corr_test['test_type'] = 'Paired t-test (matched participants)'
        else:
            corr_test = independent_ttest(dyn_corr, sta_corr, alpha)
            corr_test['test_type'] = 'Independent t-test'
        
        results['correlation_comparison'] = {
            **corr_test,
            'dynamic_mean': float(np.mean(dyn_corr)),
            'dynamic_std': float(np.std(dyn_corr, ddof=1)),
            'static_mean': float(np.mean(sta_corr)),
            'static_std': float(np.std(sta_corr, ddof=1)),
            'interpretation': self._interpret_corr_comparison(corr_test, np.mean(dyn_corr), np.mean(sta_corr))
        }
        
        # === PREDICTIVE/REACTIVE PROPORTIONS ===
        dyn_predictive = dynamic['is_predictive'].sum()
        sta_predictive = static['is_predictive'].sum()
        
        results['predictive_tracking'] = {
            'dynamic_predictive_n': int(dyn_predictive),
            'dynamic_predictive_pct': 100 * dyn_predictive / len(dynamic) if len(dynamic) > 0 else 0,
            'static_predictive_n': int(sta_predictive),
            'static_predictive_pct': 100 * sta_predictive / len(static) if len(static) > 0 else 0,
            'dynamic_n': len(dynamic),
            'static_n': len(static)
        }
        
        # === BY SIZE ANALYSIS ===
        size_col = 'size_pixels' if 'size_pixels' in xcorr_df.columns else ('size' if 'size' in xcorr_df.columns else None)
        
        if size_col:
            for size in sorted(xcorr_df[size_col].unique()):
                size_data = xcorr_df[xcorr_df[size_col] == size]
                size_dyn = size_data[size_data['condition'] == 'dynamic']
                size_sta = size_data[size_data['condition'] == 'static']
                
                if len(size_dyn) > 1 and len(size_sta) > 1:
                    if len(size_dyn) == len(size_sta):
                        size_lag_test = paired_ttest(size_dyn['optimal_lag_ms'].values, size_sta['optimal_lag_ms'].values, alpha)
                        size_corr_test = paired_ttest(size_dyn['max_correlation'].values, size_sta['max_correlation'].values, alpha)
                    else:
                        size_lag_test = independent_ttest(size_dyn['optimal_lag_ms'].values, size_sta['optimal_lag_ms'].values, alpha)
                        size_corr_test = independent_ttest(size_dyn['max_correlation'].values, size_sta['max_correlation'].values, alpha)
                    
                    results['by_size'][size] = {
                        'lag': {
                            'dynamic_mean': float(size_dyn['optimal_lag_ms'].mean()),
                            'dynamic_std': float(size_dyn['optimal_lag_ms'].std()),
                            'static_mean': float(size_sta['optimal_lag_ms'].mean()),
                            'static_std': float(size_sta['optimal_lag_ms'].std()),
                            'p_value': size_lag_test['p_value'],
                            'cohens_d': size_lag_test['cohens_d'],
                            'is_significant': size_lag_test['is_significant']
                        },
                        'correlation': {
                            'dynamic_mean': float(size_dyn['max_correlation'].mean()),
                            'dynamic_std': float(size_dyn['max_correlation'].std()),
                            'static_mean': float(size_sta['max_correlation'].mean()),
                            'static_std': float(size_sta['max_correlation'].std()),
                            'p_value': size_corr_test['p_value'],
                            'cohens_d': size_corr_test['cohens_d'],
                            'is_significant': size_corr_test['is_significant']
                        },
                        'n_dynamic': len(size_dyn),
                        'n_static': len(size_sta)
                    }
        
        results['summary'] = self._generate_xcorr_summary(results)
        return results
    
    def _interpret_lag_comparison(self, test_result: Dict, dyn_mean: float, sta_mean: float) -> str:
        """Generate interpretation for lag comparison."""
        if not test_result['is_significant']:
            return "No significant difference in tracking lag between conditions."
        if dyn_mean < sta_mean:
            return f"Dynamic condition shows SHORTER lag ({dyn_mean:.1f}ms vs {sta_mean:.1f}ms). Auditory feedback may improve response time."
        return f"Static condition shows SHORTER lag ({sta_mean:.1f}ms vs {dyn_mean:.1f}ms)."
    
    def _interpret_corr_comparison(self, test_result: Dict, dyn_mean: float, sta_mean: float) -> str:
        """Generate interpretation for correlation comparison."""
        if not test_result['is_significant']:
            return "No significant difference in tracking correlation between conditions."
        if dyn_mean > sta_mean:
            return f"Dynamic shows STRONGER correlation ({dyn_mean:.3f} vs {sta_mean:.3f}). Auditory feedback improves velocity tracking."
        return f"Static shows STRONGER correlation ({sta_mean:.3f} vs {dyn_mean:.3f})."
    
    def _generate_xcorr_summary(self, results: Dict) -> Dict[str, str]:
        """Generate overall summary of cross-correlation comparison."""
        lag_comp = results['lag_comparison']
        corr_comp = results['correlation_comparison']
        pred = results['predictive_tracking']
        
        findings = []
        if lag_comp['is_significant']:
            findings.append(f"Tracking lag differs significantly ({lag_comp['effect_interpretation']} effect, d={lag_comp['cohens_d']:.2f}).")
        else:
            findings.append("No significant lag difference between conditions.")
        
        if corr_comp['is_significant']:
            findings.append(f"Correlation strength differs significantly ({corr_comp['effect_interpretation']} effect, d={corr_comp['cohens_d']:.2f}).")
        else:
            findings.append("No significant correlation difference between conditions.")
        
        dyn_pct, sta_pct = pred['dynamic_predictive_pct'], pred['static_predictive_pct']
        if dyn_pct > 50 and sta_pct > 50:
            tracking_type = "predominantly PREDICTIVE in both conditions"
        elif dyn_pct < 50 and sta_pct < 50:
            tracking_type = "predominantly REACTIVE in both conditions"
        else:
            tracking_type = f"more PREDICTIVE with {'feedback' if dyn_pct > sta_pct else 'no feedback'} ({max(dyn_pct, sta_pct):.0f}% vs {min(dyn_pct, sta_pct):.0f}%)"
        findings.append(f"Tracking is {tracking_type}.")
        
        return {
            'main_findings': "\n".join(findings),
            'lag_result': lag_comp['interpretation'],
            'correlation_result': corr_comp['interpretation'],
            'answer_does_feedback_help': self._does_feedback_help(results)
        }
    
    def _does_feedback_help(self, results: Dict) -> str:
        """Answer: Does auditory feedback improve tracking?"""
        lag_comp = results['lag_comparison']
        corr_comp = results['correlation_comparison']
        
        better, worse, evidence = 0, 0, []
        dyn_abs, sta_abs = abs(lag_comp['dynamic_mean']), abs(lag_comp['static_mean'])
        
        if dyn_abs < sta_abs:
            better += 1
            if lag_comp['is_significant']:
                evidence.append(f"✅ Smaller absolute lag with feedback ({dyn_abs:.1f}ms vs {sta_abs:.1f}ms, p={lag_comp['p_value']:.4f})")
        else:
            worse += 1
            if lag_comp['is_significant']:
                evidence.append(f"❌ Larger absolute lag with feedback ({dyn_abs:.1f}ms vs {sta_abs:.1f}ms, p={lag_comp['p_value']:.4f})")
        
        if corr_comp['dynamic_mean'] > corr_comp['static_mean']:
            better += 1
            if corr_comp['is_significant']:
                evidence.append(f"✅ Stronger correlation with feedback ({corr_comp['dynamic_mean']:.3f} vs {corr_comp['static_mean']:.3f}, p={corr_comp['p_value']:.4f})")
        else:
            worse += 1
            if corr_comp['is_significant']:
                evidence.append(f"❌ Weaker correlation with feedback ({corr_comp['dynamic_mean']:.3f} vs {corr_comp['static_mean']:.3f}, p={corr_comp['p_value']:.4f})")
        
        if not evidence:
            evidence.append("No statistically significant differences found.")
        
        if better > worse:
            conclusion = "🎧 **Evidence suggests auditory feedback IMPROVES tracking performance.**"
        elif worse > better:
            conclusion = "🔇 **Evidence suggests auditory feedback does NOT improve tracking.**"
        else:
            conclusion = "⚖️ **Mixed evidence - no clear advantage for auditory feedback.**"
        
        return f"{conclusion}\n\nEvidence:\n" + "\n".join(evidence)

    def answer_research_questions(
        self,
        trial_metrics: pd.DataFrame,
        xcorr_results: pd.DataFrame,
        rmse_threshold: float = 50.0,
        correlation_threshold: float = 0.5,
        lag_threshold_ms: float = 500.0,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Automatically answer research questions about blob discrimination.
        
        Research Questions:
        Q1: Can participants discriminate the 21-pixel SD blob from the background?
        Q2: Can participants discriminate the 31-pixel SD blob from the background?
        Q3: Can participants discriminate the 34-pixel SD blob from the background?
        
        Uses multi-metric evidence combining:
        - RMSE (position accuracy) - lower is better
        - Max correlation (velocity tracking) - higher is better  
        - Optimal lag (response timing) - closer to 0 is better
        
        Parameters:
        -----------
        trial_metrics : pd.DataFrame
            Trial-level metrics with columns: participant_id, blob_size, condition, rmse, etc.
        xcorr_results : pd.DataFrame
            Cross-correlation results with columns: participant_id, blob_size, condition, 
            max_correlation, optimal_lag_ms
        rmse_threshold : float
            RMSE threshold for "good" tracking (default 50 pixels)
        correlation_threshold : float
            Minimum correlation for "good" velocity tracking (default 0.5)
        lag_threshold_ms : float
            Maximum acceptable lag in ms (default 500ms)
        alpha : float
            Significance level for statistical tests (default 0.05)
            
        Returns:
        --------
        Dict with keys:
            - 'questions': Dict[str, Dict] - Results for each blob size
            - 'comparison': Dict - Size comparison analysis
            - 'summary': str - Overall narrative summary
        """
        results = {
            'questions': {},
            'comparison': {},
            'summary': ''
        }
        
        # Get unique blob sizes - column is 'size' (string like '21', '31', '34')
        size_col = 'size' if 'size' in trial_metrics.columns else 'blob_size'
        blob_sizes = sorted(trial_metrics[size_col].unique())
        
        for size in blob_sizes:
            results['questions'][size] = self._analyze_discrimination_for_size(
                trial_metrics, xcorr_results, size, size_col,
                rmse_threshold, correlation_threshold, lag_threshold_ms, alpha
            )
        
        # Compare across sizes
        results['comparison'] = self._compare_discrimination_across_sizes(
            trial_metrics, xcorr_results, blob_sizes, size_col, alpha
        )
        
        # Generate overall summary
        results['summary'] = self._generate_research_summary(results)
        
        return results
    
    def _analyze_discrimination_for_size(
        self,
        trial_metrics: pd.DataFrame,
        xcorr_results: pd.DataFrame,
        blob_size: str,
        size_col: str,
        rmse_threshold: float,
        correlation_threshold: float,
        lag_threshold_ms: float,
        alpha: float
    ) -> Dict[str, Any]:
        """Analyze discrimination ability for a specific blob size."""
        
        size_trials = trial_metrics[trial_metrics[size_col] == blob_size]
        xcorr_size_col = 'size' if 'size' in xcorr_results.columns else 'blob_size' if not xcorr_results.empty else 'size'
        size_xcorr = xcorr_results[xcorr_results[xcorr_size_col] == blob_size] if not xcorr_results.empty else pd.DataFrame()
        
        analysis = {
            'blob_size': blob_size,
            'n_trials': len(size_trials),
            'n_participants': size_trials['participant_id'].nunique(),
            'evidence': [],
            'metrics': {},
            'statistical_tests': {},
            'can_discriminate': False,
            'confidence': 'low',
            'answer': ''
        }
        
        if len(size_trials) == 0:
            analysis['answer'] = f"❓ Insufficient data for {blob_size} arcmin blob."
            return analysis
        
        evidence_score = 0
        max_evidence = 0
        
        # --- Evidence 1: RMSE Analysis ---
        rmse_values = size_trials['rmse'].dropna()
        if len(rmse_values) > 1:
            max_evidence += 2
            mean_rmse = rmse_values.mean()
            std_rmse = rmse_values.std()
            
            # One-sample t-test: Is mean RMSE significantly below threshold?
            t_stat, p_value = stats.ttest_1samp(rmse_values, rmse_threshold)
            rmse_below_threshold = mean_rmse < rmse_threshold and t_stat < 0
            
            analysis['metrics']['rmse'] = {
                'mean': mean_rmse,
                'std': std_rmse,
                'threshold': rmse_threshold,
                'below_threshold': rmse_below_threshold
            }
            analysis['statistical_tests']['rmse_vs_threshold'] = {
                'test': 'one-sample t-test',
                't_statistic': t_stat,
                'p_value': p_value if t_stat < 0 else 1.0,  # One-tailed
                'significant': p_value < alpha and t_stat < 0
            }
            
            if rmse_below_threshold and p_value < alpha:
                evidence_score += 2
                analysis['evidence'].append(
                    f"✅ RMSE ({mean_rmse:.1f}±{std_rmse:.1f}px) significantly below threshold "
                    f"({rmse_threshold}px), p={p_value:.4f}"
                )
            elif mean_rmse < rmse_threshold:
                evidence_score += 1
                analysis['evidence'].append(
                    f"⚠️ RMSE ({mean_rmse:.1f}±{std_rmse:.1f}px) below threshold but not significant, p={p_value:.4f}"
                )
            else:
                analysis['evidence'].append(
                    f"❌ RMSE ({mean_rmse:.1f}±{std_rmse:.1f}px) above threshold ({rmse_threshold}px)"
                )
        
        # --- Evidence 2: Correlation Analysis ---
        if not size_xcorr.empty and 'max_correlation' in size_xcorr.columns:
            corr_values = size_xcorr['max_correlation'].dropna()
            if len(corr_values) > 1:
                max_evidence += 2
                mean_corr = corr_values.mean()
                std_corr = corr_values.std()
                
                # One-sample t-test: Is mean correlation significantly above threshold?
                t_stat, p_value = stats.ttest_1samp(corr_values, correlation_threshold)
                corr_above_threshold = mean_corr > correlation_threshold and t_stat > 0
                
                analysis['metrics']['correlation'] = {
                    'mean': mean_corr,
                    'std': std_corr,
                    'threshold': correlation_threshold,
                    'above_threshold': corr_above_threshold
                }
                analysis['statistical_tests']['correlation_vs_threshold'] = {
                    'test': 'one-sample t-test',
                    't_statistic': t_stat,
                    'p_value': p_value if t_stat > 0 else 1.0,  # One-tailed
                    'significant': p_value < alpha and t_stat > 0
                }
                
                if corr_above_threshold and p_value < alpha:
                    evidence_score += 2
                    analysis['evidence'].append(
                        f"✅ Correlation ({mean_corr:.3f}±{std_corr:.3f}) significantly above threshold "
                        f"({correlation_threshold}), p={p_value:.4f}"
                    )
                elif mean_corr > correlation_threshold:
                    evidence_score += 1
                    analysis['evidence'].append(
                        f"⚠️ Correlation ({mean_corr:.3f}±{std_corr:.3f}) above threshold but not significant, p={p_value:.4f}"
                    )
                else:
                    analysis['evidence'].append(
                        f"❌ Correlation ({mean_corr:.3f}±{std_corr:.3f}) below threshold ({correlation_threshold})"
                    )
        
        # --- Evidence 3: Lag Analysis ---
        if not size_xcorr.empty and 'optimal_lag_ms' in size_xcorr.columns:
            lag_values = size_xcorr['optimal_lag_ms'].dropna().abs()
            if len(lag_values) > 1:
                max_evidence += 2
                mean_lag = lag_values.mean()
                std_lag = lag_values.std()
                
                # One-sample t-test: Is mean lag significantly below threshold?
                t_stat, p_value = stats.ttest_1samp(lag_values, lag_threshold_ms)
                lag_below_threshold = mean_lag < lag_threshold_ms and t_stat < 0
                
                analysis['metrics']['lag'] = {
                    'mean': mean_lag,
                    'std': std_lag,
                    'threshold': lag_threshold_ms,
                    'below_threshold': lag_below_threshold
                }
                analysis['statistical_tests']['lag_vs_threshold'] = {
                    'test': 'one-sample t-test',
                    't_statistic': t_stat,
                    'p_value': p_value if t_stat < 0 else 1.0,
                    'significant': p_value < alpha and t_stat < 0
                }
                
                if lag_below_threshold and p_value < alpha:
                    evidence_score += 2
                    analysis['evidence'].append(
                        f"✅ Response lag ({mean_lag:.1f}±{std_lag:.1f}ms) significantly below threshold "
                        f"({lag_threshold_ms}ms), p={p_value:.4f}"
                    )
                elif mean_lag < lag_threshold_ms:
                    evidence_score += 1
                    analysis['evidence'].append(
                        f"⚠️ Response lag ({mean_lag:.1f}±{std_lag:.1f}ms) below threshold but not significant, p={p_value:.4f}"
                    )
                else:
                    analysis['evidence'].append(
                        f"❌ Response lag ({mean_lag:.1f}±{std_lag:.1f}ms) above threshold ({lag_threshold_ms}ms)"
                    )
        
        # --- Determine discrimination ability ---
        if max_evidence > 0:
            evidence_ratio = evidence_score / max_evidence
            
            if evidence_ratio >= 0.7:
                analysis['can_discriminate'] = True
                analysis['confidence'] = 'high'
                analysis['answer'] = f"✅ **YES** - Participants CAN discriminate the {blob_size} arcmin blob (high confidence)."
            elif evidence_ratio >= 0.5:
                analysis['can_discriminate'] = True
                analysis['confidence'] = 'moderate'
                analysis['answer'] = f"⚠️ **LIKELY YES** - Participants can probably discriminate the {blob_size} arcmin blob (moderate confidence)."
            elif evidence_ratio >= 0.3:
                analysis['can_discriminate'] = False
                analysis['confidence'] = 'moderate'
                analysis['answer'] = f"⚠️ **LIKELY NO** - Participants probably cannot discriminate the {blob_size} arcmin blob (moderate confidence)."
            else:
                analysis['can_discriminate'] = False
                analysis['confidence'] = 'high'
                analysis['answer'] = f"❌ **NO** - Participants CANNOT discriminate the {blob_size} arcmin blob (high confidence)."
            
            analysis['evidence_score'] = evidence_score
            analysis['max_evidence'] = max_evidence
            analysis['evidence_ratio'] = evidence_ratio
        
        return analysis
    
    def _compare_discrimination_across_sizes(
        self,
        trial_metrics: pd.DataFrame,
        xcorr_results: pd.DataFrame,
        blob_sizes: List[str],
        size_col: str,
        alpha: float
    ) -> Dict[str, Any]:
        """Compare discrimination performance across blob sizes."""
        
        comparison = {
            'rmse_anova': None,
            'correlation_anova': None,
            'size_ranking': [],
            'pairwise_comparisons': [],
            'interpretation': ''
        }
        
        if len(blob_sizes) < 2:
            comparison['interpretation'] = "Need at least 2 blob sizes for comparison."
            return comparison
        
        # Determine xcorr size column
        xcorr_size_col = 'size' if 'size' in xcorr_results.columns else 'blob_size' if not xcorr_results.empty else 'size'
        
        # Prepare groups for ANOVA
        rmse_groups = []
        corr_groups = []
        
        for size in blob_sizes:
            size_data = trial_metrics[trial_metrics[size_col] == size]['rmse'].dropna()
            if len(size_data) > 0:
                rmse_groups.append(size_data.values)
            
            if not xcorr_results.empty:
                size_corr = xcorr_results[xcorr_results[xcorr_size_col] == size]['max_correlation'].dropna()
                if len(size_corr) > 0:
                    corr_groups.append(size_corr.values)
        
        # RMSE ANOVA
        if len(rmse_groups) >= 2 and all(len(g) > 1 for g in rmse_groups):
            f_stat, p_value = stats.f_oneway(*rmse_groups)
            comparison['rmse_anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < alpha
            }
        
        # Correlation ANOVA  
        if len(corr_groups) >= 2 and all(len(g) > 1 for g in corr_groups):
            f_stat, p_value = stats.f_oneway(*corr_groups)
            comparison['correlation_anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < alpha
            }
        
        # Rank sizes by mean RMSE (lower is better)
        size_means = []
        for size in blob_sizes:
            mean_rmse = trial_metrics[trial_metrics[size_col] == size]['rmse'].mean()
            size_means.append((size, mean_rmse))
        
        comparison['size_ranking'] = sorted(size_means, key=lambda x: x[1])
        
        # Generate interpretation
        interp_parts = []
        
        if comparison['rmse_anova'] and comparison['rmse_anova']['significant']:
            interp_parts.append(
                f"RMSE differs significantly across blob sizes (F={comparison['rmse_anova']['f_statistic']:.2f}, "
                f"p={comparison['rmse_anova']['p_value']:.4f})."
            )
        
        if comparison['correlation_anova'] and comparison['correlation_anova']['significant']:
            interp_parts.append(
                f"Correlation differs significantly across blob sizes (F={comparison['correlation_anova']['f_statistic']:.2f}, "
                f"p={comparison['correlation_anova']['p_value']:.4f})."
            )
        
        if comparison['size_ranking']:
            best = comparison['size_ranking'][0]
            worst = comparison['size_ranking'][-1]
            interp_parts.append(
                f"Best tracking performance: {best[0]} arcmin (mean RMSE={best[1]:.1f}px). "
                f"Worst: {worst[0]} arcmin (mean RMSE={worst[1]:.1f}px)."
            )
        
        comparison['interpretation'] = " ".join(interp_parts) if interp_parts else "No significant differences found."
        
        return comparison
    
    def _generate_research_summary(self, results: Dict[str, Any]) -> str:
        """Generate a narrative summary answering the research questions."""
        
        lines = ["## Research Questions Summary\n"]
        
        # Answer each question
        q_num = 1
        for size, analysis in sorted(results['questions'].items()):
            lines.append(f"### Q{q_num}: Can participants discriminate the {size}-arcmin blob?\n")
            lines.append(f"**{analysis['answer']}**\n")
            
            if analysis['evidence']:
                lines.append("\n**Evidence:**")
                for ev in analysis['evidence']:
                    lines.append(f"- {ev}")
            
            lines.append(f"\n*Based on {analysis['n_trials']} trials from {analysis['n_participants']} participants.*\n")
            q_num += 1
        
        # Size comparison
        if results['comparison']['interpretation']:
            lines.append("\n### Cross-Size Comparison\n")
            lines.append(results['comparison']['interpretation'])
        
        # Overall conclusion
        lines.append("\n### Overall Conclusion\n")
        
        discriminable = [s for s, a in results['questions'].items() if a['can_discriminate']]
        non_discriminable = [s for s, a in results['questions'].items() if not a['can_discriminate']]
        
        if discriminable:
            lines.append(f"Participants **CAN** discriminate: {', '.join(map(str, discriminable))} arcmin blobs.")
        if non_discriminable:
            lines.append(f"Participants **CANNOT** discriminate: {', '.join(map(str, non_discriminable))} arcmin blobs.")
        
        return "\n".join(lines)
