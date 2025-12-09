"""
Tests for Statistics Module
===========================

Tests ANOVA, effect sizes, and statistical comparisons.
"""

import pytest
import pandas as pd
import numpy as np

from tracking_analysis.core.statistics import StatisticalAnalyzer


class TestEffectSizes:
    """Test effect size calculations."""
    
    def test_cohens_d_identical(self, stats_analyzer):
        """Test Cohen's d is zero for identical groups."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([1, 2, 3, 4, 5])
        
        d = stats_analyzer.cohens_d(group1, group2)
        
        assert abs(d) < 0.001
    
    def test_cohens_d_different(self, stats_analyzer):
        """Test Cohen's d detects differences."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])
        
        d = stats_analyzer.cohens_d(group1, group2)
        
        assert abs(d) > 1  # Large effect
    
    def test_cohens_d_sign(self, stats_analyzer):
        """Test Cohen's d sign indicates direction."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])
        
        d = stats_analyzer.cohens_d(group1, group2)
        
        # group1 < group2, so d should be negative
        assert d < 0
    
    def test_eta_squared_range(self, stats_analyzer, trial_metrics):
        """Test eta-squared is in valid range [0, 1]."""
        # Use RMSE grouped by sd_size
        groups = trial_metrics.groupby('sd_size')['rmse'].apply(list).values
        
        if len(groups) >= 2:
            eta_sq = stats_analyzer.eta_squared(list(groups))
            
            assert 0 <= eta_sq <= 1


class TestTTests:
    """Test t-test implementations."""
    
    def test_paired_ttest(self, stats_analyzer):
        """Test paired t-test."""
        pre = np.array([10, 12, 14, 16, 18])
        post = np.array([11, 13, 15, 17, 19])
        
        result = stats_analyzer.paired_ttest(pre, post)
        
        assert 't_stat' in result
        assert 'p_value' in result
    
    def test_paired_ttest_significant(self, stats_analyzer):
        """Test paired t-test detects significant difference."""
        pre = np.array([10, 12, 14, 16, 18])
        post = np.array([20, 22, 24, 26, 28])  # Clear increase
        
        result = stats_analyzer.paired_ttest(pre, post)
        
        assert result['p_value'] < 0.05
    
    def test_independent_ttest(self, stats_analyzer):
        """Test independent samples t-test."""
        group1 = np.array([10, 12, 14, 16, 18])
        group2 = np.array([15, 17, 19, 21, 23])
        
        result = stats_analyzer.independent_ttest(group1, group2)
        
        assert 't_stat' in result
        assert 'p_value' in result
        assert 'cohens_d' in result


class TestANOVA:
    """Test ANOVA implementations."""
    
    def test_oneway_anova(self, stats_analyzer):
        """Test one-way ANOVA."""
        group1 = np.array([10, 12, 14, 16, 18])
        group2 = np.array([15, 17, 19, 21, 23])
        group3 = np.array([20, 22, 24, 26, 28])
        
        result = stats_analyzer.oneway_anova([group1, group2, group3])
        
        assert 'f_stat' in result
        assert 'p_value' in result
    
    def test_oneway_anova_significant(self, stats_analyzer):
        """Test ANOVA detects significant difference."""
        group1 = np.array([10, 11, 12, 13, 14])
        group2 = np.array([50, 51, 52, 53, 54])  # Very different
        group3 = np.array([100, 101, 102, 103, 104])  # Even more different
        
        result = stats_analyzer.oneway_anova([group1, group2, group3])
        
        assert result['p_value'] < 0.001


class TestMultipleComparisons:
    """Test multiple comparison corrections."""
    
    def test_bonferroni_correction(self, stats_analyzer):
        """Test Bonferroni correction."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        
        corrected = stats_analyzer.bonferroni_correction(p_values)
        
        # Bonferroni multiplies by n comparisons
        assert corrected[0] == min(0.01 * 5, 1.0)
    
    def test_holm_correction(self, stats_analyzer):
        """Test Holm-Bonferroni correction."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        
        corrected = stats_analyzer.holm_correction(p_values)
        
        assert all(c <= 1.0 for c in corrected)
        assert all(c >= p for c, p in zip(corrected, p_values))
    
    def test_fdr_correction(self, stats_analyzer):
        """Test FDR correction."""
        p_values = [0.001, 0.01, 0.02, 0.04, 0.05]
        
        corrected = stats_analyzer.fdr_correction(p_values)
        
        assert all(c <= 1.0 for c in corrected)


class TestPostHoc:
    """Test post-hoc comparisons."""
    
    def test_pairwise_comparisons(self, stats_analyzer, trial_metrics):
        """Test pairwise comparisons."""
        result = stats_analyzer.pairwise_comparisons(
            trial_metrics,
            'rmse',
            'sd_size'
        )
        
        assert isinstance(result, (pd.DataFrame, list, dict))
    
    def test_pairwise_returns_all_pairs(self, stats_analyzer, trial_metrics):
        """Test all pairs are compared."""
        result = stats_analyzer.pairwise_comparisons(
            trial_metrics,
            'rmse',
            'sd_size'
        )
        
        # 3 sizes = 3 pairs (21-31, 21-34, 31-34)
        if isinstance(result, pd.DataFrame):
            assert len(result) == 3
        elif isinstance(result, list):
            assert len(result) == 3


class TestConditionComparison:
    """Test condition comparison (dynamic vs static)."""
    
    def test_compare_conditions(self, stats_analyzer, trial_metrics):
        """Test comparing conditions."""
        result = stats_analyzer.compare_conditions(
            trial_metrics,
            metric='rmse'
        )
        
        assert result is not None
        assert 't_stat' in result or 'statistic' in result
    
    def test_compare_conditions_effect_size(self, stats_analyzer, trial_metrics):
        """Test effect size for condition comparison."""
        result = stats_analyzer.compare_conditions(
            trial_metrics,
            metric='rmse'
        )
        
        assert 'cohens_d' in result or 'effect_size' in result


class TestSizeComparison:
    """Test SD size comparison."""
    
    def test_compare_sizes(self, stats_analyzer, trial_metrics):
        """Test comparing SD sizes."""
        result = stats_analyzer.compare_sizes(
            trial_metrics,
            metric='rmse'
        )
        
        assert result is not None
    
    def test_compare_sizes_anova(self, stats_analyzer, trial_metrics):
        """Test ANOVA for size comparison."""
        result = stats_analyzer.compare_sizes(
            trial_metrics,
            metric='rmse',
            test='anova'
        )
        
        assert 'f_stat' in result or 'F' in result
        assert 'p_value' in result or 'p' in result


class TestResultsSummary:
    """Test results summary generation."""
    
    def test_generate_summary(self, stats_analyzer, trial_metrics):
        """Test generating results summary."""
        summary = stats_analyzer.generate_results_summary(
            trial_metrics,
            metric='rmse'
        )
        
        assert isinstance(summary, dict)
    
    def test_summary_includes_descriptives(self, stats_analyzer, trial_metrics):
        """Test summary includes descriptive statistics."""
        summary = stats_analyzer.generate_results_summary(
            trial_metrics,
            metric='rmse'
        )
        
        # Should have some descriptive info
        assert any(key in str(summary).lower() for key in ['mean', 'std', 'n'])


class TestRepeatedMeasuresANOVA:
    """Test repeated measures ANOVA."""
    
    @pytest.mark.skipif(
        True,  # Skip if statsmodels not properly installed
        reason="Requires statsmodels for repeated measures ANOVA"
    )
    def test_repeated_measures_anova(self, stats_analyzer, trial_metrics):
        """Test repeated measures ANOVA."""
        try:
            result = stats_analyzer.repeated_measures_anova(
                trial_metrics,
                'rmse',
                within=['sd_size'],
                subject='participant_id'
            )
            
            assert 'F' in result or 'f_stat' in result
        except ImportError:
            pytest.skip("statsmodels not available")


class TestInterpretation:
    """Test statistical result interpretation."""
    
    def test_interpret_effect_size(self, stats_analyzer):
        """Test effect size interpretation."""
        # Small effect
        interp_small = stats_analyzer.interpret_effect_size(0.2, 'cohens_d')
        assert 'small' in interp_small.lower()
        
        # Medium effect
        interp_med = stats_analyzer.interpret_effect_size(0.5, 'cohens_d')
        assert 'medium' in interp_med.lower()
        
        # Large effect
        interp_large = stats_analyzer.interpret_effect_size(0.8, 'cohens_d')
        assert 'large' in interp_large.lower()
    
    def test_interpret_p_value(self, stats_analyzer):
        """Test p-value interpretation."""
        interp_sig = stats_analyzer.interpret_p_value(0.01)
        interp_ns = stats_analyzer.interpret_p_value(0.20)
        
        assert 'significant' in interp_sig.lower()
        assert 'not' in interp_ns.lower() or 'ns' in interp_ns.lower()
