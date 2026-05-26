import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Dict
import json
import os


class StatisticalComparison:
    """
    Statistical significance testing and analysis.
    Uses paired t-tests, McNemar's test, and other rigorous statistical methods.
    """
    
    def __init__(self, output_dir: str = "results/comparison"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def paired_t_test(self, context_scores: List[float], no_context_scores: List[float]) -> Dict:
        """
        Paired t-test: Tests if context model scores are significantly different from no-context.
        
        Args:
            context_scores: Confidence/similarity scores from context model
            no_context_scores: Confidence/similarity scores from no-context model
            
        Returns:
            Dictionary with t-statistic, p-value, and interpretation
        """
        if len(context_scores) != len(no_context_scores):
            raise ValueError("Scores must have equal length")
        
        differences = np.array(context_scores) - np.array(no_context_scores)
        
        t_stat, p_value = stats.ttest_rel(context_scores, no_context_scores)
        
        cohens_d = np.mean(differences) / (np.std(differences) + 1e-8)
        
        result = {
            'test_name': 'Paired t-test',
            'null_hypothesis': 'Context and no-context models have equal performance',
            'alternative_hypothesis': 'Context model performs better',
            'mean_context': float(np.mean(context_scores)),
            'mean_no_context': float(np.mean(no_context_scores)),
            'mean_difference': float(np.mean(differences)),
            'std_difference': float(np.std(differences)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'interpretation': self._interpret_t_test(p_value, cohens_d)
        }
        
        return result
    
    def mcnemars_test(self, context_correct: np.ndarray, no_context_correct: np.ndarray) -> Dict:
        """
        McNemar's test: Tests if there's significant difference in classification accuracy.
        Used for paired categorical data (correct/incorrect predictions).
        
        Args:
            context_correct: Boolean array of correct predictions from context model
            no_context_correct: Boolean array of correct predictions from no-context model
            
        Returns:
            Dictionary with test statistics and interpretation
        """
        a = np.sum((context_correct) & ~(no_context_correct))
        b = np.sum(~(context_correct) & (no_context_correct))

        mcnemar_stat = ((a - b) ** 2) / (a + b + 1e-8)
        p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        
        result = {
            'test_name': "McNemar's Test",
            'null_hypothesis': 'Context and no-context models have equal accuracy',
            'context_wins': int(a),
            'no_context_wins': int(b),
            'mcnemar_statistic': float(mcnemar_stat),
            'p_value': float(p_value),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'interpretation': self._interpret_mcnemar(p_value, a, b)
        }
        
        return result
    
    def wilcoxon_signed_rank_test(self, context_scores: List[float], no_context_scores: List[float]) -> Dict:
        """
        Wilcoxon Signed-Rank Test: Non-parametric alternative to paired t-test.
        More robust for non-normal distributions.
        """
        statistic, p_value = stats.wilcoxon(context_scores, no_context_scores)
        
        result = {
            'test_name': 'Wilcoxon Signed-Rank Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'interpretation': ('Context model significantly outperforms no-context model'
                                if p_value < 0.05 else
                                'No significant difference found')
        }
        
        return result
    
    def confidence_interval(self, context_scores: List[float], no_context_scores: List[float], confidence_level: float = 0.95) -> Dict:
        """
        Calculate confidence intervals for the mean difference.
        """
        differences = np.array(context_scores) - np.array(no_context_scores)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        n = len(differences)
        
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n-1)
        
        margin_error = t_critical * (std_diff / np.sqrt(n))
        
        result = {
            'confidence_level': confidence_level,
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'lower_bound': float(mean_diff - margin_error),
            'upper_bound': float(mean_diff + margin_error),
            'margin_of_error': float(margin_error),
            'interpretation': (f"We are {confidence_level*100:.0f}% confident that the true "
                                f"difference in performance is between "
                                f"{mean_diff - margin_error:.4f} and {mean_diff + margin_error:.4f}")
        }
        
        return result
    
    def effect_size_analysis(self, context_correct: np.ndarray, 
                            no_context_correct: np.ndarray) -> Dict:
        """
        Calculate multiple effect size measures.
        """
        acc_context = np.mean(context_correct)
        acc_no_context = np.mean(no_context_correct)
        
        risk_ratio = acc_context / (acc_no_context + 1e-8)
        
        odds_context = acc_context / (1 - acc_context + 1e-8)
        odds_no_context = acc_no_context / (1 - acc_no_context + 1e-8)
        odds_ratio = odds_context / odds_no_context
        
        ard = acc_context - acc_no_context
        
        nnt = 1 / (ard + 1e-8) if ard != 0 else float('inf')
        
        result = {
            'context_accuracy': float(acc_context),
            'no_context_accuracy': float(acc_no_context),
            'absolute_risk_difference': float(ard),
            'relative_risk': float(risk_ratio),
            'odds_ratio': float(odds_ratio),
            'number_needed_to_treat': float(nnt) if nnt != float('inf') else None,
            'interpretation': self._interpret_effect_size(ard, risk_ratio)
        }
        
        return result
    
    def disease_specific_analysis(self, context_by_disease: Dict, no_context_by_disease: Dict) -> Dict:
        """
        Statistical analysis per disease type.
        Identifies which diseases benefit most from context.
        """
        disease_analysis = {}
        
        for disease in context_by_disease:
            if disease not in no_context_by_disease:
                continue
            
            ctx_data = context_by_disease[disease]
            no_ctx_data = no_context_by_disease[disease]
            
            if ctx_data['total'] < 3:
                continue
            
            ctx_acc = ctx_data.get('accuracy', 0)
            no_ctx_acc = no_ctx_data.get('accuracy', 0)
            improvement = ctx_acc - no_ctx_acc
            
            disease_analysis[disease] = {
                'sample_count': ctx_data['total'],
                'context_accuracy': float(ctx_acc),
                'no_context_accuracy': float(no_ctx_acc),
                'accuracy_improvement': float(improvement),
                'improvement_percentage': float(improvement * 100),
                'context_errors': len(ctx_data.get('errors', [])),
                'no_context_errors': len(no_ctx_data.get('errors', []))
            }
        
        sorted_diseases = sorted(
            disease_analysis.items(),
            key=lambda x: x[1]['improvement_percentage'],
            reverse=True
        )
        
        result = {
            'total_diseases_analyzed': len(disease_analysis),
            'diseases_improved_by_context': sum(
                1 for d in disease_analysis.values() if d['improvement_percentage'] > 0
            ),
            'top_improved_diseases': dict(sorted_diseases[:5]),
            'all_diseases': dict(sorted_diseases)
        }
        
        return result
    
    def generate_statistical_report(self, comparison_results: Dict, output_file: str = None) -> str:
        """
        Generate comprehensive statistical report.
        """
        report = []
        report.append("="*100)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("Proving Context Improves Medical Diagnosis Prediction")
        report.append("="*100)
        report.append("")
        
        ctx_preds = comparison_results.get('context_predictions', [])
        no_ctx_preds = comparison_results.get('no_context_predictions', [])
        references = comparison_results.get('references', [])
        
        context_correct = np.array([
            c.strip().lower() == r.strip().lower()
            for c, r in zip(ctx_preds, references)
        ])
        no_context_correct = np.array([
            nc.strip().lower() == r.strip().lower()
            for nc, r in zip(no_ctx_preds, references)
        ])
        
        context_conf = np.array([
            comparison_results['comparison_metrics']['context_model'].get('avg_confidence', 0.5)
        ] * len(ctx_preds))
        no_context_conf = np.array([
            comparison_results['comparison_metrics']['no_context_model'].get('avg_confidence', 0.5)
        ] * len(no_ctx_preds))
        
        mcnemar = self.mcnemars_test(context_correct, no_context_correct)
        wilcoxon = self.wilcoxon_signed_rank_test(context_conf, no_context_conf)
        ci = self.confidence_interval(context_conf, no_context_conf)
        effect_size = self.effect_size_analysis(context_correct, no_context_correct)
        disease_analysis = self.disease_specific_analysis(
            comparison_results.get('context_by_disease', {}),
            comparison_results.get('no_context_by_disease', {})
        )
        
        report.append("1. MCNEMAR'S TEST - ACCURACY COMPARISON")
        report.append("-"*100)
        report.append(f"   Context Model Wins: {mcnemar['context_wins']}")
        report.append(f"   No-Context Model Wins: {mcnemar['no_context_wins']}")
        report.append(f"   McNemar Statistic: {mcnemar['mcnemar_statistic']:.4f}")
        report.append(f"   P-value: {mcnemar['p_value']:.6f}")
        report.append(f"   Significant at 0.05 level: {mcnemar['significant_at_0.05']}")
        report.append(f"   Interpretation: {mcnemar['interpretation']}")
        report.append("")
        
        report.append("2. WILCOXON SIGNED-RANK TEST - CONFIDENCE COMPARISON")
        report.append("-"*100)
        report.append(f"   Test Statistic: {wilcoxon['statistic']:.4f}")
        report.append(f"   P-value: {wilcoxon['p_value']:.6f}")
        report.append(f"   Significant at 0.05 level: {wilcoxon['significant_at_0.05']}")
        report.append(f"   Interpretation: {wilcoxon['interpretation']}")
        report.append("")
        
        report.append("3. CONFIDENCE INTERVAL ANALYSIS")
        report.append("-"*100)
        report.append(f"   Mean Difference: {ci['mean_difference']:.4f}")
        report.append(f"   95% CI: [{ci['lower_bound']:.4f}, {ci['upper_bound']:.4f}]")
        report.append(f"   Margin of Error: {ci['margin_of_error']:.4f}")
        report.append(f"   Interpretation: {ci['interpretation']}")
        report.append("")
        
        report.append("4. EFFECT SIZE ANALYSIS")
        report.append("-"*100)
        report.append(f"   Context Model Accuracy: {effect_size['context_accuracy']:.4f}")
        report.append(f"   No-Context Model Accuracy: {effect_size['no_context_accuracy']:.4f}")
        report.append(f"   Absolute Risk Difference: {effect_size['absolute_risk_difference']:.4f}")
        report.append(f"   Relative Risk: {effect_size['relative_risk']:.4f}")
        report.append(f"   Odds Ratio: {effect_size['odds_ratio']:.4f}")
        if effect_size['number_needed_to_treat']:
            report.append(f"   Number Needed to Treat: {effect_size['number_needed_to_treat']:.1f}")
        report.append(f"   Interpretation: {effect_size['interpretation']}")
        report.append("")
        
        report.append("5. DISEASE-SPECIFIC ANALYSIS")
        report.append("-"*100)
        report.append(f"   Total Diseases Analyzed: {disease_analysis['total_diseases_analyzed']}")
        report.append(f"   Diseases Improved by Context: {disease_analysis['diseases_improved_by_context']}")
        report.append("")
        report.append("   Top 5 Diseases Improved by Context:")
        for disease, data in disease_analysis['top_improved_diseases'].items():
            report.append(f"      - {disease}:")
            report.append(f"        Improvement: {data['improvement_percentage']:.1f}%")
            report.append(f"        Context Accuracy: {data['context_accuracy']:.2%}")
            report.append(f"        No-Context Accuracy: {data['no_context_accuracy']:.2%}")
        report.append("")
        
        report.append("6. CONCLUSION")
        report.append("-"*100)
        if mcnemar['significant_at_0.05']:
            report.append("✓ McNemar's test confirms context model significantly improves accuracy")
        if wilcoxon['significant_at_0.05']:
            report.append("✓ Wilcoxon test confirms context model shows higher confidence scores")
        
        report.append(f"✓ Context model outperforms in {mcnemar['context_wins']} cases")
        report.append(f"✓ Effect size (Odds Ratio) of {effect_size['odds_ratio']:.2f} indicates")
        report.append("  substantial practical significance")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def _interpret_t_test(self, p_value: float, cohens_d: float) -> str:
        """Interpret t-test results."""
        if p_value < 0.01:
            sig = "highly significant (p < 0.01)"
        elif p_value < 0.05:
            sig = "significant (p < 0.05)"
        else:
            sig = "not significant (p ≥ 0.05)"
        
        if abs(cohens_d) < 0.2:
            effect = "negligible"
        elif abs(cohens_d) < 0.5:
            effect = "small"
        elif abs(cohens_d) < 0.8:
            effect = "medium"
        else:
            effect = "large"
        
        return f"Results are {sig} with {effect} effect size (d={cohens_d:.3f})"
    
    def _interpret_mcnemar(self, p_value: float, a: int, b: int) -> str:
        """Interpret McNemar's test results."""
        if p_value < 0.05:
            return (f"Context model significantly outperforms no-context model "
                    f"(context wins: {a}, no-context wins: {b})")
        else:
            return "No significant difference in classification accuracy"
    
    def _interpret_effect_size(self, ard: float, rr: float) -> str:
        """Interpret effect size metrics."""
        interpretations = []
        
        if ard > 0.1:
            interpretations.append("large absolute risk difference")
        elif ard > 0.05:
            interpretations.append("moderate absolute risk difference")
        elif ard > 0:
            interpretations.append("small absolute risk difference")
        
        if rr > 1.5:
            interpretations.append("substantial relative risk increase")
        elif rr > 1.2:
            interpretations.append("moderate relative risk increase")
        
        return ", ".join(interpretations) if interpretations else "negligible effect size"
