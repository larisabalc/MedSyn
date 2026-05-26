import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
from difflib import SequenceMatcher
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class DiagnosisEvaluator:
    """
    Comprehensive evaluator for diagnosis models with advanced metrics.
    """
    
    def __init__(self, use_semantic_similarity=True):
        """
        Initialize evaluator with optional semantic similarity model.
        
        Args:
            use_semantic_similarity: If True, loads sentence transformer for semantic comparison
        """
        self.use_semantic_similarity = use_semantic_similarity
        self.semantic_model = None
        
        if use_semantic_similarity:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load semantic model: {e}. Continuing without semantic similarity.")
                self.use_semantic_similarity = False
    
    def compute_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """Exact string match (case-insensitive)."""
        matches = sum(
            1 for pred, ref in zip(predictions, references)
            if pred.strip().lower() == ref.strip().lower()
        )
        return matches / len(predictions) if predictions else 0.0
    
    def compute_partial_match(self, predictions: List[str], references: List[str]) -> float:
        """
        Partial match: measures string similarity using SequenceMatcher.
        Higher threshold (0.8+) means very similar, allowing for variations like "Flu" vs "Influenza".
        """
        similarities = [
            SequenceMatcher(None, pred.lower(), ref.lower()).ratio()
            for pred, ref in zip(predictions, references)
        ]
        partial_matches = sum(1 for sim in similarities if sim >= 0.8)
        return partial_matches / len(predictions) if predictions else 0.0
    
    def compute_semantic_similarity(self, predictions: List[str], references: List[str]) -> float:
        """
        Semantic similarity using sentence embeddings.
        Measures if prediction is semantically close even if not exact match.
        """
        if not self.use_semantic_similarity or not self.semantic_model:
            return None
        
        try:
            pred_embeddings = self.semantic_model.encode(predictions, convert_to_tensor=True)
            ref_embeddings = self.semantic_model.encode(references, convert_to_tensor=True)
            
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                sim = torch.nn.functional.cosine_similarity(pred_emb.unsqueeze(0), ref_emb.unsqueeze(0))
                similarities.append(sim.item())
            
            return np.mean(similarities)
        except Exception as e:
            print(f"Warning: Semantic similarity computation failed: {e}")
            return None
    
    def compute_top_k_accuracy(self, predictions: List[str], references: List[str], k: int = 3) -> float:
        """
        Top-k accuracy: Would the correct answer be in top k predictions?
        For ranking scenarios. Here we measure string similarity ranking.
        """
        correct = 0
        for pred, ref in zip(predictions, references):
            if pred.strip().lower() == ref.strip().lower():
                correct += 1
            elif SequenceMatcher(None, pred.lower(), ref.lower()).ratio() >= 0.7:
                correct += 1
        
        return correct / len(predictions) if predictions else 0.0
    
    def compute_confidence_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Compute confidence-based metrics.
        Returns metrics about prediction confidence and uncertainty.
        """
        similarities = [
            SequenceMatcher(None, pred.lower(), ref.lower()).ratio()
            for pred, ref in zip(predictions, references)
        ]
        
        correct_mask = [sim >= 0.9 for sim in similarities]
        
        metrics = {
            'avg_confidence': np.mean(similarities),
            'std_confidence': np.std(similarities),
            'confidence_correct': np.mean([s for s, c in zip(similarities, correct_mask) if c]),
            'confidence_incorrect': np.mean([s for s, c in zip(similarities, correct_mask) if not c]),
            'max_confidence': np.max(similarities),
            'min_confidence': np.min(similarities),
        }
        
        return metrics
    
    def compute_all_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Compute all available metrics.
        
        Args:
            predictions: List of predicted disease names
            references: List of ground truth disease names
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'exact_match': self.compute_exact_match(predictions, references),
            'partial_match': self.compute_partial_match(predictions, references),
            'top_k_accuracy': self.compute_top_k_accuracy(predictions, references),
        }
        
        semantic_sim = self.compute_semantic_similarity(predictions, references)
        if semantic_sim is not None:
            metrics['semantic_similarity'] = semantic_sim
        
        confidence = self.compute_confidence_metrics(predictions, references)
        metrics.update(confidence)
        
        return metrics
    
    def analyze_error_types(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Categorize errors into types for deeper analysis.
        """
        errors = {
            'total_predictions': len(predictions),
            'correct': 0,
            'incorrect': 0,
            'off_by_character_case': 0,
            'similar_disease': 0,  # e.g., Flu vs Influenza
            'completely_wrong': 0,
            'error_details': []
        }
        
        for pred, ref in zip(predictions, references):
            pred_lower = pred.strip().lower()
            ref_lower = ref.strip().lower()
            
            if pred_lower == ref_lower:
                errors['correct'] += 1
            else:
                errors['incorrect'] += 1
                
                sim = SequenceMatcher(None, pred_lower, ref_lower).ratio()
                
                if sim >= 0.8:
                    errors['similar_disease'] += 1
                    error_type = 'similar_disease'
                elif sim >= 0.5:
                    errors['off_by_character_case'] += 1
                    error_type = 'partial_similarity'
                else:
                    errors['completely_wrong'] += 1
                    error_type = 'completely_wrong'
                
                errors['error_details'].append({
                    'predicted': pred,
                    'reference': ref,
                    'similarity': sim,
                    'error_type': error_type
                })
        
        errors['accuracy'] = errors['correct'] / errors['total_predictions']
        return errors
    
    def compare_models(self, context_predictions: List[str], no_context_predictions: List[str], references: List[str]) -> Dict:
        """
        Direct comparison between context and no-context models.
        Shows where context model outperforms.
        
        Args:
            context_predictions: Predictions from context model
            no_context_predictions: Predictions from no-context model
            references: Ground truth labels
            
        Returns:
            Comparison metrics showing context advantage
        """
        context_metrics = self.compute_all_metrics(context_predictions, references)
        no_context_metrics = self.compute_all_metrics(no_context_predictions, references)
        
        comparison = {
            'context_model': context_metrics,
            'no_context_model': no_context_metrics,
            'improvements': {}
        }
        
        for key in context_metrics:
            if isinstance(context_metrics[key], (int, float)):
                improvement = context_metrics[key] - no_context_metrics.get(key, 0)
                improvement_pct = (improvement / (no_context_metrics.get(key, 0.001))) * 100 if no_context_metrics.get(key, 0) != 0 else 0
                
                comparison['improvements'][key] = {
                    'absolute': round(improvement, 4),
                    'percentage': round(improvement_pct, 2)
                }
        
        context_correct = sum(
            1 for ctx, no_ctx, ref in zip(context_predictions, no_context_predictions, references)
            if ctx.strip().lower() == ref.strip().lower() and no_ctx.strip().lower() != ref.strip().lower()
        )
        
        no_context_correct = sum(
            1 for ctx, no_ctx, ref in zip(context_predictions, no_context_predictions, references)
            if ctx.strip().lower() != ref.strip().lower() and no_ctx.strip().lower() == ref.strip().lower()
        )
        
        both_correct = sum(
            1 for ctx, no_ctx, ref in zip(context_predictions, no_context_predictions, references)
            if ctx.strip().lower() == ref.strip().lower() and no_ctx.strip().lower() == ref.strip().lower()
        )
        
        comparison['head_to_head'] = {
            'context_model_wins': context_correct,
            'no_context_model_wins': no_context_correct,
            'both_correct': both_correct,
            'both_wrong': len(references) - context_correct - no_context_correct - both_correct
        }
        
        return comparison
    
    def analyze_by_disease(self, predictions: List[str], references: List[str], disease_groups: List[str] = None) -> Dict:
        """
        Analyze performance broken down by disease type.
        Shows which diseases benefit most from context.
        
        Args:
            predictions: Model predictions
            references: Ground truth labels
            disease_groups: Optional grouping of diseases
            
        Returns:
            Performance metrics per disease
        """
        disease_performance = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': []})
        
        for pred, ref in zip(predictions, references):
            disease_performance[ref]['total'] += 1
            if pred.strip().lower() == ref.strip().lower():
                disease_performance[ref]['correct'] += 1
            else:
                disease_performance[ref]['errors'].append(pred)
        
        for disease in disease_performance:
            if disease_performance[disease]['total'] > 0:
                disease_performance[disease]['accuracy'] = (
                    disease_performance[disease]['correct'] / disease_performance[disease]['total']
                )
        
        return dict(disease_performance)
    
    def generate_report(self, context_predictions: List[str], no_context_predictions: List[str], references: List[str], output_file: str = None) -> str:
        """
        Generate comprehensive evaluation report.
        """
        comparison = self.compare_models(context_predictions, no_context_predictions, references)
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE MODEL COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append("OVERALL METRICS COMPARISON")
        report.append("-" * 80)
        report.append(f"{'Metric':<30} {'Context Model':<20} {'No-Context Model':<20} {'Improvement':<15}")
        report.append("-" * 80)
        
        for metric_name, values in comparison['improvements'].items():
            context_val = comparison['context_model'].get(metric_name, 'N/A')
            no_context_val = comparison['no_context_model'].get(metric_name, 'N/A')
            improvement = f"+{values['absolute']:.4f} ({values['percentage']:.2f}%)"
            
            if isinstance(context_val, (int, float)):
                report.append(f"{metric_name:<30} {context_val:<20.4f} {no_context_val:<20.4f} {improvement:<15}")
        
        report.append("")
        report.append("HEAD-TO-HEAD COMPARISON")
        report.append("-" * 80)
        h2h = comparison['head_to_head']
        report.append(f"Context model wins: {h2h['context_model_wins']}")
        report.append(f"No-context model wins: {h2h['no_context_model_wins']}")
        report.append(f"Both correct: {h2h['both_correct']}")
        report.append(f"Both wrong: {h2h['both_wrong']}")
        report.append("")
        
        total_tests = sum(h2h.values())
        if total_tests > 0:
            context_advantage = (h2h['context_model_wins'] / total_tests) * 100
            report.append(f"Context model advantage: {context_advantage:.2f}%")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
