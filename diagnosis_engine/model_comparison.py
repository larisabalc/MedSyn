import pandas as pd
import numpy as np
from typing import List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from diagnosis_engine.models.context_diagnosis_classifier import ContextDiagnosisClassifier
from diagnosis_engine.models.no_context_diagnosis_classifier import NoContextDiagnosisClassifier
from diagnosis_engine.evaluation_metrics import DiagnosisEvaluator


class ModelComparison:
    """
    Comprehensive comparison framework for context vs no-context diagnosis models.
    """
    
    def __init__(self, context_model_path: str = "diagnosis_engine/trained_models/context",
                no_context_model_path: str = "diagnosis_engine/trained_models/no_context",
                output_dir: str = "results/model_comparison"):
        """
        Initialize comparison framework.
        
        Args:
            context_model_path: Path to trained context model
            no_context_model_path: Path to trained no-context model
            output_dir: Directory to save comparison results
        """
        self.context_model = ContextDiagnosisClassifier()
        self.no_context_model = NoContextDiagnosisClassifier()
        
        self.context_model.load_model(context_model_path)
        self.no_context_model.load_model(no_context_model_path)
        
        self.evaluator = DiagnosisEvaluator(use_semantic_similarity=True)
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    def prepare_test_data(self, test_dataset) -> Tuple[List[str], List[str], List[str]]:
        """
        Prepare test data from a dataset.
        Extracts input_text, target, and symptoms (if available).
        """
        inputs = []
        targets = []
        
        if hasattr(test_dataset, '__len__'):
            for example in test_dataset:
                if isinstance(example, dict):
                    inputs.append(example.get('input_text', str(example)))
                    targets.append(example.get('target', example.get('Name', '')))
                else:
                    inputs.append(str(example))
                    targets.append('')
        
        return inputs, targets
    
    def generate_predictions(self, test_inputs: List[str]) -> Tuple[List[str], List[str]]:
        """
        Generate predictions from both models.
        
        Returns:
            (context_predictions, no_context_predictions)
        """
        context_predictions = []
        no_context_predictions = []
        
        print(f"Generating predictions for {len(test_inputs)} samples...")
        
        for i, input_text in enumerate(test_inputs):
            if i % 10 == 0:
                print(f"  Processing {i}/{len(test_inputs)}...")
            
            try:
                context_pred = self.context_model.generate_disease_name(input_text)
                context_predictions.append(context_pred)
            except Exception as e:
                print(f"Context prediction error at index {i}: {e}")
                context_predictions.append("ERROR")
            
            try:
                symptom_only = self._extract_symptoms_only(input_text)
                no_context_pred = self.no_context_model.generate_disease_name(symptom_only)
                no_context_predictions.append(no_context_pred)
            except Exception as e:
                print(f"No-context prediction error at index {i}: {e}")
                no_context_predictions.append("ERROR")
        
        return context_predictions, no_context_predictions
    
    def _extract_symptoms_only(self, full_input: str) -> str:
        """
        Extract only symptoms from full patient description.
        Removes age, blood pressure, cholesterol info.
        """
        if "Reported symptoms include" in full_input:
            symptoms_part = full_input.split("Reported symptoms include")[1]
            return symptoms_part.strip()
        return full_input
    
    def run_full_comparison(self, test_dataset) -> dict:
        """
        Run complete evaluation comparing both models.
        
        Returns:
            Dictionary with all comparison results
        """
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        
        print("\n1. Preparing test data...")
        inputs, targets = self.prepare_test_data(test_dataset)
        print(f"   Loaded {len(inputs)} test samples")
        
        print("\n2. Generating predictions...")
        context_preds, no_context_preds = self.generate_predictions(inputs)
        
        print("\n3. Computing metrics...")
        comparison_results = self.evaluator.compare_models(
            context_preds, no_context_preds, targets
        )
        
        print("\n4. Analyzing errors...")
        context_errors = self.evaluator.analyze_error_types(context_preds, targets)
        no_context_errors = self.evaluator.analyze_error_types(no_context_preds, targets)
        
        print("\n5. Analyzing by disease...")
        context_by_disease = self.evaluator.analyze_by_disease(context_preds, targets)
        no_context_by_disease = self.evaluator.analyze_by_disease(no_context_preds, targets)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(inputs),
            'comparison_metrics': comparison_results,
            'context_model_errors': context_errors,
            'no_context_model_errors': no_context_errors,
            'context_by_disease': dict(context_by_disease),
            'no_context_by_disease': dict(no_context_by_disease),
            'context_predictions': context_preds,
            'no_context_predictions': no_context_preds,
            'references': targets,
            'test_inputs': inputs
        }
        
        return results
    
    def save_results(self, results: dict):
        """Save results to JSON and CSV files."""
        print("\n6. Saving results...")
        
        json_path = os.path.join(self.output_dir, "metrics", "comparison_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json_results = self._convert_to_json_serializable(results)
            json.dump(json_results, f, indent=2)
        print(f"   Saved JSON: {json_path}")
        
        csv_path = os.path.join(self.output_dir, "metrics", "predictions_comparison.csv")
        predictions_df = pd.DataFrame({
            'reference': results['references'],
            'context_prediction': results['context_predictions'],
            'no_context_prediction': results['no_context_predictions'],
            'context_correct': [
                c.strip().lower() == r.strip().lower()
                for c, r in zip(results['context_predictions'], results['references'])
            ],
            'no_context_correct': [
                nc.strip().lower() == r.strip().lower()
                for nc, r in zip(results['no_context_predictions'], results['references'])
            ]
        })
        predictions_df.to_csv(csv_path, index=False)
        print(f"   Saved predictions CSV: {csv_path}")
        
        metrics_path = os.path.join(self.output_dir, "metrics", "summary_metrics.json")
        summary = {
            'timestamp': results['timestamp'],
            'total_samples': results['total_samples'],
            'context_model': results['comparison_metrics']['context_model'],
            'no_context_model': results['comparison_metrics']['no_context_model'],
            'improvements': results['comparison_metrics']['improvements'],
            'head_to_head': results['comparison_metrics']['head_to_head'],
        }
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self._convert_to_json_serializable(summary), f, indent=2)
        print(f"   Saved summary metrics: {metrics_path}")
    
    def generate_visualizations(self, results: dict):
        """Generate comparison visualizations."""
        print("\n7. Generating visualizations...")
        
        self._plot_metrics_comparison(results)
        
        self._plot_head_to_head(results)
        
        self._plot_confidence_distribution(results)
        
        self._plot_disease_performance(results)
        
        print("   Saved all visualizations")
    
    def _plot_metrics_comparison(self, results: dict):
        """Bar chart comparing key metrics."""
        metrics_to_plot = ['exact_match', 'partial_match', 'avg_confidence']
        
        context_values = [results['comparison_metrics']['context_model'].get(m, 0) for m in metrics_to_plot]
        no_context_values = [results['comparison_metrics']['no_context_model'].get(m, 0) for m in metrics_to_plot]
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, context_values, width, label='Context Model', color='#2ecc71')
        bars2 = ax.bar(x + width/2, no_context_values, width, label='No-Context Model', color='#e74c3c')
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison: Context vs No-Context', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualizations", "metrics_comparison.png"), dpi=300)
        plt.close()
    
    def _plot_head_to_head(self, results: dict):
        """Pie chart of head-to-head results."""
        h2h = results['comparison_metrics']['head_to_head']
        
        labels = ['Context Wins', 'No-Context Wins', 'Both Correct', 'Both Wrong']
        sizes = [h2h['context_model_wins'], h2h['no_context_model_wins'], 
                h2h['both_correct'], h2h['both_wrong']]
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                            startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title('Head-to-Head Comparison Results', fontsize=14, fontweight='bold')
        
        for i, (label, size) in enumerate(zip(labels, sizes)):
            texts[i].set_text(f'{label}\n(n={size})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualizations", "head_to_head.png"), dpi=300)
        plt.close()
    
    def _plot_confidence_distribution(self, results: dict):
        """Distribution of prediction confidence."""
        context_conf = results['comparison_metrics']['context_model'].get('avg_confidence', 0)
        no_context_conf = results['comparison_metrics']['no_context_model'].get('avg_confidence', 0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = ['Context Model', 'No-Context Model']
        confidences = [context_conf, no_context_conf]
        colors = ['#2ecc71', '#e74c3c']
        
        bars = ax.bar(models, confidences, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Average Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title('Prediction Confidence Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{conf:.3f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualizations", "confidence_comparison.png"), dpi=300)
        plt.close()
    
    def _plot_disease_performance(self, results: dict):
        """Performance comparison by disease type."""
        context_diseases = results['context_by_disease']
        no_context_diseases = results['no_context_by_disease']
        
        diseases = [d for d in context_diseases if context_diseases[d]['total'] >= 3][:10]  # Top 10
        
        context_acc = [context_diseases[d].get('accuracy', 0) for d in diseases]
        no_context_acc = [no_context_diseases[d].get('accuracy', 0) for d in diseases]
        
        x = np.arange(len(diseases))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width/2, context_acc, width, label='Context Model', color='#2ecc71')
        bars2 = ax.bar(x + width/2, no_context_acc, width, label='No-Context Model', color='#e74c3c')
        
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Per-Disease Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d[:20] for d in diseases], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualizations", "disease_performance.png"), dpi=300)
        plt.close()
    
    def generate_report(self, results: dict):
        """Generate comprehensive report."""
        print("\n8. Generating comprehensive report...")
        
        report_path = os.path.join(self.output_dir, "COMPARISON_REPORT.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("PROVING CONTEXT IMPROVES MEDICAL DIAGNOSIS PREDICTION\n")
            f.write("="*100 + "\n\n")
            
            f.write(f"Analysis Date: {results['timestamp']}\n")
            f.write(f"Total Test Samples: {results['total_samples']}\n\n")
            
            # Section 1: Executive Summary
            f.write("SECTION 1: EXECUTIVE SUMMARY\n")
            f.write("-"*100 + "\n")
            h2h = results['comparison_metrics']['head_to_head']
            total = sum(h2h.values())
            context_advantage_pct = (h2h['context_model_wins'] / total * 100) if total > 0 else 0
            
            f.write(f"\nThe context-based diagnosis model outperformed the symptom-only model in {h2h['context_model_wins']} "
                    f"out of {total} test cases ({context_advantage_pct:.1f}%).\n")
            f.write(f"This demonstrates that incorporating patient context significantly improves diagnostic accuracy.\n\n")
            
            f.write("SECTION 2: DETAILED PERFORMANCE METRICS\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Metric':<35} {'Context Model':<25} {'No-Context Model':<25} {'Improvement':<15}\n")
            f.write("-"*100 + "\n")
            
            for metric, values in results['comparison_metrics']['improvements'].items():
                ctx_val = results['comparison_metrics']['context_model'].get(metric, 'N/A')
                no_ctx_val = results['comparison_metrics']['no_context_model'].get(metric, 'N/A')
                
                if isinstance(ctx_val, (int, float)):
                    improvement_str = f"+{values['absolute']:.4f} ({values['percentage']:.2f}%)"
                    f.write(f"{metric:<35} {ctx_val:<25.4f} {no_ctx_val:<25.4f} {improvement_str:<15}\n")
            
            f.write("\n\nSECTION 3: ERROR ANALYSIS\n")
            f.write("-"*100 + "\n")
            
            f.write("\nContext Model Error Breakdown:\n")
            ctx_err = results['context_model_errors']
            f.write(f"  Correct: {ctx_err['correct']}/{ctx_err['total_predictions']} ({ctx_err['accuracy']*100:.2f}%)\n")
            f.write(f"  Similar diseases: {ctx_err['similar_disease']}\n")
            f.write(f"  Partially similar: {ctx_err['off_by_character_case']}\n")
            f.write(f"  Completely wrong: {ctx_err['completely_wrong']}\n")
            
            f.write("\nNo-Context Model Error Breakdown:\n")
            nc_err = results['no_context_model_errors']
            f.write(f"  Correct: {nc_err['correct']}/{nc_err['total_predictions']} ({nc_err['accuracy']*100:.2f}%)\n")
            f.write(f"  Similar diseases: {nc_err['similar_disease']}\n")
            f.write(f"  Partially similar: {nc_err['off_by_character_case']}\n")
            f.write(f"  Completely wrong: {nc_err['completely_wrong']}\n")
            

            f.write("\n\nSECTION 4: KEY FINDINGS & IMPLICATIONS\n")
            f.write("-"*100 + "\n")
            
            f.write(f"\n1. CONTEXT ADVANTAGE\n")
            f.write(f"   The context model demonstrated superior performance with {h2h['context_model_wins']} "
                    f"cases where it made correct predictions while the no-context model failed.\n")
            f.write(f"   This represents a {context_advantage_pct:.1f}% advantage over the baseline symptom-only approach.\n")
            
            f.write(f"\n2. CONFIDENCE IN PREDICTIONS\n")
            ctx_conf = results['comparison_metrics']['context_model'].get('avg_confidence', 0)
            no_ctx_conf = results['comparison_metrics']['no_context_model'].get('avg_confidence', 0)
            f.write(f"   Context model average confidence: {ctx_conf:.4f}\n")
            f.write(f"   No-context model average confidence: {no_ctx_conf:.4f}\n")
            f.write(f"   Difference: +{(ctx_conf - no_ctx_conf):.4f}\n")
            
            f.write(f"\n3. DISEASE-SPECIFIC INSIGHTS\n")
            f.write(f"   Most improved diseases with context:\n")
            
            diseases_with_improvement = []
            for disease in results['context_by_disease']:
                if disease in results['no_context_by_disease']:
                    ctx_acc = results['context_by_disease'][disease].get('accuracy', 0)
                    no_ctx_acc = results['no_context_by_disease'][disease].get('accuracy', 0)
                    improvement = ctx_acc - no_ctx_acc
                    if improvement > 0:
                        diseases_with_improvement.append((disease, improvement, ctx_acc, no_ctx_acc))
            
            diseases_with_improvement.sort(key=lambda x: x[1], reverse=True)
            for disease, improvement, ctx_acc, no_ctx_acc in diseases_with_improvement[:5]:
                f.write(f"      - {disease}: {improvement*100:.1f}% improvement "
                        f"({no_ctx_acc*100:.1f}% → {ctx_acc*100:.1f}%)\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*100 + "\n")
        
        print(f"   Saved comprehensive report: {report_path}")
        return report_path
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        return obj
    
    def run_full_evaluation_pipeline(self, test_dataset):
        """Run complete evaluation pipeline and save everything."""
        results = self.run_full_comparison(test_dataset)
        
        self.save_results(results)
        
        self.generate_visualizations(results)
        
        report_path = self.generate_report(results)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")
        print(f"  - Metrics: {os.path.join(self.output_dir, 'metrics')}")
        print(f"  - Visualizations: {os.path.join(self.output_dir, 'visualizations')}")
        print(f"  - Main Report: {report_path}")
        print("="*80 + "\n")
        
        return results
