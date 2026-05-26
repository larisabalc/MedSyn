import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from diagnosis_engine.model_comparison import ModelComparison
from diagnosis_engine.statistical_analysis import StatisticalComparison
from diagnosis_engine.models.context_diagnosis_classifier import ContextDiagnosisClassifier
import pandas as pd


def load_test_dataset(dataset_path: str = "data/synthetic/final_training_dataset.csv"):
    """
    Load test dataset from CSV.
    Expects columns: 'input_text' and 'target'
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return None
    
    df = pd.read_csv(dataset_path)
    
    dataset = [row.to_dict() for _, row in df.iterrows()]
    
    return dataset


def main():
    """
    Main evaluation pipeline.
    """
    
    print("\n" + "="*100)
    print("EVALUATION FRAMEWORK")
    print("Proving Context Improves Medical Diagnosis Prediction")
    print("="*100 + "\n")
    
    context_model_path = "diagnosis_engine/trained_models/context"
    no_context_model_path = "diagnosis_engine/trained_models/no_context"
    dataset_path = "data/synthetic/final_training_dataset.csv"
    output_dir = "results/comparison"
    
    print("CONFIGURATION:")
    print(f"  Context model: {context_model_path}")
    print(f"  No-context model: {no_context_model_path}")
    print(f"  Test dataset: {dataset_path}")
    print(f"  Output directory: {output_dir}\n")
    
    print("STEP 1: Loading test dataset...")
    dataset = load_test_dataset(dataset_path)
    
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return
    
    print(f"  Loaded {len(dataset)} samples\n")

    print("STEP 2: Running comprehensive model comparison...")
    try:
        comparison = ModelComparison(
            context_model_path=context_model_path,
            no_context_model_path=no_context_model_path,
            output_dir=output_dir
        )
        
        results = comparison.run_full_evaluation_pipeline(dataset)
        
    except Exception as e:
        print(f"Error during model comparison: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nSTEP 3: Running statistical significance tests...")
    try:
        stats_comparison = StatisticalComparison(output_dir=output_dir)
        
        stats_report_path = os.path.join(output_dir, "metrics", "STATISTICAL_ANALYSIS.txt")
        stats_report = stats_comparison.generate_statistical_report(results, stats_report_path)
        
        print(f"  Statistical report saved to: {stats_report_path}")
        
    except Exception as e:
        print(f"Error during statistical analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*100)
    print("EVALUATION COMPLETE - RESULTS SUMMARY")
    print("="*100 + "\n")
    
    h2h = results['comparison_metrics']['head_to_head']
    total = sum(h2h.values())
    context_advantage = (h2h['context_model_wins'] / total * 100) if total > 0 else 0
    
    print(f"Total samples evaluated: {len(dataset)}")
    print(f"Context model wins: {h2h['context_model_wins']} cases")
    print(f"No-context model wins: {h2h['no_context_model_wins']} cases")
    print(f"Both correct: {h2h['both_correct']} cases")
    print(f"Both wrong: {h2h['both_wrong']} cases")
    print(f"\n✓ Context model advantage: {context_advantage:.1f}%\n")

def generate_quick_comparison():
    """
    Quick comparison without full evaluation pipeline (for testing).
    """
    print("\nGenerating quick comparison for testing...\n")
    
    from diagnosis_engine.evaluation_metrics import DiagnosisEvaluator
    
    context_predictions = [
        "Diabetes",
        "Hypertension",
        "Asthma",
        "Pneumonia",
        "Gastritis",
    ]
    
    no_context_predictions = [
        "Hyperglycemia",
        "High Blood Pressure",
        "Asthma",
        "Bronchitis",
        "Stomach Ulcer",
    ]
    
    references = [
        "Diabetes",
        "Hypertension",
        "Asthma",
        "Pneumonia",
        "Gastritis",
    ]
    
    evaluator = DiagnosisEvaluator()
    results = evaluator.compare_models(context_predictions, no_context_predictions, references)
    
    print("QUICK COMPARISON RESULTS:")
    print("-" * 60)
    print(f"Context Model Metrics:")
    for metric, value in results['context_model'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nNo-Context Model Metrics:")
    for metric, value in results['no_context_model'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nImprovements with Context:")
    for metric, improvement in results['improvements'].items():
        print(f"  {metric}: +{improvement['absolute']:.4f} ({improvement['percentage']:.2f}%)")
    
    print(f"\nHead-to-Head Results:")
    h2h = results['head_to_head']
    print(f"  Context wins: {h2h['context_model_wins']}")
    print(f"  No-context wins: {h2h['no_context_model_wins']}")
    print(f"  Both correct: {h2h['both_correct']}")
    print(f"  Both wrong: {h2h['both_wrong']}")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick comparison test with sample data"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        generate_quick_comparison()
    else:
        main()
