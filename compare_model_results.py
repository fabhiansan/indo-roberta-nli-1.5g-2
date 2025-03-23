"""
Script to compare results across different NLI models.
This script reads evaluation results from multiple models and generates comparison visualizations.
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def load_model_results(results_dir):
    """
    Load results from all models in the results directory.
    
    Args:
        results_dir: Directory containing model results
        
    Returns:
        Dictionary of model results
    """
    models = {}
    
    # Get all subdirectories (one per model)
    model_dirs = [d for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d))
                  and not d.startswith('.')]
    
    for model_name in model_dirs:
        model_dir = os.path.join(results_dir, model_name)
        
        # Look for results files
        results_files = []
        for root, _, files in os.walk(model_dir):
            for file in files:
                if file.endswith('_metrics.json'):
                    results_files.append(os.path.join(root, file))
        
        if not results_files:
            print(f"No result files found for {model_name}")
            continue
        
        # Load the results
        model_results = {
            "test_lay": None,
            "test_expert": None,
            "validation": None,
            "hyperparameters": None
        }
        
        for file_path in results_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Determine which dataset this belongs to
                if 'test_lay' in file_path:
                    model_results['test_lay'] = data
                elif 'test_expert' in file_path:
                    model_results['test_expert'] = data
                elif 'validation' in file_path:
                    model_results['validation'] = data
                
        # Look for hyperparameters file
        hparams_file = os.path.join(model_dir, 'hyperparameters.json')
        if os.path.exists(hparams_file):
            with open(hparams_file, 'r') as f:
                model_results['hyperparameters'] = json.load(f)
        
        models[model_name] = model_results
    
    return models


def create_comparison_table(models, dataset="test_lay"):
    """
    Create a comparison table of model metrics.
    
    Args:
        models: Dictionary of model results
        dataset: Which dataset to compare on
        
    Returns:
        DataFrame with model comparisons
    """
    # Extract metrics from each model
    data = []
    
    for model_name, results in models.items():
        if dataset in results and results[dataset] is not None:
            metrics = results[dataset]
            
            row = {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy", None),
                "F1 Score": metrics.get("macro_f1", None),
                "Precision": metrics.get("precision", None),
                "Recall": metrics.get("recall", None)
            }
            
            # Add per-class F1 scores if available
            if "class_metrics" in metrics:
                for label, class_metrics in metrics["class_metrics"].items():
                    row[f"{label}_F1"] = class_metrics.get("f1", None)
            
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by accuracy descending
    if len(df) > 0 and "Accuracy" in df.columns:
        df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    
    return df


def plot_comparison(df, metric="Accuracy", output_dir=None):
    """
    Create a bar plot comparing models on a metric.
    
    Args:
        df: DataFrame with comparison data
        metric: Metric to plot
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    sns.barplot(x="Model", y=metric, data=df, palette="viridis")
    
    # Add value annotations
    for i, v in enumerate(df[metric]):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.title(f"Model Comparison - {metric}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save figure if output directory provided
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"comparison_{metric}.png"))
    
    plt.close()


def create_radar_chart(df, metrics, output_dir=None):
    """
    Create a radar chart comparing models across multiple metrics.
    
    Args:
        df: DataFrame with comparison data
        metrics: List of metrics to include
        output_dir: Directory to save the plot
    """
    # Number of metrics
    N = len(metrics)
    
    # Create angles for each metric
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add each model
    for i, model in enumerate(df["Model"]):
        values = [df.loc[i, metric] for metric in metrics]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("Model Comparison - Multiple Metrics")
    
    # Save figure if output directory provided
    if output_dir:
        plt.savefig(os.path.join(output_dir, "comparison_radar.png"))
    
    plt.close()


def generate_comparison_report(models, comparison_df, output_dir):
    """
    Generate a markdown report comparing all models.
    
    Args:
        models: Dictionary of model results
        comparison_df: DataFrame with model comparisons
        output_dir: Directory to save the report
    """
    report = []
    report.append("# IndoNLI Model Comparison Report")
    report.append("")
    
    # Add comparison table
    report.append("## Model Performance Comparison")
    report.append("")
    report.append(comparison_df.to_markdown(index=False))
    report.append("")
    
    # Add model details
    report.append("## Model Details")
    report.append("")
    
    for model_name, results in models.items():
        report.append(f"### {model_name}")
        report.append("")
        
        # Add hyperparameters if available
        if results['hyperparameters']:
            report.append("#### Hyperparameters")
            report.append("")
            for key, value in results['hyperparameters'].items():
                report.append(f"- **{key}**: {value}")
            report.append("")
        
        # Add test results if available
        if results['test_lay']:
            report.append("#### Test Lay Results")
            report.append("")
            report.append(f"- Accuracy: {results['test_lay'].get('accuracy', 'N/A'):.4f}")
            report.append(f"- F1 Score: {results['test_lay'].get('macro_f1', 'N/A'):.4f}")
            report.append(f"- Precision: {results['test_lay'].get('precision', 'N/A'):.4f}")
            report.append(f"- Recall: {results['test_lay'].get('recall', 'N/A'):.4f}")
            report.append("")
        
        if results['test_expert']:
            report.append("#### Test Expert Results")
            report.append("")
            report.append(f"- Accuracy: {results['test_expert'].get('accuracy', 'N/A'):.4f}")
            report.append(f"- F1 Score: {results['test_expert'].get('macro_f1', 'N/A'):.4f}")
            report.append(f"- Precision: {results['test_expert'].get('precision', 'N/A'):.4f}")
            report.append(f"- Recall: {results['test_expert'].get('recall', 'N/A'):.4f}")
            report.append("")
    
    # Add conclusion
    report.append("## Conclusion")
    report.append("")
    
    if len(comparison_df) > 0:
        best_model = comparison_df.iloc[0]["Model"]
        best_accuracy = comparison_df.iloc[0]["Accuracy"]
        report.append(f"Based on accuracy, the best performing model is **{best_model}** with an accuracy of {best_accuracy:.4f} on the test_lay dataset.")
    else:
        report.append("No model comparison data available.")
    
    # Save report
    report_path = os.path.join(output_dir, "model_comparison_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    
    print(f"Comparison report saved to {report_path}")
    
    return "\n".join(report)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare NLI model results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing model results")
    return parser.parse_args()


def main():
    """Main function to compare model results."""
    args = parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Results directory {args.results_dir} does not exist")
        return
    
    # Load model results
    models = load_model_results(args.results_dir)
    
    if not models:
        print("No model results found")
        return
    
    print(f"Found results for {len(models)} models: {', '.join(models.keys())}")
    
    # Create comparison tables
    test_lay_df = create_comparison_table(models, "test_lay")
    test_expert_df = create_comparison_table(models, "test_expert")
    
    # Save comparison tables
    test_lay_df.to_csv(os.path.join(args.results_dir, "test_lay_comparison.csv"), index=False)
    test_expert_df.to_csv(os.path.join(args.results_dir, "test_expert_comparison.csv"), index=False)
    
    # Plot comparisons
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
    for metric in metrics:
        if test_lay_df is not None and not test_lay_df.empty and metric in test_lay_df.columns:
            plot_comparison(test_lay_df, metric, args.results_dir)
        
        if test_expert_df is not None and not test_expert_df.empty and metric in test_expert_df.columns:
            plot_comparison(test_expert_df, metric, args.results_dir)
    
    # Create radar chart if enough data available
    if test_lay_df is not None and len(test_lay_df) > 0:
        available_metrics = [m for m in metrics if m in test_lay_df.columns and not test_lay_df[m].isna().any()]
        if len(available_metrics) >= 3:
            create_radar_chart(test_lay_df, available_metrics, args.results_dir)
    
    # Generate comparison report
    report = generate_comparison_report(models, test_lay_df, args.results_dir)
    
    print("\nModel comparison complete.")
    print(f"Results saved to {args.results_dir}")


if __name__ == "__main__":
    main()
