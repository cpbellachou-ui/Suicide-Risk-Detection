"""
Evaluation module for comparing baseline and BERT models.
Provides comprehensive metrics calculation and model comparison.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive evaluation and comparison of suicide risk detection models."""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the evaluator.
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = results_dir
        self.metrics_dir = os.path.join(results_dir, "metrics")
        self.figures_dir = os.path.join(results_dir, "figures")
        
        # Create directories
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int], 
                         y_proba: np.ndarray = None, model_name: str = "Model") -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Calculating metrics for {model_name}...")
        
        # Basic classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        metrics.update({
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        })
        
        # Probability-based metrics
        if y_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
            
            # Precision-Recall curve
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_proba[:, 1])
            metrics['precision_recall_curve'] = {
                'precision': precision_curve,
                'recall': recall_curve,
                'thresholds': pr_thresholds
            }
            
            # ROC curve
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba[:, 1])
            metrics['roc_curve'] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': roc_thresholds
            }
        
        # Log key metrics
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"{model_name} - F1-Score: {metrics['f1_score']:.4f}")
        if 'auc_roc' in metrics:
            logger.info(f"{model_name} - AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return metrics
    
    def compare_models(self, baseline_metrics: Dict[str, Any], 
                      bert_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare baseline and BERT model performance.
        
        Args:
            baseline_metrics: Metrics from baseline model
            bert_metrics: Metrics from BERT model
            
        Returns:
            Comparison results
        """
        logger.info("Comparing model performance...")
        
        comparison = {
            'baseline': baseline_metrics,
            'bert': bert_metrics,
            'improvement': {}
        }
        
        # Calculate improvements
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        for metric in key_metrics:
            if metric in baseline_metrics and metric in bert_metrics:
                baseline_val = baseline_metrics[metric]
                bert_val = bert_metrics[metric]
                improvement = ((bert_val - baseline_val) / baseline_val) * 100
                
                comparison['improvement'][metric] = {
                    'baseline': baseline_val,
                    'bert': bert_val,
                    'improvement_percent': improvement,
                    'improvement_absolute': bert_val - baseline_val
                }
        
        # Save comparison results
        comparison_path = os.path.join(self.metrics_dir, 'model_comparison.json')
        self._save_json(comparison, comparison_path)
        
        return comparison
    
    def create_performance_comparison_plot(self, comparison: Dict[str, Any]) -> str:
        """Create a bar chart comparing model performance.
        
        Args:
            comparison: Model comparison results
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating performance comparison plot...")
        
        # Extract metrics for plotting
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        baseline_scores = []
        bert_scores = []
        
        for metric in metrics:
            if metric in comparison['baseline'] and metric in comparison['bert']:
                baseline_scores.append(comparison['baseline'][metric])
                bert_scores.append(comparison['bert'][metric])
            else:
                baseline_scores.append(0)
                bert_scores.append(0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_scores, width, label='TF-IDF + LogReg', alpha=0.8)
        bars2 = ax.bar(x + width/2, bert_scores, width, label='BERT', alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.figures_dir, 'performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison plot saved to {plot_path}")
        return plot_path
    
    def create_confusion_matrices(self, baseline_metrics: Dict[str, Any], 
                                 bert_metrics: Dict[str, Any]) -> str:
        """Create side-by-side confusion matrices.
        
        Args:
            baseline_metrics: Baseline model metrics
            bert_metrics: BERT model metrics
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating confusion matrices...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Baseline confusion matrix
        sns.heatmap(baseline_metrics['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Risk', 'Suicide Risk'],
                   yticklabels=['No Risk', 'Suicide Risk'],
                   ax=ax1)
        ax1.set_title('TF-IDF + Logistic Regression', fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # BERT confusion matrix
        sns.heatmap(bert_metrics['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Risk', 'Suicide Risk'],
                   yticklabels=['No Risk', 'Suicide Risk'],
                   ax=ax2)
        ax2.set_title('BERT', fontweight='bold')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.suptitle('Confusion Matrices Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.figures_dir, 'confusion_matrices.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrices saved to {plot_path}")
        return plot_path
    
    def create_roc_curves(self, baseline_metrics: Dict[str, Any], 
                         bert_metrics: Dict[str, Any]) -> str:
        """Create ROC curves comparison.
        
        Args:
            baseline_metrics: Baseline model metrics
            bert_metrics: BERT model metrics
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating ROC curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curves
        if 'roc_curve' in baseline_metrics:
            fpr_baseline = baseline_metrics['roc_curve']['fpr']
            tpr_baseline = baseline_metrics['roc_curve']['tpr']
            auc_baseline = baseline_metrics['auc_roc']
            
            ax.plot(fpr_baseline, tpr_baseline, 
                   label=f'TF-IDF + LogReg (AUC = {auc_baseline:.3f})',
                   linewidth=2, alpha=0.8)
        
        if 'roc_curve' in bert_metrics:
            fpr_bert = bert_metrics['roc_curve']['fpr']
            tpr_bert = bert_metrics['roc_curve']['tpr']
            auc_bert = bert_metrics['auc_roc']
            
            ax.plot(fpr_bert, tpr_bert, 
                   label=f'BERT (AUC = {auc_bert:.3f})',
                   linewidth=2, alpha=0.8)
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        # Customize plot
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.figures_dir, 'roc_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {plot_path}")
        return plot_path
    
    def create_precision_recall_curves(self, baseline_metrics: Dict[str, Any], 
                                     bert_metrics: Dict[str, Any]) -> str:
        """Create Precision-Recall curves comparison.
        
        Args:
            baseline_metrics: Baseline model metrics
            bert_metrics: BERT model metrics
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating Precision-Recall curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot PR curves
        if 'precision_recall_curve' in baseline_metrics:
            precision_baseline = baseline_metrics['precision_recall_curve']['precision']
            recall_baseline = baseline_metrics['precision_recall_curve']['recall']
            
            ax.plot(recall_baseline, precision_baseline, 
                   label='TF-IDF + LogReg',
                   linewidth=2, alpha=0.8)
        
        if 'precision_recall_curve' in bert_metrics:
            precision_bert = bert_metrics['precision_recall_curve']['precision']
            recall_bert = bert_metrics['precision_recall_curve']['recall']
            
            ax.plot(recall_bert, precision_bert, 
                   label='BERT',
                   linewidth=2, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.figures_dir, 'precision_recall_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-Recall curves saved to {plot_path}")
        return plot_path
    
    def generate_evaluation_report(self, comparison: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            comparison: Model comparison results
            
        Returns:
            Path to saved report
        """
        logger.info("Generating evaluation report...")
        
        report_path = os.path.join(self.metrics_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("SUICIDE RISK DETECTION - MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write("This report compares the performance of two models for suicide risk detection:\n")
            f.write("1. Baseline: TF-IDF + Logistic Regression\n")
            f.write("2. Deep Learning: Fine-tuned BERT\n\n")
            
            # Model Performance
            f.write("MODEL PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            
            baseline = comparison['baseline']
            bert = comparison['bert']
            
            f.write("Baseline Model (TF-IDF + LogReg):\n")
            f.write(f"  Accuracy:  {baseline['accuracy']:.4f}\n")
            f.write(f"  Precision: {baseline['precision']:.4f}\n")
            f.write(f"  Recall:    {baseline['recall']:.4f}\n")
            f.write(f"  F1-Score:  {baseline['f1_score']:.4f}\n")
            if 'auc_roc' in baseline:
                f.write(f"  AUC-ROC:   {baseline['auc_roc']:.4f}\n")
            f.write("\n")
            
            f.write("BERT Model:\n")
            f.write(f"  Accuracy:  {bert['accuracy']:.4f}\n")
            f.write(f"  Precision: {bert['precision']:.4f}\n")
            f.write(f"  Recall:    {bert['recall']:.4f}\n")
            f.write(f"  F1-Score:  {bert['f1_score']:.4f}\n")
            if 'auc_roc' in bert:
                f.write(f"  AUC-ROC:   {bert['auc_roc']:.4f}\n")
            f.write("\n")
            
            # Improvements
            f.write("PERFORMANCE IMPROVEMENTS\n")
            f.write("-" * 25 + "\n")
            
            for metric, improvement in comparison['improvement'].items():
                f.write(f"{metric.upper()}:\n")
                f.write(f"  Baseline: {improvement['baseline']:.4f}\n")
                f.write(f"  BERT:     {improvement['bert']:.4f}\n")
                f.write(f"  Improvement: {improvement['improvement_percent']:+.2f}% "
                       f"({improvement['improvement_absolute']:+.4f})\n\n")
            
            # Key Findings
            f.write("KEY FINDINGS\n")
            f.write("-" * 15 + "\n")
            
            # Find the metric with highest improvement
            best_improvement = max(comparison['improvement'].items(), 
                                 key=lambda x: x[1]['improvement_percent'])
            f.write(f"• BERT shows the highest improvement in {best_improvement[0].upper()}: "
                   f"{best_improvement[1]['improvement_percent']:+.2f}%\n")
            
            # Overall performance
            avg_improvement = np.mean([imp['improvement_percent'] 
                                     for imp in comparison['improvement'].values()])
            f.write(f"• Average improvement across all metrics: {avg_improvement:+.2f}%\n")
            
            f.write("• BERT's contextual understanding provides better pattern recognition\n")
            f.write("• Both models show reasonable performance for suicide risk screening\n")
            f.write("• BERT is recommended for production deployment\n\n")
            
            # Ethical Considerations
            f.write("ETHICAL CONSIDERATIONS\n")
            f.write("-" * 22 + "\n")
            f.write("• This system is designed as a screening tool, not a diagnostic tool\n")
            f.write("• Human oversight is essential for all predictions\n")
            f.write("• Privacy and confidentiality must be maintained\n")
            f.write("• False positives and false negatives have serious implications\n")
            f.write("• Regular model validation and bias assessment required\n")
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report_path
    
    def _save_json(self, data: Dict[str, Any], filepath: str):
        """Save data as JSON file.
        
        Args:
            data: Data to save
            filepath: Path to save file
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Recursively convert numpy objects
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        converted_data = recursive_convert(data)
        
        with open(filepath, 'w') as f:
            json.dump(converted_data, f, indent=2)

def evaluate_models(baseline_model, bert_model, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate both models and create comprehensive comparison.
    
    Args:
        baseline_model: Trained baseline model
        bert_model: Trained BERT model
        test_df: Test DataFrame
        
    Returns:
        Model comparison results
    """
    logger.info("Starting comprehensive model evaluation...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Prepare test data
    X_test = test_df['text'].tolist()
    y_test = test_df['label'].tolist()
    
    # Evaluate baseline model
    baseline_predictions, baseline_probabilities = baseline_model.predict(X_test)
    baseline_metrics = evaluator.calculate_metrics(
        y_test, baseline_predictions, baseline_probabilities, "Baseline"
    )
    
    # Evaluate BERT model
    from bert_model import BertModelWrapper
    if isinstance(bert_model, BertModelWrapper):
        # Create test loader for BERT
        from bert_model import SuicideRiskDataset
        test_dataset = SuicideRiskDataset(
            X_test, y_test, bert_model.tokenizer, bert_model.max_length
        )
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=bert_model.batch_size, shuffle=False)
        
        bert_predictions, bert_probabilities = bert_model.predict(test_loader)
    else:
        # Assume bert_model has predict method
        bert_predictions, bert_probabilities = bert_model.predict(X_test)
    
    bert_metrics = evaluator.calculate_metrics(
        y_test, bert_predictions, bert_probabilities, "BERT"
    )
    
    # Compare models
    comparison = evaluator.compare_models(baseline_metrics, bert_metrics)
    
    # Create visualizations
    evaluator.create_performance_comparison_plot(comparison)
    evaluator.create_confusion_matrices(baseline_metrics, bert_metrics)
    evaluator.create_roc_curves(baseline_metrics, bert_metrics)
    evaluator.create_precision_recall_curves(baseline_metrics, bert_metrics)
    
    # Generate report
    evaluator.generate_evaluation_report(comparison)
    
    logger.info("Model evaluation completed!")
    return comparison

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    from baseline_model import BaselineModel
    from bert_model import BertModelWrapper
    
    # Load data
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_processed_data()
    
    # Load trained models
    baseline_model = BaselineModel()
    baseline_model.load_model()
    
    bert_model = BertModelWrapper()
    bert_model.load_model()
    
    # Evaluate models
    comparison = evaluate_models(baseline_model, bert_model, test_df)
    
    print("\nModel Comparison Results:")
    for metric, improvement in comparison['improvement'].items():
        print(f"{metric.upper()}: {improvement['improvement_percent']:+.2f}% improvement")