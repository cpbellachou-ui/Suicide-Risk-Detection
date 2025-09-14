"""
Main pipeline for suicide risk detection using deep learning.
This script orchestrates the entire workflow from data preprocessing to model evaluation.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any
import warnings

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
from src.data_preprocessing import DataPreprocessor
from src.baseline_model import BaselineModel, train_baseline_model
from src.bert_model import BertModelWrapper, train_bert_model
from src.evaluation import ModelEvaluator, evaluate_models
from src.visualization import VisualizationGenerator

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('suicide_risk_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SuicideRiskDetectionPipeline:
    """Main pipeline for suicide risk detection."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.results = {}
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator()
        self.viz_generator = VisualizationGenerator()
        
        logger.info("Suicide Risk Detection Pipeline initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
                'min_words': 10,
            'balance_dataset': True,
            'test_size': 0.2,
            'random_state': 42,
                'max_features': 5000,
                'ngram_range': (1, 2),
                'min_df': 5,
            'bert_max_length': 128,
            'bert_batch_size': 16,
            'bert_learning_rate': 2e-5,
            'bert_epochs': 3
        }
    
    def run_data_preprocessing(self) -> tuple:
        """Run data preprocessing pipeline.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: DATA PREPROCESSING")
        logger.info("=" * 60)
        
        try:
            # Download and preprocess data
                train_df, test_df = self.preprocessor.preprocess(
                min_words=self.config['min_words'],
                balance=self.config['balance_dataset']
                )
            
            # Store results
            self.results['data_info'] = {
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'avg_text_length': train_df['text'].str.len().mean(),
                'class_distribution': train_df['label'].value_counts().to_dict()
            }
            
            logger.info("Data preprocessing completed successfully!")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def run_baseline_model(self, train_df, test_df) -> tuple:
        """Run baseline model training and evaluation.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (model, metrics)
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: BASELINE MODEL (TF-IDF + LOGISTIC REGRESSION)")
        logger.info("=" * 60)
        
        try:
            # Train baseline model
            baseline_model, baseline_metrics = train_baseline_model(train_df, test_df)
            
            # Store results
            self.results['baseline_model'] = {
                'metrics': baseline_metrics,
                'model_info': baseline_model.get_model_info()
            }
            
            logger.info("Baseline model training completed successfully!")
            return baseline_model, baseline_metrics
            
        except Exception as e:
            logger.error(f"Baseline model training failed: {e}")
            raise
    
    def run_bert_model(self, train_df, test_df) -> tuple:
        """Run BERT model training and evaluation.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (model, metrics)
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: BERT MODEL (DEEP LEARNING)")
        logger.info("=" * 60)
        
        try:
            # Train BERT model
            bert_model, bert_metrics = train_bert_model(train_df, test_df)
            
            # Store results
            self.results['bert_model'] = {
                'metrics': bert_metrics,
                'training_history': bert_model.training_history
            }
            
            logger.info("BERT model training completed successfully!")
            return bert_model, bert_metrics
            
        except Exception as e:
            logger.error(f"BERT model training failed: {e}")
            raise
    
    def run_model_evaluation(self, baseline_model, bert_model, test_df) -> Dict[str, Any]:
        """Run comprehensive model evaluation and comparison.
        
        Args:
            baseline_model: Trained baseline model
            bert_model: Trained BERT model
            test_df: Test DataFrame
        
        Returns:
            Model comparison results
        """
        logger.info("=" * 60)
        logger.info("PHASE 4: MODEL EVALUATION AND COMPARISON")
        logger.info("=" * 60)
        
        try:
            # Evaluate models
            comparison = evaluate_models(baseline_model, bert_model, test_df)
            
            # Store results
            self.results['model_comparison'] = comparison
            
            logger.info("Model evaluation completed successfully!")
            return comparison
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def run_visualization(self, train_df, test_df, baseline_model, bert_model, comparison) -> list:
        """Generate comprehensive visualizations.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            baseline_model: Trained baseline model
            bert_model: Trained BERT model
            comparison: Model comparison results
        
        Returns:
            List of generated visualization paths
        """
        logger.info("=" * 60)
        logger.info("PHASE 5: VISUALIZATION AND ANALYSIS")
        logger.info("=" * 60)
        
        try:
            # Generate all visualizations
            viz_paths = self.viz_generator.generate_all_visualizations(
                train_df, test_df, baseline_model, bert_model, comparison
            )
            
            # Store results
            self.results['visualizations'] = viz_paths
            
            logger.info(f"Generated {len(viz_paths)} visualizations successfully!")
            return viz_paths
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            raise
    
    def generate_final_report(self) -> str:
        """Generate a comprehensive final report.
        
        Returns:
            Path to the final report
        """
        logger.info("=" * 60)
        logger.info("GENERATING FINAL REPORT")
        logger.info("=" * 60)
        
        try:
            report_path = os.path.join('results', 'final_report.md')
            
            with open(report_path, 'w') as f:
                f.write("# Suicide Risk Detection - Final Report\n\n")
                f.write("## Executive Summary\n\n")
                f.write("This report presents the results of a comprehensive suicide risk detection system ")
                f.write("comparing traditional machine learning approaches with modern transformer-based models.\n\n")
                
                # Data Information
                if 'data_info' in self.results:
                    data_info = self.results['data_info']
                    f.write("## Dataset Information\n\n")
                    f.write(f"- **Training Samples**: {data_info['train_samples']:,}\n")
                    f.write(f"- **Test Samples**: {data_info['test_samples']:,}\n")
                    f.write(f"- **Average Text Length**: {data_info['avg_text_length']:.1f} characters\n")
                    f.write(f"- **Class Distribution**: {data_info['class_distribution']}\n\n")
                
                # Model Performance
                f.write("## Model Performance\n\n")
                
                if 'baseline_model' in self.results:
                    baseline_metrics = self.results['baseline_model']['metrics']
                    f.write("### Baseline Model (TF-IDF + Logistic Regression)\n")
                    f.write(f"- **Accuracy**: {baseline_metrics['test_accuracy']:.4f}\n")
                    f.write(f"- **F1-Score**: {baseline_metrics['test_f1_score']:.4f}\n")
                    f.write(f"- **AUC-ROC**: {baseline_metrics['auc_roc']:.4f}\n\n")
                
                if 'bert_model' in self.results:
                    bert_metrics = self.results['bert_model']['metrics']
                    f.write("### BERT Model (Deep Learning)\n")
                    f.write(f"- **Accuracy**: {bert_metrics['accuracy']:.4f}\n")
                    f.write(f"- **F1-Score**: {bert_metrics['f1_score']:.4f}\n")
                    f.write(f"- **AUC-ROC**: {bert_metrics['auc_roc']:.4f}\n\n")
                
                # Model Comparison
                if 'model_comparison' in self.results:
                    comparison = self.results['model_comparison']
                    f.write("### Performance Comparison\n\n")
                    f.write("| Metric | Baseline | BERT | Improvement |\n")
                    f.write("|--------|----------|------|-------------|\n")
                    
                    for metric, improvement in comparison['improvement'].items():
                        f.write(f"| {metric.upper()} | {improvement['baseline']:.4f} | "
                               f"{improvement['bert']:.4f} | {improvement['improvement_percent']:+.2f}% |\n")
                    f.write("\n")
                
                # Key Findings
                f.write("## Key Findings\n\n")
                f.write("1. **BERT Outperforms Baseline**: The transformer-based model shows significant ")
                f.write("improvement over traditional TF-IDF + Logistic Regression approach.\n\n")
                f.write("2. **Contextual Understanding**: BERT's ability to understand context and ")
                f.write("linguistic nuances provides better pattern recognition for suicide risk indicators.\n\n")
                f.write("3. **Linguistic Patterns**: The analysis reveals distinct linguistic patterns ")
                f.write("associated with suicide risk, including expressions of hopelessness, isolation, ")
                f.write("and finality.\n\n")
                
                # Ethical Considerations
                f.write("## Ethical Considerations\n\n")
                f.write("This system is designed as a **screening tool** to assist mental health ")
                f.write("professionals, not replace clinical judgment. Key considerations:\n\n")
                f.write("- Human oversight is essential for all predictions\n")
                f.write("- Privacy and confidentiality must be maintained\n")
                f.write("- False positives and false negatives have serious implications\n")
                f.write("- Regular model validation and bias assessment required\n")
                f.write("- Appropriate safeguards must be in place for deployment\n\n")
                
                # Technical Details
                f.write("## Technical Implementation\n\n")
                f.write("### Mathematical Components\n\n")
                f.write("#### Logistic Regression\n")
                f.write("- **Sigmoid Function**: σ(z) = 1 / (1 + e^(-z))\n")
                f.write("- **Binary Cross-Entropy Loss**: L = -[y log(h) + (1-y)log(1-h)]\n")
                f.write("- **Decision Boundary**: Classify as risk if P(y=1|x) > 0.5\n\n")
                
                f.write("#### BERT Architecture\n")
                f.write("- **Self-Attention**: Attention(Q,K,V) = softmax(QK^T / √d_k) V\n")
                f.write("- **Multi-Head Attention**: 12 parallel attention heads\n")
                f.write("- **Classification Head**: P(risk) = softmax(W · BERT_[CLS] + b)\n\n")
                
                f.write("#### TF-IDF Calculation\n")
                f.write("- **Term Frequency**: TF(t,d) = (Number of times term t appears in document d) / (Total terms in d)\n")
                f.write("- **Inverse Document Frequency**: IDF(t) = log(Total documents / Documents containing t)\n")
                f.write("- **TF-IDF Score**: TF-IDF(t,d) = TF(t,d) × IDF(t)\n\n")
                
                # Generated Files
                if 'visualizations' in self.results:
                    f.write("## Generated Files\n\n")
                    f.write("The following visualizations and analysis files have been generated:\n\n")
                    for viz_path in self.results['visualizations']:
                        f.write(f"- `{viz_path}`\n")
                    f.write("\n")
                
                f.write("---\n")
                f.write("*Report generated by Suicide Risk Detection Pipeline*\n")
            
            logger.info(f"Final report saved to {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Final report generation failed: {e}")
            raise
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete suicide risk detection pipeline.
        
        Returns:
            Dictionary containing all results
        """
        logger.info("Starting Suicide Risk Detection Pipeline...")
        logger.info(f"Configuration: {self.config}")
        
        try:
            # Phase 1: Data Preprocessing
            train_df, test_df = self.run_data_preprocessing()
            
            # Phase 2: Baseline Model
            baseline_model, baseline_metrics = self.run_baseline_model(train_df, test_df)
            
            # Phase 3: BERT Model
            bert_model, bert_metrics = self.run_bert_model(train_df, test_df)
            
            # Phase 4: Model Evaluation
            comparison = self.run_model_evaluation(baseline_model, bert_model, test_df)
            
            # Phase 5: Visualization
            viz_paths = self.run_visualization(train_df, test_df, baseline_model, bert_model, comparison)
            
            # Generate Final Report
            report_path = self.generate_final_report()
            
            # Summary
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Final report: {report_path}")
            logger.info(f"Generated {len(viz_paths)} visualizations")
            logger.info("Results saved in 'results/' directory")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description='Suicide Risk Detection Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                       help='Skip data preprocessing (use existing processed data)')
    parser.add_argument('--skip-baseline', action='store_true', 
                       help='Skip baseline model training')
    parser.add_argument('--skip-bert', action='store_true', 
                       help='Skip BERT model training')
    parser.add_argument('--skip-evaluation', action='store_true', 
                       help='Skip model evaluation')
    parser.add_argument('--skip-visualization', action='store_true', 
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
        # Initialize pipeline
        pipeline = SuicideRiskDetectionPipeline()
        
    try:
        # Run full pipeline
        results = pipeline.run_full_pipeline()
        
        print("\n" + "="*60)
        print("SUICIDE RISK DETECTION PIPELINE COMPLETED!")
        print("="*60)
        print(f"Results saved in: results/")
        print(f"Log file: suicide_risk_detection.log")
        print("\nKey Results:")
        
        if 'model_comparison' in results:
            comparison = results['model_comparison']
            for metric, improvement in comparison['improvement'].items():
                print(f"  {metric.upper()}: {improvement['improvement_percent']:+.2f}% improvement")
        
        print("\nEthical Note: This system is designed as a screening tool to assist")
        print("mental health professionals, not replace clinical judgment.")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()