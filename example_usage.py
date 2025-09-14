"""
Example usage script for the suicide risk detection system.
Demonstrates how to use individual components and the full pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from baseline_model import BaselineModel
from bert_model import BertModel
from evaluation import ModelEvaluator
from visualization import VisualizationGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_data_preprocessing():
    """Example of data preprocessing."""
    print("=" * 60)
    print("EXAMPLE: Data Preprocessing")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Example: Process data (this would normally download from Kaggle)
    # For demo purposes, we'll create sample data
    sample_data = {
        'text': [
            "I feel so hopeless and alone. Nothing seems to matter anymore.",
            "I'm excited about my new job and looking forward to the future!",
            "I can't take this pain anymore. I want it all to end.",
            "Thanks for the help and support. I'm feeling much better now.",
            "I'm planning my suicide. This is my final goodbye to everyone.",
            "I love spending time with my family and friends. Life is good."
        ],
        'subreddit': ['SuicideWatch', 'happy', 'SuicideWatch', 'grateful', 'SuicideWatch', 'positive']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create labels
    df['label'] = (df['subreddit'] == 'SuicideWatch').astype(int)
    
    # Clean text
    df['text'] = df['text'].apply(preprocessor.clean_text)
    
    # Filter by word count
    df = preprocessor.filter_posts(df, min_words=5)
    
    print("Sample processed data:")
    print(df[['text', 'label']].head())
    print(f"\nLabel distribution: {df['label'].value_counts().to_dict()}")
    
    return df

def example_baseline_model():
    """Example of baseline model training and evaluation."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Baseline Model (TF-IDF + Logistic Regression)")
    print("=" * 60)
    
    # Create sample data
    sample_texts = [
        "I feel so hopeless and alone. Nothing seems to matter anymore.",
        "I'm excited about my new job and looking forward to the future!",
        "I can't take this pain anymore. I want it all to end.",
        "Thanks for the help and support. I'm feeling much better now.",
        "I'm planning my suicide. This is my final goodbye to everyone.",
        "I love spending time with my family and friends. Life is good.",
        "I feel like a burden to everyone around me. They'd be better off without me.",
        "I'm grateful for all the wonderful people in my life. Thank you all!",
        "I've decided to end my life tonight. This is it.",
        "I'm looking forward to my vacation next week. Can't wait!"
    ]
    
    sample_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = suicide risk, 0 = no risk
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        sample_texts, sample_labels, test_size=0.3, random_state=42, stratify=sample_labels
    )
    
    # Initialize and train model
    model = BaselineModel(max_features=1000, ngram_range=(1, 2))
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    print("Baseline Model Performance:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Get top features
    top_features = model.get_top_features(5)
    print("\nTop features:")
    for class_label, features in top_features.items():
        print(f"  Class {class_label}: {[f[0] for f in features]}")
    
    return model, metrics

def example_bert_model():
    """Example of BERT model training and evaluation."""
    print("\n" + "=" * 60)
    print("EXAMPLE: BERT Model")
    print("=" * 60)
    
    # Create sample data
    sample_texts = [
        "I feel so hopeless and alone. Nothing seems to matter anymore.",
        "I'm excited about my new job and looking forward to the future!",
        "I can't take this pain anymore. I want it all to end.",
        "Thanks for the help and support. I'm feeling much better now.",
        "I'm planning my suicide. This is my final goodbye to everyone.",
        "I love spending time with my family and friends. Life is good.",
        "I feel like a burden to everyone around me. They'd be better off without me.",
        "I'm grateful for all the wonderful people in my life. Thank you all!",
        "I've decided to end my life tonight. This is it.",
        "I'm looking forward to my vacation next week. Can't wait!"
    ]
    
    sample_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        sample_texts, sample_labels, test_size=0.3, random_state=42, stratify=sample_labels
    )
    
    # Initialize model (with reduced parameters for demo)
    model = BertModel(
        max_length=64,
        batch_size=2,
        num_epochs=1,  # Reduced for demo
        learning_rate=2e-5
    )
    
    print("Training BERT model... (This may take a few minutes)")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    print("BERT Model Performance:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Get attention weights for a sample text
    sample_text = X_test[0]
    print(f"\nSample text: '{sample_text}'")
    
    try:
        attention_weights = model.get_attention_weights(sample_text)
        print(f"Attention weights shape: {[w.shape for w in attention_weights]}")
    except Exception as e:
        print(f"Could not extract attention weights: {e}")
    
    return model, metrics

def example_evaluation():
    """Example of model evaluation and comparison."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Model Evaluation and Comparison")
    print("=" * 60)
    
    # Example metrics (replace with actual model results)
    baseline_metrics = {
        'accuracy': 0.78,
        'precision': 0.76,
        'recall': 0.78,
        'f1_score': 0.77,
        'roc_auc': 0.82
    }
    
    bert_metrics = {
        'accuracy': 0.89,
        'precision': 0.88,
        'f1_score': 0.89,
        'recall': 0.90,
        'roc_auc': 0.94
    }
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Compare models
    comparison = evaluator.compare_models(baseline_metrics, bert_metrics)
    
    print("Model Comparison:")
    print("| Metric | TF-IDF + LogReg | BERT | Improvement |")
    print("|--------|-----------------|------|-------------|")
    
    for metric, data in comparison.items():
        improvement = data['improvement_pct']
        print(f"| {metric.capitalize()} | {data['baseline']:.4f} | {data['bert']:.4f} | {improvement:+.1f}% |")
    
    # Generate report
    baseline_results = {'metrics': baseline_metrics, 'dataset_size': 1000}
    bert_results = {'metrics': bert_metrics, 'dataset_size': 1000}
    
    report = evaluator.generate_evaluation_report(baseline_results, bert_results)
    print(f"\nEvaluation Report:\n{report}")

def example_visualization():
    """Example of visualization generation."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Visualization Generation")
    print("=" * 60)
    
    # Sample data for visualization
    texts_by_class = {
        0: [
            "I'm excited about my new job and looking forward to the future!",
            "Thanks for the help and support. I'm feeling much better now.",
            "I love spending time with my family and friends. Life is good.",
            "I'm grateful for all the wonderful people in my life. Thank you all!",
            "I'm looking forward to my vacation next week. Can't wait!"
        ],
        1: [
            "I feel so hopeless and alone. Nothing seems to matter anymore.",
            "I can't take this pain anymore. I want it all to end.",
            "I'm planning my suicide. This is my final goodbye to everyone.",
            "I feel like a burden to everyone around me. They'd be better off without me.",
            "I've decided to end my life tonight. This is it."
        ]
    }
    
    # Sample feature importance
    top_features = {
        0: [('good', 0.5), ('happy', 0.4), ('help', 0.3), ('thanks', 0.2), ('future', 0.1)],
        1: [('ending', 0.6), ('alone', 0.5), ('pain', 0.4), ('hopeless', 0.3), ('burden', 0.2)]
    }
    
    # Initialize visualizer
    visualizer = VisualizationGenerator()
    
    print("Generating visualizations...")
    
    # Create visualizations
    try:
        # Feature importance plot
        visualizer.create_feature_importance_plot(top_features)
        print("✓ Feature importance plot created")
        
        # Word clouds
        visualizer.create_word_clouds(texts_by_class)
        print("✓ Word clouds created")
        
        # Linguistic pattern analysis
        visualizer.create_linguistic_pattern_analysis(texts_by_class)
        print("✓ Linguistic pattern analysis created")
        
        # Text length analysis
        visualizer.create_text_length_analysis(texts_by_class)
        print("✓ Text length analysis created")
        
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Note: Some visualizations may require additional dependencies")

def example_full_pipeline():
    """Example of running the full pipeline."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Full Pipeline")
    print("=" * 60)
    
    try:
        # Import the main pipeline
        from main import SuicideRiskDetectionPipeline
        
        # Initialize pipeline with demo configuration
        config = {
            'data': {'min_words': 5, 'balance': True, 'test_size': 0.3},
            'baseline': {'max_features': 1000, 'ngram_range': (1, 2), 'min_df': 2, 'max_iter': 100},
            'bert': {'max_length': 64, 'batch_size': 2, 'learning_rate': 2e-5, 'num_epochs': 1},
            'evaluation': {'save_results': True, 'generate_plots': True}
        }
        
        pipeline = SuicideRiskDetectionPipeline(config)
        
        print("Note: Full pipeline requires actual dataset from Kaggle")
        print("This example shows the structure and configuration")
        print("\nPipeline configuration:")
        for component, params in config.items():
            print(f"  {component}: {params}")
        
    except Exception as e:
        print(f"Pipeline initialization error: {e}")
        print("Note: Full pipeline requires the complete dataset")

def main():
    """Run all examples."""
    print("SUICIDE RISK DETECTION - EXAMPLE USAGE")
    print("=" * 60)
    print("This script demonstrates how to use the suicide risk detection system.")
    print("Note: Some examples use sample data for demonstration purposes.")
    print("=" * 60)
    
    try:
        # Run examples
        example_data_preprocessing()
        example_baseline_model()
        example_bert_model()
        example_evaluation()
        example_visualization()
        example_full_pipeline()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install required dependencies: pip install -r requirements.txt")
        print("2. Download the dataset from Kaggle")
        print("3. Run the full pipeline: python main.py")
        print("4. Check the 'results' directory for outputs")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Please check the error message and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()
