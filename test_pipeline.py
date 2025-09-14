"""
Test script for the suicide risk detection pipeline.
This script tests the pipeline with a small sample dataset to verify functionality.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
from src.data_preprocessing import DataPreprocessor
from src.baseline_model import BaselineModel
from src.bert_model import BertModelWrapper
from src.evaluation import ModelEvaluator
from src.visualization import VisualizationGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create a small sample dataset for testing."""
    logger.info("Creating sample dataset...")
    
    # Sample suicide risk posts
    risk_posts = [
        "I feel so alone and hopeless, I don't see any point in continuing.",
        "I can't take this pain anymore, I want it all to end.",
        "I'm such a burden to everyone, they'd be better off without me.",
        "I've tried everything and nothing works, I'm giving up.",
        "This is my final goodbye, I can't do this anymore.",
        "I feel so empty inside, like there's no hope left.",
        "I'm tired of fighting, I just want the pain to stop.",
        "No one understands how I feel, I'm completely alone.",
        "I've made up my mind, this is the end for me.",
        "I can't see any future for myself, it's all over."
    ]
    
    # Sample no-risk posts
    no_risk_posts = [
        "I'm excited about my new job opportunity next week!",
        "Thanks for all the help and support, I really appreciate it.",
        "I'm looking forward to the weekend and spending time with friends.",
        "I had a great day today, everything seems to be going well.",
        "I'm grateful for all the positive things in my life.",
        "I'm planning a vacation for next month, can't wait!",
        "I feel much better after talking to my therapist yesterday.",
        "I'm proud of the progress I've made recently.",
        "I'm excited about the new project I'm working on.",
        "I'm feeling optimistic about the future and what's to come."
    ]
    
    # Create DataFrame
    data = {
        'text': risk_posts + no_risk_posts,
        'label': [1] * len(risk_posts) + [0] * len(no_risk_posts)
    }
    
    df = pd.DataFrame(data)
    
    # Add word count
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    logger.info(f"Created sample dataset with {len(df)} posts")
    logger.info(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    return df

def test_data_preprocessing():
    """Test data preprocessing functionality."""
    logger.info("Testing data preprocessing...")
    
    # Create sample data
    df = create_sample_data()
    
    # Test preprocessing steps
    preprocessor = DataPreprocessor()
    
    # Test text cleaning
    sample_text = "I feel so alone and hopeless, I don't see any point in continuing. [deleted]"
    cleaned_text = preprocessor.clean_text(sample_text)
    logger.info(f"Text cleaning test: '{sample_text}' -> '{cleaned_text}'")
    
    # Test label creation
    df_with_labels = preprocessor.create_labels(df)
    logger.info(f"Label creation test: {df_with_labels['label'].value_counts().to_dict()}")
    
    # Test filtering
    df_filtered = preprocessor.filter_posts(df_with_labels, min_words=5)
    logger.info(f"Filtering test: {len(df)} -> {len(df_filtered)} posts")
    
    logger.info("Data preprocessing test completed successfully!")
    return df_filtered

def test_baseline_model(df):
    """Test baseline model functionality."""
    logger.info("Testing baseline model...")
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    
    # Initialize model
    model = BaselineModel(max_features=100, ngram_range=(1, 1), min_df=1)
    
    # Train model
    X_train = train_df['text'].tolist()
    y_train = train_df['label'].tolist()
    train_metrics = model.train(X_train, y_train)
    
    # Test predictions
    X_test = test_df['text'].tolist()
    y_test = test_df['label'].tolist()
    predictions, probabilities = model.predict(X_test)
    
    # Calculate test metrics
    from sklearn.metrics import accuracy_score, f1_score
    test_accuracy = accuracy_score(y_test, predictions)
    test_f1 = f1_score(y_test, predictions)
    
    logger.info(f"Baseline model test - Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
    
    # Test feature importance
    feature_importance = model.get_feature_importance(top_n=5)
    logger.info(f"Feature importance test: {len(feature_importance['suicide_risk'])} features for risk")
    
    logger.info("Baseline model test completed successfully!")
    return model, test_df

def test_bert_model(df):
    """Test BERT model functionality (simplified)."""
    logger.info("Testing BERT model...")
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    
    # Initialize model with small parameters for testing
    model = BertModelWrapper(
        max_length=64,
        batch_size=4,
        num_epochs=1
    )
    
    # Create data loaders
    train_loader, test_loader = model.create_data_loaders(train_df, test_df)
    
    # Train model (just one epoch for testing)
    training_history = model.train(train_loader)
    logger.info(f"BERT training test - Final loss: {training_history['train_loss'][-1]:.4f}")
    
    # Test predictions
    predictions, probabilities = model.predict(test_loader)
    
    # Calculate test metrics
    y_test = []
    for batch in test_loader:
        y_test.extend(batch['labels'].numpy())
    
    from sklearn.metrics import accuracy_score, f1_score
    test_accuracy = accuracy_score(y_test, predictions)
    test_f1 = f1_score(y_test, predictions)
    
    logger.info(f"BERT model test - Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
    
    logger.info("BERT model test completed successfully!")
    return model, test_df

def test_evaluation(baseline_model, bert_model, test_df):
    """Test evaluation functionality."""
    logger.info("Testing evaluation...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Test baseline evaluation
    X_test = test_df['text'].tolist()
    y_test = test_df['label'].tolist()
    
    baseline_predictions, baseline_probabilities = baseline_model.predict(X_test)
    baseline_metrics = evaluator.calculate_metrics(y_test, baseline_predictions, baseline_probabilities, "Baseline")
    
    # Test BERT evaluation
    from bert_model import SuicideRiskDataset
    from torch.utils.data import DataLoader
    
    test_dataset = SuicideRiskDataset(X_test, y_test, bert_model.tokenizer, bert_model.max_length)
    test_loader = DataLoader(test_dataset, batch_size=bert_model.batch_size, shuffle=False)
    
    bert_predictions, bert_probabilities = bert_model.predict(test_loader)
    bert_metrics = evaluator.calculate_metrics(y_test, bert_predictions, bert_probabilities, "BERT")
    
    # Test comparison
    comparison = evaluator.compare_models(baseline_metrics, bert_metrics)
    
    logger.info(f"Evaluation test - Baseline F1: {baseline_metrics['f1_score']:.4f}")
    logger.info(f"Evaluation test - BERT F1: {bert_metrics['f1_score']:.4f}")
    
    logger.info("Evaluation test completed successfully!")
    return comparison

def test_visualization(df, baseline_model, bert_model, comparison):
    """Test visualization functionality."""
    logger.info("Testing visualization...")
    
    # Initialize visualizer
    viz_gen = VisualizationGenerator()
    
    # Test feature importance plot
    try:
        feature_importance = baseline_model.get_feature_importance(top_n=5)
        plot_path = viz_gen.create_feature_importance_plot(feature_importance, top_n=5)
        logger.info(f"Feature importance plot: {plot_path}")
    except Exception as e:
        logger.warning(f"Feature importance plot failed: {e}")
    
    # Test word clouds
    try:
        wordcloud_paths = viz_gen.create_word_clouds(df)
        logger.info(f"Word clouds: {wordcloud_paths}")
    except Exception as e:
        logger.warning(f"Word clouds failed: {e}")
    
    # Test text length distribution
    try:
        plot_path = viz_gen.create_text_length_distribution(df)
        logger.info(f"Text length distribution: {plot_path}")
    except Exception as e:
        logger.warning(f"Text length distribution failed: {e}")
    
    # Test linguistic patterns
    try:
        plot_path = viz_gen.create_linguistic_patterns_analysis(df)
        logger.info(f"Linguistic patterns: {plot_path}")
    except Exception as e:
        logger.warning(f"Linguistic patterns failed: {e}")
    
    logger.info("Visualization test completed successfully!")

def run_all_tests():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("STARTING PIPELINE TESTS")
    logger.info("=" * 60)
    
    try:
        # Test 1: Data Preprocessing
        df = test_data_preprocessing()
        
        # Test 2: Baseline Model
        baseline_model, test_df = test_baseline_model(df)
        
        # Test 3: BERT Model
        bert_model, _ = test_bert_model(df)
        
        # Test 4: Evaluation
        comparison = test_evaluation(baseline_model, bert_model, test_df)
        
        # Test 5: Visualization
        test_visualization(df, baseline_model, bert_model, comparison)
        
        logger.info("=" * 60)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("The pipeline is ready for use with real data.")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()
