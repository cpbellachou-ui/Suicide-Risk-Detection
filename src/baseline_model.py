"""
Baseline model implementation using TF-IDF and Logistic Regression.
This serves as a traditional machine learning baseline for comparison.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class BaselineModel:
    """TF-IDF + Logistic Regression baseline model for suicide risk detection."""
    
    def __init__(self, 
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 5,
                 random_state: int = 42):
        """Initialize the baseline model.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: N-gram range for feature extraction
            min_df: Minimum document frequency for features
            random_state: Random state for reproducibility
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.random_state = random_state
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.classifier = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            C=1.0,  # L2 regularization
            solver='liblinear'
        )
        
        self.is_trained = False
        self.feature_names = None
        self.model_dir = "models/baseline"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract TF-IDF features from text.
        
        Args:
            texts: List of text documents
            
        Returns:
            TF-IDF feature matrix
        """
        return self.vectorizer.fit_transform(texts).toarray()
    
    def train(self, X_train: List[str], y_train: List[int]) -> Dict[str, Any]:
        """Train the baseline model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            
        Returns:
            Training metrics
        """
        logger.info("Training baseline model (TF-IDF + Logistic Regression)...")
        
        # Extract features
        logger.info("Extracting TF-IDF features...")
        X_train_features = self._extract_features(X_train)
        
        # Store feature names for analysis
        self.feature_names = self.vectorizer.get_feature_names_out()
        logger.info(f"Extracted {len(self.feature_names)} features")
        
        # Train classifier
        logger.info("Training Logistic Regression classifier...")
        self.classifier.fit(X_train_features, y_train)
        
        # Calculate training metrics
        y_train_pred = self.classifier.predict(X_train_features)
        train_metrics = self._calculate_metrics(y_train, y_train_pred, "Training")
        
        self.is_trained = True
        logger.info("Baseline model training completed!")
        
        return train_metrics
    
    def predict(self, X_test: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on test data.
        
        Args:
            X_test: Test text data
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        X_test_features = self.vectorizer.transform(X_test).toarray()
        
        # Make predictions
        predictions = self.classifier.predict(X_test_features)
        probabilities = self.classifier.predict_proba(X_test_features)
        
        return predictions, probabilities
    
    def evaluate(self, X_test: List[str], y_test: List[int]) -> Dict[str, Any]:
        """Evaluate the model on test data.
        
        Args:
            X_test: Test text data
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating baseline model...")
        
        # Make predictions
        y_pred, y_proba = self.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, "Test")
        
        # Add probability-based metrics
        metrics['auc_roc'] = roc_auc_score(y_test, y_proba[:, 1])
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Test AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true: List[int], y_pred: List[int], 
                          prefix: str = "") -> Dict[str, Any]:
        """Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            f'{prefix.lower()}_accuracy': accuracy_score(y_true, y_pred),
            f'{prefix.lower()}_precision': precision_score(y_true, y_pred, average='weighted'),
            f'{prefix.lower()}_recall': recall_score(y_true, y_pred, average='weighted'),
            f'{prefix.lower()}_f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """Get the most important features for each class.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with top features for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before extracting features")
        
        # Get feature coefficients
        coefficients = self.classifier.coef_[0]  # Binary classification
        
        # Get top positive and negative features
        top_positive_indices = np.argsort(coefficients)[-top_n:][::-1]
        top_negative_indices = np.argsort(coefficients)[:top_n]
        
        # Create feature importance dictionary
        feature_importance = {
            'suicide_risk': [(self.feature_names[i], coefficients[i]) 
                            for i in top_positive_indices],
            'no_risk': [(self.feature_names[i], coefficients[i]) 
                       for i in top_negative_indices]
        }
        
        return feature_importance
    
    def save_model(self, model_name: str = "baseline_model"):
        """Save the trained model to disk.
        
        Args:
            model_name: Name for the saved model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        vectorizer_path = os.path.join(self.model_dir, f"{model_name}_vectorizer.joblib")
        
        # Save model and vectorizer
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_name: str = "baseline_model"):
        """Load a trained model from disk.
        
        Args:
            model_name: Name of the saved model
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        vectorizer_path = os.path.join(self.model_dir, f"{model_name}_vectorizer.joblib")
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model files not found")
        
        # Load model and vectorizer
        self.classifier = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        # Get feature names
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture and parameters.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'TF-IDF + Logistic Regression',
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names) if self.feature_names is not None else 0,
            'classifier_params': self.classifier.get_params()
        }

def train_baseline_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[BaselineModel, Dict[str, Any]]:
    """Train the baseline model on the provided data.
    
    Args:
        train_df: Training DataFrame with 'text' and 'label' columns
        test_df: Test DataFrame with 'text' and 'label' columns
        
    Returns:
        Tuple of (trained_model, test_metrics)
    """
    # Initialize model
    model = BaselineModel()
    
    # Prepare data
    X_train = train_df['text'].tolist()
    y_train = train_df['label'].tolist()
    X_test = test_df['text'].tolist()
    y_test = test_df['label'].tolist()
    
    # Train model
    train_metrics = model.train(X_train, y_train)
    
    # Evaluate model
    test_metrics = model.evaluate(X_test, y_test)
    
    # Save model
    model.save_model()
    
    return model, test_metrics

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_processed_data()
    
    # Train baseline model
    model, metrics = train_baseline_model(train_df, test_df)
    
    # Print results
    print("\nBaseline Model Results:")
    print(f"Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"F1-Score: {metrics['test_f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    
    # Get feature importance
    feature_importance = model.get_feature_importance(top_n=10)
    print("\nTop features for suicide risk:")
    for word, score in feature_importance['suicide_risk'][:5]:
        print(f"  {word}: {score:.4f}")
    
    print("\nTop features for no risk:")
    for word, score in feature_importance['no_risk'][:5]:
        print(f"  {word}: {score:.4f}")