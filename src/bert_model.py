"""
BERT-based deep learning model for suicide risk detection.
Implements a fine-tuned BERT classifier with attention visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import os
from typing import Tuple, List, Dict, Any, Optional
import logging
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

class SuicideRiskDataset(Dataset):
    """Custom dataset for suicide risk detection."""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer: BertTokenizer, max_length: int = 128):
        """Initialize the dataset.
        
        Args:
            texts: List of text documents
            labels: List of corresponding labels
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertClassifier(nn.Module):
    """BERT-based classifier for suicide risk detection."""
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        """Initialize the BERT classifier.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(BertClassifier, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Softmax for probability output
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask, return_attention=False):
        """Forward pass through the model.
        
        Args:
            input_ids: Tokenized input sequences
            attention_mask: Attention mask for input sequences
            return_attention: Whether to return attention weights
            
        Returns:
            Model outputs and optionally attention weights
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attention
        )
        
        # Get pooled output (CLS token representation)
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification head
        logits = self.classifier(pooled_output)
        
        if return_attention:
            return logits, outputs.attentions
        else:
            return logits

class BertModelWrapper:
    """Wrapper class for BERT model training and evaluation."""
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 max_length: int = 128,
                 batch_size: int = 16,
                 learning_rate: float = 2e-5,
                 num_epochs: int = 3,
                 device: Optional[str] = None):
        """Initialize the BERT model wrapper.
        
        Args:
            model_name: Name of the pre-trained BERT model
            max_length: Maximum sequence length
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = BertClassifier().to(self.device)
        
        # Training state
        self.is_trained = False
        self.training_history = []
        self.model_dir = "models/bert"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def create_data_loaders(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for training and testing.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Create datasets
        train_dataset = SuicideRiskDataset(
            train_df['text'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer,
            self.max_length
        )
        
        test_dataset = SuicideRiskDataset(
            test_df['text'].tolist(),
            test_df['label'].tolist(),
            self.tokenizer,
            self.max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, test_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None) -> Dict[str, List[float]]:
        """Train the BERT model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Training history
        """
        logger.info("Training BERT model...")
        
        # Set model to training mode
        self.model.train()
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Calculate total training steps
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Training loop
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, scheduler, criterion)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._evaluate_epoch(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        self.is_trained = True
        self.training_history = history
        
        logger.info("BERT model training completed!")
        return history
    
    def _train_epoch(self, train_loader: DataLoader, optimizer, scheduler, criterion) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def _evaluate_epoch(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """Evaluate for one epoch.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        self.model.train()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating BERT model...")
        
        # Get true labels
        true_labels = []
        for batch in test_loader:
            true_labels.extend(batch['labels'].numpy())
        true_labels = np.array(true_labels)
        
        # Make predictions
        predictions, probabilities = self.predict(test_loader)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='weighted'),
            'recall': recall_score(true_labels, predictions, average='weighted'),
            'f1_score': f1_score(true_labels, predictions, average='weighted'),
            'auc_roc': roc_auc_score(true_labels, probabilities[:, 1]),
            'confusion_matrix': confusion_matrix(true_labels, predictions)
        }
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Test AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return metrics
    
    def get_attention_weights(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """Get attention weights for a given text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (attention_weights, tokens)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting attention weights")
        
        self.model.eval()
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get attention weights
        with torch.no_grad():
            outputs, attentions = self.model(input_ids, attention_mask, return_attention=True)
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Average attention across all layers and heads
        # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.stack(attentions)
        attention_weights = attention_weights.mean(dim=(0, 2))  # Average across layers and heads
        attention_weights = attention_weights.squeeze(0)  # Remove batch dimension
        
        return attention_weights.cpu().numpy(), tokens
    
    def save_model(self, model_name: str = "bert_model"):
        """Save the trained model.
        
        Args:
            model_name: Name for the saved model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_path = os.path.join(self.model_dir, f"{model_name}.pth")
        config_path = os.path.join(self.model_dir, f"{model_name}_config.json")
        
        # Save model state
        torch.save(self.model.state_dict(), model_path)
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'training_history': self.training_history
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Config saved to {config_path}")
    
    def load_model(self, model_name: str = "bert_model"):
        """Load a trained model.
        
        Args:
            model_name: Name of the saved model
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.pth")
        config_path = os.path.join(self.model_dir, f"{model_name}_config.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model state
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load configuration
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.training_history = config.get('training_history', [])
        
        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")

def train_bert_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[BertModelWrapper, Dict[str, Any]]:
    """Train the BERT model on the provided data.
    
    Args:
        train_df: Training DataFrame with 'text' and 'label' columns
        test_df: Test DataFrame with 'text' and 'label' columns
        
    Returns:
        Tuple of (trained_model, test_metrics)
    """
    # Initialize model
    model = BertModelWrapper()
    
    # Create data loaders
    train_loader, test_loader = model.create_data_loaders(train_df, test_df)
    
    # Train model
    training_history = model.train(train_loader)
    
    # Evaluate model
    test_metrics = model.evaluate(test_loader)
    
    # Save model
    model.save_model()
    
    return model, test_metrics

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_processed_data()
    
    # Train BERT model
    model, metrics = train_bert_model(train_df, test_df)
    
    # Print results
    print("\nBERT Model Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    
    # Get attention weights for a sample text
    sample_text = "I feel so alone and hopeless, I don't see any point in continuing."
    attention_weights, tokens = model.get_attention_weights(sample_text)
    print(f"\nAttention analysis for: '{sample_text}'")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Attention shape: {attention_weights.shape}")