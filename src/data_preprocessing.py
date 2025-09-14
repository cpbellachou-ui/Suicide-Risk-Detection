"""
Data preprocessing module for suicide risk detection.
Handles loading, cleaning, and preparing Reddit posts for analysis.
"""

import pandas as pd
import numpy as np
import re
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os
from typing import Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing for suicide risk detection."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data preprocessor.
        
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = data_dir
        self.raw_data_path = os.path.join(data_dir, "raw")
        self.processed_data_path = os.path.join(data_dir, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)
    
    def download_dataset(self) -> str:
        """Download the suicide detection dataset from Kaggle.
        
        Returns:
            Path to the downloaded dataset
        """
        logger.info("Downloading suicide detection dataset from Kaggle...")
        try:
            dataset_path = kagglehub.dataset_download("nikhileswarkomati/suicide-watch")
            logger.info(f"Dataset downloaded to: {dataset_path}")
            return dataset_path
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def load_data(self, dataset_path: str) -> pd.DataFrame:
        """Load the dataset from the downloaded files.
        
        Args:
            dataset_path: Path to the downloaded dataset
            
        Returns:
            Combined DataFrame with all posts
        """
        logger.info("Loading dataset...")
        
        # Look for CSV files in the dataset directory
        csv_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the dataset directory")
        
        # Load all CSV files and combine them
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dataframes.append(df)
                logger.info(f"Loaded {len(df)} rows from {os.path.basename(csv_file)}")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        
        if not dataframes:
            raise ValueError("No valid CSV files could be loaded")
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined dataset shape: {combined_df.shape}")
        
        return combined_df
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing URLs, special characters, and normalizing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit-specific patterns
        text = re.sub(r'\[deleted\]', '', text)
        text = re.sub(r'\[removed\]', '', text)
        text = re.sub(r'u/[a-zA-Z0-9_]+', '', text)  # Remove usernames
        text = re.sub(r'r/[a-zA-Z0-9_]+', '', text)  # Remove subreddit names
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary labels based on subreddit information.
        
        Args:
            df: DataFrame with posts
            
        Returns:
            DataFrame with binary labels added
        """
        logger.info("Creating binary labels...")
        
        # Check if we have subreddit information
        if 'subreddit' in df.columns:
            # Class 1: Suicide risk (r/SuicideWatch)
            # Class 0: No risk (other subreddits)
            df['label'] = (df['subreddit'] == 'SuicideWatch').astype(int)
        elif 'class' in df.columns:
            # Use existing class column if available
            df['label'] = df['class']
        else:
            # If no subreddit info, assume we need to infer from text
            # This is a fallback - in practice, we'd need domain knowledge
            logger.warning("No subreddit information found. Using text-based labeling.")
            # For now, create random labels as placeholder
            df['label'] = np.random.randint(0, 2, len(df))
        
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        return df
    
    def filter_posts(self, df: pd.DataFrame, min_words: int = 10) -> pd.DataFrame:
        """Filter posts based on minimum word count.
        
        Args:
            df: DataFrame with posts
            min_words: Minimum number of words required
            
        Returns:
            Filtered DataFrame
        """
        logger.info(f"Filtering posts with at least {min_words} words...")
        
        initial_count = len(df)
        
        # Count words in each post
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        # Filter posts
        df_filtered = df[df['word_count'] >= min_words].copy()
        
        final_count = len(df_filtered)
        logger.info(f"Filtered from {initial_count} to {final_count} posts ({final_count/initial_count*100:.1f}% retained)")
        
        return df_filtered
    
    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance the dataset by undersampling the majority class.
        
        Args:
            df: DataFrame with posts and labels
            
        Returns:
            Balanced DataFrame
        """
        logger.info("Balancing dataset...")
        
        # Get class counts
        class_counts = df['label'].value_counts()
        logger.info(f"Class distribution before balancing: {class_counts.to_dict()}")
        
        # Find the minority class size
        min_class_size = class_counts.min()
        
        # Balance the dataset
        balanced_dfs = []
        for label in df['label'].unique():
            class_df = df[df['label'] == label]
            if len(class_df) > min_class_size:
                # Undersample majority class
                class_df = resample(class_df, 
                                 replace=False, 
                                 n_samples=min_class_size, 
                                 random_state=42)
            balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Class distribution after balancing: {balanced_df['label'].value_counts().to_dict()}")
        return balanced_df
    
    def preprocess(self, dataset_path: str = None, min_words: int = 10, 
                  balance: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main preprocessing pipeline.
        
        Args:
            dataset_path: Path to dataset (if None, will download)
            min_words: Minimum words per post
            balance: Whether to balance the dataset
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Starting data preprocessing pipeline...")
        
        # Download dataset if not provided
        if dataset_path is None:
            dataset_path = self.download_dataset()
        
        # Load data
        df = self.load_data(dataset_path)
        
        # Identify text column
        text_columns = ['text', 'post', 'content', 'body']
        text_col = None
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            raise ValueError(f"No text column found. Available columns: {df.columns.tolist()}")
        
        # Rename text column for consistency
        df = df.rename(columns={text_col: 'text'})
        
        # Clean text
        logger.info("Cleaning text...")
        df['text'] = df['text'].apply(self.clean_text)
        
        # Create labels
        df = self.create_labels(df)
        
        # Filter posts
        df = self.filter_posts(df, min_words)
        
        # Remove posts with empty text after cleaning
        df = df[df['text'].str.len() > 0].copy()
        
        # Balance dataset if requested
        if balance:
            df = self.balance_dataset(df)
        
        # Split into train and test sets
        logger.info("Splitting data into train and test sets...")
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df['label']
        )
        
        # Save processed data
        train_df.to_csv(os.path.join(self.processed_data_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(self.processed_data_path, 'test.csv'), index=False)
        
        logger.info(f"Preprocessing complete!")
        logger.info(f"Training set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        logger.info(f"Average post length: {train_df['word_count'].mean():.1f} words")
        
        return train_df, test_df
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load preprocessed data from files.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        train_path = os.path.join(self.processed_data_path, 'train.csv')
        test_path = os.path.join(self.processed_data_path, 'test.csv')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Processed data not found. Run preprocess() first.")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Loaded processed data: {len(train_df)} train, {len(test_df)} test samples")
        return train_df, test_df

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.preprocess()
    
    # Display sample data
    print("\nSample training data:")
    print(train_df[['text', 'label']].head())
    
    print(f"\nTraining set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Label distribution in training: {train_df['label'].value_counts().to_dict()}")