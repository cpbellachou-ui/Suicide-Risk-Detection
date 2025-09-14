"""
Visualization module for suicide risk detection analysis.
Creates comprehensive visualizations for model analysis and linguistic patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Tuple
import logging
import os
from collections import Counter
import re

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """Generates comprehensive visualizations for suicide risk detection analysis."""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the visualization generator.
        
        Args:
            results_dir: Directory to save visualizations
        """
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, "figures")
        
        # Create directory
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Set default colors
        self.colors = {
            'suicide_risk': '#e74c3c',  # Red
            'no_risk': '#2ecc71',       # Green
            'baseline': '#3498db',      # Blue
            'bert': '#9b59b6'           # Purple
        }
    
    def create_feature_importance_plot(self, feature_importance: Dict[str, List[Tuple[str, float]]], 
                                     top_n: int = 20) -> str:
        """Create feature importance visualization from TF-IDF model.
        
        Args:
            feature_importance: Feature importance from baseline model
            top_n: Number of top features to display
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating feature importance plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Suicide risk features
        risk_features = feature_importance['suicide_risk'][:top_n]
        risk_words = [word for word, _ in risk_features]
        risk_scores = [score for _, score in risk_features]
        
        ax1.barh(range(len(risk_words)), risk_scores, color=self.colors['suicide_risk'], alpha=0.7)
        ax1.set_yticks(range(len(risk_words)))
        ax1.set_yticklabels(risk_words)
        ax1.set_xlabel('Feature Importance Score')
        ax1.set_title('Top Features for Suicide Risk Detection', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # No risk features
        no_risk_features = feature_importance['no_risk'][:top_n]
        no_risk_words = [word for word, _ in no_risk_features]
        no_risk_scores = [score for _, score in no_risk_features]
        
        ax2.barh(range(len(no_risk_words)), no_risk_scores, color=self.colors['no_risk'], alpha=0.7)
        ax2.set_yticks(range(len(no_risk_words)))
        ax2.set_yticklabels(no_risk_words)
        ax2.set_xlabel('Feature Importance Score')
        ax2.set_title('Top Features for No Risk Classification', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Feature Importance Analysis (TF-IDF + Logistic Regression)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.figures_dir, 'feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_path}")
        return plot_path
    
    def create_attention_heatmap(self, attention_weights: np.ndarray, tokens: List[str], 
                               text: str, model_name: str = "BERT") -> str:
        """Create attention heatmap visualization.
        
        Args:
            attention_weights: Attention weights from BERT model
            tokens: Tokenized text
            text: Original text
            model_name: Name of the model
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating attention heatmap...")
        
        # Filter out special tokens and limit length
        filtered_tokens = []
        filtered_attention = []
        
        for i, token in enumerate(tokens):
            if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##'):
                filtered_tokens.append(token)
                if i < len(attention_weights):
                    filtered_attention.append(attention_weights[i])
        
        if not filtered_tokens:
            logger.warning("No valid tokens found for attention visualization")
            return None
        
        # Limit to reasonable number of tokens
        max_tokens = 20
        if len(filtered_tokens) > max_tokens:
            filtered_tokens = filtered_tokens[:max_tokens]
            filtered_attention = filtered_attention[:max_tokens]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create attention matrix (self-attention)
        attention_matrix = np.array(filtered_attention)
        if attention_matrix.ndim == 1:
            # If 1D, create a simple bar plot
            bars = ax.bar(range(len(filtered_tokens)), attention_matrix, 
                         color=self.colors['bert'], alpha=0.7)
            ax.set_xticks(range(len(filtered_tokens)))
            ax.set_xticklabels(filtered_tokens, rotation=45, ha='right')
            ax.set_ylabel('Attention Weight')
            ax.set_title(f'BERT Attention Weights: "{text[:50]}..."', fontweight='bold')
        else:
            # If 2D, create heatmap
            im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')
            ax.set_xticks(range(len(filtered_tokens)))
            ax.set_xticklabels(filtered_tokens, rotation=45, ha='right')
            ax.set_yticks(range(len(filtered_tokens)))
            ax.set_yticklabels(filtered_tokens)
            ax.set_title(f'BERT Attention Heatmap: "{text[:50]}..."', fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.figures_dir, 'attention_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attention heatmap saved to {plot_path}")
        return plot_path
    
    def create_word_clouds(self, train_df: pd.DataFrame) -> List[str]:
        """Create word clouds for suicide risk vs no risk posts.
        
        Args:
            train_df: Training DataFrame with text and labels
            
        Returns:
            List of paths to saved word clouds
        """
        logger.info("Creating word clouds...")
        
        # Separate texts by label
        risk_texts = train_df[train_df['label'] == 1]['text'].tolist()
        no_risk_texts = train_df[train_df['label'] == 0]['text'].tolist()
        
        # Combine texts
        risk_text = ' '.join(risk_texts)
        no_risk_text = ' '.join(no_risk_texts)
        
        # Create word clouds
        wordclouds = []
        
        # Suicide risk word cloud
        risk_wc = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Reds',
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(risk_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(risk_wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud: Suicide Risk Posts', fontsize=16, fontweight='bold')
        
        risk_plot_path = os.path.join(self.figures_dir, 'wordcloud_suicide_risk.png')
        plt.savefig(risk_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        wordclouds.append(risk_plot_path)
        
        # No risk word cloud
        no_risk_wc = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Greens',
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(no_risk_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(no_risk_wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud: No Risk Posts', fontsize=16, fontweight='bold')
        
        no_risk_plot_path = os.path.join(self.figures_dir, 'wordcloud_no_risk.png')
        plt.savefig(no_risk_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        wordclouds.append(no_risk_plot_path)
        
        logger.info(f"Word clouds saved to {wordclouds}")
        return wordclouds
    
    def create_text_length_distribution(self, train_df: pd.DataFrame) -> str:
        """Create distribution of text lengths by class.
        
        Args:
            train_df: Training DataFrame with text and labels
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating text length distribution plot...")
        
        # Calculate text lengths
        train_df['text_length'] = train_df['text'].str.len()
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        risk_lengths = train_df[train_df['label'] == 1]['text_length']
        no_risk_lengths = train_df[train_df['label'] == 0]['text_length']
        
        ax1.hist(risk_lengths, bins=50, alpha=0.7, label='Suicide Risk', 
                color=self.colors['suicide_risk'], density=True)
        ax1.hist(no_risk_lengths, bins=50, alpha=0.7, label='No Risk', 
                color=self.colors['no_risk'], density=True)
        ax1.set_xlabel('Text Length (characters)')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Text Lengths', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data_for_box = [risk_lengths, no_risk_lengths]
        labels = ['Suicide Risk', 'No Risk']
        
        box_plot = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor(self.colors['suicide_risk'])
        box_plot['boxes'][1].set_facecolor(self.colors['no_risk'])
        
        ax2.set_ylabel('Text Length (characters)')
        ax2.set_title('Text Length by Class', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Text Length Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.figures_dir, 'text_length_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Text length distribution plot saved to {plot_path}")
        return plot_path
    
    def create_linguistic_patterns_analysis(self, train_df: pd.DataFrame) -> str:
        """Analyze and visualize linguistic patterns.
        
        Args:
            train_df: Training DataFrame with text and labels
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating linguistic patterns analysis...")
        
        # Define linguistic patterns to analyze
        patterns = {
            'hopelessness': ['hopeless', 'hopelessness', 'despair', 'desperate', 'no point', 'pointless'],
            'finality': ['ending', 'goodbye', 'farewell', 'last time', 'final', 'forever'],
            'burden': ['burden', 'burdensome', 'trouble', 'problem', 'waste', 'useless'],
            'isolation': ['alone', 'lonely', 'isolated', 'abandoned', 'rejected', 'unwanted'],
            'pain': ['pain', 'hurt', 'suffering', 'agony', 'torture', 'misery'],
            'future_planning': ['tomorrow', 'future', 'plan', 'hope', 'better', 'improve'],
            'support_seeking': ['help', 'advice', 'support', 'someone', 'talk', 'listen'],
            'positive_emotion': ['happy', 'joy', 'excited', 'grateful', 'thankful', 'blessed']
        }
        
        # Count patterns in each class
        pattern_counts = {}
        for pattern_name, keywords in patterns.items():
            risk_count = 0
            no_risk_count = 0
            
            for _, row in train_df.iterrows():
                text = str(row['text']).lower()
                pattern_found = any(keyword in text for keyword in keywords)
                
                if pattern_found:
                    if row['label'] == 1:
                        risk_count += 1
                    else:
                        no_risk_count += 1
            
            pattern_counts[pattern_name] = {
                'suicide_risk': risk_count,
                'no_risk': no_risk_count
            }
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        
        pattern_names = list(pattern_counts.keys())
        risk_counts = [pattern_counts[p]['suicide_risk'] for p in pattern_names]
        no_risk_counts = [pattern_counts[p]['no_risk'] for p in pattern_names]
        
        x = np.arange(len(pattern_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, risk_counts, width, label='Suicide Risk', 
                      color=self.colors['suicide_risk'], alpha=0.8)
        bars2 = ax.bar(x + width/2, no_risk_counts, width, label='No Risk', 
                      color=self.colors['no_risk'], alpha=0.8)
        
        ax.set_xlabel('Linguistic Patterns', fontsize=12)
        ax.set_ylabel('Number of Posts', fontsize=12)
        ax.set_title('Linguistic Pattern Analysis by Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pattern_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.figures_dir, 'linguistic_patterns_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Linguistic patterns analysis saved to {plot_path}")
        return plot_path
    
    def create_training_history_plot(self, training_history: Dict[str, List[float]]) -> str:
        """Create training history visualization for BERT model.
        
        Args:
            training_history: Training history from BERT model
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating training history plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(training_history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in training_history:
            ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, training_history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in training_history:
            ax2.plot(epochs, training_history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('BERT Model Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.figures_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved to {plot_path}")
        return plot_path
    
    def create_interactive_dashboard(self, comparison: Dict[str, Any], 
                                   feature_importance: Dict[str, List[Tuple[str, float]]]) -> str:
        """Create an interactive dashboard using Plotly.
        
        Args:
            comparison: Model comparison results
            feature_importance: Feature importance from baseline model
            
        Returns:
            Path to saved HTML dashboard
        """
        logger.info("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Comparison', 'Feature Importance (Suicide Risk)',
                          'Feature Importance (No Risk)', 'Confusion Matrix Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )
        
        # Performance comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        baseline_scores = [comparison['baseline'].get(m, 0) for m in metrics]
        bert_scores = [comparison['bert'].get(m, 0) for m in metrics]
        
        fig.add_trace(
            go.Bar(name='Baseline', x=metrics, y=baseline_scores, marker_color='blue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='BERT', x=metrics, y=bert_scores, marker_color='purple'),
            row=1, col=1
        )
        
        # Feature importance - Suicide Risk
        risk_features = feature_importance['suicide_risk'][:10]
        risk_words = [word for word, _ in risk_features]
        risk_scores = [score for _, score in risk_features]
        
        fig.add_trace(
            go.Bar(name='Suicide Risk Features', x=risk_scores, y=risk_words, 
                  orientation='h', marker_color='red'),
            row=1, col=2
        )
        
        # Feature importance - No Risk
        no_risk_features = feature_importance['no_risk'][:10]
        no_risk_words = [word for word, _ in no_risk_features]
        no_risk_scores = [score for _, score in no_risk_features]
        
        fig.add_trace(
            go.Bar(name='No Risk Features', x=no_risk_scores, y=no_risk_words, 
                  orientation='h', marker_color='green'),
            row=2, col=1
        )
        
        # Confusion matrices
        baseline_cm = comparison['baseline']['confusion_matrix']
        bert_cm = comparison['bert']['confusion_matrix']
        
        # Combine confusion matrices for comparison
        combined_cm = np.vstack([baseline_cm, bert_cm])
        
        fig.add_trace(
            go.Heatmap(z=combined_cm, 
                      x=['No Risk', 'Suicide Risk'],
                      y=['Baseline No Risk', 'Baseline Suicide Risk', 
                         'BERT No Risk', 'BERT Suicide Risk'],
                      colorscale='Blues'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Suicide Risk Detection - Interactive Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Save dashboard
        dashboard_path = os.path.join(self.figures_dir, 'interactive_dashboard.html')
        fig.write_html(dashboard_path)
        
        logger.info(f"Interactive dashboard saved to {dashboard_path}")
        return dashboard_path
    
    def generate_all_visualizations(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  baseline_model, bert_model, comparison: Dict[str, Any]) -> List[str]:
        """Generate all visualizations for the project.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            baseline_model: Trained baseline model
            bert_model: Trained BERT model
            comparison: Model comparison results
            
        Returns:
            List of paths to all generated visualizations
        """
        logger.info("Generating all visualizations...")
        
        generated_plots = []
        
        # Feature importance plot
        try:
            feature_importance = baseline_model.get_feature_importance(top_n=20)
            plot_path = self.create_feature_importance_plot(feature_importance)
            generated_plots.append(plot_path)
        except Exception as e:
            logger.warning(f"Failed to create feature importance plot: {e}")
        
        # Word clouds
        try:
            wordcloud_paths = self.create_word_clouds(train_df)
            generated_plots.extend(wordcloud_paths)
        except Exception as e:
            logger.warning(f"Failed to create word clouds: {e}")
        
        # Text length distribution
        try:
            plot_path = self.create_text_length_distribution(train_df)
            generated_plots.append(plot_path)
        except Exception as e:
            logger.warning(f"Failed to create text length distribution: {e}")
        
        # Linguistic patterns analysis
        try:
            plot_path = self.create_linguistic_patterns_analysis(train_df)
            generated_plots.append(plot_path)
        except Exception as e:
            logger.warning(f"Failed to create linguistic patterns analysis: {e}")
        
        # Training history plot
        try:
            if hasattr(bert_model, 'training_history') and bert_model.training_history:
                plot_path = self.create_training_history_plot(bert_model.training_history)
                generated_plots.append(plot_path)
        except Exception as e:
            logger.warning(f"Failed to create training history plot: {e}")
        
        # Interactive dashboard
        try:
            if 'feature_importance' in locals():
                dashboard_path = self.create_interactive_dashboard(comparison, feature_importance)
                generated_plots.append(dashboard_path)
        except Exception as e:
            logger.warning(f"Failed to create interactive dashboard: {e}")
        
        logger.info(f"Generated {len(generated_plots)} visualizations")
        return generated_plots

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    from baseline_model import BaselineModel
    from bert_model import BertModelWrapper
    
    # Load data
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_processed_data()
    
    # Load models
    baseline_model = BaselineModel()
    baseline_model.load_model()
    
    bert_model = BertModelWrapper()
    bert_model.load_model()
    
    # Create sample comparison data
    comparison = {
        'baseline': {'accuracy': 0.78, 'f1_score': 0.76, 'auc_roc': 0.82},
        'bert': {'accuracy': 0.89, 'f1_score': 0.87, 'auc_roc': 0.93}
    }
    
    # Generate visualizations
    viz_gen = VisualizationGenerator()
    plots = viz_gen.generate_all_visualizations(train_df, test_df, baseline_model, bert_model, comparison)
    
    print(f"Generated {len(plots)} visualizations:")
    for plot in plots:
        print(f"  - {plot}")