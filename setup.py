"""
Setup script for the Suicide Risk Detection project.
Handles installation of dependencies and environment setup.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✓ Python version: {sys.version}")

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def setup_directories():
    """Create necessary directories."""
    print("Setting up project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "models/baseline",
        "models/bert",
        "results/metrics",
        "results/figures"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_kaggle_setup():
    """Check if Kaggle is properly configured."""
    print("Checking Kaggle configuration...")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_key = kaggle_dir / "kaggle.json"
    
    if not kaggle_key.exists():
        print("⚠️  Kaggle API key not found.")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download the kaggle.json file")
        print("4. Place it in ~/.kaggle/kaggle.json")
        print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False
    else:
        print("✓ Kaggle API key found")
        return True

def check_gpu_availability():
    """Check if GPU is available for BERT training."""
    print("Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  No GPU available. BERT training will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet. GPU check will be done after installation.")
        return False

def create_env_file():
    """Create environment configuration file."""
    print("Creating environment configuration...")
    
    env_content = """# Environment Configuration for Suicide Risk Detection
# Copy this file to .env and modify as needed

# Data Configuration
MIN_WORDS=10
BALANCE_DATASET=true
TEST_SIZE=0.2

# Model Configuration
BASELINE_MAX_FEATURES=5000
BERT_MAX_LENGTH=128
BERT_BATCH_SIZE=16
BERT_LEARNING_RATE=2e-5
BERT_EPOCHS=3

# GPU Configuration
USE_GPU=true
GPU_MEMORY_FRACTION=0.8

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=suicide_risk_detection.log
"""
    
    with open("env_template.txt", "w") as f:
        f.write(env_content)
    
    print("✓ Environment template created: env_template.txt")

def run_test():
    """Run a quick test to verify installation."""
    print("Running installation test...")
    
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        import sklearn
        import torch
        import transformers
        print("✓ All core packages imported successfully")
        
        # Test basic functionality
        from src.data_preprocessing import DataPreprocessor
        from src.baseline_model import BaselineModel
        print("✓ Project modules imported successfully")
        
        print("✓ Installation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("SUICIDE RISK DETECTION - SETUP SCRIPT")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Setup directories
    setup_directories()
    
    # Install requirements
    install_requirements()
    
    # Check GPU availability
    check_gpu_availability()
    
    # Check Kaggle setup
    kaggle_ok = check_kaggle_setup()
    
    # Create environment file
    create_env_file()
    
    # Run test
    test_passed = run_test()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    
    if test_passed:
        print("✓ Installation successful!")
        print("\nNext steps:")
        print("1. Configure Kaggle API (if not done already)")
        print("2. Run: python main.py")
        print("3. Or run tests: python test_pipeline.py")
        
        if not kaggle_ok:
            print("\n⚠️  Remember to set up Kaggle API before running the main pipeline")
    else:
        print("✗ Installation had issues. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()