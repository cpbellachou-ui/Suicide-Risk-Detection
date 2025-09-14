# Suicide Risk Detection Using Deep Learning

## Project Overview
This project develops a suicide risk detection system that analyzes Reddit posts to identify individuals at risk. It compares traditional machine learning approaches with modern transformer-based models to understand which linguistic patterns indicate suicide risk in social media text.

## Research Question
"Can transformer-based deep learning models effectively identify suicide risk indicators in social media text, and what linguistic patterns distinguish at-risk individuals from the general population?"

## Dataset
- **Name**: Suicide and Depression Detection Dataset
- **Source**: Kaggle (nikhileswarkomati/suicide-watch)
- **Size**: ~232,000 Reddit posts
- **Type**: Binary classification (suicide risk vs. no risk)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Project Structure
```
suicide_risk_detection/
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned data
├── models/
│   ├── baseline/           # TF-IDF + LogReg
│   └── bert/              # Fine-tuned BERT
├── src/
│   ├── data_preprocessing.py
│   ├── baseline_model.py
│   ├── bert_model.py
│   ├── evaluation.py
│   └── visualization.py
├── results/
│   ├── metrics/           # Performance results
│   └── figures/          # Generated plots
└── main.py               # Main pipeline
```

## Ethical Considerations
This system aims to identify individuals at risk of suicide through their online communication patterns. Early detection could enable timely intervention and support. The model serves as a screening tool to assist mental health professionals, not replace clinical judgment. Privacy and ethical considerations are paramount - the system should only be deployed with appropriate safeguards and human oversight.