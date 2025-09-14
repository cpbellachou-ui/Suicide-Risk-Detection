# Quick Start Guide - Suicide Risk Detection

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Setup
```bash
python3 setup.py
```

### 3. Test the Pipeline
```bash
python3 test_pipeline.py
```

### 4. Run Full Pipeline (with real data)
```bash
# First, set up Kaggle API:
# 1. Go to https://www.kaggle.com/account
# 2. Click 'Create New API Token'
# 3. Download kaggle.json
# 4. Place in ~/.kaggle/kaggle.json
# 5. Set permissions: chmod 600 ~/.kaggle/kaggle.json

python3 main.py
```

## 📊 What You'll Get

### Generated Files
- **Models**: Trained baseline and BERT models in `models/`
- **Results**: Performance metrics in `results/metrics/`
- **Visualizations**: 6+ plots in `results/figures/`
- **Report**: Comprehensive analysis in `results/final_report.txt`

### Key Visualizations
1. **Performance Comparison**: Bar chart showing BERT vs Baseline
2. **Confusion Matrices**: Side-by-side comparison
3. **ROC Curves**: Model performance curves
4. **Feature Importance**: Top words for each class
5. **Word Clouds**: Visual word frequency analysis
6. **Linguistic Patterns**: Risk vs protective factors

## 🎯 Expected Results

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| Baseline | 75-80% | 74-79% | 80-85% |
| BERT | 85-92% | 85-92% | 90-95% |

**Improvement**: BERT shows 10-15% better performance across all metrics.

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the project directory
cd suicide_risk_detection
python3 test_pipeline.py
```

**2. Kaggle API Issues**
```bash
# Check if kaggle.json exists
ls ~/.kaggle/kaggle.json

# Set correct permissions
chmod 600 ~/.kaggle/kaggle.json
```

**3. Memory Issues (BERT)**
```bash
# Reduce batch size in config.json
"bert_batch_size": 8  # instead of 16
```

**4. CUDA/GPU Issues**
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python3 main.py
```

## 📁 Project Structure
```
suicide_risk_detection/
├── main.py              # 🚀 Start here
├── test_pipeline.py     # 🧪 Test with sample data
├── setup.py            # ⚙️ Setup environment
├── src/                # 📦 Source code
├── results/            # 📊 Generated results
└── README.md           # 📖 Full documentation
```

## 🎯 Next Steps

1. **Review Results**: Check `results/figures/` for visualizations
2. **Read Report**: Open `results/final_report.txt`
3. **Explore Code**: Look at `src/` modules
4. **Customize**: Modify `config.json` for different settings
5. **Extend**: Add new features or models

## ⚠️ Important Notes

- **Ethical Use**: This is a screening tool, not a diagnostic tool
- **Human Oversight**: Always require human review of predictions
- **Privacy**: Ensure data protection and confidentiality
- **Bias**: Be aware of potential biases in the model

## 📞 Need Help?

1. Check the full documentation in `README.md`
2. Review mathematical details in `MATHEMATICAL_DOCUMENTATION.md`
3. Run tests to verify everything works
4. Examine the generated visualizations and reports

---

**Ready to start? Run `python3 test_pipeline.py` to begin!** 🚀