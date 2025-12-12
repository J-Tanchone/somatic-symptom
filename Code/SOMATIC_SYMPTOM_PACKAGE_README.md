# Somatic Symptom Prediction Project 🏥

## 📋 Project Overview

This project uses **advanced machine learning** to predict 13 different somatic (physical) symptoms based on psychological and demographic factors. It's designed to help understand which psychological predictors (stress, mindfulness, social support, etc.) influence physical symptoms.

### What Are Somatic Symptoms?
Somatic symptoms are physical symptoms (pain, fatigue, etc.) that may be influenced by psychological factors like stress, anxiety, or emotional distress. This research helps identify psychological risk factors.

---

## 🎯 What This Project Does

### Main Goals:
1. **Predict 13 Physical Symptoms** (`physSx_1` through `physSx_13`)
   - Examples: Pain, fatigue, digestive issues, headaches, etc.

2. **Use 100+ Psychological Predictors**
   - Stress levels, Social support, Mindfulness, Self-efficacy
   - Belonging, Life satisfaction, Achievement motivation
   - And many more engineered features

3. **Train 18+ Machine Learning Models**
   - XGBoost, LightGBM, CatBoost (Gradient Boosting)
   - Random Forest, Extra Trees
   - Neural Networks (MLP, TabNet)
   - Support Vector Machines, K-Nearest Neighbors
   - Naive Bayes, LDA, QDA, Logistic Regression
   - Ensemble models (Voting, Stacking)
   - **AutoGluon** (automated ML with 100+ models)

4. **Generate Comprehensive Results**
   - Best model per symptom
   - Model performance comparisons
   - 6 professional visualizations
   - SHAP feature importance analysis
   - Saved trained models for deployment

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
# Using conda (recommended)
conda create -n somatic python=3.8
conda activate somatic

# Install from requirements file
pip install -r requirements.txt

# OR install manually
conda install -c conda-forge pandas numpy scikit-learn xgboost matplotlib seaborn
pip install catboost lightgbm imbalanced-learn optuna shap pytorch-tabnet autogluon openpyxl
```

### Step 2: Prepare Your Data

Place your data file at: `somatic-symptom/EAMMi2-Data1/EAMMi2-Data1.2.xlsx`

### Step 3: Run the Analysis

```bash
# Simple run
python somatic_symptom_prediction_complete.py

# Run in background with logging
nohup python somatic_symptom_prediction_complete.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

**Expected Runtime**: 28 hours (without AutoGluon) | 54 hours (with AutoGluon)

---

## 📊 Complete Output After Training

### 1. Trained Models 📦
```
results_ultra_optimized/trained_models/
├── physSx_1/
│   ├── physSx_1_best_model.joblib       # Ready-to-use model
│   ├── physSx_1_preprocessor.joblib     # Data transformer
│   └── physSx_1_model_info.json         # Model metadata
└── best_models_summary.json
```

### 2. Performance Reports 📈
```
results_ultra_optimized/
├── all_results.csv              # All models × all symptoms
├── best_per_symptom.csv         # Best model for each symptom
└── model_averages.csv           # Average performance by model
```

### 3. Visualizations 🎨 (6 charts at 300 DPI)
```
results_ultra_optimized/visualizations/
├── model_performance_comparison.png
├── symptom_model_heatmap.png
├── best_model_distribution.png
├── performance_distribution_boxplot.png
├── best_model_ranking.png
└── metric_correlation.png
```

### 4. SHAP Analysis 🔬 (Feature Importance)
```
results_ultra_optimized/shap_analysis/
├── physSx_1_MLP_shap.png
├── physSx_1_RandomForest_shap.png
├── physSx_1_ElasticNet_shap.png
└── shap_overall_feature_importance.csv
```

---

## ⚙️ Configuration Options

Edit these variables at the top of the script (lines 30-69):

```python
OPTUNA_TRIALS = 200              # Hyperparameter optimization depth
CV_FOLDS = 10                    # Cross-validation folds
CV_REPEATS = 3                   # Repeated CV for stability
FEATURE_SELECTION_K = 'all'      # Use all features
ENABLE_AUTOGLUON = True          # Enable AutoGluon (+26 hours)
USE_GPU = True                   # Use GPU if available
```

---

## 💾 Using Saved Models

```python
import joblib
import pandas as pd

# Load saved model
model = joblib.load('results_ultra_optimized/trained_models/physSx_1/physSx_1_best_model.joblib')
preprocessor = joblib.load('results_ultra_optimized/trained_models/physSx_1/physSx_1_preprocessor.joblib')

# Make predictions on new data
new_data = pd.read_csv('new_patients.csv')
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]

print(f"Prediction: {predictions[0]}")     # 0 or 1
print(f"Probability: {probabilities[0]:.2%}")  # e.g., "73.5%"
```

---

## 📈 Performance Metrics Explained

1. **Balanced Accuracy** - Handles class imbalance
2. **ROC-AUC** (primary metrics) - Discrimination ability (0.5 to 1.0)
3. **F1-Score** - Balance of precision and recall

**Success Thresholds**:
- ✅ Good: ≥ 60%
- ✅ Very Good: ≥ 70%
- ✅ Excellent: ≥ 75%

---

## ⚠️ Important Notes

### Runtime Warning
- **28-54 hours** to complete
- Use `nohup` or `screen` to prevent interruption
- GPU recommended but not required

### Memory Requirements
- Minimum: 8GB RAM
- Recommended: 16GB+ RAM
- With AutoGluon: 32GB+ RAM

### GPU Usage
- Automatically detects and uses GPU if available
- Works on CPU but slower (especially for deep learning)

---

## 🐛 Troubleshooting

**Issue**: Package import errors
```bash
pip install <missing_package>
```

**Issue**: CUDA out of memory
```python
USE_GPU = False  # Disable GPU in script
```

**Issue**: AutoGluon too slow
```python
ENABLE_AUTOGLUON = False  # Disable AutoGluon
```

---

## 📞 Contact & Support

**Script Version**: 1.0  
**Last Updated**: January 2025  
**Python Version**: 3.8+  
**GPU Support**: CUDA 11.0+

---

**Happy Analyzing! 🎉**
