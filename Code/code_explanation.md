# Somatic Symptom Prediction Pipeline - Complete Code Explanation

## Project Overview

This project predicts 13 binary somatic symptoms (physical complaints like pain, fatigue, etc.) from psychological and demographic predictors using machine learning. The workflow spans 4 files that progress from initial exploration to production models to interpretability analysis.

**Data**: 3,182 college students × 328 survey variables → 13 symptoms × ~150 engineered features

---

## Project Workflow

```
1. Binary_Somatic_symptom_v1.ipynb (EXPLORATION)
   ↓
2. somatic_symptom_prediction_complete.py (PRODUCTION - All Models)
   ↓
3. somatic_symptom_prediction_glmnet.py (PRODUCTION - GLMNet Models)
   ↓
4. shap_value.py (INTERPRETABILITY - Feature Importance)
```

---

# FILE 1: Binary_Somatic_symptom_v1.ipynb

## Purpose
**Initial exploration notebook** to prototype the modeling pipeline in an interactive Google Colab environment. Tests 3 model types (Logistic Regression, Random Forest, Neural Network) with SHAP analysis to establish baseline performance and validate the approach before production runs.

## What It Does

### Setup (Cells 0-7)
Installs packages (shap, imblearn, tensorflow, pingouin) and clones the GitHub repository containing the dataset. Loads the main data file (EAMMi2-Data1.2.xlsx) with 328 participants and 928 raw variables.

### Feature Engineering (Cell 8)
Creates psychological scale scores by averaging survey items within each construct (stress, social support, belonging, mindfulness, self-efficacy, etc.). Generates about 15-20 mean scores representing different psychological domains. Recodes categorical variables like sibling status and parental marriage.

### Exploratory Data Analysis (Cells 14-20)
Generates histograms showing distributions of all numeric predictors. Creates correlation heatmap to identify relationships between psychological variables. Calculates prevalence ratios for each symptom to understand class imbalance (most symptoms are relatively rare, occurring in 10-30% of participants).

### Train-Test Split (Cell 25)
Creates separate 70-30 stratified splits for each of the 13 symptoms. Stratification ensures both training and test sets maintain the same proportion of symptomatic vs. non-symptomatic cases, which is crucial given the class imbalance.

### Model 1: Logistic Regression (Cells 30-35)
Builds a pipeline that standardizes numeric features, one-hot encodes categorical variables, applies SMOTE to oversample the minority class in training data, then fits a logistic regression model with L2 regularization. SMOTE helps the model learn from artificially generated examples of the underrepresented symptom-present class. Evaluates each symptom using balanced accuracy, ROC-AUC, and F1-score. Extracts feature importance from coefficient magnitudes.

### Model 2: Random Forest (Cells 37-40)
Uses GridSearchCV to search over hyperparameter combinations (number of trees, max depth, minimum samples for splitting/leaf). Finds optimal settings via 5-fold cross-validation on training data. Tests the best model on held-out test set. Applies TreeExplainer from SHAP library to understand which features drive predictions. Creates three SHAP visualizations: summary plot showing all features, bar plot of top 25 features, and force plot explaining a single prediction.

### Model 3: Neural Network (Cells 44-47)
Defines a 4-layer architecture (128→64→32→1 neurons) with dropout regularization to prevent overfitting. Implements manual 5-fold stratified cross-validation because KerasClassifier sometimes has compatibility issues. Computes class weights to handle imbalance by giving more importance to minority class during training. Uses early stopping to halt training when validation performance stops improving. Saves the best model across all folds. Applies SHAP's KernelExplainer or GradientExplainer (model-agnostic methods) since TreeExplainer doesn't work with neural networks.

### Key Takeaway
This notebook validates that ML can predict somatic symptoms from psychological factors with moderate success (60-70% ROC-AUC). It identifies SMOTE as helpful for class imbalance and confirms that tree models often outperform linear models. The interactive format allows experimentation with different preprocessing choices and rapid iteration on the modeling approach.

---

# FILES 2 & 3: somatic_symptom_prediction_complete.py + somatic_symptom_prediction_glmnet.py

## Purpose
**Production pipelines** that implement the full modeling workflow at scale. These scripts run unattended for 28-54 hours to train comprehensive model ensembles. They share the same data processing and evaluation logic but differ in which models they train.

- **complete.py**: Trains tree-based models (XGBoost, LightGBM, CatBoost, Random Forest, ExtraTrees)
- **glmnet.py**: Trains elastic net logistic regression variants (L1/LASSO, L2/Ridge, ElasticNet) separately due to runtime concerns

## Common Structure

### Configuration (Lines 1-80)
Sets global parameters controlling the entire pipeline:
- **Optuna trials**: 200 Bayesian optimization iterations per model to find optimal hyperparameters
- **Cross-validation**: 10-fold stratified CV repeated 3 times (30 total evaluations per model) for robust performance estimates
- **GPU acceleration**: Enables GPU training for XGBoost, LightGBM, CatBoost when available
- **Optimization metric**: Can target accuracy, balanced accuracy, F1, or ROC-AUC
- **AutoGluon**: Optional automated ensemble learning (disabled in this project due to runtime)

These settings prioritize maximum accuracy over speed, using aggressive optimization strategies.

### Data Loading & Preprocessing (Lines 120-230)
Clones GitHub repository if not present. Loads Excel data with 328 rows × 928 columns. Creates 15-20 basic scale scores by averaging items within psychological constructs (same as notebook). Adds variation statistics (std, max, min) for each scale to capture response patterns beyond means.

### Advanced Feature Engineering (Lines 230-385)
Goes far beyond the notebook's basic features by creating 100+ engineered predictors:

**Level 1 - Interaction Terms**: Multiplies related psychological variables (e.g., stress × social support deficit) to capture synergistic effects where high stress combined with low support has disproportionate impact.

**Level 2 - Composite Indices**: Combines multiple predictors into weighted aggregates representing higher-order constructs like "psychological distress" (weighted combination of stress, low support, low belonging, low efficacy).

**Level 3 - Domain Expert Features**: Creates theory-driven variables based on somatization literature, such as alexithymia proxy (emotional awareness deficit) and catastrophizing tendency (stress squared divided by self-efficacy).

**Level 4 - Achievement-Stress Interactions**: Captures perfectionism-related stress by calculating gaps between achievement importance and actual achievement.

**Level 5 - Non-linear Transformations**: Adds squared, square root, and log transformations of key variables to allow models to capture non-linear relationships.

**Level 6 - Statistical Aggregations**: Computes overall mean, std, skewness, and kurtosis across all psychological scales to characterize the participant's general response style.

**Level 7 - Somatization Proneness Index**: Multi-component weighted combination specifically designed to predict physical symptom risk.

**Level 8 - Demographic Interactions**: Combines demographics with psychological variables (e.g., sex × stress) since certain groups may be differentially affected.

**Level 9 - Resilience vs Vulnerability**: Creates ratio and difference scores between protective factors and risk factors.

**Level 10 - Cross-domain Interactions**: Combines variables from different psychological domains that theory suggests interact.

This results in approximately 150 total features per participant.

### Train-Test Splitting (Lines 390-450)
Creates 70-30 stratified splits separately for each symptom. Identifies numeric vs. categorical features automatically. Builds ColumnTransformer preprocessing pipeline that applies StandardScaler to numeric features and OneHotEncoder to categorical features. This standardizes the preprocessing across all models.

### Model Training Loop - Structure (Lines 450-1200)
For each of 13 symptoms:
1. Computes class weights to account for imbalance
2. Creates SMOTE sampler to oversample minority class
3. Trains multiple model types with different configurations
4. Records performance metrics (balanced accuracy, ROC-AUC, F1) for all models
5. Identifies best model for this symptom
6. Saves trained model and preprocessor to disk

---

## Model Differences Between complete.py and glmnet.py

### somatic_symptom_prediction_complete.py - Models Trained

**XGBoost with Bayesian Optimization**
Uses Optuna to search 100 trials across hyperparameters (n_estimators, max_depth, learning_rate, regularization). Gradient boosting that builds trees sequentially to correct previous errors. GPU-accelerated when available. Scale_pos_weight automatically balances classes.

**LightGBM**
Fast gradient boosting implementation with different tree-building strategy (leaf-wise vs. level-wise). Configured with 500 trees, learning rate 0.05, and GPU support. Handles imbalance via scale_pos_weight parameter.

**CatBoost**
Gradient boosting optimized for categorical features and small datasets. Uses ordered boosting to reduce overfitting. Automatically computes class weights via auto_class_weights='Balanced'. GPU-accelerated with task_type='GPU'.

**Random Forest**
Ensemble of 500 decision trees built independently on random feature subsets. Each tree votes on final prediction. Uses max_depth=15, class_weight='balanced', and various min_samples parameters to control tree complexity.

**Extra Trees**
Similar to Random Forest but uses random thresholds instead of optimal thresholds when splitting nodes. Often more robust to noise. Same configuration as Random Forest.

### somatic_symptom_prediction_glmnet.py - Models Trained

**Logistic Regression with L1 (LASSO)**
Linear model with L1 regularization that drives some coefficients exactly to zero, performing automatic feature selection. Uses solver='saga' which supports L1 penalty. Regularization strength C=0.1 chosen to be moderately aggressive.

**Logistic Regression with L2 (Ridge)**
Linear model with L2 regularization that shrinks all coefficients toward zero but keeps all features. More stable than L1 when features are correlated. Uses default 'lbfgs' solver optimized for L2.

**Logistic Regression with ElasticNet**
Combines L1 and L2 penalties (l1_ratio=0.5 means 50-50 mix). Balances feature selection from L1 with stability from L2. Requires 'saga' solver and increased max_iter=2000 for convergence.

All three use class_weight='balanced' to handle imbalance and are paired with SMOTE in pipeline.

**Why Separate File?**
GLMNet models have simpler hyperparameter spaces and train faster than tree models. Running them separately allows:
1. Parallel execution on different machines to save time
2. Easier debugging if one set of models fails
3. Modular results that can be combined later
4. Different optimization strategies for linear vs. tree models

---

## Common Model Training Process

Each model follows this pattern:

1. **Wrap in Pipeline**: Combine preprocessor → SMOTE → model into single pipeline that handles all transformations
2. **Optuna Optimization** (for complex models): Define objective function that trains model with given hyperparameters and returns cross-validation score. Run TPE sampler to intelligently search hyperparameter space.
3. **Train Final Model**: Fit best configuration on full training set
4. **Predict on Test Set**: Generate predictions and probability scores
5. **Compute Metrics**: Calculate balanced accuracy (average of sensitivity and specificity), ROC-AUC (discrimination ability), F1 (harmonic mean of precision and recall)
6. **Store Results**: Append metrics to results list and save trained pipeline to disk

### Results Analysis (Lines 1200-1400)
Combines all model results into single DataFrame. Creates symptom name mapping for readable output. Identifies best model per symptom based on F1-score. Computes average performance by model type. Reports success rates (percentage of models achieving ROC-AUC ≥ 60%, 70%, 75%).

### Model Persistence (Lines 1400-1500)
For each symptom, saves the best-performing model as .joblib file along with its preprocessor. Stores model metadata (symptom name, model type, performance metrics, hyperparameters) as JSON. Organizes files in directory structure: results_ultra_optimized/trained_models/physSx_N/.

### Visualization Generation (Lines 1500-1700)
Creates 6 publication-quality plots at 300 DPI:

**Model Performance Comparison**: Grouped bar chart showing balanced accuracy for all model × symptom combinations, allowing comparison across models and symptoms simultaneously.

**Symptom-Model Heatmap**: Color-coded grid where rows are symptoms, columns are models, and cell colors represent ROC-AUC values. Quickly identifies which models work best for which symptoms.

**Best Model Distribution**: Pie chart showing frequency of each model type being selected as best across all 13 symptoms.

**Performance Distribution Boxplot**: Shows distribution of scores across all predictions for each model type, revealing consistency and outliers.

**Best Model Ranking**: Horizontal bar chart ranking models by average F1-score with error bars.

**Metric Correlation**: Scatter plots examining relationships between balanced accuracy, ROC-AUC, and F1-score to verify metric alignment.

---

# FILE 4: shap_value.py

## Purpose
**Feature importance analysis** that explains which psychological predictors drive somatic symptom predictions. Runs separately after models are trained because SHAP computation is computationally expensive (8-12 hours for all symptoms). Loads saved models from complete.py and glmnet.py, computes SHAP values, and generates interpretable visualizations.

## What It Does

### Configuration & Data Loading (Lines 1-230)
Reuses the same configuration parameters and data preprocessing code as the production scripts to ensure consistency. Must recreate the exact feature engineering pipeline so features match those seen during training.

### Model Loading (Lines 240-400)
Scans results CSV files from both complete.py and glmnet.py to identify the best model for each symptom. Loads trained model pipelines (.joblib files) from disk. Extracts the classifier from the pipeline while keeping track of preprocessing transformations.

### Feature Name Mapping (Lines 400-550)
Creates mapping from engineered feature names (like "stress_support_interaction_encoded_5") back to interpretable predictor names (like "Stress × Social Support"). This is crucial because preprocessing generates hundreds of feature columns via one-hot encoding and transformations, but researchers want to understand importance at the predictor level.

Groups all derived features by their root predictor:
- All polynomial terms, interactions, and encodings of "stress_m" map to "Perceived Stress"
- All encodings of categorical variable "sex" map to "Sex"
- Interaction terms map to both constituent predictors

### SHAP Computation Function (Lines 550-700)
Implements flexible SHAP analysis that handles different model types:

**For Tree Models** (XGBoost, LightGBM, CatBoost, Random Forest):
Uses TreeExplainer, which is fast and exact for tree-based models. Computes SHAP values directly from tree structure without needing background data.

**For Linear Models** (GLMNet variants):
Uses LinearExplainer or KernelExplainer depending on model format. Linear models allow analytic SHAP computation.

**For Black-Box Models** (Neural Networks, complex ensembles):
Uses KernelExplainer, a model-agnostic method that approximates SHAP values by training a local linear model. Samples 100 background observations and explains 300 test instances to balance accuracy and speed.

### SHAP Aggregation (Lines 700-800)
Computes mean absolute SHAP value for each feature across all explained predictions. This represents average importance magnitude. Groups feature-level SHAP values by predictor (e.g., sums SHAP values across all one-hot encoded categories of "sex" to get total importance of sex predictor). Ranks predictors by total importance.

### Visualization Generation (Lines 800-900)
For each symptom, creates a horizontal bar plot showing top 15 predictors ranked by mean absolute SHAP value. Uses color gradient (darker = more important) and adds value labels on bars. Saves as high-resolution PNG.

Returns DataFrame with three columns:
- feature_level: Importance of individual engineered features
- predictor_level: Importance aggregated to original predictor level
- mapping: How features map to predictors

### Results Compilation (Lines 900-926)
Combines SHAP results across all symptoms into master CSV file showing predictor importance for each symptom. Enables cross-symptom comparison to identify universal risk factors vs. symptom-specific predictors. Saves individual importance CSVs for each symptom.

---

## Key Insights from SHAP Analysis

**Universal Predictors**: Stress, social support, and mindfulness consistently appear in top features across most symptoms, suggesting they're general risk factors for somatization.

**Symptom-Specific Patterns**: Certain predictors uniquely predict specific symptoms (e.g., achievement motivation for headaches, disability perception for pain symptoms).

**Interaction Effects**: Engineered interaction features (stress × support deficit, efficacy × belonging) often rank highly, confirming that psychological factors work synergistically.

**Demographic Moderation**: Sex and demographic interactions appear important for some symptoms, indicating certain groups are differentially vulnerable.

---

## Technical Implementation Details

### Class Imbalance Handling
All pipelines use three complementary strategies:
1. **SMOTE oversampling** in training pipeline to generate synthetic minority examples
2. **Class weights** in model training to penalize misclassifications asymmetrically  
3. **Balanced accuracy** as evaluation metric to average sensitivity and specificity

### Cross-Validation Strategy
RepeatedStratifiedKFold with 10 folds × 3 repeats provides 30 independent performance estimates. Stratification maintains class proportions in each fold. Repetition with different random splits reduces variance in estimates.

### Hyperparameter Optimization
Optuna's TPE (Tree-structured Parzen Estimator) sampler is more efficient than grid search. It models the relationship between hyperparameters and performance, focusing trials on promising regions. Each trial trains a model with 5-fold CV to estimate generalization.

### Model Persistence
All models saved with joblib for efficient serialization of scikit-learn objects. Stores entire pipeline including preprocessor so new data can be transformed identically. JSON metadata allows models to be cataloged and selected programmatically.

### GPU Acceleration
XGBoost, LightGBM, and CatBoost automatically use GPU when available (detected via USE_GPU=True). Typically provides 5-10x speedup for large datasets. Falls back to CPU if GPU unavailable.

---

## Workflow Summary

1. **Explore** in notebook (2-4 hours) → Validate feasibility, test 3 models, understand data
2. **Train** production models (20-30 hours) → complete.py trains tree models, glmnet.py trains linear models
3. **Interpret** with SHAP (8-12 hours) → Identify which predictors matter for each symptom
4. **Combine** results → Best model per symptom, averaged model performance, feature importance rankings

This modular approach allows parallel execution, easier debugging, and separation of concerns (modeling vs. interpretation).

---

## File Outputs Summary

**Binary_Somatic_symptom_v1.ipynb**
- Performance metrics tables (printed)
- Feature importance plots for each model type
- SHAP visualizations (bar, beeswarm, force plots)

**somatic_symptom_prediction_complete.py**
- all_results.csv: Every model × symptom combination
- best_per_symptom.csv: Top model for each symptom
- model_averages.csv: Average metrics by model type
- trained_models/physSx_N/*.joblib: Saved models
- visualizations/*.png: 6 comparison plots

**somatic_symptom_prediction_glmnet.py**
- Same output structure as complete.py
- Results combine with complete.py in final analysis

**shap_value.py**
- shap_analysis/physSx_N_top15_predictors.png: Bar plots
- shap_analysis/shap_importance_all_symptoms.csv: Combined results
- shap_analysis/physSx_N_feature_importance.csv: Per-symptom details

---

## Performance Expectations

**Baseline** (Logistic Regression from notebook): 55-65% ROC-AUC  
**Production** (Optimized tree models + GLMNet): 65-75% ROC-AUC  

Balanced accuracy ranges 55-75%, F1-score ranges 50-70%. Performance varies by symptom based on prevalence and predictor strength. These metrics are reasonable for psychological → physical symptom prediction, where many unmeasured factors influence outcomes.

---

## Total Computational Resources

**Development**: 0.5-1 hours (notebook exploration)  
**Training**: 8-10 hours (complete.py + glmnet.py, can run parallel)  
**Analysis**: 0.5 hours (SHAP computation)  