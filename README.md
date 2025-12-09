# Predicting Risk Factors of Somatic Symptoms in Young Adults Using Machine Learning
**Author: Jessica Tanchone & Miao Yu**
This repository contains code and materials for a project modeling item-level PHQ-15 somatic symptoms in young adults using a range of machine learning approaches and SHAP-based interpretation.
## Project Overview
The Patient Health Questionnaire–15 (PHQ-15) is typically scored as a single total somatic symptom burden, but individual symptoms (e.g., fatigue, dizziness, chest pain, sleep problems) may be driven by different psychological, behavioral, and demographic factors. Collapsing them into one total score can hide meaningful heterogeneity.
**Our primary goal is to build and compare symptom-specific prediction models and to identify the most influential predictors for targeted symptoms.**
In this project, we:
- Treat each PHQ-15 symptom as a separate binary outcome (symptom present vs. absent).
- Use the EAMMi2 open dataset of young adults, including rich psychosocial scales and demographic variables.
- Build and compare symptom-specific prediction models.
- Use SHAP values to interpret models and visualize how key predictors relate to symptom risk.
## Data
Outcome variables: 13 PHQ-15 physical symptom items (e.g., stomach pain, back pain, fatigue, sleep problems, dizziness, chest pain).
Predictors (examples, not exhaustive):
- Perceived stress
- Subjective well-being
- Mindfulness
- Social support
- Need to belong
- Identity exploration and adulthood markers
- Narcissistic traits and interpersonal exploitativeness
- Social media use subscales
- Demographics (age, gender, etc.)
Sample: Young adults from the EAMMi2 (Emerging Adulthood Measured at Multiple Institutions) open dataset. Collaborators from 32 institutions recruited 4,220 respondents who started the survey.
## Method
### Prediction Setup
Problem type: Binary classification for each symptom (present vs. not present).
Evaluation:
- Balanced accuracy
- ROC-AUC
- F1 score
### Interpretability
Global feature importance:
- SHAP value summaries for each model/symptom to identify which predictors drive risk overall.
Local explanations:
- SHAP dependence and individual-level plots for selected predictors and symptoms.
## Model
For each symptom, we benchmark the following models:
- Logistic regression (GLMnet)
  - Penalized logistic regression (e.g., elastic net) as a transparent linear baseline.
- K-Nearest Neighbors (KNN)
  - Distance-based model to capture local patterns in the feature space.
- Tree-based gradient boosting and ensembles
  - XGBoost
  - CatBoost
  - LightGBM (Light Gradient Boosting)
- Random Forest
- ExtraTrees
- Voting Ensemble
  - Combines predictions from multiple strong base learners.
- Neural Network: TabNet-Deep
  - Deep learning approach tailored for tabular data with built-in feature selection/attention.
## Findings
- Across most symptoms, psychological and psychosocial variables (e.g., perceived stress, well-being, mindfulness, belonging, social conflict) show much stronger feature importance than demographics like age or gender.

