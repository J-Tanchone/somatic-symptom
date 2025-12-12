# **Executive Summary**

## **Project Overview**

This project applies interpretable machine learning to understand how psychological factors predict **13 individual somatic symptoms** from the PHQ-15. Rather than collapsing symptoms into a single total score, we modeled each symptom separately using data from over 4,000 college students. Our goal was to identify which psychosocial variables—such as stress, mindfulness, social support, self-efficacy, and well-being—best predict specific bodily symptoms, and to evaluate how predictable each symptom is across multiple algorithms. This approach provides a more symptom-specific, clinically meaningful picture of somatic presentations than traditional summary-score analyses.

## **Overall Scope of Work Completed**

* Cleaned, prepared, and engineered **166 psychological and demographic features** for model training.
* Trained and compared **nine machine-learning algorithms** (e.g., logistic regression, glmnet, random forest, gradient boosting, CatBoost).
* Modeled each of the **13 PHQ-15 symptoms independently**, generating performance metrics (F1, ROC–AUC, Balanced Accuracy) for all models.
* Identified best-performing models for each symptom.
* Applied **SHAP value analyses** to determine which psychosocial factors drive each symptom’s predictions.
* Produced a complete visualization suite, including performance comparisons and SHAP feature-importance plots.
* Summarized symptom-by-symptom insights, highlighting patterns such as:

  * **Fatigue, sleep problems, and dizziness** are the most predictable from psychosocial variables.
  * **Chest pain, fainting spells, and shortness of breath** show weak predictability, suggesting stronger medical or physiological determinants.

---

# **Individual Contributions**

## **Jessica Tanchone**

* Led the **modeling and visualization pipeline**, including comparing nine algorithms for each symptom.
* Built the cross-symptom evaluation framework, enabling direct comparison of ROC–AUC and F1 scores across symptoms.
* Designed and ran the full **SHAP workflow**, generating interpretability outputs for each best-performing model.
* Engineered a multi-level feature set (scale means, variability metrics, interactions, composite indices).
* Produced the main visualizations used for reporting, including SHAP plots and performance graphs.
* Synthesized theoretical insights connecting psychological constructs (e.g., stress, mindfulness, belonging) to somatic outcomes.
* Documented modeling code across `somatic_symptom_prediction_complete.py`, `somatic_symptom_prediction_GLMnet.py`, and `shap_value.py`.

## **Miao Yu**

* Conducted exploratory data analysis and initial feature engineering.
* Contributed to neural network model development and parameter tuning.
* Assisted with scaling, normalization, and dataset preparation.
* Helped validate model outputs and conduct error checking.

---

# **Key Findings**

* Somatic symptoms vary dramatically in their **predictability** from psychological constructs.
* Well-being, stress, and mindfulness emerged as **consistent, high-impact predictors** across multiple symptoms.
* The divergence in predictability across symptoms suggests that psychosocial interventions may target some symptoms more effectively than others.
* The interpretability framework allowed mapping specific psychological risk/protective factors onto individual symptom expressions, a step forward from global somatic scores.

---

# **Recommended Future Directions**

## **1. Expand the Predictor Space**

* Incorporate **behavioral data, EMA, social activity metrics, sleep logs**, or physiological markers if available.
* Add non-linear interactions using domain-informed feature engineering.

## **2. Improve Modeling of Low-Base-Rate Symptoms**

* Apply techniques suited for extreme imbalance (e.g., focal loss, SMOTE variants).
* Consider hierarchical or multi-task models that borrow strength across symptoms.

## **3. Clinical & Theoretical Integration**

* Use symptom-specific findings to form **hypothesis-driven psychological models** of somatic processes (e.g., stress-recovery cycles).
* Map SHAP-derived features to clinical constructs like emotion regulation, perceived control, or health anxiety.

## **4. Develop a Reproducible Package**

* Turn the full pipeline into a **reusable module** for future students or collaborators, enabling plug-and-play symptom prediction.

## **5. Longitudinal & Cross-Dataset Validation**

* Validate models on new populations or cultural groups.
* Evaluate whether predictors change across developmental stages or stress contexts.
