# Repeated-nested-cv-depression-screening
Python codebase for depression screening and vulnerability classification. Evaluates seven classifiers using repeated nested cross-validation with Lasso feature selection, SMOTE and hybrid SMOTETomek resampling, and threshold optimization strategies.

Overview: The codebase evaluates single-modality eye-movement baselines alongside multi-modal (eye-face) integration frameworks.

The pipeline implements:

Repeated Nested Cross-Validation: A rigorous nested cross-validation framework to prevent data leakage and ensure unbiased generalization performance.

Embedded Feature Selection: Lasso-based feature selection executed strictly within inner cross-validation training folds.

Class Imbalance Handling: Comparison of standard oversampling (SMOTE) and hybrid resampling (SMOTETomek combining SMOTE oversampling with Tomek Link noise reduction).

Decision Threshold Optimization: Post-hoc classification threshold tuning via Youden's Index versus the default (0.50) decision threshold.

Classifier Benchmarking: Comparative assessment across 7 baseline machine learning classifiers:CatBoost, Random Forest, AdaBoost, XGBoost, Support Vector Machine, and Logistic Regression.
