from collections import Counter
import os
import warnings
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


raw_data = 'data_depr.xlsx'
df = pd.read_excel(raw_data)

X = df.drop(columns=['ID', 'Class']).values
y = df['Class'].values
feature_names = np.array(df.drop(columns=['ID', 'Class']).columns)

base_seed = 42
base_output_dir = 'results_RF_RNV'
os.makedirs(base_output_dir, exist_ok=True)



# Fixed Nested Cross-Validation Design

OUTER_N_SPLITS = 5
INNER_N_SPLITS = 3
N_OUTER_REPEATS = 10

SMOTE_K_NEIGHBORS = 5


def get_resampler(method_name, seed):
  if method_name is None:
    return None

  if method_name == 'SMOTE':
    return SMOTE(random_state=seed, k_neighbors=SMOTE_K_NEIGHBORS)

  elif method_name == 'SMOTETomek':
    return SMOTETomek(
        random_state=seed,
        smote=SMOTE(random_state=seed, k_neighbors=SMOTE_K_NEIGHBORS),
    )
  else:
    raise ValueError(f'Unknown resampling method: {method_name}')


def resampler_label(method_name):
  return 'NoResampling' if method_name is None else method_name


def get_metrics(y_true, y_pred, y_proba):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
  acc = accuracy_score(y_true, y_pred)
  sens = recall_score(y_true, y_pred, zero_division=0)
  spec = tn / (tn + fp) if (tn + fp) > 0 else 0
  prec = precision_score(y_true, y_pred, zero_division=0)
  f1 = f1_score(y_true, y_pred, zero_division=0)
  youden_j = sens + spec - 1
  try:
    auc = roc_auc_score(y_true, y_proba)
  except Exception:
    auc = np.nan
  return acc, sens, spec, prec, f1, auc, youden_j, (tn, fp, fn, tp)


def save_summary_metrics(
    fold_results, file_path, seed, clf_name, resample_name, method_name,
    n_splits,
):
  results_df = pd.DataFrame(fold_results)
  metric_cols = [
      'Train_Accuracy', 'Train_Sensitivity', 'Train_Specificity',
      'Train_Precision', 'Train_F1', 'Train_AUC', 'Train_Youden_J',
      'Test_Accuracy', 'Test_Sensitivity', 'Test_Specificity',
      'Test_Precision', 'Test_F1', 'Test_AUC', 'Test_Youden_J',
  ]

  results_df['Repeat'] = (results_df['Fold'] - 1) // n_splits
  repeat_means = results_df.groupby('Repeat')[metric_cols].mean()

  summary_rows = []
  np.random.seed(seed)

  for col in metric_cols:
    raw_values = results_df[col].dropna().values
    repeat_values = repeat_means[col].dropna().values

    mean_val = np.mean(raw_values)
    std_val = np.std(raw_values)

    n_boot_units = len(repeat_values)
    if n_boot_units >= 2:
      boot_means = [
          np.mean(np.random.choice(repeat_values, size=n_boot_units, replace=True))
          for _ in range(1000)
      ]
      ci_lower = np.percentile(boot_means, 2.5)
      ci_upper = np.percentile(boot_means, 97.5)
    else:
      ci_lower, ci_upper = np.nan, np.nan

    summary_rows.append({
        'Classifier': clf_name,
        'Resampling_Method': resample_name,
        'Threshold_Method': method_name,
        'Metric': col,
        'Mean': mean_val,
        'Std_Dev': std_val,
        'Mean_±_SD': f'{mean_val:.4f} ± {std_val:.4f}',
        'N_Repeat_Level_Units': n_boot_units,
        '95%_CI_Lower': ci_lower,
        '95%_CI_Upper': ci_upper,
        '95%_CI_String': f'[{ci_lower:.4f}, {ci_upper:.4f}]',
    })

  summary_df = pd.DataFrame(summary_rows)
  summary_df.to_excel(file_path, index=False)
  return summary_df


def save_confusion_matrix_plot(cm_array, title, output_path):
  cm_pct = (
      cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
  ) * 100
  fig, ax = plt.subplots(figsize=(5, 4))
  sns.heatmap(cm_pct, annot=True, fmt='.2f', cmap='Greens', ax=ax, cbar=True)
  for t in ax.texts:
    t.set_text(f'{t.get_text()}%')
  ax.set_title(title)
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Actual')
  plt.tight_layout()
  plt.savefig(output_path)
  plt.close()



# Strict Nested Inner Objective

def create_strict_optuna_objective(
    X_outer_train,
    y_outer_train,
    resample_name,
    seed,
    inner_n_splits,
    threshold_search=False,
):

  inner_cv = StratifiedKFold(
      n_splits=inner_n_splits, shuffle=True, random_state=seed
  )

  def objective(trial):
    params = {
        'n_estimators': 500,  # fixed (not tuned), matches original design
        'criterion': trial.suggest_categorical(
            'criterion', ['gini', 'entropy', 'log_loss']
        ),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical(
            'max_features', ['sqrt', 'log2', None]
        ),
        'class_weight': trial.suggest_categorical(
            'class_weight', ['balanced', 'balanced_subsample', None]
        ),
        'random_state': 8,
        'n_jobs': -1,
    }

    thr = trial.suggest_float('threshold', 0.1, 0.9) if threshold_search else 0.5
    fold_scores = []

    for in_tr_idx, in_val_idx in inner_cv.split(X_outer_train, y_outer_train):
      X_in_tr, X_in_val = X_outer_train[in_tr_idx], X_outer_train[in_val_idx]
      y_in_tr, y_in_val = y_outer_train[in_tr_idx], y_outer_train[in_val_idx]

      if len(np.unique(y_in_tr)) < 2:
        continue

      imputer = SimpleImputer(strategy='median')
      X_in_tr_imp = imputer.fit_transform(X_in_tr)
      X_in_val_imp = imputer.transform(X_in_val)

      scaler = StandardScaler()
      X_in_tr_scaled = scaler.fit_transform(X_in_tr_imp)
      X_in_val_scaled = scaler.transform(X_in_val_imp)

      lasso = LogisticRegression(
          penalty='l1',
          solver='saga',
          class_weight='balanced',
          max_iter=2000,
          random_state=seed,
      )
      lasso.fit(X_in_tr_scaled, y_in_tr)
      sel_idx = [i for i, coef in enumerate(lasso.coef_[0]) if coef != 0]
      if len(sel_idx) == 0:
        sel_idx = list(np.argsort(np.abs(lasso.coef_[0]))[-5:])

      X_in_tr_sel = X_in_tr_scaled[:, sel_idx]
      X_in_val_sel = X_in_val_scaled[:, sel_idx]


      resampler = get_resampler(resample_name, seed)
      if resampler is not None:
        X_in_tr_res, y_in_tr_res = resampler.fit_resample(X_in_tr_sel, y_in_tr)
      else:
        X_in_tr_res, y_in_tr_res = X_in_tr_sel, y_in_tr

      model = RandomForestClassifier(**params)
      model.fit(X_in_tr_res, y_in_tr_res)
      y_val_proba = model.predict_proba(X_in_val_sel)[:, 1]

      y_val_pred = (y_val_proba >= thr).astype(int)
      score = balanced_accuracy_score(y_in_val, y_val_pred)

      fold_scores.append(score)

    if len(fold_scores) == 0:
      return 0.0
    return float(np.mean(fold_scores))

  return objective


# Main Benchmark Loop

resampling_methods = [
    None,
    'SMOTE',
    'SMOTETomek',
]
target_metric = 'balanced_accuracy'
outer_cv = RepeatedStratifiedKFold(
    n_splits=OUTER_N_SPLITS, n_repeats=N_OUTER_REPEATS, random_state=base_seed
)
master_summary_list = []

for resample_method in resampling_methods:

  print(
      f'RUNNING STRICT LEAK-FREE RANDOM FOREST + OPTUNA | RESAMPLER:'
      f' {resampler_label(resample_method)} | TARGET: {target_metric.upper()} | '
      f'outer_n_splits={OUTER_N_SPLITS}, inner_n_splits={INNER_N_SPLITS}'
  )


  target_dir = os.path.join(
      base_output_dir,
      resampler_label(resample_method),
      f'Optimized_{target_metric.capitalize()}',
  )
  dir_m05 = os.path.join(target_dir, 'Method_Default_Threshold_0.5')
  dir_youden = os.path.join(
      target_dir, 'Nested_Threshold_Method_YoudenJ'
  )
  os.makedirs(dir_m05, exist_ok=True)
  os.makedirs(dir_youden, exist_ok=True)

  results_05 = []
  results_youden = []
  selected_features_acc = []
  cm_05_agg = np.zeros((2, 2), dtype=float)
  cm_youden_agg = np.zeros((2, 2), dtype=float)

  for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    current_seed = base_seed + fold_idx

    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]


    # Fit Preprocessing Steps STRICTLY on Outer Train Fold

    imputer_out = SimpleImputer(strategy='median')
    X_tr_imp = imputer_out.fit_transform(X_train_outer)
    X_te_imp = imputer_out.transform(X_test_outer)

    scaler_out = StandardScaler()
    X_tr_scaled = scaler_out.fit_transform(X_tr_imp)
    X_te_scaled = scaler_out.transform(X_te_imp)

    lasso_out = LogisticRegression(
        penalty='l1',
        solver='saga',
        class_weight='balanced',
        max_iter=2000,
        random_state=current_seed,
    )
    lasso_out.fit(X_tr_scaled, y_train_outer)
    selected_indices = [
        i for i, coef in enumerate(lasso_out.coef_[0]) if coef != 0
    ]
    if len(selected_indices) == 0:
      selected_indices = list(np.argsort(np.abs(lasso_out.coef_[0]))[-5:])

    X_tr_selected = X_tr_scaled[:, selected_indices]
    X_te_selected = X_te_scaled[:, selected_indices]

    selected_feat_names = feature_names[selected_indices]
    selected_weights = lasso_out.coef_[0][selected_indices]
    selected_features_acc.extend(selected_feat_names)
    feat_weight_str = '; '.join([
        f'{name} ({weight:.4f})'
        for name, weight in zip(selected_feat_names, selected_weights)
    ])


    resampler_out = get_resampler(resample_method, current_seed)
    if resampler_out is not None:
      X_tr_res_final, y_tr_res_final = resampler_out.fit_resample(
          X_tr_selected, y_train_outer
      )
    else:
      X_tr_res_final, y_tr_res_final = X_tr_selected, y_train_outer


    # a) Default Threshold = 0.5

    study_05 = optuna.create_study(direction='maximize')
    opt_obj_05 = create_strict_optuna_objective(
        X_train_outer,
        y_train_outer,
        resample_method,
        current_seed,
        INNER_N_SPLITS,
        threshold_search=False,
    )
    study_05.optimize(opt_obj_05, n_trials=30)
    best_params_05 = study_05.best_params.copy()
    best_params_05.update(
        {'n_estimators': 500, 'random_state': 8, 'n_jobs': -1}
    )

    best_rf_05 = RandomForestClassifier(**best_params_05)
    best_rf_05.fit(X_tr_res_final, y_tr_res_final)

    y_tr_proba_05 = best_rf_05.predict_proba(X_tr_selected)[:, 1]
    y_te_proba_05 = best_rf_05.predict_proba(X_te_selected)[:, 1]

    tr_metrics_05 = get_metrics(
        y_train_outer, (y_tr_proba_05 >= 0.5).astype(int), y_tr_proba_05
    )
    te_metrics_05 = get_metrics(
        y_test_outer, (y_te_proba_05 >= 0.5).astype(int), y_te_proba_05
    )

    cm_05_agg += np.array([
        [te_metrics_05[-1][0], te_metrics_05[-1][1]],
        [te_metrics_05[-1][2], te_metrics_05[-1][3]],
    ])

    results_05.append({
        'Fold': fold_idx + 1,
        'Resampling_Method': resampler_label(resample_method),
        'Optimization_Target': target_metric,
        'Random_Seed_Used': current_seed,
        'Selected_Features_and_Weights': feat_weight_str,
        'Num_Features': len(selected_indices),
        'Applied_Threshold': 0.5,
        **{
            f'Best_{k}': v
            for k, v in best_params_05.items()
            if k not in ['random_state', 'n_jobs']
        },
        'Train_Accuracy': tr_metrics_05[0],
        'Train_Sensitivity': tr_metrics_05[1],
        'Train_Specificity': tr_metrics_05[2],
        'Train_Precision': tr_metrics_05[3],
        'Train_F1': tr_metrics_05[4],
        'Train_AUC': tr_metrics_05[5],
        'Train_Youden_J': tr_metrics_05[6],
        'Test_Accuracy': te_metrics_05[0],
        'Test_Sensitivity': te_metrics_05[1],
        'Test_Specificity': te_metrics_05[2],
        'Test_Precision': te_metrics_05[3],
        'Test_F1': te_metrics_05[4],
        'Test_AUC': te_metrics_05[5],
        'Test_Youden_J': te_metrics_05[6],
    })


    # b) Nested Threshold Tuning- Youden J

    study_yj = optuna.create_study(direction='maximize')
    opt_obj_yj = create_strict_optuna_objective(
        X_train_outer,
        y_train_outer,
        resample_method,
        current_seed,
        INNER_N_SPLITS,
        threshold_search=True,
    )
    study_yj.optimize(opt_obj_yj, n_trials=30)
    best_params_yj = study_yj.best_params.copy()
    best_threshold = best_params_yj.pop('threshold')
    best_params_yj.update(
        {'n_estimators': 500, 'random_state': 8, 'n_jobs': -1}
    )

    best_rf_yj = RandomForestClassifier(**best_params_yj)
    best_rf_yj.fit(X_tr_res_final, y_tr_res_final)

    y_tr_proba_yj = best_rf_yj.predict_proba(X_tr_selected)[:, 1]
    y_te_proba_yj = best_rf_yj.predict_proba(X_te_selected)[:, 1]

    tr_metrics_yj = get_metrics(
        y_train_outer,
        (y_tr_proba_yj >= best_threshold).astype(int),
        y_tr_proba_yj,
    )
    te_metrics_yj = get_metrics(
        y_test_outer,
        (y_te_proba_yj >= best_threshold).astype(int),
        y_te_proba_yj,
    )

    cm_youden_agg += np.array([
        [te_metrics_yj[-1][0], te_metrics_yj[-1][1]],
        [te_metrics_yj[-1][2], te_metrics_yj[-1][3]],
    ])

    results_youden.append({
        'Fold': fold_idx + 1,
        'Resampling_Method': resampler_label(resample_method),
        'Optimization_Target': target_metric,
        'Random_Seed_Used': current_seed,
        'Selected_Features_and_Weights': feat_weight_str,
        'Num_Features': len(selected_indices),
        'Applied_Threshold': best_threshold,
        'Optimal_Threshold': best_threshold,
        **{
            f'Best_{k}': v
            for k, v in best_params_yj.items()
            if k not in ['random_state', 'n_jobs']
        },
        'Train_Accuracy': tr_metrics_yj[0],
        'Train_Sensitivity': tr_metrics_yj[1],
        'Train_Specificity': tr_metrics_yj[2],
        'Train_Precision': tr_metrics_yj[3],
        'Train_F1': tr_metrics_yj[4],
        'Train_AUC': tr_metrics_yj[5],
        'Train_Youden_J': tr_metrics_yj[6],
        'Test_Accuracy': te_metrics_yj[0],
        'Test_Sensitivity': te_metrics_yj[1],
        'Test_Specificity': te_metrics_yj[2],
        'Test_Precision': te_metrics_yj[3],
        'Test_F1': te_metrics_yj[4],
        'Test_AUC': te_metrics_yj[5],
        'Test_Youden_J': te_metrics_yj[6],
    })

  # Save Output Files
  clf_label = f'RandomForest_{resampler_label(resample_method)}_{target_metric.capitalize()}'
  pd.DataFrame(results_05).to_excel(
      os.path.join(dir_m05, f'{clf_label}_Default_0.5_fold_metrics.xlsx'),
      index=False,
  )
  pd.DataFrame(results_youden).to_excel(
      os.path.join(
          dir_youden, f'{clf_label}_Nested_Threshold_fold_metrics.xlsx'
      ),
      index=False,
  )

  sum_df_05 = save_summary_metrics(
      results_05,
      os.path.join(dir_m05, f'{clf_label}_Default_0.5_summary_metrics.xlsx'),
      base_seed,
      clf_label,
      resampler_label(resample_method),
      'Default_Threshold_0.5',
      n_splits=OUTER_N_SPLITS,
  )
  save_confusion_matrix_plot(
      cm_05_agg,
      f'{clf_label} - Default 0.5 CM',
      os.path.join(
          dir_m05, f'{clf_label}_Default_0.5_confusion_matrix.png'
      ),
  )

  sum_df_youden = save_summary_metrics(
      results_youden,
      os.path.join(
          dir_youden, f'{clf_label}_Nested_Threshold_summary_metrics.xlsx'
      ),
      base_seed,
      clf_label,
      resampler_label(resample_method),
      'Nested_Threshold_Tuning',
      n_splits=OUTER_N_SPLITS,
  )
  save_confusion_matrix_plot(
      cm_youden_agg,
      f'{clf_label} - Nested Threshold CM',
      os.path.join(
          dir_youden, f'{clf_label}_Nested_Threshold_confusion_matrix.png'
      ),
  )

  master_summary_list.extend([sum_df_05, sum_df_youden])

  freq_df = pd.DataFrame(
      Counter(selected_features_acc).items(),
      columns=['Feature_Name', 'Selection_Frequency_Count'],
  )
  freq_df['Selection_Percentage'] = (
      freq_df['Selection_Frequency_Count'] / max(1, len(results_05))
  ) * 100
  freq_df.sort_values(
      by='Selection_Frequency_Count', ascending=False
  ).to_excel(
      os.path.join(target_dir, f'{clf_label}_feature_frequencies.xlsx'),
      index=False,
  )


# Save Master Summaries

master_summary_df = pd.concat(master_summary_list, ignore_index=True)
master_excel_path = os.path.join(
    base_output_dir, 'RandomForest_Optuna_Master_Summary.xlsx'
)
master_summary_df.to_excel(master_excel_path, index=False)

print('\nExecution Complete! Strict nested pipeline validation complete.')