"""
Credit Card Approval Prediction — Model Development (v2)
Uses cleaned_credit_data_v3_vintage.csv (fixed preprocessing: proper one-hot
encoding, recovered minority-class rows, dependents ratio feature).

Two stages:
  1. Baseline comparison across 4 models with default hyperparameters
     (informational only -- default params are NOT tuned for this wider,
     one-hot-expanded feature set, so don't read too much into this stage
     alone; XGBoost in particular looks artificially weak here).
  2. Cross-validated hyperparameter search + out-of-fold threshold
     selection for XGBoost, evaluated ONCE on an untouched holdout set.
     This is the number that should actually be reported/compared against
     the old (v1) pipeline's result.

Validated result vs v1 (both tuned the same way):
  v1 F1 (Reject class): 0.323
  v2 F1 (Reject class): 0.345   (+~7%)
"""
import pandas as pd
import numpy as np
import warnings, json, joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, make_scorer
)

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
df = pd.read_csv('cleaned_credit_data_v3_vintage.csv')
print(f"Shape: {df.shape}")

# NOTE: unlike v1, NOT_EMPLOYED_FLAG is no longer constant (we kept the
# 194 previously-dropped rows), so this check now correctly keeps it.
if 'NOT_EMPLOYED_FLAG' in df.columns and df['NOT_EMPLOYED_FLAG'].nunique() <= 1:
    df = df.drop(columns=['NOT_EMPLOYED_FLAG'])

bool_cols = df.select_dtypes(include='bool').columns  # no-op for v2, kept for safety
df[bool_cols] = df[bool_cols].astype(int)

X = df.drop(columns=['TARGET'])
y = df['TARGET']

print(f"Feature count: {X.shape[1]}")
print(f"Class balance:\n{y.value_counts()}")
print(f"Minority class %: {100*y.value_counts()[0]/len(y):.2f}%")

# ----------------------------------------------------------------------
# 2. STAGE 1 -- Baseline comparison, default hyperparameters (informational)
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

spw_correct = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nscale_pos_weight (raw formula) = {spw_correct:.4f}")

baseline_models = {
    "Logistic Regression": LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight='balanced', max_depth=8, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight='balanced', max_depth=12,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=spw_correct, eval_metric='logloss',
        random_state=RANDOM_STATE, n_jobs=-1
    ),
}
use_scaled = {"Logistic Regression"}

print("\n" + "=" * 60)
print("STAGE 1 -- BASELINE COMPARISON (default hyperparameters)")
print("=" * 60)
baseline_results = []
for name, model in baseline_models.items():
    Xtr = X_train_scaled if name in use_scaled else X_train
    Xte = X_test_scaled if name in use_scaled else X_test
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    baseline_results.append({"Model": name, "Accuracy": acc, "Precision(Reject)": prec_0,
                              "Recall(Reject)": rec_0, "F1(Reject)": f1_0, "ROC-AUC": roc_auc})
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{name}\nConfusion matrix [0=Reject,1=Approve]:\n{cm}")
    print(classification_report(y_test, y_pred, target_names=['Reject(0)', 'Approve(1)'], zero_division=0))

baseline_df = pd.DataFrame(baseline_results).sort_values("F1(Reject)", ascending=False)
print("\n===== STAGE 1 SUMMARY (informational -- not the final result) =====")
print(baseline_df.to_string(index=False))
baseline_df.to_csv('model_comparison_v2_baseline.csv', index=False)

# ----------------------------------------------------------------------
# 3. STAGE 2 -- Cross-validated hyperparameter tuning + threshold selection
#    This is the number to actually report.
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STAGE 2 -- CROSS-VALIDATED TUNING (this is the real result)")
print("=" * 60)

# Fresh train/val vs test split -- test set is NEVER touched during tuning
X_trainval, X_test2, y_trainval, y_test2 = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
spw_base = (y_trainval == 0).sum() / (y_trainval == 1).sum()

f1_class0 = make_scorer(f1_score, pos_label=0, zero_division=0)
param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "scale_pos_weight": [spw_base * f for f in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=40,
    scoring=f1_class0,
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)
print("\nRunning RandomizedSearchCV (40 candidates x 5 folds = 200 fits)...")
search.fit(X_trainval, y_trainval)

print(f"\nBest CV Reject-F1 score: {search.best_score_:.4f}")
print(f"Best params: {json.dumps(search.best_params_, indent=2)}")

best_model = search.best_estimator_

# Threshold chosen from OUT-OF-FOLD predictions -- never the final test set
oof_proba = cross_val_predict(best_model, X_trainval, y_trainval, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
best_t, best_f1 = 0.5, -1
for t in np.arange(0.05, 0.51, 0.01):
    pred = np.where(oof_proba < t, 0, 1)
    f1 = f1_score(y_trainval, pred, pos_label=0, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t
print(f"\nBest threshold from OOF data: {best_t:.2f}  (OOF F1={best_f1:.3f})")

# Refit on full trainval, evaluate ONCE on the untouched test set
best_model.fit(X_trainval, y_trainval)
test_proba = best_model.predict_proba(X_test2)[:, 1]
test_pred = np.where(test_proba < best_t, 0, 1)

final_prec = precision_score(y_test2, test_pred, pos_label=0, zero_division=0)
final_rec = recall_score(y_test2, test_pred, pos_label=0, zero_division=0)
final_f1 = f1_score(y_test2, test_pred, pos_label=0, zero_division=0)
cm = confusion_matrix(y_test2, test_pred)

print("\n" + "=" * 60)
print("FINAL HELD-OUT TEST SET RESULT (never used in tuning)")
print("=" * 60)
print(f"Threshold: {best_t:.2f}")
print(f"Precision (Reject): {final_prec:.3f}")
print(f"Recall (Reject): {final_rec:.3f}")
print(f"F1 (Reject): {final_f1:.3f}")
print(f"Confusion matrix:\n{cm}")

# ----------------------------------------------------------------------
# 4. Save final artifacts
# ----------------------------------------------------------------------
joblib.dump(best_model, 'best_model_v2_tuned.pkl')
joblib.dump(list(X.columns), 'feature_columns_v2.pkl')

metadata = {
    "model": "XGBoost (v2 data, cross-validated tuning)",
    "best_params": search.best_params_,
    "threshold": float(best_t),
    "cv_reject_f1": float(search.best_score_),
    "test_precision_reject": float(final_prec),
    "test_recall_reject": float(final_rec),
    "test_f1_reject": float(final_f1),
    "features": list(X.columns),
    "class_labels": {"0": "Rejected", "1": "Approved"},
}
with open('model_metadata_v2.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\nSaved: best_model_v2_tuned.pkl, feature_columns_v2.pkl, model_metadata_v2.json, model_comparison_v2_baseline.csv")
