# hyperparameter_tuning.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ================================
# 🔷 1. Load dataset
# ================================
df = pd.read_csv("login_dataset.csv")
print("✅ Dataset loaded. Shape:", df.shape)

# ================================
# 🔷 2. Encode categorical features
# ================================
le = LabelEncoder()
df['Device novelty'] = le.fit_transform(df['Device novelty'])  # Known=0, New=1

# ================================
# 🔷 3. Drop unnecessary columns
# ================================
X = df.drop(columns=[
    'Label', 'UserID', 'DeviceID', 'IP Address', 'Timestamp',  'Country', 'City'
])
y = df['Label']

# ================================
# 🔷 4. Train-Test split with stratification
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 🔷 5. Feature scaling
# ================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# 🔷 6. SMOTE Oversampling
# ================================
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train)

# ================================
# 🔷 7. Hyperparameter Tuning - Logistic Regression
# ================================
print("\n🔷 Tuning Logistic Regression...")

lr_params = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'penalty': ['l2'],        # l1 requires saga solver; l2 is standard here
    'solver': ['lbfgs', 'saga'],
    'class_weight': ['balanced', None]
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    lr_params,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

lr_grid.fit(X_train_resampled, y_train_resampled)

print("✅ Best Logistic Regression Params:", lr_grid.best_params_)
print("✅ Best ROC AUC:", lr_grid.best_score_)

# Evaluate on test set
y_pred_lr = lr_grid.best_estimator_.predict(X_test_scaled)
print("\n🔷 Logistic Regression Test Performance:")
print(classification_report(y_test, y_pred_lr))
print("ROC AUC Score:", roc_auc_score(y_test, lr_grid.best_estimator_.predict_proba(X_test_scaled)[:,1]))

# ================================
# 🔷 8. Hyperparameter Tuning - Random Forest
# ================================
print("\n🔷 Tuning Random Forest...")

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

rf_grid.fit(X_train_resampled, y_train_resampled)

print("✅ Best Random Forest Params:", rf_grid.best_params_)
print("✅ Best ROC AUC:", rf_grid.best_score_)

# Evaluate on test set
y_pred_rf = rf_grid.best_estimator_.predict(X_test_scaled)
print("\n🔷 Random Forest Test Performance:")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, rf_grid.best_estimator_.predict_proba(X_test_scaled)[:,1]))

# ================================
# 🔷 9. Confusion Matrices
# ================================
print("\n🔷 Confusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, y_pred_lr))
print("\n🔷 Confusion Matrix (Random Forest):\n", confusion_matrix(y_test, y_pred_rf))

print("\n✅ Hyperparameter tuning and model evaluation complete.")
