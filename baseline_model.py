# baseline_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
le_device = LabelEncoder()
df['Device novelty'] = le_device.fit_transform(df['Device novelty'])  # Known=0, New=1

# Encode Country and City
le_country = LabelEncoder()
le_city = LabelEncoder()

df['Country_encoded'] = le_country.fit_transform(df['Country'])
df['City_encoded'] = le_city.fit_transform(df['City'])

# ================================
# 🔷 3. Feature engineering
# ================================
# Drop unprocessed columns
X = df.drop(columns=[
    'Label',
    'UserID',
    'DeviceID',
    'IP Address',
    
    'Timestamp',
    'Country',
    'City'
])

# Add encoded country and city to X (already done above)

y = df['Label']

print("✅ Features for modeling:", list(X.columns))

# ================================
# 🔷 4. Train-Test split with stratification
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Data split completed. Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# ================================
# 🔷 5. Feature scaling (StandardScaler)
# ================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# 🔷 6. Handle class imbalance using SMOTE
# ================================
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train)

print("✅ SMOTE applied.")
print("Class distribution before SMOTE:\n", y_train.value_counts())
print("Class distribution after SMOTE:\n", pd.Series(y_train_resampled).value_counts())

# ================================
# 🔷 7. Train Logistic Regression (Baseline linear model)
# ================================
lr_model = LogisticRegression(random_state=42, class_weight='balanced')
lr_model.fit(X_train_resampled, y_train_resampled)

y_pred_lr = lr_model.predict(X_test_scaled)
print("\n🔷 Logistic Regression Performance:")
print(classification_report(y_test, y_pred_lr))
print("ROC AUC Score:", roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled)[:,1]))

# ================================
# 🔷 8. Train Random Forest (Non-linear ensemble model)
# ================================
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_resampled, y_train_resampled)

y_pred_rf = rf_model.predict(X_test_scaled)
print("\n🔷 Random Forest Performance:")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:,1]))

# ================================
# 🔷 9. Confusion Matrix comparison
# ================================
print("\n🔷 Confusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, y_pred_lr))
print("\n🔷 Confusion Matrix (Random Forest):\n", confusion_matrix(y_test, y_pred_rf))

print("\n✅ Baseline model training complete.")
