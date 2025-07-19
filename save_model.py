# save_model.py

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("login_dataset.csv")

# Encode categorical features
le_dev = LabelEncoder()
df['Device novelty'] = le_dev.fit_transform(df['Device novelty'])
df['Country_encoded'] = LabelEncoder().fit_transform(df['Country'])
df['City_encoded'] = LabelEncoder().fit_transform(df['City'])

# Prepare X and y
X = df.drop(columns=[
    'Label', 'UserID', 'DeviceID', 'IP Address', 'Timestamp',
    'Country', 'City'
])
y = df['Label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE for balanced training data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Train Random Forest model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_res, y_res)

# Save model to risk_model.pkl
with open("risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ risk_model.pkl saved successfully.")
