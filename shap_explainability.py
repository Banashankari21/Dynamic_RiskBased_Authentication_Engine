# shap_explainability.py

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

print("🔷 Loading dataset...")
df = pd.read_csv("login_dataset.csv")
print(f"✅ Dataset loaded. Shape: {df.shape}")

# Encode categorical features
le_dev = LabelEncoder()
df['Device novelty'] = le_dev.fit_transform(df['Device novelty'])
df['Country_encoded'] = LabelEncoder().fit_transform(df['Country'])
df['City_encoded'] = LabelEncoder().fit_transform(df['City'])

X = df.drop(columns=[
    'Label', 'UserID', 'DeviceID', 'IP Address', 'Timestamp',
    'Country', 'City'
])
y = df['Label']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("🔷 Applying SMOTE resampling...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)
X_res_df = pd.DataFrame(X_res, columns=X.columns)
print(f"✅ Resampled dataset shape: {X_res_df.shape}")

# Train model
print("🔷 Training Random Forest model...")
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_res, y_res)
print("✅ Model trained successfully.")

# Initialize SHAP
print("🔷 Initializing SHAP explainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_res)
print("✅ SHAP explainer initialized.")

# Global plot
print("🔷 Generating global feature importance plot...")
shap.summary_plot(shap_values, X_res_df, plot_type="bar", show=False)
plt.savefig("shap_global_importance.png")
plt.close()
print("✅ Global feature importance saved as shap_global_importance.png")

# Local plot - robust implementation using shap.plots.force
print("🔷 Generating local force plot for risky sample...")

try:
    if isinstance(shap_values, list) and len(shap_values) > 1:
        risky_class_shap_values = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        risky_class_shap_values = shap_values
        expected_value = explainer.expected_value

    risky_samples = np.where(y_res == 1)[0]

    found = False
    for idx in risky_samples:
        if idx < risky_class_shap_values.shape[0]:
            shap.initjs()

            # Generate interactive force plot
            force_plot = shap.plots.force(
                expected_value,
                risky_class_shap_values[idx],
                X_res_df.iloc[idx]
            )

            # Save as HTML (recommended for interactive force plots)
            shap.save_html("shap_local_explanation.html", force_plot)
            print("✅ Local force plot saved as shap_local_explanation.html")
            found = True
            break

    if not found:
        print("⚠️ No risky sample found within SHAP output bounds.")
except Exception as e:
    print(f"❌ Exception occurred: {e}")
