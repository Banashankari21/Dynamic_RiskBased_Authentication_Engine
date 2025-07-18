# eda_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 🔷 Load dataset
# ================================
df = pd.read_csv("login_dataset.csv")

print("✅ Dataset loaded successfully.\n")

# ================================
# 🔷 Basic Info
# ================================
print("🔷 DataFrame Info:")
print(df.info())

print("\n🔷 First 5 rows:")
print(df.head())

print("\n🔷 Describe numeric columns:")
print(df.describe())

# ================================
# 🔷 Check for nulls
# ================================
print("\n🔷 Null value count:")
print(df.isnull().sum())

# ================================
# 🔷 Class balance (Label)
# ================================
print("\n🔷 Label distribution:")
print(df['Label'].value_counts())

sns.countplot(data=df, x='Label')
plt.title("Label Distribution (0=Genuine, 1=Risky)")
plt.savefig("label_distribution.png")
plt.show()

# ================================
# 🔷 Device novelty distribution
# ================================
sns.countplot(data=df, x='Device novelty', hue='Label')
plt.title("Device Novelty vs Label")
plt.savefig("device_novelty_vs_label.png")
plt.show()

# ================================
# 🔷 Login hour deviation distribution
# ================================
sns.histplot(df['Login hour deviation'], kde=True)
plt.title("Login Hour Deviation Distribution")
plt.savefig("login_hour_deviation.png")
plt.show()

# ================================
# 🔷 Failed attempts distribution
# ================================
sns.countplot(data=df, x='Failed attempts (24h)', hue='Label')
plt.title("Failed Attempts (24h) vs Label")
plt.savefig("failed_attempts_vs_label.png")
plt.show()

print("✅ EDA complete. Graphs saved as .png files.")
