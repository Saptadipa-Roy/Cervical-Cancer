import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# -------------------------------
# Create models folder
# -------------------------------
os.makedirs("models", exist_ok=True)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("kag_risk_factors_cervical_cancer.csv")

# -------------------------------
# Data Cleaning
# -------------------------------
df.replace("?", np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.median(), inplace=True)

targets = ['Biopsy', 'Citology', 'Schiller', 'Hinselmann']

# -------------------------------
# Feature Selection (IMPORTANT)
# -------------------------------
X_full = df.drop(targets, axis=1)

# Train temp model for importance
temp_model = RandomForestClassifier(random_state=42)
temp_model.fit(X_full, df['Biopsy'])  # use Biopsy as reference

importances = temp_model.feature_importances_

# Select TOP N features
TOP_N = 10
indices = np.argsort(importances)[-TOP_N:]
selected_features = X_full.columns[indices]

# Reduce dataset
X = X_full[selected_features]

print("✅ Selected Features:")
print(selected_features.tolist())

# Save selected features (for UI)
pickle.dump(selected_features.tolist(), open("models/features.pkl", "wb"))

# -------------------------------
# Scaling (fit once)
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# PCA (optional but good)
# -------------------------------
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print("Original Features:", X.shape[1])
print("Reduced Features after PCA:", X_pca.shape[1])

# -------------------------------
# Train models
# -------------------------------
for target in targets:
    print(f"\nTraining {target}")

    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Handle imbalance
    # -------------------------------
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("After SMOTE:", np.bincount(y_train.astype(int)))

    # -------------------------------
    # Model (balanced)
    # -------------------------------
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model
    pickle.dump(model, open(f"models/{target.lower()}_model.pkl", "wb"))

# -------------------------------
# Save scaler & PCA
# -------------------------------
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(pca, open("models/pca.pkl", "wb"))

print("\n🎯 FINAL: Model trained with top features + PCA + balanced learning!")
