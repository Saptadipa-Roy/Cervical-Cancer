import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("kag_risk_factors_cervical_cancer.csv")

# Data cleaning
df.replace("?", np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.median(), inplace=True)

targets = ['Biopsy', 'Citology', 'Schiller', 'Hinselmann']

scaler = StandardScaler()

for target in targets:
    print(f"Training {target}")

    X = df.drop(target, axis=1)
    pickle.dump(X.columns.tolist(), open("models/features.pkl", "wb"))
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    pickle.dump(model, open(f"models/{target.lower()}_model.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("✅ All models trained and saved!")