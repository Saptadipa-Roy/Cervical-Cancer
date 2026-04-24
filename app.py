from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load models
biopsy_model = pickle.load(open('models/biopsy_model.pkl', 'rb'))
citology_model = pickle.load(open('models/citology_model.pkl', 'rb'))
schiller_model = pickle.load(open('models/schiller_model.pkl', 'rb'))
hinselmann_model = pickle.load(open('models/hinselmann_model.pkl', 'rb'))

scaler = pickle.load(open('models/scaler.pkl', 'rb'))
pca = pickle.load(open('models/pca.pkl', 'rb'))
feature_names = pickle.load(open('models/features.pkl', 'rb'))

# -----------------------------
# Home Route (IMPORTANT FIX)
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

# -----------------------------
# Prediction Route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        features = []
        for name in feature_names:
            value = request.form.get(name)

            if value is None or value == "":
                value = 0

            features.append(float(value))

        final_input = np.array(features).reshape(1, -1)

        # Scale + PCA
        scaled = scaler.transform(final_input)
        final_data = pca.transform(scaled)

        # Predictions
        biopsy = biopsy_model.predict_proba(final_data)[0][1]
        citology = citology_model.predict_proba(final_data)[0][1]
        schiller = schiller_model.predict_proba(final_data)[0][1]
        hinselmann = hinselmann_model.predict_proba(final_data)[0][1]

        avg = (biopsy + citology + schiller + hinselmann) / 4

        # Better threshold
        if avg >= 0.6:
            overall = "HIGH RISK 🔴"
        elif avg >= 0.3:
            overall = "MEDIUM RISK 🟠"
        else:
            overall = "LOW RISK 🟢"

        return render_template(
            'result.html',
            overall=overall,
            confidence=round(avg * 100, 2),
            biopsy=round(biopsy * 100, 2),
            citology=round(citology * 100, 2),
            schiller=round(schiller * 100, 2),
            hinselmann=round(hinselmann * 100, 2)
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
