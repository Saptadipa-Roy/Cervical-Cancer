from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# -----------------------------
# Load Models
# -----------------------------
biopsy_model = pickle.load(open('models/biopsy_model.pkl', 'rb'))
citology_model = pickle.load(open('models/citology_model.pkl', 'rb'))
schiller_model = pickle.load(open('models/schiller_model.pkl', 'rb'))
hinselmann_model = pickle.load(open('models/hinselmann_model.pkl', 'rb'))

scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# -----------------------------
# Feature Names (MUST MATCH TRAINING)
# -----------------------------
feature_names = pickle.load(open('models/features.pkl', 'rb'))
# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs
        features = [float(request.form.get(name, 0)) for name in feature_names]
        final_features = np.array(features).reshape(1, -1)

        # Scale
        final_features = scaler.transform(final_features)

        # Probabilities
        biopsy_prob = biopsy_model.predict_proba(final_features)[0][1]
        citology_prob = citology_model.predict_proba(final_features)[0][1]
        schiller_prob = schiller_model.predict_proba(final_features)[0][1]
        hinselmann_prob = hinselmann_model.predict_proba(final_features)[0][1]

        # FINAL RISK (Average)
        avg_prob = (biopsy_prob + citology_prob + schiller_prob + hinselmann_prob) / 4

        overall = "HIGH RISK 🔴" if avg_prob >= 0.5 else "LOW RISK 🟢"
        confidence = round(avg_prob * 100, 2)

        return render_template(
            'result.html',
            overall=overall,
            confidence=confidence,
            biopsy=round(biopsy_prob * 100, 2),
            citology=round(citology_prob * 100, 2),
            schiller=round(schiller_prob * 100, 2),
            hinselmann=round(hinselmann_prob * 100, 2)
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)