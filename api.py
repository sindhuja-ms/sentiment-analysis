from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from preprocess import clean_text

app = Flask(__name__)
CORS(app)  # ✅ IMPORTANT (allows frontend access)

# Load model
model = joblib.load("models/logistic.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.route("/")
def home():
    return "API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    clean = clean_text(text)
    vec = vectorizer.transform([clean])

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    return jsonify({
        "prediction": int(pred),
        "confidence": float(max(prob))
    })

# Render port fix
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
