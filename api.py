from flask import Flask, request, jsonify
import joblib
from preprocess import clean_text

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run()