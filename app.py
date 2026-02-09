from flask import Flask, render_template, request
import joblib
import os
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = os.path.join("model", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form["news"]

    # Vectorize input
    transformed_text = vectorizer.transform([news_text])

    # Prediction
    prediction_label = model.predict(transformed_text)[0]
    probability = model.predict_proba(transformed_text)[0]

    if prediction_label == 0:
        prediction = "Fake News"
        confidence = probability[0]
        result_class = "fake"
    else:
        prediction = "Real News"
        confidence = probability[1]
        result_class = "real"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence * 100, 2),
        result_class=result_class
    )

if __name__ == "__main__":
    app.run(debug=True)