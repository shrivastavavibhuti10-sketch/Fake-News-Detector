import os
import pandas as pd
import nltk
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Download NLTK resources
# -----------------------------
nltk.download("stopwords")
from nltk.corpus import stopwords

# -----------------------------
# Paths (SAFE & FIXED)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")
COMBINED_PATH = os.path.join(DATA_DIR, "fake_news_dataset.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# -----------------------------
# Load datasets
# -----------------------------
print("Loading datasets...")

fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

fake_df["label"] = 0   # Fake
true_df["label"] = 1   # Real

# -----------------------------
# Combine datasets
# -----------------------------
dataset = pd.concat([fake_df, true_df], axis=0)
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Keep only required columns
dataset = dataset[["text", "label"]]

# -----------------------------
# Save combined dataset
# -----------------------------
dataset.to_csv(COMBINED_PATH, index=False)
print("Dataset combined and saved as fake_news_dataset.csv")

# -----------------------------
# Text Vectorization
# -----------------------------
stop_words = stopwords.words("english")

vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    max_df=0.7
)

X = vectorizer.fit_transform(dataset["text"])
y = dataset["label"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# Save Model & Vectorizer
# -----------------------------
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("Model and vectorizer saved successfully")