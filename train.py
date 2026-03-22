import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

from preprocess import clean_text

# ✅ Load dataset from Google Drive
url = "https://drive.google.com/uc?id=1d2DxIm3ckFzqu6SW_giw_6DQ3MZZOBz8"

df = pd.read_csv(url)
df = df[['review', 'sentiment']]

# Remove missing values
df = df.dropna()

# Clean text
df['review'] = df['review'].astype(str)
df['review'] = df['review'].apply(clean_text)

# Convert labels
df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models
models = {
    "logistic": LogisticRegression(max_iter=200),
    "naive_bayes": MultinomialNB(),
    "random_forest": RandomForestClassifier(n_estimators=50)
}

# Create models folder
os.makedirs("models", exist_ok=True)

# Train and save
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    print(f"\n{name.upper()}")
    print(classification_report(y_test, preds))

    joblib.dump(model, f"models/{name}.pkl")

# Save vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")
