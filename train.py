import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

from preprocess import clean_text

# Load dataset (with fallback)
try:
    df = pd.read_csv("data/feedback.csv")
    df = df[['review', 'sentiment']]

except:
    print("Using fallback sample dataset")

    df = pd.DataFrame({
        "review": [
            "this product is amazing",
            "worst experience ever",
            "i love this",
            "very bad service",
            "excellent quality",
            "not good at all"
        ],
        "sentiment": [
            "positive",
            "negative",
            "positive",
            "negative",
            "positive",
            "negative"
        ]
    })

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

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models
models = {
    "logistic": LogisticRegression(),
    "naive_bayes": MultinomialNB(),
    "random_forest": RandomForestClassifier()
}

# Train and save
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    print(f"\n{name.upper()}")
    print(classification_report(y_test, preds))

    joblib.dump(model, f"models/{name}.pkl")

# Save vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")
