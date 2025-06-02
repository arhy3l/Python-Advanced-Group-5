import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Load and label data
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")
fake["label"] = 0
real["label"] = 1

# 2. Combine, shuffle, and clean
news = pd.concat([fake, real])[["title", "text", "label"]].dropna()
news = news.sample(frac=1).reset_index(drop=True)
news["content"] = news["title"] + ": " + news["text"]

# 3. Embed text using MiniLM
print("🔍 Loading MiniLM model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("🧠 Encoding embeddings (this may take a few minutes)...")
embeddings = model.encode(news["content"].tolist(), batch_size=32, show_progress_bar=True)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, news["label"].values, test_size=0.2, random_state=42
)

# 5. Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 6. Evaluate
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("✅ Model Accuracy:", accuracy)

# 7. Save model and embedder
joblib.dump(clf, "minilm_fake_news_model.pkl")
model.save("minilm_embedder")
print("💾 Model and embedder saved.")
