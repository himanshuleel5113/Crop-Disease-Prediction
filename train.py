# train_fast_nlp.py
"""
Fast training pipeline:
 - Loads disease_text_dataset.csv (columns: text_description,label)
 - Computes MiniLM embeddings (sentence-transformers)
 - Trains a LogisticRegression classifier
 - Saves 'model.pkl' and 'label_encoder.pkl' and 'embedder' info
"""

import os, joblib, time
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_CSV = "disease_text_dataset.csv"
OUT_DIR = "fast_model"
EMBEDDER_NAME = "all-MiniLM-L6-v2"  # small & fast

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_CSV)
df = df.dropna(subset=["text_description","label"]).reset_index(drop=True)
texts = df["text_description"].astype(str).tolist()
labels = df["label"].astype(str).tolist()
print(f"Samples: {len(texts)}")

print("Encoding labels...")
le = LabelEncoder()
y = le.fit_transform(labels)

print("Loading sentence-transformer embedder:", EMBEDDER_NAME)
embedder = SentenceTransformer(EMBEDDER_NAME)

# split (fast)
X_train, X_test, y_train, y_test, txt_train, txt_test = train_test_split(
    texts, y, texts, test_size=0.2, random_state=42, stratify=y
)

print("Computing embeddings (train)...")
t0 = time.time()
emb_train = embedder.encode(X_train, show_progress_bar=True, convert_to_numpy=True)
emb_test = embedder.encode(X_test, show_progress_bar=True, convert_to_numpy=True)
t1 = time.time()
print(f"Embedding time: {t1-t0:.1f}s")

print("Training LogisticRegression (solver='liblinear')...")
clf = LogisticRegression(max_iter=1000, solver="liblinear")
clf.fit(emb_train, y_train)

print("Evaluating on test set...")
y_pred = clf.predict(emb_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# Save artifacts
print("Saving artifacts to:", OUT_DIR)
joblib.dump(clf, os.path.join(OUT_DIR, "model.pkl"))
joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.pkl"))
# Save embedder name so predictor can reload same model
joblib.dump({"embedder_name": EMBEDDER_NAME}, os.path.join(OUT_DIR, "embedder_info.pkl"))

print("Done. To predict, use predictor.py or run app.py")
