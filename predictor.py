# predictor.py
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import os

MODEL_DIR = "fast_model"

def load():
    clf = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    info = joblib.load(os.path.join(MODEL_DIR, "embedder_info.pkl"))
    embedder = SentenceTransformer(info["embedder_name"])
    return embedder, clf, le

def predict(text, top_k=3):
    embedder, clf, le = load()
    emb = embedder.encode([text], convert_to_numpy=True)
    probs = clf.predict_proba(emb)[0]
    idxs = np.argsort(probs)[-top_k:][::-1]
    return [(le.inverse_transform([int(i)])[0], float(probs[i])) for i in idxs]

if __name__ == "__main__":
    txt = "Tomato leaf has yellow patches and olive-green mold underneath."
    print(predict(txt, top_k=3))
