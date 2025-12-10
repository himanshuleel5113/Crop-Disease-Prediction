# app.py (SUPER SIMPLE & CLEAN VERSION)
import streamlit as st
import predictor
import numpy as np

st.title("🌿 Crop Disease NLP Detector")
st.write("Enter a leaf symptom description and get the predicted disease.")

# Load model once
@st.cache_resource
def load_model():
    embedder, clf, le = predictor.load()
    return embedder, clf, le

embedder, clf, le = load_model()

# Text input
text = st.text_area("Write your symptom description here:", height=150)

# Simple suggestions
st.write("### Example descriptions:")
st.write("- Tomato leaf has yellow patches and olive-green mold.")
st.write("- Potato leaf shows dark water-soaked lesions.")
st.write("- Apple leaf has circular black lesions with yellow border.")
st.write("- Corn leaf has reddish-brown powdery pustules.")
st.write("- Apple leaf looks fresh, smooth, and healthy.")

# Predict button
if st.button("Predict Disease"):
    if text.strip() == "":
        st.warning("Please enter a description.")
    else:
        emb = embedder.encode([text], convert_to_numpy=True)
        probs = clf.predict_proba(emb)[0]
        idx = np.argmax(probs)

        predicted = le.inverse_transform([idx])[0]
        confidence = probs[idx]

        st.success(f"### Prediction: **{predicted}**")
        st.write(f"Confidence: **{confidence:.3f}**")

        # Show top 3
        st.write("### Top 3 predictions:")
        top3 = np.argsort(probs)[-3:][::-1]
        for i in top3:
            st.write(f"- {le.inverse_transform([i])[0]} — {probs[i]:.3f}")
