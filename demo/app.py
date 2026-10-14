"""Step 8 (M14) - Streamlit demo.  Run:  streamlit run demo/app.py

Pipeline: tweet -> clean -> zero-shot topic -> recommended classifier -> sarcastic / not.
Falls back to the keyword tagger if `transformers` is not installed.
"""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import MODELS, RESULTS, TOPICS, TOPIC_NAMES  # noqa: E402
from preprocess import clean_text  # noqa: E402
import topic_tagging as tt  # noqa: E402

st.set_page_config(page_title="Artha-Drishti", page_icon="🔎")
st.title("Artha-Drishti: topic-aware sarcasm detection")


@st.cache_resource
def load_recommender():
    p = RESULTS / "recommender.json"
    return json.load(open(p)) if p.exists() else {}


@st.cache_resource
def load_classical(name):
    return joblib.load(MODELS / f"{name}.joblib")


@st.cache_resource
def load_neural(tag="BiGRU"):
    import torch
    from neural_model import BiGRU, encode
    ck = torch.load(MODELS / f"{tag}.pt", weights_only=False)
    model = BiGRU(np.zeros(ck["emb_shape"], dtype="float32"), ck["trainable"])
    model.load_state_dict(ck["state"]); model.eval()
    return model, ck["vocab"], encode


def get_topic(text):
    try:
        scores = tt.zero_shot_scores([text], batch_size=1)[0]
    except Exception:
        scores = tt.keyword_scores([text])[0]
        st.caption("⚠ using keyword fallback tagger (zero-shot model unavailable)")
    return TOPIC_NAMES[int(np.argmax(scores))], scores


def predict(model_name, text_clean):
    if model_name.startswith("BiGRU"):
        import torch
        model, vocab, encode = load_neural(model_name)
        with torch.no_grad():
            return float(torch.sigmoid(model(encode([text_clean], vocab)))[0])
    m = load_classical(model_name)
    if hasattr(m, "predict_proba"):
        return float(m.predict_proba([text_clean])[0][1])
    return float(m.predict([text_clean])[0])


rec = load_recommender()
tweet = st.text_area("Enter a tweet", "Oh great, another Monday. Just what I needed.")
override = st.selectbox("Classifier", ["Recommended for topic"] + sorted(set(rec.values()) | {"LogReg", "SVM"}))
if st.button("Analyse") and tweet.strip():
    topic, scores = get_topic(tweet)
    model_name = rec.get(topic, "LogReg") if override == "Recommended for topic" else override
    prob = predict(model_name, clean_text(tweet))
    st.subheader(f"Topic: {topic}")
    st.write({n: round(float(s), 3) for n, s in zip(TOPIC_NAMES, scores)})
    st.write(f"Classifier used: **{model_name}**")
    st.metric("Verdict", "Sarcastic 😏" if prob >= 0.5 else "Not sarcastic", f"score {prob:.2f}")
