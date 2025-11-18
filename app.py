# app.py
import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

# Utility: clean text (same as training)
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Map label -> emoji and motivational/calm message
EMOTION_METADATA = {
    "happiness": {"emoji": "😄", "message": "Great to see you smiling! Keep going — good things are coming."},
    "sadness": {"emoji": "😔", "message": "It's okay to feel sad. Take a breath and be kind to yourself."},
    "anger": {"emoji": "😠", "message": "Take a moment — a short break can help cool things down."},
    "fear": {"emoji": "😨", "message": "You are stronger than you think. Breathe slowly and focus on the next small step."},
    "surprise": {"emoji": "😲", "message": "Surprises can be exciting — embrace the unexpected!" },
    "neutral": {"emoji": "😐", "message": "A calm moment — take it as a chance to regroup and plan."},
}

DEFAULT_MODEL_PATH = "emotion_pipeline.joblib"

@st.cache_resource
def load_pipeline(path=DEFAULT_MODEL_PATH):
    try:
        pipeline = joblib.load(path)
        return pipeline
    except Exception as e:
        return None

def predict_text(pipeline, text: str):
    c = clean_text(text)
    pred = pipeline.predict([c])[0]
    proba = None
    try:
        proba = pipeline.predict_proba([c])[0]
    except Exception:
        pass
    return pred, proba

st.set_page_config(page_title="Emotion Detection Web App", page_icon="🤖", layout="centered")
st.title("Emotion Detection Web App  🤖💬")
st.write("Upload a CSV (text,label) to train or choose a saved model. Type or pick a message and predict the emotion.")

# Sidebar: Model loading / training options
st.sidebar.header("Model & Data")
model_file = st.sidebar.file_uploader("Upload a pre-trained pipeline (.joblib)", type=["joblib"])
use_uploaded_model = False
if model_file:
    # save temporary and load
    tmp = "uploaded_pipeline.joblib"
    with open(tmp, "wb") as f:
        f.write(model_file.getbuffer())
    pipeline = load_pipeline(tmp)
    if pipeline is None:
        st.sidebar.error("Could not load uploaded joblib. Ensure it's a scikit-learn Pipeline saved with joblib.")
    else:
        use_uploaded_model = True
        st.sidebar.success("Uploaded model loaded.")
else:
    pipeline = load_pipeline(DEFAULT_MODEL_PATH)
    if pipeline is None:
        st.sidebar.warning(f"No saved pipeline found at {DEFAULT_MODEL_PATH}. You must upload a pipeline or train one below.")
    else:
        st.sidebar.success("Loaded saved pipeline.")

# Option to upload CSV to train quickly in-app
st.sidebar.subheader("Train with CSV (optional)")
csv_file = st.sidebar.file_uploader("Upload CSV (text,label) to train a new model", type=["csv"])
if csv_file is not None:
    df = pd.read_csv(csv_file)
    if 'text' not in df.columns or 'label' not in df.columns:
        st.sidebar.error("CSV must contain 'text' and 'label' columns.")
    else:
        st.sidebar.info(f"CSV contains {len(df)} rows.")
        if st.sidebar.button("Train model with uploaded CSV"):
            with st.spinner("Training... (small datasets train quickly)"):
                # lightweight training: tfidf + logistic
                from sklearn.pipeline import Pipeline
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                X = df['text'].astype(str).apply(clean_text)
                y = df['label'].astype(str)
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words='english', max_features=15000, ngram_range=(1,2))),
                    ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')),
                ])
                pipeline.fit(X, y)
                joblib.dump(pipeline, "emotion_pipeline.joblib")
                st.sidebar.success("Trained and saved pipeline as emotion_pipeline.joblib")
                st.experimental_rerun()

# Main area: allow choosing dataset messages if CSV uploaded
st.header("1) Choose or type a message")
uploaded_dataset = st.file_uploader("Upload sample dataset CSV (optional) to pick messages via dropdown", type=["csv"], key="pickdata")
selected_text = None
if uploaded_dataset is not None:
    df_sample = pd.read_csv(uploaded_dataset)
    if 'text' in df_sample.columns:
        # show only first 500
        options = df_sample['text'].astype(str).tolist()[:500]
        selected_text = st.selectbox("Select a message from uploaded CSV", options)
    else:
        st.error("CSV must have a 'text' column to pick messages.")

st.write("Or type your message below:")
user_input = st.text_area("Your message", value=(selected_text if selected_text else ""), height=120)

st.markdown("---")
st.header("2) Predict emotion")
if pipeline is None:
    st.error("No model loaded. Upload a pre-trained .joblib pipeline or train using a CSV in the sidebar.")
else:
    if st.button("Predict"):
        if not user_input or user_input.strip() == "":
            st.warning("Please type or select a message to predict.")
        else:
            with st.spinner("Predicting..."):
                label, proba = predict_text(pipeline, user_input)
                meta = EMOTION_METADATA.get(label, {"emoji":"❓","message":"No tip available for this label."})
                st.markdown(f"### Detected emotion: *{label}* {meta['emoji']}")
                if proba is not None:
                    # If classes are stored in pipeline
                    try:
                        classes = pipeline.classes_
                        probs = {c: float(p) for c,p in zip(classes, proba)}
                        probs_df = pd.DataFrame(list(probs.items()), columns=["emotion","probability"]).sort_values("probability", ascending=False)
                        st.table(probs_df)
                    except Exception:
                        pass
                st.info(meta['message'])

st.markdown("---")
st.header("Tech stack & Notes")
st.write("""
- Backend model: scikit-learn (TF-IDF + Logistic Regression).
- Save/Load model using joblib.
- If you train with your own CSV: ensure labels are consistent (e.g., happiness, sadness, anger, fear, surprise, neutral).
- For production or better accuracy: gather a larger labeled dataset and consider using more advanced models (transformers).
""")
st.caption("Developed with Python & Streamlit. 💡")