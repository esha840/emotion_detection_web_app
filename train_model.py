# train_model_fixed.py
import argparse
import pandas as pd
import re
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not present
try:
    nltk.download('stopwords', quiet=True)
    EN_STOPWORDS = set(stopwords.words('english'))
except:
    print("NLTK stopwords download failed, using empty set")
    EN_STOPWORDS = set()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove urls, mentions, hashtags, numbers, punctuation
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main(data_path: str, out_path: str, test_size: float = 0.15, random_state: int = 42):
    print(f"Starting training with data: {data_path}")
    print(f"Output will be saved to: {out_path}")
    
    try:
        # Read and validate data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns")

        # Clean text
        df['text_clean'] = df['text'].astype(str).apply(clean_text)
        X = df['text_clean']
        y = df['label'].astype(str)
        
        print(f"Data cleaned. Unique labels: {y.unique()}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Data split: {len(X_train)} train, {len(X_test)} test")

        # Pipeline: TF-IDF + LogisticRegression
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2))),
            ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')),
        ])

        print("Training model...")
        pipeline.fit(X_train, y_train)
        print("Training completed!")

        # Evaluate
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {acc:.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred, digits=4))

        # Save pipeline - THIS IS THE CRITICAL PART
        print(f"Saving pipeline to {out_path}...")
        joblib.dump(pipeline, out_path)
        
        # Verify save
        if os.path.exists(out_path):
            file_size = os.path.getsize(out_path)
            print(f"✓ SUCCESS: Pipeline saved to {out_path} ({file_size} bytes)")
        else:
            print(f"✗ ERROR: File was not created at {out_path}")
            
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Emotion Detection Model Training ===")
    
    parser = argparse.ArgumentParser(description="Train emotion detection model")
    parser.add_argument("--data", type=str, default="sample_data.csv", help="Path to CSV with columns text,label")
    parser.add_argument("--out", type=str, default="emotion_pipeline.joblib", help="Output joblib file")
    args = parser.parse_args()

    success = main(args.data, args.out)
    
    if success:
        print("🎉 Training completed successfully!")
    else:
        print("💥 Training failed!")
        sys.exit(1)