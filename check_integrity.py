import pandas as pd
import numpy as np
import joblib
import os
import re
import urllib.parse
import math
import sklearn.metrics

# --- Deepchecks Compatibility Patch ---
# 1. NumPy 2.x removed legacy aliases
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf
if not hasattr(np, 'PINF'):
    np.PINF = np.inf
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

# 2. Newer scikit-learn versions removed/renamed 'max_error'
if 'max_error' not in sklearn.metrics.get_scorer_names():
    try:
        from sklearn.metrics import make_scorer, max_error
        # Register it manually in the internal SCORERS dictionary
        import sklearn.metrics._scorer
        sklearn.metrics._scorer._SCORERS['max_error'] = make_scorer(max_error)
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration ---
MAX_LEN = 550
MODELS_DIR = 'models'
DATA_DIR = 'data'
REPORT_PATH = 'docs/deepchecks_report.html'

# --- Feature Extraction (Synced with train.py/evaluate.py) ---
def strip_protocol(url):
    url = re.sub(r'^https?://', '', str(url))
    url = re.sub(r'^www\.', '', url)
    return url.rstrip('/')

def calculate_entropy(text):
    if not text: return 0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)

def extract_hand_features(url):
    url = str(url).strip()
    c = strip_protocol(url)
    
    # 13-feature vector
    f1 = [
        len(url), len(c), url.count('.'), url.count('/'), url.count('-'), url.count('@'),
        url.count('?'), url.count('='), url.count('&'), int('//' in url[7:]),
        sum(ch.isdigit() for ch in url)/max(len(url),1),
        calculate_entropy(url)
    ]
    
    # Brand spoofing (simplified for validation script)
    domain_part = c.split('/')[0].lower()
    is_brand_spoof = 0
    for b in ['paypal', 'apple', 'microsoft', 'netflix', 'amazon']:
        if b in domain_part: # Simplified brand check
            is_brand_spoof = 1
            break
    f1.append(is_brand_spoof)
    return np.array(f1)

# --- Loading Models ---
print("📥 Loading Hybrid Models...")
vectorizer = joblib.load(os.path.join(MODELS_DIR, 'local_vectorizer.pkl'))
svm = joblib.load(os.path.join(MODELS_DIR, 'local_svm_model.pkl'))
le = joblib.load(os.path.join(MODELS_DIR, 'local_label_encoder.pkl'))
url_scaler = joblib.load(os.path.join(MODELS_DIR, 'local_url_scaler.pkl'))
tokenizer = joblib.load(os.path.join(MODELS_DIR, 'local_tokenizer.pkl'))
cnn_model = load_model(os.path.join(MODELS_DIR, 'local_hybrid_model.keras'))
meta_learner = joblib.load(os.path.join(MODELS_DIR, 'local_meta_learner_global.pkl'))

# --- Data Preparation ---
print("📂 Loading and Sampling Datasets for Validation...")
# We'll sample 2000 URLs for a quick but representative check
df_benign = pd.read_csv(os.path.join(DATA_DIR, 'definitive_benign.csv')).sample(n=1000, random_state=42)
df_benign['label'] = 'Normal'

df_malicious = pd.read_csv(os.path.join(DATA_DIR, 'definitive_malicious.csv')).sample(n=1000, random_state=42)
df_malicious['label'] = 'Phishing'

df_combined = pd.concat([df_benign, df_malicious]).sample(frac=1, random_state=42)
urls = df_combined['url'].astype(str).tolist()
labels = df_combined['label'].values

# --- Feature Reconstruction for Deepchecks ---
print("🧠 Reconstructing Meta-Features...")
processed_texts = [strip_protocol(urllib.parse.unquote(u)) for u in urls]

# 1. SVM Probs
X_tfidf = vectorizer.transform(processed_texts)
svm_probs = svm.predict_proba(X_tfidf)

# 2. CNN Probs
sequences = tokenizer.texts_to_sequences(processed_texts)
X_seq = pad_sequences(sequences, maxlen=MAX_LEN)
cnn_probs = cnn_model.predict(X_seq, batch_size=64, verbose=0)

# 3. Hand Features
hand_feats = np.array([extract_hand_features(u) for u in urls])
hand_feats_scaled = url_scaler.transform(hand_feats)

# Combine into Meta-Features DataFrame
X_meta = np.hstack([svm_probs, cnn_probs, hand_feats_scaled])
feature_names = [f'svm_p{i}' for i in range(svm_probs.shape[1])] + \
                [f'cnn_p{i}' for i in range(cnn_probs.shape[1])] + \
                [f'hand_f{i}' for i in range(hand_feats_scaled.shape[1])]

df_meta = pd.DataFrame(X_meta, columns=feature_names)
df_meta['target'] = le.transform(labels)

# --- Deepchecks Validation ---
print("🚀 Running Deepchecks Full Suite...")
# Split into "train" and "test" for drift/leakage checks
train_df, test_df = train_test_split(df_meta, test_size=0.3, random_state=42)

ds_train = Dataset(train_df, label='target', cat_features=[])
ds_test = Dataset(test_df, label='target', cat_features=[])

# Run Suite
suite = full_suite()
result = suite.run(train_dataset=ds_train, test_dataset=ds_test, model=meta_learner)

# Save Report
print(f"💾 Saving report to {REPORT_PATH}...")
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
result.save_as_html(REPORT_PATH)

print("✅ Validation Complete!")
