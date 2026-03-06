import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import os
import re
import math
import urllib.parse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("🏗️  Setting up Master Verification Environment (V2)...")

# 1. Load Models
print("📥 Loading models and preprocessors...")
vectorizer   = joblib.load('models/local_vectorizer.pkl')
svm          = joblib.load('models/local_svm_model.pkl')
le           = joblib.load('models/local_label_encoder.pkl')
url_scaler   = joblib.load('models/local_url_scaler.pkl')
tokenizer    = joblib.load('models/local_tokenizer.pkl')
cnn_model    = load_model('models/local_hybrid_model.keras')
meta_model   = joblib.load('models/local_meta_learner_global.pkl')

MAX_LEN = 550

# 2. Helpers
def strip_protocol(url):
    url = re.sub(r'^https?://', '', str(url))
    url = re.sub(r'^www\.', '', url)
    return url.rstrip('/')

def calculate_entropy(text):
    if not text: return 0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)

TARGET_BRANDS = ['paypal', 'apple', 'microsoft', 'netflix', 'amazon']
HIGH_TRUST_DOMAINS = {'google.com', 'youtube.com', 'facebook.com', 'github.com'}

def extract_features(url):
    url = str(url).strip()
    c = strip_protocol(url)
    f1 = [
        len(url), len(c), url.count('.'), url.count('/'), url.count('-'), url.count('@'),
        url.count('?'), url.count('='), url.count('&'), int('//' in url[7:]),
        sum(ch.isdigit() for ch in url)/max(len(url),1),
        calculate_entropy(url)
    ]
    domain_part = c.split('/')[0].lower()
    domain_root = '.'.join(domain_part.split('.')[-2:]) if '.' in domain_part else domain_part
    is_brand_spoof = 1 if any(b in domain_part and domain_root not in HIGH_TRUST_DOMAINS and domain_root != f"{b}.com" for b in TARGET_BRANDS) else 0
    f1.append(is_brand_spoof)
    return np.array(f1, dtype=np.float32).reshape(1, -1)

# 3. Load UNSEEN Data
print("📂 Loading UNSEEN datasets...")
# Malicious from CyberCrime Tracker (Text file, one per line)
with open('data/cybercrime_tracker.txt', 'r', encoding='latin1') as f:
    mal_urls = [line.strip() for line in f if line.strip()]

# Benign from Majestic 200k (CSV)
df_ben = pd.read_csv('data/majestic_benign_200k.csv', encoding='latin1', on_bad_lines='skip', engine='python').dropna()
ben_urls = df_ben['url'].tolist()

# Sample 2000 from each for a quick but robust test
import random
random.seed(42)
sample_mal = random.sample(mal_urls, min(2000, len(mal_urls)))
sample_ben = random.sample(ben_urls, min(2000, len(ben_urls)))

test_urls = sample_mal + sample_ben
test_labels = [1] * len(sample_mal) + [0] * len(sample_ben)

print(f"📊 Ready to verify {len(test_urls)} completely unseen URLs!")

# 4. Predict
print("🔍 Running predictions...")
preds = []
for i, url in enumerate(test_urls):
    clean = strip_protocol(url)
    
    # Probabilities
    X_vec = vectorizer.transform([clean])
    svm_p = svm.predict_proba(X_vec)
    
    seq = tokenizer.texts_to_sequences([clean])
    X_sq = pad_sequences(seq, maxlen=MAX_LEN)
    cnn_p = cnn_model.predict(X_sq, verbose=0)
    
    feat = url_scaler.transform(extract_features(url))
    
    # Meta
    meta_in = np.hstack([svm_p, cnn_p, feat])
    probs = meta_model.predict_proba(meta_in)[0]
    
    # Map back to Normal/Malicious logic
    normal_idx = list(le.classes_).index('Normal')
    malicious_confidence = 1.0 - probs[normal_idx]
    
    # Threshold 0.5 for balanced test
    preds.append(1 if malicious_confidence > 0.5 else 0)
    
    if (i+1) % 500 == 0:
        print(f"   Processed {i+1}/{len(test_urls)}...")

# 5. Results
acc = accuracy_score(test_labels, preds) * 100
prec = precision_score(test_labels, preds) * 100
rec = recall_score(test_labels, preds) * 100
f1 = f1_score(test_labels, preds) * 100

print("\n" + "="*50)
print("🏆 MASTER VERIFICATION REPORT (UNSEEN DATA)")
print("="*50)
print(f"Accuracy  : {acc:.2f}%")
print(f"Precision : {prec:.2f}%")
print(f"Recall    : {rec:.2f}%")
print(f"F1-Score  : {f1:.2f}%")
print("="*50)
