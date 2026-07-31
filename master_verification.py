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
vectorizer = joblib.load('models/rf_vectorizer.pkl')
rf_model = joblib.load('models/rf_model.pkl')

# 2. Helpers
def entropy(s):
    if not s: return 0
    p = [s.count(c) / len(s) for c in set(s)]
    return -sum(x * np.log2(x) for x in p)

def extract_features(df):
    features = pd.DataFrame()
    features['length'] = df['url'].apply(len)
    features['num_dots'] = df['url'].apply(lambda x: x.count('.'))
    features['num_hyphens'] = df['url'].apply(lambda x: x.count('-'))
    features['num_slash'] = df['url'].apply(lambda x: x.count('/'))
    features['num_question'] = df['url'].apply(lambda x: x.count('?'))
    features['num_equals'] = df['url'].apply(lambda x: x.count('='))
    features['num_at'] = df['url'].apply(lambda x: x.count('@'))
    features['num_digits'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))
    features['entropy'] = df['url'].apply(entropy)
    features['has_ip'] = df['url'].apply(lambda x: 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', x) else 0)
    
    keywords = ['login', 'admin', 'secure', 'account', 'update', 'banking', 'confirm', 'verify', 'free', 'bonus']
    for kw in keywords:
        features[f'has_{kw}'] = df['url'].apply(lambda x: 1 if kw in x.lower() else 0)
        
    return features

# 3. Load UNSEEN Data
print("📂 Loading UNSEEN dataset #2...")
df_unseen = pd.read_csv('data/real_unseen_dataset_2.csv')
test_urls = df_unseen['url'].tolist()
test_labels = df_unseen['label'].tolist()
print(f"📊 Ready to verify {len(test_urls)} completely unseen URLs!")

# 4. Predict
print("🔍 Running predictions...")
X_feat = extract_features(df_unseen)
X_tfidf = vectorizer.transform(df_unseen['url']).toarray()
X = np.hstack((X_feat.values, X_tfidf))

preds = rf_model.predict(X)

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
