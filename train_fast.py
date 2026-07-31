import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import re
import math

def entropy(s):
    p, lns = pd.Series(list(s)).value_counts() / len(s), len(s)
    return -sum(p * np.log2(p))

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

# 1. Load Data
print("Loading data...")
dfs = []
b = pd.read_csv('data/definitive_benign.csv')
b = b.rename(columns={b.columns[0]: 'url'}) if 'url' not in b.columns else b
b['label'] = 0
dfs.append(b)

m = pd.read_csv('data/definitive_malicious.csv')
m = m.rename(columns={m.columns[0]: 'url'}) if 'url' not in m.columns else m
m['label'] = 1
dfs.append(m)

p = pd.read_csv('data/verified_online.csv') # phishtank
p = p[['url']].copy()
p['label'] = 1
dfs.append(p.sample(20000, random_state=42)) # mix in some phishtank

df = pd.concat(dfs).drop_duplicates(subset=['url']).dropna().sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total URLs: {len(df)}")

# 2. Extract Features
print("Extracting features...")
X_feat = extract_features(df)
y = df['label'].values

# TF-IDF
print("TF-IDF...")
vectorizer = TfidfVectorizer(max_features=15000, analyzer='word', token_pattern=r'[a-zA-Z0-9]+', ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(df['url']).toarray()

# Combine
X = np.hstack((X_feat.values, X_tfidf))

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# 4. Train Model
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)

# 5. Evaluate
preds = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
print(f"Precision: {precision_score(y_test, preds)*100:.2f}%")
print(f"Recall: {recall_score(y_test, preds)*100:.2f}%")
import joblib
joblib.dump(rf, 'models/rf_model.pkl', compress=('gzip', 3))
joblib.dump(vectorizer, 'models/rf_vectorizer.pkl')
print("✅ Saved RF Model and Vectorizer to models/")
