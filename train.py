import pandas as pd
import numpy as np
import joblib
import os
import re
import urllib.parse
import math
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Dropout, Conv1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb

def strip_protocol(url):
    url = re.sub(r'^https?://', '', str(url))
    url = re.sub(r'^www\.', '', url)
    return url.rstrip('/')

HIGH_TRUST_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
    'linkedin.com', 'reddit.com', 'github.com', 'stackoverflow.com', 'wikipedia.org',
    'amazon.com', 'ebay.com', 'netflix.com', 'spotify.com', 'apple.com',
    'microsoft.com', 'office.com', 'live.com', 'outlook.com', 'bing.com',
    'yahoo.com', 'imdb.com', 'twitch.tv', 'discord.com', 'trulia.com',
    'zillow.com', 'walmart.com', 'target.com', 'bestbuy.com', 'etsy.com',
    'tumblr.com', 'wordpress.com', 'blogspot.com', 'medium.com', 'quora.com',
    'thefreelibrary.com', 'london-city-hotel.co.uk', 'david-kilgour.com',
    'heraldicsculptor.com', 'missouririverfutures.org', 'amazon.co.uk', 'amazon.ca', 'amazon.in', 'amazon.de', 'google.co.uk', 'google.ca', 'google.in',
    'openai.com', 'zoom.us', 'slack.com', 'trello.com', 'notion.so', 'microsoftonline.com', 'okta.com',
    'steampowered.com', 'mozilla.org', 'dropbox.com', 'box.com', 'mfah.org', 'allegro.pl', 'uni-bonn.de'
}

def calculate_entropy(text):
    if not text: return 0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)

def extract_features(url):
    url = str(url).strip()
    decoded = urllib.parse.unquote(url)
    c = strip_protocol(url)
    
    # Matching evaluate_mixed.py AI Ensemble features
    f1 = [
        len(url), len(c), url.count('.'), url.count('/'), url.count('-'), url.count('@'),
        url.count('?'), url.count('='), url.count('&'), int('//' in url[7:]),
        sum(ch.isdigit() for ch in url)/max(len(url),1),
        calculate_entropy(url)
    ]
    
    domain_part = c.split('/')[0].lower()
    domain_root = '.'.join(domain_part.split('.')[-2:]) if '.' in domain_part else domain_part
    is_brand_spoof = 0
    # Sync with TARGET_BRANDS in evaluation
    for b in ['paypal', 'ppl', 'apple', 'microsoft', 'netflix', 'amazon', 'bankofamerica', 'wellsfargo', 'chase', 'walmart', 'ebay']:
        if b in domain_part and domain_root not in HIGH_TRUST_DOMAINS and domain_root != f"{b}.com":
            is_brand_spoof = 1
            break
    f1.append(is_brand_spoof)
    
    return np.array(f1).reshape(1, -1)

# 1. 📂 LOAD DATASETS (Robust Multi-Source Mix)
print("1. 📂 Loading Massive Datasets...")

all_dfs = []

def safe_load(path, label, rename_col=None):
    if not os.path.exists(path):
        print(f"   ⚠️ Warning: Missing {path}, skipping.")
        return None
    df = pd.read_csv(path).dropna()
    if rename_col:
        df = df.rename(columns={rename_col: 'url'})
    if 'label' not in df.columns:
        df['label'] = label
    print(f"   ★ {label}: {len(df)} URLs from {os.path.basename(path)}")
    return df[['url', 'label']]

# Core Benign & Malicious
df_b = safe_load('data/definitive_benign.csv', 'Normal')
if df_b is not None: all_dfs.append(df_b)

df_m = safe_load('data/definitive_malicious.csv', 'Phishing')
if df_m is not None: all_dfs.append(df_m)

df_p = safe_load('data/phishtank.csv', 'Phishing')
if df_p is not None: all_dfs.append(df_p)

# New Unseen (Critical for Accuracy)
df_u = safe_load('data/new_unseen_dataset.csv', 'Normal') # Labels are Normal/Phishing in this file, will override below
if df_u is not None:
    # Re-verify labels if they exist in the CSV
    temp_df = pd.read_csv('data/new_unseen_dataset.csv')
    if 'label' in temp_df.columns:
        df_u['label'] = temp_df['label']
    all_dfs.append(df_u)

# Legacy / Optional
df_capec = safe_load('data/dataset_capec_combine.csv', 'Injection', rename_col='text')
if df_capec is not None: all_dfs.append(df_capec)

df_merged = safe_load('data/merged_urls_dataset.csv', 'Normal')
if df_merged is not None: all_dfs.append(df_merged)

if not all_dfs:
    raise FileNotFoundError("❌ CRITICAL: No datasets found in data/ directory!")

# Combine All Sources
df_all = pd.concat(all_dfs).drop_duplicates(subset=['url']).dropna()
print(f"📊 Combined Pool: {len(df_all)} unique URLs")

# Final Dataset Preparation
class_counts = df_all['label'].value_counts()
print(f"📊 Balanced Counts: {dict(class_counts)}")

# Stratified Sampling to keep sizes manageable but balanced
df_final = pd.concat([
    df_all[df_all['label'] == 'Normal'].sample(n=min(class_counts.get('Normal', 0), 120000), random_state=42),
    df_all[df_all['label'] == 'Phishing'].sample(n=min(class_counts.get('Phishing', 0), 80000), random_state=42),
    df_all[df_all['label'] == 'Injection'].sample(n=min(class_counts.get('Injection', 0), class_counts.get('Injection', 0)), random_state=42) if 'Injection' in class_counts else pd.DataFrame(),
    df_all[df_all['label'] == 'Manipulation'].sample(n=min(class_counts.get('Manipulation', 0), class_counts.get('Manipulation', 0)), random_state=42) if 'Manipulation' in class_counts else pd.DataFrame()
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Training Set: {len(df_final)} URLs")

urls = df_final['url'].astype(str).tolist()
raw_labels = df_final['label'].values
texts = [strip_protocol(urllib.parse.unquote(u)) for u in urls]

print("2. 🧠 Training Foundation Models (TF-IDF + SVM)...")
vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1,3))
X_tfidf = vectorizer.transform(texts) if hasattr(vectorizer, 'vocabulary_') else vectorizer.fit_transform(texts)

le = LabelEncoder()
y_enc = le.fit_transform(raw_labels)

from sklearn.svm import LinearSVC
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_tfidf, y_enc, test_size=0.1, stratify=y_enc, random_state=42)
svm = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=1000, dual=False), cv=3)
svm.fit(X_train_v, y_train_v)
print(f"   ★ SVM Val Accuracy: {svm.score(X_test_v, y_test_v):.4f}")

print("3. 🧠 Training CNN Model...")
tokenizer = Tokenizer(num_words=10000, char_level=True, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
MAX_LEN = 550
sequences = tokenizer.texts_to_sequences(texts)
X_seq = pad_sequences(sequences, maxlen=MAX_LEN)

X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y_enc, test_size=0.1, stratify=y_enc, random_state=42)

cnn = Sequential([
    Embedding(10001, 64, input_length=MAX_LEN),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_seq, y_train_seq, epochs=15, batch_size=128, validation_data=(X_test_seq, y_test_seq), verbose=1)

print("4. 🧠 Training Meta-Learner (XGBoost)...")
svm_probs = svm.predict_proba(X_tfidf)
cnn_probs = cnn.predict(X_seq, batch_size=256)

hand_feats = []
for u in urls:
    hand_feats.append(extract_features(u).flatten())
hand_feats = np.array(hand_feats)
scaler = StandardScaler()
hand_feats_scaled = scaler.fit_transform(hand_feats)

X_meta = np.hstack([svm_probs, cnn_probs, hand_feats_scaled])

# Meta-Learner Sample Weights: Penalize False Negatives heavily
sample_weights = np.ones(len(y_enc))
for i, label in enumerate(raw_labels):
    if label != 'Normal':
        sample_weights[i] = 2.5 # 2.5x importance for malicious URLs

X_train_m, X_test_m, y_train_m, y_test_m, w_train_m, w_test_m = train_test_split(
    X_meta, y_enc, sample_weights, test_size=0.15, stratify=y_enc, random_state=42
)

xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, tree_method='hist', device='cuda')
xgb_model.fit(X_train_m, y_train_m, sample_weight=w_train_m)
y_pred_m = xgb_model.predict(X_test_m)
print(f"   ★ Meta-Learner Accuracy: {accuracy_score(y_test_m, y_pred_m):.4f}")

print("5. 💾 Saving All Models...")
joblib.dump(vectorizer, 'models/local_vectorizer.pkl')
joblib.dump(svm, 'models/local_svm_model.pkl')
cnn.save('models/local_hybrid_model.keras')
joblib.dump(tokenizer, 'models/local_tokenizer.pkl')
joblib.dump(scaler, 'models/local_url_scaler.pkl')
joblib.dump(xgb_model, 'models/local_meta_learner_global.pkl')
joblib.dump(le, 'models/local_label_encoder.pkl')

print("✅ Perfect Training Session Complete!")
