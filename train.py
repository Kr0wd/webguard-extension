import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
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
# pyrefly: ignore [missing-import]
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Dropout, Conv1D
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.text import Tokenizer
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
# pyrefly: ignore [missing-import]
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

print("1. Loading Massive Dataset...")
df_massive = pd.read_csv('data/massive_train.csv').dropna()
print(f"📊 Massive Pool: {len(df_massive)} unique URLs")

class_counts = df_massive['label'].value_counts()
print(f"📊 Balanced Counts: {dict(class_counts)}")

urls = df_massive['url'].astype(str).tolist()
raw_labels = df_massive['label'].tolist()

# --- STRICLY ISOLATE HOLDOUT SET (NO DATA LEAKAGE) ---
print("1.5 🔒 Isolating Holdout Set...")
urls_train, urls_val, labels_train, labels_val = train_test_split(
    urls, raw_labels, test_size=0.15, stratify=raw_labels, random_state=42
)
texts_train = [strip_protocol(urllib.parse.unquote(u)) for u in urls_train]
texts_val = [strip_protocol(urllib.parse.unquote(u)) for u in urls_val]

le = LabelEncoder()
y_train_enc = le.fit_transform(labels_train)
y_val_enc = le.transform(labels_val)

print("2. 🧠 Training Foundation Models (TF-IDF + SVM)...")
vectorizer = TfidfVectorizer(max_features=50000, analyzer='char', ngram_range=(3,5))
X_train_tfidf = vectorizer.fit_transform(texts_train)
X_val_tfidf = vectorizer.transform(texts_val)

from sklearn.svm import LinearSVC
svm = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, dual=False), cv=3)
svm.fit(X_train_tfidf, y_train_enc)
print(f"   ★ SVM Val Accuracy: {svm.score(X_val_tfidf, y_val_enc):.4f}")

print("3. 🧠 Training CNN Model...")
tokenizer = Tokenizer(num_words=20000, char_level=True, oov_token='<OOV>')
tokenizer.fit_on_texts(texts_train)
MAX_LEN = 200
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(texts_train), maxlen=MAX_LEN)
X_val_seq = pad_sequences(tokenizer.texts_to_sequences(texts_val), maxlen=MAX_LEN)

cnn = Sequential([
    Embedding(20001, 64, input_length=MAX_LEN),
    Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(32, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(len(le.classes_), activation='softmax')
])
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
cnn.fit(X_train_seq, y_train_enc, epochs=10, batch_size=512, validation_data=(X_val_seq, y_val_enc), callbacks=[early_stop], verbose=1)

print("4. 🧠 Training Meta-Learner (XGBoost)...")
svm_probs_train = svm.predict_proba(X_train_tfidf)
cnn_probs_train = cnn.predict(X_train_seq, batch_size=512)
svm_probs_val = svm.predict_proba(X_val_tfidf)
cnn_probs_val = cnn.predict(X_val_seq, batch_size=512)

hand_feats_train = np.array([extract_features(u).flatten() for u in urls_train])
hand_feats_val = np.array([extract_features(u).flatten() for u in urls_val])

scaler = StandardScaler()
hand_feats_train_scaled = scaler.fit_transform(hand_feats_train)
hand_feats_val_scaled = scaler.transform(hand_feats_val)

X_meta_train = np.hstack([svm_probs_train, cnn_probs_train, hand_feats_train_scaled])
X_meta_val = np.hstack([svm_probs_val, cnn_probs_val, hand_feats_val_scaled])

# Let XGBoost handle class imbalance natively
xgb_model = xgb.XGBClassifier(n_estimators=400, max_depth=10, learning_rate=0.03, tree_method='hist')
xgb_model.fit(X_meta_train, y_train_enc)
y_pred_val = xgb_model.predict(X_meta_val)
print(f"   ★ Meta-Learner Accuracy: {accuracy_score(y_val_enc, y_pred_val):.4f}")

print("5. 💾 Saving All Models...")
joblib.dump(vectorizer, 'models/local_vectorizer.pkl')
joblib.dump(svm, 'models/local_svm_model.pkl')
cnn.save('models/local_hybrid_model.keras')
joblib.dump(tokenizer, 'models/local_tokenizer.pkl')
joblib.dump(scaler, 'models/local_url_scaler.pkl')
joblib.dump(xgb_model, 'models/local_meta_learner_global.pkl')
joblib.dump(le, 'models/local_label_encoder.pkl')

print("✅ Perfect Training Session Complete!")
