import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import random
import urllib.parse
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Lambda, Embedding, LSTM, Dense, Conv1D, GlobalMaxPooling1D, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

# --- CONFIGURATION ---
MAX_LEN = 500
VOCAB_SIZE = 20000   
EMBEDDING_DIM = 128  
EPOCHS = 20          
BATCH_SIZE = 64      
PAIRS_COUNT = 40000  

print("🚀 STARTING FLAWLESS TRAINING SESSION (Destroying Domain Bias)")

# --- THE FIX: BULLETPROOF DATA NORMALIZATION ---
def clean_capec_payload(text):
    """ Aggressively formats CAPEC data to look exactly like real URLs """
    text = str(text).strip()
    # 1. Remove HTTP verbs if present (e.g. "GET /login HTTP/1.1" -> "/login")
    text = re.sub(r'^(GET|POST|PUT|DELETE|OPTIONS|HEAD)\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+HTTP/.*$', '', text, flags=re.IGNORECASE)
    text = text.lstrip(' /')
    
    # 2. Inject a realistic domain so the AI learns domains are normal
    domains = ['google.com/', 'github.com/', 'example.com/', 'mysite.org/', 'dashboard.net/']
    return random.choice(domains) + text

def clean_phishtank_url(url):
    """ Strips protocol and www from PhishTank so it matches our new CAPEC data """
    url = str(url).strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    return url

def adversarial_augment(urls, labels, fraction=0.3):
    aug_urls, aug_labels = [], []
    for url, label in zip(urls, labels):
        if str(label) != 'Normal' and random.random() < fraction:
            method = random.choice(['case', 'encoding', 'param'])
            new_url = url
            try:
                if method == 'case': new_url = "".join([c.upper() if random.random() > 0.5 else c.lower() for c in url])
                elif method == 'encoding': new_url = urllib.parse.quote(url)
                elif method == 'param': new_url = f"{url}{'&' if '?' in url else '?'}v={random.randint(1,99999)}"
            except: pass
            aug_urls.append(new_url)
            aug_labels.append(label)
    return aug_urls, aug_labels

# --- 1. DATA INGESTION ---
print("\n[1/4] 📥 Loading & Aggressively Formatting Datasets...")

# A. CAPEC Data (Add domains to everything)
df1 = pd.read_csv('dataset_capec_combine.csv').dropna(subset=['text', 'label'])
capec_texts = [clean_capec_payload(t) for t in df1['text'].values]
capec_labels = df1['label'].values

# B. PhishTank Data (Strip domains down)
df2 = pd.read_csv('phishtank.csv')
phish_urls = [clean_phishtank_url(u) for u in df2['url'].dropna().values]
phish_labels = ['Phishing'] * len(phish_urls)

all_texts = list(capec_texts) + list(phish_urls)
all_labels = list(capec_labels) + list(phish_labels)

# Normalize labels
clean_labels = []
for lbl in all_labels:
    s = str(lbl).lower()
    if 'normal' in s or 'benign' in s: clean_labels.append('Normal')
    elif 'sql' in s: clean_labels.append('SQL Injection')
    elif 'xss' in s or 'script' in s: clean_labels.append('XSS')
    elif 'phish' in s: clean_labels.append('Phishing')
    else: clean_labels.append('Malicious')

# Adversarial Attacks
adv_urls, adv_lbls = adversarial_augment(all_texts, clean_labels, fraction=0.3)
all_texts.extend(adv_urls)
clean_labels.extend(adv_lbls)

df_raw = pd.DataFrame({'text': all_texts, 'label': clean_labels})

# --- STRICT BALANCING ---
print("\n⚖️ BALANCING DATASET...")
TARGET_SIZE = 15000 
balanced_dfs = []

for label in df_raw['label'].unique():
    df_sub = df_raw[df_raw['label'] == label]
    if len(df_sub) > TARGET_SIZE:
        df_sub = df_sub.sample(n=TARGET_SIZE, random_state=42)
    elif len(df_sub) < TARGET_SIZE:
        df_sub = df_sub.sample(n=TARGET_SIZE, replace=True, random_state=42)
    balanced_dfs.append(df_sub)

df_final = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
texts = df_final['text'].tolist()
labels = df_final['label'].values
print(f"✅ FINAL FLAWLESS DATASET SIZE: {len(texts)} URLs")

# --- 2. TRAIN SVM ---
print("\n[2/4] 🧠 Training SVM...")
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,3)) 
X_vec = vectorizer.fit_transform(texts)
le = LabelEncoder()
y_enc = le.fit_transform(labels)
svm_base = LinearSVC(dual=False, C=1.0, max_iter=3000)
svm = CalibratedClassifierCV(svm_base) 
svm.fit(X_vec, y_enc)
joblib.dump(vectorizer, 'local_vectorizer.pkl')
joblib.dump(svm, 'local_svm_model.pkl')
joblib.dump(le, 'local_label_encoder.pkl')
print("✅ SVM Saved.")

# --- 3. TRAIN CNN ---
print("\n[3/4] 🧠 Training CNN...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, char_level=True, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X_seq = pad_sequences(sequences, maxlen=MAX_LEN)
joblib.dump(tokenizer, 'local_tokenizer.pkl')

callbacks = [
    ModelCheckpoint('local_best_cnn.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001, verbose=1)
]

model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM, input_length=MAX_LEN),
    Conv1D(128, 5, activation='relu', padding='same'),
    BatchNormalization(), 
    Conv1D(64, 5, activation='relu'),
    GlobalMaxPooling1D(), 
    Dense(128, activation='relu'),
    Dropout(0.5), 
    Dense(len(le.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_seq, y_enc, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=callbacks)
print("✅ CNN Saved.")

# --- 4. TRAIN META-MODEL ---
print("\n[4/4] 🧠 Training Meta-Model...")
binary_labels = np.where(labels == 'Normal', 0, 1)
safe_idx = np.where(binary_labels == 0)[0]
mal_idx = np.where(binary_labels == 1)[0]

def make_ultimate_pairs(num_pairs):
    pairs_0, pairs_1, y_pairs = [], [], []
    for _ in range(num_pairs // 2):
        if len(mal_idx) > 0:
            idx1, idx2 = np.random.choice(mal_idx, 2)
            pairs_0.append(X_seq[idx1])
            pairs_1.append(X_seq[idx2])
            y_pairs.append(1) 
    for _ in range(num_pairs // 2):
        if len(mal_idx) > 0 and len(safe_idx) > 0:
            idx1 = np.random.choice(mal_idx)
            idx2 = np.random.choice(safe_idx)
            pairs_0.append(X_seq[idx1])
            pairs_1.append(X_seq[idx2])
            y_pairs.append(0) 
    return np.array(pairs_0), np.array(pairs_1), np.array(y_pairs)

X0, X1, Y_meta = make_ultimate_pairs(PAIRS_COUNT)
meta_callbacks = [
    ModelCheckpoint('local_meta_model.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

input_a = Input(shape=(MAX_LEN,))
input_b = Input(shape=(MAX_LEN,))
input_seq = Input(shape=(MAX_LEN,))
x = Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM)(input_seq)
x = Bidirectional(LSTM(64, return_sequences=False))(x) 
x = Dense(64, activation='relu')(x)
base_network = Model(input_seq, x)
vec_a = base_network(input_a)
vec_b = base_network(input_b)
dist = Lambda(lambda t: K.abs(t[0] - t[1]))([vec_a, vec_b])
out = Dense(1, activation='sigmoid')(dist)

meta_model = Model([input_a, input_b], out)
meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
meta_model.fit([X0, X1], Y_meta, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=meta_callbacks)

print("\n🎉 FLAWLESS TRAINING COMPLETE!")
