import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import re
import math
import urllib.parse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("📥 Loading models and preprocessors...")
vectorizer   = joblib.load('local_vectorizer.pkl')
svm          = joblib.load('local_svm_model.pkl')
le           = joblib.load('local_label_encoder.pkl')
url_scaler   = joblib.load('local_url_scaler.pkl')
tokenizer    = joblib.load('local_tokenizer.pkl')
cnn_model    = load_model('local_hybrid_model.keras', custom_objects={'Sequential': tf.keras.models.Sequential})
meta_model   = joblib.load('local_meta_learner_global.pkl')

MAX_LEN = 500

def strip_protocol(url):
    url = re.sub(r'^https?://', '', str(url))
    url = re.sub(r'^www\.', '', url)
    return url.rstrip('/')

def calculate_entropy(text):
    if not text: return 0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)

def extract_features(url):
    url = str(url).strip()
    decoded = urllib.parse.unquote(url)
    c = strip_protocol(url)
    RISK_TLDS = {'cfd','xyz','top','tk','gq','ml','ga','cf','pw','buzz'}
    BRAND_KEYWORDS = ['paypal','apple','microsoft','google','amazon','bank','secure','login']
    sqli     = len(re.findall(r"(?i)(union|select|insert|update|delete|drop|where|and|or|--|#|/\*|\*/)", decoded))
    xss      = len(re.findall(r"(?i)(<script|onerror|onload|alert|confirm|prompt|javascript:|eval\(|unescape\()", decoded))
    traversal = len(re.findall(r"\.\.\/|\.\.\\", decoded))
    encoded_dots   = url.lower().count('%2e%2e')
    sys_paths      = len(re.findall(r"(?i)(/etc/|/var/|/proc/|/windows/|/system32/|/bin/|/usr/)", decoded))
    special_chars  = sum(url.count(c2) for c2 in [';', "'", '"', '(', ')', '{', '}', '[', ']', '$', '*', '+', '`', '~', '|', '^'])
    path_depth     = url.count('/')
    encoded_pct    = len(re.findall(r'%[0-9a-fA-F]{2}', url))
    encoded_int    = encoded_pct / max(len(url), 1)
    wp_exploit     = int(any(p in decoded for p in ['/wp-content/plugins/', '/wp-admin/', '/wp-includes/']))
    feats = [
        len(url), len(c), url.count('.'), url.count('/'), url.count('-'), url.count('@'),
        url.count('?'), url.count('='), url.count('&'), int('//' in url[7:]),
        sum(ch.isdigit() for ch in url)/max(len(url),1),
        sum(not x.isalnum() and x not in '.-_/' for x in url)/max(len(url),1),
        int(c.split('/')[0].split('.')[-1].lower() in RISK_TLDS) if '.' in c.split('/')[0] else 0,
        max(c.split('/')[0].count('.')-1, 0),
        int(any(b in c.lower() for b in BRAND_KEYWORDS)),
        int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url))),
        int(bool(re.search(r':\d{2,5}/', url))),
        calculate_entropy(url),
        sqli, xss, traversal,
        encoded_dots, sys_paths, special_chars, path_depth,
        encoded_pct, encoded_int, wp_exploit
    ]
    return feats

def investigate_url(url):
    print(f"\n=======================================================")
    print(f"Investigating URL: {url}")
    print(f"=======================================================")
    
    decoded_url = urllib.parse.unquote(url)
    clean_url = strip_protocol(decoded_url)
    
    # 1. SVM
    X_vec = vectorizer.transform([clean_url])
    svm_proba = svm.predict_proba(X_vec)
    print(f"SVM Probabilities        : {svm_proba[0]} (classes: {svm.classes_})")
    
    # 2. CNN
    seq = tokenizer.texts_to_sequences([clean_url])
    X_sq = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    cnn_proba = cnn_model.predict(X_sq, batch_size=1, verbose=0).reshape(1, -1)
    print(f"CNN Probabilities        : {cnn_proba[0]}")
    
    # 3. Extracted Features
    X_hand = extract_features(url)
    X_hand_scaled = url_scaler.transform([X_hand])
    
    feature_names = [
        "len(url)", "len(clean_url)", "dots", "slashes", "hyphens", "ats",
        "question_marks", "equals", "amps", "double_slash", "digit_ratio",
        "special_char_ratio", "risk_tld", "subdomains", "brand_spoof", "has_ip",
        "has_port", "entropy", "sqli", "xss", "traversal", "encoded_dots",
        "sys_paths", "special_chars", "path_depth", "encoded_pct", "encoded_ratio",
        "wp_exploit"
    ]
    
    print("\nFeatures (Raw):")
    for name, val in zip(feature_names, X_hand):
        if val > 0:
            print(f"  - {name}: {val}")
    
    # 4. Meta Model (with SCALED features - this is what evaluate_mixed.py does)
    meta_features_scaled = np.hstack([svm_proba, cnn_proba, X_hand_scaled])
    all_probs_scaled = meta_model.predict_proba(meta_features_scaled)[0]
    normal_idx = list(le.classes_).index('Normal')
    malicious_proba_scaled = 1.0 - all_probs_scaled[normal_idx]
    
    print(f"\nMeta-Model Probabilities (Scaled) : {all_probs_scaled} (classes: {le.classes_})")
    print(f"Malicious Probability (Scaled)    : {malicious_proba_scaled:.4f}")

    # 5. Meta Model (with UNSCALED features - this is how it was trained)
    meta_features_raw = np.hstack([svm_proba, cnn_proba, [X_hand]])
    all_probs_raw = meta_model.predict_proba(meta_features_raw)[0]
    malicious_proba_raw = 1.0 - all_probs_raw[normal_idx]

    print(f"\nMeta-Model Probabilities (RAW)    : {all_probs_raw} (classes: {le.classes_})")
    print(f"Malicious Probability (RAW)       : {malicious_proba_raw:.4f}")

fp_urls = [
    "http://london-city-hotel.co.uk/html/londo/twins172.php",
    "http://listal.com/movie/being-john-malkovich",
    "http://rottentomatoes.com/celebrity/harry_stockwell/",
    "http://wldzb.cn/js/?ref=rxzjocnus.battle.net/d3/en/index",
    "https://google.com/",
    "http://example.com/"
]

for url in fp_urls:
    investigate_url(url)
