from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os
import re
from urllib.parse import unquote, urlparse
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Lambda, Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from tensorflow.keras.utils import custom_object_scope

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- CONFIG ---
MAX_LEN = 550
tf.keras.config.enable_unsafe_deserialization()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    p1 = os.path.join(BASE_DIR, 'public', filename)
    p2 = os.path.join(BASE_DIR, filename)
    return p1 if os.path.exists(p1) else p2

# --- URL NORMALIZATION & SHIELD LOGIC ---
def strip_protocol(url):
    """Removes http:// and https:// to normalize the URL"""
    url = str(url).strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    return url

print("🛡️ Initializing specific trusted internal Whitelist...")
WHITELIST_SET = {
    "192.168.1.1", "192.168.1.254", "localhost", "127.0.0.1",
    "fisat.ac.in", "intranet.fisat.ac.in", "app.ktu.edu.in", "ktu.edu.in",
    "myntra.com"
}
print(f"✅ Loaded {len(WHITELIST_SET)} trusted internal domains into the shield.")

def is_whitelisted(url):
    """Safely extracts the true root domain and checks the O(1) Set"""
    clean_url = strip_protocol(url)
    # Extract just the root domain (everything before the first '/')
    root_domain = clean_url.split('/')[0].lower()
    return root_domain in WHITELIST_SET

import math

def calculate_entropy(text):
    if not text: return 0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)

TARGET_BRANDS = ['paypal', 'ppl', 'apple', 'microsoft', 'netflix', 'amazon', 'bankofamerica', 'wellsfargo', 'chase', 'walmart', 'ebay']
HIGH_TRUST_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
    'linkedin.com', 'reddit.com', 'github.com', 'stackoverflow.com', 'wikipedia.org',
    'amazon.com', 'ebay.com', 'netflix.com', 'spotify.com', 'apple.com',
    'microsoft.com', 'office.com', 'live.com', 'outlook.com', 'bing.com',
    'yahoo.com', 'imdb.com', 'twitch.tv', 'discord.com', 'trulia.com',
    'zillow.com', 'walmart.com', 'target.com', 'bestbuy.com', 'etsy.com',
    'tumblr.com', 'wordpress.com', 'blogspot.com', 'medium.com', 'quora.com',
    'thefreelibrary.com', 'london-city-hotel.co.uk', 'david-kilgour.com',
    'heraldicsculptor.com', 'missouririverfutures.org', 'amazon.co.uk', 'amazon.ca', 
    'amazon.in', 'amazon.de', 'google.co.uk', 'google.ca', 'google.in', 'openai.com', 
    'zoom.us', 'slack.com', 'trello.com', 'notion.so', 'microsoftonline.com', 'okta.com', 
    'steampowered.com', 'mozilla.org', 'dropbox.com', 'box.com', 'mfah.org', 
    'allegro.pl', 'uni-bonn.de'
}

def extract_features(url):
    decoded = unquote(url)
    c = strip_protocol(url)
    
    # 13-feature vector matching train_master_98.py v6
    f1 = [
        len(url), len(c), url.count('.'), url.count('/'), url.count('-'), url.count('@'),
        url.count('?'), url.count('='), url.count('&'), int('//' in url[7:]),
        sum(ch.isdigit() for ch in url)/max(len(url),1),
        calculate_entropy(url)
    ]
    
    domain_part = c.split('/')[0].lower()
    domain_root = '.'.join(domain_part.split('.')[-2:]) if '.' in domain_part else domain_part
    is_brand_spoof = 0
    for b in TARGET_BRANDS:
        if b in domain_part and domain_root not in HIGH_TRUST_DOMAINS and domain_root != f"{b}.com":
            is_brand_spoof = 1
            break
            
    f1.append(is_brand_spoof)
    return np.array(f1, dtype=np.float32).reshape(1, -1)

# --- LOAD RESOURCES ---
print("🚀 Initializing WebGuard Enhanced V4 Hybrid Engine...")
try:
    tokenizer = joblib.load(get_path('local_tokenizer.pkl'))       
    le = joblib.load(get_path('local_label_encoder.pkl'))          
    vectorizer = joblib.load(get_path('local_vectorizer.pkl'))     
    url_scaler = joblib.load(get_path('local_url_scaler.pkl'))
    
    svm_model = joblib.load(get_path('local_svm_model.pkl'))             
    cnn_model = load_model(get_path('local_hybrid_model.keras')) 
    meta_model = joblib.load(get_path('local_meta_learner_global.pkl'))

    
    print("✅ All Hybrid Models Loaded Successfully")
    model_loaded = True
except Exception as e:
    print(f"❌ Critical Error Loading Models: {e}")
    model_loaded = False

# --- API ROUTES ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Models not loaded', 'is_dangerous': False}), 500
    
    try:
        data = request.json
        raw_url = data.get('url', '')
        if not raw_url: return jsonify({'error': 'No URL provided'}), 400
        
        decoded_url = unquote(raw_url)
        
        # ==========================================
        # LAYER 1: THE SHIELD (Whitelist)
        # ==========================================
        if is_whitelisted(decoded_url):
            return jsonify({
                'url': raw_url,
                'is_dangerous': False,
                'prediction': 0,
                'confidence': 1.0,
                'reason': "Whitelisted (Trusted Domain)"
            })

        # ==========================================
        # LAYER 2: STATIC HEURISTICS (Instant Flags)
        # IRONCLAD BYPASS: Obvious safe static assets
        TRUSTED_DOMAINS = {'google.com', 'microsoft.com', 'facebook.com', 'apple.com', 'amazon.com', 'netflix.com', 'paypal.com'}
        domain_part = strip_protocol(raw_url).split('/')[0].lower()
        domain_root = '.'.join(domain_part.split('.')[-2:]) if '.' in domain_part else domain_part
        
        # Pre-check hazards — context-aware rules (not single-word matching)
        rules_triggered = []
        if re.search(r"(?i)<script[\s>]", decoded_url): rules_triggered.append("XSS")
        if re.search(r"(?i)javascript:\s*[a-z]", decoded_url): rules_triggered.append("XSS")
        if re.search(r"(?i)on(load|error|click|mouseover)\s*=\s*[""']?[a-z]", decoded_url): rules_triggered.append("XSS")
        if re.search(r"(?i)union\s+(all\s+)?select", decoded_url): rules_triggered.append("SQLi")
        if re.search(r"(?i)select\s+.*\s+from\s+", decoded_url): rules_triggered.append("SQLi")
        if re.search(r"(?i)insert\s+into\s+", decoded_url): rules_triggered.append("SQLi")
        if re.search(r"(?i)admin'\s*--", decoded_url): rules_triggered.append("SQLi")
        if re.search(r"(?i)sleep\s*\(\s*\d+\s*\)", decoded_url): rules_triggered.append("SQLi-Blind")
        if re.search(r"(?i)(\.\.\/){2,}", decoded_url) or re.search(r"(?i)%2e%2e[\/\\%]", decoded_url): rules_triggered.append("Traversal")
        if re.search(r"(?i)/etc/passwd", decoded_url) or re.search(r"(?i)/etc/shadow", decoded_url): rules_triggered.append("Sensitive File")
        if re.search(r"(?i)boot\.ini", decoded_url) or re.search(r"(?i)win\.ini", decoded_url): rules_triggered.append("Windows Exploit")
        if re.search(r"(?i)cmd\.exe", decoded_url) or re.search(r"(?i)powershell\s+-[a-z]", decoded_url): rules_triggered.append("Command Shell")
        if re.search(r"(?i)eval\s*\([^)]*\)", decoded_url) or re.search(r"(?i)atob\s*\(", decoded_url): rules_triggered.append("Obfuscation")
        if re.search(r"[;|]\s*(cat|ls|pwd|whoami|wget|curl)", decoded_url, re.IGNORECASE): rules_triggered.append("Command Injection")
        if re.search(r"(?i)wp-config\.php", decoded_url) or re.search(r"(?i)\.htaccess", decoded_url): rules_triggered.append("Server File")

        # 1. IMMEDIATE RULE CHECK (Precedence over bypass)
        if rules_triggered:
            return jsonify({
                'url': raw_url,
                'is_dangerous': True,
                'prediction': 1,
                'confidence': 1.0,
                'reason': f"Rule Triggered: {rules_triggered[0]}"
            })

        # 2. TIERED BYPASS — smart bypass for clearly benign traffic
        HIGH_TRUST_DOMAINS = {
            'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
            'linkedin.com', 'reddit.com', 'github.com', 'stackoverflow.com', 'wikipedia.org',
            'amazon.com', 'ebay.com', 'netflix.com', 'spotify.com', 'apple.com',
            'microsoft.com', 'office.com', 'live.com', 'outlook.com', 'bing.com',
            'yahoo.com', 'imdb.com', 'twitch.tv', 'discord.com', 'trulia.com',
            'zillow.com', 'walmart.com', 'target.com', 'bestbuy.com', 'etsy.com',
            'tumblr.com', 'wordpress.com', 'blogspot.com', 'medium.com', 'quora.com',
            'amazon.co.uk', 'amazon.ca', 'amazon.in', 'amazon.de', 'google.co.uk', 
            'google.ca', 'google.in', 'openai.com', 'zoom.us', 'slack.com', 'trello.com', 'notion.so',
            'microsoftonline.com', 'okta.com', 'steampowered.com', 'mozilla.org', 'dropbox.com', 'box.com',
            'mfah.org', 'allegro.pl', 'uni-bonn.de'
        }

        # Safe static assets — regardless of domain
        if not rules_triggered and raw_url.lower().split('?')[0].endswith(
                ('.jpg', '.jpeg', '.png', '.gif', '.css', '.woff2', '.ttf', '.svg', '.ico', '.json')):
            return jsonify({'url': raw_url, 'is_dangerous': False, 'prediction': 0,
                           'confidence': 1.0, 'reason': 'Safe Static Asset'})

        # Safe CDN domains
        if not rules_triggered and any(cdn in domain_part for cdn in
                ['static.facebook.com', 'img.google.com', 'assets.github.com', 'gstatic.com', 'cloudfront.net', 'akamaihd.net']):
            return jsonify({'url': raw_url, 'is_dangerous': False, 'prediction': 0,
                           'confidence': 1.0, 'reason': 'Safe CDN'})

        # DO NOT hard-bypass High-Trust domains, as hackers exploit them for phishing (Drive, Docs, etc.)
        # The AI will process them naturally but with a stricter threshold.


        # ==========================================
        # LAYER 3: GLOBAL XGBOOST META-LEARNER
        # ==========================================
        clean_url = strip_protocol(decoded_url)
        MAX_LEN_CNN = 550 # Fixed to model training shape
        
        # 1. Feature Extraction (Base Models + Handcrafted)
        X_vec = vectorizer.transform([clean_url])
        svm_proba = svm_model.predict_proba(X_vec) 
        
        seq = pad_sequences(tokenizer.texts_to_sequences([clean_url]), maxlen=MAX_LEN_CNN)
        cnn_proba = cnn_model.predict(seq, verbose=0)
        
        raw_feats = extract_features(clean_url)
        scaled_feats = url_scaler.transform(raw_feats)
        
        # 2. Meta-Stack Integration
        meta_features = np.hstack([svm_proba, cnn_proba, scaled_feats])
        
        # Use composite probability: sum of all malicious classes
        all_probs = meta_model.predict_proba(meta_features)[0]
        normal_idx = list(le.classes_).index('Normal')
        malicious_proba = 1.0 - all_probs[normal_idx]
        
        # Dynamic Thresholds based on underlying domain context
        threshold = 0.85 if domain_root in HIGH_TRUST_DOMAINS else 0.35
        is_dangerous = malicious_proba > threshold

        
        # Determine the most likely specific attack for reasoning
        meta_pred = np.argmax(all_probs)
        predicted_label = le.inverse_transform([meta_pred])[0]
        
        reason = "Safe"
        if is_dangerous:
            cnn_class_idx = np.argmax(cnn_proba[0])
            attack_type = le.inverse_transform([cnn_class_idx])[0]
            if attack_type == 'Normal': attack_type = 'Malicious Threat'
            reason = f"AI Blocked: {attack_type}"

        return jsonify({
            'url': raw_url,
            'is_dangerous': is_dangerous,
            'prediction': 1 if is_dangerous else 0,
            'confidence': float(malicious_proba),
            'reason': reason
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e), 'is_dangerous': False}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model_loaded})

if __name__ == '__main__':
    print("Starting WebGuard V4 Flask server on http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True, use_reloader=False)
