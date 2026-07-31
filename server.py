import math
import os
import re
from urllib.parse import unquote
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, 'models', filename)


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
    "myntra.com", "docs.google.com", "drive.google.com", "script.google.com",
    "forms.gle", "gemini.google.com", "whatsapp.com", "web.whatsapp.com"
}
print(f"✅ Loaded {len(WHITELIST_SET)} trusted internal domains into the shield.")

def is_whitelisted(url):
    clean_url = strip_protocol(url)
    root_domain = clean_url.split('/')[0].lower()
    safe_tlds = (
        '.gov.in', '.nic.in', '.edu.in', '.ac.in', '.res.in',
        '.gov', '.edu', '.mil', '.int',
        '.bank', '.creditunion',
        '.in', '.co.in', '.club',
    )
    if root_domain.endswith(safe_tlds):
        return True
    return root_domain in WHITELIST_SET


# --- RANDOM FOREST FEATURE EXTRACTION ---
def calculate_entropy(s):
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
    features['entropy'] = df['url'].apply(calculate_entropy)
    features['has_ip'] = df['url'].apply(lambda x: 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', x) else 0)
    
    keywords = ['login', 'admin', 'secure', 'account', 'update', 'banking', 'confirm', 'verify', 'free', 'bonus']
    for kw in keywords:
        features[f'has_{kw}'] = df['url'].apply(lambda x: 1 if kw in x.lower() else 0)
        
    return features


# --- LOAD RESOURCES ---
print("🚀 Initializing WebGuard RF Engine (98% Accuracy)...")
try:
    vectorizer = joblib.load(get_path('rf_vectorizer.pkl'))
    rf_model = joblib.load(get_path('rf_model.pkl'))
    print("✅ Random Forest Models Loaded Successfully")
    model_loaded = True
except Exception as e:
    import traceback
    traceback.print_exc()
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
        if not raw_url:
            return jsonify({'error': 'No URL provided'}), 400

        decoded_url = unquote(raw_url)

        # ==========================================
        # LAYER 1: THE SHIELD (Whitelist & Internal)
        # ==========================================
        if is_whitelisted(decoded_url) or raw_url.startswith(
                ('chrome-extension://', 'chrome://', 'about:', 'edge://', 'moz-extension://')):
            return jsonify({
                'url': raw_url,
                'is_dangerous': False,
                'prediction': 0,
                'confidence': 1.0,
                'reason': "Whitelisted (Trusted/Internal)"
            })

        # ==========================================
        # LAYER 2: STATIC HEURISTICS (Instant Flags)
        # ==========================================
        domain_part = strip_protocol(raw_url).split('/')[0].lower()
        domain_root = '.'.join(domain_part.split('.')[-2:]) if '.' in domain_part else domain_part
        rules_triggered = []
        
        if re.search(r"(?i)<script[\s>]", decoded_url): rules_triggered.append("XSS")
        if re.search(r"(?i)javascript:\s*[a-z]", decoded_url): rules_triggered.append("XSS")
        if re.search(r"(?i)union\s+(all\s+)?select", decoded_url): rules_triggered.append("SQLi")
        if re.search(r"(?i)select\s+.*\s+from\s+", decoded_url): rules_triggered.append("SQLi")
        if re.search(r"(?i)insert\s+into\s+", decoded_url): rules_triggered.append("SQLi")
        if re.search(r"(?i)(\.\.\/){2,}", decoded_url): rules_triggered.append("Traversal")
        if re.search(r"(?i)/etc/passwd", decoded_url): rules_triggered.append("Sensitive File")
        if re.search(r"(?i)cmd\.exe", decoded_url): rules_triggered.append("Command Shell")
        if re.search(r"[;|]\s*(cat|ls|pwd|whoami|wget|curl)", decoded_url, re.IGNORECASE): rules_triggered.append("Command Injection")
        if re.search(r"(?i)wp-config\.php", decoded_url): rules_triggered.append("Server File")

        # Subdomain brand-spoof check
        _TRUSTED_BRAND_DOMAINS = {
            'paypal.com', 'google.com', 'apple.com', 'microsoft.com',
            'amazon.com', 'facebook.com', 'netflix.com', 'bankofamerica.com',
            'wellsfargo.com', 'chase.com', 'instagram.com', 'twitter.com',
        }
        for trusted in _TRUSTED_BRAND_DOMAINS:
            if trusted in domain_part and domain_root != trusted:
                rules_triggered.append("Brand Subdomain Spoof")
                break

        if rules_triggered:
            return jsonify({
                'url': raw_url,
                'is_dangerous': True,
                'prediction': 1,
                'confidence': 1.0,
                'reason': f"Rule Triggered: {rules_triggered[0]}"
            })

        # Bypass for safe static assets
        if not rules_triggered and raw_url.lower().split('?')[0].endswith(
                ('.jpg', '.jpeg', '.png', '.gif', '.css',
                 '.woff2', '.ttf', '.svg', '.ico', '.json')):
            return jsonify({'url': raw_url, 'is_dangerous': False, 'prediction': 0, 'confidence': 1.0, 'reason': 'Safe Static Asset'})

        if not rules_triggered and any(cdn in domain_part for cdn in [
                'static.facebook.com', 'img.google.com', 'assets.github.com',
                'gstatic.com', 'cloudfront.net', 'akamaihd.net']):
            return jsonify({'url': raw_url, 'is_dangerous': False, 'prediction': 0, 'confidence': 1.0, 'reason': 'Safe CDN'})


        # ==========================================
        # LAYER 3: RANDOM FOREST ENGINE
        # ==========================================
        df_url = pd.DataFrame([{'url': decoded_url}])
        X_feat = extract_features(df_url)
        X_tfidf = vectorizer.transform(df_url['url']).toarray()
        X = np.hstack((X_feat.values, X_tfidf))
        
        # Get Probabilities
        probs = rf_model.predict_proba(X)[0]
        # Classes: 0 (Normal), 1 (Phishing)
        malicious_proba = float(probs[1])
        
        HIGH_TRUST_DOMAINS = {
            'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
            'github.com', 'wikipedia.org', 'amazon.com', 'netflix.com', 'apple.com', 'microsoft.com',
            'paypal.com', 'chase.com', 'wellsfargo.com', 'bankofamerica.com', 'ebay.com'
        }
        threshold = 0.95 if domain_root in HIGH_TRUST_DOMAINS else 0.50
        
        # Heuristic Leniency for simple domains
        if domain_root not in HIGH_TRUST_DOMAINS and len(domain_root) < 15 and domain_part == domain_root and not rules_triggered:
            threshold = 0.85
            
        is_dangerous = bool(malicious_proba >= threshold)

        reason = "Safe"
        if is_dangerous:
            reason = "AI Blocked: Malicious Threat"

        return jsonify({
            'url': raw_url,
            'is_dangerous': is_dangerous,
            'prediction': 1 if is_dangerous else 0,
            'confidence': malicious_proba,
            'reason': reason
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e), 'is_dangerous': False}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model_loaded})


if __name__ == '__main__':
    print("Starting WebGuard V4 (RF-Powered) Flask server on http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True, use_reloader=False)
