import math
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import re
from urllib.parse import unquote
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- CONFIG ---
MAX_LEN = 550
tf.keras.config.enable_unsafe_deserialization()
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
    "myntra.com"
}
print(
    f"✅ Loaded {
        len(WHITELIST_SET)} trusted internal domains into the shield.")


def is_whitelisted(url):
    """Safely extracts the true root domain and checks the O(1) Set"""
    clean_url = strip_protocol(url)
    # Extract just the root domain (everything before the first '/')
    root_domain = clean_url.split('/')[0].lower()
    # High-trust structural, regional, and government TLDs that are historically very safe
    # This prevents local startups, Indian sites, and government pages from getting caught
    # in the AI's length penalization or structure checks.
    safe_tlds = (
        '.gov.in', '.nic.in', '.edu.in', '.ac.in', '.res.in',
        '.gov', '.edu', '.mil', '.int',
        '.bank', '.creditunion',
        '.in', '.co.in', '.club',
    )

    if root_domain.endswith(safe_tlds):
        return True
    return root_domain in WHITELIST_SET


def calculate_entropy(text):
    if not text:
        return 0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)


TARGET_BRANDS = [
    'paypal', 'ppl', 'apple', 'microsoft', 'netflix',
    'amazon', 'bankofamerica', 'wellsfargo', 'chase',
    'walmart', 'ebay', 'google', 'facebook', 'youtube',
    'instagram', 'twitter', 'linkedin', 'dropbox', 'spotify']
HIGH_TRUST_DOMAINS = {
    'google.com',
    'youtube.com',
    'facebook.com',
    'twitter.com',
    'instagram.com',
    'linkedin.com',
    'reddit.com',
    'github.com',
    'stackoverflow.com',
    'wikipedia.org',
    'amazon.com',
    'ebay.com',
    'netflix.com',
    'spotify.com',
    'apple.com',
    'microsoft.com',
    'office.com',
    'live.com',
    'outlook.com',
    'bing.com',
    'yahoo.com',
    'imdb.com',
    'twitch.tv',
    'discord.com',
    'trulia.com',
    'zillow.com',
    'walmart.com',
    'target.com',
    'bestbuy.com',
    'etsy.com',
    'tumblr.com',
    'wordpress.com',
    'blogspot.com',
    'medium.com',
    'quora.com',
    'thefreelibrary.com',
    'london-city-hotel.co.uk',
    'david-kilgour.com',
    'heraldicsculptor.com',
    'missouririverfutures.org',
    'amazon.co.uk',
    'amazon.ca',
    'amazon.in',
    'amazon.de',
    'google.co.uk',
    'google.ca',
    'google.in',
    'openai.com',
    'zoom.us',
    'slack.com',
    'trello.com',
    'notion.so',
    'microsoftonline.com',
    'okta.com',
    'steampowered.com',
    'mozilla.org',
    'dropbox.com',
    'box.com',
    'mfah.org',
    'allegro.pl',
    'uni-bonn.de'}


def extract_features(url):
    c = strip_protocol(url)

    # 13-feature vector matching train_master_98.py v6
    f1 = [len(url),
          len(c),
          url.count('.'),
          url.count('/'),
          url.count('-'),
          url.count('@'),
          url.count('?'),
          url.count('='),
          url.count('&'),
          int('//' in url[7:]),
          sum(ch.isdigit() for ch in url) / max(len(url),
                                                1),
          calculate_entropy(url)]

    domain_part = c.split('/')[0].lower()
    domain_root = '.'.join(domain_part.split(
        '.')[-2:]) if '.' in domain_part else domain_part
    # Domain stem = just the part before the TLD (e.g. "paypa1" from "paypa1.com")
    domain_stem = domain_root.split('.')[0] if '.' in domain_root else domain_root

    # Normalize digit-substitutions (0→o, 1→i/l, 3→e, 4→a) for fuzzy matching
    _DIGIT_MAP = str.maketrans('013458', 'oieash')

    def _normalize(s):
        return s.translate(_DIGIT_MAP)

    def _edit_dist(a, b):
        """Fast Levenshtein distance for short strings."""
        if abs(len(a) - len(b)) > 2:
            return 99
        dp = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            ndp = [i + 1]
            for j, cb in enumerate(b):
                ndp.append(min(dp[j] + (ca != cb), dp[j + 1] + 1, ndp[-1] + 1))
            dp = ndp
        return dp[-1]

    is_brand_spoof = 0
    # Exact substring check (original logic)
    for b in TARGET_BRANDS:
        if (b in domain_part and domain_root not in HIGH_TRUST_DOMAINS
                and domain_root != f"{b}.com"):
            is_brand_spoof = 1
            break

    # Subdomain brand spoof: catch paypal.com.phishing-login.ru style attacks
    # If a trusted brand name appears IN the subdomains (not the root), it's spoofed.
    if not is_brand_spoof:
        subdomains = domain_part.replace(domain_root, '').rstrip('.')
        for b in TARGET_BRANDS:
            if b in subdomains:
                is_brand_spoof = 1
                break

    # Fuzzy check with digit normalization: catch g00gle, bank0famerica, paypa1
    if not is_brand_spoof and domain_root not in HIGH_TRUST_DOMAINS:
        norm_stem = _normalize(domain_stem)
        for b in TARGET_BRANDS:
            if len(b) >= 5 and (_edit_dist(norm_stem, b) <= 2
                                or _edit_dist(domain_stem, b) <= 2):
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
        return jsonify({'error': 'Models not loaded',
                       'is_dangerous': False}), 500

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
        # IRONCLAD BYPASS: Obvious safe static assets
        domain_part = strip_protocol(raw_url).split('/')[0].lower()
        domain_root = '.'.join(domain_part.split(
            '.')[-2:]) if '.' in domain_part else domain_part

        # Pre-check hazards — context-aware rules (not single-word matching)
        rules_triggered = []
        if re.search(r"(?i)<script[\s>]", decoded_url):
            rules_triggered.append("XSS")
        if re.search(r"(?i)javascript:\s*[a-z]", decoded_url):
            rules_triggered.append("XSS")
        if re.search(
            r"(?i)on(load|error|click|mouseover)\s*=\s*["
            "']?[a-z]",
                decoded_url):
            rules_triggered.append("XSS")
        if re.search(r"(?i)union\s+(all\s+)?select", decoded_url):
            rules_triggered.append("SQLi")
        if re.search(r"(?i)select\s+.*\s+from\s+", decoded_url):
            rules_triggered.append("SQLi")
        if re.search(r"(?i)insert\s+into\s+", decoded_url):
            rules_triggered.append("SQLi")
        if re.search(r"(?i)admin'\s*--", decoded_url):
            rules_triggered.append("SQLi")
        if re.search(r"(?i)sleep\s*\(\s*\d+\s*\)", decoded_url):
            rules_triggered.append("SQLi-Blind")
        if re.search(
                r"(?i)(\.\.\/){2,}",
                decoded_url) or re.search(
                r"(?i)%2e%2e[\/\\%]",
                decoded_url):
            rules_triggered.append("Traversal")
        if re.search(
                r"(?i)/etc/passwd",
                decoded_url) or re.search(
                r"(?i)/etc/shadow",
                decoded_url):
            rules_triggered.append("Sensitive File")
        if re.search(
                r"(?i)boot\.ini",
                decoded_url) or re.search(
                r"(?i)win\.ini",
                decoded_url):
            rules_triggered.append("Windows Exploit")
        if re.search(
                r"(?i)cmd\.exe",
                decoded_url) or re.search(
                r"(?i)powershell\s+-[a-z]",
                decoded_url):
            rules_triggered.append("Command Shell")
        if re.search(
                r"(?i)eval\s*\([^)]*\)",
                decoded_url) or re.search(
                r"(?i)atob\s*\(",
                decoded_url):
            rules_triggered.append("Obfuscation")
        if re.search(
            r"[;|]\s*(cat|ls|pwd|whoami|wget|curl)",
            decoded_url,
                re.IGNORECASE):
            rules_triggered.append("Command Injection")
        if re.search(
                r"(?i)wp-config\.php",
                decoded_url) or re.search(
                r"(?i)\.htaccess",
                decoded_url):
            rules_triggered.append("Server File")

        # Subdomain brand-spoof: "paypal.com.evil.ru" — brand appears as a subdomain
        # of an untrusted root. Highly reliable phishing signal.
        _TRUSTED_BRAND_DOMAINS = {
            'paypal.com', 'google.com', 'apple.com', 'microsoft.com',
            'amazon.com', 'facebook.com', 'netflix.com', 'bankofamerica.com',
            'wellsfargo.com', 'chase.com', 'instagram.com', 'twitter.com',
        }
        for trusted in _TRUSTED_BRAND_DOMAINS:
            if trusted in domain_part and domain_root != trusted:
                rules_triggered.append("Brand Subdomain Spoof")
                break

        # Digit-substitution brand spoof: "bank0famerica" → "bankofamerica"
        _DS = str.maketrans('013458', 'oieash')
        norm_domain = domain_part.translate(_DS)
        _SAFE_ROOTS = {'google.com', 'paypal.com', 'apple.com', 'microsoft.com',
                       'amazon.com', 'netflix.com', 'facebook.com', 'bankofamerica.com'}
        for b in ['paypal', 'google', 'apple', 'microsoft', 'netflix',
                  'amazon', 'bankofamerica', 'wellsfargo', 'facebook']:
            if b in norm_domain and domain_root not in _SAFE_ROOTS:
                rules_triggered.append("Brand Digit Spoof")
                break

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
            'google.com',
            'youtube.com',
            'facebook.com',
            'twitter.com',
            'instagram.com',
            'linkedin.com',
            'reddit.com',
            'github.com',
            'stackoverflow.com',
            'wikipedia.org',
            'amazon.com',
            'ebay.com',
            'netflix.com',
            'spotify.com',
            'apple.com',
            'microsoft.com',
            'office.com',
            'live.com',
            'outlook.com',
            'bing.com',
            'yahoo.com',
            'imdb.com',
            'twitch.tv',
            'discord.com',
            'trulia.com',
            'zillow.com',
            'walmart.com',
            'target.com',
            'bestbuy.com',
            'etsy.com',
            'tumblr.com',
            'wordpress.com',
            'blogspot.com',
            'medium.com',
            'quora.com',
            'amazon.co.uk',
            'amazon.ca',
            'amazon.in',
            'amazon.de',
            'google.co.uk',
            'google.ca',
            'google.in',
            'openai.com',
            'zoom.us',
            'slack.com',
            'trello.com',
            'notion.so',
            'microsoftonline.com',
            'okta.com',
            'steampowered.com',
            'mozilla.org',
            'dropbox.com',
            'box.com',
            'mfah.org',
            'allegro.pl',
            'uni-bonn.de'}

        # Safe static assets — regardless of domain
        if not rules_triggered and raw_url.lower().split('?')[0].endswith(
                ('.jpg', '.jpeg', '.png', '.gif', '.css',
                 '.woff2', '.ttf', '.svg', '.ico', '.json')):
            return jsonify({'url': raw_url,
                            'is_dangerous': False,
                            'prediction': 0,
                            'confidence': 1.0,
                            'reason': 'Safe Static Asset'})

        # Safe CDN domains
        if not rules_triggered and any(
            cdn in domain_part for cdn in [
                'static.facebook.com',
                'img.google.com',
                'assets.github.com',
                'gstatic.com',
                'cloudfront.net',
                'akamaihd.net']):
            return jsonify({'url': raw_url,
                            'is_dangerous': False,
                            'prediction': 0,
                            'confidence': 1.0,
                            'reason': 'Safe CDN'})

        # DO NOT hard-bypass High-Trust domains, as hackers exploit them
        # for phishing (Drive, Docs, etc.)
        # The AI will process them naturally but with a stricter threshold.

        # ==========================================
        # LAYER 3: GLOBAL XGBOOST META-LEARNER
        # ==========================================
        clean_url = strip_protocol(decoded_url)
        MAX_LEN_CNN = 550  # Fixed to model training shape

        # 1. Feature Extraction (Base Models + Handcrafted)
        X_vec = vectorizer.transform([clean_url])
        svm_proba = svm_model.predict_proba(X_vec)

        seq = pad_sequences(
            tokenizer.texts_to_sequences(
                [clean_url]),
            maxlen=MAX_LEN_CNN)
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
        # Unknown/new domains need 65% confidence to block to prevent false positives
        threshold = 0.85 if domain_root in HIGH_TRUST_DOMAINS else 0.65

        # Check for brand spoofing BEFORE applying the short-domain leniency.
        # This ensures typosquats (paypa1.com, g00gle.com) never get a free pass.
        raw_feats_check = extract_features(clean_url)
        is_brand_spoof_detected = bool(raw_feats_check[0][-1] == 1.0)

        # Heuristic: Short, simple domains (startups/services: fast.com, haicabs.com)
        # with no subdomains, no static rule triggers, and NO brand spoofing detected
        # are very likely safe. Require 99% AI confidence before blocking them.
        if (len(domain_root) < 15 and domain_part == domain_root
                and not rules_triggered and not is_brand_spoof_detected):
            threshold = 0.99
            
        is_dangerous = bool(malicious_proba > threshold)

        # Determine the most likely specific attack for reasoning

        reason = "Safe"
        if is_dangerous:
            cnn_class_idx = np.argmax(cnn_proba[0])
            attack_type = le.inverse_transform([cnn_class_idx])[0]
            if attack_type == 'Normal':
                attack_type = 'Malicious Threat'
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
