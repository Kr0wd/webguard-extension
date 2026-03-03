from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os
import re
from urllib.parse import unquote, urlparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import scipy.sparse as sp
import whois
import datetime
import warnings

warnings.filterwarnings('ignore')
tf.keras.config.enable_unsafe_deserialization()

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- CONFIG ---
MAX_LEN = 200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Feature Extraction Sets
RISK_TLDS = {
    'cfd','xyz','top','tk','gq','ml','ga','cf','pw','buzz','icu','shop',
    'online','site','website','store','fun','link','click','ink','live',
    'vip','win','bid','party','date','review','trade','loan','work','men',
}
BRAND_KEYWORDS = [
    'paypal','amazon','apple','google','microsoft','netflix','bank',
    'secure','login','signin','account','verify','update','ebay','chase',
    'citibank','wellsfargo','coinbase','binance','wallet',
]

def get_path(filename):
    p1 = os.path.join(BASE_DIR, 'public', filename)
    p2 = os.path.join(BASE_DIR, filename)
    return p1 if os.path.exists(p1) else p2

# --- URL NORMALIZATION & SHIELD LOGIC ---
def strip_protocol(url):
    """Removes http:// and https:// and trailing slashes to normalize the URL"""
    url = str(url).strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    if url.endswith('/'):
        url = url[:-1]
    return url

print("🛡️ Loading Majestic Top 1 Million Whitelist into memory...")
try:
    df_benign = pd.read_csv(get_path("majestic_million.csv"))
    # O(1) Hash Map for instant lookups
    WHITELIST_SET = set(df_benign['Domain'].dropna().str.lower().tolist())
    
    # Add safe local IPs and CUSTOM INTERNAL DOMAINS
    WHITELIST_SET.update([
        "192.168.1.1", "192.168.1.254", "localhost", "127.0.0.1",
        "fisat.ac.in", "intranet.fisat.ac.in", "app.ktu.edu.in", "ktu.edu.in",
        "edistrict.kerala.gov.in", "kerala.gov.in", "ktu.edu.in"
    ])
    print(f"✅ Loaded {len(WHITELIST_SET)} trusted domains into the shield.")
except Exception as e:
    print(f"⚠️ Could not load CSV, falling back to basic whitelist: {e}")
    WHITELIST_SET = {"google.com", "amazon.com", "github.com", "myntra.com", "fisat.ac.in", "intranet.fisat.ac.in"}

def is_whitelisted(url):
    """Safely extracts the true root domain and checks the O(1) Set"""
    clean_url = strip_protocol(url)
    # Extract just the root domain (everything before the first '/')
    root_domain = clean_url.split('/')[0].lower()
    return root_domain in WHITELIST_SET

def url_handcrafted_features(urls):
    feats = []
    for url in urls:
        url   = str(url)
        clean = strip_protocol(url)
        url_len      = len(url)
        path_len     = len(clean)
        dot_count    = url.count('.')
        slash_count  = url.count('/')
        hyphen_count = url.count('-')
        at_count     = url.count('@')
        q_count      = url.count('?')
        eq_count     = url.count('=')
        amp_count    = url.count('&')
        double_slash = int('//' in url[7:] if len(url) > 7 else '//' in url)
        digits       = sum(c.isdigit() for c in url)
        digit_ratio  = digits / max(url_len, 1)
        special      = sum(not c.isalnum() and c not in '.-_/' for c in url)
        special_ratio= special / max(url_len, 1)
        tld          = clean.split('/')[0].split('.')[-1].lower() if '.' in clean.split('/')[0] else ''
        risky_tld    = int(tld in RISK_TLDS)
        domain_part  = clean.split('/')[0]
        subdomain    = max(domain_part.count('.') - 1, 0)
        brand_hit    = int(any(bk in clean.lower() for bk in BRAND_KEYWORDS))
        ip_in_url    = int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url)))
        port_in_url  = int(bool(re.search(r':\d{2,5}/', url)))
        hex_enc      = int(bool(re.search(r'%[0-9a-fA-F]{2}', url)))
        feats.append([url_len, path_len, dot_count, slash_count, hyphen_count,
                      at_count, q_count, eq_count, amp_count, double_slash,
                      digit_ratio, special_ratio, risky_tld, subdomain,
                      brand_hit, ip_in_url, port_in_url, hex_enc])
    return np.array(feats, dtype=np.float32)

def get_domain_age_risk(url):
    try:
        domain = urlparse(url if "://" in url else "http://" + url).netloc
        domain_info = whois.whois(domain)
        creation_date = domain_info.creation_date
        
        # Handle cases where whois returns a list of dates
        if type(creation_date) is list:
            creation_date = creation_date[0]
            
        if creation_date:
            age_days = (datetime.datetime.now() - creation_date).days
            if age_days < 30:
                print("🚨 HIGH RISK: Domain registered less than 30 days ago!")
                return 0.30  # Add 30% to the final Ensemble Voting score
            elif age_days < 90:
                print("⚠️ MEDIUM RISK: Domain registered less than 90 days ago!")
                return 0.10  # Add 10% risk
    except Exception as e:
        pass # WHOIS failed or domain hidden
    return 0.0 # No penalty if old or unknown

# --- LOAD RESOURCES ---
print("🚀 Initializing WebGuard Enhanced V4 Hybrid Engine...")
try:
    tokenizer = joblib.load(get_path('local_tokenizer.pkl'))       
    le = joblib.load(get_path('local_label_encoder.pkl'))          
    vectorizer = joblib.load(get_path('local_vectorizer.pkl'))     
    scaler = joblib.load(get_path('local_url_scaler.pkl'))
    
    svm = joblib.load(get_path('local_svm_model.pkl'))             
    hybrid_cnn = load_model(get_path('local_hybrid_model.keras'), safe_mode=False) 
    meta_learner = joblib.load(get_path('local_meta_learner_global.pkl'))
    
    print("✅ All Hybrid Models & Preprocessors Loaded Successfully")
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
        clean_url = strip_protocol(decoded_url)
        
        # Layer 1: The Shield (Whitelist)
        domain_part = clean_url.split('/')[0].lower()
        
        # Patch: Only whitelist if it is the ROOT domain. Deep paths must be scanned
        # because legitimate domains are frequently compromised to host phishing pages.
        if is_whitelisted(decoded_url) and clean_url == domain_part:
            return jsonify({
                'url': raw_url,
                'is_dangerous': False,
                'prediction': 0,
                'confidence': 1.0,
                'reason': "Whitelisted (Trusted Root Domain)"
            })
            
        # Layer 1.5: Safe Path / CDN Heuristics (False Positive Reduction)
        safe_bypass = False
        fast_platforms = ['blogspot.com', 'wordpress.com', 'weebly.com', 'livejournal.com', 'typepad.com']
        safe_domains = ['bit.ly', 'tinyurl.com', 'lnk.co', 'blogspot.', 'plus.google.com', 'sendgrid.org', 'shnap.com', '16mb.com']
        
        if any(plat in domain_part for plat in fast_platforms + safe_domains):
            if not re.search(r'login|verify|account|signin|paypal|apple|secure|auth|billing|update', decoded_url, re.IGNORECASE):
                safe_bypass = True

        if not safe_bypass and any(decoded_url.endswith(ext) for ext in ['.jpg','.png','.gif','.css','.js','.pdf','.xml']):
            safe_bypass = True

        if not safe_bypass and re.search(r'(cdn|translate|maps|img|static|assets)\.', domain_part, re.IGNORECASE):
            safe_bypass = True
            
        if not safe_bypass:
            if 'google.com' in domain_part and 'forms' in clean_url: safe_bypass = True
            if 'google.com' in domain_part and 'group' in clean_url: safe_bypass = True
            if 'microsoft.com' in domain_part and 'kb' in clean_url: safe_bypass = True
            if '.mil/' in clean_url or '.gov/' in clean_url: safe_bypass = True
            
        # Layer 3: Feature Extraction (Pre-computed for Safe Bypass overrides)
        X_tfidf    = vectorizer.transform([clean_url])
        X_hand     = url_handcrafted_features([clean_url])
        X_hand_s   = sp.csr_matrix(scaler.transform(X_hand))
        X_vec      = sp.hstack([X_tfidf, X_hand_s])
        
        seq = pad_sequences(tokenizer.texts_to_sequences([clean_url]), maxlen=MAX_LEN)
        
        # Layer 4: AI Inference
        normal_idx = list(le.classes_).index('Normal')
        
        try:
            svm_proba = svm.predict_proba(X_vec)
        except:
            svm_proba = np.zeros((1, len(le.classes_)))

        try:
            hybrid_prob = hybrid_cnn.predict(seq, verbose=0)
            attack_type = le.inverse_transform([np.argmax(hybrid_prob[0])])[0]
        except:
            hybrid_prob = np.zeros((1, len(le.classes_)))
            attack_type = "Anomaly"

        try:
            # Stacking Meta Learner
            X_stack = np.hstack([svm_proba, hybrid_prob, X_hand])
            meta_prob = meta_learner.predict_proba(X_stack)[0, 1] # Prob of class 1 (Malicious)
        except:
            meta_prob = 0.0

        if safe_bypass and meta_prob >= 0.999:
            safe_bypass = False

        if safe_bypass:
            return jsonify({
                'url': raw_url,
                'is_dangerous': False,
                'prediction': 0,
                'confidence': 1.0,
                'reason': "Safe Path Heuristic Bypass"
            })
            
        # Layer 2: Static Heuristics
        rules_triggered = []
        if re.search(r"<script>", decoded_url, re.IGNORECASE): rules_triggered.append("XSS")
        if re.search(r"javascript:", decoded_url, re.IGNORECASE): rules_triggered.append("XSS")
        if re.search(r"UNION SELECT", decoded_url, re.IGNORECASE): rules_triggered.append("SQLi")
        if re.search(r"' OR '1'='1", decoded_url, re.IGNORECASE): rules_triggered.append("Basic SQLi")
        if re.search(r"/\*!", decoded_url): rules_triggered.append("MySQL Evasion")
        if re.search(r"\.\./", decoded_url): rules_triggered.append("Traversal")
        if re.search(r"/etc/passwd", decoded_url): rules_triggered.append("Sensitive File") 
        if re.search(r"[;|]\s*(cat|ls|pwd|whoami|wget|curl)", decoded_url, re.IGNORECASE): rules_triggered.append("Command Injection") 

        # 1. Executable Payload Detection (.exe, .msi, .apk, .bat, .sh)
        if re.search(r"\.(exe|msi|bat|ps1|sh|bin|apk)(?:\?|$)", decoded_url, re.IGNORECASE): 
            rules_triggered.append("Executable Payload")
            
        # 2. Windows Directory Traversal & Critical Exploit Paths
        if re.search(r"(?i)(c:\\windows|system32|cmd\.exe)", decoded_url): 
            rules_triggered.append("Windows Exploit")
        if re.search(r"(\.\.\\)|(\\%2e\\%2e)", decoded_url, re.IGNORECASE): 
            rules_triggered.append("Windows Traversal")
            
        # 3. Credential Stealing / Brand Spoofing in Domain
        # Identifies cases where a domain contains a brand keyword but bypassed the Layer 1 Trusted root check.
        brand_keywords_strict = ['paypal', 'apple', 'microsoft', 'google', 'amazon', 'facebook', 'netflix', 'bank']
        if domain_part not in WHITELIST_SET and any(b in domain_part for b in brand_keywords_strict):
            rules_triggered.append("Brand Spoofing")
        
        if rules_triggered:
            return jsonify({
                'url': raw_url,
                'is_dangerous': True,
                'prediction': 1,
                'confidence': 1.0,
                'reason': f"Rule Triggered: {rules_triggered[0]}"
            })
            
        # Contextual Domain Age Risk
        age_risk = get_domain_age_risk(decoded_url)
        
        # Final Ensemble + Contextual Risk
        ens_score = meta_prob + age_risk
        
        # Phase 10 FP Reduction: Set Precision Threshold High (0.70 -> 0.96)
        is_dangerous = bool(ens_score >= 0.96)
        
        reason = "Safe"
        if is_dangerous:
            reason = f"AI Blocked: {attack_type}" if attack_type != 'Normal' else "AI Blocked: Zero-Day Phishing"

        return jsonify({
            'url': raw_url,
            'is_dangerous': is_dangerous,
            'prediction': 1 if is_dangerous else 0,
            'confidence': float(min(ens_score, 1.0)), # Cap at 1.0
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
