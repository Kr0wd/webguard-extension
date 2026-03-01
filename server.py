from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os
import re
from urllib.parse import unquote
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
MAX_LEN = 500
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

print("🛡️ Loading Majestic Top 1 Million Whitelist into memory...")
try:
    df_benign = pd.read_csv(get_path("majestic_million.csv"))
    # O(1) Hash Map for instant lookups
    WHITELIST_SET = set(df_benign['Domain'].dropna().str.lower().tolist())
    
    # Add safe local IPs and CUSTOM INTERNAL DOMAINS
    WHITELIST_SET.update([
        "192.168.1.1", "192.168.1.254", "localhost", "127.0.0.1",
        "fisat.ac.in", "intranet.fisat.ac.in", "app.ktu.edu.in", "ktu.edu.in"
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

# --- REBUILD META-MODEL ARCHITECTURE ---
def build_meta_model(vocab_size):
    input_a = Input(shape=(MAX_LEN,))
    input_b = Input(shape=(MAX_LEN,))
    input_seq = Input(shape=(MAX_LEN,))
    
    x = Embedding(vocab_size, 128)(input_seq)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dense(64, activation='relu')(x)
    
    base = Model(input_seq, x)
    vec_a = base(input_a)
    vec_b = base(input_b)
    
    def abs_diff(t): return K.abs(t[0] - t[1])
    def compute_output_shape(shapes): return shapes[0]
    dist = Lambda(abs_diff, output_shape=compute_output_shape)([vec_a, vec_b])
    
    out = Dense(1, activation='sigmoid')(dist)
    return Model([input_a, input_b], out)

# --- LOAD RESOURCES ---
print("🚀 Initializing WebGuard Hybrid Engine...")
try:
    tokenizer = joblib.load(get_path('local_tokenizer.pkl'))       
    le = joblib.load(get_path('local_label_encoder.pkl'))          
    vectorizer = joblib.load(get_path('local_vectorizer.pkl'))     
    
    svm = joblib.load(get_path('local_svm_model.pkl'))             
    cnn_model = load_model(get_path('local_best_cnn.keras'), safe_mode=False) 
    
    vocab_size = len(tokenizer.word_index) + 1
    meta_model = build_meta_model(vocab_size)
    meta_model.load_weights(get_path('local_meta_model.keras'))
    
    # Pre-calculate the SQLi signature sequence once to save API time
    sig = "UNION SELECT * FROM users"
    global_sig_seq = pad_sequences(tokenizer.texts_to_sequences([sig]), maxlen=MAX_LEN)
    
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
        # ==========================================
        rules_triggered = []
        if re.search(r"<script>", decoded_url, re.IGNORECASE): rules_triggered.append("XSS")
        if re.search(r"javascript:", decoded_url, re.IGNORECASE): rules_triggered.append("XSS")
        if re.search(r"UNION SELECT", decoded_url, re.IGNORECASE): rules_triggered.append("SQLi")
        if re.search(r"' OR '1'='1", decoded_url, re.IGNORECASE): rules_triggered.append("Basic SQLi")
        if re.search(r"/\*!", decoded_url): rules_triggered.append("MySQL Evasion")
        if re.search(r"\.\./", decoded_url): rules_triggered.append("Traversal")
        if re.search(r"/etc/passwd", decoded_url): rules_triggered.append("Sensitive File") 
        if re.search(r"[;|]\s*(cat|ls|pwd|whoami|wget|curl)", decoded_url, re.IGNORECASE): rules_triggered.append("Command Injection") 
        
        if rules_triggered:
            return jsonify({
                'url': raw_url,
                'is_dangerous': True,
                'prediction': 1,
                'confidence': 1.0,
                'reason': f"Rule Triggered: {rules_triggered[0]}"
            })

        # ==========================================
        # LAYER 3: THE AI ENSEMBLE (Tuned for False Positives)
        # ==========================================
        clean_url = strip_protocol(decoded_url)
        
        # 1. SVM Gatekeeper (Requires 70% confidence to vote Block)
        try: 
            X_vec = vectorizer.transform([clean_url])
            svm_pred_raw = svm.predict(X_vec)
            svm_conf = float(np.max(svm.predict_proba(X_vec)))
            svm_binary = 1 if (le.inverse_transform(svm_pred_raw)[0] != 'Normal' and svm_conf > 0.70) else 0
        except: 
            svm_binary, svm_conf = 0, 0.0

        # 2. CNN Deep Pattern (Requires 85% confidence to vote Block)
        seq = pad_sequences(tokenizer.texts_to_sequences([clean_url]), maxlen=MAX_LEN)
        dl_pred = cnn_model.predict(seq, verbose=0)[0]
        dl_conf = float(np.max(dl_pred))
        dl_class_idx = np.argmax(dl_pred)
        attack_type = le.inverse_transform([dl_class_idx])[0]
        
        # Only vote 'Block' if it's NOT Normal AND it is very sure about it
        cnn_binary = 1 if (attack_type != 'Normal' and dl_conf > 0.85) else 0

        # 3. Siamese Meta-Model (Requires 80% anomaly similarity to vote Block)
        sig_batch = np.repeat(global_sig_seq, len(seq), axis=0)
        meta_sim = float(meta_model.predict([seq, sig_batch], verbose=0)[0][0])
        meta_binary = 1 if meta_sim > 0.80 else 0

        # --- THE VOTING COMMITTEE ---
        # If 2 out of 3 models flag it with HIGH CONFIDENCE, block it.
        total_votes = svm_binary + cnn_binary + meta_binary
        is_dangerous = bool(total_votes >= 2)
        
        reason = "Safe"
        if is_dangerous:
            if cnn_binary == 1: reason = f"AI Blocked: {attack_type}"
            else: reason = "AI Blocked: Zero-Day Anomaly"

        return jsonify({
            'url': raw_url,
            'is_dangerous': is_dangerous,
            'prediction': 1 if is_dangerous else 0,
            'confidence': max(svm_conf, dl_conf, meta_sim),
            'reason': reason
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e), 'is_dangerous': False}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model_loaded})

if __name__ == '__main__':
    print("Starting Hybrid Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)