from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import numpy as np
import os
import sys
import re
from urllib.parse import unquote
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Lambda, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- CONFIG ---
MAX_LEN = 500
tf.keras.config.enable_unsafe_deserialization()

# --- PATHS (Adjusted to match your friend's structure) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumes models are in the same folder or a 'public' subfolder. 
# We check both to be safe.
def get_path(filename):
    p1 = os.path.join(BASE_DIR, 'public', filename)
    p2 = os.path.join(BASE_DIR, filename)
    return p1 if os.path.exists(p1) else p2

# --- REBUILD META-MODEL ARCHITECTURE ---
def build_meta_model(vocab_size):
    input_a = Input(shape=(MAX_LEN,))
    input_b = Input(shape=(MAX_LEN,))
    input_seq = Input(shape=(MAX_LEN,))
    x = Embedding(vocab_size, 32)(input_seq)
    x = LSTM(64)(x)
    x = Dense(32, activation='relu')(x)
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
    
    # Identify Normal Class
    normal_class = 'Normal'
    if 'Normal' not in le.classes_: 
        normal_class = 'benign' if 'benign' in le.classes_ else le.classes_[0]
    normal_idx = list(le.classes_).index(normal_class)
    
    print("✅ All Hybrid Models Loaded Successfully")
    model_loaded = True
except Exception as e:
    print(f"❌ Critical Error Loading Models: {e}")
    model_loaded = False

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Models not loaded', 'is_dangerous': False}), 500
    
    try:
        data = request.json
        url = data.get('url', '')
        if not url: return jsonify({'error': 'No URL provided'}), 400
        
        # --- HYBRID SCANNING LOGIC (V5) ---
        decoded_url = unquote(url)
        rules_triggered = []

        # 1. Static Rules
        if re.search(r"<script>", decoded_url, re.IGNORECASE): rules_triggered.append("XSS")
        if re.search(r"javascript:", decoded_url, re.IGNORECASE): rules_triggered.append("XSS")
        if re.search(r"UNION SELECT", decoded_url, re.IGNORECASE): rules_triggered.append("SQLi")
        if re.search(r"' OR '1'='1", decoded_url, re.IGNORECASE): rules_triggered.append("Basic SQLi")
        if re.search(r"/\*!", decoded_url): rules_triggered.append("MySQL Evasion")
        if re.search(r"\.\./", decoded_url): rules_triggered.append("Traversal")
        if re.search(r"/etc/passwd", decoded_url): rules_triggered.append("Sensitive File") 
        if re.search(r"[;|]\s*(cat|ls|pwd|whoami|wget|curl)", decoded_url, re.IGNORECASE): rules_triggered.append("Command Injection") 
        if re.search(r"\{\{.*\}\}", decoded_url): rules_triggered.append("SSTI")
        if re.search(r"=\s*(http|https|ftp)://", decoded_url, re.IGNORECASE): rules_triggered.append("RFI")

        # 2. AI Predictions
        # SVM
        try: svm_prob = float(svm.predict_proba(vectorizer.transform([url]))[0][1])
        except: svm_prob = 0.0

        # CNN
        seq = pad_sequences(tokenizer.texts_to_sequences([url]), maxlen=MAX_LEN)
        dl_pred = cnn_model.predict(seq, verbose=0)[0]
        dl_conf = float(np.max(dl_pred))
        dl_class_idx = np.argmax(dl_pred)
        is_dl_attack = (dl_class_idx != normal_idx)
        attack_type = le.inverse_transform([dl_class_idx])[0]

        # Meta
        sig = "UNION SELECT * FROM users"
        seq_sig = pad_sequences(tokenizer.texts_to_sequences([sig]), maxlen=MAX_LEN)
        meta_sim = float(meta_model.predict([seq, seq_sig], verbose=0)[0][0])

        # 3. Decision Logic
        is_dangerous = False
        reason = "Safe"
        
        # Whitelist
        safe_domains = ['google.com', 'youtube.com', 'wikipedia.org', 'amazon.com', 'github.com', 'stackoverflow.com', 'weather.com', 'dev.to', 'facebook.com', 'netflix.com']
        if any(d in url for d in safe_domains):
            is_dangerous = False
            reason = "Whitelisted"
        elif rules_triggered:
            is_dangerous = True
            reason = f"Rule: {rules_triggered[0]}"
        elif svm_prob < 0.05: 
            # Veto Power
            if is_dl_attack and dl_conf > 0.99: 
                is_dangerous = True; reason = f"AI: High Confidence {attack_type}"
            elif meta_sim > 0.90: 
                is_dangerous = True; reason = "AI: Zero-Day Anomaly"
        else:
            if is_dl_attack and dl_conf > 0.8: is_dangerous = True; reason = f"AI: {attack_type}"
            elif svm_prob > 0.80: is_dangerous = True; reason = "AI: Heuristic"
            elif meta_sim > 0.80: is_dangerous = True; reason = "AI: Anomaly"

        return jsonify({
            'url': url,
            'is_dangerous': is_dangerous,
            'prediction': 1 if is_dangerous else 0, # Maintaining his API format
            'confidence': max(svm_prob, dl_conf, meta_sim),
            'reason': reason
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e), 'is_dangerous': False}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded
    })

if __name__ == '__main__':
    print("Starting Hybrid Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)