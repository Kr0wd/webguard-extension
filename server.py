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

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing so the extension can talk to this server

# --- CONFIG ---
# Maximum length of URL sequence to consider for the Deep Learning models
MAX_LEN = 500
# Allow loading of Keras models that might have custom layers or older formats
tf.keras.config.enable_unsafe_deserialization()

# --- PATHS ---
# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Helper function to find model files. 
# It checks both a 'public' subdirectory and the current directory.
def get_path(filename):
    p1 = os.path.join(BASE_DIR, 'public', filename)
    p2 = os.path.join(BASE_DIR, filename)
    return p1 if os.path.exists(p1) else p2

# --- REBUILD META-MODEL ARCHITECTURE ---
# Neural Networks sometimes need their architecture defined in code to load weights correctly.
# This defines a "Siamese Network" (or comparison network) used for the Meta Model.
def build_meta_model(vocab_size):
    # Inputs for two different sequences (e.g., the URL to test vs a known bad signature)
    input_a = Input(shape=(MAX_LEN,))
    input_b = Input(shape=(MAX_LEN,))
    input_seq = Input(shape=(MAX_LEN,))
    
    # Shared layers to process both inputs identically
    # Embedding layer: Turns integer numbers (characters/words) into dense vectors
    x = Embedding(vocab_size, 32)(input_seq)
    # LSTM layer: specialized for sequence data (like text/URLs)
    x = LSTM(64)(x)
    x = Dense(32, activation='relu')(x)
    
    # Create the base encoder model
    base = Model(input_seq, x)
    
    # Process both inputs through the base model
    vec_a = base(input_a)
    vec_b = base(input_b)
    
    # Calculate the absolute difference between the two vector representations
    def abs_diff(t): return K.abs(t[0] - t[1])
    def compute_output_shape(shapes): return shapes[0]
    dist = Lambda(abs_diff, output_shape=compute_output_shape)([vec_a, vec_b])
    
    # Final output layer: A sigmoid classifier (0 to 1) 
    # Determines how "similar" the inputs are (or if they belong to the same class)
    out = Dense(1, activation='sigmoid')(dist)
    return Model([input_a, input_b], out)

# --- LOAD RESOURCES ---
print("🚀 Initializing WebGuard Hybrid Engine...")
try:
    # Load the preprocessing tools
    tokenizer = joblib.load(get_path('local_tokenizer.pkl'))       # Converts text -> numbers
    le = joblib.load(get_path('local_label_encoder.pkl'))          # Converts numbers -> class names (e.g. 'phishing')
    vectorizer = joblib.load(get_path('local_vectorizer.pkl'))     # Converts text -> matrix for SVM
    
    # Load the Machine Learning models
    svm = joblib.load(get_path('local_svm_model.pkl'))             # Support Vector Machine (Classical ML)
    cnn_model = load_model(get_path('local_best_cnn.keras'), safe_mode=False) # Convolutional Neural Network (Deep Learning)
    
    # Initialize and load the Meta Model (Deep Learning)
    vocab_size = len(tokenizer.word_index) + 1
    meta_model = build_meta_model(vocab_size)
    meta_model.load_weights(get_path('local_meta_model.keras'))
    
    # Identify which class index represents "Normal" / "Safe" traffic
    normal_class = 'Normal'
    if 'Normal' not in le.classes_: 
        normal_class = 'benign' if 'benign' in le.classes_ else le.classes_[0]
    normal_idx = list(le.classes_).index(normal_class)
    
    print("✅ All Hybrid Models Loaded Successfully")
    model_loaded = True
except Exception as e:
    print(f"❌ Critical Error Loading Models: {e}")
    model_loaded = False

# --- API ROUTES ---

@app.route('/predict', methods=['POST'])
def predict():
    # Check if models are ready before processing
    if not model_loaded:
        return jsonify({'error': 'Models not loaded', 'is_dangerous': False}), 500
    
    try:
        # Get the URL from the POST request data
        data = request.json
        url = data.get('url', '')
        if not url: return jsonify({'error': 'No URL provided'}), 400
        
        # --- HYBRID SCANNING LOGIC (V5) ---
        decoded_url = unquote(url) # Decode URL encoding (e.g. %20 -> space)
        rules_triggered = []

        # 1. Static Rules (Regex Checks)
        # Check for common malicious patterns directly in the URL string
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
        # Model 1: SVM (Support Vector Machine)
        # Good at detecting keyword-based anomalies
        try: svm_prob = float(svm.predict_proba(vectorizer.transform([url]))[0][1])
        except: svm_prob = 0.0

        # Model 2: CNN (Convolutional Neural Network)
        # Good at detecting patterns in character sequences
        seq = pad_sequences(tokenizer.texts_to_sequences([url]), maxlen=MAX_LEN)
        dl_pred = cnn_model.predict(seq, verbose=0)[0]
        dl_conf = float(np.max(dl_pred)) # Confidence of the top prediction
        dl_class_idx = np.argmax(dl_pred) # The class it thinks it is
        is_dl_attack = (dl_class_idx != normal_idx) # Is it NOT normal?
        attack_type = le.inverse_transform([dl_class_idx])[0] # Get the text name of the attack

        # Model 3: Meta Model (Anomaly / Signature Comparison)
        # Checks similarity to a known SQL Injection signature
        sig = "UNION SELECT * FROM users"
        seq_sig = pad_sequences(tokenizer.texts_to_sequences([sig]), maxlen=MAX_LEN)
        meta_sim = float(meta_model.predict([seq, seq_sig], verbose=0)[0][0])

        # 3. Decision Logic (Combining all signals)
        is_dangerous = False
        reason = "Safe"
        
        # Hardcoded Whitelist (Always Safe)
        safe_domains = [
            'google.com', 'youtube.com', 'wikipedia.org', 'amazon.com', 'github.com', 
            'stackoverflow.com', 'weather.com', 'dev.to', 'facebook.com', 
            'netflix.com', 'whatsapp.com', 'microsoft.com', 'apple.com',
            'chrome.com', 'shazam.com', 'instagram.com', 'linkedin.com', 'twitter.com', 'x.com', 
            'twitch.tv', 'reddit.com', 'bing.com',
            # Search & Social
            'yahoo.com', 'duckduckgo.com', 'baidu.com', 'yandex.com',
            'tiktok.com', 'pinterest.com', 'snapchat.com', 'telegram.org',
            # Tech & Dev
            'gitlab.com', 'bitbucket.org', 'npm.com', 'pypi.org', 'docker.com',
            'azure.com', 'salesforce.com', 'oracle.com', 'ibm.com',
            # Media & News
            'cnn.com', 'bbc.com', 'nytimes.com', 'theguardian.com', 'forbes.com', 'bloomberg.com',
            # Shopping & Payment
            'ebay.com', 'walmart.com', 'target.com', 'bestbuy.com', 'aliexpress.com',
            'paypal.com', 'stripe.com',
            # Other
            'spotify.com', 'adobe.com', 'dropbox.com', 'zoom.us'
        ]
        
        if any(d in url for d in safe_domains):
            is_dangerous = False
            reason = "Whitelisted"
        elif rules_triggered:
            # If a strict rule matches, flag immediately
            is_dangerous = True
            reason = f"Rule: {rules_triggered[0]}"
        elif svm_prob < 0.05: 
            # If SVM thinks it's very safe, only override if Deep Learning is extremely confident
            # This reduces false positives on normal looking URLs
            if is_dl_attack and dl_conf > 0.99: 
                is_dangerous = True; reason = f"AI: High Confidence {attack_type}"
            elif meta_sim > 0.90: 
                is_dangerous = True; reason = "AI: Zero-Day Anomaly"
        else:
            # General voting logic
            if is_dl_attack and dl_conf > 0.8: is_dangerous = True; reason = f"AI: {attack_type}"
            elif svm_prob > 0.80: is_dangerous = True; reason = "AI: Heuristic"
            elif meta_sim > 0.80: is_dangerous = True; reason = "AI: Anomaly"

        # Return the final JSON result
        return jsonify({
            'url': url,
            'is_dangerous': is_dangerous,
            'prediction': 1 if is_dangerous else 0, # Legacy support
            'confidence': max(svm_prob, dl_conf, meta_sim),
            'reason': reason
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e), 'is_dangerous': False}), 500

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded
    })

# Main entry point: Runs the server
if __name__ == '__main__':
    print("Starting Hybrid Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)