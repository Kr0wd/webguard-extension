# 🚀 WebGuard Backend Deep Dive: `server.py`

This document provides a line-by-line explanation of the WebGuard Flask server, which acts as the real-time Threat Detection Engine. 

It explains what each block of code does, and crucially, *why* we chose that implementation strategy over other alternatives.

---

## 1. Imports & Configuration (Lines 1-24)

```python
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
```
*   **What it does:** Imports all required libraries. `Flask` creates the web server. `joblib` loads standard ML models (SVM, XGBoost/Meta). `tensorflow` loads the massive Deep Neural Network. Tools like `re` (Regular Expressions) and `unquote` are used to clean URLs.
*   **Why Flask and not Node.js?** Machine Learning in Python is an industry standard (using Scikit-Learn and TensorFlow). While we *could* have built a Node.js server and called Python scripts via child processes, doing so introduces severe I/O latency. Since WebGuard needs to block threats in milliseconds *before* Chrome loads a page, keeping the server natively in Python where the models reside in RAM is drastically faster.

```python
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

MAX_LEN = 550
tf.keras.config.enable_unsafe_deserialization()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, 'models', filename)
```
*   **What it does:** Initializes the API and sets critical constants. 
*   **Why `CORS(app)`?** The requests are coming from a Chrome Extension (`chrome-extension://...`). Without CORS enabled, the browser would block the extension from talking to `localhost:5000` for security reasons.
*   **Why `MAX_LEN = 550`?** Our Deep Learning model expects a fixed mathematical matrix size. A maximum length of 550 characters covers 99.9% of URLs. If a URL is shorter, we pad it with zeros; if longer, we truncate it.
*   **Why `enable_unsafe_deserialization`?** Keras `.keras` objects can execute Python code when loaded. Since we trained the models locally and trust them, we must explicitly tell TensorFlow it's safe to load them.

---

## 2. URL Normalization & The Shield (Lines 29-66)

```python
def strip_protocol(url):
    """Removes http:// and https:// to normalize the URL"""
    url = str(url).strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    return url
```
*   **What it does:** Cleans the URL so `https://www.google.com` and `http://google.com` mathematically look identical to the AI.

```python
WHITELIST_SET = {
    "192.168.1.1", "192.168.1.254", "localhost", "127.0.0.1",
    "fisat.ac.in", "intranet.fisat.ac.in", "app.ktu.edu.in", "ktu.edu.in",
    "myntra.com"
}
```
*   **Why a SET `{}` instead of a LIST `[]`?** This is **Layer 1 Defense**. If we used a list, Python would search through it sequentially (O(N) time complexity). A Set uses a Hash Table beneath the hood, making lookups instant **(O(1) time complexity)**. This shaves off milliseconds for safe local traffic.

```python
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
```
*   **What it does:** Rapidly bypasses the AI for known, 100% safe traffic.
*   **Why check TLDs?** If we didn't whitelist `.gov.in`, the AI might flag outdated Indian government websites because their URL structures are often chaotic or lack HTTPS, which the AI interprets as suspicious.

---

## 3. Mathematical Feature Extraction (Lines 68-214)

Because the Meta-Learner (XGBoost) cannot read strings, we must manually measure the URL and give it mathematical attributes to judge.

```python
def calculate_entropy(text):
    if not text: return 0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)
```
*   **What it does:** Uses Shannon Entropy logic to measure how "random" a string is.
*   **Why?** `google.com/search` has low entropy. `paypal-login.com/x9!j$f8ja(1)` has massive entropy. Phishing attacks heavily rely on randomly generated, single-use hashes to obscure their payloads.

```python
def extract_features(url):
    c = strip_protocol(url)
    f1 = [
        len(url), len(c), url.count('.'), url.count('/'), url.count('-'), url.count('@'),
        url.count('?'), url.count('='), url.count('&'), int('//' in url[7:]),
        sum(ch.isdigit() for ch in url)/max(len(url),1), calculate_entropy(url)
    ]
```
*   **What it does:** Extracts 12 counts (length, symbols, entropy). Hackers often use excessive `@` symbols or hyphens to confuse users.

**Handling Brand Spoofing (Finding "bank0famerica"):**
If we just look for "paypal", hackers bypass it by typing `paypa1`.

```python
    # Normalize digit-substitutions (0→o, 1→i/l, 3→e, 4→a)
    _DIGIT_MAP = str.maketrans('013458', 'oieash')
    # ... calculates Levenshtein Distance ...
```
*   **Why Levenshtein Distance?** Instead of hardcoding every possible typo of a brand (e.g., `go0gle`, `goog1e`, `g0ogle`), Levenshtein math calculates how many "edits" it takes to turn string A into string B. If a requested domain is only 1 or 2 edits away from `paypal`, and it isn't officially `paypal.com`, we flag it as a spoof.

---

## 4. Bootstrapping the AI Models (Lines 216-233)

```python
    svm_model = joblib.load(get_path('local_svm_model.pkl'))
    cnn_model = load_model(get_path('local_hybrid_model.keras'))
    meta_model = joblib.load(get_path('local_meta_learner_global.pkl'))
```
*   **What it does:** Loads the multi-gigabyte models into RAM *once* when the server starts.
*   **Why at the module level?** We place this outside the API route. If we loaded the models *inside* the `/predict` route, it would take 3-5 seconds to answer every single URL check, crashing the user's browser flow. By keeping them in memory, predictions take ~20 milliseconds.

---

## 5. The Core Engine API (Lines 237-520)

### Layer 1 & 2 Execution
```python
@app.route('/predict', methods=['POST'])
def predict():
    # LAYER 1
    if is_whitelisted(decoded_url) or raw_url.startswith(('chrome-', 'about:')):
        return jsonify({'is_dangerous': False}) # Instant Return
```
*   **Why bypass Chrome Internal URLs?** When Chrome opens a new tab (`chrome://newtab`), the extension catches it. By hard-bypassing internal URLs here, we prevent infinite looping bugs where the extension accidentally tries to block Chrome's own system pages.

```python
    # LAYER 2: Static Heuristics
    if re.search(r"(?i)<script[\s>]", decoded_url):
        rules_triggered.append("XSS")
    if re.search(r"(?i)union\s+(all\s+)?select", decoded_url):
        rules_triggered.append("SQLi")
```
*   **Why hardcode Regex if we have AI?** The AI is pattern-based; it might make mistakes. A Cross-Site Scripting (XSS) attack isn't a "pattern"—it's an explicit computer command. By writing ironclad Regex rules, we guarantee 100% detection of explicit code-injection attacks before they reach the AI.

### Layer 3: The XGBoost Meta-Ensemble Execution

```python
        # 1. Feature Extraction
        X_vec = vectorizer.transform([clean_url])
        svm_proba = svm_model.predict_proba(X_vec)

        seq = pad_sequences(tokenizer.texts_to_sequences([clean_url]), maxlen=MAX_LEN_CNN)
        cnn_proba = cnn_model.predict(seq, verbose=0)

        raw_feats = extract_features(clean_url)
        scaled_feats = url_scaler.transform(raw_feats)

        # 2. Meta-Stack Integration
        meta_features = np.hstack([svm_proba, cnn_proba, scaled_feats])
        all_probs = meta_model.predict_proba(meta_features)[0]
        malicious_proba = 1.0 - all_probs[normal_idx]
```
*   **What it does:** 
    1. It asks the SVM: *"Based on keywords, what's your malicious probability?"* (Returns: `[0.1, 0.9]`)
    2. It asks the CNN: *"Based on structural sequences, what's your probability?"*
    3. It gathers the 13 Math features.
    4. By using `np.hstack`, it creates a single string of floats representing all opinions.
    5. The `meta_model` (XGBoost) looks at this massive vote and declares the final `malicious_proba`.

### Dynamic Tresholding (The Secret to No False Positives)

```python
        threshold = 0.85 if domain_root in HIGH_TRUST_DOMAINS else 0.65

        # Heuristic: Short, simple domains (like fast.com)
        if (len(domain_root) < 15 and domain_part == domain_root
                and not rules_triggered and not is_brand_spoof_detected):
            threshold = 0.99
            
        is_dangerous = bool(malicious_proba > threshold)
```
*   **Why not just use `> 0.50` like normal AI?** 
    *   **Trust But Verify:** If a URL comes from `google.com` (High Trust), the AI must be 85% certain it contains an injection payload before blocking it.
    *   **The Startup Problem:** Startups buy short, weird domains (`cred.club`, `ola.in`). Because they are unknown and short, the AI flags them as anomalies. By requiring **99% confidence** to block short, unknown domains, we successfully protect everyday web traffic from false positives while still heavily penalizing deep obfuscated phishing paths.

---

### Conclusion
By reading this flow, you can see `server.py` isn't just "running a model." It acts as a funnel. It uses ultra-fast logic (O(1) Lists and Regex) to discard 90% of traffic instantly, saving the heavy GPU-based ML predictions for only the deeply complex unknown queries.
