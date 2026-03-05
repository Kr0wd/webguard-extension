# 🧠 WebGuard ML Brain Deep Dive: `train.py` & Parameter Reasoning

This document provides a line-by-line breakdown of `train.py`, specifically focusing on **why specific mathematical values (like C=1.0, maxlen=550, Dropout=0.3) were chosen**, what the alternatives were, and why we rejected them.

---

## 1. What is Shannon Entropy? (Line 41)

```python
def calculate_entropy(text):
    if not text: return 0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)
```
Before diving into the rest of the file, let's address **Shannon Entropy**.
Invented by Claude Shannon in 1948, entropy is a mathematical formula used in cryptography to measure **Information Density** or **Chaos** in a string of text.

*   **Low Entropy (Predictable):** `google.com` (Entropy ~ 2.5). The letters make sense, they repeat predictably (o, g), and it's easy for a human to read.
*   **High Entropy (Chaotic):** `phish.com/login?token=x!9A$bcQ2p` (Entropy ~ 4.8). This is full of unique, random characters that don't repeat predictably.
*   **Why we use it:** Phishing URLs frequently use massive, randomly generated strings (Base64 encoding or session hashes) to hide their true payload from basic scanners. By calculating the entropy, we give the AI a single number that says: *"Hey, this URL is 90% mathematically chaotic. Be careful."* It is a far more reliable indicator of obfuscation than trying to guess specific bad words.

---

## 2. Inhaling the Datasets (Lines 71-137)

```python
df_benign_def = pd.read_csv('data/definitive_benign.csv')
# ... skipping 5 other CSV loads ...
```
*   **Alternative Option Rejected:** We could have just used one massive open-source dataset (like the famous Kaggle Phishing Dataset). 
*   **Why we rejected it:** Public datasets are often heavily biased. If a model only trains on one dataset, it learns the specific quirks of the humans who made that dataset, resulting in high "Accuracy" on paper, but terrible performance in the real world. By merging Majestic Million (Safe), PhishTank (Phishing), and CAPEC (Local SQLi attacks), we force the AI to understand the *entire internet*, not just one specific type of attack.

```python
# Stratified Sampling to keep sizes manageable but balanced
df_final = pd.concat([
    df_all[df_all['label'] == 'Normal'].sample(n=min(class_counts['Normal'], 120000), random_state=42),
    df_all[df_all['label'] == 'Phishing'].sample(n=min(class_counts['Phishing'], 80000), random_state=42),
    ...
```
*   **Why 120,000 Normal vs 80,000 Phishing?** 
    *   **The Problem:** The internet is 99.9% safe. If you train an AI on 99.9% safe data, the AI will just guess "Safe" 100% of the time to achieve a 99.9% score. This is called **Class Imbalance**.
    *   **The Fix:** We forcefully truncate the Safe list to 120k and the Phishing list to 80k. We maintain a *slight* bias toward Safe (1.5:1 ratio) so the AI knows Safe is more common, but it's balanced enough that the AI is forced to actually hunt for the malicious logic.

---

## 3. Training Brain 1: The SVM (Lines 139-150)

```python
vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1,3))
```
*   **What it does:** Breaks the URLs into "N-grams" (chunks of 1 to 3 characters/words), up to a maximum of 25,000 unique chunks, measuring how unique they are via TF-IDF (Term Frequency-Inverse Document Frequency).
*   **Why `ngram_range=(1,3)`?** If it was `(1,1)`, the AI would only see individual letters (too granular). If it was `(5,5)`, it would only look for massive 5-word phrases (too specific). `(1,3)` allows it to see "wp", "wp-admin", and "wp-admin-php" simultaneously.
*   **Why `max_features=25000`?** Above 25,000, the mathematical grid becomes so massive ("The Curse of Dimensionality") that the SVM slows down exponentially without any gain in accuracy. 25k is the "Golden Ratio" for URL text.

```python
svm = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=1000, dual=False), cv=3)
```
*   **Why `LinearSVC` instead of `SVC(kernel='rbf')`?** A standard radial SVM (`rbf`) calculates complex curves to separate data. For 500,000 URLs, an `rbf` kernel would literally take days to train on a standard CPU. `LinearSVC` only draws straight lines. It trains in 30 seconds and loses almost zero accuracy.
*   **Why `C=1.0`?** "C" is the Regularization Parameter. A low C (0.1) creates a "soft" boundary (underfitting). A high C (100) tries to classify every single dot perfectly, resulting in a jagged, overfitted mess that fails on new internet URLs. `1.0` is a balanced, generalized boundary.
*   **Why `CalibratedClassifierCV`?** A standard `LinearSVC` tells you "It is Phishing" (1). By wrapping it in a Calibrator, we force it to output a percentage based on standard deviation mapping: "I am 84% sure it is Phishing". We absolutely need those percentages to feed into the XGBoost Meta-Learner later.

---

## 4. Training Brain 2: Deep Learning CNN-BiLSTM (Lines 152-171)

```python
tokenizer = Tokenizer(num_words=10000, char_level=True, oov_token='<OOV>')
# ...
MAX_LEN = 550
X_seq = pad_sequences(sequences, maxlen=MAX_LEN)
```
*   **Why `char_level=True`?** Traditional Natural Language Processing (NLP) looks at whole words ("apple", "banana"). URLs don't have spaces! If we used Word-Level processing, a URL like `paypal-update` is just one meaningless word. By using Character-Level processing, the AI reads `[p, a, y, p, a, l, -, u...]`. It learns the mathematical shape of the spelling itself!
*   **Why `MAX_LEN = 550`?** 99% of URLs are under 100 characters. If we made `MAX_LEN = 2000`, the AI would waste 90% of its processing power scanning empty zeros. 550 characters catches almost all highly-obfuscated injection attacks without bloating RAM.

```python
cnn = Sequential([
    Embedding(10001, 64, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])
```
*   **Why a `Bidirectional(LSTM)` instead of a standard `RNN`?** A standard Recurrent Neural Network reads left-to-right. Phishing payloads (like an executable `.exe` file hidden in parameters) are often at the *very end* of a URL. The Bidirectional LSTM reads the URL left-to-right AND right-to-left simultaneously, meeting in the middle. It understands that `.exe` at the very end totally changes the context of `google.com` at the beginning.
*   **Why `Dropout(0.3)`?** Neural networks are notorious for "memorizing" passwords to tests (Overfitting). `Dropout(0.3)` randomly shuts off 30% of the virtual brain cells during every single training pass. It simulates "brain damage," forcing the remaining 70% of the network to work harder and learn the *actual underlying rules* of phishing rather than just memorizing specific URLs.

---

## 5. Training Brain 3: XGBoost Meta-Learner (Lines 173-200)

```python
X_meta = np.hstack([svm_probs, cnn_probs, hand_feats_scaled])
```
*   **What this does:** The XGBoost model isn't given the URL text! It is given an array of numbers looking like this: `[SVM_Guess(0.12), CNN_Guess(0.88), Entropy(4.5), Length(215), Brand_Spoof_Flag(1)]`.
*   **Why use XGBoost as the "Judge"?** XGBoost is the king of structured, tabular number data. While Deep Learning is great for reading text/images, XGBoost builds hundreds of "decision trees." It essentially learns: *"If the SVM is confused, but the Entropy is above 4.5 and the Brand Spoof Flag is triggered, override the SVM and declare it malicious."* 

```python
# Meta-Learner Sample Weights: Penalize False Negatives heavily
sample_weights = np.ones(len(y_enc))
for i, label in enumerate(raw_labels):
    if label != 'Normal':
        sample_weights[i] = 2.5 # 2.5x importance for malicious URLs
```
*   **Why `2.5`? (The Recall Penalty Strategy):**
    *   **The Default:** Standard AI treats "Accidentally blocking Google" and "Accidentally letting a virus through" as equal mathematical errors (-1 point).
    *   **The Fix:** By assigning a `2.5x` weight to malicious URLs, we tell XGBoost: *"If you miss a virus, you lose 2.5 points. If you block Google, you only lose 1 point."*
    *   **The Result:** The AI becomes completely terrified of missing viruses. It becomes historically paranoid. This maximizes our **Recall Score** (Security), ensuring nothing slips through. We then handle the slight increase in "False Positives" using the Tiered Bypass logic back in `server.py`!

```python
xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, tree_method='hist', device='cuda')
```
*   **Why `n_estimators=300` and `max_depth=8`?** If the depth is too high (e.g., 20), the trees become so complicated they memorize the noise. A depth of 8 allows for 8 "If/Then" branching questions per tree (e.g., If Entropy > 3 -> If Length > 50 -> ...), providing immense nuance. 300 sequential trees ensures the boosting algorithm has enough time to perfectly smooth out its errors.

---
### Conclusion
By meticulously choosing these mathematical boundaries—balancing classes, restricting dimensions to 25k, reading backwards/forwards with LSTMs, and artificially inflating the punishment for missing viruses by 2.5x—we created an ensemble that doesn't just "guess." It mathematically corners malicious actors from three different strategic angles.
