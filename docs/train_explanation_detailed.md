# 🧠 Deep Dive: `train.py` — Line-by-Line Explanation

This document explains every single line of the `train.py` file in complete detail — including **why** specific choices were made, **what would happen** if a different approach were used, and **what the alternatives are**.

---

## SECTION 1: Imports (Lines 1–20)

```python
import pandas as pd
```
- **What it does:** Imports the `pandas` library and gives it the alias `pd`.
- **Why pandas?** It is the industry-standard Python library for reading and working with tabular data (spreadsheets / CSV files). It gives us a `DataFrame` object — basically an in-memory spreadsheet — which makes filtering, joining, and sampling datasets effortless.
- **Alternative:** `csv` (Python built-in) is lower-level and requires manual parsing loops. `polars` is a faster Rust-powered alternative, but has a different syntax and smaller community. We use pandas because the team knows it and it integrates natively with scikit-learn.

---

```python
import numpy as np
```
- **What it does:** Imports NumPy and gives it the alias `np`.
- **Why NumPy?** All machine learning models in Python take numbers as inputs, not strings. NumPy gives us highly optimised N-dimensional "arrays" (like matrices) that are **10–100x faster** than plain Python lists for mathematical operations.
- **Alternative:** Python's `array` module exists but lacks the linear algebra features NumPy provides. There is no real alternative — deep learning and ML frameworks all require NumPy arrays internally.

---

```python
import joblib
```
- **What it does:** Imports `joblib`, a fast serialisation library.
- **Why joblib?** After training for hours, we need to "save" the model permanently so the server can load it in milliseconds. `joblib` is specifically optimised for saving large NumPy arrays (which is what all scikit-learn models contain) — it's dramatically faster than Python's built-in `pickle` library for this purpose.
- **Alternative:** `pickle` (built-in) would work but is slower for large arrays. `HDF5` (via `h5py`) is used for Keras models but not scikit-learn. We use joblib because it's the official scikit-learn recommended serialiser.

---

```python
import os
import re
import urllib.parse
import math
```
- **`os`:** Allows us to check if files or directories exist (e.g., for checking model save paths).
- **`re`:** Regular Expressions library. Used for stripping `https://` and `www.` from URLs using pattern matching.
- **`urllib.parse`:** Used to decode "URL-encoded" strings. For example, the character `%20` in a URL means a space. `urllib.parse.unquote()` converts it back to a readable space, which helps the AI see the real intent of the URL.
- **`math`:** Used specifically for `math.log2()` in the Shannon Entropy calculation.

---

```python
from sklearn.model_selection import train_test_split
```
- **What it does:** Imports the `train_test_split` function.
- **Why?** This is the gold standard way to divide a dataset. It randomly splits the dataset into a Training Set (used to teach the model) and a Test Set (a held-out portion the model has _never_ seen, used to honestly measure accuracy). Without this, you cannot know if your model has actually "learned" or simply "memorised."
- **Alternative:** Writing a manual split function (e.g., `df[:n]` and `df[n:]`) would work but wouldn't guarantee that the proportion of class labels is preserved. `train_test_split` with `stratify=y_enc` ensures a proportional random split.

---

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
- **What it does:** Imports the TF-IDF Vectorizer.
- **Why TF-IDF?** This converts raw text (URL strings) into a mathematical vector of numbers. "TF" (Term Frequency) is how often a specific word/pattern appears in a URL. "IDF" (Inverse Document Frequency) down-weights terms that appear in almost every URL (like "https") because they carry no real signal. The result is a sparse vector of `25,000` numbers that effectively represents the "vocabulary" of the URL.
- **Alternative:** `CountVectorizer` (raw count of each word) was considered but doesn't down-weight common terms, making it noisier. `HashingVectorizer` is memory-efficient but can't be pickled to disk easily. Word embeddings (Word2Vec, GloVe) are richer but much slower and overkill for short strings like URLs.

---

```python
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
```
- **`SVC`:** Originally imported but we switch to `LinearSVC` later (imported again on line 146). This is a vestigial import that doesn't cause harm.
- **`CalibratedClassifierCV`:** This is critical. `LinearSVC` is a very fast linear model that gives a hard "phishing/not phishing" output — yes or no. But the XGBoost meta-learner needs **probability scores** (e.g., "75% chance of phishing") to make a better final decision. `CalibratedClassifierCV` wraps the SVM and teaches it to output proper probability estimates using cross-validation, effectively giving us both speed and calibrated confidence.

---

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
```
- **`LabelEncoder`:** Converts string class labels (`"Normal"`, `"Phishing"`, `"Injection"`) into numbers (`0`, `1`, `2`). ML models can only work with numbers.
- **`StandardScaler`:** The 13 hand-crafted features have wildly different scales — `len(url)` might be `150`, while `calculate_entropy(url)` is between `0` and `~4`. If you feed these directly to XGBoost, the large-scale features dominate. `StandardScaler` rescales all features to have a **mean of 0 and standard deviation of 1**, putting them on an equal footing.
- **Alternative for scaling:** `MinMaxScaler` normalizes to a [0, 1] range, but `StandardScaler` is more robust to outliers. For tree-based models like XGBoost, scaling is technically less critical than for neural networks (trees split based on rank, not absolute value), but it's added here for consistency with the server-side inference pipeline.

---

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
```
- **TensorFlow + Keras:** This is the entire deep learning framework used to build and train the CNN-BiLSTM neural network.
  - `Sequential`: Defines a model where data flows in one straight line from layer to layer.
  - `Embedding`: Converts token IDs (character numbers) into rich low-dimensional vector representations. For example, the character 'a' gets a learnable 64-dimensional vector.
  - `LSTM`: Long Short-Term Memory — a type of neural network layer that reads a sequence while remembering important context across long distances. Perfect for URLs where the order of characters matters.
  - `Bidirectional`: Wraps the LSTM so it reads the URL both **left-to-right** AND **right-to-left**, giving it twice the contextual understanding.
  - `Dropout`: Randomly "turns off" a fraction of neurons during training to prevent the model from over-relying on specific neurons (a technique to prevent overfitting).
  - `EarlyStopping`: Stops training if the model's performance on the validation set stops improving, preventing over-training.
- **Alternative:** PyTorch (Facebook's framework) is equally powerful but has a lower-level API and is more common in research than production. TensorFlow is chosen here because of its excellent `.keras` format support for easy saving and loading.

---

```python
import xgboost as xgb
```
- **What it does:** Imports the XGBoost library.
- **Why XGBoost?** XGBoost is a Gradient Boosted Decision Tree algorithm — it builds a forest of decision trees sequentially, with each new tree learning to correct the mistakes of the previous ones. It is proven to achieve state-of-the-art performance on structured/tabular data (like our 13-feature vector) even with relatively small datasets.
- **Alternative:** `RandomForest` (scikit-learn) — builds trees in parallel rather than sequentially, which is faster to train but often less accurate. `LightGBM` (Microsoft's version) is even faster than XGBoost and superior for very large datasets, but XGBoost was chosen for its community support and consistent results.

---

## SECTION 2: URL Normalisation (Lines 22–25)

```python
def strip_protocol(url):
    url = re.sub(r'^https?://', '', str(url))
    url = re.sub(r'^www\.', '', url)
    return url.rstrip('/')
```
- **`str(url)`:** Converts the input to a string — protects against NaN (empty) values from pandas which are of type `float`.
- **`re.sub(r'^https?://', '', ...)`:** The `^` anchor means "only match at the start of the string." `https?` matches both `http` and `https`. This strips the protocol prefix.
- **`re.sub(r'^www\.', '', ...)`:** Strips a leading `www.` prefix. The `\.` escapes the dot so it matches a literal `.` rather than "any character."
- **`.rstrip('/')`:** Removes any trailing slash from the end of the URL, so `google.com/` and `google.com` are treated as identical.
- **Why do this?** Two different URLs that point to the same page should not be treated as different by the AI. Normalisation ensures the model learns from the core meaningful content of the URL, not superficial differences.

---

## SECTION 3: High Trust Domains Set (Lines 27–39)

```python
HIGH_TRUST_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', ...
}
```
- **What it does:** Defines a Python `Set` containing over 50 of the most trusted domains on the internet.
- **Why a Set?** Unlike a Python List (where checking if an item exists takes O(n) time — scanning every element), a Set uses a hash table, giving us O(1) constant-time lookups. This is crucial for performance inside tight loops.
- **Why this list?** Phishing attacks most commonly impersonate these brands. Knowing which domains are legitimately allowed to contain brand names (e.g., `google.com` is allowed to contain the word "google") prevents false positives in the brand-spoof detection logic.

---

## SECTION 4: Shannon Entropy (Lines 41–44)

```python
def calculate_entropy(text):
    if not text: return 0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)
```
- **`if not text: return 0`:** Guard clause for empty strings — you can't take the log of zero probabilities.
- **`set(text)`:** Gets the unique characters in the string. If the text is `"aab"`, `set` gives `{'a', 'b'}`.
- **`text.count(c) / len(text)`:** For each unique character, calculates its probability of appearing. For `"aab"`: `p('a') = 2/3`, `p('b') = 1/3`.
- **`-sum(p * math.log2(p) for p in prob)`:** This is the Shannon Entropy formula: **H = -Σ p(x) * log₂(p(x))**. The negative sign makes the result positive.
- **What it measures:** Entropy is a measure of "randomness" or "chaos." A string like `"aaaaaa"` has entropy = 0 (perfectly predictable). A randomly generated phishing domain like `"xk9q3m1.com"` has very high entropy (~3.5), because the characters are nearly random. This is one of the strongest single-feature signals in the entire system.
- **Alternative:** You could use standard deviation of character frequencies, but Shannon Entropy is the mathematically superior metric because it is information-theoretically motivated — it measures exactly how much "surprise" each character adds.

---

## SECTION 5: Feature Extraction (Lines 46–69)

```python
def extract_features(url):
    url = str(url).strip()
    decoded = urllib.parse.unquote(url)
    c = strip_protocol(url)
```
- `str(url).strip()`: Type-safe conversion and whitespace removal.
- `urllib.parse.unquote(url)`: Decodes URL encoding. The URL `%2Fadmin%2Flogin` decodes to `/admin/login`. This is necessary because attackers sometimes double-encode URLs to bypass naive text-matching filters.
- `c = strip_protocol(url)`: Creates a clean version of the URL without `https://` for domain extraction.

```python
    f1 = [
        len(url), len(c), url.count('.'), url.count('/'), url.count('-'), url.count('@'),
        url.count('?'), url.count('='), url.count('&'), int('//' in url[7:]),
        sum(ch.isdigit() for ch in url)/max(len(url),1),
        calculate_entropy(url)
    ]
```
This builds the first 12 of 13 mathematical features:

| Feature | Code | Rationale |
|---|---|---|
| **Full URL Length** | `len(url)` | Phishing URLs are typically very long due to redirection chains |
| **Clean URL Length** | `len(c)` | Same, without the protocol prefix |
| **Dot Count** | `url.count('.')` | Excessive dots indicate deep subdomains (e.g. `secure.login.paypal.com.evil.ru`) |
| **Slash Count** | `url.count('/')` | Long paths can indicate redirection bloat |
| **Hyphen Count** | `url.count('-')` | Hyphens are a classic typosquatting trick: `paypal-secure.com` |
| **@ Symbol Count** | `url.count('@')` | The `@` trick: `bank.com@evil.com` — browser goes to `evil.com` |
| **Query Marker `?`** | `url.count('?')` | Multiple query markers are unusual |
| **Key=Value Pairs `=`** | `url.count('=')` | Many key=value pairs suggest data exfiltration endpoints |
| **Param Separator `&`** | `url.count('&')` | Many & sepearators = many hidden parameters |
| **Hidden `//`** | `int('//' in url[7:])` | A `//` after the protocol position indicates a hidden redirect URL |
| **Digit Ratio** | `sum().../len(url)` | Randomly generated domains have more digits |
| **Shannon Entropy** | `calculate_entropy(url)` | Measures randomness of the URL string |

```python
    domain_part = c.split('/')[0].lower()
    domain_root = '.'.join(domain_part.split('.')[-2:]) if '.' in domain_part else domain_part
    is_brand_spoof = 0
    for b in ['paypal', 'ppl', 'apple', 'microsoft', ...]:
        if b in domain_part and domain_root not in HIGH_TRUST_DOMAINS and domain_root != f"{b}.com":
            is_brand_spoof = 1
            break
    f1.append(is_brand_spoof)
    return np.array(f1).reshape(1, -1)
```
- **`c.split('/')[0].lower()`:** Takes the part before the first `/` to isolate the domain.
- **`domain_root`:** Extracts just the last 2 parts of the domain (`paypal.com` from `secure.login.paypal.com`).
- **Brand spoof check:** If a known brand name appears inside the domain, the root is NOT the trusted domain, and the root is NOT the canonical `brand.com` — it's a spoof. The 13th feature `is_brand_spoof` captures this.
- **`np.array(f1).reshape(1, -1)`:** Converts the Python list into a NumPy row vector with shape `(1, 13)` — the format all scikit-learn models expect for a single prediction.

---

## SECTION 6: Loading Datasets (Lines 71–131)

```python
df_benign_def = pd.read_csv('data/definitive_benign.csv').dropna(subset=['url'])
df_benign_def['label'] = 'Normal'
```
- **`pd.read_csv()`:** Reads the CSV file into a DataFrame.
- **`.dropna(subset=['url'])`:** Drops any rows where the `url` column is empty. An empty URL would crash downstream code and would teach the model nothing.
- **`['label'] = 'Normal'`:** Adds/overwrites a `label` column to unify the class naming across all source datasets.

**Why multiple datasets?**
- Using a single dataset creates a biased model that only recognises the attack patterns familiar to that one collection.
- By combining **6 distinct sources** (Cisco Umbrella, Phishtank, CAPEC, etc.), the model is exposed to a dramatically wider variety of real-world URLs, making it far more robust.

```python
df_capec['label'] = df_capec['category'].apply(label_granulate)
```
- **`.apply()`:** Calls a function on every single row of a column.
- **`label_granulate` function:** The CAPEC dataset has detailed attack categories (`Injection`, `Manipulation`, `Normal`). This function maps them into our unified label format, preserving `Injection` and `Manipulation` as separate classes to make the model more granular. All others default to `'Phishing'`.

```python
df_unseen_train = df_unseen[~df_unseen['url'].isin(test_sample_urls)]
```
- **`~` (tilde):** is the pandas "NOT" operator. This creates the training portion by explicitly *removing* the rows we sampled for testing.
- **Why?** This ensures the "unseen" dataset truly remains unseen for the test. It directly prevents data leakage — the single most important concept in building trustworthy ML models.

```python
df_all = pd.concat([...]).drop_duplicates(subset=['url']).dropna()
```
- **`pd.concat()`:** Stacks all DataFrames on top of each other into one massive dataset.
- **`.drop_duplicates(subset=['url'])`:** Removes rows with identical URLs. Duplicates would unfairly bias the model — a URL that appears 5 times has 5x more influence during training.
- **`.dropna()`:** Final safety pass to remove any remaining rows with missing values.

---

## SECTION 7: Stratified Sampling (Lines 124–133)

```python
df_final = pd.concat([
    df_all[df_all['label'] == 'Normal'].sample(n=min(class_counts['Normal'], 120000), random_state=42),
    df_all[df_all['label'] == 'Phishing'].sample(n=min(class_counts['Phishing'], 80000), random_state=42),
    ...
]).sample(frac=1, random_state=42).reset_index(drop=True)
```
- **`.sample(n=...,  random_state=42)`:** Randomly selects exactly `n` rows from the DataFrame. `random_state=42` is a "seed" value that makes the selection deterministic — every run selects the exact same rows, ensuring **reproducibility**.
- **`min(class_counts['Normal'], 120000)`:** Takes whichever is smaller — the total available data, or our cap of 120,000. This prevents `IndexError` if the dataset is small.
- **Why cap at 120k Normal / 80k Phishing (not 50/50)?** Real-world internet traffic is roughly 80% benign. A 60%/40% split (120k/80k) more closely reflects reality. A perfectly balanced 50/50 dataset would train a model that thinks the internet is equally half-malicious, causing too many false positives on real traffic.
- **`sample(frac=1, ...)`:** `frac=1` means "sample 100% of the rows" — i.e., just shuffle the entire DataFrame. This is critical so the model doesn't see all `Normal` URLs first, then all `Phishing` URLs — which would cause it to train with a temporal bias.
- **`.reset_index(drop=True)`:** After shuffle-sampling, row indices would be scrambled. This reassigns clean sequential indices `0, 1, 2, 3...`.

---

## SECTION 8: TF-IDF + SVM Training (Lines 139–150)

```python
vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1,3))
X_tfidf = vectorizer.transform(texts) if hasattr(vectorizer, 'vocabulary_') else vectorizer.fit_transform(texts)
```
- **`max_features=25000`:** Only the top 25,000 most frequent and distinguishing n-grams are kept.
  - **Why 25,000?** Too few features (e.g., 5,000) means the model misses important patterns. Too many (e.g., 100,000) makes the model very large and slow without a meaningful accuracy gain. 25,000 is a well-established sweet spot from empirical testing.
  - **Alternative:** A large language model (LLM) would automatically learn a richer representation, but would require GPUs for inference and be impossibly slow for real-time URL scanning.
- **`ngram_range=(1,3)`:** The vectorizer considers individual characters/words *and* sequences of 2 and 3 of them.
  - **`1-gram`:** Single tokens. Example: `"paypal"`.
  - **`2-gram`:** Pairs of adjacent tokens. Example: `"paypal-login"`.
  - **`3-gram`:** Triplets. Example: `"paypal-login-secure"`.
  - **Why n-grams?** A 3-gram captures phrase patterns that single tokens miss. The sequence `secure login` is far more suspicious than either word alone.
- **`hasattr(vectorizer, 'vocabulary_')`:** Checks if the vectorizer has already been `fit` (trained). If it has, we only `.transform()` (apply the known vocabulary). If not, we `.fit_transform()` (learn the vocabulary AND apply it simultaneously). This pattern avoids accidentally re-training the vocabulary.

```python
le = LabelEncoder()
y_enc = le.fit_transform(raw_labels)
```
- `LabelEncoder` learns the mapping: `{'Injection': 0, 'Manipulation': 1, 'Normal': 2, 'Phishing': 3}` (alphabetical).
- `y_enc` is now a NumPy array of integers like `[2, 3, 2, 0, 1, 3, ...]`.

```python
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_tfidf, y_enc, test_size=0.1, stratify=y_enc, random_state=42
)
```
- **`test_size=0.1`:** 10% of data is held out for validation. 90% is used for training.
- **`stratify=y_enc`:** Ensures the 10% test set has the same proportion of each class as the full dataset. Without this, by bad luck you might end up with 0 `Injection` examples in your test set.

```python
svm = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=1000, dual=False), cv=3)
svm.fit(X_train_v, y_train_v)
```
- **`LinearSVC`:** A linear Support Vector Classifier, fundamentally a line-drawing algorithm that finds the widest margin between classes in high-dimensional space.
  - **Why `LinearSVC` and not `SVC(kernel='rbf')`?** The RBF kernel SVM scales as O(n²) in training time. With 200,000 samples, this is completely intractable. `LinearSVC` scales far better and achieves near-identical accuracy on text data because text is already well-represented in high-dimensional space.
- **`C=1.0`:** The "regularization" parameter. Higher C → the SVM tries harder to perfectly classify training data (risk of overfitting). Lower C → the SVM allows more misclassifications to find a wider, more generalizable margin. `C=1.0` is the mathematically proven neutral default, offering a good balance.
- **`max_iter=1000`:** Maximum optimization iterations before stopping. For most datasets, 1000 is more than sufficient to converge.
- **`dual=False`:** For datasets where `n_samples > n_features` (we have ~200k samples and 25k features), `dual=False` solves the "primal" optimization problem which is faster.
- **`cv=3`:** The calibration uses 3-fold cross-validation. The dataset is split into 3 equal parts; the model trains on 2 parts and calibrates probabilities on the 3rd, rotating 3 times.

---

## SECTION 9: CNN-BiLSTM Training (Lines 152–171)

```python
tokenizer = Tokenizer(num_words=10000, char_level=True, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
```
- **`char_level=True`:** The tokenizer maps each individual **character** to a number (e.g., `'a'=1`, `'b'=2`, `'/'=47`). This is character-level tokenization.
  - **Why char-level and not word-level?** URLs don't have meaningful English words — they have paths, numbers, and code. Character-level analysis is more appropriate for URL structure analysis. It can detect patterns like `p-a-y-p-a-1.c-o-m` (with a `1` instead of `l`) which word-level tokenization would miss.
- **`num_words=10000`:** Only the 10,000 most common characters/substrings are tracked (in practice, all ASCII characters in URLs is well under 100, so this is a generous ceiling).
- **`oov_token='<OOV>'`:** Any character not seen during training is replaced with the `<OOV>` (Out-Of-Vocabulary) token rather than crashing. This is critical for handling new, exotic phishing domain patterns.

```python
MAX_LEN = 550
sequences = tokenizer.texts_to_sequences(texts)
X_seq = pad_sequences(sequences, maxlen=MAX_LEN)
```
- **`texts_to_sequences`:** Converts each URL string into a list of integers, e.g., `"http://a.com"` → `[4, 5, 5, 3, 1, 1, ...]`.
- **`pad_sequences(..., maxlen=550)`:** Neural networks require all inputs to have the same length. Shorter URLs are padded with zeros at the start; longer URLs are truncated.
- **Why 550?** This was empirically chosen to capture 99th percentile URL length without wasting excessive memory on padding. URLs shorter than 550 chars are zero-padded (harmless); URLs longer than 550 chars are truncated from the left (the critical domain section at the start is preserved).

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

| Layer | Code | Explanation |
|---|---|---|
| **Embedding** | `Embedding(10001, 64, ...)` | Converts integer token IDs into dense 64-dimensional vectors. The model *learns* what each character means. `10001` = vocab size + 1 for the OOV token. |
| **1st BiLSTM** | `Bidirectional(LSTM(64, return_sequences=True))` | Reads the full URL character-by-character in both directions. `return_sequences=True` means it outputs a sequence, not just the final state. |
| **Dropout 0.2** | `Dropout(0.2)` | Randomly zeros out 20% of neurons during training. Prevents co-adaptation and overfitting. |
| **2nd BiLSTM** | `Bidirectional(LSTM(64))` | A second LSTM layer that summarizes the sequence output from the first into a single context vector. |
| **Dense 64 ReLU** | `Dense(64, activation='relu')` | A fully-connected layer that combines the learned sequence features. ReLU (Rectified Linear Unit) is the standard non-linear activation function. |
| **Dropout 0.3** | `Dropout(0.3)` | Stronger 30% dropout before the final classification — avoids the model memorizing training patterns. |
| **Output Layer** | `Dense(len(le.classes_), activation='softmax')` | Outputs a probability for each class. `softmax` ensures all probabilities sum to 1.0. |

```python
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
- **`optimizer='adam'`:** Adam (Adaptive Moment Estimation) is the default optimizer for almost all deep learning tasks. It automatically adjusts the learning rate during training. **Alternative:** `SGD` (Stochastic Gradient Descent) is more mathematically pure but requires manual learning rate tuning. RMSProp is similar to Adam. Adam is chosen for its robust out-of-the-box performance.
- **`loss='sparse_categorical_crossentropy'`:** The loss function measures how wrong the model's predictions are. `sparse_categorical_crossentropy` is used when labels are integers (0, 1, 2...). **Alternative:** `categorical_crossentropy` is used when labels are one-hot encoded vectors.
- **`epochs=15, batch_size=128`:** The model sees the entire dataset 15 times. Each iteration updates the model after processing 128 samples. **Alternative:** More epochs (e.g., 50) would risk overfitting without EarlyStopping.

---

## SECTION 10: XGBoost Meta-Learner (Lines 173–199)

```python
svm_probs = svm.predict_proba(X_tfidf)    # Shape: (n_samples, n_classes)
cnn_probs = cnn.predict(X_seq, batch_size=256)  # Shape: (n_samples, n_classes)
```
- Here we run the entire dataset through both already-trained models to get their **probability predictions** for every URL.
- These predictions become *features* for the XGBoost meta-learner. This is called a **Stacking Ensemble** — the meta-learner learns to combine and correct the opinions of the base models.

```python
hand_feats = []
for u in urls:
    hand_feats.append(extract_features(u).flatten())
hand_feats = np.array(hand_feats)
scaler = StandardScaler()
hand_feats_scaled = scaler.fit_transform(hand_feats)
```
- **`.flatten()`:** Converts the `(1, 13)` array from `extract_features` into a flat `(13,)` array so it can be stacked horizontally.
- **`StandardScaler().fit_transform()`:** Rescales all 13 columns to zero-mean, unit-variance. This is important because XGBoost uses gradient boosting which can be sensitive to feature scales in the learning rate.

```python
X_meta = np.hstack([svm_probs, cnn_probs, hand_feats_scaled])
```
- **`np.hstack()`:** Horizontal Stack — concatenates arrays side-by-side.
- **Result:** For each URL, the meta-learner receives a combined feature vector of:
  - `n_classes` SVM probability scores
  - `n_classes` CNN probability scores
  - `13` hand-crafted features
  - Total: `n_classes * 2 + 13` features per URL

```python
sample_weights = np.ones(len(y_enc))
for i, label in enumerate(raw_labels):
    if label != 'Normal':
        sample_weights[i] = 2.5
```
- **What this does:** Creates a weight array. All weights start at `1.0`. Any URL that is malicious gets its weight set to `2.5`.
- **Why?** This is the **Recall Penalty Strategy**. When XGBoost makes an error, the loss for missing a *malicious* URL is 2.5x more painful than incorrectly labelling a benign URL. This forces the model to prioritize **never missing a threat** (high Recall) over being perfectly precise.
- **Why 2.5x?** Through empirical testing, this factor maximized Recall while keeping Precision above the 98% production threshold. Values below 1.5 didn't improve recall enough; values above 3.0 collapsed precision too dramatically.

```python
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.05, tree_method='hist', device='cuda'
)
```

| Parameter | Value | Rationale |
|---|---|---|
| **`n_estimators=300`** | 300 trees | The number of decision trees in the ensemble. More trees = more expressions of the data. 300 is a balance between accuracy and training time. **Alternative:** 100 is faster but noticeably less accurate; 500+ shows diminishing returns. |
| **`max_depth=8`** | 8 levels | The maximum complexity of each tree. Deeper trees learn more intricate patterns but risk overfitting. 8 is deep enough for URL features. **Alternative:** `max_depth=6` is the XGBoost default; 10+ risks overfitting. |
| **`learning_rate=0.05`** | 0.05 | How much each new tree corrects the previous one. Lower rate = more careful learning, requires more trees. `0.05` paired with 300 trees gives optimal learning. **Alternative:** `0.1` trains faster but is more aggressive; `0.01` requires thousands of trees. |
| **`tree_method='hist'`** | Histogram | Faster algorithm for building trees. Bins continuous values into discrete buckets for speed. **Alternative:** `'exact'` is perfectly precise but extremely slow for > 100k samples. |
| **`device='cuda'`** | GPU | Trains the XGBoost model on the NVIDIA GPU for dramatically faster training. **Alternative:** `device='cpu'` works everywhere but can be 5-10x slower. |

---

## SECTION 11: Saving Models (Lines 201–209)

```python
joblib.dump(vectorizer, 'models/local_vectorizer.pkl')
joblib.dump(svm, 'models/local_svm_model.pkl')
cnn.save('models/local_hybrid_model.keras')
joblib.dump(tokenizer, 'models/local_tokenizer.pkl')
joblib.dump(scaler, 'models/local_url_scaler.pkl')
joblib.dump(xgb_model, 'models/local_meta_learner_global.pkl')
joblib.dump(le, 'models/local_label_encoder.pkl')
```

Each of the **7 model components** is saved independently:

| File | Library | What it contains |
|---|---|---|
| `local_vectorizer.pkl` | joblib | The TF-IDF vocabulary and IDF weights learned from training data |
| `local_svm_model.pkl` | joblib | The trained SVM + probability calibration weights |
| `local_hybrid_model.keras` | Keras | The full CNN-BiLSTM neural network (architecture + weights) |
| `local_tokenizer.pkl` | joblib | The character-to-integer mapping |
| `local_url_scaler.pkl` | joblib | The StandardScaler mean/variance for the 13 hand features |
| `local_meta_learner_global.pkl` | joblib | The trained XGBoost tree ensemble |
| `local_label_encoder.pkl` | joblib | The class label ↔ integer mapping |

- **Why `.pkl` (pickle format)?** Joblib-pickle files are the native serialization format for scikit-learn objects. They preserve the full state of the Python object so it can be loaded and called identically on a different machine.
- **Why `.keras` for the CNN?** The `.keras` format is TensorFlow's native format. It saves both the architecture (the layers) AND the weights (what the model learned) in a single file that can be loaded with one line: `load_model('local_hybrid_model.keras')`.
- **Why save them separately?** Each component needs to be loaded independently by the server during inference. Saving them separately also makes it easy to replace one model (e.g., retrain only the XGBoost) without re-loading all others.

---

> This document covers every line, parameter, and design decision in `train.py`. The core philosophy is: **use multiple strong, diverse models and combine their signals intelligently.** The SVM catches keyword patterns, the CNN-BiLSTM catches structural character-level patterns, and XGBoost is the intelligent arbiter that weighs them all.
