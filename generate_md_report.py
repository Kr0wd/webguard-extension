import pandas as pd
import urllib.parse
from deepchecks.tabular.suites import data_integrity
from deepchecks.tabular import Dataset
import os

print("📂 Loading data...")
df_benign = pd.read_csv('data/definitive_benign.csv').sample(n=1000, random_state=42)
df_benign['label'] = 'Normal'
df_malicious = pd.read_csv('data/definitive_malicious.csv').sample(n=1000, random_state=42)
df_malicious['label'] = 'Phishing'
df_combined = pd.concat([df_benign, df_malicious]).sample(frac=1, random_state=42)

urls = df_combined['url'].astype(str).tolist()
def strip_protocol(u):
    if u.startswith('http://'): return u[7:]
    if u.startswith('https://'): return u[8:]
    return u
processed_texts = [strip_protocol(urllib.parse.unquote(u)) for u in urls]

print("🧠 Reconstructing Meta-Features...")
import joblib
vectorizer = joblib.load('models/local_vectorizer.pkl')
svm = joblib.load('models/local_svm_model.pkl')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cnn_model = tf.keras.models.load_model('models/local_hybrid_model.keras')
tokenizer = joblib.load('models/local_tokenizer.pkl')

X_tfidf = vectorizer.transform(processed_texts)
svm_probs = svm.predict_proba(X_tfidf)[:, 1]

X_seq = tokenizer.texts_to_sequences(processed_texts)
X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen=550)
cnn_probs = cnn_model.predict(X_pad, verbose=0).flatten()

df_features = pd.DataFrame({
    'url': df_combined['url'].values,
    'svm_phish_prob': svm_probs,
    'cnn_phish_prob': cnn_probs,
    'url_length': [len(u) for u in processed_texts],
    'num_digits': [sum(c.isdigit() for c in u) for u in processed_texts],
    'num_special': [sum(not c.isalnum() for c in u) for u in processed_texts],
    'label': df_combined['label'].values
})

ds = Dataset(df_features, label='label', index_name='url')
print("🚀 Running Deepchecks...")
suite = data_integrity()
result = suite.run(ds)

print("📝 Generating Markdown Report...")
with open('docs/deepchecks_detailed_report.md', 'w') as f:
    f.write("# WebGuard Deepchecks Detailed Report\n\n")
    f.write(f"**Overall Status:** {'✅ PASSED' if result.passed() else '⚠️ ISSUES DETECTED'}\n\n")
    f.write("## 🔍 Check Details\n\n")
    
    for check_result in result.get_not_passed_checks():
        f.write(f"### ❌ {check_result.check.name()}\n")
        f.write(f"**Description:** {check_result.check.description()}\n")
        f.write(f"**Value:** {check_result.value}\n\n")
        
    for check_result in result.get_passed_checks():
        f.write(f"### ✅ {check_result.check.name()}\n")
        f.write(f"**Description:** {check_result.check.description()}\n")
        f.write(f"**Value:** {check_result.value}\n\n")

print("✅ Markdown generated at docs/deepchecks_detailed_report.md")
