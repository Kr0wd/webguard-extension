import pandas as pd

print("1. Compiling Real Unseen Dataset from definitive data...")
# Load definitive datasets (these were NOT used in massive_train.csv)
df_benign = pd.read_csv('data/definitive_benign.csv').dropna()
if 'label' not in df_benign.columns:
    df_benign['label'] = 0

df_malicious = pd.read_csv('data/definitive_malicious.csv').dropna()
if 'label' not in df_malicious.columns:
    df_malicious['label'] = 1

# If columns differ, align them (just need url and label)
if 'url' not in df_benign.columns:
    df_benign = df_benign.rename(columns={df_benign.columns[0]: 'url'})
if 'url' not in df_malicious.columns:
    df_malicious = df_malicious.rename(columns={df_malicious.columns[0]: 'url'})

# Sample 2000 of each
b_sample = df_benign[['url', 'label']].sample(2000, random_state=42)
m_sample = df_malicious[['url', 'label']].sample(2000, random_state=42)

df_unseen = pd.concat([b_sample, m_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
df_unseen['label'] = df_unseen['label'].apply(lambda x: 1 if str(x).lower() == 'phishing' or str(x) == '1' else 0)

df_unseen.to_csv('data/real_unseen_dataset.csv', index=False)
print(f"✅ Saved data/real_unseen_dataset.csv with {len(df_unseen)} real-world URLs.")
