import pandas as pd

print("1. Loading all definitive datasets...")
df_benign = pd.read_csv('data/definitive_benign.csv').dropna()
if 'label' not in df_benign.columns:
    df_benign['label'] = 0
if 'url' not in df_benign.columns:
    df_benign = df_benign.rename(columns={df_benign.columns[0]: 'url'})

df_malicious = pd.read_csv('data/definitive_malicious.csv').dropna()
if 'label' not in df_malicious.columns:
    df_malicious['label'] = 1
if 'url' not in df_malicious.columns:
    df_malicious = df_malicious.rename(columns={df_malicious.columns[0]: 'url'})

print("2. Identifying previously seen testing URLs...")
try:
    df_seen = pd.read_csv('data/real_unseen_dataset.csv')
    seen_urls = set(df_seen['url'].tolist())
except Exception:
    seen_urls = set()

print(f"Loaded {len(seen_urls)} previously tested URLs to exclude.")

# Filter out the seen ones
df_benign_new = df_benign[~df_benign['url'].isin(seen_urls)]
df_malicious_new = df_malicious[~df_malicious['url'].isin(seen_urls)]

print(f"Remaining Benign: {len(df_benign_new)}")
print(f"Remaining Malicious: {len(df_malicious_new)}")

# Balance the datasets to the minimum size
min_size = min(len(df_benign_new), len(df_malicious_new))
b_sample = df_benign_new[['url', 'label']].sample(min_size, random_state=99)
m_sample = df_malicious_new[['url', 'label']].sample(min_size, random_state=99)

df_unseen_2 = pd.concat([b_sample, m_sample]).sample(frac=1, random_state=99).reset_index(drop=True)
df_unseen_2['label'] = df_unseen_2['label'].apply(lambda x: 1 if str(x).lower() == 'phishing' or str(x) == '1' else 0)

df_unseen_2.to_csv('data/real_unseen_dataset_2.csv', index=False)
print(f"✅ Saved data/real_unseen_dataset_2.csv with {len(df_unseen_2)} NEW real-world URLs.")
