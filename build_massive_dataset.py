import pandas as pd
import random
import os

print("1. Loading Phishing Data...")
df_phish1 = pd.read_csv("data/verified_online.csv")
urls_phish1 = df_phish1['url'].astype(str).tolist()

df_phish2 = pd.read_csv("data/urlhaus_recent.csv", comment='#', quotechar='"', names=['id', 'dateadded', 'url', 'url_status', 'last_online', 'threat', 'tags', 'urlhaus_link', 'reporter'])
urls_phish2 = df_phish2['url'].astype(str).tolist()

phish_urls = list(set(urls_phish1 + urls_phish2))
print(f"Total Unique Phishing URLs: {len(phish_urls)}")

print("2. Loading Benign Data (Modern Benign)...")
df_benign = pd.read_csv("data/modern_benign_dataset.csv")
benign_urls = df_benign['url'].dropna().tolist()

benign_urls = list(set(benign_urls))[:len(phish_urls)]
print(f"Total Unique Benign URLs: {len(benign_urls)}")

print("3. Building Massive Dataset...")
df_massive = pd.DataFrame({
    'url': benign_urls + phish_urls,
    'label': ['Normal'] * len(benign_urls) + ['Phishing'] * len(phish_urls)
})
df_massive = df_massive.sample(frac=1, random_state=42).reset_index(drop=True)
df_massive.to_csv("data/massive_train.csv", index=False)
print(f"✅ Saved data/massive_train.csv with {len(df_massive)} total URLs.")
