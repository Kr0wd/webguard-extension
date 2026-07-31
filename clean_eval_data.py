import pandas as pd
import os

print("🧹 Starting dataset cleaning...")

# Load datasets
df_benign = pd.read_csv('data/definitive_benign.csv')
df_malicious = pd.read_csv('data/definitive_malicious.csv')

print(f"Original sizes -> Benign: {len(df_benign)}, Malicious: {len(df_malicious)}")

# 1. Resolve Data Duplicates (Drop exact duplicates within each class)
df_benign = df_benign.drop_duplicates(subset=['url'])
df_malicious = df_malicious.drop_duplicates(subset=['url'])

# 2. Resolve Conflicting Labels (Drop URLs that appear in BOTH benign and malicious)
benign_urls = set(df_benign['url'].astype(str))
malicious_urls = set(df_malicious['url'].astype(str))
conflicting_urls = benign_urls.intersection(malicious_urls)

if conflicting_urls:
    print(f"🚨 Found {len(conflicting_urls)} conflicting URLs (labeled as BOTH normal and phishing). Dropping them from both sets...")
    df_benign = df_benign[~df_benign['url'].isin(conflicting_urls)]
    df_malicious = df_malicious[~df_malicious['url'].isin(conflicting_urls)]

# 3. Resolve String Length Out of Bounds (Drop URLs > 550 chars)
df_benign = df_benign[df_benign['url'].str.len() <= 550]
df_malicious = df_malicious[df_malicious['url'].str.len() <= 550]

print(f"Cleaned sizes -> Benign: {len(df_benign)}, Malicious: {len(df_malicious)}")

# Save cleaned datasets
df_benign.to_csv('data/definitive_benign.csv', index=False)
df_malicious.to_csv('data/definitive_malicious.csv', index=False)

print("✅ Data cleaning complete. Deepchecks issues resolved!")
