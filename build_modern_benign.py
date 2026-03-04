"""
Build a high-quality MODERN benign URL dataset from:
  1. Tranco top-50k domains (fresh 2024 rankings)
  2. Majestic Million top-10k
  3. Common real-world URL patterns for popular sites

This teaches the model what current legitimate URLs look like,
dramatically reducing false positives on modern benign traffic.
"""
import pandas as pd
import numpy as np
import zipfile
import random
import re

random.seed(42)

# --- 1. Load Tranco top-50k domains ------------------------------------------
print("📥 Loading Tranco top-50k...")
with zipfile.ZipFile('tranco_top1m.zip') as z:
    with z.open(z.namelist()[0]) as f:
        df_tranco = pd.read_csv(f, header=None, names=['rank', 'domain'])

top50k = df_tranco[df_tranco['rank'] <= 50000]['domain'].tolist()
print(f"   {len(top50k)} Tranco domains loaded")

# --- 2. Load Majestic Million top-10k -----------------------------------------
print("📥 Loading Majestic Million top-10k...")
df_maj = pd.read_csv('majestic_million.csv')
top10k_maj = df_maj[df_maj['GlobalRank'] <= 10000]['Domain'].tolist()
print(f"   {len(top10k_maj)} Majestic domains loaded")

# Combine unique domains
all_trusted_domains = list(set(top50k + top10k_maj))
print(f"   Combined: {len(all_trusted_domains)} unique trusted domains")

# --- 3. Generate realistic benign URLs ---------------------------------------
print("🔨 Generating realistic benign URLs...")

# Common benign URL patterns for top domains
COMMON_PATHS = [
    '', '/', '/about', '/contact', '/home', '/index.html',
    '/products', '/services', '/blog', '/news', '/search?q=hello+world',
    '/login', '/register', '/shop', '/cart', '/faq',
    '/privacy-policy', '/terms-of-service', '/help',
    '/category/technology', '/article/2024/latest-news',
    '/en/home', '/us/en/index.html', '/docs/getting-started',
    '/images/logo.png', '/assets/main.css', '/static/bundle.js',
    '/api/v1/status', '/feeds/rss.xml',
    '/profile/john-doe', '/user/settings',
    '/videos/watch?id=abc123', '/gallery',
    '/download/report.pdf', '/files/brochure.pdf',
]

# Common query params that are benign
COMMON_QUERIES = [
    '', '?page=1', '?page=2', '?sort=price', '?filter=new',
    '?lang=en', '?ref=homepage', '?utm_source=google',
    '?id=12345', '?category=electronics', '?q=laptop',
    '?from=email&campaign=newsletter', '?tab=overview',
    '?search=shoes&color=blue', '?v=2024',
]

# HTTPS (modern) biased 80-20
def make_url(domain):
    scheme = 'https' if random.random() > 0.15 else 'http'
    path = random.choice(COMMON_PATHS)
    query = random.choice(COMMON_QUERIES) if '?' not in path else ''
    return f"{scheme}://{domain}{path}{query}"

# Generate 120k modern benign URLs from trusted domains
modern_benign = []
for _ in range(120000):
    domain = random.choice(all_trusted_domains)
    modern_benign.append(make_url(domain))

# Remove duplicates
modern_benign = list(set(modern_benign))[:100000]
print(f"   Generated {len(modern_benign)} unique modern benign URLs")

# --- 4. Sample from existing merged_urls benign (keep some for diversity) ----
print("📥 Sampling existing benign pool...")
df_merged = pd.read_csv('merged_urls_dataset.csv').dropna(subset=['url', 'label'])
existing_benign = df_merged[df_merged['label'] == 'benign']['url'].sample(
    n=30000, random_state=42
).tolist()
print(f"   {len(existing_benign)} existing benign URLs")

# --- 5. Save combined modern benign dataset ----------------------------------
all_benign = pd.DataFrame({
    'url': modern_benign + existing_benign,
    'label': 'Normal'
}).drop_duplicates(subset=['url'])

# Shuffle
all_benign = all_benign.sample(frac=1, random_state=42).reset_index(drop=True)

out_path = 'modern_benign_dataset.csv'
all_benign.to_csv(out_path, index=False)
print(f"\n✅ Saved {len(all_benign)} modern benign URLs → {out_path}")
print(f"   Sample URLs:")
for url in all_benign['url'].head(10).tolist():
    print(f"     {url}")
