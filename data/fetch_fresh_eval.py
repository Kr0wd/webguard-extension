import pandas as pd
from datasets import load_dataset
import os

print("🚀 Attempting to fetch 'm-v-p/malicious-urls' for fresh test...")

try:
    dataset = load_dataset("m-v-p/malicious-urls", split="train")
    df = dataset.to_pandas()
    
    print(f"Columns: {df.columns.tolist()}")
    
    # Common mapping for m-v-p/malicious-urls: 
    # Usually it's URL and label.
    if 'url' in df.columns and 'label' in df.columns:
        # Check label types
        print(f"Unique Labels Sample: {df['label'].unique()[:5]}")
        # Assuming label=0 is benign, 1+ is malicious
        df['label'] = df['label'].apply(lambda x: 0 if str(x) == '0' else 1)
        
    output_path = "data/fresh_verification_unseen.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Successfully saved {len(df)} URLs to {output_path}")
    
except Exception as e:
    print(f"❌ Error during download: {e}")
