import pandas as pd
from datasets import load_dataset

print("Downloading 'pirocheto/phishing-url' from HuggingFace...")
try:
    dataset = load_dataset("pirocheto/phishing-url", split="train")
    df = dataset.to_pandas()
    print("Columns:", df.columns.tolist())
    print("Label distribution:\n", df['status'].value_counts() if 'status' in df.columns else df.head())
    
    # Clean up and save to a new CSV
    if 'url' in df.columns and 'status' in df.columns:
        df = df[['url', 'status']].rename(columns={'status': 'label'})
        # 'legitimate' -> 'Normal', 'phishing' -> 'Phishing'
        df['label'] = df['label'].apply(lambda x: 'Normal' if x == 'legitimate' else 'Phishing')
        
    df.to_csv("new_unseen_dataset.csv", index=False)
    print(f"Successfully saved {len(df)} unseen URLs to 'new_unseen_dataset.csv'")
except Exception as e:
    print(f"Error downloading dataset: {e}")
