import pandas as pd
from datasets import load_dataset
import numpy as np

def download_and_prepare_data():
    print("Loading dataset 'tblard/allocine'...")
    dataset = load_dataset("tblard/allocine")
    
    np.random.seed(42)
    
    # Train: 1600 samples (800 pos, 800 neg)
    train_df = pd.DataFrame(dataset['train'])
    train_pos = train_df[train_df['label'] == 1].sample(800, random_state=42)
    train_neg = train_df[train_df['label'] == 0].sample(800, random_state=42)
    train_split = pd.concat([train_pos, train_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_split['split'] = 'train'
    
    # Val: 400 samples (200 pos, 200 neg)
    val_df = pd.DataFrame(dataset['validation'])
    val_pos = val_df[val_df['label'] == 1].sample(200, random_state=42)
    val_neg = val_df[val_df['label'] == 0].sample(200, random_state=42)
    val_split = pd.concat([val_pos, val_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_split['split'] = 'validation'
    
    # Test: 500 samples (250 pos, 250 neg)
    test_df = pd.DataFrame(dataset['test'])
    test_pos = test_df[test_df['label'] == 1].sample(250, random_state=42)
    test_neg = test_df[test_df['label'] == 0].sample(250, random_state=42)
    test_split = pd.concat([test_pos, test_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_split['split'] = 'test'
    
    full_df = pd.concat([train_split, val_split, test_split]).reset_index(drop=True)
    full_df.rename(columns={'review': 'text'}, inplace=True)
    
    full_df.to_csv("data.csv", index=False)
    print("Saved dataset to data.csv with shape:", full_df.shape)
    print("Split distribution:")
    print(full_df.groupby(['split', 'label']).size())

if __name__ == "__main__":
    download_and_prepare_data()
