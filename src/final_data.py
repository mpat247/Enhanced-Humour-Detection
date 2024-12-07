# File: src/data.py

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

class HumorDataset(Dataset):
    """
    PyTorch Dataset for Humor Classification.
    """
    def __init__(self, dataframe, tokenizer_name, max_length):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.humor_types = dataframe['type'].tolist()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        # Create a mapping from humor types to indices
        self.humor_type_to_idx = {ht: idx for idx, ht in enumerate(sorted(set(self.humor_types)))}
        self.indices = [self.humor_type_to_idx[ht] for ht in self.humor_types]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        humor_type_idx = self.indices[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'humor_type': torch.tensor(humor_type_idx, dtype=torch.long)
        }

def sharedprivate_load_and_split_data(csv_file_path, test_size=0.1, val_size=0.1, random_state=42):
    """
    Loads the dataset from a CSV file and splits it into training, validation, and test sets.
    
    Parameters:
        csv_file_path (str): Path to the CSV dataset file.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Seed used by the random number generator.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]: 
            Training, validation, test DataFrames and a mapping from humor types to indices.
    """
    print("[INFO] Loading dataset for splitting...")
    df = pd.read_csv(csv_file_path)
    print(f"[INFO] Original dataset size: {len(df)}")
    
    # Encode labels if necessary
    if 'label' not in df.columns:
        df['label'] = df['type'].astype('category').cat.codes
    
    # Split into train and temp (val + test)
    train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), stratify=df['label'], random_state=random_state)
    relative_val_size = val_size / (test_size + val_size)
    
    # Split temp into validation and test
    val_df, test_df = train_test_split(temp_df, test_size=test_size, stratify=temp_df['label'], random_state=random_state)
    
    # Create humor type to index mapping
    humor_type_to_idx = {ht: idx for idx, ht in enumerate(sorted(df['type'].unique()))}
    
    print(f"[INFO] Training set size: {len(train_df)}")
    print(f"[INFO] Validation set size: {len(val_df)}")
    print(f"[INFO] Test set size: {len(test_df)}")
    
    return train_df, val_df, test_df, humor_type_to_idx
