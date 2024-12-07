# src/data.py

import pandas as pd
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
from conceptnet_integration import final_get_conceptnet_embedding

def final_sharedprivate_load_and_split_data(csv_file_path, reduce_factor=1, focus_humor_types=None):
    """
    Loads and splits the dataset for shared-private architecture.

    Parameters:
        csv_file_path (str): Path to the CSV dataset.
        reduce_factor (float): Factor by which to reduce the dataset size (default is 1, no reduction).
        focus_humor_types (list, optional): List of humor types to include. If None, include all.

    Returns:
        train_df, val_df, test_df, humor_type_to_idx: Training, validation, and test DataFrames, 
        and mapping of humor types to indices.
    """
    print("[INFO] Loading dataset from CSV...")
    # Read the CSV file
    final_df = pd.read_csv(csv_file_path)
    print(f"[INFO] Dataset loaded with {len(final_df)} samples.")

    # Optionally filter for specific humor types
    if focus_humor_types is not None:
        final_df = final_df[final_df['type'].isin(focus_humor_types)]
        print(f"[INFO] Filtered dataset to include only humor types: {focus_humor_types}")
        print(f"[INFO] Filtered dataset now has {len(final_df)} samples.")

    # Map humor types to indices
    final_humor_types = final_df['type'].unique()
    final_humor_type_to_idx = {humor_type: idx for idx, humor_type in enumerate(final_humor_types)}
    print(f"[INFO] Humor types and their indices: {final_humor_type_to_idx}")

    # Encode humor_type to indices
    final_df['humor_type_idx'] = final_df['type'].map(final_humor_type_to_idx)

    # Create a combined stratification column
    final_df['stratify_col'] = final_df['label'].astype(str) + "_" + final_df['humor_type_idx'].astype(str)

    # Split the data into training, validation, and test sets
    print("[INFO] Splitting dataset into training, validation, and test sets...")
    final_train_df, final_temp_df = train_test_split(
        final_df,
        test_size=0.3,
        stratify=final_df['stratify_col'],
        random_state=42
    )
    final_val_df, final_test_df = train_test_split(
        final_temp_df,
        test_size=0.5,
        stratify=final_temp_df['stratify_col'],
        random_state=42
    )
    print(f"[INFO] Training set: {len(final_train_df)} samples")
    print(f"[INFO] Validation set: {len(final_val_df)} samples")
    print(f"[INFO] Test set: {len(final_test_df)} samples")

    # Reduce the dataset size if requested
    if reduce_factor > 1:
        final_train_df = final_train_df.sample(n=math.floor(len(final_train_df) / reduce_factor), random_state=42)
        final_val_df = final_val_df.sample(n=math.floor(len(final_val_df) / reduce_factor), random_state=42)
        final_test_df = final_test_df.sample(n=math.floor(len(final_test_df) / reduce_factor), random_state=42)
        print(f"[INFO] Reduced dataset size by a factor of {reduce_factor}.")

    # Drop the stratify_col as it's no longer needed
    final_train_df = final_train_df.drop(columns=['stratify_col'])
    final_val_df = final_val_df.drop(columns=['stratify_col'])
    final_test_df = final_test_df.drop(columns=['stratify_col'])

    return final_train_df, final_val_df, final_test_df, final_humor_type_to_idx

class final_HumorDataset(Dataset):
    def __init__(self, df, tokenizer_name, max_length):
        """
        Initializes the HumorDataset.

        Parameters:
            df (pd.DataFrame): DataFrame containing the dataset.
            tokenizer_name (str): Name of the BERT tokenizer to use.
            max_length (int): Maximum sequence length for tokenization.
        """
        print("[INFO] Initializing final_HumorDataset...")
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.humor_types = df['humor_type_idx'].tolist()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        print("[INFO] final_HumorDataset initialized successfully.")

    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        final_text = str(self.texts[idx])
        final_label = self.labels[idx]
        final_humor_type_idx = self.humor_types[idx]
        
        # Tokenization
        final_encoding = self.tokenizer.encode_plus(
            final_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # ConceptNet Embedding
        final_concept_embedding = final_get_conceptnet_embedding(final_text)  # [embed_dim]
        
        return {
            'input_ids': final_encoding['input_ids'].squeeze(),           # [max_length]
            'attention_mask': final_encoding['attention_mask'].squeeze(), # [max_length]
            'token_type_ids': final_encoding['token_type_ids'].squeeze(), # [max_length]
            'labels': torch.tensor(final_label, dtype=torch.long),
            'humor_type': torch.tensor(final_humor_type_idx, dtype=torch.long),  # For private BERT selection
            'concept_embeddings': final_concept_embedding  # [embed_dim]
        }
