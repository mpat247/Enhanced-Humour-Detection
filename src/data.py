# src/data.py

import pandas as pd
import os
# from datasets import Dataset
from transformers import BertTokenizer
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import math

# Initialize the tokenizer once to avoid re-initialization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_and_preprocess(file_path, dataset_name="Dataset"):
    """
    Loads and preprocesses a TSV file containing only the 'text' column.
    For the training dataset, it extracts the label from the 'text' column
    and removes rows with a label of 0 (non-humorous). It then samples the
    required number of rows and converts the DataFrame to a Hugging Face Dataset.

    Parameters:
    - file_path (str): Path to the TSV file.
    - dataset_name (str): Name of the dataset (for logging purposes).

    Returns:
    - Dataset: Hugging Face Dataset object ready for tokenization.
    """
    
    # Check if the file exists before proceeding
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{dataset_name} file not found: {file_path}")
    
    # Load the .tsv file
    try:
        # Assuming there is a header row with 'text' as the column name
        df = pd.read_csv(file_path, sep="\t", header=0, usecols=["text"], encoding="utf-8")
        print(f"Successfully loaded {dataset_name} from {file_path}")
    except ValueError as ve:
        # This error occurs if 'text' column is not found
        raise ValueError(f"{dataset_name} is missing the 'text' column.") from ve
    except Exception as e:
        raise RuntimeError(f"Error loading {dataset_name}: {e}") from e
    
    # Drop rows with missing text
    initial_count = len(df)
    df.dropna(subset=["text"], inplace=True)
    
    # Remove leading/trailing whitespace
    df["text"] = df["text"].str.strip()
    
    # Drop rows where text is empty after stripping
    df = df[df["text"].astype(bool)]
    final_count = len(df)
    dropped_rows = initial_count - final_count
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} empty or missing rows from {dataset_name}.")
    
    # Process training data
    if "train" in file_path.lower():
        # Extract the label from the 'text' column
        # Assuming the format is: "id,label,category,joke_text"
        # We'll extract the label which is the second element after splitting by ','
        def extract_label(text):
            try:
                return int(text.split(',', 2)[1])
            except (IndexError, ValueError):
                # If splitting fails or label is not an integer, return None
                return None
        
        # Apply the extraction function to create a new 'label' column
        df["label"] = df["text"].apply(extract_label)
        
        # Drop rows where label extraction failed
        before_label_count = len(df)
        df = df.dropna(subset=["label"])
        after_label_count = len(df)
        dropped_label_rows = before_label_count - after_label_count
        if dropped_label_rows > 0:
            print(f"Dropped {dropped_label_rows} rows due to failed label extraction from {dataset_name}.")
        
        # Convert 'label' column to integer type
        df["label"] = df["label"].astype(int)
        
        # Remove rows with label 0 (non-humorous)
        humorous_count = len(df[df["label"] == 1])
        non_humorous_count = len(df[df["label"] == 0])
        print(f"{dataset_name} contains {humorous_count} humorous and {non_humorous_count} non-humorous jokes before filtering.")
        
        # Keep only humorous jokes
        df = df[df["label"] == 1]
        after_filter_count = len(df)
        filtered_out = humorous_count + non_humorous_count - after_filter_count
        if filtered_out > 0:
            print(f"Filtered out {filtered_out} non-humorous jokes from {dataset_name}.")
        
        # Remove the label and preceding metadata from the 'text' column, keeping only the joke text
        # Assuming the format is "id,label,category,joke_text"
        def extract_joke_text(text):
            try:
                return text.split(',', 3)[-1].strip()
            except IndexError:
                # If splitting fails, return the original text
                return text
        
        df["text"] = df["text"].apply(extract_joke_text)
        
        # Sample 75,000 rows from the training dataset
        if len(df) < 75000:
            raise ValueError(f"Training dataset has only {len(df)} humorous jokes, which is less than the required 75,000.")
        df = df.sample(n=75000, random_state=42)
        print(f"Sampled 75,000 humorous jokes from {dataset_name}.")
    
    # Process testing data
    elif "test" in file_path.lower():
        # Sample 7,500 rows from the testing dataset
        if len(df) < 7500:
            raise ValueError(f"Testing dataset has only {len(df)} rows, which is less than the required 7,500.")
        df = df.sample(n=7500, random_state=42)
        print(f"Sampled 7,500 jokes from {dataset_name}.")
    
    else:
        print(f"No specific processing defined for {dataset_name}. Proceeding with the loaded data.")
    
    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df.reset_index(drop=True))
    print(f"Converted {dataset_name} to Hugging Face Dataset with {len(dataset)} samples.")
    
    return dataset

def tokenize_and_mask(dataset):
    """
    Tokenizes and masks the input texts for Masked Language Modeling (MLM) using BERT.
    
    Parameters:
    - dataset (Dataset): Hugging Face Dataset object containing the 'text' column.
    
    Returns:
    - Dataset: Tokenized and masked Hugging Face Dataset ready for training.
    """
    
    def tokenize_batch(batch):
        """
        Tokenizes and applies masking to a batch of texts.
        
        Parameters:
        - batch (dict): A batch from the dataset containing the 'text' key.
        
        Returns:
        - dict: A dictionary with tokenized inputs and corresponding labels.
        """
        # Tokenize the text
        encoding = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=64,
            return_attention_mask=True
        )
        
        # Convert input_ids to PyTorch tensors
        input_ids = torch.tensor(encoding["input_ids"])
        labels = input_ids.clone()
        
        # Create a random mask for 15% of the tokens
        rand = torch.rand(input_ids.shape)
        # Ensure that special tokens are not masked
        mask_arr = (rand < 0.15) & \
                   (input_ids != tokenizer.cls_token_id) & \
                   (input_ids != tokenizer.sep_token_id) & \
                   (input_ids != tokenizer.pad_token_id)
        
        # Replace 15% of the tokens with [MASK]
        input_ids[mask_arr] = tokenizer.mask_token_id
        
        # Set labels for non-masked tokens to -100 so they are ignored in loss computation
        labels[~mask_arr] = -100  # -100 is the default ignore index in PyTorch
        
        # Convert tensors back to lists for Hugging Face Datasets compatibility
        encoding["input_ids"] = input_ids.tolist()
        encoding["attention_mask"] = encoding["attention_mask"]  # already list
        encoding["labels"] = labels.tolist()
        
        return encoding
    
    # Apply the tokenize_batch function to the dataset
    tokenized_dataset = dataset.map(tokenize_batch, batched=True, batch_size=1000)
    
    print(f"Tokenization and masking completed. Dataset now has features: {tokenized_dataset.features}")
    
    return tokenized_dataset

def shared_load_and_preprocess_classification_data(file_path):
    """
    Loads and preprocesses the classification dataset, ensuring an equal number of samples for each label
    and a total of 150,000 samples.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - dataset (Dataset): Preprocessed Hugging Face Dataset with 'text' and 'labels' columns.
    - label_encoder (LabelEncoder): Fitted LabelEncoder instance for labels.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Select only 'text' and 'label' columns
    if not {'text', 'label'}.issubset(df.columns):
        raise ValueError("CSV file must contain 'text' and 'label' columns.")
    
    df = df[['text', 'label']]

    # Clean the 'text' column
    df['text'] = df['text'].astype(str).str.strip()

    # Check if there are enough samples for both labels
    label_counts = df['label'].value_counts()
    print(f"Label distribution before sampling:\n{label_counts}")

    if any(label_counts < 75000):
        raise ValueError("Not enough samples to create a balanced dataset with 150,000 rows (75,000 per label).")

    # Sample 75,000 rows for each label
    balanced_df = pd.concat([
        df[df['label'] == 0].sample(n=75000, random_state=42),
        df[df['label'] == 1].sample(n=75000, random_state=42)
    ])

    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced dataset created with {len(balanced_df)} samples.")
    print(f"Label distribution after balancing:\n{balanced_df['label'].value_counts()}")

    # Encode labels
    label_encoder = LabelEncoder()
    balanced_df['labels'] = label_encoder.fit_transform(balanced_df['label'])

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(balanced_df[['text', 'labels']])

    print(f"Loaded and preprocessed classification data. Number of samples: {len(dataset)}")

    return dataset, label_encoder


def shared_tokenize_classification_data(dataset):
    """
    Tokenizes the text data for classification using BERT.

    Parameters:
    - dataset (Dataset): Hugging Face Dataset object containing the 'text' and 'labels' columns.

    Returns:
    - tokenized_dataset (Dataset): Tokenized Hugging Face Dataset ready for training.
    """
    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=64,  # Adjust as needed
            return_attention_mask=True
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Set the format to PyTorch tensors
    tokenized_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )

    print("Tokenization for classification completed.")

    return tokenized_dataset

# for humour share private architecture

# data.py

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class HumorDataset(Dataset):
    def __init__(self, df, tokenizer_name, max_length):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.humor_types = df['humor_type_idx'].tolist()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        humor_type_idx = self.humor_types[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),          # [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(),# [max_length]
            'token_type_ids': encoding['token_type_ids'].squeeze(),# [max_length]
            'labels': torch.tensor(label, dtype=torch.long),
            'humor_type': torch.tensor(humor_type_idx, dtype=torch.long)  # For possible future use
        }

def sharedprivate_load_and_split_data(csv_file_path, reduce_factor=3):
    """
    Load and split the dataset for shared-private architecture, with optional reduction of dataset size.
    
    Parameters:
        csv_file_path (str): Path to the CSV dataset.
        reduce_factor (float): Factor by which to reduce the dataset size (default is 1.25).

    Returns:
        train_df, val_df, test_df, humor_type_to_idx: Training, validation, and test DataFrames, 
        and mapping of humor types to indices.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Map humor types to indices
    humor_types = df['type'].unique()
    humor_type_to_idx = {humor_type: idx for idx, humor_type in enumerate(humor_types)}
    
    # Encode humor_type to indices
    df['humor_type_idx'] = df['type'].map(humor_type_to_idx)
    
    # Create a combined stratification column
    df['stratify_col'] = df['label'].astype(str) + "_" + df['humor_type_idx'].astype(str)
    
    # Split the data into training, validation, and test sets
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df['stratify_col'],
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['stratify_col'],
        random_state=42
    )
    
    # Reduce the dataset size if requested
    if reduce_factor > 1:
        train_df = train_df.sample(n=math.floor(len(train_df) / reduce_factor), random_state=42)
        val_df = val_df.sample(n=math.floor(len(val_df) / reduce_factor), random_state=42)
        test_df = test_df.sample(n=math.floor(len(test_df) / reduce_factor), random_state=42)
    
    # Drop the stratify_col as it's no longer needed
    train_df = train_df.drop(columns=['stratify_col'])
    val_df = val_df.drop(columns=['stratify_col'])
    test_df = test_df.drop(columns=['stratify_col'])
    
    return train_df, val_df, test_df, humor_type_to_idx