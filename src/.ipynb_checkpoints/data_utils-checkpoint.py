# src/data_utils.py

import os
from datasets import load_dataset, DatasetDict

def load_datasets_function(train_path, test_path):
    # Check if both files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"The training dataset file {train_path} does not exist.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"The testing dataset file {test_path} does not exist.")

    # Load both datasets
    datasets = load_dataset(
        'csv',
        data_files={'train': train_path, 'test': test_path},
        delimiter='\t'
    )

    # Rename 'test' split to 'validation' for clarity
    datasets = DatasetDict({
        "train": datasets["train"],
        "validation": datasets["test"]
    })

    print(f"Loaded dataset with {len(datasets['train'])} training samples and {len(datasets['validation'])} validation samples.\n")
    return datasets

def tokenize_datasets_function(datasets, tokenizer, max_length=128):
    def tokenize_function(examples):
        # 'examples' is a dictionary where each key is a column name and each value is a list of column values
        return tokenizer(
            examples['text'],  # Replace 'text' with your actual column name if different
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_special_tokens_mask=True
        )
    
    # Apply the tokenize_function to both 'train' and 'validation' splits
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    print("Tokenization complete.\n")
    return tokenized_datasets
