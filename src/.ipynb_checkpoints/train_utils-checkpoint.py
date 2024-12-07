# src/train_utils.py

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import torch
from torch.utils.data import DataLoader

def define_training_args(
    output_dir='./models',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,
    learning_rate=1e-4,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    weight_decay=0.01,
    remove_unused_columns = None
):
    """
    Define training arguments for the Trainer.

    Automatically configures mixed precision based on the available device.
    """
    # Detect the device type
    device_type = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu").type

    # Initialize TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        weight_decay=weight_decay,
        disable_tqdm=False,        # Ensure progress bars are visible
        report_to='none',          # Disable external loggers
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        fp16=(device_type == "cuda"),  # Enable fp16 for CUDA GPUs
        bf16=(device_type == "mps"),    # Enable bf16 for MPS devices if supported
        remove_unused_columns = False
    )
    return training_args




def initialize_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    tokenizer,compute_metrics=None
):
    """
    Initialize the Trainer.

    Args:
        model (transformers.PreTrainedModel): The model to train.
        training_args (transformers.TrainingArguments): Training arguments.
        train_dataset (datasets.Dataset): The training dataset.
        eval_dataset (datasets.Dataset): The evaluation dataset.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.

    Returns:
        transformers.Trainer: Initialized Trainer.
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    return trainer


def train_model(trainer, output_dir):
    """
    Train the model using the Trainer, skipping training if the model already exists.

    Args:
        trainer (transformers.Trainer): The Trainer instance.
        output_dir (str): Path to the directory where the model is saved.

    Returns:
        transformers.Trainer: The Trainer after training.
    """
    # Check if the model directory exists and contains a saved model
    if os.path.exists(output_dir):
        model_files = [
            "pytorch_model.bin",
            "config.json",
            "tokenizer_config.json",
            "model.safetensors"  # Add model.safetensors to the list
        ]
        # Check if any of the expected files exist
        if any(os.path.isfile(os.path.join(output_dir, file)) for file in model_files):
            print(f"Model already exists at {output_dir}. Skipping training.")
            return trainer
    
    # Train the model
    print("Training the model...")
    trainer.train()
    trainer.save_model(output_dir)  # Saves the model and associated tokenizer
    print(f"Model saved to {output_dir}.")
    
    return trainer


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics: accuracy, precision, recall, and F1 score.

    Args:
        eval_pred (tuple): A tuple (predictions, labels) from the Trainer.

    Returns:
        dict: Computed metrics.
    """
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)  # Convert logits to predicted labels
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def evaluate_model(trainer, test_dataset):
    """
    Evaluate the model using the Trainer.

    Args:
        trainer (transformers.Trainer): The Trainer instance.

    Returns:
        dict: Evaluation metrics.
    """
    eval_results = trainer.predict(test_dataset=test_dataset)
    print("\nEvaluation Metrics:")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    return eval_results

def evaluate_large_dataset(trainer, eval_dataset, tokenizer, device, batch_size=16):
    """
    Custom evaluation loop for large datasets to avoid memory issues.

    Args:
        trainer (transformers.Trainer): The Trainer instance.
        eval_dataset (datasets.Dataset): The evaluation dataset.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        device (torch.device): Device to run evaluation on.
        batch_size (int): Batch size for evaluation.

    Returns:
        dict: Evaluation metrics.
    """
    model = trainer.model
    model.to(device)  # Ensure the model is moved to the correct device
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0

    # Create DataLoader for batch processing
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            # Tokenize inputs
            inputs = tokenizer(
                batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
            labels = batch["labels"].to(device)  # Move labels to device

            # Forward pass
            outputs = model(**inputs, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average="weighted")

    return {
        "eval_loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }