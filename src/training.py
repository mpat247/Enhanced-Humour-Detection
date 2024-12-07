# src/training.py

from transformers import Trainer, TrainingArguments, TrainerCallback, BertForMaskedLM, BertForSequenceClassification, AdamW,get_scheduler, get_linear_schedule_with_warmup
import torch
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from utils import sharedprivate_eval_model




class CustomPrintingCallback(TrainerCallback):
    """
    A custom callback that prints training and evaluation loss after each logging step.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss")
            eval_loss = logs.get("eval_loss")
            if loss is not None:
                print(f"Step {state.global_step}: Training Loss = {loss:.4f}")
            if eval_loss is not None:
                print(f"Step {state.global_step}: Evaluation Loss = {eval_loss:.4f}")

def initialize_trainer(
    model,
    train_dataset,
    eval_dataset,
    output_dir='./models/bert-mlm',
    logging_dir='./output/logs',
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    logging_steps=500,
    save_steps=10_000,
    eval_steps=10_000,
    save_total_limit=2,
    gradient_accumulation_steps=4
):
    """
    Initializes the Hugging Face Trainer with the specified configurations.

    If the `output_dir` contains a saved model, it loads the model and skips setting up a new trainer for training.

    Parameters:
    - model: The initial BERT MLM model instance.
    - train_dataset: The tokenized and masked training dataset.
    - eval_dataset: The tokenized and masked evaluation/testing dataset.
    - output_dir (str): Directory where model checkpoints will be saved.
    - logging_dir (str): Directory where logs will be stored.
    - learning_rate (float): Learning rate for the optimizer.
    - num_train_epochs (int): Number of training epochs.
    - per_device_train_batch_size (int): Training batch size per device.
    - per_device_eval_batch_size (int): Evaluation batch size per device.
    - logging_steps (int): Number of steps between logging.
    - save_steps (int): Number of steps between saving checkpoints.
    - eval_steps (int): Number of steps between evaluations.
    - save_total_limit (int): Maximum number of saved checkpoints.
    - gradient_accumulation_steps (int): Number of steps to accumulate gradients.

    Returns:
    - trainer: Configured Hugging Face Trainer instance if training is required, otherwise None.
    - model: The initialized or loaded BERT MLM model.
    """
    # Check if a model already exists in the output directory
    model_files = ["pytorch_model.bin", "model.safetensors"]
    model_exists = any(os.path.exists(os.path.join(output_dir, file)) for file in model_files)

    if model_exists:
        print(f"Found existing model in {output_dir}. Loading the model...")
        from transformers import BertForMaskedLM
        model = BertForMaskedLM.from_pretrained(output_dir)
        return None, model  # No need to initialize a trainer, return None for trainer
    
    print("No existing model found in output directory. Setting up for training.")
    
    # Determine mixed precision settings based on available hardware
    fp16 = torch.cuda.is_available()  # Enable fp16 if CUDA is available
    bf16 = False
    if not fp16 and torch.backends.mps.is_available():
        bf16 = True  # Enable bf16 if MPS is available

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        prediction_loss_only=True,
        fp16=fp16,
        bf16=bf16,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[CustomPrintingCallback()],
    )

    return trainer, model


def start_training(trainer):
    """
    Starts the training process using the provided Trainer instance.

    Parameters:
    - trainer: The initialized Hugging Face Trainer instance (or None if no training is required).

    Returns:
    - train_result: The result of the training process if training occurs, otherwise None.
    - model: The trained BERT MLM model (or the preloaded model if training is skipped).
    """
    # If the trainer is None, skip training
    if trainer is None:
        print("Training skipped. Pretrained model loaded.")
        return None, None

    # Start training
    print("Starting the training process...")
    train_result = trainer.train()

    # Save the final model (+ tokenizer and config) to disk
    trainer.save_model()  # Saves the tokenizer too for easy upload

    # Save training logs or metrics
    with open(os.path.join(trainer.args.output_dir, "training_results.json"), "w") as f:
        f.write(str(train_result.metrics))

    return train_result, trainer.model


def shared_initialize_classification_trainer(
    model_dir,
    train_dataset,
    eval_dataset,
    output_dir='./models/bert-classification',
    logging_dir='./output/logs_classification',
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    logging_steps=500,
    save_steps=10_000,
    eval_steps=10_000,
    save_total_limit=2,
    gradient_accumulation_steps=4
):
    """
    Initializes the Hugging Face Trainer for classification with the specified configurations.

    If the `output_dir` contains a saved classification model, it loads the model. Otherwise, it uses
    the pretrained model from `model_dir` and initializes a classification trainer.

    Parameters:
    - model_dir (str): Directory where the pretrained MLM model is saved.
    - train_dataset (Dataset): Tokenized training dataset for classification.
    - eval_dataset (Dataset): Tokenized evaluation dataset for classification.
    - output_dir (str): Directory where classification model checkpoints will be saved.
    - logging_dir (str): Directory where logs will be stored.
    - learning_rate (float): Learning rate for the optimizer.
    - num_train_epochs (int): Number of training epochs.
    - per_device_train_batch_size (int): Training batch size per device.
    - per_device_eval_batch_size (int): Evaluation batch size per device.
    - logging_steps (int): Number of steps between logging.
    - save_steps (int): Number of steps between saving checkpoints.
    - eval_steps (int): Number of steps between evaluations.
    - save_total_limit (int): Maximum number of saved checkpoints.
    - gradient_accumulation_steps (int): Number of steps to accumulate gradients.

    Returns:
    - trainer (Trainer): Configured Hugging Face Trainer instance if training is required, otherwise None.
    - model (BertForSequenceClassification): The initialized or loaded classification model.
    """
    # Check if a classification model already exists in the output directory
    model_files = ["pytorch_model.bin", "model.safetensors"]
    model_exists = any(os.path.exists(os.path.join(output_dir, file)) for file in model_files)

    if model_exists:
        print(f"Found existing classification model in {output_dir}. Loading the model...")
        model = BertForSequenceClassification.from_pretrained(output_dir)
        return None, model  # Return None for trainer, as no further training is required

    print("No existing classification model found in output directory. Initializing a new model for classification.")
    
    # Load the pretrained MLM model and adapt it for classification
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=2)  # Adjust `num_labels` as required
    
    # Determine mixed precision settings based on available hardware
    fp16 = torch.cuda.is_available()  # Enable fp16 if CUDA is available
    bf16 = False
    if not fp16 and torch.backends.mps.is_available():
        bf16 = True  # Enable bf16 if MPS is available

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        prediction_loss_only=False,
        fp16=fp16,
        bf16=bf16,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[CustomPrintingCallback()],
    )

    return trainer, model


def shared_start_training(trainer):
    """
    Starts the training process using the provided Trainer instance.

    Parameters:
    - trainer: The initialized Hugging Face Trainer instance (or None if no training is required).

    Returns:
    - train_result: The result of the training process if training occurs, otherwise None.
    - model: The trained classification model (or the preloaded model if training is skipped).
    """
    # If the trainer is None, skip training
    if trainer is None:
        print("Training skipped. Pretrained classification model loaded.")
        return None, None

    # Start training
    print("Starting the classification training process...")
    train_result = trainer.train()

    # Save the final model (+ tokenizer and config) to disk
    trainer.save_model()  # Saves the tokenizer too for easy upload

    # Save training logs or metrics
    with open(os.path.join(trainer.args.output_dir, "training_results.json"), "w") as f:
        f.write(str(train_result.metrics))

    return train_result, trainer.model

def shared_evaluate_model(model, test_dataset, batch_size=64, sample_fraction=0.5):
    """
    Evaluates the classification model on the given test dataset using MPS for faster processing.

    Parameters:
    - model (transformers.PreTrainedModel): The trained Hugging Face model.
    - test_dataset (Dataset): The tokenized Hugging Face Dataset used for evaluation.
    - batch_size (int): Batch size for evaluation (default: 64).
    - sample_fraction (float): Fraction of the dataset to evaluate on (default: 10%).

    Returns:
    - metrics (dict): A dictionary containing evaluation metrics.
    """
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import torch

    # Ensure model is on the MPS device
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    # Optional: sample a subset for quicker evaluation
    if sample_fraction < 1.0:
        dataset_size = int(len(test_dataset) * sample_fraction)
        test_dataset = test_dataset.shuffle(seed=42).select(range(dataset_size))
        print(f"Evaluating on a subset of {len(test_dataset)} samples.")

    # Use a DataLoader for batching
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    # Set model to evaluation mode
    model.eval()

    all_predictions = []
    all_labels = []

    # Iterate through the test dataset with a progress bar
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert predictions and labels to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
    }

import json
import os
# from tensorboardX import SummaryReader

def retrieve_model_functions(model_path):
    """
    Determines the type of model (MLM or Classification) from the provided path, retrieves training metrics,
    and validation metrics from TensorBoard logs, and returns the model and metrics.

    Parameters:
    - model_path (str): Path to the saved model directory.

    Returns:
    - model: The loaded model instance.
    - model_type (str): The type of the model ("classification" or "mlm").
    - metrics (dict): A dictionary containing training and validation metrics.
    """
    # Check if model directory exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model directory '{model_path}' does not exist.")

    # Determine model type
    print(f"Loading model from {model_path}...")
    try:
        model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        model_type = "classification"
        print("Model identified as a Sequence Classification model.")
    except Exception:
        try:
            model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
            model_type = "mlm"
            print("Model identified as a Masked Language Model (MLM).")
        except Exception as e:
            raise ValueError(f"Unable to determine model type from the directory '{model_path}'. Error: {str(e)}")

    # Retrieve training metrics
    metrics_file = os.path.join(model_path, "training_results.json")
    training_metrics = None
    if os.path.exists(metrics_file):
        print("Training metrics file found. Attempting to load metrics...")
        try:
            with open(metrics_file, "r") as f:
                content = f.read()
                try:
                    training_metrics = json.loads(content)
                except json.JSONDecodeError:
                    print("Warning: Metrics file is not valid JSON. Attempting fallback parsing.")
                    training_metrics = eval(content)  # Use cautiously
                    if not isinstance(training_metrics, dict):
                        raise ValueError("Parsed metrics are not a valid dictionary.")
        except Exception as e:
            print(f"Error parsing training metrics: {e}")

    # Retrieve validation metrics
    logs_dir = os.path.join(model_path, "../logs_classification")
    validation_metrics = None
    # if os.path.exists(logs_dir):
    #     print("Searching for validation metrics in TensorBoard logs...")
    #     validation_metrics = {}
    #     for file in os.listdir(logs_dir):
    #         if file.startswith("events.out.tfevents"):
    #             try:
    #                 # Use tensorboardX to read the logs
    #                 reader = SummaryReader(os.path.join(logs_dir, file))
    #                 for event in reader:
    #                     for tag in event.keys():
    #                         if "eval" in tag:  # Look for evaluation-specific keys
    #                             if tag not in validation_metrics:
    #                                 validation_metrics[tag] = []
    #                             validation_metrics[tag].append((event[tag]['step'], event[tag]['value']))
    #             except Exception as e:
    #                 print(f"Error reading TensorBoard logs: {e}")

    # Return the model, model type, and metrics
    return model, model_type, {
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics
    }

# from tensorboardX import SummaryReader
from torch.cuda.amp import autocast, GradScaler


import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

scaling_factor=1024

def sharedprivate_train_shared_bert(
    model,
    train_loader,
    val_loader,
    epochs,
    learning_rate,
    device,
    save_dir="./models/shared_bert_finetuned",
    save_interval=1,  # Save model every `save_interval` epochs
):
    """
    Trains the shared-private model's private BERT and classifier layers on the combined dataset.
    This effectively fine-tunes the private components while keeping the shared BERT frozen.

    Parameters:
        model: The shared-private model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        epochs: Number of epochs to train.
        learning_rate: Learning rate for the optimizer.
        device: Device to use for training ('cpu', 'cuda', or 'mps').
        save_dir: Directory to save checkpoints.
        save_interval: Number of epochs between saving checkpoints.

    Returns:
        model: The trained or loaded model.
    """
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = f"{save_dir}/shared_private_best_model.pt"

    ## Check if the model already exists
    if os.path.exists(best_model_path):
        print(f"Model found at {best_model_path}. Skipping training and loading the model.")
        # Load the saved checkpoint (assuming it's saved as a dictionary)
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # Attempt to load the model's state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded model_state_dict successfully.")
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)  # Directly load the state_dict
            print("Loaded state_dict successfully from checkpoint.")
        else:
            print("Error: Checkpoint format is invalid!")
            raise ValueError("The saved checkpoint does not contain a valid state_dict.")

    
    
    
    print(f"No existing model found. Starting training from scratch.")
    model.to(device)


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [
            {"params": model.classifier.parameters()},
            *[{"params": private_model.parameters()} for private_model in model.private_bert_dict.values()]
        ],
        lr=learning_rate,
    )

    # Total steps for scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_accuracy = 0.0  # Track best validation accuracy

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0

        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        for batch in tqdm(train_loader, desc="Training batches"):
            optimizer.zero_grad()

            # Move inputs and labels to the device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            humor_type_idx = batch["humor_type"].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask, token_type_ids, humor_type_idx)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Accumulate loss and calculate accuracy
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = correct_predictions.float() / len(train_loader.dataset)
        print(f"Epoch {epoch+1} - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}")

        # Evaluate on validation set
        val_metrics = sharedprivate_eval_model(model, val_loader, criterion, device, len(val_loader.dataset))
        val_accuracy = val_metrics["accuracy"]
        print(
            f"Validation Metrics - Accuracy: {val_accuracy:.4f}, "
            f"Precision: {val_metrics['precision']:.4f}, "
            f"Recall: {val_metrics['recall']:.4f}, "
            f"F1-Score: {val_metrics['f1_score']:.4f}"
        )

        # Save checkpoints periodically
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f"{save_dir}/shared_private_latest_checkpoint.pt"
            print(f"Saving checkpoint to: {checkpoint_path}")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"New best model found. Saving to: {best_model_path}")
            torch.save(model.state_dict(), best_model_path)


    return model


def sharedprivate_train_private_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    learning_rate,
    device,
    save_dir="./models/updated_shared_private",
    save_interval=1  # Save model every `save_interval` epochs
):
    """
    Trains the shared-private model's private BERT and classifier layers for a specific humor dataset.
    
    Parameters:
        model: The shared-private model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for testing data.
        epochs: Number of epochs to train.
        learning_rate: Learning rate for the optimizer.
        device: Device to use for training ('cpu', 'cuda', or 'mps').
        save_dir: Directory to save checkpoints.
        save_interval: Number of epochs between saving checkpoints.
    
    Returns:
        model: The trained model.
    """
    model.to(device)
    print(f"Training shared-private model on device: {device}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [
            {"params": model.classifier.parameters()},
            {"params": model.private_bert_dict.parameters()},
        ],
        lr=learning_rate,
    )

    # Total steps for scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_accuracy = 0.0  # Track best validation accuracy

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
    
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        for batch in tqdm(train_loader, desc="Training batches"):
            optimizer.zero_grad()
    
            # Move inputs and labels to the device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            humor_type_idx = batch["humor_type"].to(device)
    
            # Forward pass
            logits = model(input_ids, attention_mask, token_type_ids, humor_type_idx)
            loss = criterion(logits, labels) / scaling_factor  # Scale loss
    
            # Backward pass
            loss.backward()
    
            # Scale gradients back
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(scaling_factor)
    
            optimizer.step()
            scheduler.step()
    
            # Accumulate loss and calculate accuracy
            total_loss += loss.item() * scaling_factor  # Rescale for reporting
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
    
        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = correct_predictions.float() / len(train_loader.dataset)
        print(f"Epoch {epoch+1} - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}")


        # Evaluate on validation set
        val_metrics = sharedprivate_eval_model(model, val_loader, criterion, device, len(val_loader.dataset))
        val_accuracy = val_metrics["accuracy"]
        print(
            f"Validation Metrics - Accuracy: {val_accuracy:.4f}, "
            f"Precision: {val_metrics['precision']:.4f}, "
            f"Recall: {val_metrics['recall']:.4f}, "
            f"F1-Score: {val_metrics['f1_score']:.4f}"
        )

        # Save checkpoints periodically
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f"{save_dir}/shared_private_{epoch+1}.pt"
            print(f"Saving checkpoint to: {checkpoint_path}")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = f"{save_dir}/shared_private_best_model_{epoch+1}.pt"
            print(f"New best model found. Saving to: {best_model_path}")
            torch.save(model.state_dict(), best_model_path)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = sharedprivate_eval_model(model, test_loader, criterion, device, len(test_loader.dataset))
    print(
        f"Test Metrics - Accuracy: {test_metrics['accuracy']:.4f}, "
        f"Precision: {test_metrics['precision']:.4f}, "
        f"Recall: {test_metrics['recall']:.4f}, "
        f"F1-Score: {test_metrics['f1_score']:.4f}"
    )

    return model