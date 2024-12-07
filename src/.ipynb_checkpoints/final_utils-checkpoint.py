# File: src/utils.py

import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm.notebook import tqdm

def sharedprivate_eval_model(model, data_loader, loss_fn, device, n_examples):
    """
    Evaluates the model on the given data loader.

    Parameters:
        model (nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the data to evaluate.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to perform evaluation on.
        n_examples (int): Number of examples in the data_loader.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating batches"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            humor_type_idx = batch["humor_type"].to(device)
            
            logits = model(input_ids, attention_mask, token_type_ids, humor_type_idx)
            loss = loss_fn(logits, labels)
            losses.append(loss.item())
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct_predictions.double() / n_examples
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    
    return {
        "accuracy": accuracy.item(),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluates the model on the test set and prints performance metrics.

    Parameters:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform evaluation on.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    print("\n[INFO] Evaluating the model on the test set...")
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating batches"):
            # Move inputs and labels to the device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            humor_type_idx = batch["humor_type"].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask, token_type_ids, humor_type_idx)
            loss = criterion(logits, labels)

            # Accumulate loss and calculate accuracy
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

            # Collect all predictions and labels for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions.double() / len(test_loader.dataset)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": accuracy.item(),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    print(
        f"Test Metrics - Loss: {avg_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
        f"F1-Score: {metrics['f1_score']:.4f}"
    )

    return metrics
