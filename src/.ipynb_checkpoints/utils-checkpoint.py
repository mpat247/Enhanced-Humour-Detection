# src/utils.py

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def sharedprivate_eval_model(model, data_loader, loss_fn, device, n_examples):
    """
    Evaluates the shared-private model on a dataset and computes metrics.

    Parameters:
        model: The trained model.
        data_loader: DataLoader for the dataset to evaluate.
        loss_fn: Loss function.
        device: Device to use for evaluation ('cpu', 'cuda', or 'mps').
        n_examples: Number of examples in the dataset.

    Returns:
        metrics (dict): A dictionary containing accuracy, precision, recall, and F1-score.
    """
    model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating batches"):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                humor_type_idx = batch['humor_type'].to(device)

                # Forward pass
                logits = model(input_ids, attention_mask, token_type_ids, humor_type_idx)
                loss = loss_fn(logits, labels)

                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue

    accuracy = correct_predictions.float() / n_examples
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    return {
        "accuracy": accuracy.item(),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
