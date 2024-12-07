# File: src/training.py

import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import os

def sharedprivate_train_shared_bert(
    model, train_loader, val_loader, epochs, learning_rate, device, save_dir, save_interval=1
):
    """
    Trains the shared-private model by fine-tuning the shared BERT layers and classification head.
    
    Parameters:
        model (nn.Module): The shared-private model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): Device to train on.
        save_dir (str): Directory to save checkpoints.
        save_interval (int): Interval (in epochs) to save checkpoints.
    
    Returns:
        nn.Module: The trained model.
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps//2, gamma=0.1)
    
    best_val_accuracy = 0.0
    
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
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Accumulate loss and calculate accuracy
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
        
        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = correct_predictions.double() / len(train_loader.dataset)
        print(f"Epoch {epoch+1} - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}")
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        correct_val = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation batches"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)
                humor_type_idx = batch["humor_type"].to(device)
                
                logits = model(input_ids, attention_mask, token_type_ids, humor_type_idx)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct_val += torch.sum(preds == labels)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_val.double() / len(val_loader.dataset)
        
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        
        print(
            f"Validation Metrics - Accuracy: {val_accuracy:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
        )
        
        # Save checkpoints periodically
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"shared_private_epoch_{epoch+1}.pt")
            print(f"[INFO] Saving checkpoint to: {checkpoint_path}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, checkpoint_path)
        
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(save_dir, f"shared_private_best_model_epoch_{epoch+1}.pt")
            print(f"[INFO] New best model found. Saving to: {best_model_path}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, best_model_path)
    
    print("\n[INFO] Training complete.")
    return model

def sharedprivate_train_private_model(
    model, train_loader, val_loader, test_loader, epochs, learning_rate, device, save_dir, save_interval=1
):
    """
    Trains the private BERT layers for a specific humor type.
    
    Parameters:
        model (nn.Module): The shared-private model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        test_loader (DataLoader): DataLoader for the test data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): Device to train on.
        save_dir (str): Directory to save checkpoints.
        save_interval (int): Interval (in epochs) to save checkpoints.
    
    Returns:
        nn.Module: The trained model.
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps//2, gamma=0.1)
    
    best_val_accuracy = 0.0
    
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
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Accumulate loss and calculate accuracy
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
        
        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = correct_predictions.double() / len(train_loader.dataset)
        print(f"Epoch {epoch+1} - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}")
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        correct_val = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation batches"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)
                humor_type_idx = batch["humor_type"].to(device)
                
                logits = model(input_ids, attention_mask, token_type_ids, humor_type_idx)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct_val += torch.sum(preds == labels)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_val.double() / len(val_loader.dataset)
        
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        
        print(
            f"Validation Metrics - Accuracy: {val_accuracy:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
        )
        
        # Save checkpoints periodically
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"shared_private_private_epoch_{epoch+1}.pt")
            print(f"[INFO] Saving checkpoint to: {checkpoint_path}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, checkpoint_path)
        
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(save_dir, f"shared_private_private_best_model_epoch_{epoch+1}.pt")
            print(f"[INFO] New best model found. Saving to: {best_model_path}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, best_model_path)
    
    print("\n[INFO] Training complete.")
    return model
