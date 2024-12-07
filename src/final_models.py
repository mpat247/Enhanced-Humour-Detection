# src/models.py

import torch
import torch.nn as nn
from transformers import BertModel
import os

class SharedPrivateModel(nn.Module):
    def __init__(self, shared_model_path, private_model_paths, num_labels, concept_embed_dim=300, dropout_rate=0.3):
        """
        Initializes the SharedPrivateModel.

        Parameters:
            shared_model_path (str): Path to the shared BERT model or checkpoint.
            private_model_paths (dict): Dictionary mapping humor type indices to their respective private BERT model paths.
            num_labels (int): Number of output labels for classification.
            concept_embed_dim (int): Dimension of ConceptNet embeddings.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(final_SharedPrivateModel, self).__init__()
        print("[INFO] Initializing final_SharedPrivateModel...")

        # Load the shared BERT model
        if os.path.isdir(shared_model_path):
            print(f"[INFO] Loading shared BERT from directory: {shared_model_path}")
            self.final_shared_bert = BertModel.from_pretrained(shared_model_path)
        elif os.path.isfile(shared_model_path):
            print(f"[INFO] Loading shared BERT from checkpoint: {shared_model_path}")
            final_checkpoint = torch.load(shared_model_path, map_location='cpu')
            if isinstance(final_checkpoint, dict) and "model_state_dict" in final_checkpoint:
                final_state_dict = final_checkpoint["model_state_dict"]
            elif isinstance(final_checkpoint, dict):
                final_state_dict = final_checkpoint
            else:
                raise ValueError("The saved checkpoint does not contain a valid state_dict.")
            self.final_shared_bert = BertModel.from_pretrained('bert-base-uncased')
            self.final_shared_bert.load_state_dict(final_state_dict, strict=False)
        else:
            raise ValueError(f"Shared model path '{shared_model_path}' is invalid.")
        print("[INFO] Shared BERT model loaded successfully.")

        # Initialize private BERT models for each humor type
        self.final_private_bert_dict = nn.ModuleDict()
        for final_humor_type_idx, final_private_path in private_model_paths.items():
            if os.path.isdir(final_private_path):
                print(f"[INFO] Loading private BERT for type {final_humor_type_idx} from directory: {final_private_path}")
                final_private_model = BertModel.from_pretrained(final_private_path)
            elif os.path.isfile(final_private_path):
                print(f"[INFO] Loading private BERT for type {final_humor_type_idx} from checkpoint: {final_private_path}")
                final_private_checkpoint = torch.load(final_private_path, map_location='cpu')
                if isinstance(final_private_checkpoint, dict) and "model_state_dict" in final_private_checkpoint:
                    final_private_state_dict = final_private_checkpoint["model_state_dict"]
                elif isinstance(final_private_checkpoint, dict):
                    final_private_state_dict = final_private_checkpoint
                else:
                    raise ValueError("The saved checkpoint does not contain a valid state_dict.")
                final_private_model = BertModel.from_pretrained('bert-base-uncased')
                final_private_model.load_state_dict(final_private_state_dict, strict=False)
            else:
                raise ValueError(f"Private model path '{final_private_path}' is invalid for humor type {final_humor_type_idx}.")
            self.final_private_bert_dict[str(final_humor_type_idx)] = final_private_model
            print(f"[INFO] Private BERT for humor type {final_humor_type_idx} loaded successfully.")

        # Dropout layer after shared BERT
        self.final_shared_dropout = nn.Dropout(p=dropout_rate)
        print("[INFO] Dropout layer after shared BERT initialized.")

        # Dropout layer within each private BERT
        self.final_private_dropout = nn.Dropout(p=dropout_rate)
        print("[INFO] Dropout layer within private BERT initialized.")

        # Classification head: combines shared, private BERT outputs, and ConceptNet embeddings
        # Assuming BERT's hidden_size is 768 for both shared and private
        final_hidden_size = self.final_shared_bert.config.hidden_size * 2 + concept_embed_dim  # Shared + Private + ConceptNet
        self.final_classifier = nn.Sequential(
            nn.LayerNorm(final_hidden_size),
            nn.Linear(final_hidden_size, final_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(final_hidden_size // 2, final_hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(final_hidden_size // 4, num_labels),
            nn.Softmax(dim=1)
        )
        print("[INFO] Classification head initialized successfully.")

    def forward(self, input_ids, attention_mask, token_type_ids=None, humor_type_idx=None, concept_embeddings=None):
        """
        Forward pass of the model.

        Parameters:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention masks.
            token_type_ids (Tensor, optional): Token type IDs.
            humor_type_idx (Tensor, optional): Humor type indices.
            concept_embeddings (Tensor, optional): ConceptNet embeddings.

        Returns:
            logits (Tensor): Classification logits.
        """
        if humor_type_idx is None:
            raise ValueError("humor_type_idx must be provided for private BERT selection.")
        
        if concept_embeddings is None:
            raise ValueError("concept_embeddings must be provided for ConceptNet integration.")

        # Shared BERT (frozen)
        with torch.no_grad():
            final_shared_outputs = self.final_shared_bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            final_shared_pooled_output = final_shared_outputs.pooler_output  # [batch_size, hidden_size]
            print("[DEBUG] Shared BERT pooled output obtained.")

        # Apply dropout to shared BERT output
        final_shared_pooled_output = self.final_shared_dropout(final_shared_pooled_output)
        print("[DEBUG] Dropout applied to shared BERT pooled output.")

        # Initialize tensor for private pooled outputs
        final_private_pooled_output = torch.zeros_like(final_shared_pooled_output).to(input_ids.device)

        # Get unique humor types in the batch
        final_unique_types = torch.unique(humor_type_idx)
        print(f"[DEBUG] Unique humor types in the batch: {final_unique_types}")

        for final_humor_type in final_unique_types:
            final_mask = (humor_type_idx == final_humor_type)
            if final_mask.sum() == 0:
                continue
            final_private_model = self.final_private_bert_dict[str(final_humor_type.item())]
            # Process the subset of the batch corresponding to the current humor type
            final_private_outputs = final_private_model(
                input_ids=input_ids[final_mask],
                attention_mask=attention_mask[final_mask],
                token_type_ids=token_type_ids[final_mask]
            )
            final_private_pooled = final_private_outputs.pooler_output  # [num_samples_of_type, hidden_size]
            # Apply dropout to private BERT pooled output
            final_private_pooled = self.final_private_dropout(final_private_pooled)
            print(f"[DEBUG] Dropout applied to private BERT pooled output for humor type {final_humor_type.item()}.")
            final_private_pooled_output[final_mask] = final_private_pooled

        # Concatenate shared and private outputs with ConceptNet embeddings
        final_combined_output = torch.cat((final_shared_pooled_output, final_private_pooled_output, concept_embeddings), dim=1)  # [batch_size, 2*hidden_size + embed_dim]
        print("[DEBUG] Shared and private BERT outputs concatenated with ConceptNet embeddings.")

        # Apply layer normalization
        final_combined_output = self.final_classifier[0](final_combined_output)  # LayerNorm
        print("[DEBUG] Layer normalization applied to combined outputs.")

        # Pass through classification head
        final_logits = self.final_classifier[1:](final_combined_output)  # Remaining layers
        print("[DEBUG] Classification head applied.")

        return final_logits
