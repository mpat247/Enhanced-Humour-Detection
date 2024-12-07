# models.py

import torch
import torch.nn as nn
from transformers import BertModel
import os

class SharedPrivateModel(nn.Module):
    def __init__(self, shared_model_path, private_model_paths, num_labels):
        """
        Initialize the SharedPrivateModel.

        Parameters:
            shared_model_path (str): Path to the shared BERT model.
            private_model_paths (dict): Dictionary mapping humor type indices to their respective private BERT model paths.
            num_labels (int): Number of output labels for classification.
        """
        super(SharedPrivateModel, self).__init__()
        
        # Define the expected checkpoint filename
        checkpoint_filename = 'shared_private_best_model.pt'
        checkpoint_path = os.path.join(shared_model_path, checkpoint_filename)
        
        # Conditional loading based on the presence of the checkpoint file
        if os.path.isfile(checkpoint_path):
            print(f"[INFO] Found checkpoint '{checkpoint_filename}' in '{shared_model_path}'. Loading shared BERT from checkpoint.")
            
            # Initialize a standard BERT model
            self.shared_bert = BertModel.from_pretrained('bert-base-uncased')
            
            # Load the state dictionary from the checkpoint
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # It's assumed that the checkpoint contains only the shared BERT's state_dict
            # If it contains more (e.g., classifier, private BERTs), adjust accordingly
            self.shared_bert.load_state_dict(state_dict, strict=False)
            
            print("[INFO] Shared BERT loaded from checkpoint successfully.")
        else:
            print(f"[INFO] No checkpoint found in '{shared_model_path}'. Loading shared BERT using 'from_pretrained'.")
            
            # Load the BERT model directly from the specified directory
            self.shared_bert = BertModel.from_pretrained(shared_model_path)
            
            print("[INFO] Shared BERT loaded from directory successfully.")
        
        # Initialize private BERT models for each humor type
        self.private_bert_dict = nn.ModuleDict()
        for humor_type_idx, private_path in private_model_paths.items():
            private_model = BertModel.from_pretrained(private_path)
            self.private_bert_dict[str(humor_type_idx)] = private_model
        
        # Classification layer: combines shared and private BERT outputs
        hidden_size = self.shared_bert.config.hidden_size + self.shared_bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, humor_type_idx=None):
        """
        Forward pass of the model.

        Parameters:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention masks.
            token_type_ids (Tensor, optional): Token type IDs.
            humor_type_idx (Tensor, optional): Humor type indices.

        Returns:
            logits (Tensor): Classification logits.
        """
        # Shared BERT (frozen)
        with torch.no_grad():
            shared_outputs = self.shared_bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            shared_pooled_output = shared_outputs.pooler_output
        
        # Select appropriate private BERT based on humor_type_idx
        if humor_type_idx is None:
            raise ValueError("humor_type_idx must be provided for private BERT selection.")
        
        private_pooled_outputs = []
        for i in range(input_ids.size(0)):
            humor_type = str(humor_type_idx[i].item())
            private_model = self.private_bert_dict[humor_type]
            private_output = private_model(
                input_ids=input_ids[i].unsqueeze(0),
                attention_mask=attention_mask[i].unsqueeze(0),
                token_type_ids=token_type_ids[i].unsqueeze(0)
            )
            private_pooled_outputs.append(private_output.pooler_output)
        
        private_pooled_output = torch.cat(private_pooled_outputs, dim=0)
        combined_output = torch.cat((shared_pooled_output, private_pooled_output), dim=1)
        logits = self.classifier(combined_output)
        
        return logits
