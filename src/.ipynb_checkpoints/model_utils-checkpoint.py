# src/model_utils.py

from transformers import BertTokenizer, BertForMaskedLM

def initialize_tokenizer(pretrained_model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    return tokenizer

def initialize_model(pretrained_model_name='bert-base-uncased'):
    model = BertForMaskedLM.from_pretrained(pretrained_model_name)
    return model
