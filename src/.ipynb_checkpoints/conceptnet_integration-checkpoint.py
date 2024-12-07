
import spacy
from conceptnet5.query import lookup
import torch
import numpy as np
import os

# Load spaCy model for NER and noun phrase extraction
final_nlp = spacy.load("en_core_web_sm")
print("[INFO] spaCy model 'en_core_web_sm' loaded successfully.")

# Load pre-trained ConceptNet embeddings (e.g., ConceptNet Numberbatch)
# Replace 'path_to_conceptnet_embeddings' with the actual path to your embeddings
final_CONCEPTNET_EMBEDDING_PATH = './data/conceptnet_embeddings.npy'  # Example path

if os.path.exists(final_CONCEPTNET_EMBEDDING_PATH):
    final_conceptnet_embeddings = np.load(final_CONCEPTNET_EMBEDDING_PATH, allow_pickle=True).item()
    print("[INFO] Loaded ConceptNet embeddings successfully.")
else:
    final_conceptnet_embeddings = {}  # Placeholder: Implement loading mechanism
    print(f"[WARNING] ConceptNet embeddings not found at {final_CONCEPTNET_EMBEDDING_PATH}. Proceeding without embeddings.")

def final_extract_concepts(text):
    """
    Extracts key concepts from the input text using NER and noun phrase extraction.

    Parameters:
        text (str): The input text.

    Returns:
        List[str]: A list of unique concepts extracted from the text.
    """
    final_doc = final_nlp(text)
    final_concepts = set()

    # Extract named entities
    for ent in final_doc.ents:
        final_concepts.add(ent.text.lower())

    # Extract noun phrases
    for chunk in final_doc.noun_chunks:
        final_concepts.add(chunk.text.lower())

    final_concept_list = list(final_concepts)
    print(f"[DEBUG] Extracted Concepts: {final_concept_list}")
    return final_concept_list

def final_get_top_relations(concept, top_k=5):
    """
    Retrieves the top K related relations for a given concept from ConceptNet.

    Parameters:
        concept (str): The concept to query.
        top_k (int): Number of top relations to retrieve.

    Returns:
        List[dict]: A list of relation dictionaries from ConceptNet.
    """
    # Format concept for ConceptNet query
    final_concept_query = f"/c/en/{concept.replace(' ', '_')}"
    final_results = lookup(final_concept_query, limit=top_k)

    # Filter results to include only informative relations
    # Exclude overly generic relations like 'HasA', 'MadeOf', 'UsedFor'
    final_filtered_relations = [res for res in final_results if res['rel']['label'] not in ['HasA', 'MadeOf', 'UsedFor']]

    final_top_relations = final_filtered_relations[:top_k]
    print(f"[DEBUG] Retrieved Top Relations for '{concept}': {[rel['rel']['label'] for rel in final_top_relations]}")
    return final_top_relations

def final_generate_conceptnet_embedding(relations, embed_dim=300):
    """
    Generates an aggregated ConceptNet embedding from retrieved relations.

    Parameters:
        relations (List[dict]): A list of relation dictionaries.
        embed_dim (int): Dimension of the embeddings.

    Returns:
        torch.Tensor: Aggregated embedding vector.
    """
    final_embeddings = []
    for relation in relations:
        # Extract the end concept
        final_end_concept = relation['end']['term']
        if final_end_concept.startswith('/c/en/'):
            final_end_concept_clean = final_end_concept.split('/c/en/')[1].replace('_', ' ')
            # Retrieve embedding
            final_embed = final_conceptnet_embeddings.get(final_end_concept_clean.lower())
            if final_embed is not None:
                final_embeddings.append(final_embed)

    if final_embeddings:
        # Aggregate by averaging
        final_aggregated_embedding = np.mean(final_embeddings, axis=0)
        print("[DEBUG] Aggregated ConceptNet Embedding computed.")
    else:
        # If no embeddings found, return a zero vector
        final_aggregated_embedding = np.zeros(embed_dim)
        print("[WARNING] No embeddings found for retrieved relations. Using zero vector.")

    return torch.tensor(final_aggregated_embedding, dtype=torch.float)

def final_get_conceptnet_embedding(text):
    """
    Processes the input text to generate a ConceptNet embedding.

    Parameters:
        text (str): The input text.

    Returns:
        torch.Tensor: ConceptNet embedding vector.
    """
    final_concepts = final_extract_concepts(text)
    final_all_relations = []
    for final_concept in final_concepts:
        final_relations = final_get_top_relations(final_concept, top_k=5)
        final_all_relations.extend(final_relations)

    final_embedding = final_generate_conceptnet_embedding(final_all_relations)
    print("[DEBUG] ConceptNet Embedding generated for input text.")
    return final_embeddin