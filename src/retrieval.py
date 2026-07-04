import numpy as np
from typing import List, Dict, Any
from src.db import get_all_chunks
from src.config import settings

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Computes the cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
        
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def retrieve_top_k(query_embedding: List[float], top_k: int = None) -> List[Dict[str, Any]]:
    """
    Retrieves the top_k most similar chunks from the local database
    using brute-force cosine similarity.
    """
    if top_k is None:
        top_k = settings.top_k
        
    chunks = get_all_chunks()
    scored_chunks = []
    
    
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        chunk_data = chunk.copy()
        chunk_data["score"] = float(score)
        scored_chunks.append(chunk_data)
        
    # Sort by descending score
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_chunks[:top_k]