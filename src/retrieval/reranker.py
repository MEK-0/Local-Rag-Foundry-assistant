from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict, Any

# Initializes the local cross-encoder engine (will auto-download on first boot)
_rerank_model = None

def get_reranker_instance():
    global _rerank_model
    if _rerank_model is None:
        # Utilizing production standard BAAI model for edge inference optimization
        _rerank_model = CrossEncoder("BAAI/bge-reranker-base")
    return _rerank_model

def rerank_chunks(query: str, chunks: List[Dict[str, Any]], top_n: int = 4) -> List[Dict[str, Any]]:
    """
    Re-scores and re-ranks retrieved chunks utilizing a deep cross-attention transformer.
    """
    if not chunks:
        return []
        
    model = get_reranker_instance()
    
    # Formulate pairs: [[query, doc1], [query, doc2], ...]
    pairs = [[query, chunk["chunk_text"]] for chunk in chunks]
    
    # Compute cross-encoder logit scores
    scores = model.predict(pairs)
    
    # Attach scores to the chunk payload
    for idx, score in enumerate(scores):
        chunks[idx]["rerank_score"] = float(score)
        
    # Sort chunks by their new deep alignment scores in descending order
    reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    return reranked_chunks[:top_n]