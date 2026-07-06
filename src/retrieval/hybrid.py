import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.db import get_all_chunks_for_sparse

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes the classic vector dot product orientation score."""
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def hybrid_retrieve(query_text: str, query_embedding: List[float], top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
    """
    Executes a high-grade Hybrid Search combining BM25 Okapi and Dense Vector similarity.
    Fuses rankings utilizing Reciprocal Rank Fusion (RRF).
    
    Args:
        query_text (str): Raw string question for sparse keyword matching.
        query_embedding (List[float]): Generated vector for semantic evaluation.
        top_k (int): Target count of supreme document chunks to return.
        rrf_k (int): Penalty constant tracking document stability across layers (Default: 60).
    """
    # 1. Fetch all candidate documents from local storage layer
    all_chunks = get_all_chunks_for_sparse()
    if not all_chunks:
        return []
        
    # --- LAYER 1: SPARSE RETRIEVAL (BM25 OKAPI) ---
    # Tokenize corpus and query for exact keyword tracking
    corpus_tokenized = [chunk["chunk_text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(corpus_tokenized)
    
    query_tokens = query_text.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Sort chunks by raw BM25 score to calculate precise ranking indices
    sparse_ranked_indices = np.argsort(bm25_scores)[::-1]
    
    sparse_ranks = {}
    for rank_idx, chunk_arr_idx in enumerate(sparse_ranked_indices):
        chunk_id = all_chunks[chunk_arr_idx]["id"]
        # Only rank if the document actually hit at least one keyword token (score > 0)
        if bm25_scores[chunk_arr_idx] > 0:
            sparse_ranks[chunk_id] = rank_idx + 1 # Rank position starts at 1
            
    # --- LAYER 2: DENSE RETRIEVAL (COSINE SIMILARITY) ---
    dense_scores = []
    for chunk in all_chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        dense_scores.append((chunk["id"], score))
        
    # Sort dense candidates by vector distance alignment
    dense_scores_sorted = sorted(dense_scores, key=lambda x: x[1], reverse=True)
    dense_ranks = {item[0]: rank_idx + 1 for rank_idx, item in enumerate(dense_scores_sorted)}
    
    # --- LAYER 3: RECIPROCAL RANK FUSION (RRF) ---
    rrf_scores = {}
    for chunk in all_chunks:
        c_id = chunk["id"]
        
        # Calculate sparse reciprocal rank
        sparse_rank = sparse_ranks.get(c_id, None)
        sparse_score = 1.0 / (rrf_k + sparse_rank) if sparse_rank is not None else 0.0
        
        # Calculate dense reciprocal rank
        dense_rank = dense_ranks.get(c_id)
        dense_score = 1.0 / (rrf_k + dense_rank)
        
        # Combined unified fused scoring track
        rrf_scores[c_id] = sparse_score + dense_score
        
    # Sort supreme document chunks by final combined RRF metrics
    sorted_chunks_by_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Map back to full chunk payloads and decorate with telemetry tracking metadata
    final_retrieved_chunks = []
    chunk_map = {chunk["id"]: chunk for chunk in all_chunks}
    
    for chunk_id, rrf_final_score in sorted_chunks_by_rrf[:top_k]:
        target_chunk = chunk_map[chunk_id]
        
        # Expose individual pipeline layer scores for explainability dashboard tracking
        target_chunk["rrf_score"] = rrf_final_score
        target_chunk["dense_rank"] = dense_ranks.get(chunk_id, -1)
        # Handle index positioning safety boundaries cleanly
        target_chunk["sparse_rank"] = sparse_ranks.get(chunk_id, -1)
        
        final_retrieved_chunks.append(target_chunk)
        
    return final_retrieved_chunks