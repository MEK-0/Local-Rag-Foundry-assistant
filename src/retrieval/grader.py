from src.llm_client import generate_chat_response
from typing import List, Dict, Any

def grade_retrieved_chunks(query: str, chunks: List[Dict[str, Any]], confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Evaluates each chunk against the query using light heuristic patterns to prune dead weight.
    """
    valid_chunks = []
    
    for chunk in chunks:
        text_lower = chunk["chunk_text"].lower()
        query_words = [w.lower() for w in query.split() if len(w) > 3]
        
        # Calculate localized word hit match frequencies
        hit_count = sum(1 for word in query_words if word in text_lower)
        hit_ratio = hit_count / len(query_words) if query_words else 1.0
        
        # If cross encoder scored it high OR we have an ironclad keyword match fraction, keep it
        if chunk.get("rerank_score", 0.0) > -2.0 or hit_ratio >= 0.3:
            valid_chunks.append(chunk)
            
    return valid_chunks