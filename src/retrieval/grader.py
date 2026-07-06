from src.llm_client import generate_chat_response
from typing import List, Dict, Any

def grade_retrieved_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Grades retrieved chunks, enforces semantic uniqueness to prevent LLM token loops,
    and prunes dead weight. """

    valid_chunks = []
    seen_texts = set()
    
    for chunk in chunks:
        # Clean text layout slightly to evaluate strict normalized duplication
        text_normalized = " ".join(chunk["chunk_text"].lower().split())
        
        # Jaccard Similarity / Exact overlap gate to block repetitive chunks
        is_duplicate = False
        for seen_text in seen_texts:
            # If a chunk shares more than 70% of its words with an already selected chunk, block it
            words_a = set(text_normalized.split())
            words_b = set(seen_text.split())
            intersection = words_a.intersection(words_b)
            union = words_a.union(words_b)
            similarity = len(intersection) / len(union) if union else 0.0
            
            if similarity > 0.70:
                is_duplicate = True
                break
                
        if is_duplicate:
            continue
            
        # Standard query word hit tracking
        text_lower = chunk["chunk_text"].lower()
        query_words = [w.lower() for w in query.split() if len(w) > 3]
        hit_count = sum(1 for word in query_words if word in text_lower)
        hit_ratio = hit_count / len(query_words) if query_words else 1.0
        
        # Keep the chunk if it has strong cross-encoder validation or decent keyword hits
        if chunk.get("rerank_score", 0.0) > -3.0 or hit_ratio >= 0.2:
            valid_chunks.append(chunk)
            seen_texts.add(text_normalized)
            
    return valid_chunks
    