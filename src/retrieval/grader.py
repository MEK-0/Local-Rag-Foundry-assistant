from typing import List, Dict, Any

def grade_retrieved_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Grades retrieved chunks, enforces semantic uniqueness using Jaccard Similarity 
    to prevent local LLM token loops, and prunes dead weight.
    """
    valid_chunks = []
    seen_texts = set()
    
    for chunk in chunks:
        # Metni küçük harfe çevirip normalize ederek kelimelerine ayırıyoruz
        text_normalized = " ".join(chunk["chunk_text"].lower().split())
        words_current = set(text_normalized.split())
        
        is_duplicate = False
        for seen_text in seen_texts:
            words_seen = set(seen_text.split())
            
            # Jaccard Benzerlik Hesaplaması
            intersection = words_current.intersection(words_seen)
            union = words_current.union(words_seen)
            similarity = len(intersection) / len(union) if union else 0.0
            
            # Eğer iki parça %70'ten fazla benziyorsa duplicate kabul et ve engelle
            if similarity > 0.70:
                is_duplicate = True
                break
                
        if is_duplicate:
            continue
            
        # Basit anahtar kelime hit takibi (Güvenlik katmanı)
        text_lower = chunk["chunk_text"].lower()
        query_words = [w.lower() for w in query.split() if len(w) > 3]
        hit_count = sum(1 for word in query_words if word in text_lower)
        hit_ratio = hit_count / len(query_words) if query_words else 1.0
        
        # Parçayı tut: Cross-Encoder'dan iyi skor aldıysa veya kelime eşleşmesi varsa
        if chunk.get("rerank_score", 0.0) > -3.0 or hit_ratio >= 0.2:
            valid_chunks.append(chunk)
            seen_texts.add(text_normalized)
            
    return valid_chunks