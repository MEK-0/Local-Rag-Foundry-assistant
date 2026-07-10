import re
from typing import List, Dict, Any

def compress_context_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compresses long context tracks down to target sentence windows,
    while safeguarding clean high-rank structural metadata and cover pages.
    """
    compressed_chunks = []
    query_words = [w.lower() for w in query.split() if len(w) > 3]
    
    for chunk in chunks:
        # İSTİSNA KAPISI: Eğer parça kapak sayfalarına aitse cümle bazlı budama yapma,
        # sadece telif ve doküman numarası gürültülerini temizle.
        if chunk.get("page_number") in [1, 2, "1", "2"]:
            lines = chunk["chunk_text"].split("\n")
            clean_lines = []
            for line in lines:
                line_strip = line.strip()
                if any(noise in line_strip.lower() for noise in ["no part of this", "subject to change", "all rights", "b-63527en"]):
                    continue
                if len(line_strip) > 2:
                    clean_lines.append(line_strip)
            
            chunk["chunk_text"] = "\n".join(clean_lines)
            compressed_chunks.append(chunk)
            continue
            
        # İSTİSNA KAPISI: Cross-Encoder skoru çok yüksekse listelerin yapısını korumak için ham geçiş ver
        if chunk.get("rerank_score", 0.0) > 1.5:
            compressed_chunks.append(chunk)
            continue

        # Standart Metinler İçin Cümle Penceresi Oluşturma
        sentences = re.split(r'(?<=[.!?])\s+', chunk["chunk_text"])
        relevant_indices = set()
        
        for idx, sentence in enumerate(sentences):
            if any(word in sentence.lower() for word in query_words):
                relevant_indices.add(idx)
                if idx > 0: relevant_indices.add(idx - 1) # Önündeki komşu cümle
                if idx < len(sentences) - 1: relevant_indices.add(idx + 1) # Arkasındaki komşu cümle
                
        if relevant_indices:
            sorted_indices = sorted(list(relevant_indices))
            compressed_text = " ".join([sentences[i] for i in sorted_indices])
            chunk["chunk_text"] = compressed_text
            compressed_chunks.append(chunk)
            
    return compressed_chunks