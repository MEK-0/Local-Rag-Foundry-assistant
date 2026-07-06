import re
from typing import List, Dict, Any

def compress_context_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compresses long context tracks down to target sentence windows surrounding core query components.
    """
    compressed_chunks = []
    query_words = [w.lower() for w in query.split() if len(w) > 3]
    
    for chunk in chunks:
        # Split text cleanly into distinct operational sentences
        sentences = re.split(r'(?<=[.!?])\s+', chunk["chunk_text"])
        relevant_indices = set()
        
        for idx, sentence in enumerate(sentences):
            if any(word in sentence.lower() for word in query_words):
                relevant_indices.add(idx)
                # Sentence window expander: grab immediate neighbor blocks for context preservation
                if idx > 0: relevant_indices.add(idx - 1)
                if idx < len(sentences) - 1: relevant_indices.add(idx + 1)
                
        if relevant_indices:
            sorted_indices = sorted(list(relevant_indices))
            compressed_text = " ".join([sentences[i] for i in sorted_indices])
            
            # Update the chunk frame with the newly distilled high-density context
            chunk["chunk_text"] = compressed_text
            compressed_chunks.append(chunk)
        else:
            # Fallback if text metrics are uniform
            compressed_chunks.append(chunk)
            
    return compressed_chunks