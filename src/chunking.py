import os
import tiktoken
from typing import List
from src.config import settings

def get_token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def split_text_into_chunks(text: str) -> List[str]:
    """
    Splits text into token-aware chunks based on settings configuration.
    Uses a simple word-boundary approach backed by token counting.
    """
    words = text.split()
    chunks = []
    current_chunk_words = []
    
    chunk_size = settings.chunk_size_tokens
    overlap = settings.chunk_overlap_tokens
    
    for word in words:
        current_chunk_words.append(word)
        current_text = " ".join(current_chunk_words)
        
        # When we reach the token limit, save the chunk and apply overlap
        if get_token_count(current_text) >= chunk_size:
            chunks.append(current_text)
            
            # Keep the last 'overlap' amount of words to maintain context
            # (Approximation: 1 word ~ 1.3 tokens, so we keep overlap words)
            overlap_word_count = max(1, int(overlap * 0.75))
            current_chunk_words = current_chunk_words[-overlap_word_count:]
            
    # Add the remaining text as the last chunk
    if current_chunk_words:
        final_text = " ".join(current_chunk_words)
        if get_token_count(final_text) > 10: # Ignore tiny trailing chunks
            chunks.append(final_text)
            
    return chunks

def load_markdown_files(directory: str = "docs/sample_docs") -> dict:
    """
    Reads all markdown files in the specified directory.
    Returns a dictionary mapping filename to its content.
    """
    docs = {}
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return docs
        
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                docs[filename] = f.read()
                
    return docs