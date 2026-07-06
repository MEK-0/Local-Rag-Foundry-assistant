import os
import re
from typing import List, Dict, Any

def chunk_document(file_path: str, content: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Splits document content into token/character-aware chunks and attaches structural metadata.
    
    Args:
        file_path (str): The path of the source file to extract metadata like filename.
        content (str): The raw text content extracted by the specialized parser.
        chunk_size (int): Approximate target size for each text chunk (in words/characters).
        chunk_overlap (int): Overlap size between consecutive chunks to preserve context boundaries.
        
    Returns:
        List[Dict[str, Any]]: A list of structured dictionaries, each containing:
            - "chunk_text": str
            - "metadata": dict (source_file, page_number, chunk_index)
    """
    filename = os.path.basename(file_path)
    chunks_data = []
    
    # Check if the document contains explicit page markers (inserted by PDFParser)
    # e.g., "[Page 1]\nOperational steps..."
    page_segments = re.split(r'\[Page\s+(\d+)\]\n', content)
    
    if len(page_segments) > 1:
        # Document has explicit page structures (PDF mode)
        chunk_index = 0
        # re.split with capturing groups returns: [pre_text, page_num, page_content, page_num, page_content...]
        # If pre_text is empty, we skip it
        iterator = iter(page_segments)
        first_item = next(iterator, "")
        
        while True:
            try:
                page_num_str = next(iterator)
                page_content = next(iterator)
                page_num = int(page_num_str)
                
                # Split this specific page into smaller sliding-window overlapping fragments
                words = page_content.split()
                if not words:
                    continue
                    
                for i in range(0, len(words), chunk_size - chunk_overlap):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = " ".join(chunk_words)
                    
                    if chunk_text.strip():
                        chunks_data.append({
                            "chunk_text": chunk_text,
                            "metadata": {
                                "source_file": filename,
                                "page_number": page_num,
                                "chunk_index": chunk_index
                            }
                        })
                        chunk_index += 1
            except StopIteration:
                break
    else:
        # Document is a continuous stream (Markdown, DOCX, CSV/XLSX mode)
        words = content.split()
        chunk_index = 0
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():
                # For non-PDF assets, default page_number to 1 or log row indices if available
                chunks_data.append({
                    "chunk_text": chunk_text,
                    "metadata": {
                        "source_file": filename,
                        "page_number": 1, 
                        "chunk_index": chunk_index
                    }
                })
                chunk_index += 1
                
    return chunks_data