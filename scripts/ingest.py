import os
import sys
from typing import List

# Append project root directory to the Python search path for clean internal modules mapping
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import init_db, insert_chunk
from src.parser import get_parser
from src.chunking import chunk_document
from src.llm_client import get_embedding

def run_ingestion():
    """
    Orchestrates the offline multi-format ingestion pipeline.
    Parses, semantically splits, embeds, and loads documents into the hybrid SQLite store.
    """
    print("🚀 Initializing hybrid SQLite database schema...")
    init_db()
    
    source_dir = "docs/sample_docs"
    if not os.path.exists(source_dir):
        print(f"❌ Critical Error: Source directory '{source_dir}' not found.")
        return
        
    supported_extensions = ['.md', '.pdf', '.docx', '.xlsx', '.xls', '.csv']
    
    print(f"📦 Scanning local enterprise assets inside '{source_dir}'...")
    
    try:
        files = os.listdir(source_dir)
    except Exception as e:
        print(f"❌ Failed to scan directory context: {str(e)}")
        return

    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_extensions:
            continue
            
        file_path = os.path.join(source_dir, filename)
        print(f"\n📄 Ingesting Asset: {filename}")
        
        try:
            # 1. Dynamically resolve the dedicated file format parser via factory routing
            parser = get_parser(file_path)
            parsed_data = parser.parse(file_path)
            
            raw_content = parsed_data["content"]
            file_metadata = parsed_data["metadata"]
            
            # 2. Process extracted character strings into token-aware sliding fragments
            chunks_data = chunk_document(file_path, raw_content)
            print(f"   -> Extracted {len(chunks_data)} semantic fragments utilizing [{file_metadata['parser_used']}] parser.")
            
            # 3. Compute structural embedding layers and load sequentially into the hybrid storage schema
            for item in chunks_data:
                chunk_text = item["chunk_text"]
                chunk_metadata = item["metadata"]
                
                # Fetch localized vector representation from Foundry Local interface
                embedding = get_embedding(chunk_text)
                
                # Commit payload parameters directly to the database layer
                insert_chunk(
                    source_file=chunk_metadata["source_file"],
                    page_number=chunk_metadata["page_number"],
                    chunk_index=chunk_metadata["chunk_index"],
                    chunk_text=chunk_text,
                    embedding=embedding
                )
            print(f"   ✅ Successfully indexed pipeline tracks for: {filename}")
            
        except Exception as e:
            print(f"   ❌ Pipeline Fault processing file [{filename}]: {str(e)}")

if __name__ == "__main__":
    run_ingestion()