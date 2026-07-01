import os
import sys

# Add the project root to the python path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.db import init_db, insert_chunk
from src.chunking import load_markdown_files, split_text_into_chunks
from src.llm_client import get_embedding

def main():
    print(f"Starting ingestion in {settings.mode.upper()} mode...")
    
    if settings.mode != "local":
        print("Error: This script is intended for local mode SQLite ingestion.")
        print("Please use scripts/sync_azure.py for cloud mode deployments.")
        return

    print("Initializing local database...")
    init_db()

    print("Loading markdown documents...")
    docs_dir = os.path.join("docs", "sample_docs")
    docs = load_markdown_files(docs_dir)
    
    if not docs:
        print(f"No markdown files found in {docs_dir}/. Please add some and try again.")
        return

    total_chunks = 0
    for filename, content in docs.items():
        print(f"Processing '{filename}'...")
        chunks = split_text_into_chunks(content)
        
        for i, chunk in enumerate(chunks):
            print(f"  - Generating embedding for chunk {i+1}/{len(chunks)}...")
            try:
                embedding = get_embedding(chunk)
                insert_chunk(source_file=filename, chunk_text=chunk, embedding=embedding)
                total_chunks += 1
            except Exception as e:
                print(f"  - Error generating embedding for chunk: {e}")

    print(f"\nIngestion complete! Successfully stored {total_chunks} chunks in the local SQLite database.")

if __name__ == "__main__":
    main()