import sqlite3
import json
import os
from typing import List, Dict, Any

DB_PATH = "data/rag.db"

def get_db_connection():
    """Establishes a robust local connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enables column access by name like a dict
    return conn

def init_db():
    """
    Initializes the local hybrid database schema.
    Creates tables for vector storage and structured document tracks.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Core documents table with extended meta-data properties
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            page_number INTEGER DEFAULT 1,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

def insert_chunk(source_file: str, page_number: int, chunk_index: int, chunk_text: str, embedding: List[float]):
    """
    Persists a single tokenized semantic chunk into the vector layer.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Serialize float array into float-binary blob or plain JSON string depending on local preferences
    embedding_blob = json.dumps(embedding).encode('utf-8')
    
    cursor.execute("""
        INSERT INTO document_chunks (source_file, page_number, chunk_index, chunk_text, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, (source_file, page_number, chunk_index, chunk_text, embedding_blob))
    
    conn.commit()
    conn.close()

def get_all_chunks_for_sparse() -> List[Dict[str, Any]]:
    """
    Retrieves all indexed chunks to build or update the runtime BM25 sparse index token grid.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, source_file, page_number, chunk_index, chunk_text, embedding FROM document_chunks")
    rows = cursor.fetchall()
    
    chunks = []
    for row in rows:
        chunks.append({
            "id": row["id"],
            "source_file": row["source_file"],
            "page_number": row["page_number"],
            "chunk_index": row["chunk_index"],
            "chunk_text": row["chunk_text"],
            "embedding": json.loads(row["embedding"].decode('utf-8'))
        })
    
    conn.close()
    return chunks