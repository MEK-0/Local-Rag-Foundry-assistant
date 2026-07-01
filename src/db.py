import sqlite3
import json
import os
from typing import List, Dict, Any

DB_PATH = os.path.join("data", "rag.db")

def init_db() -> None:
    """
    Initializes the SQLite database and creates the necessary tables 
    if they do not exist.
    """
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # We store the embedding as a JSON string of float arrays
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            chunk_text TEXT,
            embedding TEXT 
        )
    ''')
    
    conn.commit()
    conn.close()

def insert_chunk(source_file: str, chunk_text: str, embedding: List[float]) -> None:
    """
    Inserts a single document chunk and its vector embedding into the database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    embedding_json = json.dumps(embedding)
    
    cursor.execute('''
        INSERT INTO chunks (source_file, chunk_text, embedding)
        VALUES (?, ?, ?)
    ''', (source_file, chunk_text, embedding_json))
    
    conn.commit()
    conn.close()

def get_all_chunks() -> List[Dict[str, Any]]:
    """
    Retrieves all chunks from the database to perform local cosine similarity search.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, source_file, chunk_text, embedding FROM chunks')
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "source_file": row[1],
            "chunk_text": row[2],
            "embedding": json.loads(row[3])
        })
        
    return results