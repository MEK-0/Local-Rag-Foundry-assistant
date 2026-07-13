import sqlite3
import json
import os
from typing import List, Dict, Any, Optional

DB_PATH = "data/rag.db"


def get_db_connection():
    """Establishes a robust local connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initializes the local hybrid database schema.
    Three layers:
      - documents: one row per source file, tracking a content hash so
        re-ingestion can detect "unchanged" (skip) vs "changed" (re-index)
        vs "new" (first index) without re-embedding everything every run.
      - knowledge_nodes: the full hierarchical tree (source of truth for
        parent-child expansion, headings, tables, figures, bbox/page anchors)
      - document_chunks: retrieval units produced by chunk_nodes(), each
        pointing back to its source node(s) via node_ids / parent_id
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
            node_count INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_nodes (
            id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            parent_id TEXT,
            type TEXT NOT NULL,
            level INTEGER DEFAULT 0,
            heading_path TEXT,
            content TEXT,
            page INTEGER,
            bbox TEXT,
            node_order INTEGER,
            metadata TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            source_file TEXT NOT NULL,
            page_number INTEGER DEFAULT 1,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            node_type TEXT DEFAULT 'paragraph',
            heading_path TEXT,
            node_ids TEXT,
            parent_id TEXT,
            bbox TEXT
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_parent ON knowledge_nodes(parent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_doc ON knowledge_nodes(doc_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent ON document_chunks(parent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(doc_id)")

    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Documents registry — drives incremental ingestion (hash-based dedup)
# --------------------------------------------------------------------------- #

def get_document_hash(doc_id: str) -> Optional[str]:
    """Returns the stored content hash for a doc_id, or None if never ingested."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT file_hash FROM documents WHERE doc_id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    return row["file_hash"] if row else None


def upsert_document_record(doc_id: str, filename: str, file_hash: str, node_count: int, chunk_count: int) -> None:
    """
    Records (or updates) a document's ingestion state. Called after a
    successful ingest so the next run can compare hashes and skip
    re-processing unchanged files.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (doc_id, filename, file_hash, node_count, chunk_count)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            filename = excluded.filename,
            file_hash = excluded.file_hash,
            ingested_at = CURRENT_TIMESTAMP,
            node_count = excluded.node_count,
            chunk_count = excluded.chunk_count
    """, (doc_id, filename, file_hash, node_count, chunk_count))
    conn.commit()
    conn.close()


def delete_document_data(doc_id: str) -> None:
    """
    Wipes all nodes and chunks belonging to a doc_id. Called before
    re-ingesting a file whose content hash has changed, so stale nodes/
    chunks from the previous version don't linger alongside the new ones
    (which would cause duplicate or contradictory retrieval results).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM knowledge_nodes WHERE doc_id = ?", (doc_id,))
    cursor.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()


def list_documents() -> List[Dict[str, Any]]:
    """Returns the ingestion registry — used by the dashboard to show what's indexed."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents ORDER BY ingested_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Knowledge tree (nodes) — used for parent-child expansion at retrieval time
# --------------------------------------------------------------------------- #

def insert_nodes(nodes: List[Any]) -> None:
    """Bulk-inserts KnowledgeNode objects (from parse_to_tree) into the tree table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    for node in nodes:
        d = node.to_dict() if hasattr(node, "to_dict") else node
        cursor.execute("""
            INSERT INTO knowledge_nodes (id, doc_id, parent_id, type, level, heading_path, content, page, bbox, node_order, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            d["id"], d["doc_id"], d.get("parent_id"), d["type"], d.get("level", 0),
            d.get("heading_path", ""), d.get("content", ""), d.get("page"),
            json.dumps(d.get("bbox")) if d.get("bbox") else None,
            d.get("order", 0), json.dumps(d.get("metadata", {})),
        ))
    conn.commit()
    conn.close()


def get_node(node_id: str) -> Optional[Dict[str, Any]]:
    """Fetches a single node by id (used to pull the parent's full content on demand)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM knowledge_nodes WHERE id = ?", (node_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["bbox"] = json.loads(d["bbox"]) if d["bbox"] else None
    d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
    return d


def get_children(node_id: str) -> List[Dict[str, Any]]:
    """Fetches all direct children of a node, ordered — used to reconstruct a full section."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM knowledge_nodes WHERE parent_id = ? ORDER BY node_order", (node_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Retrieval chunks (dense + sparse index source)
# --------------------------------------------------------------------------- #

def insert_chunk(chunk: Dict[str, Any], embedding: List[float]) -> None:
    """
    Persists a single retrieval chunk. `chunk` is one item produced by
    chunk_nodes() / chunk_document(): {"chunk_text": ..., "metadata": {...}}.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    meta = chunk["metadata"]
    embedding_blob = json.dumps(embedding).encode("utf-8")

    cursor.execute("""
        INSERT INTO document_chunks
            (doc_id, source_file, page_number, chunk_index, chunk_text, embedding,
             node_type, heading_path, node_ids, parent_id, bbox)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        meta.get("doc_id"), meta.get("source_file", ""), meta.get("page_number", 1),
        meta.get("chunk_index", 0), chunk["chunk_text"], embedding_blob,
        meta.get("node_type", "paragraph"), meta.get("heading_path", ""),
        json.dumps(meta.get("node_ids", [])), meta.get("parent_id"),
        json.dumps(meta.get("bbox")) if meta.get("bbox") else None,
    ))
    conn.commit()
    conn.close()


def get_all_chunks_for_sparse() -> List[Dict[str, Any]]:
    """Retrieves all indexed chunks to build/update the BM25 sparse index."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM document_chunks")
    rows = cursor.fetchall()
    conn.close()

    chunks = []
    for row in rows:
        d = dict(row)
        d["embedding"] = json.loads(d["embedding"].decode("utf-8"))
        d["node_ids"] = json.loads(d["node_ids"]) if d["node_ids"] else []
        d["bbox"] = json.loads(d["bbox"]) if d["bbox"] else None
        chunks.append(d)
    return chunks