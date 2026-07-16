import os
import sys
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import (
    init_db, insert_chunk, insert_nodes,
    get_document_hash, upsert_document_record, delete_document_data,
)
from src.parser import get_parser
from src.chunking import chunk_document, chunk_nodes
from src.llm_client import get_embedding
from src.config import settings
from src.graph_builder import build_document_graph
from src.parser.base import NodeType
from src.llm_client import get_embedding, EMBEDDING_MODEL_NAME

def run_ingestion():
    """
    Orchestrates the offline multi-format ingestion pipeline with
    hash-based incremental dedup:
      - unchanged file (hash matches stored record) -> skip entirely,
        no re-parsing, no re-embedding
      - changed or new file -> wipe any stale nodes/chunks for that
        doc_id, then re-parse/chunk/embed/index from scratch

    Two parsing paths depending on parser capability:
      - Tree-aware parsers (implement parse_to_tree, e.g. PDFDocumentParser):
        nodes persisted to knowledge_nodes, then chunk_nodes() builds
        retrieval chunks that keep heading_path / parent_id / node_type.
      - Legacy parsers (not yet migrated): fall back to parse() + chunk_document().
    """
    print("Initializing hybrid SQLite database schema...")
    init_db()

    source_dir = "docs/sample_docs"
    if not os.path.exists(source_dir):
        print(f"Critical Error: Source directory '{source_dir}' not found.")
        return

    supported_extensions = ['.md', '.pdf', '.docx', '.xlsx', '.xls', '.csv']

    print(f"Scanning local enterprise assets inside '{source_dir}'...")
    try:
        files = os.listdir(source_dir)
    except Exception as e:
        print(f"Failed to scan directory context: {str(e)}")
        return

    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_extensions:
            continue

        file_path = os.path.join(source_dir, filename)
        print(f"\nIngesting Asset: {filename}")

        try:
            doc_id = _make_doc_id(filename)
            current_hash = _file_hash(file_path)
            stored_hash = get_document_hash(doc_id)

            if stored_hash == current_hash:
                print(f"   Unchanged since last ingestion - skipping ({filename}).")
                continue

            if stored_hash is not None:
                # File existed before but content changed - clear stale
                # nodes/chunks first, otherwise old and new versions would
                # coexist and retrieval would return duplicate/contradictory chunks
                print("   Content changed - clearing previous index entries...")
                delete_document_data(doc_id)

            parser = get_parser(file_path)

            if hasattr(parser, "parse_to_tree"):
                chunks_data = _ingest_tree_aware(parser, file_path, doc_id, filename)
            else:
                chunks_data = _ingest_legacy(parser, file_path, filename, doc_id)

            print(f"   -> Extracted {len(chunks_data)} chunks for indexing.")

            indexed_count = 0
            for item in chunks_data:
                # Table/figure/warning nodes can be empty at this stage
                # (e.g. a figure awaiting vision captioning) - skip embedding
                # empty text, nothing useful to index yet
                if not item["chunk_text"].strip():
                    continue

                embedding = get_embedding(item["chunk_text"])
                insert_chunk(item, embedding)
                indexed_count += 1

            upsert_document_record(
                doc_id=doc_id,
                filename=filename,
                file_hash=current_hash,
                node_count=len(chunks_data),
                chunk_count=indexed_count,
                embedding_model=EMBEDDING_MODEL_NAME,
            )

            print(f"   Successfully indexed: {filename} ({indexed_count} chunks)")

        except Exception as e:
            print(f"   Pipeline Fault processing file [{filename}]: {str(e)}")


def _file_hash(file_path: str) -> str:
    """Content hash used to detect whether a file changed since last ingestion."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _make_doc_id(filename: str) -> str:
    """Stable doc_id derived from filename, so re-ingestion of the same file
    always maps to the same doc_id (required for hash-based dedup to work -
    otherwise every run would look like a "new" document)."""
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]


def _ingest_tree_aware(parser, file_path: str, doc_id: str, filename: str):
    nodes = parser.parse_to_tree(file_path, doc_id=doc_id)
    if not nodes:
        return []

    insert_nodes(nodes)
    chunks_data = chunk_nodes(
        nodes,
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
    )

    # chunk_nodes doesn't set metadata["doc_id"] on its own (it only carries
    # doc_id inside each node, not surfaced to the chunk-level metadata dict).
    # Set both here: doc_id for the FK/dedup column in document_chunks, and
    # source_file overwritten to the human-readable filename so telemetry/UI
    # shows "manual.pdf" instead of a hash.
    for item in chunks_data:
        item["metadata"]["doc_id"] = doc_id
        item["metadata"]["source_file"] = filename

    return chunks_data


def _ingest_legacy(parser, file_path: str, filename: str, doc_id: str):
    parsed_data = parser.parse(file_path)
    raw_content = parsed_data["content"]
    chunks_data = chunk_document(
        file_path,
        raw_content,
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
    )

    # Legacy chunker doesn't know about doc_id - attach it here so
    # insert_chunk's doc_id column and future dedup lookups stay consistent
    for item in chunks_data:
        item["metadata"]["doc_id"] = doc_id

    return chunks_data

def _ingest_tree_aware(parser, file_path: str, doc_id: str, filename: str):
    nodes = parser.parse_to_tree(file_path, doc_id=doc_id)
    if not nodes:
        return []

    insert_nodes(nodes)
    chunks_data = chunk_nodes(nodes, chunk_size=settings.chunk_size_tokens, chunk_overlap=settings.chunk_overlap_tokens)

    for item in chunks_data:
        item["metadata"]["doc_id"] = doc_id
        item["metadata"]["source_file"] = filename

    # Build the entity co-occurrence graph from this document's sections.
    # Runs after insert_nodes() since build_document_graph reads children
    # back from the DB via get_children().
    section_nodes = [n for n in nodes if n.type == NodeType.SECTION]
    if section_nodes:
        sections_processed = build_document_graph(doc_id, section_nodes)
        print(f"   -> Built entity graph from {sections_processed} sections.")

    return chunks_data

if __name__ == "__main__":
    run_ingestion()