import os
import re
from typing import List, Dict, Any, Optional

from src.parser.base import KnowledgeNode, NodeType


# --------------------------------------------------------------------------- #
# NEW: tree-aware chunking (consumes KnowledgeNode list from parse_to_tree)
# --------------------------------------------------------------------------- #

# Node types that are indexed as-is, never split (splitting a table/figure
# destroys the exact information a lookup query needs)
ATOMIC_TYPES = {NodeType.TABLE, NodeType.FIGURE, NodeType.WARNING, NodeType.NOTE, NodeType.CODE}


def chunk_nodes(
    nodes: List[KnowledgeNode],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Converts a flat KnowledgeNode list (produced by parse_to_tree) into
    retrieval-ready chunks.

    Key differences from the old fixed-size splitter:
      - Splitting respects heading boundaries (topic-boundary chunking),
        instead of blindly cutting every N words regardless of section.
      - Table / Figure / Warning / Note nodes are never split; they are
        indexed atomically so a lookup ("which grease should be used?")
        returns the whole table, not a fragment of it.
      - Every chunk keeps a reference back to its source node id(s) and
        heading_path, so compression.py can later expand to the parent
        node for full context (parent-child retrieval).

    Args:
        nodes: Flat KnowledgeNode list from a parser's parse_to_tree().
        chunk_size: Target chunk size in words for paragraph grouping.
        chunk_overlap: Word overlap between consecutive split chunks.

    Returns:
        List[Dict[str, Any]]: each item has:
            - "chunk_text": str
            - "metadata": dict (source_file, page_number, chunk_index,
              node_type, heading_path, node_ids, parent_id, bbox)
    """
    chunks_data: List[Dict[str, Any]] = []
    chunk_index = 0

    # Group paragraph nodes by their immediate parent, so grouping never
    # crosses a heading boundary (this is the "semantic" part: sections
    # are the natural topic boundary, not an arbitrary word count)
    paragraph_groups: Dict[Optional[str], List[KnowledgeNode]] = {}

    for node in nodes:
        if node.type in ATOMIC_TYPES:
            # Atomic node: emit as its own chunk immediately, no splitting
            chunks_data.append(_node_to_chunk(node, chunk_index, node_ids=[node.id]))
            chunk_index += 1
        elif node.type == NodeType.PARAGRAPH:
            paragraph_groups.setdefault(node.parent_id, []).append(node)
        # SECTION nodes are structure, not content -> not indexed as a
        # standalone chunk; their heading text lives in heading_path
        # metadata of their descendant chunks instead

    # Now split each per-section paragraph group with sliding-window overlap,
    # so long sections still get multiple chunks but short ones stay intact
    for parent_id, para_nodes in paragraph_groups.items():
        merged_text = " ".join(n.content for n in para_nodes)
        words = merged_text.split()
        if not words:
            continue

        step = max(chunk_size - chunk_overlap, 1)
        for i in range(0, len(words), step):
            window_words = words[i : i + chunk_size]
            chunk_text = " ".join(window_words).strip()
            if not chunk_text:
                continue

            ref_node = para_nodes[0]  # representative node for shared metadata
            chunks_data.append(
                _node_to_chunk(
                    ref_node,
                    chunk_index,
                    node_ids=[n.id for n in para_nodes],
                    override_text=chunk_text,
                )
            )
            chunk_index += 1

    return chunks_data


def _node_to_chunk(
    node: KnowledgeNode,
    chunk_index: int,
    node_ids: List[str],
    override_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Builds the final chunk dict from a KnowledgeNode (or a merged group)."""
    return {
        "chunk_text": override_text if override_text is not None else node.content,
        "metadata": {
            "doc_id": node.doc_id,
            "source_file": node.doc_id,
            "page_number": node.page,
            "chunk_index": chunk_index,
            "node_type": node.type.value if isinstance(node.type, NodeType) else node.type,
            "heading_path": node.heading_path,
            "node_ids": node_ids,      # used by compression.py for parent expansion
            "parent_id": node.parent_id,
            "bbox": node.bbox,
        },
    }


# --------------------------------------------------------------------------- #
# LEGACY: fixed-size text chunking (kept for parsers not yet migrated to
# parse_to_tree, e.g. markdown_parser.py / docx_parser.py / xlsx_parser.py)
# --------------------------------------------------------------------------- #

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
        # Document has explicit page structure (PDF mode)
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