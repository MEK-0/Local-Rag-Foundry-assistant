import os
from typing import List

import pandas as pd

from src.parser.base import BaseParser, KnowledgeNode, NodeType

# Beyond this many rows, a single markdown table node risks exceeding
# embedding/LLM context limits and makes retrieval too coarse (a lookup
# question would pull in a huge irrelevant table just to get one row).
# Large sheets are split into multiple table nodes instead, each still
# atomic (never split further downstream by chunking.py) but small
# enough to embed and retrieve meaningfully.
MAX_ROWS_PER_TABLE_NODE = 200


class ExcelCSVDocumentParser(BaseParser):
    def parse_to_tree(self, file_path: str, doc_id: str) -> List[KnowledgeNode]:
        """
        Converts a spreadsheet into a KnowledgeNode tree.
        Each sheet becomes a SECTION node; its data becomes one or more
        markdown TABLE nodes underneath it. Sheets larger than
        MAX_ROWS_PER_TABLE_NODE rows are split into multiple table nodes
        (each repeating the header row), instead of one node holding the
        entire sheet - keeps individual nodes retrievable and within
        reasonable embedding/context size.
        """
        ext = os.path.splitext(file_path)[1].lower()
        nodes: List[KnowledgeNode] = []
        order_counter = 0

        if ext == ".csv":
            sheets = {"CSV_Default": pd.read_csv(file_path)}
        else:
            sheets = pd.read_excel(file_path, sheet_name=None)  # all sheets

        for sheet_name, df in sheets.items():
            order_counter += 1
            section_node = KnowledgeNode(
                doc_id=doc_id,
                type=NodeType.SECTION,
                level=1,
                content=str(sheet_name),
                heading_path=str(sheet_name),
                order=order_counter,
            )
            nodes.append(section_node)

            if df.empty:
                continue

            row_chunks = self._split_rows(df, MAX_ROWS_PER_TABLE_NODE)
            total_parts = len(row_chunks)

            for part_idx, row_chunk_df in enumerate(row_chunks, start=1):
                md_table = self._df_to_markdown(row_chunk_df)
                if not md_table:
                    continue

                order_counter += 1
                part_label = f" (part {part_idx}/{total_parts})" if total_parts > 1 else ""
                nodes.append(
                    KnowledgeNode(
                        doc_id=doc_id,
                        parent_id=section_node.id,
                        type=NodeType.TABLE,
                        level=2,
                        heading_path=f"{section_node.heading_path}{part_label}",
                        content=md_table,
                        order=order_counter,
                        metadata={
                            "rows": len(row_chunk_df),
                            "columns": len(row_chunk_df.columns),
                            "row_range": (
                                f"{row_chunk_df.index[0] + 1}-{row_chunk_df.index[-1] + 1}"
                                if len(row_chunk_df) > 0 else ""
                            ),
                        },
                    )
                )

        return nodes

    @staticmethod
    def _split_rows(df: pd.DataFrame, max_rows: int) -> List[pd.DataFrame]:
        """Splits a DataFrame into consecutive row-range chunks of at most
        max_rows each. Returns [df] unchanged if it already fits."""
        if len(df) <= max_rows:
            return [df]
        return [df.iloc[i:i + max_rows] for i in range(0, len(df), max_rows)]

    @staticmethod
    def _df_to_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        headers = [str(c) for c in df.columns]
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for _, row in df.iterrows():
            cells = ["" if pd.isna(v) else str(v).replace("\n", " ") for v in row]
            md += "| " + " | ".join(cells) + " |\n"
        return md