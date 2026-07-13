import os
from typing import List

import pandas as pd

from src.parser.base import BaseParser, KnowledgeNode, NodeType


class ExcelCSVDocumentParser(BaseParser):
    def parse_to_tree(self, file_path: str, doc_id: str) -> List[KnowledgeNode]:
        """
        Converts a spreadsheet into a KnowledgeNode tree.
        Each sheet becomes a SECTION node; its full data becomes a single
        markdown TABLE node underneath it, instead of one text line per row.

        Why not per-row text lines (the old approach): a lookup question
        ("which grease for SCARA?") needs the row's columns aligned with
        their headers. A markdown table preserves that alignment for the
        LLM; a flattened "Row 5: col is 'x', col is 'y'" sentence does not,
        and it also multiplies node count unnecessarily for large sheets.
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

            md_table = self._df_to_markdown(df)
            if not md_table:
                continue

            order_counter += 1
            nodes.append(
                KnowledgeNode(
                    doc_id=doc_id,
                    parent_id=section_node.id,
                    type=NodeType.TABLE,
                    level=2,
                    heading_path=section_node.heading_path,
                    content=md_table,
                    order=order_counter,
                    metadata={"rows": len(df), "columns": len(df.columns)},
                )
            )

        return nodes

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