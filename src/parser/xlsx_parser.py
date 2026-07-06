import pandas as pd
import os
from typing import Dict, Any
from src.parser.base import BaseParser

class ExcelCSVDocumentParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Converts spreadsheet rows into structured textual descriptions 
        to maximize dense retrieval and LLM alignment.
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        # Handle both CSV and Excel extensions dynamically
        if ext == '.csv':
            df = pd.read_csv(file_path)
            sheet_names = ["CSV_Default"]
        else:
            df = pd.read_excel(file_path, sheet_name=0) # Read the primary sheet
            sheet_names = [str(sheet_name) for sheet_name in pd.ExcelFile(file_path).sheet_names]
            
        structured_lines = []
        
        # Convert spreadsheet matrix rows into descriptive semantic descriptions
        for idx, row in df.iterrows():
            row_desc = f"Row {idx + 1}: "
            items = [f"{col_name} is '{val}'" for col_name, val in row.items() if pd.notna(val)]
            row_desc += ", ".join(items) + "."
            structured_lines.append(row_desc)
            
        return {
            "content": "\n".join(structured_lines),
            "metadata": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "sheets": sheet_names,
                "parser_used": "pandas"
            }
        }