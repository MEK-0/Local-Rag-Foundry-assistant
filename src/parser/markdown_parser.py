import os
from typing import Dict, Any
from src.parser.base import BaseParser

class MarkdownDocumentParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts raw text content from Markdown (.md) files.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        return {
            "content": content,
            "metadata": {
                "file_size_bytes": os.path.getsize(file_path),
                "word_count": len(content.split()),
                "parser_used": "native_markdown"
            }
        }