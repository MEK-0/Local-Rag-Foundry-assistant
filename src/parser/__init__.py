import os
from src.parser.base import BaseParser
from src.parser.markdown_parser import MarkdownDocumentParser
from src.parser.pdf_parser import PDFDocumentParser
from src.parser.docx_parser import DocxDocumentParser
from src.parser.xlsx_parser import ExcelCSVDocumentParser

def get_parser(file_path: str) -> BaseParser:
    """
    Factory function to safely route documents to their dedicated formats parser.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.md':
        return MarkdownDocumentParser()
    elif ext == '.pdf':
        return PDFDocumentParser()
    elif ext in ['.docx', '.doc']:
        return DocxDocumentParser()
    elif ext in ['.xlsx', '.xls', '.csv']:
        return ExcelCSVDocumentParser()
    else:
        raise ValueError(f"Unsupported file extension: '{ext}'")