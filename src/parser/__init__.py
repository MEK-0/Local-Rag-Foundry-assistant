import os
from src.parsers.base import BaseParser
from src.parsers.pdf_parser import PDFDocumentParser
from src.parsers.docx_parser import DocxDocumentParser
from src.parsers.xlsx_parser import ExcelCSVDocumentParser

def get_parser(file_path: str) -> BaseParser:
    """
    Factory function to safely route documents to their dedicated formats parser.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return PDFDocumentParser()
    elif ext in ['.docx', '.doc']:
        return DocxDocumentParser()
    elif ext in ['.xlsx', '.xls', '.csv']:
        return ExcelCSVDocumentParser()
    else:
        # Fallback parser placeholder if specialized formats are not requested
        raise ValueError(f"Unsupported file extension: '{ext}'")