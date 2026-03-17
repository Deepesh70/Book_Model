import re
from typing import List, Any
import logging

logger = logging.getLogger(__name__)

class DocumentCleaner:
    def __init__(self, min_length: int = 50):
        self.min_length = min_length
        # Regex for common PDF artifacts or excessive whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        self.control_chars_pattern = re.compile(r'[\x00-\x1f\x7f-\x9f]')

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
            
        # Safely remove surrogate code points that crash Windows terminals
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        
        # Remove control characters
        text = self.control_chars_pattern.sub('', text)
        
        # Normalize whitespace (replace newlines/tabs with space and collapse)
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text

    def clean_documents(self, documents: List[Any]) -> List[Any]:
        """
        Cleans a list of LangChain Document objects.
        Filters out documents that are too short after cleaning.
        """
        cleaned_docs = []
        for doc in documents:
            if not hasattr(doc, 'page_content') or doc.page_content is None:
                continue
            
            cleaned_text = self.clean_text(str(doc.page_content))
            
            if len(cleaned_text) >= self.min_length:
                # Update the document content with cleaned version
                doc.page_content = cleaned_text
                cleaned_docs.append(doc)
            
        print(f"[INFO] Data Cleaning: {len(documents)} -> {len(cleaned_docs)} documents (filtered {len(documents) - len(cleaned_docs)})")
        return cleaned_docs

def get_default_cleaner():
    return DocumentCleaner()
