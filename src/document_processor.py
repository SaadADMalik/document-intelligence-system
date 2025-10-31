"""
Document Processing Module
Handles PDF ingestion, text extraction, cleaning, and chunking
"""
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import PyPDF2
from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.pdfparser import PDFSyntaxError
from src.utils import get_logger

logger = get_logger()


@dataclass
class Document:
    """Document data structure"""
    filename: str
    text: str
    chunks: List[str]
    metadata: Dict[str, any]
    error: Optional[str] = None


class DocumentProcessor:
    """Process PDF documents with fallback strategies"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Number of characters per chunk for semantic search
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"DocumentProcessor initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    def process_folder(self, folder_path: str) -> List[Document]:
        """
        Process all PDF files in a folder
        
        Args:
            folder_path: Path to folder containing PDFs
            
        Returns:
            List of processed Document objects
        """
        folder = Path(folder_path)
        if not folder.exists():
            logger.error(f"Folder not found: {folder_path}")
            return []
        
        pdf_files = list(folder.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
        
        documents = []
        for pdf_file in pdf_files:
            doc = self.process_document(str(pdf_file))
            documents.append(doc)
        
        # Summary
        success_count = sum(1 for d in documents if d.error is None)
        failed_count = len(documents) - success_count
        logger.success(f"Processed {success_count}/{len(documents)} documents successfully")
        if failed_count > 0:
            logger.warning(f"{failed_count} documents failed to process")
        
        return documents
    
    def process_document(self, file_path: str) -> Document:
        """
        Process a single PDF document
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document object with extracted text and chunks
        """
        filename = Path(file_path).name
        logger.info(f"Processing: {filename}")
        
        # Try extraction
        text, error = self._extract_text(file_path)
        
        if error:
            logger.error(f"  ✗ {filename}: {error}")
            return Document(
                filename=filename,
                text="",
                chunks=[],
                metadata={"file_path": file_path, "extraction_method": "failed"},
                error=error
            )
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Create chunks for semantic search
        chunks = self._create_chunks(cleaned_text)
        
        # Metadata
        metadata = {
            "file_path": file_path,
            "text_length": len(cleaned_text),
            "num_chunks": len(chunks),
            "extraction_method": "success"
        }
        
        logger.success(f"  ✓ {filename}: {len(cleaned_text)} chars, {len(chunks)} chunks")
        
        return Document(
            filename=filename,
            text=cleaned_text,
            chunks=chunks,
            metadata=metadata
        )
    
    def _extract_text(self, file_path: str) -> Tuple[str, Optional[str]]:
        """
        Extract text from PDF with fallback strategy
        
        Strategy:
        1. Try PyPDF2 (fast, works for most PDFs)
        2. Fallback to pdfminer.six (slower but handles complex PDFs)
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, error_message)
        """
        # Method 1: PyPDF2 (fast path)
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                if text.strip():  # Success if non-empty
                    return text, None
        except Exception as e:
            logger.warning(f"  PyPDF2 failed, trying pdfminer: {str(e)[:50]}")
        
        # Method 2: pdfminer.six (fallback for complex/scanned PDFs)
        try:
            text = pdfminer_extract(file_path)
            if text.strip():
                return text, None
            else:
                return "", "Empty PDF or no extractable text"
        except PDFSyntaxError:
            return "", "Corrupted PDF file (PDFSyntaxError)"
        except Exception as e:
            return "", f"Extraction failed: {str(e)[:100]}"
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove multiple whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize unicode characters
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        return text
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for semantic search
        
        Args:
            text: Cleaned text
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near chunk boundary
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.7:  # At least 70% of chunk
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap  # Overlap for context
        
        return [c for c in chunks if c]  # Remove empty chunks


def test_processor():
    """Test the document processor"""
    processor = DocumentProcessor()
    
    # Test on sample folder
    documents = processor.process_folder("data/input_documents")
    
    print(f"\n{'='*60}")
    print("DOCUMENT PROCESSOR TEST RESULTS")
    print(f"{'='*60}")
    
    for doc in documents:
        print(f"\nFile: {doc.filename}")
        print(f"  Text Length: {len(doc.text)}")
        print(f"  Chunks: {len(doc.chunks)}")
        print(f"  Error: {doc.error if doc.error else 'None'}")
        if doc.text:
            print(f"  Preview: {doc.text[:100]}...")


if __name__ == "__main__":
    test_processor()
