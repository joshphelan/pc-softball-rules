"""
PDF Loader module for processing the USSSA rulebook PDF.
"""
import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_process_pdf(pdf_path: str) -> List[Document]:
    """
    Load and process a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects
    """
    logger.info(f"Loading PDF from {pdf_path}")
    
    try:
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        logger.info(f"Successfully loaded {len(documents)} pages from PDF")
        
        # Add metadata to documents
        for i, doc in enumerate(documents):
            # Add page number to metadata if not already present
            if "page" not in doc.metadata:
                doc.metadata["page"] = i + 1
            
            # Add source to metadata
            doc.metadata["source"] = pdf_path
        
        return documents
    
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
