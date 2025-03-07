"""
Text splitter module for chunking documents.
"""
import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_markdown(markdown_text: str) -> List[Document]:
    """
    Split markdown text based on headers.
    
    Args:
        markdown_text: Markdown text to split
        
    Returns:
        List of Document objects with chunked content
    """
    logger.info("Splitting markdown text based on headers")
    
    try:
        # Define headers to split on
        headers_to_split_on = [
            ("#", "section"),
            ("##", "subsection"),
            ("###", "subsubsection"),
        ]
        
        # Create markdown splitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        # Split the markdown text
        md_header_splits = markdown_splitter.split_text(markdown_text)
        
        logger.info(f"Successfully split markdown into {len(md_header_splits)} chunks")
        
        return md_header_splits
    
    except Exception as e:
        logger.error(f"Error splitting markdown: {str(e)}")
        raise

def split_text(documents: List[Document], 
               chunk_size: int = 1000, 
               chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into chunks.
    
    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects with chunked content
    """
    logger.info(f"Splitting {len(documents)} documents into chunks of size {chunk_size} with overlap {chunk_overlap}")
    
    try:
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split documents
        chunks = text_splitter.split_documents(documents)
        
        logger.info(f"Successfully split documents into {len(chunks)} chunks")
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        raise
