"""
Embeddings module for creating vector embeddings.
"""
import logging
import os
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get embeddings model.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        Embeddings model
    """
    logger.info(f"Loading embeddings model: {model_name}")
    
    try:
        # Create embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        logger.info(f"Successfully loaded embeddings model")
        
        return embeddings
    
    except Exception as e:
        logger.error(f"Error loading embeddings model: {str(e)}")
        raise
