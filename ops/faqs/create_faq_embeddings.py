"""
Script to create embeddings from FAQ JSON content and store in ChromaDB.

This script:
1. Loads FAQ data from medical_distributor_logistics_faq.json
2. Creates embeddings using OpenAI
3. Stores embeddings in ChromaDB with collection name "faqs"
"""

import os
import json
import logging
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
FAQ_JSON_PATH = "data/medical_distributor_logistics_faq.json"
CHROMADB_DIR = "chroma_db"
COLLECTION_NAME = "faqs"


def validate_environment() -> bool:
    """Validate that all required environment variables are set.
    
    Returns:
        bool: True if all required variables are set, False otherwise
    """
    required_vars = ["OPENAI_API_KEY"]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True


def load_faq_json(file_path: str) -> List[Dict[str, str]]:
    """Load FAQ data from JSON file.
    
    Args:
        file_path (str): Path to the FAQ JSON file
        
    Returns:
        List[Dict[str, str]]: List of FAQ dictionaries with 'question' and 'answer' keys
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} FAQs from {file_path}")
        return data
        
    except FileNotFoundError:
        logger.error(f"FAQ file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise


def create_documents_from_faqs(faqs: List[Dict[str, str]]) -> List[Document]:
    """Convert FAQ dictionaries to LangChain Document objects.
    
    Args:
        faqs (List[Dict[str, str]]): List of FAQ dictionaries
        
    Returns:
        List[Document]: List of Document objects with metadata
    """
    documents = []
    
    for idx, faq in enumerate(faqs):
        # Create a combined text from question and answer for better semantic search
        # The question is the primary content, answer provides context
        content = f"Question: {faq.get('question', '')}\nAnswer: {faq.get('answer', '')}"
        
        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "question": faq.get('question', ''),
                "answer": faq.get('answer', ''),
                "index": idx,
                "source": "medical_distributor_logistics_faq"
            }
        )
        documents.append(doc)
    
    logger.info(f"Created {len(documents)} documents from FAQs")
    return documents


def create_chromadb_embeddings(documents: List[Document], 
                               persist_directory: str, 
                               collection_name: str) -> Chroma:
    """Create ChromaDB vector store with embeddings.
    
    Args:
        documents (List[Document]): List of Document objects to embed
        persist_directory (str): Directory to persist ChromaDB data
        collection_name (str): Name of the ChromaDB collection
        
    Returns:
        Chroma: ChromaDB vector store instance
    """
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        
        logger.info(f"Creating ChromaDB collection '{collection_name}' in {persist_directory}")
        
        # Create or load ChromaDB collection
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        logger.info(f"Successfully stored {len(documents)} documents in ChromaDB collection: {collection_name}")
        return vectordb
        
    except Exception as e:
        logger.error(f"Error creating ChromaDB embeddings: {e}")
        raise


def main() -> None:
    """Main function to orchestrate the FAQ embedding pipeline.
    
    The pipeline includes:
    1. Environment validation
    2. Loading FAQ JSON data
    3. Converting FAQs to Document objects
    4. Creating vector embeddings with ChromaDB
    """
    logger.info("***** Starting FAQ Embedding Pipeline *****")
    
    # Validate environment variables
    if not validate_environment():
        logger.error("Environment validation failed. Exiting...")
        return
    
    # Check if FAQ file exists
    if not os.path.exists(FAQ_JSON_PATH):
        logger.error(f"FAQ file not found: {FAQ_JSON_PATH}")
        logger.error("Please ensure the FAQ JSON file exists in the data directory.")
        return
    
    try:
        # Load FAQ data
        logger.info("Loading FAQ data from JSON file...")
        faqs = load_faq_json(FAQ_JSON_PATH)
        
        if not faqs:
            logger.error("No FAQs found in the JSON file. Exiting...")
            return
        
        # Convert FAQs to Document objects
        logger.info("Converting FAQs to Document objects...")
        documents = create_documents_from_faqs(faqs)
        
        # Create ChromaDB directory if it doesn't exist
        os.makedirs(CHROMADB_DIR, exist_ok=True)
        
        # Create embeddings and store in ChromaDB
        logger.info("Creating embeddings and storing in ChromaDB...")
        vectordb = create_chromadb_embeddings(
            documents=documents,
            persist_directory=CHROMADB_DIR,
            collection_name=COLLECTION_NAME
        )
        
        logger.info("***** Pipeline Completed Successfully! *****")
        logger.info(f"FAQs are now available in ChromaDB collection: {COLLECTION_NAME}")
        logger.info(f"ChromaDB data persisted in: {CHROMADB_DIR}")
        
    except Exception as e:
        logger.error(f"Error in FAQ embedding pipeline: {e}")
        raise


if __name__ == "__main__":
    main()

