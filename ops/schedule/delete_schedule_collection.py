"""
Script to forcefully delete the schedule collection from ChromaDB.

This script ensures the old collection is completely removed before
recreating it with updated data.
"""

import os
import logging
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

CHROMADB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "schedules"


def delete_collection_forcefully(persist_directory: str, collection_name: str) -> bool:
    """Forcefully delete a ChromaDB collection.
    
    Args:
        persist_directory (str): Directory where ChromaDB data is persisted
        collection_name (str): Name of the ChromaDB collection to delete
        
    Returns:
        bool: True if collection was deleted or didn't exist, False on error
    """
    try:
        if not os.path.exists(persist_directory):
            logger.info(f"ChromaDB directory doesn't exist: {persist_directory}. Nothing to delete.")
            return True
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)
        
        # List all collections to see what exists
        try:
            collections = client.list_collections()
            logger.info(f"Found {len(collections)} collection(s) in ChromaDB:")
            for col in collections:
                logger.info(f"  - {col.name}")
        except Exception as e:
            logger.warning(f"Could not list collections: {e}")
        
        # Try to delete the collection
        try:
            # Try to get the collection first to check if it exists
            collection = client.get_collection(name=collection_name)
            count = collection.count()
            logger.info(f"Found collection '{collection_name}' with {count} documents")
            
            # Delete the collection
            client.delete_collection(name=collection_name)
            logger.info(f"✅ Successfully deleted ChromaDB collection: {collection_name}")
            return True
        except Exception as e:
            # Collection might not exist
            error_msg = str(e).lower()
            if "does not exist" in error_msg or "not found" in error_msg:
                logger.info(f"Collection '{collection_name}' doesn't exist - nothing to delete")
                return True
            else:
                logger.error(f"Error accessing/deleting collection: {e}")
                return False
            
    except Exception as e:
        logger.error(f"Error deleting ChromaDB collection: {e}")
        return False


def main() -> None:
    """Main function to delete the schedule collection."""
    logger.info("***** Deleting Schedule Collection *****")
    logger.info(f"ChromaDB Directory: {CHROMADB_DIR}")
    logger.info(f"Collection Name: {COLLECTION_NAME}")
    
    success = delete_collection_forcefully(CHROMADB_DIR, COLLECTION_NAME)
    
    if success:
        logger.info("***** Collection Deletion Completed Successfully! *****")
        logger.info("You can now run create_schedule_embeddings.py to recreate the collection with updated data.")
    else:
        logger.error("***** Collection Deletion Failed! *****")
        logger.error("Please check the error messages above.")


if __name__ == "__main__":
    main()

