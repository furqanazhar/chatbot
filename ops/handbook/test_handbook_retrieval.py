"""
Test script to test handbook content retrieval based on user questions.

This script:
1. Loads the ChromaDB collection with handbook embeddings
2. Allows testing similarity search with user questions
3. Displays relevant handbook content with similarity scores
"""

import os
import logging
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - paths relative to project root
# Get the project root directory (two levels up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up from ops/handbook to logistics-agent

CHROMADB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "handbook"


def validate_environment() -> bool:
    """Validate that all required environment variables are set.
    
    Returns:
        bool: True if all required variables are set, False otherwise
    """
    # OpenAI API key is optional if using SentenceTransformer embeddings
    return True


def load_chromadb_collection(persist_directory: str, collection_name: str, use_openai: bool = True) -> Chroma:
    """Load existing ChromaDB collection.
    
    Args:
        persist_directory (str): Directory where ChromaDB data is persisted
        collection_name (str): Name of the ChromaDB collection
        use_openai (bool): Whether to use OpenAI embeddings
        
    Returns:
        Chroma: ChromaDB vector store instance
        
    Raises:
        FileNotFoundError: If the ChromaDB directory doesn't exist
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"ChromaDB directory not found: {persist_directory}\n"
            f"Please run create_handbook_embeddings.py first to create the embeddings."
        )
    
    try:
        if use_openai:
            embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            from chromadb.utils import embedding_functions
            embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        
        logger.info(f"Loading ChromaDB collection '{collection_name}' from {persist_directory}")
        
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
        logger.info("Successfully loaded ChromaDB collection")
        return vectordb
        
    except Exception as e:
        logger.error(f"Error loading ChromaDB collection: {e}")
        raise


def search_handbook(vectordb: Chroma, query: str, k: int = 3) -> List[Dict]:
    """Search for relevant handbook content based on user query.
    
    Args:
        vectordb (Chroma): ChromaDB vector store instance
        query (str): User's question/query
        k (int): Number of similar results to return (default: 3)
        
    Returns:
        List[Dict]: List of handbook results with content and similarity scores
    """
    try:
        # Perform similarity search with scores
        results = vectordb.similarity_search_with_score(query, k=k)
        
        handbook_results = []
        for doc, score in results:
            # Extract metadata
            metadata = doc.metadata
            
            # Calculate similarity percentage (lower score = more similar in ChromaDB)
            # Convert distance to similarity percentage (approximate)
            similarity_percentage = max(0, (1 - abs(score)) * 100) if score is not None else None
            
            handbook_result = {
                "content": doc.page_content,
                "similarity_score": float(score) if score is not None else None,
                "similarity_percentage": round(similarity_percentage, 2) if similarity_percentage else None,
                "source": metadata.get("source", "unknown"),
                "chunk_index": metadata.get("chunk_index", -1),
                "chunk_size": metadata.get("chunk_size", 0)
            }
            
            handbook_results.append(handbook_result)
        
        return handbook_results
        
    except Exception as e:
        logger.error(f"Error searching handbook: {e}")
        return []


def display_results(query: str, results: List[Dict]) -> None:
    """Display search results in a formatted way.
    
    Args:
        query (str): The original query
        results (List[Dict]): List of handbook results
    """
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    print(f"Found {len(results)} relevant section(s) from handbook:\n")
    
    for idx, result in enumerate(results, 1):
        print(f"Result {idx}:")
        print(f"  Content: {result['content'][:500]}..." if len(result['content']) > 500 else f"  Content: {result['content']}")
        if result.get('similarity_percentage'):
            print(f"  Similarity: {result['similarity_percentage']}%")
        if result.get('similarity_score') is not None:
            print(f"  Score: {result['similarity_score']:.4f}")
        print(f"  Source: {result['source']}")
        print(f"  Chunk Index: {result['chunk_index']}")
        print("-" * 80)


def interactive_test(vectordb: Chroma) -> None:
    """Interactive testing mode.
    
    Args:
        vectordb (Chroma): ChromaDB vector store instance
    """
    print("\n" + "=" * 80)
    print("Handbook Retrieval Test - Interactive Mode")
    print("=" * 80)
    print("Enter your questions to test handbook content retrieval.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            query = input("Enter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if not query:
                print("Please enter a valid question.\n")
                continue
            
            # Search for relevant handbook content
            results = search_handbook(vectordb, query, k=3)
            
            if results:
                display_results(query, results)
            else:
                print(f"\nNo relevant content found for: {query}\n")
            
            print()  # Empty line for readability
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error in interactive test: {e}")
            print(f"Error: {e}\n")


def test_predefined_queries(vectordb: Chroma) -> None:
    """Test with predefined queries.
    
    Args:
        vectordb (Chroma): ChromaDB vector store instance
    """
    test_queries = [
        "What are the safety procedures for driving?",
        "How should I handle logistics operations?",
        "What are the requirements for manpower management?",
        "What are the best practices for transportation?",
        "How do I manage inventory?",
        "What safety equipment is required?",
        "What are the procedures for vehicle maintenance?",
        "How do I handle emergency situations?",
        "What are the compliance requirements?",
        "What training is required for drivers?",
    ]
    
    print("\n" + "=" * 80)
    print("Handbook Retrieval Test - Predefined Queries")
    print("=" * 80)
    
    for query in test_queries:
        results = search_handbook(vectordb, query, k=3)
        display_results(query, results)
        print("\n")


def main() -> None:
    """Main function to run handbook retrieval tests.
    
    Supports both interactive and predefined query testing modes.
    """
    logger.info("***** Starting Handbook Retrieval Test *****")
    
    # Check embedding type (default to OpenAI, can be changed)
    USE_OPENAI = os.getenv("OPENAI_API_KEY") is not None
    
    if not USE_OPENAI:
        logger.info("OpenAI API key not found. Using SentenceTransformer embeddings.")
    
    try:
        # Load ChromaDB collection
        logger.info("Loading ChromaDB collection...")
        vectordb = load_chromadb_collection(CHROMADB_DIR, COLLECTION_NAME, use_openai=USE_OPENAI)
        
        # Ask user for test mode
        print("\n" + "=" * 80)
        print("Handbook Retrieval Test")
        print("=" * 80)
        print("Choose test mode:")
        print("1. Interactive mode (enter your own questions)")
        print("2. Predefined queries (test with sample questions)")
        print("3. Both")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            interactive_test(vectordb)
        elif choice == "2":
            test_predefined_queries(vectordb)
        elif choice == "3":
            test_predefined_queries(vectordb)
            print("\n" + "=" * 80)
            print("Now entering interactive mode...")
            interactive_test(vectordb)
        else:
            print("Invalid choice. Running predefined queries by default...")
            test_predefined_queries(vectordb)
        
        logger.info("***** Test Completed *****")
        
    except FileNotFoundError as e:
        logger.error(f"{e}")
        logger.error("Please run create_handbook_embeddings.py first to create the embeddings.")
    except Exception as e:
        logger.error(f"Error in handbook retrieval test: {e}")
        raise


if __name__ == "__main__":
    main()

