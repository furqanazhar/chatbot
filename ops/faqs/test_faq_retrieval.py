"""
Test script to test relevant FAQ retrieval based on user questions.

This script:
1. Loads the ChromaDB collection with FAQ embeddings
2. Allows testing similarity search with user questions
3. Displays relevant FAQs with similarity scores
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

# Configuration
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


def load_chromadb_collection(persist_directory: str, collection_name: str) -> Chroma:
    """Load existing ChromaDB collection.
    
    Args:
        persist_directory (str): Directory where ChromaDB data is persisted
        collection_name (str): Name of the ChromaDB collection
        
    Returns:
        Chroma: ChromaDB vector store instance
        
    Raises:
        FileNotFoundError: If the ChromaDB directory doesn't exist
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"ChromaDB directory not found: {persist_directory}\n"
            f"Please run create_faq_embeddings.py first to create the embeddings."
        )
    
    try:
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        
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


def search_faqs(vectordb: Chroma, query: str, k: int = 3) -> List[Dict]:
    """Search for relevant FAQs based on user query.
    
    Args:
        vectordb (Chroma): ChromaDB vector store instance
        query (str): User's question/query
        k (int): Number of similar results to return (default: 3)
        
    Returns:
        List[Dict]: List of FAQ results with questions, answers, and similarity scores
    """
    try:
        # Perform similarity search with scores
        results = vectordb.similarity_search_with_score(query, k=k)
        
        faq_results = []
        for doc, score in results:
            # Extract metadata
            metadata = doc.metadata
            
            # Calculate similarity percentage (lower score = more similar in ChromaDB)
            # Convert distance to similarity percentage (approximate)
            similarity_percentage = max(0, (1 - abs(score)) * 100) if score is not None else None
            
            faq_result = {
                "question": metadata.get("question", ""),
                "answer": metadata.get("answer", ""),
                "similarity_score": float(score) if score is not None else None,
                "similarity_percentage": round(similarity_percentage, 2) if similarity_percentage else None,
                "index": metadata.get("index", -1)
            }
            
            faq_results.append(faq_result)
        
        return faq_results
        
    except Exception as e:
        logger.error(f"Error searching FAQs: {e}")
        return []


def display_results(query: str, results: List[Dict]) -> None:
    """Display search results in a formatted way.
    
    Args:
        query (str): The original query
        results (List[Dict]): List of FAQ results
    """
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    print(f"Found {len(results)} relevant FAQ(s):\n")
    
    for idx, result in enumerate(results, 1):
        print(f"Result {idx}:")
        print(f"  Question: {result['question']}")
        print(f"  Answer: {result['answer']}")
        if result.get('similarity_percentage'):
            print(f"  Similarity: {result['similarity_percentage']}%")
        if result.get('similarity_score') is not None:
            print(f"  Score: {result['similarity_score']:.4f}")
        print(f"  Index: {result['index']}")
        print("-" * 80)


def interactive_test(vectordb: Chroma) -> None:
    """Interactive testing mode.
    
    Args:
        vectordb (Chroma): ChromaDB vector store instance
    """
    print("\n" + "=" * 80)
    print("FAQ Retrieval Test - Interactive Mode")
    print("=" * 80)
    print("Enter your questions to test FAQ retrieval.")
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
            
            # Search for relevant FAQs
            results = search_faqs(vectordb, query, k=3)
            
            if results:
                display_results(query, results)
            else:
                print(f"\nNo relevant FAQs found for: {query}\n")
            
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
        "How do I handle temperature-sensitive medications?",
        "What is the procedure for receiving medical supplies?",
        "How do I operate a forklift safely?",
        "What should I do if I find damaged products?",
        "How do I request time off?",
        "What is the procedure for shipping controlled substances?",
        "How do I handle expired products?",
        "What PPE is required in the warehouse?",
        "How do I process a return from a customer?",
        "What is the FIFO inventory system?",
    ]
    
    print("\n" + "=" * 80)
    print("FAQ Retrieval Test - Predefined Queries")
    print("=" * 80)
    
    for query in test_queries:
        results = search_faqs(vectordb, query, k=3)
        display_results(query, results)
        print("\n")


def main() -> None:
    """Main function to run FAQ retrieval tests.
    
    Supports both interactive and predefined query testing modes.
    """
    logger.info("***** Starting FAQ Retrieval Test *****")
    
    # Validate environment variables
    if not validate_environment():
        logger.error("Environment validation failed. Exiting...")
        return
    
    try:
        # Load ChromaDB collection
        logger.info("Loading ChromaDB collection...")
        vectordb = load_chromadb_collection(CHROMADB_DIR, COLLECTION_NAME)
        
        # Ask user for test mode
        print("\n" + "=" * 80)
        print("FAQ Retrieval Test")
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
        logger.error("Please run create_faq_embeddings.py first to create the embeddings.")
    except Exception as e:
        logger.error(f"Error in FAQ retrieval test: {e}")
        raise


if __name__ == "__main__":
    main()

