"""
Test script to test schedule retrieval based on user questions.

This script:
1. Loads the ChromaDB collection with schedule embeddings
2. Allows testing similarity search with user questions
3. Displays relevant schedule information with similarity scores
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up from ops/schedule to logistics-agent

CHROMADB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "schedules"


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
            f"Please run create_schedule_embeddings.py first to create the embeddings."
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


def search_schedules(vectordb: Chroma, query: str, k: int = 3) -> List[Dict]:
    """Search for relevant schedule information based on user query.
    
    Args:
        vectordb (Chroma): ChromaDB vector store instance
        query (str): User's question/query
        k (int): Number of similar results to return (default: 3)
        
    Returns:
        List[Dict]: List of schedule results with content and similarity scores
    """
    try:
        # Perform similarity search with scores
        results = vectordb.similarity_search_with_score(query, k=k)
        
        schedule_results = []
        for doc, score in results:
            # Extract metadata
            metadata = doc.metadata
            
            # Calculate similarity percentage (lower score = more similar in ChromaDB)
            # Convert distance to similarity percentage (approximate)
            similarity_percentage = max(0, (1 - abs(score)) * 100) if score is not None else None
            
            schedule_result = {
                "content": doc.page_content,
                "similarity_score": float(score) if score is not None else None,
                "similarity_percentage": round(similarity_percentage, 2) if similarity_percentage else None,
                "driver_id": metadata.get("driver_id", "N/A"),
                "driver_name": metadata.get("driver_name", "N/A"),
                "date": metadata.get("date", "N/A"),
                "type": metadata.get("type", "unknown"),
                "stop_number": metadata.get("stop_number", None),
                "stop_type": metadata.get("stop_type", None),
                "source_name": metadata.get("source_name", None),
                "destination_name": metadata.get("destination_name", None),
                "delivery_time": metadata.get("delivery_time", None),
                "deadline": metadata.get("deadline", None),
                "priority": metadata.get("priority", None),
                "source": metadata.get("source", "unknown")
            }
            
            schedule_results.append(schedule_result)
        
        return schedule_results
        
    except Exception as e:
        logger.error(f"Error searching schedules: {e}")
        return []


def display_results(query: str, results: List[Dict]) -> None:
    """Display search results in a formatted way.
    
    Args:
        query (str): The original query
        results (List[Dict]): List of schedule results
    """
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    print(f"Found {len(results)} relevant schedule entry(ies):\n")
    
    for idx, result in enumerate(results, 1):
        print(f"Result {idx}:")
        print(f"  Type: {result['type']}")
        print(f"  Driver: {result['driver_name']} (ID: {result['driver_id']})")
        print(f"  Date: {result['date']}")
        
        if result.get('stop_number'):
            print(f"  Stop Number: {result['stop_number']}")
            print(f"  Stop Type: {result['stop_type']}")
            if result.get('source_name'):
                print(f"  Source: {result['source_name']}")
            if result.get('destination_name'):
                print(f"  Destination: {result['destination_name']}")
            if result.get('delivery_time'):
                print(f"  Delivery Time: {result['delivery_time']}")
            if result.get('deadline'):
                print(f"  Deadline: {result['deadline']}")
            if result.get('priority'):
                print(f"  Priority: {result['priority']}")
        
        # Display content (truncated if too long)
        content = result['content']
        if len(content) > 500:
            print(f"  Content: {content[:500]}...")
        else:
            print(f"  Content: {content}")
        
        if result.get('similarity_percentage'):
            print(f"  Similarity: {result['similarity_percentage']}%")
        if result.get('similarity_score') is not None:
            print(f"  Score: {result['similarity_score']:.4f}")
        print("-" * 80)


def interactive_test(vectordb: Chroma) -> None:
    """Interactive testing mode.
    
    Args:
        vectordb (Chroma): ChromaDB vector store instance
    """
    print("\n" + "=" * 80)
    print("Schedule Retrieval Test - Interactive Mode")
    print("=" * 80)
    print("Enter your questions to test schedule retrieval.")
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
            
            # Search for relevant schedule information
            results = search_schedules(vectordb, query, k=3)
            
            if results:
                display_results(query, results)
            else:
                print(f"\nNo relevant schedule information found for: {query}\n")
            
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
        "What's my route for today?",
        # "What deliveries do I have today?",
        # "Where do I need to deliver packages?",
        # "What time is my first delivery?",
        # "What packages am I delivering to Medical City Plano?",
        # "What are my special instructions?",
        # "What temperature requirements do my packages have?",
        # "What is my delivery deadline?",
        # "What is my total distance for today?",
        # "What are the addresses for my deliveries?",
        # "What packages require temperature control?",
        # "What is my schedule for today?",
    ]
    
    print("\n" + "=" * 80)
    print("Schedule Retrieval Test - Predefined Queries")
    print("=" * 80)
    
    for query in test_queries:
        results = search_schedules(vectordb, query, k=3)
        display_results(query, results)
        print("\n")


def main() -> None:
    """Main function to run schedule retrieval tests.
    
    Supports both interactive and predefined query testing modes.
    """
    logger.info("***** Starting Schedule Retrieval Test *****")
    
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
        print("Schedule Retrieval Test")
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
        logger.error("Please run create_schedule_embeddings.py first to create the embeddings.")
    except Exception as e:
        logger.error(f"Error in schedule retrieval test: {e}")
        raise


if __name__ == "__main__":
    main()

