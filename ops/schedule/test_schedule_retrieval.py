"""
Test script to test schedule retrieval based on user questions.

This script:
1. Loads the ChromaDB collection with schedule embeddings
2. Allows testing similarity search with user questions
3. Uses LLM to synthesize search results into natural language answers
4. Displays relevant schedule information with similarity scores
"""

import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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


def generate_llm_answer(llm: ChatOpenAI, query: str, results: List[Dict]) -> Optional[str]:
    """Use LLM to synthesize search results into a natural language answer.
    
    Args:
        llm (ChatOpenAI): The LLM instance to use for answer generation
        query (str): The user's original question
        results (List[Dict]): List of schedule search results
        
    Returns:
        Optional[str]: Natural language answer, or None if generation fails
    """
    if not results:
        return None
    
    try:
        # Format the retrieved schedule information
        retrieved_info = []
        for idx, result in enumerate(results, 1):
            info_parts = [f"Result {idx}:"]
            info_parts.append(f"Type: {result['type']}")
            info_parts.append(f"Driver: {result['driver_name']} (ID: {result['driver_id']})")
            info_parts.append(f"Date: {result['date']}")
            
            if result.get('stop_number'):
                info_parts.append(f"Stop Number: {result['stop_number']}")
                info_parts.append(f"Stop Type: {result['stop_type']}")
            if result.get('source_name'):
                info_parts.append(f"Source: {result['source_name']}")
            if result.get('destination_name'):
                info_parts.append(f"Destination: {result['destination_name']}")
            if result.get('delivery_time'):
                info_parts.append(f"Delivery Time: {result['delivery_time']}")
            if result.get('deadline'):
                info_parts.append(f"Deadline: {result['deadline']}")
            if result.get('priority'):
                info_parts.append(f"Priority: {result['priority']}")
            
            info_parts.append(f"Content: {result['content']}")
            retrieved_info.append("\n".join(info_parts))
        
        retrieved_text = "\n\n".join(retrieved_info)
        
        # Create prompt for LLM
        prompt = f"""You are a helpful logistics assistant. Based on the following retrieved schedule information, answer the user's question in a clear, natural, and comprehensive way.

User Question: {query}

Retrieved Schedule Information:
{retrieved_text}

Instructions:
- Provide a clear, direct answer to the user's question
- Use the retrieved information to answer accurately
- If the information doesn't fully answer the question, say so
- Be concise but comprehensive
- Use natural, conversational language
- Include relevant details like driver names, locations, times, packages, etc. when relevant
- If multiple results are provided, synthesize them into a coherent answer

Answer:"""

        response = llm.invoke(prompt)
        answer = response.content.strip()
        
        return answer
        
    except Exception as e:
        logger.error(f"Error generating LLM answer: {e}")
        return None


def display_results(query: str, results: List[Dict], llm: Optional[ChatOpenAI] = None) -> None:
    """Display search results in a formatted way, optionally with LLM-generated answer.
    
    Args:
        query (str): The original query
        results (List[Dict]): List of schedule results
        llm (Optional[ChatOpenAI]): Optional LLM instance for generating answers
    """
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    
    # Generate LLM answer if LLM is provided
    if llm and results:
        print("\n🤖 AI Answer:")
        print("-" * 80)
        answer = generate_llm_answer(llm, query, results)
        if answer:
            print(answer)
        else:
            print("Could not generate AI answer. Showing raw results below.")
        print("-" * 80)
    
    print(f"\n📋 Found {len(results)} relevant schedule entry(ies):\n")
    
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


def interactive_test(vectordb: Chroma, llm: Optional[ChatOpenAI] = None) -> None:
    """Interactive testing mode.
    
    Args:
        vectordb (Chroma): ChromaDB vector store instance
        llm (Optional[ChatOpenAI]): Optional LLM instance for generating answers
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
            print("🔍 Searching schedule database...")
            results = search_schedules(vectordb, query, k=3)
            
            if results:
                if llm:
                    print("🤖 Generating answer...")
                display_results(query, results, llm)
            else:
                print(f"\n❌ No relevant schedule information found for: {query}\n")
            
            print()  # Empty line for readability
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error in interactive test: {e}")
            print(f"Error: {e}\n")


def test_predefined_queries(vectordb: Chroma, llm: Optional[ChatOpenAI] = None) -> None:
    """Test with predefined queries.
    
    Args:
        vectordb (Chroma): ChromaDB vector store instance
        llm (Optional[ChatOpenAI]): Optional LLM instance for generating answers
    """
    test_queries = [
        # "What's my route for today?",
        # "What deliveries do I have today?",
        # "Where do I need to deliver packages?",
        #"What time is my first delivery?",
        # "What packages am I delivering to Medical City Plano?",
        # "What are my special instructions?",
        # "What temperature requirements do my packages have?",
        # "What is my delivery deadline?",
        "What is my total distance for today?",
        # "What are the addresses for my deliveries?",
        # "What packages require temperature control?",
        # "What is my schedule for today?",
    ]
    
    print("\n" + "=" * 80)
    print("Schedule Retrieval Test - Predefined Queries")
    print("=" * 80)
    
    for idx, query in enumerate(test_queries, 1):
        print(f"\n[{idx}/{len(test_queries)}] Processing query...")
        results = search_schedules(vectordb, query, k=3)
        display_results(query, results, llm)
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
        
        # Initialize LLM for answer generation
        logger.info("Initializing LLM for answer generation...")
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Using mini for cost efficiency
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        logger.info("✅ LLM initialized successfully")
        
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
            interactive_test(vectordb, llm)
        elif choice == "2":
            test_predefined_queries(vectordb, llm)
        elif choice == "3":
            test_predefined_queries(vectordb, llm)
            print("\n" + "=" * 80)
            print("Now entering interactive mode...")
            interactive_test(vectordb, llm)
        else:
            print("Invalid choice. Running predefined queries by default...")
            test_predefined_queries(vectordb, llm)
        
        logger.info("***** Test Completed *****")
        
    except FileNotFoundError as e:
        logger.error(f"{e}")
        logger.error("Please run create_schedule_embeddings.py first to create the embeddings.")
    except Exception as e:
        logger.error(f"Error in schedule retrieval test: {e}")
        raise


if __name__ == "__main__":
    main()

