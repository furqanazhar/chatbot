"""
Script to create embeddings from schedule JSON content and store in ChromaDB.

This script:
1. Deletes existing schedule collection from ChromaDB
2. Loads schedule data from schedule.json
3. Uses LLM to convert JSON into readable, natural language text
4. Creates embeddings using OpenAI
5. Stores embeddings in ChromaDB with collection name "schedules"
"""

import os
import json
import logging
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up from ops/schedule to logistics-agent

SCHEDULE_JSON_PATH = os.path.join(SCRIPT_DIR, "schedule.json")
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


def load_schedule_json(file_path: str) -> Dict:
    """Load schedule data from JSON file.
    
    Args:
        file_path (str): Path to the schedule JSON file
        
    Returns:
        Dict: Schedule data dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded schedule data from {file_path}")
        logger.info(f"Date: {data.get('date', 'N/A')}")
        logger.info(f"Number of drivers: {len(data.get('drivers', []))}")
        return data
        
    except FileNotFoundError:
        logger.error(f"Schedule file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise


def delete_chromadb_collection(persist_directory: str, collection_name: str) -> bool:
    """Delete an existing ChromaDB collection.
    
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
        
        # Check if collection exists and delete it
        try:
            # Try to get the collection to check if it exists
            client.get_collection(name=collection_name)
            # If it exists, delete it
            client.delete_collection(name=collection_name)
            logger.info(f"✅ Successfully deleted ChromaDB collection: {collection_name}")
            return True
        except Exception as e:
            # Collection might not exist, which is fine
            logger.info(f"Collection '{collection_name}' doesn't exist or already deleted")
            return True
            
    except Exception as e:
        logger.error(f"Error deleting ChromaDB collection: {e}")
        return False


def convert_json_to_readable_text(llm: ChatOpenAI, json_data: Dict, context: str = "route") -> str:
    """Use LLM to convert JSON data into readable, natural language text.
    
    Args:
        llm (ChatOpenAI): The LLM instance to use for conversion
        json_data (Dict): The JSON data to convert
        context (str): Context about what type of data this is (e.g., "route", "stop")
        
    Returns:
        str: Readable, natural language text summary
    """
    try:
        # Create a prompt for the LLM to convert JSON to readable text
        prompt = f"""Convert the following JSON data about a logistics {context} into a clear, readable, and comprehensive natural language summary. 
        
The summary should:
- Be written in natural, conversational language
- Include all important details (driver info, locations, times, packages, instructions, etc.)
- Be well-structured and easy to read
- Use complete sentences and proper formatting
- Be suitable for semantic search and retrieval

JSON Data:
{json.dumps(json_data, indent=2)}

Provide only the natural language summary, no additional commentary or explanations."""

        response = llm.invoke(prompt)
        readable_text = response.content.strip()
        
        logger.debug(f"Converted {context} JSON to readable text ({len(readable_text)} characters)")
        return readable_text
        
    except Exception as e:
        logger.error(f"Error converting JSON to readable text: {e}")
        # Fallback to basic formatting if LLM fails
        return json.dumps(json_data, indent=2)


def create_documents_from_schedule(schedule_data: Dict, llm: ChatOpenAI) -> List[Document]:
    """Convert schedule data to LangChain Document objects using LLM to create readable text.
    
    Args:
        schedule_data (Dict): Schedule data dictionary
        llm (ChatOpenAI): LLM instance for converting JSON to readable text
        
    Returns:
        List[Document]: List of Document objects with metadata
    """
    documents = []
    date = schedule_data.get('date', 'Unknown')
    
    drivers = schedule_data.get('drivers', [])
    
    logger.info(f"Processing {len(drivers)} driver(s) with LLM conversion...")
    
    for driver_idx, driver in enumerate(drivers):
        driver_id = driver.get('driver_id', f'DRV{driver_idx + 1:03d}')
        driver_name = driver.get('driver_name', 'Unknown Driver')
        route = driver.get('route', [])
        
        logger.info(f"Processing driver {driver_idx + 1}/{len(drivers)}: {driver_name}")
        
        # Prepare route summary data for LLM conversion
        route_summary_data = {
            "driver": {
                "name": driver_name,
                "id": driver_id,
                "vehicle_type": driver.get('vehicle_type', 'Unknown'),
                "vehicle_id": driver.get('vehicle_id', 'N/A'),
                "phone": driver.get('phone', 'N/A'),
                "status": driver.get('status', 'active')
            },
            "route": {
                "date": date,
                "total_stops": driver.get('total_stops', len(route)),
                "total_distance": driver.get('total_distance', 'N/A'),
                "estimated_total_time": driver.get('estimated_total_time', 'N/A'),
                "start_time": driver.get('start_time', 'N/A'),
                "end_time": driver.get('end_time', 'N/A')
            }
        }
        
        # Use LLM to convert route summary to readable text
        route_summary = convert_json_to_readable_text(llm, route_summary_data, context="route summary")
        
        doc = Document(
            page_content=route_summary,
            metadata={
                "driver_id": driver_id,
                "driver_name": driver_name,
                "date": date,
                "type": "route_summary",
                "total_stops": driver.get('total_stops', len(route)),
                "source": "schedule.json"
            }
        )
        documents.append(doc)
        
        # Create a document for each stop/delivery
        for stop_idx, stop in enumerate(route):
            logger.info(f"  Processing stop {stop_idx + 1}/{len(route)}: Stop {stop.get('stop_number', stop_idx + 1)}")
            
            # Prepare stop data for LLM conversion
            stop_data = {
                "driver": {
                    "name": driver_name,
                    "id": driver_id,
                    "vehicle_type": driver.get('vehicle_type', 'Unknown'),
                    "vehicle_id": driver.get('vehicle_id', 'N/A')
                },
                "stop": {
                    "stop_number": stop.get('stop_number', stop_idx + 1),
                    "stop_type": stop.get('stop_type', 'delivery'),
                    "source": stop.get('source', {}),
                    "destination": stop.get('destination', {}),
                    "scheduled_pickup_time": stop.get('scheduled_pickup_time', 'N/A'),
                    "scheduled_delivery_time": stop.get('scheduled_delivery_time', 'N/A'),
                    "deadline": stop.get('deadline', 'N/A'),
                    "packages": stop.get('packages', []),
                    "special_instructions": stop.get('special_instructions', []),
                    "estimated_drive_time": stop.get('estimated_drive_time', 'N/A'),
                    "distance": stop.get('distance', 'N/A'),
                    "status": stop.get('status', 'scheduled')
                }
            }
            
            # Use LLM to convert stop data to readable text
            stop_content = convert_json_to_readable_text(llm, stop_data, context="delivery stop")
            
            doc = Document(
                page_content=stop_content,
                metadata={
                    "driver_id": driver_id,
                    "driver_name": driver_name,
                    "date": date,
                    "type": "stop",
                    "stop_number": stop.get('stop_number', stop_idx + 1),
                    "stop_type": stop.get('stop_type', 'delivery'),
                    "source_name": stop.get('source', {}).get('name', 'Unknown'),
                    "source_address": stop.get('source', {}).get('address', ''),
                    "destination_name": stop.get('destination', {}).get('name', 'Unknown'),
                    "destination_address": stop.get('destination', {}).get('address', ''),
                    "delivery_time": stop.get('scheduled_delivery_time', 'N/A'),
                    "deadline": stop.get('deadline', 'N/A'),
                    "priority": stop.get('packages', [{}])[0].get('priority', 'medium') if stop.get('packages') else 'medium',
                    "source": "schedule.json"
                }
            )
            documents.append(doc)
    
    logger.info(f"Created {len(documents)} documents from schedule data")
    logger.info(f"  - {len(drivers)} route summaries")
    logger.info(f"  - {sum(len(d.get('route', [])) for d in drivers)} stop/delivery documents")
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
    """Main function to orchestrate the schedule embedding pipeline.
    
    The pipeline includes:
    1. Environment validation
    2. Deleting existing schedule collection
    3. Loading schedule JSON data
    4. Using LLM to convert JSON to readable text
    5. Converting schedule data to Document objects
    6. Creating vector embeddings with ChromaDB
    """
    logger.info("***** Starting Schedule Embedding Pipeline *****")
    
    # Validate environment variables
    if not validate_environment():
        logger.error("Environment validation failed. Exiting...")
        return
    
    # Check if schedule file exists
    if not os.path.exists(SCHEDULE_JSON_PATH):
        logger.error(f"Schedule file not found: {SCHEDULE_JSON_PATH}")
        logger.error("Please ensure the schedule JSON file exists in the ops/schedule directory.")
        return
    
    try:
        # Delete existing schedule collection
        logger.info("Deleting existing schedule collection (if it exists)...")
        delete_chromadb_collection(CHROMADB_DIR, COLLECTION_NAME)
        
        # Initialize LLM for JSON to text conversion
        logger.info("Initializing LLM for JSON to text conversion...")
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Using mini for cost efficiency
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Load schedule data
        logger.info("Loading schedule data from JSON file...")
        schedule_data = load_schedule_json(SCHEDULE_JSON_PATH)
        
        if not schedule_data.get('drivers'):
            logger.error("No drivers found in the schedule JSON file. Exiting...")
            return
        
        # Convert schedule data to Document objects using LLM
        logger.info("Converting schedule data to readable text using LLM...")
        logger.info("This may take a few moments as each route and stop is processed...")
        documents = create_documents_from_schedule(schedule_data, llm)
        
        if not documents:
            logger.error("No documents created from schedule data. Exiting...")
            return
        
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
        logger.info(f"Schedule embeddings are now available in ChromaDB collection: {COLLECTION_NAME}")
        logger.info(f"ChromaDB data persisted in: {CHROMADB_DIR}")
        
    except Exception as e:
        logger.error(f"Error in schedule embedding pipeline: {e}")
        raise


if __name__ == "__main__":
    main()

