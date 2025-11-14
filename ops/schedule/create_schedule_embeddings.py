"""
Script to create embeddings from schedule JSON content and store in ChromaDB.

This script:
1. Loads schedule data from schedule.json
2. Creates embeddings using OpenAI
3. Stores embeddings in ChromaDB with collection name "schedules"
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


def format_stop_content(driver: Dict, stop: Dict, stop_idx: int) -> str:
    """Format a stop's information into searchable text content.
    
    Args:
        driver (Dict): Driver information
        stop (Dict): Stop/delivery information
        stop_idx (int): Index of the stop in the route
        
    Returns:
        str: Formatted text content for the stop
    """
    # Build comprehensive text content for semantic search
    content_parts = [
        f"Driver: {driver.get('driver_name', 'Unknown')} (ID: {driver.get('driver_id', 'N/A')})",
        f"Vehicle: {driver.get('vehicle_type', 'Unknown')} (ID: {driver.get('vehicle_id', 'N/A')})",
        f"Stop {stop.get('stop_number', stop_idx + 1)}: {stop.get('stop_type', 'delivery').title()}",
    ]
    
    # Source information
    source = stop.get('source', {})
    if source:
        content_parts.append(f"Pickup from: {source.get('name', 'Unknown')} at {source.get('address', 'N/A')}")
        content_parts.append(f"Contact: {source.get('contact', 'N/A')} - {source.get('phone', 'N/A')}")
    
    # Destination information
    destination = stop.get('destination', {})
    if destination:
        content_parts.append(f"Deliver to: {destination.get('name', 'Unknown')} at {destination.get('address', 'N/A')}")
        content_parts.append(f"Contact: {destination.get('contact', 'N/A')} - {destination.get('phone', 'N/A')}")
    
    # Timing information
    content_parts.append(f"Pickup time: {stop.get('scheduled_pickup_time', 'N/A')}")
    content_parts.append(f"Delivery time: {stop.get('scheduled_delivery_time', 'N/A')}")
    content_parts.append(f"Deadline: {stop.get('deadline', 'N/A')}")
    
    # Package information
    packages = stop.get('packages', [])
    if packages:
        content_parts.append(f"Packages ({len(packages)} package(s)):")
        for pkg in packages:
            pkg_parts = [
                f"  - {pkg.get('description', 'Unknown package')}",
                f"    Quantity: {pkg.get('quantity', 'N/A')}",
                f"    Type: {pkg.get('type', 'N/A')}",
                f"    Contents: {pkg.get('contents', 'N/A')}",
            ]
            if pkg.get('temperature_requirement'):
                pkg_parts.append(f"    Temperature: {pkg.get('temperature_requirement')}")
            if pkg.get('priority'):
                pkg_parts.append(f"    Priority: {pkg.get('priority')}")
            if pkg.get('weight'):
                pkg_parts.append(f"    Weight: {pkg.get('weight')}")
            content_parts.extend(pkg_parts)
    
    # Special instructions
    instructions = stop.get('special_instructions', [])
    if instructions:
        content_parts.append("Special instructions:")
        for instruction in instructions:
            content_parts.append(f"  - {instruction}")
    
    # Route information
    if stop.get('estimated_drive_time'):
        content_parts.append(f"Estimated drive time: {stop.get('estimated_drive_time')}")
    if stop.get('distance'):
        content_parts.append(f"Distance: {stop.get('distance')}")
    
    return "\n".join(content_parts)


def create_documents_from_schedule(schedule_data: Dict) -> List[Document]:
    """Convert schedule data to LangChain Document objects.
    
    Args:
        schedule_data (Dict): Schedule data dictionary
        
    Returns:
        List[Document]: List of Document objects with metadata
    """
    documents = []
    date = schedule_data.get('date', 'Unknown')
    
    drivers = schedule_data.get('drivers', [])
    
    for driver_idx, driver in enumerate(drivers):
        driver_id = driver.get('driver_id', f'DRV{driver_idx + 1:03d}')
        driver_name = driver.get('driver_name', 'Unknown Driver')
        route = driver.get('route', [])
        
        # Create a document for the driver's overall route summary
        route_summary_parts = [
            f"Driver: {driver_name} (ID: {driver_id})",
            f"Vehicle: {driver.get('vehicle_type', 'Unknown')} (ID: {driver.get('vehicle_id', 'N/A')})",
            f"Phone: {driver.get('phone', 'N/A')}",
            f"Status: {driver.get('status', 'active')}",
            f"Date: {date}",
            f"Total stops: {driver.get('total_stops', len(route))}",
            f"Total distance: {driver.get('total_distance', 'N/A')}",
            f"Estimated total time: {driver.get('estimated_total_time', 'N/A')}",
            f"Start time: {driver.get('start_time', 'N/A')}",
            f"End time: {driver.get('end_time', 'N/A')}",
        ]
        
        route_summary = "\n".join(route_summary_parts)
        
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
            stop_content = format_stop_content(driver, stop, stop_idx)
            
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
                    "destination_name": stop.get('destination', {}).get('name', 'Unknown'),
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
    2. Loading schedule JSON data
    3. Converting schedule data to Document objects
    4. Creating vector embeddings with ChromaDB
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
        # Load schedule data
        logger.info("Loading schedule data from JSON file...")
        schedule_data = load_schedule_json(SCHEDULE_JSON_PATH)
        
        if not schedule_data.get('drivers'):
            logger.error("No drivers found in the schedule JSON file. Exiting...")
            return
        
        # Convert schedule data to Document objects
        logger.info("Converting schedule data to Document objects...")
        documents = create_documents_from_schedule(schedule_data)
        
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

