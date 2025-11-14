"""
Script to extract text from PDF handbook, create chunks, generate embeddings, and store in ChromaDB.

This script:
1. Extracts text from the PDF handbook
2. Splits text into chunks using RecursiveCharacterTextSplitter
3. Generates embeddings (using OpenAI or SentenceTransformer)
4. Stores embeddings in ChromaDB with collection name "handbook"
"""

import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
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

PDF_PATH = os.path.join(PROJECT_ROOT, "data", "Manpower_Driving_and_Logistics_Handbook_04_20.pdf")
CHROMADB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "handbook"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding options
USE_OPENAI_EMBEDDINGS = True  # Set to False to use free SentenceTransformer embeddings


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ImportError: If required PDF library is not installed
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        # Try using langchain's PDF loader first
        from langchain_community.document_loaders import PyPDFLoader
        
        logger.info(f"Loading PDF using LangChain PyPDFLoader: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Combine all pages into a single text
        text = "\n\n".join([doc.page_content for doc in documents])
        logger.info(f"Extracted text from {len(documents)} pages")
        
        return text
        
    except ImportError:
        # Fallback to pypdf if langchain loader is not available
        try:
            from pypdf import PdfReader
            
            logger.info(f"Loading PDF using pypdf: {pdf_path}")
            reader = PdfReader(pdf_path)
            text = ""
            
            for i, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                text += page_text + "\n\n"
                if i % 10 == 0:
                    logger.info(f"Processed {i} pages...")
            
            logger.info(f"Extracted text from {len(reader.pages)} pages")
            return text
            
        except ImportError:
            raise ImportError(
                "Neither langchain_community.document_loaders.PyPDFLoader nor pypdf is available. "
                "Please install one of them:\n"
                "  pip install langchain-community\n"
                "  OR\n"
                "  pip install pypdf"
            )


def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    logger.info(f"Splitting text into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    logger.info(f"Created {len(chunks)} chunks from the PDF text")
    
    return chunks


def create_documents_from_chunks(chunks: List[str], source: str = "handbook.pdf") -> List[Document]:
    """Convert text chunks to LangChain Document objects.
    
    Args:
        chunks (List[str]): List of text chunks
        source (str): Source identifier for metadata
        
    Returns:
        List[Document]: List of Document objects with metadata
    """
    documents = []
    
    for idx, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": source,
                "chunk_index": idx,
                "chunk_size": len(chunk)
            }
        )
        documents.append(doc)
    
    logger.info(f"Created {len(documents)} Document objects")
    return documents


def get_embedding_function(use_openai: bool = True):
    """Get embedding function based on preference.
    
    Args:
        use_openai (bool): If True, use OpenAI embeddings. If False, use SentenceTransformer.
        
    Returns:
        Embedding function for ChromaDB
    """
    if use_openai:
        from langchain_openai import OpenAIEmbeddings
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Set USE_OPENAI_EMBEDDINGS=False to use free embeddings.")
        
        logger.info("Using OpenAI embeddings")
        return OpenAIEmbeddings(api_key=api_key)
    else:
        try:
            from chromadb.utils import embedding_functions
            
            logger.info("Using SentenceTransformer embeddings (all-MiniLM-L6-v2)")
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        except ImportError:
            raise ImportError(
                "SentenceTransformer embeddings require chromadb with sentence-transformers. "
                "Install with: pip install chromadb[sentence-transformers]"
            )


def create_chromadb_embeddings(documents: List[Document], 
                               persist_directory: str, 
                               collection_name: str,
                               use_openai: bool = True) -> Chroma:
    """Create ChromaDB vector store with embeddings.
    
    Args:
        documents (List[Document]): List of Document objects to embed
        persist_directory (str): Directory to persist ChromaDB data
        collection_name (str): Name of the ChromaDB collection
        use_openai (bool): Whether to use OpenAI embeddings
        
    Returns:
        Chroma: ChromaDB vector store instance
    """
    try:
        # Get embedding function
        if use_openai:
            embeddings = get_embedding_function(use_openai=True)
        else:
            # For SentenceTransformer, we need to use ChromaDB directly
            import chromadb
            from chromadb.utils import embedding_functions
            
            logger.info(f"Creating ChromaDB collection '{collection_name}' using SentenceTransformer")
            
            # Initialize Chroma client
            client = chromadb.PersistentClient(path=persist_directory)
            
            # Get embedding function
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Create or get collection
            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_func,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Add documents
            texts = [doc.page_content for doc in documents]
            ids = [f"handbook_chunk_{i}" for i in range(len(documents))]
            metadatas = [doc.metadata for doc in documents]
            
            collection.add(
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully stored {len(documents)} documents in ChromaDB collection: {collection_name}")
            
            # Return a Chroma instance for compatibility
            embeddings = get_embedding_function(use_openai=False)
            vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            return vectordb
        
        # Use OpenAI embeddings with LangChain Chroma
        logger.info(f"Creating ChromaDB collection '{collection_name}' using OpenAI embeddings")
        
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
    """Main function to orchestrate the handbook embedding pipeline.
    
    The pipeline includes:
    1. Extracting text from PDF
    2. Splitting text into chunks
    3. Converting chunks to Document objects
    4. Creating vector embeddings with ChromaDB
    """
    logger.info("***** Starting Handbook Embedding Pipeline *****")
    
    # Check if PDF file exists
    if not os.path.exists(PDF_PATH):
        logger.error(f"PDF file not found: {PDF_PATH}")
        logger.error("Please ensure the PDF file exists in the data directory.")
        return
    
    try:
        # Step 1: Extract text from PDF
        logger.info("Step 1: Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(PDF_PATH)
        
        if not pdf_text or len(pdf_text.strip()) == 0:
            logger.error("No text extracted from PDF. The PDF might be image-based or corrupted.")
            return
        
        logger.info(f"Extracted {len(pdf_text)} characters from PDF")
        
        # Step 2: Split text into chunks
        logger.info("Step 2: Splitting text into chunks...")
        chunks = split_text_into_chunks(pdf_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        if not chunks:
            logger.error("No chunks created from PDF text.")
            return
        
        # Step 3: Convert chunks to Document objects
        logger.info("Step 3: Converting chunks to Document objects...")
        documents = create_documents_from_chunks(chunks, source=os.path.basename(PDF_PATH))
        
        # Step 4: Create ChromaDB directory if it doesn't exist
        os.makedirs(CHROMADB_DIR, exist_ok=True)
        
        # Step 5: Create embeddings and store in ChromaDB
        logger.info("Step 4: Creating embeddings and storing in ChromaDB...")
        vectordb = create_chromadb_embeddings(
            documents=documents,
            persist_directory=CHROMADB_DIR,
            collection_name=COLLECTION_NAME,
            use_openai=USE_OPENAI_EMBEDDINGS
        )
        
        logger.info("***** Pipeline Completed Successfully! *****")
        logger.info(f"Handbook embeddings are now available in ChromaDB collection: {COLLECTION_NAME}")
        logger.info(f"ChromaDB data persisted in: {CHROMADB_DIR}")
        logger.info(f"Total chunks stored: {len(documents)}")
        
    except Exception as e:
        logger.error(f"Error in handbook embedding pipeline: {e}")
        raise


if __name__ == "__main__":
    main()

