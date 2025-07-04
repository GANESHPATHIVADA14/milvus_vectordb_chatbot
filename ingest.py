# ingest.py
import os
import logging
import sys
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from pymilvus import connections, utility

# --- Logging Configuration ---
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT", "443")  # Default for Zilliz Cloud
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "milvus_chatbot_vectordb")

if not GOOGLE_API_KEY or not MILVUS_TOKEN or not MILVUS_HOST:
    raise ValueError("Missing required API keys or Milvus endpoint in the .env file.")

# --- Other Settings ---
PDF_PATH = "/home/ganesh-pathivada/Downloads/attention.pdf"
EMBEDDING_DIM = 768  # Gemini embedding size

def main():
    logging.info("--- Starting Ingestion Process ---")

    # 1. Configure LlamaIndex with Gemini
    logging.info("Setting up LlamaIndex with Gemini models...")
    Settings.llm = Gemini(
        model="models/gemini-1.5-flash",
        api_key=GOOGLE_API_KEY
    )
    Settings.embed_model = GeminiEmbedding(
        model_name="models/embedding-001",
        api_key=GOOGLE_API_KEY
    )
    Settings.chunk_size = 1000
    Settings.chunk_overlap = 200

    # 2. Load PDF
    try:
        logging.info(f"Loading PDF from: {PDF_PATH}")
        documents = SimpleDirectoryReader(input_files=[PDF_PATH]).load_data()
        if not documents:
            logging.error("No documents loaded from PDF.")
            return
        logging.info(f"Loaded {len(documents)} document chunk(s).")
    except Exception as e:
        logging.error(f"Failed to load PDF: {e}")
        return

    # 3. Connect to Milvus Cloud
    try:
        logging.info(f"Connecting to Milvus Cloud at https://{MILVUS_HOST} ...")
        connections.connect(
            alias="default",
            uri=f"https://{MILVUS_HOST}",
            token=MILVUS_TOKEN
        )
        logging.info("Connected to Milvus.")
    except Exception as e:
        logging.error(f"Milvus connection failed: {e}")
        return

    # 4. Check or create collection
    if not utility.has_collection(MILVUS_COLLECTION_NAME):
        logging.info(f"Collection '{MILVUS_COLLECTION_NAME}' does not exist. It will be created automatically.")
    else:
        logging.info(f"Using existing Milvus collection: {MILVUS_COLLECTION_NAME}")

    # 5. Initialize Milvus Vector Store
    vector_store = MilvusVectorStore(
        collection_name=MILVUS_COLLECTION_NAME,
        dim=EMBEDDING_DIM,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 6. Create index and store vectors
    logging.info("Creating index and pushing vectors to Milvus...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    logging.info("--- Ingestion Complete ---")

if __name__ == "__main__":
    main()
