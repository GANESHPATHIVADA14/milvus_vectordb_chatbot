# main.py
import os
from fastapi import FastAPI, Query
from dotenv import load_dotenv
from pymilvus import connections
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from fastapi.middleware.cors import CORSMiddleware

# Load .env
load_dotenv()

# Env vars
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "milvus_chatbot_vectordb")
EMBEDDING_DIM = 768

# Connect to Milvus
connections.connect(
    alias="default",
    uri=f"https://{MILVUS_HOST}",
    token=MILVUS_TOKEN
)

# Setup LlamaIndex
Settings.llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=GOOGLE_API_KEY
)
Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001",
    api_key=GOOGLE_API_KEY
)

# Load vector store
vector_store = MilvusVectorStore(
    collection_name=MILVUS_COLLECTION_NAME,
    dim=EMBEDDING_DIM
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
query_engine = index.as_query_engine()

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chat")
def chat(query: str = Query(..., description="User question")):
    try:
        response = query_engine.query(query)
        return {
            "query": query,
            "response": str(response)
        }
    except Exception as e:
        return {
            "error": str(e)
        }
