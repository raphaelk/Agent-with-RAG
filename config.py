"""
Configuration parameters for Agent with RAG application.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
SAMPLE_DOCS_DIR = BASE_DIR / "sample_docs"
SERVICES_DIR = BASE_DIR / "services"
SKILLS_DIR = BASE_DIR / "skills"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure runtime directories exist
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Vector Database Files
DOC_VECTOR_DB_FILE = DATABASE_DIR / "doc_vectors.json"
SKILL_VECTOR_DB_FILE = DATABASE_DIR / "skill_vectors.json"
LOG_FILE = DATABASE_DIR / "log.json"
TELEMETRY_FILE = DATABASE_DIR / "telemetry.json"

# Score Thresholds & Execution Limits
DEFAULT_SKILL_THRESHOLD = 0.2
DEFAULT_DOC_THRESHOLD = 0.3
MIN_SKILL_SCORE = 0.2
MIN_RAG_DOC_SCORE = 0.3
DEFAULT_MAX_TURNS = 3
MAX_TURNS_LIMIT = 10

# Default Chunking Parameters
DEFAULT_CHUNK_SIZE = 500  # characters
DEFAULT_CHUNK_OVERLAP = 100  # characters

# Ollama Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_EMBEDDING_MODEL = "bge-m3:latest"

# Available Embedding Models Catalog
SUPPORTED_EMBEDDING_MODELS = [
    {
        "name": "bge-m3:latest",
        "dimensions": 1024,
        "context_window": 8192,
        "size": "1.2 GB",
        "description": "Multi-lingual, multi-functionality state-of-the-art embedding model by BAAI.",
    },
    {
        "name": "bge-large:latest",
        "dimensions": 1024,
        "context_window": 512,
        "size": "670 MB",
        "description": "High-performance English embedding model optimized for semantic search.",
    },
    {
        "name": "nomic-embed-text:latest",
        "dimensions": 768,
        "context_window": 8192,
        "size": "274 MB",
        "description": "Long-context embedding model with strong benchmark performance.",
    },
    {
        "name": "all-minilm:latest",
        "dimensions": 384,
        "context_window": 512,
        "size": "45 MB",
        "description": "Ultra-fast and lightweight embedding model for low-latency tasks.",
    },
    {
        "name": "mxbai-embed-large:latest",
        "dimensions": 1024,
        "context_window": 512,
        "size": "670 MB",
        "description": "Competitive open-source sentence embedding model from Mixedbread AI.",
    },
]

# Google AI Studio / Gemini Models
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
DEFAULT_LLM_MODEL = "gemma-4-26b-a4b-it"
GOOGLE_AI_MODELS = [
    {"id": "gemma-4-26b-a4b-it", "name": "Gemma 4 26B A4B It", "max_tokens": 8192},
    {"id": "gemini-flash-lite-latest", "name": "Gemini Flash Lite Latest", "max_tokens": 8192},
]
DEFAULT_CUSTOM_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"

# Server Configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000
DEBUG_MODE = False
