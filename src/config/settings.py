import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 300))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 50))
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE    = float(os.getenv("TEMPERATURE", 0.4))
os.environ["OPENAI_API_KEY"] = "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"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "../data/faiss_index") 

