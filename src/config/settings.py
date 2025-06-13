import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 300))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 50))
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE    = float(os.getenv("TEMPERATURE", 0.4))
os.environ["OPENAI_API_KEY"] = ""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "../data/faiss_index") 

