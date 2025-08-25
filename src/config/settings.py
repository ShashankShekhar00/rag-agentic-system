import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DEBUG = os.getenv("DEBUG", "True") == "True"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")