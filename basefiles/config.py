import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VISION_MODEL_ID = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{VISION_MODEL_ID}:generateContent"

# General LLM Settings
MAX_TOKENS = 8192
TEMPERATURE = 0.1 # Low temperature for more deterministic JSON output
