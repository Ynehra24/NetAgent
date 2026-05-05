import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Helper function to get secret from env, fallback to st.secrets
def get_secret(key):
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None

# ── Gemini API (Vision only — floor plan image extraction) ──
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
VISION_MODEL_ID = "gemini-2.5-flash"
FALLBACK_VISION_MODEL_ID = "gemini-2.5-pro"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{VISION_MODEL_ID}:generateContent"
FALLBACK_GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{FALLBACK_VISION_MODEL_ID}:generateContent"

# ── LLM API (Text generation) ──
GEMINI_TEXT_MODEL = "gemini-2.5-pro"
GEMINI_FAST_MODEL = "gemini-2.5-flash"

# General LLM Settings
MAX_TOKENS = 4096
GEMINI_MAX_TOKENS = 16384
TEMPERATURE = 0.1

