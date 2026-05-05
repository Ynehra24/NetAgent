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

# ── Groq API (All text LLM calls — fast inference) ──
GROQ_API_KEY = get_secret("GROQ_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"       # For important calls: summary, variants
GROQ_FAST_MODEL = "llama-3.1-8b-instant"     # For cheap calls: justifications, specs extraction
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# General LLM Settings
MAX_TOKENS = 4096       # Keep small for Groq free tier (12K TPM limit)
GEMINI_MAX_TOKENS = 16384  # Gemini has higher limits
TEMPERATURE = 0.1       # Low temperature for more deterministic JSON output
GROQ_CALL_DELAY = 8     # Seconds to wait between Groq calls (TPM rate limit)
