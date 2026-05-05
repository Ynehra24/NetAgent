import requests
import base64
import time
from typing import Dict, Any, Optional
from basefiles.config import (
    GEMINI_API_KEY, GEMINI_API_URL, FALLBACK_GEMINI_API_URL,
    GEMINI_TEXT_MODEL, GEMINI_FAST_MODEL,
    MAX_TOKENS, GEMINI_MAX_TOKENS, TEMPERATURE
)
from basefiles.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 5
RETRY_DELAY = 5.0  # seconds (doubles each retry)


class VisionClient:
    """
    Uses the Gemini API for image analysis (floor plan extraction).
    Groq does not have a vision model, so Gemini handles this one step.
    """
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        self.headers = {
            "Content-Type": "application/json"
        }

    def _encode_image(self, image_path: str) -> str:
        """Encode the image at the given path to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Sends the image and prompt to the Gemini API for the vision model.
        Returns the text response.
        """
        base64_image = self._encode_image(image_path)
        
        mime_type = "image/jpeg"
        if image_path.lower().endswith(".png"):
            mime_type = "image/png"
            
        gemini_payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": GEMINI_MAX_TOKENS,
            }
        }

        if logger:
            logger.info(f"Sending request to Gemini API for image {image_path}")
        else:
            print(f"Sending request to Gemini API for image {image_path}")

        url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        fallback_url_with_key = f"{FALLBACK_GEMINI_API_URL}?key={GEMINI_API_KEY}"
        
        current_url = url_with_key
        for attempt in range(MAX_RETRIES):
            response = requests.post(current_url, headers=self.headers, json=gemini_payload)
            if response.status_code == 200:
                break
                
            if response.status_code == 503 and current_url == url_with_key:
                if logger:
                    logger.warning(f"Gemini API 503 error on primary model. Switching to fallback model...")
                current_url = fallback_url_with_key
                continue
                
            if response.status_code in (429, 503) and attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (2 ** attempt)
                if logger:
                    logger.warning(f"Gemini API transient error {response.status_code}. Retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            error_msg = f"API Request failed with status code {response.status_code}: {response.text}"
            if logger:
                logger.error(error_msg)
            raise Exception(error_msg)

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            error_msg = f"Unexpected response format from API: {data}"
            if logger:
                logger.error(error_msg)
            raise Exception(error_msg) from e


class TextClient:
    """
    Uses the Gemini API for all text generation (justifications, variants, specs RAG).
    Replaces the previous Groq implementation.
    """
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        self.headers = {
            "Content-Type": "application/json"
        }

    def generate_text(self, system_prompt: str, user_prompt: str, fast: bool = False) -> str:
        """
        Sends system + user prompts to the Gemini API and returns the text response.
        fast=True  → gemini-2.5-flash (for cheap calls like justifications/specs)
        fast=False → gemini-2.5-pro (quality model, for variants and summary)
        """
        model_id = GEMINI_FAST_MODEL if fast else GEMINI_TEXT_MODEL
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"

        if logger:
            logger.info(f"Sending text request to Gemini API (model: {model_id})")

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": GEMINI_MAX_TOKENS,
            }
        }

        max_attempts = 5
        response = None
        for attempt in range(max_attempts):
            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code == 200:
                break
            
            if response.status_code in (429, 503) and attempt < max_attempts - 1:
                wait = RETRY_DELAY * (2 ** attempt)
                if logger:
                    logger.warning(f"Gemini API transient error {response.status_code}. Retrying in {wait}s (attempt {attempt+1}/{max_attempts})...")
                time.sleep(wait)
                continue
                
            error_msg = f"Gemini API failed with status code {response.status_code}: {response.text}"
            if logger:
                logger.error(error_msg)
            raise Exception(error_msg)

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            error_msg = f"Unexpected response format from Gemini API: {data}"
            if logger:
                logger.error(error_msg)
            raise Exception(error_msg) from e
