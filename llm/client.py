import requests
import base64
from typing import Dict, Any, Optional
from basefiles.config import GEMINI_API_KEY, API_URL, MAX_TOKENS, TEMPERATURE
from basefiles.logger import get_logger

logger = get_logger(__name__) if 'get_logger' in globals() else None

class VisionClient:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        # The key is passed in the URL for Gemini REST API, not headers
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
        
        # Determine mime type based on extension
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
                "maxOutputTokens": MAX_TOKENS,
            }
        }

        if logger:
            logger.info(f"Sending request to Gemini API for image {image_path}")
        else:
            print(f"Sending request to Gemini API for image {image_path}")

        url_with_key = f"{API_URL}?key={GEMINI_API_KEY}"
        response = requests.post(url_with_key, headers=self.headers, json=gemini_payload)
        
        if response.status_code != 200:
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
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        self.headers = {
            "Content-Type": "application/json"
        }

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends the text prompts to the Gemini API and returns the text response.
        """
        gemini_payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_prompt}\n\n{user_prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": MAX_TOKENS,
            }
        }

        if logger:
            logger.info("Sending text request to Gemini API")
        
        url_with_key = f"{API_URL}?key={GEMINI_API_KEY}"
        response = requests.post(url_with_key, headers=self.headers, json=gemini_payload)
        
        if response.status_code != 200:
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

