import json
import re
from typing import Dict, Any

def extract_json_from_markdown(text: str) -> Dict[str, Any]:
    """
    Extracts a JSON object from a markdown string (e.g., inside ```json ... ``` blocks).
    If no markdown block is found, it attempts to parse the raw string.
    """
    # Try to find a JSON block using regex
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    
    json_str = match.group(1) if match else text

    # Strip any leading/trailing whitespace that might cause issues
    json_str = json_str.strip()

    # Attempt to parse
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        # Provide helpful context for debugging
        raise ValueError(f"Failed to parse JSON. Extracted string was:\n{json_str}\n\nOriginal Text:\n{text}") from e
