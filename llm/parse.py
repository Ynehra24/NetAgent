import json
import re
from typing import Dict, Any

def extract_json_from_markdown(text: str) -> Dict[str, Any]:
    """
    Extracts a JSON object from a markdown string (e.g., inside ```json ... ``` blocks).
    If no markdown block is found, it attempts to parse the raw string.
    If the response is truncated, it walks back from the end to find the last parseable sub-object.
    """
    # Strategy 1: Strip markdown fences (handles ```json and plain ```)
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    json_str = match.group(1) if match else text
    json_str = json_str.strip()

    # Strategy 2: Find the first { to skip any preamble text
    start = json_str.find("{")
    if start != -1:
        json_str = json_str[start:]

    for end in range(len(json_str), 0, -1):
        candidate = json_str[:end]
        last_brace = candidate.rfind("}")
        if last_brace == -1:
            continue
        candidate = candidate[:last_brace + 1]
        
        # Try raw
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
            
        # Try appending common missing closures
        for suffix in ["}", "]}", "]}}", "]}]}"]:
            try:
                return json.loads(candidate + suffix)
            except json.JSONDecodeError:
                continue

    raise ValueError(
        f"Failed to parse any valid JSON from response.\n\n"
        f"Original Text (last 500 chars):\n{text[-500:]}"
    )
