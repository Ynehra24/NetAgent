from typing import Dict, Any
from llm.client import VisionClient
from llm.parse import extract_json_from_markdown
from basefiles.logger import get_logger

logger = get_logger(__name__) if 'get_logger' in globals() else None

SYSTEM_PROMPT = """
You are an expert architect and AI computer vision assistant.
Your task is to analyze the provided floor plan image and extract its structure and details into a strict JSON format.

We use a normalized coordinate system. You MUST map the entire floor plan to a 2D grid from `[0, 0]` (Top-Left) to `[1000, 1000]` (Bottom-Right). 
Whenever asked for a "bounding_box", provide it as an array of 4 integers: `[ymin, xmin, ymax, xmax]` representing the coordinates on this 0-1000 grid.

Please extract the following information and output it ONLY as a valid JSON object matching the exact schema below. Do not include any other commentary.

Requirements:
- Look for an explicit scale on the floor plan. If not found, use a standard interior door as the scale anchor (assume 81cm / 32in width). Describe this in `scale_anchor`.
- Identify all visible rooms and infer their likely usage. Provide a bounding box for the entire room.
- Identify any notable objects/fixtures within the rooms (e.g., Sofa, Bed, Router, Toilet, TV). Provide a bounding box for each object.
- Identify doors and windows associated with each room and provide a bounding box for each.

JSON Schema:
```json
{
  "floor_plan": {
    "scale_anchor": "Description of reference scale (e.g., 'door width' or 'explicit scale bar')",
    "rooms": [
      {
        "name": "String (e.g., 'Living Room', 'Bedroom 1')",
        "bounding_box": [ymin, xmin, ymax, xmax],
        "objects": [
          {
            "type": "String",
            "bounding_box": [ymin, xmin, ymax, xmax]
          }
        ],
        "doors": [
          {"location": "Description of where it leads", "bounding_box": [ymin, xmin, ymax, xmax]}
        ],
        "windows": [
          {"bounding_box": [ymin, xmin, ymax, xmax]}
        ]
      }
    ]
  }
}
```
"""

def extract_floorplan_features(image_path: str) -> Dict[str, Any]:
    """
    Takes a path to a floor plan image, passes it to the VLM, and returns a structured dictionary
    containing the extracted features (rooms, dimensions, objects, walls).
    """
    if logger:
        logger.info(f"Extracting features from floor plan: {image_path}")
    else:
        print(f"Extracting features from floor plan: {image_path}")

    client = VisionClient()
    
    # Use the system prompt along with a specific request
    prompt = SYSTEM_PROMPT + "\n\nPlease analyze the provided floor plan image and return the JSON."

    try:
        response_text = client.analyze_image(image_path, prompt)
        
        if logger:
            logger.debug(f"Raw VLM Response:\n{response_text}")

        # Parse the JSON from the markdown block
        json_data = extract_json_from_markdown(response_text)
        return json_data

    except Exception as e:
        if logger:
            logger.error(f"Failed to extract floor plan features: {e}")
        else:
            print(f"Failed to extract floor plan features: {e}")
        raise
