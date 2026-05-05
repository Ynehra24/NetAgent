import json
import requests
from basefiles.config import GROQ_API_KEY, GROQ_MODEL, GROQ_API_URL
from basefiles.logger import get_logger

log = get_logger(__name__)

SUMMARY_SYSTEM_PROMPT = """
You are a professional network engineer writing an executive summary report.
Given a complete RF deployment pipeline output (JSON), produce a clear, well-structured
markdown report that a client or professor can read and understand immediately.

The report MUST include these sections with proper markdown formatting:
1. **Executive Summary** — 2-3 sentence overview of the deployment
2. **Floor Plan Analysis** — How many rooms were detected, building dimensions, room types
3. **Access Point Placement** — Where each AP was placed, why, and what rooms it covers
4. **Signal Coverage Analysis** — Wall material, attenuation values, coverage radius, any dead zones
5. **Infrastructure Devices** — Router, switch, and data point placement summary
6. **Cost Analysis** — Budget vs Premium plan comparison table with costs, throughput, capacity
7. **Recommendation** — Which plan to choose and why
8. **Technical Specifications** — Equipment models, frequencies, power specs

Use tables, bold text, and clear formatting. Be concise but thorough.
Do NOT output JSON. Output ONLY clean markdown.
"""


def generate_summary(pipeline_report: dict, output_path: str = "outputs/summary_report.md") -> str:
    """
    Uses the Groq API to generate a well-formatted executive summary
    from the full pipeline report JSON.
    
    This is the final step in the chain — it takes ALL accumulated state
    and synthesizes it into a human-readable document.
    """
    log.info("=== Generating Executive Summary via Groq ===")
    
    if not GROQ_API_KEY:
        raise ValueError("GROQ_KEY is not set in the environment variables.")
    
    # Wait for TPM rate limit to reset (Groq free tier: 12K TPM)
    import time
    log.info("Waiting for Groq rate limit window to reset...")
    time.sleep(15)
    
    # Prepare a focused payload (trim verbose fields to stay within context)
    focused_report = {
        "rooms": [
            {"name": r.get("name"), "bounding_box": r.get("bounding_box")}
            for r in pipeline_report.get("parsed_layout", {}).get("floor_plan", {}).get("rooms", [])
        ],
        "ap_placements": [
            {
                "id": ap.get("id"),
                "room": ap.get("placed_in_room"),
                "position": ap.get("position"),
                "covers": ap.get("covers_rooms"),
            }
            for ap in pipeline_report.get("placement_plan", {}).get("ap_placements", [])
        ],
        "infra_devices": [
            {
                "id": d.get("id"),
                "type": d.get("type"),
                "room": d.get("placed_in_room"),
                "model": d.get("model"),
            }
            for d in pipeline_report.get("placement_plan", {}).get("infra_devices", [])
        ],
        "scale_factor": pipeline_report.get("placement_plan", {}).get("scale_factor_cm_per_unit"),
        "max_range_m": pipeline_report.get("placement_plan", {}).get("max_indoor_range_m"),
        "wall_material": pipeline_report.get("placement_plan", {}).get("wall_material"),
        "wall_attenuation_db": pipeline_report.get("placement_plan", {}).get("wall_attenuation_db"),
        "total_aps": pipeline_report.get("placement_plan", {}).get("total_aps_needed"),
        "variants": pipeline_report.get("variants"),
    }
    
    user_prompt = f"""
Here is the complete deployment pipeline output:

{json.dumps(focused_report, indent=2)}

Generate a professional executive summary report in markdown format.
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        error_msg = f"Groq API failed with status code {response.status_code}: {response.text}"
        log.error(error_msg)
        raise Exception(error_msg)
    
    data = response.json()
    summary_text = data["choices"][0]["message"]["content"]
    
    # Save to file
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(summary_text)
    
    log.info(f"Executive summary saved to {output_path}")
    log.info(f"=== Summary Generation Complete ===")
    
    return summary_text
