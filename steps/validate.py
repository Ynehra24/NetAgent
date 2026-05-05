"""
Step 3.5: Placement Validation via Web-Sourced Best Practices

Searches DuckDuckGo for real-world wiring, electrical, and AP placement
guidelines, scrapes the content, and sends it to the LLM along with the
current placement plan. The LLM then validates each device placement
against sourced best practices and suggests corrections.

This step runs AFTER placement (Step 3) and BEFORE visualization (Step 5).
"""

import json
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from llm.client import TextClient
from llm.parse import extract_json_from_markdown
from basefiles.logger import get_logger

log = get_logger(__name__)

# ── Queries to search for real-world placement best practices ──
VALIDATION_QUERIES = [
    "WiFi access point placement best practices indoor guidelines",
    "network data point ethernet wall port placement guidelines building",
    "router switch placement wiring closet best practices enterprise",
]

VALIDATION_SYSTEM_PROMPT = """\
You are a certified wireless network engineer and structured cabling specialist.
You have been given:
1. A placement plan with AP, router, switch, and data point positions on a 0-1000 grid.
2. Real-world best practice guidelines scraped from industry sources.

Your job is to VALIDATE each device placement against the sourced guidelines and flag any issues.

RULES:
- Only flag issues that are backed by the scraped guidelines. Do NOT hallucinate rules.
- Quote the specific guideline or source snippet that supports each flag.
- For each flagged device, suggest a corrected position (x, y on the 0-1000 grid) that is
  INSIDE the room's bounding_box [ymin, xmin, ymax, xmax].
- If a placement is acceptable, mark it as "ok".

Return ONLY valid JSON in this exact schema:
{
  "validation_results": [
    {
      "device_id": "<string: e.g. ap_1, dp_3, router_1>",
      "status": "<string: 'ok' | 'flagged'>",
      "issue": "<string: description of the issue, or null if ok>",
      "guideline_source": "<string: quoted snippet from the scraped text that supports the flag, or null>",
      "suggested_position": {"x": <int>, "y": <int>} or null,
      "room_name": "<string: room name>"
    }
  ],
  "general_recommendations": ["<string: any overall best-practice advice from the sources>"],
  "sources_used": ["<string: URLs that were successfully scraped>"]
}
"""


def _scrape_url(url: str) -> str:
    """Download and clean the HTML of the given URL, returning plain text."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()

        text = soup.get_text(separator=" ", strip=True)
        return text[:3000]
    except Exception as e:
        log.warning(f"Failed to scrape {url}: {e}")
        return ""


def _search_and_scrape(queries: list[str], max_results_per_query: int = 2) -> tuple[str, list[str]]:
    """
    Search DuckDuckGo for each query, scrape the top results, and return
    the combined text along with the list of source URLs.
    """
    all_text = []
    all_urls = []

    for query in queries:
        log.info(f"Searching DDG: {query}")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results_per_query))
        except Exception as e:
            log.warning(f"DDG search failed for '{query}': {e}")
            continue

        for result in results:
            url = result.get("href", "")
            if not url or url in all_urls:
                continue

            log.info(f"Scraping: {url}")
            text = _scrape_url(url)
            if text:
                all_text.append(f"--- Source: {url} ---\n{text}\n")
                all_urls.append(url)

    combined = "\n\n".join(all_text)
    return combined[:8000], all_urls


def _build_placement_summary(placement_plan: dict) -> str:
    """Build a compact text summary of all device placements for the LLM."""
    lines = []

    for ap in placement_plan.get("ap_placements", []):
        lines.append(
            f"  {ap['id']}: room='{ap['placed_in_room']}', "
            f"pos=({ap['position']['x']}, {ap['position']['y']}), "
            f"bbox={ap.get('bounding_box', 'N/A')}, "
            f"covers={ap.get('covers_rooms', [])}"
        )

    for dev in placement_plan.get("infra_devices", []):
        lines.append(
            f"  {dev['id']} ({dev['type']}): room='{dev['placed_in_room']}', "
            f"pos=({dev['position']['x']}, {dev['position']['y']})"
        )

    return "Current Device Placements:\n" + "\n".join(lines)


def _apply_corrections(placement_plan: dict, validation: dict) -> tuple[dict, int]:
    """
    Apply position corrections from validation results back into the placement plan.
    Returns the updated plan and the number of corrections applied.
    """
    corrections = 0
    results = validation.get("validation_results", [])

    # Build lookup: device_id → suggested_position
    correction_map = {}
    for r in results:
        if r.get("status") == "flagged" and r.get("suggested_position"):
            correction_map[r["device_id"]] = r["suggested_position"]

    # Apply to AP placements
    for ap in placement_plan.get("ap_placements", []):
        if ap["id"] in correction_map:
            old = ap["position"].copy()
            new_pos = correction_map[ap["id"]]

            # Validate new position is inside the room's bounding box
            bb = ap.get("bounding_box")
            if bb and len(bb) == 4:
                ymin, xmin, ymax, xmax = bb
                new_pos["x"] = max(xmin, min(xmax, new_pos["x"]))
                new_pos["y"] = max(ymin, min(ymax, new_pos["y"]))

            ap["position"] = new_pos
            corrections += 1
            log.info(
                f"Corrected {ap['id']}: ({old['x']}, {old['y']}) → "
                f"({new_pos['x']}, {new_pos['y']}) in '{ap['placed_in_room']}'"
            )

    # Apply to infrastructure devices
    for dev in placement_plan.get("infra_devices", []):
        if dev["id"] in correction_map:
            old = dev["position"].copy()
            dev["position"] = correction_map[dev["id"]]
            corrections += 1
            log.info(
                f"Corrected {dev['id']}: ({old['x']}, {old['y']}) → "
                f"({dev['position']['x']}, {dev['position']['y']}) "
                f"in '{dev['placed_in_room']}'"
            )

    return placement_plan, corrections


def validate_placement(placement_plan: dict) -> dict:
    """
    Main entry point for placement validation.

    1. Searches DuckDuckGo for placement best practices.
    2. Scrapes the top results.
    3. Sends scraped guidelines + current placements to the LLM for validation.
    4. Applies any suggested corrections.
    5. Returns the (possibly corrected) placement plan with validation metadata.
    """
    log.info("=== Placement Validation: Searching for Best Practices ===")

    # ── 1. Search & Scrape ──
    guidelines_text, source_urls = _search_and_scrape(VALIDATION_QUERIES)

    if not guidelines_text:
        log.warning("No guidelines could be scraped. Skipping validation.")
        placement_plan["validation"] = {
            "status": "skipped",
            "reason": "No best-practice sources could be fetched",
        }
        return placement_plan

    log.info(f"Scraped {len(source_urls)} source(s) ({len(guidelines_text)} chars)")

    # ── 2. Build LLM prompt ──
    placement_summary = _build_placement_summary(placement_plan)
    rooms_summary = json.dumps(
        placement_plan.get("all_rooms", []), indent=2
    )

    user_prompt = f"""\
{placement_summary}

Room Bounding Boxes (0-1000 grid, format [ymin, xmin, ymax, xmax]):
{rooms_summary}

Scale: {placement_plan.get('scale_factor_cm_per_unit', 4.0)} cm/unit
Wall material: {placement_plan.get('wall_material', 'unknown')}

--- SCRAPED BEST PRACTICE GUIDELINES ---
{guidelines_text}
--- END GUIDELINES ---

Validate every device placement against the guidelines above.
Flag any placement that violates a real guideline and suggest a corrected position INSIDE the room's bounding_box.
"""

    # ── 3. LLM Validation Call ──
    log.info("Sending placement plan to LLM for validation against scraped guidelines...")
    try:
        client = TextClient()
        raw_response = client.generate_text(VALIDATION_SYSTEM_PROMPT, user_prompt)
        validation = extract_json_from_markdown(raw_response)
    except Exception as e:
        log.warning(f"LLM validation call failed: {e}. Skipping correction.")
        placement_plan["validation"] = {
            "status": "error",
            "reason": str(e),
        }
        return placement_plan

    # ── 4. Log results ──
    results = validation.get("validation_results", [])
    flagged = [r for r in results if r.get("status") == "flagged"]
    ok_count = len(results) - len(flagged)
    log.info(f"Validation complete: {ok_count} OK, {len(flagged)} flagged")

    for r in flagged:
        log.warning(
            f"  FLAGGED {r['device_id']}: {r.get('issue', 'No description')} "
            f"[Source: {r.get('guideline_source', 'N/A')[:80]}...]"
        )

    for rec in validation.get("general_recommendations", []):
        log.info(f"  Recommendation: {rec}")

    # ── 5. Apply corrections ──
    placement_plan, num_corrections = _apply_corrections(placement_plan, validation)
    log.info(f"Applied {num_corrections} position correction(s)")

    # ── 6. Attach validation metadata ──
    placement_plan["validation"] = {
        "status": "completed",
        "sources": source_urls,
        "flagged_count": len(flagged),
        "corrections_applied": num_corrections,
        "results": results,
        "general_recommendations": validation.get("general_recommendations", []),
    }

    log.info("=== Placement Validation Complete ===")
    return placement_plan
