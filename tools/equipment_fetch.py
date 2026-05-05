import json
import os
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from basefiles.logger import get_logger
from llm.client import TextClient
from llm.parse import extract_json_from_markdown

log = get_logger(__name__)

SYSTEM_PROMPT = """
You are a networking hardware engineer. Read the following raw text scraped from a manufacturer's website or datasheet.
Extract the exact technical specifications for the networking device into JSON.
If a value is not found, estimate it based on standard industry specs for a device of this tier, but prefer exact numbers from the text.

Return exactly this JSON schema:
{
  "power_w": <float: Maximum Power Consumption in Watts>,
  "max_users": <int: Concurrent clients/users>,
  "frequency": ["<string: e.g. 2.4GHz>", "<string: e.g. 5GHz>"],
  "antenna_gain_dbi": <float: Max antenna gain>,
  "base_price_usd": <float: Estimate MSRP if not listed>
}
"""

def scrape_manual(url: str) -> str:
    """Download and clean the HTML of the given URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        return text[:4000]
    except Exception as e:
        log.warning(f"Failed to scrape {url}: {e}")
        return ""

def fetch_from_api(tier: str) -> dict | None:
    """
    Attempt to fetch live equipment specs using DuckDuckGo search + HTML scraping + LLM RAG.
    Tries up to 2 search results before falling back to local DB.
    Returns None on any failure so the fallback triggers.
    """
    try:
        query_map = {
            "budget_tier":  "TP-Link EAP225 official specifications datasheet",
            "mid_tier":     "Ubiquiti UniFi U6 Lite official specifications datasheet",
            "premium_tier": "Ubiquiti UniFi U6 Pro official specifications datasheet",
        }
        
        query = query_map.get(tier, tier)
        log.info(f"Searching web for: {query}")
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        
        if not results:
            log.warning(f"DuckDuckGo returned no results for {tier}")
            return None

        # Try each result URL until one gives usable text
        for idx, result in enumerate(results[:2]):
            url = result.get("href", "")
            if not url:
                continue
            
            log.debug(f"DuckDuckGo result {idx+1} for {tier}: {url}")
            raw_text = scrape_manual(url)
            
            if not raw_text:
                log.debug(f"Result {idx+1} gave no text (404/blocked), trying next...")
                continue
            
            # Use LLM to extract factual data from the scraped page
            log.info(f"Extracting specs from result {idx+1} via Gemini: {url}")
            llm_client = TextClient()
            response_text = llm_client.generate_text(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=f"Device: {tier}\n\nDatasheet Text:\n{raw_text}",
                fast=True
            )
            
            parsed_data = extract_json_from_markdown(response_text)
            parsed_data["source"] = url
            parsed_data["model"] = query.split("official")[0].strip()
            
            log.info(f"Successfully extracted real-world specs from result {idx+1}: {url}")
            return parsed_data
            
        log.warning(f"All {len(results[:2])} DuckDuckGo results failed for {tier}")
            
    except Exception as e:
        log.warning(f"Web search/RAG failed for {tier}: {e}")

    return None

def load_local_db() -> dict:
    db_path = "data/equipment_db.json"
    with open(db_path) as f:
        return json.load(f)

def get_specs() -> dict:
    """
    Main interface: try live DuckDuckGo RAG search, fall back to local DB.
    Always returns a complete specs dict.
    """
    local = load_local_db()
    specs = {}
    
    specs["attenuation_db"] = local.get("attenuation_db", {})
    
    tiers = ["budget_tier", "mid_tier", "premium_tier",
             "switch_budget", "switch_premium"]

    for tier in tiers:
        if tier in ["switch_budget", "switch_premium"]:
            specs[tier] = local[tier]
            continue

        live = fetch_from_api(tier)
        if live:
            specs[tier] = live
            log.info(f"Specs for {tier}: using live scraped RAG data")
        else:
            specs[tier] = local[tier]
            log.info(f"Specs for {tier}: using local DB (Search/Scrape unavailable)")

    return specs
