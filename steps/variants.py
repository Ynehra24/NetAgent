import json
from typing import Dict, Any
from llm.client import TextClient
from llm.parse import extract_json_from_markdown
from basefiles.logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """
You are a network procurement specialist and certified wireless engineer (CWNE).
You will be given:
1. A mathematically determined AP placement plan.
2. Real-world equipment specifications.
3. User capacity requirements.

Your job is to produce two concrete, costed deployment variants using ONLY the equipment 
models and prices listed in the provided specs. Do NOT invent model names or prices.

For each variant:
- Total AP cost = ap_unit_price × ap_quantity
- Total switch cost = switch_unit_price × switch_quantity
- Cabling estimate = $15 per metre of cable run (assume 3m average run per AP + 5m to switch)
- Grand total = AP cost + switch cost + cabling estimate
- Analyze user capacity: compare Total Concurrent Users against the AP's `max_users`. If total users > `max_users` * ap_quantity, it is OVERLOADED.
- Expected Throughput: Assume real-world max throughput is 300Mbps for Wi-Fi 5 (budget), and 600Mbps for Wi-Fi 6 (premium). Divide this by Total Concurrent Users to get `estimated_throughput_mbps_per_user`.

Return ONLY a valid JSON object with exactly this schema:
{
  "budget_plan": {
    "ap_model": "<exact model name>",
    "ap_quantity": <int>,
    "ap_unit_price": <float>,
    "ap_total_cost": <float>,
    "switch_model": "<exact model name>",
    "switch_quantity": 1,
    "switch_unit_price": <float>,
    "cabling_estimate_usd": <float>,
    "grand_total_usd": <float>,
    "estimated_coverage_pct": <number 0-100>,
    "max_concurrent_users": <int>,
    "estimated_throughput_mbps_per_user": <number>,
    "is_overloaded": <bool>,
    "limitations": ["<specific, honest limitation>"]
  },
  "premium_plan": {
    "ap_model": "<exact model name>",
    "ap_quantity": <int>,
    "ap_unit_price": <float>,
    "ap_total_cost": <float>,
    "switch_model": "<exact model name>",
    "switch_quantity": 1,
    "switch_unit_price": <float>,
    "cabling_estimate_usd": <float>,
    "grand_total_usd": <float>,
    "estimated_coverage_pct": <number 0-100>,
    "max_concurrent_users": <int>,
    "estimated_throughput_mbps_per_user": <number>,
    "is_overloaded": <bool>,
    "limitations": ["<specific, honest limitation>"]
  },
  "recommendation": "<one paragraph: which plan, for what reason, with what caveats>"
}
"""

def generate_variants(placement_plan: dict, specs: dict, budget_limit: float = 1000.0) -> Dict[str, Any]:
    """
    Takes the AP placement plan and real equipment specs, and calls the LLM
    to generate two costed deployment variants (budget vs. premium).
    The LLM is constrained to ONLY use equipment from the provided spec menu.
    """
    log.info("=== Generating Deployment Variants ===")
    
    num_aps = placement_plan.get("total_aps_needed", 1)
    
    # Calculate expected user density
    # Assume 3 heavy devices per room (phones, smart TVs, laptops)
    rooms_covered = set()
    for ap in placement_plan.get("ap_placements", []):
        for room in ap.get("covers_rooms", []):
            rooms_covered.add(room)
            
    total_rooms = len(rooms_covered)
    total_users = total_rooms * 3 if total_rooms > 0 else 15
    
    log.info(f"Generating variants for {num_aps} AP(s). Expected concurrent users: {total_users}. Budget: ${budget_limit:.0f}")

    # Build a COMPACT prompt (Groq free tier has 12K TPM limit)
    # Only send essential data, not the full verbose placement plan
    ap_summary = [{"id": ap["id"], "room": ap["placed_in_room"]} 
                  for ap in placement_plan.get("ap_placements", [])]
    
    # Only send pricing/specs, not the full raw data
    compact_specs = {}
    for tier in ["budget_tier", "mid_tier", "premium_tier", "switch_budget", "switch_premium"]:
        s = specs.get(tier, {})
        compact_specs[tier] = {
            "model": s.get("model", "Unknown"),
            "base_price_usd": s.get("base_price_usd", 0),
            "max_users": s.get("max_users", 0),
            "poe_ports": s.get("poe_ports"),
            "wifi_generation": s.get("wifi_generation"),
        }

    user_prompt = f"""
APs placed: {num_aps}
Rooms covered: {json.dumps(ap_summary)}

Equipment specs:
{json.dumps(compact_specs, indent=2)}

Requirements:
- APs required: {num_aps}
- Concurrent heavy users: {total_users}
- Budget limit: ${budget_limit:.0f}

Rules:
- Budget plan MUST use budget_tier APs + switch_budget switch.
- Premium plan MUST use premium_tier APs + switch_premium switch.
- Budget plan should try to stay under ${budget_limit:.0f}.
- If total_users > max_users * ap_quantity, mark is_overloaded true.
"""

    client = TextClient()
    raw = client.generate_text(SYSTEM_PROMPT, user_prompt)  # 70B for quality cost analysis
    variants = extract_json_from_markdown(raw)
    
    # Validate expected keys exist
    for key in ["budget_plan", "premium_plan", "recommendation"]:
        if key not in variants:
            raise ValueError(f"LLM response missing required key: '{key}'")

    b_total = variants["budget_plan"]["grand_total_usd"]
    p_total = variants["premium_plan"]["grand_total_usd"]
    log.info(f"Budget plan total:  ${b_total:.0f}")
    log.info(f"Premium plan total: ${p_total:.0f}")

    # ── DETERMINISTIC 'best within budget' override ──
    # Python math, not LLM opinion, decides the recommendation.
    if p_total <= budget_limit:
        best = "premium"
        reco_reason = (f"The premium plan (${p_total:.0f}) fits within the ${budget_limit:.0f} budget "
                       f"and provides the highest performance. Recommended.")
        log.info(f"Premium plan fits within budget (${p_total:.0f} ≤ ${budget_limit:.0f}) → recommending premium")
    elif b_total <= budget_limit:
        best = "budget"
        reco_reason = (f"The budget plan (${b_total:.0f}) is the best option within the ${budget_limit:.0f} budget. "
                       f"The premium plan (${p_total:.0f}) exceeds the limit.")
        log.info(f"Only budget plan fits within budget (${b_total:.0f} ≤ ${budget_limit:.0f}) → recommending budget")
    else:
        best = "budget"
        reco_reason = (f"Neither plan fits strictly within ${budget_limit:.0f}. "
                       f"The budget plan (${b_total:.0f}) is the closest option; consider negotiating or phasing the rollout.")
        log.warning(f"Both plans exceed budget (budget=${b_total:.0f}, premium=${p_total:.0f})")

    variants["recommended_plan"] = best
    variants["recommendation"] = reco_reason

    # Flag plans that exceed budget
    if b_total > budget_limit:
        variants["budget_plan"].setdefault("limitations", []).append(f"Exceeds budget limit of ${budget_limit:.0f}")
    if p_total > budget_limit:
        variants["premium_plan"].setdefault("limitations", []).append(f"Exceeds budget limit of ${budget_limit:.0f}")

    log.info("=== Variants Generation Complete ===")
    return variants
