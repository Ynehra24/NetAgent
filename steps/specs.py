from tools.equipment_fetch import get_specs
from basefiles.logger import get_logger

log = get_logger(__name__)

def fetch_and_calculate_specs(num_aps: int) -> dict:
    """
    Fetches equipment specs using the DuckDuckGo search tool,
    falling back to local DB if needed. Calculates total power draw 
    based on the required number of APs.
    """
    log.info(f"Fetching equipment specs for a deployment of {num_aps} APs")
    specs = get_specs()
    
    # Compute total power draw per plan tier
    for tier in ["budget_tier", "mid_tier", "premium_tier"]:
        if tier in specs and "power_w" in specs[tier]:
            specs[tier]["total_power_w"] = specs[tier]["power_w"] * num_aps
            specs[tier]["ap_count"] = num_aps
            
    log.info(f"Successfully loaded and calculated specs for {len(specs)} equipment categories")
    return specs
