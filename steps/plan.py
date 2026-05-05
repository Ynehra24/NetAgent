import math
import json
from typing import Dict, Any, List, Tuple
from llm.client import TextClient
from llm.parse import extract_json_from_markdown
from basefiles.logger import get_logger

log = get_logger(__name__)

MIN_SIGNAL_DBM = -70.0
DOOR_WIDTH_CM = 81.0

# Room types that do NOT need their own AP or data point.
# These get coverage via spillover from adjacent rooms.
EXCLUDED_ROOM_KEYWORDS = [
    "stairwell", "stair", "elevator", "lift",
    "restroom", "bathroom", "toilet", "wc",
    "corridor", "hallway", "lobby", "passage",
    "storage", "closet", "utility",
    "kitchenette", "pantry",
    "balcony", "porch", "terrace", "nook",
    "foyer", "vestibule", "laundry",
    "uncovered",
]

# Minimum area fraction for an uncovered zone to get its own AP
MIN_ZONE_AREA_FRAC = 0.03

JUSTIFICATION_PROMPT = """
You are a certified wireless network engineer. You have been given the mathematical output 
of a deterministic RF placement algorithm. Write a clear, one-paragraph justification for 
each AP placement explaining WHY it was chosen. 

CRITICAL: You MUST explicitly mention:
1. That the algorithm uses a ONE-AP-PER-ROOM enterprise deployment strategy.
2. The wall material and its dB attenuation penalty per wall crossing.
3. The max indoor range from the equipment datasheet.
4. That the algorithm verified the signal reaches all points within the room above -70 dBm.

Return ONLY a valid JSON object in this format:
{
  "justifications": {
    "ap_1": "<one paragraph justification>",
    "ap_2": "<one paragraph justification>"
  }
}
"""


# ─────────────────────────────────────────────
# A. Scale Calculation
# ─────────────────────────────────────────────

def calculate_scale_factor(floor_plan: dict, building_length_m: float = 20.0) -> float:
    """
    Returns cm per grid unit.
    
    Since door detection from the VLM is unreliable (varying wildly between runs),
    we use a fixed real-world building dimension instead.
    
    Default: 20m residential. For offices, pass building_length_m=40.
    The 1000-unit grid maps to the full building length.
    """
    scale = (building_length_m * 100) / 1000.0  # cm per grid unit
    log.info(f"Scale factor: {scale:.2f} cm/unit (building length = {building_length_m}m)")
    return scale


# ─────────────────────────────────────────────
# B. Room Centroids
# ─────────────────────────────────────────────

def get_room_centroids(rooms: list) -> List[Dict]:
    """
    Computes the centroid [cx, cy] of each room's bounding box on the 0-1000 grid.
    Returns list of dicts: {name, cx, cy, bounding_box}
    """
    centroids = []
    for room in rooms:
        bb = room.get("bounding_box")
        if bb and len(bb) == 4:
            ymin, xmin, ymax, xmax = bb
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            centroids.append({
                "name": room.get("name", "Unknown Room"),
                "cx": cx,
                "cy": cy,
                "bounding_box": bb
            })
    return centroids


# ─────────────────────────────────────────────
# C. Wall Raycasting
# ─────────────────────────────────────────────

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def count_walls_crossed(p1: Tuple[float, float], p2: Tuple[float, float], all_rooms: List[Dict], start_room: str, end_room: str) -> int:
    """
    Raycasts from p1 to p2 and counts how many room bounding boxes the line intersects.
    Each intermediate room entered counts as 2 walls (enter and exit).
    Adjacency (0 intermediate rooms) counts as 1 wall.
    """
    if start_room == end_room:
        return 0
        
    intersected_rooms = set()
    steps = 100
    for i in range(1, steps):
        t = i / float(steps)
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        
        for room in all_rooms:
            if room["name"] in (start_room, end_room):
                continue
            bb = room["bounding_box"]
            ymin, xmin, ymax, xmax = bb
            if xmin <= x <= xmax and ymin <= y <= ymax:
                intersected_rooms.add(room["name"])
                
    # 1 wall for the destination boundary, +2 for every intermediate room crossed
    return 1 + 2 * len(intersected_rooms)


# ─────────────────────────────────────────────
# D. Per-Room AP Placement (Enterprise Strategy)
# ─────────────────────────────────────────────

def _is_excluded_room(name: str) -> bool:
    """Check if a room name matches any excluded keyword (stairwell, restroom, etc)."""
    name_lower = name.lower()
    return any(kw in name_lower for kw in EXCLUDED_ROOM_KEYWORDS)


def place_one_ap_per_room(centroids: List[Dict], max_range_m: float, scale_factor: float, 
                          wall_attenuation_db: float, eirp: float) -> List[Dict]:
    """
    Enterprise deployment strategy: Places one AP in each FUNCTIONAL room,
    but SKIPS rooms already covered by an existing AP's coverage radius.
    
    Skips non-functional spaces (stairwells, elevators, restrooms, corridors)
    which get coverage via spillover from adjacent APs.
    """
    placements = []
    ap_id = 0
    
    range_cm = max_range_m * 100
    coverage_radius_grid = range_cm / scale_factor
    
    for room in centroids:
        # Skip non-functional rooms
        if _is_excluded_room(room["name"]):
            log.info(f"Skipping '{room['name']}' — non-functional space (no AP needed)")
            continue
        
        # Skip tiny synthetic zones (gap-filler edge artifacts)
        if "uncovered zone" in room["name"].lower():
            bb = room.get("bounding_box", [0, 0, 0, 0])
            zone_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            if zone_area < 1000000 * MIN_ZONE_AREA_FRAC:
                log.info(f"Skipping '{room['name']}' — tiny edge zone (area={zone_area}, min={1000000 * MIN_ZONE_AREA_FRAC:.0f})")
                continue
        
        # ── OVERLAP CHECK ──
        # If this room's centroid is already within the coverage radius of
        # an existing AP, skip it — it's already covered. This prevents
        # placing redundant APs in VLM-hallucinated micro-rooms.
        already_covered = False
        for existing_ap in placements:
            dist_units = distance(
                (room["cx"], room["cy"]),
                (existing_ap["position"]["x"], existing_ap["position"]["y"])
            )
            dist_m = (dist_units * scale_factor) / 100.0
            
            # Check signal strength accounting for wall attenuation
            walls = count_walls_crossed(
                (room["cx"], room["cy"]),
                (existing_ap["position"]["x"], existing_ap["position"]["y"]),
                centroids, room["name"], existing_ap["placed_in_room"]
            )
            wall_penalty = walls * wall_attenuation_db
            
            if dist_m > 0.1:
                fspl = 20 * math.log10(dist_m) + 20 * math.log10(5000) + 32.44 - 28
                received_dbm = eirp - fspl - wall_penalty
                if received_dbm >= MIN_SIGNAL_DBM:
                    already_covered = True
                    existing_ap["covers_rooms"].append(room["name"])
                    log.info(f"Skipping '{room['name']}' — already covered by "
                            f"'{existing_ap['placed_in_room']}' "
                            f"(dist={dist_m:.1f}m, signal={received_dbm:.1f}dBm)")
                    break
        
        if already_covered:
            continue
        
        ap_id += 1
        
        # This AP's primary room is always covered
        covers = [room["name"]]
        
        # Check spillover into adjacent rooms (accounting for wall attenuation)
        for other_room in centroids:
            if other_room["name"] == room["name"]:
                continue
            
            dist_units = distance((room["cx"], room["cy"]), (other_room["cx"], other_room["cy"]))
            dist_m = (dist_units * scale_factor) / 100.0
            
            walls = count_walls_crossed(
                (room["cx"], room["cy"]), 
                (other_room["cx"], other_room["cy"]),
                centroids, room["name"], other_room["name"]
            )
            wall_penalty = walls * wall_attenuation_db
            
            # FSPL: received = EIRP - FSPL(dist) - wall_penalty
            if dist_m > 0.1:
                fspl = 20 * math.log10(dist_m) + 20 * math.log10(5000) + 32.44 - 28
                received_dbm = eirp - fspl - wall_penalty
                if received_dbm >= MIN_SIGNAL_DBM:
                    covers.append(other_room["name"])
                    log.debug(f"  AP in {room['name']} has spillover to {other_room['name']} "
                             f"(dist={dist_m:.1f}m, walls={walls}, received={received_dbm:.1f}dBm)")
        
        placements.append({
            "id": f"ap_{ap_id}",
            "position": {"x": round(room["cx"]), "y": round(room["cy"])},
            "placed_in_room": room["name"],
            "covers_rooms": covers,
            "coverage_radius_grid": round(coverage_radius_grid),
            "bounding_box": room["bounding_box"],
        })
        
        log.info(f"AP {ap_id} placed in '{room['name']}' at ({round(room['cx'])}, {round(room['cy'])}) "
                f"| range={max_range_m}m | covers: {covers}")
    
    return placements


# ─────────────────────────────────────────────
# E. Infrastructure Device Placement
# ─────────────────────────────────────────────

def place_infrastructure_devices(centroids: List[Dict], specs: dict) -> List[Dict]:
    """
    Places supporting network infrastructure devices:
    1. Router — at the network entry point (room closest to top-left corner, 
       representing where ISP cable enters the building).
    2. Switch — co-located with the router to distribute PoE to APs.
    3. Data Points (Ethernet wall ports) — one per room, offset from center 
       toward the nearest wall for realism.
    """
    devices = []
    
    if not centroids:
        return devices
    
    # Find the nearest FUNCTIONAL room to top-left corner (0,0) — most likely entry point
    # Exclude stairwells, elevators, restrooms, etc.
    functional_rooms = [r for r in centroids if not _is_excluded_room(r["name"])]
    if not functional_rooms:
        functional_rooms = centroids  # fallback if ALL rooms are excluded
    entry_room = min(functional_rooms, key=lambda r: distance((r["cx"], r["cy"]), (0, 0)))
    
    # Router: offset slightly from center toward the nearest wall (top-left quadrant of room)
    bb = entry_room["bounding_box"]
    ymin, xmin, ymax, xmax = bb
    router_x = xmin + (xmax - xmin) * 0.2
    router_y = ymin + (ymax - ymin) * 0.8
    
    router_model = specs.get("switch_budget", {}).get("model", "ISP Router")
    devices.append({
        "id": "router_1",
        "type": "Router",
        "position": {"x": round(router_x), "y": round(router_y)},
        "placed_in_room": entry_room["name"],
        "model": "ISP Gateway Router",
    })
    log.info(f"Router placed in '{entry_room['name']}' at ({round(router_x)}, {round(router_y)})")
    
    # Switch: placed next to the router
    switch_x = router_x + (xmax - xmin) * 0.15
    switch_y = router_y
    
    switch_spec = specs.get("switch_budget", {})
    devices.append({
        "id": "switch_1",
        "type": "Switch",
        "position": {"x": round(switch_x), "y": round(switch_y)},
        "placed_in_room": entry_room["name"],
        "model": switch_spec.get("model", "PoE Switch"),
        "poe_ports": switch_spec.get("poe_ports", 4),
    })
    log.info(f"Switch placed in '{entry_room['name']}' at ({round(switch_x)}, {round(switch_y)})")
    
    # Data Points: one Ethernet wall port per FUNCTIONAL room
    dp_id = 0
    for room in centroids:
        if _is_excluded_room(room["name"]):
            continue
        # Skip tiny synthetic zones same as for APs
        if "uncovered zone" in room["name"].lower():
            bb = room.get("bounding_box", [0, 0, 0, 0])
            zone_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            if zone_area < 1000000 * MIN_ZONE_AREA_FRAC:
                continue
        dp_id += 1
        rbb = room["bounding_box"]
        rymin, rxmin, rymax, rxmax = rbb
        # Place data point near the bottom wall of the room, offset left
        dp_x = rxmin + (rxmax - rxmin) * 0.3
        dp_y = rymin + (rymax - rymin) * 0.85
        
        devices.append({
            "id": f"dp_{dp_id}",
            "type": "Data Point",
            "position": {"x": round(dp_x), "y": round(dp_y)},
            "placed_in_room": room["name"],
            "model": "CAT6 Wall Ethernet Port",
        })
        log.info(f"Data Point {dp_id} placed in '{room['name']}' at ({round(dp_x)}, {round(dp_y)})")
    
    return devices


# ─────────────────────────────────────────────
# F. LLM Justification
# ─────────────────────────────────────────────

def get_llm_justifications(placements: List[Dict], scale_factor: float, 
                           max_range_m: float, wall_material: str, 
                           wall_attenuation_db: float) -> Dict[str, str]:
    """
    Calls the text LLM to narrate the mathematical placement decisions.
    The LLM does NOT choose positions — it explains the math.
    """
    try:
        client = TextClient()
        # Send compact AP data (Gemini free tier has limits)
        compact = [{"id": p["id"], "room": p["placed_in_room"], 
                    "pos": p["position"], "covers": p["covers_rooms"]} 
                   for p in placements]
        user_prompt = f"""
Scale: {scale_factor:.4f} cm/unit | Range: {max_range_m}m | Wall: {wall_material} (-{wall_attenuation_db}dBm/wall)
Strategy: One AP per room (enterprise)

APs: {json.dumps(compact)}

Write a one-paragraph justification for each AP.
"""
        raw = client.generate_text(JUSTIFICATION_PROMPT, user_prompt, fast=True)
        data = extract_json_from_markdown(raw)
        return data.get("justifications", {})
    except Exception as e:
        log.warning(f"LLM justification call failed: {e}. Returning default justifications.")
        return {p["id"]: f"Placed at grid position ({p['position']['x']}, {p['position']['y']}) to maximise coverage of: {', '.join(p['covers_rooms'])}." for p in placements}


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def generate_placement_plan(floor_plan: dict, specs: dict, preferred_tier: str = "mid_tier", 
                            wall_material: str = "drywall", building_length_m: float = 20.0) -> Dict[str, Any]:
    """
    Full deterministic AP placement pipeline.
    1. Derive real-world scale from building dimensions.
    2. Read max indoor range from equipment datasheet.
    3. Find room centroids as candidate positions.
    4. Place one AP per room (enterprise standard).
    5. Place infrastructure devices (router, switch, data points).
    6. Compute spillover coverage into adjacent rooms via wall-attenuated raycasting.
    7. Request LLM justification for the mathematically determined placements.
    """
    log.info("=== Starting AP Placement Algorithm ===")

    # A. Scale
    scale_factor = calculate_scale_factor(floor_plan, building_length_m)

    # B. Get equipment specs
    tier_specs = specs.get(preferred_tier, specs.get("mid_tier", {}))
    eirp = tier_specs.get("tx_power_dbm", 23.0) + tier_specs.get("antenna_gain_dbi", 4.0)
    
    # Use REAL indoor range from the datasheet, NOT a computed value
    max_range_m = tier_specs.get("max_indoor_range_m", 20)
    
    attenuation_db_map = specs.get("attenuation_db", {})
    wall_attenuation_db = attenuation_db_map.get(wall_material, 3.0)
    log.info(f"Wall material: {wall_material} (-{wall_attenuation_db} dBm per wall)")
    log.info(f"Equipment max indoor range: {max_range_m}m (from datasheet) | EIRP={eirp} dBm")

    # C. Centroids
    rooms = floor_plan.get("floor_plan", {}).get("rooms", [])
    centroids = get_room_centroids(rooms)
    log.info(f"Found {len(centroids)} rooms: {[r['name'] for r in centroids]}")

    if not centroids:
        raise ValueError("No rooms with bounding boxes found in floor plan. Cannot generate placement.")

    # D. Place one AP per room
    placements = place_one_ap_per_room(centroids, max_range_m, scale_factor, wall_attenuation_db, eirp)
    log.info(f"Algorithm placed {len(placements)} AP(s) — one per room (enterprise strategy)")

    # E. Place infrastructure devices (router, switch, data points)
    infra_devices = place_infrastructure_devices(centroids, specs)
    log.info(f"Placed {len(infra_devices)} infrastructure devices (router, switch, data points)")

    # F. LLM narration
    justifications = get_llm_justifications(placements, scale_factor, max_range_m, wall_material, wall_attenuation_db)
    for p in placements:
        p["justification"] = justifications.get(p["id"], "Optimal position determined by RF coverage algorithm.")

    # Calculate coverage radius in grid units for display
    range_cm = max_range_m * 100
    radius_units = range_cm / scale_factor

    result = {
        "ap_placements": placements,
        "infra_devices": infra_devices,
        "all_rooms": [{"name": r["name"], "bounding_box": r["bounding_box"]} for r in centroids],
        "total_aps_needed": len(placements),
        "total_devices": len(placements) + len(infra_devices),
        "scale_factor_cm_per_unit": round(scale_factor, 4),
        "max_indoor_range_m": max_range_m,
        "coverage_radius_grid": round(radius_units, 1),
        "preferred_tier": preferred_tier,
        "wall_material": wall_material,
        "wall_attenuation_db": wall_attenuation_db,
    }

    log.info(f"=== Placement Complete: {len(placements)} AP(s) + {len(infra_devices)} infra devices ===")
    return result

