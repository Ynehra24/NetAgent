from typing import Dict, Any, List
from llm.client import VisionClient
from llm.parse import extract_json_from_markdown
from basefiles.logger import get_logger
import numpy as np
from PIL import Image

logger = get_logger(__name__) if 'get_logger' in globals() else None

SYSTEM_PROMPT = """
You are an expert architect analyzing a floor plan image.
Map the floor plan to a normalized 2D grid: [0,0] = Top-Left, [1000,1000] = Bottom-Right.

CRITICAL RULES:
- Identify EVERY distinct room/zone across the ENTIRE floor plan.
- You MUST cover ALL four quadrants of the image (top-left, top-right, bottom-left, bottom-right).
  If you only find rooms in the top half, look again — the bottom half has rooms too.
- Include offices, meeting rooms, corridors, bathrooms, kitchens, bedrooms, closets, 
  stairwells, lobbies, open office areas — EVERYTHING.
- Each room bounding_box is [ymin, xmin, ymax, xmax] on the 0-1000 grid.
- Bounding boxes must cover the FULL wall-to-wall area of each room.
- Keep the response COMPACT. Only name and bounding_box per room. No objects, no doors.
- Output ONLY valid JSON. No prose. No markdown fences.

Schema:
{"floor_plan":{"rooms":[{"name":"string","bounding_box":[ymin,xmin,ymax,xmax]}]}}
"""


# ─────────────────────────────────────────────
# Image-Based Building Boundary Detection
# ─────────────────────────────────────────────

def _detect_building_bounds(image_path: str) -> tuple:
    """
    Detect the building content boundary directly from image pixels.
    
    Returns (ymin, xmin, ymax, xmax) on the 0-1000 grid.
    
    This is the GROUND TRUTH footprint — it detects where the actual drawing
    content is (walls, text, furniture) vs. white padding. It adapts
    automatically to any image size, any building size, any amount of padding.
    """
    img = Image.open(image_path).convert('L')  # grayscale
    w, h = img.size
    
    pixels = np.array(img)
    
    # Find non-white pixels (anything darker than 240/255 is building content)
    content_mask = pixels < 240
    
    if not content_mask.any():
        if logger:
            logger.warning("No building content detected in image — using full grid")
        return (0, 0, 1000, 1000)
    
    # Find the bounding box of all content pixels
    rows_with_content = np.any(content_mask, axis=1)
    cols_with_content = np.any(content_mask, axis=0)
    
    y_min_px = int(np.argmax(rows_with_content))
    y_max_px = int(h - np.argmax(rows_with_content[::-1]))
    x_min_px = int(np.argmax(cols_with_content))
    x_max_px = int(w - np.argmax(cols_with_content[::-1]))
    
    # Map pixel coordinates to the 0-1000 grid
    ymin = int(y_min_px / h * 1000)
    xmin = int(x_min_px / w * 1000)
    ymax = int(y_max_px / h * 1000)
    xmax = int(x_max_px / w * 1000)
    
    if logger:
        logger.info(f"Image analysis: building content at "
                     f"pixels y=[{y_min_px},{y_max_px}] x=[{x_min_px},{x_max_px}] "
                     f"({w}×{h}px) → grid y=[{ymin},{ymax}] x=[{xmin},{xmax}]")
    
    return (ymin, xmin, ymax, xmax)


# ─────────────────────────────────────────────
# VLM Hardening: Room Sanitization
# ─────────────────────────────────────────────

# RELATIVE thresholds — min_area is computed dynamically from
# the actual building footprint, not from a fixed grid.
MIN_ROOM_AREA_FRAC = 0.01   # 1% of BUILDING footprint area
MAX_ASPECT_RATIO   = 12.0   # reject rooms narrower than 1:12
MAX_IOU_OVERLAP    = 0.60   # deduplicate rooms with >60% overlap


def _room_area(bb: List) -> float:
    """Area of a bounding box on the 0-1000 grid."""
    ymin, xmin, ymax, xmax = bb
    return max(0, ymax - ymin) * max(0, xmax - xmin)


def _iou(bb1: List, bb2: List) -> float:
    """Intersection-over-Union of two bounding boxes."""
    ymin = max(bb1[0], bb2[0])
    xmin = max(bb1[1], bb2[1])
    ymax = min(bb1[2], bb2[2])
    xmax = min(bb1[3], bb2[3])
    inter = max(0, ymax - ymin) * max(0, xmax - xmin)
    a1 = _room_area(bb1)
    a2 = _room_area(bb2)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def _sanitize_rooms(rooms: List[Dict], building_bounds: tuple = None) -> List[Dict]:
    """
    Post-process VLM room list to remove hallucinated entries:
      1. Drop rooms with no valid bounding box.
      2. Drop rooms whose area is below MIN_ROOM_AREA_FRAC of the BUILDING footprint.
      3. Drop rooms with extreme aspect ratios (hallucinated strips).
      4. Deduplicate: if two rooms overlap >MAX_IOU_OVERLAP, keep the larger one.
    
    The min_area threshold ADAPTS to the building size:
      - Large building filling the frame → larger min_area
      - Small building with lots of padding → smaller min_area (preserves small rooms)
    """
    if not rooms:
        return rooms

    # Compute min_area relative to BUILDING footprint, not fixed grid
    if building_bounds:
        by1, bx1, by2, bx2 = building_bounds
        footprint_area = (by2 - by1) * (bx2 - bx1)
    else:
        footprint_area = 1000 * 1000  # fallback: full grid
    
    min_area = footprint_area * MIN_ROOM_AREA_FRAC
    
    if logger:
        logger.debug(f"Sanitizer min_area: {min_area:.0f} "
                      f"(1% of footprint {footprint_area:.0f})")

    # --- Pass 1: filter by geometry ---
    valid = []
    dropped_tiny = 0
    dropped_aspect = 0

    for room in rooms:
        bb = room.get("bounding_box")
        if not bb or len(bb) != 4:
            continue

        ymin, xmin, ymax, xmax = bb
        h = ymax - ymin
        w = xmax - xmin

        # Too small?
        area = h * w
        if area < min_area:
            dropped_tiny += 1
            if logger:
                logger.debug(f"Dropped '{room.get('name')}' — area {area:.0f} < min {min_area:.0f}")
            continue

        # Extreme aspect ratio? (hallucinated thin strips)
        if h > 0 and w > 0:
            ratio = max(h / w, w / h)
            if ratio > MAX_ASPECT_RATIO:
                dropped_aspect += 1
                if logger:
                    logger.debug(f"Dropped '{room.get('name')}' — aspect ratio {ratio:.1f} > {MAX_ASPECT_RATIO}")
                continue

        valid.append(room)

    if dropped_tiny or dropped_aspect:
        if logger:
            logger.info(f"Sanitizer: dropped {dropped_tiny} tiny + {dropped_aspect} extreme-ratio rooms")

    # --- Pass 2: deduplicate overlapping rooms ---
    valid.sort(key=lambda r: _room_area(r["bounding_box"]), reverse=True)
    deduped = []
    dropped_dup = 0

    for room in valid:
        bb = room["bounding_box"]
        is_dup = False
        for kept in deduped:
            if _iou(bb, kept["bounding_box"]) > MAX_IOU_OVERLAP:
                is_dup = True
                dropped_dup += 1
                if logger:
                    logger.debug(f"Dropped '{room.get('name')}' — duplicate of '{kept.get('name')}' "
                                 f"(IoU={_iou(bb, kept['bounding_box']):.2f})")
                break
        if not is_dup:
            deduped.append(room)

    if dropped_dup:
        if logger:
            logger.info(f"Sanitizer: deduplicated {dropped_dup} overlapping rooms")

    if logger:
        logger.info(f"Sanitizer: {len(rooms)} raw → {len(deduped)} valid rooms")

    return deduped


# ─────────────────────────────────────────────
# VLM Hardening: Coverage Gap Filler
# ─────────────────────────────────────────────

def _compute_footprint(rooms: List[Dict]) -> tuple:
    """
    Compute the building footprint (bounding box of ALL rooms) on the 0-1000 grid.
    Returns (ymin, xmin, ymax, xmax) or None if no valid rooms.
    """
    valid = [r for r in rooms if r.get("bounding_box") and len(r["bounding_box"]) == 4]
    if not valid:
        return None
    fp_ymin = min(r["bounding_box"][0] for r in valid)
    fp_xmin = min(r["bounding_box"][1] for r in valid)
    fp_ymax = max(r["bounding_box"][2] for r in valid)
    fp_xmax = max(r["bounding_box"][3] for r in valid)
    return (fp_ymin, fp_xmin, fp_ymax, fp_xmax)


def _fill_coverage_gaps(rooms: List[Dict], building_bounds: tuple = None,
                        min_gap_cells: int = 2) -> List[Dict]:
    """
    Detect uncovered rectangular regions and create synthetic 'Uncovered Zone' rooms.
    
    Uses the IMAGE-DERIVED building bounds as the authoritative footprint.
    This means:
      - The footprint matches the actual drawing content, not VLM guesses.
      - White image padding is NEVER included, regardless of image size.
      - Works identically whether the building fills the frame or has huge borders.
    
    Each synthetic zone must be ADJACENT to at least one existing room on the
    raster grid, preventing L-shaped corners and isolated artifacts.
    """
    if not rooms:
        return rooms

    # Use image-derived bounds if provided, otherwise fall back to room bounds
    if building_bounds:
        fp_ymin, fp_xmin, fp_ymax, fp_xmax = building_bounds
    else:
        footprint = _compute_footprint(rooms)
        if footprint is None:
            return rooms
        fp_ymin, fp_xmin, fp_ymax, fp_xmax = footprint

    fp_h = fp_ymax - fp_ymin
    fp_w = fp_xmax - fp_xmin
    if fp_h <= 0 or fp_w <= 0:
        return rooms

    if logger:
        logger.info(f"Gap filler footprint: "
                     f"y=[{fp_ymin},{fp_ymax}] x=[{fp_xmin},{fp_xmax}] "
                     f"({fp_w:.0f}×{fp_h:.0f} units)")

    GRID = 50
    cell_h = fp_h / GRID
    cell_w = fp_w / GRID

    # Rasterize existing room bounding boxes onto the footprint grid
    covered = np.zeros((GRID, GRID), dtype=bool)
    for room in rooms:
        bb = room.get("bounding_box")
        if not bb or len(bb) != 4:
            continue
        ymin, xmin, ymax, xmax = bb
        r1 = max(0, int((ymin - fp_ymin) / cell_h))
        c1 = max(0, int((xmin - fp_xmin) / cell_w))
        r2 = min(GRID, int((ymax - fp_ymin) / cell_h))
        c2 = min(GRID, int((xmax - fp_xmin) / cell_w))
        covered[r1:r2, c1:c2] = True

    total_cells = GRID * GRID
    covered_cells = int(covered.sum())
    coverage_pct = covered_cells / total_cells * 100
    if logger:
        logger.info(f"VLM coverage: {coverage_pct:.0f}% of building footprint "
                     f"({covered_cells}/{total_cells} cells)")

    if coverage_pct >= 95:
        return rooms  # good enough

    # Find uncovered rectangular regions using greedy maximal rectangle scan
    synthetic_rooms = []
    uncovered = ~covered
    zone_id = 0

    while uncovered.any():
        rows, cols = np.where(uncovered)
        if len(rows) == 0:
            break
        r_start, c_start = int(rows[0]), int(cols[0])

        # Expand right
        c_end = c_start
        while c_end + 1 < GRID and uncovered[r_start, c_end + 1]:
            c_end += 1

        # Expand down
        r_end = r_start
        while r_end + 1 < GRID:
            if uncovered[r_end + 1, c_start:c_end + 1].all():
                r_end += 1
            else:
                break

        uncovered[r_start:r_end + 1, c_start:c_end + 1] = False

        area = (r_end - r_start + 1) * (c_end - c_start + 1)
        if area < min_gap_cells:
            continue

        # ── ADJACENCY CHECK ──
        # Only create a synthetic zone if at least one of its border cells
        # is adjacent (4-connected) to a covered cell. This ensures we only
        # fill gaps INSIDE the building, not in isolated corners.
        is_adjacent = False
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                # Only check border cells of the rectangle for performance
                if r > r_start and r < r_end and c > c_start and c < c_end:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID and 0 <= nc < GRID and covered[nr, nc]:
                        is_adjacent = True
                        break
                if is_adjacent:
                    break
            if is_adjacent:
                break

        if not is_adjacent:
            if logger:
                logger.debug(f"Skipping isolated gap at grid [{r_start}:{r_end+1}, "
                             f"{c_start}:{c_end+1}] ({area} cells) — not adjacent to any room")
            continue

        zone_id += 1
        bb = [
            int(fp_ymin + r_start * cell_h),
            int(fp_xmin + c_start * cell_w),
            int(fp_ymin + (r_end + 1) * cell_h),
            int(fp_xmin + (c_end + 1) * cell_w),
        ]
        synthetic_rooms.append({
            "name": f"Uncovered Zone {zone_id}",
            "bounding_box": bb,
        })
        if logger:
            logger.warning(f"Gap: 'Uncovered Zone {zone_id}' bb={bb} ({area} cells)")

    if synthetic_rooms:
        if logger:
            logger.info(f"Added {len(synthetic_rooms)} synthetic zones to fill coverage gaps")
        rooms.extend(synthetic_rooms)

    return rooms


# ─────────────────────────────────────────────
# Main Extraction Function
# ─────────────────────────────────────────────

def extract_floorplan_features(image_path: str) -> Dict[str, Any]:
    """
    Takes a path to a floor plan image, passes it to the VLM, and returns a structured dictionary
    containing the extracted features (rooms, dimensions, objects, walls).
    """
    if logger:
        logger.info(f"Extracting features from floor plan: {image_path}")
    else:
        print(f"Extracting features from floor plan: {image_path}")

    # STEP 0: Detect building bounds from the IMAGE ITSELF.
    # This gives us ground-truth for where the building content is,
    # independent of VLM accuracy. It drives both the sanitizer
    # (adaptive min_area) and the gap filler (authoritative footprint).
    building_bounds = _detect_building_bounds(image_path)

    client = VisionClient()
    
    # Use the system prompt along with a specific request
    prompt = SYSTEM_PROMPT + "\n\nAnalyze the floor plan image. List EVERY room you see."

    try:
        response_text = client.analyze_image(image_path, prompt)
        
        if logger:
            logger.debug(f"Raw VLM Response:\n{response_text}")

        # Parse the JSON from the markdown block
        json_data = extract_json_from_markdown(response_text)
        
        # Validate we got rooms
        rooms = json_data.get("floor_plan", {}).get("rooms", [])
        if logger:
            logger.info(f"VLM identified {len(rooms)} raw rooms")
        
        # HARDENING STEP 1: Sanitize — adaptive min_area based on building size
        rooms = _sanitize_rooms(rooms, building_bounds=building_bounds)
        
        # HARDENING STEP 2: Fill coverage gaps using image-derived footprint
        rooms = _fill_coverage_gaps(rooms, building_bounds=building_bounds)
        
        json_data["floor_plan"]["rooms"] = rooms
        
        if logger:
            logger.info(f"Final room count: {len(rooms)} "
                         f"({[r.get('name', '?') for r in rooms]})")
        
        return json_data

    except Exception as e:
        if logger:
            logger.error(f"Failed to extract floor plan features: {e}")
        else:
            print(f"Failed to extract floor plan features: {e}")
        raise
