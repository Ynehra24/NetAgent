#!/usr/bin/env python3
"""
Smoke tests for the NetAgent pipeline.
Validates that the system produces realistic, non-hallucinated results.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from steps.parse import (extract_floorplan_features, _fill_coverage_gaps, 
                         _sanitize_rooms, _compute_footprint, _detect_building_bounds)
from steps.specs import fetch_and_calculate_specs
from steps.plan import generate_placement_plan, _is_excluded_room
from steps.visualize import generate_heatmap

IMAGE_PATH = "/Users/yatharthnehva/Desktop/NetAgent/data/images/test2.png"
PASS = "✅"
FAIL = "❌"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def test_sanitizer():
    """Test 5: Room sanitizer removes hallucinated rooms."""
    print("\n=== TEST 5: Room Sanitizer ===")
    
    fake_rooms = [
        # Valid large rooms
        {"name": "Office A", "bounding_box": [0, 0, 300, 400]},       # area = 120,000 ✓
        {"name": "Office B", "bounding_box": [0, 400, 300, 800]},     # area = 120,000 ✓
        # Tiny hallucinated room (area = 2,500 = 0.25% of grid → should be dropped)
        {"name": "Speck", "bounding_box": [100, 100, 150, 150]},
        # Extreme aspect ratio strip (5 x 800 = 4,000 but ratio 160:1 → should be dropped)
        {"name": "Strip", "bounding_box": [400, 0, 405, 800]},
        # Very thin but meets area threshold (10 x 1000 = 10,000, ratio 100:1 → dropped)
        {"name": "Thin Line", "bounding_box": [500, 0, 510, 1000]},
        # Duplicate of Office A (high IoU → should be dropped)
        {"name": "Office A Copy", "bounding_box": [10, 10, 290, 390]},
    ]
    
    sanitized = _sanitize_rooms(fake_rooms)
    names = [r["name"] for r in sanitized]
    
    check("Keeps valid rooms", "Office A" in names and "Office B" in names,
          f"kept: {names}")
    check("Drops tiny hallucinated rooms", "Speck" not in names)
    check("Drops extreme aspect ratio strips", "Strip" not in names and "Thin Line" not in names)
    check("Deduplicates overlapping rooms", "Office A Copy" not in names)
    check("Total sanitized count is 2", len(sanitized) == 2, f"got {len(sanitized)}")


def test_gap_filler():
    """Test 4: Gap filler catches missing regions within the building footprint."""
    print("\n=== TEST 4: Coverage Gap Filler ===")
    
    # Simulate VLM detecting rooms only in the top half of a building
    # The building footprint spans y=0-800, x=0-1000
    # But rooms only cover y=0-400 → bottom half (y=400-800) is a gap
    fake_rooms = [
        {"name": "Room A", "bounding_box": [0, 0, 400, 500]},
        {"name": "Room B", "bounding_box": [0, 500, 400, 1000]},
        # A single room in the bottom to establish the footprint extends to y=800
        {"name": "Room C", "bounding_box": [600, 200, 800, 400]},
    ]
    
    filled = _fill_coverage_gaps(fake_rooms.copy())
    synthetic = [r for r in filled if "Uncovered" in r["name"]]
    check("Gap filler creates synthetic zones", len(synthetic) > 0, 
          f"added {len(synthetic)} zones")
    
    # Synthetic zones should stay within the footprint of detected rooms
    all_in_footprint = all(
        z["bounding_box"][0] >= 0 and z["bounding_box"][2] <= 800
        for z in synthetic
    )
    check("Synthetic zones stay within building footprint", all_in_footprint)
    
    # Full coverage should NOT create any zones
    full_rooms = [
        {"name": "Full", "bounding_box": [0, 0, 1000, 1000]},
    ]
    filled_full = _fill_coverage_gaps(full_rooms.copy())
    synthetic_full = [r for r in filled_full if "Uncovered" in r["name"]]
    check("No gaps when fully covered", len(synthetic_full) == 0)
    
    # Bounding boxes should be plain Python ints (not numpy)
    for z in synthetic:
        for val in z["bounding_box"]:
            assert type(val) == int, f"numpy leak: {type(val)}"
    check("No numpy int64 types in output", True)


def test_vlm_extraction():
    """Test 1: VLM extracts rooms covering the full floor plan."""
    print("\n=== TEST 1: VLM Room Extraction ===")
    
    floor_plan = extract_floorplan_features(IMAGE_PATH)
    rooms = floor_plan.get("floor_plan", {}).get("rooms", [])
    
    check("Room count ≥ 5", len(rooms) >= 5, f"got {len(rooms)} rooms")
    
    # Validate every room has a valid bounding box
    valid_bb = 0
    for r in rooms:
        bb = r.get("bounding_box")
        if bb and len(bb) == 4 and all(isinstance(v, (int, float)) for v in bb):
            ymin, xmin, ymax, xmax = bb
            if 0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000:
                valid_bb += 1
    check("All rooms have valid bounding boxes", valid_bb == len(rooms),
          f"{valid_bb}/{len(rooms)} valid")
    
    # Check coverage: rooms should cover at least 70% of the floor plan extent
    import numpy as np
    GRID = 50
    all_ymin = min(r["bounding_box"][0] for r in rooms if r.get("bounding_box"))
    all_ymax = max(r["bounding_box"][2] for r in rooms if r.get("bounding_box"))
    all_xmin = min(r["bounding_box"][1] for r in rooms if r.get("bounding_box"))
    all_xmax = max(r["bounding_box"][3] for r in rooms if r.get("bounding_box"))
    plan_h = all_ymax - all_ymin
    plan_w = all_xmax - all_xmin
    cell_h = plan_h / GRID
    cell_w = plan_w / GRID
    
    covered = np.zeros((GRID, GRID), dtype=bool)
    for r in rooms:
        bb = r.get("bounding_box")
        if not bb:
            continue
        ymin, xmin, ymax, xmax = bb
        r1 = max(0, int((ymin - all_ymin) / cell_h))
        c1 = max(0, int((xmin - all_xmin) / cell_w))
        r2 = min(GRID, int((ymax - all_ymin) / cell_h))
        c2 = min(GRID, int((xmax - all_xmin) / cell_w))
        covered[r1:r2, c1:c2] = True
    
    coverage = covered.sum() / (GRID * GRID) * 100
    check("Coverage ≥ 70% of floor plan", coverage >= 70, f"{coverage:.0f}%")
    
    # No numpy types should leak through
    json_str = json.dumps(floor_plan)  # will throw if int64 present
    check("JSON serializable (no numpy types)", True)
    
    return floor_plan


def test_placement_plan(floor_plan):
    """Test 2: AP placement is realistic and deterministic."""
    print("\n=== TEST 2: AP Placement Validation ===")
    
    specs = fetch_and_calculate_specs(num_aps=1)
    plan = generate_placement_plan(
        floor_plan, specs,
        preferred_tier="mid_tier",
        wall_material="concrete",
        building_length_m=40.0
    )
    
    aps = plan.get("ap_placements", [])
    infra = plan.get("infra_devices", [])
    all_rooms = plan.get("all_rooms", [])
    
    check("APs placed ≥ 3", len(aps) >= 3, f"got {len(aps)} APs")
    check("Infrastructure devices ≥ 3", len(infra) >= 3, 
          f"got {len(infra)} (router+switch+DPs)")
    check("all_rooms included in plan", len(all_rooms) > 0, f"{len(all_rooms)} rooms")
    
    # No AP should be in an excluded room
    excluded_aps = [ap for ap in aps if _is_excluded_room(ap["placed_in_room"])]
    check("No APs in stairwells/elevators/restrooms", len(excluded_aps) == 0,
          f"bad placements: {[a['placed_in_room'] for a in excluded_aps]}" if excluded_aps else "clean")
    
    # Router should NOT be in a stairwell/elevator
    routers = [d for d in infra if d["type"] == "Router"]
    check("Router exists", len(routers) >= 1)
    if routers:
        router_room = routers[0]["placed_in_room"]
        check("Router NOT in stairwell/elevator", not _is_excluded_room(router_room),
              f"placed in '{router_room}'")
    
    # Scale factor should be deterministic (4.0 for 40m building)
    scale = plan.get("scale_factor_cm_per_unit")
    check("Scale factor = 4.0 cm/unit (40m building)", scale == 4.0, f"got {scale}")
    
    # Max range should come from datasheet
    range_m = plan.get("max_indoor_range_m")
    check("Indoor range from datasheet (15-30m)", 15 <= range_m <= 30, f"got {range_m}m")
    
    # All AP positions should be within 0-1000 grid
    all_in_grid = all(
        0 <= ap["position"]["x"] <= 1000 and 0 <= ap["position"]["y"] <= 1000
        for ap in aps
    )
    check("All AP positions within 0-1000 grid", all_in_grid)
    
    # JSON serializable
    json.dumps(plan)
    check("Plan is JSON serializable", True)
    
    return plan, specs


def test_heatmap(plan):
    """Test 3: Heatmap generates successfully."""
    print("\n=== TEST 3: Heatmap Generation ===")
    
    output_path = "outputs/heatmap_smoke.png"
    result_path = generate_heatmap(IMAGE_PATH, plan, output_path)
    
    check("Heatmap file created", os.path.exists(result_path))
    
    # File should be a reasonable size (not empty, not tiny)
    size_kb = os.path.getsize(result_path) / 1024
    check("Heatmap file size > 50KB", size_kb > 50, f"{size_kb:.0f} KB")
    
    # Verify it's a valid PNG
    from PIL import Image
    img = Image.open(result_path)
    w, h = img.size
    check("Image has padding (height > original)", h > 360, f"{w}×{h}")
    
    # Clean up
    os.remove(result_path)


if __name__ == "__main__":
    print("=" * 60)
    print("  NetAgent Smoke Tests")
    print("=" * 60)
    
    test_sanitizer()   # runs fast, no API needed
    test_gap_filler()  # runs fast, no API needed
    
    floor_plan = test_vlm_extraction()
    plan, specs = test_placement_plan(floor_plan)
    test_heatmap(plan)
    
    # Summary
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed")
    if passed == total:
        print(f"  {PASS} ALL SMOKE TESTS PASSED")
    else:
        failed = [name for name, ok in results if not ok]
        print(f"  {FAIL} FAILED: {', '.join(failed)}")
    print(f"{'=' * 60}")
    
    sys.exit(0 if passed == total else 1)

