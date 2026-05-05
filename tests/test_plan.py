import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps.parse import extract_floorplan_features
from steps.specs import fetch_and_calculate_specs
from steps.plan import generate_placement_plan

def test_pipeline():
    image_path = "data/images/test1.png"
    
    print("=" * 60)
    print("STEP 1: Extracting floor plan features...")
    print("=" * 60)
    floor_plan = extract_floorplan_features(image_path)
    num_rooms = len(floor_plan.get("floor_plan", {}).get("rooms", []))
    print(f"✅ Extracted {num_rooms} rooms from floor plan\n")

    print("=" * 60)
    print("STEP 2: Fetching equipment specs (RAG tool)...")
    print("=" * 60)
    # We use a placeholder AP count of 1 initially; plan step will determine the real count
    specs = fetch_and_calculate_specs(num_aps=1)
    print(f"✅ Loaded specs for: {list(specs.keys())}\n")

    print("=" * 60)
    print("STEP 3: Running AP Placement Algorithm...")
    print("=" * 60)
    placement_plan = generate_placement_plan(floor_plan, specs, preferred_tier="mid_tier")
    
    print("\n--- Placement Plan Output ---")
    print(json.dumps(placement_plan, indent=2))
    
    print(f"\n✅ Done! {placement_plan['total_aps_needed']} AP(s) required.")
    print(f"   Scale: {placement_plan['scale_factor_cm_per_unit']} cm/unit")
    print(f"   Coverage radius: {placement_plan['coverage_radius_cm']} cm ({placement_plan['coverage_radius_cm']/100:.1f} m)")

if __name__ == "__main__":
    test_pipeline()
