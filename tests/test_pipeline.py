import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps.parse import extract_floorplan_features
from steps.specs import fetch_and_calculate_specs
from steps.plan import generate_placement_plan
from steps.variants import generate_variants
from steps.visualize import generate_heatmap
from steps.validate import validate_placement
from tools.summarizer import generate_summary

# ── Shared state dict (accumulates results across all steps) ──────────────────
state = {
    "input": {
        "floor_plan_path": "/Users/yatharthnehva/Desktop/NetAgent/data/images/test2.png",
        "budget_limit": 1000.0,
    },
    "parsed_layout": None,
    "equipment_specs": None,
    "placement_plan": None,
    "variants": None,
    "summary": None,
    "errors": [],
    "step_log": [],
}

def log_step(step, msg):
    entry = f"[{step}] {msg}"
    state["step_log"].append(entry)
    print(entry)

def run_pipeline():
    print("\n" + "=" * 60)
    print("  NetAgent Pipeline — Steps 1-6")
    print("=" * 60)

    # ── STEP 1: Visual Extraction ─────────────────────────────────
    print("\n--- STEP 1: Visual Floor Plan Extraction ---")
    try:
        floor_plan = extract_floorplan_features(state["input"]["floor_plan_path"])
        state["parsed_layout"] = floor_plan
        num_rooms = len(floor_plan.get("floor_plan", {}).get("rooms", []))
        log_step("Step 1", f"Extracted {num_rooms} rooms from floor plan")
    except Exception as e:
        state["errors"].append(f"Step 1 failed: {e}")
        print(f"❌ Step 1 failed: {e}")
        return state

    # ── STEP 2: Equipment Specs (Tool Call) ───────────────────────
    print("\n--- STEP 2: Equipment Spec Fetcher (Tool Call) ---")
    try:
        specs = fetch_and_calculate_specs(num_aps=1)
        state["equipment_specs"] = specs
        log_step("Step 2", f"Loaded specs for: {list(specs.keys())}")
    except Exception as e:
        state["errors"].append(f"Step 2 failed: {e}")
        print(f"❌ Step 2 failed: {e}")
        return state

    # ── STEP 3: AP Placement Algorithm ────────────────────────────
    print("\n--- STEP 3: Algorithmic AP Placement ---")
    try:
        placement_plan = generate_placement_plan(
            state["parsed_layout"], 
            state["equipment_specs"],
            preferred_tier="mid_tier",
            wall_material="concrete",
            building_length_m=40.0  # Office building ~40m long
        )
        state["placement_plan"] = placement_plan
        num_aps = placement_plan["total_aps_needed"]
        scale = placement_plan["scale_factor_cm_per_unit"]
        range_m = placement_plan["max_indoor_range_m"]
        log_step("Step 3", f"Placed {num_aps} AP(s) | Scale: {scale:.2f} cm/unit | Range: {range_m}m (from datasheet)")
        
        # Now recalculate specs with the real AP count
        state["equipment_specs"] = fetch_and_calculate_specs(num_aps=num_aps)
    except Exception as e:
        state["errors"].append(f"Step 3 failed: {e}")
        print(f"❌ Step 3 failed: {e}")
        return state

    # ── STEP 3.5: Validate Placements Against Best Practices ─────
    print("\n--- STEP 3.5: Validating Placements (DDG Best Practices) ---")
    try:
        state["placement_plan"] = validate_placement(state["placement_plan"])
        v = state["placement_plan"].get("validation", {})
        status = v.get("status", "unknown")
        flagged = v.get("flagged_count", 0)
        corrected = v.get("corrections_applied", 0)
        log_step("Step 3.5", f"Validation {status}: {flagged} flagged, {corrected} corrected")
    except Exception as e:
        state["errors"].append(f"Step 3.5 failed: {e}")
        print(f"⚠️ Step 3.5 failed (non-critical): {e}")

    # ── STEP 4: Generate Variants ─────────────────────────────────
    print("\n--- STEP 4: Generate Budget & Premium Variants ---")
    try:
        variants = generate_variants(
            state["placement_plan"],
            state["equipment_specs"],
            budget_limit=state["input"]["budget_limit"]
        )
        state["variants"] = variants
        b = variants["budget_plan"]["grand_total_usd"]
        p = variants["premium_plan"]["grand_total_usd"]
        log_step("Step 4", f"Budget plan: ${b:.0f} | Premium plan: ${p:.0f}")
    except Exception as e:
        state["errors"].append(f"Step 4 failed: {e}")
        print(f"❌ Step 4 failed: {e}")
        return state

    # ── STEP 5: Visualize Heatmap ─────────────────────────────────
    print("\n--- STEP 5: Rendering RF Coverage Heatmap ---")
    try:
        heatmap_path = generate_heatmap(
            state["input"]["floor_plan_path"],
            state["placement_plan"],
            output_path="outputs/heatmap.png"
        )
        log_step("Step 5", f"Heatmap saved to {heatmap_path}")
    except Exception as e:
        state["errors"].append(f"Step 5 failed: {e}")
        print(f"❌ Step 5 failed: {e}")
        return state

    # ── STEP 6: Gemini Summary Report ───────────────────────────────
    print("\n--- STEP 6: Generating Executive Summary (Gemini) ---")
    try:
        summary = generate_summary(state, output_path="outputs/summary_report.md")
        state["summary"] = summary
        log_step("Step 6", f"Executive summary generated ({len(summary)} chars)")
    except Exception as e:
        state["errors"].append(f"Step 6 failed: {e}")
        print(f"⚠️ Step 6 failed (non-critical): {e}")

    # ── Write structured final output ─────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/pipeline_report.json"
    with open(output_path, "w") as f:
        json.dump(state, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ Pipeline complete! Report saved to: {output_path}")
    if state["errors"]:
        print(f"⚠️  {len(state['errors'])} error(s) encountered")
    print("=" * 60)

    # Pretty-print the final variants summary
    if state["variants"]:
        print("\n--- Final Variants Summary ---")
        b = state["variants"]["budget_plan"]
        p = state["variants"]["premium_plan"]
        print(f"\nBudget Plan ({b['ap_model']} × {b['ap_quantity']}):")
        print(f"  Total: ${b['grand_total_usd']:.0f} | Coverage: {b['estimated_coverage_pct']}% | Max AP Users: {b['max_concurrent_users']}")
        print(f"  Throughput/User: {b.get('estimated_throughput_mbps_per_user', 'N/A')} Mbps")
        print(f"  Overloaded: {'YES ⚠️' if b.get('is_overloaded', False) else 'No ✅'}")
        print(f"  Limitations: {b['limitations'][0] if b['limitations'] else 'None'}")
        
        print(f"\nPremium Plan ({p['ap_model']} × {p['ap_quantity']}):")
        print(f"  Total: ${p['grand_total_usd']:.0f} | Coverage: {p['estimated_coverage_pct']}% | Max AP Users: {p['max_concurrent_users']}")
        print(f"  Throughput/User: {p.get('estimated_throughput_mbps_per_user', 'N/A')} Mbps")
        print(f"  Overloaded: {'YES ⚠️' if p.get('is_overloaded', False) else 'No ✅'}")
        print(f"  Limitations: {p['limitations'][0] if p['limitations'] else 'None'}")
        
        print(f"\nRecommendation:\n  {state['variants']['recommendation']}")
    
    return state

if __name__ == "__main__":
    run_pipeline()
