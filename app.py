import streamlit as st
import sys
import os
import json
import time
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page Config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="NetAgent — RF Deployment Engine",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background-color: #0d1117; }

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #161b22, #21262d);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-size: 2.2rem; font-weight: 700; color: #58a6ff; line-height: 1.2; }
.metric-sub   { font-size: 0.78rem; color: #8b949e; margin-top: 6px; letter-spacing: 0.5px; text-transform: uppercase; }

/* Plan cards */
.plan-card {
    background: linear-gradient(145deg, #161b22, #21262d);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 26px;
    margin: 6px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
}
.plan-card.budget  { border-top: 3px solid #3fb950; }
.plan-card.premium { border-top: 3px solid #f78166; }
.plan-card h3      { font-size: 1.1rem; font-weight: 600; margin-bottom: 14px; }
.plan-card p       { font-size: 0.88rem; color: #8b949e; margin: 5px 0; }
.plan-card strong  { color: #e6edf3; }

/* Step status */
.step-ok  { color: #3fb950; font-weight: 600; }
.step-err { color: #f85149; font-weight: 600; }
.step-skp { color: #d29922; font-weight: 600; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #21262d;
}

/* Device count badge */
.device-badge {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #8b949e;
    margin: 3px 3px;
}
.device-badge span { color: #58a6ff; font-weight: 700; }

/* Recommendation box */
.reco-box {
    background: linear-gradient(135deg, #0d419d22, #1f6feb33);
    border: 1px solid #1f6feb;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 10px;
    font-size: 0.9rem;
    color: #cdd9e5;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 NetAgent")
    st.markdown("**AI-Powered RF Deployment Engine**")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Floor Plan",
        type=["png", "jpg", "jpeg"],
        help="Upload a 2D architectural floor plan image"
    )

    st.markdown("### ⚙️ Deployment Settings")

    building_length = st.slider(
        "Building Length (m)", min_value=5, max_value=150, value=40,
        help="Real-world longest dimension of the building"
    )
    budget = st.number_input(
        "Budget ($)", min_value=100, max_value=50000, value=500, step=50
    )
    wall_material = st.selectbox(
        "Wall Material",
        ["drywall", "glass", "brick", "concrete", "wood_door"],
        index=0,
        help="Primary wall construction material"
    )
    preferred_tier = st.selectbox(
        "AP Equipment Tier",
        ["mid_tier", "budget_tier", "premium_tier"],
        index=0
    )

    st.markdown("---")
    run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#484f58; line-height:1.6'>
    <b>Chain:</b><br>
    1 · Gemini Vision → Room extraction<br>
    2 · DuckDuckGo → Equipment specs<br>
    3 · RF Algorithm → AP placement<br>
    4 · Groq Llama → Cost variants<br>
    5 · FSPL engine → Heatmap render<br>
    6 · Groq Llama → Executive report
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 📡 NetAgent — RF Deployment Engine")
st.markdown("*Deterministic Wi-Fi planning powered by VLM vision + FSPL signal simulation + Groq LLM narration*")
st.markdown("---")

if not uploaded_file and not run_btn:
    # ── Landing ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("🏗️", "Upload a floor plan to get started"),
        ("📡", "FSPL-based gradient RF heatmap"),
        ("💰", "Budget vs Premium cost analysis"),
        ("📝", "Groq executive summary report"),
    ]
    for col, (icon, label) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem">{icon}</div>
                <div class="metric-sub">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How the 6-step pipeline works")
    st.markdown("""
| # | Step | Technology | Description |
|---|------|-----------|-------------|
| 1 | **Visual Extraction** | Gemini Vision LLM | Identifies every room and its bounding box from the floor plan image |
| 2 | **Equipment Specs** | DuckDuckGo + web scrape + Groq RAG | Fetches real AP datasheets; tries up to 2 results before using local DB |
| 3 | **AP Placement** | FSPL + raycasting algorithm | Places exactly 1 AP per functional room; skips corridors, bathrooms, tiny zones |
| 4 | **Cost Variants** | Groq Llama 3.3 70B | Generates budget & premium plans with overload detection |
| 5 | **RF Heatmap** | PIL + NumPy signal grid | Renders a continuous Green→Red RSSI gradient over the floor plan |
| 6 | **Executive Summary** | Groq Llama 3.3 70B | Produces a professional markdown deployment report |
    """)

elif run_btn and not uploaded_file:
    st.warning("⚠️ Please upload a floor plan image first.")

elif run_btn and uploaded_file:
    # ── Save upload ───────────────────────────────────────────────────────────
    os.makedirs("data/images", exist_ok=True)
    upload_path = f"data/images/uploaded_{uploaded_file.name}"
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    state = {
        "input": {"floor_plan_path": upload_path, "budget_limit": float(budget)},
        "parsed_layout": None, "equipment_specs": None,
        "placement_plan": None, "variants": None,
        "summary": None, "errors": [],
    }

    progress = st.progress(0, text="Initialising pipeline…")
    status_log = []

    def log_status(step, msg, ok=True):
        icon = "✅" if ok else "❌"
        status_log.append(f"{icon} **Step {step}** — {msg}")

    # ── Step 1 ────────────────────────────────────────────────────────────────
    progress.progress(8, text="Step 1/6 · Extracting rooms with Gemini Vision…")
    try:
        from steps.parse import extract_floorplan_features
        with st.spinner("🔍 Analysing floor plan…"):
            floor_plan = extract_floorplan_features(upload_path)
        state["parsed_layout"] = floor_plan
        rooms = floor_plan.get("floor_plan", {}).get("rooms", [])
        num_rooms = len(rooms)
        log_status(1, f"Extracted **{num_rooms} rooms** from floor plan")
    except Exception as e:
        state["errors"].append(f"Step 1: {e}")
        log_status(1, f"Failed — `{e}`", ok=False)
        st.error(f"❌ Step 1 failed: {e}")
        st.stop()

    # ── Step 2 ────────────────────────────────────────────────────────────────
    progress.progress(20, text="Step 2/6 · Fetching equipment specs via DuckDuckGo…")
    try:
        from steps.specs import fetch_and_calculate_specs
        with st.spinner("🔧 Searching datasheets (trying up to 2 results each)…"):
            specs = fetch_and_calculate_specs(num_aps=1)
        state["equipment_specs"] = specs
        log_status(2, f"Loaded specs for **{len(specs)} equipment categories**")
    except Exception as e:
        state["errors"].append(f"Step 2: {e}")
        log_status(2, f"Failed — `{e}`", ok=False)
        st.error(f"❌ Step 2 failed: {e}")
        st.stop()

    # ── Step 3 ────────────────────────────────────────────────────────────────
    progress.progress(38, text="Step 3/6 · Running RF placement algorithm…")
    try:
        from steps.plan import generate_placement_plan
        with st.spinner("📐 Computing AP positions…"):
            placement_plan = generate_placement_plan(
                state["parsed_layout"], state["equipment_specs"],
                preferred_tier=preferred_tier,
                wall_material=wall_material,
                building_length_m=float(building_length)
            )
        state["placement_plan"] = placement_plan
        num_aps     = placement_plan["total_aps_needed"]
        infra       = placement_plan.get("infra_devices", [])
        num_router  = sum(1 for d in infra if d["type"] == "Router")
        num_switch  = sum(1 for d in infra if d["type"] == "Switch")
        num_dp      = sum(1 for d in infra if d["type"] == "Data Point")
        # Re-fetch specs with correct AP count
        state["equipment_specs"] = fetch_and_calculate_specs(num_aps=num_aps)
        log_status(3, f"Placed **{num_aps} APs** · **{num_router} Router** · "
                      f"**{num_switch} Switch** · **{num_dp} Data Points**")
    except Exception as e:
        state["errors"].append(f"Step 3: {e}")
        log_status(3, f"Failed — `{e}`", ok=False)
        st.error(f"❌ Step 3 failed: {e}")
        st.stop()

    # ── Step 4 ────────────────────────────────────────────────────────────────
    progress.progress(55, text="Step 4/6 · Generating cost variants via Groq…")
    try:
        from steps.variants import generate_variants
        with st.spinner("💰 Computing budget & premium plans…"):
            variants = generate_variants(
                state["placement_plan"], state["equipment_specs"],
                budget_limit=state["input"]["budget_limit"]
            )
        state["variants"] = variants
        bp = variants["budget_plan"]
        pp = variants["premium_plan"]
        log_status(4, f"Budget **${bp['grand_total_usd']:.0f}** · Premium **${pp['grand_total_usd']:.0f}**")
    except Exception as e:
        state["errors"].append(f"Step 4: {e}")
        log_status(4, f"Failed — `{e}`", ok=False)
        st.error(f"❌ Step 4 failed: {e}")
        st.stop()

    # ── Step 5 ────────────────────────────────────────────────────────────────
    progress.progress(72, text="Step 5/6 · Rendering RF heatmap…")
    heatmap_path = None
    try:
        from steps.visualize import generate_heatmap
        with st.spinner("🎨 Computing FSPL signal gradient…"):
            heatmap_path = generate_heatmap(
                upload_path, state["placement_plan"],
                output_path="outputs/heatmap.png"
            )
        log_status(5, "Heatmap rendered successfully")
    except Exception as e:
        state["errors"].append(f"Step 5: {e}")
        log_status(5, f"Failed — `{e}`", ok=False)

    # ── Step 6 ────────────────────────────────────────────────────────────────
    progress.progress(88, text="Step 6/6 · Generating executive summary via Groq…")
    try:
        from tools.groq_summarizer import generate_summary
        with st.spinner("📝 Writing report with Groq Llama 3.3 70B…"):
            summary = generate_summary(state, output_path="outputs/summary_report.md")
        state["summary"] = summary
        log_status(6, f"Executive summary generated ({len(summary):,} chars)")
    except Exception as e:
        state["errors"].append(f"Step 6: {e}")
        log_status(6, f"Skipped (non-critical) — `{e}`", ok=False)
        state["summary"] = None

    progress.progress(100, text="✅ Pipeline complete!")

    # Save report
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/pipeline_report.json", "w") as f:
        json.dump(state, f, indent=2)

    st.markdown("---")

    # ── Pipeline Status ───────────────────────────────────────────────────────
    with st.expander("📋 Pipeline Step Log", expanded=False):
        for line in status_log:
            st.markdown(line)

    # ── Device Count Summary ──────────────────────────────────────────────────
    st.markdown("### Device Deployment Summary")
    infra_devs  = state["placement_plan"].get("infra_devices", [])
    n_router    = sum(1 for d in infra_devs if d["type"] == "Router")
    n_switch    = sum(1 for d in infra_devs if d["type"] == "Switch")
    n_dp        = sum(1 for d in infra_devs if d["type"] == "Data Point")
    n_ap        = state["placement_plan"]["total_aps_needed"]
    n_total     = state["placement_plan"]["total_devices"]
    scale       = state["placement_plan"]["scale_factor_cm_per_unit"]
    range_m     = state["placement_plan"]["max_indoor_range_m"]
    wall_att    = state["placement_plan"]["wall_attenuation_db"]

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [n_ap, n_router, n_switch, n_dp, n_total],
        ["Access Points", "Router", "Switch", "Data Points", "Total Devices"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:12px">
        <span class="device-badge">Scale: <span>{scale:.2f} cm/grid unit</span></span>
        <span class="device-badge">AP Range: <span>{range_m} m</span></span>
        <span class="device-badge">Wall: <span>{wall_material}</span></span>
        <span class="device-badge">Attenuation: <span>-{wall_att} dBm/wall</span></span>
        <span class="device-badge">Rooms: <span>{num_rooms}</span></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📡 RF Heatmap", "💰 Cost Analysis", "📝 Report", "🔧 Raw JSON"]
    )

    # Tab 1 — Heatmap
    with tab1:
        st.markdown("### RF Coverage Heatmap")
        st.caption("Green = excellent (≥−50 dBm) · Yellow = good · Orange = weak · Red = no signal · Grey = excluded zone")
        if heatmap_path and os.path.exists(heatmap_path):
            heatmap_img = Image.open(heatmap_path)
            st.image(heatmap_img, use_container_width=True)
            with open(heatmap_path, "rb") as f:
                st.download_button("📥 Download Heatmap (PNG)", f,
                                   file_name="rf_heatmap.png", mime="image/png")
        else:
            st.error("Heatmap could not be generated.")

    # Tab 2 — Cost
    with tab2:
        st.markdown("### Budget vs Premium Deployment Plans")
        if state.get("variants"):
            bp = state["variants"]["budget_plan"]
            pp = state["variants"]["premium_plan"]

            c1, c2 = st.columns(2)
            with c1:
                overload_b = "⚠️ YES" if bp.get("is_overloaded") else "✅ No"
                lim_b = bp["limitations"][0] if bp.get("limitations") else "None"
                st.markdown(f"""
                <div class="plan-card budget">
                    <h3>💚 Budget Plan</h3>
                    <p><strong>{bp['ap_model']}</strong></p>
                    <p>APs: <strong>{bp['ap_quantity']} × ${bp['ap_unit_price']:.0f}</strong></p>
                    <p>Total Cost: <strong>${bp['grand_total_usd']:.0f}</strong></p>
                    <p>Coverage: <strong>{bp['estimated_coverage_pct']}%</strong></p>
                    <p>Max Users: <strong>{bp['max_concurrent_users']}</strong></p>
                    <p>Throughput/User: <strong>{bp.get('estimated_throughput_mbps_per_user','N/A')} Mbps</strong></p>
                    <p>Overloaded: <strong>{overload_b}</strong></p>
                    <p>Note: <em>{lim_b}</em></p>
                </div>""", unsafe_allow_html=True)

            with c2:
                overload_p = "⚠️ YES" if pp.get("is_overloaded") else "✅ No"
                lim_p = pp["limitations"][0] if pp.get("limitations") else "None"
                st.markdown(f"""
                <div class="plan-card premium">
                    <h3>🔥 Premium Plan</h3>
                    <p><strong>{pp['ap_model']}</strong></p>
                    <p>APs: <strong>{pp['ap_quantity']} × ${pp['ap_unit_price']:.0f}</strong></p>
                    <p>Total Cost: <strong>${pp['grand_total_usd']:.0f}</strong></p>
                    <p>Coverage: <strong>{pp['estimated_coverage_pct']}%</strong></p>
                    <p>Max Users: <strong>{pp['max_concurrent_users']}</strong></p>
                    <p>Throughput/User: <strong>{pp.get('estimated_throughput_mbps_per_user','N/A')} Mbps</strong></p>
                    <p>Overloaded: <strong>{overload_p}</strong></p>
                    <p>Note: <em>{lim_p}</em></p>
                </div>""", unsafe_allow_html=True)

            reco = state["variants"].get("recommendation", "")
            if reco:
                st.markdown(f'<div class="reco-box">💡 <strong>Recommendation:</strong> {reco}</div>',
                            unsafe_allow_html=True)

    # Tab 3 — Report
    with tab3:
        st.markdown("### Executive Summary Report")
        if state.get("summary"):
            st.markdown(state["summary"])
            st.download_button(
                "📥 Download Report (Markdown)", state["summary"],
                file_name="deployment_report.md", mime="text/markdown"
            )
        else:
            st.warning("Summary generation failed (Groq rate limit). Try re-running in 30 seconds.")

    # Tab 4 — Raw JSON
    with tab4:
        st.markdown("### Full Pipeline State (JSON)")
        st.json(state)
        st.download_button(
            "📥 Download JSON", json.dumps(state, indent=2),
            file_name="pipeline_report.json", mime="application/json"
        )
