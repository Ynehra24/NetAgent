# Appendix: Prompt Design & Engineering

This section details the system and user prompts used across the 6 LLM nodes in the NetAgent chain. For each step, the prompt constraints are explained, along with how the subsequent steps rely on that specific structure.

---

## 1. Vision Extraction (Gemini 2.5 Flash)
**File:** `steps/parse.py`

**System Prompt:**
> You are an expert architect analyzing a floor plan image.
> 1. Identify all functional rooms, corridors, and bounded areas.
> 2. Output the coordinates of their bounding boxes.
> 3. Use a normalized coordinate system from 0 to 1000 (0,0 is top-left, 1000,1000 is bottom-right).
> 4. Output strictly in JSON format. Do NOT wrap it in markdown fences.
> 5. Do not include doors, windows, or furniture. Only structural rooms.
> Expected JSON schema: `{"floor_plan": {"rooms": [{"name": "string", "bounding_box": [ymin, xmin, ymax, xmax]}]}}`

**User Prompt:**
> *(The raw base64 encoded floor plan image is passed as `inline_data`)*

**Why it’s shaped this way:**
The most critical constraint here is the **0 to 1000 normalized grid**. Since users upload images of vastly different resolutions (e.g., 4K renders vs low-res screenshots), pixel coordinates are meaningless for physical distance calculations. By forcing the LLM to normalize to a 1000x1000 grid, Step 3 (the placement algorithm) can reliably map these coordinates to real-world meters using the user's `building_length_m` input.

---

## 2. Equipment Spec Extraction (Gemini 2.5 Flash)
**File:** `tools/equipment_fetch.py`

**System Prompt:**
> You are a networking hardware engineer. Read the following raw text scraped from a manufacturer's website or datasheet.
> Extract the exact technical specifications for the networking device into JSON.
> If a value is not found, estimate it based on standard industry specs for a device of this tier, but prefer exact numbers from the text.
> Return exactly this JSON schema:
> `{ "power_w": <float>, "max_users": <int>, "frequency": ["<string>"], "antenna_gain_dbi": <float>, "base_price_usd": <float> }`

**User Prompt:**
> Device: {tier}
> Datasheet Text:
> {raw_html_scraped_text}

**Why it’s shaped this way:**
Web scraping returns chaotic, unstructured HTML text containing marketing fluff. We use the LLM to perform Retrieval-Augmented Generation (RAG) to pluck out just the physical integer/float values. 
**Dependency:** Step 3 depends strictly on `antenna_gain_dbi` to calculate the Effective Isotropic Radiated Power (EIRP) for the Free Space Path Loss (FSPL) math. Step 4 relies on `base_price_usd` to calculate budget constraints. If the LLM returns prose instead of JSON, the math functions will instantly crash.

---

## 3. Placement Justification (Gemini 2.5 Flash)
**File:** `steps/plan.py`

**System Prompt:**
> You are a certified wireless network engineer. You have been given the mathematical output of a deterministic RF placement algorithm.
> Write a 1-2 sentence professional justification explaining WHY the AP was placed there based on the data.
> Output ONLY valid JSON matching this schema: `{"<ap_id>": "<justification_string>"}`

**User Prompt:**
> Scale: {scale_factor} cm/unit | Range: {max_range_m}m | Wall: {wall_material}
> Strategy: One AP per room (enterprise)
> APs: {compact_json_of_ap_coordinates_and_coverage}
> Write a one-paragraph justification for each AP.

**Why it’s shaped this way:**
The LLM is explicitly told that it is narrating *existing mathematical output*, not inventing its own placements. 
**Dependency:** This narration is stored in the state dictionary and passed to Step 6, which stitches these justifications into the final executive summary table.

---

## 4. Placement Validation (Gemini 2.5 Pro)
**File:** `steps/validate.py`

**System Prompt:**
> You are a CWNE reviewing network blueprints. You are given a placement plan and raw text scraped from industry best practices.
> Evaluate each device. If a device violates the scraped guidelines, flag it and suggest a new (x,y) coordinate inside its room.
> Output schema:
> `{"validation_results": [{"device_id": "string", "status": "ok|flagged", "issue": "string", "suggested_position": {"x": int, "y": int}}]}`

**User Prompt:**
> Room Data: {rooms}
> Current Placements: {devices}
> Scraped Best Practices: {scraped_text}

**Why it’s shaped this way:**
This is the second external tool call. It acts as an autonomous QA agent. The constraint forces the LLM to output numerical `(x, y)` corrections. 
**Dependency:** The Python engine reads these `suggested_position` coordinates, clamps them mathematically to ensure the LLM didn't accidentally place the device outside the building bounding box, and overwrites the state. Step 5 (Heatmap) will render the corrected positions.

---

## 5. Cost Variants (Gemini 2.5 Pro)
**File:** `steps/variants.py`

**System Prompt:**
> You are a procurement specialist creating a dual-variant deployment proposal: Budget Tier vs Premium Tier.
> Rules: Calculate totals based on AP counts. Assume $15 per cable run. Check if user capacity is overloaded.
> Return strictly JSON with keys "budget_plan", "premium_plan", and "recommendation".

**User Prompt:**
> Placements: {compact_ap_summary}
> Specs: {compact_equipment_specs}
> Budget Limit: ${budget_limit}

**Why it’s shaped this way:**
This forces the LLM to behave like a spreadsheet, calculating line-item costs. 
**Dependency:** While the LLM generates a text `recommendation`, the Python engine intercepts the output and applies a deterministic "best within budget" logic override to ensure the recommendation never violates the user's hard budget limit.

---

## 6. Executive Summary (Gemini 2.5 Pro)
**File:** `tools/summarizer.py`

**System Prompt:**
> You are a professional network engineer writing an executive summary report.
> Given a complete RF deployment pipeline output (JSON), produce a clear, well-structured markdown report.
> MUST include these 8 sections: Executive Summary, Floor Plan Analysis, AP Placement, Coverage Analysis, Infrastructure Devices, Cost Analysis, Recommendation, Technical Specs.
> Do NOT output JSON. Output ONLY clean markdown.

**User Prompt:**
> Here is the complete deployment pipeline output:
> {focused_pipeline_state}

**Why it’s shaped this way:**
This is the only prompt in the entire chain that explicitly asks for Markdown instead of JSON. 
**Dependency:** This output is saved directly as `summary_report.md` and rendered into the Streamlit UI's "Report" tab.

---

## 🛑 Prompts Changed After Testing

### 1. Vision Hallucination Fix (Step 1)
**Original Prompt:** *"List the rooms in this floor plan and give me their coordinates."*
**Why it failed:** The Vision model would frequently detect desks, tables, and doors as individual "rooms". It would also hallucinate rooms floating outside the building boundaries. Furthermore, its coordinates were tied to pixel resolution, meaning high-res and low-res images produced completely incompatible numbers.
**The Fix:** We changed the prompt to enforce `[ymin, xmin, ymax, xmax]`, added the strict rule *"Do not include doors, windows, or furniture. Only structural rooms,"* and mandated the 0-1000 normalized grid. Despite this prompt engineering, VLMs still hallucinated occasionally, forcing us to build a Python-side `sanitizer` algorithm in Step 1 to mathematically drop tiny/overlapping zones.

### 2. Context Window Optimization (Steps 4, 5, and 6)
**Original Prompt:** *"Here is the deployment state: {json.dumps(state)}"*
**Why it failed:** Initially, we passed the entire shared `state` dictionary to the LLM for the later steps. As the pipeline progressed, the state accumulated massive amounts of data (including 100-point raycasting logs and bounding box geometries). This caused the API requests to fail with `413 Payload Too Large` or rate limit errors.
**The Fix:** We updated the User Prompts in Steps 3, 4, and 6 to construct a "compacted" payload. For example, in the Cost Variants step, instead of sending the full AP data, we mapped it to a minimal format: `ap_summary = [{"id": ap["id"], "room": ap["placed_in_room"]} for ap in placements]`. This drastically reduced the token payload while giving the LLM exactly what it needed to do the math.
