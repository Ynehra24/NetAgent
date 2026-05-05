# NetAgent: Autonomous RF Network Deployment via Multi-Step LLM Chaining

## 1. Problem Statement

NetAgent solves the problem of automated enterprise Wi-Fi network planning. Given a two-dimensional architectural floor plan image, the system produces a complete deployment blueprint: it identifies every room in the building, determines the optimal placement for wireless access points and supporting infrastructure devices, computes a radio frequency signal coverage heatmap, generates costed budget and premium equipment proposals, and synthesises everything into a professional executive summary report. The system accepts four user inputs — a floor plan image, the physical length of the building in metres, a hard budget ceiling in USD, and the dominant wall construction material — and outputs three artefacts: a structured JSON pipeline report, a visual RF heatmap overlay, and a markdown summary document.

This task genuinely benefits from multi-step chaining rather than a single prompt for several fundamental reasons. First, the problem requires multimodal input processing: the floor plan is an image, and no text-only model can extract room geometries from pixels. This necessitates a dedicated vision model step before any downstream reasoning can begin. Second, the placement of access points depends on physical signal propagation mathematics — specifically the Free Space Path Loss equation and wall attenuation raycasting — which a language model cannot execute with the precision required by RF engineers. These calculations must run in deterministic Python code, not inside a probabilistic token generator. Third, the system requires current, real-world equipment specifications and market pricing. A language model's training data is static and will hallucinate prices and antenna gain values if asked to recall them from memory. This necessitates an external tool call to retrieve live data from the web. Fourth, the final output requires synthesis across all prior stages — combining visual geometry, physics calculations, real-world prices, and placement corrections into a coherent narrative — which is precisely where a language model excels. No single prompt can perform multimodal extraction, deterministic physics, live web retrieval, and professional writing simultaneously; the chain structure allows each capability to be delegated to the component best suited for it.

## 2. Chain Design

The pipeline comprises six sequential steps connected by a shared Python dictionary called `state`. Each step reads from this dictionary and writes its results back into it, ensuring that every subsequent step has access to the cumulative knowledge the agent has gathered. At the end of execution, this dictionary is serialised to disk as the structured JSON output.

Step 1, Visual Floor Plan Extraction, receives the raw floor plan image and sends it to the Gemini 2.5 Flash vision model with a prompt that demands structured JSON output on a normalised 0–1000 coordinate grid. The model returns a list of room names and bounding boxes. This step exists as a separate stage because it is the only step that requires multimodal processing; no other step in the chain needs to see the image. After the model returns its output, a Python-side sanitisation algorithm runs three passes: it drops rooms whose area falls below one percent of the detected building footprint to eliminate hallucinated micro-zones, it removes duplicate rooms that overlap by more than sixty percent using Intersection-over-Union calculations, and it scans for uncovered rectangular gaps in the floor plan to generate synthetic zones that the vision model missed. This post-processing is critical because vision models routinely hallucinate two to five spurious rooms per floor plan, and downstream physics calculations would waste access points on non-existent spaces without this filter.

Step 2, Equipment Specification Retrieval, is the first external tool integration. It invokes the DuckDuckGo search API to find manufacturer datasheets for three tiers of networking hardware (TP-Link EAP225, Ubiquiti U6 Lite, and Ubiquiti U6 Pro), scrapes the HTML of the top search results using BeautifulSoup, and passes the raw scraped text to Gemini Flash, which extracts structured JSON fields: power draw in watts, maximum concurrent users, antenna gain in dBi, and base price in USD. This step is separated from Step 1 because it operates on entirely different input (web search results, not images) and uses an entirely different tool (HTTP scraping, not vision). If the web search fails — because DuckDuckGo returns PDF links or JavaScript-rendered pages that cannot be scraped — the system falls back to a local JSON database of manually curated specifications, ensuring the chain never breaks at this stage.

Step 3, Access Point Placement, is the computational core of the pipeline. It receives the room geometries from Step 1 and the antenna specifications from Step 2. It first calculates a scale factor that maps the abstract 0–1000 grid to real-world centimetres using the user's building length input. It then iterates through every functional room, skipping non-essential spaces such as restrooms, stairwells, and corridors using a keyword exclusion list. For each candidate room, it checks whether the room is already covered by an existing access point by computing the Free Space Path Loss at 5 GHz between the candidate room's centroid and every previously placed AP, accounting for wall crossings detected via a ten-point ray sampling algorithm. If the received signal strength at the candidate room exceeds negative seventy dBm (the standard minimum for reliable Wi-Fi), the room is marked as covered by spillover and no additional AP is placed. Otherwise, a new AP is placed at the room's geometric centre. This step also places infrastructure devices — a router near the building entry point, a PoE switch adjacent to the router, and one CAT6 data point per functional room. After all devices are positioned, Gemini Flash is called to generate human-readable justifications explaining the mathematical reasoning behind each placement. This step must be separate because it performs deterministic physics that an LLM cannot reliably execute, and because it requires the structured outputs of both Step 1 and Step 2 as inputs.

Step 3.5, Placement Validation, is the second external tool integration. It searches DuckDuckGo for enterprise Wi-Fi installation best practices, scrapes the resulting articles, and passes the combined text along with the current device coordinates to Gemini Pro. The model evaluates each device against the scraped guidelines and flags any violations, suggesting corrected coordinates. The Python engine then clamps these corrections to ensure they remain within the room's bounding box before overwriting the placement plan. This step is separated from Step 3 because it introduces an independent information source (web-scraped guidelines) that the placement algorithm has no access to, and it acts as an autonomous quality assurance layer that audits the mathematical placements against real-world installation wisdom.

Step 4, Cost Variant Generation, receives the finalised device counts and the equipment pricing from Step 2. It passes a compact payload to Gemini Pro, which calculates line-item costs for budget-tier and premium-tier deployment scenarios, including access point unit costs, switch costs, and cabling estimates at fifteen dollars per metre. Critically, while the LLM generates a textual recommendation, the Python engine overrides it with deterministic logic: if the premium plan fits within the user's budget, it is always recommended; if only the budget plan fits, that is selected; if neither fits, the cheapest option is chosen. This override exists because language models frequently make arithmetic errors or ignore hard constraints when generating financial recommendations.

Step 5, RF Heatmap Visualisation, receives the placement coordinates from Step 3 and the original floor plan image. It computes a two-dimensional signal strength grid at quarter resolution by calculating, for every pixel, the maximum received signal strength from any access point using FSPL with wall attenuation penalties. It then maps the resulting dBm values to a green-to-red colour gradient, composites it over the original floor plan image with transparency, draws device markers and labels, and appends a legend bar. This step contains no LLM calls whatsoever — it is pure NumPy and Pillow computation.

Step 6, Executive Summary Generation, receives the entire accumulated state dictionary and passes a trimmed version to Gemini Pro with instructions to produce a professional markdown report covering eight sections: executive summary, floor plan analysis, access point placement, signal coverage, infrastructure devices, cost analysis, recommendation, and technical specifications. This is the synthesis step where the LLM does what it does best — transforming a complex, multi-source data structure into coherent prose that a non-technical stakeholder can understand.

## 3. Tool Integration

NetAgent integrates the DuckDuckGo search API as its external tool, invoked at two distinct points in the chain for two entirely different purposes. The first invocation occurs in Step 2, where the tool retrieves current hardware datasheets. The second occurs in Step 3.5, where it retrieves industry best-practice guidelines for device placement. DuckDuckGo was chosen because it provides a free, unauthenticated search API that does not require API keys, making the system immediately runnable by any evaluator without additional setup. The search results are processed through a Retrieval-Augmented Generation pipeline: the system downloads the HTML of the top two results, strips away navigation, scripts, and styling using BeautifulSoup, truncates the cleaned text to four thousand characters to stay within the LLM's context window, and passes it to Gemini Flash with a structured extraction prompt. The LLM returns a JSON object with exact numerical fields, which is written directly into the shared state dictionary. This structured output is then consumed by the deterministic placement algorithm, which reads the antenna gain value to compute EIRP and the base price to enforce budget constraints. If the tool fails at any point — no search results, blocked URLs, empty scraped text, or LLM parsing errors — the system falls back to a local equipment database containing manually verified specifications, ensuring that the chain proceeds without interruption.

## 4. Limitations

The most significant limitation is the brittleness of the vision extraction step. Floor plans with heavy annotations, furniture symbols, or non-standard drawing conventions cause the vision model to either hallucinate rooms that do not exist or merge distinct physical spaces into single oversized bounding boxes. While the Python sanitiser catches mathematically impossible outputs — rooms smaller than one percent of the building footprint or rooms that overlap by more than sixty percent — it cannot detect semantically incorrect room labels or walls that the model failed to see entirely. A floor plan where two rooms share a boundary that the model misses will result in a single large bounding box, causing one access point to be placed where two were needed, producing a coverage dead zone that propagates into the heatmap and the cost analysis.

The web scraping tool is also unreliable. DuckDuckGo search results frequently link to PDF datasheets that BeautifulSoup cannot parse, or to JavaScript-heavy pages that render no text in a static HTTP response. When both top results fail, the system falls back to its local database, which may contain outdated pricing. Additionally, the wall attenuation model uses rectangular bounding-box intersection rather than true polygon geometry, meaning that L-shaped rooms, curved walls, and open-plan spaces are poorly represented. The raycasting algorithm samples only ten points along a line between two rooms, which can miss narrow walls or doorways.

## 5. Reflection

If I had more time, the highest-impact change would be replacing the bounding-box room representation with true polygon geometry. This would allow the raycasting algorithm to trace signal paths through actual wall segments rather than rectangular approximations, dramatically improving the accuracy of the FSPL calculations and the resulting heatmap. I would also refactor the pipeline's linear execution into a directed acyclic graph, running Step 1 (vision extraction) and Step 2 (equipment fetching) in parallel since they have no dependency on each other, which would halve the pipeline's wall-clock time for those stages.

I used Gemini (via Google's Antigravity coding assistant) extensively during development for debugging API integration issues, generating boilerplate for the Streamlit interface, and iterating on prompt designs. The assistant was particularly helpful in identifying that the Groq API's free-tier rate limits (100K tokens per day) were insufficient for full pipeline runs, which led to the decision to migrate entirely to the Gemini API. However, the chain architecture, the choice to use FSPL physics instead of LLM-generated placement guesses, the sanitiser algorithm design, and the deterministic budget override logic were my own engineering decisions that the assistant implemented under my direction.

---

# Appendix: Prompt Design & Engineering

This section details the system and user prompts used across the 6 LLM nodes in the NetAgent chain. For each step, the prompt constraints are explained, along with how the subsequent steps rely on that specific structure.

---

## Step 1 — Vision Extraction (Gemini 2.5 Flash)
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

**Why it's shaped this way:**
The most critical constraint here is the **0 to 1000 normalized grid**. Since users upload images of vastly different resolutions (e.g., 4K renders vs low-res screenshots), pixel coordinates are meaningless for physical distance calculations. By forcing the LLM to normalize to a 1000x1000 grid, Step 3 (the placement algorithm) can reliably map these coordinates to real-world metres using the user's `building_length_m` input. The explicit exclusion of "doors, windows, or furniture" was added after testing revealed that the vision model would detect desks, doors, and windows as individual rooms, producing dozens of spurious micro-zones that wasted access points.

**Dependency on next step:** Step 3 reads each room's `bounding_box` array directly. If the coordinates were in pixel space instead of the normalised grid, the scale factor calculation would produce incorrect metre-to-grid ratios, causing every AP coverage radius to be wrong.

---

## Step 2 — Equipment Spec Extraction (Gemini 2.5 Flash)
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

**Why it's shaped this way:**
Web scraping returns chaotic, unstructured HTML text containing marketing fluff, cookie banners, and navigation links. The LLM acts as a Retrieval-Augmented Generation (RAG) extractor, plucking out just the physical integer/float values from the noise. The "estimate if not found" instruction is a safety net: some datasheets omit antenna gain or user counts, and a missing value would crash the FSPL calculation in Step 3.

**Dependency on next step:** Step 3 reads `antenna_gain_dbi` directly to compute Effective Isotropic Radiated Power (EIRP = tx_power + antenna_gain). Step 4 reads `base_price_usd` to calculate total deployment cost. If the LLM returned prose instead of JSON, `json.loads()` would throw an exception and the chain would halt.

---

## Step 3 — Placement Justification (Gemini 2.5 Flash)
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

**Why it's shaped this way:**
The LLM is explicitly told that it is narrating *existing mathematical output*, not inventing its own placements. This is a deliberate design choice: if we allowed the LLM to choose AP positions, it would guess rather than calculate FSPL. By framing the prompt as "explain the math that already happened," we get professional-sounding narration without sacrificing the deterministic accuracy of the placement algorithm.

**Dependency on next step:** The justification strings are stored in the state dictionary and passed to Step 6 (Executive Summary), which stitches them into the final report's AP Placement section.

---

## Step 4 — Placement Validation (Gemini 2.5 Pro)
**File:** `steps/validate.py`

**System Prompt:**
> You are a Certified Wireless Network Expert (CWNE) reviewing network device placements against industry best practices.
> You are given: (a) the current device placements, (b) the room bounding boxes, and (c) raw text scraped from industry best-practice articles.
> For each device, evaluate whether its placement follows the scraped guidelines. If it violates a guideline, flag it with status "flagged", describe the specific issue, and suggest a corrected (x, y) position that is INSIDE the device's room bounding box.
> Output JSON: `{"validation_results": [{"device_id": "string", "status": "ok|flagged", "issue": "string", "suggested_position": {"x": int, "y": int}}]}`

**User Prompt:**
> Room Data: {rooms_with_bounding_boxes}
> Current Placements: {all_devices_with_coordinates}
> Scale: {scale_factor} cm/unit | Wall material: {wall_material}
> Industry Best Practices (scraped from web): {scraped_guidelines_text}

**Why it's shaped this way:**
This is the second external tool call in the chain. It acts as an autonomous QA agent that audits the deterministic placements against real-world wisdom that the math engine has no access to. The constraint forces the LLM to output numerical `(x, y)` corrections rather than vague advice like "move it closer to the wall." The Python engine then clamps these coordinates to stay within the room's bounding box, acting as a safety net against LLM hallucinations that place devices outside the building.

**Dependency on next step:** Step 5 (Heatmap) renders whichever positions are in the state dictionary. If validation corrected a device's coordinates, the heatmap automatically reflects the corrected position without any additional logic.

---

## Step 5 — Cost Variants (Gemini 2.5 Pro)
**File:** `steps/variants.py`

**System Prompt:**
> You are a Certified Wireless Network Expert (CWNE) working as a procurement specialist. Create two deployment variants: Budget and Premium.
> Use ONLY the equipment models and prices provided below. Do NOT invent or substitute models.
> Calculate: AP cost = base_price × quantity. Switch cost = base_price. Cabling = $15/m × (3m per AP + 5m to switch). Grand total = AP + switch + cabling.
> Check overload: if total_concurrent_users > max_users_per_ap × ap_quantity, set is_overloaded: true.
> Return strictly JSON: `{"budget_plan": {...}, "premium_plan": {...}, "recommendation": "string"}`

**User Prompt:**
> Number of APs: {num_aps}
> Rooms covered: {rooms_list}
> Expected users: {total_users}
> Budget limit: ${budget_limit}
> Equipment specs: {compact_specs_json}

**Why it's shaped this way:**
This prompt forces the LLM to behave like a spreadsheet, calculating line-item costs with explicit formulas provided in the prompt itself. The "Do NOT invent or substitute models" instruction was added after testing showed the LLM would sometimes recommend Cisco Meraki or Aruba devices that were never in the equipment database, producing prices that contradicted the rest of the pipeline.

**Dependency on next step:** While the LLM generates a text `recommendation`, the Python engine intercepts the output and applies a deterministic "best within budget" logic override: if the premium plan fits within the budget, it is always selected; otherwise budget is chosen. This ensures the recommendation never violates the user's hard budget constraint, regardless of what the LLM says.

---

## Step 6 — Executive Summary (Gemini 2.5 Pro)
**File:** `tools/summarizer.py`

**System Prompt:**
> You are a professional network engineer writing an executive summary report.
> Given a complete RF deployment pipeline output (JSON), produce a clear, well-structured markdown report that a client or professor can read and understand immediately.
> The report MUST include these sections with proper markdown formatting:
> 1. Executive Summary, 2. Floor Plan Analysis, 3. Access Point Placement, 4. Signal Coverage Analysis, 5. Infrastructure Devices, 6. Cost Analysis (with comparison table), 7. Recommendation, 8. Technical Specifications.
> Use tables, bold text, and clear formatting. Be concise but thorough.
> Do NOT output JSON. Output ONLY clean markdown.

**User Prompt:**
> Here is the complete deployment pipeline output:
> {focused_pipeline_state_json}

**Why it's shaped this way:**
This is the only prompt in the entire chain that explicitly asks for Markdown instead of JSON. Every prior step demanded strict JSON because the Python engine needs machine-parseable structured data. This final step reverses that constraint because its output is consumed directly by humans — it is rendered in the Streamlit UI's "Report" tab and saved as `summary_report.md`. The "focused_pipeline_state" is a trimmed version of the full state dictionary that strips verbose fields (raycasting logs, raw bounding boxes) to stay within the LLM's context window.

---

## Prompts Changed After Testing

### Change 1: Vision Hallucination Fix (Step 1)

**Original Prompt:** *"List the rooms in this floor plan and give me their coordinates."*

**What went wrong:** The Vision model would frequently detect desks, tables, and doors as individual "rooms," producing anywhere from 15 to 40 spurious zones for a floor plan that contained only 8 real rooms. It would also hallucinate rooms floating outside the building boundaries. Furthermore, its coordinates were tied to pixel resolution, meaning a 4K image and a 720p image of the same floor plan produced completely incompatible numbers that broke the scale factor calculation.

**What we changed:** We rewrote the prompt to enforce `[ymin, xmin, ymax, xmax]` bounding box format, added the strict rule *"Do not include doors, windows, or furniture. Only structural rooms,"* and mandated the 0–1000 normalised grid. Despite this prompt engineering, VLMs still hallucinated occasionally, which forced us to build a three-pass Python-side sanitiser: (1) drop rooms below 1% of building area, (2) deduplicate rooms with >60% IoU overlap, (3) scan for uncovered gaps and generate synthetic zones. This combination of prompt constraints plus programmatic post-processing reduced false rooms from ~30 per plan to 0–2.

### Change 2: Context Window Optimisation (Steps 3, 4, and 6)

**Original Prompt:** *"Here is the deployment state: {json.dumps(state)}"*

**What went wrong:** Initially, we passed the entire shared `state` dictionary to the LLM for the later steps. As the pipeline progressed, the state accumulated massive amounts of data — including 100-point raycasting logs, full bounding box arrays, and verbose justification paragraphs. This caused the API requests to fail with `413 Payload Too Large` errors and triggered aggressive rate limiting.

**What we changed:** We rewrote the user prompts in Steps 3, 4, and 6 to construct "compacted" payloads. For example, in the Cost Variants step, instead of sending the full AP data (which included raycasting results, spillover calculations, and justification text), we mapped it to a minimal format: `ap_summary = [{"id": ap["id"], "room": ap["placed_in_room"]} for ap in placements]`. In the Summary step, we built a `focused_report` dictionary that included only room names, AP positions, infrastructure device types, scale factor, and variant costs — stripping everything else. This reduced the prompt payload by approximately 70% while giving the LLM exactly the data it needed.
