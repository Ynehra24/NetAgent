# NetAgent — AI-Powered RF Deployment Engine

An autonomous LLM agent that takes a 2D floor plan image and produces a complete Wi-Fi network deployment plan — including access point placement, infrastructure device positioning, RF coverage heatmaps, and costed equipment recommendations.

Built as a **6-step LLM chain** where each step depends on the output of previous steps, with two external tool integrations (DuckDuckGo web search) and a shared state dictionary that accumulates results across the pipeline.

---

## What It Does

1. **Extracts rooms** from a floor plan image using Gemini Vision
2. **Fetches real equipment specs** via DuckDuckGo search + web scraping + LLM extraction (RAG)
3. **Places access points** using FSPL (Free Space Path Loss) physics — not LLM guesswork
4. **Validates placements** against web-sourced industry best practices (second DDG tool call)
5. **Generates budget vs premium cost plans** with deterministic budget-fit logic
6. **Renders an RF heatmap** showing signal strength as a Green→Red gradient
7. **Writes an executive summary** report in markdown

---

## Chain Architecture

```
Floor Plan Image
      │
      ▼
[Step 1] Gemini Vision → Room extraction + sanitization + gap filling
      │
      ▼
[Step 2] DuckDuckGo Search → Scrape datasheets → Gemini LLM extracts specs (Tool Call #1)
      │
      ▼
[Step 3] FSPL Algorithm → AP placement + router/switch/data points + Gemini justifications
      │
      ▼
[Step 3.5] DuckDuckGo Search → Scrape best practices → Gemini validates positions (Tool Call #2)
      │
      ▼
[Step 4] Gemini LLM → Budget vs Premium cost analysis + Python budget override
      │
      ▼
[Step 5] PIL + NumPy → RF coverage heatmap with signal gradient
      │
      ▼
[Step 6] Gemini LLM → Executive summary markdown report
      │
      ▼
Outputs: pipeline_report.json + heatmap.png + summary_report.md
```

Each step reads from and writes to a shared `state` dictionary. No step can be removed without breaking the chain.

---

## Installation

```bash
git clone https://github.com/Ynehra24/NetAgent.git
cd NetAgent
pip install -r requirements.txt
```

### API Keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
- **Gemini API Key**: Get from [Google AI Studio](https://aistudio.google.com/apikey)

---

## How to Run

### Streamlit Web UI (Recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a floor plan image, configure settings, and click **Run Pipeline**.

### Command Line

```bash
python tests/test_pipeline.py
```

Runs the full 6-step pipeline on the test image and saves outputs to `outputs/`.

---

## Input

- A 2D architectural floor plan image (PNG or JPG)
- Building length in meters (default: 40m for offices)
- Budget limit in USD (default: $1000)
- Wall material (drywall, glass, brick, concrete, wood_door)
- Preferred AP tier (budget, mid, premium)

Test images are provided in `data/images/`.

---

## Output

| File | Description |
|------|-------------|
| `outputs/pipeline_report.json` | Complete structured pipeline state (all rooms, placements, variants, errors) |
| `outputs/heatmap.png` | RF coverage heatmap overlaid on the floor plan |
| `outputs/summary_report.md` | Professional executive summary report |

---

## Project Structure

```
NetAgent/
├── app.py                      # Streamlit web interface
├── basefiles/
│   ├── config.py               # API keys, model IDs, settings
│   └── logger.py               # Centralized logging
├── llm/
│   ├── client.py               # VisionClient + TextClient (Gemini)
│   └── parse.py                # JSON extraction from LLM responses
├── steps/
│   ├── parse.py                # Step 1: VLM room extraction + sanitizer
│   ├── specs.py                # Step 2: Equipment spec orchestrator
│   ├── plan.py                 # Step 3: FSPL placement algorithm
│   ├── validate.py             # Step 3.5: DDG best-practice validation
│   ├── variants.py             # Step 4: Budget vs premium cost analysis
│   └── visualize.py            # Step 5: RF heatmap renderer
├── tools/
│   ├── equipment_fetch.py      # DuckDuckGo search + scrape + LLM RAG
│   └── summarizer.py           # Step 6: Executive summary generation
├── data/
│   ├── equipment_db.json       # Local fallback equipment database
│   └── images/                 # Test floor plan images
├── tests/
│   └── test_pipeline.py        # End-to-end pipeline runner
├── outputs/                    # Generated heatmaps, reports, JSON
├── requirements.txt
└── .env                        # API keys (not committed)
```

---

## Technologies Used

- **Gemini 2.5 Flash/Pro** — Vision model for floor plan image analysis
- **Gemini (2.5 Pro + 2.5 Flash)** — Text LLM for justifications, validation, cost analysis, and summary
- **DuckDuckGo Search** — External tool for real-world equipment specs and placement guidelines
- **BeautifulSoup** — HTML scraping of datasheet pages
- **NumPy + Pillow** — FSPL signal grid computation and heatmap rendering
- **Streamlit** — Web interface

No LangChain, LlamaIndex, or agent frameworks are used. The entire chain is built from scratch.

---

## Error Handling

| Failure | Recovery |
|---------|----------|
| Gemini 503 | Switches to fallback model (gemini-2.5-pro) |
| Gemini 429 (rate limit) | Exponential backoff (5s → 10s → 20s → 40s) |
| DuckDuckGo returns nothing | Falls back to local `equipment_db.json` |
| DuckDuckGo URL is 404 | Tries the next search result |
| LLM returns truncated JSON | Walk-back parser finds last valid sub-object |
| Validation step fails | Non-critical — pipeline continues |
| Summary step fails | Non-critical — JSON + heatmap still produced |

---

## Example

<img width="1470" height="956" alt="Screenshot 2026-05-05 at 11 05 10 PM" src="https://github.com/user-attachments/assets/07808128-38a4-4b6b-8430-bfa7b7c10e9f" />

<img width="1470" height="956" alt="Screenshot 2026-05-05 at 11 06 47 PM" src="https://github.com/user-attachments/assets/685c65f8-ebf3-4ffc-ae64-943f05717250" />

<img width="1470" height="956" alt="Screenshot 2026-05-05 at 11 06 33 PM" src="https://github.com/user-attachments/assets/59cfdc92-8575-48d0-b1fc-0f02e8fab761" />

## Limitations

- **Gemini free tier** has rate limits — heavy testing can exhaust it
- **VLM room detection** is imperfect — the sanitizer catches most hallucinations but some edge cases remain
- **Wall raycasting** uses bounding-box approximation, not actual wall geometry
- **Scale calculation** relies on user-provided building length rather than automatic detection
- **DuckDuckGo** results vary — some datasheets are PDFs that can't be scraped
