# ESG Scope 1/2/3 RAG Assistant

A retrieval-augmented generation (RAG) chatbot that extracts and verifies Scope 1, 2, and 3
carbon emissions data from corporate sustainability reports, with citation-level source
tracking and an anti-hallucination check.

Built as a personal/student project using 7 real ESG reports (~590 pages total):
Infosys, Microsoft, Siemens, Siemens Energy, and Siemens Brazil.

> Full methodology, architecture, and (real, measured) results are in
> [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md). This README is just the quickstart.

## Real measured results (not marketing numbers)

From `eval/run_eval.py` over 26 hand-verified ground-truth questions — see
[`eval/eval_report.md`](eval/eval_report.md) and
[`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) §6 for full discussion:

| Metric | Result |
|---|---|
| Mean latency | 1.52s (96.2% of responses under 2s) |
| Retrieval accuracy | 91.3% |
| Answer accuracy | 95.7% |
| Citation consistency | 69.2% |
| Numeric grounding | 80.8% |

## Stack

- **PDF parsing:** PyMuPDF (page-level text extraction)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, free, no API rate limits)
- **Vector store:** ChromaDB (persistent, local), hybrid semantic+keyword re-ranking + neighbor expansion
- **Generation:** Google Gemini (`google-genai` SDK), auto-detects an available "flash" model
  (override with `GEMINI_MODEL=gemini-3.5-flash` for higher citation fidelity at higher latency)
- **Hallucination checks:** citation-consistency + numeric-grounding (two independent, code-level checks)
- **UI:** Streamlit

## Quickstart

```bash
pip install -r requirements.txt
```

Make sure `.env` has your key:

```
GEMINI_API_KEY="your-key-here"
```

1. **Ingest the PDFs into ChromaDB** (one-time, ~a few minutes):
   ```bash
   python -m src.ingest
   ```
2. **Try it from the terminal:**
   ```bash
   python cli.py
   python cli.py --company "Microsoft"
   ```
3. **Extract structured Scope 1/2/3 data:**
   ```bash
   python -m src.emissions_extractor
   ```
4. **Verify DEGREE framework / company targets:**
   ```bash
   python -m src.degree_scorer
   ```
5. **Run the evaluation harness** (real accuracy/latency numbers):
   ```bash
   python -m eval.run_eval
   ```
6. **Launch the web app:**
   ```bash
   streamlit run app.py
   ```

## Project layout

**Start here:** [`notebooks/esg_rag_walkthrough.ipynb`](notebooks/esg_rag_walkthrough.ipynb) —
the full build story with screenshots, renders directly on GitHub.

### The pipeline — [`src/`](src/)

| File | What it does |
|---|---|
| [`config.py`](src/config.py) | Settings and the registry of all 7 reports — filenames, companies, years, chunk size, model names |
| [`ingest.py`](src/ingest.py) | Reads the PDFs page by page, chunks them, embeds locally with MiniLM, stores in ChromaDB |
| [`retriever.py`](src/retriever.py) | Vector search re-ranked with keyword matching (RRF) plus same-page neighbour expansion |
| [`rag_chain.py`](src/rag_chain.py) | Prompt construction, the Gemini call, and both hallucination checks. `ask()` lives here |
| [`emissions_extractor.py`](src/emissions_extractor.py) | Pulls Scope 1/2/3 figures for every company into a CSV |
| [`degree_scorer.py`](src/degree_scorer.py) | Checks each company against its own published targets |

### Interfaces

| File | What it does |
|---|---|
| [`app.py`](app.py) | Streamlit app — chat, company comparison, emissions dashboard, DEGREE scorecard, eval metrics |
| [`cli.py`](cli.py) | Terminal chat loop for quick testing |

### Testing — [`eval/`](eval/)

| File | What it does |
|---|---|
| [`qa_testset.json`](eval/qa_testset.json) | 26 ground-truth questions written by hand from the PDFs, including refusal cases |
| [`run_eval.py`](eval/run_eval.py) | Scores retrieval, answers, both hallucination checks and latency |
| [`eval_report.md`](eval/eval_report.md) | The measured results, per question |
| [`model_tier_comparison.md`](eval/model_tier_comparison.md) | Measured flash vs flash-lite tradeoff |

### Data and docs

| Path | What it is |
|---|---|
| [`data/emissions_summary.csv`](data/emissions_summary.csv) | Extracted Scope 1/2/3 figures, with the page each came from |
| [`data/degree_scorecard.json`](data/degree_scorecard.json) | Stated targets vs verified progress per company |
| `data/raw/` | The 7 source PDFs — not committed, see below for where to get them |
| [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) | Full write-up: methodology, architecture, results, limitations |

## Data sources

| Company | Report | Pages |
|---|---|---|
| Infosys | ESG report 2022-23 | 65 |
| Microsoft | 2024 Environmental Sustainability Report (FY23 data) | 88 |
| Siemens | Sustainability Report 2024 | 170 |
| Siemens | Sustainability Statement (ESRS/CSRD), fiscal 2025 | 117 |
| Siemens Energy | Sustainability Report 2024 | 91 |
| Siemens Energy | Sustainability Report 2024 — Performance Indicator Overview | 10 |
| Siemens (Brazil) | Institutional and ESG Report 2024 | 129 |

All are the companies' own publicly published reports, downloaded from their investor
relations / sustainability pages.

### Getting the PDFs

The PDFs aren't in this repo (58MB, and they're third-party documents). To run this
yourself, download the seven reports above from the companies' sustainability pages
and drop them into `data/raw/` using these exact filenames, which `src/config.py`
expects:

```
data/raw/
├── infosys-esg-report-2022-23.pdf
├── Microsoft-2024-Environmental-Sustainability-Report.pdf
├── sustainability-report.pdf                              # Siemens AG
├── sustainability-statement.pdf                           # Siemens AG (ESRS/CSRD)
├── se-sustainability-report-2024-pdf_Original file.pdf     # Siemens Energy
├── se-sr-2024-esg-performance-pdf_Original file.pdf         # Siemens Energy indicators
└── Relatorio-Institucional-ESG-ENG-2024.pdf                # Siemens Brazil
```

If you use different filenames, update the `REPORTS` list in `src/config.py` to match.
Then run `python -m src.ingest` to build the vector database (~1 minute).
