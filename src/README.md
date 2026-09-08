# src/

The pipeline. Everything here is imported by `app.py` and `cli.py`.

| File | What it does |
|---|---|
| `config.py` | My settings and the registry of the 7 reports — filenames, company names, reporting years, chunk sizes, model name, retrieval depth. |
| `ingest.py` | One-time setup. Reads each PDF page by page with PyMuPDF, splits into chunks that never cross a page boundary, embeds them locally with MiniLM and stores them in ChromaDB. Took 66s for 2,833 chunks. |
| `retriever.py` | Finds the relevant chunks. Vector search over a wide candidate pool, then re-ranked by fusing semantic similarity with keyword matching (RRF), plus a bonus for chunks containing every query word. Also pulls in neighbouring chunks from the same page. |
| `rag_chain.py` | Builds the prompt, calls Gemini, then runs my two hallucination checks — one verifies every cited page was actually retrieved, the other verifies every number in the answer appears verbatim in the retrieved text. `ask()` is the entry point everything else uses. |
| `emissions_extractor.py` | Asks a fixed set of Scope 1/2/3 questions for each company and writes the answers to `data/emissions_summary.csv`. |
| `degree_scorer.py` | Checks each company against its own published targets — Siemens' DEGREE framework for Siemens, their own headline commitments for the others. Writes `data/degree_scorecard.json`. |

## Order things run in

```
ingest.py  →  retriever.py  →  rag_chain.py
   (once)        (per query)      (per query)
```

`emissions_extractor.py` and `degree_scorer.py` both sit on top of `rag_chain.ask()`.
