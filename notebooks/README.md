# notebooks/

| File | What it is |
|---|---|
| `esg_rag_walkthrough.ipynb` | The full build walkthrough — how I got text out of the PDFs, chose a chunking strategy, built the vector store, and then the three retrieval bugs I hit and how I found each one. Ends with the hallucination checks, my evaluation results and the DEGREE finding. |
| `images/` | Screenshots of the running app, embedded in the notebook. |

Start here if you want to understand the project. It renders directly on GitHub with all
the outputs and screenshots, so there's nothing to run.

The two cells that call `!python -m src.ingest --rebuild` and `!python -m eval.run_eval`
will rebuild the vector database and use API quota if you execute them — the saved
outputs are already in the file, so you don't need to.
