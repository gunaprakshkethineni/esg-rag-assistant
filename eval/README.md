# eval/

How I tested the system. I wrote the questions myself by reading the source PDFs and
noting the page each answer sits on.

| File | What it does |
|---|---|
| `qa_testset.json` | My 26 ground-truth questions. Each one records the expected page, the source file, and the figure the answer should contain. Includes deliberate refusal cases where the number genuinely isn't in the report. |
| `run_eval.py` | Runs the whole set through the pipeline and scores it — retrieval accuracy, answer accuracy, refusal accuracy, both hallucination checks, and latency. |
| `eval_report.md` | The generated results, with a per-question breakdown. |
| `eval_results.json` | Same results as raw JSON, which the Streamlit app reads for its Evaluation tab. |
| `model_tier_comparison.md` | A side experiment: swapping Gemini flash-lite for standard flash gave 100% citation consistency on the 19 questions that completed before the free tier cut me off at 20 requests/day, but was 3-10x slower. |

## Current results

| Metric | Result |
|---|---|
| Retrieval accuracy | 91.3% |
| Answer accuracy | 95.7% |
| Citation consistency | 69.2% |
| Numeric grounding | 80.8% |
| Mean latency | 1.52s (96.2% under 2s) |

Run it yourself with `python -m eval.run_eval`. It paces requests to stay under the
free-tier rate limit, so a full run takes a few minutes.
