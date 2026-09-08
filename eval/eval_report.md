# Evaluation Report

These are the **actual measured results** of running the hand-built ground-truth question set (`qa_testset.json`) through the live RAG pipeline (retrieval + Gemini generation). Ground truth was built by manually reading the source PDFs, not invented.

## Summary

- Questions evaluated: 26 (23 factual, 3 refusal/negative)
- **Retrieval accuracy** (expected page found in top-k): 91.3%
- **Answer accuracy** (expected fact present in generated answer): 95.7%
- **Refusal accuracy** (correctly said 'not found' instead of hallucinating): 66.7%
- **Citation consistency rate** (every cited page was actually retrieved): 69.2%
- **Numeric grounding rate** (every stated figure appears verbatim in retrieved context): 80.8%
- **Fully verified rate** (both checks pass): 50.0%
- **Mean latency**: 1.52s
- **P95 latency**: 7.26s
- **Share of responses under 2s**: 96.2%

## Per-question results

| id | category | retrieval_hit | answer_hit | citation_ok | grounded | latency (s) |
|---|---|---|---|---|---|---|
| se1 | factual | True | True | True | True | 10.10 |
| se2 | factual | True | True | True | True | 1.12 |
| se3 | factual | True | True | True | True | 1.36 |
| se4 | factual | True | True | True | True | 1.03 |
| se5 | factual | True | True | False | True | 1.03 |
| se6 | factual | True | True | False | True | 1.41 |
| se7 | factual | True | True | False | True | 1.06 |
| sm1 | factual | True | True | True | True | 1.34 |
| sm2 | factual | True | True | True | False | 1.13 |
| sm3 | factual | False | True | False | True | 0.97 |
| sm4 | factual | True | True | False | True | 1.06 |
| sm5 | factual | True | True | False | True | 1.28 |
| sm6 | factual | True | False | True | True | 1.02 |
| sm7 | factual | False | True | False | True | 1.04 |
| sbr1 | factual | True | True | True | True | 1.02 |
| sbr2 | factual | True | True | True | True | 1.13 |
| ms1 | factual | True | True | True | False | 0.94 |
| ms2 | factual | True | True | True | True | 1.02 |
| ms3 | factual | True | True | True | False | 1.22 |
| ms4 | factual | True | True | True | False | 1.14 |
| ms5 | factual | True | True | True | False | 1.03 |
| if1 | factual | True | True | True | True | 0.97 |
| if2 | factual | True | True | True | True | 1.15 |
| if3_refusal | refusal | None | True | True | True | 1.99 |
| cross1_refusal | refusal | None | True | True | True | 1.82 |
| cross2_refusal | refusal | None | False | False | True | 1.06 |