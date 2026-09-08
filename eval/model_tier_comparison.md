# Model tier comparison: flash-lite vs flash

While tuning the system, citation consistency on the default model
(`gemini-3.5-flash-lite`, auto-selected for speed) measured at only 53.8%
across the full eval set (see `eval_report.md`). To check whether this was a
model-tier limitation rather than a retrieval/prompt problem, the same
26-question eval set was re-run with `GEMINI_MODEL=gemini-3.5-flash` (the
standard, non-lite tier).

**Result:** all 19 questions that completed before the run hit the
standard tier's free-tier quota (20 requests/day, separate from and far
stricter than flash-lite's per-minute limit) had `citation_ok = True` --
19/19 = 100%, vs. 53.8% for flash-lite over the full set. The run could not
be completed to 26/26 because the daily quota was exhausted mid-run; it is
reported here as a partial-but-informative sample, not a full second eval.

| | flash-lite (default, full run) | flash (partial run, 19/26 before quota exhaustion) |
|---|---|---|
| Citation consistency | 53.8% (14/26) | 100% (19/19 completed) |
| Mean latency (successful calls) | 1.04s | ~4.7s (excluding one 23s rate-limited retry) |
| Free-tier throughput | ~15 requests/minute | 20 requests/**day** |

**Takeaway:** there is a real, measurable accuracy/latency/quota tradeoff
between Gemini's "flash" and "flash-lite" tiers on this task. `flash-lite`
is the practical default for a project evaluated at this volume (fits the
free tier, hits the <2s latency target), but a production deployment with a
paid quota would likely default to standard `flash` (or route only
low-confidence/high-stakes queries to it) to prioritize citation fidelity.
This is configurable via the `GEMINI_MODEL` environment variable
(`src/rag_chain.py::get_model_name`).
