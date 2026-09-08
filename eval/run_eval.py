"""
This is how I test the whole thing. I run my question set through the pipeline
and score it.

What I measure:
  - retrieval accuracy: did I get back the page I know the answer is on?
  - answer accuracy: does the answer contain the figure I expected?
  - citation consistency + numeric grounding: my two hallucination checks
  - latency

I wrote all the test questions myself by reading the PDFs and noting down the page
each answer was on. I included a few where the correct behaviour is to refuse,
because the number genuinely isn't in that report - Infosys never publishes an
absolute Scope 1 figure, so if my system answers that one it's making it up.

Usage:
    python -m eval.run_eval
"""
import json
import statistics
import sys
import time
from pathlib import Path

from src.config import PROJECT_ROOT
from src.rag_chain import NOT_FOUND_MESSAGE, ask

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SECONDS_BETWEEN_CALLS = 4.5  # stay under the free-tier ~15 requests/minute limit (60s/15=4s + margin)

TESTSET_PATH = Path(__file__).parent / "qa_testset.json"
REPORT_PATH = Path(__file__).parent / "eval_report.md"
RESULTS_JSON_PATH = Path(__file__).parent / "eval_results.json"


def run_eval():
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))

    rows = []
    for case in testset:
        result = ask(case["question"], company=case.get("company"))
        retrieved_pages = {s["page_number"] for s in result["sources"]}
        retrieved_files = {s["source_file"] for s in result["sources"]}

        if case["category"] == "factual":
            retrieval_hit = case["expected_page"] in retrieved_pages and (
                case["expected_source_file"] in retrieved_files
            )
            answer_lower = result["answer"].lower()
            answer_hit = any(
                str(s).lower() in answer_lower for s in case["expected_answer_contains"]
            ) if case["expected_answer_contains"] else None
        else:  # refusal case: correct behavior is the NOT_FOUND_MESSAGE
            retrieval_hit = None
            answer_hit = result["answer"].strip() == NOT_FOUND_MESSAGE

        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "retrieval_hit": retrieval_hit,
                "answer_hit": answer_hit,
                "citation_consistent": result["citation_consistent"],
                "numerically_grounded": result["numerically_grounded"],
                "latency_seconds": result["latency_seconds"],
                "answer": result["answer"],
                "retrieved_pages": sorted(retrieved_pages),
            }
        )
        status = "OK" if (answer_hit in (True, None)) else "MISS"
        print(f"[{case['id']:>16s}] retrieval_hit={retrieval_hit} answer_hit={answer_hit} "
              f"citation_ok={result['citation_consistent']} grounded={result['numerically_grounded']} "
              f"({result['latency_seconds']:.2f}s) {status}")
        time.sleep(SECONDS_BETWEEN_CALLS)

    factual = [r for r in rows if r["category"] == "factual"]
    refusal = [r for r in rows if r["category"] == "refusal"]
    latencies = [r["latency_seconds"] for r in rows]

    retrieval_accuracy = (
        sum(1 for r in factual if r["retrieval_hit"]) / len(factual) if factual else 0.0
    )
    answer_accuracy = (
        sum(1 for r in factual if r["answer_hit"]) / len(factual) if factual else 0.0
    )
    refusal_accuracy = (
        sum(1 for r in refusal if r["answer_hit"]) / len(refusal) if refusal else 0.0
    )
    citation_consistency_rate = sum(1 for r in rows if r["citation_consistent"]) / len(rows)
    numeric_grounding_rate = sum(1 for r in rows if r["numerically_grounded"]) / len(rows)
    fully_verified_rate = sum(
        1 for r in rows if r["citation_consistent"] and r["numerically_grounded"]
    ) / len(rows)
    mean_latency = statistics.mean(latencies)
    p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    under_2s_rate = sum(1 for l in latencies if l < 2.0) / len(latencies)

    summary = {
        "n_questions": len(rows),
        "n_factual": len(factual),
        "n_refusal": len(refusal),
        "retrieval_accuracy": retrieval_accuracy,
        "answer_accuracy": answer_accuracy,
        "refusal_accuracy": refusal_accuracy,
        "citation_consistency_rate": citation_consistency_rate,
        "numeric_grounding_rate": numeric_grounding_rate,
        "fully_verified_rate": fully_verified_rate,
        "mean_latency_seconds": mean_latency,
        "p95_latency_seconds": p95_latency,
        "under_2s_rate": under_2s_rate,
    }

    RESULTS_JSON_PATH.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    write_report(summary, rows)
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFull report: {REPORT_PATH}")
    return summary


def write_report(summary, rows):
    lines = []
    lines.append("# Evaluation Report\n")
    lines.append(
        "These are the **actual measured results** of running the hand-built ground-truth "
        "question set (`qa_testset.json`) through the live RAG pipeline (retrieval + Gemini "
        "generation). Ground truth was built by manually reading the source PDFs, not "
        "invented.\n"
    )
    lines.append("## Summary\n")
    lines.append(f"- Questions evaluated: {summary['n_questions']} ({summary['n_factual']} factual, {summary['n_refusal']} refusal/negative)")
    lines.append(f"- **Retrieval accuracy** (expected page found in top-k): {summary['retrieval_accuracy']*100:.1f}%")
    lines.append(f"- **Answer accuracy** (expected fact present in generated answer): {summary['answer_accuracy']*100:.1f}%")
    lines.append(f"- **Refusal accuracy** (correctly said 'not found' instead of hallucinating): {summary['refusal_accuracy']*100:.1f}%")
    lines.append(f"- **Citation consistency rate** (every cited page was actually retrieved): {summary['citation_consistency_rate']*100:.1f}%")
    lines.append(f"- **Numeric grounding rate** (every stated figure appears verbatim in retrieved context): {summary['numeric_grounding_rate']*100:.1f}%")
    lines.append(f"- **Fully verified rate** (both checks pass): {summary['fully_verified_rate']*100:.1f}%")
    lines.append(f"- **Mean latency**: {summary['mean_latency_seconds']:.2f}s")
    lines.append(f"- **P95 latency**: {summary['p95_latency_seconds']:.2f}s")
    lines.append(f"- **Share of responses under 2s**: {summary['under_2s_rate']*100:.1f}%")
    lines.append("")
    lines.append("## Per-question results\n")
    lines.append("| id | category | retrieval_hit | answer_hit | citation_ok | grounded | latency (s) |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['category']} | {r['retrieval_hit']} | {r['answer_hit']} | "
            f"{r['citation_consistent']} | {r['numerically_grounded']} | {r['latency_seconds']:.2f} |"
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_eval()
