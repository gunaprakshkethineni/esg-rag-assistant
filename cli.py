"""My terminal chat loop. I use this for quick testing without launching Streamlit.

Usage:
    python cli.py
    python cli.py --company "Microsoft"
"""
import argparse
import sys

from src.rag_chain import ask

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", default=None)
    args = parser.parse_args()

    print("ESG RAG Assistant (CLI) - type 'exit' to quit.")
    if args.company:
        print(f"Filtering to company: {args.company}")

    while True:
        query = input("\n> ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break

        result = ask(query, company=args.company)
        print(f"\n{result['answer']}")
        print(
            f"\n[latency: {result['latency_seconds']:.2f}s | "
            f"citation_consistent: {result['citation_consistent']} | "
            f"numerically_grounded: {result['numerically_grounded']}]"
        )
        if result["sources"]:
            print("Sources retrieved:")
            for s in result["sources"]:
                dist = f"{s['distance']:.3f}" if s["distance"] is not None else "neighbor"
                print(f"  - {s['company']} {s['report_year']}, p.{s['page_number']} (dist={dist})")


if __name__ == "__main__":
    main()
