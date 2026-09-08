"""
I use this to pull Scope 1/2/3 numbers for every company into one CSV.

I ask the same set of questions for each company, then pull the number out of the
answer with a regex. My regex still isn't perfect so I keep the model's full answer
in a raw_answer column, which lets me check any value by hand.

Usage:
    python -m src.emissions_extractor
"""
import csv
import re
import sys
import time

from src.config import COMPANIES, PROJECT_ROOT
from src.rag_chain import NOT_FOUND_MESSAGE, ask

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_CSV = PROJECT_ROOT / "data" / "emissions_summary.csv"

QUERY_TEMPLATES = {
    "Scope 1": "What is the most recently reported total Scope 1 (direct) greenhouse gas emissions "
               "figure, stated as an absolute number with its unit (e.g. metric tons CO2e or ktCO2e)?",
    "Scope 2 (market-based)": "What is the most recently reported total Scope 2 market-based greenhouse "
               "gas emissions figure, stated as an absolute number with its unit?",
    "Scope 2 (location-based)": "What is the most recently reported total Scope 2 location-based "
               "greenhouse gas emissions figure, stated as an absolute number with its unit?",
    "Scope 3": "What is the most recently reported total Scope 3 greenhouse gas emissions figure "
               "(all categories combined), stated as an absolute number with its unit?",
}

SECONDS_BETWEEN_CALLS = 4.5  # stay under the free-tier ~15 requests/minute limit (60s/15=4s + margin)

# I find every number in the answer, then keep the first one that isn't part of a
# "Scope N" label or a "fiscal year YYYY" reference, and that has a unit like CO2/ton/kt
# right after it. My first version just grabbed the first number in the string and it
# kept returning "2025" and "2024" because the answers are full sentences like
# "For fiscal year 2025, ... figure is 70.7 ktCO2e" - the year comes first.
NUMBER_TOKEN = re.compile(r"\d[\d,]*\.?\d*")
UNIT_HINT = re.compile(r"co2|ton|kt\b|tco", re.IGNORECASE)
SCOPE_LABEL_BEFORE = re.compile(r"scope\s*$", re.IGNORECASE)
YEAR_LABEL_BEFORE = re.compile(r"(year|fy)\s*$", re.IGNORECASE)
UNIT_WINDOW_CHARS = 40
UNIT_TEXT_PATTERN = re.compile(r"[A-Za-z][\w .-]{0,25}")


def extract_first_number(text: str):
    for match in NUMBER_TOKEN.finditer(text):
        preceding = text[max(0, match.start() - 10): match.start()]
        if SCOPE_LABEL_BEFORE.search(preceding) or YEAR_LABEL_BEFORE.search(preceding):
            continue  # this number is a "Scope N" label or a "fiscal year YYYY" reference, not a value
        window = text[match.end(): match.end() + UNIT_WINDOW_CHARS]
        if not UNIT_HINT.search(window):
            continue
        unit_match = UNIT_TEXT_PATTERN.search(window)
        unit = unit_match.group(0).split("[")[0].strip().rstrip(")") if unit_match else None
        return match.group(0), unit or None
    return None, None


def run_extraction():
    rows = []
    for company in COMPANIES:
        for scope_label, question in QUERY_TEMPLATES.items():
            full_question = f"For {company}: {question}"
            t0 = time.perf_counter()
            result = ask(full_question, company=company)
            elapsed = time.perf_counter() - t0

            answer = result["answer"]
            found = answer.strip() != NOT_FOUND_MESSAGE
            number, unit = (None, None)
            if found:
                number, unit = extract_first_number(answer)

            source_pages = ";".join(
                f"p.{s['page_number']}" for s in result["sources"]
            ) if result["sources"] else ""

            rows.append(
                {
                    "company": company,
                    "scope": scope_label,
                    "value": number or "",
                    "unit": unit or "",
                    "found": found,
                    "citation_consistent": result["citation_consistent"],
                    "source_pages": source_pages,
                    "latency_seconds": round(elapsed, 2),
                    "raw_answer": answer.replace("\n", " "),
                }
            )
            print(f"[{company}] {scope_label}: value={number} unit={unit} found={found} ({elapsed:.2f}s)")
            time.sleep(SECONDS_BETWEEN_CALLS)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_extraction()
