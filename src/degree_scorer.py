"""
I use this to check companies against their own published targets.

DEGREE turned out to be Siemens' own framework - I found it by accident while reading
their Sustainability Report 2024 to write my test questions (section 1.1, p.8-9). It's
six areas: Decarbonization, Ethics, Governance, Resource efficiency, Equity,
Employability, and each one has a real number attached to it. So instead of inventing
my own scoring rubric I just ask my pipeline what progress they report, and compare it
against what they promised.

The other companies don't use DEGREE, so for those I check their own headline
commitments instead (Microsoft's carbon negative by 2030, and so on). I didn't want to
force a framework onto companies that never signed up to it.

Usage:
    python -m src.degree_scorer
"""
import json
import sys
import time

from src.config import PROJECT_ROOT
from src.rag_chain import ask

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_JSON = PROJECT_ROOT / "data" / "degree_scorecard.json"

# I copied these targets straight out of "Siemens Sustainability Report 2024" p.8-9.
# The baselines and targets here are what Siemens published. I don't hardcode their
# progress though - I ask my own pipeline for that below, so the comparison is
# actually doing something rather than just printing numbers I typed in.
SIEMENS_DEGREE_FRAMEWORK = {
    "D": {
        "name": "Decarbonization",
        "description": "Support the 1.5C target to fight global warming.",
        "target": "Reduce emissions in own operations by 55% by 2025 (vs. FY19 baseline of 737 kt CO2e) "
                  "and by 90% by 2030; Net Zero supply chain by 2050 with 20% reduction by 2030.",
        "verification_query": "What progress has Siemens made on reducing emissions in its own operations "
                               "compared to its target of a 55% reduction by 2025 from the fiscal year 2019 baseline?",
    },
    "E1": {
        "name": "Ethics",
        "description": "Foster a culture of trust, adhere to ethical standards, and handle data with care.",
        "target": "Train 100% of employees on the Business Conduct Guidelines every three years, by 2025.",
        "verification_query": "What share of Siemens employees have completed training on the Business Conduct Guidelines?",
    },
    "G": {
        "name": "Governance",
        "description": "Apply state-of-the-art systems for effective and responsible business conduct.",
        "target": "ESG-secured supply chain based on supplier commitment to the Supplier Code of Conduct; "
                  "long-term incentives anchored to ESG criteria.",
        "verification_query": "Has Siemens anchored ESG criteria into long-term management incentives, and are suppliers committed to the Supplier Code of Conduct?",
    },
    "R": {
        "name": "Resource efficiency",
        "description": "Achieve circularity, dematerialize, and conserve biodiversity.",
        "target": "Robust Eco Design for 100% of relevant portfolio by 2030; zero landfill waste by 2030 "
                  "(50% waste-to-landfill reduction by 2025).",
        "verification_query": "What progress has Siemens made toward its waste-to-landfill reduction target and Eco Design coverage of its portfolio?",
    },
    "E2": {
        "name": "Equity",
        "description": "Foster diversity, equity, inclusion, and community development.",
        "target": "30% female share in Top Management by 2025 (baseline FY20: 22.7%).",
        "verification_query": "What is Siemens' current female share in Top Management, compared to its 30% by 2025 target?",
    },
    "E3": {
        "name": "Employability",
        "description": "Enable people to stay resilient and relevant in a changing environment.",
        "target": "Increase digital learning hours to 25 per employee by 2025 (baseline FY20: 7 hours); "
                  "30% improvement in global LTIFR by 2025 (baseline FY20: 0.31).",
        "verification_query": "How many digital learning hours per employee has Siemens reached, and what is its current LTIFR, compared to its 2025 targets?",
    },
}

# These companies don't use DEGREE, so I check them against their own headline
# climate commitments instead.
OTHER_COMPANY_TARGETS = {
    "Microsoft": {
        "target": "Carbon negative by 2030; remove all historical carbon emissions (since 1975) by 2050.",
        "verification_query": "What progress has Microsoft made toward its commitment to be carbon negative by 2030? Has its overall emissions increased or decreased relative to its baseline year?",
    },
    "Infosys": {
        "target": "Carbon neutrality across Scope 1, 2, and 3 emissions maintained every year (ESG Vision 2030); net zero by 2040 (The Climate Pledge).",
        "verification_query": "Has Infosys maintained carbon neutrality across Scope 1, 2, and 3 emissions, and what is its net zero target year?",
    },
    "Siemens Energy": {
        "target": "Science-based net-zero target; part of Siemens Energy's own decarbonization pathway.",
        "verification_query": "What are Siemens Energy's Scope 1 and Scope 2 greenhouse gas emissions targets or reduction commitments?",
    },
    "Siemens Brazil": {
        "target": "Alignment with global Siemens DEGREE sustainability framework and decarbonization commitments.",
        "verification_query": "What sustainability or decarbonization commitments does Siemens describe for its Brazil operations?",
    },
}


SECONDS_BETWEEN_CALLS = 4.5  # stay under the free-tier ~15 requests/minute limit (60s/15=4s + margin)


def verify_siemens_degree():
    results = {}
    for letter, spec in SIEMENS_DEGREE_FRAMEWORK.items():
        result = ask(spec["verification_query"], company="Siemens")
        results[letter] = {
            "name": spec["name"],
            "description": spec["description"],
            "stated_target": spec["target"],
            "rag_verified_progress": result["answer"],
            "citation_consistent": result["citation_consistent"],
            "source_pages": [s["page_number"] for s in result["sources"]],
            "latency_seconds": round(result["latency_seconds"], 2),
        }
        print(f"[Siemens DEGREE:{letter}] {spec['name']} -> citation_consistent={result['citation_consistent']}")
        time.sleep(SECONDS_BETWEEN_CALLS)
    return results


def verify_other_companies():
    results = {}
    for company, spec in OTHER_COMPANY_TARGETS.items():
        result = ask(spec["verification_query"], company=company)
        results[company] = {
            "stated_target": spec["target"],
            "rag_verified_progress": result["answer"],
            "citation_consistent": result["citation_consistent"],
            "source_pages": [s["page_number"] for s in result["sources"]],
            "latency_seconds": round(result["latency_seconds"], 2),
        }
        print(f"[{company}] target verification -> citation_consistent={result['citation_consistent']}")
        time.sleep(SECONDS_BETWEEN_CALLS)
    return results


def run_scoring():
    scorecard = {
        "framework_source": "Siemens Sustainability Report 2024, section 1.1 'Our DEGREE sustainability "
                             "framework sets measurable ambitions' (p.8-9). Applied here only to Siemens; "
                             "other companies are checked against their own stated targets instead of being "
                             "force-fit into a framework that isn't theirs.",
        "siemens_degree": verify_siemens_degree(),
        "other_companies": verify_other_companies(),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)

    print(f"\nWrote DEGREE/target scorecard to {OUTPUT_JSON}")
    return scorecard


if __name__ == "__main__":
    run_scoring()
