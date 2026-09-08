"""
Settings and the list of reports. Everything configurable lives here.

I got the company/title/year for each report by printing page 1 of each PDF - the
filenames were useless (two of them were just "sustainability-report.pdf").
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "esg_reports"

# I picked MiniLM because it runs locally on CPU and I didn't want to pay for
# embeddings or deal with rate limits on 2800+ chunks.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150
MIN_CHARS_FOR_REAL_PAGE = 40  # I treat anything under this as an image-only page

# My list of reports. The filename has to match what's in data/raw/.
# I use report_year for the citations - it's the reporting year, not when it was published.
REPORTS = [
    {
        "filename": "infosys-esg-report-2022-23.pdf",
        "company": "Infosys",
        "report_title": "Infosys ESG report 2022-23",
        "report_year": "FY2022-23",
    },
    {
        "filename": "Microsoft-2024-Environmental-Sustainability-Report.pdf",
        "company": "Microsoft",
        "report_title": "Microsoft 2024 Environmental Sustainability Report",
        "report_year": "2024 (FY23 data)",
    },
    {
        "filename": "se-sustainability-report-2024-pdf_Original file.pdf",
        "company": "Siemens Energy",
        "report_title": "Siemens Energy Sustainability Report 2024",
        "report_year": "FY2024",
    },
    {
        "filename": "se-sr-2024-esg-performance-pdf_Original file.pdf",
        "company": "Siemens Energy",
        "report_title": "Siemens Energy Sustainability Report 2024 - Performance Indicator Overview",
        "report_year": "FY2024",
    },
    {
        "filename": "sustainability-report.pdf",
        "company": "Siemens",
        "report_title": "Siemens Sustainability Report 2024",
        "report_year": "FY2024",
    },
    {
        "filename": "sustainability-statement.pdf",
        "company": "Siemens",
        "report_title": "Siemens Sustainability Statement (ESRS/CSRD, Part of the Combined Management Report)",
        "report_year": "FY2025",
    },
    {
        "filename": "Relatorio-Institucional-ESG-ENG-2024.pdf",
        "company": "Siemens Brazil",
        "report_title": "Siemens Institutional and ESG Report 2024 (Brazil)",
        "report_year": "2024",
    },
]

COMPANIES = sorted({r["company"] for r in REPORTS})

GEMINI_GENERATION_TEMPERATURE = 0.1
# I started with 6 here. Bumped it to 10 after I noticed short questions like
# "infosys net zero target" were failing - 6 chunks just wasn't enough to go on.
# I checked the latency first and it was around 1s, so I had room to spare.
TOP_K_RETRIEVAL = 10

# If my first search comes back empty I retry with this instead (see ask()).
FALLBACK_RETRIEVAL_K = 20
