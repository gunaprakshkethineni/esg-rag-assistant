"""
Here I build the prompt out of the retrieved chunks and call Gemini.

I also run my two checks after the answer comes back. I learned pretty quickly that
telling the model "cite your sources" in the prompt isn't enough on its own - it still
gets it wrong sometimes - so instead of trusting it I verify the citations in code.
"""
import os
import re
import time
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.config import (
    COMPANIES,
    FALLBACK_RETRIEVAL_K,
    GEMINI_GENERATION_TEMPERATURE,
    TOP_K_RETRIEVAL,
)
from src.retriever import retrieve

load_dotenv()

MAX_RETRIES = 5

NOT_FOUND_MESSAGE = "I could not find this information in the provided ESG reports."

SYSTEM_INSTRUCTION = f"""You are an ESG data assistant. You answer questions ONLY using the
numbered context excerpts provided below, which come from corporate sustainability reports.

Rules (follow strictly):
1. Use ONLY facts stated in the context excerpts. Never use outside knowledge, never estimate,
   never infer a number that is not explicitly written in the context.
2. Every factual claim or number in your answer MUST end with an inline citation in the exact
   form [Company, Year, p. X], where X is the page number written in that excerpt's own header
   line (e.g. "[Excerpt 3] Siemens, FY2024, p. 8" means you must cite "p. 8", copied exactly --
   never guess, round, or reuse a page number from a different excerpt.
3. If multiple excerpts are relevant, cite each one used, with its own correct page number.
4. If the answer is not present in the context excerpts, respond with EXACTLY this sentence and
   nothing else: "{NOT_FOUND_MESSAGE}"
5. Be concise and quote exact figures and units (e.g. "1,000 metric tons CO2e") as written.
"""

CITATION_PATTERN = re.compile(r"p\.?\s*(\d+)", re.IGNORECASE)

# I use this in my grounding check below. I only look for numbers with 2+ digits so
# that I skip over the "1" in "Scope 1" and similar labels.
FACT_NUMBER_PATTERN = re.compile(r"\d[\d,]*\.?\d*")
PAGE_REF_BEFORE = re.compile(r"[Pp](?:age)?\.?\s*$")


@lru_cache(maxsize=1)
def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment/.env")
    return genai.Client(api_key=api_key)


EXCLUDE_SUBSTRINGS = ("preview", "image", "tts", "embedding", "vision", "gemma", "omni")


@lru_cache(maxsize=1)
def get_model_name():
    """I pick a flash model from whatever my API key actually has access to.

    I didn't want to hardcode a model name because they get deprecated. I can set
    GEMINI_MODEL to override this - I tested the non-lite flash model and it gave me
    much better citations, but it was 3-10x slower and the free tier only let me make
    20 requests a day on it, so I couldn't even finish my eval run.
    """
    override = os.getenv("GEMINI_MODEL")
    if override:
        return override

    client = get_client()
    candidates = []
    for m in client.models.list():
        name = m.name.replace("models/", "")
        if not m.supported_actions or "generateContent" not in m.supported_actions:
            continue
        if any(bad in name.lower() for bad in EXCLUDE_SUBSTRINGS):
            continue
        if "flash" not in name.lower():
            continue
        candidates.append(name)

    if not candidates:
        return "gemini-flash-latest"  # Google-maintained alias, always resolves to a current stable flash model

    # Prefer "flash-lite" variants: this is a factual extraction/citation task, not a reasoning
    # task, so the lighter/faster tier is the right latency/quality tradeoff. Within a tier,
    # prefer names with an explicit numeric version (e.g. gemini-2.5-flash) over unversioned
    # aliases, and prefer higher version numbers.
    def version_key(name):
        match = re.search(r"(\d+)\.(\d+)", name)
        if not match:
            return (-1, -1)
        return (int(match.group(1)), int(match.group(2)))

    lite_candidates = [c for c in candidates if "lite" in c.lower()]
    pool = lite_candidates or candidates
    pool.sort(key=version_key, reverse=True)
    return pool[0]


def generate_with_retry(client, model, prompt, config):
    """I retry with backoff on 429s here.

    The free tier gives me 15 requests a minute, so my batch scripts hit this
    constantly. My emissions extractor kept crashing halfway through until I
    added this.
    """
    delay = 5.0
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(model=model, contents=prompt, config=config)
        except genai_errors.ClientError as e:
            is_rate_limit = getattr(e, "code", None) == 429
            if not is_rate_limit or attempt == MAX_RETRIES - 1:
                raise
            wait = delay
            try:
                for d in e.details.get("error", {}).get("details", []):
                    if "retryDelay" in d:
                        wait = float(str(d["retryDelay"]).rstrip("s")) + 1.0
            except Exception:
                pass
            print(f"  [rate limited, retrying in {wait:.1f}s...]")
            time.sleep(wait)
            delay *= 2
    raise RuntimeError("unreachable")


def build_context_block(hits):
    lines = []
    for i, h in enumerate(hits, start=1):
        lines.append(
            f"[Excerpt {i}] {h['company']}, {h['report_year']}, p. {h['page_number']}\n{h['text']}"
        )
    return "\n\n".join(lines)


def check_citation_consistency(answer: str, hits):
    """My first check - every page number in the answer has to be one I actually
    retrieved.

    This catches the model citing a page it was never shown.
    """
    retrieved_pages = {h["page_number"] for h in hits}
    cited_pages = {int(p) for p in CITATION_PATTERN.findall(answer)}
    if not cited_pages:
        return True, cited_pages, retrieved_pages  # nothing to check (e.g. refusal message)
    unverified = cited_pages - retrieved_pages
    return len(unverified) == 0, cited_pages, retrieved_pages


def detect_company(text: str):
    """I pick up a company name from the question so I can filter retrieval to it.

    I found filtering is a lot more accurate than searching all 7 reports at once, but
    nobody actually bothers with the dropdown, so I detect it from the text instead.
    I sort by length first - I hit a bug where "Siemens" matched before "Siemens Energy"
    and I ended up searching the wrong company's reports entirely.
    """
    text_lower = text.lower()
    for company in sorted(COMPANIES, key=len, reverse=True):
        if company.lower() in text_lower:
            return company
    return None


def check_numeric_grounding(answer: str, hits):
    """My second check - the numbers in the answer have to appear in the retrieved text.

    I added this one later. I was clicking around in my own app and got an answer that
    cited a real page I had retrieved, but when I opened the sources panel the number
    wasn't on that page or anywhere else in the context. My citation check passed it
    completely, which is why I ended up with two separate checks instead of one.
    """
    context_text = " ".join(h["text"] for h in hits)
    context_no_commas = context_text.replace(",", "")

    ungrounded = []
    for match in FACT_NUMBER_PATTERN.finditer(answer):
        if len(match.group(0).replace(",", "").split(".")[0]) < 2:
            continue  # single-digit, e.g. a "Scope 1" label
        preceding = answer[max(0, match.start() - 6): match.start()]
        if PAGE_REF_BEFORE.search(preceding):
            continue  # this is a page number, not a fact
        value = match.group(0)
        value_no_commas = value.replace(",", "")
        if value in context_text or value_no_commas in context_no_commas:
            continue
        ungrounded.append(value)

    return len(ungrounded) == 0, ungrounded


def _answer_once(query: str, k: int, company: str | None, t0: float):
    """One retrieval + generation pass. I wrap this in ask() to add the retry."""
    hits = retrieve(query, k=k, company=company)
    if not hits:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
            "latency_seconds": time.perf_counter() - t0,
            "citation_consistent": True,
            "cited_pages": [],
            "retrieved_pages": [],
            "numerically_grounded": True,
            "ungrounded_numbers": [],
        }

    context = build_context_block(hits)
    client = get_client()
    prompt = f"Context excerpts:\n\n{context}\n\nQuestion: {query}\n\nAnswer (with citations):"

    response = generate_with_retry(
        client,
        get_model_name(),
        prompt,
        types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=GEMINI_GENERATION_TEMPERATURE,
        ),
    )
    answer = (response.text or "").strip()

    is_consistent, cited_pages, retrieved_pages = check_citation_consistency(answer, hits)
    is_grounded, ungrounded_numbers = check_numeric_grounding(answer, hits)

    latency = time.perf_counter() - t0
    return {
        "answer": answer,
        "sources": hits,
        "latency_seconds": latency,
        "citation_consistent": is_consistent,
        "cited_pages": sorted(cited_pages),
        "retrieved_pages": sorted(retrieved_pages),
        "numerically_grounded": is_grounded,
        "ungrounded_numbers": ungrounded_numbers,
    }


def ask(
    query: str,
    k: int = TOP_K_RETRIEVAL,
    company: str | None = None,
    auto_detect_company: bool = True,
):
    """This is my main entry point - ask a question, get back an answer with citations.

    Two things in here are just for handling messy input, because I noticed nobody
    types full proper sentences when they're actually using it. I guess the company
    from the question text, and if I get nothing back on the first try I retry once
    with a much bigger k before giving up.
    """
    t0 = time.perf_counter()

    if company is None and auto_detect_company:
        company = detect_company(query)

    result = _answer_once(query, k=k, company=company, t0=t0)

    if result["answer"].strip() == NOT_FOUND_MESSAGE and k < FALLBACK_RETRIEVAL_K:
        retry = _answer_once(query, k=FALLBACK_RETRIEVAL_K, company=company, t0=t0)
        # either way we return the retry - if the wider search still found nothing
        # then it genuinely isn't in the reports
        return retry

    return result
