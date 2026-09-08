"""
This is my retrieval layer. I embed the query, search ChromaDB, then re-rank the
results using both semantic similarity and keyword matching (RRF).

I started out with plain vector search here and it kept missing the emissions tables,
which was frustrating because that's the main thing I built this for. When I dug into
it I realised those pages list 20+ different metrics each, so the chunk embedding ends
up being an average of everything on the page and my query "scope 1 emissions 2024"
never ranked it highly. I added a keyword signal on top and that fixed it, without
needing to load a second model.
"""
import re
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME, TOP_K_RETRIEVAL

# I had to make this pool much bigger than I first expected. My keyword score can only
# reorder what Chroma already handed back, so if the right chunk isn't in the pool then
# no amount of keyword weighting will save it. I printed the ranks on a couple of real
# queries and found the correct chunk sitting at rank 84 and 159, which is why these
# numbers are so high. Chroma runs locally so a big pool costs me almost nothing.
CANDIDATE_MULTIPLIER = 20
MIN_CANDIDATES = 150
RRF_K = 20  # the standard is 60, I lowered it so my keyword matches count for more
COVERAGE_WEIGHT = 0.06  # I give a bonus when a chunk has every keyword from the query

STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "of", "in", "on", "for", "to",
    "and", "or", "what", "which", "did", "does", "do", "how", "much", "many",
    "at", "by", "with", "its", "this", "that", "as", "be", "has", "have", "had",
}

SCOPE_PHRASE_PATTERN = re.compile(r"scope\s*[123]", re.IGNORECASE)


def extract_keywords(query: str):
    tokens = re.findall(r"[a-zA-Z0-9%.]+", query.lower())
    keywords = {t for t in tokens if len(t) >= 3 and t not in STOPWORDS}
    scope_phrases = {p.lower().replace(" ", "") for p in SCOPE_PHRASE_PATTERN.findall(query)}
    return keywords, scope_phrases


@lru_cache(maxsize=2048)
def keyword_regex(keyword: str):
    """I match whole words here, and I allow an optional space between letters
    and digits.

    I needed the spacing part because I tested "infosys re100" and got nothing back -
    turns out the report writes it as "RE 100" with a space. Same problem with
    "scope1" vs "Scope 1". Without this those queries score zero on keywords.
    """
    parts = re.findall(r"[a-z]+|\d+", keyword)
    if len(parts) > 1:
        body = r"\s*".join(re.escape(p) for p in parts)
    else:
        body = re.escape(keyword)
    return re.compile(rf"\b{body}\b")


def keyword_score(text: str, keywords: set, scope_phrases: set) -> int:
    """I count how many of the query keywords appear in the text, whole words only.

    I was using a plain `in` check here originally and it was a real bug. "net" was
    matching inside "internet" and "network", so completely unrelated pages tied with
    the actual "net zero" page on keyword score and then beat it on the tiebreak.
    Took me a while to spot because the symptom just looked like bad answers.
    """
    text_lower = text.lower()
    score = 0
    for kw in keywords:
        if keyword_regex(kw).search(text_lower):
            score += 1
    text_scope_phrases = {p.replace(" ", "") for p in SCOPE_PHRASE_PATTERN.findall(text_lower)}
    score += 3 * len(scope_phrases & text_scope_phrases)  # exact "scope N" match is a strong signal
    return score


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def chunk_id(source_file: str, page_number: int, chunk_index: int) -> str:
    return f"{source_file}::p{page_number}::c{chunk_index}"


def expand_with_neighbors(collection, candidates_by_id: dict):
    """I also grab the chunks either side of each hit, from the same page.

    I added this after testing the Siemens DEGREE question. The sentence saying
    "six fields of action" landed in one chunk and the actual list of six landed in
    the next one, so I only ever retrieved half of it and got an incomplete answer.
    """
    neighbor_ids = set()
    for c in candidates_by_id.values():
        meta = c["meta"]
        for delta in (-1, 1):
            nid = chunk_id(meta["source_file"], meta["page_number"], meta["chunk_index"] + delta)
            if nid not in candidates_by_id:
                neighbor_ids.add(nid)

    if not neighbor_ids:
        return candidates_by_id

    fetched = collection.get(ids=list(neighbor_ids))
    for doc, meta, cid in zip(fetched["documents"], fetched["metadatas"], fetched["ids"]):
        candidates_by_id[cid] = {"text": doc, "meta": meta, "distance": None, "is_neighbor": True}

    return candidates_by_id


def retrieve(query: str, k: int = TOP_K_RETRIEVAL, company: str | None = None):
    """My main retrieval function - top-k chunks plus neighbours, re-ranked with RRF."""
    model = get_embedding_model()
    collection = get_collection()

    candidate_k = max(k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
    query_embedding = model.encode([query], convert_to_numpy=True).tolist()

    where = {"company": company} if company else None
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=candidate_k,
        where=where,
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        candidates.append({"text": doc, "meta": meta, "distance": dist})

    keywords, scope_phrases = extract_keywords(query)

    # my first ranking - by vector distance, closest first
    semantic_order = sorted(range(len(candidates)), key=lambda i: candidates[i]["distance"])
    semantic_rank = {idx: r for r, idx in enumerate(semantic_order)}

    # my second ranking - by keyword score, highest first, ties broken by distance
    kw_scores = [keyword_score(c["text"], keywords, scope_phrases) for c in candidates]
    keyword_order = sorted(
        range(len(candidates)), key=lambda i: (-kw_scores[i], candidates[i]["distance"])
    )
    keyword_rank = {idx: r for r, idx in enumerate(keyword_order)}

    # I add an extra bonus when a chunk contains all the query words. Plain RRF weights
    # both signals equally and I found that wasn't enough - when I tested
    # "what is infosys net zero target" the correct chunk had all 4 words in it but
    # still lost out, because its vector rank was so poor it dragged the score down.
    n_keywords = max(len(keywords), 1)

    def rrf_score(idx):
        coverage = kw_scores[idx] / n_keywords
        return (
            1.0 / (RRF_K + semantic_rank[idx])
            + 1.0 / (RRF_K + keyword_rank[idx])
            + COVERAGE_WEIGHT * min(coverage, 1.0)
        )

    reranked = sorted(range(len(candidates)), key=rrf_score, reverse=True)[:k]

    selected_by_id = {}
    for idx in reranked:
        c = candidates[idx]
        meta = c["meta"]
        cid = chunk_id(meta["source_file"], meta["page_number"], meta["chunk_index"])
        selected_by_id[cid] = c

    selected_by_id = expand_with_neighbors(collection, selected_by_id)

    hits = []
    for c in selected_by_id.values():
        meta = c["meta"]
        hits.append(
            {
                "text": c["text"],
                "company": meta["company"],
                "report_title": meta["report_title"],
                "report_year": meta["report_year"],
                "page_number": meta["page_number"],
                "source_file": meta["source_file"],
                "distance": c["distance"],
                "chunk_index": meta["chunk_index"],
            }
        )
    hits.sort(key=lambda h: (h["source_file"], h["page_number"], h["chunk_index"]))
    return hits
