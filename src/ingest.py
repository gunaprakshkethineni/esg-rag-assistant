"""
This is where I read the PDFs, chunk them, embed them and store them in ChromaDB.
I only need to run this once.

I made sure my chunks never cross a page boundary. I did that on purpose because it's
the only way I can show an exact page number with every answer later on.

Usage:
    python -m src.ingest            # skips if I've already done it
    python -m src.ingest --rebuild  # start over from scratch
"""
import argparse
import sys
import time

import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    COLLECTION_NAME,
    DATA_RAW_DIR,
    EMBEDDING_MODEL_NAME,
    MIN_CHARS_FOR_REAL_PAGE,
    REPORTS,
)


def chunk_text(text: str, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS):
    """I chunk with overlap here and try to break at sentence ends.

    I added the sentence-boundary logic after I saw a chunk get cut in the middle
    of a figure, which made the number useless.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # I look backwards for the last sentence or paragraph break in this window
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_report_chunks(pdf_path, report_meta):
    """I pull the chunks and metadata out of one PDF here.

    I also return any pages that came back with almost no text so I can see them
    in the log. When I ran this I got 3 of them and they turned out to be
    image-only cover pages, so nothing was actually lost.
    """
    doc = fitz.open(pdf_path)
    flagged_pages = []
    records = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        text = page.get_text("text")
        if len(text.strip()) < MIN_CHARS_FOR_REAL_PAGE:
            flagged_pages.append(page_number)
            continue
        for chunk_index, chunk in enumerate(chunk_text(text)):
            records.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source_file": report_meta["filename"],
                        "company": report_meta["company"],
                        "report_title": report_meta["report_title"],
                        "report_year": report_meta["report_year"],
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                    },
                }
            )
    doc.close()
    return records, flagged_pages


def run_ingest(rebuild: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    if collection.count() > 0 and not rebuild:
        print(
            f"Collection '{COLLECTION_NAME}' already has {collection.count()} chunks. "
            "Use --rebuild to re-ingest from scratch."
        )
        return

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    total_chunks = 0
    summary = []
    t0 = time.perf_counter()

    for report_meta in REPORTS:
        pdf_path = DATA_RAW_DIR / report_meta["filename"]
        if not pdf_path.exists():
            print(f"WARNING: missing file, skipping: {pdf_path}")
            continue

        print(f"Extracting: {report_meta['company']} - {report_meta['report_title']}")
        records, flagged_pages = extract_report_chunks(pdf_path, report_meta)

        if not records:
            print(f"  WARNING: no extractable text found in {pdf_path.name}")
            continue

        texts = [r["text"] for r in records]
        metadatas = [r["metadata"] for r in records]
        ids = [
            f"{report_meta['filename']}::p{m['page_number']}::c{m['chunk_index']}"
            for m in metadatas
        ]

        print(f"  {len(texts)} chunks, embedding...")
        embeddings = model.encode(
            texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        ).tolist()

        # I batch these because Chroma has a limit on how many I can add at once
        batch = 512
        for i in range(0, len(texts), batch):
            collection.add(
                ids=ids[i : i + batch],
                documents=texts[i : i + batch],
                metadatas=metadatas[i : i + batch],
                embeddings=embeddings[i : i + batch],
            )

        total_chunks += len(texts)
        summary.append(
            {
                "company": report_meta["company"],
                "file": report_meta["filename"],
                "chunks": len(texts),
                "flagged_pages": flagged_pages,
            }
        )
        if flagged_pages:
            print(f"  NOTE: {len(flagged_pages)} page(s) had little/no text (possible scan/image-only): {flagged_pages[:10]}{'...' if len(flagged_pages) > 10 else ''}")

    elapsed = time.perf_counter() - t0
    print("\n=== Ingestion summary ===")
    for row in summary:
        print(f"  {row['company']:<16s} {row['chunks']:>5d} chunks   ({row['file']})")
    print(f"Total chunks ingested: {total_chunks}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Collection now has {collection.count()} chunks at {CHROMA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    run_ingest(rebuild=args.rebuild)
