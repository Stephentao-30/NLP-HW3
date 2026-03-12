import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import glob
import gc
from urllib.parse import urlparse

# Constants - tuned for 5,000+ page corpus on 4GB RAM
CHUNK_SIZE = 200       # words per chunk
OVERLAP = 40           # overlapping words between consecutive chunks
MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
ENCODE_BATCH = 256
MIN_FILE_BYTES = 200   # Skip files smaller than this (noise/empty pages)
STOPWORDS = set(["the", "a", "an", "is", "in", "at", "of", "on", "and", "to"])

# Generalized URL priority subpaths - data-agnostic structural signals.
# Shorter, more structured reference pages are generally more useful than
# deep news archives or user home-pages for factoid QA.
PRIORITY_SUBPATHS = [
    "/people/", "/staff/", "/faculty/", "/leadership/",
    "/courses/", "/academics/", "/undergraduate/", "/graduate/",
    "/research/", "/Research/", "/resources/", "/honors/",
    "/awards/", "/Awards/", "/contact/", "/about/",
    "/book/", "/Directories/", "/facilities/", "/safety/",
]

# Paths that are typically low-signal for factoid QA
DEPRIORITY_SUBPATHS = [
    "/news/page/", "/category/", "/tag/",
    "/wp-content/", "/Pubs/TechRpts/",
]


def preprocess_text(text):
    """Tokenize and remove stopwords for BM25."""
    words = text.lower().split()
    return [w for w in words if w not in STOPWORDS]


def url_to_context(url):
    """Extract meaningful context words from a URL to prepend to chunks."""
    try:
        path = url.split("//")[-1].split("/", 1)[-1] if "//" in url else url
        path = path.replace(".html", "").replace(".txt", "").replace(".php", "")
        parts = []
        for seg in path.replace("/", " ").replace("_", " ").replace("-", " ").split():
            seg = seg.strip()
            if len(seg) > 1 and not seg.startswith("#") and not seg.startswith("?"):
                parts.append(seg)
        return " ".join(parts)
    except:
        return ""


def compute_url_priority(url):
    """
    Generalized, data-agnostic URL priority score.
    Based on URL depth (shorter = more authoritative) and structural path patterns.
    Returns a float in [0.5, 1.5] - stored per chunk for runtime use.
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        segments = [s for s in path.split("/") if s]
        depth = len(segments)
    except:
        return 1.0

    # Base score: shorter URLs are slightly more authoritative
    if depth <= 1:
        score = 1.2
    elif depth == 2:
        score = 1.1
    elif depth == 3:
        score = 1.0
    else:
        score = 0.95

    # Boost for structural reference pages
    for p in PRIORITY_SUBPATHS:
        if p in url:
            score = min(score + 0.2, 1.5)
            break

    # Penalty for generic archive/listing pages
    for p in DEPRIORITY_SUBPATHS:
        if p in url:
            score = max(score - 0.2, 0.5)
            break

    return round(score, 2)


def build_offline_indices():
    print("Loading corpus and chunking...")
    files = glob.glob("corpus/*.txt")
    print(f"Found {len(files)} corpus files.")

    all_chunks = []

    for file_path in files:
        file_size = os.path.getsize(file_path)
        if file_size < MIN_FILE_BYTES:
            continue

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            if not lines:
                continue

            url = lines[0].replace("URL:", "").strip()
            url_context = url_to_context(url)
            url_priority = compute_url_priority(url)
            text = " ".join([line.strip() for line in lines[1:]])
            words = text.split()

            # Sliding window with overlap
            for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
                chunk_words = words[i:i + CHUNK_SIZE]
                if len(chunk_words) < 30:
                    continue
                chunk_text = " ".join(chunk_words)
                enriched_text = f"[{url_context}] {chunk_text}" if url_context else chunk_text
                all_chunks.append({
                    "url": url,
                    "text": enriched_text,
                    "priority": url_priority,
                })

    print(f"Created {len(all_chunks)} chunks. Building FAISS index...")

    embedder = SentenceTransformer(MODEL_NAME)
    chunk_texts = [c["text"] for c in all_chunks]

    # Use IndexFlatIP (Inner Product) with L2-normalized embeddings.
    # Normalized IP == cosine similarity, and IP is slightly faster.
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)

    for start in range(0, len(chunk_texts), ENCODE_BATCH):
        end = min(start + ENCODE_BATCH, len(chunk_texts))
        batch = chunk_texts[start:end]
        embs = embedder.encode(batch, show_progress_bar=False).astype('float32')
        # L2-normalize so IP search == cosine similarity
        faiss.normalize_L2(embs)
        faiss_index.add(embs)
        if (start // ENCODE_BATCH) % 20 == 0:
            print(f"  Encoded {end}/{len(chunk_texts)} chunks...")

    print(f"FAISS index size: {faiss_index.ntotal} vectors")

    del embedder
    gc.collect()

    print("Building BM25 index...")

    tokenized_corpus = [preprocess_text(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    del tokenized_corpus
    gc.collect()

    print("Saving artifacts...")
    os.makedirs("artifacts", exist_ok=True)

    faiss.write_index(faiss_index, "artifacts/index.faiss")
    with open("artifacts/bm25_model.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open("artifacts/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    faiss_size = os.path.getsize("artifacts/index.faiss") / (1024 * 1024)
    bm25_size = os.path.getsize("artifacts/bm25_model.pkl") / (1024 * 1024)
    chunks_size = os.path.getsize("artifacts/chunks.pkl") / (1024 * 1024)
    print(f"Artifact sizes: index.faiss={faiss_size:.1f}MB, "
          f"bm25_model.pkl={bm25_size:.1f}MB, chunks.pkl={chunks_size:.1f}MB")
    print(f"Total: {faiss_size + bm25_size + chunks_size:.1f}MB")
    print("Offline indexing complete!")


if __name__ == "__main__":
    build_offline_indices()
