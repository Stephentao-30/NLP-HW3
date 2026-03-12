"""Debug retrieval for specific questions to see which chunks are returned."""
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import os

STOPWORDS = set(["the", "a", "an", "is", "in", "at", "of", "on", "and", "to"])

def preprocess_text(text):
    words = text.lower().split()
    return [w for w in words if w not in STOPWORDS]

# Load artifacts
current_dir = os.path.dirname(os.path.abspath(__file__))
faiss_index = faiss.read_index(os.path.join(current_dir, "artifacts", "index.faiss"))
with open(os.path.join(current_dir, "artifacts", "bm25_model.pkl"), "rb") as f:
    bm25 = pickle.load(f)
with open(os.path.join(current_dir, "artifacts", "chunks.pkl"), "rb") as f:
    chunks = pickle.load(f)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

chunk_texts = [c["text"] for c in chunks]

# Debug questions
queries = [
    ("q83", "Which faculty member won the MacArthur Fellow in 2010?"),
    ("q89", "Who won the COSA award in 2014?"),
    ("q98", "Who won the EIM award in 2022?"),
    ("q96", "Who won the Frontiers of Engineering Symposium invitation in 2021?"),
    ("q91", "Who won the Okawa Prize in 2013?"),
    ("q84", "Who won the National Academy of Engineering in 2024?"),
]

for qid, query in queries:
    print(f"\n{'='*80}")
    print(f"{qid}: {query}")
    
    # FAISS
    qv = embedder.encode([query]).astype('float32')
    dists, idxs = faiss_index.search(qv, 20)
    
    print(f"\n  FAISS top 10:")
    for rank, idx in enumerate(idxs[0][:10]):
        url = chunks[idx].get("url", "")
        text_snippet = chunk_texts[idx][:100]
        print(f"    {rank}: [{url}] {text_snippet}...")
    
    # BM25
    tok = preprocess_text(query)
    scores = bm25.get_scores(tok)
    bm25_top = np.argsort(scores)[::-1][:20]
    
    print(f"\n  BM25 top 10:")
    for rank, idx in enumerate(bm25_top[:10]):
        url = chunks[idx].get("url", "")
        text_snippet = chunk_texts[idx][:100]
        score = scores[idx]
        print(f"    {rank}: [{url}] (score={score:.1f}) {text_snippet}...")
