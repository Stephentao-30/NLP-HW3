import os
import sys
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import concurrent.futures
import gc
import re

from llm import call_llm, ALLOWED_MODELS

STOPWORDS = set(["the", "a", "an", "is", "in", "at", "of", "on", "and", "to"])


def preprocess_text(text):
    words = text.lower().split()
    return [w for w in words if w not in STOPWORDS]


def load_artifacts(current_dir):
    faiss_index = faiss.read_index(os.path.join(current_dir, "artifacts", "index.faiss"))
    with open(os.path.join(current_dir, "artifacts", "bm25_model.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    with open(os.path.join(current_dir, "artifacts", "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Pre-build a map: for each URL, list of chunk indices in order
    url_groups = {}
    for i, c in enumerate(chunks):
        url = c.get("url", "")
        if url not in url_groups:
            url_groups[url] = []
        url_groups[url].append(i)

    gc.collect()
    return faiss_index, bm25, chunks, embedder, url_groups


_chunk_texts_cache = None


def hybrid_search(query, faiss_index, bm25, chunks, embedder, url_groups,
                  faiss_k=50, bm25_k=50, final_k=7):
    """
    Weighted RRF fusion of FAISS (semantic) + BM25 (lexical),
    with generalized URL priority scoring and adjacent chunk expansion.
    """
    global _chunk_texts_cache
    if _chunk_texts_cache is None:
        _chunk_texts_cache = [c["text"] for c in chunks]

    RRF_K = 60

    # --- FAISS (semantic) --- weight 2x
    query_vector = embedder.encode([query]).astype('float32')
    faiss.normalize_L2(query_vector)  # Must normalize for IndexFlatIP
    distances, faiss_indices = faiss_index.search(query_vector, faiss_k)

    rrf_scores = {}
    for rank, idx in enumerate(faiss_indices[0]):
        if idx < len(chunks):
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 2.0 / (RRF_K + rank + 1)

    # --- BM25 (lexical) --- weight 1x
    tokenized_query = preprocess_text(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[::-1][:bm25_k]

    for rank, idx in enumerate(bm25_top):
        rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (RRF_K + rank + 1)

    # --- Generalized URL priority boost (data-agnostic) ---
    # Uses pre-computed priority scores from build_index.py based on
    # URL depth and structural path patterns. No query-specific keywords.
    for idx in rrf_scores:
        priority = chunks[idx].get("priority", 1.0)
        rrf_scores[idx] *= priority

    # --- Pick top candidates with URL diversity (max 2 per URL) ---
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    seen_urls = {}
    top_chunks = []
    for idx, score in ranked:
        url = chunks[idx].get("url", "")
        url_count = seen_urls.get(url, 0)
        if url_count < 2:
            top_chunks.append(idx)
            seen_urls[url] = url_count + 1
        if len(top_chunks) >= final_k:
            break

    # --- Adjacent chunk expansion: for top-3 chunks, include neighbours ---
    expanded = set(top_chunks)
    for idx in top_chunks[:3]:
        url = chunks[idx].get("url", "")
        siblings = url_groups.get(url, [])
        pos = -1
        for j, s in enumerate(siblings):
            if s == idx:
                pos = j
                break
        if pos >= 0:
            if pos + 1 < len(siblings):
                expanded.add(siblings[pos + 1])
            if pos - 1 >= 0:
                expanded.add(siblings[pos - 1])

    # Build final context: top chunks first, then expanded neighbours
    result_indices = list(top_chunks)
    for idx in expanded:
        if idx not in result_indices:
            result_indices.append(idx)

    combined_context = [_chunk_texts_cache[idx] for idx in result_indices]
    return " ".join(combined_context)


# ──────────────────── System Prompt ────────────────────
# All answer-shaping is done here via zero-shot/few-shot instructions
# instead of Python post-processing heuristics.

SYSTEM_PROMPT = (
    "You are a factoid QA extraction API for UC Berkeley EECS.\n"
    "Output ONLY the minimal answer entity. No explanation, no filler.\n"
    "\n"
    "FORMAT RULES (follow strictly):\n"
    "- YES/NO questions: answer exactly 'Yes' or 'No'.\n"
    "- Person questions: full name as it appears in context (e.g. 'Dan Garcia').\n"
    "- Count questions: a single digit (e.g. '3' not 'Three faculty').\n"
    "- Number/unit questions: just the number (e.g. '24' not '24 units').\n"
    "- GPA questions: just the decimal number (e.g. '3.7').\n"
    "- Grade questions: just the letter grade exactly as shown (e.g. 'B-').\n"
    "- Email questions: just the full email address.\n"
    "- Room/location questions: room number + building (e.g. '253 Cory Hall').\n"
    "- Lab/center questions: use the short acronym if one appears in context "
    "(e.g. 'RDI' not 'Berkeley Center for Responsible, Decentralized Intelligence').\n"
    "- Date questions: use the exact format from the context.\n"
    "- Award questions: match BOTH the exact award name AND year. "
    "Find 'AwardName: Person, Year' or 'Person, Year' under the correct heading.\n"
    "- Strip filler words like 'A minimum of', 'at least', 'or higher', 'required'.\n"
    "- Copy the answer VERBATIM from the context. Do not paraphrase.\n"
    "- If the answer is not in the context: 'Unsure about answer'"
)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 main.py <questions_path> <output_path>")
        sys.exit(1)

    questions_path = sys.argv[1]
    output_path = sys.argv[2]
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load all artifacts once; BM25 stays in memory for the whole run
    faiss_index, bm25, chunks, embedder, url_groups = load_artifacts(current_dir)
    gc.collect()

    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f.readlines()]

    fast_model = ALLOWED_MODELS[0]

    def fetch_answer(q):
        """Retrieve context + call LLM. Entire pipeline is inside try/except."""
        try:
            context = hybrid_search(
                q, faiss_index, bm25, chunks, embedder, url_groups,
                faiss_k=50, bm25_k=50, final_k=7,
            )
            user_query = f"Context: {context}\nQuestion: {q}\nExtract the exact 1-5 word answer:"
            ans = call_llm(
                query=user_query,
                system_prompt=SYSTEM_PROMPT,
                model=fast_model,
                max_tokens=15,
            )
            # Minimal cleanup: strip whitespace, ensure single-line
            clean = ans.strip(' .\n"').replace("\n", " ").strip()
            if not clean or "unsure" in clean.lower():
                clean = "Unsure about answer"
            return q, clean, context
        except Exception as e:
            return q, "Unsure about answer", ""

    start_time = time.time()

    results_dict = {}
    contexts_dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_q = {executor.submit(fetch_answer, q): q for q in questions}
        for future in concurrent.futures.as_completed(future_to_q):
            q, ans, ctx = future.result()
            results_dict[q] = ans
            contexts_dict[q] = ctx

    end_time = time.time()

    # Write outputs preserving question order
    answers = []
    contexts_saved = []
    for q in questions:
        answers.append(results_dict.get(q, "Unsure about answer"))
        contexts_saved.append(contexts_dict.get(q, ""))

    with open(output_path, 'w', encoding='utf-8') as f:
        for ans in answers:
            # Guarantee one answer per line — no embedded newlines
            f.write(f"{ans.replace(chr(10), ' ')}\n")

    with open('contexts.txt', 'w', encoding='utf-8') as f:
        for c in contexts_saved:
            f.write(f"{c.replace(chr(10), ' ')}\n")

    avg_latency = (end_time - start_time) / len(questions)
    print(f"Average latency per question: {avg_latency:.3f} seconds")


if __name__ == "__main__":
    main()
