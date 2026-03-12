from rank_bm25 import BM25Okapi

# 1. Prepare your data
# 'chunks' is a list of your 100-word text strings
# Example: ["Professor Dan Klein's office is in Soda Hall...", "The EECS department..."]
tokenized_corpus = [doc.split(" ") for doc in chunks]

# 2. Initialize the BM25 "Search Engine"
bm25 = BM25Okapi(tokenized_corpus)

# 3. Search for a question
query = "What is the office number of Dan Klein?"
tokenized_query = query.split(" ")

# Get the scores for every chunk relative to the query
doc_scores = bm25.get_scores(tokenized_query)

# Get the top 3 most relevant chunks based on keyword matching
top_chunks = bm25.get_top_n(tokenized_query, chunks, n=3)