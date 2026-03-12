import pickle

with open("artifacts/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Find chunks that mention Dawn Song + MacArthur
print("=== Chunks with Dawn Song + MacArthur ===")
for i, c in enumerate(chunks):
    text_lower = c["text"].lower()
    if "macarthur" in text_lower and "dawn song" in text_lower:
        url = c.get("url", "")
        print(f"Chunk {i}: url={url}")
        print(f"  Text: {c['text'][:300]}...")
        print()

# Find chunks with COSA + 2014
print("\n=== Chunks with COSA + 2014 ===")
for i, c in enumerate(chunks):
    text_lower = c["text"].lower()
    if "cosa" in text_lower and "2014" in text_lower:
        url = c.get("url", "")
        print(f"Chunk {i}: url={url}")
        print(f"  Text: {c['text'][:300]}...")
        print()

# Find chunks with Audrey Sillers
print("\n=== Chunks with Audrey Sillers ===")
for i, c in enumerate(chunks):
    text_lower = c["text"].lower()
    if "audrey sillers" in text_lower:
        url = c.get("url", "")
        print(f"Chunk {i}: url={url}")
        print(f"  Text: {c['text'][:300]}...")
        print()

# Find chunks with Carissa Caloud
print("\n=== Chunks with Carissa Caloud ===")
for i, c in enumerate(chunks):
    text_lower = c["text"].lower()
    if "carissa" in text_lower:
        url = c.get("url", "")
        print(f"Chunk {i}: url={url}")
        print(f"  Text: {c['text'][:300]}...")
        print()

# Find chunks with gradadmissions@
print("\n=== Chunks with gradadmissions@ ===")
for i, c in enumerate(chunks):
    if "gradadmissions@" in c["text"].lower():
        url = c.get("url", "")
        print(f"Chunk {i}: url={url}")
        print(f"  Text: {c['text'][:300]}...")
        print()

# Find chunks with Eric Fraser
print("\n=== Chunks with Eric Fraser ===")
for i, c in enumerate(chunks):
    if "eric fraser" in c["text"].lower():
        url = c.get("url", "")
        print(f"Chunk {i}: url={url}")
        print(f"  Text: {c['text'][:300]}...")
        print()
