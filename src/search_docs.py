from sentence_transformers import SentenceTransformer

# Load our documentation
with open("docs/lua.txt") as f:
    docs = f.read()

# Crude chunking, same as before
chunk_size = 1000

chunks = [docs[i : i + chunk_size] for i in range(0, len(docs), chunk_size)]

# Load a small embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Turn every chunk into a vector
embeddings = model.encode(chunks)

print("Chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print()
print("First embedding:")
print(embeddings[0][:10])
