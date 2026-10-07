from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

with open("docs/lua.txt") as f:
    docs = f.read()

chunk_size = 1000

chunks = [
    docs[i:i + chunk_size]
    for i in range(0, len(docs), chunk_size)
]

model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed the documentation
embeddings = model.encode(chunks)

# Embed our question
question = "How do I create an autocmd in Lua?"
question_embedding = model.encode(question)

# Compare the question with every chunk
scores = cos_sim(question_embedding, embeddings)[0]

# Find the 3 highest scores
best_matches = scores.argsort(descending=True)[:3]

for index in best_matches:
    print("\n--- MATCH ---")
    print("Score:", scores[index].item())
    print(chunks[index])
