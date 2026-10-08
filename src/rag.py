
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def create_chunks(text, max_size=1500):
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# 1. Load documentation and create source-aware chunks
files = [
    "docs/lua.txt",
    "docs/api.txt",
]

chunks = []

for filename in files:
    with open(filename) as f:
        docs = f.read()

    file_chunks = create_chunks(docs)

    for chunk in file_chunks:
        chunks.append({
            "source": filename,
            "text": chunk,
        })

print("Number of chunks:", len(chunks))


# 2. Load the embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True,
)


# 3. Embed the documentation
chunk_texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(chunk_texts)


# 4. Test different questions
questions = [
    "How do I create an autocmd in Lua?",
    "How do I create a Neovim autocommand?",
    "How do I use nvim_create_autocmd?",
]


for question in questions:
    # Convert the question into an embedding
    question_embedding = model.encode(question)

    # Calculate semantic similarity scores
    scores = cos_sim(question_embedding, embeddings)[0]

    # Boost chunks containing this exact API name
    keyword = "nvim_create_autocmd"

    for i, chunk in enumerate(chunks):
        if keyword in chunk["text"]:
            scores[i] += 0.3

    # Find the 5 highest-scoring chunks
    best_matches = scores.argsort(descending=True)[:5]

    print(f"\n\nQUESTION: {question}")

    for index in best_matches:
        chunk = chunks[index]

        print("\n--- MATCH ---")
        print("Score:", scores[index].item())
        print("Source:", chunk["source"])
        print(chunk["text"][:300].replace("\n", " "))


# Stop here while testing retrieval.
# We'll reconnect Qwen once we're happy with the results.
