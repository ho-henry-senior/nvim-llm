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
        chunks.append(
            {
                "source": filename,
                "text": chunk,
            }
        )

print("Number of chunks:", len(chunks))


# 2. Load the embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True,
)


# 3. Embed the documentation
chunk_texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(chunk_texts)


def retrieve(question, top_k=3):
    question_embedding = model.encode(question)
    scores = cos_sim(question_embedding, embeddings)[0].clone()

    for i, chunk in enumerate(chunks):
        if "nvim_create_autocmd" in chunk["text"]:
            scores[i] += 0.3

    best_matches = scores.argsort(descending=True)[:top_k]

    return [chunks[int(i)]["text"] for i in best_matches]

if __name__ == "__main__":
    question = "How do I create an autocmd in Lua?"

    for chunk in retrieve(question):
        print(chunk[:300])
        print("---")
