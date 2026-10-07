import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_chunks(text, max_size=1500):
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_size:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


files = [
    "docs/lua.txt",
    "docs/api.txt",
]

docs = ""

for filename in files:
    with open(filename) as f:
        docs += f.read()

chunks = create_chunks(docs)
print("Number of chunks:", len(chunks))

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True,
)

# Embed the documentation
embeddings = model.encode(chunks)

# Embed our question
question = "How do I create an autocmd in Lua?"
question_embedding = model.encode(question)

# Compare the question with every chunk
scores = cos_sim(question_embedding, embeddings)[0]

# Find the 3 highest scores
best_matches = scores.argsort(descending=True)[:5]

for index in best_matches:
    print("\n--- RETRIEVED ---")
    print("Score:", scores[index].item())
    print(chunks[index])


context = "\n\n".join(chunks[index] for index in best_matches)


prompt = f"""
You are a Neovim documentation assistant.

Answer the question using ONLY the documentation provided below.

Rules:
- Do not use knowledge that is not present in the documentation.
- Do not invent API names, parameters, fields, or behaviour.
- If the documentation does not contain enough information to answer,
  say: "The provided documentation does not contain enough information."
- Include relevant code examples when the documentation supports them.

DOCUMENTATION:

{context}

QUESTION:

{question}
"""

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True,
)

qwen = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    local_files_only=True,
).to("mps")


messages = [
    {
        "role": "user",
        "content": prompt,
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
).to("mps")

outputs = qwen.generate(
    **inputs,
    max_new_tokens=300,
)

generated = outputs[0][inputs.input_ids.shape[1] :]

answer = tokenizer.decode(
    generated,
    skip_special_tokens=True,
)

print(answer)
