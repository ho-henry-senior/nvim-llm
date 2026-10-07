import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoModelForCausalLM, AutoTokenizer

files = [
    "docs/lua.txt",
    "docs/api.txt",
]

docs = ""

for filename in files:
    with open(filename) as f:
        docs += f.read()

chunk_size = 1000

chunks = [docs[i : i + chunk_size] for i in range(0, len(docs), chunk_size)]

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
best_matches = scores.argsort(descending=True)[:3]

# for index in best_matches:
#     print("\n--- MATCH ---")
#     print("Score:", scores[index].item())
#     print(chunks[index])

context = "\n\n".join(chunks[index] for index in best_matches)


prompt = f"""
Answer the question using the Neovim documentation below.

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
