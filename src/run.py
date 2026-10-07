import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to("mps")

messages = [
    {
        "role": "user",
        "content": "Write a simple Neovim keymap in Lua to save the current file.",
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt").to("mps")

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
