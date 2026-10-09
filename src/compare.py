import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

question = (
    "How do I run a Lua function whenever "
    "a Markdown file is opened in Neovim?"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True,
)

messages = [
    {"role": "user", "content": question},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to("mps")


def test_model(label, adapter_path=None):
    # Load a fresh copy of Qwen
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        local_files_only=True,
    ).to("mps")

    if adapter_path:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=True,
        )

    model.eval()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    print(f"\n--- {label} ---")
    print(answer)

    # Free memory before loading the next version
    del model
    gc.collect()
    torch.mps.empty_cache()


test_model("Original Qwen")

test_model(
    "LoRA v1",
    "models/neovim-lora",
)

test_model(
    "LoRA v2",
    "models/neovim-lora-v2",
)
