import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

with open("data/train.jsonl") as f:
    examples = [json.loads(line) for line in f]

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    local_files_only=True,
).to("mps")

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.train()

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=0.0001,
)


def prepare_example(example):
    messages = [
        {"role": "user", "content": example["question"]},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    answer = example["answer"] + tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    answer_ids = tokenizer(
        answer,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = prompt_ids + answer_ids

    # -100 tells PyTorch to ignore these tokens when calculating loss
    labels = [-100] * len(prompt_ids) + answer_ids

    return {
        "input_ids": torch.tensor([input_ids], device="mps"),
        "labels": torch.tensor([labels], device="mps"),
    }


training_data = [prepare_example(e) for e in examples]

for epoch in range(3):
    total_loss = 0

    for example in training_data:
        optimizer.zero_grad()

        outputs = model(**example)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} | Average loss: {total_loss / len(training_data):.4f}")

model.save_pretrained("models/neovim-lora-v3")
print("Saved adapter to models/neovim-lora-v3")
