import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

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
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.train()

# Our first training example
example = """Question: How do I create an autocmd in Neovim Lua?
Answer: Use vim.api.nvim_create_autocmd to register an autocommand."""

inputs = tokenizer(
    example,
    return_tensors="pt",
).to("mps")

# Find one trainable LoRA weight
name, weight = next(
    (name, param) for name, param in model.named_parameters() if "lora_B" in name
)

before = weight.detach().clone()

# Calculate prediction error
# Set up the optimizer
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=0.001,
)

# Train for 20 steps
for step in range(20):
    optimizer.zero_grad()

    outputs = model(
        **inputs,
        labels=inputs["input_ids"],
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"Step {step + 1:2d} | Loss: {loss.item():.4f}")

# Check how much our selected weight changed
difference = (weight.detach() - before).abs().max().item()
print("Largest weight change:", difference)

# Save the trained LoRA weights
model.save_pretrained("models/neovim-lora")

print("Saved LoRA adapter to models/neovim-lora")
