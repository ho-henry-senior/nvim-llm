import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag import retrieve

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEVICE = "mps"

tests = [
    {
        "question": "How do I run Lua code every time I enter a Python buffer?",
        "expected": ["nvim_create_autocmd", "BufEnter", "*.py", "callback"],
    },
    {
        "question": "How do I run a Lua function when a Markdown file is saved?",
        "expected": ["nvim_create_autocmd", "BufWritePost", "*.md", "callback"],
    },
    {
        "question": "How do I run Lua code when a new JavaScript file is created?",
        "expected": ["nvim_create_autocmd", "BufNewFile", "*.js", "callback"],
    },
    {
        "question": "How do I run Lua code before a Python file is saved?",
        "expected": ["nvim_create_autocmd", "BufWritePre", "*.py", "callback"],
    },
    {
        "question": "How do I run Lua code after reading a Markdown file?",
        "expected": ["nvim_create_autocmd", "BufReadPost", "*.md", "callback"],
    },
]

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

all_results = []


def evaluate(label, adapter_path=None, use_rag=False):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        local_files_only=True,
    ).to(DEVICE)

    if adapter_path:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=True,
        )

    model.eval()

    passed = 0
    total_components = 0
    correct_components = 0

    print(f"\n--- {label} ---")

    for index, test in enumerate(tests, start=1):
        context = ""

        if use_rag:
            relevant_chunks = retrieve(test["question"])
            print(f"\nRetrieved documentation for: {test['question']}")

            for i, chunk in enumerate(relevant_chunks, start=1):
                print(f"\nChunk {i}:")
                print(chunk[:500])
            context = "\n\n".join(relevant_chunks)

        system_prompt = (
            "You are a Neovim Lua configuration expert. "
            "Answer using Neovim's Lua API, not Vimscript. "
            "Provide a working Lua example."
        )

        if use_rag:
            system_prompt += (
                "\n\nUse the following Neovim documentation "
                "to answer the question. "
                "Do not invent APIs or events.\n\n" + context
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test["question"]},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        answer = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        missing = [item for item in test["expected"] if item not in answer]

        correct = len(test["expected"]) - len(missing)
        total = len(test["expected"])

        success = correct == total
        passed += int(success)

        correct_components += correct
        total_components += total

        print(
            f"Test {index}: "
            f"{'PASS' if success else 'FAIL'} "
            f"| Components: {correct}/{total} "
            f"| Missing: {missing}"
        )

        all_results.append(
            {
                "model": label,
                "question": test["question"],
                "answer": answer,
                "expected": test["expected"],
                "missing": missing,
                "passed": success,
                "component_score": correct / total,
                "use_rag": use_rag,
            }
        )

    print(f"\n{label} results:")
    print(f"  Exact passes: {passed}/{len(tests)}")
    print(f"  Component accuracy: {correct_components / total_components:.1%}")
    del model
    gc.collect()
    torch.mps.empty_cache()


evaluate("Original Qwen")

evaluate(
    "LoRA v3",
    "models/neovim-lora-v3",
)

evaluate(
    "Original Qwen + RAG",
    use_rag=True,
)

evaluate(
    "LoRA v3 + RAG",
    "models/neovim-lora-v3",
    use_rag=True,
)

Path("data").mkdir(exist_ok=True)

with open("data/evaluation_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nSaved results to data/evaluation_results.json")
