import json

with open("data/train.jsonl") as f:
    examples = [json.loads(line) for line in f]

print("Training examples:", len(examples))

for example in examples:
    print("\nQuestion:", example["question"])
    print("Answer:", example["answer"])
