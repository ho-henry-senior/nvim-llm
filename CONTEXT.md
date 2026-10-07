# LLM Project Terms

Shared terminology for the Neovim configuration assistant.

## Language

**LLM (Large Language Model)**:
A model such as Qwen that generates text or code by predicting what should come
next.

**Embedding**:
A numerical representation of text that captures its meaning. Similar text
produces similar vectors.

**Vector**:
The list of numbers representing an embedding.

**Semantic search**:
Searching by meaning rather than exact words, by comparing embeddings.

**RAG (Retrieval-Augmented Generation)**:
Searching an external source, such as Neovim documentation, for relevant
information and giving it to the LLM as context before it answers. The LLM itself
isn't changed.

**LoRA (Low-Rank Adaptation)**:
An efficient way to fine-tune an existing LLM by training a small set of
additional parameters while leaving the original model mostly unchanged.
