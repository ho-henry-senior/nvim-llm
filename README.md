# nvim-llm

An early prototype of a local LLM assistant for Neovim configuration. It currently
runs a hard-coded prompt through Qwen2.5-Coder and includes a small Lua-docs
reading experiment.

https://chatgpt.com/c/6ac6095a-59d4-83ed-8029-c0b70d5d4a6a

## LLM project terms

See [LLM Project Terms](CONTEXT.md) for definitions of LLM, embedding, vector,
semantic search, RAG, and LoRA. The `all-MiniLM-L6-v2` embeddings used in
`src/search_docs.py` contain 384 numbers per vector.

In short:

- **RAG** gives the model relevant knowledge as context.
- **LoRA** changes the model's learned behaviour.

## RAG flow

The intended question-to-answer flow is:

```text
Question
   ↓
MiniLM creates an embedding
   ↓
Semantic search
   ↓
3 relevant documentation chunks
   ↓
Build a prompt containing the question and retrieved chunks
   ↓
Qwen
   ↓
Answer
```

The documentation chunks are embedded with MiniLM before searching. Semantic
search compares the question's embedding with those chunk embeddings and selects
the three highest-scoring chunks to use as context.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned improvements.

## Requirements

- Python 3.14
- An Apple Silicon Mac with PyTorch MPS support
- Internet access on the first run to download the model from Hugging Face

The project has been set up with PyTorch 2.14.1 and Transformers 5.19.0.

## Setup

From the repository root, create and activate a virtual environment, then install
the dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Check whether PyTorch can use MPS:

```sh
python src/check_gpu.py
```

Run the example prompt:

```sh
python src/run.py
```

The first run downloads `Qwen/Qwen2.5-Coder-1.5B-Instruct` from Hugging Face.
The prompt is currently hard-coded in `src/run.py`; the script does not yet
accept user input or use the Lua documentation as model context.

## Documentation sources

The `docs/` directory holds Vim and Neovim reference documentation. Neovim help
files can be copied from the installed runtime's `doc/` directory.

For a Homebrew installation, run this from the repository root to copy the API
help file:

```sh
cp "$(brew --prefix neovim)/share/nvim/runtime/doc/api.txt" docs/
```

Replace `api.txt` with another help filename, such as `lua.txt`, as needed.
This path follows the currently installed Homebrew version rather than a
specific Cellar directory.

The source installation used for the API help copy is **Neovim 0.12.5**
(Homebrew package `0.12.5_1`). When refreshing files, record the source version
here, since APIs and documentation can differ between releases. Check the
installed version with `nvim --version`.

Preserve upstream copyright and license notices when copying documentation.

## Project files

- `src/run.py` — loads the model and generates a response to an example Neovim
  keymap prompt.
- `src/check_gpu.py` — reports the installed PyTorch version and MPS availability.
- `src/read_docs.py` — reads and previews `docs/lua.txt`.
- `docs/` — Vim and Neovim reference documentation. `docs/lua.txt` contains the
  Lua documentation.
