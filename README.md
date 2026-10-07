# nvim-llm

An early prototype of a local LLM assistant for Neovim configuration. It currently
runs a hard-coded prompt through Qwen2.5-Coder and includes a small Lua-docs
reading experiment.

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

## Project files

- `src/run.py` — loads the model and generates a response to an example Neovim
  keymap prompt.
- `src/check_gpu.py` — reports the installed PyTorch version and MPS availability.
- `src/read_docs.py` — reads and previews `docs/lua.txt`.
- `docs/` — Vim and Neovim reference documentation. `docs/lua.txt` contains the
  Lua documentation.
