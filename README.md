# Custom LLM (CLI)

A lightweight local Python assistant that can:
- answer from a small trainable local knowledge base
- search the web when needed
- run in a simple CLI chat loop
- build trainable data from raw knowledge files
- train with optional GPU embedding backend

## Features
- Local retrieval-based answering from knowledge chunks
- Basic web search fallback (DuckDuckGo HTML + page snippets)
- Easy training flow from JSONL examples
- Info-to-training pipeline from `knowledge/*.md`
- Optional GPU backend with sentence-transformers + torch
- Windows batch scripts for setup, training, and chat

## Project structure
- `src/custom_llm/cli.py` - CLI chat entrypoint
- `src/custom_llm/trainer.py` - train model artifacts
- `src/custom_llm/assistant.py` - question answering orchestration
- `src/custom_llm/search.py` - web search helpers
- `src/custom_llm/build_training_data.py` - builds trainable JSONL from raw knowledge docs
- `data/training_data.jsonl` - local knowledge examples
- `knowledge/` - raw knowledge files (Python, USA, chat behavior)
- `artifacts/` - generated model files

## Quick start (Windows)
1. Run `setup.bat`
2. Run `build_info_and_train.bat`
3. Run `chat.bat`
4. Run `chat_ui.bat` to open the browser web UI

## Knowledge-first training
Instead of manually writing Q/A lines, add facts to markdown files in `knowledge/`.

Then run:
- `build_info_and_train.bat`

This will:
1. Convert `knowledge/*.md` into generated knowledge chunks in `data/training_data.jsonl`
2. Train the local retrieval model from those chunks

## GPU training (optional)
If you want GPU-accelerated embedding training:

1. Run `build_info_and_train_gpu.bat` (build + train)
or
2. Run `train_gpu.bat` (train only)

These scripts enforce:
- PyTorch 2.9.x
- CUDA 13.x runtime (targeting CUDA 13.2 wheels)
- `torch.cuda.is_available() == True`

If these checks fail, the script exits with an error so you can fix CUDA/PyTorch alignment.

## Training data format
Generated rows in `data/training_data.jsonl` are knowledge chunks, for example:

```json
{"topic":"Python","section":"Overview","text":"Topic: Python\\nSection: Overview\\nPython is ..."}
```

## CLI commands
Inside chat:
- `help` - show commands
- `gpu_status` - show CUDA + trained backend status (aliases: `gpu-status`, `gpu status`, `gpu`)
- `search: <query>` - force web search
- `retrain` - retrain using current dataset
- `exit` - quit

## Web UI
Run `chat_ui.bat` to start the local browser UI at `http://127.0.0.1:8787`.
The web UI uses the same local assistant backend as the CLI.
