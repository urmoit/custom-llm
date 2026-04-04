# Custom LLM (CLI)

A lightweight local Python assistant powered by a **fully custom GPT-style transformer** —
no pre-trained weights, no model downloads, everything trained from scratch on your own knowledge base.

## What makes it fully custom?

| Component | What it is |
|-----------|-----------|
| **Tokenizer** | Custom word-level tokenizer — vocab built from your corpus |
| **Embeddings** | Learnable token + positional embeddings initialised from scratch |
| **Attention** | Custom multi-head masked self-attention (raw tensor math, no `nn.MultiheadAttention`) |
| **Transformer blocks** | Pre-norm GPT-style blocks: LayerNorm → MHA → residual + LayerNorm → FFN → residual |
| **Training** | Cross-entropy next-token prediction, AdamW optimiser, gradient clipping |
| **Generation** | Top-k temperature sampling with causal mask |
| **Retrieval** | TF-IDF index for context injection (RAG) |

## Features
- Fully custom transformer LLM trained purely from your `knowledge/` files
- Retrieval-augmented generation: TF-IDF retrieves context, custom LLM generates answers
- Local-only — no API calls, no model downloads
- Basic web search fallback (DuckDuckGo HTML + page snippets) for current events
- Easy training flow from JSONL examples
- Info-to-training pipeline from `knowledge/*.md`
- Windows batch scripts for setup, training, and chat

## Project structure
- `src/custom_llm/model.py` - Custom GPT-style transformer (all from scratch)
- `src/custom_llm/tokenizer.py` - Custom word-level tokenizer
- `src/custom_llm/trainer.py` - Training loop for the custom LLM + TF-IDF index
- `src/custom_llm/assistant.py` - Retrieval-augmented question answering
- `src/custom_llm/cli.py` - CLI chat entrypoint
- `src/custom_llm/search.py` - Web search helpers
- `src/custom_llm/build_training_data.py` - Builds trainable JSONL from raw knowledge docs
- `data/training_data.jsonl` - Local knowledge examples
- `knowledge/` - Raw knowledge files (Python, USA, chat behaviour, and more)
- `artifacts/` - Generated model files (custom_llm.pt, tokenizer.json, vectorizer.joblib)

## Quick start (Windows)
1. Run `setup.bat`
2. Run `build_info_and_train.bat`
3. Run `chat.bat`
4. Run `chat_ui.bat` to open the browser web UI

## Custom LLM architecture
The model (`src/custom_llm/model.py`) is a ~10M parameter GPT-style decoder-only transformer:

```
vocab_size  : determined from corpus (up to 20 000 tokens)
d_model     : 512
n_heads     : 8   (head_dim = 64)
n_layers    : 6
context_len : 384 tokens
d_ff        : 1024
dropout     : 0.1
epochs      : 8
batch_size  : 4
learning_rate: 2e-4
parameters  : ~10M (varies with vocab size, typically 9.5-10.5M)
```

All attention is implemented from raw tensor multiplications — no wrapper libraries.

**Upgraded from original 3.9M parameter model** to provide richer knowledge representation,
better reasoning capabilities, and more natural conversational ability.

## Knowledge-first training
Instead of manually writing Q/A lines, add facts to markdown files in `knowledge/`.

Then run:
- `build_info_and_train.bat`

This will:
1. Convert `knowledge/*.md` into knowledge chunks in `data/training_data.jsonl`
2. Build the custom tokenizer vocabulary from those chunks
3. Train the custom LLM (next-token prediction) on the chunks
4. Build the TF-IDF retrieval index alongside the LLM (used for RAG)

## Expanded knowledge domains
The knowledge base now covers **25+ topic areas** including:

**Core subjects:**
- Science: Physics, chemistry, biology, mathematics, climate
- Technology: AI/ML, software engineering, databases, networking, cloud
- History: Ancient civilizations, world wars, modern era
- Geography: World geography, countries, regions
- Health: Medicine, nutrition, fitness, mental health
- Philosophy: Ethics, logic, epistemology, key thinkers

**New expanded areas:**
- Mathematics & Statistics: Algebra, calculus, probability, statistics
- Psychology: Cognitive biases, memory, motivation, social psychology
- Space & Astronomy: Solar system, stars, galaxies, exploration
- Personal Finance: Budgeting, investing, retirement, debt management
- Food & Cooking: Techniques, world cuisines, baking, food safety
- Literature & Writing: Classic works, poetry, narrative structure
- Music & Sound: Theory, history, production
- Art & Visual Culture: Renaissance to modern art, design, film
- Environment & Ecology: Biodiversity, climate, conservation
- Law & Justice: Constitutional, criminal, civil law basics
- Education & Learning: Study techniques, critical thinking
- Sports & Athletics: Major sports, training, Olympics
- Travel & Culture: Customs, festivals, sustainable tourism
- Relationships & Communication: Active listening, conflict resolution

## Conversational improvements
The assistant now handles **natural dialogue** without requiring web search for:
- Greetings, small talk, and farewells
- Clarification requests and follow-ups
- Agreement/understanding checks ("ok", "got it", "makes sense")
- Confusion signals ("I don't understand", "can you explain")
- Casual conversation patterns
- Yes/no responses to previous questions
- Expressions of interest ("interesting", "cool", "fascinating")
- Grammar cleanup requests
- Capability and knowledge domain questions

This enables fluid multi-turn conversations without triggering unnecessary web searches.

## Training backends

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `auto`  | Custom LLM if PyTorch available, else TF-IDF | `torch` |
| `custom` | Fully custom transformer from scratch | `torch` |
| `tfidf` | TF-IDF retrieval only (no generation) | — |
| `transformer` | GPU embedding with sentence-transformers | `torch` + CUDA |

## GPU training (optional)
If you want GPU-accelerated training:

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
{"topic":"Python","section":"Overview","text":"Topic: Python\nSection: Overview\nPython is ..."}
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

