# CoMemTree

Beyond Flat Memory: Collaborative Memory Trees for LLM-based Recommendation


## Install

```bash
pip install -U unsloth sentence-transformers faiss-cpu tqdm
```

CUDA GPU is required for the local Gemma/Unsloth inference.

## 1. Precompute local memories

```bash
python precompute_memory_create.py \
  --data_dir data/CDs \
  --sequences_file user_sequences_10_5000.json \
  --items_file items.json \
  --output precomputed/CDs/local_memories.jsonl \
  --number_of_users 1000 \
  --window_size 3 \
  --llm_batch_size 8 \
  --max_new_tokens 256
```

Use `--number_of_users 0` to process all users.

## 2. Build global memory

```bash
python build_global_memory.py \
  --precomputed precomputed/CDs/local_memories.jsonl \
  --output agent_memory/CDs/global_memory.json \
  --embedding_model Qwen/Qwen3-Embedding-0.6B \
  --embedding_batch_size 64 \
  --low_threshold 0.65 \
  --high_threshold 0.80 \
  --link_size 5 \
  --max_evolutions_per_memory 10 \
  --max_new_tokens 256 \
  --save_every 100
```

## 3. AMem4Rec baseline

Run recommendation with the constructed memory:

```bash
python inference_amem.py \
  --global_memory agent_memory/CDs/global_memory.json \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/<negative_file>.json \
  --output results/CDs/amem4rec.json \
  --k_memories 3 \
  --history_size 10 \
  --number_of_users 1000
```

## 4. NativeLLM baseline

NativeLLM uses the same candidate set and ranking LLM, but **does not retrieve memory**.

```bash
python inference_amem.py \
  --global_memory unused.json \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/<negative_file>.json \
  --output results/CDs/native_llm.json \
  --history_size 10 \
  --number_of_users 1000 \
  --no_memory
```

`unused.json` does not need to exist when `--no_memory` is enabled.

## Notes

- Default ranking model: `unsloth/gemma-3-4b-it-unsloth-bnb-4bit`
- Default embedding model: `Qwen/Qwen3-Embedding-0.6B`
- Use the same embedding model for global-memory construction and memory-based inference.
- All scripts support resume by default.
