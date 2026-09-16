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

## 2. Baseline 

### 2.1. AMem4Rec baseline
Build global memory

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

### 2.2 AMem4Rec baseline

Run recommendation with the constructed memory:

```bash
python inference_amem.py \
  --global_memory agent_memory/CDs/global_memory.json \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/user_negatives_10_5000.json \
  --output results/CDs/amem4rec.json \
  --k_memories 3 \
  --history_size 10 \
  --number_of_users 1000
```

### 2.3 NativeLLM baseline

NativeLLM uses the same candidate set and ranking LLM, but **does not retrieve memory**.

```bash
python inference_amem.py \
  --global_memory unused.json \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/user_negatives_10_5000.json \
  --output results/CDs/native_llm.json \
  --history_size 10 \
  --number_of_users 1000 \
  --no_memory
```

`unused.json` does not need to exist when `--no_memory` is enabled.


## 3. Tree Construction
Clustering
```bash
python build_reverse_behavior_tree_cluster.py \
  --input precomputed/CDs/local_memories_gemma.jsonl \
  --output-dir behavior_tree_out_cluster_k50 \
  --cluster-text-field pattern_description \
  --num-clusters 50 \
  --encoder qwen \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --max-order 5 \
  --count-mode user_normalized \
  --min-support-users 3 \
  --min-support-occurrences 3 \
  --min-jsd 0.05 \
  --smoothing-kappa 5
```


## 4. Inferencing

### 4.1 Generate behavior for test user

```bash
python precompute_test_behaviors_gemma.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --model unsloth/gemma-3-4b-it-unsloth-bnb-4bit \
  --window-size 3 \
  --max-train-interactions 30 \
  --recent-behaviors 5 \
  --max-users 300 \
  --output precomputed/CDs/test_user_behaviors_gemma.jsonl \
  --summary-output precomputed/CDs/test_user_behaviors_gemma.summary.json \
  --resume


python precompute_test_behaviors_gemma_batch.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --model unsloth/gemma-3-4b-it-unsloth-bnb-4bit \
  --window-size 3 \
  --max-train-interactions 10 \
  --batch-size 8 \
  --recent-behaviors 5 \
  --max-users 300 \
  --output precomputed/CDs/test_user_behaviors_gemma.jsonl \
  --summary-output precomputed/CDs/test_user_behaviors_gemma.summary.json \
  --resume
```

### 4.2 Testing

```bash
python infer_tree_gemma_precom.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/user_negatives_10_5000.json \
  --precomputed-behaviors precomputed/CDs/test_user_behaviors_gemma.jsonl \
  --tree-dir behavior_tree_out_cluster_k50 \
  --state-mode cluster \
  --model unsloth/gemma-3-4b-it-unsloth-bnb-4bit \
  --top-next 3 \
  --max-users 10 \
  --run-baseline \
  --output results/tree_cluster50_precomputed_test10.jsonl \
  --summary-output results/tree_cluster50_precomputed_test10_summary.json
```

## Notes

- Default ranking model: `unsloth/gemma-3-4b-it-unsloth-bnb-4bit`
- Default embedding model: `Qwen/Qwen3-Embedding-0.6B`
- Use the same embedding model for global-memory construction and memory-based inference.
- All scripts support resume by default.
