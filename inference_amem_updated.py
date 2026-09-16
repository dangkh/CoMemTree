#!/usr/bin/env python3
import argparse, json, os, random, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def normalize_items(items: Dict[Any, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in items.items()}


def get_item_info(item_id: Any, items_meta: Dict[str, Any]) -> Dict[str, str]:
    iid = str(item_id)
    info = items_meta.get(iid, {})
    category = info.get("main_cat") or info.get("category") or info.get("categories") or "Unknown"
    if isinstance(category, list):
        category = " > ".join(map(str, category[:3]))
    return {
        "item_id": iid,
        "title": str(info.get("title") or f"Unknown Item {iid}"),
        "category": str(category),
    }


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    s, e = text.find("{"), text.rfind("}") + 1
    if s != -1 and e > s:
        text = text[s:e]
    return json.loads(text)


class FlatMemoryRetriever:
    """Exact cosine retrieval using normalized vectors + FAISS IndexFlatIP."""

    def __init__(self, memory_path: str, embedding_model: str, device: str = "auto"):
        data = load_json(memory_path)
        self.memories = data.get("behavior_memories", [])
        if not self.memories:
            raise RuntimeError("Global memory contains no behavior_memories")

        embs = np.asarray([m["embedding"] for m in self.memories], dtype=np.float32)
        if embs.ndim != 2:
            raise ValueError(f"Invalid embedding matrix shape: {embs.shape}")

        embs = np.ascontiguousarray(embs)
        faiss.normalize_L2(embs)

        self.dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = SentenceTransformer(embedding_model, device=device)
        self.search_calls = 0
        self.total_search_time = 0.0

        print(f"Loaded {len(self.memories)} memories; dim={self.dim}")
        print("Retrieval: FAISS IndexFlatIP (exact cosine)")

    def retrieve(self, profile_text: str, k: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        if k <= 0:
            return [], []

        q = self.encoder.encode(
            profile_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        q = np.ascontiguousarray(np.asarray(q, dtype=np.float32).reshape(1, -1))
        if q.shape[1] != self.dim:
            raise ValueError(
                f"Query dim {q.shape[1]} != memory dim {self.dim}. "
                "Use the same embedding model used to build global memory."
            )
        faiss.normalize_L2(q)

        k = min(k, len(self.memories))
        t0 = time.perf_counter()
        scores, idx = self.index.search(q, k)
        self.total_search_time += time.perf_counter() - t0
        self.search_calls += 1

        memories, sims = [], []
        for i, s in zip(idx[0], scores[0]):
            if i >= 0:
                memories.append(self.memories[int(i)])
                sims.append(float(s))
        return memories, sims

    def stats(self):
        return {
            "num_memories": len(self.memories),
            "search_calls": self.search_calls,
            "total_search_time_sec": round(self.total_search_time, 6),
            "avg_search_time_ms": (
                round(self.total_search_time / self.search_calls * 1000, 4)
                if self.search_calls else 0.0
            ),
        }


class LocalRanker:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 1024,
        max_seq_length: int = 8192,
        load_in_4bit: bool = True,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Unsloth Gemma inference expects a CUDA GPU.")

        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            full_finetuning=False,
        )
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="gemma3",
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = int(max_seq_length)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_time = 0.0
        self.records = []

    @staticmethod
    def build_prompt(user_profile, candidates, memory_thoughts):
        memory_block = ""
        if memory_thoughts:
            memory_block = (
                "\nCollaborative memory (secondary evidence):\n"
                + json.dumps(memory_thoughts, indent=2, ensure_ascii=False)
            )

        n = len(candidates)
        return f"""You are a recommendation ranking system.

Rank ALL candidate items for the target user.

Priority:
1. Match the user's recent preferences and behavioral pattern.
2. Use category and semantic compatibility.
3. Use retrieved collaborative memory as additional evidence when relevant.

User recent history (most recent last):
{json.dumps(user_profile[-10:], indent=2, ensure_ascii=False)}
{memory_block}

Candidate items:
{json.dumps(candidates, indent=2, ensure_ascii=False)}

Requirements:
- Rank ALL {n} candidates.
- Each candidate item_id must appear exactly once.
- Do not invent item IDs.
- Return ONLY valid JSON.

{{"ranked_item_ids": ["id1", "..."], "reasoning": "1 concise sentence"}}"""

    def rank(self, user_id: str, user_profile, candidates, memory_thoughts):
        prompt = self.build_prompt(user_profile, candidates, memory_thoughts)
        messages = [
            {"role": "system", "content": "You are a recommendation ranking system."},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if text.startswith("<bos>"):
            text = text[len("<bos>"):]

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)
        n_in = int(inputs["input_ids"].shape[-1])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen = outputs[0][n_in:]
        response = self.tokenizer.decode(gen, skip_special_tokens=True)

        self.calls += 1
        self.input_tokens += n_in
        self.output_tokens += len(gen)
        self.total_time += elapsed
        self.records.append({
            "user_id": str(user_id),
            "input_tokens": n_in,
            "output_tokens": int(len(gen)),
            "inference_time_sec": round(elapsed, 4),
        })

        valid_ids = [str(x["item_id"]) for x in candidates]
        valid_set = set(valid_ids)

        try:
            obj = parse_json_response(response)
            pred = [str(x) for x in obj.get("ranked_item_ids", [])]
            cleaned, seen = [], set()
            for iid in pred:
                if iid in valid_set and iid not in seen:
                    cleaned.append(iid)
                    seen.add(iid)
            for iid in valid_ids:
                if iid not in seen:
                    cleaned.append(iid)
                    seen.add(iid)
            return cleaned, str(obj.get("reasoning", ""))
        except Exception as e:
            print(f"Warning: parse failed for user {user_id}: {e}")
            return valid_ids, ""

    def stats(self):
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "total_inference_time_sec": round(self.total_time, 4),
            "avg_inference_time_sec": (
                round(self.total_time / self.calls, 4) if self.calls else 0.0
            ),
        }


def run_one_user(
    user_id: str,
    user_data: Dict[str, Any],
    neg_data: Dict[str, Any],
    items_meta: Dict[str, Any],
    retriever: Optional[FlatMemoryRetriever],
    ranker: LocalRanker,
    k_memories: int,
    history_size: int,
    seed: int,
    candidate_row: Optional[Dict[str, Any]] = None,
):
    train_ids = [str(x) for x in user_data.get("train", [])]
    test_ids = [str(x) for x in user_data.get("test", [])]
    neg_ids = [str(x) for x in neg_data.get("test_neg", [])]

    if not test_ids:
        raise ValueError("No test item")

    history = [get_item_info(i, items_meta) for i in train_ids[-history_size:]]
    profile_text = " ".join(f"{x['title']} {x['category']}" for x in history)

    if candidate_row is not None:
        candidate_ids = [
            str(x)
            for x in candidate_row.get(
                "candidates",
                candidate_row.get("candidate_item_ids", []),
            )
        ]
        cached_target = [
            str(x)
            for x in candidate_row.get(
                "target",
                candidate_row.get(
                    "ground_truth",
                    candidate_row.get("ground_truth_item_ids", []),
                ),
            )
        ]
        if cached_target:
            test_ids = cached_target
        if not candidate_ids:
            raise ValueError(f"Empty candidate cache row for user {user_id}")
    else:
        candidate_ids = test_ids + neg_ids
        rng = random.Random(f"{seed}:{user_id}")
        rng.shuffle(candidate_ids)

    candidates = [get_item_info(i, items_meta) for i in candidate_ids]

    memory_thoughts = []
    if retriever is not None and k_memories > 0:
        memories, scores = retriever.retrieve(profile_text, k_memories)
        for m, s in zip(memories, scores):
            memory_thoughts.append({
                "thought_id": m.get("thought_id"),
                "behavior_explanation": m.get("behavior_explanation", ""),
                "pattern": m.get("pattern_description", ""),
                "similarity": round(float(s), 4),
            })

    pred, reasoning = ranker.rank(
        user_id=user_id,
        user_profile=history,
        candidates=candidates,
        memory_thoughts=memory_thoughts or None,
    )

    return {
        "user_id": str(user_id),
        "ground_truth_item_ids": test_ids,
        "candidate_item_ids": candidate_ids,
        "reranked_item_ids": pred,
        "candidate_items": candidates,
        "reranked_items": [get_item_info(i, items_meta) for i in pred],
        "retrieved_memories": memory_thoughts,
        "ranking_reasoning": reasoning,
    }


def load_candidate_file(path: str) -> Dict[str, Dict[str, Any]]:
    obj = load_json(path)
    rows: Dict[str, Dict[str, Any]] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                rows[str(k)] = v
    elif isinstance(obj, list):
        for row in obj:
            if not isinstance(row, dict):
                continue
            uid = row.get("user_id", row.get("uid", row.get("user")))
            if uid is not None:
                rows[str(uid)] = row
    if not rows:
        raise ValueError(f"Could not parse candidate file: {path}")
    return rows


def resolve_candidate_file_path(
    candidate_file: Optional[str],
    sequences_path: str,
    seed: int,
) -> str:
    if candidate_file:
        return str(candidate_file)
    seq = Path(sequences_path)
    return str(seq.with_name(f"{seq.stem}_candidates_seed{int(seed)}.json"))


def ensure_candidate_cache(
    path: str,
    users: List[str],
    sequences: Dict[str, Any],
    negatives: Dict[str, Any],
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Persist canonical candidates once and reuse the exact same order later."""
    p = Path(path)
    rows: Dict[str, Dict[str, Any]] = load_candidate_file(str(p)) if p.exists() else {}
    changed = False

    for uid in users:
        suid = str(uid)
        user_data = sequences[suid]
        neg_data = negatives.get(suid, {})
        target = [str(x) for x in user_data.get("test", [])]
        neg_ids = [str(x) for x in neg_data.get("test_neg", [])]
        if not target:
            raise ValueError(f"user={suid}: no test item for candidate cache")

        expected = target + neg_ids
        if suid in rows:
            cached = [
                str(x)
                for x in rows[suid].get(
                    "candidates",
                    rows[suid].get("candidate_item_ids", []),
                )
            ]
            cached_target = [
                str(x)
                for x in rows[suid].get(
                    "target",
                    rows[suid].get(
                        "ground_truth",
                        rows[suid].get("ground_truth_item_ids", []),
                    ),
                )
            ]
            if not cached or sorted(cached) != sorted(expected):
                raise ValueError(
                    f"user={suid}: existing candidate cache does not match "
                    "current test + test_neg. Delete/rebuild the cache or use "
                    "a different --candidate_file."
                )
            if cached_target and cached_target != target:
                raise ValueError(
                    f"user={suid}: cached target differs from current test target"
                )
            continue

        candidates = list(expected)
        rng = random.Random(f"{seed}:{suid}")
        rng.shuffle(candidates)
        rows[suid] = {
            "user_id": suid,
            "candidates": candidates,
            "target": target,
        }
        changed = True

    if changed or not p.exists():
        save_json_atomic(rows, p)
        print(f"Candidate cache saved: {p} ({len(rows)} users)")
    else:
        print(f"Candidate cache loaded: {p} ({len(rows)} users)")

    missing = [str(u) for u in users if str(u) not in rows]
    if missing:
        raise RuntimeError(
            f"Candidate cache missing {len(missing)} selected users; examples={missing[:10]}"
        )
    return rows


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--global_memory",
        type=str,
        default=None,
        help="Required only when memory retrieval is enabled.",
    )
    p.add_argument("--items", type=str, required=True)
    p.add_argument("--sequences", type=str, required=True)
    p.add_argument("--negatives", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument(
        "--candidate_file", "--candidate-file",
        dest="candidate_file",
        type=str,
        default=None,
        help=(
            "Shared candidate cache. If omitted, a deterministic cache is created "
            "next to --sequences and reused on later runs."
        ),
    )

    p.add_argument("--model_name", type=str, default="unsloth/gemma-3-4b-it-unsloth-bnb-4bit")
    p.add_argument(
        "--embedding_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    p.add_argument(
        "--embedding_device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    p.add_argument("--k_memories", type=int, default=3)
    p.add_argument("--history_size", type=int, default=10)
    p.add_argument("--number_of_users", type=int, default=0)
    p.add_argument("--start_user", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--no_memory",
        action="store_true",
        help="Native-LLM baseline: skip memory retrieval",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    for p in [args.items, args.sequences, args.negatives]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    if not args.no_memory:
        if not args.global_memory:
            raise ValueError("--global_memory is required unless --no_memory is used")
        if not os.path.exists(args.global_memory):
            raise FileNotFoundError(args.global_memory)

    items_meta = normalize_items(load_json(args.items))
    sequences = load_json(args.sequences)
    negatives = load_json(args.negatives)

    # Preserve the exact sequence-file order before applying start/count.
    user_ids = [str(u) for u in list(sequences.keys())[args.start_user:]]
    if args.number_of_users > 0:
        user_ids = user_ids[:args.number_of_users]

    candidate_file_path = resolve_candidate_file_path(
        args.candidate_file,
        args.sequences,
        args.seed,
    )
    candidate_rows = ensure_candidate_cache(
        path=candidate_file_path,
        users=user_ids,
        sequences=sequences,
        negatives=negatives,
        seed=args.seed,
    )
    args.candidate_file = candidate_file_path

    retriever = None
    if not args.no_memory:
        retriever = FlatMemoryRetriever(
            args.global_memory,
            args.embedding_model,
            args.embedding_device,
        )

    ranker = LocalRanker(
        args.model_name,
        args.max_new_tokens,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    output = Path(args.output)
    results = []
    completed = set()

    if args.resume and output.exists():
        try:
            results = load_json(str(output))
            completed = {str(x["user_id"]) for x in results}
            print(f"Resume: {len(completed)} users already completed")
        except Exception as e:
            print(f"Could not resume old output: {e}")
            results = []
            completed = set()

    pending = [u for u in user_ids if str(u) not in completed]
    print(f"Users: total={len(user_ids)}, pending={len(pending)}")

    t0 = time.perf_counter()

    for step, uid in enumerate(tqdm(pending, desc="Inference"), start=1):
        try:
            result = run_one_user(
                user_id=str(uid),
                user_data=sequences[uid],
                neg_data=negatives.get(uid, {}),
                items_meta=items_meta,
                retriever=retriever,
                ranker=ranker,
                k_memories=args.k_memories,
                history_size=args.history_size,
                seed=args.seed,
                candidate_row=candidate_rows[str(uid)],
            )
            results.append(result)
        except Exception as e:
            print(f"\nWarning: user {uid} failed: {e}")
            continue

        if args.save_every > 0 and step % args.save_every == 0:
            save_json_atomic(results, output)

    save_json_atomic(results, output)

    stats = {
        "num_results": len(results),
        "selected_user_ids": user_ids,
        "elapsed_sec_this_run": round(time.perf_counter() - t0, 4),
        "use_memory": not args.no_memory,
        "k_memories": args.k_memories,
        "history_size": args.history_size,
        "candidate_file": args.candidate_file,
        "ranking_model": args.model_name,
        "embedding_model": args.embedding_model,
        "retrieval": retriever.stats() if retriever else None,
        "llm": ranker.stats(),
    }

    stats_path = output.with_name(output.stem + "_inference_stats.json")
    tracking_path = output.with_name(output.stem + "_llm_tracking.json")

    save_json_atomic(stats, stats_path)
    save_json_atomic(
        {"summary": ranker.stats(), "records": ranker.records},
        tracking_path,
    )

    print("\nInference complete")
    print(f"Results : {output}")
    print(f"Stats   : {stats_path}")
    print(f"Tracking: {tracking_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
