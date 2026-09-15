#!/usr/bin/env python3
"""
Precompute TEST-USER behavior trajectories and Tree-AMEM augmentation evidence.

This script is intentionally tied to the current inference implementation:
    infer_tree_amem_gemma_hybrid_v2.py

It reuses the EXACT SAME:
- behavior-window construction
- local Gemma behavior extraction
- frozen hybrid-state mapping
- reverse-tree query
- tree-evidence formatting

Therefore the output can be loaded directly by the ranking-only inference script
without regenerating behaviors or remapping states.

What is precomputed per user
----------------------------
{
  "user_id": "...",
  "train_history_hash": "...",
  "behavior_windows": [...],
  "generated_behaviors": [...],
  "state_mappings": [...],
  "state_sequence": [...],
  "tree_result": {...},
  "tree_evidence": {
      "recent_behavior_states": [...],
      "matched_behavior_context": {...},
      "predicted_next_behaviors": [...]
  }
}

The final `tree_evidence` object is EXACTLY the object currently passed to
LocalGemma.rank(..., tree_evidence=tree_evidence).

No candidate items, negatives, or test targets are needed to precompute this
cache. The cache depends only on:
- train history
- behavior extraction model/prompt/code
- frozen state vocabulary
- reverse tree
- mapping settings

Example
-------
python precompute_test_behavior_tree_evidence.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --tree-dir behavior_tree_out_gemini_hybrid \
  --core-script infer_tree_amem_gemma_hybrid_v2.py \
  --model unsloth/gemma-3-4b-it-unsloth-bnb-4bit \
  --window-size 3 \
  --max-train-interactions 30 \
  --behavior-generation-mode single \
  --behavior-max-attempts 2 \
  --state-margin-threshold 0.05 \
  --output precomputed/CDs/test_user_tree_evidence_gemma.jsonl \
  --summary-output precomputed/CDs/test_user_tree_evidence_gemma.summary.json \
  --resume
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm.auto import tqdm


def jsonl_append(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_dump(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_core(path: str):
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Core inference script not found: {p}\n"
            "Pass --core-script with the path to "
            "infer_tree_amem_gemma_hybrid_v2.py"
        )

    spec = importlib.util.spec_from_file_location("amem_hybrid_core", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec from {p}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stable_json_hash(obj: Any) -> str:
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def processed_users(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()

    out: set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("precompute_ok") is True:
                    out.add(str(row["user_id"]))
            except Exception:
                continue
    return out


def load_user_ids_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None

    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            if isinstance(obj.get("users"), list):
                return [str(x) for x in obj["users"]]
            return [str(k) for k in obj]
    except Exception:
        pass

    return [x.strip() for x in text.splitlines() if x.strip()]


def select_users(
    sequences: Dict[str, Any],
    user_ids_file: Optional[str],
    max_users: int,
) -> List[str]:
    requested = load_user_ids_file(user_ids_file)

    if requested is None:
        users = list(sequences.keys())
    else:
        users = [u for u in requested if u in sequences]

    if max_users > 0:
        users = users[:max_users]
    return users


def summarize_cache(
    output_path: str | Path,
    *,
    llm_stats: Dict[str, Any],
    config: Dict[str, Any],
    elapsed_sec: float,
    attempted_this_run: int,
    failed_this_run: int,
) -> Dict[str, Any]:
    success_rows: List[Dict[str, Any]] = []
    failures = 0

    p = Path(output_path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("precompute_ok") is True:
                    success_rows.append(row)
                else:
                    failures += 1

    mapping_reasons = Counter()
    retrieval_scopes = Counter()
    matched_orders = Counter()
    state_counts = Counter()
    num_behaviors: List[int] = []
    margins: List[float] = []
    best_sims: List[float] = []

    for row in success_rows:
        mappings = row.get("state_mappings", [])
        num_behaviors.append(len(mappings))

        for m in mappings:
            mapping_reasons[str(m.get("mapping_reason", ""))] += 1
            retrieval_scopes[str(m.get("retrieval_scope", ""))] += 1
            sid = m.get("state_id")
            if sid is not None:
                state_counts[str(sid)] += 1
            if m.get("similarity_margin") is not None:
                margins.append(float(m["similarity_margin"]))
            if m.get("best_similarity") is not None:
                best_sims.append(float(m["best_similarity"]))

        tree_result = row.get("tree_result", {})
        matched_orders[str(tree_result.get("matched_order", 0))] += 1

    def quantiles(xs: List[float]) -> Dict[str, Any]:
        if not xs:
            return {"count": 0}
        a = np.asarray(xs, dtype=float)
        return {
            "count": int(len(a)),
            "min": float(np.min(a)),
            "p10": float(np.quantile(a, 0.10)),
            "p50": float(np.quantile(a, 0.50)),
            "p90": float(np.quantile(a, 0.90)),
            "max": float(np.max(a)),
        }

    return {
        "schema_version": "amem_test_tree_evidence_cache_summary_v1",
        "num_users_success": len(success_rows),
        "num_rows_failure_in_file": failures,
        "attempted_this_run": attempted_this_run,
        "failed_this_run": failed_this_run,
        "mean_behaviors_per_user": (
            float(np.mean(num_behaviors)) if num_behaviors else 0.0
        ),
        "matched_context_order_counts": dict(matched_orders),
        "state_mapping_reason_counts": dict(mapping_reasons),
        "state_retrieval_scope_counts": dict(retrieval_scopes),
        "top_mapped_states": [
            [k, int(v)] for k, v in state_counts.most_common(20)
        ],
        "best_similarity_distribution": quantiles(best_sims),
        "similarity_margin_distribution": quantiles(margins),
        "local_gemma": llm_stats,
        "elapsed_sec_this_run": float(elapsed_sec),
        "config": config,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Precompute test-user behaviors, frozen-state mappings, "
            "tree queries, and final Tree-AMEM ranking evidence."
        )
    )

    p.add_argument("--items", required=True)
    p.add_argument("--sequences", required=True)
    p.add_argument(
        "--core-script",
        default="infer_tree_amem_gemma_hybrid_v2.py",
        help="Current inference implementation reused for exact compatibility.",
    )

    p.add_argument(
        "--tree-dir",
        default="behavior_tree_out_gemini_hybrid",
    )
    p.add_argument("--states-json", default=None)
    p.add_argument("--state-embeddings", default=None)
    p.add_argument("--tree-json", default=None)

    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-device", default="auto")
    p.add_argument("--embedding-batch-size", type=int, default=64)

    p.add_argument("--state-low-threshold", type=float, default=None)
    p.add_argument("--state-high-threshold", type=float, default=None)
    p.add_argument("--state-top-k", type=int, default=None)
    p.add_argument("--verifier-min-confidence", type=float, default=None)
    p.add_argument("--state-margin-threshold", type=float, default=0.05)

    p.add_argument(
        "--model",
        default="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    )
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--behavior-max-new-tokens", type=int, default=512)
    p.add_argument("--verifier-max-new-tokens", type=int, default=96)

    p.add_argument("--window-size", type=int, default=3)
    p.add_argument("--max-train-interactions", type=int, default=30)
    p.add_argument("--top-next", type=int, default=3)
    p.add_argument(
        "--behavior-generation-mode",
        choices=["single", "multi"],
        default="single",
    )
    p.add_argument("--behavior-max-attempts", type=int, default=2)

    p.add_argument("--user-ids-file", default=None)
    p.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="0 = all selected users",
    )

    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default=None)
    p.add_argument("--failures-output", default=None)
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    core = load_core(args.core_script)

    tree_dir = Path(args.tree_dir)
    states_json = args.states_json or str(tree_dir / "behavior_states.json")
    state_embeddings = (
        args.state_embeddings
        or str(tree_dir / "behavior_state_embeddings.npy")
    )
    tree_json = args.tree_json or str(tree_dir / "behavior_tree.json")

    for p in [args.items, args.sequences, states_json, state_embeddings, tree_json]:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    output = Path(args.output)
    summary_output = Path(
        args.summary_output or (str(output) + ".summary.json")
    )
    failures_output = Path(
        args.failures_output or (str(output) + ".failures.jsonl")
    )

    if output.exists() and not args.resume:
        raise RuntimeError(
            f"{output} already exists. Delete it or use --resume."
        )

    items_meta = core.json_load(args.items)
    try:
        items_meta = {int(k): v for k, v in items_meta.items()}
    except Exception:
        pass

    raw_sequences = core.json_load(args.sequences)
    sequences = {str(k): v for k, v in raw_sequences.items()}

    user_ids = select_users(
        sequences,
        user_ids_file=args.user_ids_file,
        max_users=args.max_users,
    )

    already_done = processed_users(output) if args.resume else set()
    pending = [u for u in user_ids if u not in already_done]

    vocab = core.HybridStateVocabulary(
        states_json=states_json,
        state_embeddings_npy=state_embeddings,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        low_threshold=args.state_low_threshold,
        high_threshold=args.state_high_threshold,
        top_k=args.state_top_k,
        verifier_min_confidence=args.verifier_min_confidence,
    )
    tree = core.ReverseBehaviorTree(tree_json, vocab)

    # rank_max_new_tokens is unused here but required by the shared LocalGemma.
    llm = core.LocalGemma(
        model=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        temperature=args.temperature,
        behavior_max_new_tokens=args.behavior_max_new_tokens,
        rank_max_new_tokens=64,
        verifier_max_new_tokens=args.verifier_max_new_tokens,
        seed=args.seed,
    )

    config = {
        "items": str(args.items),
        "sequences": str(args.sequences),
        "core_script": str(Path(args.core_script).resolve()),
        "tree_dir": str(tree_dir),
        "states_json": str(states_json),
        "state_embeddings": str(state_embeddings),
        "tree_json": str(tree_json),
        "model": args.model,
        "window_size": args.window_size,
        "max_train_interactions": args.max_train_interactions,
        "top_next": args.top_next,
        "behavior_generation_mode": args.behavior_generation_mode,
        "behavior_max_attempts": args.behavior_max_attempts,
        "state_margin_threshold": args.state_margin_threshold,
        "effective_state_low_threshold": vocab.low_threshold,
        "effective_state_high_threshold": vocab.high_threshold,
        "effective_state_top_k": vocab.top_k,
        "effective_verifier_min_confidence": vocab.verifier_min_confidence,
        "embedding_model": vocab.embedding_model_name,
        "seed": args.seed,
    }

    # Fingerprints help detect stale caches if history/tree/code changes.
    fingerprints = {
        "core_script_sha256": file_sha256(args.core_script),
        "behavior_states_sha256": file_sha256(states_json),
        "behavior_state_embeddings_sha256": file_sha256(state_embeddings),
        "behavior_tree_sha256": file_sha256(tree_json),
        "config_sha256": stable_json_hash(config),
    }

    print("=" * 96)
    print("PRECOMPUTE TEST-USER BEHAVIORS + TREE EVIDENCE")
    print("=" * 96)
    print(f"selected users      : {len(user_ids)}")
    print(f"already cached      : {len(user_ids) - len(pending)}")
    print(f"pending             : {len(pending)}")
    print(f"output              : {output}")
    print(f"tree states         : {len(vocab.state_by_id)}")
    print(
        f"mapping thresholds  : low={vocab.low_threshold:.3f}, "
        f"high={vocab.high_threshold:.3f}, "
        f"margin={args.state_margin_threshold:.3f}"
    )

    t0 = time.time()
    failed_this_run = 0

    pbar = tqdm(
        pending,
        desc="Precompute test-user tree evidence",
        unit="user",
        dynamic_ncols=True,
    )

    for uid in pbar:
        user_data = sequences[uid]
        train_ids = list(user_data.get("train", []))

        try:
            windows = core.build_behavior_windows(
                train_item_ids=train_ids,
                items_meta=items_meta,
                window_size=args.window_size,
                max_train_interactions=args.max_train_interactions,
            )

            tasks = [
                core.BehaviorTask(
                    task_id=f"{uid}_W{i}",
                    interaction_sequence=w,
                )
                for i, w in enumerate(windows)
            ]

            behaviors = (
                llm.generate_behaviors(
                    tasks,
                    mode=args.behavior_generation_mode,
                    max_attempts=args.behavior_max_attempts,
                )
                if tasks else []
            )

            mappings = (
                vocab.map_behaviors(
                    behaviors,
                    llm,
                    margin_threshold=args.state_margin_threshold,
                )
                if behaviors else []
            )

            state_sequence = [
                str(m["state_id"]) for m in mappings
            ]

            tree_result = tree.query(
                state_sequence=state_sequence,
                top_next=args.top_next,
            )

            # THIS is the exact object currently used to augment ranking.
            tree_evidence = core.build_tree_evidence(
                mappings,
                tree_result,
            )

            used_train_ids = train_ids[-args.max_train_interactions:]

            row = {
                "schema_version": "amem_test_tree_evidence_cache_v1",
                "precompute_ok": True,
                "user_id": uid,
                "train_history_item_ids": [str(x) for x in used_train_ids],
                "train_history_hash": stable_json_hash(
                    [str(x) for x in used_train_ids]
                ),
                "window_size": args.window_size,
                "max_train_interactions": args.max_train_interactions,
                "behavior_windows": windows,
                "generated_behaviors": behaviors,
                "state_mappings": mappings,
                "state_sequence": state_sequence,
                "tree_result": tree_result,
                "tree_evidence": tree_evidence,
                "fingerprints": fingerprints,
            }

            jsonl_append(output, row)

            pbar.set_postfix(
                behaviors=len(behaviors),
                order=tree_result.get("matched_order", 0),
                next=len(tree_result.get("next_behaviors", [])),
                refresh=False,
            )

        except Exception as e:
            failed_this_run += 1
            failure = {
                "schema_version": "amem_test_tree_evidence_cache_v1",
                "precompute_ok": False,
                "user_id": uid,
                "error_type": type(e).__name__,
                "error": str(e),
                "fingerprints": fingerprints,
            }
            jsonl_append(failures_output, failure)
            print(
                f"\n[WARN] user={uid} failed: {type(e).__name__}: {e}",
                flush=True,
            )

    pbar.close()

    summary = summarize_cache(
        output,
        llm_stats=llm.stats(),
        config=config,
        elapsed_sec=time.time() - t0,
        attempted_this_run=len(pending),
        failed_this_run=failed_this_run,
    )
    summary["fingerprints"] = fingerprints

    json_dump(summary_output, summary)

    print("\nSummary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nCache    : {output}")
    print(f"Failures : {failures_output}")
    print(f"Summary  : {summary_output}")


if __name__ == "__main__":
    main()
