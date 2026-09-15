#!/usr/bin/env python3
"""
Ranking-only Tree-AMEM inference from PRECOMPUTED test-user tree evidence.

Expected cache:
    output of precompute_test_behavior_tree_evidence.py

This script does NOT:
- regenerate test-user behaviors
- load Qwen
- remap behaviors to states
- query the reverse tree

It simply loads each user's precomputed `tree_evidence` object and passes it
unchanged to the current LocalGemma.rank(...).

This is useful for testing multiple ranking models/settings while holding the
test-user behavior representation and tree evidence fixed.

Example
-------
python infer_tree_amem_gemma_from_precomputed.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/user_negatives_10_5000.json \
  --precomputed-evidence precomputed/CDs/test_user_tree_evidence_gemma.jsonl \
  --core-script infer_tree_amem_gemma_hybrid_v2.py \
  --model unsloth/gemma-3-4b-it-unsloth-bnb-4bit \
  --max-users 10 \
  --run-baseline \
  --output results/tree_gemma_precomputed_test10.jsonl \
  --summary-output results/tree_gemma_precomputed_test10_summary.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm.auto import tqdm


def load_core(path: str):
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    spec = importlib.util.spec_from_file_location("amem_hybrid_core", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def load_evidence_cache(path: str | Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("precompute_ok") is not True:
                continue
            uid = str(row["user_id"])
            rows[uid] = row
    return rows


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
                if row.get("user_id") is not None:
                    out.add(str(row["user_id"]))
            except Exception:
                continue
    return out


def aggregate_metrics(
    rows: List[Dict[str, Any]],
    key: str,
) -> Optional[Dict[str, float]]:
    vals = [r.get(key) for r in rows if isinstance(r.get(key), dict)]
    if not vals:
        return None

    out: Dict[str, float] = {}
    for metric in [
        "recall@5",
        "recall@10",
        "recall@20",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
    ]:
        xs = [float(v[metric]) for v in vals if v.get(metric) is not None]
        out[metric] = float(np.mean(xs)) if xs else 0.0

    ranks = [
        float(v["target_rank"])
        for v in vals
        if v.get("target_rank") is not None
    ]
    out["mean_target_rank"] = float(np.mean(ranks)) if ranks else float("nan")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank candidates using precomputed Tree-AMEM evidence."
    )

    p.add_argument("--items", required=True)
    p.add_argument("--sequences", required=True)
    p.add_argument("--negatives", required=True)
    p.add_argument("--candidate-file", default=None)

    p.add_argument("--precomputed-evidence", required=True)
    p.add_argument(
        "--core-script",
        default="infer_tree_amem_gemma_hybrid_v2.py",
    )

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
    p.add_argument("--rank-max-new-tokens", type=int, default=768)

    p.add_argument("--max-train-interactions", type=int, default=30)
    p.add_argument("--max-users", type=int, default=0)
    p.add_argument("--user-ids-file", default=None)

    p.add_argument(
        "--run-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default=None)
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

    evidence_cache = load_evidence_cache(args.precomputed_evidence)
    if not evidence_cache:
        raise ValueError("No successful users found in precomputed evidence cache")

    items_meta, user_sequences, user_negatives = core.load_amem_data(
        args.items,
        args.sequences,
        args.negatives,
    )
    candidate_rows = core.load_candidate_file(args.candidate_file)

    requested = core.load_user_ids_file(args.user_ids_file)
    if requested is None:
        users = [u for u in user_sequences if u in evidence_cache]
    else:
        users = [
            str(u)
            for u in requested
            if str(u) in user_sequences and str(u) in evidence_cache
        ]

    if args.max_users > 0:
        users = users[:args.max_users]

    done = processed_users(args.output) if args.resume else set()
    pending = [u for u in users if u not in done]

    llm = core.LocalGemma(
        model=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        temperature=args.temperature,
        behavior_max_new_tokens=64,   # unused
        rank_max_new_tokens=args.rank_max_new_tokens,
        verifier_max_new_tokens=64,   # unused
        seed=args.seed,
    )

    print("=" * 96)
    print("RANKING FROM PRECOMPUTED TEST-USER TREE EVIDENCE")
    print("=" * 96)
    print(f"cached users        : {len(evidence_cache)}")
    print(f"selected users      : {len(users)}")
    print(f"already ranked      : {len(users) - len(pending)}")
    print(f"pending             : {len(pending)}")
    print(f"ranking model       : {args.model}")

    rows_this_run: List[Dict[str, Any]] = []
    failures = 0
    t0 = time.time()

    pbar = tqdm(
        pending,
        desc="Gemma ranking from cached evidence",
        unit="user",
        dynamic_ncols=True,
    )

    for uid in pbar:
        try:
            user_data = user_sequences[uid]
            negative_data = user_negatives.get(uid, {})

            candidate_ids, ground_truth = core.get_candidates_for_user(
                user_id=uid,
                user_data=user_data,
                negative_data=negative_data,
                candidate_file_rows=candidate_rows,
                seed=args.seed,
            )

            candidate_items = [
                core.get_item_info(i, items_meta)
                for i in candidate_ids
            ]

            train_ids = list(user_data.get("train", []))
            history_items = [
                core.get_item_info(i, items_meta)
                for i in train_ids[-args.max_train_interactions:]
            ]

            cache_row = evidence_cache[uid]
            tree_evidence = cache_row["tree_evidence"]

            tree_ranked, tree_rank_meta = llm.rank(
                history_items=history_items,
                candidate_items=candidate_items,
                tree_evidence=tree_evidence,
                call_type="tree_ranking_precomputed",
            )
            tree_metrics = core.ranking_metrics(
                tree_ranked,
                ground_truth,
            )

            baseline_ranked = None
            baseline_meta = None
            baseline_metrics = None

            if args.run_baseline:
                baseline_ranked, baseline_meta = llm.rank(
                    history_items=history_items,
                    candidate_items=candidate_items,
                    tree_evidence=None,
                    call_type="baseline_ranking",
                )
                baseline_metrics = core.ranking_metrics(
                    baseline_ranked,
                    ground_truth,
                )

            row = {
                "schema_version": "amem_precomputed_tree_ranking_v1",
                "user_id": uid,
                "candidate_item_ids": candidate_ids,
                "ground_truth_item_ids": ground_truth,
                "tree_evidence_cache_user_id": uid,
                "tree_evidence": tree_evidence,
                "cached_state_sequence": cache_row.get("state_sequence", []),
                "cached_state_mappings": cache_row.get("state_mappings", []),
                "cached_tree_result": cache_row.get("tree_result", {}),
                "tree_ranked_item_ids": tree_ranked,
                "tree_rank_meta": tree_rank_meta,
                "tree_metrics": tree_metrics,
                "baseline_ranked_item_ids": baseline_ranked,
                "baseline_rank_meta": baseline_meta,
                "baseline_metrics": baseline_metrics,
            }

            jsonl_append(args.output, row)
            rows_this_run.append(row)

            pbar.set_postfix(
                order=cache_row.get("tree_result", {}).get("matched_order", 0),
                rank=tree_metrics.get("target_rank"),
                refresh=False,
            )

        except Exception as e:
            failures += 1
            print(
                f"\n[WARN] user={uid} failed: {type(e).__name__}: {e}",
                flush=True,
            )

    pbar.close()

    # Aggregate over the whole output file, not just this run.
    all_rows: List[Dict[str, Any]] = []
    output_path = Path(args.output)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_rows.append(json.loads(line))
                    except Exception:
                        pass

    tree_metrics = aggregate_metrics(all_rows, "tree_metrics")
    baseline_metrics = aggregate_metrics(all_rows, "baseline_metrics")

    tree_minus_baseline = None
    if tree_metrics is not None and baseline_metrics is not None:
        tree_minus_baseline = {
            "recall@5": tree_metrics["recall@5"] - baseline_metrics["recall@5"],
            "recall@10": tree_metrics["recall@10"] - baseline_metrics["recall@10"],
            "recall@20": tree_metrics["recall@20"] - baseline_metrics["recall@20"],
            "ndcg@5": tree_metrics["ndcg@5"] - baseline_metrics["ndcg@5"],
            "ndcg@10": tree_metrics["ndcg@10"] - baseline_metrics["ndcg@10"],
            "ndcg@20": tree_metrics["ndcg@20"] - baseline_metrics["ndcg@20"],
            "mean_target_rank_improvement": (
                baseline_metrics["mean_target_rank"]
                - tree_metrics["mean_target_rank"]
            ),
        }

    order_counts = Counter(
        str(
            r.get("cached_tree_result", {}).get(
                "matched_order", 0
            )
        )
        for r in all_rows
    )

    summary = {
        "schema_version": "amem_precomputed_tree_ranking_summary_v1",
        "num_users": len(all_rows),
        "failed_this_run": failures,
        "tree_metrics": tree_metrics,
        "baseline_metrics": baseline_metrics,
        "tree_minus_baseline": tree_minus_baseline,
        "matched_context_order_counts": dict(order_counts),
        "local_gemma": llm.stats(),
        "elapsed_sec_this_run": float(time.time() - t0),
        "config": {
            "items": args.items,
            "sequences": args.sequences,
            "negatives": args.negatives,
            "candidate_file": args.candidate_file,
            "precomputed_evidence": args.precomputed_evidence,
            "model": args.model,
            "max_train_interactions": args.max_train_interactions,
            "run_baseline": args.run_baseline,
            "seed": args.seed,
        },
    }

    summary_output = (
        args.summary_output
        or (str(args.output) + ".summary.json")
    )
    json_dump(summary_output, summary)

    print("\nSummary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutput  : {args.output}")
    print(f"Summary : {summary_output}")


if __name__ == "__main__":
    main()
