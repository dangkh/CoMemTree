#!/usr/bin/env python3
"""
Inspect the EXACT ranking prompts used by Tree-AMEM vs NativeLLM.

This script does NOT call any LLM.
It reconstructs the final ranking prompt from an existing inference result JSONL.

It is intended for diagnosing cases where Tree-AMEM underperforms NativeLLM/AMEM.

For each selected user it exports:
- recent raw history
- precomputed Gemma behaviors
- mapped states
- matched tree context
- predicted next behaviors
- candidates + ground-truth marker (diagnostic only; NOT included in prompt)
- exact Tree-AMEM SYSTEM + USER prompt
- exact NativeLLM SYSTEM + USER prompt
- target ranks if available

The prompt template below is copied from infer_tree_gemma_precom_batch.py
(_build_rank_prompt), so the reconstructed prompt matches the current ranking code.

Examples
--------
First 6 successful samples:

python inspect_tree_ranking_prompts.py \
  --results results/tree_cluster50_300.jsonl \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --max-train-interactions 10 \
  --num-samples 6 \
  --selection first \
  --output-dir prompt_debug/cluster50_first6

If the result file contains both tree and baseline metrics, inspect cases where
Tree-AMEM worsened target rank:

python inspect_tree_ranking_prompts.py \
  --results results/tree_hybrid_with_baseline.jsonl \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --max-train-interactions 10 \
  --num-samples 8 \
  --selection degraded \
  --output-dir prompt_debug/degraded8
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =============================================================================
# Generic I/O
# =============================================================================

def json_load(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                print(
                    f"[WARN] skip bad JSON line {line_no}: "
                    f"{type(e).__name__}: {e}"
                )
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


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
            return [str(k) for k in obj.keys()]
    except Exception:
        pass

    return [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]


# =============================================================================
# Item helpers
# =============================================================================

def load_items(path: str) -> Dict[Any, Dict[str, Any]]:
    raw = json_load(path)
    try:
        return {
            int(k): v
            for k, v in raw.items()
        }
    except Exception:
        return raw


def resolve_item_key(
    item_id: Any,
    items_meta: Dict[Any, Dict[str, Any]],
) -> Optional[Any]:
    if item_id in items_meta:
        return item_id

    s = str(item_id)

    if s in items_meta:
        return s

    try:
        i = int(s)
        if i in items_meta:
            return i
    except Exception:
        pass

    return None


def get_item_info(
    item_id: Any,
    items_meta: Dict[Any, Dict[str, Any]],
) -> Dict[str, str]:
    k = resolve_item_key(
        item_id,
        items_meta,
    )

    if k is None:
        return {
            "item_id": str(item_id),
            "title": f"Item {item_id}",
            "category": "Unknown",
        }

    info = items_meta[k]

    return {
        "item_id": str(item_id),
        "title": str(
            info.get(
                "title",
                f"Item {item_id}",
            )
        ),
        "category": str(
            info.get(
                "main_cat",
                info.get(
                    "category",
                    "Unknown",
                ),
            )
        ),
    }


# =============================================================================
# EXACT CURRENT RANKING PROMPT TEMPLATE
# =============================================================================

SYSTEM_PROMPT = (
    "You are a recommendation ranking system. "
    "Return valid JSON only."
)


def build_rank_prompt(
    *,
    history_items: List[Dict[str, str]],
    candidate_items: List[Dict[str, str]],
    tree_evidence: Optional[Dict[str, Any]],
) -> Tuple[str, str, List[str]]:
    """
    Exact copy of LocalGemma._build_rank_prompt() from the current inference code.
    """
    ids = [
        str(x["item_id"])
        for x in candidate_items
    ]
    n = len(ids)

    evidence_block = ""

    if tree_evidence is not None:
        evidence_block = f"""
Target user's inferred recent behaviors and collaborative trajectory evidence:
{json.dumps(tree_evidence, ensure_ascii=False, indent=2)}

Interpretation:
- recent_behavior_states describes the target user's own inferred recent behavior.
- predicted_next_behaviors comes from collaborative trajectory statistics.
- probabilities are relative collaborative weights for the predicted next behaviors.
Use this evidence only when compatible with the user's own observed history.
Do not let collaborative evidence override strong personal evidence.
"""

    prompt = f"""Rank ALL candidate items for the target user.

Priority:
1. Recent observed history and recency.
2. Target user's inferred recent behavior.
3. Semantic/category fit.
4. Collaborative predicted-next-behavior evidence as an auxiliary signal.

Recent history (most recent last):
{json.dumps(history_items[-10:], ensure_ascii=False, indent=2)}
{evidence_block}
Candidates:
{json.dumps(candidate_items, ensure_ascii=False, indent=2)}

Rank ALL {n} IDs exactly once.
Do not invent or omit IDs.
Do not preserve input order by default.

Return JSON only:
{{"ranked_item_ids":[...],"reasoning":"one concise sentence"}}
"""

    return SYSTEM_PROMPT, prompt, ids


# =============================================================================
# Diagnostics
# =============================================================================

def target_rank_from_row(
    row: Dict[str, Any],
    key: str,
) -> Optional[int]:
    metrics = row.get(key)

    if not isinstance(metrics, dict):
        return None

    rank = metrics.get("target_rank")

    if rank is None:
        return None

    try:
        return int(rank)
    except Exception:
        return None


def select_rows(
    rows: List[Dict[str, Any]],
    *,
    selection: str,
    num_samples: int,
    seed: int,
    user_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    usable = [
        row
        for row in rows
        if row.get("user_id") is not None
        and row.get("candidate_item_ids")
    ]

    if user_ids is not None:
        by_uid = {
            str(r["user_id"]): r
            for r in usable
        }

        return [
            by_uid[u]
            for u in user_ids
            if u in by_uid
        ][:num_samples]

    if selection == "first":
        return usable[:num_samples]

    if selection == "random":
        rng = random.Random(seed)
        if len(usable) <= num_samples:
            return usable
        return rng.sample(
            usable,
            num_samples,
        )

    if selection in {
        "degraded",
        "improved",
        "largest_drop",
    }:
        scored = []

        for row in usable:
            tree_rank = target_rank_from_row(
                row,
                "tree_metrics",
            )
            baseline_rank = target_rank_from_row(
                row,
                "baseline_metrics",
            )

            if (
                tree_rank is None
                or baseline_rank is None
            ):
                continue

            # Positive delta = Tree is worse.
            delta = (
                tree_rank
                - baseline_rank
            )

            if (
                selection == "degraded"
                and delta <= 0
            ):
                continue

            if (
                selection == "improved"
                and delta >= 0
            ):
                continue

            scored.append(
                (
                    delta,
                    row,
                )
            )

        if selection in {
            "degraded",
            "largest_drop",
        }:
            scored.sort(
                key=lambda x: x[0],
                reverse=True,
            )
        else:
            scored.sort(
                key=lambda x: x[0],
            )

        return [
            row
            for _, row in scored[
                :num_samples
            ]
        ]

    raise ValueError(
        f"Unknown selection: {selection}"
    )


def compact_behavior(
    b: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "window_index": b.get(
            "window_index"
        ),
        "behavior_signature": b.get(
            "behavior_signature"
        ),
        "mechanism": b.get(
            "mechanism"
        ),
        "scope": b.get(
            "scope"
        ),
        "direction": b.get(
            "direction"
        ),
        "confidence": b.get(
            "confidence"
        ),
        "pattern_description": b.get(
            "pattern_description"
        ),
        "behavior_explanation": b.get(
            "behavior_explanation"
        ),
    }


def build_sample_text(
    *,
    row: Dict[str, Any],
    history_items: List[Dict[str, str]],
    candidate_items: List[Dict[str, str]],
    ground_truth: List[str],
    tree_system: str,
    tree_prompt: str,
    baseline_system: str,
    baseline_prompt: str,
) -> str:
    uid = str(
        row["user_id"]
    )

    tree_rank = target_rank_from_row(
        row,
        "tree_metrics",
    )
    baseline_rank = target_rank_from_row(
        row,
        "baseline_metrics",
    )

    rank_delta = None

    if (
        tree_rank is not None
        and baseline_rank is not None
    ):
        rank_delta = (
            tree_rank
            - baseline_rank
        )

    behaviors = [
        compact_behavior(x)
        for x in (
            row.get(
                "generated_behaviors"
            )
            or []
        )
    ]

    mappings = (
        row.get(
            "state_mappings"
        )
        or []
    )

    tree_result = (
        row.get(
            "tree_result"
        )
        or {}
    )

    tree_evidence = (
        row.get(
            "tree_evidence"
        )
    )

    candidates_diagnostic = []

    gt_set = set(
        str(x)
        for x in ground_truth
    )

    for x in candidate_items:
        y = dict(x)
        y["is_ground_truth"] = (
            str(x["item_id"])
            in gt_set
        )
        candidates_diagnostic.append(
            y
        )

    blocks = []

    blocks.append(
        "=" * 110
    )
    blocks.append(
        f"USER: {uid}"
    )
    blocks.append(
        "=" * 110
    )

    blocks.append(
        "\n[RESULT SUMMARY]"
    )
    blocks.append(
        json.dumps(
            {
                "tree_target_rank": tree_rank,
                "baseline_target_rank": baseline_rank,
                "rank_delta_tree_minus_baseline": rank_delta,
                "matched_order": (
                    tree_result.get(
                        "matched_order"
                    )
                ),
                "matched_support_users": (
                    tree_result.get(
                        "matched_support_users"
                    )
                ),
                "matched_support_occurrences": (
                    tree_result.get(
                        "matched_support_occurrences"
                    )
                ),
                "num_predicted_next_behaviors": len(
                    tree_result.get(
                        "next_behaviors",
                        [],
                    )
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    blocks.append(
        "\n[RAW RECENT HISTORY]"
    )
    blocks.append(
        json.dumps(
            history_items,
            ensure_ascii=False,
            indent=2,
        )
    )

    blocks.append(
        "\n[PRECOMPUTED GEMMA BEHAVIORS]"
    )
    blocks.append(
        json.dumps(
            behaviors,
            ensure_ascii=False,
            indent=2,
        )
    )

    blocks.append(
        "\n[STATE MAPPINGS]"
    )
    blocks.append(
        json.dumps(
            mappings,
            ensure_ascii=False,
            indent=2,
        )
    )

    blocks.append(
        "\n[TREE QUERY RESULT]"
    )
    blocks.append(
        json.dumps(
            tree_result,
            ensure_ascii=False,
            indent=2,
        )
    )

    blocks.append(
        "\n[EXACT TREE EVIDENCE INSERTED INTO PROMPT]"
    )
    blocks.append(
        json.dumps(
            tree_evidence,
            ensure_ascii=False,
            indent=2,
        )
    )

    blocks.append(
        "\n[CANDIDATES - DIAGNOSTIC VIEW ONLY; GROUND TRUTH MARKER IS NOT IN ACTUAL PROMPT]"
    )
    blocks.append(
        json.dumps(
            candidates_diagnostic,
            ensure_ascii=False,
            indent=2,
        )
    )

    blocks.append(
        "\n" + "#" * 110
    )
    blocks.append(
        "# EXACT TREE-AMEM RANKING PROMPT"
    )
    blocks.append(
        "#" * 110
    )
    blocks.append(
        "\n[SYSTEM]"
    )
    blocks.append(
        tree_system
    )
    blocks.append(
        "\n[USER]"
    )
    blocks.append(
        tree_prompt
    )

    blocks.append(
        "\n" + "#" * 110
    )
    blocks.append(
        "# EXACT NATIVE-LLM BASELINE PROMPT"
    )
    blocks.append(
        "#" * 110
    )
    blocks.append(
        "\n[SYSTEM]"
    )
    blocks.append(
        baseline_system
    )
    blocks.append(
        "\n[USER]"
    )
    blocks.append(
        baseline_prompt
    )

    blocks.append(
        "\n" + "#" * 110
    )
    blocks.append(
        "# PROMPT SIZE"
    )
    blocks.append(
        "#" * 110
    )
    blocks.append(
        json.dumps(
            {
                "tree_prompt_chars": len(
                    tree_prompt
                ),
                "baseline_prompt_chars": len(
                    baseline_prompt
                ),
                "extra_chars_from_tree_evidence": (
                    len(tree_prompt)
                    - len(baseline_prompt)
                ),
            },
            indent=2,
        )
    )

    return "\n".join(
        blocks
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Dump exact Tree-AMEM and NativeLLM ranking prompts "
            "from existing inference results."
        )
    )

    p.add_argument(
        "--results",
        required=True,
        help=(
            "Inference JSONL containing tree_evidence, "
            "candidate_item_ids, etc."
        ),
    )

    p.add_argument(
        "--items",
        required=True,
    )

    p.add_argument(
        "--sequences",
        required=True,
    )

    p.add_argument(
        "--max-train-interactions",
        type=int,
        default=10,
    )

    p.add_argument(
        "--num-samples",
        type=int,
        default=6,
    )

    p.add_argument(
        "--selection",
        choices=[
            "first",
            "random",
            "degraded",
            "improved",
            "largest_drop",
        ],
        default="first",
        help=(
            "degraded/largest_drop require both tree_metrics "
            "and baseline_metrics in the results file."
        ),
    )

    p.add_argument(
        "--user-ids-file",
        default=None,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    rows = load_jsonl(
        args.results
    )

    items_meta = load_items(
        args.items
    )

    raw_sequences = json_load(
        args.sequences
    )

    sequences = {
        str(k): v
        for k, v in raw_sequences.items()
    }

    requested_users = (
        load_user_ids_file(
            args.user_ids_file
        )
    )

    selected = select_rows(
        rows,
        selection=args.selection,
        num_samples=args.num_samples,
        seed=args.seed,
        user_ids=requested_users,
    )

    if not selected:
        raise ValueError(
            "No rows matched the requested selection. "
            "If using --selection degraded/improved, confirm that "
            "the result file contains baseline_metrics."
        )

    out_dir = Path(
        args.output_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    all_samples = []

    for sample_idx, row in enumerate(
        selected,
        start=1,
    ):
        uid = str(
            row["user_id"]
        )

        if uid not in sequences:
            print(
                f"[WARN] user={uid} missing from sequences; skipped"
            )
            continue

        user_data = sequences[
            uid
        ]

        train_ids = list(
            user_data.get(
                "train",
                [],
            )
        )[
            -args.max_train_interactions:
        ]

        history_items = [
            get_item_info(
                x,
                items_meta,
            )
            for x in train_ids
        ]

        candidate_ids = [
            str(x)
            for x in row.get(
                "candidate_item_ids",
                [],
            )
        ]

        candidate_items = [
            get_item_info(
                x,
                items_meta,
            )
            for x in candidate_ids
        ]

        ground_truth = [
            str(x)
            for x in row.get(
                "ground_truth_item_ids",
                [],
            )
        ]

        tree_evidence = row.get(
            "tree_evidence"
        )

        (
            tree_system,
            tree_prompt,
            _,
        ) = build_rank_prompt(
            history_items=history_items,
            candidate_items=candidate_items,
            tree_evidence=tree_evidence,
        )

        (
            baseline_system,
            baseline_prompt,
            _,
        ) = build_rank_prompt(
            history_items=history_items,
            candidate_items=candidate_items,
            tree_evidence=None,
        )

        text = build_sample_text(
            row=row,
            history_items=history_items,
            candidate_items=candidate_items,
            ground_truth=ground_truth,
            tree_system=tree_system,
            tree_prompt=tree_prompt,
            baseline_system=baseline_system,
            baseline_prompt=baseline_prompt,
        )

        safe_uid = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            uid,
        )

        sample_path = (
            out_dir
            / f"{sample_idx:02d}_{safe_uid}.txt"
        )

        sample_path.write_text(
            text,
            encoding="utf-8",
        )

        all_samples.append(
            text
        )

        tree_rank = target_rank_from_row(
            row,
            "tree_metrics",
        )

        baseline_rank = target_rank_from_row(
            row,
            "baseline_metrics",
        )

        summary_rows.append({
            "sample_index": sample_idx,
            "user_id": uid,
            "file": str(sample_path),
            "tree_target_rank": tree_rank,
            "baseline_target_rank": baseline_rank,
            "rank_delta_tree_minus_baseline": (
                tree_rank - baseline_rank
                if (
                    tree_rank is not None
                    and baseline_rank is not None
                )
                else None
            ),
            "matched_order": (
                (
                    row.get(
                        "tree_result"
                    )
                    or {}
                ).get(
                    "matched_order"
                )
            ),
            "tree_prompt_chars": len(
                tree_prompt
            ),
            "baseline_prompt_chars": len(
                baseline_prompt
            ),
            "extra_chars_from_tree_evidence": (
                len(tree_prompt)
                - len(baseline_prompt)
            ),
        })

    combined_path = (
        out_dir
        / "ALL_PROMPT_SAMPLES.txt"
    )

    combined_path.write_text(
        (
            "\n\n\n"
            + "\n\n".join(
                all_samples
            )
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    json_dump(
        out_dir
        / "prompt_debug_summary.json",
        {
            "results": args.results,
            "selection": args.selection,
            "requested_num_samples": (
                args.num_samples
            ),
            "actual_num_samples": len(
                summary_rows
            ),
            "max_train_interactions": (
                args.max_train_interactions
            ),
            "samples": summary_rows,
            "combined_file": str(
                combined_path
            ),
        },
    )

    print(
        f"[DONE] dumped {len(summary_rows)} samples"
    )

    for x in summary_rows:
        print(
            f"  sample={x['sample_index']:02d} "
            f"user={x['user_id']} "
            f"tree_rank={x['tree_target_rank']} "
            f"baseline_rank={x['baseline_target_rank']} "
            f"delta={x['rank_delta_tree_minus_baseline']} "
            f"order={x['matched_order']} "
            f"extra_chars={x['extra_chars_from_tree_evidence']}"
        )

    print(
        f"\nCombined prompts: {combined_path}"
    )
    print(
        f"Summary         : "
        f"{out_dir / 'prompt_debug_summary.json'}"
    )


if __name__ == "__main__":
    import re
    main()
