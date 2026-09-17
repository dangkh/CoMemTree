#!/usr/bin/env python3
"""
BM25 baseline matched to the current CoMemTree evaluation protocol.

FAIRNESS / NO-LEAKAGE DEFAULTS
------------------------------
1) Select the SAME evaluation users as CoMemTree:
   - by default, intersect sequence users with successful users in
     --eval-behaviors, preserving sequence-file order;
   - then take --max-users (default: 300).
   - alternatively pass --user-ids-file with a frozen user list.

2) Build the BM25 user query ONLY from:
       train[-history_size:]
   with history_size=10 by default.
   Validation/test interactions are never used in the query.

3) BM25 IDF/document statistics are built ONLY from the static item metadata
   catalog in --items. They are NOT computed from each user's test candidate set.

4) Evaluation uses the SAME candidate construction as CoMemTree:
       test_ids + test_neg
       random.Random(f"{seed}:{user_id}").shuffle(candidates)
   Or pass --candidate-file if CoMemTree used an explicit frozen candidate file.

5) Default item text mirrors the information exposed by CoMemTree ranking:
       title + category
   Missing metadata contributes no lexical terms. Description can be enabled
   explicitly with --use-description, but it is OFF by default for fairness.

Outputs
-------
<output_dir>/
  bm25_metrics.json
  bm25_rankings.jsonl
  fixed_candidates_snapshot.json

Example
-------
python bm25_comemtree_baseline.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/user_negatives_10_5000.json \
  --eval-behaviors precomputed/CDs/test_user_behaviors_gemma.jsonl \
  --max-users 300 \
  --history-size 10 \
  --expected-candidates 20 \
  --seed 42 \
  --k1 1.5 \
  --b 0.75 \
  --output-dir results/bm25_cd_300
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


# =============================================================================
# Generic I/O
# =============================================================================

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if not isinstance(x, list):
        x = [x]
    return [str(v) for v in x]


# =============================================================================
# User selection -- same intent as CoMemTree
# =============================================================================

def load_successful_behavior_users(path: str | Path) -> Set[str]:
    users: Set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception:
                continue

            uid = row.get("user_id", row.get("uid"))
            if uid is None:
                continue

            if row.get("precompute_ok") is False:
                continue

            behaviors = row.get("generated_behaviors")
            if isinstance(behaviors, list) and len(behaviors) == 0:
                continue

            users.add(str(uid))

    return users


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
            return [str(x) for x in obj.keys()]
    except Exception:
        pass

    return [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]


def select_users(
    sequences: Dict[str, Dict[str, Any]],
    eval_behaviors: Optional[str],
    user_ids_file: Optional[str],
    max_users: int,
) -> List[str]:
    requested = load_user_ids_file(user_ids_file)

    if requested is not None:
        users = [
            uid
            for uid in requested
            if uid in sequences
        ]

        if eval_behaviors:
            behavior_users = load_successful_behavior_users(eval_behaviors)
            users = [
                uid
                for uid in users
                if uid in behavior_users
            ]

    elif eval_behaviors:
        behavior_users = load_successful_behavior_users(eval_behaviors)

        # Preserve sequence-file order, matching the current CoMemTree selection.
        users = [
            uid
            for uid in sequences
            if uid in behavior_users
        ]

    else:
        users = list(sequences.keys())

    if max_users > 0:
        users = users[:max_users]

    if not users:
        raise ValueError("No users selected.")

    return users


# =============================================================================
# Candidate construction -- same as CoMemTree
# =============================================================================

def load_candidate_file(
    path: Optional[str],
) -> Optional[Dict[str, Dict[str, Any]]]:
    if not path:
        return None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".jsonl":
        rows: Dict[str, Dict[str, Any]] = {}

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)
                uid = row.get("user_id", row.get("uid"))

                if uid is not None:
                    rows[str(uid)] = row

        return rows

    obj = load_json(p)

    if isinstance(obj, dict) and isinstance(obj.get("users"), dict):
        obj = obj["users"]

    if not isinstance(obj, dict):
        raise ValueError(
            "Candidate file must be a JSON object or JSONL rows."
        )

    return {
        str(k): v
        for k, v in obj.items()
    }


def get_candidates_for_user(
    uid: str,
    user_data: Dict[str, Any],
    negative_data: Dict[str, Any],
    candidate_rows: Optional[Dict[str, Dict[str, Any]]],
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Exact default CoMemTree protocol:
        candidates = test_ids + test_neg
        random.Random(f"{seed}:{uid}").shuffle(candidates)
    """
    if candidate_rows is not None and uid in candidate_rows:
        row = candidate_rows[uid]

        candidates = as_str_list(
            row.get(
                "candidates",
                row.get(
                    "candidate_item_ids",
                    [],
                ),
            )
        )

        targets = as_str_list(
            row.get(
                "target",
                row.get(
                    "targets",
                    row.get(
                        "target_item_ids",
                        user_data.get("test", []),
                    ),
                ),
            )
        )

        if not candidates:
            raise ValueError(
                f"user={uid}: empty candidate-file row"
            )

        return candidates, targets

    test_ids = as_str_list(
        user_data.get("test", [])
    )

    neg_ids = as_str_list(
        negative_data.get("test_neg", [])
    )

    if not test_ids:
        raise ValueError(
            f"user={uid}: no test item"
        )

    candidates = test_ids + neg_ids

    rng = random.Random(
        f"{seed}:{uid}"
    )
    rng.shuffle(candidates)

    return candidates, test_ids


# =============================================================================
# Item metadata and text construction
# =============================================================================

TOKEN_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)


def normalize_items(items: Any) -> Dict[str, Dict[str, Any]]:
    """
    Current CDs items.json is expected to be keyed by item/ASIN.
    This helper also tolerates a list-of-dicts representation.
    """
    if isinstance(items, dict):
        return {
            str(k): (v if isinstance(v, dict) else {})
            for k, v in items.items()
        }

    if isinstance(items, list):
        out: Dict[str, Dict[str, Any]] = {}

        for row in items:
            if not isinstance(row, dict):
                continue

            iid = (
                row.get("item_id")
                or row.get("item")
                or row.get("asin")
                or row.get("id")
            )

            if iid is None:
                continue

            out[str(iid)] = row

        return out

    raise ValueError(
        "Unsupported items.json structure."
    )


def clean_category(info: Dict[str, Any]) -> str:
    category = (
        info.get("main_cat")
        or info.get("category")
        or info.get("categories")
        or ""
    )

    if isinstance(category, list):
        if category and isinstance(category[0], list):
            category = " ".join(
                str(v)
                for v in category[0]
                if v
            )
        else:
            category = " ".join(
                str(v)
                for v in category
                if v
            )

    category = str(category or "").strip()

    if category.lower() in {
        "",
        "unknown",
        "none",
        "null",
        "nan",
    }:
        return ""

    return category


def flatten_text_field(x: Any) -> str:
    if x is None:
        return ""

    if isinstance(x, str):
        return x

    if isinstance(x, list):
        parts: List[str] = []

        for v in x:
            if isinstance(v, list):
                parts.extend(
                    str(z)
                    for z in v
                    if z
                )
            elif v:
                parts.append(str(v))

        return " ".join(parts)

    if isinstance(x, dict):
        return " ".join(
            str(v)
            for v in x.values()
            if v
        )

    return str(x)


def item_text(
    iid: str,
    items_meta: Dict[str, Dict[str, Any]],
    use_description: bool,
    use_feature: bool,
) -> str:
    """
    Default = title + category, matching CoMemTree ranking information.
    Missing metadata => empty text, not item ID.
    """
    info = items_meta.get(str(iid), {})

    if not info:
        return ""

    title = (
        info.get("title")
        or info.get("name")
        or ""
    )

    category = clean_category(info)

    parts = [
        str(title or "").strip(),
        category,
    ]

    if use_feature:
        parts.append(
            flatten_text_field(
                info.get("feature")
            )
        )

    if use_description:
        parts.append(
            flatten_text_field(
                info.get("description")
            )
        )

    return " ".join(
        p.strip()
        for p in parts
        if p and p.strip()
    )


def tokenize(text: str) -> List[str]:
    return [
        t.lower()
        for t in TOKEN_RE.findall(
            text or ""
        )
        if t.strip()
    ]


# =============================================================================
# BM25
# =============================================================================

class BM25Index:
    """
    Self-contained BM25 implementation.

    IDF:
        log(1 + (N - df + 0.5) / (df + 0.5))
    This keeps IDF non-negative and is commonly used in BM25 implementations.
    """

    def __init__(
        self,
        docs: Dict[str, List[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)

        self.doc_tf: Dict[str, Counter] = {}
        self.doc_len: Dict[str, int] = {}
        self.df: Counter = Counter()

        for iid, tokens in docs.items():
            tf = Counter(tokens)

            self.doc_tf[str(iid)] = tf
            self.doc_len[str(iid)] = len(tokens)

            for term in tf.keys():
                self.df[term] += 1

        self.N = len(self.doc_tf)

        if self.N == 0:
            raise ValueError(
                "BM25 corpus is empty."
            )

        self.avgdl = (
            sum(self.doc_len.values())
            / float(self.N)
        )

        self.idf: Dict[str, float] = {}

        for term, df in self.df.items():
            self.idf[term] = math.log(
                1.0
                + (
                    self.N
                    - float(df)
                    + 0.5
                )
                / (
                    float(df)
                    + 0.5
                )
            )

    def score_tokens(
        self,
        query_tokens: Sequence[str],
        doc_tokens: Sequence[str],
    ) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0

        tf = Counter(doc_tokens)
        dl = len(doc_tokens)

        # Repeated query terms reflect repeated evidence in user history.
        qtf = Counter(query_tokens)

        score = 0.0

        for term, q_count in qtf.items():
            f = tf.get(term, 0)

            if f <= 0:
                continue

            idf = self.idf.get(term)

            if idf is None:
                # Term unseen in the static item-metadata corpus.
                continue

            denom = (
                f
                + self.k1
                * (
                    1.0
                    - self.b
                    + self.b
                    * (
                        dl
                        / max(
                            self.avgdl,
                            1e-12,
                        )
                    )
                )
            )

            term_score = (
                idf
                * (
                    f
                    * (
                        self.k1
                        + 1.0
                    )
                )
                / denom
            )

            score += (
                float(q_count)
                * term_score
            )

        return float(score)


def build_bm25_catalog(
    items_meta: Dict[str, Dict[str, Any]],
    use_description: bool,
    use_feature: bool,
) -> Tuple[BM25Index, Dict[str, List[str]], Dict[str, str]]:
    """
    Build corpus statistics ONLY from static items.json metadata.
    No user test/candidate information is used.
    """
    doc_tokens: Dict[str, List[str]] = {}
    doc_texts: Dict[str, str] = {}

    for iid in items_meta.keys():
        text = item_text(
            iid=iid,
            items_meta=items_meta,
            use_description=use_description,
            use_feature=use_feature,
        )

        tokens = tokenize(text)

        doc_texts[iid] = text
        doc_tokens[iid] = tokens

    return doc_tokens, doc_texts


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    ranks: Sequence[int],
    ks: Sequence[int],
) -> Dict[str, float]:
    if not ranks:
        return {}

    out: Dict[str, float] = {}
    n = float(len(ranks))

    for k in ks:
        hits = sum(
            r <= k
            for r in ranks
        )

        hit = (
            hits
            / n
        )

        ndcg = (
            sum(
                (
                    1.0
                    / math.log2(
                        r + 1.0
                    )
                )
                if r <= k
                else 0.0
                for r in ranks
            )
            / n
        )

        out[f"Hit@{k}"] = hit
        out[f"Recall@{k}"] = hit
        out[f"NDCG@{k}"] = ndcg

    out["MRR"] = (
        sum(
            1.0 / r
            for r in ranks
        )
        / n
    )

    out["MeanRank"] = float(
        np.mean(ranks)
    )

    out["MedianRank"] = float(
        np.median(ranks)
    )

    return out


# =============================================================================
# Main
# =============================================================================

def main(args: argparse.Namespace) -> None:
    # ---------------------------------------------------------
    # Train-time/static inputs only
    # ---------------------------------------------------------
    items_raw = load_json(args.items)
    sequences_raw = load_json(args.sequences)

    items_meta = normalize_items(
        items_raw
    )

    sequences = {
        str(k): v
        for k, v in sequences_raw.items()
    }

    users = select_users(
        sequences=sequences,
        eval_behaviors=args.eval_behaviors,
        user_ids_file=args.user_ids_file,
        max_users=args.max_users,
    )

    if (
        args.strict_user_count
        and args.max_users > 0
        and len(users) != args.max_users
    ):
        raise ValueError(
            f"Expected exactly {args.max_users} users, "
            f"selected {len(users)}."
        )

    # Build corpus IDF from static metadata only.
    doc_tokens, doc_texts = build_bm25_catalog(
        items_meta=items_meta,
        use_description=args.use_description,
        use_feature=args.use_feature,
    )

    bm25 = BM25Index(
        docs=doc_tokens,
        k1=args.k1,
        b=args.b,
    )

    # Build user query only from last <= history_size TRAIN interactions.
    user_queries: Dict[str, Dict[str, Any]] = {}

    for uid in users:
        train_ids = as_str_list(
            sequences[uid].get(
                "train",
                [],
            )
        )

        if args.history_size > 0:
            train_ids = train_ids[
                -args.history_size:
            ]

        if not train_ids:
            raise ValueError(
                f"user={uid}: no usable train interactions"
            )

        history_texts: List[str] = []

        for iid in train_ids:
            text = item_text(
                iid=iid,
                items_meta=items_meta,
                use_description=args.use_description,
                use_feature=args.use_feature,
            )

            if text.strip():
                history_texts.append(
                    text.strip()
                )

        query_text = " ".join(
            history_texts
        )

        query_tokens = tokenize(
            query_text
        )

        user_queries[uid] = {
            "train_history": train_ids,
            "query_text": query_text,
            "query_tokens": query_tokens,
            "n_history_with_metadata": len(
                history_texts
            ),
        }

    print("=" * 80)
    print("BM25 -- CoMemTree strict-fair baseline")
    print("=" * 80)
    print(f"Selected users               : {len(users)}")
    print(f"Max train history/user       : {args.history_size}")
    print(f"Static BM25 catalog items    : {bm25.N}")
    print(f"BM25 avg document length     : {bm25.avgdl:.3f}")
    print(f"k1 / b                       : {args.k1} / {args.b}")
    print(f"Use feature                  : {args.use_feature}")
    print(f"Use description              : {args.use_description}")
    print("Query source                  : TRAIN only")
    print("Uses old train outside cutoff: NO")
    print("Uses validation in query      : NO")
    print("Uses test/GT in query         : NO")
    print("Uses candidate set for IDF    : NO")
    print("=" * 80)

    # ---------------------------------------------------------
    # Evaluation inputs are loaded only now.
    # ---------------------------------------------------------
    negatives_raw = load_json(
        args.negatives
    )

    negatives = {
        str(k): v
        for k, v in negatives_raw.items()
    }

    candidate_rows = load_candidate_file(
        args.candidate_file
    )

    fixed_candidates: Dict[
        str,
        Dict[str, Any],
    ] = {}

    ranking_rows: List[
        Dict[str, Any]
    ] = []

    ranks: List[int] = []
    candidate_counts: List[int] = []

    n_missing_metadata_candidates = 0
    n_missing_metadata_targets = 0
    n_zero_query_users = 0

    if any(
        len(user_queries[u]["query_tokens"]) == 0
        for u in users
    ):
        n_zero_query_users = sum(
            len(user_queries[u]["query_tokens"]) == 0
            for u in users
        )

    for uid in users:
        candidates, targets = get_candidates_for_user(
            uid=uid,
            user_data=sequences[uid],
            negative_data=negatives.get(
                uid,
                {},
            ),
            candidate_rows=candidate_rows,
            seed=args.seed,
        )

        if (
            args.strict_one_target
            and len(targets) != 1
        ):
            raise ValueError(
                f"user={uid}: expected exactly 1 test target, "
                f"got {len(targets)}"
            )

        if (
            args.expected_candidates > 0
            and len(candidates) != args.expected_candidates
        ):
            raise ValueError(
                f"user={uid}: expected "
                f"{args.expected_candidates} candidates, "
                f"got {len(candidates)}"
            )

        if len(set(candidates)) != len(candidates):
            raise ValueError(
                f"user={uid}: duplicate candidate IDs detected"
            )

        for t in targets:
            if t not in candidates:
                raise ValueError(
                    f"user={uid}: target {t} is missing from candidates"
                )

        fixed_candidates[uid] = {
            "target": (
                targets[0]
                if len(targets) == 1
                else targets
            ),
            "targets": targets,
            "candidates": candidates,
        }

        candidate_counts.append(
            len(candidates)
        )

        query_tokens = user_queries[
            uid
        ]["query_tokens"]

        score_values: List[float] = []
        missing_metadata: List[str] = []

        for iid in candidates:
            # If iid is absent from items.json, CoMemTree would still keep the
            # candidate with fallback metadata. BM25 gets no lexical evidence,
            # so its document is empty and score is exactly zero.
            tokens = doc_tokens.get(
                iid,
                [],
            )

            if not tokens:
                missing_metadata.append(
                    iid
                )

            score = bm25.score_tokens(
                query_tokens=query_tokens,
                doc_tokens=tokens,
            )

            score_values.append(
                score
            )

        n_missing_metadata_candidates += len(
            missing_metadata
        )

        if any(
            t in missing_metadata
            for t in targets
        ):
            n_missing_metadata_targets += 1

        # Stable deterministic tie-break:
        # preserve original CoMemTree candidate order.
        order = sorted(
            range(len(candidates)),
            key=lambda j: (
                -score_values[j],
                j,
            ),
        )

        ranked_candidates = [
            candidates[j]
            for j in order
        ]

        target_ranks = [
            ranked_candidates.index(t) + 1
            for t in targets
            if t in ranked_candidates
        ]

        if not target_ranks:
            raise RuntimeError(
                f"user={uid}: no target found after ranking"
            )

        rank = min(
            target_ranks
        )

        ranks.append(
            rank
        )

        ranking_rows.append({
            "user_id": uid,
            "train_history": user_queries[
                uid
            ]["train_history"],
            "n_history_with_metadata": user_queries[
                uid
            ]["n_history_with_metadata"],
            "query_token_count": len(
                query_tokens
            ),
            "targets": targets,
            "candidates": candidates,
            "scores_in_candidate_order": score_values,
            "missing_metadata_candidates": missing_metadata,
            "ranked_candidates": ranked_candidates,
            "rank_position": rank,
        })

    metrics = compute_metrics(
        ranks=ranks,
        ks=[
            1,
            3,
            5,
            10,
            15,
            20,
        ],
    )

    total_eval_candidates = sum(
        candidate_counts
    )

    diagnostics = {
        "n_users": len(users),
        "candidate_count_min": min(
            candidate_counts
        ),
        "candidate_count_max": max(
            candidate_counts
        ),
        "n_eval_candidates": total_eval_candidates,
        "n_candidates_without_text_metadata": (
            n_missing_metadata_candidates
        ),
        "candidate_missing_metadata_rate": (
            n_missing_metadata_candidates
            / total_eval_candidates
            if total_eval_candidates
            else 0.0
        ),
        "n_targets_without_text_metadata": (
            n_missing_metadata_targets
        ),
        "target_missing_metadata_rate": (
            n_missing_metadata_targets
            / len(users)
            if users
            else 0.0
        ),
        "n_zero_query_users": n_zero_query_users,
        "zero_query_user_rate": (
            n_zero_query_users
            / len(users)
            if users
            else 0.0
        ),
    }

    out_dir = Path(
        args.output_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        out_dir
        / "fixed_candidates_snapshot.json",
        {
            "seed": args.seed,
            "construction": (
                "explicit candidate_file"
                if args.candidate_file
                else (
                    'test + test_neg; '
                    'random.Random(f"{seed}:{uid}").shuffle'
                )
            ),
            "n_users": len(users),
            "users": fixed_candidates,
        },
    )

    write_jsonl(
        out_dir
        / "bm25_rankings.jsonl",
        ranking_rows,
    )

    summary = {
        "method": "BM25",
        "protocol": (
            "CoMemTree_strict_fair"
        ),
        "inputs": {
            "items": args.items,
            "sequences": args.sequences,
            "negatives": args.negatives,
            "eval_behaviors": (
                args.eval_behaviors
            ),
            "user_ids_file": (
                args.user_ids_file
            ),
            "candidate_file": (
                args.candidate_file
            ),
        },
        "fairness": {
            "n_users": len(users),
            "history_size_max": (
                args.history_size
            ),
            "query_source": (
                "last <= history_size TRAIN interactions only"
            ),
            "uses_old_train_outside_cutoff": False,
            "uses_validation_in_query": False,
            "uses_test_in_query": False,
            "uses_test_candidates_for_idf": False,
            "idf_corpus": (
                "static items.json metadata only"
            ),
            "candidate_seed": (
                args.seed
            ),
            "candidate_order": (
                "same as CoMemTree"
            ),
        },
        "hyperparameters": {
            "k1": args.k1,
            "b": args.b,
            "use_feature": (
                args.use_feature
            ),
            "use_description": (
                args.use_description
            ),
            "tokenizer": (
                "lowercase alphanumeric regex"
            ),
        },
        "diagnostics": diagnostics,
        "metrics": metrics,
    }

    save_json(
        out_dir
        / "bm25_metrics.json",
        summary,
    )

    print(
        "\nEVALUATION DIAGNOSTICS"
    )
    print(
        json.dumps(
            diagnostics,
            indent=2,
        )
    )

    print(
        "\nRESULTS"
    )
    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(
        f"\nSaved to: {out_dir}"
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--items",
        default="data/CDs/items.json",
    )

    p.add_argument(
        "--sequences",
        default=(
            "data/CDs/"
            "user_sequences_10_5000.json"
        ),
    )

    p.add_argument(
        "--negatives",
        default=(
            "data/CDs/"
            "user_negatives_10_5000.json"
        ),
    )

    p.add_argument(
        "--eval-behaviors",
        default=(
            "precomputed/CDs/"
            "test_user_behaviors_gemma.jsonl"
        ),
        help=(
            "Used only to recover the exact CoMemTree evaluation-user set. "
            "Behavior text is NEVER used by BM25."
        ),
    )

    p.add_argument(
        "--user-ids-file",
        default=None,
        help=(
            "Optional explicit frozen user list. "
            "Preferred for final paper runs."
        ),
    )

    p.add_argument(
        "--candidate-file",
        default=None,
        help=(
            "Optional explicit frozen candidate file. "
            "If omitted, reproduce CoMemTree candidate construction."
        ),
    )

    p.add_argument(
        "--max-users",
        type=int,
        default=300,
    )

    p.add_argument(
        "--history-size",
        type=int,
        default=10,
    )

    p.add_argument(
        "--expected-candidates",
        type=int,
        default=20,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--strict-user-count",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--strict-one-target",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # BM25 standard defaults.
    p.add_argument(
        "--k1",
        type=float,
        default=1.5,
    )

    p.add_argument(
        "--b",
        type=float,
        default=0.75,
    )

    # OFF by default to stay aligned with CoMemTree item information.
    p.add_argument(
        "--use-feature",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    p.add_argument(
        "--use-description",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    p.add_argument(
        "--output-dir",
        default="results/bm25_cd_300",
    )

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
