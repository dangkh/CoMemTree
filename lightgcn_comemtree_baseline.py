#!/usr/bin/env python3
"""
LightGCN baseline matched to the current CoMemTree evaluation protocol.

STRICT FAIRNESS DEFAULTS
------------------------
1) Select the SAME 300 users as CoMemTree:
   - by default, intersect sequence users with the successful users in
     --eval-behaviors, preserving sequence-file order;
   - then take --max-users (default: 300).

2) Train only on up to each selected user's last 10 TRAIN interactions:
       train[-history_size:]
   A user with fewer than 10 uses all available train interactions.
   No older-train/val/test information is used by the training loss or sampler.

3) Only AFTER fitting is complete, evaluate on the SAME candidate construction
   as CoMemTree:
       test_ids + test_neg
       random.Random(f"{seed}:{user_id}").shuffle(candidates)
   Or pass --candidate-file if the CoMemTree run used an explicit frozen
   candidate file.

4) Same held-out target and same candidate order. For the default CDs protocol,
   --expected-candidates=20 enforces 1 GT + 19 negatives.

5) LightGCN uses only IDs/interactions. Item metadata is NOT used for scoring.
   --items is used only to define the catalog/item embedding table.

Outputs
-------
<output_dir>/
  lightgcn_metrics.json
  lightgcn_rankings.jsonl
  fixed_candidates_snapshot.json

Example
-------
python lightgcn_comemtree_baseline.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/user_negatives_10_5000.json \
  --eval-behaviors precomputed/CDs/test_user_behaviors_gemma_dual_v2_300.jsonl \
  --max-users 300 \
  --history-size 10 \
  --expected-candidates 20 \
  --seed 42 \
  --embedding-dim 64 \
  --n-layers 3 \
  --epochs 300 \
  --lr 1e-3 \
  --reg-weight 1e-4 \
  --output-dir results/lightgcn_cd_300x10
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange


# ---------------------------------------------------------------------
# I/O and reproducibility
# ---------------------------------------------------------------------

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if not isinstance(x, list):
        x = [x]
    return [str(v) for v in x]


# ---------------------------------------------------------------------
# User selection: mirror CoMemTree
# ---------------------------------------------------------------------

def load_successful_behavior_users(path: str | Path) -> Set[str]:
    """
    CoMemTree inference uses users that occur in both sequences and the
    successful precomputed behavior cache. We only need the user IDs here.
    """
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

            # Match the intent of the CoMemTree cache loader: only usable rows.
            ok = row.get("precompute_ok")
            behaviors = row.get("generated_behaviors")
            if ok is False:
                continue
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

    return [x.strip() for x in text.splitlines() if x.strip()]


def select_users(
    sequences: Dict[str, Dict[str, Any]],
    eval_behaviors: Optional[str],
    user_ids_file: Optional[str],
    max_users: int,
) -> List[str]:
    requested = load_user_ids_file(user_ids_file)

    if requested is not None:
        users = [uid for uid in requested if uid in sequences]
        if eval_behaviors:
            behavior_users = load_successful_behavior_users(eval_behaviors)
            users = [uid for uid in users if uid in behavior_users]
    elif eval_behaviors:
        behavior_users = load_successful_behavior_users(eval_behaviors)
        # IMPORTANT: preserve sequence-file order, matching CoMemTree.
        users = [uid for uid in sequences if uid in behavior_users]
    else:
        users = list(sequences.keys())

    if max_users > 0:
        users = users[:max_users]

    if not users:
        raise ValueError("No users selected.")

    return users


# ---------------------------------------------------------------------
# Candidate construction: mirror CoMemTree exactly
# ---------------------------------------------------------------------

def load_candidate_file(path: Optional[str]) -> Optional[Dict[str, Dict[str, Any]]]:
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
                if uid is None:
                    continue
                rows[str(uid)] = row
        return rows

    obj = load_json(p)
    if isinstance(obj, dict) and isinstance(obj.get("users"), dict):
        obj = obj["users"]
    if not isinstance(obj, dict):
        raise ValueError("Candidate file must be a JSON object or JSONL rows.")
    return {str(k): v for k, v in obj.items()}


def get_candidates_for_user(
    uid: str,
    user_data: Dict[str, Any],
    negative_data: Dict[str, Any],
    candidate_rows: Optional[Dict[str, Dict[str, Any]]],
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Exact CoMemTree default:
        candidates = test_ids + test_neg
        random.Random(f"{seed}:{uid}").shuffle(candidates)
    """
    if candidate_rows is not None and uid in candidate_rows:
        row = candidate_rows[uid]
        candidates = as_str_list(
            row.get("candidates", row.get("candidate_item_ids", []))
        )
        targets = as_str_list(
            row.get(
                "target",
                row.get(
                    "targets",
                    row.get("target_item_ids", user_data.get("test", [])),
                ),
            )
        )
        if not candidates:
            raise ValueError(f"user={uid}: empty candidate-file row")
        return candidates, targets

    test_ids = as_str_list(user_data.get("test", []))
    neg_ids = as_str_list(negative_data.get("test_neg", []))

    if not test_ids:
        raise ValueError(f"user={uid}: no test item")

    candidates = test_ids + neg_ids
    rng = random.Random(f"{seed}:{uid}")
    rng.shuffle(candidates)
    return candidates, test_ids


# ---------------------------------------------------------------------
# Catalog parsing
# ---------------------------------------------------------------------

def catalog_item_ids(items_obj: Any) -> List[str]:
    if isinstance(items_obj, dict):
        return [str(k) for k in items_obj.keys()]

    if isinstance(items_obj, list):
        out = []
        for row in items_obj:
            if not isinstance(row, dict):
                continue
            iid = (
                row.get("item_id")
                or row.get("item")
                or row.get("asin")
                or row.get("id")
            )
            if iid is not None:
                out.append(str(iid))
        return out

    raise ValueError("Unsupported items.json structure.")


# ---------------------------------------------------------------------
# LightGCN
# ---------------------------------------------------------------------

class LightGCN(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        n_layers: int,
        norm_adj: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.norm_adj = norm_adj

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ego = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        all_layers = [ego]
        x = ego

        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            all_layers.append(x)

        final = torch.stack(all_layers, dim=0).mean(dim=0)
        users, items = torch.split(final, [self.n_users, self.n_items], dim=0)
        return users, items


def build_normalized_adj(
    n_users: int,
    n_items: int,
    edges: Sequence[Tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """
    Symmetric normalized bipartite adjacency:
       D^{-1/2} A D^{-1/2}
    """
    n_nodes = n_users + n_items

    src: List[int] = []
    dst: List[int] = []

    for u, i in edges:
        ii = n_users + i
        src.extend([u, ii])
        dst.extend([ii, u])

    if not src:
        raise ValueError("Training graph has no edges.")

    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)

    deg = torch.bincount(src_t, minlength=n_nodes).float()
    deg_inv_sqrt = torch.zeros_like(deg)
    nz = deg > 0
    deg_inv_sqrt[nz] = torch.pow(deg[nz], -0.5)

    values = deg_inv_sqrt[src_t] * deg_inv_sqrt[dst_t]
    indices = torch.stack([src_t, dst_t], dim=0)

    adj = torch.sparse_coo_tensor(
        indices,
        values,
        size=(n_nodes, n_nodes),
        dtype=torch.float32,
    ).coalesce()

    return adj.to(device)


def sample_negatives(
    user_indices: np.ndarray,
    user_forbidden: List[Set[int]],
    n_items: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    One negative per positive edge.

    Strict no-leakage rule:
    only the TRAIN positives actually used to build the graph are excluded.
    Validation/test/candidate information is never consulted by this sampler.
    """
    neg = np.empty(len(user_indices), dtype=np.int64)

    for k, u in enumerate(user_indices):
        forbidden = user_forbidden[int(u)]
        while True:
            j = int(rng.integers(0, n_items))
            if j not in forbidden:
                neg[k] = j
                break

    return neg


def train_lightgcn(
    model: LightGCN,
    train_edges: Sequence[Tuple[int, int]],
    user_forbidden: List[Set[int]],
    n_items: int,
    epochs: int,
    lr: float,
    reg_weight: float,
    seed: int,
) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = model.user_embedding.weight.device

    edge_users_np = np.asarray([u for u, _ in train_edges], dtype=np.int64)
    edge_pos_np = np.asarray([i for _, i in train_edges], dtype=np.int64)

    users_t = torch.tensor(edge_users_np, dtype=torch.long, device=device)
    pos_t = torch.tensor(edge_pos_np, dtype=torch.long, device=device)

    rng = np.random.default_rng(seed)
    losses: List[float] = []

    model.train()
    pbar = trange(1, epochs + 1, desc="LightGCN training")

    for epoch in pbar:
        neg_np = sample_negatives(
            edge_users_np, user_forbidden, n_items, rng
        )
        neg_t = torch.tensor(neg_np, dtype=torch.long, device=device)

        user_final, item_final = model.propagate()

        u_e = user_final[users_t]
        p_e = item_final[pos_t]
        n_e = item_final[neg_t]

        pos_scores = torch.sum(u_e * p_e, dim=1)
        neg_scores = torch.sum(u_e * n_e, dim=1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

        # Standard LightGCN L2 regularization on ego embeddings.
        u0 = model.user_embedding(users_t)
        p0 = model.item_embedding(pos_t)
        n0 = model.item_embedding(neg_t)
        reg = (
            u0.pow(2).sum()
            + p0.pow(2).sum()
            + n0.pow(2).sum()
        ) / (2.0 * len(train_edges))

        loss = bpr + reg_weight * reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        value = float(loss.detach().cpu())
        losses.append(value)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            pbar.set_postfix(loss=f"{value:.5f}")

    return losses


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(ranks: Sequence[int], ks: Sequence[int]) -> Dict[str, float]:
    if not ranks:
        return {}

    out: Dict[str, float] = {}
    n = float(len(ranks))

    for k in ks:
        hits = sum(r <= k for r in ranks)
        hit = hits / n

        ndcg = sum(
            (1.0 / math.log2(r + 1.0)) if r <= k else 0.0
            for r in ranks
        ) / n

        out[f"Hit@{k}"] = hit
        # One ground-truth item/user => Recall@K == Hit@K.
        out[f"Recall@{k}"] = hit
        out[f"NDCG@{k}"] = ndcg

    out["MRR"] = sum(1.0 / r for r in ranks) / n
    out["MeanRank"] = float(np.mean(ranks))
    out["MedianRank"] = float(np.median(ranks))
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    sequences_raw = load_json(args.sequences)
    items_raw = load_json(args.items)

    sequences = {str(k): v for k, v in sequences_raw.items()}

    users = select_users(
        sequences=sequences,
        eval_behaviors=args.eval_behaviors,
        user_ids_file=args.user_ids_file,
        max_users=args.max_users,
    )

    if args.strict_user_count and len(users) != args.max_users:
        raise ValueError(
            f"Expected exactly {args.max_users} users, selected {len(users)}."
        )

    # Build the item universe WITHOUT consulting val/test/candidates.
    #
    # `items.json` is treated as the dataset catalog, which is available to all
    # methods. We may additionally add TRAIN-only IDs in case a train ID is
    # missing from metadata. Held-out interactions never expand the train-time
    # item universe.
    catalog = catalog_item_ids(items_raw)
    item_ids: List[str] = []
    seen_items: Set[str] = set()

    def add_item(iid: str) -> None:
        iid = str(iid)
        if iid not in seen_items:
            seen_items.add(iid)
            item_ids.append(iid)

    for iid in catalog:
        add_item(iid)

    for uid in users:
        train_ids_for_catalog = as_str_list(sequences[uid].get("train", []))
        if args.history_size > 0:
            train_ids_for_catalog = train_ids_for_catalog[-args.history_size:]
        for iid in train_ids_for_catalog:
            add_item(iid)

    user_to_idx = {uid: idx for idx, uid in enumerate(users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    # STRICT training graph: selected users x last history_size TRAIN interactions only.
    train_edges: List[Tuple[int, int]] = []
    train_histories: Dict[str, List[str]] = {}

    for uid in users:
        train_ids = as_str_list(sequences[uid].get("train", []))
        if args.history_size > 0:
            train_ids = train_ids[-args.history_size:]

        if not train_ids:
            raise ValueError(f"user={uid}: no usable train interactions")

        # CoMemTree semantics: history_size is a MAXIMUM cutoff, not an
        # exact-length requirement. A user with 9 interactions uses all 9.
        if len(train_ids) > args.history_size:
            raise AssertionError(
                f"user={uid}: internal error, history exceeds cutoff "
                f"{args.history_size}"
            )

        train_histories[uid] = train_ids
        uidx = user_to_idx[uid]

        for iid in train_ids:
            train_edges.append((uidx, item_to_idx[iid]))

    # STRICT NO-LEAKAGE negative-sampling exclusion:
    # only the exact TRAIN positives used in the graph are known at training.
    # Do NOT consult:
    #   - older train interactions outside the history cutoff,
    #   - validation interactions,
    #   - test target,
    #   - test negatives/candidate list.
    user_forbidden: List[Set[int]] = [set() for _ in users]
    for uid in users:
        uidx = user_to_idx[uid]
        user_forbidden[uidx] = {
            item_to_idx[iid]
            for iid in train_histories[uid]
            if iid in item_to_idx
        }

    norm_adj = build_normalized_adj(
        n_users=len(users),
        n_items=len(item_ids),
        edges=train_edges,
        device=device,
    )

    model = LightGCN(
        n_users=len(users),
        n_items=len(item_ids),
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        norm_adj=norm_adj,
    ).to(device)

    print("=" * 80)
    print("LightGCN -- CoMemTree strict-fair baseline")
    print("=" * 80)
    print(f"Device                    : {device}")
    print(f"Selected users            : {len(users)}")
    train_lengths = [len(train_histories[uid]) for uid in users]
    print(f"Max train interactions/user: {args.history_size}")
    print(f"Actual train length min/max: {min(train_lengths)}/{max(train_lengths)}")
    print(f"Total train edges          : {len(train_edges)}")
    print(f"Catalog/item nodes        : {len(item_ids)}")
    print(f"Candidate seed            : {args.seed}")
    print(f"Embedding dim             : {args.embedding_dim}")
    print(f"LightGCN layers           : {args.n_layers}")
    print(f"Epochs                    : {args.epochs}")
    print("Uses old train outside cutoff: NO")
    print("Uses val in training         : NO")
    print("Uses test/GT in training     : NO")
    print("Uses candidates in training  : NO")
    print("Negative sampler knows       : TRAIN positives only")
    print("Candidate construction       : EXACT CoMemTree protocol")
    print("=" * 80)

    losses = train_lightgcn(
        model=model,
        train_edges=train_edges,
        user_forbidden=user_forbidden,
        n_items=len(item_ids),
        epochs=args.epochs,
        lr=args.lr,
        reg_weight=args.reg_weight,
        seed=args.seed,
    )

    # -----------------------------------------------------------------
    # Evaluation data are loaded ONLY AFTER training is complete.
    # Nothing below this line can influence LightGCN fitting.
    # -----------------------------------------------------------------
    negatives_raw = load_json(args.negatives)
    negatives = {str(k): v for k, v in negatives_raw.items()}
    candidate_rows = load_candidate_file(args.candidate_file)

    fixed_candidates: Dict[str, Dict[str, Any]] = {}
    candidate_counts: List[int] = []

    for uid in users:
        candidates, targets = get_candidates_for_user(
            uid=uid,
            user_data=sequences[uid],
            negative_data=negatives.get(uid, {}),
            candidate_rows=candidate_rows,
            seed=args.seed,
        )

        if args.strict_one_target and len(targets) != 1:
            raise ValueError(
                f"user={uid}: expected exactly 1 test target, got {len(targets)}"
            )

        if len(set(candidates)) != len(candidates):
            raise ValueError(f"user={uid}: duplicate candidate IDs detected")

        for t in targets:
            if t not in candidates:
                raise ValueError(
                    f"user={uid}: target {t} is missing from candidates"
                )

        if (
            args.expected_candidates > 0
            and len(candidates) != args.expected_candidates
        ):
            raise ValueError(
                f"user={uid}: expected {args.expected_candidates} candidates, "
                f"got {len(candidates)}"
            )

        fixed_candidates[uid] = {
            "target": targets[0] if len(targets) == 1 else targets,
            "targets": targets,
            "candidates": candidates,
        }
        candidate_counts.append(len(candidates))

    print(
        f"Evaluation candidates min/max: "
        f"{min(candidate_counts)}/{max(candidate_counts)}"
    )

    # Final propagation once; rank only within each user's frozen candidate set.
    model.eval()
    with torch.no_grad():
        user_final, item_final = model.propagate()

    ranking_rows: List[Dict[str, Any]] = []
    ranks: List[int] = []

    for uid in users:
        uidx = user_to_idx[uid]
        row = fixed_candidates[uid]
        candidates = [str(x) for x in row["candidates"]]
        targets = [str(x) for x in row["targets"]]

        missing_from_train_catalog = [
            iid for iid in candidates if iid not in item_to_idx
        ]
        if missing_from_train_catalog:
            raise ValueError(
                f"user={uid}: evaluation candidate(s) missing from items.json/"
                f"train-time catalog: {missing_from_train_catalog[:5]}. "
                "Do not add them from test data before training; fix the dataset "
                "catalog instead."
            )

        cidx = torch.tensor(
            [item_to_idx[iid] for iid in candidates],
            dtype=torch.long,
            device=device,
        )

        scores = torch.mv(item_final[cidx], user_final[uidx])
        score_values = scores.detach().cpu().tolist()

        # Stable deterministic tie-break: preserve original candidate order.
        order = sorted(
            range(len(candidates)),
            key=lambda j: (-score_values[j], j),
        )
        ranked_candidates = [candidates[j] for j in order]

        target_ranks = [
            ranked_candidates.index(t) + 1
            for t in targets
            if t in ranked_candidates
        ]
        if not target_ranks:
            raise RuntimeError(f"user={uid}: no target found after ranking")

        rank = min(target_ranks)
        ranks.append(rank)

        ranking_rows.append(
            {
                "user_id": uid,
                "train_history": train_histories[uid],
                "targets": targets,
                "candidates": candidates,
                "scores_in_candidate_order": score_values,
                "ranked_candidates": ranked_candidates,
                "rank_position": rank,
            }
        )

    metrics = compute_metrics(ranks, ks=[1, 3, 5, 10, 15, 20])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        out_dir / "fixed_candidates_snapshot.json",
        {
            "seed": args.seed,
            "construction": (
                "explicit candidate_file"
                if args.candidate_file
                else 'test + test_neg; random.Random(f"{seed}:{uid}").shuffle'
            ),
            "n_users": len(users),
            "users": fixed_candidates,
        },
    )
    append_jsonl(out_dir / "lightgcn_rankings.jsonl", ranking_rows)

    summary = {
        "method": "LightGCN",
        "protocol": "CoMemTree_strict_fair_300x10",
        "inputs": {
            "items": args.items,
            "sequences": args.sequences,
            "negatives": args.negatives,
            "eval_behaviors": args.eval_behaviors,
            "user_ids_file": args.user_ids_file,
            "candidate_file": args.candidate_file,
        },
        "fairness": {
            "n_users": len(users),
            "history_size_max": args.history_size,
            "train_length_min": min(len(train_histories[u]) for u in users),
            "train_length_max": max(len(train_histories[u]) for u in users),
            "total_train_edges": len(train_edges),
            "train_source": "last <= history_size interactions from train only",
            "uses_old_train_outside_cutoff_in_training": False,
            "uses_val_in_training": False,
            "uses_test_in_training": False,
            "uses_test_target_to_filter_bpr_negatives": False,
            "uses_candidates_in_training": False,
            "train_item_universe": "items.json catalog + used train IDs only",
            "bpr_negative_exclusion": "only positives in the training graph",
            "candidate_seed": args.seed,
            "candidate_count_min": min(candidate_counts),
            "candidate_count_max": max(candidate_counts),
            "candidate_order": "same as CoMemTree",
        },
        "hyperparameters": {
            "embedding_dim": args.embedding_dim,
            "n_layers": args.n_layers,
            "epochs": args.epochs,
            "lr": args.lr,
            "reg_weight": args.reg_weight,
            "seed": args.seed,
        },
        "training": {
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
        },
        "metrics": metrics,
    }

    save_json(out_dir / "lightgcn_metrics.json", summary)

    print("\nRESULTS")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Same raw inputs as CoMemTree.
    p.add_argument("--items", default="data/CDs/items.json")
    p.add_argument(
        "--sequences",
        default="data/CDs/user_sequences_10_5000.json",
    )
    p.add_argument(
        "--negatives",
        default="data/CDs/user_negatives_10_5000.json",
    )

    # This is the safest way to select exactly the same evaluation users.
    p.add_argument(
        "--eval-behaviors",
        default="precomputed/CDs/test_user_behaviors_gemma_dual_v2_300.jsonl",
        help=(
            "Used only to recover the exact CoMemTree user set. "
            "Behavior text is NEVER used by LightGCN."
        ),
    )
    p.add_argument(
        "--user-ids-file",
        default=None,
        help="Optional explicit frozen user list. If supplied, it takes precedence.",
    )

    # Optional explicit frozen candidate file. If absent, reproduce CoMemTree.
    p.add_argument("--candidate-file", default=None)

    # CoMemTree protocol.
    p.add_argument("--max-users", type=int, default=300)
    p.add_argument("--history-size", type=int, default=10)
    p.add_argument("--expected-candidates", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # Fail loudly if the data do not match the intended protocol.
    p.add_argument(
        "--strict-user-count",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--strict-history-size",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Deprecated compatibility flag. CoMemTree treats history-size as "
            "a maximum cutoff, so exact length is not required."
        ),
    )
    p.add_argument(
        "--strict-one-target",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # LightGCN.
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg-weight", type=float, default=1e-4)
    p.add_argument("--device", default="auto")

    p.add_argument(
        "--output-dir",
        default="results/lightgcn_cd_300x10",
    )

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
