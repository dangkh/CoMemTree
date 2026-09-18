#!/usr/bin/env python3
"""
Unified recommendation baselines for the current CoMemTree protocol.

Included methods
----------------
  lightgcn
  bert4rec
  sasrec

Shared protocol
---------------
- Same selected evaluation users as CoMemTree.
- Use only the last <= history_size TRAIN interactions per selected user.
- No val/test/candidate information is used by the training objective.
- Evaluation candidates are:
      test_ids + test_neg
      random.Random(f"{seed}:{user_id}").shuffle(candidates)
- Default CDs protocol:
      max_users=300
      history_size=10
      expected_candidates=20
      seed=42

Usage
-----
LightGCN:
  python comemtree_baselines.py lightgcn \
    --items data/CDs/items.json \
    --sequences data/CDs/user_sequences_10_5000.json \
    --negatives data/CDs/user_negatives_10_5000.json \
    --eval-behaviors precomputed/CDs/test_user_behaviors_gemma.jsonl

BERT4Rec:
  python comemtree_baselines.py bert4rec \
    --items data/CDs/items.json \
    --sequences data/CDs/user_sequences_10_5000.json \
    --negatives data/CDs/user_negatives_10_5000.json \
    --eval-behaviors precomputed/CDs/test_user_behaviors_gemma.jsonl

SASRec:
  python comemtree_baselines.py sasrec \
    --items data/CDs/items.json \
    --sequences data/CDs/user_sequences_10_5000.json \
    --negatives data/CDs/user_negatives_10_5000.json \
    --eval-behaviors precomputed/CDs/test_user_behaviors_gemma.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import trange


# =============================================================================
# Shared helpers
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


def catalog_item_ids(items_obj: Any) -> List[str]:
    """
    Current Amazon items.json is normally keyed by item/ASIN.
    Also tolerate list-of-dicts.
    """
    if isinstance(items_obj, dict):
        return [str(k) for k in items_obj.keys()]

    if isinstance(items_obj, list):
        out: List[str] = []
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


# =============================================================================
# Shared user selection
# =============================================================================

def load_successful_behavior_users(path: str | Path) -> Set[str]:
    """
    Behavior text is never used by these baselines.
    This file is used only to recover the exact CoMemTree evaluation-user set.
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

        # Preserve sequence-file order, matching current CoMemTree behavior.
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


def build_train_histories(
    sequences: Dict[str, Dict[str, Any]],
    users: Sequence[str],
    history_size: int,
) -> Dict[str, List[str]]:
    train_histories: Dict[str, List[str]] = {}

    for uid in users:
        train_ids = as_str_list(
            sequences[uid].get("train", [])
        )

        if history_size > 0:
            train_ids = train_ids[-history_size:]

        if not train_ids:
            raise ValueError(
                f"user={uid}: no usable train interactions"
            )

        train_histories[uid] = train_ids

    return train_histories


def build_train_item_vocabulary(
    items_raw: Any,
    train_histories: Dict[str, List[str]],
    users: Sequence[str],
) -> List[str]:
    """
    Preserves the vocabulary policy of the uploaded baseline files:
      static items.json catalog + allowed TRAIN IDs.

    Val/test/candidate IDs never expand the vocabulary before fitting.
    """
    item_ids: List[str] = []
    seen_items: Set[str] = set()

    def add_item(iid: str) -> None:
        iid = str(iid)
        if iid not in seen_items:
            seen_items.add(iid)
            item_ids.append(iid)

    for iid in catalog_item_ids(items_raw):
        add_item(iid)

    for uid in users:
        for iid in train_histories[uid]:
            add_item(iid)

    if not item_ids:
        raise ValueError("Empty train-time item vocabulary.")

    return item_ids


# =============================================================================
# Shared candidate construction
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
    if candidate_rows is not None and uid in candidate_rows:
        row = candidate_rows[uid]

        candidates = as_str_list(
            row.get(
                "candidates",
                row.get("candidate_item_ids", []),
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


def validate_candidates(
    uid: str,
    candidates: List[str],
    targets: List[str],
    expected_candidates: int,
    strict_one_target: bool,
) -> None:
    if strict_one_target and len(targets) != 1:
        raise ValueError(
            f"user={uid}: expected exactly 1 test target, "
            f"got {len(targets)}"
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

    if (
        expected_candidates > 0
        and len(candidates) != expected_candidates
    ):
        raise ValueError(
            f"user={uid}: expected {expected_candidates} candidates, "
            f"got {len(candidates)}"
        )


def load_evaluation_candidates(
    args: argparse.Namespace,
    sequences: Dict[str, Dict[str, Any]],
    users: Sequence[str],
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    """
    Called only AFTER the selected model has finished fitting.

    Returns:
      fixed_candidates
      negatives
    """
    negatives_raw = load_json(args.negatives)
    negatives = {
        str(k): v
        for k, v in negatives_raw.items()
    }

    candidate_rows = load_candidate_file(
        args.candidate_file
    )

    fixed_candidates: Dict[str, Dict[str, Any]] = {}

    for uid in users:
        candidates, targets = get_candidates_for_user(
            uid=uid,
            user_data=sequences[uid],
            negative_data=negatives.get(uid, {}),
            candidate_rows=candidate_rows,
            seed=args.seed,
        )

        validate_candidates(
            uid=uid,
            candidates=candidates,
            targets=targets,
            expected_candidates=args.expected_candidates,
            strict_one_target=args.strict_one_target,
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

    return fixed_candidates, negatives


# =============================================================================
# Shared metrics/output
# =============================================================================

def compute_metrics(
    ranks: Sequence[int],
    ks: Sequence[int] = (1, 3, 5, 10, 15, 20),
) -> Dict[str, float]:
    if not ranks:
        return {}

    out: Dict[str, float] = {}
    n = float(len(ranks))

    for k in ks:
        hit = (
            sum(r <= k for r in ranks)
            / n
        )

        ndcg = (
            sum(
                (
                    1.0
                    / math.log2(r + 1.0)
                )
                if r <= k
                else 0.0
                for r in ranks
            )
            / n
        )

        out[f"Hit@{k}"] = hit

        # Current protocol has one ground-truth item/user.
        out[f"Recall@{k}"] = hit

        out[f"NDCG@{k}"] = ndcg

    out["MRR"] = (
        sum(1.0 / r for r in ranks)
        / n
    )

    out["MeanRank"] = float(
        np.mean(ranks)
    )

    out["MedianRank"] = float(
        np.median(ranks)
    )

    return out


def save_candidate_snapshot(
    out_dir: Path,
    args: argparse.Namespace,
    fixed_candidates: Dict[str, Dict[str, Any]],
) -> None:
    save_json(
        out_dir / "fixed_candidates_snapshot.json",
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
            "n_users": len(fixed_candidates),
            "users": fixed_candidates,
        },
    )


def common_protocol_summary(
    args: argparse.Namespace,
    users: Sequence[str],
    train_histories: Dict[str, List[str]],
) -> Dict[str, Any]:
    return {
        "n_users": len(users),
        "history_size_max": args.history_size,
        "train_length_min": min(
            len(train_histories[u])
            for u in users
        ),
        "train_length_max": max(
            len(train_histories[u])
            for u in users
        ),
        "train_source": (
            "last <= history_size interactions from train only"
        ),
        "uses_old_train_outside_cutoff": False,
        "uses_validation_in_training": False,
        "uses_test_in_training": False,
        "uses_candidates_in_training": False,
        "candidate_seed": args.seed,
        "candidate_order": "same as CoMemTree",
    }


def load_common_train_inputs(
    args: argparse.Namespace,
) -> Tuple[
    Any,
    Dict[str, Dict[str, Any]],
    List[str],
    Dict[str, List[str]],
]:
    """
    Load only static/train-side inputs before fitting.

    IMPORTANT:
    --negatives and --candidate-file are intentionally NOT loaded here.
    """
    items_raw = load_json(args.items)
    sequences_raw = load_json(args.sequences)

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

    train_histories = build_train_histories(
        sequences=sequences,
        users=users,
        history_size=args.history_size,
    )

    return (
        items_raw,
        sequences,
        users,
        train_histories,
    )


# =============================================================================
# LightGCN
# =============================================================================

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

        self.user_embedding = nn.Embedding(
            n_users,
            embedding_dim,
        )

        self.item_embedding = nn.Embedding(
            n_items,
            embedding_dim,
        )

        nn.init.normal_(
            self.user_embedding.weight,
            std=0.1,
        )

        nn.init.normal_(
            self.item_embedding.weight,
            std=0.1,
        )

    def propagate(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ego = torch.cat(
            [
                self.user_embedding.weight,
                self.item_embedding.weight,
            ],
            dim=0,
        )

        all_layers = [ego]
        x = ego

        for _ in range(self.n_layers):
            x = torch.sparse.mm(
                self.norm_adj,
                x,
            )
            all_layers.append(x)

        final = torch.stack(
            all_layers,
            dim=0,
        ).mean(dim=0)

        users, items = torch.split(
            final,
            [
                self.n_users,
                self.n_items,
            ],
            dim=0,
        )

        return users, items


def build_normalized_adj(
    n_users: int,
    n_items: int,
    edges: Sequence[Tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    n_nodes = n_users + n_items

    src: List[int] = []
    dst: List[int] = []

    for u, i in edges:
        ii = n_users + i
        src.extend([u, ii])
        dst.extend([ii, u])

    if not src:
        raise ValueError(
            "Training graph has no edges."
        )

    src_t = torch.tensor(
        src,
        dtype=torch.long,
    )

    dst_t = torch.tensor(
        dst,
        dtype=torch.long,
    )

    deg = torch.bincount(
        src_t,
        minlength=n_nodes,
    ).float()

    deg_inv_sqrt = torch.zeros_like(
        deg
    )

    nz = deg > 0
    deg_inv_sqrt[nz] = torch.pow(
        deg[nz],
        -0.5,
    )

    values = (
        deg_inv_sqrt[src_t]
        * deg_inv_sqrt[dst_t]
    )

    indices = torch.stack(
        [src_t, dst_t],
        dim=0,
    )

    adj = torch.sparse_coo_tensor(
        indices,
        values,
        size=(n_nodes, n_nodes),
        dtype=torch.float32,
    ).coalesce()

    return adj.to(device)


def sample_lightgcn_negatives(
    user_indices: np.ndarray,
    user_forbidden: List[Set[int]],
    n_items: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Only TRAIN positives actually used in the graph are forbidden.
    Val/test/candidates are never consulted.
    """
    neg = np.empty(
        len(user_indices),
        dtype=np.int64,
    )

    for k, u in enumerate(user_indices):
        forbidden = user_forbidden[int(u)]

        while True:
            j = int(
                rng.integers(
                    0,
                    n_items,
                )
            )

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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    device = (
        model.user_embedding.weight.device
    )

    edge_users_np = np.asarray(
        [u for u, _ in train_edges],
        dtype=np.int64,
    )

    edge_pos_np = np.asarray(
        [i for _, i in train_edges],
        dtype=np.int64,
    )

    users_t = torch.tensor(
        edge_users_np,
        dtype=torch.long,
        device=device,
    )

    pos_t = torch.tensor(
        edge_pos_np,
        dtype=torch.long,
        device=device,
    )

    rng = np.random.default_rng(
        seed
    )

    losses: List[float] = []

    model.train()

    pbar = trange(
        1,
        epochs + 1,
        desc="LightGCN training",
    )

    for epoch in pbar:
        neg_np = sample_lightgcn_negatives(
            edge_users_np,
            user_forbidden,
            n_items,
            rng,
        )

        neg_t = torch.tensor(
            neg_np,
            dtype=torch.long,
            device=device,
        )

        user_final, item_final = (
            model.propagate()
        )

        u_e = user_final[users_t]
        p_e = item_final[pos_t]
        n_e = item_final[neg_t]

        pos_scores = torch.sum(
            u_e * p_e,
            dim=1,
        )

        neg_scores = torch.sum(
            u_e * n_e,
            dim=1,
        )

        bpr = -F.logsigmoid(
            pos_scores - neg_scores
        ).mean()

        u0 = model.user_embedding(
            users_t
        )

        p0 = model.item_embedding(
            pos_t
        )

        n0 = model.item_embedding(
            neg_t
        )

        reg = (
            u0.pow(2).sum()
            + p0.pow(2).sum()
            + n0.pow(2).sum()
        ) / (
            2.0
            * len(train_edges)
        )

        loss = (
            bpr
            + reg_weight * reg
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()
        optimizer.step()

        value = float(
            loss.detach().cpu()
        )

        losses.append(value)

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            pbar.set_postfix(
                loss=f"{value:.5f}"
            )

    return losses


def run_lightgcn(
    args: argparse.Namespace,
) -> None:
    set_seed(args.seed)

    device = torch.device(
        args.device
        if args.device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    (
        items_raw,
        sequences,
        users,
        train_histories,
    ) = load_common_train_inputs(args)

    item_ids = build_train_item_vocabulary(
        items_raw,
        train_histories,
        users,
    )

    user_to_idx = {
        uid: idx
        for idx, uid in enumerate(users)
    }

    item_to_idx = {
        iid: idx
        for idx, iid in enumerate(item_ids)
    }

    train_edges: List[Tuple[int, int]] = []

    for uid in users:
        uidx = user_to_idx[uid]

        for iid in train_histories[uid]:
            train_edges.append(
                (
                    uidx,
                    item_to_idx[iid],
                )
            )

    user_forbidden: List[Set[int]] = [
        set()
        for _ in users
    ]

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

    print("=" * 88)
    print("LightGCN -- CoMemTree strict-fair baseline")
    print("=" * 88)
    print(f"Device                     : {device}")
    print(f"Selected users             : {len(users)}")
    print(f"Max train history/user     : {args.history_size}")
    print(
        f"Actual train length min/max: "
        f"{min(len(train_histories[u]) for u in users)}/"
        f"{max(len(train_histories[u]) for u in users)}"
    )
    print(f"Total train edges          : {len(train_edges)}")
    print(f"Train-time item vocabulary : {len(item_ids)}")
    print(f"Embedding dim              : {args.embedding_dim}")
    print(f"Layers                     : {args.n_layers}")
    print(f"Epochs                     : {args.epochs}")
    print("Uses val/test in training  : NO")
    print("=" * 88)

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

    # Evaluation inputs loaded only after fitting.
    fixed_candidates, _ = (
        load_evaluation_candidates(
            args,
            sequences,
            users,
        )
    )

    model.eval()

    with torch.no_grad():
        user_final, item_final = (
            model.propagate()
        )

    ranking_rows: List[Dict[str, Any]] = []
    ranks: List[int] = []

    for uid in users:
        uidx = user_to_idx[uid]

        row = fixed_candidates[uid]

        candidates = [
            str(x)
            for x in row["candidates"]
        ]

        targets = [
            str(x)
            for x in row["targets"]
        ]

        known_positions = [
            j
            for j, iid in enumerate(candidates)
            if iid in item_to_idx
        ]

        # Preserve uploaded LightGCN cold-start policy.
        score_values = [
            0.0
            for _ in candidates
        ]

        if known_positions:
            known_item_indices = torch.tensor(
                [
                    item_to_idx[candidates[j]]
                    for j in known_positions
                ],
                dtype=torch.long,
                device=device,
            )

            known_scores = torch.mv(
                item_final[known_item_indices],
                user_final[uidx],
            ).detach().cpu().tolist()

            for j, s in zip(
                known_positions,
                known_scores,
            ):
                score_values[j] = float(s)

        cold_start_candidates = [
            iid
            for iid in candidates
            if iid not in item_to_idx
        ]

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
                f"user={uid}: target missing after ranking"
            )

        rank = min(target_ranks)
        ranks.append(rank)

        ranking_rows.append({
            "user_id": uid,
            "train_history": train_histories[uid],
            "targets": targets,
            "candidates": candidates,
            "scores_in_candidate_order": score_values,
            "cold_start_candidates": cold_start_candidates,
            "n_cold_start_candidates": len(
                cold_start_candidates
            ),
            "target_is_cold_start": any(
                t not in item_to_idx
                for t in targets
            ),
            "ranked_candidates": ranked_candidates,
            "rank_position": rank,
        })

    metrics = compute_metrics(ranks)

    total_eval_candidates = sum(
        len(r["candidates"])
        for r in ranking_rows
    )

    total_cold_start = sum(
        int(r["n_cold_start_candidates"])
        for r in ranking_rows
    )

    cold_start_targets = sum(
        bool(r["target_is_cold_start"])
        for r in ranking_rows
    )

    diagnostics = {
        "n_eval_candidates": total_eval_candidates,
        "n_cold_start_candidates": total_cold_start,
        "cold_start_candidate_rate": (
            total_cold_start
            / total_eval_candidates
            if total_eval_candidates
            else 0.0
        ),
        "n_cold_start_targets": cold_start_targets,
        "cold_start_target_rate": (
            cold_start_targets
            / len(ranking_rows)
            if ranking_rows
            else 0.0
        ),
        "cold_start_policy": (
            "zero embedding => score 0.0 after model freeze"
        ),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_candidate_snapshot(
        out_dir,
        args,
        fixed_candidates,
    )

    write_jsonl(
        out_dir / "lightgcn_rankings.jsonl",
        ranking_rows,
    )

    summary = {
        "method": "LightGCN",
        "protocol": "CoMemTree_strict_fair",
        "inputs": {
            "items": args.items,
            "sequences": args.sequences,
            "negatives": args.negatives,
            "eval_behaviors": args.eval_behaviors,
            "user_ids_file": args.user_ids_file,
            "candidate_file": args.candidate_file,
        },
        "fairness": {
            **common_protocol_summary(
                args,
                users,
                train_histories,
            ),
            "total_train_edges": len(train_edges),
            "train_item_vocabulary": (
                "items.json catalog + allowed TRAIN IDs only"
            ),
            "bpr_negative_exclusion": (
                "only positives in the training graph"
            ),
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
            "final_loss": (
                losses[-1]
                if losses
                else None
            ),
            "min_loss": (
                min(losses)
                if losses
                else None
            ),
        },
        "diagnostics": diagnostics,
        "metrics": metrics,
    }

    save_json(
        out_dir / "lightgcn_metrics.json",
        summary,
    )

    print("\nCOLD-START DIAGNOSTICS")
    print(
        json.dumps(
            diagnostics,
            indent=2,
        )
    )

    print("\nRESULTS")
    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(f"\nSaved to: {out_dir}")


# =============================================================================
# BERT4Rec
# =============================================================================

class BertTrainSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: List[List[int]],
    ) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self,
        idx: int,
    ) -> List[int]:
        return self.sequences[idx]


class Bert4RecCollator:
    """
    Dynamic Cloze masking on TRAIN sequences only.

    0 = PAD
    1 = MASK
    real item embedding IDs start from 2
    """

    def __init__(
        self,
        max_len: int,
        mask_prob: float,
        n_real_items: int,
        seed: int,
    ) -> None:
        self.max_len = int(max_len)
        self.mask_prob = float(mask_prob)
        self.n_real_items = int(n_real_items)
        self.rng = random.Random(seed)

    def _random_real_item_embedding_id(
        self,
    ) -> int:
        return self.rng.randint(
            2,
            self.n_real_items + 1,
        )

    def __call__(
        self,
        batch: Sequence[List[int]],
    ) -> Dict[str, torch.Tensor]:
        bsz = len(batch)

        input_ids = torch.zeros(
            (bsz, self.max_len),
            dtype=torch.long,
        )

        labels = torch.full(
            (bsz, self.max_len),
            fill_value=-100,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (bsz, self.max_len),
            dtype=torch.bool,
        )

        for b, seq in enumerate(batch):
            seq = seq[-self.max_len:]

            if not seq:
                continue

            L = len(seq)

            input_ids[b, :L] = torch.tensor(
                seq,
                dtype=torch.long,
            )

            attention_mask[b, :L] = True

            selected: List[int] = []

            for pos in range(L):
                if (
                    self.rng.random()
                    < self.mask_prob
                ):
                    selected.append(pos)

            if not selected:
                selected = [
                    self.rng.randrange(L)
                ]

            for pos in selected:
                original = int(
                    input_ids[b, pos].item()
                )

                labels[b, pos] = original

                r = self.rng.random()

                if r < 0.8:
                    input_ids[b, pos] = 1

                elif r < 0.9:
                    input_ids[
                        b,
                        pos,
                    ] = (
                        self._random_real_item_embedding_id()
                    )

                else:
                    pass

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class BERT4Rec(nn.Module):
    def __init__(
        self,
        n_real_items: int,
        max_len: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.n_real_items = int(
            n_real_items
        )

        self.max_len = int(
            max_len
        )

        self.hidden_size = int(
            hidden_size
        )

        self.item_embedding = nn.Embedding(
            n_real_items + 2,
            hidden_size,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            max_len,
            hidden_size,
        )

        self.embedding_norm = nn.LayerNorm(
            hidden_size
        )

        self.embedding_dropout = nn.Dropout(
            dropout
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

        self.output_norm = nn.LayerNorm(
            hidden_size
        )

        self.output_bias = nn.Parameter(
            torch.zeros(
                n_real_items
            )
        )

        self.reset_parameters()

    def reset_parameters(
        self,
    ) -> None:
        nn.init.normal_(
            self.item_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        with torch.no_grad():
            self.item_embedding.weight[
                0
            ].zero_()

        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L = input_ids.shape

        pos = torch.arange(
            L,
            device=input_ids.device,
        ).unsqueeze(0).expand(
            B,
            L,
        )

        x = (
            self.item_embedding(
                input_ids
            )
            + self.position_embedding(
                pos
            )
        )

        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        key_padding_mask = (
            ~attention_mask.bool()
        )

        x = self.encoder(
            x,
            src_key_padding_mask=(
                key_padding_mask
            ),
        )

        return self.output_norm(x)

    def masked_logits(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        mask = labels != -100

        masked_hidden = hidden[mask]

        masked_labels_embedding_ids = (
            labels[mask]
        )

        if masked_hidden.numel() == 0:
            raise RuntimeError(
                "No masked positions in batch."
            )

        real_item_weights = (
            self.item_embedding.weight[2:]
        )

        logits = (
            masked_hidden
            @ real_item_weights.t()
            + self.output_bias
        )

        target_classes = (
            masked_labels_embedding_ids
            - 2
        )

        return (
            logits,
            target_classes,
        )

    @torch.no_grad()
    def score_real_item_classes(
        self,
        mask_hidden: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        emb_ids = class_ids + 2

        item_vecs = self.item_embedding(
            emb_ids
        )

        bias = self.output_bias[
            class_ids
        ]

        return (
            item_vecs
            @ mask_hidden
            + bias
        )


def run_bert4rec(
    args: argparse.Namespace,
) -> None:
    set_seed(args.seed)

    device = torch.device(
        args.device
        if args.device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    (
        items_raw,
        sequences,
        users,
        train_histories,
    ) = load_common_train_inputs(args)

    item_ids = build_train_item_vocabulary(
        items_raw,
        train_histories,
        users,
    )

    item_to_class = {
        iid: idx
        for idx, iid in enumerate(item_ids)
    }

    item_to_embedding_id = {
        iid: idx + 2
        for iid, idx in item_to_class.items()
    }

    encoded_train_sequences: List[
        List[int]
    ] = []

    for uid in users:
        encoded_train_sequences.append([
            item_to_embedding_id[iid]
            for iid in train_histories[uid]
        ])

    dataset = BertTrainSequenceDataset(
        encoded_train_sequences
    )

    model_max_len = (
        args.history_size + 1
        if args.history_size > 0
        else (
            max(
                len(x)
                for x in encoded_train_sequences
            )
            + 1
        )
    )

    collator = Bert4RecCollator(
        max_len=model_max_len,
        mask_prob=args.mask_prob,
        n_real_items=len(item_ids),
        seed=args.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    model = BERT4Rec(
        n_real_items=len(item_ids),
        max_len=model_max_len,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("=" * 88)
    print("BERT4Rec -- CoMemTree strict-fair baseline")
    print("=" * 88)
    print(f"Device                     : {device}")
    print(f"Selected users             : {len(users)}")
    print(f"Max train history/user     : {args.history_size}")
    print(
        f"Actual train length min/max: "
        f"{min(len(train_histories[u]) for u in users)}/"
        f"{max(len(train_histories[u]) for u in users)}"
    )
    print(f"Train-time item vocabulary : {len(item_ids)}")
    print(f"Model max sequence length  : {model_max_len}")
    print(f"Hidden size                : {args.hidden_size}")
    print(f"Layers / heads             : {args.n_layers} / {args.n_heads}")
    print(f"Mask probability           : {args.mask_prob}")
    print(f"Batch size                 : {args.batch_size}")
    print(f"Epochs                     : {args.epochs}")
    print("Uses val/test in training  : NO")
    print("=" * 88)

    epoch_losses: List[float] = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        running_loss = 0.0
        n_batches = 0

        for batch in loader:
            input_ids = batch[
                "input_ids"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            hidden = model.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits, targets = (
                model.masked_logits(
                    hidden=hidden,
                    labels=labels,
                )
            )

            loss = F.cross_entropy(
                logits,
                targets,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            running_loss += float(
                loss.detach().cpu()
            )

            n_batches += 1

        mean_loss = (
            running_loss
            / max(
                n_batches,
                1,
            )
        )

        epoch_losses.append(
            mean_loss
        )

        if (
            epoch == 1
            or epoch % args.log_every == 0
            or epoch == args.epochs
        ):
            print(
                f"epoch={epoch:04d} "
                f"loss={mean_loss:.6f}"
            )

    model.eval()

    # Evaluation inputs loaded only after fitting.
    fixed_candidates, _ = (
        load_evaluation_candidates(
            args,
            sequences,
            users,
        )
    )

    ranking_rows: List[
        Dict[str, Any]
    ] = []

    ranks: List[int] = []

    total_oov_candidates = 0
    total_oov_targets = 0

    with torch.no_grad():
        for uid in users:
            row = fixed_candidates[
                uid
            ]

            candidates = [
                str(x)
                for x in row["candidates"]
            ]

            targets = [
                str(x)
                for x in row["targets"]
            ]

            history_ids = (
                train_histories[uid][
                    -args.history_size:
                ]
                if args.history_size > 0
                else train_histories[uid]
            )

            encoded_history = [
                item_to_embedding_id[iid]
                for iid in history_ids
            ]

            eval_seq = (
                encoded_history
                + [1]
            )

            if (
                len(eval_seq)
                > model_max_len
            ):
                eval_seq = eval_seq[
                    -model_max_len:
                ]

            input_ids = torch.zeros(
                (1, model_max_len),
                dtype=torch.long,
                device=device,
            )

            attention_mask = torch.zeros(
                (1, model_max_len),
                dtype=torch.bool,
                device=device,
            )

            L = len(eval_seq)

            input_ids[
                0,
                :L,
            ] = torch.tensor(
                eval_seq,
                dtype=torch.long,
                device=device,
            )

            attention_mask[
                0,
                :L,
            ] = True

            hidden = model.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            mask_hidden = hidden[
                0,
                L - 1,
            ]

            score_values = [
                float("-inf")
                for _ in candidates
            ]

            known_positions: List[int] = []
            known_class_ids: List[int] = []
            oov_candidates: List[str] = []

            for j, iid in enumerate(
                candidates
            ):
                class_id = item_to_class.get(
                    iid
                )

                if class_id is None:
                    oov_candidates.append(
                        iid
                    )
                else:
                    known_positions.append(
                        j
                    )

                    known_class_ids.append(
                        class_id
                    )

            if known_class_ids:
                class_tensor = torch.tensor(
                    known_class_ids,
                    dtype=torch.long,
                    device=device,
                )

                known_scores = (
                    model.score_real_item_classes(
                        mask_hidden=mask_hidden,
                        class_ids=class_tensor,
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )

                for j, s in zip(
                    known_positions,
                    known_scores,
                ):
                    score_values[j] = float(s)

            total_oov_candidates += len(
                oov_candidates
            )

            target_is_oov = any(
                t not in item_to_class
                for t in targets
            )

            if target_is_oov:
                total_oov_targets += 1

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
                    f"user={uid}: target missing after ranking"
                )

            rank = min(
                target_ranks
            )

            ranks.append(rank)

            ranking_rows.append({
                "user_id": uid,
                "train_history": history_ids,
                "targets": targets,
                "candidates": candidates,
                "scores_in_candidate_order": score_values,
                "oov_candidates": oov_candidates,
                "n_oov_candidates": len(
                    oov_candidates
                ),
                "target_is_oov": target_is_oov,
                "ranked_candidates": ranked_candidates,
                "rank_position": rank,
            })

    metrics = compute_metrics(
        ranks
    )

    total_eval_candidates = sum(
        len(r["candidates"])
        for r in ranking_rows
    )

    diagnostics = {
        "n_users": len(users),
        "n_eval_candidates": total_eval_candidates,
        "n_oov_candidates": total_oov_candidates,
        "oov_candidate_rate": (
            total_oov_candidates
            / total_eval_candidates
            if total_eval_candidates
            else 0.0
        ),
        "n_oov_targets": total_oov_targets,
        "oov_target_rate": (
            total_oov_targets
            / len(users)
            if users
            else 0.0
        ),
        "oov_policy": (
            "-inf score; held-out IDs do not expand vocabulary before training"
        ),
    }

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_candidate_snapshot(
        out_dir,
        args,
        fixed_candidates,
    )

    write_jsonl(
        out_dir / "bert4rec_rankings.jsonl",
        ranking_rows,
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "item_ids": item_ids,
            "item_to_class": item_to_class,
            "config": vars(args),
            "model_max_len": model_max_len,
        },
        out_dir / "bert4rec_model.pt",
    )

    summary = {
        "method": "BERT4Rec",
        "protocol": "CoMemTree_strict_fair",
        "inputs": {
            "items": args.items,
            "sequences": args.sequences,
            "negatives": args.negatives,
            "eval_behaviors": args.eval_behaviors,
            "user_ids_file": args.user_ids_file,
            "candidate_file": args.candidate_file,
        },
        "fairness": {
            **common_protocol_summary(
                args,
                users,
                train_histories,
            ),
            "train_item_vocabulary": (
                "items.json catalog + allowed TRAIN IDs only"
            ),
        },
        "hyperparameters": {
            "hidden_size": args.hidden_size,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "dropout": args.dropout,
            "mask_prob": args.mask_prob,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
        "training": {
            "final_loss": (
                epoch_losses[-1]
                if epoch_losses
                else None
            ),
            "min_loss": (
                min(epoch_losses)
                if epoch_losses
                else None
            ),
        },
        "diagnostics": diagnostics,
        "metrics": metrics,
    }

    save_json(
        out_dir / "bert4rec_metrics.json",
        summary,
    )

    print("\nEVALUATION DIAGNOSTICS")
    print(
        json.dumps(
            diagnostics,
            indent=2,
        )
    )

    print("\nRESULTS")
    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(f"\nSaved to: {out_dir}")


# =============================================================================
# SASRec
# =============================================================================

class SASRecTrainDataset(Dataset):
    def __init__(
        self,
        sequences: List[List[int]],
    ) -> None:
        self.sequences = [
            seq
            for seq in sequences
            if len(seq) >= 2
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self,
        idx: int,
    ) -> List[int]:
        return self.sequences[idx]


class SASRecCollator:
    def __init__(
        self,
        max_len: int,
    ) -> None:
        self.max_len = int(max_len)

    def __call__(
        self,
        batch: Sequence[List[int]],
    ) -> Dict[str, torch.Tensor]:
        bsz = len(batch)

        input_ids = torch.zeros(
            (bsz, self.max_len),
            dtype=torch.long,
        )

        labels = torch.full(
            (bsz, self.max_len),
            fill_value=-100,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (bsz, self.max_len),
            dtype=torch.bool,
        )

        for b, seq in enumerate(batch):
            seq = seq[
                -(self.max_len + 1):
            ]

            inp = seq[:-1]
            tgt = seq[1:]

            if len(inp) > self.max_len:
                inp = inp[
                    -self.max_len:
                ]

                tgt = tgt[
                    -self.max_len:
                ]

            L = len(inp)

            input_ids[
                b,
                :L,
            ] = torch.tensor(
                inp,
                dtype=torch.long,
            )

            labels[
                b,
                :L,
            ] = torch.tensor(
                tgt,
                dtype=torch.long,
            )

            attention_mask[
                b,
                :L,
            ] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        max_len: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.n_items = int(
            n_items
        )

        self.max_len = int(
            max_len
        )

        self.hidden_size = int(
            hidden_size
        )

        self.item_embedding = nn.Embedding(
            n_items + 1,
            hidden_size,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            max_len,
            hidden_size,
        )

        self.embedding_dropout = nn.Dropout(
            dropout
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

        self.final_norm = nn.LayerNorm(
            hidden_size
        )

        self.output_bias = nn.Parameter(
            torch.zeros(
                n_items
            )
        )

        self.reset_parameters()

    def reset_parameters(
        self,
    ) -> None:
        nn.init.normal_(
            self.item_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        with torch.no_grad():
            self.item_embedding.weight[
                0
            ].zero_()

        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L = input_ids.shape

        positions = torch.arange(
            L,
            device=input_ids.device,
        ).unsqueeze(0).expand(
            B,
            L,
        )

        x = (
            self.item_embedding(
                input_ids
            )
            + self.position_embedding(
                positions
            )
        )

        x = self.embedding_dropout(x)

        causal_mask = torch.triu(
            torch.ones(
                (L, L),
                dtype=torch.bool,
                device=input_ids.device,
            ),
            diagonal=1,
        )

        key_padding_mask = (
            ~attention_mask.bool()
        )

        x = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=(
                key_padding_mask
            ),
        )

        return self.final_norm(x)

    def next_item_logits(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        real_item_weights = (
            self.item_embedding.weight[1:]
        )

        return (
            hidden
            @ real_item_weights.t()
            + self.output_bias
        )

    @torch.no_grad()
    def score_item_classes(
        self,
        hidden: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        embedding_ids = (
            class_ids + 1
        )

        item_vecs = self.item_embedding(
            embedding_ids
        )

        bias = self.output_bias[
            class_ids
        ]

        return (
            item_vecs
            @ hidden
            + bias
        )


def run_sasrec(
    args: argparse.Namespace,
) -> None:
    set_seed(args.seed)

    device = torch.device(
        args.device
        if args.device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    (
        items_raw,
        sequences,
        users,
        train_histories,
    ) = load_common_train_inputs(args)

    item_ids = build_train_item_vocabulary(
        items_raw,
        train_histories,
        users,
    )

    item_to_class = {
        iid: idx
        for idx, iid in enumerate(item_ids)
    }

    item_to_embedding_id = {
        iid: idx + 1
        for iid, idx in item_to_class.items()
    }

    encoded_sequences: List[
        List[int]
    ] = []

    for uid in users:
        encoded_sequences.append([
            item_to_embedding_id[iid]
            for iid in train_histories[uid]
        ])

    dataset = SASRecTrainDataset(
        encoded_sequences
    )

    if len(dataset) == 0:
        raise ValueError(
            "No sequence has at least two TRAIN interactions."
        )

    model_max_len = (
        args.history_size
        if args.history_size > 0
        else max(
            len(x)
            for x in encoded_sequences
        )
    )

    collator = SASRecCollator(
        max_len=model_max_len
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    model = SASRec(
        n_items=len(item_ids),
        max_len=model_max_len,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    n_train_pairs = sum(
        max(
            len(train_histories[u]) - 1,
            0,
        )
        for u in users
    )

    print("=" * 88)
    print("SASRec -- CoMemTree strict-fair baseline")
    print("=" * 88)
    print(f"Device                     : {device}")
    print(f"Selected users             : {len(users)}")
    print(f"Trainable users (len >= 2) : {len(dataset)}")
    print(f"Max train history/user     : {args.history_size}")
    print(
        f"Actual train length min/max: "
        f"{min(len(train_histories[u]) for u in users)}/"
        f"{max(len(train_histories[u]) for u in users)}"
    )
    print(f"Next-item training pairs   : {n_train_pairs}")
    print(f"Train-time item vocabulary : {len(item_ids)}")
    print(f"Model max sequence length  : {model_max_len}")
    print(f"Hidden size                : {args.hidden_size}")
    print(f"Layers / heads             : {args.n_layers} / {args.n_heads}")
    print(f"Batch size                 : {args.batch_size}")
    print(f"Epochs                     : {args.epochs}")
    print("Uses val/test in training  : NO")
    print("=" * 88)

    epoch_losses: List[float] = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        running_loss = 0.0
        n_batches = 0

        for batch in loader:
            input_ids = batch[
                "input_ids"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            hidden = model.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            valid = labels != -100

            if not valid.any():
                continue

            valid_hidden = hidden[
                valid
            ]

            target_classes = (
                labels[valid]
                - 1
            )

            logits = model.next_item_logits(
                valid_hidden
            )

            loss = F.cross_entropy(
                logits,
                target_classes,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            running_loss += float(
                loss.detach().cpu()
            )

            n_batches += 1

        if n_batches == 0:
            raise RuntimeError(
                "No valid SASRec training batches."
            )

        mean_loss = (
            running_loss
            / n_batches
        )

        epoch_losses.append(
            mean_loss
        )

        if (
            epoch == 1
            or epoch % args.log_every == 0
            or epoch == args.epochs
        ):
            print(
                f"epoch={epoch:04d} "
                f"loss={mean_loss:.6f}"
            )

    model.eval()

    # Evaluation inputs loaded only after fitting.
    fixed_candidates, _ = (
        load_evaluation_candidates(
            args,
            sequences,
            users,
        )
    )

    ranking_rows: List[
        Dict[str, Any]
    ] = []

    ranks: List[int] = []

    total_oov_candidates = 0
    total_oov_targets = 0

    with torch.no_grad():
        for uid in users:
            row = fixed_candidates[
                uid
            ]

            candidates = [
                str(x)
                for x in row["candidates"]
            ]

            targets = [
                str(x)
                for x in row["targets"]
            ]

            history_ids = (
                train_histories[uid][
                    -model_max_len:
                ]
            )

            encoded_history = [
                item_to_embedding_id[iid]
                for iid in history_ids
            ]

            L = len(
                encoded_history
            )

            input_ids = torch.zeros(
                (1, model_max_len),
                dtype=torch.long,
                device=device,
            )

            attention_mask = torch.zeros(
                (1, model_max_len),
                dtype=torch.bool,
                device=device,
            )

            input_ids[
                0,
                :L,
            ] = torch.tensor(
                encoded_history,
                dtype=torch.long,
                device=device,
            )

            attention_mask[
                0,
                :L,
            ] = True

            hidden = model.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            final_hidden = hidden[
                0,
                L - 1,
            ]

            score_values = [
                float("-inf")
                for _ in candidates
            ]

            known_positions: List[int] = []
            known_class_ids: List[int] = []
            oov_candidates: List[str] = []

            for j, iid in enumerate(
                candidates
            ):
                class_id = item_to_class.get(
                    iid
                )

                if class_id is None:
                    oov_candidates.append(
                        iid
                    )
                else:
                    known_positions.append(
                        j
                    )

                    known_class_ids.append(
                        class_id
                    )

            if known_class_ids:
                class_tensor = torch.tensor(
                    known_class_ids,
                    dtype=torch.long,
                    device=device,
                )

                known_scores = (
                    model.score_item_classes(
                        hidden=final_hidden,
                        class_ids=class_tensor,
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )

                for j, s in zip(
                    known_positions,
                    known_scores,
                ):
                    score_values[j] = float(s)

            total_oov_candidates += len(
                oov_candidates
            )

            target_is_oov = any(
                t not in item_to_class
                for t in targets
            )

            if target_is_oov:
                total_oov_targets += 1

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
                    f"user={uid}: target missing after ranking"
                )

            rank = min(
                target_ranks
            )

            ranks.append(rank)

            ranking_rows.append({
                "user_id": uid,
                "train_history": history_ids,
                "targets": targets,
                "candidates": candidates,
                "scores_in_candidate_order": score_values,
                "oov_candidates": oov_candidates,
                "n_oov_candidates": len(
                    oov_candidates
                ),
                "target_is_oov": target_is_oov,
                "ranked_candidates": ranked_candidates,
                "rank_position": rank,
            })

    metrics = compute_metrics(
        ranks
    )

    total_eval_candidates = sum(
        len(r["candidates"])
        for r in ranking_rows
    )

    diagnostics = {
        "n_users": len(users),
        "n_trainable_users": len(dataset),
        "n_next_item_training_pairs": n_train_pairs,
        "n_eval_candidates": total_eval_candidates,
        "n_oov_candidates": total_oov_candidates,
        "oov_candidate_rate": (
            total_oov_candidates
            / total_eval_candidates
            if total_eval_candidates
            else 0.0
        ),
        "n_oov_targets": total_oov_targets,
        "oov_target_rate": (
            total_oov_targets
            / len(users)
            if users
            else 0.0
        ),
        "oov_policy": (
            "-inf score; held-out IDs do not expand vocabulary before training"
        ),
    }

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_candidate_snapshot(
        out_dir,
        args,
        fixed_candidates,
    )

    write_jsonl(
        out_dir / "sasrec_rankings.jsonl",
        ranking_rows,
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "item_ids": item_ids,
            "item_to_class": item_to_class,
            "config": vars(args),
            "model_max_len": model_max_len,
        },
        out_dir / "sasrec_model.pt",
    )

    summary = {
        "method": "SASRec",
        "protocol": "CoMemTree_strict_fair",
        "inputs": {
            "items": args.items,
            "sequences": args.sequences,
            "negatives": args.negatives,
            "eval_behaviors": args.eval_behaviors,
            "user_ids_file": args.user_ids_file,
            "candidate_file": args.candidate_file,
        },
        "fairness": {
            **common_protocol_summary(
                args,
                users,
                train_histories,
            ),
            "train_item_vocabulary": (
                "items.json catalog + allowed TRAIN IDs only"
            ),
        },
        "hyperparameters": {
            "hidden_size": args.hidden_size,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
        "training": {
            "n_trainable_users": len(dataset),
            "n_next_item_training_pairs": n_train_pairs,
            "final_loss": (
                epoch_losses[-1]
                if epoch_losses
                else None
            ),
            "min_loss": (
                min(epoch_losses)
                if epoch_losses
                else None
            ),
        },
        "diagnostics": diagnostics,
        "metrics": metrics,
    }

    save_json(
        out_dir / "sasrec_metrics.json",
        summary,
    )

    print("\nEVALUATION DIAGNOSTICS")
    print(
        json.dumps(
            diagnostics,
            indent=2,
        )
    )

    print("\nRESULTS")
    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(f"\nSaved to: {out_dir}")


# =============================================================================
# CLI
# =============================================================================

def add_common_args(
    p: argparse.ArgumentParser,
) -> None:
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
            "Used only to recover the exact CoMemTree user IDs. "
            "Behavior text is NEVER used by the baseline model."
        ),
    )

    p.add_argument(
        "--user-ids-file",
        default=None,
        help=(
            "Optional explicit frozen user list. "
            "If supplied, it takes precedence."
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

    p.add_argument(
        "--device",
        default="auto",
    )


def add_transformer_args(
    p: argparse.ArgumentParser,
) -> None:
    p.add_argument(
        "--hidden-size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--n-layers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--n-heads",
        type=int,
        default=2,
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=10,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified LightGCN / BERT4Rec / SASRec baselines "
            "for the CoMemTree protocol."
        )
    )

    sub = parser.add_subparsers(
        dest="method",
        required=True,
    )

    # LightGCN
    p_lg = sub.add_parser(
        "lightgcn",
        help="Run LightGCN baseline.",
    )

    add_common_args(p_lg)

    p_lg.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
    )

    p_lg.add_argument(
        "--n-layers",
        type=int,
        default=3,
    )

    p_lg.add_argument(
        "--epochs",
        type=int,
        default=300,
    )

    p_lg.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    p_lg.add_argument(
        "--reg-weight",
        type=float,
        default=1e-4,
    )

    p_lg.add_argument(
        "--output-dir",
        default="results/lightgcn_cd_300x10",
    )

    # BERT4Rec
    p_bert = sub.add_parser(
        "bert4rec",
        help="Run BERT4Rec baseline.",
    )

    add_common_args(p_bert)
    add_transformer_args(p_bert)

    p_bert.add_argument(
        "--mask-prob",
        type=float,
        default=0.2,
    )

    p_bert.add_argument(
        "--output-dir",
        default="results/bert4rec_cd_300",
    )

    # SASRec
    p_sas = sub.add_parser(
        "sasrec",
        help="Run SASRec baseline.",
    )

    add_common_args(p_sas)
    add_transformer_args(p_sas)

    p_sas.add_argument(
        "--output-dir",
        default="results/sasrec_cd_300",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.method == "lightgcn":
        run_lightgcn(args)

    elif args.method == "bert4rec":
        run_bert4rec(args)

    elif args.method == "sasrec":
        run_sasrec(args)

    else:
        raise ValueError(
            f"Unsupported method: {args.method}"
        )


if __name__ == "__main__":
    main()
