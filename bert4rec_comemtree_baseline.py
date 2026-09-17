#!/usr/bin/env python3
"""
BERT4Rec baseline matched to the current CoMemTree evaluation protocol.

STRICT FAIRNESS / NO-TEST-LEAKAGE DEFAULTS
------------------------------------------
1) Use the SAME evaluation users as CoMemTree:
   - recover user IDs from --eval-behaviors only, or
   - preferably pass a frozen --user-ids-file.
   Behavior text is NEVER used by BERT4Rec.

2) Train ONLY from each selected user's:
       train[-history_size:]
   with history_size=10 by default.
   No validation/test interactions are used by the training objective.

3) Training objective:
   BERT4Rec-style Cloze / masked-item prediction on TRAIN sequences only.

4) Evaluation:
       input = train[-history_size:] + [MASK]
   Score exactly the SAME candidate set as CoMemTree:
       test_ids + test_neg
       random.Random(f"{seed}:{uid}").shuffle(candidates)

5) Train-time item vocabulary:
       static items.json catalog
       + item IDs appearing in the selected users' allowed TRAIN histories.
   Test/validation/candidate IDs never expand the vocabulary before fitting.
   Evaluation candidates outside this vocabulary are OOV and receive -inf score.
   OOV rates are reported explicitly.

6) No early stopping on test data. Hyperparameters are fixed by command line.

Outputs
-------
<output_dir>/
  bert4rec_metrics.json
  bert4rec_rankings.jsonl
  fixed_candidates_snapshot.json
  bert4rec_model.pt

Example
-------
CUDA_VISIBLE_DEVICES=1 python bert4rec_comemtree_baseline.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/user_negatives_10_5000.json \
  --eval-behaviors precomputed/CDs/test_user_behaviors_gemma.jsonl \
  --max-users 300 \
  --history-size 10 \
  --expected-candidates 20 \
  --seed 42 \
  --hidden-size 64 \
  --n-layers 2 \
  --n-heads 2 \
  --dropout 0.2 \
  --mask-prob 0.2 \
  --batch-size 64 \
  --epochs 200 \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --output-dir results/bert4rec_cd_300
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


# =============================================================================
# I/O / reproducibility
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

        # Preserve sequence-file order as in the current CoMemTree pipeline.
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
# Item vocabulary
# =============================================================================

def catalog_item_ids(items_obj: Any) -> List[str]:
    """
    Current CDs items.json is keyed by ASIN/item ID.
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

    raise ValueError(
        "Unsupported items.json structure."
    )


# =============================================================================
# Dataset: dynamic BERT4Rec masking
# =============================================================================

class TrainSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: List[List[int]],
    ) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> List[int]:
        return self.sequences[idx]


class Bert4RecCollator:
    """
    Dynamic Cloze masking on TRAIN sequences only.

    Special token IDs:
      PAD  = 0
      MASK = 1
      real items start from 2

    Returned labels use:
      -100 for unmasked positions
      real embedding IDs (>=2) for masked positions
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

    def _random_real_item_embedding_id(self) -> int:
        # Real item embedding IDs are [2, n_real_items + 1].
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
            # Training sequence length is <= history_size, which is <= max_len.
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
                if self.rng.random() < self.mask_prob:
                    selected.append(pos)

            # At least one masked training target per user sequence.
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

                # Standard BERT replacement:
                # 80% [MASK], 10% random item, 10% unchanged.
                if r < 0.8:
                    input_ids[b, pos] = 1
                elif r < 0.9:
                    input_ids[b, pos] = (
                        self._random_real_item_embedding_id()
                    )
                else:
                    pass

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# =============================================================================
# BERT4Rec
# =============================================================================

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

        self.n_real_items = int(n_real_items)
        self.max_len = int(max_len)
        self.hidden_size = int(hidden_size)

        # Embedding IDs:
        #   0 PAD
        #   1 MASK
        #   2..n_real_items+1 real items
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

        # One bias per real item. Output weights are tied to real item embeddings.
        self.output_bias = nn.Parameter(
            torch.zeros(n_real_items)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.item_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

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
        ).unsqueeze(0).expand(B, L)

        x = (
            self.item_embedding(input_ids)
            + self.position_embedding(pos)
        )

        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        # Transformer uses True for padding positions.
        key_padding_mask = ~attention_mask.bool()

        x = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask,
        )

        return self.output_norm(x)

    def masked_logits(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = labels != -100

        masked_hidden = hidden[mask]
        masked_labels_embedding_ids = labels[mask]

        if masked_hidden.numel() == 0:
            raise RuntimeError(
                "No masked positions in batch."
            )

        # Real-item output weights only: embedding rows 2:.
        real_item_weights = (
            self.item_embedding.weight[2:]
        )

        logits = (
            masked_hidden
            @ real_item_weights.t()
            + self.output_bias
        )

        # Convert embedding IDs 2..V+1 to class IDs 0..V-1.
        target_classes = (
            masked_labels_embedding_ids
            - 2
        )

        return logits, target_classes

    @torch.no_grad()
    def score_real_item_classes(
        self,
        mask_hidden: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        mask_hidden: [H]
        class_ids:   [C], values 0..n_real_items-1
        """
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

    # -----------------------------------------------------------------
    # Train-time/static inputs only.
    # -----------------------------------------------------------------
    items_raw = load_json(
        args.items
    )

    sequences_raw = load_json(
        args.sequences
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

    # -----------------------------------------------------------------
    # Strict train histories: last <= history_size TRAIN only.
    # -----------------------------------------------------------------
    train_histories: Dict[str, List[str]] = {}

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

        train_histories[uid] = train_ids

    # -----------------------------------------------------------------
    # Train-time item vocabulary:
    # items.json static catalog + allowed TRAIN IDs only.
    # No val/test/candidate IDs are added here.
    # -----------------------------------------------------------------
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
        raise ValueError(
            "Empty train-time item vocabulary."
        )

    # class id: 0..n_items-1
    item_to_class = {
        iid: idx
        for idx, iid in enumerate(item_ids)
    }

    # embedding id: class + 2
    item_to_embedding_id = {
        iid: idx + 2
        for iid, idx in item_to_class.items()
    }

    # -----------------------------------------------------------------
    # Encode train sequences.
    # -----------------------------------------------------------------
    encoded_train_sequences: List[List[int]] = []

    for uid in users:
        seq = [
            item_to_embedding_id[iid]
            for iid in train_histories[uid]
        ]

        encoded_train_sequences.append(
            seq
        )

    dataset = TrainSequenceDataset(
        encoded_train_sequences
    )

    # +1 allows all history items plus final [MASK] at inference.
    model_max_len = (
        args.history_size + 1
        if args.history_size > 0
        else max(
            len(x)
            for x in encoded_train_sequences
        ) + 1
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

    print("=" * 90)
    print("BERT4Rec -- CoMemTree strict-fair baseline")
    print("=" * 90)
    print(f"Device                       : {device}")
    print(f"Selected users               : {len(users)}")
    print(f"Max train history/user       : {args.history_size}")
    print(
        f"Actual train length min/max  : "
        f"{min(len(train_histories[u]) for u in users)}/"
        f"{max(len(train_histories[u]) for u in users)}"
    )
    print(f"Train-time item vocabulary   : {len(item_ids)}")
    print(f"Model max sequence length    : {model_max_len}")
    print(f"Hidden size                  : {args.hidden_size}")
    print(f"Layers / heads               : {args.n_layers} / {args.n_heads}")
    print(f"Mask probability             : {args.mask_prob}")
    print(f"Batch size                   : {args.batch_size}")
    print(f"Epochs                       : {args.epochs}")
    print("Uses old train outside cutoff: NO")
    print("Uses validation in training  : NO")
    print("Uses test/GT in training     : NO")
    print("Uses candidates in training  : NO")
    print("=" * 90)

    # -----------------------------------------------------------------
    # Training.
    # -----------------------------------------------------------------
    epoch_losses: List[float] = []

    for epoch in range(1, args.epochs + 1):
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

            logits, targets = model.masked_logits(
                hidden=hidden,
                labels=labels,
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
            / max(n_batches, 1)
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

    # MODEL IS NOW FROZEN.
    model.eval()

    # -----------------------------------------------------------------
    # Evaluation data are loaded only after fitting.
    # -----------------------------------------------------------------
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

    total_oov_candidates = 0
    total_oov_targets = 0

    with torch.no_grad():
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
                        f"user={uid}: target {t} missing from candidates"
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

            # ---------------------------------------------------------
            # BERT4Rec next-item inference:
            # history + [MASK]
            # ---------------------------------------------------------
            history_ids = train_histories[
                uid
            ][
                -args.history_size:
            ]

            encoded_history = [
                item_to_embedding_id[iid]
                for iid in history_ids
            ]

            eval_seq = (
                encoded_history
                + [1]  # [MASK]
            )

            if len(eval_seq) > model_max_len:
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

            # Final non-padding token is [MASK].
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

            for j, iid in enumerate(candidates):
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

            # Stable tie-break by original candidate order.
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
                "train_history": history_ids,
                "targets": targets,
                "candidates": candidates,
                "scores_in_candidate_order": score_values,
                "oov_candidates": oov_candidates,
                "n_oov_candidates": len(
                    oov_candidates
                ),
                "target_is_oov": (
                    target_is_oov
                ),
                "ranked_candidates": (
                    ranked_candidates
                ),
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
        "n_eval_candidates": (
            total_eval_candidates
        ),
        "n_oov_candidates": (
            total_oov_candidates
        ),
        "oov_candidate_rate": (
            total_oov_candidates
            / total_eval_candidates
            if total_eval_candidates
            else 0.0
        ),
        "n_oov_targets": (
            total_oov_targets
        ),
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
        / "bert4rec_rankings.jsonl",
        ranking_rows,
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "item_ids": item_ids,
            "item_to_class": (
                item_to_class
            ),
            "config": vars(args),
            "model_max_len": (
                model_max_len
            ),
        },
        out_dir
        / "bert4rec_model.pt",
    )

    summary = {
        "method": "BERT4Rec",
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
            "train_source": (
                "last <= history_size TRAIN interactions only"
            ),
            "uses_old_train_outside_cutoff": False,
            "uses_validation_in_training": False,
            "uses_test_in_training": False,
            "uses_candidates_in_training": False,
            "candidate_seed": (
                args.seed
            ),
            "candidate_order": (
                "same as CoMemTree"
            ),
            "train_item_vocabulary": (
                "items.json catalog + allowed TRAIN IDs only"
            ),
        },
        "hyperparameters": {
            "hidden_size": (
                args.hidden_size
            ),
            "n_layers": (
                args.n_layers
            ),
            "n_heads": (
                args.n_heads
            ),
            "dropout": (
                args.dropout
            ),
            "mask_prob": (
                args.mask_prob
            ),
            "batch_size": (
                args.batch_size
            ),
            "epochs": (
                args.epochs
            ),
            "lr": args.lr,
            "weight_decay": (
                args.weight_decay
            ),
            "grad_clip": (
                args.grad_clip
            ),
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
        "diagnostics": (
            diagnostics
        ),
        "metrics": metrics,
    }

    save_json(
        out_dir
        / "bert4rec_metrics.json",
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
            "Used only to recover the exact CoMemTree user IDs. "
            "Behavior text is NEVER used by BERT4Rec."
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

    # BERT4Rec.
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
        "--mask-prob",
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

    p.add_argument(
        "--device",
        default="auto",
    )

    p.add_argument(
        "--output-dir",
        default="results/bert4rec_cd_300",
    )

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
