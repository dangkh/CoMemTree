#!/usr/bin/env python3
"""
Build the SAME collaborative variable-order reverse suffix tree, but use
KMeans clustering as the STATE-INDUCTION ABLATION.

Purpose
-------
This script is designed to test whether the semantic/structured state induction
is better than a generic geometric grouping of the SAME upstream Gemma behavior
memories.

Main controlled change:
    Gemma local behavior memory
        -> Qwen embedding of behavior text
        -> KMeans(K=50)
        -> cluster ID is the discrete behavior state
        -> SAME reverse suffix tree

No mechanism/scope taxonomy is used for clustering.
No Gemini/Gemma verifier is used for clustering.
No LLM call is made by this script.

Default input
-------------
    precomputed/CDs/local_memories_gemma.jsonl

Default clustering text
-----------------------
    pattern_description

This is intentionally the raw Gemma behavior representation BEFORE the
hierarchical/canonical state-induction stage.

Outputs
-------
    cluster_behavior_inputs.jsonl
    behavior_states.json
    behavior_state_embeddings.npy
    memory_state_assignments.jsonl
    user_behavior_sequences.json
    context_observations.jsonl
    behavior_tree.json
    tree_stats.json

The filenames match the main tree builder where possible so downstream cluster
inference can reuse the same general interface.

Example
-------
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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from tqdm.auto import tqdm

# =============================================================================
# Basic I/O / math
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def jsonl_dump(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        return arr / max(n, eps)
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(n, eps)


def entropy_bits(dist: Dict[str, float], eps: float = 1e-12) -> float:
    vals = np.asarray([v for v in dist.values() if v > eps], dtype=np.float64)
    if len(vals) == 0:
        return 0.0
    vals = vals / vals.sum()
    return float(-(vals * np.log2(vals)).sum())


def js_divergence_bits(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence using log2. Range [0,1]."""
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    pv = np.asarray([p.get(k, 0.0) for k in keys], dtype=np.float64)
    qv = np.asarray([q.get(k, 0.0) for k in keys], dtype=np.float64)
    pv = pv / max(float(pv.sum()), eps)
    qv = qv / max(float(qv.sum()), eps)
    m = 0.5 * (pv + qv)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > eps
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], eps))))

    return 0.5 * kl(pv, m) + 0.5 * kl(qv, m)


def normalize_dist(weights: Dict[str, float]) -> Dict[str, float]:
    z = float(sum(weights.values()))
    if z <= 0:
        return {}
    return {str(k): float(v / z) for k, v in weights.items() if v > 0}


def interpolate_dist(
    local: Dict[str, float],
    parent: Dict[str, float],
    lam: float,
) -> Dict[str, float]:
    keys = set(local) | set(parent)
    out = {
        k: lam * local.get(k, 0.0) + (1.0 - lam) * parent.get(k, 0.0)
        for k in keys
    }
    return normalize_dist(out)


def stable_context_id(context: Tuple[str, ...]) -> str:
    if not context:
        return "ROOT"
    h = hashlib.sha1("|".join(context).encode("utf-8")).hexdigest()[:12]
    return f"CTX_{len(context)}_{h}"


def read_user_ids(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return {str(x) for x in obj}
        if isinstance(obj, dict):
            if "users" in obj and isinstance(obj["users"], list):
                return {str(x) for x in obj["users"]}
            return {str(x) for x in obj.keys()}
    except json.JSONDecodeError:
        pass
    return {line.strip() for line in text.splitlines() if line.strip()}




# =============================================================================
# Gemma raw-memory input
# =============================================================================

REQUIRED_FIELDS = {"user_id", "window_index"}


def make_memory_id(row: Dict[str, Any], row_idx: int) -> str:
    if row.get("memory_id") is not None:
        return str(row["memory_id"])
    if row.get("precompute_id") is not None:
        return f"M{row['precompute_id']}"
    if row.get("thought_id") is not None:
        return f"T{row['thought_id']}"
    return f"MROW{row_idx}"


def clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def choose_cluster_text(row: Dict[str, Any], field: str) -> str:
    pattern = clean_text(row.get("pattern_description"))
    explanation = clean_text(row.get("behavior_explanation"))
    signature = clean_text(row.get("behavior_signature"))

    if field == "pattern_description":
        text = pattern
    elif field == "behavior_signature":
        text = signature
    elif field == "combined":
        text = (
            f"Behavior pattern: {pattern}. "
            f"Evidence summary: {explanation}"
        )
    else:
        raise ValueError(field)

    return clean_text(text)


def load_memories(
    path: str,
    cluster_text_field: str,
    include_users: Optional[Set[str]],
    exclude_users: Optional[Set[str]],
    skip_invalid: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if not line.strip():
                continue

            try:
                x = json.loads(line)
            except Exception as e:
                if skip_invalid:
                    print(
                        f"[WARN] invalid JSON line {row_idx + 1}: {e}",
                        file=sys.stderr,
                    )
                    continue
                raise

            missing = REQUIRED_FIELDS - set(x)
            if missing:
                if skip_invalid:
                    print(
                        f"[WARN] line {row_idx + 1} missing {sorted(missing)}; skipped",
                        file=sys.stderr,
                    )
                    continue
                raise ValueError(
                    f"line {row_idx + 1} missing {sorted(missing)}"
                )

            # Respect upstream parse failures if the field exists.
            if x.get("parse_ok") is False:
                if skip_invalid:
                    continue
                raise ValueError(
                    f"line {row_idx + 1}: parse_ok=False"
                )

            uid = str(x["user_id"])
            if include_users is not None and uid not in include_users:
                continue
            if exclude_users is not None and uid in exclude_users:
                continue

            cluster_text = choose_cluster_text(
                x,
                cluster_text_field,
            )
            if not cluster_text:
                if skip_invalid:
                    print(
                        f"[WARN] line {row_idx + 1}: empty "
                        f"{cluster_text_field}; skipped",
                        file=sys.stderr,
                    )
                    continue
                raise ValueError(
                    f"line {row_idx + 1}: empty {cluster_text_field}"
                )

            y = dict(x)
            y["memory_id"] = make_memory_id(x, row_idx)
            y["user_id"] = uid
            y["window_index"] = int(x["window_index"])
            y["cluster_text"] = cluster_text
            y["_row_idx"] = row_idx
            rows.append(y)

    if not rows:
        raise ValueError("No valid memories found after filtering")

    return rows


# =============================================================================
# Embeddings
# =============================================================================

def encode_qwen(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "Need sentence-transformers for --encoder qwen. "
            "Install: pip install -U sentence-transformers transformers accelerate"
        ) from e

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"[INFO] embedding model={model_name} device={device}"
    )

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return np.asarray(emb, dtype=np.float32)


def encode_tfidf(
    texts: List[str],
    max_features: int,
) -> np.ndarray:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:
        raise RuntimeError(
            "Need scikit-learn for --encoder tfidf"
        ) from e

    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=max_features,
    )

    x = vec.fit_transform(texts).toarray().astype(np.float32)
    return l2_normalize(x)


def create_embeddings(
    args: argparse.Namespace,
    memories: List[Dict[str, Any]],
) -> np.ndarray:
    if args.embeddings_npy:
        x = np.load(args.embeddings_npy)
        if len(x) != len(memories):
            raise ValueError(
                f"--embeddings-npy has {len(x)} rows but "
                f"filtered input has {len(memories)} memories"
            )
        return l2_normalize(
            np.asarray(x, dtype=np.float32)
        )

    texts = [m["cluster_text"] for m in memories]

    if args.encoder == "qwen":
        return encode_qwen(
            texts,
            args.embedding_model,
            args.batch_size,
            args.device,
        )

    if args.encoder == "tfidf":
        return encode_tfidf(
            texts,
            args.tfidf_max_features,
        )

    raise ValueError(args.encoder)


# =============================================================================
# KMeans state induction
# =============================================================================

def run_kmeans(
    embeddings: np.ndarray,
    num_clusters: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise RuntimeError(
            "Need scikit-learn for KMeans clustering"
        ) from e

    if num_clusters <= 1:
        raise ValueError("--num-clusters must be > 1")
    if num_clusters > len(embeddings):
        raise ValueError(
            f"--num-clusters={num_clusters} > "
            f"num_memories={len(embeddings)}"
        )

    km = KMeans(
        n_clusters=num_clusters,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        verbose=0,
    )

    labels = km.fit_predict(embeddings)
    centers = l2_normalize(
        np.asarray(km.cluster_centers_, dtype=np.float32)
    )

    return (
        np.asarray(labels, dtype=np.int64),
        centers,
        float(km.inertia_),
    )


def build_cluster_states(
    memories: List[Dict[str, Any]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[int, str],
    List[np.ndarray],
]:
    state_rows: List[Dict[str, Any]] = []
    idx_to_state: Dict[int, str] = {}
    centroid_rows: List[np.ndarray] = []

    for cluster_idx in range(len(centers)):
        member_indices = np.where(
            labels == cluster_idx
        )[0].tolist()

        if not member_indices:
            # Standard KMeans should not leave empty clusters, but be safe.
            continue

        sid = f"C{cluster_idx:03d}"
        centroid = centers[cluster_idx]
        mat = embeddings[member_indices]
        sims = mat @ centroid
        medoid_local = int(np.argmax(sims))
        medoid_idx = member_indices[medoid_local]
        medoid = memories[medoid_idx]

        support_users = sorted({
            str(memories[i]["user_id"])
            for i in member_indices
        })

        examples = list(dict.fromkeys(
            memories[i]["cluster_text"]
            for i in member_indices
        ))[:12]

        keyword_counts: Counter = Counter()
        for i in member_indices:
            kws = memories[i].get("keywords") or []
            if not isinstance(kws, list):
                kws = [str(kws)]
            for kw in kws:
                s = clean_text(kw).lower()
                if s:
                    keyword_counts[s] += 1

        state_rows.append({
            "state_id": sid,
            "cluster_id": int(cluster_idx),
            "canonical_text": medoid["cluster_text"],
            "canonical_behavior_signature": medoid["cluster_text"],
            "representative_pattern_description": clean_text(
                medoid.get("pattern_description")
            ),
            "representative_behavior_explanation": clean_text(
                medoid.get("behavior_explanation")
            ),
            "medoid_memory_id": medoid["memory_id"],
            "cluster_text_examples": examples,
            "member_count": len(member_indices),
            "support_user_count": len(support_users),
            "support_users": support_users,
            "memory_ids": [
                memories[i]["memory_id"]
                for i in member_indices
            ],
            "top_keywords": [
                k
                for k, _ in keyword_counts.most_common(12)
            ],
            "mean_similarity_to_centroid": float(
                np.mean(sims)
            ),
            "min_similarity_to_centroid": float(
                np.min(sims)
            ),
            "max_similarity_to_centroid": float(
                np.max(sims)
            ),
        })

        centroid_rows.append(
            centroid.astype(np.float32)
        )

        for i in member_indices:
            idx_to_state[i] = sid

    # Preserve cluster-index/state-index alignment.
    state_rows.sort(
        key=lambda x: int(x["cluster_id"])
    )
    centroid_rows = [
        centers[int(row["cluster_id"])].astype(np.float32)
        for row in state_rows
    ]

    return state_rows, idx_to_state, centroid_rows


def cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    try:
        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score,
        )

        if len(set(labels.tolist())) > 1:
            out["silhouette_cosine"] = float(
                silhouette_score(
                    embeddings,
                    labels,
                    metric="cosine",
                )
            )
            out["calinski_harabasz"] = float(
                calinski_harabasz_score(
                    embeddings,
                    labels,
                )
            )
            out["davies_bouldin"] = float(
                davies_bouldin_score(
                    embeddings,
                    labels,
                )
            )
    except Exception as e:
        out["metric_error"] = (
            f"{type(e).__name__}: {e}"
        )

    return out


# =============================================================================
# Ordered user sequences
# =============================================================================


def build_user_sequences(
    memories: List[Dict[str, Any]],
    idx_to_state: Dict[int, str],
    collapse_consecutive: bool,
) -> Dict[str, Dict[str, Any]]:
    by_user: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, m in enumerate(memories):
        by_user[str(m["user_id"])].append((idx, m))

    out: Dict[str, Dict[str, Any]] = {}
    for uid, items in by_user.items():
        items.sort(key=lambda x: (int(x[1]["window_index"]), int(x[1]["_row_idx"])))
        entries: List[Dict[str, Any]] = []
        last_sid: Optional[str] = None

        for idx, m in items:
            sid = idx_to_state[idx]
            if collapse_consecutive and sid == last_sid:
                entries[-1]["memory_ids"].append(m["memory_id"])
                entries[-1]["window_indices"].append(int(m["window_index"]))
                continue
            entries.append({
                "state_id": sid,
                "memory_ids": [m["memory_id"]],
                "window_indices": [int(m["window_index"])],
            })
            last_sid = sid

        out[uid] = {
            "user_id": uid,
            "length": len(entries),
            "states": [e["state_id"] for e in entries],
            "entries": entries,
        }
    return out


# =============================================================================
# Context -> next observations
# =============================================================================


@dataclass
class ContextStats:
    occurrence_counts: Counter = field(default_factory=Counter)
    # user_target_counts[user][target] = number occurrences for this user/context/target
    user_target_counts: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    support_users: Set[str] = field(default_factory=set)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    seen: int = 0

    def add(
        self,
        uid: str,
        target: str,
        evidence: Dict[str, Any],
        max_evidence: int,
        rng: random.Random,
    ) -> None:
        self.occurrence_counts[target] += 1
        self.user_target_counts[uid][target] += 1
        self.support_users.add(uid)
        self.seen += 1

        # bounded unbiased evidence sample
        if max_evidence <= 0:
            return
        if len(self.evidence) < max_evidence:
            self.evidence.append(evidence)
        else:
            j = rng.randint(0, self.seen - 1)
            if j < max_evidence:
                self.evidence[j] = evidence


def suffix_parent(context: Tuple[str, ...]) -> Tuple[str, ...]:
    """Chronological [A,B,C] -> [B,C] -> [C] -> ROOT."""
    if len(context) <= 1:
        return ()
    return context[1:]


def accumulate_contexts(
    user_sequences: Dict[str, Dict[str, Any]],
    max_order: int,
    max_evidence: int,
    seed: int,
) -> Dict[Tuple[str, ...], ContextStats]:
    rng = random.Random(seed)
    stats: Dict[Tuple[str, ...], ContextStats] = defaultdict(ContextStats)

    for uid, rec in user_sequences.items():
        seq = rec["states"]
        entries = rec["entries"]
        if len(seq) < 2:
            continue

        for t in range(1, len(seq)):
            target = seq[t]

            root_ev = {
                "user_id": uid,
                "context_states": [],
                "reverse_tree_path": [],
                "target_state": target,
                "target_memory_ids": entries[t]["memory_ids"],
                "target_window_indices": entries[t]["window_indices"],
            }
            stats[()].add(uid, target, root_ev, max_evidence, rng)

            for L in range(1, min(max_order, t) + 1):
                context = tuple(seq[t-L:t])  # chronological
                ev = {
                    "user_id": uid,
                    "context_states": list(context),
                    "reverse_tree_path": list(reversed(context)),
                    "context_memory_ids": [entries[j]["memory_ids"] for j in range(t-L, t)],
                    "context_window_indices": [entries[j]["window_indices"] for j in range(t-L, t)],
                    "target_state": target,
                    "target_memory_ids": entries[t]["memory_ids"],
                    "target_window_indices": entries[t]["window_indices"],
                }
                stats[context].add(uid, target, ev, max_evidence, rng)

    return stats


def collaborative_weights(acc: ContextStats, count_mode: str) -> Dict[str, float]:
    if count_mode == "occurrence":
        return {str(k): float(v) for k, v in acc.occurrence_counts.items()}

    if count_mode == "distinct_user":
        # Each user can vote at most once per target. A user may vote for multiple targets.
        target_users: Dict[str, Set[str]] = defaultdict(set)
        for uid, counter in acc.user_target_counts.items():
            for target in counter:
                target_users[target].add(uid)
        return {target: float(len(users)) for target, users in target_users.items()}

    if count_mode == "user_normalized":
        # Each user contributes total mass exactly 1 for a given context.
        weights: Dict[str, float] = defaultdict(float)
        for uid, counter in acc.user_target_counts.items():
            total = float(sum(counter.values()))
            if total <= 0:
                continue
            for target, cnt in counter.items():
                weights[target] += float(cnt) / total
        return dict(weights)

    raise ValueError(f"Unknown count_mode={count_mode}")


# =============================================================================
# Context records, smoothing, active predictive contexts
# =============================================================================


def build_context_records(
    stats: Dict[Tuple[str, ...], ContextStats],
    state_rows: List[Dict[str, Any]],
    count_mode: str,
    smoothing_kappa: float,
    min_support_users: int,
    min_support_occurrences: int,
    min_jsd: float,
    promote_order1: bool,
    top_next: int,
) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    if () not in stats:
        raise ValueError("No transitions found; ROOT distribution unavailable")

    state_text = {x["state_id"]: x["canonical_text"] for x in state_rows}
    records: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    for ctx in sorted(stats.keys(), key=lambda c: (len(c), c)):
        acc = stats[ctx]
        weights = collaborative_weights(acc, count_mode)
        raw_dist = normalize_dist(weights)
        support_users = len(acc.support_users)
        occurrences = int(sum(acc.occurrence_counts.values()))

        if ctx == ():
            parent_ctx = None
            lam = 1.0
            smooth_dist = raw_dist
            parent_raw = {}
            jsd_raw = 0.0
            eligible = True
            active = True
            reason = "root"
        else:
            parent_ctx = suffix_parent(ctx)
            parent = records[parent_ctx]
            parent_raw = parent["raw_distribution_dict"]
            parent_smooth = parent["smoothed_distribution_dict"]
            lam = float(support_users / (support_users + smoothing_kappa)) if smoothing_kappa > 0 else 1.0
            smooth_dist = interpolate_dist(raw_dist, parent_smooth, lam)
            jsd_raw = js_divergence_bits(raw_dist, parent_raw)
            eligible = (
                support_users >= min_support_users
                and occurrences >= min_support_occurrences
            )
            if not eligible:
                active = False
                reason = "insufficient_support"
            elif len(ctx) == 1 and promote_order1:
                active = True
                reason = "order1_base_context"
            else:
                active = bool(jsd_raw >= min_jsd)
                reason = "predictive_gain" if active else "low_predictive_gain"

        def dist_rows(dist: Dict[str, float]) -> List[Dict[str, Any]]:
            items = sorted(dist.items(), key=lambda x: (-x[1], x[0]))
            if top_next > 0:
                items = items[:top_next]
            return [
                {
                    "state_id": sid,
                    "probability": float(p),
                    "state_text": state_text.get(sid, ""),
                    "occurrence_count": int(acc.occurrence_counts.get(sid, 0)),
                    "support_user_count": int(sum(1 for u in acc.user_target_counts if sid in acc.user_target_counts[u])),
                }
                for sid, p in items
            ]

        records[ctx] = {
            "node_id": stable_context_id(ctx),
            "context_states": list(ctx),                    # chronological
            "reverse_tree_path": list(reversed(ctx)),       # actual traversal path
            "context_order": len(ctx),
            "suffix_parent_context": list(parent_ctx) if parent_ctx is not None else None,
            "suffix_parent_id": stable_context_id(parent_ctx) if parent_ctx is not None else None,
            "support_user_count": support_users,
            "support_occurrence_count": occurrences,
            "occurrence_next_counts": {str(k): int(v) for k, v in acc.occurrence_counts.items()},
            "collaborative_next_weights": {str(k): float(v) for k, v in weights.items()},
            "raw_distribution_dict": raw_dist,
            "smoothed_distribution_dict": smooth_dist,
            "raw_next_distribution": dist_rows(raw_dist),
            "smoothed_next_distribution": dist_rows(smooth_dist),
            "raw_entropy_bits": entropy_bits(raw_dist),
            "smoothed_entropy_bits": entropy_bits(smooth_dist),
            "smoothing_lambda": lam,
            "jsd_raw_to_suffix_parent_bits": jsd_raw,
            "eligible_support": bool(eligible),
            "active_predictive": bool(active),
            "activation_reason": reason,
            "evidence": acc.evidence,
        }

    # nearest active suffix for each node
    for ctx, rec in records.items():
        if ctx == ():
            rec["active_backoff_context"] = None
            rec["active_backoff_id"] = None
            continue
        p = suffix_parent(ctx)
        while p != () and not records[p]["active_predictive"]:
            p = suffix_parent(p)
        # ROOT is always active
        rec["active_backoff_context"] = list(p)
        rec["active_backoff_id"] = stable_context_id(p)

    return records


# =============================================================================
# Physical reverse tree serialization
# =============================================================================


def build_reverse_tree(records: Dict[Tuple[str, ...], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build ONE physical reverse tree.

    For chronological context [A,B,C], reverse_tree_path = [C,B,A].
    Therefore the node is reached as ROOT -> C -> B -> A.

    IMPORTANT: structural nodes are retained even if active_predictive=False.
    This ensures a deeper path is traversable and inference can track the deepest
    active node encountered along that path.
    """
    nodes: Dict[str, Dict[str, Any]] = {}

    for ctx in sorted(records.keys(), key=lambda c: (len(c), c)):
        rec = dict(records[ctx])
        rec["children"] = {}  # token -> child node id
        # Internal dict copies not needed in public JSON twice.
        rec.pop("raw_distribution_dict", None)
        rec.pop("smoothed_distribution_dict", None)
        nodes[rec["node_id"]] = rec

    # Every context [A,B,C] has a structural reverse parent [B,C].
    # In reverse traversal, [B,C] path is C->B, and child token A reaches ABC.
    for ctx in sorted(records.keys(), key=lambda c: (len(c), c)):
        if ctx == ():
            continue
        node_id = stable_context_id(ctx)
        if len(ctx) == 1:
            parent_ctx = ()
            edge_token = ctx[-1]       # ROOT -> C
        else:
            parent_ctx = suffix_parent(ctx)
            edge_token = ctx[0]        # BC -> ABC adds A at the reverse-tree end
        parent_id = stable_context_id(parent_ctx)
        nodes[parent_id]["children"][edge_token] = node_id
        nodes[node_id]["structural_parent_id"] = parent_id
        nodes[node_id]["incoming_reverse_token"] = edge_token

    nodes["ROOT"]["structural_parent_id"] = None
    nodes["ROOT"]["incoming_reverse_token"] = None

    return {
        "root_id": "ROOT",
        "nodes": nodes,
        "num_structural_nodes": len(nodes),
        "num_active_predictive_nodes": sum(1 for n in nodes.values() if n["active_predictive"]),
    }


# =============================================================================
# Optional self-check: ensure reverse traversal semantics are correct
# =============================================================================


def validate_tree_structure(tree: Dict[str, Any], records: Dict[Tuple[str, ...], Dict[str, Any]]) -> None:
    nodes = tree["nodes"]
    for ctx in records:
        if ctx == ():
            continue
        cur = "ROOT"
        for token in reversed(ctx):
            children = nodes[cur]["children"]
            if token not in children:
                raise AssertionError(
                    f"Broken reverse path for chronological context {ctx}; "
                    f"at node {cur}, missing token {token}"
                )
            cur = children[token]
        expected = stable_context_id(ctx)
        if cur != expected:
            raise AssertionError(f"Reverse traversal reached {cur}, expected {expected} for {ctx}")


# =============================================================================
# CLI
# =============================================================================




# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "KMeans-state ablation: raw Gemma behavior memory -> "
            "Qwen -> KMeans(K) -> same reverse behavior tree"
        )
    )

    p.add_argument(
        "--input",
        default="precomputed/CDs/local_memories_gemma.jsonl",
        help="Raw Gemma behavior-memory JSONL before canonical state induction.",
    )
    p.add_argument(
        "--output-dir",
        default="behavior_tree_out_cluster_k50",
    )

    p.add_argument("--include-users", default=None)
    p.add_argument("--exclude-users", default=None)
    p.add_argument("--strict-input", action="store_true")

    p.add_argument(
        "--cluster-text-field",
        choices=[
            "pattern_description",
            "behavior_signature",
            "combined",
        ],
        default="pattern_description",
        help=(
            "Text embedded for KMeans. For the controlled Gemma ablation, "
            "pattern_description is the default."
        ),
    )

    p.add_argument(
        "--encoder",
        choices=["qwen", "tfidf"],
        default="qwen",
    )
    p.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    p.add_argument("--embeddings-npy", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--tfidf-max-features",
        type=int,
        default=8000,
    )

    p.add_argument(
        "--num-clusters",
        type=int,
        default=50,
    )
    p.add_argument(
        "--kmeans-n-init",
        type=int,
        default=20,
    )
    p.add_argument(
        "--kmeans-max-iter",
        type=int,
        default=300,
    )

    # SAME tree hyperparameters as the semantic-state builder.
    p.add_argument(
        "--collapse-consecutive-states",
        action="store_true",
    )
    p.add_argument("--max-order", type=int, default=5)
    p.add_argument(
        "--count-mode",
        choices=[
            "user_normalized",
            "distinct_user",
            "occurrence",
        ],
        default="user_normalized",
    )
    p.add_argument(
        "--smoothing-kappa",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--min-support-users",
        type=int,
        default=3,
    )
    p.add_argument(
        "--min-support-occurrences",
        type=int,
        default=3,
    )
    p.add_argument(
        "--min-jsd",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--no-promote-order1",
        action="store_true",
    )
    p.add_argument(
        "--max-evidence",
        type=int,
        default=8,
    )
    p.add_argument(
        "--top-next",
        type=int,
        default=20,
    )
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    include_users = read_user_ids(
        args.include_users
    )
    exclude_users = read_user_ids(
        args.exclude_users
    )

    print("[1/8] Load raw Gemma behavior memories")
    memories = load_memories(
        path=args.input,
        cluster_text_field=args.cluster_text_field,
        include_users=include_users,
        exclude_users=exclude_users,
        skip_invalid=not args.strict_input,
    )

    num_users = len({
        m["user_id"] for m in memories
    })

    print(
        f"      memories={len(memories)} "
        f"users={num_users}"
    )
    print(
        f"      cluster_text_field="
        f"{args.cluster_text_field}"
    )

    print("[2/8] Save cluster inputs")
    jsonl_dump(
        [
            {
                "memory_id": m["memory_id"],
                "user_id": m["user_id"],
                "window_index": m["window_index"],
                "cluster_text": m["cluster_text"],
                "pattern_description": m.get(
                    "pattern_description"
                ),
                "behavior_explanation": m.get(
                    "behavior_explanation"
                ),
                "behavior_signature": m.get(
                    "behavior_signature"
                ),
                "keywords": m.get("keywords"),
            }
            for m in memories
        ],
        out_dir / "cluster_behavior_inputs.jsonl",
    )

    print("[3/8] Embed Gemma behavior text")
    embeddings = create_embeddings(
        args,
        memories,
    )
    embeddings = l2_normalize(embeddings)
    print(
        f"      embedding_shape={embeddings.shape}"
    )

    print(
        f"[4/8] KMeans state induction "
        f"(K={args.num_clusters})"
    )
    labels, centers, inertia = run_kmeans(
        embeddings=embeddings,
        num_clusters=args.num_clusters,
        seed=args.seed,
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
    )

    state_rows, idx_to_state, centroid_rows = (
        build_cluster_states(
            memories=memories,
            embeddings=embeddings,
            labels=labels,
            centers=centers,
        )
    )

    state_centroids = np.stack(
        centroid_rows,
        axis=0,
    ).astype(np.float32)

    quality = cluster_quality(
        embeddings,
        labels,
    )
    quality["kmeans_inertia"] = inertia

    print(
        f"      states={len(state_rows)} "
        f"inertia={inertia:.4f}"
    )
    if "silhouette_cosine" in quality:
        print(
            f"      silhouette_cosine="
            f"{quality['silhouette_cosine']:.4f}"
        )

    state_payload = {
        "metadata": {
            "input": os.path.abspath(
                args.input
            ),
            "num_memories": len(memories),
            "num_states": len(state_rows),
            "state_induction": "kmeans",
            "num_clusters": args.num_clusters,
            "cluster_text_field": (
                args.cluster_text_field
            ),
            "encoder": args.encoder,
            "embedding_model": (
                args.embedding_model
                if args.encoder == "qwen"
                else None
            ),
            "state_representation": (
                "KMeans cluster over Gemma behavior-text embeddings"
            ),
            "cluster_quality": quality,
        },
        "states": state_rows,
    }

    json_dump(
        state_payload,
        out_dir / "behavior_states.json",
    )
    np.save(
        out_dir / "behavior_state_embeddings.npy",
        state_centroids,
    )

    assignment_rows: List[Dict[str, Any]] = []
    for idx, m in enumerate(memories):
        sid = idx_to_state[idx]
        cluster_idx = int(labels[idx])
        centroid = centers[cluster_idx]
        sim = float(
            embeddings[idx] @ centroid
        )

        assignment_rows.append({
            "memory_id": m["memory_id"],
            "precompute_id": m.get(
                "precompute_id"
            ),
            "user_id": m["user_id"],
            "window_index": m["window_index"],
            "state_id": sid,
            "cluster_id": cluster_idx,
            "cluster_similarity": sim,
            "cluster_text": m["cluster_text"],
            "pattern_description": m.get(
                "pattern_description"
            ),
            "behavior_explanation": m.get(
                "behavior_explanation"
            ),
            "behavior_signature": m.get(
                "behavior_signature"
            ),
            "keywords": m.get("keywords"),
            "state_induction": "kmeans",
        })

    jsonl_dump(
        assignment_rows,
        out_dir / "memory_state_assignments.jsonl",
    )

    print("[5/8] Reconstruct ordered cluster-state sequences")
    user_sequences = build_user_sequences(
        memories,
        idx_to_state,
        collapse_consecutive=(
            args.collapse_consecutive_states
        ),
    )

    json_dump(
        {
            "metadata": {
                "num_users": len(user_sequences),
                "collapse_consecutive_states": (
                    args.collapse_consecutive_states
                ),
                "state_induction": "kmeans",
                "num_clusters": args.num_clusters,
            },
            "users": user_sequences,
        },
        out_dir / "user_behavior_sequences.json",
    )

    observed_transitions = sum(
        max(0, u["length"] - 1)
        for u in user_sequences.values()
    )

    print(
        f"      observed_adjacent_transitions="
        f"{observed_transitions}"
    )

    if observed_transitions <= 0:
        raise ValueError(
            "No transitions after sequence construction"
        )

    print(
        "[6/8] Generate overlapping "
        "context -> next observations"
    )
    ctx_stats = accumulate_contexts(
        user_sequences=user_sequences,
        max_order=args.max_order,
        max_evidence=args.max_evidence,
        seed=args.seed,
    )

    print(
        f"      unique_contexts_including_root="
        f"{len(ctx_stats)}"
    )

    print(
        "[7/8] Estimate collaborative next distributions "
        "+ suffix smoothing"
    )
    records = build_context_records(
        stats=ctx_stats,
        state_rows=state_rows,
        count_mode=args.count_mode,
        smoothing_kappa=args.smoothing_kappa,
        min_support_users=args.min_support_users,
        min_support_occurrences=(
            args.min_support_occurrences
        ),
        min_jsd=args.min_jsd,
        promote_order1=(
            not args.no_promote_order1
        ),
        top_next=args.top_next,
    )

    jsonl_dump(
        [
            records[c]
            for c in sorted(
                records,
                key=lambda x: (len(x), x),
            )
        ],
        out_dir / "context_observations.jsonl",
    )

    print("[8/8] Build ONE physical reverse suffix tree")
    tree = build_reverse_tree(records)
    validate_tree_structure(
        tree,
        records,
    )

    tree_payload = {
        "metadata": {
            "tree_type": (
                "collaborative_variable_order_"
                "reverse_suffix_behavior_tree"
            ),
            "single_tree": True,
            "time_direction_for_learning": (
                "chronological"
            ),
            "index_direction": "reverse_suffix",
            "count_mode": args.count_mode,
            "smoothing": {
                "method": (
                    "suffix_parent_interpolation"
                ),
                "lambda": (
                    "support_users/"
                    "(support_users+kappa)"
                ),
                "kappa": args.smoothing_kappa,
            },
            "activation": {
                "min_support_users": (
                    args.min_support_users
                ),
                "min_support_occurrences": (
                    args.min_support_occurrences
                ),
                "min_jsd_raw_to_suffix_parent_bits": (
                    args.min_jsd
                ),
                "promote_order1": (
                    not args.no_promote_order1
                ),
                "structural_nodes_are_retained_when_inactive": True,
            },
            "input": os.path.abspath(args.input),
            "num_users": len(user_sequences),
            "num_memories": len(memories),
            "num_states": len(state_rows),
            "state_identity": "kmeans_cluster_id",
            "state_representation": (
                f"KMeans(K={args.num_clusters}) over "
                f"{args.cluster_text_field} embeddings"
            ),
            "observed_adjacent_transitions": (
                observed_transitions
            ),
            "max_order": args.max_order,
        },
        **tree,
    }

    json_dump(
        tree_payload,
        out_dir / "behavior_tree.json",
    )

    by_order = defaultdict(
        lambda: {
            "structural": 0,
            "active": 0,
        }
    )
    for ctx, rec in records.items():
        order = len(ctx)
        by_order[order]["structural"] += 1
        if rec["active_predictive"]:
            by_order[order]["active"] += 1

    state_sizes = [
        int(x["member_count"])
        for x in state_rows
    ]
    largest_state_size = (
        max(state_sizes)
        if state_sizes
        else 0
    )

    raw_self_transitions = 0
    raw_total_transitions = 0
    distinct_states_per_user: List[int] = []

    for u in user_sequences.values():
        seq = u["states"]
        distinct_states_per_user.append(
            len(set(seq))
        )
        for a, b in zip(
            seq[:-1],
            seq[1:],
        ):
            raw_total_transitions += 1
            raw_self_transitions += int(
                a == b
            )

    cluster_size_distribution = {
        "min": int(np.min(state_sizes)),
        "p25": float(np.quantile(
            state_sizes, 0.25
        )),
        "p50": float(np.quantile(
            state_sizes, 0.50
        )),
        "p75": float(np.quantile(
            state_sizes, 0.75
        )),
        "max": int(np.max(state_sizes)),
    }

    stats_payload = {
        "input": os.path.abspath(args.input),
        "num_memories": len(memories),
        "num_users": len(user_sequences),
        "num_behavior_states": len(state_rows),
        "state_induction": "kmeans",
        "num_clusters": args.num_clusters,
        "cluster_text_field": (
            args.cluster_text_field
        ),
        "cluster_quality": quality,
        "cluster_size_distribution": (
            cluster_size_distribution
        ),
        "mean_memories_per_state": float(
            np.mean(state_sizes)
        ),
        "median_memories_per_state": float(
            np.median(state_sizes)
        ),
        "num_singleton_states": int(
            sum(1 for n in state_sizes if n == 1)
        ),
        "largest_state_size": int(
            largest_state_size
        ),
        "largest_state_ratio": float(
            largest_state_size / len(memories)
        ),
        "observed_adjacent_transitions": (
            observed_transitions
        ),
        "self_transition_count": (
            raw_self_transitions
        ),
        "self_transition_ratio": float(
            raw_self_transitions
            / raw_total_transitions
        ) if raw_total_transitions else 0.0,
        "mean_distinct_states_per_user": float(
            np.mean(distinct_states_per_user)
        ) if distinct_states_per_user else 0.0,
        "num_users_with_one_state": int(
            sum(
                1
                for x in distinct_states_per_user
                if x == 1
            )
        ),
        "num_structural_context_nodes_including_root": (
            len(records)
        ),
        "num_active_predictive_nodes_including_root": (
            tree["num_active_predictive_nodes"]
        ),
        "contexts_by_order": {
            str(k): v
            for k, v
            in sorted(by_order.items())
        },
        "config": vars(args),
    }

    json_dump(
        stats_payload,
        out_dir / "tree_stats.json",
    )

    print("\n[DONE] Files")
    for name in [
        "cluster_behavior_inputs.jsonl",
        "behavior_states.json",
        "behavior_state_embeddings.npy",
        "memory_state_assignments.jsonl",
        "user_behavior_sequences.json",
        "context_observations.jsonl",
        "behavior_tree.json",
        "tree_stats.json",
    ]:
        print(f"  {out_dir / name}")

    print("\nKey stats")
    print(json.dumps({
        "num_memories": len(memories),
        "num_users": len(user_sequences),
        "num_states": len(state_rows),
        "largest_state_size": int(
            largest_state_size
        ),
        "largest_state_ratio": float(
            largest_state_size / len(memories)
        ),
        "observed_adjacent_transitions": (
            observed_transitions
        ),
        "structural_nodes": len(records),
        "active_predictive_nodes": (
            tree["num_active_predictive_nodes"]
        ),
        "self_transition_ratio": float(
            raw_self_transitions
            / raw_total_transitions
        ) if raw_total_transitions else 0.0,
        "mean_distinct_states_per_user": float(
            np.mean(distinct_states_per_user)
        ) if distinct_states_per_user else 0.0,
        "silhouette_cosine": (
            quality.get("silhouette_cosine")
        ),
        "kmeans_inertia": inertia,
    }, indent=2))


if __name__ == "__main__":
    main()
