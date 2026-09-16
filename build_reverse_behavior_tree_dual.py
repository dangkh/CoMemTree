#!/usr/bin/env python3
"""
Build collaborative reverse behavior tree DIRECTLY from dual-view precomputed
behavior memories.

Expected input schema
---------------------
Each memory must already contain BOTH:

A) Ranking-oriented semantic view
   - recommendation_behavior
   - semantic_focus

B) Tree-oriented trajectory view
   - trajectory_signature
   - behavior_signature        (alias is accepted)
   - mechanism
   - scope
   - direction
   - confidence

IMPORTANT DESIGN RULE
---------------------
Only the TRAJECTORY view is used to create cluster embeddings / state identity.
recommendation_behavior and semantic_focus are preserved only for auditing and
later ranking evidence. They never enter KMeans.

Pipeline
--------
dual-view local memories
    -> abstract trajectory text
    -> hard bucket by mechanism (default)
    -> KMeans within compatible buckets
    -> confidence/margin filtering
    -> ordered accepted state segments
    -> collaborative variable-order reverse suffix tree

Ambiguous/rejected assignments BREAK a temporal segment rather than being
silently removed and bridging the states on either side. This avoids creating
false transitions across an uncertain memory.
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
DUAL_REQUIRED_FIELDS = {
    "recommendation_behavior",
    "mechanism",
    "scope",
    "direction",
}
DUAL_SCHEMA_NAME = "dual_view_semantic_plus_trajectory"


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


def normalize_behavior_value(x: Any) -> str:
    """Normalize categorical behavior attributes without inventing semantics."""
    s = clean_text(x).lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or "unknown"


def normalize_mechanism(x: Any) -> str:
    """
    Canonicalize only obvious surface variants from the dual-view prompt.
    Keep behavior mechanisms distinct; do not infer semantics.
    """
    s = normalize_behavior_value(x)
    aliases = {
        "switching": "shifting",
        "shift": "shifting",
        "cross category exploration": "cross category exploration",
        "cross-category exploration": "cross category exploration",
        "collection completion": "collection expansion",
        "preference deepening": "deepening",
        "preference refinement": "refinement",
        "repeat": "repetition",
    }
    return aliases.get(s, s)


def normalize_scope(x: Any) -> str:
    """Collapse obvious surface variants of scope labels."""
    s = normalize_behavior_value(x)
    aliases = {
        "same artist": "same creator",
        "same artists": "same creator",
        "same collection series": "same series",
        "same collection": "same series",
        "same series collection": "same series",
        "same broad category": "same category",
        "same broad categories": "same category",
        "same categories": "same category",
        "related category": "adjacent category",
        "cross category": "cross category",
        "mixed": "mixed",
    }
    return aliases.get(s, s)


def normalize_direction(x: Any) -> str:
    s = normalize_behavior_value(x)
    aliases = {
        "switch": "shift",
        "shifting": "shift",
        "deepening": "deepen",
        "broadening": "broaden",
        "narrowing": "narrow",
        "repetition": "repeat",
        "returning": "return",
    }
    return aliases.get(s, s)


def behavior_profile(row: Dict[str, Any]) -> Dict[str, str]:
    return {
        "mechanism": normalize_mechanism(row.get("mechanism")),
        "scope": normalize_scope(row.get("scope")),
        "direction": normalize_direction(row.get("direction")),
    }


def behavior_bucket_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """Full structured profile retained for diagnostics/state summaries."""
    p = behavior_profile(row)
    return (p["mechanism"], p["scope"], p["direction"])


def clustering_bucket_key(
    row: Dict[str, Any],
    constraint_level: str,
) -> Tuple[str, ...]:
    """
    Hard compatibility key used by constrained KMeans.

    Default mechanism-only prevents grossly incompatible behaviors from sharing
    a state while scope/direction/trajectory_signature still split geometrically.
    """
    p = behavior_profile(row)
    if constraint_level == "mechanism":
        return (p["mechanism"],)
    if constraint_level == "mechanism_scope":
        return (p["mechanism"], p["scope"])
    if constraint_level == "full":
        return (p["mechanism"], p["scope"], p["direction"])
    raise ValueError(
        f"Unknown constraint_level={constraint_level}"
    )


def abstract_behavior_text(profile: Dict[str, str]) -> str:
    return (
        f"mechanism={profile['mechanism']}; "
        f"scope={profile['scope']}; "
        f"direction={profile['direction']}"
    )


def get_trajectory_signature(row: Dict[str, Any]) -> str:
    # New schema uses trajectory_signature. behavior_signature is an alias.
    return clean_text(
        row.get("trajectory_signature")
        or row.get("behavior_signature")
    )


def choose_cluster_text(
    row: Dict[str, Any],
    field: str,
) -> str:
    """
    IMPORTANT: recommendation_behavior / semantic_focus NEVER enter this text.
    """
    signature = get_trajectory_signature(row)
    p = behavior_profile(row)

    if field in {
        "trajectory_signature",
        "behavior_signature",
    }:
        text = signature
    elif field == "structured_combined":
        text = (
            f"mechanism={p['mechanism']}; "
            f"scope={p['scope']}; "
            f"direction={p['direction']}; "
            f"trajectory={signature}"
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
    """
    Load already-precomputed dual-view memories.

    No LLM structuring is performed here.
    """
    rows: List[Dict[str, Any]] = []
    skipped_invalid_schema = 0
    skipped_parse_fail = 0

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
                        f"[WARN] line {row_idx + 1} missing "
                        f"{sorted(missing)}; skipped",
                        file=sys.stderr,
                    )
                    continue
                raise ValueError(
                    f"line {row_idx + 1} missing {sorted(missing)}"
                )

            if x.get("parse_ok") is False:
                skipped_parse_fail += 1
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

            # Accept the explicit trajectory field or its compatibility alias.
            trajectory_signature = get_trajectory_signature(x)

            missing_dual = [
                field
                for field in DUAL_REQUIRED_FIELDS
                if not clean_text(x.get(field))
            ]
            if not trajectory_signature:
                missing_dual.append("trajectory_signature/behavior_signature")

            if missing_dual:
                skipped_invalid_schema += 1
                msg = (
                    f"line {row_idx + 1} is not a valid dual-view memory; "
                    f"missing/empty={missing_dual}"
                )
                if skip_invalid:
                    print(f"[WARN] {msg}; skipped", file=sys.stderr)
                    continue
                raise ValueError(msg)

            y = dict(x)
            y["memory_id"] = make_memory_id(x, row_idx)
            y["user_id"] = uid
            y["window_index"] = int(x["window_index"])

            # Normalize dual fields without overwriting raw semantic evidence.
            y["trajectory_signature"] = trajectory_signature
            y["behavior_signature"] = trajectory_signature
            y["recommendation_behavior"] = clean_text(
                x.get("recommendation_behavior")
            )

            semantic_focus = x.get("semantic_focus")
            if semantic_focus is None:
                semantic_focus = x.get("keywords", [])
            if not isinstance(semantic_focus, list):
                semantic_focus = [str(semantic_focus)] if semantic_focus else []
            y["semantic_focus"] = [
                clean_text(v)
                for v in semantic_focus
                if clean_text(v)
            ][:12]

            prof = behavior_profile(y)
            y["mechanism"] = prof["mechanism"]
            y["scope"] = prof["scope"]
            y["direction"] = prof["direction"]
            y["behavior_profile"] = prof
            y["behavior_bucket_key"] = list(
                behavior_bucket_key(y)
            )

            cluster_text = choose_cluster_text(
                y,
                cluster_text_field,
            )
            if not cluster_text:
                msg = (
                    f"line {row_idx + 1}: empty cluster text "
                    f"for field={cluster_text_field}"
                )
                if skip_invalid:
                    print(f"[WARN] {msg}; skipped", file=sys.stderr)
                    continue
                raise ValueError(msg)

            y["cluster_text"] = cluster_text
            y["_row_idx"] = row_idx
            rows.append(y)

    if not rows:
        raise ValueError(
            "No valid dual-view memories found after filtering"
        )

    print(
        f"      skipped_parse_fail={skipped_parse_fail} "
        f"skipped_invalid_dual_schema={skipped_invalid_schema}"
    )

    return rows


def dual_schema_stats(
    memories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    mechanism_counts = Counter(
        m["mechanism"] for m in memories
    )
    scope_counts = Counter(
        m["scope"] for m in memories
    )
    direction_counts = Counter(
        m["direction"] for m in memories
    )

    unknown_profile_count = sum(
        1
        for m in memories
        if "unknown" in behavior_bucket_key(m)
    )

    return {
        "schema": DUAL_SCHEMA_NAME,
        "num_memories": len(memories),
        "unknown_profile_count": int(unknown_profile_count),
        "unknown_profile_ratio": float(
            unknown_profile_count / len(memories)
        ),
        "mechanism_counts": dict(
            mechanism_counts.most_common()
        ),
        "scope_counts": dict(
            scope_counts.most_common()
        ),
        "direction_counts": dict(
            direction_counts.most_common()
        ),
    }


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

def _fit_kmeans_block(
    x: np.ndarray,
    k: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise RuntimeError("Need scikit-learn for KMeans clustering") from e

    if k <= 0 or k > len(x):
        raise ValueError(f"invalid k={k} for block size={len(x)}")

    if k == 1:
        raw_center = np.mean(x, axis=0, keepdims=True).astype(np.float32)
        labels = np.zeros(len(x), dtype=np.int64)
        diff = x - raw_center[0]
        inertia = float(np.sum(diff * diff))
        return labels, l2_normalize(raw_center), inertia

    km = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        verbose=0,
    )
    labels = km.fit_predict(x)
    centers = l2_normalize(np.asarray(km.cluster_centers_, dtype=np.float32))
    return np.asarray(labels, dtype=np.int64), centers, float(km.inertia_)


def _allocate_clusters_to_buckets(
    bucket_sizes: Dict[Tuple[str, ...], int],
    requested_k: int,
) -> Dict[Tuple[str, ...], int]:
    """
    Allocate at least one cluster per behavior bucket, then split the largest
    expected cluster until the requested/effective total K is reached.

    If the number of behavior buckets is larger than requested_k, the effective
    K is increased to the number of buckets instead of merging incompatible
    behavior profiles.
    """
    if not bucket_sizes:
        raise ValueError("No behavior buckets available")
    if requested_k <= 1:
        raise ValueError("--num-clusters must be > 1")

    if len(bucket_sizes) > requested_k:
        raise ValueError(
            f"Hard constraint creates {len(bucket_sizes)} buckets > requested K={requested_k}. "
            "Use --constraint-level mechanism (recommended), increase K, or use global mode."
        )
    effective_k = min(int(requested_k), int(sum(bucket_sizes.values())))

    alloc = {key: 1 for key in bucket_sizes}
    remaining = effective_k - len(alloc)

    while remaining > 0:
        candidates = [
            key for key, size in bucket_sizes.items()
            if alloc[key] < size
        ]
        if not candidates:
            break
        # Split the bucket with the largest expected members/cluster.
        key = max(
            candidates,
            key=lambda z: (bucket_sizes[z] / alloc[z], bucket_sizes[z], z),
        )
        alloc[key] += 1
        remaining -= 1

    return alloc


def run_global_kmeans(
    embeddings: np.ndarray,
    num_clusters: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[int, Dict[str, Any]], Dict[str, Any]]:
    labels, centers, inertia = _fit_kmeans_block(
        embeddings, num_clusters, seed, n_init, max_iter
    )
    meta = {
        int(i): {
            "cluster_id": int(i),
            "cluster_mode": "global",
            "behavior_profile": None,
            "behavior_bucket_key": None,
        }
        for i in range(len(centers))
    }
    return labels, centers, inertia, meta, {
        "requested_num_clusters": int(num_clusters),
        "effective_num_clusters": int(len(centers)),
        "num_behavior_buckets": None,
        "bucket_cluster_allocation": None,
    }


def run_constrained_kmeans(
    memories: List[Dict[str, Any]],
    embeddings: np.ndarray,
    num_clusters: int,
    seed: int,
    n_init: int,
    max_iter: int,
    constraint_level: str,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """
    HARD constraint according to constraint_level.
    Recommended default is mechanism-only.
    """
    bucket_to_indices: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for idx, m in enumerate(memories):
        bucket_to_indices[clustering_bucket_key(m, constraint_level)].append(idx)

    bucket_sizes = {k: len(v) for k, v in bucket_to_indices.items()}
    allocation = _allocate_clusters_to_buckets(bucket_sizes, num_clusters)

    effective_k = sum(allocation.values())

    labels = np.full(len(memories), -1, dtype=np.int64)
    centers: List[np.ndarray] = []
    cluster_meta: Dict[int, Dict[str, Any]] = {}
    total_inertia = 0.0
    global_cluster_id = 0

    for bucket_idx, key in enumerate(sorted(bucket_to_indices.keys())):
        indices = bucket_to_indices[key]
        local_x = embeddings[indices]
        local_k = allocation[key]
        local_labels, local_centers, inertia = _fit_kmeans_block(
            local_x,
            local_k,
            seed + bucket_idx,
            n_init,
            max_iter,
        )
        total_inertia += inertia

        for local_cluster_id in range(local_k):
            gid = global_cluster_id
            global_cluster_id += 1
            centers.append(local_centers[local_cluster_id].astype(np.float32))
            cluster_meta[gid] = {
                "cluster_id": gid,
                "cluster_mode": "constrained",
                "behavior_profile": None,
                "behavior_bucket_key": list(key),
                "bucket_size": len(indices),
                "clusters_in_bucket": local_k,
                "local_cluster_id": local_cluster_id,
            }
            for pos, mem_idx in enumerate(indices):
                if int(local_labels[pos]) == local_cluster_id:
                    labels[mem_idx] = gid

    if np.any(labels < 0):
        bad = np.where(labels < 0)[0][:10].tolist()
        raise AssertionError(f"Unassigned memories after constrained KMeans: {bad}")

    centers_arr = np.stack(centers, axis=0).astype(np.float32)
    allocation_json = {
        " | ".join(key): int(v)
        for key, v in sorted(allocation.items())
    }
    info = {
        "requested_num_clusters": int(num_clusters),
        "effective_num_clusters": int(len(centers_arr)),
        "num_behavior_buckets": int(len(bucket_to_indices)),
        "constraint_level": constraint_level,
        "bucket_cluster_allocation": allocation_json,
        "bucket_sizes": {
            " | ".join(key): int(size)
            for key, size in sorted(bucket_sizes.items())
        },
    }
    return labels, centers_arr, float(total_inertia), cluster_meta, info


def run_kmeans(
    memories: List[Dict[str, Any]],
    embeddings: np.ndarray,
    num_clusters: int,
    seed: int,
    n_init: int,
    max_iter: int,
    cluster_mode: str,
    constraint_level: str,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[int, Dict[str, Any]], Dict[str, Any]]:
    if num_clusters > len(embeddings):
        raise ValueError(
            f"--num-clusters={num_clusters} > num_memories={len(embeddings)}"
        )
    if cluster_mode == "global":
        return run_global_kmeans(
            embeddings, num_clusters, seed, n_init, max_iter
        )
    if cluster_mode == "constrained":
        return run_constrained_kmeans(
            memories, embeddings, num_clusters, seed, n_init, max_iter, constraint_level
        )
    raise ValueError(cluster_mode)


def assignment_diagnostics(
    memories: List[Dict[str, Any]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    cluster_meta: Dict[int, Dict[str, Any]],
    cluster_mode: str,
    constraint_level: str,
    min_similarity: float,
    min_margin: float,
) -> List[Dict[str, Any]]:
    """
    Compute top-1/top-2 cosine similarity and margin among COMPATIBLE clusters.
    In constrained mode, compatible means same behavior bucket. In global mode,
    all clusters are compatible.
    """
    bucket_to_clusters: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    if cluster_mode == "constrained":
        for cid, meta in cluster_meta.items():
            key = tuple(meta["behavior_bucket_key"])
            bucket_to_clusters[key].append(int(cid))

    rows: List[Dict[str, Any]] = []
    all_ids = list(range(len(centers)))

    for idx, m in enumerate(memories):
        assigned = int(labels[idx])
        key = clustering_bucket_key(m, constraint_level)
        candidate_ids = (
            sorted(bucket_to_clusters[key])
            if cluster_mode == "constrained"
            else all_ids
        )
        if assigned not in candidate_ids:
            raise AssertionError(
                f"assigned cluster {assigned} not compatible with behavior bucket {key}"
            )

        sims = np.asarray(
            [float(embeddings[idx] @ centers[cid]) for cid in candidate_ids],
            dtype=np.float64,
        )
        order = np.argsort(-sims)
        top1_pos = int(order[0])
        top1_cid = int(candidate_ids[top1_pos])
        top1_sim = float(sims[top1_pos])

        if len(order) >= 2:
            top2_pos = int(order[1])
            top2_cid: Optional[int] = int(candidate_ids[top2_pos])
            top2_sim: Optional[float] = float(sims[top2_pos])
        else:
            top2_cid = None
            top2_sim = None

        # Confidence is defined relative to the ACTUAL assigned cluster, not
        # merely top1-vs-top2. This remains valid even if standard KMeans'
        # Euclidean assignment differs slightly from cosine-to-normalized-center.
        assigned_sim = float(embeddings[idx] @ centers[assigned])
        alternative_sims = [
            float(embeddings[idx] @ centers[cid])
            for cid in candidate_ids
            if cid != assigned
        ]
        if alternative_sims:
            best_alternative_sim: Optional[float] = max(alternative_sims)
            margin: Optional[float] = float(assigned_sim - best_alternative_sim)
            margin_ok = margin >= min_margin
        else:
            best_alternative_sim = None
            margin = None
            margin_ok = True

        nearest_agrees = bool(top1_cid == assigned)
        accepted = bool(
            assigned_sim >= min_similarity
            and margin_ok
        )

        global_sims = embeddings[idx] @ centers.T
        global_order = np.argsort(-global_sims)
        global_top1 = int(global_order[0])
        global_top2 = int(global_order[1]) if len(global_order) > 1 else None

        rows.append({
            "memory_index": idx,
            "assigned_cluster_id": assigned,
            "assigned_similarity": assigned_sim,
            "compatible_top1_cluster_id": top1_cid,
            "compatible_top1_similarity": top1_sim,
            "compatible_top2_cluster_id": top2_cid,
            "compatible_top2_similarity": top2_sim,
            "compatible_similarity_margin": margin,
            "best_alternative_similarity": best_alternative_sim,
            "num_compatible_clusters": len(candidate_ids),
            "nearest_compatible_agrees_with_assignment": nearest_agrees,
            "global_top1_cluster_id": global_top1,
            "global_top1_similarity": float(global_sims[global_top1]),
            "global_top2_cluster_id": global_top2,
            "global_top2_similarity": (
                float(global_sims[global_top2]) if global_top2 is not None else None
            ),
            "assignment_accepted": accepted,
            "rejection_reason": (
                None if accepted else (
                    "low_similarity" if assigned_sim < min_similarity else
                    "low_margin"
                )
            ),
        })

    return rows


def build_cluster_states(
    memories: List[Dict[str, Any]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    cluster_meta: Dict[int, Dict[str, Any]],
    assignment_info: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[int, str],
    List[np.ndarray],
]:
    state_rows: List[Dict[str, Any]] = []
    idx_to_state: Dict[int, str] = {}
    centroid_rows: List[np.ndarray] = []

    accepted_mask = np.asarray(
        [bool(x["assignment_accepted"]) for x in assignment_info], dtype=bool
    )

    for cluster_idx in range(len(centers)):
        member_indices = np.where(labels == cluster_idx)[0].tolist()
        if not member_indices:
            continue

        sid = f"C{cluster_idx:03d}"
        centroid = centers[cluster_idx]
        mat = embeddings[member_indices]
        sims = mat @ centroid
        medoid_local = int(np.argmax(sims))
        medoid_idx = member_indices[medoid_local]
        medoid = memories[medoid_idx]

        accepted_member_indices = [i for i in member_indices if accepted_mask[i]]
        support_users = sorted({str(memories[i]["user_id"]) for i in member_indices})
        accepted_support_users = sorted({
            str(memories[i]["user_id"]) for i in accepted_member_indices
        })

        examples = list(dict.fromkeys(
            memories[i]["cluster_text"] for i in member_indices
        ))[:12]

        keyword_counts: Counter = Counter()
        signature_counts: Counter = Counter()
        semantic_focus_counts: Counter = Counter()
        recommendation_examples: List[str] = []

        for i in member_indices:
            sig = get_trajectory_signature(
                memories[i]
            )
            if sig:
                signature_counts[sig] += 1

            for sf in memories[i].get(
                "semantic_focus",
                [],
            ):
                s = clean_text(sf).lower()
                if s:
                    semantic_focus_counts[s] += 1

            rb = clean_text(
                memories[i].get(
                    "recommendation_behavior"
                )
            )
            if rb and rb not in recommendation_examples:
                recommendation_examples.append(rb)

            kws = memories[i].get("keywords") or []
            if not isinstance(kws, list):
                kws = [str(kws)]
            for kw in kws:
                s = clean_text(kw).lower()
                if s:
                    keyword_counts[s] += 1

        meta = dict(cluster_meta[int(cluster_idx)])
        # Always summarize each learned cluster using its dominant FULL structured
        # profile. The hard constraint may only be mechanism-level.
        profiles = [behavior_bucket_key(memories[i]) for i in member_indices]
        majority_key, majority_count = Counter(profiles).most_common(1)[0]
        profile = {
            "mechanism": majority_key[0],
            "scope": majority_key[1],
            "direction": majority_key[2],
        }
        meta["majority_profile_fraction"] = float(majority_count / len(member_indices))

        canonical = abstract_behavior_text(profile)
        dominant_signature = (
            signature_counts.most_common(1)[0][0]
            if signature_counts else ""
        )

        state_rows.append({
            "state_id": sid,
            "cluster_id": int(cluster_idx),
            "cluster_mode": meta.get("cluster_mode"),
            "behavior_profile": profile,
            "behavior_bucket_key": meta.get("behavior_bucket_key"),
            "mechanism": profile["mechanism"],
            "scope": profile["scope"],
            "direction": profile["direction"],
            # IMPORTANT: abstract canonical text prevents content leakage from
            # the medoid example into downstream next-behavior prompts.
            "canonical_text": canonical,
            "canonical_behavior_signature": dominant_signature,
            "representative_signature": dominant_signature,
            # Diagnostics only: never required in the ranking prompt.
            "representative_pattern_description": clean_text(
                medoid.get("pattern_description")
            ),
            "representative_behavior_explanation": clean_text(
                medoid.get("behavior_explanation")
            ),
            "medoid_memory_id": medoid["memory_id"],
            "cluster_text_examples": examples,
            "member_count": len(member_indices),
            "accepted_member_count": len(accepted_member_indices),
            "rejected_member_count": len(member_indices) - len(accepted_member_indices),
            "support_user_count": len(support_users),
            "accepted_support_user_count": len(accepted_support_users),
            "support_users": support_users,
            "accepted_support_users": accepted_support_users,
            "memory_ids": [memories[i]["memory_id"] for i in member_indices],
            "accepted_memory_ids": [
                memories[i]["memory_id"] for i in accepted_member_indices
            ],
            # Semantic fields below are DIAGNOSTICS ONLY.
            # They were NOT used to construct the cluster embedding/state.
            "semantic_evidence_is_state_identity": False,
            "top_semantic_focus": [
                k
                for k, _
                in semantic_focus_counts.most_common(20)
            ],
            "recommendation_behavior_examples": (
                recommendation_examples[:8]
            ),
            "top_keywords": [
                k
                for k, _
                in keyword_counts.most_common(12)
            ],
            "mean_similarity_to_centroid": float(np.mean(sims)),
            "min_similarity_to_centroid": float(np.min(sims)),
            "max_similarity_to_centroid": float(np.max(sims)),
            "cluster_metadata": meta,
        })

        centroid_rows.append(centroid.astype(np.float32))
        for i in accepted_member_indices:
            idx_to_state[i] = sid

    state_rows.sort(key=lambda x: int(x["cluster_id"]))
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
    """
    Build accepted state sequences.

    CRITICAL:
    An ambiguous/rejected memory BREAKS the temporal segment.
    We do NOT delete it and connect the states before/after it, because that
    would fabricate a transition that was never observed.
    """
    by_user: Dict[
        str,
        List[Tuple[int, Dict[str, Any]]],
    ] = defaultdict(list)

    for idx, m in enumerate(memories):
        by_user[str(m["user_id"])].append(
            (idx, m)
        )

    out: Dict[str, Dict[str, Any]] = {}

    for uid, items in by_user.items():
        items.sort(
            key=lambda x: (
                int(x[1]["window_index"]),
                int(x[1]["_row_idx"]),
            )
        )

        segments: List[Dict[str, Any]] = []
        current_entries: List[Dict[str, Any]] = []
        skipped_memory_ids: List[str] = []
        last_sid: Optional[str] = None

        def flush_segment() -> None:
            nonlocal current_entries, last_sid
            if current_entries:
                segments.append({
                    "states": [
                        e["state_id"]
                        for e in current_entries
                    ],
                    "entries": current_entries,
                    "length": len(current_entries),
                })
            current_entries = []
            last_sid = None

        for idx, m in items:
            sid = idx_to_state.get(idx)

            if sid is None:
                skipped_memory_ids.append(
                    m["memory_id"]
                )
                flush_segment()
                continue

            if (
                collapse_consecutive
                and sid == last_sid
                and current_entries
            ):
                current_entries[-1][
                    "memory_ids"
                ].append(m["memory_id"])
                current_entries[-1][
                    "window_indices"
                ].append(
                    int(m["window_index"])
                )
                continue

            current_entries.append({
                "state_id": sid,
                "memory_ids": [
                    m["memory_id"]
                ],
                "window_indices": [
                    int(m["window_index"])
                ],
            })
            last_sid = sid

        flush_segment()

        # Compatibility view only. NEVER use this flattened list for transition
        # counting because it can cross segment boundaries.
        flat_entries = [
            e
            for seg in segments
            for e in seg["entries"]
        ]

        out[uid] = {
            "user_id": uid,
            "length": len(flat_entries),
            "states": [
                e["state_id"]
                for e in flat_entries
            ],
            "entries": flat_entries,
            "segments": segments,
            "num_segments": len(segments),
            "num_skipped_ambiguous_memories": len(
                skipped_memory_ids
            ),
            "skipped_ambiguous_memory_ids": (
                skipped_memory_ids
            ),
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
    stats: Dict[
        Tuple[str, ...],
        ContextStats,
    ] = defaultdict(ContextStats)

    for uid, rec in user_sequences.items():
        segments = rec.get("segments")

        if segments is None:
            # Backward compatibility for older sequence payloads.
            segments = [{
                "states": rec.get("states", []),
                "entries": rec.get("entries", []),
            }]

        for segment_index, seg in enumerate(
            segments
        ):
            seq = seg["states"]
            entries = seg["entries"]

            if len(seq) < 2:
                continue

            for t in range(1, len(seq)):
                target = seq[t]

                root_ev = {
                    "user_id": uid,
                    "segment_index": segment_index,
                    "context_states": [],
                    "reverse_tree_path": [],
                    "target_state": target,
                    "target_memory_ids": (
                        entries[t]["memory_ids"]
                    ),
                    "target_window_indices": (
                        entries[t]["window_indices"]
                    ),
                }
                stats[()].add(
                    uid,
                    target,
                    root_ev,
                    max_evidence,
                    rng,
                )

                for L in range(
                    1,
                    min(max_order, t) + 1,
                ):
                    context = tuple(
                        seq[t-L:t]
                    )
                    ev = {
                        "user_id": uid,
                        "segment_index": segment_index,
                        "context_states": list(
                            context
                        ),
                        "reverse_tree_path": list(
                            reversed(context)
                        ),
                        "context_memory_ids": [
                            entries[j][
                                "memory_ids"
                            ]
                            for j in range(
                                t-L,
                                t,
                            )
                        ],
                        "context_window_indices": [
                            entries[j][
                                "window_indices"
                            ]
                            for j in range(
                                t-L,
                                t,
                            )
                        ],
                        "target_state": target,
                        "target_memory_ids": (
                            entries[t][
                                "memory_ids"
                            ]
                        ),
                        "target_window_indices": (
                            entries[t][
                                "window_indices"
                            ]
                        ),
                    }
                    stats[context].add(
                        uid,
                        target,
                        ev,
                        max_evidence,
                        rng,
                    )

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
    state_meta = {
        x["state_id"]: {
            "behavior_profile": x.get("behavior_profile"),
            "mechanism": x.get("mechanism"),
            "scope": x.get("scope"),
            "direction": x.get("direction"),
        }
        for x in state_rows
    }
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
                    "behavior_profile": state_meta.get(sid, {}).get("behavior_profile"),
                    "mechanism": state_meta.get(sid, {}).get("mechanism"),
                    "scope": state_meta.get(sid, {}).get("scope"),
                    "direction": state_meta.get(sid, {}).get("direction"),
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
            "Dual-view behavior cluster tree: trajectory view -> Qwen -> "
            "constrained KMeans -> reverse behavior tree"
        )
    )

    p.add_argument(
        "--input",
        default="precomputed/CDs/local_memories_gemma_dual_300.jsonl",
        help="Dual-view precomputed behavior-memory JSONL.",
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
            "structured_combined",
            "trajectory_signature",
            "behavior_signature",
        ],
        default="structured_combined",
        help=(
            "Text embedded for KMeans. structured_combined is recommended and "
            "uses ONLY mechanism/scope/direction + trajectory_signature. "
            "Semantic recommendation fields are never embedded."
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
        help=(
            "Requested TOTAL number of clusters. With the default mechanism-level "
            "constraint, effective K remains equal to requested K whenever feasible."
        ),
    )
    p.add_argument(
        "--cluster-mode",
        choices=["constrained", "global"],
        default="constrained",
        help=(
            "constrained = hard bucket by --constraint-level before KMeans; "
            "global = old unconstrained ablation."
        ),
    )
    p.add_argument(
        "--constraint-level",
        choices=["mechanism", "mechanism_scope", "full"],
        default="mechanism",
        help=(
            "Hard constraint for constrained KMeans. Default mechanism is recommended; "
            "scope/direction remain in the embedding and can split clusters naturally."
        ),
    )
    p.add_argument(
        "--min-cluster-similarity",
        type=float,
        default=0.55,
        help="Reject a training assignment below this cosine similarity.",
    )
    p.add_argument(
        "--min-cluster-margin",
        type=float,
        default=0.02,
        help=(
            "Reject a training assignment when top1-top2 cosine margin among "
            "compatible clusters is below this value."
        ),
    )
    p.add_argument(
        "--keep-ambiguous-assignments",
        action="store_true",
        help=(
            "Diagnostic override: keep low-confidence assignments in tree "
            "sequences. Default is to skip them."
        ),
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

    print("[1/8] Load dual-view precomputed behavior memories")
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

    print("[2/8] Validate dual-view schema")
    schema_stats = dual_schema_stats(
        memories
    )
    print(
        f"      unknown_profile_ratio="
        f"{schema_stats['unknown_profile_ratio']:.3f}"
    )
    print(
        f"      mechanisms="
        f"{len(schema_stats['mechanism_counts'])}"
    )

    print("[3/8] Save cluster inputs")
    jsonl_dump(
        [
            {
                "memory_id": m["memory_id"],
                "user_id": m["user_id"],
                "window_index": m["window_index"],
                "cluster_text": m["cluster_text"],

                # Semantic/ranking view (NOT used for clustering).
                "recommendation_behavior": m.get(
                    "recommendation_behavior"
                ),
                "semantic_focus": m.get(
                    "semantic_focus"
                ),

                # Trajectory/tree view.
                "trajectory_signature": m.get(
                    "trajectory_signature"
                ),
                "behavior_signature": m.get(
                    "behavior_signature"
                ),
                "behavior_explanation": m.get(
                    "behavior_explanation"
                ),
                "mechanism": m.get("mechanism"),
                "scope": m.get("scope"),
                "direction": m.get("direction"),
                "behavior_profile": m.get("behavior_profile"),
                "behavior_bucket_key": m.get("behavior_bucket_key"),
                "keywords": m.get("keywords"),
            }
            for m in memories
        ],
        out_dir / "cluster_behavior_inputs.jsonl",
    )

    print("[4/8] Embed trajectory behavior text")
    embeddings = create_embeddings(
        args,
        memories,
    )
    embeddings = l2_normalize(embeddings)
    print(
        f"      embedding_shape={embeddings.shape}"
    )

    print(
        f"[5/8] KMeans state induction "
        f"(mode={args.cluster_mode}, requested_K={args.num_clusters})"
    )
    labels, centers, inertia, cluster_meta, cluster_info = run_kmeans(
        memories=memories,
        embeddings=embeddings,
        num_clusters=args.num_clusters,
        seed=args.seed,
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
        cluster_mode=args.cluster_mode,
        constraint_level=args.constraint_level,
    )

    assignment_info = assignment_diagnostics(
        memories=memories,
        embeddings=embeddings,
        labels=labels,
        centers=centers,
        cluster_meta=cluster_meta,
        cluster_mode=args.cluster_mode,
        constraint_level=args.constraint_level,
        min_similarity=args.min_cluster_similarity,
        min_margin=args.min_cluster_margin,
    )
    if args.keep_ambiguous_assignments:
        for row in assignment_info:
            row["assignment_accepted"] = True
            row["rejection_reason"] = None

    state_rows, idx_to_state, centroid_rows = build_cluster_states(
        memories=memories,
        embeddings=embeddings,
        labels=labels,
        centers=centers,
        cluster_meta=cluster_meta,
        assignment_info=assignment_info,
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

    accepted_count = sum(1 for x in assignment_info if x["assignment_accepted"])
    rejected_count = len(assignment_info) - accepted_count
    margins = [
        x["compatible_similarity_margin"]
        for x in assignment_info
        if x["compatible_similarity_margin"] is not None
    ]
    quality["assignment_quality"] = {
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "accepted_ratio": float(accepted_count / len(assignment_info)),
        "min_cluster_similarity": args.min_cluster_similarity,
        "min_cluster_margin": args.min_cluster_margin,
        "keep_ambiguous_assignments": args.keep_ambiguous_assignments,
        "mean_compatible_margin": float(np.mean(margins)) if margins else None,
        "p10_compatible_margin": float(np.quantile(margins, 0.10)) if margins else None,
        "p50_compatible_margin": float(np.quantile(margins, 0.50)) if margins else None,
    }

    print(
        f"      states={len(state_rows)} "
        f"effective_K={cluster_info['effective_num_clusters']} "
        f"inertia={inertia:.4f}"
    )
    print(
        f"      accepted_assignments={accepted_count}/{len(assignment_info)} "
        f"({accepted_count / len(assignment_info):.3f})"
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
            "state_induction": "constrained_kmeans" if args.cluster_mode == "constrained" else "global_kmeans",
            "requested_num_clusters": args.num_clusters,
            "effective_num_clusters": cluster_info["effective_num_clusters"],
            "cluster_mode": args.cluster_mode,
            "constraint_level": args.constraint_level,
            "constraint_fields": (
                ["mechanism"] if args.constraint_level == "mechanism" else
                ["mechanism", "scope"] if args.constraint_level == "mechanism_scope" else
                ["mechanism", "scope", "direction"]
            ),
            "input_behavior_schema": schema_stats,
            "semantic_fields_used_for_clustering": False,
            "cluster_info": cluster_info,
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
                "Behavior-constrained KMeans (configurable hard constraint) over structured behavior embeddings" if args.cluster_mode == "constrained" else "Global KMeans over Gemma behavior-text embeddings"
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
        cluster_idx = int(labels[idx])
        sid = f"C{cluster_idx:03d}"
        diag = assignment_info[idx]

        assignment_rows.append({
            "memory_id": m["memory_id"],
            "precompute_id": m.get("precompute_id"),
            "user_id": m["user_id"],
            "window_index": m["window_index"],
            "state_id": sid,
            "cluster_id": cluster_idx,
            "cluster_similarity": diag["assigned_similarity"],
            "cluster_text": m["cluster_text"],

            # Ranking-oriented semantic view; stored only for downstream use.
            "recommendation_behavior": m.get(
                "recommendation_behavior"
            ),
            "semantic_focus": m.get(
                "semantic_focus"
            ),

            # Tree-oriented trajectory view.
            "trajectory_signature": m.get(
                "trajectory_signature"
            ),
            "behavior_signature": m.get(
                "behavior_signature"
            ),
            "behavior_explanation": m.get(
                "behavior_explanation"
            ),
            "mechanism": m.get("mechanism"),
            "scope": m.get("scope"),
            "direction": m.get("direction"),
            "behavior_profile": m.get("behavior_profile"),
            "behavior_bucket_key": m.get("behavior_bucket_key"),
            "compatible_top1_cluster_id": diag["compatible_top1_cluster_id"],
            "compatible_top1_similarity": diag["compatible_top1_similarity"],
            "compatible_top2_cluster_id": diag["compatible_top2_cluster_id"],
            "compatible_top2_similarity": diag["compatible_top2_similarity"],
            "similarity_margin": diag["compatible_similarity_margin"],
            "best_alternative_similarity": diag["best_alternative_similarity"],
            "num_compatible_clusters": diag["num_compatible_clusters"],
            "nearest_compatible_agrees_with_assignment": diag["nearest_compatible_agrees_with_assignment"],
            "global_top1_cluster_id": diag["global_top1_cluster_id"],
            "global_top1_similarity": diag["global_top1_similarity"],
            "global_top2_cluster_id": diag["global_top2_cluster_id"],
            "global_top2_similarity": diag["global_top2_similarity"],
            "assignment_accepted": diag["assignment_accepted"],
            "rejection_reason": diag["rejection_reason"],
            "keywords": m.get("keywords"),
            "state_induction": (
                "constrained_kmeans" if args.cluster_mode == "constrained"
                else "global_kmeans"
            ),
        })

    jsonl_dump(
        assignment_rows,
        out_dir / "memory_state_assignments.jsonl",
    )

    print("[6/8] Reconstruct accepted state segments")
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
                "state_induction": (
                    "constrained_kmeans" if args.cluster_mode == "constrained"
                    else "global_kmeans"
                ),
                "requested_num_clusters": args.num_clusters,
                "effective_num_clusters": cluster_info["effective_num_clusters"],
                "cluster_mode": args.cluster_mode,
                "min_cluster_similarity": args.min_cluster_similarity,
                "min_cluster_margin": args.min_cluster_margin,
                "keep_ambiguous_assignments": args.keep_ambiguous_assignments,
            },
            "users": user_sequences,
        },
        out_dir / "user_behavior_sequences.json",
    )

    observed_transitions = sum(
        max(0, seg["length"] - 1)
        for u in user_sequences.values()
        for seg in u.get("segments", [])
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
        "[7/8] Generate overlapping "
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
        "[8/8] Estimate collaborative next distributions "
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

    print("      Build ONE physical reverse suffix tree")
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
            "state_identity": "trajectory_behavior_constrained_kmeans_cluster_id",
            "input_behavior_schema": DUAL_SCHEMA_NAME,
            "semantic_fields_used_for_clustering": False,
            "state_representation": (
                f"{args.cluster_mode} KMeans(requested_K={args.num_clusters}, "
                f"effective_K={cluster_info['effective_num_clusters']}) over "
                f"{args.cluster_text_field} embeddings"
            ),
            "constraint_level": args.constraint_level,
            "constraint_fields": (
                ["mechanism"] if args.constraint_level == "mechanism" else
                ["mechanism", "scope"] if args.constraint_level == "mechanism_scope" else
                ["mechanism", "scope", "direction"]
            ),
            "assignment_filter": {
                "min_similarity": args.min_cluster_similarity,
                "min_margin": args.min_cluster_margin,
                "keep_ambiguous": args.keep_ambiguous_assignments,
            },
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
        for seg in u.get(
            "segments",
            [],
        ):
            seg_seq = seg["states"]
            for a, b in zip(
                seg_seq[:-1],
                seg_seq[1:],
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
        "state_induction": ("constrained_kmeans" if args.cluster_mode == "constrained" else "global_kmeans"),
        "requested_num_clusters": args.num_clusters,
        "effective_num_clusters": cluster_info["effective_num_clusters"],
        "cluster_mode": args.cluster_mode,
        "constraint_level": args.constraint_level,
            "constraint_fields": (
                ["mechanism"] if args.constraint_level == "mechanism" else
                ["mechanism", "scope"] if args.constraint_level == "mechanism_scope" else
                ["mechanism", "scope", "direction"]
            ),
        "cluster_info": cluster_info,
        "cluster_text_field": (
            args.cluster_text_field
        ),
        "input_behavior_schema": (
            DUAL_SCHEMA_NAME
        ),
        "semantic_fields_used_for_clustering": False,
        "dual_schema_stats": schema_stats,
        "cluster_quality": quality,
        "accepted_assignment_count": accepted_count,
        "rejected_assignment_count": rejected_count,
        "accepted_assignment_ratio": float(accepted_count / len(assignment_info)),
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
        "num_segments": int(
            sum(
                len(u.get("segments", []))
                for u in user_sequences.values()
            )
        ),
        "num_skipped_ambiguous_memories": int(
            sum(
                u.get(
                    "num_skipped_ambiguous_memories",
                    0,
                )
                for u in user_sequences.values()
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
        "effective_num_clusters": cluster_info["effective_num_clusters"],
        "accepted_assignment_ratio": float(accepted_count / len(assignment_info)),
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
