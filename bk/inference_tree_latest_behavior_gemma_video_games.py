#!/usr/bin/env python3
"""
Inference for the dual-view collaborative behavior Tree with a Video-Games-specific ranking prompt.

Core idea
---------
The dual-view precompute intentionally separates:

    WHAT the target user currently prefers
        = latest recommendation_behavior + semantic_focus

from:

    HOW that preference is expected to evolve next
        = collaborative Tree predicted abstract state
          (mechanism + scope + trajectory_signature)

The final ranking LLM receives BOTH directly. There is NO extra LLM behavior
composer, so the method uses only one ranking LLM call per condition.

Pipeline
--------
precomputed test behaviors
    -> trajectory-only mapping to frozen KMeans states
    -> query reverse suffix Tree
    -> latest semantic behavior (WHAT)
       + predicted abstract transition/state (HOW)
    -> candidate ranking

Controlled ranking conditions
-----------------------------
1) Native:
      history + candidates

2) Latest-only:
      history + latest observed semantic behavior + candidates

3) Tree (main):
      history + latest observed semantic behavior
      + Tree predicted next abstract state + candidates

All three conditions use the SAME candidate order and the SAME ranking template.
Only the available behavior blocks differ.

Important implementation details
--------------------------------
- No behavior generation at inference time.
- No GT/test item is used to construct behavior evidence.
- Candidate construction matches the current pipeline:
      test_ids + test_neg
      shuffle with random.Random(f"{seed}:{user_id}")
- Test behavior -> state mapping uses the SAME structured trajectory text used
  by build_reverse_behavior_tree_dual.py.
- When the Tree was trained with mechanism-constrained KMeans, test mapping is
  restricted to states with the same mechanism whenever compatible states exist.
- Ambiguous/rejected test mappings BREAK the recent state suffix by default,
  matching the segmented transition construction used by the current Tree builder.
- Semantic fields from other users / cluster medoids are NEVER injected into the
  ranking prompt. The predicted state contributes abstract transition information only.
- Incomplete rankings are NOT retried. They are sanitized like the native baseline.
  Summary still reports parse-complete quality for audit.

Expected inputs
---------------
Dual test behavior cache:
    precompute_test_behaviors_gemma_dual_v2.py output

Tree directory:
    build_reverse_behavior_tree_dual.py output containing:
      behavior_states.json
      behavior_state_embeddings.npy
      behavior_tree.json

Example
-------
python inference_tree_latest_behavior_gemma.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/user_negatives_10_5000.json \
  --precomputed-behaviors precomputed/CDs/test_user_behaviors_gemma_dual_v2_300.jsonl \
  --tree-dir behavior_tree_out_dual_v2_k50 \
  --output results/tree_latest_behavior_cd_300.jsonl \
  --model unsloth/gemma-3-4b-it-unsloth-bnb-4bit \
  --max-users 300 \
  --top-next 1 \
  --history-size 10 \
  --llm-batch-size 8 \
  --run-native \
  --run-latest-only \
  --seed 42
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Initialize Unsloth before any library that may import transformers/peft
# (e.g. sentence_transformers used by the Qwen encoder).
# This applies patches only; it does NOT load the Gemma model yet.
import unsloth  # noqa: F401

import torch
from tqdm.auto import tqdm


SCHEMA_VERSION = "dual_tree_latest_behavior_inference_v1"
RANK_PROMPT_VERSION = "video_games_what_plus_how_v1"


# =============================================================================
# Generic helpers
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def normalize_label(x: Any) -> str:
    # Mirror the dual Tree builder: punctuation is not semantic.
    s = clean_text(x).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s).strip() or "unknown"


def normalize_mechanism(x: Any) -> str:
    s = normalize_label(x)
    aliases = {
        "switching": "shifting",
        "shift": "shifting",
        "cross category exploration": "cross category exploration",
        "collection completion": "collection expansion",
        "preference deepening": "deepening",
        "preference refinement": "refinement",
        "repeat": "repetition",
    }
    return aliases.get(s, s)


def normalize_scope(x: Any) -> str:
    s = normalize_label(x)
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
    }
    return aliases.get(s, s)


def normalize_direction(x: Any) -> str:
    s = normalize_label(x)
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


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 1e-12, n, 1.0)


def extract_json_value(text: str) -> Any:
    """Recover the first complete JSON object/array from model output."""
    s = str(text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    starts = []
    for opener, closer in [("{", "}"), ("[", "]")]:
        pos = s.find(opener)
        if pos >= 0:
            starts.append((pos, opener, closer))

    if not starts:
        raise ValueError("No JSON value found")

    start, opener, closer = min(starts, key=lambda z: z[0])
    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])

    raise ValueError("Unbalanced JSON")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                x = json.loads(line)
            except Exception:
                continue
            if isinstance(x, dict):
                rows.append(x)
    return rows


def processed_users_from_jsonl(path: str | Path) -> set[str]:
    out: set[str] = set()
    for row in read_jsonl(path):
        if (
            row.get("schema_version") == SCHEMA_VERSION
            and row.get("rank_prompt_version") == RANK_PROMPT_VERSION
            and row.get("user_id") is not None
        ):
            out.add(str(row["user_id"]))
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
            return [str(x) for x in obj.keys()]
    except Exception:
        pass
    return [x.strip() for x in text.splitlines() if x.strip()]


# =============================================================================
# Data / items / candidates
# =============================================================================

def normalize_items(items: Dict[Any, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in items.items()}


def get_item_info(
    item_id: Any,
    items_meta: Dict[str, Any],
) -> Dict[str, str]:
    iid = str(item_id)
    info = items_meta.get(iid, {})

    category = (
        info.get("main_cat")
        or info.get("category")
        or info.get("categories")
        or "Unknown"
    )
    if isinstance(category, list):
        if category and isinstance(category[0], list):
            category = " > ".join(
                str(v)
                for v in category[0][:5]
            )
        else:
            category = " > ".join(
                str(v)
                for v in category[:5]
            )

    title = (
        info.get("title")
        or info.get("name")
        or f"Item {iid}"
    )

    return {
        "item_id": iid,
        "title": str(title),
        "category": str(category),
    }


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
                uid = str(
                    row.get("user_id")
                    or row.get("uid")
                )
                rows[uid] = row
        return rows

    obj = load_json(p)
    if isinstance(obj, list):
        return {
            str(
                row.get("user_id")
                or row.get("uid")
            ): row
            for row in obj
            if isinstance(row, dict)
        }
    if isinstance(obj, dict):
        # Could already be user_id -> row.
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in obj.items():
            if isinstance(v, dict):
                out[str(k)] = v
        return out

    raise ValueError(
        f"Unsupported candidate file format: {path}"
    )


def as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if not isinstance(x, list):
        x = [x]
    return [str(v) for v in x]


def get_candidates_for_user(
    uid: str,
    user_data: Dict[str, Any],
    negative_data: Dict[str, Any],
    candidate_rows: Optional[Dict[str, Dict[str, Any]]],
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Default path EXACTLY matches current Oracle/native candidate construction:
        test_ids + test_neg
        random.Random(f"{seed}:{uid}").shuffle(...)
    """
    if (
        candidate_rows is not None
        and uid in candidate_rows
    ):
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
                    "ground_truth",
                    row.get(
                        "ground_truth_item_ids",
                        user_data.get("test", []),
                    ),
                ),
            )
        )
        if not candidates:
            raise ValueError(
                f"user={uid}: empty candidate file row"
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
# Dual behavior cache
# =============================================================================

def load_precomputed_behaviors(
    path: str,
) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            if row.get("precompute_ok") is not True:
                continue

            uid = str(row["user_id"])
            behaviors = row.get(
                "generated_behaviors",
                [],
            )
            if not isinstance(behaviors, list):
                continue

            # Ensure chronological order.
            behaviors = sorted(
                [
                    dict(x)
                    for x in behaviors
                    if isinstance(x, dict)
                ],
                key=lambda x: int(
                    x.get("window_index", 0)
                ),
            )

            if not behaviors:
                continue

            row = dict(row)
            row["generated_behaviors"] = behaviors
            rows[uid] = row

    if not rows:
        raise ValueError(
            f"No successful users in behavior cache: {path}"
        )

    return rows


def validate_behavior_cache_history(
    uid: str,
    cache_row: Dict[str, Any],
    user_data: Dict[str, Any],
    max_train_interactions: int,
    strict: bool,
) -> None:
    """
    Validate that the precomputed behavior cache belongs to the same user history.

    IMPORTANT:
    The cache is authoritative about how many train interactions were used during
    precompute. Inference may use a different --history-size or even a different
    --max-train-interactions for diagnostics; that alone must NOT invalidate an
    otherwise correct cache.

    Validation order:
      1) Compare with the cache's own max_train_interactions, when recorded.
      2) Otherwise compare against a suffix of the current train history having
         exactly len(cached) items.
      3) If neither matches, raise/warn with useful diagnostics.
    """
    cached = [
        str(x)
        for x in cache_row.get(
            "train_history_item_ids",
            [],
        )
    ]

    current_all = [
        str(x)
        for x in list(
            user_data.get("train", [])
        )
    ]

    if not cached:
        msg = (
            f"user={uid}: precomputed behavior row has no "
            "train_history_item_ids; cannot verify cache/history identity"
        )
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}", flush=True)
        return

    cache_limit_raw = cache_row.get(
        "max_train_interactions",
        None,
    )

    cache_limit: Optional[int] = None
    try:
        if cache_limit_raw is not None:
            cache_limit = int(
                cache_limit_raw
            )
            if cache_limit <= 0:
                cache_limit = None
    except Exception:
        cache_limit = None

    # Primary comparison: reproduce exactly the precompute slicing rule.
    if cache_limit is not None:
        expected_from_cache_config = (
            current_all[
                -cache_limit:
            ]
        )
        if cached == expected_from_cache_config:
            return

    # Robust fallback: the cached IDs themselves tell us exactly how many
    # interactions were used. This also handles older caches lacking metadata.
    expected_same_length = (
        current_all[
            -len(cached):
        ]
        if cached
        else []
    )

    if cached == expected_same_length:
        return

    # Diagnostic information for a genuine mismatch.
    first_diff = None
    for i, (a, b) in enumerate(
        zip(
            cached,
            expected_same_length,
        )
    ):
        if a != b:
            first_diff = {
                "index": int(i),
                "cached": a,
                "current": b,
            }
            break

    if (
        first_diff is None
        and len(cached)
        != len(expected_same_length)
    ):
        first_diff = {
            "index": min(
                len(cached),
                len(expected_same_length),
            ),
            "cached": (
                cached[
                    len(expected_same_length)
                ]
                if len(cached)
                > len(expected_same_length)
                else None
            ),
            "current": (
                expected_same_length[
                    len(cached)
                ]
                if len(expected_same_length)
                > len(cached)
                else None
            ),
        }

    msg = (
        f"user={uid}: behavior cache history truly mismatches current sequence; "
        f"cached_n={len(cached)}, current_train_n={len(current_all)}, "
        f"cache_max_train_interactions={cache_limit_raw!r}, "
        f"inference_max_train_interactions={max_train_interactions}, "
        f"first_diff={first_diff}, "
        f"cached_tail={cached[-3:]}, "
        f"current_expected_tail={expected_same_length[-3:]}. "
        "This usually means the cache and --sequences file were produced from "
        "different dataset/split versions."
    )

    if strict:
        raise ValueError(msg)

    print(
        f"[WARN] {msg}",
        flush=True,
    )


def latest_behavior_view(
    behaviors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    WHAT block for ranking.

    Deliberately use the TARGET USER'S latest semantic evidence only.
    Do not use semantic examples from a cluster or another user.
    """
    b = behaviors[-1]

    semantic_focus = b.get(
        "semantic_focus",
        b.get("keywords", []),
    )
    if not isinstance(semantic_focus, list):
        semantic_focus = (
            [str(semantic_focus)]
            if semantic_focus
            else []
        )

    return {
        "recommendation_behavior": clean_text(
            b.get(
                "recommendation_behavior",
                b.get(
                    "pattern_description",
                    "",
                ),
            )
        ),
        "semantic_focus": [
            clean_text(x)
            for x in semantic_focus
            if clean_text(x)
        ][:8],
        # Audit only; ranking prompt mainly uses the semantic fields above.
        "observed_trajectory_signature": clean_text(
            b.get(
                "trajectory_signature",
                b.get(
                    "behavior_signature",
                    "",
                ),
            )
        ),
        "window_index": int(
            b.get("window_index", 0)
        ),
    }


# =============================================================================
# Qwen encoder + frozen KMeans state mapping
# =============================================================================

class QwenEncoder:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.model_name = str(model_name)
        self.device = str(device)
        self.batch_size = int(batch_size)

        print(
            f"[INFO] Loading state encoder: "
            f"{self.model_name} on {self.device}"
        )

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

    def encode(
        self,
        texts: List[str],
    ) -> np.ndarray:
        if not texts:
            return np.zeros(
                (0, 1),
                dtype=np.float32,
            )
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(
            emb,
            dtype=np.float32,
        )

    def release(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ClusterStateVocabulary:
    """
    Frozen cluster vocabulary produced by build_reverse_behavior_tree_dual.py.
    """

    def __init__(
        self,
        states_json: str,
        state_embeddings_npy: str,
        embedding_model: Optional[str],
        embedding_device: str,
        embedding_batch_size: int,
        cluster_text_field: Optional[str],
        min_similarity: Optional[float],
        min_margin: Optional[float],
        respect_mapping_filter: bool,
    ) -> None:
        payload = load_json(states_json)

        if not (
            isinstance(payload, dict)
            and isinstance(
                payload.get("states"),
                list,
            )
        ):
            raise ValueError(
                "behavior_states.json must contain {metadata, states}"
            )

        self.metadata = payload.get(
            "metadata",
            {},
        )
        self.rows = [
            dict(x)
            for x in payload["states"]
        ]

        # Builder writes states sorted by cluster_id and centroid rows in that
        # same cluster order. Sort explicitly here for safety.
        self.rows.sort(
            key=lambda x: int(
                x.get(
                    "cluster_id",
                    str(
                        x.get(
                            "state_id",
                            "C0",
                        )
                    ).lstrip("C") or 0,
                )
            )
        )

        self.state_by_id = {
            str(r["state_id"]): r
            for r in self.rows
        }

        centroids = np.asarray(
            np.load(state_embeddings_npy),
            dtype=np.float32,
        )
        if len(centroids) != len(self.rows):
            raise ValueError(
                f"centroid/state mismatch: "
                f"{len(centroids)} != {len(self.rows)}"
            )
        self.centroids = l2_normalize(
            centroids
        )

        self.embedding_model_name = str(
            embedding_model
            or self.metadata.get(
                "embedding_model"
            )
            or "Qwen/Qwen3-Embedding-0.6B"
        )
        self.cluster_text_field = str(
            cluster_text_field
            or self.metadata.get(
                "cluster_text_field"
            )
            or "structured_combined"
        )

        quality = (
            self.metadata
            .get("cluster_quality", {})
            .get("assignment_quality", {})
        )

        self.min_similarity = float(
            min_similarity
            if min_similarity is not None
            else quality.get(
                "min_cluster_similarity",
                0.55,
            )
        )
        self.min_margin = float(
            min_margin
            if min_margin is not None
            else quality.get(
                "min_cluster_margin",
                0.02,
            )
        )
        self.respect_mapping_filter = bool(
            respect_mapping_filter
        )

        self.constraint_level = str(
            self.metadata.get(
                "constraint_level",
                "mechanism",
            )
        )

        self.state_mechanisms = [
            normalize_mechanism(
                r.get("mechanism")
            )
            for r in self.rows
        ]

        self.encoder = QwenEncoder(
            self.embedding_model_name,
            embedding_device,
            embedding_batch_size,
        )

    def _trajectory_text(
        self,
        behavior: Dict[str, Any],
    ) -> str:
        """
        MUST mirror choose_cluster_text() in the dual Tree builder.
        Semantic recommendation fields NEVER enter this text.
        """
        mechanism = normalize_mechanism(
            behavior.get("mechanism")
        )
        scope = normalize_scope(
            behavior.get("scope")
        )
        direction = normalize_direction(
            behavior.get("direction")
        )
        trajectory = clean_text(
            behavior.get(
                "trajectory_signature",
                behavior.get(
                    "behavior_signature",
                    "",
                ),
            )
        )

        if self.cluster_text_field in {
            "trajectory_signature",
            "behavior_signature",
        }:
            text = trajectory
        elif (
            self.cluster_text_field
            == "structured_combined"
        ):
            text = (
                f"mechanism={mechanism}; "
                f"scope={scope}; "
                f"direction={direction}; "
                f"trajectory={trajectory}"
            )
        else:
            raise ValueError(
                f"Unsupported cluster_text_field="
                f"{self.cluster_text_field!r}; "
                "this inference expects the dual Tree builder."
            )

        if not clean_text(text):
            raise ValueError(
                "Empty trajectory text for state mapping"
            )
        return clean_text(text)

    def state_evidence(
        self,
        sid: str,
    ) -> Dict[str, Any]:
        """
        HOW block.

        Never expose representative_pattern_description, top_semantic_focus, or
        recommendation_behavior_examples from other cluster members.
        """
        r = self.state_by_id.get(
            str(sid),
            {},
        )

        trajectory = clean_text(
            r.get(
                "canonical_behavior_signature",
                r.get(
                    "representative_signature",
                    "",
                ),
            )
        )
        if not trajectory:
            trajectory = clean_text(
                r.get("canonical_text")
            )

        return {
            "state_id": str(sid),
            "mechanism": normalize_mechanism(
                r.get("mechanism")
            ),
            "scope": normalize_scope(
                r.get("scope")
            ),
            "trajectory_signature": trajectory,
            # Audit only. Not required by the final ranking prompt.
            "direction": normalize_direction(
                r.get("direction")
            ),
        }

    def map_users(
        self,
        behaviors_by_user: Dict[
            str,
            List[Dict[str, Any]],
        ],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Encode all test behaviors in one pass, then map each behavior to the
        nearest compatible frozen centroid.
        """
        flat: List[Tuple[str, int, Dict[str, Any]]] = []
        texts: List[str] = []

        for uid, behaviors in behaviors_by_user.items():
            for i, b in enumerate(behaviors):
                flat.append(
                    (uid, i, b)
                )
                texts.append(
                    self._trajectory_text(b)
                )

        embeddings = self.encoder.encode(
            texts
        )
        all_scores = (
            embeddings
            @ self.centroids.T
        )

        out: Dict[
            str,
            List[Dict[str, Any]],
        ] = defaultdict(list)

        all_indices = list(
            range(len(self.rows))
        )

        for row_idx, (
            uid,
            behavior_index,
            behavior,
        ) in enumerate(flat):
            mechanism = normalize_mechanism(
                behavior.get("mechanism")
            )

            compatible = [
                i
                for i, m in enumerate(
                    self.state_mechanisms
                )
                if m == mechanism
            ]

            # Current Tree builder defaults to mechanism-constrained KMeans.
            # If an unseen/unknown mechanism has no bucket, use global fallback.
            candidate_indices = (
                compatible
                if compatible
                else all_indices
            )
            retrieval_scope = (
                "same_mechanism_centroids"
                if compatible
                else "global_centroid_fallback"
            )

            scores = all_scores[
                row_idx,
                candidate_indices,
            ]
            order_local = np.argsort(
                -scores
            )

            best_local = int(
                order_local[0]
            )
            best_idx = int(
                candidate_indices[
                    best_local
                ]
            )
            best_sim = float(
                scores[best_local]
            )

            second_sim: Optional[float] = None
            second_idx: Optional[int] = None

            if len(order_local) > 1:
                second_local = int(
                    order_local[1]
                )
                second_idx = int(
                    candidate_indices[
                        second_local
                    ]
                )
                second_sim = float(
                    scores[second_local]
                )

            margin = (
                best_sim - second_sim
                if second_sim is not None
                else None
            )

            accepted = (
                best_sim
                >= self.min_similarity
            )
            reasons: List[str] = []

            if not compatible:
                accepted = False
                reasons.append(
                    "missing_compatible_mechanism_bucket"
                )

            if (
                best_sim
                < self.min_similarity
            ):
                accepted = False
                reasons.append(
                    "low_similarity"
                )

            if (
                len(candidate_indices) > 1
                and margin is not None
                and margin < self.min_margin
            ):
                accepted = False
                reasons.append(
                    "low_margin"
                )

            if not self.respect_mapping_filter:
                accepted = True
                reasons = []

            sid = str(
                self.rows[
                    best_idx
                ]["state_id"]
            )

            out[uid].append({
                "behavior_index": int(
                    behavior_index
                ),
                "window_index": int(
                    behavior.get(
                        "window_index",
                        behavior_index,
                    )
                ),
                "state_id": sid,
                "state_evidence": (
                    self.state_evidence(sid)
                ),
                "behavior_mechanism": (
                    mechanism
                ),
                "behavior_scope": (
                    normalize_scope(
                        behavior.get(
                            "scope"
                        )
                    )
                ),
                "behavior_direction": (
                    normalize_direction(
                        behavior.get(
                            "direction"
                        )
                    )
                ),
                "behavior_trajectory_signature": (
                    clean_text(
                        behavior.get(
                            "trajectory_signature",
                            behavior.get(
                                "behavior_signature",
                                "",
                            ),
                        )
                    )
                ),
                "cluster_text": texts[
                    row_idx
                ],
                "retrieval_scope": (
                    retrieval_scope
                ),
                "best_similarity": (
                    best_sim
                ),
                "second_similarity": (
                    second_sim
                ),
                "similarity_margin": (
                    margin
                ),
                "num_compatible_states": len(
                    candidate_indices
                ),
                "assignment_accepted": bool(
                    accepted
                ),
                "rejection_reason": (
                    "+".join(reasons)
                    if reasons
                    else None
                ),
                "second_state_id": (
                    str(
                        self.rows[
                            second_idx
                        ]["state_id"]
                    )
                    if second_idx
                    is not None
                    else None
                ),
            })

        # Preserve chronological order.
        for uid in out:
            out[uid].sort(
                key=lambda x: (
                    int(
                        x.get(
                            "window_index",
                            0,
                        )
                    ),
                    int(
                        x.get(
                            "behavior_index",
                            0,
                        )
                    ),
                )
            )

        return dict(out)

    def release_encoder(self) -> None:
        self.encoder.release()


def latest_accepted_state_suffix(
    mappings: List[Dict[str, Any]],
) -> List[str]:
    """
    Reproduce the current Tree training semantics:
    a rejected/ambiguous mapping breaks the sequence.
    """
    suffix: List[str] = []
    for m in mappings:
        if m.get(
            "assignment_accepted"
        ):
            suffix.append(
                str(m["state_id"])
            )
        else:
            suffix = []
    return suffix


# =============================================================================
# Reverse suffix Tree
# =============================================================================

class ReverseBehaviorTree:
    def __init__(
        self,
        tree_json: str,
        vocab: ClusterStateVocabulary,
    ) -> None:
        payload = load_json(tree_json)
        self.payload = payload
        self.nodes: Dict[
            str,
            Dict[str, Any],
        ] = payload["nodes"]
        self.root_id = str(
            payload.get(
                "root_id",
                "ROOT",
            )
        )
        self.vocab = vocab
        self.max_order = int(
            payload.get(
                "metadata",
                {},
            ).get(
                "max_order",
                5,
            )
        )

        if self.root_id not in self.nodes:
            raise ValueError(
                f"Tree root {self.root_id!r} missing"
            )

    def query(
        self,
        state_sequence: List[str],
        top_next: int,
    ) -> Dict[str, Any]:
        recent = list(
            state_sequence[
                -self.max_order:
            ]
        )

        cur = self.root_id
        deepest_structural = (
            self.root_id
        )
        deepest_active = (
            self.root_id
            if self.nodes[
                self.root_id
            ].get(
                "active_predictive",
                True,
            )
            else None
        )

        traversed_reverse: List[str] = []
        stop_reason = "history_exhausted"

        for token in reversed(recent):
            children = (
                self.nodes[cur]
                .get(
                    "children",
                    {},
                )
            )
            if token not in children:
                stop_reason = (
                    "missing_reverse_edge"
                )
                break

            cur = str(
                children[token]
            )
            deepest_structural = cur
            traversed_reverse.append(
                token
            )

            if self.nodes[
                cur
            ].get(
                "active_predictive",
                False,
            ):
                deepest_active = cur

        if deepest_active is None:
            deepest_active = (
                self.root_id
            )

        structural_node = self.nodes[
            deepest_structural
        ]
        active_node = self.nodes[
            deepest_active
        ]

        dist = (
            active_node.get(
                "smoothed_next_distribution"
            )
            or active_node.get(
                "raw_next_distribution"
            )
            or []
        )

        preds = []
        for x in dist[
            :max(
                0,
                int(top_next),
            )
        ]:
            sid = str(
                x.get("state_id")
            )
            ev = (
                self.vocab
                .state_evidence(sid)
            )
            preds.append({
                **ev,
                "probability": float(
                    x.get(
                        "probability",
                        0.0,
                    )
                ),
                "support_user_count": int(
                    x.get(
                        "support_user_count",
                        0,
                    )
                ),
                "occurrence_count": int(
                    x.get(
                        "occurrence_count",
                        0,
                    )
                ),
            })

        return {
            "query_state_sequence": (
                state_sequence
            ),
            "query_recent_suffix": (
                recent
            ),
            "reverse_tokens_traversed": (
                traversed_reverse
            ),
            "stop_reason": (
                stop_reason
            ),
            "deepest_structural_node_id": (
                deepest_structural
            ),
            "deepest_structural_order": int(
                structural_node.get(
                    "context_order",
                    0,
                )
            ),
            "matched_node_id": (
                deepest_active
            ),
            "matched_context_state_ids": (
                active_node.get(
                    "context_states",
                    [],
                )
            ),
            "matched_order": int(
                active_node.get(
                    "context_order",
                    0,
                )
            ),
            "matched_support_users": int(
                active_node.get(
                    "support_user_count",
                    0,
                )
            ),
            "matched_support_occurrences": int(
                active_node.get(
                    "support_occurrence_count",
                    0,
                )
            ),
            "activation_reason": (
                active_node.get(
                    "activation_reason",
                    "",
                )
            ),
            "smoothing_lambda": float(
                active_node.get(
                    "smoothing_lambda",
                    1.0,
                )
            ),
            "predicted_next_states": (
                preds
            ),
        }


def build_tree_rank_evidence(
    latest_behavior: Dict[str, Any],
    tree_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Only target-user semantics + abstract collaborative transition.
    """
    predicted = []

    for rank, p in enumerate(
        tree_result.get(
            "predicted_next_states",
            [],
        ),
        start=1,
    ):
        predicted.append({
            "rank": rank,
            "state_id": p.get(
                "state_id"
            ),
            "probability": round(
                float(
                    p.get(
                        "probability",
                        0.0,
                    )
                ),
                6,
            ),
            "mechanism": p.get(
                "mechanism"
            ),
            "scope": p.get(
                "scope"
            ),
            "trajectory_signature": (
                p.get(
                    "trajectory_signature"
                )
            ),
            "support_user_count": int(
                p.get(
                    "support_user_count",
                    0,
                )
            ),
        })

    return {
        "latest_observed_behavior": {
            "recommendation_behavior": (
                latest_behavior.get(
                    "recommendation_behavior",
                    "",
                )
            ),
            "semantic_focus": (
                latest_behavior.get(
                    "semantic_focus",
                    [],
                )
            ),
        },
        "predicted_next_transition": (
            predicted
        ),
        "tree_context": {
            "matched_order": int(
                tree_result.get(
                    "matched_order",
                    0,
                )
            ),
            "support_users": int(
                tree_result.get(
                    "matched_support_users",
                    0,
                )
            ),
        },
    }


# =============================================================================
# Ranking prompt
# =============================================================================

def format_history(
    items: List[Dict[str, str]],
) -> str:
    return "\n".join(
        f"{i}. {x['title']} | {x['category']}"
        for i, x in enumerate(
            items,
            start=1,
        )
    )


def format_candidates(
    items: List[Dict[str, str]],
) -> str:
    return "\n".join(
        f"[{i}] {x['item_id']} | "
        f"{x['title']} | {x['category']}"
        for i, x in enumerate(
            items,
            start=1,
        )
    )


def build_rank_prompt(
    *,
    history_items: List[Dict[str, str]],
    candidate_items: List[Dict[str, str]],
    latest_behavior: Optional[Dict[str, Any]],
    predicted_transition: Optional[
        List[Dict[str, Any]]
    ],
) -> str:
    """
    Video-Games-specific ranking prompt.

    Controlled comparison is preserved:
      Native      = history + candidates
      Latest-only = history + latest target-user behavior + candidates
      Tree        = history + latest target-user behavior
                    + predicted abstract transition + candidates

    Only the interpretation instructions are specialized for Video Games.
    No extra metadata or test information is introduced.
    """
    ids = [
        str(x["item_id"])
        for x in candidate_items
    ]

    latest_block = (
        "Not provided."
    )
    if latest_behavior is not None:
        latest_block = json.dumps(
            {
                "recommendation_behavior": (
                    latest_behavior.get(
                        "recommendation_behavior",
                        "",
                    )
                ),
                "semantic_focus": (
                    latest_behavior.get(
                        "semantic_focus",
                        [],
                    )
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

    transition_block = (
        "Not provided."
    )
    if predicted_transition:
        transition_block = json.dumps(
            predicted_transition,
            ensure_ascii=False,
            indent=2,
        )

    return f"""You are ranking VIDEO GAME candidate items for the user's NEXT choice.

AVAILABLE SIGNALS

1) RECENT GAME HISTORY
These are the user's observed game interactions, ordered from older to newer.

{format_history(history_items)}

2) LATEST OBSERVED BEHAVIOR — WHAT the user currently prefers
{latest_block}

3) PREDICTED NEXT TRANSITION — HOW the current preference is expected to evolve
{transition_block}

VIDEO-GAME INTERPRETATION RULES

- Use only information explicitly visible in the recent history, latest behavior,
  predicted transition, candidate title, and candidate category.
- The LATEST OBSERVED BEHAVIOR is the semantic anchor. First determine the user's
  current game preference from it and from the most recent history.
- The PREDICTED NEXT TRANSITION is an ABSTRACT behavioral change. It does NOT name
  the next game directly. Interpret it RELATIVE TO the user's current game preference
  before comparing candidates.
- Do NOT match candidates directly to abstract words such as "deepening",
  "collection expansion", "persistence", or "adjacent exploration".
- Do NOT import semantic content from another user or from a cluster representative.
- Candidate input order is RANDOM and has no ranking meaning.

HOW TO INTERPRET COMMON TRANSITIONS FOR VIDEO GAMES

- deepening:
  Prefer candidates that continue the user's current game interest more specifically,
  such as a closely related title, franchise/series continuation, closely related
  subgenre, gameplay style, theme, or category WHEN such evidence is visible in the
  supplied titles/categories.

- collection expansion:
  Prefer candidates that extend an already observed game series, franchise, edition,
  closely related collection, or strongly connected set of titles WHEN that relation
  is supported by the supplied titles/categories.

- persistence:
  Prefer candidates that remain close to the user's established recent game preference
  rather than introducing a large semantic shift.

- adjacent exploration:
  Prefer candidates that are near the current preference but introduce a plausible
  neighboring game category, theme, style, or related title. The move should be
  adjacent, not arbitrary.

- If another mechanism is provided, interpret it analogously as a change RELATIVE TO
  the user's current game preference, not as a standalone content label.

- Scope constrains how far the transition should move:
  * same collection/series -> stay very close to a visible series/franchise relation;
  * same creator -> use only if creator information is actually visible;
  * same subcategory -> remain within a narrow current game type/category;
  * same category -> remain within the broader current game category;
  * adjacent category -> allow a nearby category shift;
  * mixed/unknown -> rely more strongly on the target user's recent semantic evidence.

- If multiple predicted transitions are provided, earlier entries have higher
  collaborative probability and should receive more weight.
- A high-probability transition is still only auxiliary evidence. If it conflicts with
  strong target-user evidence, prioritize the target user's own recent preference.
- Never invent a franchise, platform, genre, gameplay property, developer, or creator
  that is not inferable from the provided title/category text.

CANDIDATE ITEMS

{format_candidates(candidate_items)}

DECISION PROCEDURE

1. Infer the target user's current GAME preference from recent history and, when
   provided, the latest observed behavior.
2. If a predicted transition is provided, translate that abstract transition into the
   expected NEXT preference relative to the current game preference.
3. For each candidate, judge how well its visible title/category realizes that expected
   next preference.
4. Give the predicted transition meaningful weight only when it helps discriminate
   candidates supported by the user's own preference evidence.
5. Rank all remaining candidates by decreasing compatibility with the target user's
   recent game preference.

OUTPUT REQUIREMENTS

- Rank ALL {len(ids)} candidate IDs exactly once.
- Use every candidate ID exactly once.
- Do not invent IDs.
- Do not omit IDs.
- Return JSON only.

{{"ranked_item_ids":["item_id_1","item_id_2","..."],"reasoning":"one concise sentence"}}
"""

def sanitize_ranking(
    predicted: Any,
    valid_ids: Sequence[str],
) -> Tuple[List[str], bool, int]:
    valid = [
        str(x)
        for x in valid_ids
    ]
    valid_set = set(valid)

    if not isinstance(
        predicted,
        list,
    ):
        predicted = []

    out: List[str] = []
    seen = set()

    for x in predicted:
        iid = str(x)
        if (
            iid in valid_set
            and iid not in seen
        ):
            out.append(iid)
            seen.add(iid)

    model_returned = len(out)
    complete = (
        model_returned
        == len(valid)
    )

    # Fallback is retained ONLY so every row has a usable ranking.
    # parse_complete=False is propagated and strict metrics exclude the row.
    for iid in valid:
        if iid not in seen:
            out.append(iid)
            seen.add(iid)

    return (
        out,
        complete,
        model_returned,
    )


# =============================================================================
# Batched local Gemma ranker
# =============================================================================

@dataclass
class CallStat:
    call_type: str
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    parse_ok: bool
    complete_ranking: bool


class LocalGemmaRanker:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
        rank_max_new_tokens: int,
        temperature: float,
        seed: int,
    ) -> None:
        from unsloth import FastModel
        from unsloth.chat_templates import (
            get_chat_template,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Local Unsloth Gemma ranking expects CUDA."
            )

        print(
            f"[INFO] Loading ranking model: "
            f"{model_name}",
            flush=True,
        )
        print(
            "[INFO] Loading Gemma model weights/tokenizer; "
            "ranking starts immediately after this step.",
            flush=True,
        )

        self.model, self.tokenizer = (
            FastModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                full_finetuning=False,
            )
        )
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="gemma3",
        )
        self.tokenizer.padding_side = (
            "left"
        )

        if (
            self.tokenizer.pad_token_id
            is None
        ):
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )

        self.model.eval()
        self.device = next(
            self.model.parameters()
        ).device

        print(
            f"[INFO] Gemma loaded on {self.device}. "
            "Starting ranking batches.",
            flush=True,
        )
        self.max_seq_length = int(
            max_seq_length
        )
        self.rank_max_new_tokens = int(
            rank_max_new_tokens
        )
        self.temperature = float(
            temperature
        )
        self.calls: List[CallStat] = []

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

    def _render(
        self,
        prompt: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a recommendation ranking system. "
                    "Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        if text.startswith("<bos>"):
            text = text[len("<bos>"):]

        return text

    def _generate_batch_raw(
        self,
        prompts: List[str],
        call_type: str,
    ) -> List[str]:
        if not prompts:
            return []

        rendered = [
            self._render(p)
            for p in prompts
        ]

        batch = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)

        input_width = int(
            batch["input_ids"].shape[1]
        )

        kwargs: Dict[str, Any] = {
            "max_new_tokens": (
                self.rank_max_new_tokens
            ),
            "use_cache": True,
            "pad_token_id": (
                self.tokenizer.pad_token_id
            ),
            "eos_token_id": (
                self.tokenizer.eos_token_id
            ),
        }

        if self.temperature > 0:
            kwargs.update({
                "do_sample": True,
                "temperature": (
                    self.temperature
                ),
                "top_p": 0.95,
            })
        else:
            kwargs["do_sample"] = False

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model.generate(
                **batch,
                **kwargs,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (
            time.perf_counter()
            - t0
        )

        generated = outputs[
            :,
            input_width:,
        ]

        raws = (
            self.tokenizer
            .batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

        # Detailed parse/complete stats are added later.
        per_sample_time = (
            elapsed
            / max(
                1,
                len(prompts),
            )
        )

        for gen_row in generated:
            self.calls.append(
                CallStat(
                    call_type=call_type,
                    input_tokens=input_width,
                    output_tokens=int(
                        gen_row.shape[0]
                    ),
                    elapsed_sec=float(
                        per_sample_time
                    ),
                    parse_ok=False,
                    complete_ranking=False,
                )
            )

        return [
            str(x).strip()
            for x in raws
        ]

    def _parse_rank(
        self,
        raw: str,
        valid_ids: List[str],
    ) -> Tuple[
        List[str],
        Dict[str, Any],
    ]:
        parse_ok = False
        reasoning = ""
        obj: Any = None

        try:
            obj = extract_json_value(
                raw
            )
            parse_ok = isinstance(
                obj,
                dict,
            )
        except Exception:
            obj = None

        if isinstance(
            obj,
            dict,
        ):
            (
                ranking,
                complete,
                returned_n,
            ) = sanitize_ranking(
                obj.get(
                    "ranked_item_ids",
                    [],
                ),
                valid_ids,
            )
            reasoning = clean_text(
                obj.get(
                    "reasoning",
                    "",
                )
            )
        else:
            (
                ranking,
                complete,
                returned_n,
            ) = sanitize_ranking(
                [],
                valid_ids,
            )

        return ranking, {
            "parse_ok": bool(
                parse_ok
            ),
            "parse_complete": bool(
                parse_ok and complete
            ),
            "model_returned_valid_ids": int(
                returned_n
            ),
            "num_candidates": len(
                valid_ids
            ),
            "reasoning": reasoning,
            "raw_output": raw,
        }

    def rank_jobs(
        self,
        jobs: List[Dict[str, Any]],
        batch_size: int,
        call_type: str,
        max_retries: int,
    ) -> List[
        Tuple[
            List[str],
            Dict[str, Any],
        ]
    ]:
        if not jobs:
            return []

        final: List[
            Optional[
                Tuple[
                    List[str],
                    Dict[str, Any],
                ]
            ]
        ] = [
            None
            for _ in jobs
        ]

        batch_size = max(
            1,
            int(batch_size),
        )

        batch_starts = list(
            range(
                0,
                len(jobs),
                batch_size,
            )
        )

        pbar = tqdm(
            batch_starts,
            desc=call_type,
            unit="batch",
            dynamic_ncols=True,
        )

        for start in pbar:
            end = min(
                start + batch_size,
                len(jobs),
            )
            pbar.set_postfix(
                users=f"{start + 1}-{end}/{len(jobs)}",
                refresh=False,
            )
            chunk = jobs[
                start:end
            ]

            prompts = [
                build_rank_prompt(
                    history_items=j[
                        "history_items"
                    ],
                    candidate_items=j[
                        "candidate_items"
                    ],
                    latest_behavior=j.get(
                        "latest_behavior"
                    ),
                    predicted_transition=j.get(
                        "predicted_transition"
                    ),
                )
                for j in chunk
            ]

            try:
                raws = (
                    self._generate_batch_raw(
                        prompts,
                        call_type,
                    )
                )
            except Exception as e:
                print(
                    f"[WARN] {call_type} "
                    f"batch {start}:{end} failed: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                raws = [
                    ""
                    for _ in chunk
                ]

            for local_idx, (
                job,
                prompt,
                raw,
            ) in enumerate(
                zip(
                    chunk,
                    prompts,
                    raws,
                )
            ):
                global_idx = (
                    start
                    + local_idx
                )
                valid_ids = [
                    str(
                        x["item_id"]
                    )
                    for x in job[
                        "candidate_items"
                    ]
                ]

                ranking, meta = (
                    self._parse_rank(
                        raw,
                        valid_ids,
                    )
                )

                # No individual retry: keep the first batched generation result.
                # If the model omitted/duplicated IDs, sanitize_ranking() has already
                # appended missing valid IDs in original candidate order, matching
                # the native baseline behavior.
                attempts = 1
                meta["attempts"] = attempts
                meta["call_type"] = (
                    call_type
                )
                final[
                    global_idx
                ] = (
                    ranking,
                    meta,
                )

        pbar.close()

        if any(
            x is None
            for x in final
        ):
            raise RuntimeError(
                "Internal ranking result missing"
            )

        return [
            x
            for x in final
            if x is not None
        ]

    def stats(self) -> Dict[str, Any]:
        if not self.calls:
            return {
                "model": None,
                "generation_records": 0,
            }

        by_type = Counter(
            x.call_type
            for x in self.calls
        )

        return {
            "model": str(
                getattr(
                    self.model,
                    "name_or_path",
                    "local_gemma",
                )
            ),
            "generation_records": len(
                self.calls
            ),
            "input_tokens_approx": int(
                sum(
                    x.input_tokens
                    for x in self.calls
                )
            ),
            "output_tokens_approx": int(
                sum(
                    x.output_tokens
                    for x in self.calls
                )
            ),
            "elapsed_sec": float(
                sum(
                    x.elapsed_sec
                    for x in self.calls
                )
            ),
            "call_type_counts": dict(
                by_type
            ),
        }


# =============================================================================
# Metrics
# =============================================================================

def target_rank(
    ranking: Sequence[str],
    target: str,
) -> int:
    ranking = [
        str(x)
        for x in ranking
    ]
    try:
        return (
            ranking.index(
                str(target)
            )
            + 1
        )
    except ValueError:
        return len(
            ranking
        ) + 1


def metrics_from_ranks(
    ranks: List[int],
    ks: Sequence[int] = (
        1,
        3,
        5,
        10,
    ),
) -> Optional[Dict[str, float]]:
    if not ranks:
        return None

    n = len(ranks)
    out: Dict[str, float] = {}

    for k in ks:
        out[f"Hit@{k}"] = float(
            sum(
                r <= k
                for r in ranks
            )
            / n
        )
        out[f"NDCG@{k}"] = float(
            sum(
                (
                    1.0
                    / math.log2(
                        r + 1
                    )
                )
                if r <= k
                else 0.0
                for r in ranks
            )
            / n
        )

    out["MRR"] = float(
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
    out["NumUsers"] = int(n)
    return out


def aggregate_mode(
    rows: List[Dict[str, Any]],
    mode: str,
    strict_complete: bool,
) -> Optional[Dict[str, float]]:
    ranks: List[int] = []

    for row in rows:
        rec = row.get(
            "rankings",
            {},
        ).get(
            mode
        )
        if not isinstance(
            rec,
            dict,
        ):
            continue

        if (
            strict_complete
            and not rec.get(
                "parse_complete"
            )
        ):
            continue

        rank = rec.get(
            "target_rank"
        )
        if rank is not None:
            ranks.append(
                int(rank)
            )

    return metrics_from_ranks(
        ranks
    )


def metric_gain(
    a: Optional[Dict[str, float]],
    b: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    if not a or not b:
        return None

    out: Dict[str, float] = {}

    for k in a:
        if (
            k not in b
            or k == "NumUsers"
        ):
            continue

        if k in {
            "MeanRank",
            "MedianRank",
        }:
            # Positive = A better.
            out[k] = float(
                b[k] - a[k]
            )
        else:
            out[k] = float(
                a[k] - b[k]
            )

    return out


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Dual-view Tree inference for Video Games: latest semantic behavior (WHAT) "
            "+ predicted abstract state (HOW) -> game-specific ranking."
        )
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
        "--negatives",
        required=True,
    )
    p.add_argument(
        "--candidate-file",
        default=None,
    )

    p.add_argument(
        "--precomputed-behaviors",
        required=True,
    )
    p.add_argument(
        "--tree-dir",
        required=True,
    )
    p.add_argument(
        "--states-json",
        default=None,
    )
    p.add_argument(
        "--state-embeddings",
        default=None,
    )
    p.add_argument(
        "--tree-json",
        default=None,
    )

    p.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Defaults to behavior_states.json metadata."
        ),
    )
    p.add_argument(
        "--embedding-device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )
    p.add_argument(
        "--embedding-batch-size",
        type=int,
        default=128,
    )
    p.add_argument(
        "--cluster-text-field",
        choices=[
            "structured_combined",
            "trajectory_signature",
            "behavior_signature",
        ],
        default=None,
    )

    p.add_argument(
        "--mapping-min-similarity",
        type=float,
        default=None,
        help=(
            "Default: training threshold from behavior_states.json."
        ),
    )
    p.add_argument(
        "--mapping-min-margin",
        type=float,
        default=None,
        help=(
            "Default: training threshold from behavior_states.json."
        ),
    )
    p.add_argument(
        "--respect-mapping-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rejected mappings break the recent Tree context, matching training."
        ),
    )

    p.add_argument(
        "--model",
        default=(
            "unsloth/gemma-3-4b-it-"
            "unsloth-bnb-4bit"
        ),
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
    )
    p.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--rank-max-new-tokens",
        type=int,
        default=1024,
    )
    p.add_argument(
        "--rank-retries",
        type=int,
        default=0,
        help="Deprecated compatibility option. Retries are disabled and this value is ignored.",
    )
    p.add_argument(
        "--llm-batch-size",
        type=int,
        default=8,
    )

    p.add_argument(
        "--history-size",
        type=int,
        default=10,
    )
    p.add_argument(
        "--max-train-interactions",
        type=int,
        default=30,
        help=(
            "Used only to validate the precomputed behavior cache."
        ),
    )
    p.add_argument(
        "--top-next",
        type=int,
        default=1,
        help=(
            "Number of Tree-predicted abstract next states shown to ranking. "
            "Default 1 matches the latest-behavior + predicted-state formulation."
        ),
    )

    p.add_argument(
        "--run-native",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also rank History + Candidates."
        ),
    )
    p.add_argument(
        "--run-latest-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also rank History + Latest Behavior + Candidates."
        ),
    )

    p.add_argument(
        "--strict-cache-history",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--user-ids-file",
        default=None,
    )
    p.add_argument(
        "--start-user",
        type=int,
        default=0,
    )
    p.add_argument(
        "--max-users",
        type=int,
        default=0,
    )

    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--summary-output",
        default=None,
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tree_dir = Path(
        args.tree_dir
    )
    states_json = (
        args.states_json
        or str(
            tree_dir
            / "behavior_states.json"
        )
    )
    state_embeddings = (
        args.state_embeddings
        or str(
            tree_dir
            / "behavior_state_embeddings.npy"
        )
    )
    tree_json = (
        args.tree_json
        or str(
            tree_dir
            / "behavior_tree.json"
        )
    )

    required = [
        args.items,
        args.sequences,
        args.negatives,
        args.precomputed_behaviors,
        states_json,
        state_embeddings,
        tree_json,
    ]
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(
                path
            )

    items_meta = normalize_items(
        load_json(args.items)
    )
    sequences_raw = load_json(
        args.sequences
    )
    negatives_raw = load_json(
        args.negatives
    )

    sequences = {
        str(k): v
        for k, v in sequences_raw.items()
    }
    negatives = {
        str(k): v
        for k, v in negatives_raw.items()
    }

    behavior_cache = (
        load_precomputed_behaviors(
            args.precomputed_behaviors
        )
    )
    candidate_rows = (
        load_candidate_file(
            args.candidate_file
        )
    )

    requested = load_user_ids_file(
        args.user_ids_file
    )

    if requested is None:
        users = [
            uid
            for uid in sequences
            if uid in behavior_cache
        ]
    else:
        users = [
            str(uid)
            for uid in requested
            if (
                str(uid) in sequences
                and str(uid)
                in behavior_cache
            )
        ]

    users = users[
        int(args.start_user):
    ]
    if args.max_users > 0:
        users = users[
            :int(args.max_users)
        ]

    if not users:
        raise ValueError(
            "No selected users overlap sequences and behavior cache"
        )

    output_path = Path(
        args.output
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = Path(
        args.summary_output
        or (
            str(
                output_path.with_suffix(
                    ""
                )
            )
            + "_summary.json"
        )
    )

    if (
        output_path.exists()
        and not args.resume
    ):
        raise RuntimeError(
            f"{output_path} already exists. "
            "Delete it or use --resume."
        )

    done = (
        processed_users_from_jsonl(
            output_path
        )
        if args.resume
        else set()
    )

    pending = [
        uid
        for uid in users
        if uid not in done
    ]

    print("=" * 96)
    print(
        "DUAL-VIEW TREE INFERENCE: "
        "LATEST BEHAVIOR (WHAT) + PREDICTED STATE (HOW)"
    )
    print("=" * 96)
    print(
        f"selected users       : {len(users)}"
    )
    print(
        f"already completed    : {len(users) - len(pending)}"
    )
    print(
        f"pending              : {len(pending)}"
    )
    print(
        f"precomputed behavior : {args.precomputed_behaviors}"
    )
    print(
        f"tree dir             : {args.tree_dir}"
    )
    print(
        f"top-next             : {args.top_next}"
    )
    print(
        f"run latest-only      : {args.run_latest_only}"
    )
    print(
        f"run native           : {args.run_native}"
    )

    # -------------------------------------------------------------------------
    # Phase 1: validate behavior cache + frozen state mapping + Tree query.
    # Do this before loading Gemma so the Qwen encoder can be released.
    # -------------------------------------------------------------------------
    print(
        "\n[1/3] Map precomputed test behaviors to frozen states"
    )

    behaviors_by_user: Dict[
        str,
        List[Dict[str, Any]],
    ] = {}

    for uid in pending:
        cache_row = behavior_cache[
            uid
        ]
        validate_behavior_cache_history(
            uid=uid,
            cache_row=cache_row,
            user_data=sequences[uid],
            max_train_interactions=(
                args.max_train_interactions
            ),
            strict=(
                args.strict_cache_history
            ),
        )
        behaviors_by_user[uid] = list(
            cache_row[
                "generated_behaviors"
            ]
        )

    vocab = ClusterStateVocabulary(
        states_json=states_json,
        state_embeddings_npy=(
            state_embeddings
        ),
        embedding_model=(
            args.embedding_model
        ),
        embedding_device=(
            args.embedding_device
        ),
        embedding_batch_size=(
            args.embedding_batch_size
        ),
        cluster_text_field=(
            args.cluster_text_field
        ),
        min_similarity=(
            args.mapping_min_similarity
        ),
        min_margin=(
            args.mapping_min_margin
        ),
        respect_mapping_filter=(
            args.respect_mapping_filter
        ),
    )

    mappings_by_user = (
        vocab.map_users(
            behaviors_by_user
        )
    )

    tree = ReverseBehaviorTree(
        tree_json=tree_json,
        vocab=vocab,
    )

    prepared: List[
        Dict[str, Any]
    ] = []

    for uid in pending:
        user_data = sequences[
            uid
        ]
        behaviors = behaviors_by_user[
            uid
        ]
        mappings = mappings_by_user.get(
            uid,
            [],
        )

        state_suffix = (
            latest_accepted_state_suffix(
                mappings
            )
        )

        tree_result = tree.query(
            state_sequence=state_suffix,
            top_next=args.top_next,
        )

        latest = latest_behavior_view(
            behaviors
        )

        (
            candidate_ids,
            targets,
        ) = get_candidates_for_user(
            uid=uid,
            user_data=user_data,
            negative_data=(
                negatives.get(
                    uid,
                    {},
                )
            ),
            candidate_rows=(
                candidate_rows
            ),
            seed=args.seed,
        )

        candidate_items = [
            get_item_info(
                iid,
                items_meta,
            )
            for iid in candidate_ids
        ]

        train_ids = [
            str(x)
            for x in user_data.get(
                "train",
                [],
            )
        ]
        history_items = [
            get_item_info(
                iid,
                items_meta,
            )
            for iid in train_ids[
                -int(
                    args.history_size
                ):
            ]
        ]

        predicted_transition = (
            tree_result.get(
                "predicted_next_states",
                [],
            )
        )

        prepared.append({
            "user_id": uid,
            "behaviors": behaviors,
            "latest_behavior": (
                latest
            ),
            "state_mappings": (
                mappings
            ),
            "tree_query_state_suffix": (
                state_suffix
            ),
            "tree_result": (
                tree_result
            ),
            "tree_rank_evidence": (
                build_tree_rank_evidence(
                    latest,
                    tree_result,
                )
            ),
            "predicted_transition": (
                predicted_transition
            ),
            "candidate_ids": (
                candidate_ids
            ),
            "candidate_items": (
                candidate_items
            ),
            "targets": (
                targets
            ),
            "history_items": (
                history_items
            ),
        })

    # Free Qwen before loading Gemma.
    vocab.release_encoder()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Phase 2: controlled ranking.
    # -------------------------------------------------------------------------
    print(
        "\n[2/3] Controlled LLM ranking"
    )

    ranker = LocalGemmaRanker(
        model_name=args.model,
        max_seq_length=(
            args.max_seq_length
        ),
        load_in_4bit=(
            args.load_in_4bit
        ),
        rank_max_new_tokens=(
            args.rank_max_new_tokens
        ),
        temperature=(
            args.temperature
        ),
        seed=args.seed,
    )

    tree_jobs = [
        {
            "history_items": x[
                "history_items"
            ],
            "candidate_items": x[
                "candidate_items"
            ],
            "latest_behavior": x[
                "latest_behavior"
            ],
            "predicted_transition": x[
                "predicted_transition"
            ],
        }
        for x in prepared
    ]

    print(
        f"[RANK] Tree condition: {len(tree_jobs)} users, "
        f"batch_size={args.llm_batch_size}",
        flush=True,
    )
    tree_rankings = ranker.rank_jobs(
        tree_jobs,
        batch_size=args.llm_batch_size,
        call_type="tree_latest_plus_transition",
        max_retries=0,
    )

    latest_rankings = None
    if args.run_latest_only:
        latest_jobs = [
            {
                "history_items": x[
                    "history_items"
                ],
                "candidate_items": x[
                    "candidate_items"
                ],
                "latest_behavior": x[
                    "latest_behavior"
                ],
                "predicted_transition": None,
            }
            for x in prepared
        ]
        print(
            f"[RANK] Latest-only condition: {len(latest_jobs)} users, "
            f"batch_size={args.llm_batch_size}",
            flush=True,
        )
        latest_rankings = (
            ranker.rank_jobs(
                latest_jobs,
                batch_size=(
                    args.llm_batch_size
                ),
                call_type=(
                    "latest_behavior_only"
                ),
                max_retries=0,
            )
        )

    native_rankings = None
    if args.run_native:
        native_jobs = [
            {
                "history_items": x[
                    "history_items"
                ],
                "candidate_items": x[
                    "candidate_items"
                ],
                "latest_behavior": None,
                "predicted_transition": None,
            }
            for x in prepared
        ]
        print(
            f"[RANK] Native condition: {len(native_jobs)} users, "
            f"batch_size={args.llm_batch_size}",
            flush=True,
        )
        native_rankings = (
            ranker.rank_jobs(
                native_jobs,
                batch_size=(
                    args.llm_batch_size
                ),
                call_type="native",
                max_retries=0,
            )
        )

    # -------------------------------------------------------------------------
    # Phase 3: save rows + aggregate summary.
    # -------------------------------------------------------------------------
    print(
        "\n[3/3] Save results and summary"
    )

    for i, item in enumerate(
        tqdm(
            prepared,
            desc="write results",
        )
    ):
        uid = item[
            "user_id"
        ]
        target = (
            str(
                item["targets"][0]
            )
            if item["targets"]
            else None
        )

        tree_ranking, tree_meta = (
            tree_rankings[i]
        )

        rankings: Dict[
            str,
            Dict[str, Any],
        ] = {
            "tree": {
                "ranked_item_ids": (
                    tree_ranking
                ),
                "target_rank": (
                    target_rank(
                        tree_ranking,
                        target,
                    )
                    if target
                    is not None
                    else None
                ),
                **tree_meta,
            }
        }

        if latest_rankings is not None:
            (
                latest_ranking,
                latest_meta,
            ) = latest_rankings[
                i
            ]
            rankings[
                "latest_only"
            ] = {
                "ranked_item_ids": (
                    latest_ranking
                ),
                "target_rank": (
                    target_rank(
                        latest_ranking,
                        target,
                    )
                    if target
                    is not None
                    else None
                ),
                **latest_meta,
            }

        if native_rankings is not None:
            (
                native_ranking,
                native_meta,
            ) = native_rankings[
                i
            ]
            rankings[
                "native"
            ] = {
                "ranked_item_ids": (
                    native_ranking
                ),
                "target_rank": (
                    target_rank(
                        native_ranking,
                        target,
                    )
                    if target
                    is not None
                    else None
                ),
                **native_meta,
            }

        row = {
            "schema_version": (
                SCHEMA_VERSION
            ),
            "rank_prompt_version": (
                RANK_PROMPT_VERSION
            ),
            "user_id": uid,
            "ground_truth_item_ids": (
                item["targets"]
            ),
            "candidate_item_ids": (
                item["candidate_ids"]
            ),
            "history_items": (
                item["history_items"]
            ),

            # WHAT: target-user semantics.
            "latest_observed_behavior": (
                item[
                    "latest_behavior"
                ]
            ),

            # Mapping sequence used to query the Tree.
            "state_mappings": (
                item[
                    "state_mappings"
                ]
            ),
            "tree_query_state_suffix": (
                item[
                    "tree_query_state_suffix"
                ]
            ),

            # HOW: collaborative next transition.
            "tree_result": (
                item[
                    "tree_result"
                ]
            ),
            "tree_rank_evidence": (
                item[
                    "tree_rank_evidence"
                ]
            ),

            "rankings": rankings,
        }

        append_jsonl(
            output_path,
            row,
        )

    all_rows = [
        row
        for row in read_jsonl(
            output_path
        )
        if (
            row.get(
                "schema_version"
            )
            == SCHEMA_VERSION
            and row.get(
                "rank_prompt_version"
            )
            == RANK_PROMPT_VERSION
        )
    ]

    modes = ["tree"]
    if args.run_latest_only:
        modes.append(
            "latest_only"
        )
    if args.run_native:
        modes.append(
            "native"
        )

    metrics_all = {
        mode: aggregate_mode(
            all_rows,
            mode,
            strict_complete=False,
        )
        for mode in modes
    }
    metrics_complete_only = {
        mode: aggregate_mode(
            all_rows,
            mode,
            strict_complete=True,
        )
        for mode in modes
    }

    parse_quality = {}
    for mode in modes:
        recs = [
            row.get(
                "rankings",
                {},
            ).get(mode)
            for row in all_rows
        ]
        recs = [
            x
            for x in recs
            if isinstance(
                x,
                dict,
            )
        ]
        parse_quality[
            mode
        ] = {
            "num_rows": len(
                recs
            ),
            "complete_count": int(
                sum(
                    bool(
                        x.get(
                            "parse_complete"
                        )
                    )
                    for x in recs
                )
            ),
            "complete_ratio": (
                float(
                    sum(
                        bool(
                            x.get(
                                "parse_complete"
                            )
                        )
                        for x in recs
                    )
                    / len(recs)
                )
                if recs
                else None
            ),
        }

    order_counts = Counter()
    root_fallbacks = 0
    total_tree_rows = 0
    mapping_total = 0
    mapping_accepted = 0
    mapping_rejections = Counter()
    mapping_scope_counts = Counter()

    for row in all_rows:
        tr = row.get(
            "tree_result"
        )
        if isinstance(
            tr,
            dict,
        ):
            total_tree_rows += 1
            order = int(
                tr.get(
                    "matched_order",
                    0,
                )
            )
            order_counts[
                str(order)
            ] += 1
            if order == 0:
                root_fallbacks += 1

        for m in row.get(
            "state_mappings",
            [],
        ):
            mapping_total += 1
            if m.get(
                "assignment_accepted"
            ):
                mapping_accepted += 1
            else:
                mapping_rejections[
                    str(
                        m.get(
                            "rejection_reason",
                            "unknown",
                        )
                    )
                ] += 1

            mapping_scope_counts[
                str(
                    m.get(
                        "retrieval_scope",
                        "unknown",
                    )
                )
            ] += 1

    summary = {
        "schema_version": (
            "dual_tree_latest_behavior_inference_summary_v1"
        ),
        "rank_prompt_version": (
            RANK_PROMPT_VERSION
        ),
        "method": (
            "latest target-user semantic behavior (WHAT) "
            "+ collaborative Tree predicted abstract state (HOW)"
        ),
        "num_users": len(
            all_rows
        ),
        "ranking_conditions": (
            modes
        ),

        "metrics_all_rows": (
            metrics_all
        ),
        "metrics_parse_complete_only": (
            metrics_complete_only
        ),
        "parse_quality": (
            parse_quality
        ),

        "tree_minus_latest_only": (
            metric_gain(
                metrics_complete_only.get(
                    "tree"
                ),
                metrics_complete_only.get(
                    "latest_only"
                ),
            )
            if args.run_latest_only
            else None
        ),
        "tree_minus_native": (
            metric_gain(
                metrics_complete_only.get(
                    "tree"
                ),
                metrics_complete_only.get(
                    "native"
                ),
            )
            if args.run_native
            else None
        ),
        "latest_only_minus_native": (
            metric_gain(
                metrics_complete_only.get(
                    "latest_only"
                ),
                metrics_complete_only.get(
                    "native"
                ),
            )
            if (
                args.run_latest_only
                and args.run_native
            )
            else None
        ),

        "tree_diagnostics": {
            "matched_context_order_counts": (
                dict(order_counts)
            ),
            "root_fallback_ratio": (
                float(
                    root_fallbacks
                    / total_tree_rows
                )
                if total_tree_rows
                else None
            ),
            "top_next": int(
                args.top_next
            ),
        },

        "state_mapping_diagnostics": {
            "total_mappings": int(
                mapping_total
            ),
            "accepted_mappings": int(
                mapping_accepted
            ),
            "accepted_ratio": (
                float(
                    mapping_accepted
                    / mapping_total
                )
                if mapping_total
                else None
            ),
            "rejection_reasons": (
                dict(
                    mapping_rejections
                )
            ),
            "retrieval_scope_counts": (
                dict(
                    mapping_scope_counts
                )
            ),
            "respect_mapping_filter": bool(
                args.respect_mapping_filter
            ),
            "min_similarity": float(
                vocab.min_similarity
            ),
            "min_margin": float(
                vocab.min_margin
            ),
            "constraint_level": (
                vocab.constraint_level
            ),
        },

        "ranking_llm": (
            ranker.stats()
        ),

        "files": {
            "output": str(
                output_path
            ),
            "summary": str(
                summary_path
            ),
            "precomputed_behaviors": (
                args.precomputed_behaviors
            ),
            "states_json": (
                states_json
            ),
            "state_embeddings": (
                state_embeddings
            ),
            "tree_json": (
                tree_json
            ),
        },

        "config": vars(args),
    }

    dump_json(
        summary_path,
        summary,
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"\nResults : {output_path}"
    )
    print(
        f"Summary : {summary_path}"
    )


if __name__ == "__main__":
    main()
