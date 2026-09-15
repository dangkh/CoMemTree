#!/usr/bin/env python3
"""
Build a Flat-AMem global memory from TEXT-ONLY precomputed local memories.

Expected input
--------------
JSONL produced by:
    precompute_memory_create_gemma_text.py

Each input record contains semantic memory text / metadata only:
    - precompute_id
    - user_id / user_order / window_index
    - interaction_sequence
    - behavior_explanation
    - pattern_description
    - keywords
    - parse_ok / raw_response
and intentionally contains NO embedding.

Pipeline
--------
1. Load all local-memory JSONL records.
2. Compose one canonical text per local memory.
3. Batch-encode ALL required local memories once using the embedding model.
4. Replay local memories sequentially in precompute_id order:
       exact cosine top-k
       -> LLM link decision
       -> LLM evolution
       -> store/discard
5. If an existing global memory evolves, only that memory is re-embedded.
6. FAISS IndexFlatIP + IndexIDMap2 is used for EXACT flat cosine search.
   This is NOT ANN and NOT Tree Memory.

Why this design
---------------
The precompute artifact remains independent of the embedding model. You can
change --embedding_model and rebuild global memory without rerunning Gemma
memory extraction.

Suggested packages
------------------
pip install -U unsloth sentence-transformers faiss-cpu tqdm

Example
-------
python build_global_memory_flat_faiss_gemma_text.py \
  --precomputed precomputed/CDs/local_memories_gemma_text.jsonl \
  --output agent_memory/CDs/global_flat_faiss_gemma_text.json \
  --model_name /home/hkieu/.cache/huggingface/hub/models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe \
  --embedding_model Qwen/Qwen3-Embedding-0.6B \
  --embedding_batch_size 64 \
  --low_threshold 0.65 \
  --high_threshold 0.8 \
  --link_size 5 \
  --max_evolutions_per_memory 10 \
  --max_new_tokens 256 \
  --save_every 100

Important
---------
Use a NEW output file when changing:
    - embedding model
    - threshold settings
    - linking/evolution LLM
    - precomputed input
unless you intentionally want to resume the exact same run.
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch

# IMPORTANT: import Unsloth before libraries that import Transformers.
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


EXPECTED_PRECOMPUTE_SCHEMA = "amem_local_memory_text_v2"


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class UserInteraction:
    item_id: str
    item_name: str
    item_category: str
    action_type: str
    rating: Optional[float] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BehaviorMemory:
    thought_id: int
    interaction_sequence: List[UserInteraction]
    behavior_explanation: str
    pattern_description: str
    keywords: List[str]
    embedding: np.ndarray

    links: List[int] = field(default_factory=list)
    timestamp: Optional[str] = None
    evolution_count: int = 0
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    max_evolutions: Optional[int] = None
    last_evolved_timestamp: Optional[str] = None

    # Useful provenance; ignored by the old AMem logic but helpful for analysis.
    source_user_id: Optional[str] = None
    source_user_order: Optional[int] = None
    source_window_index: Optional[int] = None
    source_precompute_id: Optional[int] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        self.embedding = np.asarray(self.embedding, dtype=np.float32)

    def can_evolve(self) -> bool:
        if self.max_evolutions is None:
            return True
        return self.evolution_count < self.max_evolutions

    def record_evolution(
        self,
        update_type: str,
        old_values: Dict[str, Any],
        new_values: Dict[str, Any],
        reasoning: str,
    ) -> None:
        self.evolution_count += 1
        self.last_evolved_timestamp = datetime.now().isoformat()
        self.evolution_history.append({
            "evolution_number": self.evolution_count,
            "timestamp": self.last_evolved_timestamp,
            "update_type": update_type,
            "old_values": old_values,
            "new_values": new_values,
            "reasoning": reasoning,
        })

    def to_dict(self) -> Dict[str, Any]:
        x = asdict(self)
        x["embedding"] = self.embedding.astype(np.float32).tolist()
        x["interaction_sequence"] = [
            asdict(v) for v in self.interaction_sequence
        ]
        return x

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorMemory":
        d = dict(data)
        d["embedding"] = np.asarray(d["embedding"], dtype=np.float32)
        d["interaction_sequence"] = [
            v if isinstance(v, UserInteraction) else UserInteraction(**v)
            for v in d.get("interaction_sequence", [])
        ]
        return cls(**d)


# =============================================================================
# Tracking
# =============================================================================

@dataclass
class LLMCallRecord:
    call_id: int
    call_type: str
    input_tokens: int
    output_tokens: int
    inference_time_sec: float
    timestamp: str


@dataclass
class LLMTracker:
    records: List[LLMCallRecord] = field(default_factory=list)
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_inference_time: float = 0.0

    def record(
        self,
        call_type: str,
        input_tokens: int,
        output_tokens: int,
        elapsed: float,
    ) -> None:
        self.total_calls += 1
        self.total_input_tokens += int(input_tokens)
        self.total_output_tokens += int(output_tokens)
        self.total_inference_time += float(elapsed)

        self.records.append(
            LLMCallRecord(
                call_id=self.total_calls,
                call_type=call_type,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                inference_time_sec=round(float(elapsed), 4),
                timestamp=datetime.now().isoformat(),
            )
        )

    def summary(self) -> Dict[str, Any]:
        by_type: Dict[str, Any] = {}

        for r in self.records:
            if r.call_type not in by_type:
                by_type[r.call_type] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "inference_time_sec": 0.0,
                }

            x = by_type[r.call_type]
            x["calls"] += 1
            x["input_tokens"] += r.input_tokens
            x["output_tokens"] += r.output_tokens
            x["inference_time_sec"] = round(
                x["inference_time_sec"] + r.inference_time_sec,
                4,
            )

        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": (
                self.total_input_tokens + self.total_output_tokens
            ),
            "total_inference_time_sec": round(
                self.total_inference_time,
                4,
            ),
            "avg_inference_time_sec": (
                round(
                    self.total_inference_time / self.total_calls,
                    4,
                )
                if self.total_calls else 0.0
            ),
            "by_call_type": by_type,
        }


@dataclass
class SearchStats:
    search_calls: int = 0
    exact_comparisons: int = 0
    total_search_time_sec: float = 0.0
    max_pool_size_searched: int = 0

    add_calls: int = 0
    total_add_time_sec: float = 0.0

    update_calls: int = 0
    total_update_time_sec: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "search_calls": self.search_calls,
            "exact_comparisons": self.exact_comparisons,
            "total_search_time_sec": round(
                self.total_search_time_sec,
                6,
            ),
            "avg_search_time_ms": (
                round(
                    self.total_search_time_sec
                    / self.search_calls
                    * 1000.0,
                    4,
                )
                if self.search_calls else 0.0
            ),
            "max_pool_size_searched": (
                self.max_pool_size_searched
            ),
            "add_calls": self.add_calls,
            "total_add_time_sec": round(
                self.total_add_time_sec,
                6,
            ),
            "avg_add_time_ms": (
                round(
                    self.total_add_time_sec
                    / self.add_calls
                    * 1000.0,
                    4,
                )
                if self.add_calls else 0.0
            ),
            "update_calls": self.update_calls,
            "total_update_time_sec": round(
                self.total_update_time_sec,
                6,
            ),
            "avg_update_time_ms": (
                round(
                    self.total_update_time_sec
                    / self.update_calls
                    * 1000.0,
                    4,
                )
                if self.update_calls else 0.0
            ),
        }


# =============================================================================
# Canonical memory text
# =============================================================================

def compose_memory_text(
    behavior_explanation: str,
    pattern_description: str,
    keywords: List[str],
) -> str:
    """
    Keep the same semantic combination used by the original AMem create/evolve
    code, but centralize it so local and evolved memories use exactly the same
    representation rule.
    """
    return (
        f"{behavior_explanation} "
        f"{pattern_description} "
        f"{' '.join(str(x) for x in keywords)}"
    ).strip()


# =============================================================================
# Prompts
# =============================================================================

def build_linking_prompt(
    new_behavior: str,
    new_pattern: str,
    nearest_info: List[Dict[str, Any]],
    low_threshold: float,
) -> str:
    return f"""You are a memory linking agent. Decide which past patterns are semantically related to the new pattern.

New Pattern:
- Behavior: {new_behavior}
- Pattern: {new_pattern}

Candidate Memories:
{json.dumps(nearest_info, indent=2, ensure_ascii=False)}

Based on shared categories, preferences, or behavioral structure, determine which candidates should be linked to the new pattern.

Return ONLY valid JSON:
{{"should_link": true/false, "linked_thought_ids": [...], "reasoning": "1 sentence"}}"""


def build_evolution_prompt(
    new_behavior: str,
    new_pattern: str,
    mem_info: List[Dict[str, Any]],
) -> str:
    return f"""You are a collaborative memory evolution expert.
Your goal is to reinforce shared cross-user patterns with minimal, concise changes.

New Pattern:
- Behavior: {new_behavior}
- Pattern: {new_pattern}

Candidate Memories (with evolution count):
{json.dumps(mem_info, indent=2, ensure_ascii=False)}

Rules:
- Update only if the new pattern meaningfully strengthens or refines a shared category, preference, or sequence.
- Keep updated text very concise.
- If no meaningful improvement, keep the original text.

Return ONLY valid JSON:
{{"should_evolve": true/false, "updates": [{{"thought_id": id, "behavior_explanation": "updated text or null", "pattern_description": "updated text or null", "reasoning": "1 sentence"}}]}}"""


def build_mem_info_with_decay(
    memories: List[BehaviorMemory],
    max_evolutions: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "thought_id": m.thought_id,
            "behavior_explanation": m.behavior_explanation,
            "pattern": m.pattern_description,
            "evolution_count": m.evolution_count,
            "update_weight": round(
                max(
                    0.2,
                    1.0
                    - (
                        m.evolution_count
                        / max(max_evolutions, 1)
                    )
                    * 0.8,
                ),
                2,
            ),
        }
        for m in memories
    ]


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()

    if "```json" in text:
        text = (
            text.split("```json", 1)[1]
            .split("```", 1)[0]
            .strip()
        )
    elif "```" in text:
        text = (
            text.split("```", 1)[1]
            .split("```", 1)[0]
            .strip()
        )

    start = text.find("{")
    end = text.rfind("}") + 1

    if start != -1 and end > start:
        text = text[start:end]

    return json.loads(text)


# =============================================================================
# Exact cosine index
# =============================================================================

class ExactCosineFaissIndex:
    """
    Exact flat cosine search.

    IndexFlatIP performs an exhaustive inner-product scan. With L2-normalized
    vectors, inner product == cosine similarity.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)

        if self.dim <= 0:
            raise ValueError(
                f"Invalid embedding dimension: {dim}"
            )

        self.index = faiss.IndexIDMap2(
            faiss.IndexFlatIP(self.dim)
        )
        self.stats = SearchStats()

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def normalize(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        arr = np.asarray(
            x,
            dtype=np.float32,
        )

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if (
            arr.ndim != 2
            or arr.shape[1] != self.dim
        ):
            raise ValueError(
                f"Expected (*,{self.dim}), got {arr.shape}"
            )

        arr = np.ascontiguousarray(
            arr,
            dtype=np.float32,
        )
        faiss.normalize_L2(arr)
        return arr

    def add(
        self,
        thought_id: int,
        embedding: np.ndarray,
    ) -> None:
        vec = self.normalize(embedding)
        ids = np.asarray(
            [thought_id],
            dtype=np.int64,
        )

        t0 = time.perf_counter()
        self.index.add_with_ids(vec, ids)
        elapsed = time.perf_counter() - t0

        self.stats.add_calls += 1
        self.stats.total_add_time_sec += elapsed

    def add_many(
        self,
        thought_ids: List[int],
        embeddings: np.ndarray,
    ) -> None:
        if not thought_ids:
            return

        vecs = self.normalize(embeddings)
        ids = np.asarray(
            thought_ids,
            dtype=np.int64,
        )

        if len(ids) != len(vecs):
            raise ValueError(
                "thought_ids and embeddings length mismatch"
            )

        t0 = time.perf_counter()
        self.index.add_with_ids(vecs, ids)
        elapsed = time.perf_counter() - t0

        self.stats.add_calls += len(ids)
        self.stats.total_add_time_sec += elapsed

    def update(
        self,
        thought_id: int,
        embedding: np.ndarray,
    ) -> None:
        vec = self.normalize(embedding)
        ids = np.asarray(
            [thought_id],
            dtype=np.int64,
        )

        t0 = time.perf_counter()

        self.index.remove_ids(ids)
        self.index.add_with_ids(vec, ids)

        elapsed = time.perf_counter() - t0

        self.stats.update_calls += 1
        self.stats.total_update_time_sec += elapsed

    def search(
        self,
        query: np.ndarray,
        k: int,
    ) -> Tuple[List[int], List[float]]:
        if self.ntotal == 0 or k <= 0:
            return [], []

        q = self.normalize(query)
        k_actual = min(
            int(k),
            self.ntotal,
        )

        pool_size = self.ntotal

        t0 = time.perf_counter()
        scores, ids = self.index.search(
            q,
            k_actual,
        )
        elapsed = time.perf_counter() - t0

        self.stats.search_calls += 1
        self.stats.exact_comparisons += pool_size
        self.stats.total_search_time_sec += elapsed
        self.stats.max_pool_size_searched = max(
            self.stats.max_pool_size_searched,
            pool_size,
        )

        result_ids: List[int] = []
        result_scores: List[float] = []

        for thought_id, score in zip(
            ids[0].tolist(),
            scores[0].tolist(),
        ):
            if thought_id == -1:
                continue

            result_ids.append(
                int(thought_id)
            )
            result_scores.append(
                float(score)
            )

        return result_ids, result_scores


# =============================================================================
# Input
# =============================================================================

def load_precomputed_jsonl(
    path: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception as exc:
                raise ValueError(
                    f"Invalid JSONL line {line_no}: {exc}"
                ) from exc

            if "embedding" in rec:
                raise RuntimeError(
                    f"Input line {line_no} contains `embedding`. "
                    "This builder expects the NEW text-only precompute schema."
                )

            schema = rec.get(
                "schema_version"
            )

            if (
                schema is not None
                and schema != EXPECTED_PRECOMPUTE_SCHEMA
            ):
                raise RuntimeError(
                    f"Unexpected schema_version={schema!r} "
                    f"at line {line_no}; expected "
                    f"{EXPECTED_PRECOMPUTE_SCHEMA!r}."
                )

            required = [
                "precompute_id",
                "behavior_explanation",
                "pattern_description",
                "keywords",
                "interaction_sequence",
            ]

            missing = [
                key
                for key in required
                if key not in rec
            ]

            if missing:
                raise RuntimeError(
                    f"Missing fields {missing} "
                    f"at JSONL line {line_no}"
                )

            records.append(rec)

    records.sort(
        key=lambda r: int(
            r["precompute_id"]
        )
    )

    # Require unique, increasing IDs.
    ids = [
        int(r["precompute_id"])
        for r in records
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate precompute_id values detected."
        )

    return records


def canonical_text_from_record(
    rec: Dict[str, Any],
) -> str:
    keywords = rec.get(
        "keywords",
        [],
    )

    if not isinstance(keywords, list):
        keywords = [str(keywords)]

    return compose_memory_text(
        behavior_explanation=str(
            rec.get(
                "behavior_explanation",
                "",
            )
        ),
        pattern_description=str(
            rec.get(
                "pattern_description",
                "",
            )
        ),
        keywords=[
            str(x) for x in keywords
        ],
    )


def record_to_memory(
    rec: Dict[str, Any],
    embedding: np.ndarray,
) -> BehaviorMemory:
    interactions: List[UserInteraction] = []

    for x in rec.get(
        "interaction_sequence",
        [],
    ):
        interactions.append(
            UserInteraction(
                item_id=str(
                    x.get(
                        "item_id",
                        "",
                    )
                ),
                item_name=str(
                    x.get(
                        "item_name",
                        "",
                    )
                ),
                item_category=str(
                    x.get(
                        "item_category",
                        "Unknown",
                    )
                ),
                action_type=str(
                    x.get(
                        "action_type",
                        "purchase",
                    )
                ),
                rating=x.get("rating"),
                timestamp=x.get("timestamp"),
                metadata=x.get(
                    "metadata",
                    {},
                ),
            )
        )

    keywords = rec.get(
        "keywords",
        [],
    )

    if not isinstance(keywords, list):
        keywords = [str(keywords)]

    pid = int(
        rec["precompute_id"]
    )

    return BehaviorMemory(
        thought_id=pid,
        interaction_sequence=interactions,
        behavior_explanation=str(
            rec.get(
                "behavior_explanation",
                "",
            )
        ),
        pattern_description=str(
            rec.get(
                "pattern_description",
                "",
            )
        ),
        keywords=[
            str(x) for x in keywords
        ],
        embedding=np.asarray(
            embedding,
            dtype=np.float32,
        ),
        source_user_id=str(
            rec.get(
                "user_id",
                "",
            )
        ),
        source_user_order=(
            int(rec["user_order"])
            if rec.get("user_order") is not None
            else None
        ),
        source_window_index=(
            int(rec["window_index"])
            if rec.get("window_index") is not None
            else None
        ),
        source_precompute_id=pid,
    )


# =============================================================================
# Builder
# =============================================================================

class FlatAMemBuilder:
    def __init__(
        self,
        model_name: str,
        embedding_model_name: str,
        low_threshold: float,
        high_threshold: float,
        max_new_tokens: int,
        llm_max_seq_length: int,
        load_in_4bit: bool,
        embedding_device: str,
    ):
        if low_threshold >= high_threshold:
            raise ValueError(
                "low_threshold must be < high_threshold"
            )

        self.model_name = model_name
        self.embedding_model_name = (
            embedding_model_name
        )
        self.low_threshold = float(
            low_threshold
        )
        self.high_threshold = float(
            high_threshold
        )
        self.max_new_tokens = int(
            max_new_tokens
        )

        # ------------------------------------------------------------------
        # Gemma / Unsloth linking + evolution LLM
        # ------------------------------------------------------------------

        print(
            f"Loading Gemma linking/evolution LLM: "
            f"{model_name}"
        )

        self.model, self.tokenizer = (
            FastModel.from_pretrained(
                model_name=model_name,
                max_seq_length=llm_max_seq_length,
                load_in_4bit=load_in_4bit,
                full_finetuning=False,
            )
        )

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="gemma3",
        )
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )

        try:
            FastModel.for_inference(
                self.model
            )
        except Exception:
            pass

        self.model.eval()

        self.llm_device = next(
            self.model.parameters()
        ).device

        print(
            f"Gemma device: {self.llm_device}"
        )

        # ------------------------------------------------------------------
        # Embedding model
        # ------------------------------------------------------------------

        if embedding_device == "auto":
            # Default to CPU if Gemma is on CUDA to reduce VRAM pressure.
            # User can explicitly request CUDA.
            emb_device = (
                "cpu"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            emb_device = embedding_device

        print(
            f"Loading embedding model: "
            f"{embedding_model_name} "
            f"(device={emb_device})"
        )

        self.embedding_model = (
            SentenceTransformer(
                embedding_model_name,
                device=emb_device,
            )
        )

        self.embedding_device = emb_device

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------

        self.behavior_memories: List[
            BehaviorMemory
        ] = []

        self._id2idx: Dict[
            int,
            int,
        ] = {}

        self.faiss_index: Optional[
            ExactCosineFaissIndex
        ] = None

        self.llm_tracker = LLMTracker()

        self.processed_local_memories = 0
        self.stored_memories = 0
        self.discarded_memories = 0
        self.update_decisions = 0
        self.store_decisions = 0
        self.evolved_batches = 0

    # ------------------------------------------------------------------
    # Gemma generation
    # ------------------------------------------------------------------

    def gemma_generate(
        self,
        prompt: str,
        role_prompt: str,
        call_type: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": role_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        if text.startswith("<bos>"):
            text = text[len("<bos>"):]

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.llm_device)

        input_width = int(
            inputs["input_ids"].shape[1]
        )
        input_token_count = int(
            inputs["attention_mask"]
            .sum()
            .item()
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=(
                    self.max_new_tokens
                ),
                use_cache=True,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                ),
                eos_token_id=(
                    self.tokenizer.eos_token_id
                ),
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter() - t0
        )

        generated = outputs[0][
            input_width:
        ]

        output_token_count = int(
            len(generated)
        )

        self.llm_tracker.record(
            call_type=call_type,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            elapsed=elapsed,
        )

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        embeddings = (
            self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            )
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def encode_one(
        self,
        text: str,
    ) -> np.ndarray:
        emb = self.embedding_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            emb,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Index / memory state
    # ------------------------------------------------------------------

    def _ensure_index(
        self,
        dim: int,
    ) -> None:
        if self.faiss_index is None:
            self.faiss_index = (
                ExactCosineFaissIndex(dim)
            )
        elif self.faiss_index.dim != dim:
            raise RuntimeError(
                f"Embedding dimension changed: "
                f"{self.faiss_index.dim} -> {dim}"
            )

    def _rebuild_id2idx(
        self,
    ) -> None:
        self._id2idx = {
            m.thought_id: idx
            for idx, m
            in enumerate(
                self.behavior_memories
            )
        }

    def _rebuild_faiss_from_memory(
        self,
    ) -> None:
        if not self.behavior_memories:
            self.faiss_index = None
            return

        dim = int(
            self.behavior_memories[0]
            .embedding.shape[-1]
        )

        self.faiss_index = (
            ExactCosineFaissIndex(dim)
        )

        ids = [
            m.thought_id
            for m in self.behavior_memories
        ]

        embeddings = np.stack(
            [
                m.embedding
                for m
                in self.behavior_memories
            ],
            axis=0,
        ).astype(np.float32)

        self.faiss_index.add_many(
            ids,
            embeddings,
        )

    def get_memory_by_id(
        self,
        thought_id: int,
    ) -> Optional[BehaviorMemory]:
        idx = self._id2idx.get(
            int(thought_id)
        )

        if idx is None:
            return None

        return self.behavior_memories[
            idx
        ]

    def store_memory(
        self,
        memory: BehaviorMemory,
    ) -> None:
        self._ensure_index(
            memory.embedding.shape[-1]
        )

        self.behavior_memories.append(
            memory
        )

        self._id2idx[
            memory.thought_id
        ] = (
            len(self.behavior_memories)
            - 1
        )

        # Incremental add: no full matrix rebuild.
        self.faiss_index.add(
            memory.thought_id,
            memory.embedding,
        )

        self.stored_memories += 1

    def replace_memory_embedding(
        self,
        memory: BehaviorMemory,
        new_embedding: np.ndarray,
    ) -> None:
        memory.embedding = np.asarray(
            new_embedding,
            dtype=np.float32,
        )

        self._ensure_index(
            memory.embedding.shape[-1]
        )

        # Incremental update of one thought_id only.
        self.faiss_index.update(
            memory.thought_id,
            memory.embedding,
        )

    # ------------------------------------------------------------------
    # Threshold / strategy
    # ------------------------------------------------------------------

    def get_adaptive_thresholds(
        self,
    ) -> Tuple[float, float]:
        n = len(
            self.behavior_memories
        )

        scale = min(
            n / 1000.0,
            1.0,
        )

        low = (
            self.low_threshold
            + 0.05 * scale
        )

        high = (
            self.high_threshold
            - 0.05 * scale
        )

        if low >= high:
            return (
                self.low_threshold,
                self.high_threshold,
            )

        return low, high

    def determine_memory_strategy(
        self,
        max_similarity: float,
        topk_similarities: List[float],
    ) -> Tuple[bool, bool, str]:
        if not topk_similarities:
            return (
                False,
                True,
                "store_only_no_memories",
            )

        low, high = (
            self.get_adaptive_thresholds()
        )

        total = len(
            topk_similarities
        )

        high_sims = [
            s
            for s in topk_similarities
            if s >= high
        ]

        low_sims = [
            s
            for s in topk_similarities
            if s < low
        ]

        high_percent = (
            len(high_sims) / total
        )

        low_percent = (
            len(low_sims) / total
        )

        if max_similarity < low:
            return (
                False,
                True,
                "store_only_completely_new",
            )

        if max_similarity < high:
            return (
                True,
                True,
                "update_and_store_partial_overlap",
            )

        if high_percent >= 0.6:
            return (
                True,
                False,
                "update_only_likely_duplicate",
            )

        if low_percent >= 0.5:
            return (
                False,
                True,
                "store_only_few_relevant",
            )

        if high_percent >= 0.4:
            return (
                True,
                True,
                "update_and_store_common_pattern",
            )

        return (
            True,
            True,
            "update_and_store_ambiguous",
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def find_similar_memories(
        self,
        new_memory: BehaviorMemory,
        k: int,
        verbose: bool,
    ) -> Tuple[
        List[int],
        List[float],
        List[int],
        bool,
        bool,
        str,
    ]:
        n = len(
            self.behavior_memories
        )

        if n == 0:
            return (
                [],
                [],
                [],
                False,
                True,
                "store_only_no_memories",
            )

        assert self.faiss_index is not None

        ids, sims = (
            self.faiss_index.search(
                new_memory.embedding,
                min(k, n),
            )
        )

        max_similarity = (
            sims[0]
            if sims
            else 0.0
        )

        (
            should_update,
            should_store,
            zone,
        ) = self.determine_memory_strategy(
            max_similarity,
            sims,
        )

        low, high = (
            self.get_adaptive_thresholds()
        )

        update_eligible_ids = [
            thought_id
            for thought_id, sim
            in zip(ids, sims)
            if sim >= low
        ]

        if verbose:
            total = len(sims)

            high_count = sum(
                1
                for s in sims
                if s >= high
            )

            medium_count = sum(
                1
                for s in sims
                if low <= s < high
            )

            low_count = sum(
                1
                for s in sims
                if s < low
            )

            print(
                f"  -> Max similarity: "
                f"{max_similarity:.4f}"
            )

            if total:
                print(
                    "  -> Distribution: "
                    f"high={high_count}/{total} "
                    f"({100 * high_count / total:.1f}%), "
                    f"medium={medium_count}/{total} "
                    f"({100 * medium_count / total:.1f}%), "
                    f"low={low_count}/{total} "
                    f"({100 * low_count / total:.1f}%)"
                )

            print(
                f"  -> Zone: {zone}"
            )

            strategy_parts = []

            if should_update:
                strategy_parts.append(
                    "UPDATE"
                )

            if should_store:
                strategy_parts.append(
                    "STORE"
                )

            print(
                "  -> Strategy: "
                + " + ".join(strategy_parts)
            )

        return (
            ids,
            sims,
            update_eligible_ids,
            should_update,
            should_store,
            zone,
        )

    # ------------------------------------------------------------------
    # Linking
    # ------------------------------------------------------------------

    def link_behavior_memories(
        self,
        new_memory: BehaviorMemory,
        k: int,
        wo_link: bool,
        verbose: bool,
    ) -> Tuple[
        List[int],
        bool,
        bool,
        str,
    ]:
        if not self.behavior_memories:
            return (
                [],
                False,
                True,
                "store_only",
            )

        (
            ids,
            sims,
            update_eligible_ids,
            should_update,
            should_store,
            zone,
        ) = self.find_similar_memories(
            new_memory,
            k=k,
            verbose=verbose,
        )

        if should_update:
            self.update_decisions += 1

        if should_store:
            self.store_decisions += 1

        if not should_update:
            return (
                [],
                False,
                should_store,
                zone,
            )

        if not update_eligible_ids:
            if verbose:
                print(
                    "  -> No candidate >= "
                    "low threshold; skip linking LLM."
                )

            return (
                [],
                False,
                should_store,
                zone,
            )

        if wo_link:
            return (
                update_eligible_ids,
                True,
                should_store,
                zone,
            )

        if "store_only" in zone:
            return (
                [],
                False,
                should_store,
                zone,
            )

        id_to_sim = dict(
            zip(ids, sims)
        )

        nearest_info = []

        for thought_id in (
            update_eligible_ids
        ):
            mem = self.get_memory_by_id(
                thought_id
            )

            if mem is None:
                continue

            nearest_info.append({
                "thought_id": thought_id,
                "behavior_explanation": (
                    mem.behavior_explanation
                ),
                "pattern": (
                    mem.pattern_description
                ),
                "similarity": round(
                    float(
                        id_to_sim[
                            thought_id
                        ]
                    ),
                    4,
                ),
            })

        low, _ = (
            self.get_adaptive_thresholds()
        )

        prompt = build_linking_prompt(
            new_behavior=(
                new_memory.behavior_explanation
            ),
            new_pattern=(
                new_memory.pattern_description
            ),
            nearest_info=nearest_info,
            low_threshold=low,
        )

        try:
            response = self.gemma_generate(
                prompt=prompt,
                role_prompt=(
                    "You are a behavioral "
                    "memory modeling system. "
                    "Return only valid JSON."
                ),
                call_type="memory_linking",
            )

            result = parse_json_response(
                response
            )

            if not result.get(
                "should_link",
                False,
            ):
                if verbose:
                    print(
                        "  -> LLM linking decision: "
                        "SKIP"
                    )

                return (
                    [],
                    False,
                    should_store,
                    zone,
                )

            raw_ids = result.get(
                "linked_thought_ids",
                [],
            )

            valid_ids: List[int] = []

            for value in raw_ids:
                try:
                    thought_id = int(
                        value
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if (
                    thought_id
                    in update_eligible_ids
                ):
                    valid_ids.append(
                        thought_id
                    )

            if verbose:
                print(
                    f"  -> LLM verified "
                    f"{len(valid_ids)} linked memories"
                )

            return (
                valid_ids,
                bool(valid_ids),
                should_store,
                zone,
            )

        except Exception as exc:
            # IMPORTANT: do not use search_all fallback.
            # A failed LLM call must not silently change the method.
            print(
                "  ! Linking error "
                f"[{type(exc).__name__}]: "
                f"{repr(exc)}"
            )

            return (
                [],
                False,
                should_store,
                "linking_error_skip",
            )

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve_behavior_memories(
        self,
        new_memory: BehaviorMemory,
        linked_ids: List[int],
        max_evolutions_per_memory: int,
        embedding_batch_size: int,
        verbose: bool,
    ) -> Optional[bool]:
        if not linked_ids:
            return None

        linked_memories = [
            self.get_memory_by_id(
                thought_id
            )
            for thought_id
            in linked_ids
        ]

        linked_memories = [
            m
            for m in linked_memories
            if m is not None
        ]

        if not linked_memories:
            return None

        evolvable: List[
            BehaviorMemory
        ] = []

        for mem in linked_memories:
            mem.max_evolutions = (
                max_evolutions_per_memory
            )

            if mem.can_evolve():
                evolvable.append(
                    mem
                )
            elif verbose:
                print(
                    f"  -> Memory "
                    f"{mem.thought_id} "
                    f"reached max evolutions."
                )

        if not evolvable:
            return False

        mem_info = (
            build_mem_info_with_decay(
                evolvable,
                max_evolutions=(
                    max_evolutions_per_memory
                ),
            )
        )

        prompt = build_evolution_prompt(
            new_behavior=(
                new_memory.behavior_explanation
            ),
            new_pattern=(
                new_memory.pattern_description
            ),
            mem_info=mem_info,
        )

        try:
            response = self.gemma_generate(
                prompt=prompt,
                role_prompt=(
                    "You are a collaborative "
                    "memory evolution expert. "
                    "Return only valid JSON."
                ),
                call_type="memory_evolution",
            )

            result = parse_json_response(
                response
            )

            if not result.get(
                "should_evolve",
                False,
            ):
                return True

            placeholders = {
                "updated text",
                "updated text or null",
                "null",
                "...",
                "<updated text>",
                "none",
            }

            pending = []

            for upd in result.get(
                "updates",
                [],
            ):
                try:
                    thought_id = int(
                        upd.get(
                            "thought_id"
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                memory = (
                    self.get_memory_by_id(
                        thought_id
                    )
                )

                if memory is None:
                    continue

                # Do not allow the LLM to update an arbitrary memory
                # outside the linked/evolvable candidate set.
                if thought_id not in {
                    m.thought_id
                    for m in evolvable
                }:
                    continue

                old_values = {
                    "behavior_explanation": (
                        memory.behavior_explanation
                    ),
                    "pattern_description": (
                        memory.pattern_description
                    ),
                }

                updated = False
                update_type = []

                new_expl = upd.get(
                    "behavior_explanation"
                )

                if (
                    new_expl
                    and str(
                        new_expl
                    ).strip().lower()
                    not in placeholders
                ):
                    memory.behavior_explanation = (
                        str(new_expl).strip()
                    )
                    update_type.append(
                        "behavior_explanation"
                    )
                    updated = True

                # New prompt asks for pattern_description. Keep backward
                # compatibility with older responses using new_pattern.
                new_pattern = upd.get(
                    "pattern_description",
                    upd.get(
                        "new_pattern"
                    ),
                )

                if (
                    new_pattern
                    and str(
                        new_pattern
                    ).strip().lower()
                    not in placeholders
                ):
                    memory.pattern_description = (
                        str(
                            new_pattern
                        ).strip()
                    )
                    update_type.append(
                        "pattern"
                    )
                    updated = True

                if not updated:
                    continue

                new_values = {
                    "behavior_explanation": (
                        memory.behavior_explanation
                    ),
                    "pattern_description": (
                        memory.pattern_description
                    ),
                }

                pending.append((
                    memory,
                    old_values,
                    new_values,
                    str(
                        upd.get(
                            "reasoning",
                            "",
                        )
                    ),
                    update_type,
                ))

            if not pending:
                return True

            # Re-embed ONLY evolved memories, in one batch.
            evolved_texts = [
                compose_memory_text(
                    memory.behavior_explanation,
                    memory.pattern_description,
                    memory.keywords,
                )
                for (
                    memory,
                    *_,
                ) in pending
            ]

            evolved_embeddings = (
                self.encode_texts(
                    evolved_texts,
                    batch_size=(
                        embedding_batch_size
                    ),
                    show_progress_bar=False,
                )
            )

            for (
                pending_item,
                embedding,
            ) in zip(
                pending,
                evolved_embeddings,
            ):
                (
                    memory,
                    old_values,
                    new_values,
                    reasoning,
                    update_type,
                ) = pending_item

                self.replace_memory_embedding(
                    memory,
                    embedding,
                )

                memory.record_evolution(
                    update_type=", ".join(
                        update_type
                    ),
                    old_values=old_values,
                    new_values=new_values,
                    reasoning=reasoning,
                )

            self.evolved_batches += 1
            return True

        except Exception as exc:
            print(
                "  ! Evolution error "
                f"[{type(exc).__name__}]: "
                f"{repr(exc)}"
            )
            return None

    # ------------------------------------------------------------------
    # Process one local memory
    # ------------------------------------------------------------------

    def process_local_memory(
        self,
        new_memory: BehaviorMemory,
        link_size: int,
        wo_evolving: bool,
        wo_link: bool,
        max_evolutions_per_memory: int,
        embedding_batch_size: int,
        verbose: bool,
    ) -> Dict[str, Any]:
        self.processed_local_memories += 1

        if wo_evolving:
            self.store_memory(
                new_memory
            )

            return {
                "zone": (
                    "store_only_wo_evolving"
                ),
                "stored": True,
                "updated": False,
                "linked_ids": [],
            }

        (
            linked_ids,
            should_update,
            should_store,
            zone,
        ) = self.link_behavior_memories(
            new_memory=new_memory,
            k=link_size,
            wo_link=wo_link,
            verbose=verbose,
        )

        if (
            should_update
            and linked_ids
        ):
            new_memory.links = list(
                linked_ids
            )

            evolved = (
                self.evolve_behavior_memories(
                    new_memory=(
                        new_memory
                    ),
                    linked_ids=linked_ids,
                    max_evolutions_per_memory=(
                        max_evolutions_per_memory
                    ),
                    embedding_batch_size=(
                        embedding_batch_size
                    ),
                    verbose=verbose,
                )
            )

            if (
                evolved is False
                and not should_store
            ):
                should_store = True

        if should_store:
            self.store_memory(
                new_memory
            )
        else:
            self.discarded_memories += 1

        return {
            "zone": zone,
            "stored": bool(
                should_store
            ),
            "updated": bool(
                should_update
                and linked_ids
            ),
            "linked_ids": linked_ids,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_memory(
        self,
        filepath: str,
        processed_precompute_id: int,
        source_precomputed: str,
        embedding_model_name: str,
        run_config: Dict[str, Any],
        completed: bool,
    ) -> None:
        path = Path(filepath)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata = {
            "save_timestamp": (
                datetime.now().isoformat()
            ),
            "completed": bool(
                completed
            ),
            "processed_precompute_id": int(
                processed_precompute_id
            ),
            "processed_local_memories": int(
                self.processed_local_memories
            ),
            "num_memories": len(
                self.behavior_memories
            ),
            "source_precomputed": (
                source_precomputed
            ),
            "precompute_schema": (
                EXPECTED_PRECOMPUTE_SCHEMA
            ),
            "embedding_model": (
                embedding_model_name
            ),
            "search_type": (
                "exact_cosine_flat"
            ),
            "search_backend": (
                "faiss.IndexIDMap2(IndexFlatIP)"
            ),
            "search_stats": (
                self.faiss_index.stats.summary()
                if self.faiss_index is not None
                else SearchStats().summary()
            ),
            "llm_stats": (
                self.llm_tracker.summary()
            ),
            "construction_stats": {
                "stored_memories": (
                    self.stored_memories
                ),
                "discarded_memories": (
                    self.discarded_memories
                ),
                "update_decisions": (
                    self.update_decisions
                ),
                "store_decisions": (
                    self.store_decisions
                ),
                "evolved_batches": (
                    self.evolved_batches
                ),
            },
            "run_config": run_config,
        }

        next_thought_id = (
            max(
                (
                    m.thought_id
                    for m
                    in self.behavior_memories
                ),
                default=-1,
            )
            + 1
        )

        data = {
            "behavior_memories": [
                m.to_dict()
                for m
                in self.behavior_memories
            ],
            # Kept for compatibility with the old AMem memory JSON schema.
            "user_interaction_history": [],
            "next_thought_id": (
                next_thought_id
            ),
            "metadata": metadata,
        }

        temp_path = path.with_suffix(
            path.suffix + ".tmp"
        )

        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

        os.replace(
            temp_path,
            path,
        )

    def load_memory(
        self,
        filepath: str,
    ) -> Dict[str, Any]:
        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        self.behavior_memories = [
            BehaviorMemory.from_dict(
                x
            )
            for x in data.get(
                "behavior_memories",
                [],
            )
        ]

        self._rebuild_id2idx()
        self._rebuild_faiss_from_memory()

        metadata = data.get(
            "metadata",
            {},
        )

        self.processed_local_memories = int(
            metadata.get(
                "processed_local_memories",
                0,
            )
        )

        construction = metadata.get(
            "construction_stats",
            {},
        )

        self.stored_memories = int(
            construction.get(
                "stored_memories",
                len(
                    self.behavior_memories
                ),
            )
        )

        self.discarded_memories = int(
            construction.get(
                "discarded_memories",
                0,
            )
        )

        self.update_decisions = int(
            construction.get(
                "update_decisions",
                0,
            )
        )

        self.store_decisions = int(
            construction.get(
                "store_decisions",
                0,
            )
        )

        self.evolved_batches = int(
            construction.get(
                "evolved_batches",
                0,
            )
        )

        return metadata

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def evolution_statistics(
        self,
    ) -> Dict[str, Any]:
        if not self.behavior_memories:
            return {}

        counts = [
            m.evolution_count
            for m in self.behavior_memories
        ]

        return {
            "total_memories": len(
                counts
            ),
            "total_evolutions": int(
                sum(counts)
            ),
            "avg_evolutions_per_memory": float(
                np.mean(counts)
            ),
            "max_evolutions": int(
                max(counts)
            ),
            "never_evolved": int(
                sum(
                    c == 0
                    for c in counts
                )
            ),
            "evolved_once": int(
                sum(
                    c == 1
                    for c in counts
                )
            ),
            "evolved_multiple": int(
                sum(
                    c > 1
                    for c in counts
                )
            ),
        }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Flat-AMem global memory from text-only "
            "Gemma precomputed local memories."
        )
    )

    parser.add_argument(
        "--precomputed",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=(
            "unsloth/"
            "gemma-3-4b-it-unsloth-bnb-4bit"
        ),
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default=(
            "Qwen/"
            "Qwen3-Embedding-0.6B"
        ),
    )

    parser.add_argument(
        "--embedding_device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help=(
            "auto defaults to CPU to leave GPU memory for Gemma. "
            "Use cuda explicitly if you have enough VRAM."
        ),
    )

    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--low_threshold",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--high_threshold",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--link_size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max_evolutions_per_memory",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--llm_max_seq_length",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--wo_evolving",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--wo_link",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    precomputed_path = Path(
        args.precomputed
    )

    output_path = Path(
        args.output
    )

    if not precomputed_path.exists():
        raise FileNotFoundError(
            f"Precomputed JSONL not found: "
            f"{precomputed_path}"
        )

    records = load_precomputed_jsonl(
        str(precomputed_path)
    )

    if not records:
        raise RuntimeError(
            "No precomputed local memories found."
        )

    print("=" * 80)
    print("FLAT-AMEM GLOBAL BUILD FROM TEXT-ONLY PRECOMPUTE")
    print("=" * 80)
    print(
        f"Precomputed records : "
        f"{len(records)}"
    )
    print(
        f"Input               : "
        f"{precomputed_path}"
    )
    print(
        f"Output              : "
        f"{output_path}"
    )
    print(
        f"Link/evolve LLM     : "
        f"{args.model_name}"
    )
    print(
        f"Embedding model     : "
        f"{args.embedding_model}"
    )
    print(
        f"Embedding device    : "
        f"{args.embedding_device}"
    )
    print(
        "Search              : "
        "FAISS IndexFlatIP exact cosine"
    )

    builder = FlatAMemBuilder(
        model_name=args.model_name,
        embedding_model_name=(
            args.embedding_model
        ),
        low_threshold=(
            args.low_threshold
        ),
        high_threshold=(
            args.high_threshold
        ),
        max_new_tokens=(
            args.max_new_tokens
        ),
        llm_max_seq_length=(
            args.llm_max_seq_length
        ),
        load_in_4bit=(
            args.load_in_4bit
        ),
        embedding_device=(
            args.embedding_device
        ),
    )

    run_config = {
        "model_name": args.model_name,
        "embedding_model": (
            args.embedding_model
        ),
        "low_threshold": (
            args.low_threshold
        ),
        "high_threshold": (
            args.high_threshold
        ),
        "link_size": args.link_size,
        "max_evolutions_per_memory": (
            args.max_evolutions_per_memory
        ),
        "max_new_tokens": (
            args.max_new_tokens
        ),
        "llm_max_seq_length": (
            args.llm_max_seq_length
        ),
        "load_in_4bit": (
            args.load_in_4bit
        ),
        "wo_evolving": (
            args.wo_evolving
        ),
        "wo_link": (
            args.wo_link
        ),
        "embedding_batch_size": (
            args.embedding_batch_size
        ),
        "seed": args.seed,
    }

    last_processed_id = -1

    if (
        args.resume
        and output_path.exists()
    ):
        print(
            "\nResume enabled: "
            "loading existing global memory..."
        )

        metadata = builder.load_memory(
            str(output_path)
        )

        # Guard against accidentally resuming a different experiment.
        old_source = metadata.get(
            "source_precomputed"
        )

        old_embedding = metadata.get(
            "embedding_model"
        )

        if (
            old_source
            and Path(old_source).resolve()
            != precomputed_path.resolve()
        ):
            raise RuntimeError(
                "Existing output was built from a different "
                "precomputed file. Use a new --output."
            )

        if (
            old_embedding
            and old_embedding
            != args.embedding_model
        ):
            raise RuntimeError(
                "Existing output used a different embedding model. "
                "Use a new --output instead of resuming."
            )

        old_config = metadata.get(
            "run_config",
            {},
        )

        config_keys = [
            "low_threshold",
            "high_threshold",
            "link_size",
            "max_evolutions_per_memory",
            "wo_evolving",
            "wo_link",
        ]

        for key in config_keys:
            if (
                key in old_config
                and old_config[key]
                != run_config[key]
            ):
                raise RuntimeError(
                    f"Cannot resume: setting {key!r} changed "
                    f"from {old_config[key]!r} to {run_config[key]!r}."
                )

        last_processed_id = int(
            metadata.get(
                "processed_precompute_id",
                -1,
            )
        )

        print(
            f"Resume after precompute_id="
            f"{last_processed_id}"
        )

    pending_records = [
        rec
        for rec in records
        if int(
            rec["precompute_id"]
        ) > last_processed_id
    ]

    if not pending_records:
        print(
            "Nothing to do: global memory build is complete."
        )
        return

    # =========================================================================
    # IMPORTANT NEW DESIGN:
    # batch-encode all pending local-memory TEXTS once before sequential replay.
    # =========================================================================

    print(
        f"\nBatch encoding "
        f"{len(pending_records)} local memories..."
    )

    local_texts = [
        canonical_text_from_record(
            rec
        )
        for rec in pending_records
    ]

    embedding_start = (
        time.perf_counter()
    )

    local_embeddings = (
        builder.encode_texts(
            texts=local_texts,
            batch_size=(
                args.embedding_batch_size
            ),
            show_progress_bar=True,
        )
    )

    embedding_elapsed = (
        time.perf_counter()
        - embedding_start
    )

    print(
        f"Local-memory embedding complete: "
        f"shape={local_embeddings.shape}, "
        f"time={embedding_elapsed:.2f}s"
    )

    if (
        len(local_embeddings)
        != len(pending_records)
    ):
        raise RuntimeError(
            "Embedding count does not match pending record count."
        )

    # Sequential AMem global construction.
    build_start = (
        time.perf_counter()
    )

    progress = tqdm(
        range(
            len(pending_records)
        ),
        desc="Building global memory",
    )

    for local_idx in progress:
        rec = pending_records[
            local_idx
        ]

        embedding = local_embeddings[
            local_idx
        ]

        new_memory = record_to_memory(
            rec,
            embedding,
        )

        result = (
            builder.process_local_memory(
                new_memory=new_memory,
                link_size=(
                    args.link_size
                ),
                wo_evolving=(
                    args.wo_evolving
                ),
                wo_link=(
                    args.wo_link
                ),
                max_evolutions_per_memory=(
                    args.max_evolutions_per_memory
                ),
                embedding_batch_size=(
                    args.embedding_batch_size
                ),
                verbose=(
                    not args.quiet
                ),
            )
        )

        last_processed_id = int(
            rec["precompute_id"]
        )

        progress.set_postfix({
            "pool": len(
                builder.behavior_memories
            ),
            "zone": (
                result["zone"][:18]
            ),
            "stored": int(
                result["stored"]
            ),
        })

        processed_this_run = (
            local_idx + 1
        )

        if (
            args.save_every > 0
            and processed_this_run
            % args.save_every == 0
        ):
            builder.save_memory(
                filepath=str(
                    output_path
                ),
                processed_precompute_id=(
                    last_processed_id
                ),
                source_precomputed=str(
                    precomputed_path.resolve()
                ),
                embedding_model_name=(
                    args.embedding_model
                ),
                run_config=run_config,
                completed=False,
            )

    build_elapsed = (
        time.perf_counter()
        - build_start
    )

    builder.save_memory(
        filepath=str(
            output_path
        ),
        processed_precompute_id=(
            last_processed_id
        ),
        source_precomputed=str(
            precomputed_path.resolve()
        ),
        embedding_model_name=(
            args.embedding_model
        ),
        run_config=run_config,
        completed=True,
    )

    stats = {
        "completed": True,
        "precomputed_file": str(
            precomputed_path.resolve()
        ),
        "output_file": str(
            output_path.resolve()
        ),
        "num_precomputed_records": (
            len(records)
        ),
        "num_encoded_this_run": (
            len(pending_records)
        ),
        "local_embedding_shape": list(
            local_embeddings.shape
        ),
        "local_embedding_time_sec": round(
            embedding_elapsed,
            4,
        ),
        "global_build_time_sec": round(
            build_elapsed,
            4,
        ),
        "num_global_memories": len(
            builder.behavior_memories
        ),
        "construction": {
            "processed_local_memories": (
                builder.processed_local_memories
            ),
            "stored_memories": (
                builder.stored_memories
            ),
            "discarded_memories": (
                builder.discarded_memories
            ),
            "update_decisions": (
                builder.update_decisions
            ),
            "store_decisions": (
                builder.store_decisions
            ),
            "evolved_batches": (
                builder.evolved_batches
            ),
        },
        "search": (
            builder.faiss_index.stats.summary()
            if builder.faiss_index
            is not None
            else SearchStats().summary()
        ),
        "llm": (
            builder.llm_tracker.summary()
        ),
        "evolution": (
            builder.evolution_statistics()
        ),
        "run_config": run_config,
    }

    stats_path = (
        output_path.with_name(
            output_path.stem
            + "_build_stats.json"
        )
    )

    with stats_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(
        f"Global memory      : "
        f"{output_path}"
    )
    print(
        f"Build stats        : "
        f"{stats_path}"
    )
    print(
        f"Global pool size   : "
        f"{len(builder.behavior_memories)}"
    )
    print(
        f"Local embed time   : "
        f"{embedding_elapsed:.2f}s"
    )
    print(
        f"Sequential build   : "
        f"{build_elapsed:.2f}s"
    )

    if builder.faiss_index is not None:
        print(
            "\nExact cosine stats:"
        )
        print(
            json.dumps(
                builder.faiss_index
                .stats
                .summary(),
                indent=2,
            )
        )

    print("\nLLM stats:")
    print(
        json.dumps(
            builder.llm_tracker
            .summary(),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
